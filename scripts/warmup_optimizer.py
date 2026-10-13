"""Paired 500-step optimizer-state rounding trials on the existing SlimPajama data."""
import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
import sys
import time
import numpy as np
import torch
from benchmark_bridge import ROOT, SOURCE, config_for, factory_namespace
sys.path.insert(0,str(ROOT))
os.environ.setdefault('TIKTOKEN_CACHE_DIR',str(ROOT/'.tmp'))
from models.llama import Llama
from quantization.state_simulation import StateSimulation, variants
from quantization.packed_optimizer import PackedOptimizer
from checkpoint_io import save_checkpoint_snapshot

def write_json(path,value):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str),encoding='utf-8'); tmp.replace(path)

def run(args):
    out=args.output; out.mkdir(parents=True,exist_ok=True)
    if (out/'status.json').exists(): raise ValueError('Refusing to overwrite an existing trial')
    step=0
    try:
        write_json(out/'status.json',dict(status='initializing',pid=os.getpid(),optimizer=args.optimizer,variant=args.variant))
        torch.set_num_threads(4); torch.manual_seed(0); np.random.seed(1337)
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        cfg=config_for(args.optimizer)
        cfg.batch_size=args.microbatch; cfg.acc_steps=32//args.microbatch
        cfg.dropout=0.; cfg.bias=False; cfg.untied_embeds=False; cfg.vocab_size=50304
        cfg.init_std=.02; cfg.rmsnorm_eps=1e-5
        model=Llama(cfg).cuda()
        named=dict(model.named_parameters())
        groups=[{**g,'params':[named[n] for n in g['params']]} for g in model.get_parameter_group_specs(cfg)]
        magma_ids={id(p) for n,p in named.items() if cfg.magma_scope=='all' or 'attn' in n.lower() or 'mlp' in n.lower()}
        ns=factory_namespace(); opt=ns['build_optimizer'](cfg,model,groups,magma_ids)
        simulation=(PackedOptimizer if args.backend=='packed' else StateSimulation)(opt,args.optimizer,args.variant)
        scheduler=ns['build_scheduler'](cfg,opt,groups)
        if args.optimizer=='sf-adamw': opt.train()
        rng=np.random.default_rng(1337)
        root=ROOT/'data/slimpajama-1p5b'
        train=np.memmap(root/'train.bin',dtype=np.uint16,mode='r'); val=np.memmap(root/'val.bin',dtype=np.uint16,mode='r')
        protocol=dict(optimizer=args.optimizer,variant=args.variant,backend=args.backend,
            resource_claim='Actual payload/scales storage; per-parameter decoded workspace; engineering smoke, not a fused-kernel benchmark' if args.backend=='packed' else 'FP32 simulation storage; no compression gains',
            parameter_count=sum(p.numel() for p in model.parameters()),model='Llama 123.59M',
            parameter_dtype='float32',gradient_dtype='float32',autocast='bfloat16',
            microbatch=args.microbatch,acc_steps=cfg.acc_steps,sequence_length=512,nominal_tokens_per_update=16384,
            full_scheduler_steps=91553,warmup_steps=3000,smoke_stop=args.steps,seed=0,data_seed=1337,grad_clip=.5,
            eval_interval=200,eval_sequences=32,checkpoint_policy='one rolling checkpoint per optimizer at steps 250/500' if args.checkpoint_root else 'preflight without checkpoint',
            config=vars(cfg),torch_version=torch.__version__,gpu=torch.cuda.get_device_name(),
            data={s:json.loads((root/f'{s}_manifest.json').read_text()) for s in ['train','val']})
        write_json(out/'resolved_protocol.json',protocol)
        write_json(out/'resolved_quantization.json',dict(backend=args.backend,state_codecs=variants(args.optimizer)[args.variant],group_size=128,min_numel=4096,rounding='RNE',scale_dtype='float32',quantization_time='after upstream logical state writes',persistent_fp32_shadow=False,per_parameter_workspace=args.backend=='packed'))
        snapshot=out/'source_snapshot'; snapshot.mkdir(exist_ok=True)
        source_paths=[Path(__file__),ROOT/'scripts/benchmark_bridge.py',ROOT/'quantization/state_simulation.py',ROOT/'quantization/adamw.py',ROOT/'quantization/packed_optimizer.py']
        for path in source_paths:
            shutil.copy2(path,snapshot/path.name)
        if hasattr(simulation,'prodigy_transformed_source'):
            (snapshot/'prodigy_transformed.py').write_text(simulation.prodigy_transformed_source,encoding='utf-8')
        write_json(out/'provenance.json',dict(benchmark_commit='7776f6b0b189707911d43f11853c40851fd058a8',
            sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}))
        def batch(data,starts):
            return tuple(torch.from_numpy(np.stack([data[i+shift:i+shift+512] for i in starts]).astype(np.int64)).cuda() for shift in [0,1])
        log=(out/'metrics.jsonl').open('a',encoding='utf-8')
        def metric(step,name,value,split):
            log.write(json.dumps(dict(run_id=out.name,optimizer=args.optimizer,variant=args.variant,optimizer_step=step,nominal_tokens=step*16384,effective_tokens=step*16384,metric_name=name,value=value,split=split,measurement_protocol='new_control_smoke_'+args.backend))+'\n'); log.flush()
        def evaluate(step):
            model.eval()
            if args.optimizer=='sf-adamw': opt.eval()
            total=correct=count=0
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                for offset in range(0,32,args.microbatch):
                    x,y=batch(val,np.arange(32)[offset:offset+args.microbatch]*512)
                    r=model(x,y,get_logits=True); total+=r['loss'].item()*y.numel()
                    correct+=(r['logits'].argmax(-1)==y).sum().item(); count+=y.numel()
            loss=total/count
            for key,value in [('val_loss',loss),('val_pp',2.71828**loss),('ppl_exp_exact',math.exp(loss)),('val_acc',correct/count)]: metric(step,key,value,'val')
            if args.optimizer=='sf-adamw': opt.train()
            model.train(); return loss
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start=time.monotonic()
        def save_checkpoint(step):
            if args.checkpoint_root is None: return
            folder=args.checkpoint_root/args.optimizer; folder.mkdir(parents=True,exist_ok=True)
            temporary=folder/'last.pt.partial'; target=folder/'last.pt'
            save_checkpoint_snapshot(dict(model=model.state_dict(),optimizer=opt.state_dict(),
                scheduler=None if scheduler is None else scheduler.state_dict(),step=step,protocol=protocol,
                data_rng=rng.bit_generator.state,cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),temporary)
            temporary.replace(target)
            write_json(folder/'checkpoint.json',dict(optimizer=args.optimizer,variant=args.variant,backend=args.backend,step=step,bytes=target.stat().st_size,source_run=str(out)))
        final_val=None
        for step in range(1,args.steps+1):
            opt.zero_grad(set_to_none=True); total=torch.zeros((),device='cuda')
            starts=rng.integers(0,len(train)-513,size=32)
            for offset in range(0,32,args.microbatch):
                x,y=batch(train,starts[offset:offset+args.microbatch])
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss=model(x,y,precond_flag=(args.optimizer=='newton-muon' and opt.precond_flag_for_step()))['loss']
                (loss/cfg.acc_steps).backward(); total+=loss.detach()/cfg.acc_steps
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.5,error_if_nonfinite=True)
            if args.optimizer=='sophiag': opt.step(bs=cfg.sophia_bs*512)
            else: opt.step()
            if scheduler is not None: scheduler.step()
            if args.optimizer=='sophiag':
                opt.zero_grad(set_to_none=True)
                if step%cfg.precondition_frequency==0:
                    # Pinned upstream legacy_last_microbatch estimator, with its /acc_steps scaling.
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        r=model(x,y,get_logits=True)
                        sampled=torch.distributions.Categorical(logits=r['logits']).sample()
                        sampled_loss=torch.nn.functional.cross_entropy(r['logits'].view(-1,r['logits'].size(-1)),sampled.view(-1))
                    (sampled_loss/cfg.acc_steps).backward(); opt.update_hessian()
                    if args.backend=='simulation': simulation.round_states('hessian')
                    opt.zero_grad(set_to_none=True)
            if args.optimizer.startswith('mars-'):
                opt.zero_grad(set_to_none=True); opt.update_last_grad()
            value=total.item()
            if not math.isfinite(value): raise FloatingPointError('nonfinite loss')
            metric(step,'train_loss',value,'train'); metric(step,'lr',opt.param_groups[0]['lr'],'train')
            metric(step,'grad_norm',norm.item(),'train')
            if step%200==0 or step==args.steps: final_val=evaluate(step)
            if step%250==0: save_checkpoint(step)
            if step%10==0 or step==1:
                state=dict(status='running',optimizer=args.optimizer,variant=args.variant,step=step,target=args.steps,seconds=time.monotonic()-start,pid=os.getpid(),loss=value)
                write_json(out/'status.json',state); print(json.dumps(state),flush=True)
        torch.cuda.synchronize(); elapsed=time.monotonic()-start
        inventory=simulation.inventory(named.items()); write_json(out/'state_inventory.json',inventory)
        state_bytes=sum(i['payload_bytes']+i['metadata_bytes'] for i in inventory)
        summary=dict(status='completed',steps=args.steps,seconds=elapsed,tokens_per_sec_including_eval=args.steps*16384/elapsed,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            persistent_moment_bytes=sum(i['payload_bytes']+i['metadata_bytes'] for i in inventory if i['role'] in {'m','v','h','z','matrix_momentum'}),
            persistent_optimizer_state_bytes=state_bytes,final_train_loss=value,final_val_loss=final_val,
            backend=args.backend,max_simultaneously_decoded_parameters=getattr(simulation,'max_active_parameters',None),
            max_decoded_numel=getattr(simulation,'max_decoded_numel',None),formal_resource_probe=False,quality_conclusion='500-step engineering smoke only')
        write_json(out/'summary.json',summary); write_json(out/'status.json',dict(status='completed',step=args.steps,pid=os.getpid()))
        print(json.dumps(summary),flush=True)
    except Exception as exc:
        write_json(out/'status.json',dict(status='failed',step=step,error_type=type(exc).__name__,error=str(exc),pid=os.getpid()))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--optimizer',required=True); p.add_argument('--variant',required=True)
    p.add_argument('--backend',choices=['packed','simulation'],default='packed')
    p.add_argument('--checkpoint-root',type=Path)
    p.add_argument('--steps',type=int,default=500); p.add_argument('--microbatch',type=int,default=2); p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.steps<1 or args.steps>500 or args.microbatch<1 or 32%args.microbatch: p.error('invalid steps/microbatch')
    if args.optimizer.startswith('gn-'): p.error('GN integration requires separate inner-loop protocol validation')
    run(args)
