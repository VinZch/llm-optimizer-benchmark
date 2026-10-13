"""Full 91,553-update paired control; upstream dense 123.59M Llama."""
import gc
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"vendor/benchmark/src")]
os.environ.setdefault("TIKTOKEN_CACHE_DIR",str(ROOT/".tmp"))
from models.llama import Llama
from quantization.adamw import QuantizedAdamW,stored_bytes
from checkpoint_io import save_checkpoint_snapshot
from full_run_utils import verify_data, evaluation_batch, rewind_metrics

VARIANTS={"A0":("fp32","fp32"),"A1":("bf16","fp32"),"A2":("fp32","bf16"),"A3":("fp8","fp32"),"A4":("fp32","fp8"),"A5":("fp8","bf16"),"A6":("fp8","fp8")}

def write_json(path,value):
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding="utf-8")
    tmp.replace(path)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--variant",choices=list(VARIANTS)+["native"],default="A0")
    parser.add_argument("--steps",type=int,default=91553)
    parser.add_argument("--microbatch",type=int,default=8)
    parser.add_argument("--data-root",type=Path,default=ROOT/"data/slimpajama-1p5b")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--synthetic-preflight",action="store_true")
    parser.add_argument("--resume",type=Path)
    args=parser.parse_args()
    if args.steps<=0 or args.steps>91553 or args.microbatch<=0 or 32%args.microbatch:
        raise ValueError("invalid steps/microbatch; effective batch must remain 32")
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/"status.json").exists() and not args.resume:
        raise ValueError("output exists; resume explicitly or choose a new directory")
    if args.resume and (args.output/"status.json").exists() and json.loads((args.output/"status.json").read_text())["status"]=="completed":
        raise ValueError("run already completed; do not overwrite its results with an empty resume")
    status_path=args.output/"status.json"
    write_json(status_path,dict(status="initializing",variant=args.variant))
    try:
        torch.set_num_threads(4)
        torch.manual_seed(0); np.random.seed(1337)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        config=SimpleNamespace(vocab_size=50304,sequence_length=512,n_embd=768,n_head=12,n_layer=12,multiple_of=256,rmsnorm_eps=1e-5,dropout=0.,bias=False,untied_embeds=False,init_std=.02,moe=False,opt="adamw",parallel_block=False,mlp_dim_exp_factor=1.)
        model=Llama(config).cuda()
        parameters=dict(model.named_parameters())
        group_specs=model.get_parameter_group_specs(config)
        groups=[{**spec,"params":[parameters[n] for n in spec["params"]]} for spec in group_specs]
        common=dict(lr=5e-4,betas=(.9,.95),eps=1e-8,weight_decay=.1)
        if args.variant=="native":
            opt=torch.optim.AdamW(groups,**common,fused=True)
        else:
            m,v=VARIANTS[args.variant]
            opt=QuantizedAdamW(groups,**common,m_codec=m,v_codec=v)
        scheduler=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=[5e-4]*2,total_steps=91553,pct_start=3000/91553,anneal_strategy="cos",cycle_momentum=False,div_factor=100.,final_div_factor=1.)
        rng=np.random.default_rng(1337)
        manifests={}
        if not args.synthetic_preflight:
            for split in ["train","val"]:
                manifests[split]=json.loads((args.data_root/f"{split}_manifest.json").read_text())
            for split in ["train","val"]:
                verify_data(args.data_root/f"{split}.bin",manifests[split])
            train=np.memmap(args.data_root/"train.bin",dtype=np.uint16,mode="r")
            val=np.memmap(args.data_root/"val.bin",dtype=np.uint16,mode="r")
        else:
            train=rng.integers(0,50257,size=100000,dtype=np.uint16); val=train.copy()
        def batch(data,starts):
            x=np.stack([data[i:i+512] for i in starts]).astype(np.int64)
            y=np.stack([data[i+1:i+513] for i in starts]).astype(np.int64)
            return torch.from_numpy(x).cuda(),torch.from_numpy(y).cuda()
        protocol=dict(status="new_control",historical_reproduction=False,variant=args.variant,model=vars(config),parameter_count=sum(p.numel() for p in model.parameters()),parameter_dtype=str(next(model.parameters()).dtype),gradient_dtype="float32",autocast="bfloat16",microbatch=args.microbatch,acc_steps=32//args.microbatch,sequence_length=512,nominal_tokens_per_update=16384,full_scheduler_steps=91553,warmup_steps=3000,training_target_steps=91553,grad_clip=.5,lr=5e-4,betas=[.9,.95],eps=1e-8,weight_decay=.1,seed=0,data_seed=1337,new_choices=["betas/epsilon from fixed upstream defaults","microbatch with gradient accumulation","fixed source dataset prefix","periodic evaluation: 32 fixed sequences; final: all validation targets","PyTorch/Windows/backend versions"],torch_version=torch.__version__,cuda_runtime=torch.version.cuda,gpu=torch.cuda.get_device_name(),attention_backend="upstream SDPA with autocast; dispatch by PyTorch",compile=False,ddp=False,tf32=False,data=manifests,synthetic_preflight=args.synthetic_preflight,measurement_protocol="full training; logs/evaluation/checkpoints included; not formal resource probe")
        write_json(args.output/"resolved_protocol.json",protocol)
        write_json(args.output/"resolved_quantization.json",dict(variant=args.variant,state_codecs=VARIANTS.get(args.variant),group_size=128,min_numel=4096,rounding="RNE",scale_dtype="float32",permanent_fp32_shadow=False))
        write_json(args.output/"provenance.json",dict(benchmark_commit="7776f6b0b189707911d43f11853c40851fd058a8",design_commit="6d7abf437819475447993eacb95d912034e6d05b",adapter_sha256=hashlib.sha256((ROOT/"quantization/adamw.py").read_bytes()).hexdigest(),runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),checkpoint_io_sha256=hashlib.sha256((ROOT/"scripts/checkpoint_io.py").read_bytes()).hexdigest()))
        protocol["evaluation"]={"periodic_sequences":32,"interval":200,"final_targets":"all len(val)-1 targets, padded final sequence masked"}
        start_step=0
        prior_seconds=0.
        prior_allocated=prior_reserved=0
        if args.resume:
            ckpt=torch.load(args.resume,map_location="cpu",weights_only=False)
            if ckpt["protocol"]!=protocol: raise ValueError("resume protocol mismatch")
            model.load_state_dict(ckpt["model"]); opt.load_state_dict(ckpt["optimizer"]); scheduler.load_state_dict(ckpt["scheduler"])
            rng.bit_generator.state=ckpt["data_rng"]
            torch.set_rng_state(ckpt["cpu_rng"].cpu()); torch.cuda.set_rng_state_all([t.cpu() for t in ckpt["cuda_rng"]])
            start_step=ckpt["step"]
            prior_seconds=ckpt.get("elapsed_seconds",0.)
            prior_allocated=ckpt.get("peak_allocated_bytes",0)
            prior_reserved=ckpt.get("peak_reserved_bytes",0)
            if start_step>=args.steps: raise ValueError("checkpoint already reached requested stop")
            del ckpt
            gc.collect()
            rewind_metrics(args.output/"metrics.jsonl",start_step)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        started=time.monotonic()
        losses=[]
        metric_file=(args.output/"metrics.jsonl").open("a",encoding="utf-8")
        def metric(step,name,value,split):
            metric_file.write(json.dumps(dict(run_id=args.output.name,optimizer_step=step,nominal_tokens=step*16384,effective_tokens=step*16384,metric_name=name,value=value,split=split,measurement_protocol="new_control_full"))+"\n")
            metric_file.flush()
        def evaluate(step, full=False):
            model.eval(); loss_sum=correct=count=0
            starts=np.arange(0,len(val)-1,512) if full else np.arange(32)*512
            with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
                for offset in range(0,len(starts),args.microbatch):
                    ex,ey=evaluation_batch(val,starts[offset:offset+args.microbatch])
                    x,y=torch.from_numpy(ex).cuda(),torch.from_numpy(ey).cuda()
                    result=model(x,y,get_logits=True)
                    valid=(y!=-1); targets=int(valid.sum().item())
                    loss_sum+=result["loss"].item()*targets
                    correct+=((result["logits"].argmax(-1)==y)&valid).sum().item(); count+=targets
            metric(step,"evaluated_targets",count,"val")
            loss=loss_sum/count
            metric(step,"val_loss",loss,"val"); metric(step,"val_pp",2.71828**loss,"val"); metric(step,"ppl_exp_exact",math.exp(loss),"val"); metric(step,"val_acc",correct/count,"val")
            model.train(); return loss
        checkpoint=args.output/"checkpoints"; checkpoint.mkdir(exist_ok=True)
        def save_checkpoint(step):
            target=checkpoint/"last.pt"
            temporary=checkpoint/"last.pt.partial"
            save_checkpoint_snapshot(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),step=step,data_rng=rng.bit_generator.state,cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),protocol=protocol,elapsed_seconds=prior_seconds+time.monotonic()-started,peak_allocated_bytes=max(prior_allocated,torch.cuda.max_memory_allocated()),peak_reserved_bytes=max(prior_reserved,torch.cuda.max_memory_reserved())),temporary)
            temporary.replace(target)
        final_val=None
        for step in range(start_step+1,args.steps+1):
            opt.zero_grad(set_to_none=True)
            loss_acc=torch.zeros((),device="cuda")
            starts=rng.integers(0,len(train)-513,size=32)
            for offset in range(0,32,args.microbatch):
                x,y=batch(train,starts[offset:offset+args.microbatch])
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    loss=model(x,y)["loss"]
                (loss/(32//args.microbatch)).backward()
                loss_acc+=loss.detach()/(32//args.microbatch)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.5,error_if_nonfinite=True)
            opt.step(); scheduler.step()
            loss_value=loss_acc.item()
            if not math.isfinite(loss_value): raise FloatingPointError("nonfinite loss")
            losses.append(loss_value)
            metric(step,"train_loss",loss_value,"train")
            if step%200==0 or step==args.steps:
                final_val=evaluate(step,full=(step==91553))
            if step%1000==0:
                save_checkpoint(step)
            if step%10==0 or step==1:
                print(json.dumps(dict(variant=args.variant,step=step,loss=loss_value,lr=scheduler.get_last_lr()[0],seconds=time.monotonic()-started,peak_gib=torch.cuda.max_memory_allocated()/2**30)),flush=True)
                write_json(status_path,dict(status="running",step=step,target=args.steps,seconds=prior_seconds+time.monotonic()-started))
        torch.cuda.synchronize()
        elapsed=time.monotonic()-started
        inventory=[]
        for name,p in parameters.items():
            for key,s in opt.state[p].items():
                if key not in ["exp_avg","exp_avg_sq"]: continue
                if args.variant=="native":
                    entry=dict(payload_bytes=s.numel()*s.element_size(),metadata_bytes=0,stored_dtype=str(s.dtype),codec="native_fp32")
                else:
                    entry=dict(payload_bytes=s["payload"].numel()*s["payload"].element_size(),metadata_bytes=0 if s["scales"] is None else s["scales"].numel()*4,stored_dtype=str(s["payload"].dtype),codec=s["codec"],writes=s["writes"])
                inventory.append(dict(canonical_parameter_name=name,shape=list(p.shape),branch="adamw",state_key=key,role="m" if key=="exp_avg" else "v",actually_used=True,protected_reason="numel<4096" if p.numel()<4096 else None,**entry))
        write_json(args.output/"state_inventory.json",inventory)
        summary=dict(status="completed" if args.steps==91553 else "paused",steps=args.steps,seconds=prior_seconds+elapsed,session_seconds=elapsed,updates_timed=args.steps-start_step,tokens_per_sec_including_eval=(args.steps-start_step)*16384/elapsed,peak_allocated_bytes=max(prior_allocated,torch.cuda.max_memory_allocated()),peak_reserved_bytes=max(prior_reserved,torch.cuda.max_memory_reserved()),persistent_moment_bytes=sum(i["payload_bytes"]+i["metadata_bytes"] for i in inventory),final_train_loss=losses[-1] if losses else None,final_val_loss=final_val,synthetic_preflight=args.synthetic_preflight,quality_conclusion="paired full training; assess all variants and seeds before conclusions",final_evaluated_targets=(len(val)-1 if args.steps==91553 else 16384),formal_resource_probe=False)
        write_json(args.output/"summary.json",summary)
        save_checkpoint(args.steps)
        write_json(status_path,dict(status=summary["status"],step=args.steps,target=91553)); print(json.dumps(summary),flush=True)
    except Exception as exc:
        write_json(status_path,dict(status="failed",error_type=type(exc).__name__,error=str(exc)))
        raise

if __name__=="__main__": main()
