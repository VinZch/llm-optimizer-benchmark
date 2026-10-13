"""Materialize ordered, reviewable experiment matrix."""
import sys
from pathlib import Path
from benchmark_bridge import ROOT, config_for
sys.path.insert(0,str(ROOT))
from quantization.state_simulation import variants
from warmup_optimizer import write_json

ORDER=['lion','signum','cadamw','adopt','ademamix','adamw-magma','sf-adamw','prodigy',
       'mars-adamw','mars-lion','mars-shampoo','sophiag','muon','d-muon','muon-magma',
       'softeq-k2000-muon','soap','newton-muon']

def main():
    root=ROOT/'runs/optimizer-warmup'; root.mkdir(parents=True,exist_ok=True)
    runs=[]
    for name in ORDER:
        cfg=config_for(name)
        for variant,codecs in variants(name).items():
            runs.append(dict(optimizer=name,variant=variant,steps=500,microbatch=2,codecs=codecs,
                             backend='simulation',lr=cfg.lr,scheduler=cfg.scheduler))
    # GN has 8 inner updates/outer update and requires JVP-aware batching. Never masquerade as AdamW.
    deferred=[dict(optimizer=n,variants=list(variants(n)),status='pending_inner_loop_validation',
                   reason='GN 8-inner-step JVP objective requires its own accumulation and memory validation') for n in ['gn-prox','gn-full']]
    matrix=dict(protocol='original AdamW 500-step engineering smoke',deadline='2026-10-14T22:00:00+08:00',
                ordering='all variants of one optimizer consecutively',runs=runs,deferred=deferred,
                prior_adamw='A0/A3 full and A0/A3/A5 500-step archived separately; A5 full stopped by user',
                scheduler_steps=91553,warmup_steps=3000,tokens_per_update=16384,seed=0,data_seed=1337,
                checkpoint_policy='retain all metrics/configs; no per-trial model checkpoints to stay within disk capacity',
                failure_policy='record failed trial; no retry; baseline failure blocks that optimizer family; continue independent families')
    write_json(root/'matrix.json',matrix)
    lines=['# 124M optimizer quantization warmup matrix','',
           'Deadline: 2026-10-14 22:00 Asia/Shanghai. Run each optimizer’s variants consecutively.',
           '500 updates each; SlimPajama fixed 1.5B-token file; 16,384 targets/update; seed 0/data seed 1337.',
           'Same model, FP32 weights/gradients, BF16 autocast, clip 0.5, cosine horizon 91,553 and warmup 3,000 as the original AdamW smoke.',
           'Optimizer-specific lr/betas/options come from the pinned upstream 124m shell recipes and parser defaults. SF uses its original internal warmup without external scheduling.',
           '', '## Representation and limitations', '',
           'This first broad pass is **state-rounding simulation**: round selected states after their original final writes, immediately decode to FP32, and use those rounded states on the next step. No unrounded shadow drives subsequent updates. Actual persistent storage is FP32. Memory and speed measurements describe simulation overhead and cannot demonstrate packed-storage savings.',
           'Unselected states stay in their original representation: AdEMAMix slow EMA; Prodigy s/p0/scalars; Magma scale; SOAP GG/Q; MARS history and backup; Muon backup and Newton preconditioners. Sophia h is rounded only on Hessian refresh. SF train/eval never rounds z.',
           'No full-run ranking from a 500-step single-seed smoke. GN-prox/full are included as pending protocol integration, not silently substituted. AdamW full A5 remains disabled.',
           'Keep all logs, metrics, configurations and state inventories. These short trials do not retain model checkpoints (disk budget); existing A0/A3 checkpoints remain archived.',
           '', '## Matrix', '', '|Order|Optimizer|Configurations|', '|---:|---|---|']
    for i,name in enumerate(ORDER): lines.append(f'|{i+1}|{name}|'+', '.join(variants(name))+'|')
    lines += ['', 'A0=(m32,v32), A1=(mBF16,v32), A2=(m32,vBF16), A3=(mFP8,v32), A4=(m32,vFP8), A5=(mFP8,vBF16), A6=(mFP8,vFP8).',
              'L0/L1/L2 = selected momentum FP32/BF16/FP8. H0–H6 use Hessian in place of v.',
              'S0=(z32,v32), S1=(z32,vBF16), S2=(z32,vFP8), S3=(zBF16,v32), S4=(zBF16,vBF16).',
              '', '## Measurements', '', 'metrics.jsonl keeps the AdamW field schema: optimizer_step, nominal/effective tokens, train_loss each update; val_loss, val_pp, ppl_exp_exact, val_acc at steps 200/400/500. Additional lr and grad_norm each update.',
              'summary.json keeps seconds, tokens_per_sec_including_eval, peak_allocated_bytes, peak_reserved_bytes, persistent_moment_bytes and final train/val loss; persistent_optimizer_state_bytes includes protected auxiliary states. State bytes are actual FP32 simulation storage. Each group receives combined CSV and SVG loss curves.']
    (ROOT/'WARMUP_MATRIX.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'{len(runs)} planned trials across {len(ORDER)} optimizer variants; 2 GN methods pending integration')

if __name__=='__main__': main()
