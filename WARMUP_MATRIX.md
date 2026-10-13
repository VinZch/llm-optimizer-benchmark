# 124M optimizer quantization warmup matrix

Deadline: 2026-10-14 22:00 Asia/Shanghai. Run each optimizer’s variants consecutively.
500 updates each; SlimPajama fixed 1.5B-token file; 16,384 targets/update; seed 0/data seed 1337.
Same model, FP32 weights/gradients, BF16 autocast, clip 0.5, cosine horizon 91,553 and warmup 3,000 as the original AdamW smoke.
Optimizer-specific lr/betas/options come from the pinned upstream 124m shell recipes and parser defaults. SF uses its original internal warmup without external scheduling.

## Representation and limitations

This first broad pass is **state-rounding simulation**: round selected states after their original final writes, immediately decode to FP32, and use those rounded states on the next step. No unrounded shadow drives subsequent updates. Actual persistent storage is FP32. Memory and speed measurements describe simulation overhead and cannot demonstrate packed-storage savings.
Unselected states stay in their original representation: AdEMAMix slow EMA; Prodigy s/p0/scalars; Magma scale; SOAP GG/Q; MARS history and backup; Muon backup and Newton preconditioners. Sophia h is rounded only on Hessian refresh. SF train/eval never rounds z.
No full-run ranking from a 500-step single-seed smoke. GN-prox/full are included as pending protocol integration, not silently substituted. AdamW full A5 remains disabled.
Keep all logs, metrics, configurations and state inventories. These short trials do not retain model checkpoints (disk budget); existing A0/A3 checkpoints remain archived.

## Matrix

|Order|Optimizer|Configurations|
|---:|---|---|
|1|lion|L0, L1, L2|
|2|signum|L0, L1, L2|
|3|cadamw|A0, A1, A2, A3, A4, A5, A6|
|4|adopt|A0, A1, A2, A3, A4, A5, A6|
|5|ademamix|A0, A1, A2, A3, A4, A5, A6|
|6|adamw-magma|A0, A1, A2, A3, A4, A5, A6|
|7|sf-adamw|S0, S1, S2, S3, S4|
|8|prodigy|A0, A1, A2, A3, A4, A5, A6|
|9|mars-adamw|A0, A1, A2, A3, A4, A5, A6|
|10|mars-lion|L0, L1, L2|
|11|mars-shampoo|L0, L1, L2|
|12|sophiag|H0, H1, H2, H3, H4, H5, H6|
|13|muon|L0, L1, L2|
|14|d-muon|L0, L1, L2|
|15|muon-magma|L0, L1, L2|
|16|softeq-k2000-muon|L0, L1, L2|
|17|soap|A0, A1, A2, A3, A4, A5, A6|
|18|newton-muon|L0, L1, L2|

A0=(m32,v32), A1=(mBF16,v32), A2=(m32,vBF16), A3=(mFP8,v32), A4=(m32,vFP8), A5=(mFP8,vBF16), A6=(mFP8,vFP8).
L0/L1/L2 = selected momentum FP32/BF16/FP8. H0–H6 use Hessian in place of v.
S0=(z32,v32), S1=(z32,vBF16), S2=(z32,vFP8), S3=(zBF16,v32), S4=(zBF16,vBF16).

## Measurements

metrics.jsonl keeps the AdamW field schema: optimizer_step, nominal/effective tokens, train_loss each update; val_loss, val_pp, ppl_exp_exact, val_acc at steps 200/400/500. Additional lr and grad_norm each update.
summary.json keeps seconds, tokens_per_sec_including_eval, peak_allocated_bytes, peak_reserved_bytes, persistent_moment_bytes and final train/val loss; persistent_optimizer_state_bytes includes protected auxiliary states. State bytes are actual FP32 simulation storage. Each group receives combined CSV and SVG loss curves.
