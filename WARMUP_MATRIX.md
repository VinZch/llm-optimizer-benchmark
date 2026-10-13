# 124M optimizer quantization warmup matrix

Deadline: 2026-10-14 22:00 Asia/Shanghai. Run each optimizer’s variants consecutively.
500 updates each; SlimPajama fixed 1.5B-token file; 16,384 targets/update; seed 0/data seed 1337.
Same model, FP32 weights/gradients, BF16 autocast, clip 0.5, cosine horizon 91,553 and warmup 3,000 as the original AdamW smoke.
Optimizer-specific lr/betas/options come from the pinned upstream 124m shell recipes and parser defaults. SF uses its original internal warmup without external scheduling.

## Representation and limitations

This suite uses **actual packed optimizer states**, as requested by the user. FP8 E4M3FN payloads/BF16 tensors remain low precision between updates; FP8 scales are FP32, blocks contain 128 elements, tensors below 4096 elements stay FP32. Decode only one parameter at a time, update in FP32, then encode after final use. No persistent FP32 shadow and no full-model decoding.
The previous simulation suite is stopped and retained separately under runs/optimizer-warmup. All packed configurations restart from the same seed in runs/optimizer-warmup-packed; simulation results are not mixed into these comparisons.
Per-parameter execution preserves group clocks and Magma mask order. Prodigy retains its global d reduction, moving the independent m/v recurrence to the final parameter pass using the original old d. Newton preconditioner refresh remains once per original step. This is a Python reference adapter, not an optimized fused kernel.
Unselected states stay in their original representation: AdEMAMix slow EMA; Prodigy s/p0/scalars; Magma scale; SOAP GG/Q; MARS history and backup; Muon backup and Newton preconditioners. Sophia h is rounded only on Hessian refresh. SF train/eval never rounds z.
No full-run ranking from a 500-step single-seed smoke. GN-prox/full are included as pending protocol integration, not silently substituted. AdamW full A5 remains disabled.
Keep all logs, metrics, configurations, source snapshots and state inventories. Save a rolling model/optimizer/RNG/scheduler checkpoint per optimizer at steps 250 and 500; later variants overwrite this rolling checkpoint. Existing AdamW A0/A3 checkpoints remain archived. This avoids exhausting disk space.

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
summary.json keeps seconds, tokens_per_sec_including_eval, peak_allocated_bytes, peak_reserved_bytes, persistent_moment_bytes and final train/val loss; persistent_optimizer_state_bytes includes protected auxiliary states. Count actual low-precision payload and scale bytes. Each group receives combined CSV and SVG loss curves. Peak memory includes workspace/activations; state-byte reductions do not imply the same reduction in total peak memory or speedups.

## Validation

CUDA checks cover all 18 adapters and all 88 configurations against unmodified native FP32 and recurrent-rounding references; additional 256-step replay, SF mode changes, Sophia Hessian refresh, packed dtype/byte inventory and checkpoint resume. Full 124M memory preflight precedes launch. Tests do not prove long-run convergence.
