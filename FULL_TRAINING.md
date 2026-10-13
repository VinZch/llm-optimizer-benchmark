# RTX 4060 AdamW 完整训练

本次授权：A0 (mFP32/vFP32)、A3 (mFP8/vFP32)、A5 (mFP8/vBF16) 各 91,553 步，按此顺序在单卡运行。每组从 seed 0 初始化重新开始；已有 500 步报告与 checkpoint 保留。

使用完整上游仓库固定提交 7776f6b0b189707911d43f11853c40851fd058a8 的 Llama：12 层、12 头、768 hidden，共 123,587,328 参数。模型源码与 vendor 副本忽略 CRLF 后一致。协议继承本地新配对对照：SlimPajama 1,500,004,352 个训练 token，GPT-2 tokenizer，sequence 512，microbatch 8 × accumulate 4，每步 16,384 target，warmup 3,000，cosine scheduler 91,553，LR 5e-4，betas (0.9,0.95)，eps 1e-8，weight decay 0.1，clip 0.5。训练按固定 RNG 随机抽取文档拼接文件中的窗口，累计 target 预算不表示每个数据 token 恰好遍历一次。

每 200 步评估与 smoke 相同的固定 32 条序列；最后覆盖整个验证文件的 9,999,999 个 next-token target，尾部不足 512 的序列以 -1 屏蔽。不能将最终 loss 与仅 16,384 target 的 smoke loss 直接混为同一评估。

每 1,000 步覆盖保存 CPU snapshot checkpoint，并在阶段停止或完成时保存；恢复保留模型、压缩状态 dtype、scheduler 与 CPU/CUDA/数据 RNG，删除 checkpoint 之后尚未持久化的指标再重放更新。启动前逐字节核对训练与验证文件 SHA256。

```powershell
.\scripts\run_adamw_full.ps1
# 中断后，确认旧进程已经退出，再显式恢复：
.\scripts\run_adamw_full.ps1 -Resume
```

状态：runs/adamw-full/suite-status.json；各组 status.json、metrics.jsonl、checkpoints/last.pt；后台输出 stdout.log/stderr.log。完成三组后生成 FULL_RESULTS.md 和 runs/adamw-full/comparison.json。原方法设计的多 seed 验收需另跑 seed 1/2，本次只启动与 smoke 一致的 seed 0 三组。

按此前实测估算每组约 31 小时，三组约 4 天，实际受温度、功耗和最终评估影响。电脑须保持运行；checkpoint 支持中断后手动恢复。
