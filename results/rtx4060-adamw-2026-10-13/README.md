# RTX 4060 AdamW A0 / A3 完整实验归档

两组均完成 91,553 次更新、累计 1,500,004,352 target。模型为上游 Llama 123,587,328 参数，seed 0，SlimPajama 固定前缀；最终验证覆盖独立验证文件的 9,999,999 个 target。

|配置|最终 val loss|耗时小时|持久 moment MiB|峰值 allocated GiB|
|---|---:|---:|---:|---:|
|A0|3.41535549|29.157|942.90|6.0513|
|A3|3.41468677|30.307|593.05|5.7183|

A0=mFP32/vFP32；A3=mFP8/vFP32。A3 状态节省约 37.10%。单 seed 的细小 loss 差异不能证明方法优劣。此实验是本地新配对对照，数据、评估与硬件并非历史严格复现。

## 产物

- adamw-full/A0、adamw-full/A3：完整逐步日志、配置、状态清单、来源和结果。
- adamw-A0-500、adamw-A3-500：此前真实数据 500 步验证。
- archive-A0-before-checkpoint-fix：修复 checkpoint storage 保留问题前的 A0，资源数据已排除；保留原始排除声明。
- preflight-A0-mb8：随机数据功能检查，不作为真实训练结果。
- full-suite-*：原始整队列输出，其中可能含用户改策略前短暂启动的 A5 记录。
- data/：数据来源和 SHA256；源语料不是本实验产生的结果，未重复上传。

所有仍保留的 checkpoint 存于 [GitHub Release](https://github.com/VinZch/llm-optimizer-benchmark/releases/tag/rtx4060-adamw-a0-a3-2026-10-13)；逐文件大小、SHA256 和下载地址见 artifact-manifest.json。last.pt 在训练过程中覆盖保存，因此不存在全部历史步数的 checkpoint。

A5 已按用户新策略停止，另开展其他优化器 warmup。
