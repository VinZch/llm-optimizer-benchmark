# RTX 4060 本地 AdamW 验证

本地实现范围：固定 benchmark 提交 `7776f6b0b189707911d43f11853c40851fd058a8` 的 dense Llama 模型，123,587,328 参数；新增 AdamW 持久状态编码与独立 smoke runner。上游模型文件保留，`models/utils.py` 仅保留 dense Llama 所需注册表。许可证见 `vendor/benchmark/LICENSE`。

这是一套新的配对对照，不能称为历史严格复现。历史门户未记录数据文件、betas、epsilon、完整评估预算；本地显式采用固定源码默认 betas=(0.9,0.95)、epsilon=1e-8，以及独立固定的数据前缀和验证序列。

## 环境和数据

- RTX 4060 Laptop，8188 MiB；驱动 566.07。
- 项目 `.venv`：Python 3.13，PyTorch 2.9.1+cu126。系统 CUDA Toolkit 12.6 已安装；该参考实现不需要自行编译 CUDA 内核。
- SlimPajama 原仓库当前不可用；使用社区重上传 `gmongaras/SlimPajama-627B_Reupload`，固定 revision `c34c22dbb10ae6b264a2f357a909d1a537141b36`。镜像端点重定向至官方 HF 存储。manifest 记录原始文件及 token 文件的 SHA256。
- 训练目标 1,500,004,352 token；验证集 10,000,000 token。取各独立 split 的固定文档前缀，以 GPT-2 分词，每篇后附 EOT。此选择不等同于已知历史数据，也不代表全部 SlimPajama。
- `data/slimpajama-1p5b/*_preparation_status.json` 是准备进度；只有最终 `*_manifest.json` 和 `.bin` 发布后才可训练。

数据转换命令（源分片下载及 SHA256 校验通过后）：

```powershell
.\.venv\Scripts\python.exe scripts/prepare_slimpajama.py --split train --tokens 1500004352 --source data/raw/train-00000.parquet
.\.venv\Scripts\python.exe scripts/prepare_slimpajama.py --split val --tokens 10000000 --source data/raw/validation-00000.parquet
```

当前任务的自动流水线状态保存在 `runs/pipeline-status.json`，训练阶段各 run 的 `status.json` 保存更新进度。流水线遇到校验失败、OOM 或数值异常会停止并记录，不自动改变实验配置。

## 验证

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_adamw -v
.\scripts\run_adamw_smoke.ps1
```

初始集合为 A0（mFP32/vFP32）、A3（mFP8/vFP32）、A5（mFP8/vBF16），每个 500 个 optimizer update，同 seed 和数据 RNG。A1、A2、A4、A6 已支持，可显式传 `-Variants A0,A1,A2,A3,A4,A5,A6` 扩展。

microbatch=8、acc_steps=4；保持每次更新 16,384 token、序列 512、FP32 参数与梯度、BF16 autocast，完整 scheduler=91,553 步、warmup=3,000。500 步仅暂停，不能将 scheduler 总长度改为 500。LR=5e-4、weight decay=0.1、grad_clip=0.5；上游参数分组保留。每 200 步和最后评估同一组 32 条验证序列。

FP8 为 block128 E4M3FN、FP32 upward-rounded scale、RNE；小于 4096 元素的状态保持 FP32。本步更新用解码后的 FP32 临时状态，参数更新后保存压缩状态，无常驻 FP32 影子状态。全零块保存正零与 scale=1；溢出或非有限状态报错，不静默饱和。

每个 run 输出 resolved protocol、quantization、provenance、state inventory、metrics、summary、status 和 checkpoint。恢复命令需指向原目录及 checkpoint；codec/dtype 保留，协议不一致时报错，不自动重编码。

checkpoint 先建立 CPU 快照，再存档并清理 PyTorch pickler 的循环引用，防止旧 CUDA storage 延续到后续更新。压缩 payload 在 CPU 快照和恢复时保持原 dtype；专门测试旧 storage 释放和 FP8/BF16 序列化。

```powershell
.\.venv\Scripts\python.exe scripts/smoke_adamw.py --variant A0 --steps 500 --microbatch 8 --output runs/adamw-A0-500 --resume runs/adamw-A0-500/checkpoints/last.pt
```

500 步 loss/PPL 不用于判断最终收敛。这里的耗时含日志、评估和 checkpoint，不能作为方法规范中 3000+1000 步、3 次重复的正式资源探针。原生 fused AdamW 另通过 `--variant native` 支持，用于后续速度参照；本轮量化质量与 A0 reference 配对。

`runs/preflight-*` 使用随机 token 做短 GPU 冒烟检查，不能作为真实数据的 500 步结果。
