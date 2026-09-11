# Prefix-TTT

在完整 LLaVA-1.5-7B 上研究 Local-32 + token-inclusive Prefix-TTT。稳定任务书与当前状态：

- [任务书](docs/experiments/2026-09-09-prefix-ttt/plan.md)
- [主账本](docs/experiments/2026-09-10-prefix-ttt-h100/ledger.md)
- [阶段总结](docs/experiments/2026-09-10-prefix-ttt-h100/experiment_summary.md)
- [论文解读](docs/experiments/2026-09-09-prefix-ttt/paper_reading.md)
- [仓库审计](docs/experiments/2026-09-09-prefix-ttt/repo_audit.md)

## 环境与 CPU 验证

项目使用 uv，普通包来自阿里镜像，torch/torchvision 来自官方 cu128。已安装 CUDA 构建不代表本机当前有 GPU。

```bash
uv sync --locked
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --locked pytest -m 'not gpu'
uv run --locked python -m prefix_ttt.audit_environment
```

固定配置在 `configs/base.json`。数据直接使用 `data/llava-v1.5-assets-v1/`；`preference/` 始终只读。运行时 LLaVA 来自本项目维护的 `third_party/llava/`，不导入参考目录。

## 当前实现边界

已提供严格 HF→原 LLaVA 权重桥接、同步展开 metadata、标签兼容检查、reference/FLA 算子接口、真实块批量 Local-32、缓存存储/Local分段基础、混合 attention 训练路径、LoRA冻结审计、A/B loss基础、数据审计/分组清单和最小 SQLite 调度器。

CPU tiny 模型已验证前向、反传、optimizer/RNG恢复和混合缓存greedy生成；reference 仅用于小测，不用于正式训练。FLA GPU 算子验收 17 项通过；A 与 E2 两阶段已在 16×H100（两台主机）上从零训完，MME/POPE 分数与逐请求成本（prefill、TPOT、峰值显存、缓存分解）在同一次运行内采集，数字与结论见 2026-09-10 主账本与阶段总结。混合模型仅支持cached greedy单beam，不支持的生成模式明确报错；图像只编码一次，decode只处理新token。

BF16权重桥接只转换参数，保留原FP32 RoPE频率buffer。使用 `load_checkpoint(..., dtype=torch.bfloat16)` 后仅 `.to(device)`；不要再次整体 `.bfloat16()` 或 `.to(dtype=...)`，以免改变原始RoPE精度。

## 挂卡重启后

先阅读同一本主账本，检查已有队列/结果和工作树，确认实际物理GPU ID及空闲状态，再验证cu128环境。不要重复已经通过的GPU smoke或直接启动完整训练。SQLite 调度器只做单机 GPU 分配，不支持跨主机；本阶段的两机训练用 `docs/experiments/2026-09-10-prefix-ttt-h100/scripts/run_multinode.sh` 手动会合，仓库内不再保留队列与服务配置。

2026-09-10 起，本机与 `h100-1` 通过宿主机 NFS 共享 `/data/shared/weights/prefix-ttt`（训练产物与评测结果都放在这里），跨机任务用同一份固定清单。
