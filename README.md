# Prefix-TTT

在完整 LLaVA-1.5-7B 上研究 Local-32 + token-inclusive Prefix-TTT。稳定任务书与当前状态：

- [任务书](docs/experiments/2026-09-09-prefix-ttt/plan.md)
- [主账本](docs/experiments/2026-09-09-prefix-ttt/ledger.md)
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

CPU tiny 模型已验证前向、反传、optimizer/RNG恢复和混合缓存greedy生成；reference 仅用于小测，不用于正式训练。FLA GPU 数值与性能、完整7B回归、正式A/B runner和benchmark输入适配尚需继续实施或验证。混合模型仅支持cached greedy单beam，不支持的生成模式明确报错；图像只编码一次，decode只处理新token。

BF16权重桥接只转换参数，保留原FP32 RoPE频率buffer。使用 `load_checkpoint(..., dtype=torch.bfloat16)` 后仅 `.to(device)`；不要再次整体 `.bfloat16()` 或 `.to(dtype=...)`，以免改变原始RoPE精度。

## 挂卡重启后

先阅读同一本主账本，检查已有队列/结果和工作树，确认实际物理GPU ID及空闲状态，再验证cu128环境。不要重复已经通过的GPU smoke或直接启动完整训练。当前容器经用户批准由pixi Supervisor托管SQLite调度器，配置在 `configs/supervisor/`，操作及恢复边界见[Supervisor说明](docs/experiments/2026-09-09-prefix-ttt/supervisor.md)。旧systemd样例保留，不是当前前置要求。

本轮没有提交训练或质量benchmark任务；具体CPU审计完成数以主账本最终记录为准。
