# Prefix-TTT

在完整 LLaVA-1.5-7B 上研究 Local-32 + token-inclusive Prefix-TTT。稳定任务书与当前状态：

- [任务书](docs/experiments/2026-09-09-prefix-ttt/plan.md)
- [主账本](docs/experiments/2026-09-10-prefix-ttt-h100/ledger.md)
- [阶段总结](docs/experiments/2026-09-10-prefix-ttt-h100/experiment_summary.md)
- [论文解读](docs/experiments/2026-09-09-prefix-ttt/paper_reading.md)
- [仓库审计](docs/experiments/2026-09-09-prefix-ttt/repo_audit.md)

## 目录

- `src/prefix_ttt/`：模型、算子、训练与运行时实现。
- `scripts/experiments/`：按项目组织的实验启动、测量和分析工具。
- `tests/`：生产行为回归测试。
- `docs/experiments/`：实验账本、论文解读、总结、简要结果及 `images/` 图表。
- `artifacts/experiments/`：本地原始结果、日志、trace、源码快照和测试报告；既有调度记录仍保留原 artifacts 路径。

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

先阅读同一本主账本，检查已有队列/结果和工作树，确认实际物理GPU ID及空闲状态，再验证cu128环境。不要重复已经通过的GPU smoke或直接启动完整训练。SQLite 调度器只做单机 GPU 分配，不支持跨主机；两机训练入口为 `scripts/experiments/prefix_ttt_h100/run_multinode.sh`，同项目的评测与分析脚本在相邻目录。Kernel 优化工具位于 `scripts/experiments/prefix_ttt_kernel/`，全量微调工具位于 `scripts/experiments/prefix_ttt_fullft/`。

2026-09-10 起，本机与 `h100-1` 通过宿主机 NFS 共享 `/data/shared/weights/prefix-ttt`（训练产物与评测结果都放在这里），跨机任务用同一份固定清单。

## 多机环境配置（主机名 + NFS 共享）

三台机器（`h100-3` = 本机/导出方、`h100-1`、`h100-2`）共用同一套脚本，**每个节点的配置都可以复现，不需要手工改文件**。
脚本位于 `scripts/experiments/prefix_ttt_h100/`，均可重复执行（幂等）：

| 脚本 | 作用 | 运行位置 |
| --- | --- | --- |
| `cluster-hosts.sh` | 把三个节点的「名字 → 地址」写进 `/etc/hosts`（唯一来源，新增节点只改这里）；`HOSTS_FILE=<文件>` 可对副本干跑 | 每个节点，需 root |
| `nfs-server-export.sh` | 把 `/data/shared/weights/prefix-ttt` 以 `rw,sync,no_subtree_check,all_squash,anonuid=1001,anongid=1002` 导出给 `$CLIENTS`（默认 `h100-1`） | 导出方（本机），需 root |
| `nfs-client-setup.sh` | 客户端把同一路径挂到**同名路径**；若本机就是导出方则识别为本地目录；`PERSIST=1`（默认）额外写 `/etc/fstab`（`_netdev`），避免重启后静默写进本地盘；最后写 `.probe-$(hostname)` 验证可写 | 每个客户端，需 root |

新增/重建一个节点的完整流程：

```bash
# 1) 代码：用 git，不要手工同步文件
git clone git@github.com:greed-106/prefix-ttt.git /data/mjyang/code/llm/prefix-ttt
git -C /data/mjyang/code/llm/prefix-ttt checkout <branch>

# 2) 主机名映射（每个节点各跑一次）
sudo bash scripts/experiments/prefix_ttt_h100/cluster-hosts.sh

# 3) 导出方：把新节点加入导出列表并生效
sudo CLIENTS="h100-1 h100-2" bash scripts/experiments/prefix_ttt_h100/nfs-server-export.sh

# 4) 新节点：挂载共享目录
sudo bash scripts/experiments/prefix_ttt_h100/nfs-client-setup.sh
findmnt -T /data/shared/weights/prefix-ttt        # 应显示来自 h100-3:/data/shared/...
```

已按此流程配置完成：`h100-1`、`h100-2` 均挂在同一路径，导出列表包含 `h100-1`、`h100-2`；`cluster-hosts.sh`
写入的 `h100-2 = 172.18.1.73` 与实际解析一致。

数据与环境的重建方法（**代码之外**才需要搬运）：模型权重、数据集与 pyproject 里的依赖都在
`data/llava-v1.5-assets-v1/`（约 75 GB）与项目 `.venv`（约 7.5 GB，绝对路径固定所以可整目录拷贝）；
`artifacts/cpu/fixed_manifest.json` 是训练入口的必需清单，新节点也要有。上述内容走
`172.18.1.x` LAN 的 `rsync`/`tar | ssh`，实测约 **500 MB/s**（约 79 GB / 2.5 分钟）。
**RDMA 目前不可用**：本机 IPoIB 接口 `ibp*` 为 down，且两机之间存储网 `bond_stor`（100.80.2.x）不通；
要用需两端 root 起 IPoIB 或配 NFS-over-RDMA，对一次性拷贝不值得。

注意：`uv` 必须可用（`uv run --locked ...` 是唯一入口）。本机在 `/data/mjyang/.local/bin/uv`，h100-1 用
系统自带的 `/snap/bin/uv`，h100-2 由本项目拷贝二进制并在 `~/.local/bin/uv` 建软链；正式任务不依赖
`.bashrc`，启动脚本与 Supervisor 配置都显式设置 `PATH`。
