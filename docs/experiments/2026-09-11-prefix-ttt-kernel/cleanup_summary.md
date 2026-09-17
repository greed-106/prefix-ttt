# 保留优化的代码与测试整理

本阶段按用户要求整理现有实现，保留已采用的直接 Local、合并 AB 投影和无 mask FLA。性能研究暂停，本轮只检查重构是否保持现有行为与性能，不使用 CUDA Graph。

## 代码归属

| 之前 | 现在 | 实质变化 |
| --- | --- | --- |
| `prefill.features_pair(module, q, k)` | `FeatureReadout.features_prefill(q, k)` | serving 权重及其使用者归于同一类，原 BMM 与舍入顺序不变 |
| `prefill.dense_local` | `local.dense_local` | Local 的直接、打包、缓存路径归于同一模块，计算原样迁移 |
| `prefill.dense_prefix` 与 `fla_prefix` | `fla_prefix(valid=None)` 的 no-grad 路径 | 共用输入校验和后端边界，最终 FP32 状态检查仅保留一处 |

删除搬空的 `ops/prefill.py`，模型少一个 prefix 分支；相对本阶段开始的已优化版本，生产 Python 净减 34 行。没有新增配置、长度档位或形状特判。已有 B1 packed decode 分支仍保留，不把“没有新增特判”写成“没有任何形状特判”。

FP32 参数/state、非持久 BF16 serving buffer、请求级有效性判断、padding/holes/finished、分段 prefill 和 decode 接续保持原契约。训练继续走原 packed FLA；没有把推理无 mask 路径扩大到训练。

## 测试精简

常规 `tests/` 删除五个只服务实验脚本或未投产候选的文件：`test_decode_bench.py`、`test_decode_profile.py`、`test_decode_local.py`、`test_short_ops.py`、`test_mamba_benchmark.py`。其中包含本轮刚补的基准辅助测试，随后按用户的精简要求删除。实验脚本、候选实现和历史结果仍保留；删除的是常规测试入口。

另删除两个只断言 SDPA 调用次数/reshape 的结构测试和一个重复的 CPU FLA 拒绝测试。保留独立数值 oracle、梯度、精度/状态合同、缓存边界和输入污染检查。维护中的测试由 24 文件、2847 行降为 19 文件、2408 行，净减 **439 行**。已跑实验测试源码压缩归档于 `artifacts/mamba-kernel/retired-experiment-tests.tar.gz`，不建立新的常规测试目录。

## 验证

- 修改前：122 项相关 CPU 测试、84 项 GPU 算子及集成测试通过。
- 接口重构后、精简前：相同 122 项 CPU 测试通过。
- 最终精简后：完整 CPU 套件 **199 passed**，GPU 算子及集成回归 **84 passed**。
- 两长度的 prefill 加 128 步 decode，共 **258 次**逐步比较，输出均 finite，输出和全部缓存均 exact，旧 LayerState 不污染；FP32 state、计数器及请求标记符合契约。

同卡对照使用本次修改前源码快照，不以早期未优化版本作参照。固定真实 E2 层权重，双方共享 QKV/O 模块，特征参数相同；B1、640/1536 token，prefill 与连续 128 步 decode 各预热 2 次、交替采样 5×5 次。正确性逐步检查输出、全部缓存、旧 LayerState 不污染和计数器更新。prefill 计时含新建缓存，decode 前缀在计时外；所有计时为普通 eager。

| 前缀长度 | 阶段 | 重构前 | 重构后 | 中位耗时变化 |
| ---: | --- | ---: | ---: | ---: |
| 640 | prefill（ms） | 1.286816 | 1.293056 | +0.485% |
| 1536 | prefill（ms） | 1.312480 | 1.313888 | +0.107% |
| 640 | decode（ms/步） | 1.006764 | 1.007802 | +0.103% |
| 1536 | decode（ms/步） | 1.006411 | 1.006553 | +0.014% |

共 200 个计时样本，四项中位耗时变化都小于 0.5%，视为性能基本持平；各轮原始样本保留，不声称提速。四项显存统计在两臂完全相同。结果只覆盖所列完整层形状，不代表重新验证了整模型性能或 MME/POPE 质量，也不构成新的 Flash 胜出结论。本阶段无运行失败；测试脚本在提交前修正了对原地递增计数器的错误不变性要求，未改变生产计数器。

相关配置、日志、源码身份和报告见 [主账本](ledger.md)、[CPU 报告](../../../artifacts/experiments/prefix_ttt_kernel/evidence/mamba-cleanup/final-cpu.xml)、[GPU 报告](../../../artifacts/experiments/prefix_ttt_kernel/evidence/mamba-cleanup/after-gpu.xml)、[完整配对数据](../../../artifacts/experiments/prefix_ttt_kernel/evidence/mamba-cleanup/paired.json)。三项 GPU 任务均由 Supervisor/SQLite 管理，提交后相关 Python 源码哈希未变化。
