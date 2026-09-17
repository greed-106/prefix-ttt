# 普通 eager 热点证据与代码可读性审查

2026-09-17，用户明确要求后续不使用CUDA Graph，以实际测量决定优化方向，并避免生产代码变成特判和候选实现的集合。本次据此审计了既有性能证据与代码增量，补充真实E2普通eager时间线。历史Graph实验保留为历史记录，不再作为后续实现或验收方向。

## 1. 原推荐方向有热点依据，但优先级证据不够

之前提出“AB特征GEMM与SiLU/RMS融合”，依据来自[B1/1536生产单层profile](../../../artifacts/experiments/prefix_ttt_kernel/evidence/mamba-final/profile.json)。它使用真实E2层权重、固定随机激活，执行普通eager生产路径，得到如下GPU kernel时间。

| 工作 | GPU kernel时间 |
|---|---:|
| 共同的Q/K/V/O投影 | 268.833微秒 |
| 两次SiLU/RMS | 83.970微秒 |
| FLA状态与输出 | 50.208微秒 |
| Local Flash | 45.920微秒 |
| readout | 43.776微秒 |
| 两次AB投影 | 28.319微秒 |
| 全部kernel合计 | 675.553微秒 |

AB投影与SiLU/RMS占kernel合计约16.6%，所以它是有实测依据的候选。此前表述为“最值得优先实现”过强：尚未测过融合实现，没有证据证明它比减少主机调用、张量处理或缓存开销更值得投入。1536下的96 MiB仅是按中间张量尺寸计算的逻辑写读量，没有硬件计数器证明全部流量进入HBM，更不能直接转换为端到端时间收益。

已有消融其实提供了不同的优先级线索。B1/1536旧TTT为2.234704 ms，仅合并AB投影为2.186048 ms，降低2.18%；仅采用直接Local与无mask FLA为1.379312 ms，降低38.28%；两者结合为1.320672 ms。此前大收益主要来自取消不必要的数据处理，单独合并GEMM的收益小得多。不同消融存在交互，不能把降幅简单相加。

另一个反例来自已经运行过的1warp实验。以下只重新分析其中`eager-*`样本，没有重新运行Graph；该实验是已知全有效的optimized路径，不含生产有效性主机判定。

| B/T | 原配置 | 1warp | 延迟降低 | 五轮都改善 |
|---|---:|---:|---:|---|
| 1/640 | 1.308912 ms | 1.306336 ms | 0.197% | 否 |
| 1/1536 | 1.314816 ms | 1.313104 ms | 0.130% | 否 |
| 4/1536 | 2.688352 ms | 2.541344 ms | 5.468% | 是 |
| 4/2048 | 3.436368 ms | 3.244960 ms | 5.570% | 是 |

1warp同时影响feature与readout，不能单独归因于feature。它在目标B1短序列上没有稳定实质收益，因此当前没有理由为这两个形状增加生产分派。[证据复核JSON](../../../artifacts/experiments/prefix_ttt_kernel/evidence/mamba-eager-audit/review.json)保存了来源哈希、全部eager配置和分轮数字。

## 2. 需要补齐的是实际请求中的归因

已有profile仅包含B1/1536和32768的单层各一次调用，启用了shape和memory记录。其CPU表包含约1.37 ms的`Activity Buffer Request`采集成本，不能把这个条目当成模型热点，也不能把不同运行的kernel合计从未profile的层延迟中相减，声称剩下的是CPU耗时。CPU父范围、GPU注释范围与实际kernel也不能重复累计。

生产cache.begin中的全有效判定确实出现在1536时间线：`aten::item`约27.912微秒、1字节DtoH约2.336微秒、`cudaStreamSynchronize`约7.770微秒。这些事件存在嵌套，不能相加；判定每请求一次，不是每层一次。相应optimized/production消融约有28微秒差距，但单点证据不足以解释整模型的主要剩余开销。

本次补测采用真实E2文本请求，B1、640/1536、fresh cache、普通eager。关闭shape、memory、stack记录，在profiler中先预热两步，再记录三个active调用；总prefill、decoder层、cache begin/finish和TTT四阶段分开标记。原始时间线用于观察调用、同步和设备执行关联，绝对延迟仍以独立未profile的基准为准。三个active调用可检查采集内重复性，但不能消除profiler和标记造成的扰动。

补测已完成。两个长度各采集三个active请求，均为每请求1720个kernel；共10320个kernel全部通过correlation关联到CPU runtime，再归到最内层阶段，没有未关联项或请求外kernel。以下是每请求23个TTT层的对应阶段累计时间，取三次请求的中位数，单位ms。CPU范围与GPU执行可以重叠，两个时间列不能相加。

| 长度 | 阶段 | GPU kernel累计时间 | CPU调用范围时间（inclusive） |
|---:|---|---:|---:|
| 640 | AB投影与SiLU/RMS | 1.198 | 4.930 |
| 640 | Local与尾块缓存准备 | 0.640 | 4.092 |
| 640 | prefix（含ETA及FLA调用） | 0.573 | 7.601 |
| 640 | readout | 0.444 | 1.860 |
| 1536 | AB投影与SiLU/RMS | 2.597 | 5.103 |
| 1536 | Local与尾块缓存准备 | 1.216 | 4.077 |
| 1536 | prefix（含ETA及FLA调用） | 1.404 | 7.462 |
| 1536 | readout | 1.021 | 1.842 |

特征映射仍是这四项中最大的GPU工作，因此原候选有依据；但prefix是CPU调用范围最大的阶段，其GPU工作反而较小。native `ChunkSimpleGLAFunction` 的CPU self在23层累计约5.49/5.43 ms每请求，prefix内CUDA launch API区间并集仅约0.299/0.294 ms。不能把差异笼统说成CUDA launch开销，更不能假定7.5 ms或5.4 ms可以全部省掉。Python包装、Triton调用准备、入口检查和采集扰动都需要进一步区分。

这一证据将下一步调整为**先对FLA调用链的主机工作做隔离验证**：保持现有计算和kernel，检查是否能精简调用链，再在没有profiler的完整层和真实模型中测量净收益。AB融合保留为GPU侧候选，当前不预先投入复杂的新GEMM实现。每请求两次内部stream同步各约8微秒，这份数据也不足以支持继续为它们新增有效性特判。

原始三次分布、CPU/GPU范围以及来源哈希见[profile结果](../../../artifacts/experiments/prefix_ttt_kernel/evidence/mamba-eager-profile/results.json)和[阶段归因](../../../artifacts/experiments/prefix_ttt_kernel/evidence/mamba-eager-profile/attribution.json)；[归因脚本](../../../artifacts/experiments/prefix_ttt_kernel/evidence/mamba-eager-profile/attribution.py)保证每个kernel只归属一次，decoder/整请求统计是另一个包含子阶段的视图，不能再加到阶段上。两种长度的logits均finite。原始Chrome trace保留在`artifacts/eager-profile/mamba-eager-profile/`，摘要记录其SHA256。

## 3. 代码复杂度的顾虑成立

Mamba优化相对`eb3d2c6`在生产中新增116行、删除2行，净增114行，含空行和注释；其中97行来自新`ops/prefill.py`。新增三个算子函数、一个请求级瞬时布尔标记，以及每层2 MiB的AB推理副本。这个计数是本次可读性整理之前的审计值。下述重复边界已在后续[代码整理](cleanup_summary.md)中合并，原审计记录保留以说明修改依据。

当前生产没有Graph、1warp或按640/1536等长度分档。已有的精确`(1,1,32,128)` decode packed路径来自更早阶段，也应计入未来的维护审视。不能只审查新文件，忽略既有特判的累积。

主要问题是职责分散和重复，而非单看代码行数：

- `generation.py`判定全有效初始请求，`hybrid.py`再判定dtype/头维/backend，`prefill.py`重复检查部分资格条件。理解一条路径需要跨三个文件。
- `dense_prefix`与现有`fla_prefix`重复CUDA、shape、tile和FP32状态契约校验，实际差别主要是是否创建和应用mask。
- 一个dense-prefill标记同时控制Local、AB投影和FLA，而AB投影本身不依赖Local的全有效语义。继续向这个模式添加候选会扩大耦合。
- Local三路选择曾写成嵌套条件表达式。本次已展开成明确的`if/elif/else`，条件与调用不变；模块说明中的“candidates”也改为准确的已采用算子描述。

有些分支必须保留其语义。padding、空洞和finished行的Local分块按有效token逻辑位置计算；分段prefill要接续旧尾块；decode与分块prefill计算不同；训练要保留梯度；FP32 master/state与BF16副本有数值约定。简化应减少重复表达这些契约的地方，不能把支持这些输入的逻辑直接删掉。

后续代码收敛应围绕已有算子边界进行。例如，让已有FLA入口明确处理无mask输入，有机会移除重复的dense包装；但它涉及梯度路径、空输入及冻结基准依赖，需要独立验证。本次没有顺手做这项语义性重构，也没有新建策略类或分派框架。

## 4. 后续采用标准

先由真实普通eager测量列出候选，注明输入范围、CPU/GPU证据和不确定性。再隔离一个变更做消融，检查完整层及实际模型请求中的稳定收益和正确性。性能报告必须说明收益能否覆盖目标B1短序列，不能用B4或核心算子结果替代。

生产实现应替换或合并原有算子路径，保持数学计算流程容易辨认。warp、tile或特定测试长度不应不断变成顶层forward的新分支。收益不稳定或很小的候选留在实验记录；若仍需要复杂的并行路径，就必须明确说明获得了什么可复现的收益，以及承担了什么维护成本。

本次的决定是：停止使用Graph；撤回AB融合的既定优先级；先补实际热点证据；进行小范围可读性整理并验证，而后按测量决定一个候选。没有删除历史实验，也没有以“整理”为由取消已有mask、缓存或训练能力。

本次六项GPU分派/缓存集成测试通过，覆盖dense、padding、空洞、空行、finished，以及分段prefill接续decode。两个新GPU作业均成功，仅使用物理GPU7；配置、源码快照、日志和状态见[可读性回归](../../../artifacts/experiments/prefix_ttt_kernel/evidence/mamba-eager-readability/manifest.json)、[实际profile配置](../../../artifacts/experiments/prefix_ttt_kernel/evidence/mamba-eager-profile/manifest.json)与[队列状态](../../../artifacts/experiments/prefix_ttt_kernel/evidence/mamba-eager-profile/queue-status.json)。本轮没有提交或推送Git。
