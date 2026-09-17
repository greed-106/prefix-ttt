# Prefix-TTT kernel 实验主账本

目录迁移说明（2026-09-17）：实验工具现位于仓库根目录下的 `scripts/experiments/prefix_ttt_kernel/`；原始证据及源码快照现位于 `artifacts/experiments/prefix_ttt_kernel/evidence/`；文档和 `images/` 保留本目录。下文旧 `scripts/`、`evidence/` 路径均为当时运行记录，完整映射与验证见末尾「纠正文档与代码目录职责」。

本项目启动于 2026-09-11，稳定目录不变。2026-09-17 续接 Mamba 启发的 prefill 优化。
旧阶段事实保留在 [原任务书](plan.md) 与 [原总结](experiment_summary.md)，本文件不重建缺失的历史逐条记录。

**当前状态：2026-09-17 代码、测试与目录整理完成。** 已保留现有优化并合并重复边界，重构前后普通 eager 完整层性能基本持平；常规测试净减 439 行。实验工具、原始结果与源码快照已移出文档目录，路径/哈希/CPU 回归验证通过。全部队列终态，Supervisor 关闭；本次目录迁移未运行 GPU。decode 超过静态 KV Flash 的目标尚未达成，既有测量见 [decode 阶段总结](decode_summary.md)、[后续路线](decode_plan.md)及[代码整理结果](cleanup_summary.md)。

## 2026-09-17：续接启动

- 目标：保持 Prefix-TTT 数学定义，在固定输入完整注意力替换层上超过 PyTorch 2.7.1 Flash SDPA；报告 512–32768 token 曲线及稳定领先区间，不预先承诺结果。
- 授权：用户要求实施方案；主攻推理 prefill，允许经验证的归约浮点误差，仅物理 GPU 7。未授权 commit/push。
- 开工核对：已读 AGENTS.md、原计划/总结及成功证据；HEAD `eb3d2c61054686b596b2e407767f546ac99e9525`；工作树仅用户新上传 Mamba PDF 未跟踪。
- 资源：GPU 0–6 有其他作业，GPU 7 为 1 MiB/0%；未发现存活 Supervisor、项目消费者或已有 SQLite。旧 packed 的 MME/POPE 已成功，禁止将其作为新任务重复提交。
- 当前代码支持：FLA 分块前缀、通用 mask/缓存、单 token packed feature；尚无本轮全有效 prefill 专用路径。训练与 decode 不属于新内核实现范围。
- 关键决策：Mamba 的融合、减少状态 HBM 往返和分块思想可迁移；其小对角状态 scan 不直接套用矩阵状态。现有 FLA 是本轮基线，不从零复制其算法。
- 分工：主线程负责集成/调度/账本/验收；独立 agent 分别负责中文阅读教学、实验基准、新 prefill 算子与测试。agent 不自行启动 GPU。
- 运行项：暂无 GPU 任务；下一步建立基准和 Supervisor 独立队列，先正确性及 profile，再消融和正式曲线。
- 失败项：无。
- 产物：本目录新阶段使用 `evidence/mamba-*`、`scripts/mamba_*`；运行数据库与原始日志放 `artifacts/mamba-kernel/`。旧阶段文件不覆盖。

### 环境与第一项准备变更

- 已启动独立 Supervisor，配置归档 `evidence/mamba-environment/`；正式实验使用每阶段不可变 manifest/SQLite，不更改调度器实现。
- `FeatureReadout.prepare_inference` 新增不进入 state_dict 的 A/B 输出维拼接 BF16 副本，供两次 prefill GEMM 候选使用；FP32 master 与旧 decode 路径不变。是否最终保留由消融决定。
- 正在运行 CPU 基线（包含现有调度器与真实 Supervisor 生命周期测试），日志 `evidence/mamba-environment/cpu-baseline.log`；CUDA 对该进程不可见。

### 基线检查结果

- CPU 基线：175 passed、17 deselected，19.34 秒；包含 Supervisor/调度器已有验证。
- 原 FLA GPU 接口：13 passed（日志及 JUnit 已保留），输出/FP32 状态、分段与梯度通过。此项由独立 Supervisor 消费者在 GPU 7 完成，启动终端已退出。
- 当前发现：Local-32 对全有效 prefill 仍进行 nonzero/cumsum/index_copy 和缓存拼接；这构成直接分块候选的具体动机。是否是主要耗时仍待 profile。

### prefill 候选算子实现

- 新增 `ops/prefill.py`：两次 A/B 合并投影、全有效 Local-32 直接分块和标准缓存输出、跳过 mask 的锁定 FLA 调用；ETA 仍仅在原后端应用一次。
- 新增专项测试覆盖 FP64 local、非连续/尾块、缓存接续、BF16 features、FLA 四个 tile 与 FP64 oracle。实现尚未接入生产 forward，GPU 验证待调度。

### 文档与可复现测量实现

- 新增 [Mamba 阅读](mamba_reading.md)、[分块与硬件教学](teaching_mamba_prefix.md)，已检查公式定界符和相对链接；论文结论与本轮待验证候选分开。
- 新增 `scripts/mamba_bench.py`，包括原提交冻结 baseline、强制 Flash SDPA、相同投影/固定输入、核心与完整层、原始分轮计时、显存和独立 profile。先 smoke 验证后再正式矩阵。
- 新算子 CPU 专项：27 passed；GPU 专项正由 `ops` 队列执行。

### 生产候选接线（待验收，不代表已采用）

- 首次 cached prefill 在 `cache.begin` 执行一次全有效判定，存为私有瞬时标记；不在每层同步。padding、空洞、finished 行、分段调用不进入直接 Local 路径。
- BF16、头维 128、FLA、空初始缓存条件下，hybrid 调用合并投影/直接 Local/无 mask FLA；其他输入及 autograd 保持原路径。cache.finish 清理瞬时标记，缓存张量格式不变。
- 这包含一次明确 GPU→CPU 判定成本，不能宣称整个 pipeline 无同步；最终基准须测真实生产分派而不只测已知全有效实验分支。

- 算子 GPU/CPU 专项已完成：88 passed（其中 61 项 GPU），119.25 秒；四种 FLA tile 与 FP64 oracle 达到原 2% 输出/1% 状态误差界，未放宽容差。
- 新增生产接线测试：全有效、padding、空洞、空行、finished、分段 prefill 接续 decode、旧 state 不污染；待队列执行。

### 生产接线与真实权重准备验证

- GPU 生产接线测试 6 passed，19.78 秒；有效性分派与缓存语义通过。
- 新增 `scripts/mamba_weights.py`，已在 CPU 导出 E2 layer 1 的七项权重至 `artifacts/mamba-kernel/weights/layer1.pt`；来源与张量哈希在同名 `.pt.json`。不是新训练或模型权重下载。
- 合并顺序等同现有推理：HF 基底转 BF16，FP32 LoRA B@A 乘 2、转 BF16 后加到 BF16 基底；TTT master 保持 FP32。源 checkpoint 为完整 E2/B，5182 步。
- 基准审查修正项：Flash 后端选择上下文移出计时；原 baseline 的缓存类和依赖需要冻结/一致性验证；真实生产分派单列 production，防止已知全有效实验分支遗漏判定开销。

### Smoke 与审计结果

- `smoke` 成功：B1、512/1536 token，所有候选的层输出/最终状态与原提交逐位一致。仅 1 轮×2 次计时，不能作为正式性能结论。
- Smoke CUDA Events 中位数：baseline 1.889/2.206 ms，production 1.297/1.339 ms，MHA 0.353/0.586 ms。短序列仍未超过 MHA。
- Profile 证实 Local 打包和缓存重建开销大；候选减少了相关索引。Smoke 的 kernel 计数混入 profiler GPU user annotations，已在脚本修正为 Chrome `cat=kernel`，正式计数以后续结果为准，旧 JSON 不追改。
- 新增 `scripts/mamba_report.py`（表格/图片归档）与 `scripts/mamba_quality_compare.py`（逐回答/身份/官方分数对照）；均通过 CPU fixture 验证。
- 当前正式校准：真实 E2 权重全长度/批次正确性，以及四种 FLA tile 的同卡性能消融。脚本输出保存源码哈希，原始 Chrome trace 改存 artifacts，本目录保存汇总引用。

### 校准正确性完成、计量脚本缺陷修复

- 真实 E2 层权重在全部 18 个 B/L 配置（B1/B4，512–32768）通过；production 与原版输出及最终状态逐位一致。
- `tile-ablation` 在第一个 shape 的显存记账阶段失败，未保存任何完整性能 shape：`AttributeError: 'function' object has no attribute 'tensor_bytes'`。原因是 Tensor 自身有 storage 方法，被误识别为 HybridCache。
- 有依据的一次修复：cache_bytes 先识别 Tensor，再识别 HybridCache 属性。保留原失败日志/JSON，另建修复后 tile 作业；不重跑成功的真实层正确性任务。

- 修复验证：CPU tensor/tuple/DynamicCache/HybridCache 四类记账通过，并新增持久回归测试 `test_mamba_benchmark.py`。报告生成器补齐临时显存、持久缓存、正确性和实际 kernel 数汇总；每层新增 prepared buffer 为 2 MiB，23 层总计 46 MiB。

### 四种块大小消融完成

- 修复后 `tile-fixed` 成功，8 个 B/L 配置、每项 20 次预热及 5×20 次计时；原失败作业未重试，使用新不可变作业 ID。
- 32768 token：无 mask tile64 为 B1 1.173 ms、B4 4.738 ms；tile128 为 1.104/4.326 ms；Flash SDPA 核心为 24.198/99.886 ms。这只是核心比较，不含 QKV、特征、Local、门控与 O。
- tile16/32 在长序列更慢，边界状态写读更多；tile128 在短序列不占优。下一步完整层消融后再决定是否采用128，当前生产候选仍为64。

- 基准新增显式 `tile128` 实验候选：仅改变 optimized 的 FLA tile，便于同进程逐项消融；production 未改变。默认 CLI 比较改为 baseline/production/MHA，防止默认测量遗漏生产判定成本。

### 完整层消融与采用决定

- 203 项 CPU 回归通过、84 项 GPU 标记项排除，20.30 秒；含最终生产接线、基准记账回归及原梯度/调度器/Supervisor测试。
- B1/T32768 完整层消融：baseline 21.580 ms、production 12.232 ms、MHA 32.795 ms，已观察到完整层 2.68 倍于指定 Flash SDPA。短长度尚未领先，不泛化至所有长度。
- tile128 在完整层仅约 ±0.3%（已完成配置），并带来约 0.17% 相对 L2 输出误差；不满足有意义独立收益，**不采用128生产分派**，保留既有 tile64。其工程支持与实验候选没有删除。
- 采用候选：两次 AB 合并 GEMM；全有效直接 Local-32+缓存；无 mask FLA；每请求一次真实有效性判定。生产 public API/checkpoint/训练/decode 算法不变。
- 正在准备最终真实 E2 单层曲线、独立随机权重四层、真实权重 decode、正式 profile 与新版本 MME/POPE；所有任务仍仅GPU7串行。

- 完整层消融已全部成功；最后 B4/T32768 的 optimized64→128 为 49.318→48.885 ms（0.9%），仍不足以采用。该配置 production 49.596 ms、MHA 131.687 ms。

### 最终单层曲线完成

- `final/curve` 的18个配置完成。B1 T16384/T32768：production 6.299/12.223 ms，Flash 9.739/32.174 ms；B4：24.134/49.052 ms，对39.146/131.356 ms。
- 两个batch的首个**实测稳定领先采样点均为16384**；8192约1.02–1.04倍，未达到1.05验收线。跨越区间只界定为8192–16384，不声称精确交叉长度。
- 原TTT在32768也已比Flash快；本轮进一步降低其完整层延迟约43%，不能描述为本轮首次发明了长序列复杂度优势。
- 显存限制：B1/T32768 production峰值增量2583.35 MiB，高于Flash1296.28 MiB；但持久缓存仅2.53 MiB，对Flash512 MiB。需同时报告临时状态与持久缓存，不能笼统声称峰值显存更省。
- 超过项目2048 token固定训评展开协议的配置仅验证算子硬件性能，不代表模型可直接长上下文生成；底座资产max_position_embeddings为4096。

### 多层、decode、profile 与首项质量回归完成

- 四层所有6项配置完成，32K相对Flash为2.65–2.71倍；独立随机权重，不是完整Transformer端到端实验。
- Decode两项完成：旧版约1.030 ms/步，production 1.032/1.035 ms/步；无实质加速，也没有观察到有意义的性能回归。
- 正式profile：1536 kernel数119→61，32768为120→60，Flash均17；三方均观测到Flash SDPA执行（TTT用于Local分支）。
- MME成功且逐条比较通过：2374条、0回答变化、0缺失/新增/身份变化；Perception 1429.4892957182874、Cognition 278.2142857142857均不变。对照 `evidence/mamba-final/mme-comparison.json`。
- POPE正在运行；此前完成的任务不重复提交。已生成[完整测量报告](mamba_measurements.md)和4张相对引用归档图（含正式曲线/消融）；新增[阶段总结](mamba_summary.md)，其中明确标记POPE待验收，未声称全部质量通过。

### 独立最终审查

- 生产实现与分派/缓存/精度审查未发现实质缺陷；最终队列提交后源码哈希核对无变化。
- 文档数值逐格核对通过，并修正术语：导出layer1是零基decoder索引1；2048是项目训评协议，不是底座4096位置配置；正式profiler只抽查B1/1536与32768，其他MHA配置由强制后端约束。
- 当前独立GPU用例完成80项（13原FLA+61新算子+6集成）；另4项既有varlen梯度将排在质量任务之后，避免与性能/质量任务争用GPU。

### 全部质量验收通过

- 最终队列7项全部成功。POPE的9000条回答、覆盖与样本身份完全一致；Accuracy 0.8501111111111112、F1 0.8357081963220071不变。
- MME+POPE合计11374条回答无变化，四项官方主指标均相同；`quality-comparison.json` 的 exact_answers=true、score_nonregression=true。保留已验证生产候选，不需回退。
- 剩余4项变长梯度测试单独由varlen队列运行；不重复执行已成功的核心性能/质量/算子专项。

### 最终验收与归档

- 4项varlen输出/状态/梯度测试通过（81.80秒）；本轮独立GPU测试总计84项，CPU回归203项，无测试容差放宽。
- 最终源码与任务提交时SHA256一致；所有队列均为终态。成功任务没有重复提交，唯一失败为修复前tile计量脚本；全部作业分配均为物理GPU7。
- 状态/事件导出 `evidence/mamba-final/queue-status.json`，Supervisor最终状态留档；任务完成后关闭本轮创建的Supervisor管理器，GPU已释放。原SQLite、配置和日志保留，恢复时仍跳过成功任务。
- 新文档已检查数学定界符配对、旧定界符、裸LaTeX和相对链接；图表归档且视觉检查通过，git diff --check通过。
- 完成交付：[论文阅读](mamba_reading.md)、[教学推导](teaching_mamba_prefix.md)、[完整测量](mamba_measurements.md)、[阶段总结](mamba_summary.md)。旧2026-09-11总结仅增加续接链接，旧结果不覆盖。
- 下一步：本阶段无必需剩余任务。短序列/临时显存优化、训练反向与长上下文质量属于后续研究；本轮没有创建Git commit或推送。

### 队列提交：gpu-baseline

- 作业：existing-gpu-acceptance；GPU 7，单卡并发 1，无自动重试。
- 配置/源码哈希：`evidence/mamba-gpu-baseline/`；SQLite/原始日志：`artifacts/mamba-kernel/gpu-baseline/`。
- 状态：已由 Supervisor 启动；完成与失败另追加实测记录。

### 队列提交：ops

- 作业：prefill-ops；GPU 7，单卡并发 1，无自动重试。
- 配置/源码哈希：`evidence/mamba-ops/`；SQLite/原始日志：`artifacts/mamba-kernel/ops/`。
- 状态：已由 Supervisor 启动；完成与失败另追加实测记录。

### 队列提交：integration

- 作业：production-integration；GPU 7，单卡并发 1，无自动重试。
- 配置/源码哈希：`evidence/mamba-integration/`；SQLite/原始日志：`artifacts/mamba-kernel/integration/`。
- 状态：已由 Supervisor 启动；完成与失败另追加实测记录。

### 队列提交：smoke

- 作业：smoke；GPU 7，单卡并发 1，无自动重试。
- 配置/源码哈希：`evidence/mamba-smoke/`；SQLite/原始日志：`artifacts/mamba-kernel/smoke/`。
- 状态：已由 Supervisor 启动；完成与失败另追加实测记录。

### 队列提交：calibration

- 作业：real-correctness, tile-ablation；GPU 7，单卡并发 1，无自动重试。
- 配置/源码哈希：`evidence/mamba-calibration/`；SQLite/原始日志：`artifacts/mamba-kernel/calibration/`。
- 状态：已由 Supervisor 启动；完成与失败另追加实测记录。

### 队列提交：tile-fixed

- 作业：tile-ablation-fixed；GPU 7，单卡并发 1，无自动重试。
- 配置/源码哈希：`evidence/mamba-tile-fixed/`；SQLite/原始日志：`artifacts/mamba-kernel/tile-fixed/`。
- 状态：已由 Supervisor 启动；完成与失败另追加实测记录。

### 队列提交：ablation

- 作业：complete-layer-ablation；GPU 7，单卡并发 1，无自动重试。
- 配置/源码哈希：`evidence/mamba-ablation/`；SQLite/原始日志：`artifacts/mamba-kernel/ablation/`。
- 状态：已由 Supervisor 启动；完成与失败另追加实测记录。

### 队列提交：final

- 作业：curve, profile, stack, decode, e2-mme, e2-pope, quality-compare；GPU 7，单卡并发 1，无自动重试。
- 配置/源码哈希：`evidence/mamba-final/`；SQLite/原始日志：`artifacts/mamba-kernel/final/`。
- 状态：已由 Supervisor 启动；完成与失败另追加实测记录。

### 队列提交：varlen

- 作业：varlen-gradients；GPU 7，单卡并发 1，无自动重试。
- 配置/源码哈希：`evidence/mamba-varlen/`；SQLite/原始日志：`artifacts/mamba-kernel/varlen/`。
- 状态：已由 Supervisor 启动；完成与失败另追加实测记录。

## 2026-09-17：历史对齐与短序列续研

- 用户请求：确认相对历史数据是否更快，研究短序列也超过Flash的路径。沿用同目录、同数学约束与物理GPU7授权。
- 开工复核：AGENTS/账本/工作树已读；旧9个队列均终态（唯一失败为已记录的计量缺陷），无消费者/管理器存活，GPU7为1MiB/0%，不重复提交旧成功任务。
- 已有单层证据：640为1.891→1.305ms，1536为2.201→1.324ms；分别降31%/40%。历史58.8/81.7ms是整模型，不直接相减或外推本轮整模型加速。
- 本轮范围：先补相同协议整模型old-packed/current配对；再对双方均采用CUDA Graph的完整层进行诊断，分离发射成本与GPU执行负担；试验逐元素尾部1warp替代4warps。暂不改生产、数学方法、状态精度或baseline后端。
- 新增实验 `scripts/short_ops.py` 与 `tests/test_short_ops.py`：只复用旧Triton kernel、归约与舍入，仅改变launch的warp数；4项CPU测试通过，16项GPU逐位一致测试待统一队列。
- `scripts/short_graph.py`由独立agent实现；主线程负责真实整模型配对、统一调度与结论。不把Graph TTT对eager MHA，也不将已知全有效Graph实验冒充已完成生产Graph服务。

### 短序列实验代码就绪

- 新增 `scripts/short_graph.py`：TTT/Flash双方分别eager和Graph计时，覆盖换输入、重复replay和恢复输入的输出/全部缓存检查。CPU编译、CLI与比较fixture通过；不涉及生产Graph接线。
- 已重启本轮Supervisor管理器，核对9个历史队列全部终态；仅提交新的1warp专项和Graph smoke。整模型历史对齐脚本由独立agent实现，统一调度，避免GPU争用。

### 队列提交：short-smoke

- 作业：warp1-gpu-tests, graph-smoke；GPU 7，单卡并发 1，无自动重试。
- 配置/源码哈希：`evidence/mamba-short-smoke/`；SQLite/原始日志：`artifacts/mamba-kernel/short-smoke/`。
- 状态：已由 Supervisor 启动；完成与失败另追加实测记录。

### 短序列专项与Graph smoke通过

- `short-smoke`两项成功：16项GPU逐位一致测试通过（BF16/FP32、尾块/stride/无效行）；Graph的三种方案均通过换输入、重放、恢复输入及全缓存校验。
- B1/T512 smoke观察：Graph optimized约0.327ms、1warp约0.316ms、Graph Flash约0.163ms；这是1轮×2次预检，不能作为正式收益。启动开销明显，但双方Graph后TTT仍较慢。
- 下一步：固定源码执行B1/B4、128/256/512/640/1024/1536/2048的公平Graph正式测量（20预热，5×20采样）；整模型对齐脚本同步准备。生产代码不变。

### 队列提交：short-graph

- 作业：short-graph-curve；GPU 7，单卡并发 1，无自动重试。
- 配置/源码哈希：`evidence/mamba-short-graph/`；SQLite/原始日志：`artifacts/mamba-kernel/short-graph/`。
- 状态：已由 Supervisor 启动；完成与失败另追加实测记录。

### 公平Graph正式测量完成，历史整模型配对就绪

- `short-graph`14个配置成功（B1/B4，128–2048），每项20预热及5×20采样，共8400条。728个浮点字段、560个整数/布尔字段比较全equal；包含换输入与重复replay，无缓存累积。
- B1/640 Graph+1warp 0.361616ms，对Graph Flash 0.196112ms；B1/1536为0.666992，对0.425840ms。所有短长度均未超过同模式Flash；1warp的Graph独立加速约1.020–1.064倍。
- 新增并CPU检查 `scripts/short_history.py`，复用原request/summarize，冻结eb3d2c6的四个方法，同一个真实E2实例配对old-packed/current；保留当前cache构造及class identity，避免super闭包失效。每轮保留完整采样logits差异与重复漂移，资产身份核对已有manifest；下一任务补整模型640/1536、128步、两预热三轮及同期E0。

### 队列提交：short-history

- 作业：whole-model-history；GPU 7，单卡并发 1，无自动重试。
- 配置/源码哈希：`evidence/mamba-short-history/`；SQLite/原始日志：`artifacts/mamba-kernel/short-history/`。
- 状态：已由 Supervisor 启动；完成与失败另追加实测记录。

### 短序列结果审计与制图

- 独立只读审计确认14点8400样本齐全、全部输出/缓存比较exact、源码哈希一致；双方共用QKV/O权重，Graph含设备缓存初始化。固定全有效且长度均为32倍数，不外推动态形状、padding、并发请求或生产缓存生命周期。
- 新增 `scripts/short_plot.py`，直接读取成功证据并写入本目录 `images/short_graph.png`，不产生新的GPU实验。
- 下一候选预算：特征AB投影+SiLU/RMS epilogue融合可在B1/1536去掉96MiB中间张量写读；改变cuBLAS实现须单独验收归约误差。现有FLA输出每CTA只含64个V通道，128维RMS不能简单附加，优先级低于特征融合。

### 历史整模型配对完成

- `short-history/whole-model-history`成功，同一真实E2、640/1536前缀、128 teacher-forced步、两次预热、三轮交替。三轮每个长度全部129位置×32064词表logits逐位一致，maxabs/relL2均0，输出finite。
- 整模型prefill：640旧packed重测53.999454→当前33.390816ms；1536为79.082558→49.674271ms，降38.2%/37.2%。历史58.800/81.738ms同时保留，但不把跨运行全部差异归因本轮代码。
- 同期E0为23.346432/47.957153ms，当前E2尚未超过E0。TPOT旧/新为27.279776/27.289984与27.380383/27.334801ms，基本不变；不能把历史32ms到本次27ms声称为prefill优化带来的decode收益。
- 所有本轮GPU任务终态；下一步完成研究总结、图表/证据归档及文档审计，不重复成功实验。

### 短序列研究总结成稿

- 新增 `short_sequence_summary.md`，区分历史整语言模型、同进程配对、完整注意力层、Graph实验；记录两个短序列候选的成功与未达目标原因，并给出下一项特征GEMM尾部融合的流量预算、精度门槛与未执行状态。
- Mamba阶段总结新增续研链接；图表视觉检查通过。注意真实整模型logits词表维度32064；文本forward不含视觉编码器/图像投影。
- 当前全部四项新GPU任务成功；正在归档日志、队列状态及检查新增文档链接/公式，生产代码本阶段没有新增变更。

### 短序列阶段验收与关闭

- 四项新GPU作业全部成功，GPU7已释放（1MiB/0%），无任务运行或等待；日志复制至各自evidence，状态与事件导出 `evidence/mamba-short-history/queue-status.json`，Supervisor状态已保存并关闭本次管理器。
- 运行源码哈希核对一致；新增脚本编译、git diff --check、70个单层表格数字逐格对照、文档数学定界符/本地链接检查通过，图已视觉检查。
- 整模型基线由独立agent只读审查：冻结方法覆盖热路径差异，42份源码hash一致；6次新旧logits对比和12项真正跨轮重复均exact。
- 完成交付：[历史性能与短序列研究总结](short_sequence_summary.md)、Graph曲线图片、完整配置/样本/测试与队列日志。当前生产继续沿用上一阶段已验收实现；本阶段未接入生产Graph或1warp，没有声称实现短序列领先。
- 下一步研究建议：先实现AB特征GEMM与SiLU/RMS尾部融合，逐级验证数值与整模型质量，再决定生产采用。本阶段必需工作已完成；没有Git commit或push。

- 最后独立文档审查通过：14行Graph表、历史整模型数值、40/96MiB流量预算、精度与Graph适用边界、17个相对链接均一致，无实质问题。

## 2026-09-17：普通eager证据与代码可读性审查

- 用户新约束：后续一律不使用CUDA Graph；优化方向必须来自实际测量，关注冗余、特判与可读性。已写入AGENTS.md；历史Graph代码/结果保留，但不再运行或作为新优化验收方式，没有声称已删除。
- 开工核对：AGENTS、账本、工作树及12个SQLite队列已查；队列均终态，唯一失败为原计量缺陷。GPU7空闲，未重复提交旧成功基准。
- 证据修正：AB+SiLU/RMS在B1/1536生产单层profile为28.319+83.970us，占kernel合计16.6%；只能列为候选热点，尚不足以证明优先级。96MiB是可消除的逻辑张量读写量，未用硬件计数器证明全部进入HBM。
- 只取旧实验中的eager数据：1warp在B1/640、1536完整层仅降0.197%/0.130%，五轮涨跌混合；不支持为这两个目标形状增加生产分派。
- 独立代码审计：Mamba增量相对eb3d2c6为+116/-2行（含97行新prefill.py），新增3算子、1请求bool和1权重副本；没有生产Graph/1warp/640等长度分档。确有重复FLA校验、跨文件资格判断及嵌套三元，可读性担忧成立。
- 本轮最小生产整理：仅把Local三路嵌套三元展开if/elif/else，执行条件和调用不变；prefill模块说明改为已采用算子。暂缓语义性重构和新融合；现有6项GPU分派/缓存集成测试将验证此改动。
- 新测量准备：补真实E2整语言模型640/1536普通eager重复CPU/GPU时间线，关闭shape/memory/stack记录，区分cache、特征、Local、FLA和readout；此前仅有B1/1536和32768单层各一次profile，不足以定位真实整模型主机瓶颈。主线程统一GPU7调度，独立agent只写实验脚本。

- 已启动本次Supervisor；短序列总结追加后续禁用Graph的约束，原优先融合建议改为待验证候选。可读性改动的专项验证使用新源码和独立任务ID，属于修改后回归，不重跑旧性能曲线。

### 队列提交：eager-readability

- 作业：readability-integration；GPU 7，单卡并发 1，无自动重试。
- 配置/源码哈希：`evidence/mamba-eager-readability/`；SQLite/原始日志：`artifacts/mamba-kernel/eager-readability/`。
- 状态：已由 Supervisor 启动；完成与失败另追加实测记录。

- 可读性整理回归 `eager-readability/readability-integration`成功：6项GPU集成测试通过，覆盖dense/padding/holes/empty/finished及分段prefill接续decode。新源码校验与日志将在阶段结束归档。
- 新增 `eager_review.md` 与 `evidence/mamba-eager-audit/review.json`：纠正热点/优先级区别，列出普通eager的真实微小收益、代码复杂度问题和必须保留的语义；整模型profile结果明确待完成。

- 新增并冻结178行实验脚本 `scripts/eager_profile.py`，复用现有模型加载/资产身份/输入协议；不新增生产profiling逻辑，标记上下文退出后还原。CPU编译与无CUDA的CLI检查通过。两种长度各外部预热2次、profile预热2次、active3次；JSON明确这些是有采集扰动的归因数据，不能当成新延迟基准。

### 队列提交：eager-profile

- 作业：real-eager-profile；GPU 7，单卡并发 1，无自动重试。
- 配置/源码哈希：`evidence/mamba-eager-profile/`；SQLite/原始日志：`artifacts/mamba-kernel/eager-profile/`。
- 状态：已由 Supervisor 启动；完成与失败另追加实测记录。

### 真实普通eager时间线完成

- `eager-profile/real-eager-profile`成功，两长度各3次active完整请求，共各5160实际kernel；logits finite，metadata cuda_graph=false，shape/memory/stack采集均关闭，Activity Buffer Request归因项为0。
- 初读：每请求kernel总量640约21.968ms，1536约46.743ms；这是profile下GPU工作之和，不与未profile延迟直接相减。
- CPU self里ChunkSimpleGLAFunction在23个TTT层累计约5.49/5.43ms每请求；prefix阶段inclusive约7.60/7.53ms，包含Python/backend/launch等，不能全部解释为可消除的包装开销。cache.begin约0.225/0.210ms，finish约0.129/0.118ms。
- 下一步只读关联Chrome runtime与kernel及最内层阶段，查归因覆盖，完善报告。暂无新增优化实现或性能胜出声明。

### 归因、决策与阶段完成

- 新增只读分析 `evidence/mamba-eager-profile/attribution.py` 与JSON。共10320个kernel逐个correlation→CPU runtime→最内层阶段关联，未关联0、请求外0；保留三个请求分布及trace哈希，CPU父/子范围与GPU注释不重复相加。
- 23个TTT层每请求GPU阶段中位数：640 features/local/prefix/readout为1.198/0.640/0.573/0.444ms；1536为2.597/1.216/1.404/1.021ms。对应CPU inclusive prefix约7.601/7.462ms，最大；prefix内launch API仅约0.299/0.294ms，不能笼统归因CUDA launch或承诺可省7.5ms。
- 决策：优先隔离验证FLA调用链主机工作是否可精简，并用无profiler的完整层/模型A/B确认收益；AB融合只保留为有GPU热点依据的待验证候选。本阶段完成测量与代码审计，没有实现新的优化分支。
- 已完成 `eager_review.md`，更新旧总结后续约束链接；AGENTS已记录后续禁用Graph和测量驱动、避免堆叠特判的原则。
- 两个新GPU任务成功，日志与queue-status归档，提交时源码SHA256无变化；本次Supervisor已关闭，GPU7释放。下一步做文档数值/链接与diff检查，完成最终交付；无Git提交/推送。

- 最终检查通过：10320个kernel归因独占完备与总和守恒、trace/分析源码SHA256、16个新阶段表格数值、数学定界符及本地链接、Python编译和git diff --check。阶段全部工作完成；详细结论见 `eager_review.md`。

## 2026-09-17：decode优先目标与资源授权

- 用户明确主要想在decode速度超越Flash，并授权所需实验使用当前空闲物理GPU。已写AGENTS；资源从仅GPU7扩大为启动时确认空闲的卡，GPU ID显式传调度器，不使用已有占用的卡。
- 开工复核：AGENTS/主账本/工作树及14个SQLite队列已读，全部终态，唯一历史失败已记录；GPU1/3/5/6/7当前1–2MiB，GPU0/2/4仍占用。主线程统一调度，subagent只做代码与只读研究。
- 关键纠偏：decode单token走recurrent_decode，不执行FLA chunk。此前FLA prefill主机7.46ms热点不作为本阶段优化优先级。Flash缓存decode每步读取历史KV，计算随历史长度线性，不是每步重新做N² prefill。
- 既有基准：单层B1/640、1536当前1.031769/1.034765ms每步，DynamicCache Flash0.272978/0.272704ms；真实模型TPOT当前约27.29/27.33ms，对同期E0约10.97/11.47ms。差距约3.8倍单层、2.4倍整模型，小幅kernel调参不足以保证胜出。
- 新增 `scripts/decode_profile.py` 与 `tests/test_decode_profile.py`：真实E2先prefill，再连续128 teacher-forced步，独立请求预热，正式第18–20步active profile；标记cache/Local/feature/recurrent/readout，记录每步cache递增。2项CPU测试、编译/CLI通过，GPU待统一调度。
- 正在准备公平单层decode基准：新增预分配KV Flash，避免靠DynamicCache每步cat整段KV获得虚假优势；上界消融仅实验全active条件跳过重复finished保护，必须验证全部输出/cache，不是可部署路径。
- 调度提交助手增加可选GPU ID参数，默认仍7，不修改调度器逻辑；性能对照同卡串行，优先GPU7以衔接旧环境。下一步真实decode profile后排序候选。

### 队列提交：decode-profile

- 作业：real-decode-profile；GPU 7，单卡并发 1，无自动重试。
- 配置/源码哈希：`evidence/mamba-decode-profile/`；SQLite/原始日志：`artifacts/mamba-kernel/decode-profile/`。
- 状态：已由 Supervisor 启动；完成与失败另追加实测记录。

### 真实decode热点与公平基准就绪

- `decode-profile/real-decode-profile`成功：两长度各3个连续active单token步，共11838个kernel（每步1973）；全部独占归因，无未关联或步外kernel。完整128步cache递增、129位置logits finite。原始trace在`artifacts/decode-profile/mamba-decode-profile/`，分析及哈希在`evidence/mamba-decode-profile/attribution.{py,json}`。
- B1/1536每步23个TTT层累计CPU inclusive中位数：Local 6.465ms、features 4.118ms、recurrent 4.040ms、TTT set_layer 2.133ms；均受profile扰动，不能解释为可直接消除的时间。Local实际用PyTorch efficient/CUTLASS attention，每层12个kernel；核心GPU合计约0.160ms、两次scatter约0.193ms，其余准备约0.365ms。
- 新增并CPU验证`decode_bench.py`及2项测试：production、DynamicCache Flash、预分配StaticKV Flash、全active cache保护上界四臂。计时含连续decode、只排除prefill；每个样本重建缓存；正确性逐步检查输出/全部缓存和旧state不污染。正式目标以StaticKV Flash为准，上界不作为可部署实现。
- 根据实际Local热点，准备仅融合KV复制/scatter、可见mask和计数更新的实验候选`decode_local.py`；SDPA本体和数学不改，无B1/特定长度分派，无生产接线。CPU/autograd保留参考路径，专项GPU测试由主线程统一提交。
- 下一步：公平基准smoke，随后单变量候选GPU正确性及完整层A/B；依据收益再决定整模型验证。

### 队列提交：decode-smoke

- 作业：static-flash-decode-smoke；GPU 7，单卡并发 1，无自动重试。
- 配置/源码哈希：`evidence/mamba-decode-smoke/`；SQLite/原始日志：`artifacts/mamba-kernel/decode-smoke/`。
- 状态：已由 Supervisor 启动；完成与失败另追加实测记录。

### 公平decode smoke通过；Local准备融合候选冻结

- `decode-smoke/static-flash-decode-smoke`成功：B1/640连续8步的StaticKV vs DynamicCache、cache诊断上界 vs production全部输出和缓存exact，旧state未污染。短smoke仅验证协议，不作为正式收益。
- `decode_local.py`候选88行、2项CPU测试及CPU Triton interpreter索引检查通过；18项GPU测试待队列。融合的是非浮点归约准备工作，原SDPA、旧缓存不变、无效token写不可见slot的行为均保留。未接入生产。
- 候选SHA256 `cf29f054472c62bdcda878b56ba0b77cfabde489b6448dfcccf8db8e91b2c307`；接下来先GPU专项，再加入完整层五臂配对，避免重复跑原四臂正式曲线。

### 队列提交：decode-local-tests

- 作业：local-prepare-exactness；GPU 7，单卡并发 1，无自动重试。
- 配置/源码哈希：`evidence/mamba-decode-local-tests/`；SQLite/原始日志：`artifacts/mamba-kernel/decode-local-tests/`。
- 状态：已由 Supervisor 启动；完成与失败另追加实测记录。

- `decode_bench.py`加入第五臂`local_prepare`，只在decode patch实验候选；逐步输出与全部缓存相等检查纳入候选，source记录含候选文件。旧四臂smoke源码已另存`evidence/mamba-decode-smoke/decode_bench.py`，保留运行时证据。
- 独立只读审计通过：静态KV有效范围/位置/causal语义和所有归因hash一致；每步模型内部一次stream同步约9us存在，报告仅声称无人为逐步同步。样本ms/step是128步平均值，分位数来自请求样本，不能叫单token延迟分位。
- 完整层invalid/finished保护专项与真实模型连续128步脚本正在准备；重新查阅原历史measure：每步record event并clone GPU logits，CPU复制在整个循环之后；新连续协议只保留一对event，采用128步平均值，不直接与旧单步中位数相减。

### Local专项编译失败与一次修复

- `decode-local-tests/local-prepare-exactness`失败：18项均止于Triton编译，尚未执行候选kernel；锁定Triton版本不支持对显式`tl.constexpr`包装的tuple stride作索引（`constexpr object is not subscriptable`）。不是数值不一致、OOM或资源故障。原日志/源码已保留于该evidence。
- 允许一次有依据修复：将stride tuple拆成标量constexpr参数，与仓库现有kernel写法一致；随后用新任务ID验证，禁止自动重试。

- 新增中文`decode_plan.md`，先记录已完成归因、StaticKV公平协议、失败原因与按证据排序的路线；正式性能结果明确尚待执行，不声称decode获胜。

- Local stride编译修复完成并冻结SHA256 `846f0573a101051a0c71ed9504fe6fb4512fb19b2070773753b0e3aab1c3a7a1`；增加第19项完整层测试覆盖B4、padding prefill、持续/中途finished、交替invalid、4步跨wrap和旧全部缓存不污染。测试源码SHA256 `5dc8222227c82296a2f9a69a5027608fe9ad23b44f49efd35fb89b73f20836d6`。

### 队列提交：decode-local-fixed

- 作业：local-prepare-fixed-exactness；GPU 7，单卡并发 1，无自动重试。
- 配置/源码哈希：`evidence/mamba-decode-local-fixed/`；SQLite/原始日志：`artifacts/mamba-kernel/decode-local-fixed/`。
- 状态：已由 Supervisor 启动；完成与失败另追加实测记录。

### Local候选正确性通过，正式decode曲线启动

- `decode-local-fixed/local-prepare-fixed-exactness`成功：19项GPU专项全部通过，覆盖BF16/FP32、stride、B1/B4、跨32边界，以及完整层invalid/finished和旧缓存不污染。修复仅编译参数表示，无归约数学改变。
- 五臂正式配对：B1前缀640/1536/8192/32768/65536，B4前缀640/1536；每项128连续步、两次预热、5轮×5次采样。前缀640/1536是实用目标，长长度仅单层硬件扩展性，不声称模型长上下文质量。两个作业同GPU7串行，不与其它性能测量并行。

### 队列提交：decode-curve

- 作业：b1-decode-curve, b4-decode-check；GPU 7，单卡并发 1，无自动重试。
- 配置/源码哈希：`evidence/mamba-decode-curve/`；SQLite/原始日志：`artifacts/mamba-kernel/decode-curve/`。
- 状态：已由 Supervisor 启动；完成与失败另追加实测记录。

- 新增只读制图脚本`decode_plot.py`，直接读取正式B1/B4五臂证据，生成文档相对路径`images/decode_layer.png`；当前等待曲线任务完成，不运行额外GPU实验。

- 新增并冻结`decode_model.py`（SHA256 `55789bb0a72eb40c4d793512c4167ee198e652111e65269af390c28a7c7d1d51`），CLI/编译/CPU时序fixture通过；同一真实E2，prefill与patch选择在计时外，一对event测128连续步，GPU logits最后统一CPU核对，两warmup五round交替。没有人为逐步同步。
- B1正式五点全部成功：640/1536 Local候选约降2.89%，cache诊断上界约降6%；尚不足弥合静态KV Flash差距。32768下DynamicCache Flash1.260ms、StaticKV仅0.307ms，进一步确认不能靠全KV重复制成本宣称TTT领先。B4仍在运行，随后在同卡执行真实模型配对。

### 队列提交：decode-model

- 作业：real-model-local-prepare；GPU 7，单卡并发 1，无自动重试。
- 配置/源码哈希：`evidence/mamba-decode-model/`；SQLite/原始日志：`artifacts/mamba-kernel/decode-model/`。
- 状态：已由 Supervisor 启动；完成与失败另追加实测记录。

### 正式单层decode曲线完成

- `decode-curve`两个作业成功：7个形状×5臂×25样本共875个连续128步样本；每个形状三对正确性均128步exact，共2688次逐步输出/cache比较。所有Local候选五轮均快于production，但仅2.80–3.43%下降；没有配置超过静态KV Flash。
- B1/640：production1.037933、Local1.007895、Flash static0.285517ms/步；1536：1.042040、1.011893、0.284991。B1/65536 Local1.017390仍慢于static0.478947；不会把DynamicCache的2.384778ms当作充分胜出证据。
- cache上界相对production仅降5.54–6.35%，无法单独弥合差距，不建议绕过finished语义或为这项堆形状分支。
- `images/decode_layer.png`已由证据生成并视觉检查；19项GPU回归与正式曲线日志/SQLite状态已归档，提交时既有源码hash无变化。真实E2配对正在同GPU7执行。

- 新增中文`decode_summary.md`，单层表格直接从成功JSON生成，区分静态/动态KV、实用长度/硬件长序列、候选/生产。根据2.8–3.4%单层收益和可读性约束，决定本阶段不增加生产分派；整模型配对完成后补入结果。

### 真实模型配对完成与阶段决策

- `decode-model/real-model-local-prepare`成功：B1/640生产26.770309→Local26.087273ms/步（降2.55%），1536为26.922852→26.171539（降2.79%）。两长度各五轮全部更快，129位置×32064词表logits全部exact，最大绝对误差0；cache长度/seen/valid/finished/FP32状态检查通过。
- 本协议一对event包连续128步；旧历史为逐步event与GPU logits clone、循环后CPU复制及单步中位数，不把旧27.3ms与本次生产26.8ms跨协议差值归因于候选。此阶段不重跑E0、MME或POPE，不声称图像质量已重验。
- 决策：Local的单层约3%、整模型约2.5–2.8%收益真实但不足弥合Flash差距；为保持可读性，生产不新增优化分派，实验候选保留。缓存全跳过上界约6%，次要于进一步隔离特征/recurrent固定调用开销，不能用移除finished保护来冒充优化。
- `decode_plan.md`与独立阶段总结`decode_summary.md`已更新，完整命令/样本/日志/queue-status归档；全部新GPU任务终态，唯一新失败为已修复的Triton编译参数问题。下一步独立文档/数值审计、关闭本次Supervisor；无commit/push。

- 关闭前复核全部20个阶段SQLite均终态，无等待/运行GPU任务；日志、状态和旧源码快照已归档。Supervisor状态存`evidence/mamba-decode-model/supervisor-final.txt`，已关闭本次管理器。正式曲线/模型提交时源码hash未变；smoke与失败任务使用已归档旧版本，不将无关后来实验源码变化误报为运行时变更。

### Decode首轮最终验收

- 独立只读审计通过：7行单层表、2行模型表、5行profile表与JSON逐格一致；875样本/2688比较/19GPU测试和10次模型全词表exact核对通过；曲线42项、模型44项源码hash与当前一致。
- 两份中文文档13个相对链接及数学定界符、8份实验/测试Python语法、git diff --check通过；图表已视觉检查。补充旧历史允许TF32、本轮关闭的配置差异，所有收益使用本轮同进程配对，避免跨阶段归因。
- GPU7最终1MiB/0%，所有20个队列终态，本次Supervisor已关闭。阶段完成但decode超越Flash的长期目标尚未达成；下一阶段应测特征/recurrent固定调用开销，而非预先承诺小融合即可领先。没有新增生产路径，没有Git commit/push。

## 2026-09-17：保留优化的代码整理

- 用户明确要求清理现有实现，保留已经采用的优化和性能。目标：消除重复 FLA 边界、让投影和 Local 实现回到所属模块；不增加策略配置，不改 kernel、归约顺序、FP32 state 或缓存语义。
- 开工复核 AGENTS、账本、工作树、20 个 SQLite 队列和遗留进程：全部队列终态，无消费者或 GPU 任务残留；历史两个失败均已有记录。GPU7 空闲（1MiB/0%），用于本轮同卡回归，无 CUDA Graph。
- 修改前源码及相关测试/脚本复制至 `evidence/mamba-cleanup/before/`，用于与当前已优化实现配对；这不是与早期未优化版本的比较。历史结果不改写。
- 主线程负责生产重构、接口与验证；独立 agent 只读审计投影归属和脚本依赖。当前尚未修改生产代码或运行新验证；下一步冻结测试基线并实施最小整理。
- 修改前 CPU 回归完成：122 passed、65 GPU deselected，报告 `evidence/mamba-cleanup/before-cpu.xml`。新增 `artifacts/mamba-kernel/cleanup-before-jobs.json`，GPU 回归显式导入冻结快照，不受后续工作树修改影响；即将通过 Supervisor/SQLite 调度。
- 生产整理已实施：AB 投影原计算移入 `FeatureReadout.features_prefill`；直接 Local 原计算移入 `ops.local.dense_local`；无 mask FLA 合入 `fla_prefix(valid=None)` 的 no-grad 路径，训练仍使用原 packed 流程。最终 state 的 FP32 检查统一在 `_run_kernel`，删掉重复 `dense_prefix` 边界和已搬空的 `ops/prefill.py`，模型少一个 prefix 分支。既有三项优化、packed decode、请求级有效性检查全部保留，没有新增配置或适用形状。
- 当前正在同步调用者并准备修改后回归；修改前 GPU 回归使用独立快照运行。另委派独立 agent 只编辑 `scripts/cleanup_bench.py`，实现当前优化版本重构前后的 eager 完整层配对，不运行 GPU。
- 已更新 `test_prefill_ops.py` 使用归位后的接口；无 mask FLA 仍与显式全有效 mask 路径、FP64 oracle 比较，避免合并后变成自比较。已有 CPU autograd、空序列、尾块、stride、非零初态与 decode 接续覆盖保留。
- 新增 `scripts/cleanup_bench.py`：完整层 prefill 与连续 128 步 decode 分别 2 次预热、5×5 交替样本，比较本次修改前快照与 live，检查每一步输出/所有 cache tensor exact 及旧 tensor 不污染，保存输入/权重/源码哈希和原始样本；当前尚未提交 GPU 性能任务。
- 另一独立 agent 仅同步 `mamba_bench.py`、`eager_profile.py`、`short_ops.py` 及对应 CPU 测试，维护历史基线的冻结依赖，不放宽历史身份校验。
- 修改前 GPU 回归成功：84 passed（snapshot 算子/集成、FLA 长度/因果/分段/梯度），`cleanup-before/before-gpu-regression` 退出码 0，报告 `evidence/mamba-cleanup/before-gpu.xml`。修改后相同 CPU 选择亦 122 passed，报告 `after-cpu.xml`。
- 性能脚本运行前审查发现计数器的合同区别：`seen_tokens` 本来就按请求原地递增，不应要求不可变；旧 tensor 不污染检查限定为旧 LayerState，计数器仍检查两实现相等及实际步数。此为提交前脚本修正，不是已发生 GPU 失败。
- 上述性能脚本修正已完成，使用真实 CPU 缓存原地计数器的 129 步 fixture 全通过。历史脚本同步完成：`mamba_bench` 为 Git baseline 冻结 FLA/Local 整个 namespace、保留其余传递依赖校验，core 亦使用冻结 FLA；当前算子/profile 调用改为新归属。`eager_profile`/`short_ops` 引用同步，历史结果不变。
- 已准备 `artifacts/mamba-kernel/cleanup-after-jobs.json`：修改后 84 项 GPU 回归与完整层 eager 配对，两任务在 GPU7 串行、无自动重试。正确性和性能失败均保留产物；性能目标是保持本次重构前已优化版本，不是重新争取超过 Flash。
- 用户追加清理不必要测试。独立审计后从常规 `tests/` 删除 5 个仅服务实验脚本/未投产候选的文件：`test_decode_bench.py`、`test_decode_profile.py`、`test_decode_local.py`、`test_short_ops.py`、`test_mamba_benchmark.py`（含本轮刚补的 3 项基准辅助测试）。删除时共 474 行、49 个参数化实例；已跑源码压缩保存于 `artifacts/mamba-kernel/retired-experiment-tests.tar.gz`，仅作历史材料，不再作为维护中的测试或另建常规测试入口。实验脚本/候选能力及既有结果仍保留。
- 同时删除 `test_prefill_ops` 重复的 CPU FLA 拒绝测试，以及该文件和 `test_ops` 中两个只锁定一次 SDPA 调用/具体 reshape 的结构测试。保留数值 oracle、训练梯度、BF16/FP32 合同、padding/finished、缓存接续与旧 state 保护，未用通用 fixture 层替代删除的冗余代码。
- 同步脚本曾通过 8 项 CPU 专项（16 GPU deselected）；新增的辅助测试随后按用户要求删除，不能算作最终套件数量。接下来运行精简后生产测试，并执行已准备的 eager 性能对照。
- 精简后完整 CPU 套件通过：199 passed、84 GPU deselected（18.83s），报告 `evidence/mamba-cleanup/final-cpu.xml`。维护中的测试由本轮开始时 24 文件/2847 行降至 19 文件/2408 行，净减 439 行；不把本轮新增后又删掉的 68 行辅助测试计作原有代码减少。GPU 回归正在执行，性能任务同卡排队。
- 新增中文 `cleanup_summary.md` 记录代码归属、实际删除对象与验证范围；更新算子 README 中已过时的“尚未集成/待GPU验证”说明，以及旧审计/总结的后续链接。GPU/性能栏仍明确待完成，不预写收益。独立 AST/字节审计确认 AB/Local 数学顺序、既有 FeatureReadout 方法、cache/generation 与 fused kernels 保持不变。
- 最终 GPU 回归成功：84 passed、100 deselected，129.33s，报告 `evidence/mamba-cleanup/after-gpu.xml`；`cleanup-after` 两项任务均退出码 0。
- eager 完整层配对成功：B1/640 prefill 1.286816→1.293056ms（+0.485%），decode 1.006764→1.007802ms/步（+0.103%）；B1/1536 prefill 1.312480→1.313888ms（+0.107%），decode 1.006411→1.006553ms/步（+0.014%）。每项 5×5 交替样本，共 200 样本；全部中位变化小于 0.5%，本轮视为性能基本持平，不宣称提速或新的 Flash 胜出。
- 两长度各 prefill+128 decode 共 258 次逐步比较，输出均 finite，输出和全部缓存均 exact，旧 LayerState 未污染、FP32 state 与计数器均符合契约。原始数据 `evidence/mamba-cleanup/paired.json`；配置/日志/SQLite 状态归档于 `evidence/mamba-cleanup-before/` 和 `evidence/mamba-cleanup-after/`。提交后 Python 源码哈希未变化。
- 阶段总结已补齐最终结果，四项显存统计两臂亦完全相同。测试净减数量按初始源码 SHA256 复核：原 `test_mamba_benchmark.py` 为 23 行，本轮临时新增 68 行后整文件删除，故净减为 439 行（纠正中途估计的 444 行）。接下来只做文档/产物审计和关闭本次 Supervisor，不追加性能实验。
- 收尾复核：22 个阶段队列均终态，本轮 3 个 GPU 任务全部成功；Supervisor 最终状态已保存并关闭，无消费者/测试遗留进程，GPU7 释放（1MiB/0%）。文档相对链接、数学定界符、JUnit 数量与 `git diff --check` 通过；队列总状态 `evidence/mamba-cleanup/all-queue-status.json`，退休实验测试压缩包 SHA256 `ba435b857931eb80903e82b740933eabb933ce4a1171bcd37351386b0c421ed1`。代码/测试整理完成，无 Git commit/push。
- 独立最终审计通过：从原始 200 样本重算四项中位数/百分比逐格一致，258 次比较与四项显存字典核对一致；JUnit、删除数量、生产净减及文档链接全部通过。已收紧 finite 的措辞，使其只对应实际检查的输出字段。阶段完成。

### 队列提交：cleanup-before

- 作业：before-gpu-regression；GPU 7，单卡并发 1，无自动重试。
- 配置/源码哈希：`evidence/mamba-cleanup-before/`；SQLite/原始日志：`artifacts/mamba-kernel/cleanup-before/`。
- 状态：已由 Supervisor 启动；完成与失败另追加实测记录。

### 队列提交：cleanup-after

- 作业：after-gpu-regression, paired-layer-performance；GPU 7，单卡并发 1，无自动重试。
- 配置/源码哈希：`evidence/mamba-cleanup-after/`；SQLite/原始日志：`artifacts/mamba-kernel/cleanup-after/`。
- 状态：已由 Supervisor 启动；完成与失败另追加实测记录。

## 2026-09-17：纠正文档与代码目录职责

- 上述 cleanup-before/after 三项 GPU 作业均已成功，最终验证见前节。本次用户指出文档目录不应存放脚本、源码副本；此前把稳定文档目录扩展为代码/运行产物目录是组织错误，本轮纠正。
- 开工已重读 AGENTS、相关账本、工作树及已有结果；22 个 SQLite 队列均终态，无相关遗留进程。只迁移文件及修复路径，不启动 GPU、不重复成功实验，不改变生产计算。
- 目录约定已补入 AGENTS/README：生产实现归 `src/prefix_ttt/`，实验工具归 `scripts/experiments/<project>/`；原始证据、配置、日志与源码快照归 `artifacts/experiments/<project>/`，文档保留说明、图表与简要汇总。
- 主线程负责原始证据完整迁移、文档链接及验证；两个子任务分别仅处理 kernel 工具、h100/fullft 工具。历史结果/队列中记录的旧命令和源码哈希不改写，迁移映射另记；图像继续留在原文档 `images/`。下一步核对移动前后内容哈希、脚本导入/CLI、Shell语法、文档链接与CPU回归。
- 原始证据已原样移动：`2026-09-09-prefix-ttt/evidence/` → `artifacts/experiments/prefix_ttt/evidence/`；`2026-09-10-prefix-ttt-h100/evidence/` → `artifacts/experiments/prefix_ttt_h100/evidence/`；本项目 `evidence/` → `artifacts/experiments/prefix_ttt_kernel/evidence/`。旧前缀均位于 `docs/experiments/`。逐文件字节数与 SHA256 移动前后一致，完整清单 `artifacts/experiments/layout-migration.json`。代码快照仅作为 artifacts 中的历史材料，不再作为 docs 下的源码树。
- 工具迁移完成：本项目 16 个脚本与原 `artifacts/mamba-kernel/submit.py` 归 `scripts/experiments/prefix_ttt_kernel/`；h100 的 7 个脚本归 `scripts/experiments/prefix_ttt_h100/`；fullft 的 3 个脚本归 `scripts/experiments/prefix_ttt_fullft/`。旧空脚本目录已移除。kernel 仅两绘图脚本和提交助手需要路径改动，h100/fullft 仅修复 Shell 仓库根解析与跨脚本调用，计算逻辑不变。
- 已修正 13 份文档中的现行链接/路径；历史账本正文和结果记录的运行时路径保持原文，通过本节映射解释迁移。补丁初次因 hunk 格式被拒绝、未应用任何修改，修正格式后成功；无文件内容丢失或实验失败。
- 子任务验证通过：17 个 kernel 文件语法、6 模块 CPU 导入、11 个非 Graph CLI；提交助手只核对路径和源码扫描、不启动调度；两绘图工具实际读取迁移结果并拦截保存，目的地仍是文档 images。h100/fullft 的 7 个 Shell `bash -n`、3 个 Python AST/CLI 通过，未执行训练、评测、NFS 或 GPU。
- 已更新本地 Supervisor 22 个程序配置的 manifest 文件位置，使其指向移动后的同一字节文件；只改运行入口路径，不改历史 evidence 中的配置快照、manifest 内容、SQLite 身份或终态，不启动消费者。
- 完整 CPU 回归通过：199 passed、84 GPU deselected（19.17s），报告归 `artifacts/experiments/prefix_ttt_kernel/layout-cpu.xml`。生产 Python 与上一阶段已验证版本逐文件哈希相同；本轮没有重跑 GPU，原性能结论仍对应原始验收。
- 最终独立审计通过：文档目录递归无 Python/Shell、源码树、原始日志/测试报告/运行配置；27 个工具归 scripts（含提交助手），312 个历史文件逐字节保留于 artifacts。120 个 Markdown 本地链接已核对，唯一不存在项是用户早已明确删除并要求保留历史引用的 `migration_from_a40.md`，不属于迁移缺陷。22 个 Supervisor manifest 新路径均存在且与 SQLite 保存身份一致，终态未变。生产代码、计算优化和图片内容未改；`git diff --check` 通过。目录整理完成，无 Git commit/push。

## 本地提交

- 用户明确授权将上述改动合并为一个本地 commit，并明确不推送。提交范围包括已验证的 prefill 优化与清理、必要生产测试、实验脚本归位、中文文档/图表、Mamba 参考论文及协作规范；原始运行产物按现有 `.gitignore` 保留在本地 artifacts，不纳入提交。
- 提交前复核工作树与队列终态，生产 Python 与验收源码哈希一致；复核已完成的 199 项 CPU、84 项 GPU 报告均无失败，不重复运行成功实验。提交身份为 `Codex <codex@openai.com>`，提交信息为 `Optimize Prefix-TTT prefill and organize experiment code`；提交结果以 Git 记录为准，不执行 push。
