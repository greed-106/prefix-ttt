# Prefix-TTT 推理 kernel 优化总结

本阶段于 2026-09-11 完成，最终采用 packed 版本。在不改变已训练方法的前提下，prefill 延迟降低 23.1%–25.9%，TPOT 降低 13.4%–14.3%；MME、POPE 共 11374 条回答与合并 LoRA 后的 postrefactor 基线完全一致，但性能仍未追平 E0。

## 1. 优化对象与数值边界

注意力计算减少不等于整模型等比例加速：MLP 与 QKV/O 投影仍在，TTT 又增加特征映射、FP32 状态读写、归一化和门控。原实现还有重复权重转换、细碎算子及 prefill 变长打包的同步成本。本阶段减少这些成本，没有重新训练或改变注意力定义。

状态仍为先写后读的 additive update，不是 delta rule：

$$
S_t=S_{t-1}+2^{-7}\phi(k_t)v_t^\top,\qquad m_t=\phi(q_t)^\top S_t.
$$

特征仍为 SiLU 双投影乘积再 RMS，门控仍为有符号线性门控；Local-32 仍按有效位置计数进行分块因果注意力。23 个 TTT 层、9 个完整注意力层、FP32 master 和 checkpoint 均不变。

| 保留实现 | 作用与边界 |
| --- | --- |
| SiLU/乘积/RMS、readout 尾部融合 | 减少中间张量和 launch，保留 BF16 舍入点；生产 D=128 复现当前 ATen 求和顺序 |
| FP32 状态写入融合 | 不修改旧状态，读出仍用原 FP32 cuBLAS einsum |
| Q/K RoPE 融合 | 保留乘法舍入和 HF 输出 strides，禁 FMA；混合 dtype 回退 |
| 无梯度 FLA dense masked prefill | 避免变长打包；无效输入置零，训练和物理空序列保留原路径 |
| 准备期 BF16 权重副本 | 避免每 token 转换；不进入 state_dict，重载/修改权重后需重新准备 |
| 单步 packed 特征投影 | 四次 GEMM 合为一次 batched GEMM；限已准备的 B1/T1/H32/D128 BF16 |

准备期副本合计 97.75 MiB，参数 master 与请求缓存结构、容量不变。融合路径保留 autograd 回退；FLA dense 分支本身覆盖 FP16，不能泛称所有 FP16 路径均未变。padding/空洞可能改变 BF16 chunk 边界，不承诺任意布局下 dense 与 packed 都逐位相同。

## 2. 失败方案与修正

首版自定义 feature matvec 和完全融合的 recurrent 读出更快，但真实 logits 最大绝对差达到 0.171875。正式 MME 有 25/2374 条回答改变，Perception 从 1429.4893 降至 1413.1216，Cognition 从 278.2143 降至 269.6429；POPE 的小幅上涨不能抵消该回归，因此两项生产候选已实际删除。

不同 GEMM/归约顺序、TF32、RMS 求和顺序以及 RoPE layout 对后续 cuBLAS 算法的影响，都可能造成漂移。最终恢复原 GEMM/读出归约，只融合可保留求值顺序的操作。早期验证还出现过 Triton stride 编译错误、测试 oracle 假设错误和旧 forward 误绑定新 FLA 的问题，均已修正；受污染的早期 ablation 不作为最终正确性证据。最终 kernel 计数排除了 Memcpy/Memset。

## 3. 最终性能与质量

真实 E2 checkpoint 在同进程交替运行原提交、未 packed 和最终 packed 路径，前缀为 640/1536 token，各追加 128 个 teacher-forced decode，重复三轮。prefill 取三轮中位数，TPOT 取三轮共 384 个单步的合并中位数。数值比较取末轮保存的 129 个 logits 位置，两个长度与原提交及未 packed 路径均逐位一致、finite。

| 前缀长度 | E0 prefill / TPOT（ms） | 原 E2 | 未 packed | 最终 packed |
| --- | --- | --- | --- | --- |
| 640 | 22.886 / 11.407 | 79.336 / 37.786 | 58.751 / 34.734 | 58.800 / 32.733 |
| 1536 | 48.085 / 11.676 | 106.334 / 37.484 | 81.997 / 34.111 | 81.738 / 32.134 |

汇总见 [640 结果](../../../artifacts/experiments/prefix_ttt_kernel/evidence/packed-probe/640.json)、[1536 结果](../../../artifacts/experiments/prefix_ttt_kernel/evidence/packed-probe/1536.json)。packed 本身约省 2 ms/token，不改善 prefill；最终 TPOT 仍为 E0 的 2.75–2.87 倍。真正 kernel 数在两长度下相同：prefill 5136→3066，decode 3182→1963；E0 为 1364/1492。计时与 profiler 独立运行，不能直接相减计算 CPU 开销；forward-loop wall 含 logits 采集，不是完整 greedy 请求延迟。cache 字段仅为容量统计。

| 任务 | 官方分数（与 postrefactor 相同） | 回答变化 / 缺失 |
| --- | --- | --- |
| MME | Perception 1429.4892957182874；Cognition 278.2142857142857 | 0 / 0，共 2374 条 |
| POPE | Accuracy 0.8501111111111112；F1 0.8357081963220071 | 0 / 0，共 9000 条 |

见[最终 MME 对照](../../../artifacts/experiments/prefix_ttt_kernel/evidence/quality-packed/mme-comparison.txt)、[最终 POPE 对照](../../../artifacts/experiments/prefix_ttt_kernel/evidence/quality-packed/pope-comparison.txt)。这是最终 packed 版本独立评测，原始结果仍在 `/data/shared/weights/prefix-ttt/eval-kernel/packed/`。MME 峰值 allocated 中位数从 13.515 增至 13.607 GiB，请求缓存 147.781 MiB 不变；跨运行评测延迟不替代同进程 A/B。

## 4. 复现与限制

保留[性能测量脚本](../../../scripts/experiments/prefix_ttt_kernel/measure.py)、[性能配置](../../../artifacts/experiments/prefix_ttt_kernel/evidence/packed-probe/manifest.json)和[质量配置](../../../artifacts/experiments/prefix_ttt_kernel/evidence/quality-packed/manifest.json)。测量依赖仓库中可读取的基线提交 `16d67fe`、锁定依赖及本地模型/数据；ZIP 或不含该提交的浅克隆不一定满足条件。配置保留历史输出路径，不应覆盖已有结果。原始日志、逐步采样、详细 profiler 和中间诊断产物已移出仓库，精简 JSON 的保留值没有重算。

清理前完整验证为 CPU **170 passed**、GPU **138 passed**，覆盖舍入、dtype/layout、状态递推、缓存 finished 保护与梯度等。七份新增专项测试已按用户要求移出，这些数字是历史验证，不代表当前仍有相同覆盖；清理后保留的原有 CPU 测试为 **167 passed**，GPU 未重跑。

实际验证环境为 PyTorch 2.7.1、Triton 3.3.1 和 H100，换版本或后端应重新验证。若继续优化，需要分析剩余主机包装、动态缓存分配及共同骨干成本；静态缓存/CUDA Graph 等方向本阶段未实施，也未证明能够超越 E0。

2026-09-17 续接阶段见 [Mamba 启发的 prefill 优化总结](mamba_summary.md)：完整注意力替换层在长序列超过指定 Flash SDPA，MME/POPE回归一致。该层级实验不替代本页的完整模型E0/E2比较。
