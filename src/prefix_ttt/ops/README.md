# 算子接口与数值边界

`reference.py` 是小规模正确性 oracle；顺序及分块路径均为 token-inclusive，固定 eta=1/128，FP32 累积（FP64 输入保留 FP64 以供严格测试）。不用于正式训练性能报告。

`FeatureReadout` 管理 FP32 特征参数和非持久 BF16 推理副本。`features_pair` 保留通用投影及已有 packed decode；`features_prefill` 使用拼接 A/B 列的两次 BMM，再调用相同的 SiLU/乘积/RMS 融合算子。权重重载后需重新 `prepare_inference()`。

`local_attention` 将有效 token 按逻辑位置打包为 32 长度块，以一次批量 SDPA 计算。`local_attention_cached` 接续未满块；`local_attention_decode` 处理已有缓存上的单 token。`dense_local` 直接处理从逻辑零开始的全有效输入，省去打包，生成相同格式的尾部缓存。它不接受旧缓存，调用者必须先确认输入有效性。

`PrefixTTTAttention` 统一选择这些路径。全有效初始请求由 `TransformersHybridCache.begin` 每请求判断一次；模型继续检查 backend、dtype、头维与旧状态。padding、holes、finished 和后续分段保留通用路径，finished 状态保护由请求缓存负责。

`fla_prefix` 是统一 FLA 入口，使用锁定 commit `c51953382397da5c3b7b8a41e568915b703e2934`：tile=64 调用 `chunk_linear_attn(normalize=False)`，其余 16/32/128 调用无 gate 的 `chunk_simple_gla(chunk_size=...)`。输入为 `[B,T,H,D]`，`scale=1`，V 在接口内部只除一次 128。no-grad 且 `valid=None` 时直接执行无 mask kernel；显式 mask 的推理将无效 Q/K/V 清零以保持状态；训练按有效 token gather 和独立 `cu_seqlens` 处理。初态与返回终态使用 FP32。

单步 `recurrent_step` 显式禁用 autocast，以向量化 FP32 外积和读出提供 CPU/GPU 数学路径；生产融合 decode 使用 `recurrent_decode`。均不以 Python 循环代替完整序列 FLA。验证范围和同卡 eager 对照见[整理记录](../../../docs/experiments/2026-09-11-prefix-ttt-kernel/cleanup_summary.md)。

固定源码来源为 https://github.com/fla-org/flash-linear-attention ，MIT；基础安装依赖为 `transformers>=4.45.0` 与 `einops`，现有项目显式锁定的 torch/triton 满足该版本 CUDA extra 的最低要求。没有复制其 trainer。
