# CPU 算子接口边界

`reference.py` 是小规模正确性 oracle；顺序及分块路径均为 token-inclusive，固定 eta=1/128，FP32 累积（FP64 输入保留 FP64 以供严格测试）。不用于正式训练性能报告。

`local_attention` 将有效 token 按逻辑位置打包为 32 长度块，以一次批量 SDPA 调用计算，尾部物理 padding 输出丢弃。`local_attention_cached` 接续请求当前未满块，支持任意分段和逐 token 调用；返回独立 LocalCache，完整块完成后不保留其 KV。finished 行输出为零且不增加计数或写 KV。

这两个接口尚未安装至真实 LLaVA attention，LocalCache 也不等同于 Transformers Cache。GPU smoke 仍须检查实际 SDPA/FLA backend 和分支耗时；CPU 测试通过不能替代这一验证。

`fla_prefix` 使用锁定 commit `c51953382397da5c3b7b8a41e568915b703e2934`：tile=64 调用 `chunk_linear_attn(normalize=False)`，其余 16/32/128 调用同版本无 gate `chunk_simple_gla(chunk_size=...)`。输入始终为 `[B,T,H,D]`，`scale=1`，V 在接口内部只除一次 128。padding 通过有效 token gather 与独立 `cu_seqlens` 边界处理；全空行不传入 kernel，保留其 initial state。

单步 `recurrent_step` 使用显式禁用 autocast 的向量化 FP32 外积和读出，CPU 可检验数学输出与梯度；不是对完整序列的 Python loop fallback。FLA 的 GPU 数值、varlen kernel 和性能均待挂卡验证；完整模型生成仍须在集成阶段完成。

固定源码来源为 https://github.com/fla-org/flash-linear-attention ，MIT；基础安装依赖为 `transformers>=4.45.0` 与 `einops`，现有项目显式锁定的 torch/triton 满足该版本 CUDA extra 的最低要求。没有复制其 trainer。
