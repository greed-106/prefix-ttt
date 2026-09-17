# Decode 优先：测量、首个候选与后续路线

本阶段延续同一 kernel 项目，目标改为普通 eager 的单 token decode；全程不使用 CUDA Graph。用户已授权使用启动时确认空闲的 GPU。本阶段已完成归因、875个单层序列样本与真实模型五轮配对，结果见[阶段总结](decode_summary.md)：Local准备融合仅使单层降低2.80–3.43%、整模型降低2.55–2.79%；未超过静态KV Flash。候选保留在实验目录，生产不增加分派。

## 比较的对象与成功标准

缓存 decode 与 prefill 是不同问题。对已有长度 $L$ 的上下文，Flash 每次只计算一个 query，读取历史 KV，单步 attention 随 $L$ 线性增长；不能把它称为每步重新执行平方复杂度的 prefill。Prefix-TTT 则更新定长矩阵状态，同时执行 Local-32、特征映射和门控读出。其算法优势是状态部分不随历史增长，短上下文是否更快还取决于固定开销。

比较采用同一 H100、BF16 输入、相同真实 checkpoint Q/K/V/O 权重，以及相同固定输入。TTT 状态与参数 master 保持 FP32。两者是不同的 attention 函数，性能公平不代表输出相等或模型质量等价；exact 校验针对优化前后 TTT，以及两种 KV 管理方式的 Flash。

主基线为锁定 PyTorch 2.7.1 的 `FLASH_ATTENTION` SDPA，预分配容量为前缀加后续生成步数，只让有效 KV 参与计算。不依赖每步 `cat` 整段 KV 的成本来制造优势。没有安装并测量独立 FlashAttention-3 或 FlashInfer，结论仅限本基线。

每个样本先建立独立缓存，再计时连续 128 次单 token forward；计时包含 QKV/O 投影、RoPE、特征、Local、递推、读出及缓存更新。prefill、初始缓存分配、编译预热和正确性检查不计入 decode 时间。单层每步执行一次 cache.begin/finish，真实模型则整模型每步执行一次，不能把单层延迟直接乘以 23。

验收分两级：候选对当前生产实现至少有稳定的完整层收益，并在真实模型连续 decode 中复现；“超越 Flash”则要求相对静态 KV Flash 至少快 5%，五轮中每轮均领先。保留 wall/CUDA 原始样本与分轮结果；每次 128 步的平均 TPOT 分布不是单 token 尾延迟分布。

## 实际 decode 路径给出的证据

当前单 token 走 `recurrent_decode`，不调用 prefill 的 FLA chunk。Mamba 风格优化之前获得的 prefill 加速，不能作为 decode 收益；历史同进程整模型配对也表明 decode 基本不变。

已完成真实 E2 的 B1、640/1536 前缀测量。先进行独立请求预热，再对新请求连续生成 128 个 teacher-forced token，仅采集第 18–20 步。两份 trace 共 11838 个 kernel，每步 1973 个，全部能够通过 correlation 唯一关联到 CPU 调用与最内层阶段，无未关联或步外 kernel。

以下是 1536 前缀时三步中位数，各行累计了模型中 23 个 TTT 层的同一阶段，单位 ms：

| 阶段 | CPU inclusive | GPU kernel 合计 | Launch API |
|---|---:|---:|---:|
| Local | 6.465 | 0.718 | 0.963 |
| 特征映射 | 4.118 | 0.248 | 0.336 |
| 递推更新与读出 | 4.040 | 0.177 | 0.269 |
| TTT cache.set_layer | 2.133 | 0.303 | 0.319 |
| 门控读出尾部 | 1.623 | 0.041 | 0.079 |

这些是带 profiler 扰动的归因，CPU inclusive 包含嵌套调用，不能与 GPU 合计相加，也不能直接当作可节省时间。模型内部每步还有一次约 9 微秒的 stream 同步；本实验没有人为插入逐步同步，并不表示整个模型没有同步。

Local 每层有 12 个 kernel。由于 boolean mask，实际后端是 PyTorch efficient/CUTLASS attention。23 层 Local 核心 GPU 合计约 0.160 ms，两次 scatter 约 0.193 ms，其余 mask/计数等准备约 0.365 ms。因此首个候选针对准备链，而不是凭复杂度推断 attention 核心应当先重写。

原始证据：[profile 结果](../../../artifacts/experiments/prefix_ttt_kernel/evidence/mamba-decode-profile/results.json)、[独占归因](../../../artifacts/experiments/prefix_ttt_kernel/evidence/mamba-decode-profile/attribution.json)、[分析脚本](../../../artifacts/experiments/prefix_ttt_kernel/evidence/mamba-decode-profile/attribution.py)。原始 trace 路径与 SHA256 保存在归因文件中。

## 首个候选与诊断消融

`decode_local.py` 将旧 KV 复制、新 token scatter、visible mask 和 seen/length 更新融合为一个 Triton kernel，后续保留原 SDPA 调用。它不重写 softmax、投影或 recurrent 浮点归约，不修改旧缓存，也不增加 B1 或特定长度分派。当前为实验实现，尚未接入生产。

首次 GPU 专项停在 Triton 编译：当前锁定版本不支持对显式 constexpr tuple stride 作索引。日志与源码保留；修复为标量 stride 参数后，19项GPU测试及正式曲线2688次逐步输出/cache比较通过，不把CPU interpreter检查当成GPU编译通过。

同时比较 `cache_upper_bound`：只在全 active、单 TTT 层实验中跳过 set_layer 的校验和 finished 合并，估计这部分最多有多少优化空间。它不能投入生产，因为 Local 即使 valid=false 也会写入一个不可见 slot，finished 行完整冻结依赖原 set_layer；也不能沿用到保留全注意力的层。

## 后续优化的决策顺序

1. **Local准备融合已完成验证。** BF16/FP32、stride、invalid、finished、跨32-token边界、旧缓存不污染均通过，完整层和真实模型仅约3%收益。本阶段保留实验候选，后续统一整理decode时可考虑替换现有准备链，避免额外生产分派。
2. **缓存职责降为次要候选。** 上界消融实测仅5.54–6.35%，不能单独弥合差距。若统一整理更新契约，可合并重复保护，但必须保留完整finished语义，不引入每步`.any().item()`的all-active特判；此项不再作为主要突破方向。
3. **下一项主研究是特征与recurrent的固定调用开销。** 先隔离测量主机包装和重复metadata处理，再测试合并 Q/K 的 SiLU/RMS 尾部调用、精简重复主机包装，保留现有 cuBLAS 投影和 FP32 读出归约。按单变量验证收益，不能以 kernel 数量下降代替 TPOT 下降。
4. **判断短上下文能否达到目标。** 若上述可安全替换的开销仍不足，才考虑较大的 decode 数据流重整或数值实现改动，并单独做质量验收。历史自定义 feature matvec 和全融合 recurrent 曾产生最大 0.171875 的 logits 误差、改变 25/2374 个 MME 答案，不能重复把“数值接近”当作部署通过。

长上下文单层扫描仅用于研究硬件交叉点。项目训练/评估协议主要在 2048 内，基底配置上限为 4096；超出这些范围的计时不能证明长上下文能力或质量。真实模型含 23 个 TTT 层和 9 个全注意力层，公共 MLP、权重读取与保留全注意力的开销也限制最终端到端收益。

代码组织上，以替换已有准备链为目标，训练/prefill/decode 按语义分工；不为每个 batch、长度或 tile 加一个生产分支。失败候选留在实验目录和账本，只有完整层/模型及正确性同时通过的实现才考虑合入。
