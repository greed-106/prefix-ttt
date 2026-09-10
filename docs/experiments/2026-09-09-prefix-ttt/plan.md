# LLaVA-1.5-7B Prefix-TTT 任务书

本文件归档用户批准的原始规格与修订决定，是持续实施的规范，不是已运行结果。状态以 [ledger.md](ledger.md) 和原始日志为准。启动日期为 2026-09-09，跨日不更换项目目录。

2026-09-10托管修订：用户批准以pixi安装的Supervisor替代当前容器不可用的用户systemd，仍使用现有SQLite队列、显式argv/环境、GPU白名单及失败不自动重试语义。不改变模型或实验顺序。消费者异常后人工核对遗留进程再恢复，禁止自动重启风暴；容器重建后的启动入口及数据库持久性另行确认。操作见supervisor.md，旧systemd配置只保留作参考。

## 1. 目标、边界与顺序

研究固定大小关联状态能否以有限适配成本接替大部分 Full Attention，并获得实际缓存或延迟收益。主框架是完整 LLaVA-1.5-7B：保留视觉 token 数及原 FFN，9 层 Full MSA，因此整模型不是严格线性复杂度或常数缓存。

不做全量微调、token 删除、动态选层、视频、卷积、SWA 退火、Muon、动量、遗忘门、RL 或扩大完整消融矩阵；不预先宣称首创或质量/速度必然提升。

修订后的主线：

```text
仓库/环境/数据审计 → CPU/reference 测试 → 最小 GPU 闭环
→ E0 质量基线 → 50,000 条阶段 A → A 后整模型切换诊断
→ E1/E2 同预算 Pilot → 各自恢复并完成 B 的一遍 SFT
→ 最终质量/效率评测 → 原定小预算布局和机制诊断
```

当前阶段仅完成环境准备与 CPU 可验证工作。用户挂载 GPU 并重启后继续；不在 CPU 上冒充运行完整训练或 GPU 验收。

## 2. 工程来源、环境与完整权重

实际只读参考目录为 `preference/`：LLaVA `c121f0432da27facab705978f83c4ada465e46fd`；KV Binding LaCT `1c2109fef1cf707c1025643d73684fe5e0920cc6`；Spatial-TTT `e2e33a62b6f92c33b7e24ff042be9737d05c5bdb`。LaCT 虽在 main，已包含指定并行变体，不自动切分支。

独立复制所需代码并保留逐文件许可和来源。禁止运行时从只读参考目录 import；LLaVA 原模型、图像处理、展开、labels、dataset、LoRA/SFT 和评测组织继续复用。不迁移到 LaCT 的 trainer/语言模型框架。

本地数据根为 `data/llava-v1.5-assets-v1/`，源包保留。模型是 `llava-hf/llava-1.5-7b-hf` revision `b234b804b114d9e37bb655e11cbbb5f5e971b7a9` 的完整 HF 格式 checkpoint，不能直接传给原 LLaVA loader。增加严格 bridge：映射 LLM、lm_head、projector 和 embedded CLIP，逐 tensor 检查覆盖和形状；不使用缺权重静默随机初始化。保留全部 32064 行 embedding/head，不为 tokenizer 长度裁剪。确认 32 层、hidden=4096、32 heads、D=128，不符则停止。

保持本地 processor 的 336 CLIP resize/center crop、patch 选择，不能无说明换为 pad。固定原 LLaVA v1 对话模板，HF 对照接收同一显式 prompt。先依次回归 pixel values、ViT/projector、展开 embeddings、全 MSA logits 和生成，再安装新 attention。

使用独立 uv 环境。初始版本组合：Python 3.10、torch 2.7.1/cu128、torchvision 0.22.1、Triton 3.3.1、Transformers 4.51.3、PEFT 0.15.2、Accelerate 1.6.0。普通包用用户指定阿里源，torch/torchvision 用官方 cu128。记录实际版本及 import `__file__`，禁止全局盲目升降级。FLA 候选锁定 `c51953382397da5c3b7b8a41e568915b703e2934`，接口和 GPU 数值验证后才称后端可用。正式 argv 使用绝对 uv，环境不依赖 `.bashrc`。

## 3. A9-T23 与精确算子

所有层号 0-based：

```python
full_attention_layers = [0, 3, 7, 11, 15, 19, 23, 27, 31]
ttt_layers = [i for i in range(32) if i not in full_attention_layers]
candidate_ttt_layers = [i for i in range(32)
    if i not in [3, 7, 11, 15, 19, 23, 27, 31]]
```

冻结 ViT/projector 和 LLM 原参数；保留原 QKV/o_proj、FFN、residual、Norm、embedding 和 lm_head。原 input RMSNorm 后的 x 产生共享 QKV，原 RoPE 后 Q/K 同时进入 Local-32 与 TTT。TTT 经无 affine RMSNorm 后乘每 token/head 的有符号 gate，两分支相加、拼 head，原 o_proj 只调用一次。anchor 不安装无用分支。

Q/K/V 布局 `[B,T,H,D]`，H=32，D=R=128；状态 `[B,H,R,D]`。每层每 head 独立 A/B 矩阵 `[H,128,128]`，Q/K 共享映射，不跨 head/层共享：

$$
\phi(x)=\mathrm{RMSNorm}_{\mathrm{no\ affine}}[\mathrm{SiLU}(xA_\phi)\odot(xB_\phi)].
$$

A/B 正态 std=$1/\sqrt{128}$，无 bias，按层确定种子，外层训练但单序列内不更新。新增 feature/readout Norm eps=1e-6，统计 FP32 再转计算 dtype。原 Norm 不变。

$$
S_0=0,\qquad \eta=1/128,\qquad
S_t=S_{t-1}+\eta\phi(k_t)^Tv_t,\qquad m_t=\phi(q_t)S_t.
$$

先写当前 K/V 再读，包含 j≤t。无 learnable S0、MSE delta-rule、分母、归一化写回、遗忘、动量或推理 optimizer。S 是请求私有状态，FP32 累积，不入 AdamW、不跨样本保存、不在 tile 间 detach。

gate 是 `Linear(4096,32,bias=False)`，零初始化，无 sigmoid/tanh，无序列平均：

$$
g_t=W_gx_t,\qquad r_{t,h}=g_{t,h}\mathrm{RMSNorm}(m_{t,h}).
$$

gate 打开后验证 feature map 梯度；A→B 不重置 gate。首层 MSA 不为后续 TTT 初始化非零状态。零 gate 的 Local-32 不等于 Full Attention。

计算 tile 默认 C=64，测试16/32/64/128，不改变数学可见性：

$$
\Delta S_c=\widehat K_c^T(\eta V_c),\quad
S_c^{in}=S_0+\sum_{b<c}\Delta S_b,
$$
$$
M_c=\widehat Q_cS_c^{in}+\mathrm{tril}(\widehat Q_c\widehat K_c^T)(\eta V_c).
$$

不得缺第二项、加 softmax 或物化全 token 五维状态。官方 chunk-exclusive 和论文附录 chunk-inclusive 不作为本 token-inclusive 算子的等价目标。

## 4. Local-32、FLA、缓存和真实后端

Local 可见性按展开后的有效 token 0-based p 定义：j≤t 且 floor(pj/32)=floor(pt/32)。不是滑动窗口；图文交界不断开，padding 不计数，缩放 $1/\sqrt{128}$。Local 与 TTT 可以重复读取同一历史位置。

训练/prefill 将有效 token 分成不重叠块并合并到 batch 维，实际批量 SDPA 的 attention 矩阵最大32×32，再 scatter 回原布局。禁止全段 T×T mask 模拟局部，禁止逐 token/逐块 Python attention 循环。尾块正确mask。分段 prefill 接续未满 Local 块，decode 只读本块≤32 KV。

reference 仅用于小测。FLA 封装接收原始 V，内部除128一次；显式 scale=1、normalize=False。当前 linear_attn 无 chunk_size 公共参数，tile 变体使用等价无gate simple_gla 接口，不硬传未知参数。varlen 展平需独立序列边界。后端不兼容报错，不用 reference 静默跑正式训练。

HybridCache：全局每样本 seen/next position/finished；MSA 完整 KV；TTT FP32 S + 当前 Local 块 KV。无 pending chunk，tail 全部入 S；Local 边界只清 KV 不清 S。K 只做一次 RoPE，位置不依赖第0层或Local长度。图像只在 prompt 编码一次，decode 只算新 token。支持 greedy、num_beams=1、异长 batch、选择/重排及finished停止写；不支持 beam/speculative 时明确报错。

core 显式输入/输出状态，无 self.state；训练 use_cache=False 不关闭更新。冻结基座不得 no_grad 整个 decoder。non-reentrant block checkpointing 与关闭时梯度一致，重算不写推理缓存。

GPU smoke 用 profiler/CUDA events 确认实际 SDPA/FLA kernel、块形状及有无意外全长 attention；预热后报告 QKV/RoPE、Local、feature、Prefix-TTT、gate/readout、o_proj、packing/scatter、cache 和总 prefill/decode 耗时，区分 profiler开销与正式计时。没有 GPU 不声称已验证这些项。

## 5. 数据、标签归因与训练量

初查665298条：624610单图、40688纯文本，不等于最终有效训练量。完整检查缺图/解码、schema、重复组、来源、真实展开长度与有效监督。

三类问题分开：

1. 数据问题：缺图、损坏、结构/原始回答缺失，有证据后统一排除。
2. 协议不适用：正确预处理后视觉段被2048上限截断或回答全部落在截断外，记录原因后统一排除。
3. 预处理错误：tokenizer/template/角色/token边界/mask错误导致监督丢失，必须修复并重审，不能当坏样本删除。

诊断保留原角色/回答、格式化prompt、token IDs和监督区间、展开前后及截断后 shifted target 数、预处理版本。原始有回答却丢监督的样本阻塞相关预处理验收。fixtures覆盖图文/纯文本、单多轮、空回答、特殊token和截断边界。

不跨样本packing。训练右padding，左右padding均测试。image mask不能由labels=-100推断，text为所有有效非视觉位置。展开后总长≤2048，非“2048文本+图像”；视觉段完整且至少一个 shifted assistant target。logits[t]监督labels[t+1]。使用同一manifest，不各实验静默跳样本。

按图像/原对话关联组留出约2048 dev，整组可略超并记录；纯文本内容哈希分组。dev仅未参与本次适配，不声称基座未见。A从train固定分层抽50000条，允许与B train重叠，不与dev重叠。保存集合/顺序/配置哈希。

microbatch=1，有效batch128，accumulation=128/(world_size×microbatch)，不整除报错，末批不补重复样本。A每样本先模态均衡再step样本平均；B按整个accumulation group全部有效shifted targets总数平均，考虑DDP平均world size因子。做累积对等效大batch梯度测试。恢复保存全局游标，不换microbatch重开epoch。

## 6. 两阶段、整模型诊断和公平 Pilot

### 阶段 A

24候选层含第0层，主模型仅23层。教师原完整LLaVA eval/no_grad，Full输出继续主干，旁路不能污染后续教师。相同教师QKV、o_proj之前：

$$
\Delta_{\ell,i}=A^{full}_{\ell,i}-A^{local32}_{\ell,i},\qquad
R_{\theta,\ell,i}\approx\operatorname{stopgrad}(\Delta_{\ell,i}).
$$

不是block差，也不能仅提远程V。每样本非空模态m：

$$
e_{\ell,i,m}=\frac{\operatorname{mean}_{t\in m,h,d}(R-\Delta)^2}
{\operatorname{mean}_{t\in m,h,d}(A^{full})^2+10^{-6}}.
$$

双模态各1/2，单模态权重1。层内样本均值、层间loss相加反传，日志另报层平均；不除24。按层单独clip=1。教师逐层推进、旁路enable_grad/backward后释放，不保留24层图、不缓存全数据QKV、不用inference-mode target。

只训A/B/gate，50000条一遍约391更新；末批实际归一化。保存24层参数、种子/schema、逐层/模态raw/normalized误差与gate/state诊断。A→B只迁移参数，optimizer/scheduler新建。

### A 后整模型切换诊断

安装A9-T23加载23层A参数，LoRA零增量，固定dev子集比较E0：全有效assistant CE/KL、图文生成、prefill/decode、逐层gate/state/readout尺度、mask/cache/save-reload。区分逐层拟合与整模型组合误差。质量下降是研究结果，不设benchmark门槛，不自动改结构/损失。

### 阶段 B 与 Pilot

全部32层LLM q/k/v/o及gate/up/down投影注入标准LoRA：r32/alpha64/dropout0/bias none/零增量，完整路径白名单，不匹配ViT。TTT直接训练，不再套LoRA。所有初始化后重新冻结审计，按参数ID去重，保存全名/shape/dtype/grad/group。预计23层新增约27.1M、LoRA约80M，以实际为准。新参数FP32 master+BF16 autocast，基座BF16，S FP32。

唯一目标为全部有效shifted assistant targets的token平均CE；无KL/MSE附加训练loss，教师不常驻。dev teacher KL仅诊断，覆盖全部有效回答位置。

E1/E2采用相同B train顺序、分组、step及Pilot预算。B开始前按完整train一遍设total_steps，warmup/cosine基于完整B。处理≥50000条后的首个完整step保存，通常50048。两者都完成Pilot并固定dev比较后，分别恢复各自完整状态继续剩余B。不得重置scheduler或再跑全epoch。E2的A额外计算单列，不称总算力相等。

| 超参数 | A | B |
|---|---|---|
| 新模块LR | 1e-4 | 1e-4 |
| LoRA LR | 无 | 2e-5 |
| AdamW betas/eps | (0.9,0.95)/1e-8 | 同左 |
| weight decay | 矩阵0.01，bias/Norm0 | 同左 |
| scheduler/warmup | cosine/阶段步数3% | cosine/完整B步数3% |
| clip | 各候选层1.0 | 全trainable合计1.0 |
| batch/seed | 128/42 | 128/42，LoRA独立确定种子 |

checkpoint包括模型/optimizer/scheduler/RNG/global_step/samples_seen/数据游标。NaN、因果错误、损坏数据、真实OOM等是阻塞，不能悄悄缩图/缩序列/截断梯度或升级全量微调。

## 7. 实验矩阵和评测

E0=原32MSA不训练；E1=32MSA LoRA一遍B；E2=A9-T23 A+B。先E0、A与切换诊断，再两者同预算Pilot，最后连续完成完整B。

原定小预算E3 anchors=[1,3,7,11,15,19,23,27,31]；E4=[3,7,11,15,19,23,27,31]；E5按A9 anchors其余Local-only无A。E3/E4复用对应A候选、LoRA重新零增量，不从E2换层。与E2相同Pilot预算和完整B规划scheduler；无等预算完整训练不作正式结构优劣结论。仅按需做固定dev每32token重置S/修改远历史KV等机制干预，明确分布改变，首轮不扩展完整消融。

E0提前完成MME/POPE/GQA，锁定代码/split/prompt/processor/greedy num_beams1/max_new_tokens128，后续不按分数调协议。MME报告Perception/Cognition，POPE P/R/F1，GQA accuracy，不混量纲平均。包内cache路径映射另存，不修改来源manifest。保存每样本ID/答案/raw/解析状态，失败截断不静默剔除。最终主结果用B结束checkpoint。

效率：展开prefill1024/2048、batch1/4、固定decode128；总位置不超基座有效上限，否则所有模型统一调整并记录，不暗改RoPE。编译预热+同步，median/p90、allocated/reserved、真实cache bytes、samples/s和有效tokens/s；LLM-only与含ViT/projector端到端分开。质量与固定工作量性能输出分开，精度和LoRA merge策略相同，先验证merge前后。

缓存估算每TTT S=2MiB/样本，Local KV≤0.5MiB；N2048的A9约345.5MiB vs原1024MiB，仅容量估算非实测。两张A100 80GB是预算，GPU ID由用户指定后配置，microbatch1先实测接近2048的forward/backward/step/save-resume，留15%–20%余量，A/B/eval分别估时。

## 8. 验收、调度解耦与交付

算子长度1/2/31/32/33/63/64/65/127/128/129、N<C、随机非零initial：顺序/分块/FLA输出+final+Q/K/V/S0梯度，tile变化、未来扰动不影响过去、过去影响未来、padding等价、任意分段/tail、checkpoint梯度。因果fixture gate非零。FP64 rtol1e-9/atol1e-10，FP32 rtol1e-4/atol1e-5。BF16对相同量化输入FP32 oracle，relative Frobenius输出≤2%、state≤1%、grad≤5%，同时报max abs，超限定位不放宽。整模型容差结合原MSA路径基线锁定，不由benchmark反调。

真实GPU smoke用128条训练样本，图文/文本、左右padding、异长/独立、截断、请求重置、保存重载、零增量原模型回归、冻结参数未变、新参数有限梯度、gate打开后feature梯度、后部回答loss指导早期视觉KV。debug权重丢弃。无GPU仅标CPU测试通过，生成/显存/性能未运行。

三条解耦工作线：模型闭环；SQLite/systemd；文档审计。最小账本与机器可读配置日志即时记录，完整论文解读/教材不阻塞smoke。小型smoke可由主线程直接以相同argv/env运行，正式长程任务必须调度器。

参考FastWAM `c13e1534ece96093c95e0bff48d4ee6506ec40d9` MIT标准库骨架：单consumer锁、固定manifest、SQLite事务原子占全部GPU、显式env/argv shell=False、PID身份、日志/退出码、成功skip、失败不自动retry、timeout/中断进程组清理。只补必要resume和多卡，不做动态追加/UI/优先级。SQLite放运行主机可靠本地文件系统，产物可共享盘。正式用户systemd服务绝对uv、环境显式、KillMode控制组；检查资源/并发/env/成功失败超时中断恢复及脱终端存活。

当前systemd包已安装但userbus不可用；记录限制，不伪装服务验收成功，不阻塞CPU闭环。GPU重启交接需恢复本目录主账本、检查worktree/队列/既有成功结果、验证硬件和环境，再继续，不重复成功任务。

## 9. 论文阅读与文档

本地四篇：KV Binding（v4，20页）、Test-Time Training Done Right（v1，32页）、Spatial-TTT（19页）、ViT3（11页主文）。分别记录问题/动机/推导/实验/局限和我们的分析，不能把论文配置当本项目必需条件。

关键阅读边界：LaCT硬件收益依赖其大chunk/state；KV Binding固定feature最后矩阵特例可推加性算子，但chunk-exclusive代码非token-prefix；Spatial共享QKV/anchors/零gate可参考，pending/tail只读和视频卷积不沿用；ViT3非因果整图更新不直接用于decoder，其梯度洞见用于K/V反传检查。含Muon求和变换的数学疑点单列，不影响本项目排除Muon后的定义。不声称读过未提供附录或未视觉核对的图形细节。

本目录保存任务书、主账本、repo/environment审计、分章教学与阶段总结；图像随引用文档复制到images并相对引用。明确区分已实现、已执行、待验证；原始日志在artifacts，关键结论和测试摘要归档到文档。结束CPU阶段提供重启恢复入口，不将未实现的GPU/整模型功能标为完成。

## 10. 任务书参考入口

这些链接用于来源定位；实际实现锁本地commit，不将在线main随时间变化的内容当作已运行版本。LoLCATs仅作为attention transfer→low-rank adaptation思路参考，具体残差目标、布局和预算由本任务书固定。

- [KV Binding官方入口](https://github.com/nv-tlabs/tttla)
- [官方chunk-exclusive并行变体](https://github.com/JunchenLiu77/LaCT/blob/tttla/lact_llm/lact_model/_ttt_operation_impl/only_w1_no_wn_parallel.py)
- Spatial-TTT：本地PDF第5、7页的共享QKV/anchor/迁移初始化，另见paper_reading.md。
- [FLA causal linear attention](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/linear_attn/chunk.py)
- [PyTorch checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html)
- [PEFT LoRA](https://huggingface.co/docs/peft/en/developer_guides/lora)
- [LLaVA多模态展开](https://github.com/haotian-liu/LLaVA/blob/main/llava/model/llava_arch.py)
- [LoLCATs](https://github.com/HazyResearch/lolcats)
