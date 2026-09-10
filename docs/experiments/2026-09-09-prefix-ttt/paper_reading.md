# 从 KV Binding 到 Prefix-TTT：四篇论文的研究脉络与适用边界

本文以仓库内 PDF 为依据，页码均为 PDF 阅读器的 1-based 物理页码；ViT3 的 PDF 第 1 页对应会议印刷页 51。论文中的测量值是作者报告，不是 Prefix-TTT 的复现实验结果。本阶段重新核对了主要方法、关键附录及实验表格；GPU 实验尚未执行。

本地来源是 [KV Binding](../../paper/Test-Time%20Training%20with%20KV%20Binding%20Is%20Secretly%20Linear%20Attention.pdf)、[Test-Time Training Done Right](../../paper/Test-Time%20Training%20Done%20Right.pdf)、[Spatial-TTT](../../paper/Spatial-TTT%20Streaming%20Visual-based%20Spatial%20Intelligence%20with%20Test-Time%20Training.pdf) 和 [ViT3](../../paper/Han_ViT3_Unlocking_Test-Time_Training_in_Vision_CVPR_2026_paper.pdf)。前两篇本地版本分别标注 arXiv:2602.21204v4 和 arXiv:2505.23884v1；不将在线最新版内容无声混入本地依据。

## 第一章：要压缩的究竟是什么

Full Attention 保存上下文每个位置的 K/V，再由当前 query 为这些历史位置分配权重。这样保留了逐位置可寻址的信息，但解码缓存随序列增长。TTT 的出发点是把历史压缩进一个固定大小函数的参数：以 K 为输入、V 为学习信号更新这个函数，再对 Q 求值。这里的“训练”发生在一次请求内部，更新的是请求私有状态，并不意味着推理时必须启动 AdamW，也不意味着将用户输入写入永久模型参数。

四篇论文分别回答不同问题。LaCT 关注如何把在线更新组织成 GPU 擅长的计算；KV Binding 追问这些更新是否真的需要复杂的内层优化解释；ViT3 研究非因果视觉任务中哪些内层设计容易学习；Spatial-TTT 则讨论如何把这种状态机制装进已预训练的多模态模型，并用适当监督使状态保留空间信息。

Prefix-TTT 的问题更窄：在完整 LLaVA-1.5-7B 中，固定大小关联状态能否在有限迁移和 LoRA 成本下接替大部分 Full Attention。我们保留所有 token、全部原 FFN 和 9 层 Full Attention，所以“单个 TTT 层状态恒定”不能推出“整个模型缓存恒定”，也不能推出端到端严格线性复杂度。

## 第二章：KV Binding 的化简为什么成立，又在哪里停止成立

### 2.1 从内层更新推导外积状态

KV Binding 第 5 页定理 5.1 假设内层函数最后一层是无 bias 的线性映射：

$$
f(x)=\phi(x;\Theta)W.
$$

对 K/V 关联目标做一次梯度下降时，最后矩阵的梯度必然包含“隐特征的转置乘输出梯度”。若使用负点积目标，输出梯度就是负 V；若我们进一步固定序列内的特征映射参数，便得到：

$$
\mathcal L_{\mathrm{inner}}=-\langle\phi(k)W,v\rangle,\qquad
\nabla_W\mathcal L_{\mathrm{inner}}=-\phi(k)^\top v,
$$

$$
S_t=S_{t-1}+\eta\phi(k_t)^\top v_t,\qquad
m_t=\phi(q_t)S_t.
$$

这里“固定特征映射”只表示单序列内不更新；其参数仍通过最终任务损失接受外层训练。将状态递推展开：

$$
m_t=\phi(q_t)S_0+
\eta\sum_{j\le t}\langle\phi(q_t),\phi(k_j)\rangle v_j.
$$

这个式子直接说明当前输出如何依赖历史 V，也说明为何无需运行一个真正的内层 optimizer。计算外积并累加状态就是该优化步骤的闭式实现。它没有 softmax，也没有对历史权重求和的分母；不同 K 写入同一个有限矩阵，因此会发生关联干扰，不能把状态视为无损 KV 缓存。

论文定理还允许特征映射本身随历史变化，或者将一般损失的输出梯度视为有效 V。但这种“线性形式的读出”并不保证可以使用普通、固定核的并行线性 attention 实现。第 20 页附录 I 明确区分：动态非线性特征和每次更新后的状态归一化，会引入递推依赖，破坏简单求和的结合结构。对 Prefix-TTT 而言，不使用状态写回归一化、动量和 Muon，是为了固定本项目的可验证算子，而不是声称这些机制对所有任务都无效。

### 2.2 “记忆悖论”支持什么结论

论文第 3–4 页观察到：推理时增加内层更新次数虽降低内层 loss，却可能损害下游指标；将下降改为上升后重新训练，模型仍能取得接近甚至略好的表现。第 4 页表 1 中，LaCT-LLM baseline perplexity 为 16.43，gradient-ascent 变体为 16.19。关键限定是上升版本从训练阶段就采用同一规则，并非给训练好的模型临时翻转梯度符号。

我们的理解是：外层模型学习的是一整套数据相关计算，内层 loss 的优化方向和迭代次数也是这套计算的一部分。临时多走几步可能造成分布偏移，负号也可能被可训练投影吸收。这些证据足以反对“内层拟合越好，下游必然越好”的简单解释，但不足以证明所有 TTT 都没有任何记忆功能。论文的经验对象主要是 LaCT 和 ViTTT，理论也有最后层线性、无 bias 的假设，不能扩展成对全部测试时适应方法的否定。

第 8 页表 2 的逐步消融从 16.43 baseline 到仅更新最后矩阵的 15.93，再到完整简化版本的 16.80。后者与最优简化中间点差 0.87，与 baseline 差 0.37；引用结果时必须说清比较对象。第 9 页报告最高约 4 倍的是特定 TTT 层吞吐提升，端到端训练提升为 1.19 倍。它们都不是 Prefix-TTT 的速度预估。

### 2.3 三种 prefix 必须分开

论文第 18–19 页附录 H 对 chunk 输出使用包含当前整个 chunk 更新的状态。这样的 chunk-inclusive 表达若直接用于逐 token 自回归，会让同一 chunk 早期位置读到后续位置的信息。它可以适合其他可见性定义，但不能直接成为我们的因果 oracle。

本地 KV Binding 修改版 LaCT 的 `only_w1_no_wn_parallel.py` 第 93–94 行则使用 `cumsum(dw1)-dw1`，生成更新当前 chunk 之前的状态。它是 chunk-exclusive：同一 chunk 每个位置读取同一个旧状态。此形式避免当前 chunk 未来泄漏，但没有本项目要求的块内当前前缀。

Prefix-TTT 明确定义 token-inclusive。一个执行 tile 内必须同时存在历史块状态项和块内下三角项：

$$
M_c=\widehat Q_cS_c^{\mathrm{in}}+
\operatorname{tril}(\widehat Q_c\widehat K_c^\top)(\eta V_c).
$$

例如序列只有两个 token、尚未跨 tile、初始状态为零时，chunk-exclusive 读出仍是零，而 token-inclusive 的第一个位置已写入自己的 K/V，第二个位置包含两次写入。这是短序列“状态有没有真实工作”的直接测试，不必等 benchmark 才发现差异。改变执行 tile 不能改变可见性；修改同 tile 的未来 K/V 不得影响较早位置，修改过去 K/V 则应影响后续位置。

本章留下的研究问题是：在有限状态容量和固定特征映射下，跨块信息能保留多少。首轮仅采用既定 Local-only 和临时状态重置诊断，不扩展为特征维度、优化器或架构的大搜索。

## 第三章：LaCT 的硬件论证不是“大 chunk 永远更好”

### 3.1 小矩阵为什么可能比大矩阵更慢

Done Right 第 3 页分析大小为 $h\times h$ 的状态乘以 $b\times h$ 的 chunk。按照 BF16 的理想读写量，其计算与访存比为：

$$
\rho=\frac{2h^2b}{2h^2+4hb}
=\frac{b}{1+2b/h}\le\min(h/2,b).
$$

当 chunk 很小或状态很小时，算术强度有限，GPU 可能主要等待数据搬运。把很多在线更新聚合成大矩阵操作，可以摊薄状态读写开销，也让大状态和更复杂更新规则更易执行。论文使用从 2K 到 1M token 的不同任务 chunk，并讨论 context parallelism；这些属于任务数据结构和算子共同决定的执行设计。

第 4–7 页区分 update/apply 顺序及等效可见性。语言模型采用 shifted block-wise causal，让当前 chunk 读取历史状态，再用共享 QKV 的 sliding-window attention 补足局部因果关系。新视角合成中可以先聚合多个输入视图再读取目标视图，视频扩散则有干净与带噪 chunk 的专用顺序。不同数据不能只换 chunk size 就认为算子相同。

### 3.2 实验说明的收益与迁移限制

第 8 页表 2 的新视角合成实验，在 A100、48 张 512×512 输入图像、约 196K 输入 token 上报告 full-attention prefill 16.1 秒与 LaCT 1.4 秒。这是长图像集合压缩任务，并非 LLaVA 的 2048-token 图文对话。

第 8–9 页语言实验包含 760M/3B 模型、40B/60B 训练 token、32768 序列长度及 2048/4096 sliding window。表 3 的 3B 吞吐中，Full Transformer 为 4.1K tokens/s，LaCT GD 为 5.0K，Muon 为 4.3K；更复杂更新未必更快，其目标是质量与状态表达能力的取舍。第 10 页消融报告大状态和 Muon 在相应设置中有益，这与 KV Binding 在另一套消融中发现简化有效并不构成简单逻辑矛盾：训练预算、状态规模、测量对象和逐步修改组合都不同。

Prefix-TTT 不借用其 2K chunk 更新语义，也不引入 Muon。我们借用的是“执行形状必须能让硬件并行”的标准。因此 Local-32 必须真实收集为最多 32 个位置的块批量矩阵运算，不能构造整段 attention 再套稀疏 mask，也不能逐块 Python 调用冒充高效实现。

即使 Prefix-TTT 的复杂度较低，32×32 小块也可能受到启动、packing、scatter 和读写开销限制。GPU smoke 应同时观察真实 SDPA/FLA kernel、各分支耗时与整体耗时。保留原 FFN、9 个 anchor、ViT 和 projector 意味着 attention 层加速不会等比例成为端到端加速。这一验证属于首轮既定工作，不需要添加新的性能模型结构。

## 第四章：Spatial-TTT 教会我们如何迁移，而不是规定层比例

### 4.1 预训练知识与新状态之间的接口

Spatial-TTT 第 5 页指出，直接替换全部 attention 可能破坏预训练跨模态语义，所以使用混合 TTT 与 Full Attention anchor。TTT 分支与局部 attention 共享 QKV，让新模块接收原模型已有的表示，而不是另学一套输入空间。第 7 页进一步说明基座为 Qwen3-VL-2B-Instruct，gate 零初始化，scale/shift 分别初始化为一和零。

但其初始化保持原模型行为还依赖最初足够大的 attention window：论文先令窗口覆盖训练上下文，再退火到 chunk 大小。Prefix-TTT 从 Local-32 起步，即使 gate 为零，输出也只是 Local Attention，绝不等价于原 Full Attention。因此我们需要阶段 A 的 attention 残差迁移，以及 A 后提前执行的整模型切换诊断。旁路在教师分布上拟合得好，不保证所有层一起替换后仍有相同输入分布。

原任务书的 A9 布局、首尾 anchor 和额外预热第 0 层是本项目假设，Spatial-TTT 的固定 3:1 并不能证明它最优。复用原 QKV 和 o_proj 也不代表任意替换都能保存原语义，必须用完整基座回归与最终质量评测检验。

### 4.2 监督决定状态学会保存什么

第 6–7 页强调稀疏空间 QA 的短答案可能只监督少量局部信息，因此作者加入完整场景描述，使状态得到覆盖全场景的训练信号，再做空间 VQA。这提出一个重要机制问题：早期视觉 K/V 的写入，能否收到后部答案的有效梯度。

Prefix-TTT 不复用这些空间视频数据，也不新增描述任务。我们在既定图文指令数据上检查 shifted assistant targets，并验证后部答案 loss 能反向指导早期视觉写入。user/system/视觉 token 没有直接 CE 标签，不代表它们可以从状态更新中删除；它们可以通过影响后续输出接受间接监督。

本地 PDF 第 7 页 §3.4 提到 2M 空间 VQA，而同页 §4.1 及第 8 页将第二阶段数据描述为 3M。本文保留这一版本内表述差异，不替作者选择一个数字。无论是哪一数字，其训练规模与我们的 50K A 加一遍 LLaVA 指令 SFT 都不是相同预算。

### 4.3 缓存与实验证据的边界

第 7 页的 dual cache 包含滑动窗口 KV 和待满 chunk 才提交更新的 pending KV。这适配其 chunk 延迟语义；Prefix-TTT 不采用 pending chunk，半 tile prefill 结束时最终状态也必须吸收所有有效 prompt token。Local 块跨界只清空 Local KV，不清除全局状态；全局位置不能从 Local 缓存长度推导。

第 8–11 页报告 VSI-Bench 64.4、MindCube-Tiny 76.2；VSI-Bench 去掉 anchor 后降至 53.9，去掉空间卷积降至 62.1，去掉 dense data 降至 61.3。这些是其数据、模型和训练下的消融结果，支持“集成与监督很重要”，不证明卷积或相同比例是 LLaVA 改造的必要条件。

论文第 11 页表 5 报告 1024 帧时 Spatial-TTT 11.9 GB、799.4 TFLOPs，对照 Qwen3-VL-2B 21.2 GB、1403.1 TFLOPs。论文有“近线性”与“constant-memory streaming”的表述，但保留完整历史的 anchor 本身仍带来随长度增长的 KV 与二次 prefill 项。对 Prefix-TTT 必须按实际缓存 tensor 与计算路径报告，不将混合系统表述为严格常数缓存。我们的缓存容量估算也不能当作实测训练显存。

本章的研究方向限于既定同预算 E1/E2 Pilot、A 后整模型诊断及后续小预算 anchor 对照；不扩展到视频、3D 卷积或窗口退火。

## 第五章：ViT3 对梯度链路的提醒比视觉配方更可迁移

### 5.1 为什么必须验证 K/V 梯度

ViT3 第 3–4 页将内层更新视为外层计算图的一部分。设内层预测为 $\widehat V$，则输出任务对 V 投影的梯度需要穿过内层 loss 的混合导数。如果使用 MAE，预测误差梯度大多是常值符号，相关混合导数几乎处处为零，值投影可能失去有效信号。

第 4 页表 1 中，dot product 和 MSE 分别达到 78.9% 和 79.2% ImageNet top-1，MAE 为 76.5%。这不是证明所有任务必须用负点积，而是说明前向“可以运行”和外层“可以训练所需路径”是两件事。

Prefix-TTT 的闭式外积实现消除了显式求梯度的内层程序，但没有消除这些依赖：

$$
S_t=S_{t-1}+\eta\phi(k_t)^\top v_t.
$$

后部 loss 必须能穿过状态累加，到达过去的 K/V 和特征映射。逐 tile detach、错误地在 no_grad 下执行整个冻结 decoder、缺失 final-state backward，都会让前向看似正确而训练目标无法到达早期模块。因此算子测试必须包含 K、V 和 initial state 梯度；整模型测试还必须检查早期视觉写入。gate 零初始化时 feature map 最初梯度可能为零，打开 gate 后再要求非零梯度，不能把初始化行为误判为实现故障。

### 5.2 六条视觉经验及其范围

第 4–6 页系统研究内层 loss、batch/epoch、学习率、模型宽度、深度和卷积。单轮整图 full-batch 更新在其视觉实验中表现良好；多步更新会降低吞吐，过多时不稳定。增宽内层可改善效果，但增深未必更好：三层 MLP 训练 loss 更高，作者将其归因为优化困难。简化 GLU 与 depthwise convolution 在表达能力和计算成本之间取得较好折中。

第 7 页最终 ViT3 使用一个卷积 head，其余 head 使用简化 GLU，并在每个样本内做整图、非因果的更新。它不是在语言序列中对每个 token 严格只看前缀。整图 full-batch、学习率 1.0 和卷积适合性的经验不能直接转为 Prefix-TTT 的规则；我们继续固定 eta=1/128、R=128、无卷积、因果 token-prefix。

第 6–8 页实验涵盖 ImageNet 分类、COCO 检测、ADE20K 分割和 DiT 图像生成。作者报告 ViT3-S 81.6% top-1，对照 DeiT-S 79.8%；同时指出分割性能仍低于高度优化的 TransNeXt。第 8 页的 4.6 倍速度和 90.3% 显存降低对应 RTX3090、高分辨率图像、特定小模型，不能用于估算两张 A100 上 LLaVA 的训练或生成成本。

真正可迁移的问题是：固定容量状态如何在有限训练预算内学会保留对任务重要的信息，以及我们是否保留了学习所需的梯度路径。首轮通过既定 feature/gate 诊断和累积梯度一致性测试回答工程部分，质量部分由 E0/E1/E2 实验回答。

## 第六章：将论文思想落实为可证伪的实验

四篇论文共同支持把状态、可见性、硬件执行和外层监督分开分析。Prefix-TTT 因而首先固定数学定义，再分别验证输出与梯度、LLaVA 集成、真实训练与生成。不能仅凭 loss 下降判断成功：错标签可能使监督样本被误删，零 gate 可能掩盖未来泄漏，chunk-exclusive 可能在短序列中完全不使用当前 K/V，完整前缀重算也可能伪装成可工作的 cache。

先完成 E0 质量基线，目的是确认原始模型、预处理和评测都能复现同一计算路径。A 后整模型切换诊断检验逐层残差迁移是否能组合；E1/E2 同预算 Pilot 将额外 SFT 的效果与结构改造区分。两者都使用完整 B 规划的 scheduler，并从自己的 Pilot 断点继续，不重开训练轨迹。

正式报告必须将论文声称、我们的算子推导、CPU 已通过的测试、GPU 尚未运行的验证和最终研究结果分别陈述。预期缓存更小属于结构推算，速度更快必须来自实测；质量下降也是有效结果。当前不存在支持 Prefix-TTT 已超过基线、已获得速度收益或具备首创性的实验结论。
