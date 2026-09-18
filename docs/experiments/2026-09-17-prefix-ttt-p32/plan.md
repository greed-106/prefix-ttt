# P32 纯 Prefix-TTT 主模型训练计划（`full-ttt` 分支）

启动日期 2026-09-17。本目录是「把 LLaVA-1.5-7B 的 32 个 LLM 子层全部换成纯 Prefix-TTT」这一长程实验的稳定目录。
前一阶段（E0/E2/E2-full 混合布局的训练、评测与推理优化）的账本在
`docs/experiments/2026-09-10-prefix-ttt-h100/` 与 `docs/experiments/2026-09-12-prefix-ttt-fullft/`，保持原样，本目录不复制其内容。

## 0. 起点与事实（已勘察确认）

- 当前 `h100` @ `60be803`，工作树干净；E0/E2/E2-full 已训完并评测：E0 MME 1479.6432/349.2857、POPE 0.8548/0.8397；E2 1429.4893/278.2143；E2-full 1456.9744/297.5000。
- 现有布局是 A9-T23（9 个全注意力 anchor + 23 个 Local-32+TTT 层，`hybrid.py:20`、`configs/base.json`）。
- 现有 A 阶段目标是残差 $T\approx A_{\text{Full}}-A_{\text{Local}}$（`transfer.py:66-75`）；读出带 RMS 归一化（`features.py:93-95`）；B 阶段只有 CE、没有教师 KL（`sft.py:217-223`）。
- 数据固定且已审计：train 663248 / dev 2048 / A 50000；metadata 层已强制单图样本（`llava_arch.py:240`）。
- 本机 8×H100 全空闲；h100-1 的 8 卡被他人任务占用，本轮不接入。
- benchmark 协议：lmms-eval 0.7.2、batch_size=1、带内打点；跨跑批的延迟数字不可比，成本必须在独占条件下同批采集。

**本轮只做一件事：把 LLaVA-1.5-7B 的 32 个 LLM 子层全部换成纯 Prefix-TTT，并把它训到可用。**

## 1. 目标与判据

| 判据 | 条件 |
| --- | --- |
| 完成性 | A 阶段 `complete=true`、391 步；B 阶段 `complete=true`、5182 步、`samples_seen=663248`；loss/grad_norm 全程有限 |
| 纯度 | 学生 LLM 前向内 `scaled_dot_product_attention` 调用数为 0；缓存只有 `state`，请求级恰好 64 MiB（32×32×128×128×4，FP32） |
| 正确性 | 因果性、前缀等价、生成一致性、状态隔离、梯度路径五项检查全通过（§6） |
| 质量 | MME / POPE 官方分数与成本（独占条件、E0 同条件复测）采集完毕，与 E0 同协议对照；不设人为及格线，也不改配方试凑 |
| 证据 | 逐层迁移诊断（full/vision 误差、state/memory/readout/gate 幅度）与训练轨迹落盘，结论可逐条复查 |

## 2. 分支、目录与边界

1. **第一步就建分支**：确认工作树干净后 `git checkout -b full-ttt`；此步完成前不修改任何代码。
2. 稳定实验目录 `docs/experiments/2026-09-17-prefix-ttt-p32/`（`plan.md`、`ledger.md`、结尾 `experiment_summary.md`）；脚本放 `scripts/experiments/prefix_ttt_p32/`；原始日志/快照放 `artifacts/experiments/prefix_ttt_p32/`。
3. 产物：`/data/shared/weights/prefix-ttt/training/{A-P32,P32}/`、评测 `eval-p32/`。本机训练，h100-1 后续接入时同一命令加环境变量即可（§5.2）。
4. **不动**：`configs/base.json`、`configs/full.json`（既有 checkpoint 的 `config_sha256` 身份）、两个历史文档目录、E2 的 Local-32 代码路径、`third_party/llava`。
5. 不提交、不推送（除当轮明确授权）。

## 3. 代码改造（按子系统）

### 3.1 纯 TTT 算子与模型层

- `ops/features.py`：`readout(x, memory, *, local=None, normalize=True)`。`normalize=False` 时输出 $y_t=g_t m_t$（不再对 $m_t$ 做 RMSNorm）；`normalize=True` 保留旧行为。特征映射 $\phi(x)=\mathrm{RMSNorm}[\mathrm{SiLU}(xA_\phi)\odot(xB_\phi)]$ **不变**。
- `ops/fused_features.py`：`readout(...)` 增加 `HAS_LOCAL` / `NORMALIZE` 两个 `tl.constexpr`（`HAS_LOCAL=0` 时不加载 local，指针传 `memory`），小形状回调路径同步。只影响推理。
- `model/hybrid.py`：新增 `LAYOUT_ANCHORS = {'E2': FULL_ATTENTION_LAYERS, 'P32': ()}`。**唯一布局开关是 anchor 列表**：`install_prefix_ttt(..., full_attention_layers=...)` 内部推得 `pure = not full_attention_layers`，据此设每层 `use_local=not pure`、`readout_normalize=not pure`。传空列表即纯 TTT，不存在「忘记传开关而静默退回归一化读出」的可能；E2 数字与回归测试因此保持原样。
- `PrefixTTTAttention.forward`：`use_local=False` 时完全不调用 `local_attention*`/`dense_local`，只写 `LayerState(state=...)`（不分配 key/value/window），读出走 `local=None, normalize=False`；prefill/decode/fused 三条分支按同一开关收口。

### 3.2 A 阶段：完整 attention 迁移 + 视觉贡献监督（`transfer.py`、`training.py`）

- `training.py` 新增：
  - `full_attention_transfer_loss(readout, target, valid)`：对教师**完整** attention 输出（原 `o_proj` 的输入）的归一化 MSE，视觉 query 与文本 query 分别按各自教师能量归一化后取平均（沿用现有均衡结构，**去掉 local 项**）。
  - `visual_transfer_loss(readout_visual, target_visual, positions, valid, energy)`：在选定的文本位置上比较 $Y^{S,V}$ 与 $Y^{T,V}$，**分母统一用教师完整输出能量**（不除以可能很小的视觉贡献能量）。
  - 删除因改动失效的 `residual_transfer_loss` 及其测试，并在账本记录该语义变更。
- 教师视觉贡献 $Y^{T,V}_t=\sum_{j\le t,\,j\in\mathcal V}a^T_{tj}v_j$：在 `o_proj` 前钩子里用**同一个** `attention_mask`、同一个 `scale=attention.scaling`、同一 `is_causal` 判据做一次 `F.scaled_dot_product_attention(q,k,v\odot m_{\text{vis}})`，softmax 分母不变；**不做**「只留视觉 K/V 重新 softmax」。同一次前向内用同一套参数重算 `sdpa(q,k,v)` 与捕获输出比对（只记不训练、超差即报错），证明 mask/scale 语义与教师一致。
- 学生视觉贡献 $S^{V}_t=\frac1{128}\sum_{j\le t,j\in\mathcal V}\phi(k_j)^\top v_j$，$Y^{S,V}_t=g_t\phi(q_t)S^V_t$：同一 `fla_prefix` 调用把 $v$ 在非视觉位置置零再跑一遍（第二次前缀 pass），读出沿用同一 $g_t$。
- 视觉监督位置 $P_b$：`valid & ~image_token_mask` 且位于最后一张图像之后的文本位置中，按索引顺序取**最多 64 个**；无图像样本不参与视觉项（仍参与 full 项）。这是本轮已标注的固定设计选择。
- `transfer.py`：钩子改为新目标；`identity` 增加 `objective='full_attention+visual'`（常量，非浮点旋钮），旧 A checkpoint 因此不可 resume（已完成，属有意保护）；每层诊断在现有基础上增加 `memory_rms`、`readout_rms`（守住「读出取消 RMS 后幅度随前缀增长」这一风险），其余诊断保留。

### 3.3 B 阶段：CE + 教师 KL（`sft.py`）

$$
\mathcal L_B=\mathcal L_{\text{CE}}+\lambda_{KD}\tau^2\,\mathrm{Mean}_{t\in\mathcal A}D_{\mathrm{KL}}\big(p^T_t(\tau)\Vert p^S_t(\tau)\big),\qquad \lambda_{KD}=1,\ \tau=1
$$

- 载入第二份冻结教师（bf16、`eval()`、`requires_grad_(False)`），与学生共用同一份 `inputs_embeds`/`attention_mask`；教师前向在 `no_grad + autocast(bf16)` 内完成，随即算完 KL 并释放教师 logits（逐 micro-batch，峰值只多一份 `[B,T,32064]`）。
- KL 与 CE 共用 shifted-label 对齐与同一全局 target 计数归一化（`world / global_target_count`），只在 `labels[:,1:] != -100` 处计算；教师只给目标分布，不注入 hidden states、不生成替代回答。
- 触发条件 `config['training']['kd_weight'] > 0`：`base.json`/`full.json` 无此键 → 旧路径逐字节不变；`configs/p32.json` 设为 1.0。
- 训练范围：冻结 ViT/projector/LLM 原权重，训练 32 层的特征映射、gate 与全部 32 层 LoRA（rank 32、alpha 64、dropout 0，7 个投影）。学生 decoder **不进** `no_grad()`。优化器分组、手动梯度 all_reduce、`clip_grad_norm_`、调度器、checkpoint 全部复用现成代码。

### 3.4 配置、身份与入口

- 新增 `configs/p32.json`：`full_attention_layers: []`、`candidate_ttt_layers: [0..31]`、`training.micro_batch_size: 8`、`training.kd_weight: 1.0`、`training.kd_temperature: 1.0`，其余与 `base.json` 同值。
- `config.py`：anchor 校验改为「必须等于 `LAYOUT_ANCHORS` 中某一个（`()` 或 E2 名单）」，继续校验 `candidate_ttt_layers ∪ anchors == range(32)`；`kd_weight`/`kd_temperature` 作为「出现才校验」的镜像键（与 `base_lr` 同机制）。
- `sft.py`：`--layout` 增加 `P32`，安装时传 `config['full_attention_layers']`；`identity` 增加 `kd_weight`、`kd_temperature`。
- `lmms_model.py`：接受 `layout='P32'`，用 `LAYOUT_ANCHORS[state['layout']]` 安装；LoRA 分支保持「装载 → `load_trainable` → 合并 → 去 PEFT 包装」。`instrument.py` 无需改动（全部分桶自然落入 `ttt_state_bytes`）。

### 3.5 测试（只做能守住主训练正确性的最小集）

- 纯度：LLM 前向内 `F.scaled_dot_product_attention` 调用数为 0；prefill 后缓存只有 `state`，字节数 = batch×64 MiB。
- 因果性 / 前缀等价 / 生成一致性 / 状态隔离：把现有 `test_model_hybrid.py`、`test_prefill_integration.py` 用例按 `anchors=()` 参数化（不新增重复用例）。
- 梯度路径：后部回答位置的 CE/KL 梯度能非零传到视觉位置的 $k_j$ 写入。
- 新损失与 KL 的 CPU 单测（归一化口径、mask 对齐、梯度）；P32 checkpoint 的加载/合并用例；`configs/p32.json` 的校验用例。
- 目标：`uv run --locked pytest -m 'not gpu'` 全绿（现有 175 项不回归）+ GPU 套件（FLA 验收 + P32 smoke）通过。

## 4. 训练配置

| 项 | A 阶段 | B 阶段 |
| --- | --- | --- |
| 数据 | `manifest['A']`（50000，一遍，391 步） | `manifest['train']`（663248，一遍，5182 步） |
| 可训练 | 32 层特征映射 + gate | 同上 + 32 层 LoRA |
| 教师 | 冻结原 LLaVA（给 $Y^T$、$Y^{T,V}$） | 冻结原 LLaVA（只给目标分布） |
| 损失 | $\mathcal L_{\text{full}}+\mathcal L_{\text{vision}}$ | CE + KL（$\lambda_{KD}=1,\tau=1$） |
| 学习率 | 新模块 $10^{-4}$；3% warmup + cosine（按各自完整轨迹） | 新模块 $10^{-4}$、LoRA $2\times10^{-5}$；同调度 |
| 其他 | AdamW (0.9,0.95)、eps $10^{-8}$、矩阵 wd 0.01、clip 1.0、BF16 主计算 + FP32 状态/新参数 master | 同左 |
| 有效 batch / 长度 | 128 / 2048（含 576 视觉 token） | 同左 |
| Pilot | — | 391 步（5 万样本）存档，`--resume` 续跑剩余样本，**不清零 gate、不重新 warmup** |

## 5. 执行顺序

1. **建分支 `full-ttt`** → 建实验目录并把本计划写入 `plan.md`。
2. 实现 §3.1–§3.4 → CPU 套件全绿。
3. 3 步 smoke（Stage A 与 Stage B 各一次，输出到 scratch，`--max-steps`）→ 记录峰值显存、步时、纯度与五项检查；**据此定 micro_batch_size**（8 为默认，超预算则 4，累积步数自动相应调整）。
4. Stage A 正式跑（391 步，`training/A-P32/`）→ 检查逐层误差单调下降、grad 有限、`complete=true`。
5. Stage B 正式跑（5182 步，`training/P32/`）→ 391 步 Pilot 存档并检查 loss 与 grad_norm，再 `--resume` 续跑；每 250 步存权重。
6. 评测：MME + POPE（P32，独占 GPU、带内打点），并用**同一条件**复测 E0 以取得可比成本；GQA 可选、不阻塞。
7. 写账本、`experiment_summary.md`，更新 `README.md`；教学/推导文档与总结一起补齐。

### 5.1 单机 8 卡为默认

本机 8 卡全空闲，Stage B 用 `--standalone --nproc_per_node=8`（micro 8 → 累积 2），Stage A 同规模。预计：A 30–60 min；B 12–14 h（含教师前向）。GPU ID 由启动脚本的 `CUDA_VISIBLE_DEVICES` 显式指定，训练代码不含卡数假设。

### 5.2 为 16 卡预留空间（不写死单机）

- 复用现有 `scripts/experiments/prefix_ttt_h100/run_multinode.sh`：`NNODES` / `GPUS` / `NODE_RANK` / `MASTER_HOST` / `MASTER_PORT` 全部走环境变量，`NNODES=1` 即单机运行，`NNODES=2` 时 h100-1 执行同一脚本、同一份配置。
- 累积步数由 `accumulation_steps(world, micro)` 计算（world=8→2、16→1、4→4，均整除 128），配置与脚本中不出现固定 world size；`micro_batch_size`、`dataloader_workers` 仍来自配置。
- 账本已实测「目标函数与 micro 分组无关」（跨 world 4→16 的第 391 步 CE 相差 0.4%），因此 h100-1 空闲后**可以**从中断处用 16 卡续跑；默认保持单次运行内 world size 不变，若切换则在账本记录。

## 6. 正式训练前的关键检查

| 检查 | 必须满足 |
| --- | --- |
| 因果性 | 非零 gate 下未来 token 不影响更早输出 |
| 前缀等价 | 顺序、分块、不同 tile 的输出与梯度一致（reference vs FLA） |
| 生成一致性 | 整段 prefill、任意切段、逐 token decode 一致 |
| 状态隔离 | padding、finished、新请求、异长 batch 正确处理 |
| 梯度路径 | 后部回答 loss 能传回前部视觉写入 |
| 最终纯度 | 学生不调用 Full/SWA/Local softmax attention，不分配历史 KV（查真正执行的运算，不是模块名） |

## 7. 风险与缓解

| 风险 | 缓解 |
| --- | --- |
| 读出取消 RMS 后幅度随前缀增长 | 逐层记录 `state_rms`/`memory_rms`/`readout_rms`；不在主训练里静默切换归一化版本 |
| 视觉监督的第二次前缀 pass | 只在 A 阶段、只在 32 层；实测步时后决定是否需要更省的写法 |
| 学生 + 教师同卡显存 | 逐 micro-batch 释放教师 logits；micro 8→4 是已验证等价的安全阀；smoke 先实测 |
| 教师 mask/scale 与捕获输出不一致 | 每次前向重算 `sdpa(q,k,v)` 比对，超差即报错 |
| 训练发散或非有限 | 保留 `torch.isfinite` 硬检查；预算内允许一次降 lr 重跑并如实记录 |
| 成本口径被并发污染 | 正式成本测量只在独占 GPU、单任务条件下做，E0 同条件复测 |

## 8. 本轮不做（后置，不阻塞主训练）

- **P32-Basic（去掉视觉监督的消融）**：本轮不实现、不训练；视觉监督作为主模型的一部分直接使用，消融留到主结果出来之后。
- Full-MSA-LoRA 对照、P32 全量微调、H32/A9-T23 新配方、E1/E2 的重训或复现。
- 三个迁移节点的独立 dev 评测工具与生成样例协议（等主模型训完再补；训练日志已提供逐层与逐步证据）。
- 不新增数据集/QA/caption，不用教师回答替换标签，不制造长上下文，不改 prompt 与评测协议。
- 不做 CUDA graph，不引入新依赖或改库版本，不改 `third_party/llava`。

## 9. 假设（如有不符请指出）

1. 测试床为本机 8×H100；任务书中「两张 A100 80GB」按本轮确认作废，h100-1 暂不接入。
2. 运行方式沿用 `torchrun`（单机 `--standalone`，双机用现成的 `run_multinode.sh`）；SQLite 调度器当前未运行且只支持单机分卡，本轮不经过它。
3. 视觉监督位置规则固定为「最后一张图像之后的文本位置中前 64 个」，$\lambda_V=1$ 不做浮点旋钮。
4. 训练超参（lr、batch、长度、调度）沿用既有已验证配方，只改架构与损失，便于与 E0/E2/E2-full 对照。
