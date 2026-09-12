# Prefix-TTT 全量微调实验主账本

本目录 `docs/experiments/2026-09-12-prefix-ttt-fullft/` 是「把 E2 的 LoRA 微调换成全量微调」
这一实验项目的稳定目录，启动日期 2026-09-12。跨日继续时持续更新本账本，不再另建目录。

关联目录（保持原样，不追加内容）：

- `docs/experiments/2026-09-10-prefix-ttt-h100/`：本机环境重建、A/B 训练、E0/E2 评测、推理优化与代码审计。
- `docs/experiments/2026-09-11-prefix-ttt-kernel/`：推理 kernel 专项。
- `docs/experiments/2026-09-09-prefix-ttt/`：A40 旧机的历史账本。

## 当前状态

最后更新 2026-09-12 01:50。

- 阶段：**规划完成，等待用户批准后开始实现**。方案见本目录 `plan.md`。
- 目标：同样的数据、manifest、步数与评测协议下，用全量微调替换 E2 的 LoRA，观察 MME / POPE 能否提高。
  主判据 MME perception > 1429.4893（E2），目标 ≥ 1479.6432（E0）。
- 关键决策（用户 2026-09-12 决定）：
  1. 可训练范围：**冻结 ViT，训练 mm_projector + LLM + TTT**。
  2. **只跑 TTT 全量微调一臂**，不跑「无 TTT 的纯 LLaVA 全量微调」对照臂。
  3. 并行实现：初版为「自研 ZeRO-1 风格分片 AdamW」；用户追问「为什么不用现成库」后**改为
     `torch.distributed.optim.ZeroRedundancyOptimizer`（torch 内建 ZeRO-1）+ 原版 `torch.optim.AdamW`
     + fp32 参数**，自研优化器代码降为 0 行。显存账由 31.7 GiB 变为 **54.1 GiB/卡**（约 75%），
     理由与备选记录在 `plan.md` 4.2 节。
- 侦察结论（2026-09-12 01:27–01:50，只读，未改任何文件）：
  - 工作树干净，`h100` 分支 HEAD = `1c61b28`（kernel 实验提交，作者 `Codex`），与 `origin/h100` 一致。
    `1c61b28` 不是本次任务的产物，未触碰。
  - 本机 8×H100 全空闲（0 MiB 占用）；h100-1（`cucloud-server1`）SSH 可达、8 卡全空闲、NFS 已挂载、
    `data/llava-v1.5-assets-v1` 与 `artifacts/cpu/fixed_manifest.json` 都在、venv 可用。
  - 模型 7.1705 B 参数；全量 AdamW 全复制需 106.85 GiB/卡 > 80 GB，必须分片；分片后约 31.7 GiB/卡。
  - bf16 参数直接更新不可行：lr=2e-5 的更新量比 $|w|\approx0.02$ 处 bf16 的 ulp（约 1.5e-4）小 7 倍，
    因此 fp32 master 是正确性要求。
  - `torch.distributed.broadcast_coalesced` 在 torch 2.7.1 **不存在**；`all_reduce_coalesced`、
    `all_gather_into_tensor`、`reduce_scatter_tensor` 存在。方案据此改用 all-gather 回写。
  - h100-1 检出停在 `49acbe2` 且工作树有 6 个脏文件。逐个 SHA256 比对：`configs/base.json` 与本机 HEAD
    完全一致，其余 5 个不匹配任何历史提交，是本机那次重构的中途快照被 rsync 过去的残留，
    **不是第三方改动**；同步前仍先打包备份再重置。
- 运行项：无。尚未启动任何训练或评测任务，未提交任何 GPU 作业。
- 失败项：无。
- 产物位置：本目录 `plan.md`；训练与评测产物路径规划见 `plan.md` 第 8 节（尚未创建）。
- 下一步：等用户批准 → 实现分片优化器与单元测试 → 三级冒烟（1/8/16 卡）→ 重跑 A 阶段 → 正式 B 阶段
  → MME/POPE 评测 → 三臂对照与总结。

## 2026-09-12 规划（本账本首条）

- 用户提出新目标：把之前数据上的 LoRA 微调换成全量微调，仍用本机与 h100-1 共 16 卡并行训练，
  先做好规划。
- 按 `AGENTS.md` 完成开工前四步：读本文件、读稳定实验目录主账本、查调度器与已有结果、检查工作树。
  - 调度器：Supervisor 未在运行（`supervisorctl` socket 不存在），仓库内 `configs/` 已无 supervisor 配置，
    单机 SQLite 调度器代码 `src/prefix_ttt/scheduler.py` 保留但本阶段不启用。16 卡跨两机作业沿用既有做法：
    两端 `scripts/run_multinode.sh` + `setsid nohup`，产物写共享 NFS。这是对 `AGENTS.md`
    「正式大型实验交给项目任务调度器」的一处偏离，理由是调度器为单机单卡并发模型、无法编排跨机 torchrun；
    既有的 A/E2 正式训练也是这么跑的。如需改走调度器请用户告知。
  - 已有结果：E0（基座）MME 1479.6432 / 349.2857、POPE 0.8548 / 0.8397；E2（TTT+LoRA，合并口径）
    MME 1429.4893 / 278.2143、POPE 0.8501 / 0.8357。本实验不重算这些数字。
- 产出：`plan.md`（目标与判据、唯一变量、资源账、工程改造清单、10 步执行与验证、风险、产物、明确不做）。
- 待用户批准后才开始修改代码；本轮未创建除 `plan.md`、`ledger.md` 之外的任何文件。

## 2026-09-12 优化器选型复议：改用 torch 内建 ZeRO-1（用户要求库优先）

- 用户提问「为什么分片优化器要自己实现？torch 或别的库没有吗」。**该质疑成立**：初版方案只把 FSDP2
  列为备选，遗漏了 torch 内建的 `torch.distributed.optim.ZeroRedundancyOptimizer`（即 ZeRO-1）。
- 查证结果（torch 2.7.1 实机）：
  - `torch.distributed.optim.ZeroRedundancyOptimizer` 存在，`_partition_parameters()` 把参数分给各 rank，
    每卡只为自己的分片建内层优化器（动量天然只存 1/16），`step()` 后由 owner rank 广播更新后的参数
    （`_broadcast_params_from_rank`，默认逐参数 async broadcast，`parameters_as_bucket_view=True` 时按桶）。
  - **更正上一轮的说法**：`dist.broadcast_coalesced` 公开别名确实不存在，但私有的
    `dist._broadcast_coalesced` **存在**（DDP 与 ZeRO 内部使用）。上一轮「broadcast_coalesced 不存在」
    的表述不准确。
  - 该库**不提供混合精度 master weight**：内层优化器按参数原 dtype 建立状态。
- 定量验证（4096²、$|w|\sim0.02$、lr=$2\times10^{-5}$、bf16 梯度、5 步 AdamW）：
  - bf16 参数 → `exp_avg` dtype = bf16，5 步后平均 $|\Delta|$ = 3.667e-05，**未改变元素 69.5702%**；
  - fp32 参数 → 状态 fp32，平均 $|\Delta|$ = 9.999e-05，未改变元素 0.0000%；
  - $|w|\approx0.02$ 处 bf16 的 ulp = 1.221e-04，单步更新量 2e-5 仅为 0.16 ulp。
  - 结论：把 bf16 模型直接交给现成优化器会让全量微调**静默失效**，这不是精度略差的问题。
- **决策（用户 2026-09-12）**：用库、尽量少自写 → 采用 ZeRO-1 + **fp32 参数**（参数本身即 master，
  无需自定优化器）。代价是显存从 31.7 GiB 升到 **54.1 GiB/卡**；换来自研优化器代码 0 行、
  训练循环与梯度归约语义与 E2 完全一致、数值上即标准 AMP 配方。
- 参数量更正：以 HF checkpoint 的 safetensors 头重算，总计 **7.0634 B**
  （`language_model` 6.7389 B、`vision_tower` 0.3035 B、`multi_modal_projector` 0.0210 B）。
  初稿引用的 7.1705 B 来自 A40 旧机 PEFT 包装后的审计结果，存在重复计数，**已作废**。
  可训练（LLM + projector）= 6.7599 B。
- 新发现的固有限制（已写入方案 5.3 与风险表）：`ZeroRedundancyOptimizer.state_dict()` 在未
  `consolidate_state_dict()` 时抛异常，而 consolidation 会把 16 个分片全部 broadcast 到单个 rank
  （约 80 GiB）→ 显存放不下。替代路径是内层 `zero.optim.state_dict()` 写逐卡分片；
  若往返测试不通过则退回「只存权重、崩溃后冷启动 Adam」，并如实记录，不假装支持精确 resume。
- 下一步：在 `sft.py` 接入 `--trainable full` 与 ZeRO-1，先跑 CPU 测试，再做 1/8/16 卡三级探针。

## 2026-09-12 写码前的 ZeRO 设计探针（两进程 gloo，CPU，临时脚本未入库）

目的：在写正式代码前把选型里唯一没被证实的部分证掉。脚本放在 `/tmp/zero-probe/`，
结论将以正式测试固化进 `tests/`。

- **发现一（硬约束）**：`ZeroRedundancyOptimizer` 要求所有参数 dense type 一致，混合 dtype 在构造时
  直接报 `ValueError: ... got both torch.BFloat16Tensor and torch.FloatTensor`。
  - 影响：E2 式的「bf16 基座 + fp32 TTT」混合布局**不能**交给 ZeRO；而全模型以 fp32 加载后
    LLM / projector / TTT 统一为 fp32，天然满足。这条约束与 4.2 节已选的 fp32 方案相容，
    同时说明「bf16 参数 + 库」这条近路在工程上是被库直接堵死的。
  - 冻结的 ViT 不进优化器，dtype 不受约束。
- **发现二（通过项）**：全 fp32、两个参数组（lr 2e-5 / 1e-4，weight_decay 0.01 / 0）下：
  构造通过；`LambdaLR` 的学习率更新正确传到内层优化器（两组按 0.5 同步衰减）；
  每步 ZeRO 回写后各 rank 参数 `torch.equal` 全等；冻结参数平均位移 0；
  `parameters_as_bucket_view=True` 可用；`zero_grad(set_to_none=True)` 与外部 `clip_grad_norm_`
  的既有调用顺序不变。
- **发现三（存档路径）**：公开 `state_dict()` 未 consolidate 时确实抛 `RuntimeError`（限制属实，
  consolidation 需把 16 个分片 gather 到单卡，约 80 GiB，显存放不下）；
  内层 `zero.optim.state_dict()` → CPU → `load_state_dict()` 往返**通过**，逐卡分片存档可行。
  踩坑记录：只搬 `state` 字段而丢掉 `param_groups` 会在加载时报 `KeyError: 'param_groups'`
  （该报错最初出现在探针自身，不是库缺陷，已修正后复测通过）。
- 产出：`plan.md` 5.1 节新增「已验证」表与硬约束说明，5.3 节与风险表同步更新。
- 下一步：在 `sft.py` 接入 `--trainable full` 与 ZeRO-1，把上述结论写成 `tests/` 正式测试，
  再跑 1/8/16 卡三级探针。本轮未修改任何仓库内代码，未启动任何 GPU 任务。

## 2026-09-12 混合精度答疑与实测（用户提问：为什么不能用 FP16 做 SFT）

- 用户提问：是否必须 fp32 加载？为什么不在 fp16 上 SFT？哪种是最佳实践？用户自述没有训练经验。
- 实测（4096²、$\mathcal{N}(0,0.02^2)$ 权重、梯度 $\mathcal{N}(0,10^{-3})$、AdamW lr 2e-5 / betas
  (0.9,0.95) / eps 1e-8 / wd 0.01，5 步）：
  - fp32 参数：5 步 mean|Δ| = 9.9991e-05，未改变元素 **0.00%**（正确答案：$5\eta$，每元素都在动）；
  - bf16 参数：mean|Δ| = 3.6670e-05，未改变元素 **69.57%** —— 静默失效，不报错、loss 仍在降；
  - fp16 参数：mean|Δ| = **NaN**，未改变元素 0.49%，$v$ 有 **56% 恰为 0、44% 为次正规**。
- fp16 失败机理（已写入教学文档）：$v$ 是梯度的平方（$10^{-6}\sim10^{-10}$），而 fp16 最小正规数为
  6.1e-05 → 大量下溢；且 $\epsilon=10^{-8}$ 在 fp16 中直接为 0 → $\sqrt{v}+\epsilon=0$ → 除零 → NaN。
  这与 bf16 的失效方式完全不同：**bf16 坏在精度，fp16 坏在范围**。
- 其他实测：
  - `F.linear` 在 autocast 下用 fp32 权重与 bf16 权重**逐位相同**（`torch.equal` 为真，最大差 0.0）
    → FP32 存储不改变矩阵乘的计算精度与速度。
  - `F.embedding` **不遵循**该规律：fp32 权重 → fp32 输出，bf16 权重 → bf16 输出。
  - 残差流 dtype 只由**输入**决定：fp32 权重 + bf16 输入 → 残差流仍是 bf16。
    故唯一风险点是多模态嵌入的起始 dtype，已列为单卡探针的必测项（激活显存最坏 +4 GiB）。
  - 基座 checkpoint 为 **F16 存储**（686 张量，`config.torch_dtype=float16`）→ 以 fp32 加载是无损加宽，
    E0/E2 的 bf16 加载是有损收窄（10 位尾数 → 7 位）。这一点此前未记录。
- **关于"最佳实践"的澄清**（写入教学文档第 4 章）：教科书最优是「bf16 参数 + 分片 fp32 master」
  （DeepSpeed / FSDP2 的做法，约 30 GiB/卡）；本项目选的是「fp32 参数 + autocast + ZeRO-1」，
  贵约 22 GiB，换来零自研优化器代码与完全不变的训练循环。这是**有意的取舍，不是最优方案**。
- 产出：新增教学文档 `mixed_precision.md`（5 章：四个精度决策 / 为什么 master 必须 fp32 /
  低精度用在哪 / 本项目的推理链与代价 / 常见误解清单）；`plan.md` 4.2 与第 6 节步骤 2 同步更新。
- 仍未修改任何仓库内代码，未启动任何 GPU 任务。

## 2026-09-12 代码实现完成（`--trainable full` + ZeRO-1）

用户指示「继续推进训练计划」。本轮完成代码实现与 CPU 验证，未启动正式 GPU 任务。

- 改动清单（6 个文件 + 2 个新文件）：
  - `src/prefix_ttt/model/trainability.py`：新增 `FULL_FINETUNE_PREFIXES`（显式白名单：
    `model.layers.` / `model.embed_tokens.` / `model.norm.` / `model.mm_projector.` / `lm_head.`）与
    `VISION_TOWER_PREFIX`；新增 `enable_full_finetuning(model)`（先全冻结，再按白名单打开，最后**校验**
    没有任何白名单之外的参数可训练，返回可训练元素数）；`audit_parameters` 增加 `allow_base` 开关与
    `base` 类别 —— `allow_base=False` 时行为与改动前逐字一致。
  - `src/prefix_ttt/training.py`：新增常量 `BASE_LR = 2e-5`；`optimizer_groups` 增加 `allow_base` 参数
    与 `kind → lr` 显式映射（`new_module`/`base`/`lora`）。
  - `src/prefix_ttt/config.py`：`load_config(path, *, require_base_lr=False)`；`base_lr` 键**存在即校验**，
    `require_base_lr=True` 时缺失即报错。`configs/base.json` 不受影响。
  - `configs/full.json`：由 `base.json` 文本级插入一行生成，共有键逐字相同；`config_sha256` 由
    `bc5ec1e2…`（未变，与既有 A/E2 身份一致）变为 `845c4851…`。
  - `src/prefix_ttt/sft.py`：新增 `--trainable {lora,full}`（默认 `lora`）；full 分支以 fp32 加载模型、
    不调用 `install_lora`、构造 `ZeroRedundancyOptimizer`；**训练循环、梯度 `all_reduce`、
    `clip_grad_norm_`、调度器、日志全部复用原有代码**；identity 增加 `trainable_mode` 字段。
  - `src/prefix_ttt/lmms_model.py`：`trainable_mode == 'full'` 时走「装 TTT → 不装 LoRA →
    `load_state_dict` 全量权重 → 不合并」；LoRA 分支逐字未动。
  - `src/prefix_ttt/runtime.py`：新增 `to_cpu`（把嵌套优化器状态搬到主机再序列化）。
  - `tests/test_full_finetuning.py`（新，4 项）：白名单正确性（ViT 冻结 / LLM+projector 可训练 /
    TTT 保持 fp32）、`audit_parameters` 的 `allow_base` 语义与 LoRA 路径不回归、
    参数组 LR 映射（`BASE_LR` / `NEW_MODULE_LR`）、以及**两进程 gloo** 下 ZeRO-1 的混合 dtype 拒绝、
    `LambdaLR` 传递、各 rank 参数逐位一致、逐卡分片往返与「分片恰好划分参数集」。
- 验证：CPU 套件 **171 passed / 0 failed**（原 167 项零回归 + 新增 4 项）。
- **一处键名冲突在实现中发现并修正**：identity 里的 `trainable`（字符串）会被存档时的权重字典
  `trainable` 覆盖，导致 resume 身份校验永远失败。已把 identity 字段改名为 `trainable_mode`，
  权重键保持 `trainable` 以复用 `load_trainable`。

### 顺带发现并修复一个既有缺陷（非本次改动引入）

- 现象：`load_manifest` 抛 `NameError: name 'digest_json' is not defined`。
- 定性：**HEAD 上已存在的缺陷**，`git status` 可证本轮未触碰这三个文件。成因是 `16d67fe` 那次重构把
  `digest_json` 抽到 `digests.py` 时，修了 `manifests.py` 的 import，漏了另外三个文件：
  - `data_pipeline.py` 缺 `digest_json` → **正式训练入口 `sft.py` 第一步就崩**；
  - `pilot_diagnostic.py`、`switch_diagnostic.py` 缺 `digest_file`。
- 为何十天没被发现：重构的验证是**评测路径的逐样本比对**，而评测不读 manifest；`tests/test_data_pipeline.py`
  当时只覆盖 `micro_batches`，`load_manifest` 零覆盖。**重构的验证范围没有覆盖训练入口**，这是流程漏洞，
  不是运气问题。
- 处理：三处各补一个导入（有依据的一次修复）。验证 `load_manifest` 实跑通过，`manifest_sha256 =
  9280270bf2453897c3ff60f2265dfe84630326fc9dfab78f0fc256bdfac63d4b`，与账本既有记录一致；
  并新增 3 项测试（合成 manifest，避免每次哈希 500 MB 标注文件）：正常路径、篡改顺序、篡改标注、
  split 交叠，耗时合计 2.8 s。全套件 **171 passed**。

### 方案更正：单卡显存探针不可行

- 原方案第 6 节步骤 2 写的是「单卡探针」。**该步骤在 ZeRO 下不可能成立**：world=1 时"分片"就是全部，
  优化器状态 50.4 GiB + fp32 参数 26.3 GiB + fp32 梯度 25.2 GiB 远超单卡 80 GiB。
- 正确的最小规模是 8 卡：优化器分片 6.3 GiB，稳态合计 **57.8 GiB/卡**（16 卡为 54.6 GiB）。
  探针改为「8 卡单机」，与冒烟合并为一步。已同步 `plan.md`。

## 2026-09-12 探针失败两次，定位并修掉两个真问题（未启动正式 B）

### 第一次 8 卡探针（E1+full，micro=8）：两个失败

- **失败一：`RuntimeError: Missing trainable gradient`**（6 个 rank）。
  根因：`prepare_sample` 把整段多模态展开包在 `torch.no_grad()` 里，因此 `embed_tokens` 与
  `mm_projector` 的权重**永远拿不到梯度**。LoRA 路径从未暴露这一点，因为 `install_lora` 本来就冻结这两者。
  修复：`prepare_sample(..., trainable_embedding=False)`，全量路径传 True，用
  `torch.set_grad_enabled(...)` 替代固定的 `no_grad`。**不需要给 ViT 打补丁**：ViT 参数已冻结且输入
  不需要梯度，autograd 根本不会为它建图，所以梯度正好停在 ViT 边界、流经 projector 与 embed。
- **失败二：CUDA OOM**，峰值 78.6 GiB（其中活跃 ~70 GiB、碎片 7.35 GiB）。
  处置：`configs/full.json` 的 `micro_batch_size` 8→4（账本已证明目标函数与 micro 分组无关），
  并在启动环境加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

### 第二次 8 卡探针（micro=4 + expandable_segments）：仍贴 80 GiB，且第 2 步不再推进

- 现象：8 张卡显存全部 ~80 GiB，部分 rank 100% 占用、部分 0%，步时从 7.2 s 恶化到停滞。已终止。
- **分阶段显存测量**（8 卡，逐步打印 allocated/peak/reserved）给出关键事实：
  - `load_checkpoint(fp32)` + `to(device)` → **26.41 GiB**（与预测 26.31 一致）
  - 建 ZeRO 优化器后 → **仍 26.41 GiB**（优化器状态首次 `step()` 才分配，符合库设计）
  - 4 个 micro-batch 数据就绪 → 27.02 GiB
  - 单 micro-batch 形状 `[4, 1121~1316, 4096]`（**序列长度约 1100–1300，不是 2048**）
  - 结论：稳态预测 26.4 + 25.2（fp32 梯度）+ 6.3（w=8 的优化器分片）= **57.9 GiB 是对的**，
    超出部分来自 fp32 残差流带来的激活膨胀。
- **真正的拦路虎：Triton `AssertionError: Both operands must be same dtype. Got fp32 and bf16`**，
  堆栈为 `hybrid.py:95 → ops/fla.py:62 fla_prefix`，即 Prefix-TTT 的 FLA kernel。
  （E1 探针没有 TTT 层，所以看不到这个错。）

### 根因（逐个 dtype 实测确认，非推测）

1. 全量微调下基座是 fp32，`F.embedding` 在 fp32 权重下返回 **fp32**（`F.linear` 则返回 bf16，两者行为不同），
   于是 `inputs_embeds` 是 fp32 → 整条残差流 fp32。
2. HF 的 `LlamaRotaryEmbedding` 会把 cos/sin **转成 hidden 的 dtype** → 残差流 fp32 时 cos/sin 也是 fp32
   → 旋转之后 q/k 被提升为 **fp32**（v 不经旋转，仍是 autocast 转出的 bf16）。
3. `hybrid.py` 的 `amp_enabled = q.dtype in (bf16, fp16)` 因此为 **False** →
   `features_pair` 在 fp32 下算出 **qf/kf 为 fp32**，与 fp32 的 v… 不，v 是 bf16 → FLA kernel 报错。
   （注：这一步此前一直误判为"q 就是 fp32"；实测 spy 打印 q=bf16、k=bf16、v=bf16，
   说明提升发生在**旋转之后**，不是投影处。）
- 为什么 E2/LoRA 从未见过：那条路径基座是 bf16 → 残差流 bf16 → cos/sin bf16 → q/k bf16 → 一切匹配。

### 修复与验证

- 修复：`prepare_sample` 在全量路径把多模态嵌入 `embeds.to(torch.bfloat16)`。
  这把残差流恢复成与 E2 逐位同构的 bf16，**FP32 权重在 autocast 下每个投影本来就转 bf16，所以变的是存储
  dtype 而不是数值**；同时把残差流的激活字节减半，正面缓解 OOM。
- 验证：1 卡真实模型探针，`inputs_embeds dtype = torch.bfloat16`，**前向成功**（修复前同一探针报
  CompilationError）。
- 未采用的做法（记录理由）：改 `hybrid.py` 让它容忍 fp32 q/k —— 会触碰已验证的 kernel 路径，
  且解决不了残差流翻倍的显存问题。

### h100-1 同步

- 先打包备份其 6 个脏文件到 `/data/shared/weights/prefix-ttt/scratch/h100-1-dirty-20260912.tar.gz`
  （10808 B），再 rsync `src/`、`configs/`、`tests/`。
- 校验：`sft.py`、`data_pipeline.py`、`model/trainability.py`、`training.py`、`configs/full.json`
  五个文件 SHA256 两端逐字相同。

### 已启动

- A 阶段正式训练（`configs/full.json`，`--output training/A-full`），两端 `run_multinode.sh` 0/1，
  16 卡，超时 6 h，日志 `/tmp/A-full-node{0,1}.log`。

## 2026-09-12 A 阶段完成 + E2-full 三步冒烟 + 正式 B 启动

### A 阶段（全量微调的 TTT 初始化）

- 命令：两端 `run_multinode.sh 0|1 prefix_ttt.transfer --config configs/full.json
  --manifest artifacts/cpu/fixed_manifest.json --save-every 25 --output training/A-full`，16 卡，6 h 超时。
- 结果：`complete=true`、`global_step=391`、`samples_seen=50000`、`config_sha256=2c51936e…`
  （与 `configs/full.json` 一致）、`manifest_sha256=9280270b…`；稳态步时中位数 **1.05 s**，全程约 8 分钟，
  峰值显存 15.1 GiB。产物 `training/A-full/`（latest.pt 325877387 B、result.json、run.json、steps.jsonl）。

### E2-full 三步冒烟（16 卡跨机）

- 命令：`launch_b.sh <0|1> --output scratch/smoke-e2full --max-steps 3`，两端。
- 结果：三步全部成功，无任何报错。
  - step 1 loss=8.853707 grad_norm=126.93 31.90s；step 2 loss=9.055559 grad_norm=101.61 3.02s；
    step 3 loss=9.070159 grad_norm=156.61 2.39s。
- **重要对照（澄清一个看似异常的 loss）**：正式 E2（LoRA）轨迹的前三步是
  `8.849594 / 9.058605 / 9.080996`，与本次 `8.853707 / 9.055559 / 9.070159` 几乎逐位吻合，
  梯度范数同为 80–160 量级。因此 **loss≈8.9 是 E2 布局本身的起点，不是缺陷**；
  E1（无 TTT）的 0.70 不能用来判断 E2 是否正常。步时也基本持平（E2 为 33.15/3.05/2.07 s）。

### 正式 B 阶段已启动

- 命令：两端 `launch_b.sh <0|1>`（`--layout E2 --trainable full --config configs/full.json
  --stage-a-checkpoint training/A-full/latest.pt --output training/E2-full --save-every 250`），
  16 卡，6 h 超时，环境含 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
- 启动时间 2026-09-12 02:42；step 1 loss=8.853707（与冒烟逐位一致，确定性成立）。
- 显存：稳态约 **78.4 GiB / 79.19 GiB（98.5%）**，余量不足 1 GiB。这是本阶段最大的运行风险，
  已如实记录；缓解手段是 `--save-every 250` 提供的权重恢复点，以及必要时把 `micro_batch_size` 降到 2
  并重跑 A（改配置会改 `config_sha256`，A 必须同配置重跑，代价约 8 分钟）。
- 未解释的显存增量：稳态 78.4 GiB 比"26.4 参数 + 25.2 梯度 + 3.15 优化器分片 = 54.75 GiB"高出约 23 GiB。
  主要嫌疑是 **autocast 为 fp32 权重逐次生成的 bf16 副本被反向保留**（按 32 层 × 7 投影估算约 13 GiB），
  该项与 microbatch 无关，因此降低 micro 只能收回其中一部分。**该推断尚未实测确认**，不作为结论。

## 2026-09-12 架构返工：bf16 参数 + 分片 fp32 master（B 重启成功）

### 为什么必须返工：micro 不是那个杠杆（实测）

| 手段 | 峰值显存 |
| --- | --- |
| fp32 + micro=8 | 78.99 GiB |
| fp32 + micro=4（减半） | **78.40 GiB** |
| bf16 参数 + 分片 fp32 master | **42.1 GiB** |

micro 从 8 降到 4 只回收 **0.6 GiB**，因为 fp32 训练下的显存几乎全是"模型尺寸"项：
fp32 参数 26.4 + fp32 梯度 25.2 + 优化器分片 3.2 + autocast 权重副本 ~13 ≈ 68 GiB 与批量无关，
只有约 5 GiB 激活随 micro 变。**用户质疑"为什么不直接调小 micro"——该质疑合理，此表即为回答。**

### 返工内容

- 新增 `src/prefix_ttt/optim.py::MasterWeightAdamW`（45 行）：AdamW 的 fp32 master 与 m/v 存在
  **`optimizer.state`** 里。ZeRO 只分片优化器状态；若 master 作为模块参数存在，会被 ZeRO 的 step 后广播
  复制到所有 rank。放进 state 才可分片，且模块存储保持 bf16（正是前向原本的计算 dtype）。
  更新算式与 `torch.optim.AdamW` 逐位一致（含 `exp_avg.lerp_(grad, 1-beta1)` 的写法）。
- `sft.py`：模型改回 **bf16 加载**；`full` 分支按参数类别拆两个优化器——
  `base`（bf16）走 `ZeroRedundancyOptimizer(MasterWeightAdamW)`，`new_module`（fp32 TTT，27M）走普通
  `AdamW`；两个 `LambdaLR` 同步；`zero_grad`/`step` 遍历两个优化器；存档分 `base`/`ttt` 两段。
- 新增 2 个测试：与 `torch.optim.AdamW` 在 fp32 参数上 20 步**逐位相等**；bf16 存储下 master 的位移
  与 naive bf16 的对比（实测 master mean|Δ|=6.27e-05、零元素未变；naive 1.91e-05、**70.9% 元素不动**）。
- CPU 套件 **175 passed / 0 failed**。两端 `sft.py`、`optim.py` SHA256 逐字相同。
- **`configs/full.json` 未改动，`config_sha256` 不变，因此已完成的 A-full 继续有效，无需重跑。**

### 结果（B 重启后第 11 步实测）

| | 返工前 | 返工后 |
| --- | --- | --- |
| 显存 | 78.4 GiB（98.5%，第 12 步卡死，功耗 147 W） | **42.1 GiB（53%）** |
| 步时 | 2.64 s | **2.26 s** |
| 功耗 | 147 W（自旋） | **480 W（真实计算）** |
| step 11 loss | 7.718246 | 7.719872 |

步时变快的原因是不再有每次前向的 fp32→bf16 权重转换。

## 2026-09-12 B 阶段完成 + 三臂评测结果（本轮实验的主结果）

### 训练完成

- `result.json`：`complete=true`、`global_step=5182`、`samples_seen=663248`（训练集全量）、
  `config_sha256=2c51936e…`、`stage_a_sha256=547cd695…`、`world_size=16`、`diagnostic_only=false`。
- 03:01 → 06:36，墙钟 **3.48 h**，稳态 2.28 s/步，无 OOM、无非有限 loss。
- 产物：`training/E2-full/latest.pt`（14.2 GB）+ 16 个 `latest.pt.optim-rank*`、`pilot.pt` + 16 个分片。

### 训练曲线：全量微调收敛更快更低（同 5182 步、同数据顺序）

| | E2 (LoRA) | E2-full |
| --- | --- | --- |
| step 100 loss | 2.2056 | **1.2911** |
| step 500 loss | 1.0930 | **0.9287** |
| 最后一步 loss | 0.7636 | **0.7321** |
| 末 50 步均值 | 0.7745 | **0.7339** |
| grad_norm 中位数 | 2.258 | 1.992 |
| 非有限 loss | 0 | 0 |
| 墙钟 | 3.83 h | **3.48 h** |

### 官方分数（lmms-eval 0.7.2，同一 prompt 与打分器）

| 臂 | MME perception | MME cognition | POPE accuracy | POPE F1 |
| --- | --- | --- | --- | --- |
| E0（不训练） | 1479.6432 | 349.2857 | 0.8548 | 0.8397 |
| E2（LoRA） | 1429.4893 | 278.2143 | 0.8501 | 0.8357 |
| **E2-full（本次）** | **1456.9744** | **297.5000** | **0.8541** | **0.8392** |

- 相对 E2：MME perception **+27.4851**、cognition **+19.2857**、POPE acc **+0.0040**、F1 **+0.0035** —— 四项全部提高。
- 相对 E0：MME perception −22.6688、cognition −51.7857、POPE acc −0.0007、F1 −0.0005。

**按方案第 1 节的判据**：
- ✅ 主判据达成（MME perception > 1429.4893）；
- ✅ 次要判据达成（POPE accuracy > 0.8501），且与 E0 的差距缩到 0.0007（9000 条里约 6 条），可视为持平；
- ❌ 目标未达成（MME perception < 1479.6432）——全量微调追回了 E0 差距的 **54.8 %**
  （E2 落后 50.15 → E2-full 落后 22.67），但没有追平。

### 推理成本与一处必须澄清的不可比

- **缓存逐位相同**：E2 与 E2-full 均为 Full KV 90.3 + TTT 状态 46.0 + Local-32 11.5 = **147.8 MiB**（MME 口径），
  与"架构未变"一致；峰值显存 13.52 vs 13.61 GiB，差异在噪声量级。
- **prefill/TPOT 出现了不应有的差异**：E2 78.92/34.66 ms，E2-full 62.58/31.47 ms。
  两臂推理路径完全相同，**该差异不可归因于训练方式**。上一阶段账本已明确记录：跑批之间不可比
  （E2 的最终跑批为 4 并发，E2-full 为 2 并发）。
- 处置：已在 GPU 2/3 以**同样的 2 并发条件**重测 E2 权重（输出 `eval-recheck/`），
  用同条件数字替换后再写结论。**在重测完成前，不声称 E2-full 的推理速度有任何变化。**

### 同条件重测：结论（写死，不再用原跑批的延迟数字）

- 在 GPU 2/3 以**与 E2-full 相同的 2 并发**重测 E2 权重（`eval-recheck/`）：
  - 分数**逐位复现**：MME 1429.4893 / 278.2143，POPE 0.8501 / 0.8357 —— 证明重测条件可靠。
  - 成本：E2 的 prefill 由原记录的 **78.92 → 62.10 ms**（MME 口径），TPOT 34.66 → 31.12 ms。
- 同条件下 E2 vs E2-full：MME 62.10/31.12 vs 62.58/31.47（1.008×/1.011×）；
  POPE 62.29/31.67 vs 60.65/30.27（0.974×/0.956×）。**差异在两个任务上符号相反、量级 ±1–4 %**，
  即**无系统性差异**，与"两臂推理架构完全相同"一致。
- 因此本阶段**不声称**全量微调改变了推理速度；原跑批的 21 % 差异已定性为并发造成的测量假象。

### 本阶段正式产物

- `experiment_summary.md`：实验总结（目标与判据、唯一变量、三次架构演进、三臂结果、成功与失败、
  失败原因分析、五条工程记录、后续方向、产物位置）。
- `mixed_precision.md`：混合精度教学文档（5 章）。
- `metrics-summary.json`：三臂分数与成本的机器可读汇总（由绘图脚本生成）。
- `images/`：`cost_by_task.png`、`cache_by_task.png`、`score_vs_cost.png`（均已按最终口径重绘，
  成本使用同条件重测的数据）。
- `scripts/`：`launch_b.sh`、`run_eval_full.sh`、`plot_three_arms.py`。

### 本阶段最终结论

把 LoRA 换成全量微调，在本数据与配方下把 MME perception 从 1429.4893 提到 **1456.9744**
（追回与基座差距的 54.8 %），MME cognition 从 278.2143 提到 297.5000，POPE accuracy 从 0.8501 提到
**0.8541**（与基座 0.8548 的差距缩到 9000 条里约 6 条），且推理成本不变。
但 **TTT 混合架构相对原始 LLaVA-1.5 仍有约 22.7 分 MME perception 与约 51.8 分 cognition 的差距，
全量微调不足以弥合**。是否值得继续投入取决于该差距能否由架构改动而非训练方式消除——
这需要补做「无 TTT 的纯 LLaVA 全量微调」对照臂，本阶段按用户决定未运行。
