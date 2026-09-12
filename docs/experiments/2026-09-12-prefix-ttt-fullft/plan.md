# Prefix-TTT 全量微调实验方案（E2-full）

启动日期 2026-09-12。本目录是「把 E2 的 LoRA 微调换成全量微调」这一长程实验的稳定目录；
前一阶段（环境重建、A/B 训练、E0/E2 评测、推理 kernel 优化与代码审计）的账本在
`docs/experiments/2026-09-10-prefix-ttt-h100/`，推理 kernel 专项在
`docs/experiments/2026-09-11-prefix-ttt-kernel/`，两者保持原样，本目录不复制其内容。

## 1. 目标与成功判据

**目标**：在完全相同的训练数据、manifest、步数与评测协议下，把 E2 的 LoRA 适配器换成对基座权重的
全量微调，观察 MME / POPE 官方分数能否提高。

**成功判据**（按重要性排序，全部以官方分数为准）：

| 判据 | 阈值 | 含义 |
| --- | --- | --- |
| 主判据 | MME perception > 1429.4893 | 超过 E2（LoRA），说明全量微调这一改动本身有效 |
| 目标 | MME perception ≥ 1479.6432 | 追平未经训练的基座 E0，即 TTT + 全量微调不再是净损失 |
| 次要 | POPE accuracy > 0.8501 | 与 E2 同向 |
| 完成性 | `complete=true`、5182 步、loss 与 grad_norm 全程有限 | 没有配置、日志和状态文件支撑的结果不写入结论 |

**反判据**：若 MME perception 落在 1429.4893 以下，则结论是「在该数据与配方下 LoRA 已足够，
全量微调不带来收益」，如实记录，不改配方反复试凑。

## 2. 基线（不得重算，直接引用）

| 臂 | 训练方式 | MME perception | MME cognition | POPE accuracy | POPE F1 |
| --- | --- | --- | --- | --- | --- |
| E0 | 不训练（官方 LLaVA-1.5-7B） | 1479.6432 | 349.2857 | 0.8548 | 0.8397 |
| E2 | A 阶段 + TTT + LoRA | 1429.4893 | 278.2143 | 0.8501 | 0.8357 |

E2 是在**未合并的 LoRA 适配器**下训练、在**合并后**的权重上评测的；E2 相对 E0 在 MME 上低 50.15 分，
这是本实验想要挽回的差额。E1（只 LoRA、无 TTT）从未训练，本阶段也不训练。

## 3. 唯一变量

相对 E2，只改一件事：**被优化的参数集合从「LoRA 适配器 + TTT」变成「基座 LLM + mm_projector + TTT」**。

保持不变（逐项钉死，任何一项被改动都会破坏归因）：

- 数据：`artifacts/cpu/fixed_manifest.json` 的同一份 train/dev/A 划分，同一 `data_root`。
- 步数：`total_steps=5182`、pilot 在 step 391；`effective_batch_size=128`、`micro_batch_size=8`、
  `dataloader_workers=4`。
- 优化配方：cosine + `warmup_fraction=0.03`、`grad_clip=1.0`、`betas=(0.9,0.95)`、`eps=1e-8`、
  矩阵 `weight_decay=0.01`、`new_module_lr=1e-4`、**新增** `base_lr=2e-5`。
- 种子：`SEED=42`（数据顺序与初始化），A 阶段照旧。
- 评测：同一 lmms-eval 0.7.2 官方 MME / POPE、同一 prompt 与打分器、同一带内打点口径。
- **梯度聚合语义**：仍然保留 `sft.py` 现有的「逐参数 `all_reduce` 求和 + 手动 token 归一化」，
  不交给 ZeRO 或 FSDP 改写归约顺序，使梯度与 E2 在同一套语义下产生。

关于 `base_lr=2e-5`：这是 LLaVA-1.5 二阶段全量微调的官方学习率，也与本配方既有的 `lora_lr` 数值相同，
因此不构成新的调参自由度。`weight_decay=0.01` 对基座矩阵是新增作用面，但在 lr=$2\times10^{-5}$ 下
整段轨迹的衰减因子约为 $\exp(-5182 \times 2\times10^{-5} \times 0.01) \approx 0.999$，可以忽略，故保持配方一致。

**可训练范围（用户 2026-09-12 决定）**：冻结 ViT（`vision_tower`，0.3035 B），训练 mm_projector（0.0210 B）
+ LLM（6.7389 B）+ TTT（约 27.1M 新参数）。ViT 继续以 `eval()` 模式前向，与 E2 相同。

**命名**：布局不变（仍是 `--layout E2` 的 TTT 层安排），训练方式变成全量，产物目录用
`/data/shared/weights/prefix-ttt/training/E2-full/`；不新造 E3 编号，避免与任务书的 E0/E1/E2 语义冲突。
代码侧用正交开关 `--trainable {lora,full}` 表达，默认 `lora`，旧命令行为不变。

## 4. 资源账

### 4.1 参数与显存

参数量以 HF checkpoint 的 safetensors 头为准（2026-09-12 实测）：总计 **7.0634 B**——
`language_model` 6.7389 B、`vision_tower` 0.3035 B、`multi_modal_projector` 0.0210 B。

> 更正：本方案初稿写的 7.1705 B / 106.85 GiB 来自 A40 旧机的 `trainable_params.json`，
> 那是 PEFT 包装后的审计结果且存在重复计数，**已作废**，本节数字全部以 safetensors 头重算。

### 4.2 显存方案：ZeRO-1 + fp32 参数（用户 2026-09-12 选定）

用户要求「用库、尽量少自己写」，因此采用 **torch 内建的
`torch.distributed.optim.ZeroRedundancyOptimizer`（ZeRO-1）+ 原版 `torch.optim.AdamW`**，
模型以 **fp32** 加载：参数本身就是 master，不需要额外的 master 副本，也不需要自定优化器。

| 项目 | 每卡 |
| --- | --- |
| fp32 可训练参数（LLM + projector，复制） | 25.18 GiB |
| fp32 梯度（复制） | 25.18 GiB |
| Adam m / v（ZeRO 分片 1/16） | 3.15 GiB |
| ViT（bf16，冻结） | 0.57 GiB |
| **稳态合计** | **54.1 GiB** |
| 现有 LoRA 训练实测 | 16–20 GiB |

激活值与 E2 同量级（梯度检查点已开启，实测约 3–7 GiB），预计峰值 **57–61 GiB / 80 GiB（约 75%）**。
`micro_batch_size` 是安全阀：账本已证明目标函数与 micro 分组无关，必要时从 8 降到 4 即可砍掉一半激活。

**为什么必须是 fp32 参数，而不是把 bf16 参数直接交给库**：`torch.optim.AdamW` 的状态 dtype 跟随参数
dtype，bf16 参数会得到 bf16 动量，更新也在 bf16 里完成。2026-09-12 实测（4096² 权重、$|w|\sim0.02$、
lr=$2\times10^{-5}$、bf16 梯度、5 步）：

```text
bf16 参数：5 步后 |Δ| 平均 = 3.667e-05，未改变元素 69.5702%
fp32 参数：5 步后 |Δ| 平均 = 9.999e-05，未改变元素 0.0000%
|w|≈0.02 处 bf16 的 ulp = 1.221e-04；单步更新量 2e-5 = 0.16 ulp
```

即 bf16 参数下近七成权重在 5 步里一个 bit 都不动，全量微调会静默失效。这与既有代码一致：
`install_lora` 早已把 LoRA 与 TTT 参数转成 fp32 再训练，基座因冻结才留在 bf16。

fp16 参数更糟：同条件下 $v$ 有 **56% 恰好为 0、44% 掉进次正规数**，且 $\epsilon=10^{-8}$ 在 fp16 中
直接等于 0，导致 $\sqrt{v}+\epsilon=0$ → 5 步内参数出现 **NaN**。因此"把权重存成 fp16 再优化"不是
精度略差，而是数值上不成立；AMP 的 fp16 配方一直保留 fp32 master 正是这个原因。

**FP32 存储不等于 FP32 计算**：本方案的矩阵乘仍然全走 bf16 张量核心（`autocast` 转换），
实测 `F.linear` 在 autocast 下用 fp32 权重与用 bf16 权重**逐位相同**（`torch.equal` 为真）。
另外实测基座 checkpoint 本身为 **F16 存储**（686 张量，`config.torch_dtype=float16`），
故以 fp32 加载是**无损加宽**，而 E0/E2 的 bf16 加载是有损收窄。

原理与完整推导见本目录教学文档 `mixed_precision.md`（第 1–5 章）。

**被放弃的备选（记录理由，不实施）**：

- bf16 参数 + 自定 `MasterWeightAdamW`（fp32 master 存进 `optimizer.state`，由 ZeRO 一并分片）：
  显存约 30 GiB，比选定方案少 24 GiB，但需要自己写 40–60 行优化器。用户要求尽量少自写，故不采用；
  若选定方案出现显存不足，这是第一顺位的回退。
- FSDP2（`fully_shard`，约 6.7 GiB/卡）：能力足够且也是零自定优化器代码，但要把手写 all_reduce 循环、
  checkpoint 格式、以及自定义 TTT 前向 / FLA Triton kernel 的 DTensor 兼容全部重接，回归面最大。
- 不用 DeepSpeed：不在 `uv.lock` 中，自带 launcher 与配置体系，且会与手写训练循环冲突。

### 4.3 通信预算

跨机 RDMA 实测：64 MiB all-reduce 0.533–0.568 ms，有效聚合带宽约 230 GB/s。ZeRO-1 每步两次通信：

| 通信 | 数据量 | 估计耗时 |
| --- | --- | --- |
| 梯度 all-reduce（fp32，现有循环原有动作） | 25.18 GiB | 约 0.11 s |
| 参数回写：owner rank 逐个广播更新后的参数 | 25.18 GiB | 约 0.11 s |
| 合计 | | **约 0.22 s/步** |

现有 LoRA 训练稳态 0.92–1.05 s/步；全量微调还增加反向中基座权重的梯度计算（约 +25%）、
fp32 参数在 autocast 下的逐层 bf16 转换开销与优化器更新，预计 **1.6–2.4 s/步**，5182 步约 **2.3–3.5 h**。

## 5. 工程改造清单

原则：E1/E2 的既有代码路径必须逐字节保持不变；新增能力全部走 `--trainable full` 分支。
**不新增任何自研优化器数学**。

### 5.1 `sft.py`

- 新增 `--trainable {lora,full}`，默认 `lora`。
- `full` 分支与 `lora` 分支的差异只有四处：模型以 fp32 加载；不调用 `install_lora`；改用
  `ZeroRedundancyOptimizer(optimizer_groups(...), torch.optim.AdamW, betas=..., eps=...)`；
  checkpoint 内容不同。**训练循环、梯度 all_reduce、`clip_grad_norm_`、调度器、日志全部复用现有一段代码。**
- identity 字典新增 `trainable` 字段，使全量与 LoRA 的 checkpoint 互相不可 resume。
  **已知代价**：已完成且 `complete=true` 的 E2 轨迹将无法再 resume（identity 多一个键），
  这是有意的保护而非缺陷；当前不需要 resume 它。

**已用 gloo 探针在写正式代码前验证完毕**（2026-09-12，两进程 CPU，临时脚本 `/tmp/zero-probe/`，
结论将固化为 `tests/` 中的正式测试）：

| 验证项 | 结果 |
| --- | --- |
| 多参数组（`base` / `new_module`，不同 lr 与 weight_decay）分片 | 通过 |
| `LambdaLR` 的学习率更新传递到内层优化器 | 通过（两组的 lr 按 0.5 同步衰减） |
| ZeRO 每步回写后各 rank 参数是否逐位一致 | 通过（`torch.equal` 全等） |
| 冻结参数是否被误改 | 通过（平均位移 0） |
| `parameters_as_bucket_view=True` | 可用 |
| `optimizer.zero_grad(set_to_none=True)` 与外部 `clip_grad_norm_` 顺序 | 通过 |
| 逐卡优化器状态存档往返 `zero.optim.state_dict()` → CPU → `load_state_dict()` | 通过 |
| 公开 `state_dict()` 未 consolidate 时的行为 | 确认抛 `RuntimeError`（限制属实） |

**硬约束（探针发现，必须遵守）**：`ZeroRedundancyOptimizer` 要求所有参数的 dense type 一致，
混合 dtype 直接在构造时报错——

```text
ValueError: ZeroRedundancyOptimizer only supports using the same dense type for all
parameters but got both torch.BFloat16Tensor and torch.FloatTensor
```

这条约束与 4.2 节选择的 fp32 参数方案正好相容（LLM / projector / TTT 加载后统一为 fp32），
但也意味着**「bf16 基座 + fp32 TTT」这种 E2 式混合布局不能交给 ZeRO**。冻结的 ViT 不进入优化器，
其 dtype 不受该约束。

`parameters_as_bucket_view=True` 可把逐参数广播合并为分桶广播，作为探针阶段的性能开关，默认先关。

### 5.2 `model/trainability.py`

- 新增 `enable_full_finetuning(model, new_parameters)`：按显式白名单打开 LLM 与 mm_projector 的
  `requires_grad_`，TTT 新参数保持 fp32 并打开，ViT 保持冻结；不使用「除 ViT 外全部打开」的模糊写法。
- `audit_parameters` 增加 `base` 类别，使全量微调不触发「Unexpected trainable base parameter」。
- `optimizer_groups` 增加 `base` → `BASE_LR` 映射；`training.py` 新增常量 `BASE_LR = 2e-5`。

### 5.3 checkpoint 与评测加载

- **权重**：每 250 步由 rank 0 存一次 `latest.pt`，内容为**转为 bf16 的完整 state_dict**
  （约 14.4 GB），评测直接消费。转 bf16 与 E2 的合并口径一致（推理本来就是 bf16），
  fp32 精度只保留在优化器分片里。按 837 MB/s 实测写入约 17 s / 次。
- **优化器状态**：`ZeroRedundancyOptimizer.state_dict()` 在未 `consolidate_state_dict()` 时直接抛异常
  （已实测确认），而 consolidation 会把 16 个分片全部 broadcast 到单个 rank（约 80 GiB）→ 显存放不下，
  **不可用**。已实测可行的替代路径是内层本地优化器 `zero.optim.state_dict()` 写逐卡分片文件
  （约 8 行），往返 `state_dict()` → CPU → `load_state_dict()` 通过。注意必须保存**整个** state dict
  （含 `param_groups`），只搬 `state` 字段会在加载时报 `KeyError: 'param_groups'`。
  默认策略：**权重每 250 步、优化器分片仅在 pilot 与结束时写**。该取舍源于 ZeRO-1 的设计，
  不是本项目的实现缺陷。
- `lmms_model.py`：`state['trainable'] == 'full'` 时走「装 TTT → 不装 LoRA → 载入全量权重 → 不合并」；
  LoRA 分支保持原样。

### 5.4 配置

新增 `configs/full.json`（`configs/base.json` 不得改动：它是既有 A/E2 checkpoint 的 `config_sha256`
身份来源）。`full.json` 在 base 基础上增加 `training.base_lr=0.00002`、`training.trainable="full"`，
`config.py` 的配方校验同步认识这两个键。

### 5.5 明确不动

`transfer.py`（A 阶段）、`hybrid.py` 的前向数学、`ops/` 下全部 kernel、`data_pipeline.py` 的数据划分、
`runtime.py` 的 `save_atomic` / RNG 语义、`sft.py` 的 `lora` 分支。

## 6. 执行步骤与验证

1. **接入 ZeRO-1 与 `--trainable full`** → 验证：`uv run --locked pytest -m 'not gpu'` 全绿
   （现有 167 项不得回归）；新增测试覆盖 5.1 的三点确认项与「多参数组 + 混合 dtype 分片」。
2. **8 卡单机探针与冒烟**（真实模型、fp32、2 步、写 scratch）→ 验证：实测峰值显存与 4.2 节预测同量级；
   单步耗时给出 16 卡外推；确认权重存档与重载往返一致。
   **为什么不是单卡**：ZeRO-1 在 world=1 时分片即为全部，优化器状态 50.4 GiB + fp32 参数 26.3 GiB +
   梯度 25.2 GiB 远超单卡 80 GiB；8 卡时优化器分片降到 6.3 GiB、稳态合计 57.8 GiB，才是可行的最小规模。
   同时**实测「fp32 参数的整模型前向」与「bf16 参数前向」的差异**：已知 `F.embedding` 在 fp32 权重下
   返回 fp32（`F.linear` 逐位相同），因此残差流可能以 fp32 起始、激活显存最多翻倍（约 +4 GiB）。
   探针需给出真实峰值与逐层 dtype 证据；若超出预算，安全阀是 `micro_batch_size` 8→4。
3. **8 卡单机 2 步冒烟** → 验证：loss / grad_norm 有限；ZeRO 回写后各 rank 参数逐位一致。
4. **16 卡跨机 3 步冒烟**（`run_multinode.sh`，两端）→ 验证：rendezvous、跨机 all-reduce 与参数广播、
   共享 NFS 落盘与重载全部可用。
5. **同步 h100-1 代码** → 验证：两端 `src/`、`configs/`、`scripts/` 的 SHA256 逐一相同。同步前先把
   h100-1 的 6 个脏文件打包备份到 `/data/shared/weights/prefix-ttt/scratch/h100-1-dirty-20260912.tar.gz`，
   再 `git fetch && git checkout -f h100`。已核实这些脏文件是本机重构中途快照被 rsync 过去的残留
   （`configs/base.json` 与本机 HEAD 完全一致；其余 5 个不匹配任何历史提交），不是第三方改动。
6. **重跑 A 阶段**（`configs/full.json`，391 步，约 12 min）→ 验证：`complete=true`，输出
   `training/A-full/`。因为 `config_sha256` 变了，既有 A checkpoint 不能复用，重跑比放宽校验便宜且干净。
7. **正式 B 阶段**（`--layout E2 --trainable full`，5182 步，预计 2.3–3.5 h）→ 验证：逐步 loss /
   grad_norm 有限且在 E2 同量级；pilot 与最终 checkpoint 完整。
8. **评测**：E2-full 的 MME + POPE，带内打点，与 E0/E2 同协议 → 验证：分数落入判据区间；成本四项
   与 E2 合并口径一致（架构未变，不应有系统差异）。
9. **对照与总结**：三臂（E0 / E2 / E2-full）分数与成本对照表、曲线图、中文阶段总结；账本逐条更新。

## 7. 风险与缓解

| 风险 | 影响 | 缓解 |
| --- | --- | --- |
| 显存到 75%，长样本批次 OOM | 训练中断 | 步骤 2/3/4 三级探针实测；`micro_batch_size` 8→4 是已验证等价的安全阀 |
| ZeRO 与本项目多参数组 / 混合 dtype 不兼容 | 训练起不来 | **已验证**：多参数组可用；混合 dtype 被库拒绝，故全模型 fp32 加载（见 5.1 硬约束）；失败即回退到 4.2 节记录的 `MasterWeightAdamW` 方案 |
| fp32 参数在 autocast 下的逐层转换开销 | 步时膨胀 | 步骤 2 实测；必要时评估 `parameters_as_bucket_view` |
| 全量微调发散（loss spike / 非有限） | 轨迹作废 | 保留 `torch.isfinite` 硬检查；预算内允许一次降 lr 重跑并如实记录 |
| 优化器状态存档路径脆弱 | 崩溃后只能冷启动恢复 | **已验证**逐卡分片往返可用；仍保守地把优化器存档限制在 pilot 与结束，权重每 250 步 |
| h100-1 代码不同步导致两端行为不同 | 结果不可信 | 步骤 5 的哈希逐一比对；脏文件先备份再重置 |
| identity 增加字段使旧 E2 轨迹不可 resume | 既有轨迹的数据完整性 | E2 已 `complete=true`，不需要 resume；在账本中显式记录该影响 |
| 结果低于 E2 | 目标未达成 | 按第 1 节反判据如实记录，不试凑配方 |

## 8. 产物

- 训练：`/data/shared/weights/prefix-ttt/training/A-full/`、`training/E2-full/`（`latest.pt`、
  `steps.jsonl`、`run.json`、`result.json`）。
- 评测：`/data/shared/weights/prefix-ttt/eval-full/{e2full-mme,e2full-pope}/` 与 `cost/` 逐请求记录。
- 文档：本目录 `plan.md`、`ledger.md`、`experiment_summary.md`、`metrics-summary.json`、`images/`、`scripts/`。
- 代码：`sft.py` 的 `full` 分支、`trainability` 扩展、新配置与测试；E1/E2 路径零改动。

## 9. 明确不做

- 不训练对照臂「无 TTT 的纯 LLaVA 全量微调」（用户 2026-09-12 决定只跑一臂）。
- 不自己写分片优化器数学（用户 2026-09-12 决定优先用库）。
- 不改 `configs/base.json`，不重算 E0/E2 的任何既有数字。
- 不改动 `transfer.py`，不重构 `hybrid.py` 前向。
- 不引入 DeepSpeed / bitsandbytes 等新依赖，不改任何库版本。
- 不在本轮做数据集扩充、prompt 调整或评测协议变更；分数变化只归因于训练方式。
