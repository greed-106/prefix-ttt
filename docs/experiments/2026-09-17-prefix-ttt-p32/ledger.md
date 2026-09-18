# P32 纯 Prefix-TTT 主账本

本文件是本实验项目的唯一主账本，逐条记录代码改动、训练提交、实验完成或失败、关键决策与产物位置。
方案见同目录 [plan.md](plan.md)；前两阶段账本见
`docs/experiments/2026-09-10-prefix-ttt-h100/ledger.md` 与 `docs/experiments/2026-09-12-prefix-ttt-fullft/ledger.md`。

## 2026-09-17 立项与分支创建

- 用户提出新一轮目标：把 LLaVA-1.5-7B 的**全部 32 个 LLM self-attention 子层**替换为纯 Prefix-TTT，
  不保留 Full MSA / SWA / Local-32；只使用现有 LLaVA-665k 数据；原模型只作为教师与实验对照。
- 用户在本轮明确的两项决策：
  1. **算力**：先只用本机 8×H100，h100-1 暂不接入，但配置与脚本不得写死单机（为后续 16 卡留空间）。
  2. **范围**：本轮不实现也不训练 P32-Basic（去掉视觉监督的消融）、不做 Full-MSA-LoRA 对照、
     不做 P32 全量微调；集中精力快速完成主模型训练，其余不阻塞主训练的工作全部后置。
- 已按用户要求**先建分支再改代码**：工作树干净（`git status --porcelain` 为空）时从 `h100` @ `60be803`
  创建本地分支 `full-ttt`。未提交、未推送。
- 建立稳定实验目录 `docs/experiments/2026-09-17-prefix-ttt-p32/`（本账本与 `plan.md`）、
  脚本目录 `scripts/experiments/prefix_ttt_p32/`、原始产物目录 `artifacts/experiments/prefix_ttt_p32/`。
- 勘察结论（写入 plan.md 第 0 节）：现有 A 阶段目标是残差 $T\approx A_{\text{Full}}-A_{\text{Local}}$，
  读出带 RMS 归一化，B 阶段无教师 KL；数据固定为 train 663248 / dev 2048 / A 50000；
  本机 8×H100 全空闲；h100-1 的 8 卡被 `mkxiang` 的 RoboTwin/VLA 任务占用（7–10 GiB/卡）。

### 本轮训练配置（初值，smoke 后可能按实测调整 micro 批次）

| 项 | 值 |
| --- | --- |
| 布局 | `P32`：32 层全为 Prefix-TTT，anchor 列表为空 |
| A 阶段 | `manifest['A']` 50000 样本 / 391 步，目标 $\mathcal L_{\text{full}}+\mathcal L_{\text{vision}}$ |
| B 阶段 | `manifest['train']` 663248 样本 / 5182 步，损失 CE + KL（$\lambda_{KD}=1,\tau=1$） |
| 启动规模 | 单机 8×H100（`NNODES=1`、`GPUS=8`），累积步数由 world size 推出 |

下一步：实现纯 TTT 算子路径（plan.md §3.1）。

## 2026-09-17 代码改造（plan.md §3.1–§3.5，未提交）

按 plan 的分层实现，全部改动都在 `full-ttt` 分支上；`h100` 分支与既有 checkpoint 不受影响。

### 纯 TTT 算子与模型层（§3.1）

- `ops/features.py`：`FeatureReadout.readout(x, memory, *, local=None, normalize=True)`。`normalize=False`
  时读出为 $y_t=g_t m_t$（P32 主配置，取消对 $m_t$ 的 RMSNorm）；`normalize=True` 保留原混合布局行为。
  特征映射 $\phi(x)=\mathrm{RMSNorm}[\mathrm{SiLU}(xA_\phi)\odot(xB_\phi)]$ 未改。
- `ops/fused_features.py`：Triton 读出 kernel 新增 `NORMALIZE`/`HAS_LOCAL` 两个 `tl.constexpr`
  （`HAS_LOCAL=0` 时不加载 local，指针传 memory），小形状回调路径同步；只影响推理。
- `model/hybrid.py`：新增 `LAYOUT_ANCHORS = {'E2': FULL_ATTENTION_LAYERS, 'P32': ()}`；
  `install_prefix_ttt` 以 anchor 列表为唯一布局开关，内部推出 `use_local`/`readout_normalize`。
  `PrefixTTTAttention` 在 `use_local=False` 时完全不调用 local 路径，缓存只写 `LayerState(state=...)`。
- **语义变更（有意）**：A 阶段目标由残差 $T\approx A_{\text{Full}}-A_{\text{Local}}$ 改为
  $T\approx A_{\text{Full}}$，`training.residual_transfer_loss` 及其单测随之删除（E2 的 A 阶段语义留在 `h100` 分支）。

### A 阶段：完整 attention 迁移 + 视觉贡献监督（§3.2）

- `training.py`：新增 `full_attention_transfer_loss`（视觉/文本 query 分别按教师能量归一化后取平均）、
  `visual_transfer_loss`（分子分母统一用教师完整输出能量）、`visual_queries`（图像之后前 64 个文本 query）。
- `transfer.py`：钩子改为新目标；教师视觉贡献用**同一个** `attention_mask`/`scale`/`is_causal` 的
  SDPA、只把非视觉位置的 V 置零得到，softmax 分母不变；学生视觉贡献由第二次前缀 pass（V 掩码）得到。
  每步记录 `teacher_reconstruction`（重算 SDPA 与教师捕获输出的相对差，超过 0.05 直接报错），
  逐层诊断新增 `memory_rms`、`readout_rms`、`vision_*`。
- A 阶段 `identity` 增加 `objective='full_attention+visual'`，旧 A checkpoint 因此不可 resume（已完成，属有意保护）。

### B 阶段：CE + 教师 KL（§3.3）

- `training.py`：新增 `kd_loss`（forward KL、shifted-label 对齐、与 CE 同一全局 target 计数归一化、
  $\tau^2$ 缩放）与常量 `KD_WEIGHT=1.0`、`KD_TEMPERATURE=1.0`。
- `sft.py`：`kd_weight > 0` 时装载第二份冻结教师（bf16、`eval()`、`requires_grad_(False)`），
  与学生共用同一份 `inputs_embeds`；逐 micro-batch 先算教师前向并立即求 KL，再释放教师 logits。
  `base.json`/`full.json` 无 `kd_weight` 键，旧路径行为不变。

### 配置与入口（§3.4）

- 新增 `configs/p32.json`：anchor 列表为空、候选层 0–31、`micro_batch_size=8`、`kd_weight=1.0`、
  `kd_temperature=1.0`。`config_sha256 = 924c5253a7b30cd0…`。
- `config.py`：anchor 校验放宽为「必须是 `LAYOUT_ANCHORS` 中的某一个」，`kd_weight`/`kd_temperature`
  与 `base_lr` 同机制（出现才校验）。
- `sft.py`：`--layout` 增加 `P32`，安装时传配置的 anchor 列表，并断言 `--layout` 与配置一致。
- `lmms_model.py`：接受 `layout='P32'`，按 checkpoint 的布局安装对应 anchor 列表。

### 测试（§3.5）

- 更新：`test_model_hybrid.py`（纯 TTT 下无 local 窗口）、`test_transfer.py`（新目标 + 教师重建一致性）、
  `test_training.py`（新损失与 `visual_queries`）、`test_lmms_model.py`（新增 P32 布局参数）、`test_sft.py`。
- 新增：`test_pure_prefix_ttt_has_no_softmax_attention_and_no_kv`（TorchDispatchMode 统计
  `scaled_dot_product_attention` 调用为 0，缓存只有 FP32 状态且字节数精确）、
  `test_late_answer_loss_reaches_early_visual_writes`（梯度路径）、
  `test_kd_is_forward_kl_on_shifted_targets_with_one_scale`、两个 `configs/p32.json` 校验用例。
- **CPU 套件 207 passed / 0 failed**（`OMP_NUM_THREADS=1 uv run --locked pytest -m 'not gpu'`）。

下一步：启动脚本与 3 步 GPU smoke（显存、步时、纯度）。

## 2026-09-18 启动脚本、单卡 smoke 与 A 阶段正式开训

### 启动脚本（不写死单机）

- 新增 `scripts/experiments/prefix_ttt_p32/launch_a.sh` 与 `launch_b.sh`：只负责设置
  `PYTORCH_CUDA_ALLOC_CONF`、`OMP/MKL_NUM_THREADS` 与 argv，然后 exec 现成的
  `scripts/experiments/prefix_ttt_h100/run_multinode.sh`。
  `NNODES` / `NODE_RANK` / `GPUS` / `MASTER_HOST` / `MASTER_PORT` / `CUDA_VISIBLE_DEVICES` /
  `OUT` / `STAGE_A` 全部由环境变量提供，默认单机 8 卡；h100-1 空闲后同一脚本加 `NNODES=2 NODE_RANK=1`
  即可，无代码改动。累积步数由 `accumulation_steps(world, micro)` 推出（world=8→2、16→1）。
- 新增 `scripts/experiments/prefix_ttt_p32/run_eval_p32.sh`：官方 lmms-eval MME/POPE + 带内打点，
  `RUN="gpu:label:task ..."` 选择任务；`p32` 标签带 `--checkpoint`，`e0` 标签不带。

### 单卡 smoke（`--max-steps 1`，输出到 `artifacts/experiments/prefix_ttt_p32/smoke-a/`）

命令：`CUDA_VISIBLE_DEVICES=0 NNODES=1 GPUS=1 OUT=artifacts/experiments/prefix_ttt_p32/smoke-a \
bash scripts/experiments/prefix_ttt_p32/launch_a.sh 0 --max-steps 1`

| 观测 | 数值/结论 |
| --- | --- |
| 教师重建一致性 | `teacher_reconstruction = 0.0`（重算 SDPA 与教师捕获输出**逐位相同**，证明 mask/scale 语义与教师一致） |
| 峰值显存 | 18.93 GiB（teacher bf16 权重 + 单层反向图 + 32 层分支）；单卡 |
| 步时 | 63.1 s（world=1、micro 8、累积 16），即约 3.9 s/micro-batch |
| 32 层梯度 | 全部非零、有限（grad_norm 20.9–166.8） |
| 初始诊断 | `normalized_sample_mean≈1.0`（gate 为 0 时读出为 0，误差等于教师能量）；`gate_rms=readout_rms=0`；`vision_normalized` 在 0.03–0.67 之间，随层而异 |
| 状态幅度 | `state_rms` 0.048（层 0）→ 1.17（层 30）、`memory_rms` 0.35→5.4，随深度增长而非随序列长度爆炸 |
| 产物 | `latest.pt` 453 MB、`run.json`（`objective='full_attention+visual'`、`diagnostic_only=true`）、`steps.jsonl`、`result.json` |

**A 阶段目标的物理含义得到确认**：`normalized_sample_mean ≈ 1` 说明第 1 步的读出确实为零、误差等于
教师完整输出的全部能量；`vision_normalized` 是教师视觉贡献占完整输出能量的比例（层 27 约 3%、
层 21 约 67%），这正是辅助监督要约束的量。

### A 阶段正式开训

- 命令：`CUDA_VISIBLE_DEVICES=0..7 NNODES=1 GPUS=8 OUT=/data/shared/weights/prefix-ttt/training/A-P32 \
  setsid nohup bash scripts/experiments/prefix_ttt_p32/launch_a.sh 0 > /tmp/p32-a.log 2>&1 &`
- 启动后实测：稳态 **2.9 s/步**（world=8、micro 8、累积 2），391 步预计约 19 分钟；
  第 19 步的 `gate_rms` 已从 0 升到 0.006–0.017，`readout_rms` 升到 0.006–0.042，说明 gate 正在学习。
- 本机 8 卡独占；h100-1 未接入。

## 2026-09-18 A 阶段完成、GPU 验收、B 阶段存档死锁的定位与修复

### A 阶段正式结果（`/data/shared/weights/prefix-ttt/training/A-P32/`）

`result.json`：`complete=true`、`global_step=391`、`samples_seen=50000`、`world_size=8`、
`objective='full_attention+visual'`、`config_sha256=924c5253…`。墙钟 00:47→01:06（19 分钟），
稳态 2.8 s/步、峰值显存 18.6 GiB/卡；`teacher_reconstruction` 全程 **0.0**（重算 SDPA 与教师捕获输出逐位相同）。

逐层诊断（全 32 层平均，第 1 步 → 第 391 步）：

| 指标 | 第 1 步 | 第 391 步 |
| --- | --- | --- |
| 归一化完整输出误差 | 1.228 | **0.733** |
| 视觉 query 误差 | 0.999 | 0.477 |
| 文本 query 误差 | 0.999 | 0.751 |
| 视觉贡献误差 | 0.246 | **0.115** |
| `gate_rms` | 0 | 0.0184 |
| `readout_rms` | 0 | 0.0434 |
| `state_rms` | 0.708 | 0.716（未随步数发散） |
| `grad_norm` 均值 | 57.7 | 3.79 |

层 0 拟合最好（1.48→0.27），深层较差（0.63–0.80）。**读出幅度仍远小于教师 attention 输出
（`readout_rms≈0.043` 对教师约 1），即 A 阶段只恢复了约 27% 的输出能量**——这是 B 阶段起点
`loss=44.1`、`grad_norm≈5000` 的直接原因，作为已知事实记录，不在本轮追加 A 阶段预算。

### GPU 验收

- GPU 套件 `uv run --locked pytest -m gpu`：**84 passed**（132 s），含 FLA 算子验收与分段 prefill/decode。
- 纯度与一致性检查 `scripts/experiments/prefix_ttt_p32/check_purity.py`（真实 7B + A 阶段权重，
  输出 `artifacts/experiments/prefix_ttt_p32/purity.json`）：

| 项 | 结果 |
| --- | --- |
| LLM 前向内 `scaled_dot_product_attention` 调用 | **0** |
| 请求级缓存字节 | **67 108 864 B = 64 MiB**（= 32 层 × 32 头 × 128 × 128 × 4，FP32 状态） |
| 历史 KV 字节 | 0 |
| 整段 vs 分段 prefill（logits 相对差） | 5.1% |
| 整段 vs 逐 token decode | 6.7% |
| 分段 vs 整段 的最终状态相对差 | 1.4% |

  对照：**E0（原模型）同方法的相对差是 0.7%（分段）/ 2.0%（逐 token）**，即 P32 的分块敏感性约为
  E0 的 3 倍，来自 FLA 分块核在 bf16 下的归约顺序差异；不是结构缺口（结构缺口会给出 O(1) 相对差）。
  评测走的就是 cached 路径，自洽。

### B 阶段存档死锁：定位与修复（本轮最重要的工程记录）

**症状**：B 阶段正式命令在跑完第 1 步后停止推进；rank 0 的 GPU 空闲、其余 7 张卡 100% 利用率但只有
约 115 W（NCCL 自旋等待的签名），**日志里没有任何 traceback**。stage A 与它的 16 次存档从未出现该问题。

**定位过程**（每一步都由证据推动，不是猜测）：

1. 给 `save()` 插入阶段打印（`gathering-rng` / `built` / `written`）→ 卡点位于打印之后、文件出现之前；
2. 把 `save()` 里的 `dist.gather_object` 换成**每卡 RNG sidecar**（`runtime.save_rng`/`load_rng`，
   与全量微调已有的 `latest.pt.optim-rank*` 同一模式）→ 卡点前移，说明集合通信只是其中一环；
3. 再把周期性存档拆成「只存权重」与「带优化器状态」两种，并在两者之间加打印 →
   3 步 smoke 显示：第 1 步（只存权重）**成功**、第 2/3 步 **5.23 s / 4.15 s**、
   第 3 步（`step == stop`，带优化器）**再次卡住**。

**根因**：`sft.py` 的 LoRA 分支存档引用了**未定义的名字 `optimizer`**（它只作为
`schedulers = [LambdaLR(optimizer, …) for optimizer in optimizers]` 的推导循环变量存在，
Python 3 不会把它泄漏到函数作用域）。该 NameError 只在 rank 0 触发，异常退到 `finally` 的
`dist.destroy_process_group()` 时，其余 rank 仍在集合通信中等待，**进程组析构死锁**，
于是既没有 traceback、也没有正常退出。全量微调分支用 `optimizers[0].optim…` 绕开了该名字，
所以上一轮 E2-full 没有暴露它；这是最近一次重构引入的潜伏缺陷，此后未再跑过 LoRA 分支的 B 阶段。

**修复**：`sft.py` 两处改为显式 `optimizers[0]`（resume 的 `load_state_dict` 与存档的 `state_dict`），
不引入别名。另用 pyflakes 扫过 `src/`，只剩两处无关的既有告警（`lmms_run` 的注册导入、全量分支的
`enabled` 局部变量）。

**同时保留的两项改进**：

- 存档路径**不再有集合通信**：每卡写自己的 `latest.pt.rng-rank{rank}`，resume 各读各的；缺失 sidecar
  时该 rank 保留初始 RNG 并打印提示。
- 周期性存档**只存权重**（约 0.35 GB，原约 1.5 GB），优化器状态只在 `pilot.pt` 与结束时写；
  resume 若遇到没有优化器状态的滚动存档，会打印「Adam restarts cold」而不是报 KeyError。

**验证**：3 步 smoke（`--max-steps 3`，输出 `artifacts/experiments/prefix_ttt_p32/smoke-b/`）三步全部完成、
三次存档均打印 `weights → built → written`、`result.json` 正常写出、进程正常退出、显存归零。

### B 阶段正式训练（Supervisor 托管）

- 新增 `configs/supervisor/supervisord.conf`：管理器 socket/pid/日志在 `/var/tmp/prefix-ttt-1000/`，
  程序 `p32-a` / `p32-b` 使用绝对 uv 路径、显式 `PATH`/`CUDA_VISIBLE_DEVICES`/`OUT`/`STAGE_A`，
  `autostart=false`、`autorestart=false`、`startretries=0`（失败不自动重试，先查后启）。
- 启动：`/data/mjyang/.pixi/bin/supervisord -c configs/supervisor/supervisord.conf` 后
  `supervisorctl -c … start p32-b`。**训练进程独立于终端与 harness 会话**；`status` 返回码 3 表示有程序处于
  STOPPED，不代表管理器不可用。
- 首次启动实测：**约 4–5 s/步**（第 1 步 42 s 为首次 kernel 编译），5182 步预计约 **6.5 小时**，
  显存 42–52 GiB/卡（学生 + 教师两份 bf16 权重 + 激活 + KD logits）。
- 待观察：起点 `loss=44.13`、`grad_norm≈5000`（A 阶段预热不足），需盯住前 155 步 warmup 与
  前 391 步 pilot 是否稳定下降。

## 2026-09-18 B 阶段完成：训练轨迹与负面信号

- `result.json`：`complete=true`、`global_step=5182`、`samples_seen=663248`、`world_size=8`、
  `pilot_reached=true`、`diagnostic_only=false`；Supervisor 状态 `EXITED`（正常退出）。
  墙钟 01:38→08:35（6 小时 57 分），中位 4.77 s/步，显存稳态 42–52 GiB/卡、功耗 500–610 W。
  产物：`latest.pt`（1.41 GB，含优化器状态）、`pilot.pt`、`steps.jsonl`（5182 行）、`run.json`、
  每卡 `*.rng-rank*`。
- 391 步 pilot 正常存档；全程无非有限 loss / 梯度。

### 训练 loss 与 E2（上一轮 LoRA 版混合布局）同窗口对比

| 步区间 | P32（纯 TTT） | E2（A9-T23 混合，LoRA） |
| --- | --- | --- |
| 1–50 | 31.2108 | 6.5135 |
| 50–100 | 23.9825 | 3.1028 |
| 100–200 | 22.6173 | 1.7896 |
| 200–400 | 19.4907 | 1.2452 |
| 400–800 | 14.9370 | 1.0319 |
| 800–1600 | 10.1539 | 0.8969 |
| 1600–3200 | 6.9343 | 0.8130 |
| 3200–6400 | 5.7346 | 0.7757 |
| 4800–9600 | **5.6700** | **0.7700** |
| 末 100 步 | 5.6229 | ~0.77 |

**P32 从 44 降到约 5.7 后基本平台化，比 E2 高约 7 倍**。这是本轮最重要的负面结果，必须在总结中
如实呈现，并且它先于任何评测分数就说明：按当前配方，纯 TTT 连训练集都没有拟合到 E2 的量级。

可能原因（按当前证据排序，尚未区分）：

1. **预热不足**：A 只恢复到教师 attention 输出能量的约 27%，B 的预算大量花在"从坏起点爬出"；
   支持证据是起点 loss 44 与 grad_norm 5066，以及 loss 在前 1000 步的快速下降；
2. **参数化容量**：每层只有一个固定 $1/128$ 尺度的外积记忆、读出无归一化、没有 Local-32 / anchor 兜底，
   可能无法表达 softmax 的尖锐选择；
3. **优化预算**：LoRA $2\times10^{-5}$、新模块 $10^{-4}$、5182 步，对更难的问题可能不足。

区分 1 与 2 的判据：评测结束后用 `scripts/experiments/prefix_ttt_p32/compare_readout.py` 量出
**B 之后每层读出与教师 attention 输出的幅度比**——仍远小于 1 指向 1，接近 1 指向 2。

### 评测（GPU 7，串行）

- 7 号卡是当前唯一可用卡（0–6 号被其他进程占用约 1 GiB/卡）；MME 与 POPE **串行**在同一张卡上跑，
  避免并发污染延迟口径。产物：`/data/shared/weights/prefix-ttt/eval-p32/{p32/{mme,pope},cost/}`。
- 加载路径 smoke：4 条 MME 样本全部正常生成（`--limit 4`，输出 `/tmp/p32-smoke-eval`），证明
  P32 checkpoint 的装载（装 TTT → 装 LoRA → `load_trainable` → 合并 → 去 PEFT 包装）可用。

## 2026-09-18 评测结果、成本、以及教师 KL 被放大 8 倍的实现缺陷（本轮主结论）

### 官方分数（lmms-eval 0.7.2，独占 GPU 7，MME/POPE 串行）

| 模型 | MME Perception | MME Cognition | POPE Accuracy | POPE F1 |
| --- | --- | --- | --- | --- |
| E0 | 1479.64 | 349.29 | 0.8548 | 0.8397 |
| E2（LoRA） | 1429.49 | 278.21 | 0.8501 | 0.8357 |
| E2-full | 1456.97 | 297.50 | 0.8541 | 0.8392 |
| **P32** | **984.84** | **286.79** | **0.8024** | **0.7886** |

POPE 的 precision 0.848 / recall 0.737，即倾向回答 no、漏检物体。MME 子任务里
`posters 0.425`、`artwork 0.540`、`celebrity 0.638`、`landmark 0.698`、`scene 0.785`，
认知类 `commonsense_reasoning 0.586`、`text_translation 0.650`。

### 推理成本（中位数，独占条件）

| 模型 | Prefill | TPOT | 峰值显存 | 缓存 |
| --- | ---: | ---: | ---: | --- |
| E0 | 23.0 / 22.8 ms | 12.8 / 12.4 ms | 13.56 GiB | 321.0 MiB（全 KV） |
| E2 | 62.1 / 62.3 ms | 31.1 / 31.7 ms | 13.61 GiB | 147.8 MiB |
| **P32** | **40.0 / 39.5 ms** | **26.5 / 26.4 ms** | 13.65 GiB | **64.0 MiB（纯状态，与长度无关）** |

缓存数字与设计完全一致（$32\times32\times128\times128\times4=64$ MiB）；P32 比 E2 更快，
但相对 E0 是 1.74× prefill、2.07× TPOT。

### 实现缺陷：`kd_loss` 的 world_size 因子（我的错误，已修复）

- **症状**：训练日志 loss 平台约 5.7，但用最终权重直接测量只有 1.28–1.66，差 3–4 倍。
- **根因**：`sft.py` 把 `world` 传给了 `kd_loss`，而 `kd_loss` 会乘 `world_size`；本项目的梯度归约是
  手动 sum all_reduce，`token_normalized_ce` 在调用点用默认 `world_size=1`。有效目标因此是
  $\text{CE}+8\,\text{KL}$，而非方案规定的 $\text{CE}+\text{KL}$。
- **A/B 证据**（同权重、同 128 样本、第 392 步，从 `pilot.pt` 恢复）：
  原代码 loss **14.5922** / grad_norm 103.09（与原训练日志逐位一致）；
  修复后 loss **3.7918** / grad_norm 23.05。
  账目核对：最后一批样本 CE 1.138 + 8×0.524 = 5.33 ≈ 日志第 5182 步的 5.03。
- **修复**：`training.kd_loss` 删除 `world_size` 参数，调用点同步；单测改为断言各 rank 贡献之和等于
  一次前向的值。`tests/test_sft.py` 通过。
- **影响**：本轮权重是在偏离方案的目标函数下训练的，**其分数不能作为纯 TTT 架构的公平检验**。

### 真实损失与读出幅度诊断（区分"预热不足"与"表达力不足"）

| 数据（最终权重） | 学生 CE | 学生 KL | CE+KL | 教师 CE |
| --- | ---: | ---: | ---: | ---: |
| dev 前 48 条 | 1.005 | 0.233 | 1.238 | 0.757 |
| train 前 64 条（batch 8） | 1.035 | 0.245 | 1.280 | 0.759 |
| train 最后 64 条 | 1.138 | 0.524 | 1.662 | 0.593 |

- 训练/评测前向一致性核对：同一批数据在 grad 开启（FLA 打包路径）与关闭（dense 掩码路径）下
  CE/KL 完全相同（1.0313 / 0.1945），**排除了"训练用了退化前向"的可能**。
- 读出幅度比（学生 o_proj 输入 RMS / 教师 attention 输出 RMS，4 条 dev 样本）：均值 **1.36**
  （min 0.38、max 7.17；层 0/1/2 = 3.73/7.17/3.55，深层 0.54–0.83）。
  A 阶段结束时该比值约为 0.04。**结论：幅度不足已被 B 阶段自行修好，剩余差距在方向/内容层面。**

### 下一步（建议，未执行）

1. **用修复后的目标函数重跑 B 阶段**（8 卡约 7 小时），A 阶段不动，保持单一变量；
   当前只有 7 号卡可用，8 卡重跑需等 0–6 号卡释放。
2. 若重跑后 Perception 仍远低于 E2，再按单变量依次检验结构改动（恢复读出归一化 / 保留短窗或 anchor /
   提高状态表达力），不把多个变量混在一起。
3. 本轮已把结果与缺陷写入 `experiment_summary.md`。

## 2026-09-18 r2 任务书与提交

- 新增本目录 `plan_r2.md`：第二轮（A/B 都从零重跑）的任务书，含每阶段的调整、依据、验收判据、
  代码改动清单、预算与算力待定项。第一轮的 `plan.md`、`ledger.md`、`experiment_summary.md` 保持原样。
- 单卡 A 的可行性结论（用户提问）：技术上等价、不影响 B——B 的 A checkpoint 身份校验不含 world size，
  且账本已实测跨 world 的目标函数一致（第 391 步 CE 差 0.4%）。代价是单卡每步要串行 16 个 micro-batch：
  1 遍约 6–7 h、3 遍约 19–21 h（8 卡分别是 20 min / 55 min）。因此推荐等 8 卡；若短期只有 7 号卡，
  可单卡跑 1 遍 A 并放弃 r2 的"A 加长"。
- **提交**：本轮全部代码、配置、脚本与文档按用户 2026-09-18 授权提交到 `full-ttt` 分支，作者
  `Codex <codex@openai.com>`，提交标题 `Replace all 32 LLM attention layers with pure Prefix-TTT`。
  **未推送**（推送需要当轮明确授权）。
- 分支状态（提交前核对）：`h100` 的 `eb3d2c6`、`60be803` 两个 commit **已在本地提交但未推送**
  （`origin/h100` 仍停在 `1c61b28`）；`full-ttt` 从 `60be803` 分出，因此已包含这两个 commit。
- 当前状态：**等待 8 卡空闲**（0–6 号卡被他人任务占用，7 号卡空闲）；用户 2026-09-18 决定暂不继续推进。
