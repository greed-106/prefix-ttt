# P32 纯 Prefix-TTT 阶段总结：训练、评测、成本与一个必须记录的实现缺陷

本文件是 `docs/experiments/2026-09-17-prefix-ttt-p32/` 这一阶段的实验总结，对应主账本
[ledger.md](ledger.md)、方案 [plan.md](plan.md)。数字都能在「原始数据」一节列出的文件中复查。

## 1. 这一阶段要回答什么

把 LLaVA-1.5-7B 的**全部 32 个 LLM self-attention 子层**换成纯 Prefix-TTT（不保留 Full MSA、
SWA 或 Local-32），只继承 QKV/RoPE/o_proj 与原 FFN，用两阶段训练把原模型的能力迁移过去：

- **A 阶段**：50k 样本上逐层迁移教师的**完整 attention 输出**，并额外监督教师的**视觉来源贡献**；
- **B 阶段**：纯 P32 + rank-32 LoRA，用原始回答 CE 加教师 KL 训练一遍 LLaVA-665k。

原模型只作为冻结教师与实验对照。数据、模板、预处理与评测协议全部沿用前两轮。

## 2. 做了什么

| 项 | 内容 |
| --- | --- |
| 分支 | 从 `h100` @ `60be803` 新建 `full-ttt`（先建分支后改代码） |
| 算子 | 纯 TTT 路径：`use_local=False`、读出取消 RMS 归一化（$y_t=g_t m_t$）、缓存只留 FP32 状态 |
| A 阶段 | 目标由残差 $A_{\text{Full}}-A_{\text{Local}}$ 改为 $A_{\text{Full}}$，新增视觉贡献项（$\lambda_V=1$，图像后前 64 个文本 query 位置） |
| B 阶段 | 新增教师 KL（$\lambda_{KD}=1$、$\tau=1$），冻结 ViT/projector/LLM 原权重，训练特征映射 + gate + 32 层 LoRA |
| 验证 | CPU 套件 207 passed、GPU 套件 84 passed、纯度检查（无 softmax attention、缓存恰好 64 MiB）通过 |
| 训练 | A：8 卡 391 步 19 分钟；B：8 卡 5182 步 6 小时 57 分，均由 Supervisor 托管（独立于终端与智能体会话） |
| 评测 | 官方 lmms-eval 0.7.2 MME/POPE，独占单卡串行、带内打点 |

## 3. 结果

### 3.1 官方分数

| 模型 | MME Perception | MME Cognition | POPE Accuracy | POPE F1 |
| --- | --- | --- | --- | --- |
| E0（原始 LLaVA-1.5-7B） | 1479.64 | 349.29 | 0.8548 | 0.8397 |
| E2（A9-T23 + LoRA） | 1429.49 | 278.21 | 0.8501 | 0.8357 |
| E2-full（A9-T23 + 全量微调） | 1456.97 | 297.50 | 0.8541 | 0.8392 |
| **P32（纯 TTT + LoRA）** | **984.84（−33.4%）** | **286.79** | **0.8024** | **0.7886** |

- **Perception 掉三分之一**，是"细粒度视觉判别"这一类的严重退化（POPE 的 recall 只有 0.737，
  倾向于回答 no、漏检物体）；
- **Cognition 286.79 略高于 E2**，说明不是整体崩坏。

### 3.2 推理成本（独占 GPU、同一协议，中位数）

| 模型 | Prefill | TPOT | 峰值显存 | 缓存（MME 口径） |
| --- | ---: | ---: | ---: | --- |
| E0 | 23.0 ms | 12.8 ms | 13.56 GiB | 321.0 MiB（全 KV，随长度增长） |
| E2 | 62.1 ms | 31.1 ms | 13.61 GiB | 147.8 MiB（KV 90.3 + 状态 46.0 + 窗口 11.5） |
| **P32** | **40.0 ms** | **26.5 ms** | 13.65 GiB | **64.0 MiB（纯 FP32 状态，与长度无关）** |

缓存的账目与设计完全一致：$32\times32\times128\times128\times4=64$ MiB，每层只保存关联状态，
不保存历史 KV；纯度检查用真实 7B 前向确认"LLM 内 `scaled_dot_product_attention` 调用数为 0"。
代价是同长度下 prefill 是 E0 的 1.74 倍、TPOT 的 2.07 倍（但**比 E2 更快**）。

### 3.3 训练轨迹与真实损失

- A 阶段：教师重建一致性**全程 0.0**；归一化完整输出误差 1.228 → 0.733，视觉贡献误差 0.246 → 0.115；
  `state_rms` 稳定（0.708 → 0.716，未发散）。
- B 阶段日志 loss 从 44.13 降到约 5.7 后平台化，比 E2 的 0.77 高约 7 倍。**这个数字不是真实目标值**，
  原因见 §4。
- 用最终权重在真实训练批次上直接测量（`scripts/.../dev_loss.py`）：

| 数据 | 学生 CE | 学生 KL | 学生 CE+KL | 教师 CE |
| --- | ---: | ---: | ---: | ---: |
| dev 前 48 条 | 1.005 | 0.233 | 1.238 | 0.757 |
| train 前 64 条（组批 8） | 1.035 | 0.245 | 1.280 | 0.759 |
| **train 最后一批（663168 起 64 条）** | 1.138 | 0.524 | **1.662** | 0.593 |

  即：模型的真实目标是 **CE + KL ≈ 1.28–1.66**，比教师低 0.25–0.55 nat/token。

### 3.4 读出幅度诊断（判断"预热不足"还是"表达力不足"）

用最终权重比较每层读出（o_proj 输入）与教师 attention 输出的 RMS：

| 层 | 0 | 1 | 2 | 8 | 16 | 24 | 31 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 学生/教师 | 3.73 | 7.17 | 3.55 | 0.83 | 0.56 | 0.58 | 0.54 |

**平均 1.36**（A 阶段结束时该比值约 0.04）。结论：**"幅度不足"已被 B 阶段自行修好**，
剩余差距是**方向/内容**层面的——线性注意力状态无法复现 softmax 那种尖锐的、基于内容的选取。

## 4. 必须记录的实现缺陷：教师 KL 被放大了 8 倍

**症状**：训练日志 loss 平台（约 5.7）与用最终权重直接测出的目标值（约 1.3–1.7）对不上。

**定位**：`sft.py` 调用 `kd_loss(..., world, ...)` 把 `world_size=8` 传了进去，而 `kd_loss` 内部会乘
`world_size`；但本项目的梯度归约是**手动 sum all_reduce**，`token_normalized_ce` 在调用点用的就是
默认 `world_size=1`。于是有效目标是 $\text{CE}+8\,\text{KL}$ 而不是方案钉死的 $\text{CE}+\text{KL}$。

**证据（A/B，同一权重同一批样本）**：从 `pilot.pt` 恢复、跑第 392 步：

| 代码 | loss | grad_norm |
| --- | ---: | ---: |
| 原训练（KL×8） | **14.5922** | 103.09 |
| 修复后（KL×1） | **3.7918** | 23.05 |

原训练日志第 392 步记录的正是 14.5922，与 probe 逐位一致，说明这是唯一差别。
账目也完全对上：最后一批样本 CE 1.138 + 8×0.524 = 5.33 ≈ 日志第 5182 步的 5.03。

**处置**：`training.kd_loss` 去掉 `world_size` 参数、`sft.py` 调用点同步修改，单测改为断言
"各 rank 贡献之和等于一次前向的值"；`tests/test_sft.py` 通过。

**结论的影响**：本轮的 P32 权重是在**与方案不符的目标函数**下训练的（蒸馏项过重、梯度被 KL 主导）。
因此 §3.1 的分数**不能当作纯 TTT 架构的公平检验**，只能当作"该配方下的一次结果"。

## 5. 顺带修掉的两个工程问题

1. **存档路径里的集合通信**：`save()` 原本用 `dist.gather_object` 收集各卡 RNG 状态；改为
   **每卡写自己的 `*.rng-rank{rank}` sidecar**，保存不再需要集合通信，慢的 peer 不能再拖死存档。
2. **周期性存档减负**：滚动 checkpoint 只存权重（约 0.35 GB，原约 1.5 GB），优化器状态只在
   pilot 与结束时写；resume 遇到没有优化器状态的存档会打印"Adam restarts cold"而非报 KeyError。

## 6. 下一步（建议）

1. **用修好的目标函数重跑 B 阶段**（8 卡约 7 小时），A 阶段保持不变（单一变量）。当前唯一可用的
   7 号卡不足以做 8 卡重跑，需要等 0–6 号卡释放。
2. 若重跑后 Perception 仍显著低于 E2，则证据指向**结构**而非配方，届时再按单变量依次检验：
   恢复读出归一化、给部分层保留短窗/anchor、或提高状态表达力（可学习写入尺度、更高秩的特征映射）。
3. 不建议在重跑前同时改动 A 的预算：那会把"KL 修复"与"预热加长"两个变量混在一起。

## 7. 原始数据与产物

- 训练：`/data/shared/weights/prefix-ttt/training/A-P32/`、`training/P32/`（`latest.pt`、`pilot.pt`、
  `steps.jsonl`、`run.json`、`result.json`）。
- 评测：`/data/shared/weights/prefix-ttt/eval-p32/{p32/{mme,pope},cost/}`。
- 检查与诊断：`artifacts/experiments/prefix_ttt_p32/{purity.json,readout-ratio.json,dev-loss-*.json}`、
  日志 `/tmp/p32-*.log`、Supervisor 日志 `/var/tmp/prefix-ttt-1000/`。
- 工具：`scripts/experiments/prefix_ttt_p32/{launch_a.sh,launch_b.sh,run_eval_p32.sh,check_purity.py,compare_readout.py,dev_loss.py}`。
- 托管：`configs/supervisor/supervisord.conf`（`p32-a` / `p32-b`，`autostart=false`、`autorestart=false`）。
