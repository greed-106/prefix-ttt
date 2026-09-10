# 正式 E0 基线与阶段 A 队列

2026-09-10后续更新：E0/A已全部完成，结果见e0_a_summary.md。当前新队列为configs/jobs-pilot.json：A整模型切换诊断→四卡E2 Pilot→四卡E1 Pilot，活动库queue-pilot.sqlite3；旧E0/A库保留。两Pilot均391步/50048样本，共用固定数据顺序与完整5182步scheduler，首步/每25步保存，后续从checkpoint继续。以下E0/A启动过程为历史。

2026-09-10，在四A40容量闭环与Supervisor接入完成后，用户要求加快推进、避免重复验证。本轮只补真实7B基座一致性及新入口的必要短测，随后提交正式队列。

## 已完成的前置

HF与桥接模型的首次回归因为测试强制HF eager而桥接为SDPA，projector误差超限。对齐双方实际SDPA后，图文projector/展开/logits、零增量LoRA与E0/E2非零门控缓存检查通过，原阈值未放宽。初次失败与修正后结果分别保存在artifacts/gpu/e0-regression.json和e0-regression-aligned.json。

新训练入口复用原LLaVA的LazySupervisedDataset、collator和多模态展开；data_pipeline仅核对固定manifest及提供原始索引/暂存适配，不建立新数据管线。B支持完整scheduler、Pilot和严格恢复；A通过逐层原投影hook取得o_proj前Full输出，旁路单独反传并释放，教师继续使用原Full输出。CPU必要检查证明教师输出不变、旁路梯度存在、token归一化与恢复逻辑；没有重跑已完成的容量或调度大套件。

## 当前队列配置与实际状态

2026-09-10评测框架切换后，四卡MME限量8条入口短测退出0，样本ID 0–7唯一完整。Supervisor正式消费者随后已启动；SQLite确认MME running并占用四卡，POPE/GQA/A queued。configs/jobs.json共4项，依次为官方lmms-eval==0.7.2的MME、POPE、GQA和阶段A。每项申请物理GPU 0/1/2/3全部四卡；评测通过accelerate启动4个进程，由框架分发样本并汇总，不再使用自定义分片/合并任务。活动数据库位置为/var/tmp/prefix-ttt-1000/queue.sqlite3；启动不等于全量评测或训练完成。

阶段A入口检查三个benchmark的原生lmms结果是否全量、非debug，并验证原生指标及逐样本日志完整性；不自行重新评分，也没有质量分数门槛。失败不自动重试，缺失或不完整基线阻止A训练。

正式基线使用官方0.7.2任务、提示与评分，模型适配器继承官方LLaVA生成方法，仅接入项目完整checkpoint加载。使用vicuna_v1模板、CLIP336处理、greedy/num_beams1；官方任务默认MME/GQA max_new_tokens=16、POPE=128，不统一覆盖为128。GQA12578、POPE9000、MME2374。入口限量预跑仅为debug，不计正式基线。

旧自定义评测入口、测试、配置及其artifacts/e0、artifacts/e0-preflight、旧源码/数据库快照和旧队列日志已按用户要求删除，旧结果不作为本阶段质量证据。原始benchmark、权重、训练manifest及已通过的CPU/GPU容量证据保留。lmms直接复用现有离线数据；HF_DATASETS_CACHE与LMMS_EVAL_DATASETS_CACHE均显式指向打包数据中的benchmarks/hf-home/datasets。

## 接下来的训练轨迹

阶段A计划在完整E0通过后自动接续：24候选层，固定50000条A顺序，四卡各micro1、有效batch128，391步，最后不足组使用实际样本数。每层独立裁剪，FP32新参数与状态、BF16计算；首步、每25步与最后保存原子latest.pt，包含optimizer/scheduler/RNG/样本游标。当前queued，不宣称已执行首步。

阶段A计划产物路径为artifacts/training/A/latest.pt及steps.jsonl/result.json；新E0原生结果与samples日志分别写入artifacts/lmms/e0/{mme,pope,gqa}/下的模型名称子目录。原始进程日志在队列目录。旧artifacts/run_sources/e0-a/与artifacts/queue/e0-a-initial.sqlite3已删除，不能作为新队列恢复来源；新源码快照为artifacts/run_sources/lmms072-e0-a/，初始数据库备份为artifacts/queue/lmms072-e0-a-initial.sqlite3。初始备份不是活动库，也不包含后续完成状态。

B入口已实现但未排队。A完成后先做整模型切换诊断，再执行E1/E2同预算Pilot，并从各自完整B scheduler继续；不能把容量debug或不完整A权重加载为正式E2起点。不要在当前不可变队列运行时修改其manifest追加任务。

Supervisor支持脱离终端运行，不保证容器重建后自动恢复。计划内停机前须正常停止、归档一致性数据库和checkpoint；数据库丢失时不得新建空队列重跑成功实验。
