# Prefix-TTT 主账本

目录迁移说明（2026-09-17）：本目录原 `evidence/` 已原样移至仓库根目录下的 `artifacts/experiments/prefix_ttt/evidence/`，逐文件字节与 SHA256 已核对。下文中的旧路径保留为运行时历史记录；实验事实不变。迁移详见 [kernel 主账本](../2026-09-11-prefix-ttt-kernel/ledger.md)。

## 2026-09-10 仅 E2 完整 B 续训准备

已确认真实恢复更新：E2第392步、samples_seen50176、targets11683、loss1.11955523、裁剪前grad_norm9.242995，均有限；LoRA/新增模块LR分别1.9891392e-5/9.9456961e-5，承接391步而非重启warmup。四卡约17.5–17.9GB显存，持续计算。当前只有E2完整B续训运行，E1没有训练排队。目标累计5182步/663248样本，当前Supervisor持久托管。

单独E2恢复任务已提交并RUNNING，独占0/1/2/3。初始一致性库备份artifacts/queue/e2-resume-initial.sqlite3，修正后的执行配置/源码快照artifacts/run_sources/e2-resume/。实际第392步待核查。

门槛定向5项测试通过（8.05s），核实旧续训失败后E2仍恰好391条步骤日志/末步391，未重复更新。新configs/jobs-e2-resume.json只含e2-b-resume-5182，核对E2通过报告后从pilot.pt恢复；Supervisor切至独立queue-e2-resume.sqlite3。旧诊断失败库与日志全部保留，无自动重试E1。

首批诊断结果：E2 passed，固定8条dev加权CE1.35726383（A后10.11157515），缓存检查通过。E1缓存step1最大绝对差0.265625>原0.25，相对0.01658869<0.02，退出1；没有放宽阈值，也不声称E1诊断通过。双报告门槛因此令E2续训在加载模型前失败，未发生更新。由于用户仅授权E2续训，修正过度耦合门槛为必须核对“所恢复模型”的独立通过报告：E2不依赖E1工程验收；E1问题保留、后续训练仍不提交。该修正不改变E2前向/训练/阈值；准备单独新E2恢复任务，不重跑成功诊断。

Supervisor已启动新队列，SQLite确认e2-pilot-diagnostic running，e1-pilot-diagnostic与e2-b-continue-5182 queued。仅E2有续训任务；E1条目仅只读诊断。快照artifacts/run_sources/e2-full/，初始一致性库备份artifacts/queue/e2-full-initial.sqlite3。诊断/实际恢复训练结果待核查。

SFT定向CPU测试5项通过，新增诊断入口CLI导入通过。E2原Pilot run/result/steps已归档至artifacts/training/E2/pilot-record，后续steps继续追加；原pilot.pt保持不变。Supervisor配置已改指jobs-e2-full.json/queue-e2-full.sqlite3，即将启动，E1不在续训清单中。

新增pilot_diagnostic.py及configs/jobs-e2-full.json：E2/E1各一次固定dev工程诊断，随后仅四卡E2从不变pilot.pt恢复至5182步。诊断顺序占四资源槽避免与后续训练重叠；E1没有训练任务。续训输出沿用E2目录，旧Pilot运行元数据将先归档，pilot.pt不覆盖；活动数据库使用新的queue-e2-full.sqlite3，保留旧成功库。

用户明确要求先诊断两个Pilot、只提交E2续训，不提交E1对照组续训。核实两个旧队列全成功、Supervisor EXITED、四A40空闲。新增可选Pilot报告门槛：两报告通过工程检查、配置/manifest与391步/50048样本一致，E2恢复checkpoint SHA必须与实际诊断对象一致；不设质量分数门槛。准备固定8条dev简短CE/生成/cache诊断，原benchmark不重跑，E1仅诊断且保留原checkpoint。

## 2026-09-10 Pilot 完成核查

queue-pilot.sqlite3的切换诊断、E2 Pilot、E1 Pilot均succeeded/exit0，Supervisor消费者13:55正常结束，四GPU空闲。两模型均正式391步/50048样本，逐步step/samples_seen/targets完全对应，loss/grad_norm日志全部有限。E2耗时约3小时26分、E1约2小时36分。

E2末步训练CE1.10585，末20步均值1.14615；E1分别0.542736/0.573154。这些是训练loss，不是dev或benchmark；E2质量恢复程度尚未评测。两模型pilot.pt/latest.pt均CPU mmap加载成功，保存global_step=391、samples_seen=50048、scheduler.last_epoch=391、四rank RNG与optimizer状态。complete=false表示完整B未结束，并非Pilot失败。

阶段总结pilot_summary.md；本轮仅状态/产物核查，无新任务提交。下一步固定dev与生成诊断，然后按完整轨迹恢复剩余4791步/613200样本，不重置scheduler、不重跑成功Pilot。

## 2026-09-10 后续启动准备

正式E2 Pilot已完成2步/256样本，loss8.84736、9.05157，梯度均有限（日志grad_norm为裁剪前105.45/83.81，实际按1.0裁剪）；首步latest.pt约1.2GB已原子保存。首步55.77s含启动编译，第二步31.30s，不用两步外推精确总时长。四A40真实训练中、E1 queued，Supervisor持续托管。本轮目标已从提交落实到真实SFT optimizer更新；后续核查Pilot而非重跑。

切换诊断已退出0，status=passed（仅有限值/cache工程检查）：固定8条dev全部回答token加权E0 CE0.26838286、A9切换CE10.11157515、teacher KL9.77045207，存在明显质量退化，不能称迁移保持质量。不按质量门槛自动改结构/停止；按原计划继续Pilot检验恢复。原始报告 artifacts/training/switch-A/result.json。SQLite确认E2 Pilot running、E1 queued，正式首步待核查。

新队列已由Supervisor启动，SQLite确认 a-whole-model-switch running（四槽独占），e2-pilot-50048/e1-pilot-50048 queued。执行源码/配置快照 artifacts/run_sources/pilot/，初始一致性库备份 artifacts/queue/pilot-initial.sqlite3。尚不宣称诊断通过或SFT首步完成。

switch_diagnostic.py已完成并通过语法/CLI导入检查：固定dev各4图文/纯文本，E0与正式A切换模型的全部有效回答位置CE/KL，每模态短greedy/cache检查沿用0.25绝对/2%相对阈值，无质量门槛。Supervisor配置改指jobs-pilot.json与queue-pilot.sqlite3，旧configs/jobs.json和queue.sqlite3作为成功历史保留。准备启动诊断和Pilot正式队列。

configs/jobs-pilot.json已建立：切换诊断申请四卡资源槽作为独占屏障（实际单卡诊断），随后E2/E1各四卡Pilot；独立queue-pilot.sqlite3，保留旧完成库不修改。SFT定向4项CPU测试通过；未增加消融或修改预算。

只读Pilot审查未发现固定四卡梯度/AMP/恢复阻塞；补充A配置SHA、完整步数和样本数校验，防止误载同manifest的不同配置。保存间隔设25步并保留首步保存。

用户授权启动后续。已核实旧SQLite四项全成功、Supervisor消费者EXITED、四GPU空闲，不重复E0/A。准备独立不可变 switch→E2 Pilot→E1 Pilot 队列，旧数据库保留。新增SFT整模型切换报告身份门槛，失败/缺失报告阻止训练；首步也保存latest用于真实启动核查。Pilot按完整5182步scheduler运行至391步/50048样本，不用max-steps debug模式。切换诊断入口并行实施中，尚未提交。

## 2026-09-10 07:48 完成状态核查

SQLite 四项任务全部 succeeded/exit_code=0；Supervisor消费者03:24正常结束，四A40当前均无显存占用。官方原生完整性检查再次通过：MME2374、POPE9000、GQA12578，limit=None、样本ID完整唯一。E0：MME Perception1481.122749/Cognition341.785714；POPE Precision93.552812%、Recall75.777778%、F1 83.732351%；GQA Accuracy61.273652%。

A result.json 确认 complete=true、diagnostic_only=false、50000样本/391步、world_size4及固定manifest SHA一致；steps.jsonl恰好391条，latest.pt存在（339875595字节）。调度器记录A耗时4854.16秒（约80.9分钟），本次未重新加载大checkpoint做额外模型验收。阶段总结见 e0_a_summary.md。

下一步仍为A后整模型切换诊断→E1/E2同预算Pilot→连续完整B；B未提交。本轮仅核查和记录，没有重跑成功任务或提交新任务。A完成不等于整模型质量达标。

## lmms 0.7.2 最新验证

training_progress.md、resume.md、supervisor.md 已同步官方0.7.2四项队列、当前运行状态及新备份路径；旧CPU移交清单明确为历史，不要求重复已通过测试。

正式 MME 已进入真实生成，四rank任务数594/594/593/593，总2374；检查时rank0进度179/594，并非只启动空进程。原生日志 /var/tmp/prefix-ttt-1000/e0-lmms072-mme.log。完整分数尚未产出。

正式新队列已启动：SQLite 核实 e0-lmms072-mme running，独占物理0/1/2/3；POPE、GQA、A queued。活动库 /var/tmp/prefix-ttt-1000/queue.sqlite3，初始一致性备份 artifacts/queue/lmms072-e0-a-initial.sqlite3，执行配置/源码及SHA快照 artifacts/run_sources/lmms072-e0-a/。旧自定义代码、预测与旧队列产物已删除且不恢复；新诊断证据与原始数据保留。A/B尚未开始训练，B仍等待A后整模型切换诊断。

四卡 accelerate MME limit=8 短测退出0，官方汇总原始2374/effective8，样本ID为0–7恰好一次；证据 artifacts/lmms/preflight-072-four.log 及同名目录原生输出。该结果仅diagnostic，不计质量基线。现在提交 configs/jobs.json 四项正式任务，经 Supervisor/SQLite 启动，不重复容量验收。

阶段 A 的旧自定义汇总门槛已替换为官方原生 results/sample 完整性检查：三个固定任务、limit=None、完整唯一 doc_id、固定样本数和有限原生指标，不重评分、不设分数门槛。定向 CPU 测试 8 passed / 1 deselected（9.33s）。四卡官方框架短测进行中。

新 configs/jobs.json 已准备：三个官方任务各使用四卡 accelerate 分发，随后四卡 A；MME/GQA 沿用官方 max_new_tokens=16，POPE=128，不统一覆盖。显式固定两种 datasets cache 环境变量，完全离线。当前仅配置，等待四卡短测与原生结果门槛完成后启动。

MME 两条真实GPU生成、官方评分与原生JSON/JSONL保存已完成；末尾表格打印误引用 evaluator.make_table 导致退出失败，已按官方 utils.make_table 修复。该结果为 limit=2 的诊断，不作E0；下一步四卡框架短测并重提交正式E0→A。

## 目标与固定目录

在完整 LLaVA-1.5-7B 上验证 A9-T23 的质量、状态容量与实际延迟取舍。任务定义见 [plan.md](plan.md)。本项目跨日持续更新本目录，不另建每日实验目录。

## 当前状态

lmms0.7.2的registry_v2适配完成，4项CPU参数恢复/官方方法继承检查通过；MME/POPE官方任务离线加载通过，GQA官方process_docs预构建派生缓存中。发现官方wheel索引无关video_holmes缺模板，采用公开TaskManager(include_defaults=False,include_path=三个已安装官方任务目录)，不复制修改task/scorer。新增lmms_run.py仅调用官方simple_evaluate/EvaluationTracker并自然传播异常；避免默认CLI吞异常和全任务索引问题，不实现评测循环或评分。uv默认组包含eval，防止普通uv命令意外移除评测依赖。

用户指定检查并使用lmms-eval0.7.2。已核对官方wheel SHA256 fb437fc40064f92e20f752d4d298bf77d72cb30ac9c841f78b6c5126c9bc8bc8与PyPI一致；0.7.2的MME/POPE/GQA YAML齐全，dataset_path及GQA二级图像加载仍为lmms-lab，可直接对应现有缓存。此前0.7.3命名变化不能外推0.7.2，现改锁0.7.2＋datasets4.0.0，准备官方新注册接口适配和必要加载验证。旧队列保持停止，旧自定义产物已清除，新队列尚未提交。

选择正式兼容版本lmms-eval0.5.0＋datasets4.0.0：核对0.5官方wheel的MME/POPE/GQA任务及GQA图像加载均仍指向lmms-lab，与现有缓存身份一致；4.0原生支持缓存List特征且兼容pyarrow19。0.6 wheel缺少所需YAML不采用，0.7仓库名不同不做未证明别名。正在uv锁定重同步，不改官方任务或数据内容。

lmms-eval0.7.3安装完成，但官方任务dataset_path已改为lmms-lab-encoder，而本地缓存属于lmms-lab；不做未经验证的仓库数据别名，正在选择兼容缓存的官方发行版。datasets3.6对GQA缓存的List特征元数据不兼容，拟固定支持该schema的4.0版，不改Arrow或缓存元数据。新增lmms_model.py为官方Llava最小加载适配，继承官方generate_until，不含自定义任务/评分；版本最终确认前未执行GPU评测。

旧自定义E0预测、MME预跑与旧运行源码快照已删除（约7MB），不保留可恢复副本；队列与日志继续按明确路径清理，原始benchmark缓存不动。lmms-eval选定PyPI固定0.7.3（Python>=3.10），datasets固定3.6.0以兼容已有Arrow缓存；添加uv eval依赖组，保持torch/transformers等核心锁定，准备阿里源下载，不做高频轮询。

用户要求废弃自定义benchmark并删除旧产物，改用lmms-eval后重新提交。已正常停止Supervisor消费者，GQA四分片此前完成、POPE四分片中断、A从未启动，四GPU全部释放。已删除旧evaluate.py、audit_benchmarks.py、test_evaluate.py、evaluation.json和旧jobs.json；正在清理旧预测/预跑/队列与源码快照，不保留这些作废产物。数据包、模型、固定manifest和仍有效训练/算子证据不动。lmms-eval官方task/scorer与最小模型适配接入中，尚未重提交。

Supervisor正式消费者RUNNING（pid9692），SQLite确认e0-gqa-0/1/2/3分别独占物理0/1/2/3运行，显存约14GB/卡；其余12项queued。初始一致性SQLite备份artifacts/queue/e0-a-initial.sqlite3，代码/配置快照及SHA位于artifacts/run_sources/e0-a/，不是新运行数据库。当前已从准备转入正式E0质量基线；A等待完整基线，不声称已开始A参数更新。

正式队列已提交并交Supervisor启动：configs/jobs.json固定16项（E0 GQA/POPE/MME各4分片、3合并、A50000）。SQLite=/var/tmp/prefix-ttt-1000/queue.sqlite3；各分片占1卡，合并占4卡仅作前序任务完成的资源屏障（CPU评分、不测GPU），确保A不早于全部分片退出；A检查三完整汇总，不以分数通过。A四卡micro1/global128、391步、50k一次、首步及每25步保存，超时48小时，不自动重试。成功容量测试未重复提交，B尚未排队，等待A后整模型切换诊断。正式产物artifacts/e0/及artifacts/training/A/。

E0必要回归已通过（e0-regression-aligned.json）：相同SDPA后端的projector/展开/图文logits、零LoRA、E0与非零gate E2 cache检查；阈值未放宽。评测入口2条真实MME预跑通过但明确diagnostic_only，不作质量基线。A入口transfer.py已完成，CPU2项验证原教师输出不变/旁路梯度/基线完整门槛；加入首步checkpoint和训练峰值记录用于直接监控正式首步，不追加重复大规模smoke。准备固定E0分片→完整合并→四卡A队列，A先验证三项非debug完整baseline。

必要7B回归首轮projector差异失败（relative7.18%，未放宽阈值）：定位到测试显式强制HF eager，但桥接基座实际SDPA，比较后端不一致。修正测试为HF默认SDPA并记录/校验双方后端后复验至e0-regression-aligned.json/log；原失败证据保留，尚不宣称修正后通过。E0 eval同时修复基座缺省mm_use_im_start_end的getattr读取，10项原局部测试通过。B入口进一步锁config和A权重SHA，拒绝debug A checkpoint。

新增gpu_regression.py、sft.py/data_pipeline.py、evaluate.py及固定evaluation配置，局部CPU检查2+3+10项通过；B正式入口支持固定全轨迹/Pilot/恢复，E0评测支持MME2374/POPE9000/GQA12578与分片严格合并。当前仅启动物理GPU0的必要7B E0回归（artifacts/gpu/e0-regression.log/json），不是重复容量测试；A入口实施中。E1/E2评测加载后续单独接入，不声称现E0入口已支持训练权重。

继续推进：用户要求缩减非必要测试、优先训练。已检查Supervisor运行且消费者STOPPED、无正式队列，四A40空闲；不重跑容量/调度验收。并行实施最小7B E0回归、固定manifest的B训练入口、离线benchmark适配，主线程实施A逐层残差迁移。新增data_pipeline.py复用原LazySupervisedDataset/展开/标签，只增加固定manifest索引和CPU暂存适配；正式训练尚未启动。

Supervisor接入收尾：主daemon PID8208、PPID1、无TTY，跨启动终端退出存活已验证；正式消费者STOPPED且无正式manifest/数据库，四GPU空闲。配置、日志与隔离测试状态已归档evidence/supervisor/，未修改平台启动入口。用户本次托管替换已完成；容器重建后的持久卷/自动启动不在本次已验证结论内，后续不再以systemd缺失阻塞模型实施。

Supervisor集成与调度器回归最终13项通过（8.49s）：脱离启动session、真实uv/环境/四卡分配标签、成功失败超时、重启不重跑、正常停止清理独立任务及孙进程、强杀消费者后恢复拒绝重复派发。artifacts/supervisor/integration-final.log保留证据；所有测试进程清理完毕，主Supervisor保持运行、正式消费者STOPPED。同步plan/AGENTS/README/resume及scheduler文档，systemd不再是本容器推进前置；未声明容器重建自动恢复或正式训练已开始。

Supervisor主服务已启动（socket权限0600、运行目录0700），正式消费者STOPPED，未创建空正式队列或提交训练。隔离集成首轮1通过/1失败：Supervisor组TERM与uv转发产生重复信号，第二次KeyboardInterrupt打断scheduler清理，DB遗留running。测试进程已清理；初始日志artifacts/supervisor/integration-initial.log。已针对性修复scheduler首次终止后及清理期间忽略重复TERM/INT，完成后恢复原handler；准备复验，不更改队列分配/重试规则。

2026-09-10 Supervisor接入开始：用户明确批准替代user systemd，已定位pixi Supervisor4.3.0，当前无运行中的Supervisor/队列，既有GPU短测不重复提交。更新AGENTS授权与configs/supervisor/supervisord.conf，正式消费者autostart=false/不自动重启，待manifest准备后显式启动；本轮不提交训练。/var/tmp为本地overlay，支持本容器运行，但容器重建持久性未经保证，不能当永久数据库卷。准备隔离CPU生命周期验证。

2026-09-10本轮GPU容量任务完成：四A40 E2 DDP有效batch128、256条真实近2048样本、2次AdamW更新及rank0保存重载全部通过。最高allocated16.77GiB/reserved27.88GiB，四rank梯度一致，feature门控打开后梯度非零，无OOM。短测退出码0，当前无GPU任务，四卡已释放。GPU无需更换；正式A/B尚未启动，用户systemd不可用仍是持久托管阻塞，另需按原计划完成正式runner和E0/生成等前置验收。完整总结gpu_capacity.md。

2026-09-10 GPU恢复：用户授权使用全部四张空闲A40（物理0/1/2/3，每卡49140MiB，无NVLink，PCIe PXB/PIX）。驱动580.65.06；CPU quota现为12核。已检查工作树及已有产物，既有队列目录不存在，无成功GPU任务可重复提交。开始FLA GPU验收与真实7B SFT smoke入口实施；尚未证明SFT可运行。当前PID1=bash，user systemd总线仍不可用，正式长任务托管未满足，但不阻塞短时GPU smoke。

2026-09-09：CPU准备阶段已完成。最终统一测试145通过、4项GPU跳过；全量图片与标签审计、固定清单和哈希核对完成。无正式A/B训练或质量benchmark结果。用户挂载GPU并重启后按resume.md继续；阶段总结见cpu_summary.md。

## 完成项

- E2单A40真实7B训练smoke通过2个AdamW step（展开2023/2020），梯度有限；step1 gate有梯度/feature为零符合零门控初始化，step2 feature梯度非零。step2 decoder训练1.886s，allocated峰值17122100224B、reserved17834180608B，权重/optimizer保存重载通过。首步73.27s含编译，不作为稳态速度。日志artifacts/gpu/e2-single.log与e2-single/rank-0.json；此短测有效batch1，不是正式batch128轨迹或A后权重。下一步独占四卡DDP验证。

- 2026-09-10 FLA GPU验收17项通过：原4项tile16/32/64/128输出/状态/全部输入梯度通过（294.60s含首次编译）；新增13项长度/因果/过去影响/分段尾状态/跨状态梯度通过（29.26s）。证据artifacts/gpu/fla-tests-fixed.log、acceptance.log/xml。没有放宽误差阈值，没有reference回退。GPU0/2测试进程已结束，GPU1单卡7B诊断仍运行。

- 开发包本地校验：Ubuntu libpython3.10-dev 3.10.12-1~22.04.18 amd64，4765042字节，SHA256 c058675011ec4b64f5ad803d74b5fe7de0c720c3700683040da68d1fb5b44ed9，与apt元数据一致；传输源http://mirrors.aliyun.com/ubuntu/pool/main/p/python3.10/libpython3.10-dev_3.10.12-1~22.04.18_amd64.deb，来源身份Ubuntu。项目内提取，不安装到系统。configs/gpu-environment.json保存显式CUDA/CPATH/离线环境；尚未部署systemd服务。CPU回归已启动至artifacts/gpu/cpu-regression.log。

- GPU恢复环境采集已可导入FLA，无需升级锁定依赖；修正audit_environment中的CPU阶段硬编码GPU状态，新增可见设备名称/容量/计算能力。环境审计本身不声称完成GPU测试。

- 2026-09-10四卡NCCL正确性验证通过（sum=10），64MiB all-reduce预热后10次均值约4.77ms/rank，日志artifacts/gpu/nccl.log；不是SFT吞吐。configs/base.json明确授权物理0/1/2/3。开发头文件已本地提取，首次CPATH漏架构include父目录导致一次失败，修正后继续FLA测试，失败日志均保留。

- 阅读根 AGENTS.md，确认原三个参考仓库工作树干净；实际参考目录为 `preference/`。
- 确认 data 已解压，源 TAR 保留；本次不重复解压、不删除或迁移数据。
- 建立本账本、稳定实验目录和独立 uv 工程配置。
- 普通依赖采用用户本轮指定的阿里 PyPI 源，覆盖 AGENTS 的默认清华源；PyTorch/torchvision 使用官方 cu128 index。

## 运行项

当前无运行任务。以下2026-09-10运行过程记录为历史状态，均已结束；最终状态以本账本顶部及gpu_capacity.md为准。

- 四卡E2有效batch128第1步已完成，四rank grad_norm一致25.087963，全部梯度有限，71,313有效shifted targets，序列2000–2048；约82秒含编译，各rank reserved峰值约27.64GiB。继续第2步和保存重载，尚未将整体任务记为通过。

- 已启动四卡E2 DDP短测：2步×有效batch128=256条真实train样本，每卡micro1/accum32，固定近2048筛选且不改正式manifest。输出artifacts/gpu/e2-four-gb128.log与同名目录，各卡独占，无其他GPU任务。独立branch-profile已完成，Local实际使用pytorch_flash::flash_fwd_kernel，SDPA Q/K/V形状[64,32,32,128]（真实64个32-token块），不是2048²掩码。

- 新增gpu_branch_profile.py，独立记录生产形状[1,2048,32,128]的Local/FLA预热后前向耗时、实际CUDA kernel名称及SDPA输入形状；它是算子剖析，不冒充整模型SFT性能。尚待前三项短时任务结束后运行，以避免性能测量资源干扰。

- GPU短时任务由主线程统一分卡：原FLA tile验收物理0，E2单卡7B容量smoke物理1，扩展13项验收物理2，互不重叠。7B严格加载686张量成功、107085824可训练参数；训练step尚进行中。四卡DDP须待这些短时任务结束后独占全部卡，不提交正式长任务。

- 2026-09-10新增gpu_smoke.py（真实checkpoint+原数据集路径、近2048、E1/E2、DDP、梯度/AdamW/保存重载诊断）与test_gpu_acceptance.py（13项GPU扩展验收）。未把短时诊断当正式SFT轨迹。CPU回归145通过/4未选；FLA tile16/32已通过，其余进行中。

CPU阶段收尾时无运行任务。GPU恢复后的运行项见本节顶部；下面保留CPU历史实施记录，不表示旧任务仍在运行。

### 实施记录（CPU首批，尚待统一运行测试）

- 已归档完整任务书及四项修订至 plan.md；初始化独立Git分支 prefix-ttt。
- uv sync 已启动，日志 artifacts/cpu/uv-sync.log；尚未宣称安装成功。
- cpu_ops 已实现 reference、真实批量Local、feature/gate、FLA lazy adapter和cache存储基础，附CPU测试；完整Hybrid生成/分段Local尚未集成。
- model_bridge 已复制独立LLaVA，增加严格HF bridge、展开metadata、LoRA白名单和v1标签错误fail-closed；仅语法检查通过，运行测试待依赖。
- cpu_scheduler 已完成最小SQLite与服务模板，10项标准库CPU子进程测试通过；统一pytest及日志归档待执行，systemd/GPU未运行。
- uv环境安装成功：torch2.7.1+cu128、torchvision0.22.1+cu128、Transformers4.51.3，uv.lock已生成；长下载期间无高频轮询。
- 算子首轮66项CPU测试通过（ops-tests.log）；正在补Local分段缓存和finished写保护。
- 主线程新增configs/base.json、A模态均衡loss、B全局token归一化/DDP因子、完整轨迹scheduler基础和CPU梯度对照测试；这不是完整A/B trainer。
- 模型首次8项中1失败，发现legacy=false下v1旧偏移64≠真实65；已按原渲染前缀修复，扩展后13项CPU通过。无样本因该错误删除。
- 算子补测曾73通过1失败（旧fixture结束请求后才初始化cache）；纠正fixture顺序，最终74项通过。
- 实际7B头部/meta模型严格审计686 tensors、7,063,427,072参数，全覆盖无shape错误；没有7B前向。
- 调度器锁环境10项通过，并修复恢复时空PID identity比较的边界问题。
- 新增离线数据审计入口，结构/完整图片解码/原v1标签分开记录；审计不自动创建正式train/dev清单。准备全量执行。
- FLA固定git commit已加入uv sources，下一次sync安装，不宣称GPU验收。
- FLA sync成功，安装0.6.0并锁定指定git commit。新增environment审计入口与repo_audit.md。
- A/B loss基础3项CPU通过，包括空microbatch和不等长DDP加权；完整trainer仍未实现。
- reference发现外层autocast会降低einsum精度，已显式关闭并补回归；最新算子78通过、4项GPU跳过。
- 独立hybrid训练attention已实现并通过tiny测试，真实Local、共享RoPE后QKV、零gate和LoRA梯度可测；Hybrid生成adapter待实现。
- 四篇论文解读已归档paper_reading.md，调度说明scheduler.md已完成；它们未阻塞模型实现。
- 环境审计已落盘environment.lock.json：CUDA build12.8/nvcc12.8.93，设备数0。FLA导入报0 active drivers，标为无GPU下的后端未验证，不全局改CPU版PyTorch。
- 前512条真实原始数据的v1标签审计全部valid；正在全量665298条审计与349034张图片完整解码。
- tiny图文混合训练路径已通过LoRA反传、早期KV/feature梯度、checkpoint开关和strict保存重装；hybrid.generate明确报未实现，禁止全前缀重算伪装cache。
- 首次统一套件111通过/4 GPU跳过，pytest.log及pytest-results.xml；后续新增测试另复验。
- 全量标签审计已8192条valid；审计独立源码快照artifacts/cpu/audit_data.executed.py留存（SHA256见命令记录）。当前运行采用已确认的576 patch计数；下一版入口改为从config推导相同值，并修复decode失败需非零退出的归因边界。不因此更改或删除原数据。
- 全量标签审计推进至180224条valid，CPU cgroup实际配额4核（400000/100000），持续运行中。
- 新增manifest生成器并20项测试通过：完整审计门槛、传递分组、来源命名空间ID、分层配额、执行源码快照与当前代码哈希分列；尚未冻结全量清单。
- README提供可复现CPU入口与重启边界；新增数据审计分类fixtures。完整checkpoint SHA256计算已启动至artifacts/cpu/checkpoint-sha256.txt。
- 新增生产规模meta参数审计入口，验证23层新增参数及32层LoRA精确计数/冻结集合，不加载7B权重、不执行模型。
- 全部349034张图片完整RGB解码成功，缺图0/解码失败0（data-decode.log及data/data-audit.json）。
- optimizer真实step和完整RNG/optimizer/scheduler恢复测试1项通过，冻结基座逐tensor bit-exact不变；数据归因fixtures3项通过。
- 三个完整模型shard SHA256已计算完毕，checkpoint-sha256.txt保留，不等于已执行7B前向。
- 新增benchmark离线路径映射/字节SHA/Arrow可读性审计入口，原manifest不改，未运行质量评测。
- GQA/POPE/MME所有9个Arrow文件字节/SHA/读取验证完成，benchmark_paths.json归档。
- B optimizer分组基础新增固定新模块/LoRA不同LR与矩阵decay及白名单检查；补Pilot最小样本数边界，附CPU测试。
- Hybrid Transformers cache/greedy已接入：模型组29项通过，首段全padding布局6项通过，ViT一次/q后续1、finished/重排、首层TTT均有tiny CPU证据；早期“cache未实现”是历史状态，现已替代。
- 缓存测试首轮有6个测试代码NameError（冻结断言误放fixture末尾），纠正后通过，原日志model-cache-tests-initial.log保留。
- BF16 base/FP32新参数无外层AMP测试通过，hybrid内部新增投影明确AMP；上下文长度检查改为总prompt+generated上限。
- BF16 bridge首次与HF正式加载对照发现RoPE buffer被整体dtype cast量化（失败日志bridge-bf16-rope-initial.log），已仅转换parameters，保留原FP32频率；高位置至2047图文logits bit-exact通过，加载后仅.to(device)。
- 最终统一CPU套件145 passed / 4 GPU skipped（46.49s），pytest.log与pytest-results.xml完整保存；普通环境依赖下载不做短间隔轮询。
- 小型关键证据（统一测试log/XML、模型hash/header、环境与RoPE失败/修复日志）已复制至本目录evidence，便于随文档保存；大规模逐行数据审计继续保存在artifacts。
- 全量标签665298条完成：665296 valid、2原始空assistant，预处理错误0；147条展开长度>2048仍保有截断后有效监督。
- 固定清单已生成并核对：train663248/dev2048/A50000，集合无重复、A属于train且dev不重叠，原annotation SHA与asset_manifest一致。完整清单SHA246586be1ddd3d0cea09e17047e6d658c5e9a38637654e46906605578707a93b。
- CPU阶段总结、data_manifest_summary.json和resume.md完成；正式B规划5182步、warmup156、Pilot391步/50048样本，未执行训练。

- 主线程、cpu_ops、model_bridge、cpu_scheduler的本阶段任务均完成并集成。服务样例未安装/启动，无GPU任务。

## 失败项与环境限制

- 2026-09-10首次FLA GPU测试失败：Triton编译缺少Python.h（artifacts/gpu/fla-tests.log），不是GPU OOM。系统无python-dev且sudo需要密码，准备从系统已配置阿里apt源提取匹配3.10.12开发头文件到项目内，通过显式CPATH使用，不升级系统依赖。新增gpu_communication.py用于四卡NCCL短时正确性/通信计时，尚待执行。

- systemd 249 已安装；`systemctl --user list-units` 因缺少用户总线和运行目录失败。仅正式持久托管验收受阻，不阻塞 CPU 或最小 GPU smoke。
- CPU阶段未挂载可用GPU是历史限制；2026-09-10已挂载四A40，GPU算子与单卡训练显存已验证，真实模型生成仍未验收。

## 关键决策

- 不直接 import 只读参考目录；保留原 LLaVA 数据/标签/评测路径，只在独立副本增量适配。
- 标签兼容错误必须修复，不能当坏数据删除；Local-32 必须真实块批量计算。
- 顺序为最小闭环 → E0 → A → 整模型切换诊断 → E1/E2 同预算 Pilot → 连续完成 B。
- 调度、完整文档与模型实现解耦；最小账本和原始配置/日志始终保留。
- 长下载不进行短间隔高频轮询，等待期间继续独立实现工作。

## 产物位置

- 正式队列与下一阶段说明training_progress.md；E0已持续写出逐样本结果（检查时GQA合计872条），不是仅启动空进程。A/B入口src/prefix_ttt/transfer.py与sft.py，模型定义/预算不扩展，B未排队。

- Supervisor部署说明supervisor.md；正式配置configs/supervisor/supervisord.conf（主服务已运行，消费者STOPPED），隔离测试tests/test_supervisor.py；原始日志artifacts/supervisor/，归档evidence/supervisor/。重复TERM缺陷修复位于src/prefix_ttt/scheduler.py。

- 本轮GPU容量验收叙述与完整入口命令：gpu_capacity.md；四卡正式有效batch短测仍在运行，未提前宣称通过。

- 工程：`src/prefix_ttt/`、`third_party/llava/`、`tests/`、`pyproject.toml`、`uv.lock`。
- 文档：本目录；运行证据：`artifacts/cpu/`及本目录evidence。
- 数据：`data/llava-v1.5-assets-v1/`；原文件只读使用。

## 下一步

GPU恢复后以gpu_capacity.md及本账本最新状态为准，不重复成功的容量任务。完成真实7B E0回归/生成、正式runner与评测适配及用户systemd持久服务验收后，按E0→A→切换诊断→E1/E2同预算Pilot推进，不将debug容量权重用于正式训练。
# 2026-09-10 lmms 入口续接

- 0.7.2 首次 GPU preflight 在加载模型前失败：官方 `simple_evaluate` 实际要求 model_args 字符串，dict 触发 strip AttributeError；日志 artifacts/lmms/preflight-072.log。
- 已改为官方逗号分隔参数格式，保留原任务、生成和评分；准备一次有依据的重验。Supervisor 消费者 STOPPED，旧 SQLite 已删除，没有重复提交训练。
