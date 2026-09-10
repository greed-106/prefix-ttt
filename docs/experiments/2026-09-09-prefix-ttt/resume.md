# 挂载 GPU 并重启后的恢复入口

当前实际活动队列更新为configs/jobs-e2-resume.json / queue-e2-resume.sqlite3，仅E2完整B续训。先前queue-e2-full.sqlite3中E2诊断通过、E1缓存数值阈值检查失败、E2被双报告门槛拦截且未更新；原日志保留。现E2仅依赖自身通过报告，E1未提交训练、失败未隐瞒。最新恢复进度查ledger和E2/steps.jsonl，不从已过时Pilot起点重复训练。

最新授权：两个Pilot完成后，用户要求先诊断再仅续训E2。当前configs/jobs-e2-full.json、/var/tmp/prefix-ttt-1000/queue-e2-full.sqlite3含两项Pilot只读诊断及一项E2完整B续训；E1续训未提交。E2从pilot.pt第391步/50048样本恢复至5182步，原Pilot元数据已归档E2/pilot-record，steps继续追加。勿恢复旧队列重复Pilot；中断后须按最新checkpoint与日志审计，不能再次从391重跑已完成续训。

2026-09-10后续队列已启动：当前Supervisor使用configs/jobs-pilot.json和/var/tmp/prefix-ttt-1000/queue-pilot.sqlite3，切换诊断→E2 Pilot→E1 Pilot。旧queue.sqlite3的E0/A四项已全部成功，严禁重跑。Pilot保持完整5182步scheduler，止于391步/50048样本；下一次全量续训须从各自latest.pt恢复，不重新开始epoch。最新状态以ledger与SQLite为准。

最新运行入口：先读training_progress.md及ledger.md。旧自定义评测队列已停止，代码与旧结果已删除；configs/jobs.json现为lmms-eval==0.7.2的三个官方benchmark及四卡A，共4项。四卡8条MME短测通过后消费者已启动，SQLite确认MME running、POPE/GQA/A queued；实时状态以主账本和SQLite为准。活动库路径/var/tmp/prefix-ttt-1000/queue.sqlite3，旧库已按用户要求清除，不能拿旧备份恢复新队列。新源码快照artifacts/run_sources/lmms072-e0-a/，初始数据库备份artifacts/queue/lmms072-e0-a-initial.sqlite3不包含后续完成状态。先查询Supervisor和SQLite状态，不重复GPU容量测试；B尚未排队，等待A后切换诊断。

2026-09-10最新托管决定：用户已批准Supervisor替代systemd，主服务已部署，入口与恢复边界见supervisor.md。下文systemd限制是历史记录，不再要求等待systemd才能推进；服务器/容器重启后仍必须先确认数据库及遗留任务，不能丢库后创建空队列重跑。

2026-09-10四张A40的真实SFT容量闭环已通过，最新结果先读gpu_capacity.md和ledger.md。物理0/1/2/3已由用户授权，配置见configs/base.json。显式CPATH补丁见configs/gpu-environment.json；不再等待user systemd。

## 先读取状态，不重复工作

稳定目录不变。先读本目录 ledger.md、plan.md 和 cpu_summary.md（CPU收尾时生成），检查Git工作树以及现有SQLite队列/成功产物。不要重新解压、重下权重、重建dev或更换训练顺序。

CPU原始证据在 `artifacts/cpu/`，权重完整SHA256在 `checkpoint-sha256.txt`，环境和可训练参数清单在本目录。CPU tiny通过不是完整7B/GPU通过。

## 环境恢复（仅缺失或损坏时执行同步）

```bash
cd /data/ymj/code/llm/prefix-ttt
/data/ymj/.pixi/bin/uv sync --locked
/data/ymj/.pixi/bin/uv run --locked --no-sync python -m prefix_ttt.audit_environment --output artifacts/cpu/environment-after-reboot.json
```

不要更改已锁定依赖来规避尚未诊断的问题。普通包沿用当前锁定阿里源配置，PyTorch/torchvision为cu128；FLA锁定git commit，eval默认依赖组锁定lmms-eval==0.7.2与datasets==4.0.0。CPU时期FLA的`0 active drivers`是历史状态，已有GPU验证证据，不要据此重复验收或改装CPU版PyTorch。不要默认重跑完整CPU套件。

正式评测的HF_DATASETS_CACHE与LMMS_EVAL_DATASETS_CACHE必须同时指向data/llava-v1.5-assets-v1/benchmarks/hf-home/datasets，并设置HF_DATASETS_OFFLINE=1、HF_HUB_OFFLINE=1；完整绝对路径见configs/jobs.json。使用已生成的官方GQA派生缓存，不重建原数据或自定义评分器。

严格桥接通过 `load_checkpoint(path, dtype=torch.bfloat16)` 设置参数精度，此后仅 `.to(device)`。不要再次整体 `.bfloat16()`/`.to(dtype=...)`，否则原FP32 RoPE buffers会被量化。新增feature/gate/LoRA参数保持FP32 master，BF16计算，S FP32累积。

## GPU与服务前置检查

确认真实GPU型号/容量、驱动、用户指定的物理ID与空闲状态，再配置 `CUDA_VISIBLE_DEVICES`。当前授权的是四张A40物理0/1/2/3，不是预算中的两张A100。显式设置CUDA_HOME、PATH、LD_LIBRARY_PATH、CPATH和共享HF缓存，不能依赖source bashrc。

Supervisor托管已完成脱离终端与停止/恢复验证，按supervisor.md恢复，不再要求systemd用户manager作为前置。SQLite目录须是可靠本地文件系统，不要直接把共享/data作为WAL数据库落点。容器重建不保证保留/var/tmp，恢复前必须核对数据库和已有结果。

## 历史CPU移交清单与当前接续

CPU移交时列出的FLA GPU测试、真实checkpoint回归、四卡容量与保存恢复、A/B入口实现已在后续推进，详细证据见ledger.md及gpu_capacity.md；该历史清单不再是要求重复执行的待办。正式A/B训练尚未因此被视为完成。

当前lmms必要入口验证已通过，新4项队列已启动：官方MME、POPE、GQA各用accelerate四卡执行，随后四卡A。采用官方默认生成配置（MME/GQA最多16新token，POPE128）、生成与评分；检查全量原生结果，不使用旧自定义分数。之后A后整模型诊断、E1/E2同预算Pilot，两者各自恢复同一完整B scheduler继续剩余一遍，最后评测与原定小预算对照。

质量下降保留为研究结果；实现错误/NaN/真实OOM等记录后停止相关任务，不自动扩模型、缩图像或序列、截断梯度或扩大消融矩阵。
