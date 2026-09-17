# Supervisor 托管 SQLite 调度器

当前状态（2026-09-10评测框架切换）：旧自定义评测队列已停止，旧代码、结果与SQLite已按用户要求删除。新configs/jobs.json包含lmms-eval==0.7.2官方MME、POPE、GQA各四卡accelerate任务，随后四卡阶段A，共4项；四卡MME限量8条短测通过后消费者已启动，SQLite确认MME running、POPE/GQA/A queued。实时进度以training_progress.md和ledger.md最新记录为准，不重复启动任务。

## 本轮完成了什么

2026-09-10用户批准在当前容器使用已由pixi安装的Supervisor4.3.0，替代不可用的user systemd。现有SQLite调度器继续负责GPU白名单、原子多卡分配、显式argv/环境、任务状态和恢复检查；没有迁移模型、训练管线或任务数据库格式，也没有重跑成功的GPU实验。

主Supervisor已启动并脱离启动终端，正式程序`prefix-ttt-queue`已启动。当前使用`configs/jobs.json`，不创建空manifest：调度器的manifest不可变。环境模板在`configs/supervisor/jobs.example.json`，其中jobs为空只是示例，不能直接当正式清单执行。当前旧库清除是用户要求的评测切换操作，不是日常恢复策略；新队列建立后仍禁止删库重跑。新源码快照artifacts/run_sources/lmms072-e0-a/与初始数据库备份artifacts/queue/lmms072-e0-a-initial.sqlite3供追溯，初始备份不代表实时任务状态。

配置文件为仓库内`configs/supervisor/supervisord.conf`。程序固定使用绝对uv路径和四卡0/1/2/3、每卡并发1；训练子进程的环境来自manifest，不继承Supervisor的环境，必须同时填写manifest.env。HF_HOME共用`/data/ymj/.cache/huggingface`；评测另外显式设置HF_DATASETS_CACHE与LMMS_EVAL_DATASETS_CACHE到打包benchmark的datasets目录，并启用离线模式。评测依赖已通过uv锁定0.7.2，不重新下载现有权重和benchmark。

## 操作入口

以下命令不依赖source bashrc。status显示STOPPED时supervisorctl退出码3是程序状态，不表示Supervisor管理器不可连接。

```bash
# 查看当前状态与管理器PID
/data/ymj/.pixi/bin/supervisorctl -c /data/ymj/code/llm/prefix-ttt/configs/supervisor/supervisord.conf status
/data/ymj/.pixi/bin/supervisorctl -c /data/ymj/code/llm/prefix-ttt/configs/supervisor/supervisord.conf pid

# 仅在正式manifest、已有队列/成功结果、GPU空闲和数据入口检查完成后启动
/data/ymj/.pixi/bin/supervisorctl -c /data/ymj/code/llm/prefix-ttt/configs/supervisor/supervisord.conf start prefix-ttt-queue

# 正常停止：给消费者清理所有已启动任务的时间
/data/ymj/.pixi/bin/supervisorctl -c /data/ymj/code/llm/prefix-ttt/configs/supervisor/supervisord.conf stop prefix-ttt-queue
```

只有确认没有已有管理器、运行目录及数据库完好时，才重新启动管理器，禁止对仍运行的服务重复执行：

```bash
/data/ymj/.pixi/bin/supervisord -c /data/ymj/code/llm/prefix-ttt/configs/supervisor/supervisord.conf
```

启用`autostart=false`、`autorestart=false`、`startretries=0`。队列执行完正常退出，不维持空轮询；失败训练不会重试，消费者异常也不盲目拉起。恢复时检查旧PID身份、任务进程组和SQLite状态，再显式启动；发现旧任务还活着或处于不确定claimed窗口，现有调度器会拒绝派发。不要把running/claimed手工改成queued来绕过保护，也不要删除数据库重建。

## 停止行为与修复

任务由调度器以独立session/进程组启动，Supervisor的stopasgroup/killasgroup只能覆盖消费者的组，不能替代systemd的cgroup整树清理。正常TERM必须由消费者的finally清理任务；120秒stopwaitsecs为该路径留出时间。消费者SIGKILL或硬崩溃可能留下任务，恢复保护会阻止重复训练，但不会自动接管或杀死旧训练。任务不得自行daemonize或另建脱离原任务组的session。

真实集成首轮发现重复TERM缺陷：Supervisor组信号与uv转发叠加，第二个KeyboardInterrupt中断了清理。修复仅涉及首次TERM/INT后和finally期间忽略重复终止信号，结束时恢复原handler；不改变任务重试或资源语义。初始失败日志保留，最终调度器10项＋Supervisor3项全部通过，8.49秒。

验证覆盖真实uv消费者、脱离启动session、四卡标签分配（CPU fixture，不初始化CUDA）、环境传递、成功/失败/超时、重复启动跳过已处理任务、停止清理任务及孙进程、强杀消费者后旧任务存活时拒绝重派。测试daemon及子进程已清理，生产管理器没有被测试停止。

## 数据库与重启边界

`/var/tmp/prefix-ttt-1000`属于ymj且权限0700，控制socket0600，不开放TCP管理端口。该位置是本地overlay文件系统，SQLite和单消费者锁已在同类本地路径实测；当前SQLite使用默认rollback journal，不需要开启共享文件系统WAL。

本地overlay支持当前容器生命周期内的队列，但**不保证容器被平台重建后保留**。Supervisor能跨SSH/终端断开运行，不能跨容器销毁保持进程，也没有自动安装平台启动钩子。正式长训练前仍需确认平台对本地卷的保留策略；计划内关机前应在停止消费者并核对任务后归档SQLite一致性备份、配置、日志及checkpoint。恢复时若数据库缺失，停止并从已验证备份恢复，不能创建空库重跑。若需自动随容器启动，须由平台接入启动入口，本轮未修改容器PID1、crontab或平台配置。

## 证据

本轮集成日志与测试配置/状态数据库/消费者日志归档于仓库根目录下的 `artifacts/experiments/prefix_ttt/evidence/supervisor/`；原始日志位于`artifacts/supervisor/`。三个最终测试目录也原样保留在`/var/tmp/prefix-ttt-supervisor-test-*`，路径见integration-final.log。这些是隔离的CPU测试队列，不是正式训练数据库，不得拿来续跑模型训练。

systemd样例没有删除，仅不作为当前容器的必需前置。正式训练runner与E0/生成等验收仍按原计划推进；Supervisor部署成功不代表正式训练已开始。
