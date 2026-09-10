# SQLite 调度器：CPU 实施与部署边界

2026-09-10更新：用户批准改由Supervisor托管，部署与实测见[supervisor.md](supervisor.md)。本文件下述systemd内容保留为CPU阶段历史/替代平台参考，不再构成当前必须启用systemd的前置条件。SQLite任务语义不变；重复终止信号清理缺陷已在Supervisor真实集成测试中发现并修复。

本阶段新增独立 `prefix_ttt.scheduler`，参考 FastWAM 的标准库 SQLite 队列结构、argv 校验及进程组管理，未在运行时导入 FastWAM。参考来源为 `experiments/robotwin/schedule_robotwin_seed_search.py`，commit `c13e1534ece96093c95e0bff48d4ee6506ec40d9`；MIT 完整声明归档于 [FASTWAM_LICENSE](../../../configs/systemd/FASTWAM_LICENSE)。本项目重新实现了事务内多卡分配和恢复保护。

## 当前代码支持

- 固定 JSON manifest、严格 FIFO、单 consumer 的 `flock`；SQLite `jobs` 和 `events` 保留命令、环境配置摘要、状态、GPU 分配、进程身份、时间、退出码和日志。整个 manifest 与资源配置计算 SHA256，恢复时不允许修改或追加任务。
- `gpu_ids` 为允许的物理 GPU 索引，每卡默认并发 1，可配置。事务内一次 claim 全部所需 GPU，不拆分双卡任务、不超额占用。本地数据库应位于已确认支持 SQLite 锁语义的本地文件系统；部署时需核实挂载类型。
- 子进程必须以绝对 `uv` 路径开头的 argv 描述，经 `Popen(shell=False, start_new_session=True)` 启动。子进程环境仅来自 manifest，另注入分配后的 `CUDA_VISIBLE_DEVICES` 和队列根目录，不继承父进程的未声明变量，不读取 `.bashrc`。
- 成功任务跳过，失败、超时、中断任务均不自动重试。正常 SIGTERM/KeyboardInterrupt 清理整个任务进程组，保留尚未启动的 queued 任务供下一次恢复。
- 恢复使用 boot ID、PID start ticks 和进程组检查。发现旧进程或子进程仍在运行即拒绝派发；确认进程已消失后将旧 running 标为 interrupted。claim 与记录 PID 之间的崩溃窗口保持 claimed 并拒绝自动恢复，需要人工核实日志及进程后处理，不冒险重复训练。

## 固定 manifest 接口

```json
{
  "cwd": "/data/ymj/code/llm/prefix-ttt",
  "env": {
    "HOME": "/data/ymj",
    "PATH": "/usr/local/cuda-12.8/bin:/data/ymj/.pixi/bin:/usr/local/bin:/usr/bin:/bin",
    "CUDA_HOME": "/usr/local/cuda-12.8",
    "LD_LIBRARY_PATH": "/usr/local/cuda-12.8/lib64",
    "HF_ENDPOINT": "https://hf-mirror.com"
  },
  "jobs": [{
    "id": "example-only",
    "argv": ["/data/ymj/.pixi/bin/uv", "run", "--frozen", "--no-sync", "python", "-c", "print('fixture')"],
    "gpu_count": 2,
    "timeout_seconds": 60
  }]
}
```

这是 schema 示例，不是已排队的训练任务。正式配置还应显式提供共用缓存目录和运行命令需要的其他环境。GPU 编号必须由用户确认并检查当前空闲情况；调度器只约束其自身队列，不能排斥其他用户的外部进程。

## 本阶段验证与未执行项

CPU tests 通过 `uv run --no-project --offline python -m unittest discover -s tests -p test_scheduler.py -v` 执行（临时 `PYTHONPATH=src`），10 项通过。fixture 调用真实 CPU 子进程，GPU 编号仅作为分配标签，不初始化 CUDA。覆盖 GPU 白名单、单卡并发、原子双卡分配、单 consumer、显式环境不泄漏、成功/失败/超时、spawn 失败、immutable manifest、恢复身份、整组清理，以及实际 SIGTERM 后的新 consumer 恢复。

systemd 样例位于 `configs/systemd/prefix-ttt-queue.service.example`，未安装、未启用、未启动；实际用户 bus、重启后 user manager、脱离终端存活、GPU 进程和峰值显存仍未验证。挂载 GPU 后先完成最小 GPU 闭环，正式长程任务前再完成这些部署验收。文档及调度器不阻塞直接运行短时 GPU smoke。

## 环境锁定后的复验与依赖边界

在项目 uv 环境中再次执行 `uv run --locked pytest tests/test_scheduler.py`，10 项通过，完整日志保存于 `artifacts/cpu/scheduler-tests.log`。调度器自身只依赖 Python 标准库和 Linux 的 `/proc`、`flock`、进程组；无需 PyTorch、FLA 或 GPU 即可启动 CPU fixture。正式 job 则通过相同 uv 锁文件获得其训练依赖。单元测试的真实进程场景不等于 systemd 生命周期已经通过。

恢复采用保守协议而非自动接管遗留训练：live job 保持占卡并阻止新的 consumer 工作；dead job 标为 interrupted，保留 checkpoint 给后续显式安排的恢复任务。不可自动修改已有 manifest 来重试。launch 后进程很快退出可能使 start identity 来不及读取；此时依靠 PID 对应进程组存活检查，不把两个空 identity 错判为活进程。

正式 job 必须以前台进程运行，不自行 daemonize 或创建脱离任务进程组的 session。服务配置的 `KillMode=control-group` 负责完整服务 cgroup 的停机清理，但当前 CPU 测试只覆盖正常子进程组。队列资源限制是同一数据库内的协调，不提供跨数据库或外部进程的 GPU 全局互斥；同一服务器正式实验共用一个队列数据库。重启后须重新检查物理 GPU 编号、空闲状态和本地数据库挂载，不能沿用测试中的虚拟资源标签直接提交训练。

当前 `/data` 和 home 位于共享 Lustre，不能将 home 下的默认状态目录视为本地 SQLite 存储。服务样例改为显式 `/var/tmp/prefix-ttt-1000/queue.sqlite3`；本阶段未创建该目录。部署前必须用实际挂载检查确认该路径是支持 SQLite/flock 的本地文件系统，确认目录属于运行用户且其他用户不可写，同时核对 UID 与样例中的 1000 一致。还需检查 tmpfiles/租赁主机清理策略，确认 `/var/tmp` 跨计划内重启保留；若不能保证，选择另一个已确认持久的本地卷。不能用共享路径替代，或在数据库丢失后静默创建空队列重新训练。
