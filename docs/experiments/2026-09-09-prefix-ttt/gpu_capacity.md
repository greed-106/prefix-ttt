# 四张 A40 的 SFT 容量验收

后续托管更新：用户已批准并部署Supervisor，见[supervisor.md](supervisor.md)。下文关于systemd的阻塞描述保留为GPU容量测试当时的记录；当前不再等待systemd，仍需确认容器重建的数据库持久性和完成正式训练前置验收。

## 本轮目标与边界

2026-09-10用户授权使用物理GPU 0/1/2/3，实际各为48GB A40，无NVLink，通过PCIe连接。本轮先回答这些卡能否执行既定7B Prefix-TTT＋LoRA训练，不跳过E0、阶段A及其后整模型切换诊断，不把容量测试权重当正式B权重。

原任务书的两张80GB A100只是预算参考，不是算子或模型要求。本轮不改模型、图像、2048上限、LoRA rank或全局batch128；四卡时每卡micro-batch1、accumulation32。实际冻结基座BF16，新增与LoRA参数FP32，TTT状态FP32。

## 已执行结果

- 四卡NCCL求和正确，64MiB all-reduce预热后10次均值约4.77ms。此项只验证通信，不代表训练吞吐。
- CPU回归145项通过。FLA GPU共17项通过，覆盖tile16/32/64/128、规定长度、非零初态、因果性、历史影响和跨段状态梯度；没有回退reference或放宽阈值。
- 单卡完整E2执行2次真实AdamW更新，序列展开2023/2020，可训练参数107085824。第二步feature梯度非零；权重和optimizer保存重载通过。第二步decoder训练1.886s、allocated峰值15.95GiB、reserved16.61GiB；首步含编译73.27s不能当稳态性能。单卡诊断batch1，不是正式batch128训练。
- 独立分支剖析使用[1,2048,32,128] BF16输入：Local实际SDPA输入[64,32,32,128]，捕获PyTorch Flash Attention kernel；FLA捕获chunk_fwd_kernel_h/o。预热后10次前向均值Local1.151ms、FLA1.568ms，包含适配层，不包含特征映射与完整模型反向，不能外推最终加速比。

四卡DDP有效batch128验收通过：每卡micro1/accum32，2次AdamW更新共256条固定train记录，展开长度2000–2048。第二步四rank梯度范数均为12.13770294，feature梯度非零且一致；rank0完成权重及optimizer保存重载，四rank状态均passed，进程退出码0。第二步decoder训练约58.17s/128条（不含前置视觉与数据准备，不外推全训练ETA）；最大allocated16.77GiB、reserved27.88GiB，有显存余量。全部debug权重标记diagnostic_only，不用于后续正式实验。短时任务已退出，四卡空闲。

结论：这四张A40已实证支持本配置的SFT更新，无需因显存或PCIe通信更换GPU。这不是完整665k训练稳定性、最终质量、生成或端到端性能验收。正式A/B长训练尚未启动。

## 环境修复与持久运行阻塞

驱动580.65.06支持锁定的torch2.7.1+cu128。首次Triton失败原因是缺少Python.h，已将匹配的Ubuntu3.10.12开发包提取到项目artifacts/gpu/python-dev，通过显式CPATH使用；没有修改系统Python或依赖锁。源、字节数及SHA256见ledger.md，显式环境见configs/gpu-environment.json。

系统虽然安装systemd软件包，但当前PID1是bash、没有可连接的user manager。正式长训练按规范须使用SQLite＋用户systemd服务，目前尚未满足。更换GPU不能修复这一容器服务问题；不能擅自改为nohup/tmux持久托管。最小GPU诊断已独立执行，没有等待服务部署。

## 证据与复现

原始产物在artifacts/gpu/：environment-four-a40.json、nccl.log、cpu-regression.log、fla-tests-fixed.log、acceptance.log/xml、branch-profile.log、e2-single/及e2-four-gb128/。完整命令如下；使用configs/gpu-environment.json中的显式环境，单卡任务另设置对应CUDA_VISIBLE_DEVICES。不要通过source bashrc初始化任务。

```bash
uv run --locked python -m torch.distributed.run --standalone --nproc_per_node=4 -m prefix_ttt.gpu_communication
uv run --locked pytest tests/test_ops.py -m gpu -x -v
uv run --locked pytest tests/test_gpu_acceptance.py -x -v --junitxml=artifacts/gpu/acceptance.xml
uv run --locked python -m prefix_ttt.gpu_branch_profile
uv run --locked python -m prefix_ttt.gpu_smoke --layout E2 --steps 2 --output artifacts/gpu/e2-single
uv run --locked python -m torch.distributed.run --standalone --nproc_per_node=4 -m prefix_ttt.gpu_smoke --layout E2 --steps 2 --accumulation 32 --output artifacts/gpu/e2-four-gb128
```

计时/峰值重置在冻结视觉展开之后，因此表述为decoder训练阶段，不能当端到端或完整训练稳定峰值。仍需真实模型生成与E0回归、完整runner/恢复轨迹验收；当前入口只是短时SFT容量诊断。
