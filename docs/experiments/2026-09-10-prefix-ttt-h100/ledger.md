# Prefix-TTT H100 主机主账本

目录迁移说明（2026-09-17）：原 `scripts/` 的 7 个工具已移至仓库根目录下的 `scripts/experiments/prefix_ttt_h100/`，原 `evidence/` 移至 `artifacts/experiments/prefix_ttt_h100/evidence/`；图片仍在本目录 `images/`。已修复工具路径并通过 Shell/Python 语法、CLI 检查，没有重新执行实验。下文旧路径保留为历史记录，迁移详见 [kernel 主账本](../2026-09-11-prefix-ttt-kernel/ledger.md)。

本目录 `docs/experiments/2026-09-10-prefix-ttt-h100/` 是本项目在**本机 `cucloud-server3`**（用户 `mjyang`，8×H100 80GB，裸金属主机）的稳定实验目录，启动日期 2026-09-10。本机的后续记录持续写入本账本，不按自然日另建目录。

前一阶段记录位于 `docs/experiments/2026-09-09-prefix-ttt/`，对应**另一台机器上的早期环境**（用户 `ymj`，四张 A40 48GB，路径 `/data/ymj/code/llm/prefix-ttt`）。那份账本保持原样作为历史，不追加本机内容；2026-09-10 迁移时已将本机环境审计从旧目录移至本目录 `environment.lock.json`，旧目录未被修改。两台机器是各自独立的检出，`data/`、`artifacts/`、`preference/` 不共享。

## 当前状态

最后更新 2026-09-11 14:40。

- 环境：按 `pyproject.toml` / `uv.lock` 安装并锁定，GPU 算子验收 17 项通过；两台 H100 主机（`cucloud-server3` / `h100-1`）通过宿主机 NFS 共享 `/data/shared/weights/prefix-ttt`，多机 NCCL/RDMA 已实测。
- 训练：已放弃 A40 迁移方案，改为按任务书在本机重训。A 阶段 391 步与 B/E2 阶段 5182 步（16 卡跨两机，micro_batch 8，全局 batch 128）均已完成，`complete=true`；权重在 `/data/shared/weights/prefix-ttt/training/{A,E2}`。
- 推理口径：**LoRA 在加载后于内存中一次性合并进基座权重，这是唯一的推理路径**（运行时开关与对照代码已删除）；训练侧 LoRA 仍与基座分开，checkpoint 保留原始适配器，不提供导出合并权重的接口。
- 性能优化：① LoRA 合并（TPOT ×1.33）与 ③ Local-32 decode 专用路径（×1.24）已落地，合计 TPOT ×1.74 / kernel 数 −43%；② torch.compile 融合与 ④ FLA fused recurrent 经实测为负收益，已回退（详见 2026-09-11 两条记录）。
- 评测：E0（基座）与 E2 的 MME、POPE 官方分数已产出；E2 采用合并口径后为 MME 1429.4893 / 278.2143、POPE 0.8501 / 0.8357；E2 GQA 经用户决定不跑，E1 不跑。带内打点的跑批共 22748 条逐请求记录，另有优化后的对照跑批。
- 工程质量：已完成一轮审计与重构（分层、常量单点化、配方校验），CPU 167 passed / GPU 17 passed；重构经逐样本比对确认零行为变化。
- 产出：`experiment_summary.md`（阶段总结）、`metrics-summary.json`、`images/`（三张图，已按最终口径重绘）、`scripts/`（评测、绘图、对照脚本）。
- 待办：本地改动尚未提交（用户未授权）；`gpu_smoke.py`/`scheduler.py`/两个诊断模块的去留、`transfer.py` 手抄前向的收敛、`configs/base.json` 33 个装饰键的最终处理方式，均待用户决定。
- 仍缺：`preference/` 参考仓库（不影响训练与评测入口）。

## 2026-09-10 环境重建与验证（本账本首条）

- 环境严格按 `pyproject.toml` / `uv.lock` 安装，未改任何库版本：`uv sync --locked` 解析 161 包、安装 159 包；执行前后 `uv.lock` SHA256 前缀均为 `7426306b361e2b3c`，`git status` 干净，`pyproject.toml` 未修改。
- 关键版本：torch 2.7.1+cu128、torchvision 0.22.1+cu128、triton 3.3.1、transformers 4.51.3、peft 0.15.2、accelerate 1.6.0、flash-linear-attention git `c51953382397da5c3b7b8a41e568915b703e2934`、lmms-eval 0.7.2 + datasets 4.0.0（eval 组）、pytest 8 系（dev 组）。venv 解释器为系统 `/usr/bin/python3.10`（3.10.12），满足 `requires-python >=3.10,<3.11`。
- CPU 套件：`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --locked pytest -m 'not gpu'` → 156 passed、12 skipped、0 failed（10.76s）。12 项 skip 的原因已逐条核实为缺本地资产而非缺陷：9 项缺本地 tokenizer/权重（`test_data_audit.py` 3 项、`test_model_bridge.py` 6 项），3 项因未安装 Supervisor（`test_supervisor.py`）。
- 环境审计：`uv run --locked python -m prefix_ttt.audit_environment --output docs/experiments/2026-09-10-prefix-ttt-h100/environment.lock.json` → 包导入错误 0，CUDA available=true、device_count=8，8 张 NVIDIA H100 80GB HBM3（85028372480 B，compute capability 9.0）。审计入口默认输出指向旧实验目录的 `environment.lock.json`，在本容器必须显式 `--output` 到本目录，避免覆盖旧容器证据。
- 本机资源：8×H100 两两 NV18 互联（`nvidia-smi topo -m`），192 逻辑核，2TB 内存，驱动 570.211.01，CUDA Toolkit 12.8.93（`/usr/local/cuda-12.8`），`/data` 余量约 12T。系统已装 `libpython3.10-dev`（3.10.12-1~22.04.17），`/usr/include/python3.10/Python.h` 存在，旧容器为编译 Triton 而项目内提取开发头文件的阻塞在本机不复现。
- 沙箱干扰已排除，不记为环境缺陷：在文件沙箱限制下，`tests/test_scheduler.py` 有 2 项失败、`nvidia-smi` 报 NVML Unknown Error、`torch.cuda` 不可用；原因是子进程 `uv` 写入 `~/.cache/uv` 被拒、以及 `/dev/nvidia*` 的 O_RDWR 打开被拒。放开权限后复验 `tests/test_scheduler.py` 10/10 通过、`nvidia-smi` 列出 8 张 H100、`torch.cuda.is_available()=True`。

## 2026-09-10 GPU 算子验收（FLA/Triton）与 flash 依赖澄清

- 命令：`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 uv run --locked pytest -m gpu -v --durations=0`，只占用物理 0 号卡；运行前后各卡显存 0 MiB，未与他人任务重叠。
- 结果：17 passed、168 deselected、0 failed。冷 Triton 缓存首跑 366.08s，热缓存复跑 82.22s，取证运行（`-v --durations=0`）82.50s；逐项结果与耗时见 `evidence/gpu-acceptance.log`。
- 覆盖范围：`tests/test_ops.py::test_fla_gpu_varlen_output_state_and_gradients` 的 tile 16/32/64/128 四项（输出、状态、全部输入梯度），以及 `tests/test_gpu_acceptance.py` 的长度 1/2/31/32/33/63/64/65/127/128/129、未来因果与过去影响、分段尾状态与跨状态梯度共 13 项。未放宽任何误差阈值，也未回退 reference 实现。
- 结论：本机 Triton 3.3.1 能真实编译并执行 FLA kernel，`flash-linear-attention` 0.6.0（git `c5195338…`）在 8×H100 上可用；JIT 缓存在 `~/.triton`（约 250M）。这是算子层验收，**不代表**整模型前向、SFT 或评测已可运行。
- flash 依赖澄清（两个易混概念）：
  - `flash-linear-attention`（`import fla`）是本项目正式依赖，锁在 git rev `c51953382397da5c3b7b8a41e568915b703e2934`，依赖 triton 3.3.1 做 kernel JIT。
  - `flash_attn`（Dao-AILab flash-attention）**不在依赖中、也未安装**：`import flash_attn` 报 `ModuleNotFoundError`。它只被 vendored LLaVA 的上游入口引用（`third_party/llava/llava/train/train_mem.py`、`train/llama_flash_attn_monkey_patch.py`、`serve/model_worker.py --use-flash-attn`）；`src/prefix_ttt` 与 `tests/` 中 `import flash_attn` / `flash_attention_2` 出现 0 次，Local-32 使用 `F.scaled_dot_product_attention`（`ops/local.py`），模型加载取默认后端；审计中该字段即记为 `not_installed; PyTorch SDPA selected`。
  - 因此在本项目入口下出现 “No module named 'flash_attn'” 属于误走上游旧入口，不是依赖缺失；不要把 flash-attn 的预编译轮子当成本环境前置条件。
- 沙箱现象复述（不记为环境缺陷）：受限环境下 `import fla` 曾报 `RuntimeError: 0 active drivers ([]). There should only be one.`，同源现象为 `nvidia-smi` NVML Unknown Error 和 `fla` 的 “Triton is not supported on current platform, roll back to CPU”；放开权限后全部消失，最终审计 `package_import_errors` 为空。
- 遗留项（未改配置）：`configs/gpu-environment.json` 的 `CPATH` 指向本机不存在的 `/data/ymj/.../artifacts/gpu/python-dev/...`。本机系统自带 `libpython3.10-dev`（3.10.12-1~22.04.17），Triton 编译不再需要项目内提取头文件，该行在跑正式任务前应清理。

## 2026-09-10 迁移资产到位与校验（20:24–20:36 到位）

- `data/llava-v1.5-assets-v1/` 共 69G：`datasets/llava-665k` 46G、`benchmarks` 10G（含可移植 `hf-home`）、`models/llava-1.5-7b-hf-b234b804…` 14G。校验以包内 `asset_manifest.json` 声明的身份为准，结果如下：
  - 标注 `llava_v1_5_mix665k.json` SHA256 = `ce959ce6e23073ee1cd1a8a2ef1c633768c10d4174327b8b2dc7113b91af6cf8`，与声明一致。
  - `integrity/acquisition_manifest.json` = `204fede1488e26e48e521902e38de1f34b5cedae0c967e2b4a394dcf831abc00`、`integrity/llava665k_image_allowlist.json` = `34afcea4b67774fff3810859925c1a685c113df2658b1b91ca0135a5eef6ddca`，均一致。
  - 图片实际文件数 349034，与声明一致；模型目录实际文件数 17、文件字节和 14131147625，与声明一致。
  - 三个权重分片 SHA256 与旧容器证据 `docs/experiments/2026-09-09-prefix-ttt/evidence/checkpoint-sha256.txt` 逐条相同：`c11dbf01…`、`46df6c6e…`、`4f06177c…`，即与旧机为同一字节对象。
  - 本次**未做**的事：未重新逐张解码 349034 张图片，未重跑 665298 条标签审计；上述校验只证明包内声明的哈希、文件数与字节数成立。
- `artifacts/migrations/from-a40-20260910-step1075/`：`sha256sum -c SHA256SUMS` 7/7 全部 OK，`E2-resume.pt` 校验通过；`fixed_manifest.json` SHA256 = `246586be1ddd3d0cea09e17047e6d658c5e9a38637654e46906605578707a93b`，与旧账本及 `data_manifest_summary.json` 记录一致。
- 交接指南 `migration_from_a40.md` 已到位，明确四项必须适配（仅 B 恢复时跳过 A 读取、world_size 4→8 显式迁移与 RNG 策略、manifest 旧绝对路径重定位且保留原字节与身份键、历史 Pilot 报告与本次恢复验收分离），并要求接管前与用户/旧机协调停止时机。旧机训练仍在继续，接管边界（1075 步或更晚固定 checkpoint）未定。

## 2026-09-10 flash-linear-attention 取源问题诊断与兜底

- 现象归类：报错发生在 `uv sync --locked` 阶段而非运行阶段。`flash-linear-attention` 是这份 lock 里**唯一**的 git 源依赖（`source = { git = "https://github.com/fla-org/flash-linear-attention.git?rev=c51953382397da5c3b7b8a41e568915b703e2934" }`），其余包分别来自阿里 PyPI 镜像与官方 cu128 索引；因此当 github.com 取源超时/被拒时，同步会恰好停在这个包上。
- 排除构建缺陷：锁定 rev 的 `pyproject.toml` 中 `[build-system] requires = ["setuptools>=64", "wheel"]`，`dependencies = ["transformers>=4.45.0", "einops"]`，构建期不导入 torch，也不依赖 `flash_attn`；`setup.py` 仅做 setuptools 打包。空缓存实测（`UV_CACHE_DIR` 指向空目录）从 github.com 取源并构建成功，耗时 7.2s。
- 现状核对：本机 `.venv` 中 FLA 的 `direct_url.json` 记录 `commit_id = c51953382397da5c3b7b8a41e568915b703e2934`，与 lock 完全一致；`uv sync --locked` 稳定输出 Resolved 161 / Checked 159 的无改动结果；`UV_OFFLINE=1 uv sync --locked` 在缓存可用时 0.04s 通过（exit 0），可作为断网复核手段。
- 兜底产物：由锁定 rev 的同一份源码构建 `artifacts/wheels/flash_linear_attention-0.6.0-py3-none-any.whl`（1406450 B，SHA256 `d481d7cd01a820cfb0fe25c16e2bbb2a0669be83025501f30dd46d22eb0c7e84`），并与 `.venv` 中已安装的 FLA 逐文件比对：515 个 `.py` 文件哈希全等；该 wheel 可在干净 venv 中用 `--no-deps` 安装并正确报告 0.6.0。
- 恢复路径与残余风险：若 uv 缓存失效且 github.com 不可达，可临时 `uv pip install --no-deps artifacts/wheels/flash_linear_attention-0.6.0-py3-none-any.whl`（内容与锁定 rev 一致，但 `direct_url` 会记为本地文件，待 GitHub 可达时用 `uv sync --locked` 复原为锁定源）。该断网路径**未实测**，只验证了 wheel 内容一致与可安装。若要彻底消除该单点，需要改动 `pyproject.toml` 的 source 或配置 git 镜像映射，二者都需用户批准，本轮未做。

## 2026-09-10 A40 迁移资产可用性核查（只读，未改代码）

目的：按 `migration_from_a40.md` 第 1 步确认迁移资产是否到位、能否直接使用。全部为只读操作，未改代码、未跑 GPU、未提交任务。

- 到位情况：7 个载荷文件齐全，`sha256sum -c SHA256SUMS` 7/7 OK；数据包 69G 与模型权重分片 SHA256 均与旧机证据一致（见上节）。
- `E2-resume.pt` 内部一致性与指南表格逐项吻合，CPU 加载成功：`stage=B`、`layout=E2`、`global_step=1075`、`samples_seen=137600`（恰为 1075×128）、`total_steps=5182`、`world_size=4`、`complete=False`、`diagnostic_only=False`、`manifest_sha256=246586be…`、`config_sha256=4ff7ea01…`、`stage_a_sha256=d379eb00…`、`scheduler.last_epoch=1075`、`base_lrs=[2e-05, 1e-04]`、`_last_lr=[1.8395e-05, 9.1975e-05]`、`rng_by_rank` 长度 4（每份含 cuda/numpy/python/torch）。
- 张量状态：`trainable` 517 个张量全部 `torch.float32` 且全部有限；其中 `prefix_ttt` 69 个 = 23 层 × {`a_phi`(32,128,128)、`b_phi`(32,128,128)、`gate_weight`(32,4096)}，LoRA 448 个，与「A9-T23」一致；`optimizer` 含 517 条 state（`exp_avg`/`exp_avg_sq` 均为 FP32，含 `step`）与 2 个参数组。故 FP32 master 与 optimizer 状态完整，不需要重新 warmup。
- 数据侧可用：标注 `llava_v1_5_mix665k.json` 665298 条、SHA256 与 manifest 声明一致；图片根目录存在，按固定清单随机抽 300 条记录（275 条含图）逐一检查图片文件，缺失 0。
- **直接使用仍有 4 处代码阻塞，与指南一致，已在源码逐条确认**：
  - A：`src/prefix_ttt/sft.py:96` 对 `layout=E2` 强制要求 `--stage-a-checkpoint`，`:117` 在建模前 `digest_file()` 该文件，`:133` 强制加载并校验 A 完整性。A 权重不在迁移范围内，故必须新增「仅 B 恢复」路径。
  - B：`sft.py:157` 用 `identity` 逐字段与 checkpoint 比较，`world_size` 由 4 变 8 会直接抛 `Resume trajectory identity differs`；`sft.py:165` 的 `checkpoint['rng_by_rank'][rank]` 在 rank≥4 时越界。必须显式迁移模式与 RNG 映射策略。
  - C：`fixed_manifest.json` 的 `annotation` 是旧机绝对路径 `/data/ymj/code/llm/prefix-ttt/data/...`，`data_pipeline.py:26-28` 会 `resolve()` 后按该键查 `inputs` 并 `digest_file()`，在本机必然失败。**无法用符号链接绕过**：`/data` 属 root 且 sudo 被 `no new privileges` 禁用，本机不能创建 `/data/ymj`；因此路径重定位必须在代码内完成（保留 manifest 原字节与身份键，读取新路径后按原 SHA 校验）。
  - D：`sft.py:122` 的 `check_pilot_diagnostics(paths, identity, args.resume)` 会把恢复文件 SHA 与第 391 步 Pilot 报告比对，不能原样用于 1075 步恢复点；须区分历史 Pilot 前置与本次恢复验收。
- 附带发现（不阻塞）：manifest `inputs` 的 4 条中，3 条指向旧机 `artifacts/cpu/labels-full/label-audit.jsonl`、`labels-full/data-audit.json`、`data/data-audit.json`，这三份没有随迁移包传入；但 `load_manifest` 只校验 annotation 一条，故不影响训练，仅作为来源身份记录保留。

## 2026-09-10 迁移阻塞点逐条复核（只读，未改代码）

针对用户提出的四点意见逐条回到源码核对，结论如下（仍未修改任何代码）。

- A 冗余确认：`transfer.py:213` 中 A 只保存 TTT 分支的 `features`（`branch.state_dict()`）；E2 恢复路径上 `sft.py:131` 先新建 TTT 模块、`:133-143` 读入 A 的 features、`:144` 装 LoRA，随后 `:155-159` 的 `load_trainable(model, checkpoint['trainable'])` 会把**全部 517 个可训练张量**（含 69 个 `prefix_ttt`）整体覆盖。故对**恢复**路径而言 A 文件读取确属冗余，可去掉；非恢复的 E2 初始化仍必须保留该校验。`identity['stage_a_sha256']` 仍须保留（迁移时取自 checkpoint 的 `d379eb00…`），不得置空。
- world_size 4→8 数学自适应确认：`sft.py:23` 的 `sample_group = order[cursor:cursor+128][rank::world_size]` 在 8 卡下每卡 16 条（4 卡为 32 条），128 条分组与全局样本顺序不变；`training.py:11-15` 的 `accumulation_steps(8,1)=16` 仅做整除校验；`sft.py:195-197` 的 `targets` 先按 rank 求和再 `all_reduce` SUM，`:206` 的 `token_normalized_ce` 按**全局** target 数归一，`:212-218` 梯度用手动 `all_reduce` SUM，`:226-227` 的 loss 同样全局求和，`:172-176` 保存时按 world 申请 8 份 RNG。因此 batch 语义（micro 1 × 8 卡 × 累积 16 = 128）无需改动即成立，`:225` 的游标与卡数无关，第 1076 步仍取 `train[137600:137728]`。真正阻塞的只有 `sft.py:157` 的 identity 比较与 `:165` 的 `rng_by_rank[rank]` 越界。
- manifest 批量替换评估（不推荐）：`data_pipeline.py:11` 硬编码 `MANIFEST_SHA256='246586be…'`，`:15-18` 校验文件字节，改写文件会立即抛 `Fixed manifest SHA256 mismatch`；`sft.py:114/157` 会把新 SHA 与 checkpoint 记录的 `246586be…` 比较而拒绝恢复；迁移包内 `source-run.json`、`source-switch-A.json` 同样记录旧 SHA。更关键的是 `--require-switch-diagnostic` 为 `required=True`（`sft.py:82`），而迁移来的 `source-switch-A.json` 恰好满足 `status=passed`、`stage_a_sha256=d379eb00…`、`manifest_sha256=246586be…`、`config_sha256=4ff7ea01…`，**无需改动即可通过校验**；一旦改写 manifest，这份唯一的 A 阶段来源凭证随之失效。推荐替代方案：保持 manifest 字节不变，在 `load_manifest` 内做读取期重定位（原字符串仍作 `inputs` 的键，把 `/data/ymj/code/llm/prefix-ttt` 前缀映射到当前项目根，再按原键 SHA 校验重定位后的文件，本机该文件哈希已确认为 `ce959ce6…`），并把重定位后的路径交给 `build_dataset`。
- Pilot 门槛确认不阻塞：`--require-pilot-diagnostics` 为可选参数（`sft.py:85`），新命令不传即可；迁移来的 `source-E2-pilot-diagnostic.json` 中 `checkpoint_sha256=9dc14688…` 对应第 391 步 pilot，本就不适用于 1075 步恢复点，不再作为门槛使用。
- 待实施的最小改动清单（等用户确认后动手）：① `data_pipeline.py` 增加读取期 annotation 重定位；② `sft.py` 增加显式迁移模式，使 E2 恢复可不传 `--stage-a-checkpoint`（`stage_a_sha256` 取自 checkpoint）、允许 world_size 4→8 并记录旧/新值、为 rank≥旧 world_size 定义确定性 RNG 策略；③ 启动改用 `--nproc_per_node=8`，保留 `--require-switch-diagnostic` 指向迁移报告，去掉 `--require-pilot-diagnostics`。同时须在账本明确：每卡样本集合与梯度累积顺序已改变，**不得**声称与 4 卡轨迹逐位复现。

## 2026-09-10 移除硬编码 manifest SHA256 与 Pilot 诊断门槛（代码改动，已测）

- 动机：用户指出代码内硬编码哈希不可接受、Pilot 诊断门槛无用。全项目检索确认 64 位十六进制字面量在 `src/` 中**仅有一处**，且只作为 `load_manifest` 的默认参数；Pilot 门槛涉及 1 个函数、1 个参数与 6 处测试引用。
- 改动（`git diff --stat` = 3 文件、1 insertion(+)、51 deletions(-)）：
  - `src/prefix_ttt/data_pipeline.py`：删除 `MANIFEST_SHA256` 常量、`expected_sha256` 形参及其比较分支；`load_manifest(path)` 仍计算并返回文件 SHA 供恢复身份使用。四个调用点（`sft.py`、`transfer.py`、`switch_diagnostic.py`、`pilot_diagnostic.py`）本来就只传一个参数，无需改动。
  - `src/prefix_ttt/sft.py`：删除 `check_pilot_diagnostics` 函数、`--require-pilot-diagnostics` 参数、对应 `parser.error` 分支与调用。
  - `tests/test_sft.py`：删除 `test_pilot_diagnostic_gate`。
- 验证：`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --locked pytest -m 'not gpu'` → 164 passed、3 skipped、17 deselected、0 failed（11.26s）。通过数由 156 升至 164 有两个原因：删除 1 个用例；`data/` 到位后，原先因缺本地 tokenizer/权重而 skip 的 9 个用例（`test_data_audit.py` 3 个、`test_model_bridge.py` 6 个）开始真实执行并通过。剩余 3 个 skip 仍为未安装 Supervisor。
- 仍然保留的校验（未删）：manifest 自身的 split 唯一性与 `order_sha256`、`dev`/`train`/`A` 分离性、annotation 相对 `inputs` 的摘要校验；恢复时 `identity` 比较仍会拒绝 manifest/配置/stage-A 不一致的 checkpoint。
- 因此失去的保护：新建（非恢复）运行时不再有外部钉死的 manifest 身份，无法自动发现清单文件被替换。若日后需要，应改为显式命令行参数传入，而不是写死在代码里。
- 未处理：`configs/jobs-e2-full.json`、`configs/jobs-e2-resume.json` 仍含 `--require-pilot-diagnostics` 与旧机 `/data/ymj` 路径；它们是旧容器的提交记录，本次未修改（若复用会因未知参数报错）。`--require-switch-diagnostic` 仍为 `required=True`，本次未删。
- 仍未实施：迁移适配三项——annotation 读取期重定位、E2 恢复可不传 `--stage-a-checkpoint`、world_size 4→8 与新增 rank 的 RNG 策略。

## 2026-09-10 移除 switch 诊断门槛并同步清理配置参数（代码改动，已测）

- 动机：用户认为微调过程不需要历史报告作为前置，文件身份不应由执行者手工用哈希确认。
- 代码改动：
  - `src/prefix_ttt/sft.py`：删除 `check_switch_diagnostic` 函数、`--require-switch-diagnostic` 参数（原为 `required=True`）及其调用。
  - `tests/test_sft.py`：删除 `test_switch_gate`。
- 配置同步：用 AST 提取各模块的 `add_argument` 清单，与 `configs/*.json` 中每个任务的 argv 逐项比对，据此删除已不存在的参数，共 4 个任务 5 处：
  - `configs/jobs-e2-full.json::e2-b-continue-5182`：`--require-switch-diagnostic` + 路径、`--require-pilot-diagnostics` + 2 个路径；
  - `configs/jobs-e2-resume.json::e2-b-resume-5182`：同上（Pilot 为 1 个路径）；
  - `configs/jobs-pilot.json::e2-pilot-50048` 与 `::e1-pilot-50048`：各 1 处 `--require-switch-diagnostic` + 路径。
  - 其余任务（`prefix_ttt.pilot_diagnostic`、`prefix_ttt.switch_diagnostic`、`prefix_ttt.transfer`，以及 accelerate 的 `prefix_ttt.lmms_run`）参数全部仍然有效，未改动；配置中的 `/data/ymj` 旧路径属旧容器历史记录，也未改动。
- 验证：全部 `configs/*.json` 可解析；复跑比对脚本 → 含未知参数的任务数 0；`pytest -m 'not gpu'` → 163 passed、3 skipped、17 deselected、0 failed（11.23s）。`src/`、`tests/` 中无 switch 门槛残留，`sft.py` 无因删除而失效的 import（`digest_file` 仍用于 `stage_a_sha256`）。
- 保留说明：`identity` 内的 manifest/config/stage-A 摘要仍会计算并与 checkpoint 比较，这是**恢复一致性校验**，全自动、不要求执行者提供任何文件或哈希；本轮删除的是「要求执行者手工提供报告与哈希」的前置门槛。至此 `sft` 入口不再有任何执行者提供的哈希类前置。
- 仍未实施：迁移适配三项（annotation 读取期重定位、E2 恢复可不传 `--stage-a-checkpoint`、world_size 4→8 与新增 rank 的 RNG 策略）。

## 完成项

- 独立 uv 工程环境安装并锁定（`uv sync --locked`），锁文件与 `pyproject.toml` 零改动。
- GPU 算子验收 17 项全部通过（tile 16/32/64/128 与长度/因果/分段梯度），证明 FLA + Triton 在本机真实可用；证据 `evidence/gpu-acceptance.log`。此项仅限算子层。
- 全量 CPU 套件 156 项通过，无失败；skip 项归因明确。
- 环境审计落盘本目录 `environment.lock.json`：导入错误 0，8×H100 可见，compute capability 9.0。
- 迁移包 7/7 校验通过，数据包按 `asset_manifest.json` 的哈希、图片数 349034、模型 17 文件/14131147625 B 校验通过；三个权重分片与旧容器证据同 SHA256。
- FLA 取源问题完成归因（lock 中唯一 git 源依赖 + github.com 可达性），并落盘内容一致的兜底 wheel；`UV_OFFLINE=1 uv sync --locked` 可离线复核。
- A40 迁移资产完成只读可用性核查：checkpoint 可加载且内部自洽（517 张量全 FP32 有限、optimizer 517 条、scheduler 1075、4 份 RNG）、标注与图片可解析、4 处代码阻塞已定位到具体行号。

## 运行项

无。本容器当前没有任何训练、评测或队列任务在运行，也没有已提交队列。

## 失败项与环境限制

- `preference/` 参考仓库仍缺失（不影响训练与评测入口，只影响 `audit_environment` 的参考仓库字段与论文/审计文档）。数据与迁移副本已到位；`artifacts/` 下目前只有迁移副本与 wheels，没有旧机的训练产物、队列库与 benchmark 输出。
- `configs/base.json` 的 `gpu_ids [0,1,2,3]` 与 `gpu_status` 仍是旧机器「四张 A40 48GB」口径；`configs/gpu-environment.json` 的 `CPATH` 仍指向 `/data/ymj/...`。本机为 8×H100 80GB，两处配置在真正跑任务前必须先定授权范围与路径。
- Supervisor 未安装（pixi 全局仅有 nvitop/unzip），SQLite 调度器暂无旧容器那种持久托管；`test_supervisor.py` 3 项因此 skip。

## 关键决策

- 使用 `pyproject.toml` / `uv.lock` 确定的环境，不自行升降任何库版本。
- 旧容器账本 `docs/experiments/2026-09-09-prefix-ttt/` 只作历史保留，本容器记录一律写本目录。
- 环境审计输出显式指定本目录，不覆盖旧容器的 `environment.lock.json`。
- 数据与权重获取、参考仓库恢复、Supervisor 安装、GPU 授权与配置更新，均等用户决策，不擅自执行。
- 不为了绕开 `flash-linear-attention` 的取源问题而改动 `pyproject.toml`/`uv.lock` 或替换来源版本；只落盘由锁定 rev 构建、逐文件校验一致的兜底 wheel。

## 产物位置

- 本目录：`ledger.md`、`environment.lock.json`（本机环境审计）、`evidence/gpu-acceptance.log`、`migration_from_a40.md`（旧机交接指南，他人写入，本智能体未修改）。
- 接收资产：`data/llava-v1.5-assets-v1/`（69G）、`artifacts/migrations/from-a40-20260910-step1075/`（含 `E2-resume.pt`、`fixed_manifest.json`、`SHA256SUMS` 与四份 source-*.json/jsonl）、兜底 wheel `artifacts/wheels/flash_linear_attention-0.6.0-py3-none-any.whl`。
- 工程与环境：`src/prefix_ttt/`、`third_party/llava/`、`tests/`、`pyproject.toml`、`uv.lock`、`.venv/`。
- 旧容器历史记录：`docs/experiments/2026-09-09-prefix-ttt/`。

## 下一步

数据、权重与迁移副本已到位并校验，不再需要下载；后续按 `migration_from_a40.md` 推进，但每一步都等用户确认：① 确定本机授权的物理 GPU，并更新 `configs/base.json` 的 `gpu_ids`/`gpu_status` 与 `configs/gpu-environment.json` 的 `CPATH` 等本机路径（语义训练配置保持原样，只覆盖部署侧）；② 与用户/旧机协调接管边界（直接接 1075 步，或停止旧机后补传更晚的固定 checkpoint）——旧机仍在训练，两边后续更新不得混用；③ 实施四项最小恢复适配并做短时 8 卡恢复验证（严格加载 TTT/LoRA、FP32 master、原 RoPE buffer，冻结基座 BF16 且加载后只做 device 移动）；④ 视需要安装 Supervisor 恢复 SQLite 调度器持久托管。在适配与验证完成前不提交正式训练，也不把算子验收通过表述为「整模型可用」。

另需用户决策：是否为了消除 `flash-linear-attention` 的 GitHub 单点，改动 `pyproject.toml` 的 source 或配置 git 镜像映射（当前仅落盘了内容一致的兜底 wheel，未改动任何配置）。

## 2026-09-10 从 A40 接收 E2 迁移副本（未启动）

- 用户要求仅迁移E2，A权重已包含于E2更新后的TTT参数，A/E1权重均不传；本机智能体负责适配及实际恢复，旧机智能体只传文件和指南。
- 已通过rsync接收 `artifacts/migrations/from-a40-20260910-step1075/`，约1.294GB；SHA256SUMS所列7个有效载荷文件全部远端验证OK。恢复checkpoint为E2-resume.pt，step1075/samples_seen137600、scheduler1075、来源world_size4、517个可训练张量及optimizer状态、四rank RNG。
- 指南见 [migration_from_a40.md](migration_from_a40.md)。包括仅B恢复时跳过A读取、4→8显式迁移/RNG策略、路径重定位保留manifest哈希、历史Pilot报告与新机恢复验收分离。当前代码不能不经适配直接运行。
- 本次没有改本机代码/依赖、没有提交本机训练或GPU测试，没有删除旧机任何文件。旧机训练仍继续：接管前协调停止或选择固定1075步为边界，不能混用两边后续更新。旧主账本中的资产缺失描述是此前环境搭建时状态；data目录现在存在，但本次没有全量校验data包。
- 不需重建train/dev manifest、不需要先跑baseline或重跑A/Pilot。固定manifest原文件随包传入；E1仅保留旧机历史，不在此次新机训练范围。


## 2026-09-10 补充：用户要求的 A 权重已补传

用户随后明确要求补传A checkpoint，因此上文“不传A/缺少A”的描述仅代表首次迁移，不再代表当前资产。E1仍未传输。

- 新增文件：`artifacts/migrations/from-a40-20260910-step1075/A-complete.pt`，339875595字节（约325MiB），来自旧机 `artifacts/training/A/latest.pt`。
- 这是正式A的391步/50000样本完整checkpoint；SHA256：`d379eb00342f97e15b8b660beda59104627aee6fde256d07c6ff6f84945038a9`，与E2-resume.pt中stage_a_sha256一致。附独立 `A-SHA256SUMS`，不覆盖原E2迁移校验清单。
- 如需沿用原sft.py的A初始化路径，可将stage-a-checkpoint指向此文件；随后仍必须严格加载E2-resume.pt中的更新后TTT/LoRA和optimizer/scheduler，不能用A权重覆盖已恢复E2。
- 此补传使“跳过A文件读取”不再是必须实施的迁移改动。4→8 world_size/RNG、绝对路径、配置身份与恢复诊断的适配仍然需要；不能据此原样启动旧命令。
- 不重跑A，不改变E2迁移恢复点1075；未启动远程训练，未删除或停止本地数据/任务。实际恢复仍由远端智能体按用户指示执行。

## 2026-09-10 A 权重独立校验、TTT 层数差异与 world_size 适配（代码改动，已测）

- A 权重独立复核（与上节互补，不重复其结论）：`sha256sum -c A-SHA256SUMS` 通过，实测 SHA256 = `d379eb00342f97e15b8b660beda59104627aee6fde256d07c6ff6f84945038a9`，与 E2 checkpoint 的 `stage_a_sha256` 一致；内部字段满足 `sft.py` 校验（`stage=A`、`complete=True`、`diagnostic_only=False`、`manifest_sha256=246586be…`、`config_sha256=4ff7ea01…`、`global_step=391`、`samples_seen=50000=len(manifest['A'])`）。故 `--stage-a-checkpoint` 直接可用，指南「A. 仅恢复 B 时跳过 A 初始化」一项确认无需实施。
- 层数差异（记录，未修改）：A 的 `features` 覆盖 **24** 层（含 layer 0，72 张量），E2 可训练 TTT 覆盖 **23** 层（69 张量，不含 layer 0）。原因：A 按 `config['candidate_ttt_layers']`（24 项、含 0）建分支（`transfer.py:182`），而 E2 的 `install_prefix_ttt` 按 `FULL_ATTENTION_LAYERS=(0,3,7,…,31)` 排除锚点层（`hybrid.py:14,103-107`）。载入 A 时只对存在 `prefix_ttt` 的层赋值，layer 0 分支被静默忽略：不报错、不参与 B，A 阶段约 1/24 的算力对应这个未使用分支。layer 0 同时出现在 `candidate_ttt_layers` 与 `full_attention_layers`，是否为配置笔误待用户判断；不影响迁移。
- world_size 4→8 适配（本轮实施）：`identity` 不再包含 `world_size`（`total_steps`/`manifest`/`config`/`stage_a`/`layout`/`diagnostic_only` 仍严格比较）；`world_size` 改为显式写入 `run.json`、`latest.pt`/`pilot.pt` 状态与 `result.json`；恢复时打印旧→新 world_size，并对 `rng_by_rank` 做长度保护——有存档的 rank 恢复原状态，新增 rank 保留进程启动时的确定性初始状态。batch 语义不变（micro 1 × world × 累积 = 128，八卡每卡 16 样本），无需其他改动。
- 验证：`pytest -m 'not gpu'` → 163 passed、3 skipped、17 deselected、0 failed（11.17s）。未跑 GPU 恢复验证。
- 路径问题（待用户决策，未实施）：硬编码常量已删除，故改写 manifest 不再触发「Fixed manifest SHA256 mismatch」；但 `sft.py` 恢复时仍比较 `identity['manifest_sha256']`（由清单文件字节计算）与 checkpoint 内记录值，**改写文件会改变该值并拒绝恢复**。即「批量替换路径」与「保留自动身份校验」二者不可兼得。若沿用自动校验，则保持 manifest 字节不变、在读取期把清单记录的数据根重定位到本机 `config['data_root']` 并按原 `inputs` 摘要校验（约 8 行 + 4 处调用点传 `data_root`），`build_dataset` 无需改动。

## 2026-09-10 迁移适配实施与 8 卡恢复验证（代码改动 + GPU 实测）

用户决定：layer 0 差异按最小改动修复、路径采用「清单字节不动 + 读取期重定位」(a)、继续完成 world_size 适配。

- layer 0 修复（代码，未动配置）：`transfer.py` 建分支时排除 `config['full_attention_layers']`（新增 `anchors` 一行、生成式加一个过滤条件），使 A 与 E2 的 TTT 层集合一致（均为 23 层）。**不能改 `configs/base.json`**：实测其 `digest_json = 4ff7ea01…` 与 checkpoint 记录的 `config_sha256` 相同，且与迁移来的 `source-base.json` 字节相同；改动配置会使恢复身份校验失败。
- 读取期重定位 (a)：`data_pipeline.py` 新增 `local_path(recorded, data_root)`，`load_manifest(path, data_root)` 用清单原字符串作 `inputs` 键、把记录路径重定位到本机数据根后按原摘要校验，并把重定位结果写回内存中的 `manifest['annotation']`（`build_dataset` 无需改动）；四个调用点（`sft.py`、`transfer.py`、`switch_diagnostic.py`、`pilot_diagnostic.py`）改传 `config['data_root']`。清单文件字节不变。
- 新增 `sft.py --stop-after N`：从起点（恢复点）再跑 N 步即停，**不进入训练身份**；用于短时恢复验证（`--max-steps` 会把 `diagnostic_only` 置真而拒绝恢复正式 checkpoint，不能用于此目的）。
- 验证一（CPU）：`pytest -m 'not gpu'` → 163 passed、3 skipped、17 deselected、0 failed（11.15s）。重定位实测：`load_manifest` 返回值 = `246586be…`（与 checkpoint 一致）、磁盘清单字节未变、`annotation` 解析为 `data/llava-v1.5-assets-v1/datasets/llava-665k/llava_v1_5_mix665k.json` 且文件存在、train/dev/A = 663248/2048/50000。
- 验证二（8×H100 实跑，只跑 2 步，输出到 scratch 目录 `artifacts/verify/e2-8gpu-smoke/`，不写正式轨迹）：命令为 `torch.distributed.run --standalone --nproc_per_node=8 -m prefix_ttt.sft --layout E2 --config configs/base.json --manifest artifacts/migrations/from-a40-20260910-step1075/fixed_manifest.json --stage-a-checkpoint …/A-complete.pt --resume …/E2-resume.pt --output artifacts/verify/e2-8gpu-smoke --save-every 100 --stop-after 2`，退出码 0，日志 `artifacts/verify/e2-8gpu-smoke.log`。
  - 恢复正确性：首个新 step = **1076**、`samples_seen = 137728`，与指南要求一致；`world_size 4 -> 8` 提示按 rank 打印；无 OOM，无 FLA/Triton 编译失败。
  - 两个 step 全部有限：step1076 loss 0.876779、grad_norm 3.120507、耗时 102.12s（含首次 Triton 编译与图像加载）；step1077 loss 0.909872、grad_norm 4.873047、耗时 **8.42s**。学习率 `[1.839e-05, 9.196e-05]` 承接余弦调度、未重启 warmup。
  - 与旧机对照（`source-steps.jsonl` 末三行为 step1088/1089/1090，说明旧机已训练到 1090 步且仍在继续）：loss 0.899/0.908/0.927、grad_norm 3.85/8.57/3.42、约 32.2s/step。本机 8 卡 8.42s/step 约为旧机 4 卡 32.2s/step 的 1/3.8，loss 处于同一区间；因每卡样本集合与梯度累积顺序改变，**不声称逐位复现**。
  - 保存重载：`latest.pt` 1285768519 B，CPU 加载校验 `global_step=1077`、`samples_seen=137856`、`world_size=8`、`rng_by_rank` 8 份、`trainable` 517、`optimizer.state` 517、`scheduler.last_epoch=1077`、`manifest/config/stage_a` 三个摘要与来源一致、`complete=False`、`diagnostic_only=False`。
- 边界提醒：旧机仍在训练（源日志已到 1090 步）。若从 1075 接管，旧机 1076 步之后的更新不属于本机轨迹，两边不得合并 optimizer 或宣称同一条权威轨迹；接管边界仍需用户与旧机确认。
- 未做：未提交正式训练队列，未安装 Supervisor，未改动 configs 中的旧机路径，未测整模型质量。

## 2026-09-10 面向后续重训的收尾硬化（代码改动，已测）

用户问「以后重新训练是否能避免这些问题」，据此排查并对齐两处仍会复现的遗留。

- `transfer.py` 与 `sft.py` 对齐：A 阶段 `identity` 同样移除 `world_size`，改为显式写入 `run.json`、`latest.pt`、`result.json`。此前只有 B 阶段做了该调整，A 阶段仍会因卡数变化拒绝续训。`switch_diagnostic.py` 的期望字段不含 `world_size`，不受影响。
- `sft.py` 载入 A 特征时增加显式提示：计算模型实际安装 TTT 的层集合，若 A checkpoint 含多余层则打印 `Phase A checkpoint carries unused feature layers: [...]`。原先该循环静默跳过非 TTT 层，正是它把 layer 0 的多余分支掩盖了数月；现在新 A 训练层集合已一致（23 层，不会触发提示），而历史 checkpoint（24 层）会明确提示。
- 验证：`pytest -m 'not gpu'` → 163 passed、3 skipped、17 deselected、0 failed（11.27s）。
- 结论（面向未来重训）：不会再复现的问题包括——A 与 B 的 TTT 层集合不一致、清单重新生成需要改代码里的固定 SHA、开训前必须准备并通过诊断报告、换机器后清单内绝对路径全部失效、以及训练卡数变化导致无法续训。仍然保留且属刻意的约束——恢复时的自动身份校验（manifest/config/stage-A 摘要、预算、layout）、新起 E2 必须提供 A、以及 `configs/base.json` 摘要与 checkpoint 绑定（改动配置即无法续训；若要重训应同时重新生成清单与配置，并可顺手清掉 `candidate_ttt_layers` 中多余的 `0` 与 A40 口径的 `gpu_ids`/`gpu_status`）。

## 2026-09-10 放弃 A40 迁移、改为按任务书重训（用户决定）

- 用户决定：删除全部 A40 迁移产物，按任务书 `docs/experiments/2026-09-09-prefix-ttt/plan.md` 从零重训；本轮修改另开 `h100` 分支并推送 GitHub，以便在另一台主机配置多机环境。
- 删除对象与证据（删除前已记录文件与字节数，删除后 `artifacts/` 仅余 1.4M）：
  - `artifacts/migrations/from-a40-20260910-step1075/`（1.6G）：`E2-resume.pt` 1285708991 B、`A-complete.pt` 339875595 B、`fixed_manifest.json` 8468345 B、`source-steps.jsonl` 231046 B、`source-switch-A.json` 5151 B、`source-E2-pilot-diagnostic.json` 2872 B、`source-run.json` 2736 B、`source-base.json` 1230 B、`SHA256SUMS` 600 B、`A-SHA256SUMS` 80 B。
  - `artifacts/verify/`（1.2G）：`e2-8gpu-smoke/latest.pt` 1285768519 B、`trainable_params.json` 302589 B、`e2-8gpu-smoke.log` 11428 B、`run.json` 2542 B、`result.json` 465 B、`steps.jsonl` 428 B。该验证针对已删除的迁移 checkpoint，其结论（world_size 4→8、路径重定位、`--stop-after`）对现行代码仍然有效。
  - **未删除**：`data/llava-v1.5-assets-v1/`（69G，重训必需）、`artifacts/wheels/`（FLA 兜底 wheel，1.4M）、全部 `docs/`。
- 一处自我纠正：任务书 `plan.md:41-44` 明确 `candidate_ttt_layers` 为 24 项（含 layer 0）、`ttt_layers` 为 23 项，并规定「anchor 不安装无用分支」。因此配置中的 24 项**符合任务书**，不应修改；真正的问题是 `transfer.py` 未排除锚点，本轮代码修复方向正确。
- 配置清理：`configs/base.json` 删除 `gpu_ids` 与 `gpu_status` 两个键（无任何代码读取，且为旧机 A40 口径；按 AGENTS.md，允许使用的 GPU 属脚本/调度参数）。重训不受 config 摘要绑定影响。
- 验证：`pytest -m 'not gpu'` → 163 passed、3 skipped、17 deselected、0 failed（11.24s）；`configs/base.json` 仍为合法 JSON，顶层 21 键。
- 分支与推送：`h100` 分支 commit `e8b1f11`（作者 `DeepSeek <noreply@deepseek.com>`）已推送至 `git@github.com:greed-106/prefix-ttt.git`，远端 `refs/heads/h100` 指向同一提交；`main` 未改动。
- 删除 `migration_from_a40.md`（用户决定，1 个文件）：该指南随迁移包写入，面向已放弃的 A40 checkpoint 接续；其内容要点仍保存在本账本前述各节。本账本中引用该文件路径的旧句保留为历史记录，文件本身不再存在。

## 2026-09-10 重训前置：数据审计与固定清单生成（进行中）

- 必要性与耗时结论（基于代码与实测，非估算）：
  - **两者都是硬门槛，不能跳过**。`manifests.py:69-80` 在生成清单前强制校验：标签审计必须 `labels.complete=True`、`checked == counts.records`（665298）、`preprocessing_verified=True` 且无 `preprocessing_error`；图片审计必须 `missing_images == []`、`decode.checked == unique_images`（349034）、`decode.failures == []`，且两份摘要的记录数与唯一图片数一致。因此想得到固定清单，标签审计与图片解码审计都必须完整跑完。
  - 实测吞吐（本机 192 逻辑核）：标注全量扫描 6.4s；标签审计 16384 条 / 64 workers 用时 17.05s（含 6.4s 扫描与 worker 启动），折合约 1500 条/s，全量 665298 条约 **7–9 分钟**；图片解码 8192 张随机样本 32 线程 3.34s（约 2450 张/s），全量 349034 张约 **2.4–3.3 分钟**；清单生成本身为单进程 ijson + 并查集遍历，预计数分钟。
  - 因此本机全 CPU 的端到端预计 **约 15–25 分钟**，瓶颈是标签审计。作为对照，旧容器 CPU 配额仅 4 核，标签审计持续数小时，与本机差距来自核数而非代码。
- 已启动流水线（后台，输出 `artifacts/cpu/`）：① `audit_data --output artifacts/cpu/data --decode --workers 32`；② `audit_data --output artifacts/cpu/labels-full --labels --workers 64`；③ `manifests --output artifacts/cpu/fixed_manifest.json`。日志 `data-decode.log`、`labels-audit.log`、`manifest-build.log`。
- 待核查：生成后应比对 `order_sha256` 的 train/dev/A 三项是否等于旧记录（`0a2bd76a…`、`1298c9ec…`、`a63faa46…`）——若相等，可证明新清单的划分与旧受审清单一致；注意清单文件整体 SHA 预计与旧值不同（执行源码快照等字段随本机而变），这属预期，不代表划分改变。






## 2026-09-10 重训前置完成：审计与固定清单（实测结果）

- 流水线实际耗时：图片解码审计 2m41.6s（349034 张，32 workers）、标签审计 5m54.9s（665298 条，64 workers）、清单生成 26.9s，端到端约 **9 分钟**，优于事前的 15–25 分钟估计；无需 GPU。
- 审计结果（全部满足 `manifests.py` 的门槛，无一项放宽）：
  - 图片：`records=665298`、`unique_images=349034`、`missing_images=[]`、`decode.checked=349034`、`decode.failures=[]`。
  - 标签：`checked=665298`、`complete=True`、`preprocessing_verified=True`，`statuses={'valid': 665296, 'data_no_assistant_content': 2}`、`length_bins={'<=2048': 665149, '>2048': 147}`——与旧机记录（665296 valid、2 条空 assistant、147 条超长）完全一致。
- 新固定清单：`artifacts/cpu/fixed_manifest.json`（8468237 B），`counts` 与旧清单逐项相同（original 665298、valid 665296、train 663248、dev 2048、A 50000、excluded 2、valid_groups 389196）。
  - **`order_sha256` 三项与旧记录完全相等**：train `0a2bd76a…`、dev `1298c9ec…`、A `a63faa46…`，即划分与旧受审清单一致。
  - 整体 SHA256 = `9280270bf2453897c3ff60f2265dfe84630326fc9dfab78f0fc256bdfac63d4b`，与旧值 `246586be…` 不同。差异来源已定位为**来源与环境元数据而非划分**：① `annotation` 与 `inputs` 的四个键由 `/data/ymj/...` 变为 `/data/mjyang/...`；② `audit_execution_source_sha256` 在本机为空 `{}`（本机不存在旧机那份 `artifacts/cpu/audit_data.executed.py` 执行源码快照）。
  - 已用现行代码验证可用：`load_manifest(path, 'data/llava-v1.5-assets-v1')` 返回同一 SHA、annotation 正确解析、`trajectory(663248)` = 5182 步 / pilot 391 步 / 50048 样本。
- 多机注意：两台机器应使用**同一份**清单文件（本机 SHA256 见上），或各自生成后核对 `order_sha256` 三项——该三项由数据与种子决定，是划分的真正不变量；整体 SHA 会随路径与快照字段变化。

## 2026-09-10 清理 A40 残留与跨主机就绪

- h100-1（`cucloud-server1`，8×H100，2TB 内存，`/data` 余 13T）已由用户配好环境（`h100` 分支 `b1c4f40`、venv 就位）。
- 数据同步与校验（rsync 内网 550 MB/s）：
  - `data/llava-v1.5-assets-v1/` 共 72,679,799,380 B、349103 文件，用时 2m05s；远端文件数 349103 与本机相同。
  - 远端 annotation SHA256 = `ce959ce6…`、图片 349034 张、三个权重分片 `c11dbf01…`/`46df6c6e…`/`4f06177c…`，全部与本机及旧机证据一致。
  - 固定清单 SHA256 = `9280270bf2453897…` 与本机相同，另同步 `artifacts/cpu` 审计产物（183M）。
  - 远端端到端校验通过：`load_manifest` 返回同一 SHA、`order_sha256` 三项一致、annotation 解析到本地文件、`trajectory` = 5182/391/50048。
  - 首轮 rsync 因远端缺少 `data/` 父目录失败一次（rsync 不建多级父目录），补建后成功；失败记录保留在 `artifacts/cpu/rsync-data.log` 之前的输出中。
- 按用户决定删除的旧机配置（共 5 个文件）：`configs/jobs.json`、`configs/jobs-pilot.json`、`configs/jobs-e2-full.json`、`configs/jobs-e2-resume.json`、`configs/gpu-environment.json`。删除理由：路径全为 `/data/ymj/...`，且指向已删除的 `artifacts/training/{A,E1,E2}`、`pilot-diagnostic/*`、`artifacts/lmms/e0/*`；`src/` 与 `tests/` 无任何引用。历史可从 git 找回。
- `tests/test_supervisor.py` 改为自动探测：`shutil.which` 查找 `uv`/`supervisord`/`supervisorctl`，支持 `PREFIX_TTT_UV`/`PREFIX_TTT_SUPERVISORD`/`PREFIX_TTT_SUPERVISORCTL` 显式覆盖，找不到时仍 skip。删除硬编码的 `/data/ymj/.pixi/bin/*`。
- 验证：`pytest -m 'not gpu'` → **166 passed、17 deselected、0 skipped、0 failed**（17.56s）。skip 由 3 降为 0 是因为本机现已安装 pixi 全局 Supervisor 4.3.0（`/data/mjyang/.pixi/bin/supervisord`），3 个生命周期用例真实执行并通过。
- 用户决定保留 `docs/experiments/2026-09-09-prefix-ttt/` 全部 736K 历史（含任务书 `plan.md`、旧机证据与旧 `environment.lock.json`）。
- 仍待决策的残留：`configs/supervisor/jobs.example.json`、`configs/supervisor/supervisord.conf`、`configs/systemd/prefix-ttt-queue.service.example` 仍含 `/data/ymj` 路径，且 `supervisord.conf` 指向已删除的 `configs/jobs-e2-resume.json`（现已失效）。
- 多机限制记录：现有 SQLite 调度器只做单机 GPU 分配（本地 `gpu_ids` + 本地队列库），**不支持跨主机调度**；双机运行需要手动跨机 `torchrun`，或先扩展调度器。

## 2026-09-10 多机 NCCL/RDMA 验证与多机训练支持（代码改动，未提交）

- 两节点环境：`cucloud-server3`（172.18.1.184，本机）与 `cucloud-server1`（172.18.1.156，h100-1），各 8×H100 80GB、192 核、2TB 内存；均具备 `mlx5_0..13` 与 `mlx5_bond_0` IB HCA（`PORT_ACTIVE`，InfiniBand link layer），NCCL 2.26.2 与 torch 2.7.1+cu128 完全一致；以太网 RTT 0.7ms。IPoIB 接口（`ibp*`）全部 DOWN，但 NCCL 走 verbs 不需要 IPoIB。
- 验证一（每节点 1 卡、2 rank）：NCCL 选中 NET/IB，LID 27/51、`subnet-prefix 33022`（同一 IB 子网）；`sum=3.0` 正确；64MiB all-reduce 2.45ms / 2.48ms。
- 验证二（每节点 8 卡、16 rank）：ranks 0–7 位于 server3、8–15 位于 server1；全部 `sum=136.0`（=16×17/2）正确；64MiB all-reduce **1.677–1.721 ms**（≈40 GB/s 有效带宽）；两端均使用 NET/IB（NCCL 识别 11 个 IB/RoCE 设备）。
- 已知非阻塞项：日志为 `GDR 0`，即未启用 GPUDirect RDMA；原因是 `nvidia_peermem` 未加载（模块文件存在：`/lib/modules/5.15.0-122-generic/updates/dkms/nvidia-peermem.ko`，版本 570.211.01 与驱动匹配）。数据经主机内存中转但带宽仍有约 40 GB/s，相对 8s 级步时可忽略。有 sudo 时可直接 `modprobe nvidia_peermem` 尝试启用（见后续更正节）。
- 代码改动（最小化，3 文件 +17/−7）：
  - `sft.py` 与 `transfer.py`：`dist.init_process_group('nccl', device_id=device)` 显式绑定本 rank 设备；`world > 1` 时打印 `{host, rank, world_size, local_rank, device}`，便于跨机核对 rank→主机映射。
  - `gpu_communication.py`：同样绑定 `device_id`；输出新增 `host` 与全局 `rank`、并把原字段改为 `local_rank`（原先把 LOCAL_RANK 标为 rank，多机下会误导）。
- 验证三（用新代码复跑 16 rank）：两端退出码 0；rank→host 映射正确（0–7 cucloud-server3，8–15 cucloud-server1）；sum 全部 136.0；1.68–1.72 ms。
- CPU 套件：166 passed、17 deselected、0 failed（16.93s）。
- 多机启动方式（不得使用 `--standalone`）：两台机器分别执行
  `torchrun --nnodes=2 --nproc_per_node=8 --node_rank=<0|1> --master_addr=172.18.1.184 --master_port=<port> -m prefix_ttt.sft ...`（A 阶段把模块换成 `prefix_ttt.transfer`）。
- 重要限制（本次未改代码，仅记录）：两机 `/data` 均为本地盘（server3 为 xfs、server1 为 ext4），**没有共享文件系统**；rank 0 的 `--output`、`steps.jsonl`、checkpoint 只会落在 rank 0 所在节点，跨机训练需人工回收/同步，或改用共享存储。
- h100-1 代码状态：该机 checkout 仍为 `c8501f7`（不含 h100 提交）。本次为跨机验证仅 rsync 了上述 3 个文件到其工作树（`git status` 显示 3 个 M）；正式训练前需让该机取得 `h100` 分支代码。
- 未做：未提交、未推送（遵循 AGENTS.md 的提交与推送授权规则）。

## 2026-09-10 跨机共享存储方案（宿主机 NFS，待用户配置）

- 需求：两台机器需要一个共同的 checkpoint 落盘位置，避免不同节点各自写本地盘后再人工搬运。
- 环境事实（已实测）：
  - 两机 `/data` 均为本地 LVM（server3 xfs、server1 ext4），**不存在现成共享文件系统**；h100-1 上 `/data/shared/weights/prefix-ttt` 原本不存在。
  - 两台机器都是**裸金属主机**（PID 1 = systemd、`systemd-detect-virt` = none、根文件系统分别是 `/dev/sda4` 与本地 LVM，无容器标记文件）。此前"容器内无法 mount"的判断基于错误前提，已更正，见下节。
  - `mjyang` 在 server3 上 **sudo 免密可用**（`sudo -n id` → uid=0），在 server1 上 sudo **需要密码**（`sudo -n` 报 password required）。因此 server3 侧的系统级配置可由本智能体直接完成，server1 侧需要用户执行。
  - `/etc/hosts` 为 root:adm 0644 的普通文件（非独立挂载），有 sudo 即可修改；`nvidia_peermem` 模块存在（`nvidia-peermem.ko` 570.211.01）但未加载，有 sudo 即可 `modprobe`。
  - 两端 `mjyang` 的 uid/gid 相同（1001/1002，另含 `fastwam` 1003），但**不作为前提**：导出使用 `all_squash,anonuid=1001,anongid=1002`，任何 uid 的客户端都可写；目标目录 owner 为 `mjyang`、setgid 到 `fastwam`。
  - 两机互访带宽实测 550 MB/s（72.7GB/2m05s）；h100-1 → 本机的 `ssh h100-3`（172.18.1.184）通道亦可用。
- 选定方案 A：宿主机会话在 server3 导出 `/data/shared/weights/prefix-ttt`，在 server1 挂载到**同一路径**；此后训练零代码改动，`--output` 指向该路径即可，两端看到同一份数据。
- 注意：该挂载点只用于 checkpoint 与训练日志等大文件；训练数据（34.9 万张图片）继续两端各自本地，避免小文件走 NFS 造成慢速。
- 待办：server3 侧配置可由本智能体执行，server1 侧需用户执行（sudo 需密码）；配置完成后做跨机写入验证（从 h100-1 写入文件并在 server3 确认可见、核对 uid 与吞吐）。

## 2026-09-10 更正：本机为裸金属主机，不是容器

- 起因：用户质疑"容器内 `/etc/hosts` 不可写"的说法。复查证据：`/proc/1/comm` = `systemd`（而非容器里的 bash/init 替身）、无 `/.dockerenv` 与 `/run/.containerenv`、根文件系统为 `/dev/sda4` ext4（非 overlayfs）、`systemd-detect-virt` = `none`，两端一致。因此**两台机器都是裸金属主机**。
- 错误来源有两处：① 旧账本（另一台 A40 机器）记录"PID1=bash、user systemd 不可用"，我未加验证就把它当成本机事实沿用；② 更早一次 `sudo` 失败发生在 DSH 文件沙箱仍生效时（沙箱会设置 `no_new_privs`，令 setuid 提权失败），我据此推断"无法提权"，但沙箱解除后 `sudo -n id` 直接返回 uid=0。
- 更正后的结论：`/etc/hosts` 可用 sudo 修改，NFS 可用 sudo 挂载，`nvidia_peermem` 可用 sudo 加载。此前提出的替代方案——用 `ssh -G` 从 `~/.ssh/config` 解析会合地址、以及"宿主机挂载再传播进容器"——**均无必要**；正确做法是把两台机器的主机名统一写进各自的 `/etc/hosts`，NFS 与 `torchrun --master_addr` 都直接使用主机名，任何命令中都不出现 IP。
- 本账本中"容器"一词此前多处指代本机，属用词错误；标题与开篇已改为"主机/机器"，其余历史条目保留原文以如实反映当时的判断。

## 2026-09-10 共享存储脚本与别名解析（方案 A 落地，未提交）

- 用户确认采用方案 A（宿主机 NFS），并具备 sudo 权限；要求命令中不再出现 IP、且尽量用户级、从节点可快速配置。
- 关键实测与决策：
  - **UID 不必一致**：导出加 `all_squash,anonuid=1001,anongid=1002`，所有客户端写入统一映射为该 uid/gid，从节点无需与其对齐（此前两端 uid 恰好相同只是巧合，不作为前提）。
  - **别名解析分两层**：宿主机 `/etc/hosts` 里的主机名供 NFS 使用（`/etc/exports` 与 `mount` 命令因此不含 IP）；容器内**无法**解析这些别名——`HOSTALIASES` 实测不生效，容器 `/etc/hosts` 不可写（root 所有），且其中 `cucloud-server3` 映射到 `127.0.1.1`，若用作 `--master_addr` 会绑到回环。
  - 因此 `--master_addr` 由启动脚本用 `ssh -G <alias>` 从 `~/.ssh/config` 解析：实测 `ssh -G h100-1` → `172.18.1.156`；本机缺少 `h100-3` 自身别名时脚本明确报错，不会静默使用错误地址。
- 新增脚本（`docs/experiments/2026-09-10-prefix-ttt-h100/scripts/`，已 `bash -n` 检查，未提交）：
  - `nfs-server-export.sh`：在存储宿主（h100-3）以 sudo 运行；写入 `/etc/hosts` 别名、安装 nfs-kernel-server、以 `all_squash` 导出 `/data/shared/weights/prefix-ttt` 给 `h100-1`。
  - `nfs-client-setup.sh`：在每个训练节点以 sudo 运行、幂等；写入别名、安装 nfs-common、按别名挂载到同一路径并写探针文件。
  - `run_multinode.sh`：纯用户级、无需 root；从 `~/.ssh/config` 解析会合地址后调用 `torchrun`，命令中不含 IP。
- 待用户执行：两台宿主机分别跑上述两个脚本；另需在 server3（本机）的 `~/.ssh/config` 增加 `h100-3` 指向自身的别名，启动脚本才能在本机解析。
- 未做：未提交、未推送；未在宿主机执行任何 sudo 操作；未挂载任何文件系统。

## 2026-09-10 server3 侧共享存储与 GDR 配置（系统级改动，已执行）

- 经用户授权执行两项：
  - `sudo bash docs/experiments/2026-09-10-prefix-ttt-h100/scripts/nfs-server-export.sh`：向 `/etc/hosts` 写入两条映射（`172.18.1.184 cucloud-server3 h100-3`、`172.18.1.156 cucloud-server1 h100-1`）；安装 nfs-kernel-server；导出 `/data/shared/weights/prefix-ttt` 给 `cucloud-server1`。`exportfs -v` 实测选项为 `rw,sync,no_subtree_check,all_squash,anonuid=1001,anongid=1002,root_squash,sec=sys`；目录权限 2775。
  - `sudo modprobe nvidia_peermem`：模块加载成功（`nvidia_peermem 16384 0`，由 `nvidia` 引用）。
- 验证：
  - `showmount -e localhost` → `/data/shared/weights/prefix-ttt cucloud-server1`。
  - NCCL 通道对比（2 节点 × 1 卡，NCCL_DEBUG=INFO）：server3 侧 8 条通道全部为 `NET/IB/11/GDRDMA`；h100-1 侧（未加载 peermem）为 `NET/IB/13`，无 GDRDMA。
  - 64MiB all-reduce：server3 2.478ms、h100-1 2.461ms；与加载前（2.448/2.480ms）无实质差异——该尺寸下 GDR 不改变时延，主要节省 CPU 与主机内存带宽。
- 待用户在 h100-1 执行（该机 sudo 需密码）：
  - `sudo modprobe nvidia_peermem`
  - `sudo bash /data/mjyang/code/llm/prefix-ttt/docs/experiments/2026-09-10-prefix-ttt-h100/scripts/nfs-client-setup.sh`
- 未做：未配置模块开机自动加载（`/etc/modules-load.d`）；未做更大消息尺寸的 GDR A/B；未在 h100-1 执行任何 sudo 操作。

## 2026-09-10 客户端脚本增加「同机」与「占用」判定（已实测）

- 结论确认：所有计算节点都是客户端，服务端只提供中心存储；**服务端本机不应挂载自己的导出**，直接用本地目录。理由：同一份数据经本地路径与 NFS 路径两种方式访问会导致属性缓存不一致；本机 NFS 服务重启时硬挂载可能挂住本地进程；且白白绕一圈网络栈。
- `nfs-client-setup.sh` 改为三分支判定（按优先级）：
  1. 已是挂载点：读取 `findmnt -no SOURCE --mountpoint`，来源不是 `h100-3:<SHARE>` 则**报错退出**，避免误用别的来源。
  2. 未挂载但本机在导出该目录（`exportfs -s | grep -F "$SHARE"`）：判定为服务端本机，**跳过挂载**，使用本地目录。
  3. 未挂载且本机不是导出方：若目标目录已存在且**非空**，**报错退出**（挂载会遮蔽本地数据）；否则按需装 `nfs-common`、建目录、以别名挂载，最后 `findmnt` + 写探针文件自证。
- 实测（在 server3 上以 sudo 运行）：
  - 分支 2 命中：输出 `is exported by this host; using the local directory, nothing to mount`，未执行任何挂载；探针写入成功（`.probe-cucloud-server3`）。
  - 分支 3 的占用保护：用临时目录（内含一个文件）替换 `SHARE` 后运行，脚本以退出码 1 报 `already holds local data; mounting over it would hide that data`，`mountpoint` 确认未挂载、原文件完好。
  - 分支 1 与正常的客户端挂载路径需在 h100-1 上实测（该机 sudo 需密码）。
- 未做：未提交、未推送；未在 h100-1 执行任何 sudo 操作。

## 2026-09-10 多机支持提交并同步到 h100-1

- 本机提交 `49acbe2`（作者 `Codex <codex@openai.com>`，7 文件 +194/−10），内容：多机 `device_id` 绑定与启动日志、`scripts/` 三个脚本、账本（含 NCCL/RDMA 实测与"本机是裸金属而非容器"的更正）。已推送，远端 `refs/heads/h100` = `49acbe2`。
- h100-1 同步：该机原在 `main` @ `c8501f7`，且带有此前 rsync 产生的 3 个已修改文件与未跟踪的脚本目录。处理方式：`git checkout -- src/prefix_ttt/` 丢弃本地改动、`rm -rf docs/experiments/2026-09-10-prefix-ttt-h100`（提交中已包含）→ `git fetch origin && git checkout -B h100 origin/h100`。现该机位于 `h100` @ `49acbe2`，工作树干净。
- 一致性校验：两端 `sft.py`、`transfer.py`、`gpu_communication.py` 与三个脚本的 SHA256 完全相同；脚本在 h100-1 上保留可执行位（`rwxrwxr-x`）。
- 写入路径与解析的关系（澄清）：训练写的是**挂载后的本地路径**，NFS 客户端在挂载时已完成解析，写入过程不再涉及名字解析或 IP；仍需解析的只有两处一次性动作——挂载时的 `h100-3:<SHARE>` 与 `torchrun --master_addr=h100-3`。两者都由脚本写入的 `/etc/hosts` 满足。
- 待办：在 h100-1 以 sudo 运行 `nfs-client-setup.sh` 完成挂载（该机 sudo 需密码）；随后由本机核实探针文件可见性与属主、跨机写入吞吐。

## 2026-09-10 共享存储联通性验证（h100-1 已挂载，全部通过）

- 用户在 h100-1 以 sudo 执行 `nfs-client-setup.sh`：输出 `mounted h100-3:/data/shared/weights/prefix-ttt at /data/shared/weights/prefix-ttt`；`findmnt` 显示 `nfs4 rw,relatime`；探针 `.probe-cucloud-server1` 写入成功。
- server3 侧验证结果：
  - **双向可见**：h100-1 创建的探针在 server3 可见；server3 创建的文件在 h100-1 可见。
  - **`all_squash` 生效**：h100-1 创建的 `.probe-cucloud-server1` 与 2GB 测试文件在 server3 上属主均为 `mjyang:fastwam`（uid 1001），与客户端自身 uid 无关。
  - **squash 语义严格**：用 `sudo` 在 server3 创建的文件属主为 `root:fastwam`，h100-1 经 NFS **读取被拒**（客户端凭据被映射为 1001:1002，既非 owner 也不在 `fastwam` 组）。结论：**共享目录中的文件必须由 `mjyang` 创建，不得用 sudo 写入**；训练以 mjyang 身份运行，符合该要求。
  - **吞吐**：h100-1 → server3 写入 **837 MB/s**（2GB/2.57s）、读取 **1.1 GB/s**（server3 侧 `drop_caches` 后）；server3 本地写入对照 **1.7 GB/s**。按 1.3GB checkpoint 估算，跨机写入约 1.6s。
  - **原子性**：从 h100-1 执行「写 `.tmp` + rename」成功，server3 侧只出现最终文件、无 `.tmp` 残留——与 `sft.py` 的保存路径一致，可安全用于 checkpoint 落盘。
- 清理：删除 root 属主的测试残留与 dd 测试文件；共享目录现仅保留 `.probe-cucloud-server1` 与 `training/` 目录（均为 mjyang 属主）。
- 未做：h100-1 未加载 `nvidia_peermem`（GDR 目前仅 server3 侧生效）；本次账本更新未提交。

## 2026-09-10 两端启用 GDR 后的 RDMA 复测（更正此前的判断）

- 用户已在 h100-1 执行 `sudo modprobe nvidia_peermem`；两端现均为已加载状态（server3 与 h100-1）。
- 16 卡（8+8）复测结果：
  - 通道：两端各 8 个 IB 设备、每个 32 条通道，**全部为 `NET/IB/.../GDRDMA`**（此前仅 server3 侧为 GDRDMA，h100-1 侧为 `NET/IB/nn` 无 GDR）。
  - 正确性：两端全部 rank `sum=136.0`（=16×17/2），退出码 0。
  - 时延：64MiB all-reduce **0.533–0.568 ms**，此前（仅单端 GDR）为 1.677–1.721 ms，**提升约 3 倍**。
- **更正**：此前记录"GDR 不改变时延、主要节省 CPU 与主机内存带宽"，其依据是只启用单端 GPR 的测量，属不公平对比。双端启用后时延下降约 3 倍，该结论作废。按 64MiB/0.54ms 估算的有效聚合带宽约 230 GB/s，与单节点 8 个 IB HCA 多轨并行相符。
- 与训练的关联：B 阶段每步梯度 all-reduce 的通信开销本已相对 8s 级步时可忽略，此提升进一步压缩通信占比；跨机扩展的瓶颈现在更可能在数据加载与计算，而非网络。

## 2026-09-10 多机训练 smoke 通过（16 卡，E1 layout）

- 命令：两端各执行 `scripts/run_multinode.sh <0|1> prefix_ttt.sft --config configs/base.json --manifest artifacts/cpu/fixed_manifest.json --layout E1 --output /data/shared/weights/prefix-ttt/training/E1-smoke --max-steps 2 --save-every 1`。选 E1（仅 LoRA）是因为 `transfer.py` 的 E0 基线门槛尚未满足（见下节）。
- 结果：两端退出码 0；rank→主机映射正确（ranks 0–7 = `cucloud-server3`，8–15 = `cucloud-server1`，world_size=16）；训练 2 步：step1 loss 0.700951、grad_norm 0.081375、5.59s；step2 loss 0.581590、grad_norm 0.080431、4.14s；梯度与 loss 全部有限。
- 产物（写在共享 NFS，属主 mjyang）：`latest.pt` 960207747 B、`result.json`、`run.json`、`steps.jsonl`、`trainable_params.json`。
- 保存/重载验证：CPU 加载 `latest.pt` 成功，`layout=E1`、`world_size=16`、`global_step=2`、`samples_seen=256`、`trainable` 448 张量（E1 无 TTT，与 E2 的 448 LoRA + 69 TTT = 517 一致）、`optimizer.state` 448、`rng_by_rank` **16 份**、`diagnostic_only=True`（`--max-steps` 所致，符合预期）。`run.json` 记录 world_size=16、total_steps=5182。
- 结论：多机 rendezvous、数据管线、跨机梯度 all_reduce、共享存储落盘与重载**全部可用**，多机训练机制已验证。

## 2026-09-10 训练进程的 RDMA/GPU-Direct 实测与效率诊断

- 用户提问：正在跑的 A 训练是否真的走 RDMA、是否 GPU Direct。在运行中的进程上直接取证（非微基准）：
  - **RDMA 在用**：每个 rank 进程持有 **22 个 `/dev/infiniband/*` verbs 句柄**；IB 端口计数器显示真实流量（mlx5_0 单设备 15s 内约 127 MB 量级，8 个 rail 同时活跃）；**TCP 字节计数在 3s 与 20s 采样中完全不变**（3604/3620 → 3604/3620），说明节点间 TCP 只承担 rendezvous/控制面，**数据不走 TCP**。
  - **GPU Direct 在用**：训练期间 `nvidia_peermem` 引用计数为 **721**（GPU 显存已注册供 peer DMA），与此前 NCCL 日志中的 `NET/IB/.../GDRDMA` 一致。
  - 计数器单位已校准：`port_xmit_data` 为 4 字节单位（由字节增量/包增量≈461 与 MTU 4096 反推，平均包约 1.8 KB）。流量呈**突发**特征（每步末尾集中 all_reduce），短窗口采样波动大，早期 20s 采样低估了实际值。
- 效率诊断（用户观察到"每卡功耗不到 300W，远低于 700W TDP"）：
  - 实测：SM 利用率 38–58%、显存带宽利用率 10–18%、功耗 233–274 W、显存占用约 20 GB/80 GB；**每个 rank 进程恰好占用约 1 个 CPU 核（101–105%）**。
  - 结论：瓶颈不在网络、也不在卡数，而在**串行且单线程的数据预处理**——`prepare_sample` 在训练循环内逐个样本同步执行（图像解码/缩放、tokenize、多模态展开），期间 GPU 空转；且 `micro_batch_size` 被代码硬编码为 1（`accumulation_steps(world, 1)`，config 中的该字段从未被读取），每步每卡仅 8 个样本、逐个前向反传，kernel 小、启动开销占比高。
  - A 阶段整体仍很快：391 步约 12 分钟（旧机 4×A40 为 80.9 分钟）；但 B 阶段 5182 步在此效率下会显著拉长，值得先优化。
- 可选的优化方向（尚未实施）：① 数据预取/重叠（把样本准备移到后台线程或 DataLoader worker，与 GPU 计算重叠）——不改变数学与协议，安全性最高；② `micro_batch>1`——利用率提升最大，但需改数据路径且偏离任务书的 micro=1 协议，影响与既有 E1/E2 设计的可比性；③ 用两个独立 8 卡作业替代一个 16 卡作业——单卡效率更高，但单条轨迹墙钟更慢；④ 允许 CPU 侧多线程（当前 `OMP_NUM_THREADS=1`）以加速图像预处理，代价小、可快速试。

## 2026-09-10 micro_batch 与数据预取的审计（未实施改动）

- **现状**：`configs/base.json` 的 `training.micro_batch_size` 是**死字段**——全代码仅 `accumulation_steps(world, 1)` 出现，micro 被硬编码为 1（grep 确认无其他读取点）。每步每卡样本数由 `sample_group(order, cursor, rank, world)` = 128/world 决定（16 卡时 8 个）。
- **对 Prefix-TTT 正确性的影响：目标函数与 micro 分组无关**。
  - B 阶段：`token_normalized_ce` 返回 `Σ_micro CE / T_global`（调用处不传 world_size，故因子为 1），梯度经手动 `all_reduce` SUM 聚合 → 等价于全局 token 均值梯度。8 个样本拆成 8×micro1 或 4×micro2，总和相同。
  - A 阶段：`residual_transfer_loss(...).sum() / self.denominator`（denominator=128 全局样本数）→ 全局样本均值，同样与分组无关。
  - `clip_grad_norm_` 在 all_reduce 之后每步执行一次，不受分组影响。
  - 结论：**micro>1 不改变 Prefix-TTT 的公式、也不改变优化目标**。
- **但必须区分"数学不变"与"实现需验证"**：
  - 算子层已为批量+padding 设计：`local_attention` 用 `valid.nonzero` 只打包有效 token，且按行分块（`rows * blocks * block_size`）保证不同序列不混块，尾部零槽在因果掩码下不影响有效 query；`fla_prefix` 用 `_pack` + `cu_seqlens = valid.sum(1)` 做变长分段；B 阶段把 `prefix_valid_mask=metadata['valid_mask']` 传入混合模型。故架构上支持 micro>1，**但 padding 是否会污染 TTT 状态必须实测验证**。
  - A 阶段的 `TransferHooks` 诊断**假定每次前向对应一个样本**（`diagnostics[key] += value` 后除以 denominator=128 样本；`visual_samples` 每次前向计 1）→ micro>1 会让这些**日志**按 micro 倍数缩放（仅日志，不影响训练）。
  - 数据路径需真正批量化：`prepare_sample` 目前构造 `collate([dataset[index]])`（单样本），需改为多样本且 metadata 形状为 [B,T]。
  - 数值上归约顺序改变 → 与 micro=1 不再逐位可比。
  - 协议上任务书规定 micro=1，旧 A40 运行亦为 micro=1 → 改变后与既有设计与旧结果的协议一致性受影响，需在论文口径中说明。
- **若采用 micro>1，要求的等价性验证**：固定同一 128 样本组，micro=1 与 micro=2 各跑一步，比较 loss、grad_norm 与若干参数梯度（应在 bf16 容差内一致）；另做 padding 泄漏测试（单样本单独前向 vs 与更短样本同批，logits 必须一致）。
- **数据预取/重叠的含义**（回答用户提问）：指让"样本准备（图像解码、缩放、tokenize、多模态展开）"与"GPU 前向反传"并行——当前二者在同一线程内严格串行，GPU 需等 CPU 备好下一个样本。实现方式为有界队列 + 后台线程（PIL/torchvision 会释放 GIL）或 DataLoader worker。样本、顺序与数学完全不变，**协议风险为零**；按当前 SM 约 45%、每 rank 占满 1 核推算，预期收益可观。
- 建议顺序：先做「预取重叠」与「放开 CPU 侧线程」（协议不变），实测收益后再决定是否需要 micro>1。

## 2026-09-10 micro-batch + DataLoader 改造与逐卡 micro 支持（已实测）

用户要求：允许按 GPU 吞吐调整每卡 micro_batch_size、自动反算梯度累积，同时保持 global batch=128、每步样本集合、损失归一化、裁剪时机、学习率轨迹与数据进度不变。

- 实现（改动仅限"取数据"路径）：
  - `data_pipeline.micro_batches(order, cursor, rank, world, micro)`：把该 rank 在固定 128 组中的份额 `order[cursor:cursor+128][rank::world]` 重新分块，**集合与顺序不变**。
  - `data_pipeline.batch_loader(...)`：`DataLoader(batch_size=None, collate_fn=collate, num_workers=config.training.dataloader_workers)`，样本准备（图像解码/缩放/分词/展开）移入 CPU worker 并与 GPU 计算重叠。
  - `prepare_sample(base, batch, device)`：改为接收已 collate 的批；修正了原实现丢弃的 None 过滤（LLaVA 展开后 `input_ids` 等可能为 None）。
  - `sft.py`/`transfer.py`：`micro` 读自 config，并支持逐 rank 覆盖 `PREFIX_TTT_MICRO_BATCH`；`batches_per_step = accumulation_steps(world, micro)` 自动反算。
  - `transfer.py` 诊断：per-forward 平均量改为"先按样本求平均、再对样本求和"，使日志与 micro 分组无关（否则 state_rms/visual/text 统计会随前向次数缩放）。
- 配置：`micro_batch_size: 1 → 8`、新增 `dataloader_workers: 4`（world=16 时每卡每步正好 8 个样本，故 micro 上限为 8）。
- 逐条验证：
  - **global batch=128**：`accumulation_steps` 校验 `128 % (world×micro)==0`；micro=1/2/4/8 → 累积次数 8/4/2/1，每 rank 样本数恒为 8；16 个 rank 的并集恰好等于同一组 0..127（已用脚本断言）。
  - **样本集合/损失归一化/裁剪时机/学习率轨迹/数据进度**：diff 确认这些代码行**未被触碰**；`token_normalized_ce` 的 `targets` 仍是全局 all_reduce 后的 token 总数。
  - **异构 micro 安全**：每步集合通信次数与 micro 无关（sft：`all_reduce(targets)`×1 + 每参数×N + `all_reduce(loss_sum)`×1，均在微批循环之外；transfer：每参数×N + 每层诊断×23）→ 允许不同 rank 用不同 micro 而不破坏通信顺序。
  - **等价性实测**（同一 128 样本组、同起点）：micro=8 vs micro=1，step 1 全部指标相对差 <1.5%（`grad_norm` 5.5e-4、`normalized_sample_mean` 4.7e-4），`visual_samples`/`text_samples` 完全相同；micro=4 vs micro=1 在 3 步 × 3 层 × 10 指标中 87/90 项在 2% 内，超差 3 项均为 step 3 的 `gate_rms`（绝对值差 7e-5，属两步优化后的累积数值差异）。
  - **性能**：稳态 2.10–2.23 s/step（micro=1）→ 0.92–1.05 s/step（micro=4/8），约 **2.2×**；显存仅 16–20 GiB/80 GiB。
- 已归档：micro=1 的 A 运行移至 `training/A-micro1-reference`（便于对照）。micro=8 的正式 A 正在 `training/A` 运行。
- 未做：未实现"按实测吞吐自动选择 micro"的自适应调参（当前为人工通过 config 或 `PREFIX_TTT_MICRO_BATCH` 指定）；未提交、未推送。

## 2026-09-10 micro-batch 改造后的 A 阶段完成与 B 阶段启动

- 用户决定：micro_batch_size 仅由 config 人工指定，不做逐卡覆盖、不做自适应调参。据此已回退 `PREFIX_TTT_MICRO_BATCH` 覆盖（`src/` 中已无残留），保留 `accumulation_steps(world, micro)` 的自动反算。
- **A 阶段（micro=8 + DataLoader）完整跑完**：391/391 步、`complete=true`、`diagnostic_only=false`、samples_seen=50000、world_size=16、`config_sha256=bc5ec1e2…`（新配置）；耗时 01:56→02:03 共 **7 分钟**（micro=1 时为 15 分钟）。checkpoint 校验：23 层 × 3 = 69 张量、optimizer state 69 条、stage=A、文件 SHA `e83e7e32…`。
- **DataLoader 已确认工作**：每个 rank 进程下有 **4 个 `pt_data_worker`** 子进程（各约占 4% CPU），此前主进程独占 1 核 100% 的现象消失；GPU 侧 SM 利用率由 38–58% 升至 **89–94%**，功耗由 233–274 W 升至 **434–479 W**（H100 TDP 700 W）。
- **E2 smoke（3 步，18 卡… 实为 16 卡）**通过：loss 8.84959 / 9.05861 / 9.08356，grad_norm 107.39 / 82.69 / 137.30，稳态 2.05 s/步；checkpoint 校验 layout=E2、world_size=16、`trainable` **517**（448 LoRA + 69 TTT）、`rng_by_rank` 16 份。
  - 交叉验证：旧机 4×A40（micro=1、world=4）的 E2 首两步 loss 为 **8.84736 / 9.05157**，与本机 16×H100（micro=8、world=16）的 **8.84959 / 9.05861** 在千分位一致，说明数据管线、A 初始化与损失归一化在不同硬件与分组下可复现。
- **B 阶段正式训练已启动**（`setsid nohup` 脱离会话，两端 node_rank 0/1）：输出 `/data/shared/weights/prefix-ttt/training/E2`，`--save-every 25`，超时 6 小时。日志 `/tmp/B-run/node{0,1}.log`。
  - 首个 44 步：首步 33.1 s（含编译），后续中位数 **2.69 s/步**；按此估算剩余 5138 步约 **3.8 小时**。loss 8.85 → 4.54，grad_norm 正常，学习率轨迹 5.64e-06（LoRA）/ 2.82e-05（新增模块）随 warmup 上升。
  - 启动插曲：node 1 首次启动因远端缺少 `/tmp/B-run` 目录失败，补建后重启成功；node 0 在会合点等待并自动接入。
- 未做：未提交、未推送本次代码与账本改动。

## 2026-09-10 阶段 B（E2 完整轨迹）训练完成

- **结果**：5182/5182 步、`complete=true`、`diagnostic_only=false`、`samples_seen=663248`（全训练集）、world_size=16、`global_step=5182`；产物位于共享存储 `/data/shared/weights/prefix-ttt/training/E2/`（`latest.pt` 1285875223 B、`pilot.pt` 1285873117 B、`steps.jsonl`、`run.json`、`result.json`、`trainable_params.json`）。
- **耗时**：02:05 启动 → 06:00 结束，约 **3 小时 55 分**；首步 33.1s（含编译），稳态中位数 **2.63 s/步**、均值 2.65 s/步（含每 25 步一次约 1.3GB 的 NFS checkpoint 写入）。
- **健康度**：step 1..5182 连续无缺；loss 与 grad_norm **无任何非有限值**；loss 曲线 前 20 步均值 8.1691 → Pilot(371–391) 1.1448 → 1000 附近 0.9081 → 中途 0.7961 → 末 20 步 0.7732。末步学习率为 0（余弦调度在 total_steps 处归零，符合设计）。
- **终态 checkpoint 校验**：`stage=B`、`layout=E2`、`complete=True`、`world_size=16`、`rng_by_rank` 16 份、`trainable` **517** 张量且**全部 FP32 且全部有限**、`optimizer.state` 517、`scheduler.last_epoch=5182`、`diagnostic_only=False`；`manifest/config/stage_a` 三个摘要与 A 阶段及配置一致（stage_a `e83e7e32…`）。
- **跨机复现对照（重要验证）**：第 391 步（Pilot）末步 CE 旧机 4×A40（world=4, micro=1）为 **1.10585**、末 20 步均值 **1.14615**；本机 16×H100（world=16, micro=8）为 **1.10140** 与 **1.14717**，分别相差 0.4% 与 0.09%。在不同硬件、不同卡数、不同微批分组下达到该一致性，说明固定清单、数据顺序、A 初始化与损失归一化均忠实。
- 评测入口 `lmms_model.py` 支持 `--checkpoint` 加载阶段 B 的完整可训练张量（LoRA + TTT）；E0 基线在本机尚未运行。
- 未做：未提交、未推送。


## 2026-09-10 评测内打点：性能指标与分数同批采集

- 用户决定：**只在 benchmark 里测 prefill 延迟、TPOT、峰值显存、缓存占用**（不做受控长度扫描、暂不测 FLOPs，E1 暂不训练）。
- 实现（不改任务/prompt/评分，官方口径不变）：
  - 新增 `src/prefix_ttt/instrument.py`：`ForwardMeter` 用 forward pre/post hook 对模型**每次前向**用 CUDA event 计时（首次=prefill，其余=decode），并读取最终 `past_key_values` 做缓存分解（Full KV / TTT state / Local 窗口）；`cache_buckets` 同时支持混合缓存与 E0 的普通 DynamicCache。
  - `lmms_model.PrefixTTTLava` 增加可选 `measure=<jsonl>`：官方 `Llava.generate_until` 内部按 batch_size=1 **逐样本**调用 `self.model.generate(...)`，因此只包这一层即可每请求记一条；多 rank 时按 rank 分文件。
  - `lmms_run.py` 增加 `--measure`，透传为 model_args。
  - `profile_inference.py` 改为复用 `instrument.cache_buckets`（去重）。
  - 修复过程中的三处实际缺陷：forward hook 未开 `with_kwargs` 导致签名不匹配；输出目录不存在时写入失败；LLaVA 展开后是 3 维 `inputs_embeds`，长度读取需兼容。
- 冒烟验证（E0/MME，limit=4，单卡）：每样本一条记录，`prefill_tokens≈637–645`、prefill 23–25ms（首样本 99ms 含编译）、TPOT 12.3ms、峰值 13.56 GiB、缓存 319–323 MiB。缓存量与解析值一致（32 层 × 2 × 32 头 × ~640 token × 128 × 2B ≈ 320 MiB）。
- 未打点版评测已完成 5/6（e0 三个 + e2 的 mme/pope），仅 e2/gqa 仍在跑；打点版评测已用独立输出根 `/data/shared/weights/prefix-ttt/eval-instrumented/` 启动 5 个任务（GPU 0,1,2,4,5），e2/gqa 待 GPU 3 释放后补跑。
- 已记录的初步分数（未打点版，官方 lmms-eval）：E0 MME Perception 1479.6432 / Cognition 349.2857、POPE Accuracy 0.8548 / F1 0.8397；E2 MME Perception 1415.7322 / Cognition 287.1429。**E2 在 MME 上低于 E0**（Perception −4.3%、Cognition −17.8%），耗时 247s→537s（2.2×），需在总结文档中如实呈现。
- 方法学注意：打点版与未打点版并发运行会争用 CPU/PCIe，绝对延迟偏高；E0 与 E2 在同一条件下测量，横向可比。后续将用一次空闲机器上的单任务复测量化该影响。

## 2026-09-11 评测完成、flash attention 归因与阶段总结文档

- **打点版跑批全部完成**：e0-mme（2374）、e0-pope（9000）、e2-mme（2374）、e2-pope（9000），共 **22748** 条逐请求记录，目录 `/data/shared/weights/prefix-ttt/eval-instrumented/`。e0-gqa 此前被终止，2283 条为半截数据；GQA 无 E2 对照，不进入结论。
- **打点未干扰评测（逐位一致）**：E0 MME 1479.6432 / 349.2857、POPE 0.8548 / 0.8397；E2 MME 1415.7322 / 287.1429、POPE 0.8507 / 0.8363，与未打点版完全相同；官方评测耗时 E0 244.3 s / 505.6 s，E2 531.1 s / 1534.1 s。
- **成本中位数**（同任务、同输入长度）：MME（642 token）E0 prefill 23.0 ms、TPOT 12.8 ms、峰值 13.562 GiB、缓存 321.0 MiB；E2 prefill 101.5 ms、TPOT 64.3 ms、峰值 13.846 GiB、缓存 147.8 MiB（90.3 KV + 46.0 状态 + 11.5 窗口）。POPE（637 token）E0 22.8 / 12.4 / 13.562 GiB / 318.5 MiB，E2 95.8 / 58.7 / 13.846 GiB / 147.1 MiB。
- **并发争用已量化，无需重测**：e2-pope 自身前 3000 条（四任务并发期）与后 3000 条（独占期）对比，prefill 95.9 对 96.0 ms、TPOT 58.8 对 58.8 ms，无可测差异。原计划的"空闲机器单任务复测"据此取消。
- **flash attention 归因**（660 token、单卡、GPU 7 空闲）：E0 与 E2 都是 `attn_implementation=sdpa`，命中的是 PyTorch 自带的 `pytorch_flash::flash_fwd_kernel`；Dao-AILab `flash_attn` 未安装、也非依赖。计入的 FLOPs E0 8777 G 对 E2 8794 G，其中 flash 部分 228.4 G 对 72.7 G（E2 的注意力算力只有三分之一）；kernel launch 次数 prefill 1363 对 7506、decode 1395 对 6411（4.6×），与 TPOT 之比 5.0× 吻合。结论：**慢在 kernel 数量与访存，不在算术量**。
- **绘图脚本重写**：`scripts/plot_metrics.py` 原按已废弃的 `profile_inference` 长度扫描 JSON 编写，数据源改为 benchmark 的逐请求 cost JSONL。用户决定只用跑批数据；benchmark 输入长度全落在 634–663 token，长度曲线做不出来，改为逐任务分布图。产出 `images/{cost_by_task,cache_by_task,score_vs_cost}.png` 与 `metrics-summary.json`。
- **长度基准修正**：早期日志 E0 记的是展开后位置数、E2 记的是纯文本 token 数。本总结统一用缓存字节反推（E0 524288 B/token、E2 147456 B/token），并用 2374 对同序样本交叉验证（长度差均值 −0.001）。`instrument.py` 已改为直接记录 `prefill_positions`；现有日志产生于该改动之前，故无此字段，反推法继续有效。
- **阶段总结已写**：`experiment_summary.md`，含结论「E2 在 640 token 上 prefill 慢 4.4×、TPOT 慢 5.0×、缓存只有 46%，MME Cognition 掉 17.8%」。
- 未做（用户决定）：E1 训练、E2 的 GQA、benchmark 内的 FLOPs 测量。`profile_inference.py` 保留未删。本地改动仍未提交、未推送。

## 2026-09-11 代码清理：需求变更残留审计与重构

- 用户要求检查代码是否干净、有无需求变更遗留的死代码。用两个只读 subagent 做独立审计（符号级死代码审计；CLI/配置/脚本/测试一致性审计），关键结论由本智能体复核。
- **审计结论：需求变更的删除没有留下孤儿函数。** `check_baselines`、`--require-baseline`、`MANIFEST_SHA256`、`expected_sha256`、`check_switch_diagnostic`、`check_pilot_diagnostics` 在 `src/`、`tests/`、`configs/`、`scripts/` 中全部 0 命中（仅本账本散文保留删除记录）；42 个 CLI flag 无失效引用，15 个测试文件无失效 import，5 个脚本的入口与参数全部存在。
- 本次改动（外科手术式，未触碰无关代码）：
  - `sft.py` 删除 `sample_group`：micro-batch 改造后模块内已无调用者，其唯一语义（128 组按 rank 切分）已被 `data_pipeline.micro_batches` 内联覆盖。
  - `transfer.py` 删除随之失效的 `sample_group` 未使用 import。
  - `tests/test_sft.py` 移除该函数的导入与用例；新增 `tests/test_data_pipeline.py`，把同一条划分不变量改测 `micro_batches`（含分块与顺序），覆盖面不减少。
  - `instrument.py` 删除 `ForwardMeter.close()`、`self._handles`、`prefill_input_tokens`/`self.tokens`：前者无调用者（meter 每进程只建一次、无需摘钩子）；后者无人读取，且它的基准不一致正是上一次长度混淆的来源，只保留语义明确的 `prefill_positions`。
  - `scripts/run_eval_instrumented.sh`：默认 RUN 去掉 GQA（E2 未跑、无对照），修正头部注释的 GPU 映射。
  - `README.md`：修正因需求变更而失真的四处——主账本链接指向旧目录、`configs/supervisor/` 与 systemd 样例"保留"的说法（实际已删）、"FLA GPU 数值与性能/正式 A/B runner 尚需实施"的过期状态、"本轮没有提交训练或 benchmark"；并补充两台主机共享存储的现状。
  - 统一新建文件权限（`.py` 644、`.sh` 755），与仓库既有模式一致。
- 验证：`pytest -m 'not gpu'` → 159 passed、17 deselected、0 failed（改动前后同为 0 失败；数量变化可逐项归因：删除基线门槛用例 8 个、`sample_group` 用例 1 个，新增 2 个，Supervisor 3 个由 skip 转为执行）。GPU 冒烟：清理后的 `ForwardMeter` 仍每请求一条记录且含 `prefill_positions`（64 token 用例，缓存 32.5 MiB 与解析值一致）。
- 只报告、未改动（等用户决定）：
  - `profile_inference.py`：整模块无消费者，其长度扫描方案已被"只在 benchmark 内测量"取代；该文件未被 git 跟踪，删除不可恢复，故先询问。
  - `scripts/run_eval_parallel.sh`：未打点版，与打点版协议相同、分数逐位一致，属重复入口。
  - `configs/base.json` 有 28 个键在 `src/` 无读取点（值在代码中硬编码且当前恰好一致，如 `lora.rank/alpha`、`training.*_lr`、`betas`、`grad_clip`、`max_expanded_length`、`generation.*`）；风险是"改了配置但行为不变"，但该文件参与 `config_sha256` 身份，改动会让现有 A/E2 checkpoint 无法恢复，故不动。
  - 与本轮变更无关的历史死代码（保留）：`bridge.audit_checkpoint_headers`、`ops/fla.FLA_COMMIT`、`generation.TransformersHybridCache.tensor_bytes` 全仓无引用，但都早于本轮变更。`sft.py` 中 `if group:` 的 else 分支在本轮改造后不可达（固定清单尾组为 80 条 ≥ 16 个 rank，空组不会出现；且一旦出现会先因流耗尽报错），但该分支改造前就存在，未改。
  - 本账本中指向已删文件的历史句（`migration_from_a40.md` 的链接、`configs/supervisor/*` 的描述）按"保留历史记录"的既定决定未回改。
- 用户决定后执行的删除（2026-09-11）：
  - `src/prefix_ttt/profile_inference.py`（183 行）：无消费者，保留只会让读者以为还有受控长度扫描这条路。
  - `docs/experiments/2026-09-10-prefix-ttt-h100/scripts/run_eval_parallel.sh`：与打点版重复；打点版是超集（同样产出官方分数 + 成本日志），头部注释改为不再引用它。
  - 三个与本轮变更无关的历史死符号：`bridge.audit_checkpoint_headers`（bridge.py 尾部整函数，33 行，含其函数内 import）、`ops/fla.py` 的 `FLA_COMMIT` 常量（FLA 提交哈希仍在 `ops/README.md` 与 `audit_environment.py` 的版本字符串中，未丢失）、`generation.TransformersHybridCache.tensor_bytes` 属性。`cache.HybridCache.tensor_bytes` 因被 `tests/test_cache.py` 使用而保留。
  - 处理后复核：`grep -rn` 三者在 `src/`、`tests/`、`configs/`、`scripts/`、`README.md` 中 0 命中；`pytest -m 'not gpu'` → 159 passed、17 deselected、0 failed；`CUDA_VISIBLE_DEVICES=7 pytest -m gpu` → **17 passed**（82.64s，FLA/Triton 路径在删除 `FLA_COMMIT` 后仍完整可用）。
  - `configs/base.json` **未改动**：按用户决定保留为配方记录，并在阶段总结中说明这些键当前不生效。

## 2026-09-11 SDPA 后端对比、decode kernel 归因与 LoRA 合并实测

- 动机：用户问「是否该关掉 SDPA 的 flash attention 改用朴素实现」「prefix-TTT 的计算后端是什么」「为什么 kernel 调用次数更多」。
- **SDPA 后端对比**（640 token 文本前缀、单卡、预热后 3 次中位数、`sdpa_kernel` 强制选择）：

  | 模型 | 后端 | prefill | TPOT | 峰值显存 | 每请求 kernel |
  | --- | --- | --- | --- | --- | --- |
  | E0 | flash（默认） | 26.8 ms | 17.1 ms | 13.88 GiB | 12146 |
  | E0 | mem-efficient | 25.5 ms | 16.7 ms | 13.88 GiB | 11890 |
  | E0 | math（朴素） | 37.2 ms | 21.3 ms | 14.00 GiB | 15474 |
  | E2 | flash（默认） | 114.4 ms | 81.1 ms | 14.03 GiB | 51322 |
  | E2 | mem-efficient | 114.9 ms | 80.8 ms | 14.03 GiB | 51249 |
  | E2 | math（朴素） | 124.5 ms | 89.2 ms | 14.10 GiB | 56488 |

  结论：朴素实现确实内置于 SDPA（`SDPBackend.MATH`），但关掉 flash 对 E0 的伤害（prefill +39%、TPOT +25%）远大于对 E2 的（+9%、+10%），等于单方面拖慢基线，且 MATH 会物化 $L \times L$ 分数矩阵（8k 上下文每层约 4 GB），与项目要避免的 $O(L^2)$ 代价相悖。**本智能体建议维持默认**（两者同口径接受 PyTorch 的默认选择），待用户确认；若要与使用 eager attention 的历史数字对齐，必须 E0/E2 同时用 math 重跑，现有数据不可混用。
- **prefix-TTT 计算后端**：prefill/训练走 FLA Triton（`fla_prefix` → `chunk_linear_attn` tile 64 或 `chunk_simple_gla` tile 16/32/128，FP32 状态进出）；**decode（t=1）走手写 ATen**（`recurrent_step`：关 autocast、`.float()` 后两个 einsum）；Local-32 与 9 个 anchor 层走 `F.scaled_dot_product_attention`（命中 flash）。`chunk_prefix` 参考实现只用于测试。
- **decode kernel 归因**（`TorchDispatchMode` 计数一个 decode step 的 dispatcher op）：E0 2440 ops（实测 1492 kernel），E2 10215 ops（实测 5953 kernel），比值 4.2×/4.0×。E2 分解：TTT 层自身投影/rotary/mask/o_proj 2208（96/层）、`features()` 1702（74/层）、Local-32 包装 943、Local-32 打包+SDPA 759、`recurrent_step` 851、`readout()` 322、`set_layer` 256，其余 3174 为 9 个 anchor 层 + MLP + 采样。即 **TTT 层每 token 约 306 个 op，而 E0 的普通层只有 76 个**。
  三个结构性原因：① 23 层每层都要做局部注意力、两组特征映射、递推与读出门控；② LoRA 覆盖每层全部 7 个投影（`trainability.py` 校验 `layers × 7`），每个包装后的 Linear = 基座 mm + 2 个小 mm，使每 token 的 mm 由 224 升到 672；③ FP32/bf16 混用导致大量 cast 与逐元素 kernel（`rms_no_affine` 每层调 3 次、`recurrent_step` cast 6 次，`_to_copy` 单步 900 次 vs E0 的 133）。另有 t=1 时无用的训练期簿记（`nonzero`/`cumsum`/`index_copy`，即 cub DeviceSelect/Scan kernel 的来源）。
  量级关系：**TPOT ≈ 每 kernel 约 10 µs × kernel 条数**（E0 benchmark 1395→12.8 ms、E2 6411→64.3 ms），所以 decode 慢的本质是 kernel 条数。
- **LoRA 合并实测**（推理时把 `B@A*scaling` 并入 bf16 基座权重、绕开 LoRA 分支，224 个投影）：prefill 114.0 → 96.4 ms（−15%）、TPOT 80.3 → 61.9 ms（−23%）、每请求 kernel 51323 → 38778（−25%）；8 个 greedy token 逐位不变。这条与「TPOT ∝ kernel 数」互相印证。代价是把增量舍入进 bf16 权重（数值略有变化），当前代码仍保持未合并（`lmms_model.py` 刻意不合并、也不引入 PEFT 的 generate 包装）。
- 未实施：融合 `features()`/RMS 的 cast 链、Local-32 的 decode 专用路径、`recurrent_step` 改用 FLA fused recurrent kernel、跨层批处理、CUDA graph / `torch.compile`。以上均为候选方案，需要用户决定是否投入。
- 证据脚本为一次性 `/tmp` 脚本（已删除），数据来源为 GPU 7 单卡实测；`pytest` 与 GPU 用例未受本轮影响。

## 2026-09-11 推理 kernel 优化：①③ 落地、②④ 实测为负收益（用户批准的四项）

- 用户决定：按 ①→②→③→④ 顺序优化 kernel，不关闭 FA2；用**单个最快的 benchmark**（E2/MME，单卡约 8–9 分钟）作为每步回归指标。
- **方法学更正（重要）**：benchmark 跑批之间的绝对延迟**不可跨跑批比较**。基线 `e2-mme` 是在 4 个任务并发时采集的；晚些时候同一份未优化代码单独跑，prefill 就低了约 6%。因此每项优化的加速改用**同一进程内切换开关的 A/B**（同一张卡、同一前缀、同一进程，4 次中位数），benchmark 跑批只用于正确性（官方分数 + 逐样本回答）。此前"并发无可测影响"的结论据此更正。
- **单进程 A/B 结果**（640 token 文本前缀 + 8 个新 token，GPU 6 独占）：

  | 变体 | prefill | TPOT | kernel/请求 | TPOT 加速 |
  | --- | --- | --- | --- | --- |
  | 基线（当前报告口径） | 115.6 ms | 81.3 ms | 51323 | ×1.00 |
  | ① LoRA 合并 | 94.9 | 61.2 | 38777 | ×1.33 |
  | ③ Local-32 decode 专用路径 | 115.3 | 65.4 | 41983 | ×1.24 |
  | **①+③（最终保留）** | **94.5** | **46.7** | **29438** | **×1.74** |
  | ①+③+④ | 94.4 | 48.9 | 28310 | ×1.67（更慢，已回退） |

- **① LoRA 合并（保留）**：新增 `trainability.merge_lora_weights`（校验 `layers×7` 完整后把 `B@A×scaling` 并入基座权重，并用 base layer 替换包装模块）；`lmms_model` 增 `merge_lora` 开关（默认开），`lmms_run` 增 `--no-merge-lora` 供 A/B。单测 `test_lora_merge_is_exact_and_removes_lora_branches` 校验合并后的权重与 FP32 精确值逐位相等、forward 与未合并一致、LoRA 分支消失。
- **③ Local-32 decode 专用路径（保留）**：`ops/local.py` 新增 `local_attention_decode`：解码时只把新 token 写进 32 槽缓冲，用布尔 mask 表达可见集合，取代 `nonzero`/`cumsum`/`index_copy` 打包（每层 71 → 13 个 kernel）。`hybrid.py` 在 `t==1` 且有缓存时走它。单测 `test_cached_local_attention_decode_matches_the_packed_path`：90 步、跨多个 32 块边界、带空洞，与打包路径逐 token 对齐（1e-4 内），缓存可见尾部逐位相同、计数完全一致。
- **② features/readout 融合（两次尝试，全部回退）**：
  - 换写法（einsum → 批量 matmul、`F.rms_norm` 代替显式链）：kernel 数**一个没少**（被替换的 permute/view 是纯 metadata，不产生 kernel），TPOT 不变 → 回退。
  - `torch.compile` 融合：kernel 29441 → 26129（RMS）/ 25209（连 features+readout）**减少**，但 TPOT 45.8 → **52.7 / 50.6 ms 变慢**。微基准给出原因：解码形状下一次 RMS 的墙钟耗时为显式链 32.9 µs、`F.rms_norm` 29.8 µs、编译版 **47.4 µs**——Inductor 每次调用的 guard/分派开销超过了省下的 6 次小 kernel 启动 → 回退。
- **④ FLA fused recurrent（实测后回退）**：用 `fla.ops.linear_attn.fused_recurrent_linear_attn`（`scale=1.0`、`v/128`、`normalize=False`）替换 `recurrent_step`。等价性实测：输出相对误差 1.07e-7、状态 0（bf16 输入）/ 6e-9（fp32 输入），非法行状态不变。但 kernel 29438 → 28310 的同时 TPOT 46.7 → **48.9 ms 变慢**（FLA 每次调用的 Python 侧预处理约 100 µs×23 层）→ 回退。
- **结论**：在这个尺寸（batch=1、单 token、32×128 张量）下，真正有效的是**减少工作量**（①让 448 个 LoRA 矩阵乘消失）与**用同一个原语折叠整条算子链**（③不引入新的框架调用开销）；把"很多小 kernel 换成一个大的框架 kernel"（Inductor / FLA 调用）是亏的。原因是这些 kernel 的单次开销里，框架的 Python/分派成本大于 GPU 执行成本。
- **MME 正确性回归**（单卡单任务，逐样本回答比对）：

  | 跑批 | MME Perception | MME Cognition | 回答变化 | 说明 |
  | --- | --- | --- | --- | --- |
  | 基线（记录值） | 1415.7322 | 287.1429 | — | 未优化 |
  | ① merge-lora | 1429.4893 (+13.76) | 278.2143 (−8.93) | 22/2374 | bf16 舍入增量导致 |
  | ③ no-merge | 逐位不变 | 逐位不变 | 0/2374 | 数值中立 |
  | ①+③ 最终 | 1429.4893 | 278.2143 | 22/2374（与①单独的 22 条完全相同） | ③ 未额外改变任何回答 |

  最终跑批的成本中位数（该跑批独占机器）：prefill 101.5 → 78.5 ms（×1.29）、TPOT 64.3 → **34.4 ms（×1.87）**、峰值显存 13.845 → 13.515 GiB（LoRA 参数被合并后释放）、缓存占用不变。基线的绝对值含并发影响，故 ×1.87 略高于同进程 A/B 的 ×1.74。
- 新增可复用脚本：`scripts/run_e2_mme_check.sh`（单卡跑 E2/MME + 成本日志，自动与基线比对）、`scripts/compare_eval_runs.py`（分数、逐样本回答、成本中位数三项对比）。
- **待用户决定**：合并 LoRA 是否作为正式口径（会使 E2 的 MME 与 POPE 数字变化，POPE 需重跑）；③ 与 ① 无关，可随时作为默认。
- 未提交、未推送。

## 2026-09-11 LoRA 合并固化为唯一推理路径 + 工程审计与重构

- 用户决定：① 合并作为默认推理路径，但**训练侧仍保持 LoRA 分开**，**不提供导出合并权重的接口**（只在内存中合并一次），并**删除运行时的合并开关与"合并/未合并两份计算"的对照代码**，避免困惑；② 随后做工程审计与重构（分层、设计、去硬编码、去历史包袱）；③ 重构完成后重跑 benchmark。
- **合并固化**：`lmms_model.PrefixTTTLava` 删除 `merge_lora` 参数与字符串解析，加载 LoRA 后无条件调用 `merge_lora_weights`；`lmms_run.py` 删除 `--no-merge-lora`；`self.merged_lora` 计数属性删除（合并函数自身在数量不符时会抛错）。测试改为断言"合并后的 q_proj 权重 = 基座 bf16 权重 + A/B 折出的增量"（E1/E2 两种 layout 都测），并保留 FP32 精确性单测。GPU 冒烟：适配器加载后模型内 `lora_` 模块数 0、可训练参数 0、生成正常。
- **审计**：两个只读 subagent 分别做「分层与重复」与「硬编码与配置」审计，结论要点：
  - 内部依赖图**无环**，`ops/` 零内部依赖、`model → ops` 单向；但有 4 处分层违规：推理 `lmms_model` 依赖训练入口 `sft`、A 阶段 `transfer` 依赖 B 入口 `sft`、`pilot_diagnostic → switch_diagnostic → gpu_regression` 顺带拖入 PIL/HF-Llava，以及 `third_party/llava` 反向 import `prefix_ttt`（后者限制模型层重构，未动）。
  - 重复：`sft.py` 与 `transfer.py` 归一化后完全相同 67 行；`pilot` 与 `switch` 51 行；`transfer.py:57-75` 手抄 `hybrid.py:55-92` 的 attention 前向（审计标记为全仓最高风险重复）；`gpu_smoke.py:161-179` 手抄 `prepare_sample`。
  - 硬编码：anchor 名单双源（`hybrid.py` 常量 vs `base.json`，只有 `transfer.py` 读 config）；`1/128` 六处字面量；Local-32 的 `32` 十七处以上；全局 batch `128` 十一处；`2048` 承担四种不同语义；`-100` 十一处。
  - **关键约束**：`config_sha256 = digest_json(configs/base.json)` 已写入 A/E2 checkpoint，**改动该文件即破坏 resume 身份**，故审计建议不动 `base.json`。
- **已执行的重构（纯搬运 / 命名 / 加断言，行为与数值不变）**：
  - 新增 `src/prefix_ttt/runtime.py`：`distributed_context`（消除 sft/transfer 各 13 行重复的分布式初始化）、`seed_everything`、`rng_state`/`restore_rng`/`gather_rng`、`save_atomic`、`load_trainable`，以及 `SEED`/`LATEST`/`PILOT` 常量。`lmms_model`、`pilot_diagnostic` 改从这里取 `load_trainable`，**推理路径不再 import 训练入口**（实测：`import prefix_ttt.lmms_model` 现在拉起 15 个内部模块、0 个 ijson，且不含 `sft`）。
  - 新增 `src/prefix_ttt/digests.py`：`digest_file`/`digest_json` 从 220 行、依赖 ijson 的 `manifests.py` 搬出。
  - 新增 `src/prefix_ttt/config.py`：`load_config()` 读取配方并**逐项校验 29 个"装饰键"与代码常量一致**（eta、local_block_size、tile_size、max_expanded_length、num_heads/head_dim/feature_dim、LoRA 参数、全部优化器超参、generation 默认值、seed 等），再校验 anchor 名单与"候选层覆盖全部非 anchor 层"；sft/transfer/两个诊断/两个 GPU 工具统一改用它（原来各自 `json.loads` 两次）。新增 `tests/test_config.py` 6 个用例证明漂移会被拒绝。
  - 常量单点化：`ops/__init__.py` 从"无人使用的重导出"改为 `ETA`(2⁻⁷)、`LOCAL_BLOCK_SIZE`(32)、`TILE_SIZE`(64) 的唯一来源；`EFFECTIVE_BATCH_SIZE`、优化器超参（`NEW_MODULE_LR`/`LORA_LR`/`ADAM_BETAS`/`ADAM_EPS`/`MATRIX_WEIGHT_DECAY`/`WARMUP_FRACTION`/`GRAD_CLIP`）、`PILOT_MIN_SAMPLES` 进 `training.py`；`IGNORE_INDEX` 进 `model/labels.py`（避免轻量模块为 `-100` 去 import 3 秒、4445 个模块的 `llava`）；`MAX_EXPANDED_LENGTH`/`PINNED_SHAPE`/`IMAGE_ASPECT_RATIO` 进 `model/bridge.py`；`LORA_*` 进 `trainability.py`；`CONV_TEMPLATE` 进 `data_pipeline.py`；`A_STAGE_SAMPLES` 进 `manifests.py`；`MAX_NEW_TOKENS`/`NUM_BEAMS`/`DO_SAMPLE` 进 `model/generation.py`；`RMS_EPS` 进 `ops/features.py`；`LLM_PROJECTIONS_PER_LAYER` 统一 install 与 merge 的 `×7`。
  - 校验：`ETA` 的替换经逐位比较确认与 `/128` 完全相同（fp32 与 bf16）；CPU 套件 161 → **167 passed / 0 failed**（新增 6 个配置校验用例）；GPU 套件 **17 passed**（82.2s）。
- **故意未做（需要取舍或会改变行为）**：`transfer.py` 手抄的 attention 前向（要求先写等价性测试）；`gpu_smoke.py` / `scheduler.py` / 两个诊断模块的去留；`configs/base.json` 的任何编辑（会破坏 A/E2 的 resume 身份）；`third_party/llava` 的反向依赖。
- **benchmark 重跑（用户第 ③ 条）**：见下条。

## 2026-09-11 benchmark 重跑（合并口径 + 两项优化后）

- 命令：`scripts/run_benchmark_check.sh postrefactor <task>`，单卡 GPU 7 独占，先 MME 后 POPE（不并发，保证成本数字可比）；产物在 `/data/shared/weights/prefix-ttt/eval-optimized/postrefactor/`。
- **与旧基线（未合并 LoRA、未优化）对照**：

  | 任务 | 指标 | 旧 | 新 | 变化 |
  | --- | --- | --- | --- | --- |
  | MME | Perception | 1415.7322 | **1429.4893** | +13.7571 |
  | MME | Cognition | 287.1429 | **278.2143** | −8.9286 |
  | POPE | Accuracy | 0.8507 | **0.8501** | −0.0006 |
  | POPE | F1 | 0.8363 | **0.8357** | −0.0006 |

  - 逐样本回答变化：MME 22/2374（0.9%）、POPE 41/9000（0.46%），全部来自合并 LoRA 的那一次 bf16 舍入；单独测 ③ 时回答逐位不变（见上一条）。
  - 成本中位数：MME prefill 101.472 → 78.917 ms（×1.29）、TPOT 64.276 → 34.660 ms（×1.85）；POPE prefill 95.807 → 78.084 ms（×1.23）、TPOT 58.688 → 34.271 ms（×1.71）；峰值显存 13.845 → 13.51 GiB（LoRA 参数释放）；缓存占用不变。
  - 端到端墙钟：MME 531 → 434 s，POPE 1534 → 1170 s。
- **重构无回归的证据**：把 `postrefactor` 与"优化后、重构前"的 `opt-lora-fastdecode` 跑批逐样本比对——分数逐位相同、**2374/2374 条回答完全一致**、成本中位数差异 <1%（prefill 78.53 → 78.92 ms、TPOT 34.35 → 34.66 ms）。
- 汇总产物同步更新：`eval-final/`（用符号链接把 E0 的旧结果与 E2 的新结果合成一个根，供绘图脚本消费）、重新生成的 `images/*.png` 与 `metrics-summary.json`、`experiment_summary.md` 的 4.1/4.2/4.3/5/8 节数字。
- 仍待用户决定：是否删除 `gpu_smoke.py`、`scheduler.py`（及其两个测试）、`pilot_diagnostic.py`/`switch_diagnostic.py` 等历史模块；是否把 `transfer.py` 手抄的 attention 前向收敛到 `hybrid.py`（需先加等价性测试）。未提交、未推送。

## 2026-09-11 决策落地：删除 gpu_smoke、保留排队器、暂不改 transfer、单 commit 提交并推送

- 用户决定：① **删除** `src/prefix_ttt/gpu_smoke.py`；② **保留** `scheduler.py`（单机 SQLite 排队器，后续可能启用）；③ **暂不改** `transfer.py` 手抄的 attention 前向（先不做等价性测试与抽取）；④ 本轮全部工作合并为**一个 commit** 并推送到远端。
- 删除对象与理由：`gpu_smoke.py`（257 行）自述为 "Disposable real-checkpoint SFT capacity diagnostic; never a formal B run"，是 A40 时代回答"4×48GB 能否放下近 2048 token 的真实 7B 样本"的诊断（旧账本 `docs/experiments/2026-09-09-prefix-ttt/gpu_capacity.md:37-38` 记录了它的单卡与 4 卡 DDP 用法）。澄清一处易混：**本机的多机 smoke 不是它**——两次 smoke（16 卡 E1、3 步 E2）都用 `prefix_ttt.sft --max-steps` 经 `run_multinode.sh` 启动，与正式训练同一个入口；`gpu_smoke.py` 在本机现行流程里没有任何调用点。
- 保留理由：`scheduler.py` 与 DDP 无关（单机 GPU 队列），本机从未启用，但后续可能使用；当前只有 `tests/test_scheduler.py`、`tests/test_supervisor.py` import 它。
- 验证：删除后 CPU 套件 **167 passed / 0 failed**；`src/`、`tests/`、`configs/`、`README.md` 中对 `gpu_smoke` 的引用 0 命中。旧实验目录 `gpu_capacity.md` 中记录其运行命令的句子按"历史记录不回改"的约定保留。
- 提交：单个 commit，作者 `Codex <codex@openai.com>`，推送到 `origin` 的 `h100` 分支。推送前核对：本地与远端 `refs/heads/h100` 均为 `49acbe2`（无分叉，普通快进推送，未使用 force）。

## 2026-09-12 后续工作移交到新的实验目录（全量微调）

- 用户提出新目标：把数据上的 LoRA 微调换成**全量微调**，仍用本机与 h100-1 共 16 卡并行，观察 benchmark
  分数能否提高。
- 该工作已按 `AGENTS.md` 建立新的稳定实验目录 `docs/experiments/2026-09-12-prefix-ttt-fullft/`，
  方案与账本写入该目录（`plan.md`、`ledger.md`）。**本目录（h100）自本条起不再追加全量微调的内容**，
  保持为环境重建 / A-B 训练 / E0-E2 评测 / 推理优化 / 代码审计阶段的完整记录。
- 本目录的既有结论仍然有效并被新实验引用：E0 MME 1479.6432 / 349.2857、POPE 0.8548 / 0.8397；
  E2（合并口径）MME 1429.4893 / 278.2143、POPE 0.8501 / 0.8357；E2 成本 prefill 78.9 ms、
  TPOT 34.7 ms、峰值 13.51 GiB、缓存 147.8 MiB。
- 本阶段未修改任何代码、未启动任何 GPU 任务。本机 8 卡与 h100-1 8 卡在 2026-09-12 01:30 均空闲。

## 2026-09-19 多机能力回迁到 h100 分支（full-ttt → h100）

- 目标：把 `full-ttt` 分支上修好的多节点通信问题与"三台主机一键配置 NFS + Hosts"脚本搬回 `h100` 分支，供后续多机训练使用。本次未启动任何 GPU 任务、未评测、未改动训练数学。
- 已回迁（`h100` 现比 `origin/h100` 多 8 个本地 commit；`origin/h100` 仍为 `60be803`，未推送）：
  - 直接 cherry-pick 4 个脚本提交（`1657fab`、`65061cb`、`36a97fe`、`75f654b`）：新增 `scripts/experiments/prefix_ttt_h100/cluster-hosts.sh` 作为三机名字映射的唯一来源，`nfs-server-export.sh`/`nfs-client-setup.sh` 改为 `SERVER`/`SHARE`/`CLIENTS`/`PERSIST` 参数化且幂等，客户端把挂载写进 `/etc/fstab`（`_netdev`）避免重启后静默写本地盘，README 增补新节点配置流程。
  - cherry-pick `0b6dbf3`：`run_multinode.sh` 支持 elastic rendezvous（`elastic` 作为节点参数），各节点 GPU 数可以不同，静态模式行为不变。
  - 手工移植 `f04d2ef`：`training.all_reduce_gradients` 按 (dtype, device) 分桶归约，替换 SFT 循环里逐张量 `all_reduce` + `isfinite` 同步；未移植该提交里 p32 supervisor 程序的两行改动（该文件在 h100 上由 `7adde0d` 删除）。
  - 手工移植 `6bfa6ac` 的 RNG 部分：`runtime.save_rng/load_rng` 取代集合式 `gather_rng`，保存路径不再需要集合通信，慢节点或异常节点无法再挂住别人的 checkpoint；`sft.py`/`transfer.py` 恢复时先读 sidecar，读不到再回退旧 checkpoint 的 `rng_by_rank` 载荷，因此既有 checkpoint（如 `E2/latest.pt`，实测 16 份 `rng_by_rank`）仍逐 rank 精确恢复。这两处手工移植各带一个新增/扩展的 CPU 测试。
- 未回迁：P32 模型改造（`6bfa6ac` 主体）、`47e70d2`/`d8af8c2` 的 p32 实验改动、`configs/p32.json` 与 `docs/experiments/2026-09-17-prefix-ttt-p32/` 账本；这些仍只在 `full-ttt` 上。
- 验证：4 个脚本与 `README.md` 相对 `full-ttt` 逐字节相同（SHA256 比对），`runtime.py` 与 `full-ttt` 相同；`cluster-hosts.sh` 在合成 hosts 文件上验证"长名已在、短名缺失仍会写入"（原 bug）与二次运行幂等，在 `/etc/hosts` 副本上验证三机条目齐全时报 present；`run_multinode.sh` 的 elastic/static 两条分支用 stub `uv` 打印实参核对；`tests/test_gradient_reduction.py`（2 rank gloo）通过；全量 CPU 套件 201 passed / 84 GPU deselected；6 个 Shell 脚本 `bash -n` 通过。NFS 挂载与真实多机 rendezvous 未在本机执行（需要 root 与三台主机同时在线）。
- AGENTS.md 改为只承载跨实验长期规范：新增"持久化托管一律用 Supervisor，主机缺该程序时用 `pixi global install supervisor` 安装"与"除跨主机同步代码外不得自行 commit"；把 2026-09-17 的"decode 优先、GPU 资源授权"这类单轮目标移出本文件，原文仍完整保留在 `docs/experiments/2026-09-11-prefix-ttt-kernel/ledger.md` 的 2026-09-17 各节；"不使用 CUDA Graph"按用户当轮要求同样从本文件删除，该约束现只作为那一轮的范围记录留在 kernel 账本中；并新增一条"单轮实验目标写在该实验目录"的元规则。
- 推送：用户于 2026-09-19 当轮明确授权推送 `h100`。推送前核对本地与 `origin/h100` 无分叉（远端为 `60be803`），使用普通快进推送，未使用 `--force`、未改写远端历史；`origin/h100` 前移到本轮 8 个多机回迁提交的头部。
