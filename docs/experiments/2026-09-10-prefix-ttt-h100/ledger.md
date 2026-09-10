# Prefix-TTT H100 主机主账本

本目录 `docs/experiments/2026-09-10-prefix-ttt-h100/` 是本项目在**本机 `cucloud-server3`**（用户 `mjyang`，8×H100 80GB，裸金属主机）的稳定实验目录，启动日期 2026-09-10。本机的后续记录持续写入本账本，不按自然日另建目录。

前一阶段记录位于 `docs/experiments/2026-09-09-prefix-ttt/`，对应**另一台机器上的早期环境**（用户 `ymj`，四张 A40 48GB，路径 `/data/ymj/code/llm/prefix-ttt`）。那份账本保持原样作为历史，不追加本机内容；2026-09-10 迁移时已将本机环境审计从旧目录移至本目录 `environment.lock.json`，旧目录未被修改。两台机器是各自独立的检出，`data/`、`artifacts/`、`preference/` 不共享。

## 当前状态

本容器 Python 环境已按 `pyproject.toml` / `uv.lock` 安装并验证通过，GPU 算子验收 17 项通过。`data/llava-v1.5-assets-v1/`（69G）与 A40 迁移副本 `artifacts/migrations/from-a40-20260910-step1075/` 已于 2026-09-10 20:24–20:36 到位并完成哈希/数量校验（见「迁移资产到位与校验」）。仍缺：`preference/` 参考仓库、Supervisor（持久托管）、本机 GPU 授权与 `configs/` 更新。按迁移指南，当前代码**不能不经适配**直接在 8 卡上恢复 E2 训练。

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
