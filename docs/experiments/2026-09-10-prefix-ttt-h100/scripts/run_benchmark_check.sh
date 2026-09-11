#!/usr/bin/env bash
# One-run regression check: evaluate E2 on one benchmark with the cost log, then
# compare score, per-sample responses and per-request cost against the recorded
# baseline in the same command.
#
#   scripts/run_benchmark_check.sh <tag> [task] [extra prefix_ttt.lmms_run args...]
#   GPU=6 scripts/run_benchmark_check.sh opt-1-3 pope
#
# MME is the fastest task (2374 requests, ~9 minutes) and is the default; POPE
# takes ~26 minutes. Task, prompts and scoring come from the pinned lmms-eval.
set -euo pipefail

TAG=${1:?usage: run_benchmark_check.sh <tag> [task] [extra args...]}
TASK=${2:-mme}
shift 2
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
HFHOME=$REPO/data/llava-v1.5-assets-v1/benchmarks/hf-home
MODEL=$REPO/data/llava-v1.5-assets-v1/models/llava-1.5-7b-hf-b234b804b114d9e37bb655e11cbbb5f5e971b7a9
CHECKPOINT=/data/shared/weights/prefix-ttt/training/E2/latest.pt
BASELINE=/data/shared/weights/prefix-ttt/eval-instrumented
OUT=/data/shared/weights/prefix-ttt/eval-optimized/$TAG
LOG=${LOG_DIR:-/tmp/eval-optimized}
GPU=${GPU:-7}

export HF_HOME=$HFHOME HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=$HFHOME/datasets LMMS_EVAL_DATASETS_CACHE=$HFHOME/datasets
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$LOG" "$OUT/cost"
cd "$REPO"

echo "== running e2/$TASK for tag '$TAG' on GPU $GPU -> $OUT"
CUDA_VISIBLE_DEVICES=$GPU "$(command -v uv)" run --locked python -m torch.distributed.run \
    --standalone --nproc_per_node=1 -m prefix_ttt.lmms_run \
    --pretrained "$MODEL" --checkpoint "$CHECKPOINT" --tasks "$TASK" \
    --output-path "$OUT/e2/$TASK" --measure "$OUT/cost/e2-$TASK.jsonl" "$@" \
    > "$LOG/$TAG-$TASK.log" 2>&1
echo "== evaluation finished; comparing with the baseline"
"$(command -v uv)" run --locked python \
    docs/experiments/2026-09-10-prefix-ttt-h100/scripts/compare_eval_runs.py \
    --baseline "$BASELINE" --candidate "$OUT" --label e2 --task "$TASK"
