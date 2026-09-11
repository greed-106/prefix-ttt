#!/usr/bin/env bash
# The single-GPU evaluations behind this stage's numbers: each job writes its official
# lmms-eval results plus a per-request cost log (prefill latency, TPOT, peak memory,
# cache breakdown). Tasks, prompts and scoring are untouched.
#
#   GPU 1: e0/pope  GPU 2: e0/mme     GPU 4: e2/pope  GPU 5: e2/mme
# GQA is out of scope for this stage: E2 was cancelled, so it has no counterpart.
# Launch subsets with: RUN="1:e0:pope 2:e0:mme" scripts/run_eval_instrumented.sh
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
HFHOME=$REPO/data/llava-v1.5-assets-v1/benchmarks/hf-home
MODEL=$REPO/data/llava-v1.5-assets-v1/models/llava-1.5-7b-hf-b234b804b114d9e37bb655e11cbbb5f5e971b7a9
CHECKPOINT=/data/shared/weights/prefix-ttt/training/E2/latest.pt
OUT=/data/shared/weights/prefix-ttt/eval-instrumented
LOG=${LOG_DIR:-/tmp/eval-instr}

export HF_HOME=$HFHOME HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=$HFHOME/datasets LMMS_EVAL_DATASETS_CACHE=$HFHOME/datasets
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$LOG" "$OUT" "$OUT/cost"
cd "$REPO"

launch() {  # spec = gpu:label:task
    local gpu label task checkpoint="" args
    IFS=: read -r gpu label task <<<"$1"
    [ "$label" = e2 ] && checkpoint=$CHECKPOINT
    args=(--pretrained "$MODEL" --tasks "$task" --output-path "$OUT/$label/$task"
          --measure "$OUT/cost/$label-$task.jsonl")
    [ -n "$checkpoint" ] && args+=(--checkpoint "$checkpoint")
    echo "  GPU $gpu: $label/$task -> $OUT/cost/$label-$task.jsonl"
    CUDA_VISIBLE_DEVICES=$gpu setsid nohup timeout 21600 \
        "$(command -v uv)" run --locked python -m torch.distributed.run --standalone \
        --nproc_per_node=1 -m prefix_ttt.lmms_run "${args[@]}" \
        > "$LOG/$label-$task.log" 2>&1 < /dev/null &
}

SPEC=${RUN:-"1:e0:pope 2:e0:mme 4:e2:pope 5:e2:mme"}
echo "launching instrumented evaluations"
for spec in $SPEC; do launch "$spec"; done
wait
echo "instrumented evaluations finished"
