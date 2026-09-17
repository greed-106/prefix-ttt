#!/usr/bin/env bash
# The single-GPU evaluations behind the full fine-tuning numbers: the same official
# lmms-eval MME/POPE protocol and the same in-band cost log as the E0/E2 stage, so
# the three arms stay comparable. E0 and E2 are not re-run: their scores are fixed.
#
#   GPU 0: e2full/mme  GPU 1: e2full/pope      (override with RUN="0:e2full:mme")
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
HFHOME=$REPO/data/llava-v1.5-assets-v1/benchmarks/hf-home
MODEL=$REPO/data/llava-v1.5-assets-v1/models/llava-1.5-7b-hf-b234b804b114d9e37bb655e11cbbb5f5e971b7a9
CHECKPOINT=${CHECKPOINT:-/data/shared/weights/prefix-ttt/training/E2-full/latest.pt}
OUT=${OUT:-/data/shared/weights/prefix-ttt/eval-full}
LOG=${LOG_DIR:-/tmp/eval-full}

export HF_HOME=$HFHOME HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=$HFHOME/datasets LMMS_EVAL_DATASETS_CACHE=$HFHOME/datasets
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$LOG" "$OUT" "$OUT/cost"
cd "$REPO"

launch() {  # spec = gpu:label:task
    local gpu label task args
    IFS=: read -r gpu label task <<<"$1"
    args=(--pretrained "$MODEL" --tasks "$task" --output-path "$OUT/$label/$task"
          --checkpoint "$CHECKPOINT" --measure "$OUT/cost/$label-$task.jsonl")
    echo "  GPU $gpu: $label/$task -> $OUT/cost/$label-$task.jsonl"
    CUDA_VISIBLE_DEVICES=$gpu setsid nohup timeout 21600 \
        "$(command -v uv)" run --locked python -m torch.distributed.run --standalone \
        --nproc_per_node=1 -m prefix_ttt.lmms_run "${args[@]}" \
        > "$LOG/$label-$task.log" 2>&1 < /dev/null &
}

SPEC=${RUN:-"0:e2full:mme 1:e2full:pope"}
echo "launching full fine-tuning evaluations (checkpoint=$CHECKPOINT)"
for spec in $SPEC; do launch "$spec"; done
wait
echo "evaluations finished"
