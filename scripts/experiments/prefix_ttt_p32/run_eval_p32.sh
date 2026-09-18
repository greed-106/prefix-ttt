#!/usr/bin/env bash
# Official lmms-eval MME/POPE for the pure Prefix-TTT model, plus an E0 re-measure
# under identical conditions. Each job writes the official results and a per-request
# cost log (prefill latency, TPOT, peak memory, cache breakdown). Tasks, prompts and
# scoring are untouched.
#
#   RUN="0:p32:mme 1:p32:pope 2:e0:mme 3:e0:pope" run_eval_p32.sh
#
# Serial by default: the previous stage's ledger proved that concurrent runs make
# the latency numbers incomparable, so cost reporting needs one task per GPU.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
HFHOME=$REPO/data/llava-v1.5-assets-v1/benchmarks/hf-home
MODEL=$REPO/data/llava-v1.5-assets-v1/models/llava-1.5-7b-hf-b234b804b114d9e37bb655e11cbbb5f5e971b7a9
CHECKPOINT=${CHECKPOINT:-/data/shared/weights/prefix-ttt/training/P32/latest.pt}
OUT=${OUT:-/data/shared/weights/prefix-ttt/eval-p32}
LOG=${LOG_DIR:-/tmp/eval-p32}

export HF_HOME=$HFHOME HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=$HFHOME/datasets LMMS_EVAL_DATASETS_CACHE=$HFHOME/datasets
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$LOG" "$OUT" "$OUT/cost"
cd "$REPO"

launch() {  # spec = gpu:label:task
    local gpu label task args
    IFS=: read -r gpu label task <<<"$1"
    args=(--pretrained "$MODEL" --tasks "$task" --output-path "$OUT/$label/$task"
          --measure "$OUT/cost/$label-$task.jsonl")
    [ "$label" = p32 ] && args+=(--checkpoint "$CHECKPOINT")
    echo "  GPU $gpu: $label/$task -> $OUT/cost/$label-$task.jsonl"
    CUDA_VISIBLE_DEVICES=$gpu setsid nohup timeout 21600 \
        "$(command -v uv)" run --locked python -m torch.distributed.run --standalone \
        --nproc_per_node=1 -m prefix_ttt.lmms_run "${args[@]}" \
        > "$LOG/$label-$task.log" 2>&1 < /dev/null &
}

SPEC=${RUN:-"0:p32:mme"}
echo "launching evaluations into $OUT"
for spec in $SPEC; do launch "$spec"; done
wait
echo "done"
