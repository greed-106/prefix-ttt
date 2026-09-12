#!/usr/bin/env bash
# Launch one node of the full fine-tuning B stage. Usage: launch_b.sh <0|1> [extra sft args...]
set -uo pipefail
NODE=${1:?node rank}; shift
REPO=/data/mjyang/code/llm/prefix-ttt
SCRIPT=$REPO/docs/experiments/2026-09-10-prefix-ttt-h100/scripts/run_multinode.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"
exec "$SCRIPT" "$NODE" prefix_ttt.sft \
  --config configs/full.json \
  --manifest artifacts/cpu/fixed_manifest.json \
  --layout E2 --trainable full \
  --stage-a-checkpoint /data/shared/weights/prefix-ttt/training/A-full/latest.pt \
  --output /data/shared/weights/prefix-ttt/training/E2-full \
  --save-every 250 "$@"
