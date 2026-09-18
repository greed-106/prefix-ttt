#!/usr/bin/env bash
# Phase B of the pure Prefix-TTT layout: train the feature maps, gates and LoRA of
# all 32 layers with the original answers (CE) plus the frozen teacher's KL.
#
#   launch_b.sh <node_rank> [extra sft args...]
#
# Single host (default) is NNODES=1 with all visible GPUs; the same script runs
# unchanged on a second host:
#   host 0: NNODES=2 NODE_RANK=0 GPUS=8 MASTER_HOST=h100-3 launch_b.sh 0
#   host 1: NNODES=2 NODE_RANK=1 GPUS=8 MASTER_HOST=h100-3 launch_b.sh 1
#
# OUT / STAGE_A / GPUS / CUDA_VISIBLE_DEVICES / MASTER_* are environment variables.
set -uo pipefail
NODE=${1:?node rank}; shift
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
exec "$REPO/scripts/experiments/prefix_ttt_h100/run_multinode.sh" "${NODE_RANK:-$NODE}" \
  prefix_ttt.sft \
  --config configs/p32.json \
  --manifest artifacts/cpu/fixed_manifest.json \
  --layout P32 --trainable lora \
  --stage-a-checkpoint "${STAGE_A:-/data/shared/weights/prefix-ttt/training/A-P32/latest.pt}" \
  --output "${OUT:-/data/shared/weights/prefix-ttt/training/P32}" \
  --save-every 250 "$@"
