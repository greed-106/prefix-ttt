#!/usr/bin/env bash
# Phase A of the pure Prefix-TTT layout: transfer the teacher's complete attention
# output plus its vision-only contribution into all 32 layers.
#
#   launch_a.sh <node_rank> [extra transfer args...]
#
# Single host (default) is NNODES=1 with all visible GPUs; the same script runs
# unchanged on a second host once one is available:
#   host 0: NNODES=2 NODE_RANK=0 GPUS=8 MASTER_HOST=h100-3 launch_a.sh 0
#   host 1: NNODES=2 NODE_RANK=1 GPUS=8 MASTER_HOST=h100-3 launch_a.sh 1
#
# OUT / GPUS / CUDA_VISIBLE_DEVICES / MASTER_* are all environment variables, so
# nothing here assumes a single machine or a fixed number of cards.
set -uo pipefail
NODE=${1:?node rank}; shift
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
exec "$REPO/scripts/experiments/prefix_ttt_h100/run_multinode.sh" "${NODE_RANK:-$NODE}" \
  prefix_ttt.transfer \
  --config configs/p32.json \
  --manifest artifacts/cpu/fixed_manifest.json \
  --output "${OUT:-/data/shared/weights/prefix-ttt/training/A-P32}" \
  --save-every 100 "$@"
