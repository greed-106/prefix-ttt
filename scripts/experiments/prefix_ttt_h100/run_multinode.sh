#!/usr/bin/env bash
# Launch one node of a multi-node run. Host names resolve through /etc/hosts,
# so no IP appears in the command line and no ssh alias is involved.
#
#   scripts/experiments/prefix_ttt_h100/run_multinode.sh <node_rank> <module> [module args...]
#   scripts/experiments/prefix_ttt_h100/run_multinode.sh 0 prefix_ttt.transfer --config configs/base.json ...
#   scripts/experiments/prefix_ttt_h100/run_multinode.sh 1 prefix_ttt.sft --layout E2 --output ... --resume ...
#
# Nodes with EQUAL GPU counts use the static form: pass the node rank. Static mode
# computes the world size locally as nnodes * nproc_per_node, so every node must
# contribute the same number and the ranks have to be handed out explicitly.
#
# Nodes with DIFFERENT GPU counts pass the literal `elastic` instead of a rank:
# the c10d rendezvous sums each node's local size, which was measured to work for
# 1/3/4 ranks per node (WORLD=8). No node_rank is given, so the master must start
# first to become rank 0 (the rank that writes checkpoints).
#
#   launcher: NODE_RANK=elastic NNODES=3 GPUS=<this node's gpus> bash run_multinode.sh elastic ...
#
# Override with MASTER_HOST, MASTER_PORT, NNODES, GPUS, RDZV_ID.
set -euo pipefail

NODE_RANK=${1:?usage: run_multinode.sh <node_rank|elastic> <module> [args...]}
MODULE=${2:?usage: run_multinode.sh <node_rank|elastic> <module> [args...]}
shift 2

MASTER_HOST=${MASTER_HOST:-h100-3}
MASTER_PORT=${MASTER_PORT:-29500}
NNODES=${NNODES:-2}
GPUS=${GPUS:-8}
RDZV_ID=${RDZV_ID:-prefix-ttt}

getent hosts "$MASTER_HOST" >/dev/null || {
    echo "error: $MASTER_HOST does not resolve; add it to /etc/hosts (see scripts/experiments/prefix_ttt_h100/nfs-client-setup.sh)" >&2
    exit 1
}

if [ "$NODE_RANK" = elastic ]; then
    echo "rendezvous=elastic  master=$MASTER_HOST:$MASTER_PORT  nnodes=$NNODES  local_gpus=$GPUS  module=$MODULE" >&2
    exec uv run --locked python -m torch.distributed.run \
        --rdzv_backend=c10d --rdzv_endpoint="$MASTER_HOST:$MASTER_PORT" --rdzv_id="$RDZV_ID" \
        --nnodes="$NNODES:$NNODES" --nproc_per_node="$GPUS" --max-restarts=0 \
        -m "$MODULE" "$@"
fi

echo "node_rank=$NODE_RANK  master=$MASTER_HOST:$MASTER_PORT  module=$MODULE" >&2
exec uv run --locked python -m torch.distributed.run \
    --nnodes="$NNODES" --nproc_per_node="$GPUS" --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_HOST" --master_port="$MASTER_PORT" \
    -m "$MODULE" "$@"
