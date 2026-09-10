#!/usr/bin/env bash
# Launch one node of a multi-node run. Host names resolve through /etc/hosts,
# so no IP appears in the command line and no ssh alias is involved.
#
#   scripts/run_multinode.sh <node_rank> <module> [module args...]
#   scripts/run_multinode.sh 0 prefix_ttt.transfer --config configs/base.json ...
#   scripts/run_multinode.sh 1 prefix_ttt.sft --layout E2 --output ... --resume ...
#
# Override with MASTER_HOST, MASTER_PORT, NNODES, GPUS.
set -euo pipefail

NODE_RANK=${1:?usage: run_multinode.sh <node_rank> <module> [args...]}
MODULE=${2:?usage: run_multinode.sh <node_rank> <module> [args...]}
shift 2

MASTER_HOST=${MASTER_HOST:-h100-3}
MASTER_PORT=${MASTER_PORT:-29500}
NNODES=${NNODES:-2}
GPUS=${GPUS:-8}

getent hosts "$MASTER_HOST" >/dev/null || {
    echo "error: $MASTER_HOST does not resolve; add it to /etc/hosts (see scripts/nfs-client-setup.sh)" >&2
    exit 1
}

echo "node_rank=$NODE_RANK  master=$MASTER_HOST:$MASTER_PORT  module=$MODULE" >&2
exec uv run --locked python -m torch.distributed.run \
    --nnodes="$NNODES" --nproc_per_node="$GPUS" --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_HOST" --master_port="$MASTER_PORT" \
    -m "$MODULE" "$@"
