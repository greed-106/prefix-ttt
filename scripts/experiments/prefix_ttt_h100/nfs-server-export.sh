#!/usr/bin/env bash
# Run with sudo on the host that owns the shared directory (h100-3).
# Exports it read-write to every compute node; every client write is squashed to
# one uid/gid, so clients never have to match this host's user ids. Idempotent.
#
#   sudo bash nfs-server-export.sh
#   sudo CLIENTS="h100-1 h100-2" bash nfs-server-export.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SHARE=${SHARE:-/data/shared/weights/prefix-ttt}
CLIENTS=${CLIENTS:-h100-1}
ANON_UID=${ANON_UID:-1001}
ANON_GID=${ANON_GID:-1002}

bash "$SCRIPT_DIR/cluster-hosts.sh"

command -v exportfs >/dev/null || apt-get install -y nfs-kernel-server
mkdir -p "$SHARE"
chmod 2775 "$SHARE"

for client in $CLIENTS; do
    LINE="$SHARE $client(rw,sync,no_subtree_check,all_squash,anonuid=$ANON_UID,anongid=$ANON_GID)"
    if grep -qF "$LINE" /etc/exports; then
        echo "present: $LINE"
    else
        echo "$LINE" >>/etc/exports
        echo "added:   $LINE"
    fi
done
exportfs -ra
showmount -e localhost
