#!/usr/bin/env bash
# Run with sudo on the host that owns the shared directory (h100-3).
# Exports it read-write to the training node; every client write is squashed to
# one uid/gid, so clients never have to match this host's user ids.
set -euo pipefail

SHARE=/data/shared/weights/prefix-ttt
CLIENT=h100-1
ANON_UID=1001
ANON_GID=1002

# Both machines get the same name -> address mapping.
grep -qw h100-3 /etc/hosts || echo "172.18.1.184  cucloud-server3 h100-3" >> /etc/hosts
grep -qw h100-1 /etc/hosts || echo "172.18.1.156  cucloud-server1 h100-1" >> /etc/hosts

command -v exportfs >/dev/null || apt-get install -y nfs-kernel-server
mkdir -p "$SHARE"
chmod 2775 "$SHARE"

LINE="$SHARE $CLIENT(rw,sync,no_subtree_check,all_squash,anonuid=$ANON_UID,anongid=$ANON_GID)"
grep -qF "$LINE" /etc/exports || echo "$LINE" >> /etc/exports
exportfs -ra
showmount -e localhost
