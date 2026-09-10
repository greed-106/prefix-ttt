#!/usr/bin/env bash
# Run with sudo on every compute node. The node that exports the share keeps
# using its local directory; every other node mounts the same path over NFS, so
# the training command looks identical on every node. Idempotent.
set -euo pipefail

SERVER=h100-3
SHARE=/data/shared/weights/prefix-ttt
SOURCE="$SERVER:$SHARE"

# Every machine gets the same name -> address mapping.
grep -qw h100-3 /etc/hosts || echo "172.18.1.184  cucloud-server3 h100-3" >> /etc/hosts
grep -qw h100-1 /etc/hosts || echo "172.18.1.156  cucloud-server1 h100-1" >> /etc/hosts

exports_here() {
    command -v exportfs >/dev/null && exportfs -s 2>/dev/null | grep -qF "$SHARE"
}

if mountpoint -q "$SHARE"; then
    current=$(findmnt -no SOURCE --mountpoint "$SHARE")
    if [ "$current" != "$SOURCE" ]; then
        echo "error: $SHARE is already mounted from '$current', expected '$SOURCE'" >&2
        exit 1
    fi
    echo "$SHARE is already mounted from $current"
elif exports_here; then
    echo "$SHARE is exported by this host; using the local directory, nothing to mount"
else
    if [ -d "$SHARE" ] && [ -n "$(ls -A "$SHARE")" ]; then
        echo "error: $SHARE already holds local data; mounting over it would hide that data" >&2
        echo "       move or delete that content, then rerun" >&2
        exit 1
    fi
    command -v mount.nfs >/dev/null || apt-get install -y nfs-common
    mkdir -p "$SHARE"
    mount -t nfs "$SOURCE" "$SHARE"
    echo "mounted $SOURCE at $SHARE"
fi

findmnt -T "$SHARE"
touch "$SHARE/.probe-$(hostname)"
echo "OK: $SHARE is usable; wrote .probe-$(hostname)"
