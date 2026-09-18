#!/usr/bin/env bash
# Run with sudo on every compute node. The node that exports the share keeps
# using its local directory; every other node mounts the same path over NFS, so
# the training command looks identical on every node. Idempotent.
#
#   sudo bash nfs-client-setup.sh
#   sudo SERVER=h100-3 SHARE=/data/shared/weights/prefix-ttt bash nfs-client-setup.sh
#
# PERSIST=1 (default) also records the mount in /etc/fstab with _netdev, so a
# rebooted node cannot silently write checkpoints into a local directory; set
# PERSIST=0 to mount for this boot only.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SERVER=${SERVER:-h100-3}
SHARE=${SHARE:-/data/shared/weights/prefix-ttt}
PERSIST=${PERSIST:-1}
SOURCE="$SERVER:$SHARE"

# Every machine gets the same name -> address mapping.
bash "$SCRIPT_DIR/cluster-hosts.sh"

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
    if [ "$PERSIST" = 1 ]; then
        LINE="$SOURCE $SHARE nfs defaults,_netdev 0 0"
        if grep -qF "$SOURCE $SHARE" /etc/fstab; then
            echo "present: fstab entry"
        else
            echo "$LINE" >>/etc/fstab
            echo "added:   fstab entry ($LINE)"
        fi
    fi
fi

findmnt -T "$SHARE"
touch "$SHARE/.probe-$(hostname)"
echo "OK: $SHARE is usable; wrote .probe-$(hostname)"
