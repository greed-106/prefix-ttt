#!/usr/bin/env bash
# Write the same name -> address map on every compute node, so NFS mounts and
# torchrun rendezvous can use host names instead of addresses. Idempotent.
#
#   sudo bash cluster-hosts.sh
#   HOSTS_FILE=/tmp/hosts.test bash cluster-hosts.sh    # dry run against a copy
#
# Keep the list here as the single source of truth: the NFS export and client
# scripts call this file, so a new node only has to be added once.
set -euo pipefail

HOSTS_FILE=${HOSTS_FILE:-/etc/hosts}
NODES=(
    "172.18.1.184 cucloud-server3 h100-3"
    "172.18.1.156 cucloud-server1 h100-1"
    "172.18.1.73  cucloud-server2 h100-2"
)

[ -w "$HOSTS_FILE" ] || { echo "error: $HOSTS_FILE is not writable; run with sudo" >&2; exit 1; }

for entry in "${NODES[@]}"; do
    read -r address names <<<"$entry"
    marker=${names%% *}
    if grep -qw "$marker" "$HOSTS_FILE"; then
        echo "present: $entry"
        continue
    fi
    printf '%s  %s\n' "$address" "$names" >>"$HOSTS_FILE"
    echo "added:   $entry"
done

if [ "$HOSTS_FILE" = /etc/hosts ]; then
    echo "--- name resolution ---"
    for entry in "${NODES[@]}"; do
        read -r address names <<<"$entry"
        getent hosts "${names##* }" >/dev/null || { echo "error: ${names##* } does not resolve" >&2; exit 1; }
    done
    echo "all names resolve"
else
    echo "dry run against $HOSTS_FILE: skipping resolution check"
fi
