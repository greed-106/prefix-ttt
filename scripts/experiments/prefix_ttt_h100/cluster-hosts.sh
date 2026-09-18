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

resolved_address() {
    # A name that does not resolve makes getent exit non-zero; under `set -e` a
    # failing command substitution would abort the whole script silently, which is
    # exactly the case this function has to report. Always exit 0 here.
    local answer
    answer=$(getent hosts "$1" 2>/dev/null || true)
    printf '%s' "$answer" | awk 'NR==1{print $1}'
}

for entry in "${NODES[@]}"; do
    read -r address names <<<"$entry"
    # Key on the SHORT name, which is what the training commands use. It is not the
    # first field: the stock 127.0.1.1 line already carries the long hostname, so
    # keying on that calls every entry "present" and adds nothing.
    alias=${names##* }
    if [ "$HOSTS_FILE" = /etc/hosts ]; then
        found=$(resolved_address "$alias")
        if [ "$found" = "$address" ]; then
            echo "present: $entry"
            continue
        fi
        if [ -n "$found" ]; then
            echo "warning: $alias resolves to $found, not $address; fix that line by hand" >&2
            continue
        fi
    elif grep -qw "$alias" "$HOSTS_FILE"; then
        echo "present: $entry (dry run)"
        continue
    fi
    printf '%s  %s\n' "$address" "$names" >>"$HOSTS_FILE"
    echo "added:   $entry"
done

if [ "$HOSTS_FILE" = /etc/hosts ]; then
    echo "--- name resolution ---"
    for entry in "${NODES[@]}"; do
        read -r address names <<<"$entry"
        alias=${names##* }
        [ "$(resolved_address "$alias")" = "$address" ] || {
            echo "error: $alias does not resolve to $address" >&2; exit 1; }
    done
    echo "all names resolve to their pinned addresses"
else
    echo "dry run against $HOSTS_FILE: skipping resolution check"
fi
