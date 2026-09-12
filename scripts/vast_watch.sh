#!/bin/bash
# Poll all vast.ai instances for a FOUND line; print it and (optionally)
# destroy the fleet.
#
#   ./scripts/vast_watch.sh                # watch only
#   AUTO_DESTROY=1 ./scripts/vast_watch.sh # destroy all instances on find
set -euo pipefail
cd "$(dirname "$0")/.."

INTERVAL=${INTERVAL:-60}
AUTO_DESTROY=${AUTO_DESTROY:-0}
# Only the instances this hunt created (recorded by vast_launch.sh). Monitoring
# and destruction are restricted to this list so unrelated account instances are
# never touched.
FLEET_FILE=${FLEET_FILE:-.vast_fleet}

if [ ! -f "$FLEET_FILE" ]; then
    echo "no fleet file ($FLEET_FILE); run vast_launch.sh first, or set FLEET_FILE" >&2
    echo "refusing to operate on the whole account" >&2
    exit 1
fi

ids() {
    grep -E '^[0-9]+$' "$FLEET_FILE" | sort -u
}

while true; do
    found=""
    for id in $(ids); do
        log=$(vastai logs "$id" --tail 400 2>/dev/null || true)
        if grep -q '^FOUND' <<<"$log"; then
            echo "=== instance $id ==="
            grep '^FOUND\|^ssh-ed25519' <<<"$log" | sort -u
            found=$id
        fi
    done
    if [ -n "$found" ]; then
        echo
        echo "Recover the private key locally:"
        echo "  python3 tools/keytool.py privkey <seed-hex> ~/.ssh/id_ed25519_vanity"
        if [ "$AUTO_DESTROY" = 1 ]; then
            for id in $(ids); do
                echo "destroying instance $id"
                vastai destroy instance "$id"
            done
        else
            echo "Fleet still running/BILLING. Destroy with:"
            echo "  vastai show instances; vastai destroy instance <id>"
        fi
        exit 0
    fi
    date "+%H:%M:%S no match yet; sleeping ${INTERVAL}s"
    sleep "$INTERVAL"
done
