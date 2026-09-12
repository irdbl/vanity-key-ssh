#!/bin/bash
# Poll all vast.ai instances for a FOUND line; print it and (optionally)
# destroy the fleet.
#
#   ./scripts/vast_watch.sh                # watch only
#   AUTO_DESTROY=1 ./scripts/vast_watch.sh # destroy all instances on find
set -euo pipefail

INTERVAL=${INTERVAL:-60}
AUTO_DESTROY=${AUTO_DESTROY:-0}

ids() {
    vastai show instances --raw | python3 -c "
import json, sys
for i in json.load(sys.stdin):
    print(i['id'])
"
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
