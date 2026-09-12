#!/bin/bash
# Runs one gpu_vanity process per GPU and waits for the first FOUND.
# The seed only ever appears in this machine's stdout/logs — pull it with
# `vastai logs <id>`. If NTFY_TOPIC is set, a content-free ping (no seed,
# no suffix details beyond what you chose to put in the topic name) is sent
# so you know to fetch logs and tear the fleet down.
set -u
cd "$(dirname "$0")/.."
SUFFIX=${SUFFIX:?set SUFFIX, e.g. SUFFIX=++pham}
TABLE=${TABLE:-table.bin}
CI=${CI:-}   # CI=1: case-insensitive letter matching (multi-target F2)
OUT=out
mkdir -p "$OUT"

NGPU=$(nvidia-smi -L | wc -l)
echo "starting $NGPU miner(s) for suffix '$SUFFIX'"
pids=()
for i in $(seq 0 $((NGPU - 1))); do
    # Process substitution (not a pipe) so $! is the miner's PID, not tee's:
    # killing tee would leave the GPU searcher running until a later pipe write.
    # stdbuf execs gpu_vanity, so the recorded PID is the miner itself.
    stdbuf -oL ./bin/gpu_vanity --suffix "$SUFFIX" ${CI:+--ci} --table "$TABLE" --device "$i" \
        > >(stdbuf -oL tee "$OUT/gpu_$i.log") 2>&1 &
    pids+=($!)
done

while sleep 5; do
    if grep -qs '^FOUND' "$OUT"/gpu_*.log; then
        kill "${pids[@]}" 2>/dev/null
        wait "${pids[@]}" 2>/dev/null
        break
    fi
    alive=0
    for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
    if [ "$alive" = 0 ]; then
        echo "all miners exited without FOUND (crash?)" >&2
        exit 1
    fi
done

grep -hs '^FOUND\|^ssh-ed25519' "$OUT"/gpu_*.log
if [ -n "${NTFY_TOPIC:-}" ]; then
    curl -s -m 10 -d "vanity hunt: match found, fetch logs and destroy fleet" \
        "https://ntfy.sh/$NTFY_TOPIC" >/dev/null || true
fi

# Keep republishing the result so it stays in recent log output until the
# instance is destroyed.
while true; do
    echo "=== RESULT (destroy this instance after saving the seed) ==="
    grep -hs '^FOUND\|^ssh-ed25519' "$OUT"/gpu_*.log
    sleep 300
done
