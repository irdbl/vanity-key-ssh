#!/bin/bash
# Launch a distributed hunt on vast.ai.
#
#   SUFFIX=++pham COUNT=8 ./scripts/vast_launch.sh
#
# Env knobs:
#   SUFFIX      (required) base64 suffix to hunt
#   COUNT       total GPUs to rent (instances may have several) [4]
#   GPU         gpu_name filter                       [RTX_4090]
#   MAX_DPH     max $/hr per GPU                      [0.50]
#   REPO        public git URL instances clone        [git remote origin]
#   CI          set to 1 for case-insensitive letter matching
#   NTFY_TOPIC  optional ntfy.sh topic for a content-free "found" ping
#   BID         set to 1 for interruptible bids        [off]
#               (memoryless search loses only the in-flight launch if outbid;
#                typically 30-50% cheaper than on-demand)
#
# Requires: pip install vastai; vastai set api-key <key>
# Random search is memoryless, so instances need no coordination — each picks
# a random 128-bit base and searches independently; duplicate work is
# negligible (~0 probability of overlap).
set -euo pipefail
cd "$(dirname "$0")/.."

SUFFIX=${SUFFIX:?set SUFFIX, e.g. SUFFIX=++pham}
COUNT=${COUNT:-4}
GPU=${GPU:-RTX_4090}
MAX_DPH=${MAX_DPH:-0.50}
REPO=${REPO:-$(git remote get-url origin)}
NTFY_TOPIC=${NTFY_TOPIC:-}
CI=${CI:-}   # CI=1: case-insensitive hunt (see README)
IMAGE=${IMAGE:-nvidia/cuda:12.4.1-devel-ubuntu22.04}
BID=${BID:-}
# Instances created by this hunt are recorded here so vast_watch.sh destroys
# only this fleet, never unrelated instances in the account. Appends across
# reruns that top up the fleet; delete it to start a fresh hunt record.
FLEET_FILE=${FLEET_FILE:-.vast_fleet}

ONSTART="apt-get update && apt-get install -y --no-install-recommends git python3 curl ca-certificates && \
git clone --depth 1 '$REPO' /app && cd /app && \
python3 tools/gen_table.py table16.bin --wide && make gpu GPU_ARCH='-arch=native' && \
SUFFIX='$SUFFIX' CI='$CI' NTFY_TOPIC='$NTFY_TOPIC' TABLE=table16.bin bash scripts/entrypoint.sh"

echo "searching offers (COUNT=$COUNT GPUs total, <\$$MAX_DPH/hr per GPU)..."
# Notes from live testing + AUDIT2 F19:
# - verified 4090s are scarce on vast; filter on reliability instead
# - offers go stale fast; without --cancel-unavail a failed schedule silently
#   creates a STOPPED instance husk, so pass it and walk down the offer list
# - pick a CUDA image the host driver supports (cuda_max_good)
# - expected $ is proportional to $/key, not $/hr, so rank offers by
#   dph_total / (num_gpus * rate). Multi-GPU hosts amortize setup and shrink
#   the teardown blast radius: COUNT means total GPUs, not instances
# - entrypoint runs one gpu_vanity process per GPU: require cpu_cores >= GPUs
OFFERS=$(vastai search offers \
    "${GPU:+gpu_name=$GPU} rentable=true reliability>0.95" \
    -o 'dph' --raw | python3 -c "
import json, sys
# Mkeys/s planning rates: RTX 4090 measured (AUDIT2 F18); others scaled by
# INT32 lane count (AUDIT2 F19 table). Unknown GPUs get a conservative rate
# so only genuine bargains outrank the known cards.
RATE = {'RTX 4090': 200, 'RTX 4080': 120, 'RTX 3090': 90, 'RTX 3080': 75,
        'RTX 2080 Ti': 65, 'RTX 3060': 32, 'RTX 5090': 260, 'A100': 95, 'H100': 145}
rows = []
for o in json.load(sys.stdin):
    n = o.get('num_gpus') or 1
    dph = o['dph_total']
    if dph / n > $MAX_DPH or (o.get('cpu_cores_effective') or 0) < n:
        continue
    rate = RATE.get(o.get('gpu_name'), 50)
    rows.append((dph / (n * rate), o['id'], round(dph, 3), o.get('cuda_max_good') or 99, n))
rows.sort()
for r in rows[:40]:
    print(r[1], r[2], r[3], r[4])
")
if [ -z "$OFFERS" ]; then
    echo "no offers matched; raise MAX_DPH or change GPU" >&2
    exit 1
fi

total_dph=0
gpus=0
while read -r id dph cuda ngpu; do
    [ "$gpus" -ge "$COUNT" ] && break
    img=$IMAGE
    if python3 -c "import sys; sys.exit(0 if float('$cuda') < 12.4 else 1)"; then
        img="nvidia/cuda:12.2.2-devel-ubuntu22.04"
    fi
    bid_args=()
    if [ -n "$BID" ]; then bid_args=(--bid "$dph"); fi
    echo "renting offer $id (x$ngpu gpu, \$$dph/hr${BID:+, interruptible bid}, cuda<=$cuda, $img)..."
    if out=$(vastai create instance "$id" \
        --image "$img" \
        --disk 16 \
        --cancel-unavail \
        ${bid_args[@]+"${bid_args[@]}"} \
        --onstart-cmd "$ONSTART" \
        --raw 2>&1) && python3 -c "
import json, sys
d = json.loads('''$out''')
sys.exit(0 if d.get('success') else 1)" 2>/dev/null; then
        echo "$out"
        iid=$(python3 -c "import json,sys; print(json.loads('''$out''').get('new_contract',''))" 2>/dev/null || true)
        if [ -n "$iid" ]; then
            echo "$iid" >>"$FLEET_FILE"
        else
            echo "  WARNING: could not record instance id for offer $id; destroy it manually" >&2
        fi
        gpus=$((gpus + ngpu))
        total_dph=$(python3 -c "print(round($total_dph + $dph, 3))")
    else
        echo "  offer $id unavailable, trying next"
    fi
done <<<"$OFFERS"

if [ "$gpus" -lt "$COUNT" ]; then
    echo "WARNING: only $gpus/$COUNT GPUs scheduled; rerun later or raise MAX_DPH" >&2
fi

echo
echo "recorded $(wc -l <"$FLEET_FILE" 2>/dev/null | tr -d ' ') instance id(s) in $FLEET_FILE"
echo "fleet running at ~\$$total_dph/hr. Monitor with: ./scripts/vast_watch.sh"
echo "IMPORTANT: instances keep billing until destroyed — vast_watch.sh with AUTO_DESTROY=1"
echo "tears the whole fleet down when a match is found."
