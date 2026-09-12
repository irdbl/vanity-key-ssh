#!/bin/bash
# Launch a distributed hunt on vast.ai.
#
#   SUFFIX=++pham COUNT=8 ./scripts/vast_launch.sh
#
# Env knobs:
#   SUFFIX      (required) base64 suffix to hunt
#   COUNT       instances to rent                     [4]
#   GPU         gpu_name filter                       [RTX_4090]
#   MAX_DPH     max $/hr per instance                 [0.50]
#   REPO        public git URL instances clone        [git remote origin]
#   NTFY_TOPIC  optional ntfy.sh topic for a content-free "found" ping
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
IMAGE=${IMAGE:-nvidia/cuda:12.4.1-devel-ubuntu22.04}

ONSTART="apt-get update && apt-get install -y --no-install-recommends git python3 curl ca-certificates && \
git clone --depth 1 '$REPO' /app && cd /app && \
python3 tools/gen_table.py table.bin && make gpu GPU_ARCH='-arch=native' && \
SUFFIX='$SUFFIX' NTFY_TOPIC='$NTFY_TOPIC' bash scripts/entrypoint.sh"

echo "searching offers: $GPU, <\$$MAX_DPH/hr..."
# Notes from live testing:
# - verified 4090s are scarce on vast; filter on reliability instead
# - offers go stale fast; without --cancel-unavail a failed schedule silently
#   creates a STOPPED instance husk, so pass it and walk down the offer list
# - pick a CUDA image the host driver supports (cuda_max_good)
OFFERS=$(vastai search offers \
    "gpu_name=$GPU num_gpus=1 rentable=true reliability>0.95 dph<$MAX_DPH" \
    -o 'dph' --raw | python3 -c "
import json, sys
for o in json.load(sys.stdin):
    print(o['id'], round(o['dph_total'], 3), o.get('cuda_max_good') or 99)
")
if [ -z "$OFFERS" ]; then
    echo "no offers matched; raise MAX_DPH or change GPU" >&2
    exit 1
fi

total_dph=0
rented=0
while read -r id dph cuda; do
    [ "$rented" -ge "$COUNT" ] && break
    img=$IMAGE
    if python3 -c "import sys; sys.exit(0 if float('$cuda') < 12.4 else 1)"; then
        img="nvidia/cuda:12.2.2-devel-ubuntu22.04"
    fi
    echo "renting offer $id (\$$dph/hr, cuda<=$cuda, $img)..."
    if out=$(vastai create instance "$id" \
        --image "$img" \
        --disk 16 \
        --cancel-unavail \
        --onstart-cmd "$ONSTART" \
        --raw 2>&1) && python3 -c "
import json, sys
d = json.loads('''$out''')
sys.exit(0 if d.get('success') else 1)" 2>/dev/null; then
        echo "$out"
        rented=$((rented + 1))
        total_dph=$(python3 -c "print(round($total_dph + $dph, 3))")
    else
        echo "  offer $id unavailable, trying next"
    fi
done <<<"$OFFERS"

if [ "$rented" -lt "$COUNT" ]; then
    echo "WARNING: only $rented/$COUNT instances scheduled; rerun later or raise MAX_DPH" >&2
fi

echo
echo "fleet running at ~\$$total_dph/hr. Monitor with: ./scripts/vast_watch.sh"
echo "IMPORTANT: instances keep billing until destroyed — vast_watch.sh with AUTO_DESTROY=1"
echo "tears the whole fleet down when a match is found."
