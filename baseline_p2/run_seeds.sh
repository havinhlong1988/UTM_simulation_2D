#!/usr/bin/env bash
# baseline_p2: FROZEN baseline = Phase-0 baseline + A6 demand-capacity balancing
# (DCB_MODE, share=1.5). Run all 5 D2 seeds, copy metrics.json into baseline_p2/metrics/.
#   PYTHON=~/anaconda3/envs/utm/bin/python bash baseline_p2/run_seeds.sh
#   python baseline_p2/aggregate.py
set -e
PYTHON="${PYTHON:-python}"
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$HERE/.." && pwd)"
M="$HERE/metrics"; mkdir -p "$M"; cd "$ROOT"
for S in 12345 54321 11111 22222 33333; do
  echo "=== seed $S start $(date +%H:%M:%S) ==="
  "$PYTHON" 09_simulate_agents_2d.py --param-file params/baseline_p2.params --seed "$S" \
      --no-animation --no-html --no-pyvista > "$M/log_seed$S.txt" 2>&1
  cp output/09_baseline_p2/metrics.json "$M/metrics_seed$S.json"
  echo "=== seed $S done  $(date +%H:%M:%S) ==="
done
echo "ALL_SEEDS_DONE -> next: $PYTHON baseline_p2/aggregate.py"
