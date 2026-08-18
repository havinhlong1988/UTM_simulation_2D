#!/usr/bin/env bash
# Phase-0 baseline: run all 5 seeds, copy metrics.json after each into phase0/metrics/.
#
# Usage (from the repo root, with the `utm` conda env ACTIVE):
#     conda activate utm
#     bash phase0/run_seeds.sh
#     python phase0/aggregate.py
#
# `python` must resolve to the utm interpreter (numpy/pandas/matplotlib). Override
# with:  PYTHON=/path/to/python bash phase0/run_seeds.sh
# NOTE: do NOT hard-code a macOS miniforge path here -- this must stay portable.
set -e
PYTHON="${PYTHON:-python}"
HERE="$(cd "$(dirname "$0")" && pwd)"      # phase0/
ROOT="$(cd "$HERE/.." && pwd)"             # repo root
M="$HERE/metrics"
mkdir -p "$M"
cd "$ROOT"
for S in 12345 54321 11111 22222 33333; do
  echo "=== seed $S start $(date +%H:%M:%S) ==="
  "$PYTHON" 09_simulate_agents_2d.py --param-file params/baseline_p0.params --seed "$S" \
      --no-animation --no-html --no-pyvista > "$M/log_seed$S.txt" 2>&1
  cp output/09_phase0_baseline/metrics.json "$M/metrics_seed$S.json"
  echo "=== seed $S done  $(date +%H:%M:%S) -> metrics_seed$S.json ==="
done
echo "ALL_SEEDS_DONE  ->  next: $PYTHON phase0/aggregate.py"
