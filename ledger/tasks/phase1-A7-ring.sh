#!/usr/bin/env bash
# phase1-A7-ring: A/B the ring-entry metering (RING_METER) over the 5 D2 seeds vs
# the frozen Phase-0 baseline. RING_METER is OFF by default in the code, so the
# ONLY difference from baseline_p0 is params/phase1_A7_ring.params. Judges the
# revised gates (progress.md §5): G1' min_same_lane_gap_m >= 80m is the target.
# Exit 0 = ran; verdict (KEEP/KILL) is printed and journaled, not an exit code.
set -euo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
M="phase1/A7/metrics"; mkdir -p "$M"
for S in 12345 54321 11111 22222 33333; do
  echo "[A7] seed $S $(date +%H:%M:%S)"
  "$PYTHON" 09_simulate_agents_2d.py --param-file params/phase1_A7_ring.params \
      --seed "$S" --no-animation --no-html --no-pyvista > "$M/log_seed$S.txt" 2>&1
  cp output/09_phase1_A7_ring/metrics.json "$M/metrics_seed$S.json"
done
"$PYTHON" - <<'PY'
import json, glob, statistics as st
def med(xs): return st.median(xs)
A=[json.load(open(f)) for f in sorted(glob.glob("phase1/A7/metrics/metrics_seed*.json"))]
B=json.load(open("phase0/baseline_summary.json"))
def col(rows,k): return [r.get(k) for r in rows]
keys=["min_same_lane_gap_m","n_completed","conflict_samples","n_battery_dead",
      "total_hold_minutes","peak_backlog","mean_wait_for_leg_s","n_reroutes"]
print(f"{'metric':22}{'A7 median':>14}{'A7 range':>20}")
gaps=col(A,"min_same_lane_gap_m")
for k in keys:
    xs=[x for x in col(A,k) if isinstance(x,(int,float))]
    if xs: print(f"{k:22}{med(xs):>14.2f}{f'{min(xs):.1f}..{max(xs):.1f}':>20}")
gl=any(r.get("gridlock") for r in A)
print("gridlock any:", gl)
g1=min(gaps)>=80.0
print(f"\nG1' min_same_lane_gap >=80 (all seeds): min={min(gaps):.1f} -> {'PASS' if g1 else 'FAIL'}")
print(f"G2 no gridlock: {'PASS' if not gl else 'FAIL'}")
PY
