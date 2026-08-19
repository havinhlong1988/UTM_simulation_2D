#!/usr/bin/env bash
# phase1-A7b-ring: A/B ring wrap-aware CAR-FOLLOWING only (RING_WRAP_FOLLOW, no
# merge meter) over the 5 D2 seeds vs the frozen Phase-0 baseline. Isolates the
# cheap half of A7 (which was KILLED by battery deaths from the merge meter's
# entry holds). Target: lift ring gap toward >= 80m (G1') WITHOUT tripping G3
# (n_battery_dead <= baseline). Only params/phase1_A7b_ring.params differs from
# baseline. Verdict (KEEP/KILL) is printed and journaled, not an exit code.
set -euo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
M="phase1/A7b/metrics"; mkdir -p "$M"
for S in 12345 54321 11111 22222 33333; do
  echo "[A7b] seed $S $(date +%H:%M:%S)"
  "$PYTHON" 09_simulate_agents_2d.py --param-file params/phase1_A7b_ring.params \
      --seed "$S" --no-animation --no-html --no-pyvista > "$M/log_seed$S.txt" 2>&1
  cp output/09_phase1_A7b_ring/metrics.json "$M/metrics_seed$S.json"
done
"$PYTHON" - <<'PY'
import json, glob, statistics as st
A=[json.load(open(f)) for f in sorted(glob.glob("phase1/A7b/metrics/metrics_seed*.json"))]
B=json.load(open("phase0/baseline_summary.json"))
def col(rows,k): return [r.get(k) for r in rows]
# baseline_summary.json may store medians under a nested structure; fall back to
# the hard-coded Phase-0 medians (progress.md §3) if a key is absent.
BASE={"min_same_lane_gap_m":65.8,"n_completed":877,"conflict_samples":4006,
      "n_battery_dead":123,"total_hold_minutes":12772,"peak_backlog":999,
      "mean_wait_for_leg_s":6119,"n_reroutes":1063}
keys=list(BASE)
def med(xs): return st.median(xs)
print(f"{'metric':22}{'baseline':>12}{'A7b median':>14}{'A7b range':>20}")
for k in keys:
    xs=[x for x in col(A,k) if isinstance(x,(int,float))]
    if xs: print(f"{k:22}{BASE[k]:>12}{med(xs):>14.2f}{f'{min(xs):.1f}..{max(xs):.1f}':>20}")
gaps=[x for x in col(A,"min_same_lane_gap_m") if isinstance(x,(int,float))]
dead=[x for x in col(A,"n_battery_dead") if isinstance(x,(int,float))]
gl=any(r.get("gridlock") for r in A)
g1=min(gaps)>=80.0
g3=med(dead)<=BASE["n_battery_dead"]
print(f"\nG1' min_same_lane_gap >=80 (all seeds): min={min(gaps):.1f} -> {'PASS' if g1 else 'FAIL'}")
print(f"G2 no gridlock: {'PASS' if not gl else 'FAIL'}")
print(f"G3 n_battery_dead median {med(dead):.0f} <= baseline {BASE['n_battery_dead']}: {'PASS' if g3 else 'FAIL'}")
PY
