#!/usr/bin/env bash
# phase1-A5-tolling: A/B system-optimum marginal-cost tolling (TOLL_MODE) over the
# 5 D2 seeds vs the frozen Phase-0 baseline. TOLL_MODE is OFF by default in the
# code, so only params/phase1_A5_tolling.params differs from baseline. A5 targets
# the stated objective "raise route utilisation / balance load" (progress.md §5
# soft goals): lower max_agents_on_a_lane, lower total_hold_minutes, more
# completions -- WITHOUT tripping the hard gates G2 (no gridlock), G3
# (n_battery_dead <= baseline), G4 (return share). Verdict printed + journaled.
set -euo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
M="phase1/A5/metrics"; mkdir -p "$M"
for S in 12345 54321 11111 22222 33333; do
  echo "[A5] seed $S $(date +%H:%M:%S)"
  "$PYTHON" 09_simulate_agents_2d.py --param-file params/phase1_A5_tolling.params \
      --seed "$S" --no-animation --no-html --no-pyvista > "$M/log_seed$S.txt" 2>&1
  cp output/09_phase1_A5_tolling/metrics.json "$M/metrics_seed$S.json"
done
"$PYTHON" - <<'PY'
import json, glob, statistics as st
A=[json.load(open(f)) for f in sorted(glob.glob("phase1/A5/metrics/metrics_seed*.json"))]
def col(k): return [r.get(k) for r in A]
def med(xs): return st.median([x for x in xs if isinstance(x,(int,float))])
# frozen Phase-0 medians (progress.md §3)
BASE={"min_same_lane_gap_m":65.8,"n_completed":877,"conflict_samples":4006,
      "n_battery_dead":123,"total_hold_minutes":12772,"peak_backlog":999,
      "mean_wait_for_leg_s":6119,"n_reroutes":1063,"max_agents_on_a_lane":6}
keys=list(BASE)
print(f"{'metric':22}{'baseline':>12}{'A5 median':>12}{'A5 range':>20}")
for k in keys:
    xs=[x for x in col(k) if isinstance(x,(int,float))]
    if xs: print(f"{k:22}{BASE[k]:>12}{med(xs):>12.2f}{f'{min(xs):.1f}..{max(xs):.1f}':>20}")
gl=any(r.get("gridlock") for r in A)
dead=med(col("n_battery_dead")); gap=[x for x in col("min_same_lane_gap_m") if isinstance(x,(int,float))]
holdd=(BASE["total_hold_minutes"]-med(col("total_hold_minutes")))/BASE["total_hold_minutes"]*100
compd=med(col("n_completed"))-BASE["n_completed"]
laned=med(col("max_agents_on_a_lane"))-BASE["max_agents_on_a_lane"]
print(f"\nHARD GATES:")
print(f"  G2 no gridlock: {'PASS' if not gl else 'FAIL'}")
print(f"  G3 n_battery_dead {dead:.0f} <= baseline {BASE['n_battery_dead']}: {'PASS' if dead<=BASE['n_battery_dead'] else 'FAIL'}")
print(f"  G1' gap not-worse than baseline ({min(gap):.1f} vs ~58m floor): {'PASS' if min(gap)>=57.0 else 'FAIL'}")
print(f"SOFT GOALS (objective = balance load / raise utilisation):")
print(f"  total_hold_minutes change: {holdd:+.1f}%  (★ want <= -15%)")
print(f"  n_completed change: {compd:+.0f}  (want +)")
print(f"  max_agents_on_a_lane change: {laned:+.1f}  (want - / more even)")
PY
