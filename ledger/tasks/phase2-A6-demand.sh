#!/usr/bin/env bash
# phase2-A6-demand: A/B demand-capacity balancing (DCB_MODE) over the 5 D2 seeds
# vs the frozen Phase-0 baseline. DCB_MODE is OFF by default in the code, so only
# params/phase2_A6_dcb.params differs. A6 attacks the launch-backlog bottleneck
# that A7/A5 could not (peak_backlog=999): meter launches per origin corridor so
# the fleet spreads instead of draining one corridor first. Judges hard gates
# (G2 gridlock, G3 battery, G1' gap not-worse) + ★ goals (holds -15%, completions
# +, load evenness). Verdict printed + journaled.
set -euo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
M="phase2/A6/metrics"; mkdir -p "$M"
for S in 12345 54321 11111 22222 33333; do
  echo "[A6] seed $S $(date +%H:%M:%S)"
  "$PYTHON" 09_simulate_agents_2d.py --param-file params/phase2_A6_dcb.params \
      --seed "$S" --no-animation --no-html --no-pyvista > "$M/log_seed$S.txt" 2>&1
  cp output/09_phase2_A6_dcb/metrics.json "$M/metrics_seed$S.json"
done
"$PYTHON" - <<'PY'
import json, glob, statistics as st
A=[json.load(open(f)) for f in sorted(glob.glob("phase2/A6/metrics/metrics_seed*.json"))]
def col(k): return [r.get(k) for r in A]
def med(xs): return st.median([x for x in xs if isinstance(x,(int,float))])
BASE={"min_same_lane_gap_m":65.8,"n_completed":877,"conflict_samples":4006,
      "n_battery_dead":123,"total_hold_minutes":12772,"peak_backlog":999,
      "mean_wait_for_leg_s":6119,"n_reroutes":1063,"max_agents_on_a_lane":6}
print(f"{'metric':22}{'baseline':>12}{'A6 median':>12}{'A6 range':>20}{'delta%':>10}")
for k in list(BASE):
    xs=[x for x in col(k) if isinstance(x,(int,float))]
    if not xs: continue
    m=med(xs); d=(m-BASE[k])/BASE[k]*100 if BASE[k] else 0
    print(f"{k:22}{BASE[k]:>12}{m:>12.2f}{f'{min(xs):.1f}..{max(xs):.1f}':>20}{d:>+9.1f}%")
gl=any(r.get("gridlock") for r in A)
dead=med(col("n_battery_dead")); gap=[x for x in col("min_same_lane_gap_m") if isinstance(x,(int,float))]
holdd=(med(col("total_hold_minutes"))-BASE["total_hold_minutes"])/BASE["total_hold_minutes"]*100
compd=med(col("n_completed"))-BASE["n_completed"]
print(f"\nHARD GATES:  G2 gridlock {'PASS' if not gl else 'FAIL'} | "
      f"G3 battery {dead:.0f}<= {BASE['n_battery_dead']} {'PASS' if dead<=BASE['n_battery_dead'] else 'FAIL'} | "
      f"G1' gap {min(gap):.1f} not-worse {'PASS' if min(gap)>=57.0 else 'FAIL'}")
print(f"★ GOALS:  hold {holdd:+.1f}% (want<=-15) | completed {compd:+.0f} (want+) | "
      f"battery {(dead-BASE['n_battery_dead'])/BASE['n_battery_dead']*100:+.1f}%")
kept = (not gl) and dead<=BASE['n_battery_dead'] and min(gap)>=57.0 and (holdd<=-15 or compd>0)
print(f"\nKEEP/KILL (§5): {'KEEP' if kept else 'review'} "
      f"(hard gates pass + >=1 ★ clears: hold<=-15% OR completions up)")
PY
