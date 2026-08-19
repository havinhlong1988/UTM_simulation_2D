#!/usr/bin/env bash
# phase3-A4-speed: A/B speed control (SPEED_CONTROL) over the 5 D2 seeds vs the
# CURRENT frozen baseline baseline_p1 (DCB on). SPEED_CONTROL is OFF by default in
# the code, so only params/phase3_A4_speed.params differs. Goal: smooth stop-and-go
# to recover some of the makespan/wait DCB traded away, without failing the hard
# gates. Verdict printed + journaled.
set -euo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
M="phase3/A4/metrics"; mkdir -p "$M"
for S in 12345 54321 11111 22222 33333; do
  echo "[A4] seed $S $(date +%H:%M:%S)"
  "$PYTHON" 09_simulate_agents_2d.py --param-file params/phase3_A4_speed.params \
      --seed "$S" --no-animation --no-html --no-pyvista > "$M/log_seed$S.txt" 2>&1
  cp output/09_phase3_A4_speed/metrics.json "$M/metrics_seed$S.json"
done
"$PYTHON" - <<'PY'
import json, glob, statistics as st
A=[json.load(open(f)) for f in sorted(glob.glob("phase3/A4/metrics/metrics_seed*.json"))]
B=json.load(open("baseline_p1/baseline_summary.json"))["metrics"]
def med(k): return st.median([r[k] for r in A if isinstance(r.get(k),(int,float))])
def bmed(k): return B[k]["median"]
keys=["min_same_lane_gap_m","n_completed","conflict_samples","n_battery_dead",
      "total_hold_minutes","peak_backlog","mean_wait_for_leg_s","sim_end_hours",
      "n_reroutes","max_agents_on_a_lane"]
print(f"{'metric':22}{'baseline_p1':>13}{'A4 median':>12}{'A4 range':>18}{'delta%':>9}")
for k in keys:
    xs=[r.get(k) for r in A if isinstance(r.get(k),(int,float))]
    if not xs: continue
    m=med(k); bb=bmed(k); d=(m-bb)/bb*100 if bb else 0
    print(f"{k:22}{bb:>13.2f}{m:>12.2f}{f'{min(xs):.1f}..{max(xs):.1f}':>18}{d:>+8.1f}%")
gl=any(r.get("gridlock") for r in A)
dead=med("n_battery_dead"); gap=[r["min_same_lane_gap_m"] for r in A]
mk=(med("sim_end_hours")-bmed("sim_end_hours"))/bmed("sim_end_hours")*100
wt=(med("mean_wait_for_leg_s")-bmed("mean_wait_for_leg_s"))/bmed("mean_wait_for_leg_s")*100
print(f"\nHARD GATES: G2 gridlock {'PASS' if not gl else 'FAIL'} | "
      f"G3 battery {dead:.0f}<= {bmed('n_battery_dead'):.0f} {'PASS' if dead<=bmed('n_battery_dead') else 'FAIL'} | "
      f"G1' gap {min(gap):.1f} not-worse {'PASS' if min(gap)>=57.0 else 'FAIL'}")
print(f"GOAL (recover DCB cost): makespan {mk:+.1f}% | mean_wait {wt:+.1f}%  (want both DOWN)")
PY
