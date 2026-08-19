#!/usr/bin/env bash
# phase2-A6-sweep: full 5-seed sweep of DCB_CORRIDOR_SHARE to locate the knee of
# the throughput <-> latency tradeoff before freezing an A6 config. Writes
# per-run metrics under phase2/A6_sweep/sh<share>/ and prints a median table.
set -euo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
BASE="params/phase2_A6_dcb.params"
TMP="$(mktemp -d)"
for SH in 0.75 1.0 1.25 1.5; do
  P="$TMP/sh$SH.params"
  sed "s/^DCB_CORRIDOR_SHARE .*/DCB_CORRIDOR_SHARE  = $SH/" "$BASE" > "$P"
  M="phase2/A6_sweep/sh$SH/metrics"; mkdir -p "$M"
  for S in 12345 54321 11111 22222 33333; do
    echo "[sweep share=$SH] seed $S $(date +%H:%M:%S)"
    "$PYTHON" 09_simulate_agents_2d.py --param-file "$P" --seed "$S" \
        --no-animation --no-html --no-pyvista > "$M/log_seed$S.txt" 2>&1
    cp output/09_phase2_A6_dcb/metrics.json "$M/metrics_seed$S.json"
  done
done
"$PYTHON" - <<'PY'
import json, glob, statistics as st
BASE={"n_completed":877,"total_hold_minutes":12772,"n_battery_dead":123,
      "conflict_samples":4006,"mean_wait_for_leg_s":6119,"sim_end_hours":5.12,
      "n_reroutes":1063}
def med(rows,k): return st.median([r[k] for r in rows if isinstance(r.get(k),(int,float))])
print(f"{'share':>6} | {'completed':>10} {'hold_min':>9} {'dead':>5} {'conflict':>8} {'mwait':>7} {'makespan':>8} {'reroute':>7}")
print(f"{'BASE':>6} | {BASE['n_completed']:>10} {BASE['total_hold_minutes']:>9.0f} {BASE['n_battery_dead']:>5} {BASE['conflict_samples']:>8} {BASE['mean_wait_for_leg_s']:>7.0f} {BASE['sim_end_hours']:>8.2f} {BASE['n_reroutes']:>7}")
for SH in ["0.75","1.0","1.25","1.5"]:
    R=[json.load(open(f)) for f in sorted(glob.glob(f"phase2/A6_sweep/sh{SH}/metrics/metrics_seed*.json"))]
    if not R: continue
    print(f"{SH:>6} | {med(R,'n_completed'):>10.0f} {med(R,'total_hold_minutes'):>9.0f} "
          f"{med(R,'n_battery_dead'):>5.0f} {med(R,'conflict_samples'):>8.0f} "
          f"{med(R,'mean_wait_for_leg_s'):>7.0f} {med(R,'sim_end_hours'):>8.2f} {med(R,'n_reroutes'):>7.0f}")
print("\n(medians of 5 seeds; deltas vs BASE row above)")
PY
echo "SWEEP_DONE"
