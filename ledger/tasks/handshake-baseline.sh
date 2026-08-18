#!/usr/bin/env bash
# handshake-baseline: confirm THIS machine reproduces the frozen Phase-0 baseline
# for seed 12345, so cross-machine numbers can be trusted (guards env drift).
# Exit 0 = match, 1 = mismatch. Run via ledger/relay.py or: bash ledger/tasks/handshake-baseline.sh
set -euo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"          # repo root
PYTHON="${PYTHON:-python}"

echo "[handshake] host=$(uname -s) running baseline seed 12345 ..."
"$PYTHON" 09_simulate_agents_2d.py --param-file params/baseline_p0.params \
    --seed 12345 --no-animation --no-html --no-pyvista > /tmp/handshake_run.log 2>&1

"$PYTHON" - <<'PY'
import json, sys
m = json.load(open("output/09_phase0_baseline/metrics.json"))
# frozen reference for seed 12345 (from phase0/baseline_summary.json)
checks = [
    ("n_completed",         m.get("n_completed"),         869,   0),      # exact
    ("gridlock",            m.get("gridlock"),            False, None),   # exact bool
    ("min_same_lane_gap_m", m.get("min_same_lane_gap_m"), 62.88, 1.0),    # +-1 m
    ("conflict_samples",    m.get("conflict_samples"),    3999,  80),     # +-2%
]
ok = True
for name, got, ref, tol in checks:
    if tol is None:
        good = (got == ref)
    else:
        good = (got is not None) and (abs(got - ref) <= tol)
    ok = ok and good
    print(f"  {name:22} got={got!s:<12} ref={ref!s:<8} {'OK' if good else 'MISMATCH'}")
print("[handshake]", "MATCH — env parity confirmed" if ok else "MISMATCH — investigate env/numpy drift before trusting cross-machine numbers")
sys.exit(0 if ok else 1)
PY
