#!/usr/bin/env bash
# Branch a_fmm: everything after the shared random map (01).
# Stage 01 is SHARED -- run it once, outside this script.
set -euo pipefail
PY="${PY:-$HOME/anaconda3/envs/utm/bin/python}"
s(){ echo; echo "=========== $* ==========="; }
s "02a route plan";                     $PY 02a_route_plan_fmm.py
s "03a density + traffic nodes";        $PY 03a_route_density_fmm.py
s "04a master corridor";                $PY 04a_master_corridor_fmm.py
s "05a corridor network + roundabouts"; $PY 05a_corridor_network_fmm.py
s "06a costmap (econ/air/ground on the 05 network)"; $PY 06a_costmap_fmm.py
s "07a sim (pass 1, assessed costmap)";              $PY 07a_simulate_fmm.py
s "06a costmap (+ measured pass-1 traffic)";         $PY 06a_costmap_fmm.py
s "07a sim (pass 2, full costmap)";                  $PY 07a_simulate_fmm.py
echo; echo "branch a_fmm done -> output/a_fmm/"
