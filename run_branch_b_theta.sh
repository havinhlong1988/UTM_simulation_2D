#!/usr/bin/env bash
# Branch b_theta: everything after the shared random map (01).
# Stage 01 is SHARED -- run it once, outside this script.
set -euo pipefail
PY="${PY:-$HOME/anaconda3/envs/utm/bin/python}"
s(){ echo; echo "=========== $* ==========="; }
s "02b route plan";                     $PY 02b_route_plan_thetastar.py
s "03b density + traffic nodes";        $PY 03b_route_density_thetastar.py
s "04b master corridor";                $PY 04b_master_corridor_thetastar.py
s "05b corridor network + roundabouts"; $PY 05b_corridor_network_thetastar.py
s "07b sim (pass 1, no costmap)";       $PY 07b_simulate_thetastar.py
s "06b costmap from pass-1 traffic";    $PY 06b_costmap_thetastar.py
s "07b sim (pass 2, with costmap)";     $PY 07b_simulate_thetastar.py
echo; echo "branch b_theta done -> output/b_theta/"
