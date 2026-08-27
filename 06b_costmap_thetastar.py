#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""06b_costmap_thetastar.py -- branch (b_theta) : slowness cost-map

Thin branch launcher: runs the shared engine `src/engine_costmap.py` on this
branch's corridor network. The engine prices the network with four layers --
economic balance, air operational safety, ground safety and (optionally)
measured traffic -- see params/b_theta/06_costmap.params.

The traffic layer is OPTIONAL: this stage runs straight after 05 with the
three assessed layers, and folds the measured density in on a re-run once a
pass-1 simulation has produced trajectories.csv.
"""
import runpy, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
B = "b_theta"
if __name__ == "__main__":
    sys.argv = [str(HERE / "src" / "engine_costmap.py"),
                "--param-file", f"params/{B}/06_costmap.params",
                "--traffic", f"output/{B}/07_agent_sim_scheduling/trajectories.csv",
                "--corridor-dir", f"output/{B}/05_corridor_network",
                "--corridor-param-file", f"params/{B}/05_corridor_network.params",
                "--out-dir", f"output/{B}/06_costmap",
                *sys.argv[1:]]
    runpy.run_path(str(HERE / "src" / "engine_costmap.py"), run_name="__main__")
