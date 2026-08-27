#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""06a_costmap_fmm.py -- branch (a_fmm) : slowness cost-map

Thin branch launcher: runs the shared engine `engine_costmap.py` on this
branch's corridor network. The engine prices the network with four layers --
economic balance, air operational safety, ground safety and (optionally)
measured traffic -- see params/a_fmm/06_costmap.params.

The traffic layer is OPTIONAL: this stage runs straight after 05 with the
three assessed layers, and folds the measured density in on a re-run once a
pass-1 simulation has produced trajectories.csv.
"""
import runpy, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
B = "a_fmm"
if __name__ == "__main__":
    sys.argv = [str(HERE / "engine_costmap.py"),
                "--param-file", f"params/{B}/06_costmap.params",
                "--traffic", f"output/{B}/07_agent_sim_scheduling/trajectories.csv",
                "--corridor-dir", f"output/{B}/05_corridor_network",
                "--corridor-param-file", f"params/{B}/05_corridor_network.params",
                "--out-dir", f"output/{B}/06_costmap",
                *sys.argv[1:]]
    runpy.run_path(str(HERE / "engine_costmap.py"), run_name="__main__")
