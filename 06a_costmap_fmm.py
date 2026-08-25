#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""06a_costmap_fmm.py -- branch (a_fmm) : traffic cost-map

Thin branch launcher: runs the shared engine `engine_costmap.py` on this
branch's pass-1 traffic and corridor network. The engine takes explicit paths
rather than a params file, so they are spelled out here.
"""
import runpy, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
B = "a_fmm"
if __name__ == "__main__":
    sys.argv = [str(HERE / "engine_costmap.py"),
                "--traffic", f"output/{B}/07_agent_sim/trajectories.csv",
                "--corridor-dir", f"output/{B}/05_corridor_network",
                "--corridor-param-file", f"params/{B}/05_corridor_network.params",
                "--out-dir", f"output/{B}/06_costmap",
                *sys.argv[1:]]
    runpy.run_path(str(HERE / "engine_costmap.py"), run_name="__main__")
