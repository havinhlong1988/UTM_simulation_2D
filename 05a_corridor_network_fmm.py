#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""05a_corridor_network_fmm.py -- branch (a_fmm) : corridor network + roundabouts

Thin branch launcher: runs the shared engine `engine_corridor_network.py` with this branch's
parameters, so both branches execute the SAME code and differ only in their
inputs and output tree. Edit the parameters in `params/a_fmm/05_corridor_network.params`.
"""
import runpy, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if __name__ == "__main__":
    sys.argv = [str(HERE / "engine_corridor_network.py"), "--param-file", "params/a_fmm/05_corridor_network.params", *sys.argv[1:]]
    runpy.run_path(str(HERE / "engine_corridor_network.py"), run_name="__main__")
