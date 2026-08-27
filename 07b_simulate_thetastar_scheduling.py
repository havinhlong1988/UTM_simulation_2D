#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""07b_simulate_thetastar_scheduling.py -- branch (b_theta) : coordination simulation

Thin branch launcher: runs the shared engine `src/engine_simulate_scheduling.py` with this branch's
parameters, so both branches execute the SAME code and differ only in their
inputs and output tree. Edit the parameters in `params/b_theta/07_simulate_scheduling.params`.
"""
import runpy, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if __name__ == "__main__":
    sys.argv = [str(HERE / "src" / "engine_simulate_scheduling.py"), "--param-file", "params/b_theta/07_simulate_scheduling.params", *sys.argv[1:]]
    runpy.run_path(str(HERE / "src" / "engine_simulate_scheduling.py"), run_name="__main__")
