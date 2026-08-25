#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""07b_simulate_thetastar.py -- branch (b_theta) : coordination simulation

Thin branch launcher: runs the shared engine `engine_simulate.py` with this branch's
parameters, so both branches execute the SAME code and differ only in their
inputs and output tree. Edit the parameters in `params/b_theta/07_simulate.params`.
"""
import runpy, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if __name__ == "__main__":
    sys.argv = [str(HERE / "engine_simulate.py"), "--param-file", "params/b_theta/07_simulate.params", *sys.argv[1:]]
    runpy.run_path(str(HERE / "engine_simulate.py"), run_name="__main__")
