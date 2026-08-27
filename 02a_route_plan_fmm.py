#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""02a_route_plan_fmm.py -- branch (a_fmm) : route planning (FMM)

Thin branch launcher: runs the shared engine `src/engine_route_plan.py` with this branch's
parameters, so both branches execute the SAME code and differ only in their
inputs and output tree. Edit the parameters in `params/a_fmm/02_route_plan.params`.
"""
import runpy, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if __name__ == "__main__":
    sys.argv = [str(HERE / "src" / "engine_route_plan.py"), "--param-file", "params/a_fmm/02_route_plan.params", *sys.argv[1:]]
    runpy.run_path(str(HERE / "src" / "engine_route_plan.py"), run_name="__main__")
