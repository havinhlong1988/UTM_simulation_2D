#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""03b_route_density_thetastar.py -- branch (b_theta) : route density + traffic nodes

Thin branch launcher: runs the shared engine `engine_route_density.py` with this branch's
parameters, so both branches execute the SAME code and differ only in their
inputs and output tree. Edit the parameters in `params/b_theta/03_route_density.params`.
"""
import runpy, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if __name__ == "__main__":
    sys.argv = [str(HERE / "engine_route_density.py"), "--param-file", "params/b_theta/03_route_density.params", *sys.argv[1:]]
    runpy.run_path(str(HERE / "engine_route_density.py"), run_name="__main__")
