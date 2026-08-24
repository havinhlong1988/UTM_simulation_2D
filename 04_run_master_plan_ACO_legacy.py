
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_run_master_plan_v5_plot_only_snapshots.py

v5 adds:
    - RUN_MODE = all | plot_only
    - figures/PSO_snap
    - figures/ACO_snap
    - figures/corridors
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    from src.routeplan_PSO_ACO import run_master_plan, load_params, MODULE_VERSION
except Exception:
    from routeplan_PSO_ACO import run_master_plan, load_params, MODULE_VERSION


def main() -> None:
    params_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("params/routeplan_PSO_ACO.params")
    print(f"[routerplan] runner uses module version: {MODULE_VERSION}")
    params = load_params(params_file)
    run_master_plan(params)


if __name__ == "__main__":
    main()
