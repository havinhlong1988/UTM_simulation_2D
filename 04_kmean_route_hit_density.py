#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
04_kmean_route_hit_density_v4.py

Version: v4_strict_route_density_mesh

This runner imports src.kmean_v4_strict_route_density_mesh. It is provided to avoid any cache or
file-overwrite confusion with older src/kmean.py files.

Run:
    python 04_kmean_route_hit_density_v4.py params/kmean_v4_strict_route_density_mesh.params

Expected important figure:
    output/04_kmean_route_hit_density/figures/00_trafic_density.png
"""

from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from src.kmean import load_params, run_route_hit_density_kmeans, MODULE_VERSION

SCRIPT_VERSION = "v4_strict_route_density_mesh"  # VISIBLE VERSION TAG - 2026-07-01


def main() -> None:
    print(f"[kmean] 04_kmean_route_hit_density_v4.py version: {SCRIPT_VERSION}")
    print(f"[kmean] imported module version: {MODULE_VERSION}")
    params_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("params/kmean.params")
    params = load_params(params_file)
    run_route_hit_density_kmeans(params)


if __name__ == "__main__":
    main()
