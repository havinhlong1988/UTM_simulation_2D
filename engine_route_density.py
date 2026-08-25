#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
engine_route_density.py

Route-density analysis for the step-02 route network (02_route_plan.py).

Finds the HIGH-DENSITY areas of the planned routes -- where many corridors run
close together -- which are the natural candidates for traffic-node / roundabout
placement (step 05/06/07).

Method (corridor-coverage density, route + buffer band + circular radius):
  1. Rasterise every route polyline from output/02_route_plan/route_points.csv
     onto the step-01 grid (centreline cells). hit_center(cell) = #routes whose
     centreline passes through the cell.
  2. For each route, grow its centreline by the CORRIDOR BAND half-extent
     (WIDTH/2 + BUFFER, read from step-02 metrics.json) PLUS a circular
     NODE_RADIUS_M, and vote on every covered cell -> hit_cover(cell) = #routes
     whose band+radius footprint reaches the cell. So a node's density counts the
     routes through it, the buffers over it, and their circular neighbourhood.
     Coverage never spills into no-fly cells.
  3. (Optional) extra Gaussian smoothing of DENSITY_SIGMA_M m; 0 = off (the disk
     already spreads density). Select traffic-node areas by RELATIVE density
     (density / the busiest genuine junction) with a TWO-TIER bar: the inner-city
     core needs rel >= TN_REL_INNER (0.5), the near-boundary belt (within
     BOUNDARY_MARGIN_M of a map edge) only rel >= TN_REL_BOUNDARY (0.25), so hubs
     are placed out toward the edges too. Connected components -> AREAS; each
     area's density PEAK is a candidate hub, then filtered (not at a DB/DK, has
     obstacle clearance, spaced >= TN_MIN_SEPARATION_M apart for an even layout).

Runs directly (parameters embedded in the header):
    python engine_route_density.py
    python engine_route_density.py --sigma-m 150 --percentile 92

Inputs
------
    output/02_route_plan/route_points.csv          routes (pair, alt, seq, x, y)
    output/01_random_node_map/*.xyz                grid + no-fly + DB/DK (optional)

Outputs (OUT_DIR)
-----------------
    route_density.npz          hit_center + hit_cover + density grids (+ extent, dx)
    high_density_areas.csv     per area: centroid, peak/mean density, size, hits
    traffic_nodes.csv          the picked traffic nodes (region, rel_peak, spacing)
    relief_nodes.csv           bypass fill nodes (inner-core lattice + 4 corners)
    network_nodes.csv          combined layout: objectives + traffic + relief nodes
    node_hit_count.csv         per cell: x, y, routes_through, buffer_cover, density
    figures/00_route_density.png        density heatmap + routes + hot areas
    figures/01_high_density_areas.png   the thresholded high-density regions
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, label, center_of_mass, distance_transform_edt

THIS_DIR = Path(__file__).resolve().parent
VERSION = "v1"

# ======================================================================
# PARAMETERS  (run defaults -- EDIT HERE; each is overridden by its CLI flag)
# ======================================================================
PARAMETERS: dict = {
    # ---- inputs / outputs ----
    "ROUTE_POINTS_FILE": "output/_engine_default/02_route_plan/route_points.csv",
    "RISK_XYZ":          "output/01_random_node_map/random_2d_node_riskmap_seed2102359706.xyz",
    # step-02 metrics.json -> corridor width/buffer (band half-extent). If missing,
    # ROUTE_WIDTH_M / ROUTE_BUFFER_M below are used.
    "ROUTE_METRICS_FILE": "output/_engine_default/02_route_plan/metrics.json",
    "OUT_DIR":           "output/_engine_default/03_route_density",
    # ---- what counts as a route ----
    # include the diversified alternatives (True -> corridor COVERAGE density) or
    # only the primary route per pair (#alt0 -> backbone density).
    "INCLUDE_ALTERNATIVES": True,
    # ---- density model: route + BUFFER band + circular radius expansion ----
    # Each route votes on every cell its full CORRIDOR BAND covers (centreline +-
    # band half-extent = WIDTH/2 + BUFFER), then that footprint is grown by a
    # circular NODE_RADIUS_M so the density spreads into the neighbourhood. A
    # cell's density = number of route corridors (band + radius) that reach it.
    "BUFFER_DENSITY":      True,      # False -> old behaviour (centreline hits only)
    "NODE_RADIUS_M":       100.0,     # circular expansion of each band into neighbours (m)
    "ROUTE_WIDTH_M":        50.0,     # fallback usable corridor width if no metrics file
    "ROUTE_BUFFER_M":       12.5,     # fallback safety buffer if no metrics file
    # ---- density (optional extra grid-KDE smoothing on top of the coverage) ----
    "DENSITY_SIGMA_M":        0.0,    # Gaussian bandwidth (m); 0 = off (disk does the spread)
    "NOFLY_SLOWNESS":         10.0,   # step-01 slowness >= this marks a no-fly cell
    # ---- high-density area extraction: RELATIVE-density two-tier threshold ----
    # A cell is high-density when its RELATIVE density (density / a reference peak
    # = the busiest genuine, non-terminal junction) clears a bar that depends on
    # WHERE the cell is: the busy INNER-CITY core needs a high bar, the quieter
    # NEAR-BOUNDARY belt a lower one, so hubs are still placed out toward the map
    # edges and the network stays evenly covered.
    "TN_THRESHOLD_MODE":  "relative", # "relative" (two-tier) | "percentile" (legacy)
    "TN_REL_INNER":         0.40,     # rel-density bar in the inner-city core
    "TN_REL_BOUNDARY":      0.20,     # rel-density bar in the near-boundary belt
    "BOUNDARY_MARGIN_M":   800.0,     # a cell within this of a map edge is "near-boundary"
    "HIGH_DENSITY_PERCENTILE": 90.0,  # used only when TN_THRESHOLD_MODE = "percentile"
    "MIN_AREA_CELLS":            4,    # ignore high-density blobs smaller than this
    # ---- Traffic Node (TN) selection: greedy spaced peak-picking ----
    # Objectives (DB/DK) are seeded as nodes, so new TN keep TN_MIN_SEPARATION_M
    # from them too -> terminals + TN form one evenly-spaced network.
    "TN_OBJECTIVE_MERGE_M":    300.0,  # radius around a DB/DK excluded from the reference-peak
    "TN_OBSTACLE_CLEARANCE_M":  75.0,  # a TN must be >= this from any no-fly (node r 25 + corridor 50)
    "TN_OBSTACLE_SHIFT_MAX_M": 150.0,  # (unused by greedy picker; kept for the percentile legacy)
    "TN_MIN_SEPARATION_M":     500.0,  # spacing between ALL nodes incl. objectives (smaller -> more)
    "TN_MAJOR_TOP_K":            15,     # the K busiest kept TN are 'major', the rest 'minor'
    "TN_MAX":                     0,    # cap on total TN (0 = no cap; count set by threshold+spacing)
    # ---- relief / bypass fill nodes (offload the busy network) ----
    # extra, SPARSER nodes placed OFF the busy corridors to seed BYPASS routes:
    # a lattice over the inner-core gaps + one node near each of the 4 map corners,
    # spaced FILL_SPACING_FACTOR x TN_MIN_SEPARATION_M apart.
    "FILL_RELIEF_ENABLE":  True,
    "FILL_SPACING_FACTOR":  1.5,       # relief spacing = factor x TN_MIN_SEPARATION_M (=750 m)
    "FILL_INNER":          True,       # lattice-fill the inner-core gaps
    "FILL_CORNERS":        True,       # a node near each of the 4 map corners
    "CORNER_INSET_M":      400.0,      # how far inside each corner to anchor the corner node
    "FILL_MIN_GAP_M":      300.0,      # min gap from a main node (keep relief off, not on, them)
    "MAKE_FIGURES":            True,
}


# ----------------------------------------------------------------------
def parse_value(v: str):
    s = v.strip()
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return s


def load_params(path: Path) -> dict:
    params: dict = {}
    if not path.exists():
        return params
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line and "=" in line:
            k, v = line.split("=", 1)
            params[k.strip()] = parse_value(v.strip())
    return params


def pget(params, key, default):
    return params.get(key, default)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Route-density analysis: find the high-density areas of the "
                    "step-02 route network. Defaults come from the header PARAMETERS.")
    p.add_argument("--param-file", default=None,
                   help="optional params file that overrides the header PARAMETERS.")
    p.add_argument("--sigma-m", type=float, default=None, help="Gaussian bandwidth (m).")
    p.add_argument("--percentile", type=float, default=None,
                   help="high-density threshold percentile.")
    p.add_argument("--no-figures", action="store_true")
    return p.parse_args()


# ----------------------------------------------------------------------
def load_grid(xyz_path: Path, nofly_slw: float):
    """(nofly[ny,nx], x0, y0, dx, nx, ny, objectives{label:(x,y)}) from step-01 .xyz."""
    df = pd.read_csv(xyz_path, sep=r"\s+")
    xs = np.sort(df["x"].unique()); ys = np.sort(df["y"].unique())
    dx = float(np.median(np.diff(xs)))
    x0, y0 = float(xs[0]), float(ys[0])
    nx, ny = len(xs), len(ys)
    ix = np.rint((df["x"].to_numpy() - x0) / dx).astype(int)
    iy = np.rint((df["y"].to_numpy() - y0) / dx).astype(int)
    slw = np.zeros((ny, nx), float)
    slw[iy, ix] = df["slowness"].to_numpy(float)
    nofly = slw >= nofly_slw
    obj = {}
    if "label" in df.columns:
        lab = df["label"].astype(str)
        sel = df[lab.str.startswith(("DB", "DK"))]
        obj = {str(r.label): (float(r.x), float(r.y)) for r in sel.itertuples()}
    return nofly, x0, y0, dx, nx, ny, obj


def _bres(a, b):
    (i0, j0), (i1, j1) = a, b
    di, dj = abs(i1 - i0), abs(j1 - j0)
    si = 1 if i1 > i0 else -1
    sj = 1 if j1 > j0 else -1
    i, j, err, out = i0, j0, di - dj, []
    while True:
        out.append((i, j))
        if i == i1 and j == j1:
            return out
        e2 = 2 * err
        if e2 > -dj:
            err -= dj; i += si
        if e2 < di:
            err += di; j += sj


def snap_to_clearance(px, py, clr, x0, y0, dx, nx, ny, need_m, max_shift_m):
    """Return (x, y, clearance) of the nearest cell with clearance >= need_m within
    max_shift_m of (px, py); or None if the pocket is too tight. Keeps a TN off
    obstacles so the corridor geometry fits around it."""
    ix = int(np.clip(round((px - x0) / dx), 0, nx - 1))
    iy = int(np.clip(round((py - y0) / dx), 0, ny - 1))
    if clr[iy, ix] >= need_m:
        return px, py, float(clr[iy, ix])
    R = int(round(max_shift_m / dx))
    best, best_d = None, 1e18
    for a in range(max(0, iy - R), min(ny, iy + R + 1)):
        for b in range(max(0, ix - R), min(nx, ix + R + 1)):
            if clr[a, b] >= need_m:
                d = (a - iy) ** 2 + (b - ix) ** 2
                if d < best_d:
                    best_d, best = d, (b, a)
    if best is None:
        return None
    b, a = best
    return x0 + b * dx, y0 + a * dx, float(clr[a, b])


def rasterize_route(xy, x0, y0, dx, nx, ny):
    """Distinct grid cells (iy, ix) a route polyline passes through (Bresenham)."""
    ij = [(int(np.clip(round((y - y0) / dx), 0, ny - 1)),
           int(np.clip(round((x - x0) / dx), 0, nx - 1))) for x, y in xy]
    cells = set()
    for a, b in zip(ij, ij[1:]):
        cells.update(_bres(a, b))
    if len(ij) == 1:
        cells.add(ij[0])
    return cells


# ----------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    params = dict(PARAMETERS)
    if args.param_file:
        pf = THIS_DIR / args.param_file
        if pf.exists():
            params.update(load_params(pf))
        else:
            print(f"[warn] --param-file {pf} not found; using header PARAMETERS")
    if args.sigma_m is not None:
        params["DENSITY_SIGMA_M"] = args.sigma_m
    if args.percentile is not None:
        params["HIGH_DENSITY_PERCENTILE"] = args.percentile
    make_fig = (not args.no_figures) and bool(pget(params, "MAKE_FIGURES", True))

    out_dir = THIS_DIR / str(pget(params, "OUT_DIR", "output/_engine_default/03_route_density"))
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    sigma_m = float(pget(params, "DENSITY_SIGMA_M", 100.0))
    pct = float(pget(params, "HIGH_DENSITY_PERCENTILE", 90.0))
    min_cells = int(pget(params, "MIN_AREA_CELLS", 4))
    incl_alt = bool(pget(params, "INCLUDE_ALTERNATIVES", True))

    print("=" * 66)
    print(f"engine_route_density.py  {VERSION}")
    print(f"Density model  : route + buffer band + circular radius  "
          f"(extra KDE sigma {sigma_m:.0f} m)")
    print(f"Output dir     : {out_dir}")
    print("=" * 66)

    # ---- grid + no-fly + objectives from step 01 ----
    nofly, x0, y0, dx, nx, ny, obj_xy = load_grid(
        THIS_DIR / str(pget(params, "RISK_XYZ", "")),
        float(pget(params, "NOFLY_SLOWNESS", 10.0)))
    extent = [x0, x0 + nx * dx, y0, y0 + ny * dx]
    print(f"Grid          : {nx} x {ny} @ {dx:.0f} m")

    # ---- corridor geometry (band half-extent = WIDTH/2 + BUFFER) ----
    # prefer the exact geometry step 02 planned with (metrics.json); else params.
    band_half = 0.5 * float(pget(params, "ROUTE_WIDTH_M", 50.0)) \
        + float(pget(params, "ROUTE_BUFFER_M", 12.5))
    mfile = THIS_DIR / str(pget(params, "ROUTE_METRICS_FILE", ""))
    if mfile.exists():
        try:
            cm = json.loads(mfile.read_text(encoding="utf-8")).get("corridor", {})
            band_half = float(cm.get("half_extent_m", band_half))
        except Exception:
            pass
    node_radius = float(pget(params, "NODE_RADIUS_M", 100.0))
    buffer_density = bool(pget(params, "BUFFER_DENSITY", True))
    foot_r = (band_half + node_radius) if buffer_density else 0.0
    passable = ~nofly

    # ---- routes -> density field: route + BUFFER band + circular radius ----
    # hit_center = #routes whose CENTRELINE passes through a cell.
    # hit_cover  = #routes whose CORRIDOR BAND (centreline +- band_half) grown by
    #              NODE_RADIUS_M reaches the cell -> "routes + buffer + circular
    #              neighbourhood". Coverage never spills into no-fly cells.
    pts = pd.read_csv(THIS_DIR / str(pget(params, "ROUTE_POINTS_FILE", "")))
    if not incl_alt and "alt" in pts.columns:
        pts = pts[pts["alt"] == 0]
    hit_center = np.zeros((ny, nx), float)
    hit_cover = np.zeros((ny, nx), float)
    n_routes = 0
    for _pair, g in pts.groupby("pair"):
        xy = g.sort_values("seq")[["x", "y"]].to_numpy()
        if len(xy) < 1:
            continue
        cmask = np.zeros((ny, nx), bool)
        for (iy, ix) in rasterize_route(xy, x0, y0, dx, nx, ny):
            cmask[iy, ix] = True
        hit_center[cmask] += 1.0
        if foot_r > 0:                               # band + circular radius footprint
            d = distance_transform_edt(~cmask) * dx
            foot = (d <= foot_r) & passable          # never counts a no-fly cell
        else:
            foot = cmask
        hit_cover[foot] += 1.0
        n_routes += 1
    hit = hit_cover                                   # density signal used downstream
    print(f"Routes        : {n_routes}   band +-{band_half:.0f} m + r{node_radius:.0f} m"
          f"   (max coverage {int(hit_cover.max())}, max through {int(hit_center.max())})")

    # ---- optional extra grid-KDE smoothing on top of the coverage field ----
    if sigma_m > 0:
        density = gaussian_filter(hit_cover, sigma=max(sigma_m / dx, 1e-3))
    else:
        density = hit_cover.copy()
    density[nofly] = 0.0

    # ---- high-density mask: RELATIVE two-tier (inner-city vs near-boundary) ----
    covered = hit > 0
    mode = str(pget(params, "TN_THRESHOLD_MODE", "relative")).lower()
    rel_inner = float(pget(params, "TN_REL_INNER", 0.5))
    rel_bound = float(pget(params, "TN_REL_BOUNDARY", 0.25))
    bnd_margin = float(pget(params, "BOUNDARY_MARGIN_M", 800.0))
    merge_m = float(pget(params, "TN_OBJECTIVE_MERGE_M", 300.0))

    # near-boundary belt = cells within bnd_margin of any map edge; rest = inner core
    row_d = np.minimum(np.arange(ny), ny - 1 - np.arange(ny)).astype(float)
    col_d = np.minimum(np.arange(nx), nx - 1 - np.arange(nx)).astype(float)
    edge_m = np.minimum(row_d[:, None], col_d[None, :]) * dx
    boundary = edge_m < bnd_margin

    if mode == "relative":
        # reference peak = densest GENUINE junction: exclude terminal (DB/DK)
        # neighbourhoods, which are trivially the densest but are dropped as TN.
        obj_mask = np.zeros((ny, nx), bool)
        for (ox_, oy_) in obj_xy.values():
            jj = int(np.clip(round((ox_ - x0) / dx), 0, nx - 1))
            ii = int(np.clip(round((oy_ - y0) / dx), 0, ny - 1))
            obj_mask[ii, jj] = True
        far_obj = (distance_transform_edt(~obj_mask) * dx >= merge_m) \
            if obj_mask.any() else np.ones((ny, nx), bool)
        ref_cells = passable & far_obj & (density > 0)
        d_ref = float(density[ref_cells].max()) if ref_cells.any() else float(max(density.max(), 1.0))
        d_ref = max(d_ref, 1e-9)
        rel = density / d_ref                              # relative density in ~[0,1]
        thr_field = np.where(boundary, rel_bound, rel_inner)
        mask = passable & (rel >= thr_field)
        thr = round(d_ref * rel_inner, 3)                  # inner absolute bar (reference)
        print(f"High-density  : relative  inner>= {rel_inner:.2f} / boundary>= {rel_bound:.2f}"
              f"  (ref peak {d_ref:.1f}, boundary margin {bnd_margin:.0f} m)")
    else:
        rel = density / max(float(density.max()), 1e-9)
        thr = float(np.percentile(density[covered], pct)) if covered.any() else 0.0
        mask = passable & (density >= thr)
        print(f"High-density  : percentile P{pct:.0f} = {thr:.3f}")
    lab_arr, n_lab = label(mask)
    areas = []
    for lb in range(1, n_lab + 1):
        cells = (lab_arr == lb)
        n = int(cells.sum())
        if n < min_cells:
            continue
        cy, cx = center_of_mass(density * cells)     # density-weighted centroid
        wx = x0 + cx * dx
        wy = y0 + cy * dx
        masked = np.where(cells, density, -np.inf)   # density PEAK cell (true junction)
        piy, pix = np.unravel_index(int(np.argmax(masked)), masked.shape)
        areas.append({
            "area_id": len(areas) + 1,
            "cx": round(float(wx), 1), "cy": round(float(wy), 1),
            "peak_x": round(float(x0 + pix * dx), 1),
            "peak_y": round(float(y0 + piy * dx), 1),
            "peak_density": round(float(density[cells].max()), 3),
            "rel_peak": round(float(rel[cells].max()), 3),        # relative density at peak
            "region": "boundary" if bool(boundary[piy, pix]) else "inner",
            "mean_density": round(float(density[cells].mean()), 3),
            "n_cells": n, "area_m2": round(n * dx * dx, 0),
            "total_hits": int(hit[cells].sum()),
        })
    areas.sort(key=lambda a: -a["peak_density"])
    for i, a in enumerate(areas, 1):
        a["area_id"] = i
    print(f"High-density  : {len(areas)} area(s)")
    for a in areas[:8]:
        print(f"  area {a['area_id']:>2}: ({a['cx']:.0f},{a['cy']:.0f}) [{a['region']:>8}]  "
              f"rel={a['rel_peak']:.2f}  peak={a['peak_density']:.1f}  cells={a['n_cells']}")

    # ---- select Traffic Nodes (TN) by GREEDY spaced peak-picking over the mask ----
    # Not one node per blob (that collapses as thresholds drop and blobs merge):
    # instead take EVERY high-density cell that can host a node, place the densest
    # one, suppress everything within TN_MIN_SEPARATION_M, and repeat. Dense regions
    # are thus TILED with multiple, evenly-spaced nodes -> a dense hub layout whose
    # count is driven by how big the dense area is and the spacing, not the blob
    # count. The DB/DK OBJECTIVES are seeded as network nodes too, so every new TN
    # keeps the FULL TN_MIN_SEPARATION_M from them as well and the combined layout
    # (terminals + traffic nodes) stays evenly distributed. Filters: obstacle
    # clearance, min spacing from every existing node.
    obst_clear = float(pget(params, "TN_OBSTACLE_CLEARANCE_M", 75.0))
    sep_m = float(pget(params, "TN_MIN_SEPARATION_M", 400.0))
    major_k = int(pget(params, "TN_MAJOR_TOP_K", 3))
    tn_max = int(pget(params, "TN_MAX", 0))
    clr = distance_transform_edt(~nofly) * dx
    obj_pts = list(obj_xy.values())

    # candidates = high-density cells that are placeable (clear of obstacles),
    # ranked densest-first
    cand = mask & passable & (clr >= obst_clear)
    ci, cj = np.where(cand)
    order = np.argsort(-density[ci, cj], kind="stable")
    tn: list[dict] = []
    dropped = {"objective": 0, "separation": 0}
    sep2 = sep_m * sep_m
    for idx in order:
        iy, ix = int(ci[idx]), int(cj[idx])
        px, py = x0 + ix * dx, y0 + iy * dx
        if any((px - ox) ** 2 + (py - oy) ** 2 < sep2 for ox, oy in obj_pts):  # 1) too near a terminal node
            dropped["objective"] += 1; continue
        if any((px - t["x"]) ** 2 + (py - t["y"]) ** 2 < sep2 for t in tn):    # 2) too near another TN
            dropped["separation"] += 1; continue
        tn.append({"x": round(px, 1), "y": round(py, 1),
                   "region": "boundary" if bool(boundary[iy, ix]) else "inner",
                   "rel_peak": round(float(rel[iy, ix]), 3),
                   "total_hits": int(hit[iy, ix]), "peak_density": round(float(density[iy, ix]), 3),
                   "clearance_m": round(float(clr[iy, ix]), 1), "from_area": int(lab_arr[iy, ix])})
        if tn_max and len(tn) >= tn_max:
            break
    for i, t in enumerate(tn, 1):
        t["tn_id"] = i
        t["tn_class"] = "major" if i <= major_k else "minor"
    print(f"Traffic nodes : {len(tn)} kept "
          f"(dropped {dropped['objective']} at objective, {dropped['separation']} too-close)  ->  "
          f"major {min(major_k, len(tn))} / minor {max(0, len(tn) - major_k)}")
    for t in tn:
        print(f"  TN{t['tn_id']:>2} [{t['tn_class']:>5}/{t['region']:>8}] "
              f"({t['x']:.0f},{t['y']:.0f})  rel={t['rel_peak']:.2f}  "
              f"clear={t['clearance_m']:.0f} m")

    # ---- relief / bypass fill nodes: inner-core lattice + 4 corners ----
    # Extra, SPARSER nodes forming their OWN grid (spacing = FILL_SPACING_FACTOR x
    # TN_MIN_SEPARATION_M) to seed BYPASS routes that offload the dense network: a
    # lattice over the inner core + one anchor near each of the 4 map corners. Each
    # relief node is obstacle-clear, keeps the relief grid spacing from other relief
    # nodes, and stays >= FILL_MIN_GAP_M off the main nodes so it is not a duplicate
    # but a genuine OFF-corridor alternative.
    relief: list[dict] = []
    fill_fac = float(pget(params, "FILL_SPACING_FACTOR", 1.5))
    fill_sep = fill_fac * sep_m
    if bool(pget(params, "FILL_RELIEF_ENABLE", True)):
        fill_sep2 = fill_sep * fill_sep
        min_gap = float(pget(params, "FILL_MIN_GAP_M", 0.6 * sep_m))   # keep off main nodes
        min_gap2 = min_gap * min_gap
        main_xy = [(ox, oy) for ox, oy in obj_pts] + [(t["x"], t["y"]) for t in tn]
        relief_xy: list[tuple] = []

        def _try_add(px, py, kind, force=False):
            jx = int(np.clip(round((px - x0) / dx), 0, nx - 1))
            iyy = int(np.clip(round((py - y0) / dx), 0, ny - 1))
            if clr[iyy, jx] < obst_clear:                        # must clear obstacles
                return False
            if not force:
                if any((px - qx) ** 2 + (py - qy) ** 2 < min_gap2 for qx, qy in main_xy):
                    return False                                 # don't stack on a main node
                if any((px - qx) ** 2 + (py - qy) ** 2 < fill_sep2 for qx, qy in relief_xy):
                    return False                                 # keep the relief grid spacing
            relief.append({"x": round(px, 1), "y": round(py, 1), "kind": kind,
                           "region": "boundary" if bool(boundary[iyy, jx]) else "inner",
                           "clearance_m": round(float(clr[iyy, jx]), 1)})
            relief_xy.append((px, py))
            return True

        # one anchor per map corner (forced -- these are explicitly wanted)
        if bool(pget(params, "FILL_CORNERS", True)):
            inset = float(pget(params, "CORNER_INSET_M", 400.0))
            for cxm, cym in [(x0 + inset, y0 + inset),
                             (x0 + nx * dx - inset, y0 + inset),
                             (x0 + inset, y0 + ny * dx - inset),
                             (x0 + nx * dx - inset, y0 + ny * dx - inset)]:
                snapped = snap_to_clearance(cxm, cym, clr, x0, y0, dx, nx, ny,
                                            obst_clear, fill_sep)
                if snapped:
                    _try_add(snapped[0], snapped[1], "corner", force=True)

        # inner-core relief lattice at fill_sep pitch (its own grid, off the main nodes)
        if bool(pget(params, "FILL_INNER", True)):
            for yy in np.arange(y0 + fill_sep, y0 + ny * dx, fill_sep):
                for xx in np.arange(x0 + fill_sep, x0 + nx * dx, fill_sep):
                    jx = int(np.clip(round((xx - x0) / dx), 0, nx - 1))
                    iyy = int(np.clip(round((yy - y0) / dx), 0, ny - 1))
                    if boundary[iyy, jx]:                        # inner core only
                        continue
                    snapped = snap_to_clearance(xx, yy, clr, x0, y0, dx, nx, ny,
                                                obst_clear, fill_sep / 3.0)
                    if snapped:
                        _try_add(snapped[0], snapped[1], "inner-fill")
    for i, r in enumerate(relief, 1):
        r["fill_id"] = i
    n_inner_fill = sum(1 for r in relief if r["kind"] == "inner-fill")
    n_corner = sum(1 for r in relief if r["kind"] == "corner")
    print(f"Relief nodes  : {len(relief)}  (inner-fill {n_inner_fill}, corners {n_corner})"
          f"   spacing {fill_fac:.1f}x sep = {fill_sep:.0f} m")

    # ---- save ----
    np.savez_compressed(out_dir / "route_density.npz",
                        hit=hit, hit_center=hit_center, hit_cover=hit_cover,
                        density=density, x0=x0, y0=y0, dx=dx, thr=thr)
    pd.DataFrame(areas).to_csv(out_dir / "high_density_areas.csv", index=False)
    tn_cols = ["tn_id", "tn_class", "region", "rel_peak", "x", "y", "total_hits",
               "peak_density", "clearance_m", "from_area"]
    pd.DataFrame(tn, columns=tn_cols).to_csv(out_dir / "traffic_nodes.csv", index=False)
    # combined node network: objectives (terminals) + traffic nodes, one even layout
    net_rows = []
    for lbl, (ox_, oy_) in obj_xy.items():
        jj = int(np.clip(round((ox_ - x0) / dx), 0, nx - 1))
        ii = int(np.clip(round((oy_ - y0) / dx), 0, ny - 1))
        net_rows.append({"node_id": str(lbl), "type": "terminal",
                         "kind": "DB" if str(lbl).startswith("DB") else "DK",
                         "x": round(float(ox_), 1), "y": round(float(oy_), 1),
                         "region": "boundary" if bool(boundary[ii, jj]) else "inner"})
    for t in tn:
        net_rows.append({"node_id": f"TN{t['tn_id']}", "type": "traffic",
                         "kind": t["tn_class"], "x": t["x"], "y": t["y"],
                         "region": t["region"]})
    for r in relief:
        net_rows.append({"node_id": f"RN{r['fill_id']}", "type": "relief",
                         "kind": r["kind"], "x": r["x"], "y": r["y"],
                         "region": r["region"]})
    pd.DataFrame(net_rows, columns=["node_id", "type", "kind", "x", "y", "region"]
                 ).to_csv(out_dir / "network_nodes.csv", index=False)
    pd.DataFrame(relief, columns=["fill_id", "kind", "region", "x", "y", "clearance_m"]
                 ).to_csv(out_dir / "relief_nodes.csv", index=False)
    # master-plan input for 04_run_master_plan.py (src.routeplan_PSO_ACO):
    # it needs the FULL step-01 node model (every cell, with slowness so the
    # planner can build its no-fly clearance KD-tree and validate corridors),
    # NOT just candidates. We pass the whole step-01 .xyz through and only ADD:
    #   is_candidate    True at the TN/RN cells (the hubs the planner may use)
    #   route_hit_count corridor coverage at each cell (density weighting)
    # DB/DK objective labels are already in the .xyz 'label' column.
    mp = pd.read_csv(THIS_DIR / str(pget(params, "RISK_XYZ", "")), sep=r"\s+")
    mj = np.clip(np.rint((mp["x"].to_numpy(float) - x0) / dx).astype(int), 0, nx - 1)
    mi = np.clip(np.rint((mp["y"].to_numpy(float) - y0) / dx).astype(int), 0, ny - 1)

    def _cell(px, py):
        return (int(np.clip(round((py - y0) / dx), 0, ny - 1)),
                int(np.clip(round((px - x0) / dx), 0, nx - 1)))

    # candidate_type per cell: TN major -> "major"; TN minor + relief -> "minor".
    # (05_run_master_corridor_theta.py's TN buffer keys off major/minor radii;
    #  relief/bypass nodes act as secondary "minor" anchors.)
    cand_type: dict = {}
    cand_label: dict = {}
    for t in tn:
        c = _cell(t["x"], t["y"])
        cand_type[c] = "major" if t["tn_class"] == "major" else "minor"
        cand_label[c] = f"TN{t['tn_id']}"
    for r in relief:
        c = _cell(r["x"], r["y"])
        if c not in cand_type:
            cand_type[c] = "minor"
            cand_label[c] = f"RN{r['fill_id']}"
    mp["is_candidate"] = [(i, j) in cand_type for i, j in zip(mi, mj)]
    mp["candidate_type"] = [cand_type.get((i, j), "") for i, j in zip(mi, mj)]
    mp["candidate_id"] = [cand_label.get((i, j), "") for i, j in zip(mi, mj)]
    mp["route_hit_count"] = hit_cover[mi, mj].astype(int)
    mp.to_csv(out_dir / "master_plan_input_nodes.csv", index=False)
    print(f"Master input  : {len(mp)} model nodes, {int(mp['is_candidate'].sum())} candidates "
          f"({sum(1 for v in cand_type.values() if v == 'major')} major, "
          f"{sum(1 for v in cand_type.values() if v == 'minor')} minor)")
    iy, ix = np.where(covered)
    pd.DataFrame({"x": x0 + ix * dx, "y": y0 + iy * dx,
                  "routes_through": hit_center[iy, ix].astype(int),   # centreline votes
                  "buffer_cover": hit_cover[iy, ix].astype(int),      # band + radius votes
                  "density": np.round(density[iy, ix], 4)}
                 ).to_csv(out_dir / "node_hit_count.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps({
        "version": VERSION, "n_routes": n_routes, "include_alternatives": incl_alt,
        "density_model": {"buffer_density": buffer_density,
                          "band_half_m": round(band_half, 1),
                          "node_radius_m": node_radius,
                          "footprint_radius_m": round(foot_r, 1),
                          "extra_kde_sigma_m": sigma_m},
        "max_routes_through_cell": int(hit_center.max()),
        "max_buffer_cover_cell": int(hit_cover.max()),
        "threshold_mode": mode,
        "relative_threshold": {"inner": rel_inner, "boundary": rel_bound,
                               "boundary_margin_m": bnd_margin,
                               "reference_peak": round(float(d_ref), 3)} if mode == "relative" else None,
        "high_density_percentile": pct if mode == "percentile" else None,
        "inner_abs_threshold": round(thr, 4),
        "n_high_density_areas": len(areas), "n_traffic_nodes": len(tn),
        "n_tn_inner": sum(1 for t in tn if t["region"] == "inner"),
        "n_tn_boundary": sum(1 for t in tn if t["region"] == "boundary"),
        "n_objectives": len(obj_xy),
        "n_relief_nodes": len(relief), "n_relief_inner": n_inner_fill, "n_relief_corner": n_corner,
        "n_network_nodes": len(obj_xy) + len(tn) + len(relief),
        "objectives_as_nodes": True,
        "relief": {"enabled": bool(pget(params, "FILL_RELIEF_ENABLE", True)),
                   "spacing_factor": fill_fac, "spacing_m": round(fill_sep, 1),
                   "corner_inset_m": float(pget(params, "CORNER_INSET_M", 400.0))},
        "tn": {"objective_merge_m": merge_m, "obstacle_clearance_m": obst_clear,
               "min_separation_m": sep_m, "major_top_k": major_k},
        "grid": {"nx": nx, "ny": ny, "dx_m": dx},
    }, indent=2))
    print(f"Saved         : route_density.npz, high_density_areas.csv, traffic_nodes.csv, "
          f"relief_nodes.csv, network_nodes.csv, master_plan_input_nodes.csv, node_hit_count.csv")

    # ---- figures ----
    if make_fig:
        def base(ax, field, title, cmap):
            fld = np.where(nofly, np.nan, field)
            im = ax.imshow(fld, origin="lower", extent=extent, cmap=cmap, zorder=1)
            if mode == "relative":                         # inner-city / near-boundary divide
                ax.add_patch(plt.Rectangle(
                    (x0 + bnd_margin, y0 + bnd_margin),
                    nx * dx - 2 * bnd_margin, ny * dx - 2 * bnd_margin,
                    fill=False, ls="--", ec="#00e5ff", lw=1.4, zorder=7))
                ax.text(x0 + bnd_margin + 30, y0 + ny * dx - bnd_margin - 60,
                        f"inner  (rel>= {rel_inner:.2f})", color="#00b8cc",
                        fontsize=8, zorder=7)
                ax.text(x0 + 40, y0 + 40, f"boundary belt  (rel>= {rel_bound:.2f})",
                        color="#00b8cc", fontsize=8, zorder=7)
            for nid, (x, y) in obj_xy.items():
                is_db = str(nid).startswith("DB")
                ax.scatter([x], [y], s=55, marker="s" if is_db else "^",
                           c="#c0392b" if is_db else "#1f6f3f", edgecolors="k",
                           linewidths=0.5, zorder=6)
            for t in tn:                                   # traffic nodes (major/minor)
                mj = t["tn_class"] == "major"
                ax.scatter([t["x"]], [t["y"]], s=240 if mj else 130, marker="o",
                           facecolors="none", edgecolors="#ff8c00" if mj else "#ffd400",
                           linewidths=2.4 if mj else 1.6, zorder=8)
                ax.annotate(f"TN{t['tn_id']}", (t["x"], t["y"]),
                            textcoords="offset points", xytext=(6, 4), fontsize=8,
                            weight="bold", color="#ff8c00" if mj else "#8a6d00", zorder=9)
            for r in relief:                               # relief / bypass fill nodes
                is_corner = r["kind"] == "corner"
                ax.scatter([r["x"]], [r["y"]], s=170 if is_corner else 90, marker="D",
                           facecolors="none", edgecolors="#d000d0",
                           linewidths=2.2 if is_corner else 1.4, zorder=8)
                ax.annotate(f"RN{r['fill_id']}", (r["x"], r["y"]),
                            textcoords="offset points", xytext=(5, 4), fontsize=7,
                            color="#a000a0", zorder=9)
            ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
            ax.set_title(title)
            return im

        fig, ax = plt.subplots(figsize=(12, 11))
        im = base(ax, density,
                  f"Route density (route + buffer band +-{band_half:.0f} m + r{node_radius:.0f} m) "
                  f"+ high-density areas", "viridis")
        fig.colorbar(im, ax=ax, shrink=0.75).set_label("corridor coverage (routes through band+radius)")
        fig.tight_layout(); fig.savefig(out_dir / "figures" / "00_route_density.png", dpi=130)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 11))
        _ttl = (f"Traffic-node areas (rel>= {rel_inner:.2f} inner / {rel_bound:.2f} boundary)"
                if mode == "relative" else f"High-density areas (density >= P{pct:.0f})")
        im = base(ax, np.where(mask, density, np.nan), _ttl, "hot")
        fig.colorbar(im, ax=ax, shrink=0.75).set_label("density (high-density cells only)")
        fig.tight_layout(); fig.savefig(out_dir / "figures" / "01_high_density_areas.png", dpi=130)
        plt.close(fig)
        print(f"Figures       : figures/00_route_density.png, figures/01_high_density_areas.png")
    print(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
