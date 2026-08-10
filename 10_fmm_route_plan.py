#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_fmm_route_plan.py

Risk/conflict-aware route planning with the FAST MARCHING METHOD (FMM).

A new, standalone planning stage (steps 02/06/07 stay intact for comparison).
It builds a weighted Eikonal cost field over the 2D map and, for every
objective pair, plans the route that MINIMISES a blend of travel time, ground
RISK and traffic CONFLICT:

    cost = W_TIME*1 + W_RISK*risk_norm(step01) + W_CONFLICT*conflict_norm(step08)

FMM propagates the accumulated cost from each destination (one solve per
destination); each origin's route is the steepest-descent path back down that
field.  No-fly cells (step 01) are obstacles.

Inputs
------
    output/01_random_node_map/*.xyz     risk_total + slowness (no-fly) grid
    output/08_costmap/slowness_costmap.npz   traffic-density slowness (step 08)
    output/07_corridor_network/*         objectives + pair list (+ lanes overlay)

Outputs (output/10_fmm_routes/)
-------------------------------
    fmm_routes.csv        pair, seq, x, y, risk, conflict per route point
    route_summary.csv     per-pair length / mean risk / mean conflict / cost
    metrics.json          run-level summary + weights
    cost_field.png        weighted impedance + all routes
    risk_field.png        risk term + routes
    conflict_field.png    conflict term + routes
    arrival_field.png     one example FMM arrival-cost field

Run
---
    python 10_fmm_route_plan.py
    python 10_fmm_route_plan.py --param-file params/fmm_route_plan.params \\
        --w-risk 3 --w-conflict 2 --pairs all
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

from src.fmm import eikonal_fmm, backtrack
from src.maprule import add_map_rule

THIS_DIR = Path(__file__).resolve().parent
VERSION = "v1"


# ----------------------------------------------------------------------
# params helpers (same tiny format as the other stages)
# ----------------------------------------------------------------------
def parse_value(v: str):
    s = v.strip()
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def load_params(path: Path) -> dict:
    params: dict = {}
    if not path.exists():
        return params
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        params[k.strip()] = parse_value(v.strip())
    return params


def pget(params, key, default):
    return params.get(key, default)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FMM risk/conflict-aware route planner.")
    p.add_argument("--param-file", default="params/fmm_route_plan.params")
    p.add_argument("--w-time", type=float, default=None)
    p.add_argument("--w-risk", type=float, default=None)
    p.add_argument("--w-conflict", type=float, default=None)
    p.add_argument("--pairs", choices=["corridor", "all"], default=None)
    p.add_argument("--no-figures", action="store_true")
    return p.parse_args()


# ----------------------------------------------------------------------
# grid loading
# ----------------------------------------------------------------------
def load_risk_grid(xyz_path: Path):
    """Return (risk[ny,nx], slw01[ny,nx], x0, y0, dx, nx, ny) from a step-01
    node riskmap .xyz (regular grid)."""
    df = pd.read_csv(xyz_path, sep=r"\s+")
    xs = np.sort(df["x"].unique())
    ys = np.sort(df["y"].unique())
    dx = float(np.median(np.diff(xs)))
    x0, y0 = float(xs[0]), float(ys[0])
    nx, ny = len(xs), len(ys)
    ix = np.rint((df["x"].to_numpy() - x0) / dx).astype(int)
    iy = np.rint((df["y"].to_numpy() - y0) / dx).astype(int)
    risk = np.zeros((ny, nx), float)
    slw01 = np.zeros((ny, nx), float)
    risk[iy, ix] = df["risk_total"].to_numpy(float)
    slw01[iy, ix] = df["slowness"].to_numpy(float)
    return risk, slw01, x0, y0, dx, nx, ny


def resample_costmap(npz_path: Path, x0, y0, dx, nx, ny):
    """Sample step-09 slowness onto the risk grid (nearest). Missing -> 1.0."""
    if not npz_path.exists():
        return np.ones((ny, nx), float)
    z = np.load(npz_path)
    g = z["slowness"].astype(float)
    cx0, cy0, cres = float(z["x0"]), float(z["y0"]), float(z["res"])
    cny, cnx = g.shape
    xs = x0 + dx * np.arange(nx)
    ys = y0 + dx * np.arange(ny)
    gx, gy = np.meshgrid(xs, ys)
    cix = np.clip(((gx - cx0) / cres).astype(int), 0, cnx - 1)
    ciy = np.clip(((gy - cy0) / cres).astype(int), 0, cny - 1)
    return g[ciy, cix]


def _norm(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Min-max normalise ``a`` to [0,1] using only ``mask`` cells for the range."""
    v = a[mask]
    if v.size == 0:
        return np.zeros_like(a)
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def snap_to_flyable(ix, iy, passable, nx, ny, max_r=12):
    """Nearest passable cell to (ix,iy) within a small radius."""
    if passable[iy, ix]:
        return iy, ix
    for r in range(1, max_r + 1):
        best = None
        best_d = 1e18
        for a in range(max(0, iy - r), min(ny, iy + r + 1)):
            for b in range(max(0, ix - r), min(nx, ix + r + 1)):
                if passable[a, b]:
                    d = (a - iy) ** 2 + (b - ix) ** 2
                    if d < best_d:
                        best_d, best = d, (a, b)
        if best is not None:
            return best
    return iy, ix


def smooth_xy(xy: np.ndarray, win: int) -> np.ndarray:
    if win is None or win < 2 or len(xy) < win:
        return xy
    k = np.ones(win) / win
    x = np.convolve(xy[:, 0], k, mode="same")
    y = np.convolve(xy[:, 1], k, mode="same")
    x[0], y[0], x[-1], y[-1] = xy[0, 0], xy[0, 1], xy[-1, 0], xy[-1, 1]
    return np.column_stack([x, y])


# ----------------------------------------------------------------------
# plotting
# ----------------------------------------------------------------------
def draw_objectives(ax, obj_xy):
    for nid, (x, y) in obj_xy.items():
        is_db = str(nid).startswith("DB")
        ax.scatter([x], [y], s=70, marker="s" if is_db else "^",
                   c="#c0392b" if is_db else "#1f6f3f", edgecolors="k",
                   linewidths=0.6, zorder=8)
        ax.annotate(str(nid), (x, y), textcoords="offset points",
                    xytext=(5, 4), fontsize=7, weight="bold", zorder=9)


def field_figure(field, extent, routes, obj_xy, title, cbar_label, cmap, out_png):
    fig, ax = plt.subplots(figsize=(12, 11))
    im = ax.imshow(field, origin="lower", extent=extent, cmap=cmap, zorder=1)
    for xy in routes.values():
        ax.plot(xy[:, 0], xy[:, 1], "-", color="#1020ff", lw=1.1, alpha=0.85, zorder=6)
    draw_objectives(ax, obj_xy)
    fig.colorbar(im, ax=ax, shrink=0.75).set_label(cbar_label)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    add_map_rule(ax, extent[0], extent[2], extent[1], extent[3])
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    params = load_params(THIS_DIR / args.param_file)

    w_time = args.w_time if args.w_time is not None else float(pget(params, "W_TIME", 1.0))
    w_risk = args.w_risk if args.w_risk is not None else float(pget(params, "W_RISK", 2.0))
    w_conf = args.w_conflict if args.w_conflict is not None else float(pget(params, "W_CONFLICT", 1.5))
    pair_source = args.pairs or str(pget(params, "PAIR_SOURCE", "corridor"))
    make_fig = (not args.no_figures) and bool(pget(params, "MAKE_FIGURES", True))

    corridor_dir = THIS_DIR / str(pget(params, "CORRIDOR_DIR", "output/07_corridor_network"))
    out_dir = THIS_DIR / str(pget(params, "OUT_DIR", "output/10_fmm_routes"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 66)
    print(f"10_fmm_route_plan.py  {VERSION}")
    print(f"Weights       : time={w_time}  risk={w_risk}  conflict={w_conf}")
    print(f"Pairs         : {pair_source}")
    print(f"Output dir    : {out_dir}")
    print("=" * 66)

    # ---- risk grid + no-fly ----
    risk, slw01, x0, y0, dx, nx, ny = load_risk_grid(
        THIS_DIR / str(pget(params, "RISK_XYZ", "")))
    nofly_slw = float(pget(params, "NOFLY_SLOWNESS", 10.0))
    nofly = slw01 >= nofly_slw
    passable = ~nofly
    extent = [x0, x0 + nx * dx, y0, y0 + ny * dx]
    print(f"Grid          : {nx} x {ny} @ {dx:.0f} m  "
          f"({int(nofly.sum())} no-fly / {nx*ny} cells)")

    # ---- conflict term from step-09 costmap ----
    slw08 = resample_costmap(
        THIS_DIR / str(pget(params, "COST_MAP_FILE", "")), x0, y0, dx, nx, ny)
    conflict = (1.0 - slw08) if bool(pget(params, "CONFLICT_FROM_COSTMAP", True)) \
        else np.zeros((ny, nx))

    # ---- risk field: raw step-01 risk PLUS a decaying hazard buffer ----
    # risk_total is non-zero only inside no-fly/RA cells, so on the flyable
    # area it is flat 0 and gives FMM nothing to steer by. Add a proximity
    # risk that is 1 at a no-fly boundary and decays with distance, so routes
    # keep clearance from hazards (RISK_BUFFER_M sets the decay length).
    buf_m = float(pget(params, "RISK_BUFFER_M", 300.0))
    if buf_m > 0:
        dist_m = distance_transform_edt(passable) * dx
        risk_prox = np.exp(-dist_m / buf_m)
    else:
        risk_prox = np.zeros((ny, nx))
    risk_field = np.maximum(risk, risk_prox)

    # ---- normalised, weighted impedance field ----
    r_hat = _norm(risk_field, passable)
    c_hat = _norm(conflict, passable)
    floor = float(pget(params, "COST_FLOOR", 0.05))
    cost = w_time * 1.0 + w_risk * r_hat + w_conf * c_hat + floor
    if bool(pget(params, "NOFLY_AS_OBSTACLE", True)):
        cost[nofly] = np.inf
    else:
        cost[nofly] += float(pget(params, "NOFLY_PENALTY", 50.0))

    # ---- objectives -> grid cells ----
    nodes = pd.read_csv(corridor_dir / "network_nodes.csv")
    obj = nodes[nodes["kind"] == "objective"]
    obj_xy = {str(r.net_id): (float(r.x), float(r.y)) for r in obj.itertuples()}
    obj_ij = {}
    for nid, (x, y) in obj_xy.items():
        ix = int(np.clip(round((x - x0) / dx), 0, nx - 1))
        iy = int(np.clip(round((y - y0) / dx), 0, ny - 1))
        obj_ij[nid] = snap_to_flyable(ix, iy, passable, nx, ny)

    # ---- pair list ----
    if pair_source == "all":
        ids = list(obj_xy)
        pairs = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1:]]
        corridor_len = {}
    else:
        pr = pd.read_csv(corridor_dir / "pair_routes.csv")
        pairs, corridor_len = [], {}
        for r in pr.itertuples():
            a, b = str(r.pair).split("_to_")
            pairs.append((a, b))
            corridor_len[(a, b)] = float(r.length_m)
    print(f"Objectives    : {len(obj_xy)}   pairs to plan: {len(pairs)}")

    # ---- FMM once per destination, backtrack every origin ----
    win = int(pget(params, "ROUTE_SMOOTH_WIN", 3))
    dests = sorted({b for _, b in pairs})
    T_cache = {}
    for b in dests:
        src = np.zeros((ny, nx), bool)
        src[obj_ij[b]] = True
        T_cache[b] = eikonal_fmm(cost, src, dx)

    routes_xy = {}
    rows_pts, rows_sum = [], []
    n_ok = 0
    for (a, b) in pairs:
        path = backtrack(T_cache[b], obj_ij[a])
        if len(path) < 2:
            print(f"  ! {a}->{b}: unreachable")
            continue
        n_ok += 1
        ij = np.array(path)
        xy = np.column_stack([x0 + ij[:, 1] * dx, y0 + ij[:, 0] * dx]).astype(float)
        seg = np.hypot(*np.diff(xy, axis=0).T)
        length_m = float(seg.sum())
        rk = risk_field[ij[:, 0], ij[:, 1]]
        cf = conflict[ij[:, 0], ij[:, 1]]
        pair = f"{a}_to_{b}"
        routes_xy[pair] = smooth_xy(xy, win)
        for s, (px, py) in enumerate(xy):
            rows_pts.append({"pair": pair, "seq": s, "x": round(px, 2),
                             "y": round(py, 2), "risk": round(float(rk[s]), 4),
                             "conflict": round(float(cf[s]), 4)})
        rows_sum.append({
            "pair": pair, "n_pts": len(xy), "length_m": round(length_m, 1),
            "mean_risk": round(float(rk.mean()), 4),
            "max_risk": round(float(rk.max()), 4),
            "mean_conflict": round(float(cf.mean()), 4),
            "fmm_cost": round(float(T_cache[b][obj_ij[a]]), 3),
            "corridor_length_m": round(corridor_len.get((a, b), float("nan")), 1),
        })

    pd.DataFrame(rows_pts).to_csv(out_dir / "fmm_routes.csv", index=False)
    summ = pd.DataFrame(rows_sum)
    summ.to_csv(out_dir / "route_summary.csv", index=False)
    print(f"Planned       : {n_ok}/{len(pairs)} routes")
    if len(summ):
        print(f"Route length  : mean {summ['length_m'].mean():.0f} m  "
              f"(corridor mean {summ['corridor_length_m'].mean():.0f} m)")
        print(f"Risk exposure : mean {summ['mean_risk'].mean():.3f}  "
              f"max {summ['max_risk'].max():.3f}")
        print(f"Conflict expo : mean {summ['mean_conflict'].mean():.3f}")

    metrics = {
        "version": VERSION,
        "weights": {"time": w_time, "risk": w_risk, "conflict": w_conf},
        "grid": {"nx": nx, "ny": ny, "dx_m": dx,
                 "map_w_m": nx * dx, "map_h_m": ny * dx},
        "n_nofly_cells": int(nofly.sum()),
        "n_pairs": len(pairs),
        "n_routes": n_ok,
        "pair_source": pair_source,
        "nofly_as_obstacle": bool(pget(params, "NOFLY_AS_OBSTACLE", True)),
        "mean_route_length_m": None if not len(summ) else round(float(summ["length_m"].mean()), 1),
        "mean_corridor_length_m": None if not len(summ) else round(float(summ["corridor_length_m"].mean()), 1),
        "mean_risk_exposure": None if not len(summ) else round(float(summ["mean_risk"].mean()), 4),
        "mean_conflict_exposure": None if not len(summ) else round(float(summ["mean_conflict"].mean()), 4),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved         : {out_dir/'fmm_routes.csv'}, route_summary.csv, metrics.json")

    # ---- figures ----
    if make_fig:
        cost_disp = np.where(np.isfinite(cost), cost, np.nan)
        field_figure(cost_disp, extent, routes_xy, obj_xy,
                     f"FMM weighted cost field  (time {w_time} / risk {w_risk} / conflict {w_conf})  + routes",
                     "impedance (normalised, no-fly = white)", "viridis",
                     out_dir / "cost_field.png")
        field_figure(np.where(passable, r_hat, np.nan), extent, routes_xy, obj_xy,
                     "Risk term (step 01) + FMM routes", "risk (normalised)",
                     "Reds", out_dir / "risk_field.png")
        field_figure(np.where(passable, c_hat, np.nan), extent, routes_xy, obj_xy,
                     "Conflict term (step 08 traffic) + FMM routes", "conflict (normalised)",
                     "Oranges", out_dir / "conflict_field.png")
        b0 = dests[0]
        Td = T_cache[b0]
        field_figure(np.where(np.isfinite(Td), Td, np.nan), extent,
                     {k: v for k, v in routes_xy.items() if k.endswith(f"_to_{b0}")},
                     obj_xy, f"FMM arrival-cost field to {b0}", "accumulated cost",
                     "magma", out_dir / "arrival_field.png")
        print(f"Figures       : cost_field.png, risk_field.png, conflict_field.png, arrival_field.png")
    print(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
