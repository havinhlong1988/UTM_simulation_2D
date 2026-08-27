#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
engine_route_plan.py

Risk/conflict-aware route planning over a weighted 2D cost field.

Runs directly with the PARAMETERS embedded in the header below -- no params file
needed:

    python engine_route_plan.py
    python engine_route_plan.py --planner fmm --diversify-k 4
    python engine_route_plan.py --planner theta --diversify-k 3 --pairs all

Precedence: CLI flag > --param-file (optional) > the PARAMETERS block here. Edit
the PARAMETERS dict below to change the run defaults.

It builds a weighted Eikonal impedance field over the map and, for every
objective pair, plans the route that MINIMISES a blend of travel time, ground
RISK and traffic CONFLICT:

    cost = W_TIME*1 + W_RISK*risk_norm(step01) + W_CONFLICT*conflict_norm(step08)

SELECTABLE SOLVER (PLANNER):
    "fmm"   -- Fast Marching: propagate accumulated cost from the destination,
               steepest-descent path back down the field (smooth, field-following).
    "theta" -- Theta* (any-angle A* on the SAME cost field, line-of-sight string
               pull) -- piecewise-linear least-cost paths, for comparison.

LOCK-AND-RE-SEARCH (DIVERSIFY_K > 1): after a path is found it is "locked" (a
usage penalty is stamped in a corridor around it), then the search is repeated so
each alternative is pushed onto a FRESH route. LOCK_SCOPE="global" carries the
usage across ALL pairs (spreads the whole network so high-density directions get
parallel corridors, incl. along the border ring); "per_pair" gives each pair K
spatially-separated alternatives. Works with either solver. No-fly cells are
obstacles.

This is the FIRST planning stage: it plans on the COST-FREE free-space volume of
step 01 (risk + travel time only) to generate an efficient route network -- the
seed the corridor network (07) is then built from. It needs ONLY step 01; the
step-07 corridor network and the step-08 traffic costmap do not exist yet and are
optional (used only if present, e.g. on a later re-plan).

Inputs
------
    output/01_random_node_map/*.xyz     risk_total + slowness grid AND the DB/DK
                                        objective points (from the label column)
    output/07_costmap/slowness_costmap.npz   OPTIONAL traffic-conflict term (08)

Outputs (OUT_DIR)
-----------------
    route_points.csv      pair, alt, seq, x, y, risk, conflict per route point
    route_summary.csv     per-pair length / mean risk / mean conflict
    metrics.json          run-level summary + planner/lock config + weights
    cost_field.png        weighted impedance + all routes
    risk_field.png        risk term + routes
    conflict_field.png    conflict term + routes
    arrival_field.png     one example FMM arrival-cost field
    route_network.html    interactive map: pan/zoom, toggle obstacles/corridor/
                          buffer bands, filter by pair, hover for route info
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

# This engine lives in src/ but is anchored on the PROJECT ROOT: every path it
# resolves -- params files, output trees, and the root-level stage scripts -- is
# written relative to the root, not to src/. Put the root on sys.path too, so
# `from src.x import y` works whether this file is run through its launcher
# (runpy from the root) or directly as `python src/engine_*.py`.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.fmm import eikonal_fmm, backtrack
from src.maprule import add_map_rule
from src.route_html import render_route_html, load_candidate_nodes

THIS_DIR = _ROOT          # project root: all params/output paths hang off it
VERSION = "v2"

# ======================================================================
# PARAMETERS  (run defaults -- edit here; overridden by --param-file / CLI)
# ======================================================================
PARAMETERS: dict = {
    # ---- inputs / outputs ----
    # step 01 is the ONLY required input (risk grid + DB/DK objectives from labels)
    "RISK_XYZ":      "output/01_random_node_map/random_2d_node_riskmap_seed2102359706.xyz",
    # No step-08 traffic costmap yet: the costmap is UNIFORM (same slowness for
    # every node), which is routing-neutral -> this first plan is on the cost-free
    # volume. Point COST_MAP_FILE at a real step-08 .npz + turn CONFLICT_FROM_COSTMAP
    # on later to use the traffic map instead.
    "COSTMAP_UNIFORM_SLOWNESS": 1.0,   # uniform slowness for all nodes (1.0 = full speed)
    "COST_MAP_FILE": "",               # optional later step-08 costmap (unused now)
    "CORRIDOR_DIR":  "",               # unused now (step 07 not built yet)
    "OUT_DIR":       "output/_engine_default/02_route_plan",
    "PAIR_SOURCE":   "corridor",     # "corridor" = every DB->DK pair | "all" = every objective pair
    # ---- cost-field weights ----
    "W_TIME":     1.0,               # travel time / path length
    "W_RISK":     2.0,               # obstacle + RA risk exposure (step 01)
    "W_CONFLICT": 1.5,               # traffic-density / conflict exposure (step 08)
    "COST_FLOOR": 0.05,              # min impedance on a free cell (FMM needs cost > 0)
    "RISK_BUFFER_M": 300.0,          # decay length of the hazard-proximity risk (0 = off)
    # ---- corridor geometry: each route is a BAND, not a bare line ----
    "ROUTE_WIDTH_M":  50.0,          # usable corridor width (centreline +- WIDTH/2 = 25 m)
    "ROUTE_BUFFER_M": 12.5,          # safety buffer OUTSIDE the width -> half-extent 37.5 m
    # a centreline is only allowed where WIDTH/2 + BUFFER of clearance to no-fly
    # exists, so the whole corridor + buffer fits in free space.
    "NOFLY_AS_OBSTACLE": True,       # True: no-fly = infinite cost; False: add NOFLY_PENALTY
    "NOFLY_SLOWNESS":    10.0,       # step-01 slowness >= this marks a no-fly cell
    "NOFLY_PENALTY":     50.0,       # soft no-fly cost when NOFLY_AS_OBSTACLE is False
    "CONFLICT_FROM_COSTMAP": False,  # OFF: plan on the cost-free volume (risk+time only).
                                     # ON needs the step-08 costmap (a later re-plan).
    "ROUTE_SMOOTH_WIN":  3,          # moving-average window (cells) for plotted routes; 0/1 = raw
    "MAKE_FIGURES":      True,
    "MAKE_HTML":         True,        # interactive route_network.html (pan/zoom + toggles)
    # ---- planner selection ----
    "PLANNER":    "fmm",             # "fmm" (Eikonal field) | "theta" (any-angle A*/Theta*)
    # ---- network build order + crossing control ----
    # Plan the PREDICTED-BUSIEST corridors first (find first -> lock first), so the
    # trunk routes claim the straightest paths and later routes detour around them.
    "PRIORITIZE_BY_DENSITY": True,
    # penalty (added cost per existing corridor) a new route pays for CROSSING a
    # corridor already laid down -> it takes a longer detour instead. 0 disables.
    "CROSS_PENALTY": 12.0,
    # ---- lock-and-re-search (path diversification) ----
    "DIVERSIFY_K":      8,           # alternatives per pair (1 = single path, off)
    "LOCK_PENALTY":     4.0,         # soft-lock: cost added per unit usage along a locked route
    "LOCK_HALFWIDTH_M": 75.0,       # half-width (m) of the locked corridor stamped into usage
    "LOCK_MODE":        "soft",      # "soft" (penalty) | "hard" (mask the corridor as no-fly)
    "LOCK_SCOPE":       "global",    # "global" (spread ALL pairs) | "per_pair" (K per pair)
}


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
    p = argparse.ArgumentParser(
        description="Risk/conflict-aware route planner (FMM or Theta*, with "
                    "optional lock-and-re-search diversification).")
    p.add_argument("--param-file", default=None,
                   help="optional params file that overrides the header PARAMETERS.")
    p.add_argument("--w-time", type=float, default=None)
    p.add_argument("--w-risk", type=float, default=None)
    p.add_argument("--w-conflict", type=float, default=None)
    p.add_argument("--pairs", choices=["corridor", "all"], default=None)
    p.add_argument("--planner", choices=["fmm", "theta"], default=None,
                   help="single-path solver: FMM field or Theta* (any-angle A*).")
    p.add_argument("--diversify-k", type=int, default=None,
                   help="alternatives per pair via lock-and-re-search (1 = off).")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--no-html", action="store_true")
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


def load_objectives(xyz_path: Path, prefixes=("DB", "DK")) -> dict:
    """Read the routing objectives straight from the step-01 riskmap .xyz -- the
    nodes whose `label` starts with one of `prefixes` (DB bases, DK docks). This
    is what lets step 02 run right after step 01, before any corridor network."""
    df = pd.read_csv(xyz_path, sep=r"\s+")
    lab = df["label"].astype(str)
    sel = df[lab.str.startswith(tuple(prefixes))]
    return {str(r.label): (float(r.x), float(r.y)) for r in sel.itertuples()}


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


def _clearance_at_pts(pts, clearance_m, x0, y0, dx):
    """Sample the clearance field (distance to the no-fly FOOTPRINT, m) at each
    (x, y) point via nearest cell."""
    ny, nx = clearance_m.shape
    ci = np.clip(np.rint((pts[:, 1] - y0) / dx).astype(int), 0, ny - 1)
    cj = np.clip(np.rint((pts[:, 0] - x0) / dx).astype(int), 0, nx - 1)
    return clearance_m[ci, cj]


def smooth_xy(xy: np.ndarray, win: int, clearance_m=None, req_clear: float = 0.0,
              x0: float = 0.0, y0: float = 0.0, dx: float = 1.0) -> np.ndarray:
    """Moving-average smooth of a route centreline.

    When ``clearance_m`` (distance from each cell to the no-fly FOOTPRINT, in m)
    is supplied, any smoothed vertex -- or segment midpoint -- that would pull
    the corridor closer than ``req_clear`` to an obstacle is reverted to its raw
    (planned, guaranteed-clear) position. Smoothing therefore can never cut a
    corner into a no-fly zone, so the corridor + buffer band stays clear."""
    xy = np.asarray(xy, float)
    if win is None or win < 2 or len(xy) < win:
        return xy
    k = np.ones(win) / win
    x = np.convolve(xy[:, 0], k, mode="same")
    y = np.convolve(xy[:, 1], k, mode="same")
    x[0], y[0], x[-1], y[-1] = xy[0, 0], xy[0, 1], xy[-1, 0], xy[-1, 1]
    sm = np.column_stack([x, y])
    if clearance_m is None or req_clear <= 0 or len(sm) != len(xy):
        return sm
    # revert vertices that lose the required clearance ...
    bad = _clearance_at_pts(sm, clearance_m, x0, y0, dx) < req_clear
    sm[bad] = xy[bad]
    # ... then also any segment whose midpoint dips below it (corner-cut guard)
    for _ in range(4):
        mids = 0.5 * (sm[:-1] + sm[1:])
        seg_bad = _clearance_at_pts(mids, clearance_m, x0, y0, dx) < req_clear
        if not seg_bad.any():
            break
        nb = np.zeros(len(sm), bool)
        nb[:-1] |= seg_bad
        nb[1:] |= seg_bad
        sm[nb] = xy[nb]
    return sm


# ----------------------------------------------------------------------
# Theta* (any-angle A*) planner over the SAME weighted cost field as FMM.
# Selectable alternative to FMM: A* on the 8-connected grid (edge cost = mean
# cell impedance x segment length), then a line-of-sight "string pull" so the
# path is any-angle / piecewise-linear (Theta*-style), then rasterised back to a
# dense cell list so it can be scored and stamped exactly like an FMM path.
# ----------------------------------------------------------------------
_SQ2 = math.sqrt(2.0)
_NB8 = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, _SQ2), (-1, 1, _SQ2), (1, -1, _SQ2), (1, 1, _SQ2)]


def _astar_grid(cost, passable, src, dst, dx):
    ny, nx = cost.shape
    si, sj = src
    di, dj = dst
    fin = cost[np.isfinite(cost)]
    cmin = float(fin.min()) if fin.size else 0.0          # admissible heuristic scale
    def h(i, j):
        return math.hypot(i - di, j - dj) * dx * cmin
    g = {src: 0.0}
    came: dict = {}
    pq = [(h(si, sj), 0.0, src)]
    while pq:
        _f, gu, u = heapq.heappop(pq)
        if u == dst:
            break
        if gu > g.get(u, 1e18):
            continue
        ui, uj = u
        cu = cost[ui, uj]
        for dii, djj, step in _NB8:
            vi, vj = ui + dii, uj + djj
            if vi < 0 or vj < 0 or vi >= ny or vj >= nx or not passable[vi, vj]:
                continue
            cv = cost[vi, vj]
            if not (np.isfinite(cu) and np.isfinite(cv)):
                continue
            ng = gu + 0.5 * (cu + cv) * step * dx
            v = (vi, vj)
            if ng < g.get(v, 1e18):
                g[v] = ng
                came[v] = u
                heapq.heappush(pq, (ng + h(vi, vj), ng, v))
    if dst != src and dst not in came:
        return []
    path = [dst]
    cur = dst
    while cur != src:
        cur = came.get(cur)
        if cur is None:
            return []
        path.append(cur)
    path.reverse()
    return path


def _line_cells(a, b):
    """Bresenham cells from a to b inclusive."""
    (i0, j0), (i1, j1) = a, b
    di, dj = abs(i1 - i0), abs(j1 - j0)
    si = 1 if i1 > i0 else -1
    sj = 1 if j1 > j0 else -1
    i, j = i0, j0
    err = di - dj
    out = []
    while True:
        out.append((i, j))
        if i == i1 and j == j1:
            return out
        e2 = 2 * err
        if e2 > -dj:
            err -= dj
            i += si
        if e2 < di:
            err += di
            j += sj


def _los(passable, a, b):
    for (i, j) in _line_cells(a, b):
        if not passable[i, j]:
            return False
    return True


def _string_pull(path, passable):
    """Drop intermediate vertices whose skip stays in line-of-sight (Theta*)."""
    if len(path) < 3:
        return path
    out = [path[0]]
    anchor = 0
    for i in range(1, len(path) - 1):
        if not _los(passable, path[anchor], path[i + 1]):
            out.append(path[i])
            anchor = i
    out.append(path[-1])
    return out


def _rasterize(vertices):
    if not vertices:
        return []
    dense = [vertices[0]]
    for a, b in zip(vertices, vertices[1:]):
        dense.extend(_line_cells(a, b)[1:])
    return dense


def plan_theta(cost, passable, src, dst, dx):
    p = _astar_grid(cost, passable, src, dst, dx)
    if len(p) < 2:
        return []
    return _rasterize(_string_pull(p, passable))


def plan_one(cost_eff, src_ij, dst_ij, planner, passable, dx):
    """Dispatch a single origin->dest path on the current (possibly locked) cost
    field. FMM: propagate from the destination, steepest-descent from the origin.
    Theta*: any-angle A* on the same field. Returns a dense [(i,j), ...] path."""
    if planner == "theta":
        return plan_theta(cost_eff, passable, src_ij, dst_ij, dx)
    src = np.zeros(cost_eff.shape, bool)
    src[dst_ij] = True
    T = eikonal_fmm(cost_eff, src, dx)
    return backtrack(T, src_ij)


def stamp_usage(usage, path_ij, dx, halfwidth_m):
    """'Lock' a found path: raise a usage penalty in a corridor of half-width
    halfwidth_m around it, so the next search is steered onto a fresh route."""
    if not path_ij:
        return
    pm = np.zeros(usage.shape, bool)
    for (i, j) in path_ij:
        pm[i, j] = True
    d = distance_transform_edt(~pm) * dx
    usage[d <= halfwidth_m] += 1.0


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


def _corridor_polygon(xy, half_m):
    """Closed left+right offset polygon of a polyline at +-half_m metres."""
    xy = np.asarray(xy, float)
    if len(xy) < 2 or half_m <= 0:
        return None
    d = np.diff(xy, axis=0)
    seglen = np.hypot(d[:, 0], d[:, 1])
    seglen[seglen < 1e-9] = 1e-9
    snx, sny = -d[:, 1] / seglen, d[:, 0] / seglen        # segment unit normals
    vn = np.zeros_like(xy)                                # per-vertex normal (avg)
    vn[:-1, 0] += snx; vn[:-1, 1] += sny
    vn[1:, 0] += snx; vn[1:, 1] += sny
    ln = np.hypot(vn[:, 0], vn[:, 1]); ln[ln < 1e-9] = 1e-9
    vn[:, 0] /= ln; vn[:, 1] /= ln
    left = xy + vn * half_m
    right = xy - vn * half_m
    return np.vstack([left, right[::-1]])


def field_figure(field, extent, routes, obj_xy, title, cbar_label, cmap, out_png,
                 corridor_half_m=0.0, buffer_half_m=0.0, emphasize=None):
    """Plot ``field`` with every route overlaid. If ``emphasize`` is a set of
    route keys, those keys are drawn bold blue and the rest are dimmed gray --
    used by the arrival-cost figure to show the WHOLE network (same count as the
    other fields) while highlighting the routes that actually descend the field.
    ``emphasize=None`` keeps the original behaviour (all routes bold blue)."""
    fig, ax = plt.subplots(figsize=(12, 11))
    im = ax.imshow(field, origin="lower", extent=extent, cmap=cmap, zorder=1)
    for key, xy in routes.items():
        emph = (emphasize is None) or (key in emphasize)
        col = "#1020ff" if emph else "#9aa0a6"
        line_a = 0.9 if emph else 0.30
        line_w = 1.0 if emph else 0.6
        line_z = 6 if emph else 3
        if buffer_half_m > 0:                            # outer buffer band (faint)
            poly = _corridor_polygon(xy, buffer_half_m)
            if poly is not None:
                ax.fill(poly[:, 0], poly[:, 1], color="#1020ff", alpha=0.10, lw=0, zorder=4)
        if corridor_half_m > 0:                          # usable corridor width
            poly = _corridor_polygon(xy, corridor_half_m)
            if poly is not None:
                ax.fill(poly[:, 0], poly[:, 1], color="#1020ff", alpha=0.22, lw=0, zorder=5)
        ax.plot(xy[:, 0], xy[:, 1], "-", color=col, lw=line_w, alpha=line_a, zorder=line_z)
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


def _route_pair_alt(key: str):
    """Split a route key 'A_to_B#altK' (or 'A_to_B') into (pair, alt_int)."""
    if "#alt" in key:
        pair, a = key.split("#alt", 1)
        try:
            return pair, int(a)
        except ValueError:
            return pair, 0
    return key.split("#", 1)[0], 0


def order_pairs_by_density(pairs, obj_ij, ny, nx, dx, band_m):
    """Order pairs by PREDICTED route density: build an overlap field from every
    pair's straight-line corridor band, then rank each pair by the mean overlap
    along its own line. Busiest (trunk) corridors come first, so they are planned
    -- and locked -- first, and lighter routes detour around them."""
    ii = np.arange(ny, dtype=float)[:, None]
    jj = np.arange(nx, dtype=float)[None, :]
    density = np.zeros((ny, nx), float)
    bands = {}
    for p in pairs:
        (ai, aj), (bi, bj) = obj_ij[p[0]], obj_ij[p[1]]
        di, dj = float(bi - ai), float(bj - aj)
        l2 = di * di + dj * dj
        if l2 < 1.0:
            bands[p] = np.zeros((ny, nx), bool)
            continue
        t = np.clip(((ii - ai) * di + (jj - aj) * dj) / l2, 0.0, 1.0)
        perp = np.hypot(ii - (ai + t * di), jj - (aj + t * dj)) * dx
        m = perp <= band_m
        bands[p] = m
        density += m
    scores = {p: (float(density[bands[p]].mean()) if bands[p].any() else 0.0) for p in pairs}
    return sorted(pairs, key=lambda p: -scores[p]), scores


def _seg_cross(p1, p2, p3, p4):
    """True if open segments p1p2 and p3p4 properly cross."""
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])
    d1 = ccw(p3, p4, p1); d2 = ccw(p3, p4, p2)
    d3 = ccw(p1, p2, p3); d4 = ccw(p1, p2, p4)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def count_crossings(routes: dict) -> int:
    """Number of PRIMARY route pairs whose centrelines cross (share no endpoint)."""
    prim = {k: v for k, v in routes.items() if k.endswith("#alt0") or "#alt" not in k}
    items = list(prim.values())
    ends = [(tuple(np.round(v[0], 1)), tuple(np.round(v[-1], 1))) for v in items]
    n = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if ends[i][0] in ends[j] or ends[i][1] in ends[j]:
                continue                       # meet at a shared objective, not a crossing
            A, B = items[i], items[j]
            crossed = False
            for s in range(len(A) - 1):
                for t in range(len(B) - 1):
                    if _seg_cross(A[s], A[s + 1], B[t], B[t + 1]):
                        crossed = True
                        break
                if crossed:
                    break
            n += int(crossed)
    return n


# ----------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    # header PARAMETERS are the base; an optional --param-file overrides them;
    # explicit CLI flags (below) override both.
    params = dict(PARAMETERS)
    if args.param_file:
        pf = THIS_DIR / args.param_file
        if pf.exists():
            params.update(load_params(pf))
        else:
            print(f"[warn] --param-file {pf} not found; using header PARAMETERS")

    w_time = args.w_time if args.w_time is not None else float(pget(params, "W_TIME", 1.0))
    w_risk = args.w_risk if args.w_risk is not None else float(pget(params, "W_RISK", 2.0))
    w_conf = args.w_conflict if args.w_conflict is not None else float(pget(params, "W_CONFLICT", 1.5))
    pair_source = args.pairs or str(pget(params, "PAIR_SOURCE", "corridor"))
    make_fig = (not args.no_figures) and bool(pget(params, "MAKE_FIGURES", True))
    make_html = (not args.no_html) and bool(pget(params, "MAKE_HTML", True))

    # ---- planner selection + lock-and-re-search (path diversification) ----
    planner = (args.planner or str(pget(params, "PLANNER", "fmm"))).lower()
    diversify_k = max(1, args.diversify_k if args.diversify_k is not None
                      else int(pget(params, "DIVERSIFY_K", 1)))
    lock_penalty = float(pget(params, "LOCK_PENALTY", 4.0))     # soft-lock weight
    lock_halfwidth = float(pget(params, "LOCK_HALFWIDTH_M", 150.0))
    lock_mode = str(pget(params, "LOCK_MODE", "soft")).lower()  # soft | hard
    lock_scope = str(pget(params, "LOCK_SCOPE", "global")).lower()  # global | per_pair
    prioritize_by_density = bool(pget(params, "PRIORITIZE_BY_DENSITY", True))
    cross_penalty = float(pget(params, "CROSS_PENALTY", 12.0))  # cost to cross a laid corridor

    out_dir = THIS_DIR / str(pget(params, "OUT_DIR", "output/_engine_default/02_route_plan"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 66)
    print(f"engine_route_plan.py  {VERSION}")
    print(f"Planner       : {planner}"
          + (f"  +lock-and-re-search K={diversify_k} "
             f"({lock_mode}, hw={lock_halfwidth:.0f}m, scope={lock_scope})"
             if diversify_k > 1 else "  (single path)"))
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

    # ---- conflict term = 1 - costmap slowness ----
    # No traffic map yet: the costmap is UNIFORM (COSTMAP_UNIFORM_SLOWNESS for every
    # node), so the conflict term is a constant -> it does not bias routing and this
    # plan stays on the cost-free volume (risk + time). Only if CONFLICT_FROM_COSTMAP
    # is on AND a real step-08 .npz exists do we read a spatially-varying map.
    cmf = str(pget(params, "COST_MAP_FILE", ""))
    if bool(pget(params, "CONFLICT_FROM_COSTMAP", False)) and cmf \
            and (THIS_DIR / cmf).exists():
        conflict = 1.0 - resample_costmap(THIS_DIR / cmf, x0, y0, dx, nx, ny)
    else:
        uniform_slw = float(pget(params, "COSTMAP_UNIFORM_SLOWNESS", 1.0))
        conflict = np.full((ny, nx), 1.0 - uniform_slw, dtype=float)

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

    # ---- corridor width + buffer: constrain the centreline so the whole band fits ----
    # clearance = distance to the nearest no-fly cell (m). A centreline is only
    # allowed where clearance >= WIDTH/2 + BUFFER, so the corridor plus its buffer
    # never touches an obstacle/RA.
    route_width = float(pget(params, "ROUTE_WIDTH_M", 100.0))
    route_buffer = float(pget(params, "ROUTE_BUFFER_M", 50.0))
    req_clear = 0.5 * route_width + route_buffer
    # clearance = distance to the no-fly FOOTPRINT boundary. distance_transform_edt
    # measures to the nearest no-fly CELL CENTRE, so it over-reports the true gap
    # to the obstacle region (each no-fly cell is a dx-square, whose nearest CORNER
    # sits up to half a diagonal, 0.5*dx*sqrt(2), inside that centre distance).
    # Subtract the half-diagonal so the corridor + buffer band is tested against
    # the footprint conservatively for ANY approach angle and can never intrude
    # into a no-fly zone. Also cap by distance to the MAP BOUNDARY so the band
    # never leaves the map either.
    row_d = np.minimum(np.arange(ny), ny - 1 - np.arange(ny)).astype(float)
    col_d = np.minimum(np.arange(nx), nx - 1 - np.arange(nx)).astype(float)
    border_clear = np.minimum(row_d[:, None], col_d[None, :]) * dx
    half_diag = 0.5 * dx * math.sqrt(2.0)
    gap_m = np.clip(distance_transform_edt(passable) * dx - half_diag, 0.0, None)
    clearance_m = np.minimum(gap_m, border_clear)
    route_ok = clearance_m >= req_clear
    passable_route = passable & route_ok
    cost[~route_ok] = np.inf                 # centreline may only run where the band fits
    print(f"Corridor      : width {route_width:.0f} m + buffer {route_buffer:.0f} m "
          f"-> need {req_clear:.0f} m clearance ({int(route_ok.sum())}/{passable.sum()} cells qualify)")

    # ---- objectives (DB/DK) straight from the step-01 riskmap ----
    obj_xy = load_objectives(THIS_DIR / str(pget(params, "RISK_XYZ", "")))
    obj_ij = {}
    for nid, (x, y) in obj_xy.items():
        ix = int(np.clip(round((x - x0) / dx), 0, nx - 1))
        iy = int(np.clip(round((y - y0) / dx), 0, ny - 1))
        # snap the endpoint to the nearest cell that can host the full-width corridor;
        # fall back to any flyable cell if none is near (tight objective pocket).
        obj_ij[nid] = snap_to_flyable(ix, iy, passable_route, nx, ny)
        if not passable_route[obj_ij[nid]]:
            obj_ij[nid] = snap_to_flyable(ix, iy, passable, nx, ny)
    # endpoints must always be enterable/leavable even if they sit in a tight
    # pocket that cannot host the full corridor width.
    for ij in set(obj_ij.values()):
        passable_route[ij] = True
        if not np.isfinite(cost[ij]):
            cost[ij] = w_time * 1.0 + floor

    # ---- pair list ----
    # "corridor" = every DB->DK delivery pair (the network we want to build);
    # "all" = every objective pair. No corridor lengths yet (step 07 not built).
    corridor_len: dict = {}
    if pair_source == "all":
        ids = list(obj_xy)
        pairs = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1:]]
    else:
        db_ids = [k for k in obj_xy if k.startswith("DB")]
        dk_ids = [k for k in obj_xy if k.startswith("DK")]
        pairs = [(a, b) for a in db_ids for b in dk_ids]
    print(f"Objectives    : {len(obj_xy)} (DB/DK)   pairs to plan: {len(pairs)}")

    # priority: plan the predicted-busiest corridors FIRST (find first -> lock
    # first), so the trunk routes claim the straightest paths.
    if prioritize_by_density and pairs:
        pairs, _scores = order_pairs_by_density(pairs, obj_ij, ny, nx, dx, req_clear)
        top = " > ".join(f"{a}->{b}" for a, b in pairs[:3])
        print(f"Priority order: {top}" + (" > ..." if len(pairs) > 3 else ""))

    # ---- plan each pair; optionally K alternatives via lock-and-re-search ----
    # For every pair the base cost field is planned by the chosen solver; each
    # found path is "locked" (a usage penalty stamped in a corridor around it) so
    # the next search is pushed onto a fresh route. LOCK_SCOPE=global carries the
    # usage across ALL pairs (spreads the whole network); per_pair resets it so
    # each pair just gets K spatially-separated alternatives.
    win = int(pget(params, "ROUTE_SMOOTH_WIN", 3))
    usage_global = np.zeros((ny, nx), float)
    # corridor_usage accumulates every laid corridor at its FULL width; crossing
    # (or overlapping) it costs cross_penalty, so later routes take a longer detour
    # instead of cutting across -> minimises crossings. Half-width = corridor half.
    corridor_usage = np.zeros((ny, nx), float)
    cross_half = 0.5 * route_width

    routes_xy = {}
    rows_pts, rows_sum = [], []
    n_ok = 0
    for (a, b) in pairs:
        usage = usage_global if lock_scope == "global" else np.zeros((ny, nx), float)
        for k in range(diversify_k):
            # base cost + crossing penalty for cutting across existing corridors
            base = cost
            if cross_penalty > 0.0 and np.any(corridor_usage > 0):
                base = cost + cross_penalty * corridor_usage
            if lock_penalty > 0.0 and np.any(usage > 0):
                if lock_mode == "hard":
                    cost_eff = base.copy()
                    blocked = (usage > 0)
                    blocked[obj_ij[a]] = False       # never seal the endpoints
                    blocked[obj_ij[b]] = False
                    cost_eff[blocked] = np.inf
                else:
                    cost_eff = base + lock_penalty * usage
            else:
                cost_eff = base

            path = plan_one(cost_eff, obj_ij[a], obj_ij[b], planner, passable_route, dx)
            if len(path) < 2:
                if k == 0:
                    print(f"  ! {a}->{b}: unreachable")
                break

            ij = np.array(path)
            xy = np.column_stack([x0 + ij[:, 1] * dx, y0 + ij[:, 0] * dx]).astype(float)
            seg = np.hypot(*np.diff(xy, axis=0).T)
            length_m = float(seg.sum())
            rk = risk_field[ij[:, 0], ij[:, 1]]
            cf = conflict[ij[:, 0], ij[:, 1]]
            clr = clearance_m[ij[:, 0], ij[:, 1]]
            pair = f"{a}_to_{b}" if diversify_k == 1 else f"{a}_to_{b}#alt{k}"
            routes_xy[pair] = smooth_xy(xy, win, clearance_m, req_clear, x0, y0, dx)
            n_ok += 1
            for s, (px, py) in enumerate(xy):
                rows_pts.append({"pair": pair, "alt": k, "seq": s,
                                 "x": round(px, 2), "y": round(py, 2),
                                 "risk": round(float(rk[s]), 4),
                                 "conflict": round(float(cf[s]), 4)})
            rows_sum.append({
                "pair": pair, "alt": k, "n_pts": len(xy), "length_m": round(length_m, 1),
                "width_m": round(route_width, 1),
                "buffer_m": round(route_buffer, 1),
                "half_extent_m": round(req_clear, 1),         # width/2 + buffer
                "min_clearance_m": round(float(clr.min()), 1),  # >= half_extent (band fits)
                "corridor_area_m2": round(length_m * route_width, 0),
                "mean_risk": round(float(rk.mean()), 4),
                "max_risk": round(float(rk.max()), 4),
                "mean_conflict": round(float(cf.mean()), 4),
                "corridor_length_m": round(corridor_len.get((a, b), float("nan")), 1),
            })
            # lock this route so the next alternative (and, if global, later pairs)
            # steer around it
            if diversify_k > 1 or lock_scope == "global":
                stamp_usage(usage, [tuple(p) for p in path], dx, lock_halfwidth)
            # record the laid corridor at full width so later routes avoid crossing it
            if cross_penalty > 0.0:
                stamp_usage(corridor_usage, [tuple(p) for p in path], dx, cross_half)

    pd.DataFrame(rows_pts).to_csv(out_dir / "route_points.csv", index=False)
    summ = pd.DataFrame(rows_sum)
    summ.to_csv(out_dir / "route_summary.csv", index=False)
    n_crossings = count_crossings(routes_xy)
    print(f"Planned       : {n_ok}/{len(pairs)} routes   crossings: {n_crossings}")
    if len(summ):
        print(f"Route length  : mean {summ['length_m'].mean():.0f} m  "
              f"(corridor mean {summ['corridor_length_m'].mean():.0f} m)")
        print(f"Risk exposure : mean {summ['mean_risk'].mean():.3f}  "
              f"max {summ['max_risk'].max():.3f}")
        print(f"Conflict expo : mean {summ['mean_conflict'].mean():.3f}")

    metrics = {
        "version": VERSION,
        "planner": planner,
        "diversify_k": diversify_k,
        "lock": {"penalty": lock_penalty, "halfwidth_m": lock_halfwidth,
                 "mode": lock_mode, "scope": lock_scope} if diversify_k > 1 else None,
        "corridor": {"width_m": route_width, "buffer_m": route_buffer,
                     "half_extent_m": req_clear},
        "weights": {"time": w_time, "risk": w_risk, "conflict": w_conf},
        "grid": {"nx": nx, "ny": ny, "dx_m": dx,
                 "map_w_m": nx * dx, "map_h_m": ny * dx},
        "n_nofly_cells": int(nofly.sum()),
        "n_pairs": len(pairs),
        "n_routes": n_ok,
        "n_crossings": n_crossings,
        "prioritize_by_density": prioritize_by_density,
        "cross_penalty": cross_penalty,
        "pair_source": pair_source,
        "nofly_as_obstacle": bool(pget(params, "NOFLY_AS_OBSTACLE", True)),
        "mean_route_length_m": None if not len(summ) else round(float(summ["length_m"].mean()), 1),
        "mean_corridor_length_m": None if not len(summ) else round(float(summ["corridor_length_m"].mean()), 1),
        "mean_risk_exposure": None if not len(summ) else round(float(summ["mean_risk"].mean()), 4),
        "mean_conflict_exposure": None if not len(summ) else round(float(summ["mean_conflict"].mean()), 4),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved         : {out_dir/'route_points.csv'}, route_summary.csv, metrics.json")

    # ---- figures ----
    if make_fig:
        cost_disp = np.where(np.isfinite(cost), cost, np.nan)
        field_figure(cost_disp, extent, routes_xy, obj_xy,
                     f"Route network  (width {route_width:.0f} m + buffer {route_buffer:.0f} m)  "
                     f"on the cost-free volume",
                     "impedance (normalised, no-fly = white)", "viridis",
                     out_dir / "cost_field.png",
                     corridor_half_m=0.5 * route_width, buffer_half_m=req_clear)
        field_figure(np.where(passable, r_hat, np.nan), extent, routes_xy, obj_xy,
                     "Risk term (step 01) + FMM routes", "risk (normalised)",
                     "Reds", out_dir / "risk_field.png")
        field_figure(np.where(passable, c_hat, np.nan), extent, routes_xy, obj_xy,
                     "Conflict term (step 08 traffic) + FMM routes", "conflict (normalised)",
                     "Oranges", out_dir / "conflict_field.png")
        b0 = sorted({b for _, b in pairs})[0]
        src0 = np.zeros((ny, nx), bool)
        src0[obj_ij[b0]] = True
        Td = eikonal_fmm(cost, src0, dx)             # illustrative cost landscape
        # Show the WHOLE network (same route count as the other fields), but
        # emphasise the routes that actually descend THIS field (the ones that
        # end at b0); the rest are dimmed. The arrival field is destination-
        # specific, so only the b0 routes truly follow its gradient.
        to_b0 = {k for k in routes_xy if k.split("#")[0].endswith(f"_to_{b0}")}
        field_figure(np.where(np.isfinite(Td), Td, np.nan), extent,
                     routes_xy, obj_xy,
                     f"FMM arrival-cost field to {b0}  "
                     f"(all {len(routes_xy)} routes; {len(to_b0)} to {b0} bold)",
                     "accumulated cost", "magma", out_dir / "arrival_field.png",
                     emphasize=to_b0)
        print(f"Figures       : cost_field.png, risk_field.png, conflict_field.png, arrival_field.png")

    if make_html:
        model_nodes = load_candidate_nodes(THIS_DIR / str(pget(params, "RISK_XYZ", "")))
        html_fields = [
            ("cost", np.where(np.isfinite(cost), cost, np.nan), "viridis"),
            ("risk", np.where(passable, r_hat, np.nan), "jet"),
            ("conflict", np.where(passable, c_hat, np.nan), "Oranges"),
        ]
        render_route_html(
            out_dir / "route_network.html", routes_xy, obj_xy, nofly, extent, dx,
            route_width, req_clear,
            {"title": "Route network — step 02 (engine_route_plan.py)",
             "planner": planner, "diversify_k": diversify_k,
             "n_routes": n_ok, "n_pairs": len(pairs),
             "w_time": w_time, "w_risk": w_risk, "w_conflict": w_conf},
            nodes=model_nodes, fields=html_fields)
        print(f"HTML          : route_network.html")
    print(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
