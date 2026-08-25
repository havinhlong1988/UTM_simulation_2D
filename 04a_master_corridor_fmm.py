#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04a_master_corridor_fmm.py  --  FMM MASTER PLANNER

Risk-aware master-route planning over a weighted 2D cost field, using the FAST
MARCHING METHOD (FMM). This is the FMM counterpart of the Theta* master corridor
planner (04b_master_corridor_thetastar.py): both read step-03's
master_plan_input_nodes.csv and plan DB->DK routes, so FMM vs Theta* can be
compared fairly at the master stage. (Replaces the former PSO/ACO master planner,
kept as 04_run_master_plan_ACO_legacy.py.)

Runs directly with the PARAMETERS embedded in the header below -- no params file
needed:

    python 04a_master_corridor_fmm.py
    python 04a_master_corridor_fmm.py --planner fmm --diversify-k 8
    python 04a_master_corridor_fmm.py --planner theta --diversify-k 8 --pairs all

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

This is the MASTER planning stage (after step-03 density): it plans DB->DK master
routes on the weighted risk field with FMM + K-diversification, the FMM sibling of
the Theta* corridor planner (05). By default it plans risk + travel time only (no
density-conflict), matching 05's cost basis for a fair FMM-vs-Theta* comparison.

Inputs
------
    output/03_route_density/master_plan_input_nodes.csv   grid (risk_total +
        slowness) AND the DB/DK objective points (from the label column). A
        step-01 riskmap .xyz also works -- delimiter is auto-detected.
    output/07_costmap/slowness_costmap.npz   OPTIONAL traffic-conflict term

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
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

from src.fmm import eikonal_fmm, backtrack
from src.maprule import add_map_rule
from src.route_html import render_route_html, load_candidate_nodes

THIS_DIR = Path(__file__).resolve().parent
VERSION = "v2"

# ======================================================================
# PARAMETERS  (run defaults -- edit here; overridden by --param-file / CLI)
# ======================================================================
PARAMETERS: dict = {
    # ---- inputs / outputs ----
    # FMM master planner reads step-03's master_plan_input_nodes.csv (the SAME
    # input as the Theta* master planner 05, for a fair FMM-vs-Theta* comparison):
    # a full grid node table with risk_total + slowness + DB/DK objective labels.
    # A step-01 riskmap .xyz also works (delimiter auto-detected).
    "RISK_XYZ":      "output/a_fmm/03_route_density/master_plan_input_nodes.csv",
    # No step-08 traffic costmap yet: the costmap is UNIFORM (same slowness for
    # every node), which is routing-neutral -> this first plan is on the cost-free
    # volume. Point COST_MAP_FILE at a real step-08 .npz + turn CONFLICT_FROM_COSTMAP
    # on later to use the traffic map instead.
    "COSTMAP_UNIFORM_SLOWNESS": 1.0,   # uniform slowness for all nodes (1.0 = full speed)
    "COST_MAP_FILE": "",               # optional later step-08 costmap (unused now)
    "CORRIDOR_DIR":  "",               # unused now (step 07 not built yet)
    "OUT_DIR":       "output/a_fmm/04_master_corridor",
    "PAIR_SOURCE":   "corridor",     # "corridor" = every DB->DK pair | "all" = every objective pair
    # ---- cost-field weights ----
    "W_TIME":     1.0,               # travel time / path length
    "W_RISK":     2.0,               # obstacle + RA risk exposure (step 01)
    "W_CONFLICT": 1.5,               # traffic-density / conflict exposure (step 08)
    "COST_FLOOR": 0.05,              # min impedance on a free cell (FMM needs cost > 0)
    "RISK_BUFFER_M": 300.0,          # decay length of the hazard-proximity risk (0 = off)
    # ---- TN attraction: make cells FAR from a traffic/relief node cost more, so
    # the FMM geodesic is pulled to thread the TN/RN network (routes USE the nodes
    # from step 03 instead of cutting straight DB->DK). cost += WEIGHT*(1 - exp(
    # -dist_to_nearest_node / RADIUS)). Higher WEIGHT / RADIUS -> stronger pull
    # (but more detour/turning). Attracts to ALL step-03 candidates (TN + RN).
    "TN_ATTRACT_ENABLE":   True,
    "TN_ATTRACT_RADIUS_M": 100.0,    # attraction well radius per node/edge (m)
    "TN_ATTRACT_WEIGHT":   9.0,      # extra impedance on cells far from the network (higher = stronger pull to TN)
    # each TN is a 100 m-DIAMETER zone. The route is made TANGENT to it GEOMETRICALLY
    # by FILLET below (arc radius = TN_DIAMETER/2), which also minimises sharp turns.
    # TN_CORE_PENALTY is a soft cost-field alternative (repel the interior) but it
    # ADDS turns, so it is off by default now that the geometric fillet does the job.
    "TN_DIAMETER_M":       100.0,    # traffic-node zone diameter (m)
    "TN_CORE_PENALTY":       0.0,    # cost-field interior repulsion (0 = off; use FILLET instead)
    # ---- geometric corner FILLET: round sharp corners into arcs TANGENT to the
    # TN circles (radius = TN_DIAMETER/2) -> tangent pass-by + fewer/smoother turns.
    "FILLET_ENABLE":       False,
    "FILLET_RADIUS_M":      50.0,    # arc radius (= TN zone radius); tangent to the 100 m circle
    # low-cost EDGES between nearby obstacle-free node pairs -> the cheap region is
    # a CONNECTED TN network, so FMM threads TN -> TN (junction routing, not just
    # hugging node zones). TN_EDGE_MAX_M caps edge length.
    "TN_EDGE_ENABLE":      True,
    "TN_EDGE_MAX_M":       800.0,    # only connect nodes within this distance
    "TN_USED_RADIUS_M":     75.0,    # a node counts as USED if a route passes within this
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
    "SAVE_PAIR_FIGURES": True,   # one PNG per objective pair in figures/ (like 05)
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
    # ---- bidirectional routing + hard minimum separation (mirrors 05) ----
    # Each pair is planned in BOTH directions (a->b and b->a), interleaved so
    # neither direction claims every good corridor first. A candidate is REJECTED
    # (and re-searched under a harder lock) unless it is genuinely separated from
    # the routes already accepted for that pair: at most X% of its length -- minus
    # the shared endpoint driveways -- may run closer than MIN_SEPARATION_M.
    "BIDIRECTIONAL":       True,
    "PAIR_DIRECTION_ORDER": "interleaved",   # "interleaved" | "sequential"
    "SEPARATION_ENABLE":   True,
    # Separation is measured CENTRELINE-to-CENTRELINE. Safety rule: the MAIN
    # ROUTE bands (the usable corridor, centreline +- ROUTE_WIDTH/2) must never
    # overlap; the outer BUFFER bands MAY overlap. So the minimum centreline gap
    # is ROUTE_WIDTH_M (bands just touch) plus a margin.
    # 0 = auto: ROUTE_WIDTH_M + SEPARATION_MARGIN_M.
    "MIN_SEPARATION_M":      0.0,
    "SEPARATION_MARGIN_M":  10.0,   # extra clearance on top of the touching distance
    # MAIN-BAND overlap budget. Traffic naturally converges at the objectives
    # (dock/base), so some corridor overlap there is unavoidable and acceptable:
    # allow the main bands to overlap over at most this share of a route's length
    # (0 = never allowed, 100 = unconstrained). Outside the budget the candidate is
    # re-searched under a harder lock, then dropped.
    "SEPARATION_MAIN_OVERLAP_PCT": 15.0,
    "SEPARATION_MAX_VIOLATION_PCT_LONG":  10.0,
    "SEPARATION_MAX_VIOLATION_PCT_SHORT":  5.0,
    "SEPARATION_LONG_ROUTE_M":          3000.0,
    "SEPARATION_ENDPOINT_SKIP_M":        300.0,   # dense terminal aprons are exempt
    "SEPARATION_MAX_RETRY":                 3,    # re-searches under an escalating lock
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
    """Return (risk[ny,nx], slw01[ny,nx], x0, y0, dx, nx, ny) from a regular
    grid node table -- a step-01 riskmap .xyz (whitespace) OR step-03's
    master_plan_input_nodes.csv (comma). Delimiter is auto-detected."""
    df = pd.read_csv(xyz_path, sep=None, engine="python")
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
    """Read the routing objectives (nodes whose `label` starts with DB/DK) from
    the same node table -- step-01 .xyz or step-03 master_plan_input_nodes.csv.
    Delimiter is auto-detected."""
    df = pd.read_csv(xyz_path, sep=None, engine="python")
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


def _rdp(pts: np.ndarray, eps: float) -> np.ndarray:
    """Ramer-Douglas-Peucker simplification -> the polyline's real corners."""
    pts = np.asarray(pts, float)
    if len(pts) < 3:
        return pts
    a, b = pts[0], pts[-1]
    ab = b - a
    L = float(np.hypot(*ab))
    if L < 1e-9:
        d = np.hypot(*(pts - a).T)
    else:
        d = np.abs((pts[:, 0] - a[0]) * ab[1] - (pts[:, 1] - a[1]) * ab[0]) / L
    i = int(np.argmax(d))
    if d[i] > eps:
        left = _rdp(pts[:i + 1], eps)
        right = _rdp(pts[i:], eps)
        return np.vstack([left[:-1], right])
    return np.vstack([a, b])


def fillet_route(xy, radius, clearance_m, req_clear, x0, y0, dx, min_deg=8.0, simplify_tol=40.0):
    """Round the sharp corners of a route into circular arcs of ``radius`` that are
    TANGENT to both incoming/outgoing segments (so the centreline is tangent to a
    radius-circle at each turn -- a smooth pass-by that MINIMISES sharp turns). The
    route is first simplified to its real corners; each corner deflecting more than
    ``min_deg`` becomes an arc, unless the arc would drop the corridor clearance
    below ``req_clear`` (then the sharp, already-clear corner is kept). radius = TN
    zone radius makes the track tangent to the 100 m TN circles at its turns."""
    xy = np.asarray(xy, float)
    if len(xy) < 3 or radius <= 0:
        return xy
    corners = _rdp(xy, simplify_tol)
    if len(corners) < 3:
        return xy
    ny_, nx_ = clearance_m.shape

    def clear(pt):
        j = int(np.clip(round((pt[0] - x0) / dx), 0, nx_ - 1))
        i = int(np.clip(round((pt[1] - y0) / dx), 0, ny_ - 1))
        return clearance_m[i, j] >= req_clear

    out = [corners[0]]
    for k in range(1, len(corners) - 1):
        A, V, B = corners[k - 1], corners[k], corners[k + 1]
        va, vb = A - V, B - V
        la, lb = float(np.hypot(*va)), float(np.hypot(*vb))
        if la < 1e-6 or lb < 1e-6:
            out.append(V); continue
        ua, ub = va / la, vb / lb
        theta = float(np.arccos(np.clip(np.dot(ua, ub), -1.0, 1.0)))   # angle between segments
        if np.degrees(np.pi - theta) < min_deg or theta < 1e-3:        # nearly straight -> keep
            out.append(V); continue
        half = theta / 2.0
        t = min(radius / max(np.tan(half), 1e-6), 0.48 * la, 0.48 * lb)
        if t < dx:                                                     # too tight to fillet
            out.append(V); continue
        r_eff = t * np.tan(half)
        bis = ua + ub
        nb = float(np.hypot(*bis))
        if nb < 1e-9:
            out.append(V); continue
        bis /= nb
        cc = V + bis * (t / max(np.cos(half), 1e-6))                   # arc centre
        p1, p2 = V + ua * t, V + ub * t
        a1 = np.arctan2(p1[1] - cc[1], p1[0] - cc[0])
        a2 = np.arctan2(p2[1] - cc[1], p2[0] - cc[0])
        da = (a2 - a1 + np.pi) % (2 * np.pi) - np.pi                   # signed shorter arc
        n = max(2, int(abs(da) * r_eff / max(0.5 * dx, 1e-6)))
        arc = [cc + r_eff * np.array([np.cos(a1 + da * s / n), np.sin(a1 + da * s / n)])
               for s in range(n + 1)]
        if all(clear(p) for p in arc):
            out.append(p1); out.extend(arc[1:-1]); out.append(p2)
        else:
            out.append(V)
    out.append(corners[-1])
    return np.asarray(out)


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


def draw_nodes(ax, nodes):
    """Overlay the step-03 traffic/relief nodes on a field figure: TN = orange
    circle, RN = magenta diamond, with the node label. Nodes flagged as UNUSED
    (5-tuple with used=False) are drawn hollow."""
    for item in (nodes or []):
        x, y, lbl, kind = item[0], item[1], item[2], item[3]
        used = bool(item[4]) if len(item) > 4 else True
        col = "#e8710a" if kind == "TN" else "#d000d0"
        ax.scatter([x], [y], s=120, marker="o" if kind == "TN" else "D",
                   facecolors=(col if used else "none"), edgecolors=col,
                   linewidths=1.8, zorder=10)
        ax.annotate(str(lbl), (x, y), textcoords="offset points", xytext=(6, 4),
                    fontsize=7, weight="bold",
                    color=("#a04000" if kind == "TN" else "#900090"), zorder=11)


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
                 corridor_half_m=0.0, buffer_half_m=0.0, emphasize=None, nodes=None):
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
    draw_nodes(ax, nodes)
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


def _resample_m(xy, step_m=25.0):
    """Resample a polyline at a fixed spacing (for distance comparisons)."""
    xy = np.asarray(xy, float)
    if len(xy) < 2:
        return xy
    d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    if d[-1] < step_m:
        return xy
    t = np.arange(0.0, d[-1], step_m)
    return np.column_stack([np.interp(t, d, xy[:, 0]), np.interp(t, d, xy[:, 1])])


def separation_violation_pct(new_xy, ref_xy, min_sep_m, endpoint_skip_m=150.0):
    """Percentage of ``new_xy``'s length that runs closer than ``min_sep_m`` to
    ``ref_xy``. Endpoint driveways (first/last endpoint_skip_m) are excluded --
    routes of a pair necessarily share their terminals. Mirrors 05's
    backup_separation_stats(): resample at 25 m, point-to-polyline distance."""
    A = _resample_m(new_xy, 25.0)
    B = _resample_m(ref_xy, 25.0)
    if len(A) < 2 or len(B) < 2:
        return 0.0
    d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(A, axis=0).T))]
    keep = (d >= endpoint_skip_m) & (d <= max(d[-1] - endpoint_skip_m, 0.0))
    A = A[keep]
    if len(A) == 0:
        return 0.0
    dist = np.min(np.hypot(A[:, 0][:, None] - B[:, 0][None, :],
                           A[:, 1][:, None] - B[:, 1][None, :]), axis=1)
    return 100.0 * float((dist < min_sep_m).sum()) / float(len(A))


def separation_ok(new_xy, refs, min_sep_m, max_pct_long, max_pct_short,
                  long_route_m=3000.0, endpoint_skip_m=150.0):
    """True if ``new_xy`` is genuinely separated from every already-accepted route
    in ``refs``. Long routes get a looser allowance than short ones (same rule as
    05's BACKUP_SEPARATION_MAX_VIOLATION_PCT_LONG / _SHORT)."""
    if not refs or min_sep_m <= 0:
        return True, 0.0
    L = float(np.hypot(*np.diff(np.asarray(new_xy, float), axis=0).T).sum())
    allow = max_pct_long if L >= long_route_m else max_pct_short
    worst = 0.0
    for ref in refs:
        pct = separation_violation_pct(new_xy, ref, min_sep_m, endpoint_skip_m)
        worst = max(worst, pct)
        if pct > allow:
            return False, worst
    return True, worst


def pair_figure(pair_name, routes, extent, nofly, obj_xy, nodes, out_png,
                route_width=50.0, req_clear=37.5, tn_diameter=100.0):
    """One figure per objective pair (same idea as 05's plot_pair_routes): the
    pair's own K alternatives drawn over the no-fly map, with the corridor band,
    the TN/RN zones (100 m circles) and every objective labelled."""
    fig, ax = plt.subplots(figsize=(11, 10))
    ny_, nx_ = nofly.shape
    ax.imshow(np.where(nofly, 1.0, np.nan), origin="lower", extent=extent,
              cmap="Greys", vmin=0, vmax=1.6, zorder=1)

    # TN/RN zones: 100 m-diameter circle + label (used = filled, unused = hollow)
    for item in (nodes or []):
        x, y, lbl, kind = item[0], item[1], item[2], item[3]
        used = bool(item[4]) if len(item) > 4 else True
        col = "#e8710a" if kind == "TN" else "#d000d0"
        ax.add_patch(plt.Circle((x, y), 0.5 * tn_diameter, fill=False, ec=col,
                                lw=1.2, ls="-", alpha=0.75, zorder=6))
        ax.scatter([x], [y], s=90, marker="o" if kind == "TN" else "D",
                   facecolors=(col if used else "none"), edgecolors=col,
                   linewidths=1.6, zorder=7)
        ax.annotate(str(lbl), (x, y), textcoords="offset points", xytext=(7, 5),
                    fontsize=7, weight="bold",
                    color=("#a04000" if kind == "TN" else "#900090"), zorder=8)

    # this pair's alternatives: alt0 bold, the rest thinner/cyan
    # one colour family per DIRECTION: forward = blues, backward = warm
    dir_keys = sorted({_route_pair_alt(k)[0] for k in routes})
    fams = {d: plt.get_cmap(c) for d, c in zip(dir_keys, ["winter", "autumn", "cool"])}
    for key, xy in sorted(routes.items()):
        dkey, alt = _route_pair_alt(key)
        prim = (alt == 0)
        shade = fams.get(dkey, plt.get_cmap("cool"))(0.10 + 0.75 * min(alt / 7.0, 1.0))
        poly = _corridor_polygon(xy, 0.5 * route_width)
        if poly is not None:
            ax.fill(poly[:, 0], poly[:, 1], color=shade,
                    alpha=0.18 if prim else 0.07, lw=0, zorder=3)
        ax.plot(xy[:, 0], xy[:, 1], "-", color=shade,
                lw=2.6 if prim else 1.3, alpha=0.95 if prim else 0.8, zorder=5,
                label=f"{dkey}  alt {alt}" + (" *" if prim else ""))

    draw_objectives(ax, obj_xy)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"{pair_name}  --  {len(routes)} FMM route alternatives "
                 f"(corridor {route_width:.0f} m + buffer {req_clear - 0.5*route_width:.0f} m)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9, ncol=2)
    add_map_rule(ax, extent[0], extent[2], extent[1], extent[3])
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)


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

    out_dir = THIS_DIR / str(pget(params, "OUT_DIR", "output/04_master_corridor_plan_FMM"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 66)
    print(f"04a_master_corridor_fmm.py (FMM master)  {VERSION}")
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

    # ---- TN attraction: raise the cost of cells far from a TN/RN node so the FMM
    # geodesic is pulled to thread the step-03 node network (use TN + bridges) ----
    if bool(pget(params, "TN_ATTRACT_ENABLE", False)):
        _cnodes = load_candidate_nodes(THIS_DIR / str(pget(params, "RISK_XYZ", "")))
        if _cnodes:
            src_mask = np.zeros((ny, nx), bool)      # attraction sources: nodes (+ edges)
            node_mask = np.zeros((ny, nx), bool)     # node CENTRES only (for core repulsion)
            cells = []
            for (px, py, _lbl, _kind) in _cnodes:
                jj = int(np.clip(round((px - x0) / dx), 0, nx - 1))
                ii = int(np.clip(round((py - y0) / dx), 0, ny - 1))
                src_mask[ii, jj] = True
                node_mask[ii, jj] = True
                cells.append((ii, jj))
            # LOW-COST EDGES between nearby nodes (obstacle-free straight segments):
            # rasterise each accepted TN-TN edge into the attraction sources, so the
            # cheap region forms a CONNECTED network and FMM threads node -> node.
            n_edges = 0
            if bool(pget(params, "TN_EDGE_ENABLE", True)):
                passable_los = ~nofly
                emax_c = float(pget(params, "TN_EDGE_MAX_M", 800.0)) / dx
                emax2 = emax_c * emax_c
                for a in range(len(cells)):
                    ia, ja = cells[a]
                    for b in range(a + 1, len(cells)):
                        ib, jb = cells[b]
                        if (ia - ib) ** 2 + (ja - jb) ** 2 > emax2:
                            continue
                        if _los(passable_los, (ia, ja), (ib, jb)):     # edge stays clear of no-fly
                            for (ci, cj) in _line_cells((ia, ja), (ib, jb)):
                                src_mask[ci, cj] = True
                            n_edges += 1
            radius = float(pget(params, "TN_ATTRACT_RADIUS_M", 250.0))
            weight = float(pget(params, "TN_ATTRACT_WEIGHT", 3.0))
            d_tn = distance_transform_edt(~src_mask) * dx
            attract = np.exp(-d_tn / max(radius, 1e-6))     # 1 on a node/edge, decays outward
            cost = cost + weight * (1.0 - attract)          # cells far from the network cost MORE
            # ---- TN core: each node is a 100 m-diameter ZONE. Repel the INTERIOR so
            # the route centreline skirts the ring TANGENTIALLY (touches the circle,
            # does not cross the centre) -> the cheapest track is the tangent ring,
            # which curves smoothly past the node -> fewer sharp turns. ----
            tn_diam = float(pget(params, "TN_DIAMETER_M", 100.0))
            core_w = float(pget(params, "TN_CORE_PENALTY", 5.0))
            if core_w > 0 and tn_diam > 0:
                R = 0.5 * tn_diam
                d_center = distance_transform_edt(~node_mask) * dx   # dist to nearest TN CENTRE
                core = np.exp(-(d_center / (0.6 * R)) ** 2)          # bump peaking at the centre
                cost = cost + core_w * core
            print(f"TN attraction : {len(cells)} nodes, {n_edges} edges  radius {radius:.0f} m  "
                  f"weight {weight:.1f}  |  core zone {tn_diam:.0f} m, penalty {core_w:.1f}")

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
    fillet_on = bool(pget(params, "FILLET_ENABLE", True))
    fillet_r = float(pget(params, "FILLET_RADIUS_M", 0.5 * float(pget(params, "TN_DIAMETER_M", 100.0))))
    usage_global = np.zeros((ny, nx), float)
    # corridor_usage accumulates every laid corridor at its FULL width; crossing
    # (or overlapping) it costs cross_penalty, so later routes take a longer detour
    # instead of cutting across -> minimises crossings. Half-width = corridor half.
    corridor_usage = np.zeros((ny, nx), float)
    cross_half = 0.5 * route_width

    routes_xy = {}
    rows_pts, rows_sum = [], []
    n_ok = 0
    # ---- separation / bidirectional settings (mirrors 05) ----
    bidir = bool(pget(params, "BIDIRECTIONAL", True))
    dir_order = str(pget(params, "PAIR_DIRECTION_ORDER", "interleaved")).lower()
    sep_on = bool(pget(params, "SEPARATION_ENABLE", True))
    sep_m = float(pget(params, "MIN_SEPARATION_M", 0.0))
    sep_touch = route_width          # centreline gap at which the main bands just touch
    if sep_m <= 0:
        sep_m = sep_touch + float(pget(params, "SEPARATION_MARGIN_M", 10.0))
    elif sep_m < sep_touch:
        print(f"  [warn] MIN_SEPARATION_M={sep_m:.0f} m < corridor width {sep_touch:.0f} m "
              f"-> main route bands WILL overlap")
    sep_pct_long = float(pget(params, "SEPARATION_MAX_VIOLATION_PCT_LONG", 10.0))
    sep_pct_short = float(pget(params, "SEPARATION_MAX_VIOLATION_PCT_SHORT", 5.0))
    sep_long_m = float(pget(params, "SEPARATION_LONG_ROUTE_M", 3000.0))
    sep_skip_m = float(pget(params, "SEPARATION_ENDPOINT_SKIP_M", 150.0))
    sep_retry = int(pget(params, "SEPARATION_MAX_RETRY", 3))
    sep_main_pct = float(pget(params, "SEPARATION_MAIN_OVERLAP_PCT", 15.0))
    n_rejected = 0

    for (a, b) in pairs:
        usage = usage_global if lock_scope == "global" else np.zeros((ny, nx), float)
        accepted: list = []          # accepted geometries of THIS pair (both directions)
        # forward a->b and backward b->a, interleaved so neither direction can
        # claim every good corridor first (same intent as 05's PAIR_DIRECTION_ORDER)
        dirs = [("fwd", a, b)] + ([("bwd", b, a)] if bidir else [])
        if dir_order == "sequential":
            schedule = [(k, d) for d in dirs for k in range(diversify_k)]
        else:
            schedule = [(k, d) for k in range(diversify_k) for d in dirs]

        for k, (dname, src_id, dst_id) in schedule:
            sm = None
            for attempt in range(sep_retry + 1):
                # base cost + crossing penalty for cutting across existing corridors
                base = cost
                if cross_penalty > 0.0 and np.any(corridor_usage > 0):
                    base = cost + cross_penalty * corridor_usage
                if lock_penalty > 0.0 and np.any(usage > 0):
                    if lock_mode == "hard":
                        cost_eff = base.copy()
                        blocked = (usage > 0)
                        blocked[obj_ij[src_id]] = False   # never seal the endpoints
                        blocked[obj_ij[dst_id]] = False
                        cost_eff[blocked] = np.inf
                    else:
                        # escalate the soft lock on each separation retry, so the
                        # re-search is pushed further off the routes it clashed with
                        cost_eff = base + lock_penalty * (1.0 + attempt) * usage
                else:
                    cost_eff = base

                path = plan_one(cost_eff, obj_ij[src_id], obj_ij[dst_id], planner,
                                passable_route, dx)
                if len(path) < 2:
                    break

                ij = np.array(path)
                xy = np.column_stack([x0 + ij[:, 1] * dx, y0 + ij[:, 0] * dx]).astype(float)
                cand = smooth_xy(xy, win, clearance_m, req_clear, x0, y0, dx)
                if fillet_on:
                    cand = fillet_route(cand, fillet_r, clearance_m, req_clear, x0, y0, dx)

                if not sep_on:
                    sm = cand
                    break
                ok, worst = separation_ok(cand, accepted, sep_m, sep_pct_long,
                                          sep_pct_short, sep_long_m, sep_skip_m)
                if ok and sep_main_pct < 100.0:
                    # MAIN-BAND budget: the corridors may overlap, but only over
                    # SEPARATION_MAIN_OVERLAP_PCT of the length (checked BOTH ways,
                    # terminal aprons already excluded by sep_skip_m).
                    for ref in accepted:
                        ov = max(separation_violation_pct(cand, ref, route_width, sep_skip_m),
                                 separation_violation_pct(ref, cand, route_width, sep_skip_m))
                        if ov > sep_main_pct:
                            ok = False
                            break
                if ok:
                    sm = cand
                    break
                # too close to an accepted route of this pair: lock it harder and retry
                stamp_usage(usage, [tuple(pt) for pt in path], dx, max(lock_halfwidth, sep_m))

            if sm is None:
                n_rejected += 1
                continue

            pair = f"{a}_to_{b}" if (diversify_k == 1 and not bidir) else \
                   f"{src_id}_to_{dst_id}#alt{k}"
            routes_xy[pair] = sm
            accepted.append(sm)
            n_ok += 1
            # sample fields at the FINAL (smoothed + filleted) route geometry, so the
            # CSV / summary reflect the actual planned corridor centreline.
            si = np.clip(np.rint((sm[:, 1] - y0) / dx).astype(int), 0, ny - 1)
            sj = np.clip(np.rint((sm[:, 0] - x0) / dx).astype(int), 0, nx - 1)
            rk = risk_field[si, sj]
            cf = conflict[si, sj]
            clr = clearance_m[si, sj]
            length_m = float(np.hypot(*np.diff(sm, axis=0).T).sum()) if len(sm) > 1 else 0.0
            for s, (px, py) in enumerate(sm):
                rows_pts.append({"pair": pair, "alt": k, "direction": dname, "seq": s,
                                 "x": round(float(px), 2), "y": round(float(py), 2),
                                 "risk": round(float(rk[s]), 4),
                                 "conflict": round(float(cf[s]), 4)})
            rows_sum.append({
                "pair": pair, "alt": k, "direction": dname,
                "n_pts": len(sm), "length_m": round(length_m, 1),
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
            if diversify_k > 1 or bidir or lock_scope == "global":
                stamp_usage(usage, [tuple(pt) for pt in path], dx, lock_halfwidth)
            # record the laid corridor at full width so later routes avoid crossing it
            if cross_penalty > 0.0:
                stamp_usage(corridor_usage, [tuple(pt) for pt in path], dx, cross_half)

    if sep_on:
        print(f"Separation    : min {sep_m:.0f} m centreline-to-centreline "
              f"(main bands {route_width:.0f} m wide -> {sep_m - route_width:+.0f} m gap; "
              f"buffers may overlap)  allow {sep_pct_short:.0f}%/{sep_pct_long:.0f}%, main-band overlap <= {sep_main_pct:.0f}%  "
              f"-> {n_rejected} rejected")

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

    # ---- which model nodes (TN/RN) are USED: a route passes within TN_USED_RADIUS_M ----
    model_nodes = load_candidate_nodes(THIS_DIR / str(pget(params, "RISK_XYZ", "")))
    used_r = float(pget(params, "TN_USED_RADIUS_M", 75.0))
    allpts = np.vstack(list(routes_xy.values())) if routes_xy else np.zeros((0, 2))
    marked = []
    for (px, py, lbl, kind) in model_nodes:
        used = bool(len(allpts)) and float(np.hypot(allpts[:, 0] - px, allpts[:, 1] - py).min()) <= used_r
        marked.append((px, py, lbl, kind, used))
    n_used = sum(1 for m in marked if m[4])
    print(f"Node usage    : {n_used}/{len(marked)} TN/RN used by routes (within {used_r:.0f} m)")

    # ---- figures ----
    if make_fig:
        cost_disp = np.where(np.isfinite(cost), cost, np.nan)
        field_figure(cost_disp, extent, routes_xy, obj_xy,
                     f"Route network  (width {route_width:.0f} m + buffer {route_buffer:.0f} m)  "
                     f"on the cost-free volume",
                     "impedance (normalised, no-fly = white)", "viridis",
                     out_dir / "cost_field.png",
                     corridor_half_m=0.5 * route_width, buffer_half_m=req_clear, nodes=marked)
        field_figure(np.where(passable, r_hat, np.nan), extent, routes_xy, obj_xy,
                     "Risk term (step 01) + FMM routes", "risk (normalised)",
                     "Reds", out_dir / "risk_field.png", nodes=marked)
        field_figure(np.where(passable, c_hat, np.nan), extent, routes_xy, obj_xy,
                     "Conflict term (step 08 traffic) + FMM routes", "conflict (normalised)",
                     "Oranges", out_dir / "conflict_field.png", nodes=marked)
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
                     emphasize=to_b0, nodes=marked)
        print(f"Figures       : cost_field.png, risk_field.png, conflict_field.png, arrival_field.png")

        if bool(pget(params, "SAVE_PAIR_FIGURES", True)):
            fig_dir = out_dir / "figures"
            fig_dir.mkdir(parents=True, exist_ok=True)
            # group BOTH directions of an objective pair into one figure (like 05)
            by_pair: dict = {}
            for key, xy in routes_xy.items():
                p0 = _route_pair_alt(key)[0]
                u, v = p0.split("_to_") if "_to_" in p0 else (p0, "")
                by_pair.setdefault("_".join(sorted([u, v])) if v else p0, {})[key] = xy
            for pname, prts in sorted(by_pair.items()):
                pair_figure(pname, prts, extent, nofly, obj_xy, marked,
                            fig_dir / f"{pname}_fmm_routes.png",
                            route_width=route_width, req_clear=req_clear,
                            tn_diameter=float(pget(params, "TN_DIAMETER_M", 100.0)))
            print(f"Pair figures  : {len(by_pair)} in {fig_dir}")

    if make_html:
        html_fields = [
            ("cost", np.where(np.isfinite(cost), cost, np.nan), "viridis"),
            ("risk", np.where(passable, r_hat, np.nan), "jet"),
            ("conflict", np.where(passable, c_hat, np.nan), "Oranges"),
        ]
        render_route_html(
            out_dir / "route_network.html", routes_xy, obj_xy, nofly, extent, dx,
            route_width, req_clear,
            {"title": "FMM master network — step 04 (04a_master_corridor_fmm.py)",
             "planner": planner, "diversify_k": diversify_k,
             "n_routes": n_ok, "n_pairs": len(pairs),
             "w_time": w_time, "w_risk": w_risk, "w_conflict": w_conf},
            nodes=marked, fields=html_fields)
        print(f"HTML          : route_network.html")
    print(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
