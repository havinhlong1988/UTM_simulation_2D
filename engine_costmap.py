#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
engine_costmap.py -- stage 06 : SLOWNESS cost-map for the corridor network.

The map is INDEPENDENT of where the routes go -- it does not change the
geometry of the network, only how expensive each place in it is:

    true velocity = slowness(x, y) * base drone velocity      (stage 07 sim)
    conflict term = 1 - slowness(x, y)                        (FMM re-plan)

v2 replaces the single traffic-density heuristic with a FOUR-LAYER
assessment of the airspace, all of it evaluated ON THE NETWORK stage 05
actually built:

  (1) ECONOMIC BALANCE  `econ`   -- cân bằng kinh tế của khu vực
      Value of serving a place versus what it costs to serve it.
        value  V = demand kernels around the DB bases / DK docks
                   + expected corridor throughput (how many DB->DK pairs
                     route over each leg, from pair_routes.csv)
        cost   C = energy/deadhead (distance from the nearest DB)
                   + infrastructure (distance from the built network)
        balance B = V - C  in [-1, 1]; the penalty is (1 - B)/2, so an area
        whose value outweighs its cost is cheap to fly and one that costs
        more than it returns is expensive.
      An EQUITY term (ECON_EQUITY_W) additionally penalises the already
      saturated cores, so new capacity spreads instead of piling onto the
      same few corridors. The Gini coefficient of V over the network is
      reported in metrics.json as the economic-balance indicator.

  (2) AIR OPERATIONAL SAFETY  `air`  -- an toàn vận hành trên không
      Mid-air/encounter exposure implied by the network topology:
        * expected traffic flow per leg (pair_routes) -> encounter density
        * merge & junction hotspots: roundabouts weighted by their entry
          count, network nodes weighted by their degree
        * proximity to restricted airspace (RA) -> infringement exposure

  (3) GROUND SAFETY  `ground`  -- an toàn mặt đất
      Third-party risk on the ground under the corridor (SORA-flavoured):
        exposure  = built-up/obstacle density + the DB/DK ground aprons
        sheltering discounts the part of that exposure that is indoors
        the result is spread over the ballistic/glide IMPACT footprint
        and scaled up where no emergency landing zone (FLZ) is in reach.

  (4) TRAFFIC  `traffic`  -- measured, OPTIONAL
      Gaussian-smoothed density of the stage-07 pass-1 trajectories, with
      holding (jammed) samples weighted up. If no trajectories.csv exists
      the layer is simply dropped, so stage 06 now runs straight after
      stage 05 -- v1 crashed without a pass-1 simulation.

    composite risk = sum(W_i * layer_i) / sum(W_i)          in [0, 1]
    slowness       = MAX_SLOWNESS - risk * (MAX_SLOWNESS - MIN_SLOWNESS)

Corridor confinement
--------------------
Only cells INSIDE a corridor keep their assessed slowness. A cell is
"inside" when it lies within CORRIDOR_HALF_WIDTH metres of any lane centre-
line (stage 05's lane_nodes.csv). Everything OUTSIDE the corridor network is
forced to OUTSIDE_SLOWNESS (a crawl), so agents that stray off the corridors
are slowed to a near halt and effectively cannot fly out of the network.

Roundabouts are TWO concentric circulating corridors (mirrors 05): an outer
ring lane at radius_m and an inner ring lane at radius_m - lane_gap. BOTH ring
bands are part of the network and keep their normal slowness. The disk
ENCLOSED by the inner ring is not a lane, so its cost is RAISED (slowness
forced to RING_INTERIOR_SLOWNESS) -- agents circulate on the rings and cannot
cut straight across the roundabout centre.

Pipeline
--------
    05x_corridor_network   -> the network this map is assessed on
    06x_costmap  (this)    -> slowness cost-map           [no sim needed]
    07x_simulate           -> speeds modulated by the map
    (optional) re-run 06 with --traffic once a pass-1 sim exists, to fold
    the MEASURED traffic layer in on top of the assessed ones.

Run
---
    python engine_costmap.py --param-file params/a_fmm/06_costmap.params
    python engine_costmap.py --w-econ 1 --w-air 2 --w-ground 2 --w-traffic 0

Outputs (--out-dir)
-------------------
    slowness_costmap.npz    slowness[ny,nx], x0, y0, res  (+ every layer)
    slowness_costmap.png    the field with the corridor network overlaid
    cost_layers.png         the four layers + composite + slowness
    costmap.html            interactive viewer -- swap the component maps,
                            toggle every model-node family (DB/DK/TN/RBT/FLZ/
                            RA/model grid), hover to read all layers at once
    network_assessment.csv  per-leg econ / air / ground / risk / slowness
    metrics.json            weights, layer stats, economic Gini, worst legs
    density.png             the measured traffic layer (only with --traffic)
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
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.spatial import cKDTree

from src.maprule import add_map_rule
from src.costmap_html import render_costmap_html

THIS_DIR = Path(__file__).resolve().parent
VERSION = "v2"

# ======================================================================
# PARAMETERS  (run defaults -- edit here; overridden by --param-file / CLI)
# ======================================================================
PARAMETERS: dict = {
    # ---- grid + slowness range ----
    "RES_M":                  40.0,   # cost-map cell size
    "SMOOTH_SIGMA_M":        150.0,   # Gaussian smoothing of the density layers
    "MIN_SLOWNESS":           0.01,   # slowest a corridor cell may get
    "MAX_SLOWNESS":           1.00,   # full speed
    "CORRIDOR_HALF_WIDTH_M":  60.0,   # cells this close to a lane are "in network"
    "OUTSIDE_SLOWNESS":       0.01,   # crawl outside the network
    "RING_INTERIOR_SLOWNESS": None,   # None -> same as OUTSIDE_SLOWNESS

    # ---- layer blend (a weight of 0 drops the layer) ----
    "W_ECON":                  1.0,   # economic balance
    "W_AIR":                   1.5,   # air operational safety
    "W_GROUND":                1.5,   # ground safety
    "W_TRAFFIC":               1.0,   # measured traffic (needs --traffic)
    # "network" -> stretch the composite over its in-corridor range so the
    # sim sees the full speed band; "flyable" -> keep the whole-map scaling.
    "NORMALISE_ON":      "network",

    # ---- (1) economic balance ----
    "ECON_DEMAND_RADIUS_M":  700.0,   # decay length of a service-demand kernel
    "ECON_DB_WEIGHT":          1.0,   # a drone base radiates this much demand
    "ECON_DK_WEIGHT":          0.6,   # a docking station this much
    "ECON_VALUE_NODE_W":       1.0,   # weight of the node demand in V
    "ECON_VALUE_FLOW_W":       1.0,   # weight of the corridor throughput in V
    "ECON_ENERGY_RADIUS_M": 1500.0,   # deadhead energy decay from the nearest DB
    "ECON_INFRA_RADIUS_M":   400.0,   # cost of being far from built infrastructure
    "ECON_COST_ENERGY_W":      1.0,
    "ECON_COST_INFRA_W":       1.0,
    "ECON_EQUITY_W":          0.35,   # 0 = pure value/cost balance, 1 = pure equity
    "ECON_SATURATION_Q":      0.75,   # V above this quantile counts as saturated

    # ---- (2) air operational safety ----
    "AIR_FLOW_W":              1.0,   # encounter density from the expected flow
    "AIR_JUNCTION_W":          1.0,   # merge/junction hotspots
    "AIR_RA_W":                0.8,   # restricted-airspace infringement
    "AIR_JUNCTION_SIGMA_M":  180.0,   # decay length of a junction hotspot
    "AIR_RA_BUFFER_M":       250.0,   # decay length of the RA proximity term

    # ---- (3) ground safety ----
    "GROUND_EXPOSURE_SIGMA_M": 150.0, # smoothing of the built-up/obstacle proxy
    "GROUND_APRON_W":          0.6,   # ground people/assets at the DB/DK aprons
    "GROUND_APRON_RADIUS_M": 250.0,
    "GROUND_SHELTER_FACTOR":  0.35,   # fraction of exposure shielded by structures
    "GROUND_IMPACT_RADIUS_M": 120.0,  # ballistic/glide lateral footprint
    "GROUND_FLZ_REACH_M":    900.0,   # emergency-landing reach; beyond it, no relief
    "GROUND_FLZ_W":            1.0,   # how much an unreachable FLZ inflates the risk

    # ---- (4) measured traffic ----
    "HOLD_WEIGHT":             3.0,   # extra weight for holding (jammed) samples

    "MAKE_FIGURES":           True,
    "MAKE_HTML":              True,   # interactive costmap.html viewer
}


# ----------------------------------------------------------------------
# params helpers (same tiny format as the other stages)
# ----------------------------------------------------------------------
def parse_value(v: str):
    s = v.strip()
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    if s.lower() == "none":
        return None
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Economic / air-safety / ground-safety slowness cost-map.")
    p.add_argument("--param-file", default="",
                   help="optional params file overriding the PARAMETERS block")
    # ---- paths ----
    p.add_argument("--traffic", default="",
                   help="OPTIONAL agent position samples (t_s,agent_id,x,y,...) "
                        "from a pass-1 sim; absent -> the traffic layer is dropped")
    p.add_argument("--corridor-dir", default="output/06_corridor_network")
    p.add_argument("--out-dir", default="output/07_costmap")
    p.add_argument("--risk-xyz", default="",
                   help="stage-01 riskmap .xyz (obstacle/RA/FLZ layers); empty -> "
                        "RISK_XYZ from the param file, else the newest "
                        "output/01_random_node_map/*.xyz")
    p.add_argument("--corridor-param-file", default="params/corridor_network.params",
                   help="stage 05's params, read for ROUNDABOUT_LANE_GAP_M so the "
                        "inner ring lane matches the geometry 05 built")
    # ---- grid / slowness ----
    p.add_argument("--res", type=float, default=None, help="grid cell size (m)")
    p.add_argument("--sigma", type=float, default=None, help="Gaussian sigma (m)")
    p.add_argument("--min-slowness", type=float, default=None)
    p.add_argument("--max-slowness", type=float, default=None)
    p.add_argument("--hold-weight", type=float, default=None,
                   help="extra weight for holding (jammed) traffic samples")
    p.add_argument("--corridor-half-width", type=float, default=None,
                   help="cells within this distance (m) of a lane centreline are "
                        "'inside' a corridor and keep their assessed slowness")
    p.add_argument("--outside-slowness", type=float, default=None,
                   help="slowness forced on cells outside every corridor "
                        "(low = agents crawl and cannot leave the network)")
    p.add_argument("--roundabout-lane-gap", type=float, default=None,
                   help="gap (m) between the outer and inner ring lane; "
                        "default reads ROUNDABOUT_LANE_GAP_M from the param file")
    p.add_argument("--ring-interior-slowness", type=float, default=None,
                   help="slowness forced on the disk inside each roundabout's "
                        "inner ring; default = outside-slowness")
    # ---- layer weights ----
    p.add_argument("--w-econ", type=float, default=None)
    p.add_argument("--w-air", type=float, default=None)
    p.add_argument("--w-ground", type=float, default=None)
    p.add_argument("--w-traffic", type=float, default=None)
    p.add_argument("--normalise-on", choices=["network", "flyable"], default=None)
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--ring-as-area", action="store_true",
                   help="price each roundabout's whole disc as corridor (stage 07 "
                        "flies them as 2-D ORCA zones), keeping only the island out")
    p.add_argument("--no-html", action="store_true")
    return p.parse_args()


def resolve_params(args) -> dict:
    """PARAMETERS block < --param-file < CLI flag."""
    P = dict(PARAMETERS)
    if args.param_file:
        P.update(load_params(THIS_DIR / args.param_file))
    cli = {
        "RES_M": args.res, "SMOOTH_SIGMA_M": args.sigma,
        "MIN_SLOWNESS": args.min_slowness, "MAX_SLOWNESS": args.max_slowness,
        "HOLD_WEIGHT": args.hold_weight,
        "CORRIDOR_HALF_WIDTH_M": args.corridor_half_width,
        "OUTSIDE_SLOWNESS": args.outside_slowness,
        "RING_INTERIOR_SLOWNESS": args.ring_interior_slowness,
        "W_ECON": args.w_econ, "W_AIR": args.w_air,
        "W_GROUND": args.w_ground, "W_TRAFFIC": args.w_traffic,
        "NORMALISE_ON": args.normalise_on,
    }
    P.update({k: v for k, v in cli.items() if v is not None})
    if args.no_figures:
        P["MAKE_FIGURES"] = False
    if args.no_html:
        P["MAKE_HTML"] = False
    if args.ring_as_area:
        P["RING_AS_AREA"] = True
    return P


def load_lane_gap(param_file: Path, default: float = 50.0) -> float:
    """Read ROUNDABOUT_LANE_GAP_M from stage 05's corridor params so the costmap's
    inner ring matches the ring geometry 05 actually built."""
    if not param_file.exists():
        return default
    ns: dict = {}
    exec(compile(param_file.read_text(encoding="utf-8"), str(param_file), "exec"), {}, ns)
    return float(ns.get("ROUNDABOUT_LANE_GAP_M", default))


def find_risk_xyz(explicit: str, params: dict) -> Path | None:
    """--risk-xyz > RISK_XYZ in the param file > newest stage-01 .xyz."""
    for cand in (explicit, str(params.get("RISK_XYZ", "") or "")):
        if cand:
            p = THIS_DIR / cand
            if p.exists():
                return p
    pool = sorted((THIS_DIR / "output/01_random_node_map").glob("*.xyz"),
                  key=lambda q: q.stat().st_mtime, reverse=True)
    return pool[0] if pool else None


# ----------------------------------------------------------------------
# small field helpers
# ----------------------------------------------------------------------
def _norm(a: np.ndarray, mask: np.ndarray | None = None, hi_pct: float = 99.0) -> np.ndarray:
    """Min-to-p99 normalise ``a`` into [0,1], using only ``mask`` for the range.
    The percentile cap keeps one hot cell from flattening the whole field."""
    v = a[mask] if mask is not None else a.ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.zeros_like(a)
    lo = float(v.min())
    hi = float(np.percentile(v, hi_pct))
    if hi - lo < 1e-12:
        hi = float(v.max())
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def _gini(v: np.ndarray) -> float:
    """Gini coefficient of a non-negative distribution (0 = perfectly even)."""
    v = np.asarray(v, float).ravel()
    v = np.sort(v[np.isfinite(v) & (v >= 0)])
    if v.size == 0 or v.sum() <= 0:
        return float("nan")
    n = v.size
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * v)) / (n * np.sum(v)) - (n + 1.0) / n)


def dist_to_points(pts, gx, gy) -> np.ndarray:
    """Metres from every cell centre to the nearest of ``pts``."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    if len(pts) == 0:
        return np.full(gx.shape, np.inf)
    d, _ = cKDTree(pts).query(np.column_stack([gx.ravel(), gy.ravel()]), k=1)
    return d.reshape(gx.shape)


def kernel_sum(pts, weights, gx, gy, radius_m: float) -> np.ndarray:
    """Sum of exponentially decaying kernels w*exp(-d/radius) over ``pts``."""
    out = np.zeros(gx.shape, float)
    for (px, py), w in zip(np.asarray(pts, float).reshape(-1, 2), weights):
        if w == 0:
            continue
        out += float(w) * np.exp(-np.hypot(gx - px, gy - py) / max(radius_m, 1e-6))
    return out


def resample_lane(xy: np.ndarray, step: float) -> tuple[np.ndarray, float]:
    """Densely resample a polyline; returns (points, total length)."""
    seg = np.hypot(*np.diff(xy, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    n = max(2, int(np.ceil(total / max(step, 1e-6))) + 1)
    t = np.linspace(0.0, total, n)
    return np.column_stack([np.interp(t, s, xy[:, 0]),
                            np.interp(t, s, xy[:, 1])]), total


def stamp_lane_weights(lanes: pd.DataFrame, weights: dict, xc, yc, res) -> np.ndarray:
    """Accumulate each leg's weight along its lane centrelines onto the grid.
    The stamp is per unit length, so a long leg spreads its flow rather than
    concentrating it."""
    ny, nx = len(yc), len(xc)
    g = np.zeros((ny, nx), float)
    x0, y0 = xc[0] - 0.5 * res, yc[0] - 0.5 * res
    step = max(1.0, res / 2.0)
    for (leg_id, _lane), grp in lanes.groupby(["leg_id", "lane"]):
        w = float(weights.get(leg_id, 0.0))
        if w <= 0:
            continue
        xy = grp.sort_values("seq")[["x", "y"]].to_numpy(float)
        if len(xy) < 2:
            continue
        pts, total = resample_lane(xy, step)
        if total <= 0:
            continue
        ix = np.clip(((pts[:, 0] - x0) / res).astype(int), 0, nx - 1)
        iy = np.clip(((pts[:, 1] - y0) / res).astype(int), 0, ny - 1)
        np.add.at(g, (iy, ix), w * total / max(len(pts) - 1, 1))
    return g


def sample_xyz(df: pd.DataFrame, cols, gx, gy) -> dict:
    """Nearest-neighbour sample stage-01 .xyz columns onto the cost-map grid."""
    xs = np.sort(df["x"].unique())
    ys = np.sort(df["y"].unique())
    dx = float(np.median(np.diff(xs)))
    x0, y0 = float(xs[0]), float(ys[0])
    nx0, ny0 = len(xs), len(ys)
    ix = np.rint((df["x"].to_numpy(float) - x0) / dx).astype(int)
    iy = np.rint((df["y"].to_numpy(float) - y0) / dx).astype(int)
    jx = np.clip(np.rint((gx - x0) / dx).astype(int), 0, nx0 - 1)
    jy = np.clip(np.rint((gy - y0) / dx).astype(int), 0, ny0 - 1)
    out = {}
    for c in cols:
        base = np.zeros((ny0, nx0), float)
        base[iy, ix] = df[c].to_numpy(float)
        out[c] = base[jy, jx]
    return out


# ----------------------------------------------------------------------
# corridor geometry (unchanged from v1)
# ----------------------------------------------------------------------
def corridor_mask(lanes: pd.DataFrame, xc: np.ndarray, yc: np.ndarray,
                  half_width: float, rings: pd.DataFrame | None = None,
                  lane_gap: float = 0.0, ring_as_area: bool = False,
                  island_frac: float = 0.35) -> tuple[np.ndarray, np.ndarray]:
    """Return (inside, ring_interior), each a Boolean grid[ny,nx].

    `inside` is True where a cell centre lies within `half_width` metres of any
    lane centreline OR of a roundabout ring lane. Lane polylines are densely
    resampled so a nearest-point KD-tree query approximates the true distance-
    to-polyline.

    Each roundabout is TWO concentric circulating corridors (mirrors 05's ring
    model): an OUTER lane at radius_m and an INNER lane at radius_m - lane_gap
    (floored at 0.3*radius_m). BOTH annular bands |dist_to_centre - r| <=
    half_width are added -- the leg lanes stop AT the outer ring boundary, so
    without this the ring lanes read as off-corridor.

    `ring_interior` is the disk ENCLOSED by the inner ring (dist_to_centre <
    r_in - half_width): it is not part of any lane, and the caller raises its
    cost so agents cannot cut straight across the roundabout centre."""
    step = max(2.0, half_width / 3.0)
    pts = []
    for (_leg_id, _lane), g in lanes.groupby(["leg_id", "lane"]):
        xy = g.sort_values("seq")[["x", "y"]].to_numpy(float)
        if len(xy) < 1:
            continue
        if len(xy) == 1:
            pts.append(xy)
            continue
        pts.append(resample_lane(xy, step)[0])
    lane_pts = np.vstack(pts)
    tree = cKDTree(lane_pts)
    gx, gy = np.meshgrid(xc, yc)                       # [ny,nx]
    dist, _ = tree.query(np.column_stack([gx.ravel(), gy.ravel()]), k=1)
    inside = (dist <= half_width).reshape(gx.shape)
    ring_interior = np.zeros_like(inside)
    if rings is not None and len(rings):              # add the ring circulating bands
        for r in rings.itertuples():
            dcen = np.hypot(gx - float(r.center_x), gy - float(r.center_y))
            r_out = float(r.radius_m)
            if ring_as_area:
                # stage 07 flies these rings as 2-D ORCA manoeuvring AREAS, so the
                # whole disc out to the buffer edge is usable corridor and only the
                # kept-clear island at the centre is priced as non-lane. Pricing the
                # disc as off-corridor would crawl every agent that uses the very
                # area the ring was widened to provide.
                inside |= dcen <= r_out + half_width
                ring_interior |= dcen < island_frac * r_out
                continue
            r_in = max(0.3 * r_out, r_out - lane_gap)  # mirrors 05 (build_figure ring draw)
            inside |= np.abs(dcen - r_out) <= half_width
            inside |= np.abs(dcen - r_in) <= half_width
            ring_interior |= dcen < (r_in - half_width)
    return inside, ring_interior


def leg_flow(corridor_dir: Path, legs: pd.DataFrame, base: float = 1.0) -> dict:
    """Expected relative flow per leg = how many successful DB->DK pair routes
    use it (stage 05's pair_routes.csv), plus a base for the patrol traffic
    every leg carries."""
    flow = {str(l): base for l in legs["leg_id"]}
    pr_file = corridor_dir / "pair_routes.csv"
    if not pr_file.exists():
        return flow
    pr = pd.read_csv(pr_file)
    if "success" in pr.columns:
        pr = pr[pr["success"].astype(bool)]
    for row in pr.itertuples():
        for leg in str(getattr(row, "legs", "") or "").split(";"):
            leg = leg.strip()
            if leg:
                flow[leg] = flow.get(leg, base) + 1.0
    return flow


# ----------------------------------------------------------------------
# the four assessment layers
# ----------------------------------------------------------------------
def economic_layer(P, gx, gy, flow_field, db_xy, dk_xy, d_net_m, mask):
    """(1) cân bằng kinh tế -- value of serving a place vs the cost of serving it.

    Returns (econ_penalty, value, cost); all normalised to [0,1] on ``mask``."""
    v_nodes = (kernel_sum(db_xy, [P["ECON_DB_WEIGHT"]] * len(db_xy), gx, gy,
                          P["ECON_DEMAND_RADIUS_M"])
               + kernel_sum(dk_xy, [P["ECON_DK_WEIGHT"]] * len(dk_xy), gx, gy,
                            P["ECON_DEMAND_RADIUS_M"]))
    value = _norm(P["ECON_VALUE_NODE_W"] * _norm(v_nodes, mask)
                  + P["ECON_VALUE_FLOW_W"] * _norm(flow_field, mask), mask)

    d_db = dist_to_points(db_xy, gx, gy)
    c_energy = 1.0 - np.exp(-d_db / max(P["ECON_ENERGY_RADIUS_M"], 1e-6))
    c_infra = 1.0 - np.exp(-d_net_m / max(P["ECON_INFRA_RADIUS_M"], 1e-6))
    cost = _norm(P["ECON_COST_ENERGY_W"] * c_energy
                 + P["ECON_COST_INFRA_W"] * c_infra, mask)

    balance = value - cost                       # in [-1, 1]; > 0 = value wins
    penalty = (1.0 - balance) / 2.0              # in [0, 1]; high = uneconomic

    # equity: the already-saturated cores are penalised so new capacity spreads
    q = float(np.clip(P["ECON_SATURATION_Q"], 0.0, 0.999))
    thr = float(np.quantile(value[mask], q)) if np.any(mask) else 1.0
    sat = np.clip((value - thr) / max(1.0 - thr, 1e-6), 0.0, 1.0)
    w_eq = float(np.clip(P["ECON_EQUITY_W"], 0.0, 1.0))
    return np.clip((1.0 - w_eq) * penalty + w_eq * sat, 0.0, 1.0), value, cost


def air_layer(P, gx, gy, flow_field, rings, node_deg_xy, ra_mask, res, mask):
    """(2) an toàn vận hành trên không -- encounter, merge and infringement
    exposure implied by the network topology."""
    enc = _norm(flow_field, mask)

    junc = np.zeros(gx.shape, float)
    sig = P["AIR_JUNCTION_SIGMA_M"]
    if rings is not None and len(rings):
        n_ent = (rings["n_entries"].to_numpy(float)
                 if "n_entries" in rings.columns else np.ones(len(rings)))
        junc += kernel_sum(rings[["center_x", "center_y"]].to_numpy(float),
                           n_ent, gx, gy, sig)
    if len(node_deg_xy):
        pts = np.asarray([p for p, _ in node_deg_xy], float)
        wts = [max(d - 2.0, 0.0) for _, d in node_deg_xy]   # only real junctions
        junc += kernel_sum(pts, wts, gx, gy, sig)
    junc = _norm(junc, mask)

    if ra_mask.any():
        d_ra = distance_transform_edt(~ra_mask) * res
        ra_prox = np.exp(-d_ra / max(P["AIR_RA_BUFFER_M"], 1e-6))
    else:
        ra_prox = np.zeros(gx.shape, float)

    air = (P["AIR_FLOW_W"] * enc + P["AIR_JUNCTION_W"] * junc
           + P["AIR_RA_W"] * ra_prox)
    return _norm(air, mask), enc, junc, ra_prox


def ground_layer(P, gx, gy, obst_frac, apron_xy, flz_xy, res, mask):
    """(3) an toàn mặt đất -- third-party risk on the ground under the corridor."""
    sig_cells = P["GROUND_EXPOSURE_SIGMA_M"] / res
    built = gaussian_filter(obst_frac, sigma=sig_cells)      # people/assets proxy
    apron = kernel_sum(apron_xy, [1.0] * len(apron_xy), gx, gy,
                       P["GROUND_APRON_RADIUS_M"])
    exposure = _norm(built, mask) + P["GROUND_APRON_W"] * _norm(apron, mask)

    # sheltering: part of the exposed population is indoors, under a roof
    shelter = np.clip(P["GROUND_SHELTER_FACTOR"] * _norm(built, mask), 0.0, 1.0)
    exposure = exposure * (1.0 - shelter)

    # ballistic / glide footprint: the crash is felt around the track, not only
    # directly beneath it
    exposure = gaussian_filter(exposure, sigma=P["GROUND_IMPACT_RADIUS_M"] / res)

    # emergency-landing relief: no FLZ in reach -> nothing mitigates the fall
    d_flz = dist_to_points(flz_xy, gx, gy)
    flz_gap = 1.0 - np.exp(-d_flz / max(P["GROUND_FLZ_REACH_M"], 1e-6))
    ground = exposure * (1.0 + P["GROUND_FLZ_W"] * flz_gap)
    return _norm(ground, mask), _norm(exposure, mask), flz_gap


def traffic_layer(P, traffic_csv: Path, xedges, yedges, res, mask):
    """(4) measured traffic density from a pass-1 simulation (optional)."""
    traf = pd.read_csv(traffic_csv)
    w = np.ones(len(traf))
    if "holding" in traf.columns and P["HOLD_WEIGHT"] != 1.0:
        w = np.where(traf["holding"].astype(bool), P["HOLD_WEIGHT"], 1.0)
    H, _, _ = np.histogram2d(traf["x"].to_numpy(float), traf["y"].to_numpy(float),
                             bins=[xedges, yedges], weights=w)
    dens = gaussian_filter(H.T, sigma=P["SMOOTH_SIGMA_M"] / res)
    return _norm(dens, mask), dens, len(traf)


# ----------------------------------------------------------------------
# plotting
# ----------------------------------------------------------------------
def draw_network(ax, corridor_dir: Path, lane_gap: float = 0.0, labels: bool = True):
    """Light overlay of lane centrelines + roundabout rings + objectives.
    Each roundabout draws BOTH circulating lanes (outer at radius_m, inner at
    radius_m - lane_gap), matching 05's two-ring model and the corridor mask."""
    lanes = pd.read_csv(corridor_dir / "lane_nodes.csv")
    for leg_id in lanes["leg_id"].unique():
        for lane in ("A", "B"):
            g = lanes[(lanes["leg_id"] == leg_id) & (lanes["lane"] == lane)].sort_values("seq")
            xy = g[["x", "y"]].to_numpy(float)
            if len(xy) >= 2:
                ax.plot(xy[:, 0], xy[:, 1], "-", color="0.5", lw=0.6, alpha=0.7, zorder=5)
    ring_file = corridor_dir / "roundabouts.csv"
    if ring_file.exists():
        from matplotlib.patches import Circle
        for r in pd.read_csv(ring_file).itertuples():
            c = (float(r.center_x), float(r.center_y))
            r_out = float(r.radius_m)
            r_in = max(0.3 * r_out, r_out - lane_gap)  # mirrors 05 (inner ring lane)
            for rr in (r_out, r_in):
                ax.add_patch(Circle(c, rr, fill=False, ec="0.5", lw=0.6, alpha=0.7, zorder=5))
    nodes = pd.read_csv(corridor_dir / "network_nodes.csv")
    obj = nodes[nodes["kind"] == "objective"]
    for r in obj.itertuples():
        is_db = str(r.net_id).startswith("DB")
        ax.scatter([r.x], [r.y], s=70 if labels else 24, marker="s" if is_db else "^",
                   c="#c0392b" if is_db else "#1f6f3f", edgecolors="k",
                   linewidths=0.5, zorder=7)
        if labels:
            ax.annotate(str(r.net_id), (r.x, r.y), textcoords="offset points",
                        xytext=(5, 4), fontsize=7, weight="bold", zorder=8)


def layers_figure(panels, extent, corridor_dir, lane_gap, out_png):
    """One page with every assessment layer side by side."""
    n = len(panels)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 5.9 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, field, cmap, vmin, vmax, cbl) in zip(axes, panels):
        im = ax.imshow(field, origin="lower", extent=extent, cmap=cmap,
                       vmin=vmin, vmax=vmax, zorder=1)
        draw_network(ax, corridor_dir, lane_gap, labels=False)
        fig.colorbar(im, ax=ax, shrink=0.8).set_label(cbl, fontsize=8)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    P = resolve_params(args)
    corridor_dir = THIS_DIR / args.corridor_dir
    out_dir = THIS_DIR / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    res = float(P["RES_M"])

    print("=" * 70)
    print(f"engine_costmap.py  {VERSION}   (economic / air / ground / traffic)")
    print(f"Corridor dir  : {args.corridor_dir}")
    print(f"Output dir    : {args.out_dir}")
    print("=" * 70)

    # ---- the network this map prices must exist ----
    need = ["network_nodes.csv", "network_legs.csv", "lane_nodes.csv"]
    missing = [f for f in need if not (corridor_dir / f).exists()]
    if missing:
        raise SystemExit(
            f"stage 05 output missing in {args.corridor_dir}: {', '.join(missing)}.\n"
            "Build the corridor network first (05a/05b) -- the cost-map prices "
            "that network, it cannot be built without it.")

    # ---- bounding box from the corridor network ----
    nodes = pd.read_csv(corridor_dir / "network_nodes.csv")
    legs = pd.read_csv(corridor_dir / "network_legs.csv")
    lanes = pd.read_csv(corridor_dir / "lane_nodes.csv")
    ring_file = corridor_dir / "roundabouts.csv"
    rings = pd.read_csv(ring_file) if ring_file.exists() else None

    pad = 5 * res
    x0 = float(min(nodes["x"].min(), lanes["x"].min())) - pad
    y0 = float(min(nodes["y"].min(), lanes["y"].min())) - pad
    x1 = float(max(nodes["x"].max(), lanes["x"].max())) + pad
    y1 = float(max(nodes["y"].max(), lanes["y"].max())) + pad
    nx = int(np.ceil((x1 - x0) / res))
    ny = int(np.ceil((y1 - y0) / res))
    xedges = x0 + res * np.arange(nx + 1)
    yedges = y0 + res * np.arange(ny + 1)
    xc = x0 + res * (np.arange(nx) + 0.5)
    yc = y0 + res * (np.arange(ny) + 0.5)
    gx, gy = np.meshgrid(xc, yc)
    extent = [x0, x0 + nx * res, y0, y0 + ny * res]
    print(f"Grid          : {nx} x {ny} cells @ {res:.0f} m")

    # ---- corridor mask (the network this map is assessed on) ----
    lane_gap = (args.roundabout_lane_gap if args.roundabout_lane_gap is not None
                else load_lane_gap(THIS_DIR / args.corridor_param_file))
    half_w = float(P["CORRIDOR_HALF_WIDTH_M"])
    ring_area = bool(P.get("RING_AS_AREA", False))
    island_frac = float(P.get("RING_ISLAND_FRAC", 0.35))
    inside, ring_interior = corridor_mask(lanes, xc, yc, half_w, rings, lane_gap,
                                          ring_as_area=ring_area,
                                          island_frac=island_frac)
    d_net_m = distance_transform_edt(~inside) * res
    print(f"Corridor mask : half-width {half_w:.0f} m, {inside.mean()*100:.1f}% of "
          f"cells inside ({0 if rings is None else len(rings)} roundabouts x2 ring "
          f"lanes, lane gap {lane_gap:.0f} m)")

    # ---- stage-01 layers (obstacles / RA / no-fly / FLZ) ----
    risk_xyz = find_risk_xyz(args.risk_xyz, P)
    flz_xy: list = []
    flz_lbl: list = []
    model_dx = res
    if risk_xyz is None:
        print("Riskmap       : NONE FOUND -- ground/RA terms fall back to zero")
        obst_frac = np.zeros((ny, nx))
        ra_mask = np.zeros((ny, nx), bool)
        nofly = np.zeros((ny, nx), bool)
    else:
        rdf = pd.read_csv(risk_xyz, sep=r"\s+")
        xs01 = np.sort(rdf["x"].unique())
        model_dx = float(np.median(np.diff(xs01))) if len(xs01) > 1 else res
        s = sample_xyz(rdf, ["obstacle_flag", "ra_flag", "slowness"], gx, gy)
        obst_frac = s["obstacle_flag"]
        ra_mask = s["ra_flag"] > 0.5
        nofly = s["slowness"] >= 10.0
        if "label_prefix" in rdf.columns:
            f = rdf[rdf["label_prefix"] == "FLZ"]
            flz_xy = list(zip(f["x"].astype(float), f["y"].astype(float)))
            flz_lbl = [(float(r.x), float(r.y), str(r.label)) for r in f.itertuples()]
        print(f"Riskmap       : {risk_xyz.name}  "
              f"({int(ra_mask.sum())} RA / {int(nofly.sum())} no-fly cells, "
              f"{len(flz_xy)} FLZ)")
    passable = ~nofly

    # normalisation domain: the flyable airspace (honest across the whole map)
    dom = passable
    if not dom.any():
        dom = np.ones((ny, nx), bool)

    # ---- expected flow on the established network ----
    flow = leg_flow(corridor_dir, legs)
    flow_field = gaussian_filter(stamp_lane_weights(lanes, flow, xc, yc, res),
                                 sigma=P["SMOOTH_SIGMA_M"] / res)
    n_pairs = sum(1 for v in flow.values() if v > 1.0)
    print(f"Network flow  : {len(legs)} legs, {n_pairs} carry DB->DK pair routes, "
          f"max leg load {max(flow.values()):.0f}")

    # ---- objective / junction geometry ----
    obj = nodes[nodes["kind"] == "objective"]
    db_xy = list(zip(obj[obj["net_id"].astype(str).str.startswith("DB")]["x"].astype(float),
                     obj[obj["net_id"].astype(str).str.startswith("DB")]["y"].astype(float)))
    dk_xy = list(zip(obj[obj["net_id"].astype(str).str.startswith("DK")]["x"].astype(float),
                     obj[obj["net_id"].astype(str).str.startswith("DK")]["y"].astype(float)))
    deg = pd.concat([legs["a_id"], legs["b_id"]]).value_counts().to_dict()
    xy_of = {str(r.net_id): (float(r.x), float(r.y)) for r in nodes.itertuples()}
    node_deg_xy = [(xy_of[k], float(v)) for k, v in deg.items() if str(k) in xy_of]
    if not flz_xy:
        flz_xy = dk_xy                      # docks are the fallback landing option

    # ---- (1) (2) (3) assessed layers ----
    econ, econ_value, econ_cost = economic_layer(P, gx, gy, flow_field, db_xy, dk_xy,
                                                 d_net_m, dom)
    air, air_enc, air_junc, air_ra = air_layer(P, gx, gy, flow_field, rings,
                                               node_deg_xy, ra_mask, res, dom)
    ground, ground_exp, flz_gap = ground_layer(P, gx, gy, obst_frac,
                                               list(db_xy) + list(dk_xy), flz_xy,
                                               res, dom)

    # ---- (4) measured traffic (optional) ----
    traffic = np.zeros((ny, nx))
    dens_s = None
    n_traf = 0
    traffic_path = (THIS_DIR / args.traffic) if args.traffic else None
    if traffic_path is not None and traffic_path.exists():
        traffic, dens_s, n_traf = traffic_layer(P, traffic_path, xedges, yedges, res, dom)
        print(f"Traffic layer : {n_traf} samples from {args.traffic}")
    else:
        if args.traffic:
            print(f"Traffic layer : SKIPPED (no {args.traffic}); "
                  f"assessed layers only -- re-run after a pass-1 sim to fold it in")
        else:
            print("Traffic layer : not requested")

    # ---- composite risk -> slowness ----
    # Re-scale every layer over the BLEND DOMAIN before mixing. Without this the
    # layers arrive with different in-corridor spreads (the economic balance
    # varies far less across a built network than the safety terms do), so the
    # nominal weights would not be the effective ones.
    blend_dom = inside if (str(P["NORMALISE_ON"]) == "network" and inside.any()) else dom
    econ_b = _norm(econ, blend_dom, hi_pct=100.0)
    air_b = _norm(air, blend_dom, hi_pct=100.0)
    ground_b = _norm(ground, blend_dom, hi_pct=100.0)
    traffic_b = _norm(traffic, blend_dom, hi_pct=100.0) if dens_s is not None else traffic
    econ, air, ground, traffic = econ_b, air_b, ground_b, traffic_b

    W = {"econ": float(P["W_ECON"]), "air": float(P["W_AIR"]),
         "ground": float(P["W_GROUND"]),
         "traffic": float(P["W_TRAFFIC"]) if dens_s is not None else 0.0}
    wsum = sum(W.values())
    if wsum <= 0:
        raise SystemExit("every layer weight is 0 -- nothing to build a cost-map from")
    risk = np.clip((W["econ"] * econ + W["air"] * air + W["ground"] * ground
                    + W["traffic"] * traffic) / wsum, 0.0, 1.0)

    lo, hi = float(P["MIN_SLOWNESS"]), float(P["MAX_SLOWNESS"])
    slowness = np.clip(hi - risk * (hi - lo), lo, hi)

    # ---- confine to corridors: keep the assessed slowness INSIDE, crawl OUTSIDE ----
    out_s = float(P["OUTSIDE_SLOWNESS"])
    interior_slow = (float(P["RING_INTERIOR_SLOWNESS"])
                     if P["RING_INTERIOR_SLOWNESS"] is not None else out_s)
    slowness = np.where(inside, slowness, out_s)
    # raise cost inside each roundabout's inner ring (the enclosed disk is not a
    # lane); applied AFTER the corridor confinement so it overrides any lane that
    # clips through the centre
    slowness = np.where(ring_interior, interior_slow, slowness)
    print(f"Weights       : econ {W['econ']}  air {W['air']}  ground {W['ground']}  "
          f"traffic {W['traffic']}   (normalised on {P['NORMALISE_ON']})")
    print(f"In-corridor   : risk mean {risk[inside].mean():.3f} "
          f"[{risk[inside].min():.3f}, {risk[inside].max():.3f}]  ->  slowness "
          f"[{slowness[inside].min():.3f}, {slowness[inside].max():.3f}]")

    np.savez(out_dir / "slowness_costmap.npz",
             slowness=slowness.astype(np.float32),
             x0=x0, y0=y0, res=res,
             risk=risk.astype(np.float32),
             econ=econ.astype(np.float32),
             air=air.astype(np.float32),
             ground=ground.astype(np.float32),
             traffic=traffic.astype(np.float32),
             inside=inside,
             weights=np.array([W["econ"], W["air"], W["ground"], W["traffic"]]))
    print(f"Saved         : {out_dir/'slowness_costmap.npz'}")

    # ---- per-leg assessment ON the established network ----
    rows = []
    step = max(1.0, res / 2.0)
    for leg in legs.itertuples():
        lid = str(leg.leg_id)
        g = lanes[(lanes["leg_id"] == lid) & (lanes["lane"] == "A")].sort_values("seq")
        xy = g[["x", "y"]].to_numpy(float)
        if len(xy) < 2:
            continue
        pts, _ = resample_lane(xy, step)
        jx = np.clip(((pts[:, 0] - x0) / res).astype(int), 0, nx - 1)
        jy = np.clip(((pts[:, 1] - y0) / res).astype(int), 0, ny - 1)
        rows.append({
            "leg_id": lid, "a_id": leg.a_id, "b_id": leg.b_id,
            "leg_type": leg.leg_type, "length_m": round(float(leg.length_m), 1),
            "pair_flow": round(flow.get(lid, 0.0), 1),
            "econ": round(float(econ[jy, jx].mean()), 4),
            "air": round(float(air[jy, jx].mean()), 4),
            "ground": round(float(ground[jy, jx].mean()), 4),
            "traffic": round(float(traffic[jy, jx].mean()), 4),
            "risk": round(float(risk[jy, jx].mean()), 4),
            "risk_max": round(float(risk[jy, jx].max()), 4),
            "slowness": round(float(slowness[jy, jx].mean()), 4),
        })
    assess = pd.DataFrame(rows).sort_values("risk", ascending=False)
    assess.to_csv(out_dir / "network_assessment.csv", index=False)
    print(f"Assessment    : {len(assess)} legs -> network_assessment.csv "
          f"(worst {assess.iloc[0]['leg_id']} risk {assess.iloc[0]['risk']:.3f})"
          if len(assess) else "Assessment    : no legs")

    # ---- metrics ----
    gini = _gini(econ_value[inside]) if inside.any() else float("nan")
    def _stat(a):
        v = a[inside] if inside.any() else a.ravel()
        return {"mean": round(float(v.mean()), 4), "min": round(float(v.min()), 4),
                "max": round(float(v.max()), 4), "p90": round(float(np.percentile(v, 90)), 4)}
    metrics = {
        "version": VERSION,
        "grid": {"nx": nx, "ny": ny, "res_m": res, "x0": x0, "y0": y0},
        "weights": W,
        "normalise_on": str(P["NORMALISE_ON"]),
        "corridor": {"half_width_m": half_w,
                     "frac_cells_inside": round(float(inside.mean()), 4),
                     "n_legs": int(len(legs)),
                     "n_roundabouts": 0 if rings is None else int(len(rings)),
                     "lane_gap_m": lane_gap},
        "traffic_samples": int(n_traf),
        "layers": {"econ": _stat(econ), "air": _stat(air), "ground": _stat(ground),
                   "traffic": _stat(traffic), "risk": _stat(risk),
                   "slowness": _stat(slowness)},
        "economic_balance": {
            "value_gini_on_network": None if not np.isfinite(gini) else round(gini, 4),
            "mean_value": round(float(econ_value[inside].mean()), 4) if inside.any() else None,
            "mean_cost": round(float(econ_cost[inside].mean()), 4) if inside.any() else None,
            "net_balance": round(float((econ_value - econ_cost)[inside].mean()), 4) if inside.any() else None,
        },
        "worst_legs": assess.head(5)[["leg_id", "risk", "econ", "air", "ground"]]
                            .to_dict("records") if len(assess) else [],
        "safest_legs": assess.tail(5)[["leg_id", "risk", "econ", "air", "ground"]]
                             .to_dict("records") if len(assess) else [],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Economic Gini : {gini:.3f} (0 = value spread evenly over the network)")
    print(f"Saved         : {out_dir/'metrics.json'}, network_assessment.csv")

    # ---- figures ----
    # nan_out is shared with the HTML viewer below, so it is defined even when
    # the PNGs are skipped (MAKE_FIGURES and MAKE_HTML are independent).
    nan_out = lambda a: np.where(passable, a, np.nan)
    if bool(P["MAKE_FIGURES"]):
        fig, ax = plt.subplots(figsize=(12, 11))
        im = ax.imshow(slowness, origin="lower", extent=extent, cmap="RdYlGn",
                       vmin=lo, vmax=hi, zorder=1)
        ax.contour(xc, yc, inside.astype(float), levels=[0.5],
                   colors="k", linewidths=0.6, linestyles="--", zorder=6)
        draw_network(ax, corridor_dir, lane_gap)
        cb = fig.colorbar(im, ax=ax, shrink=0.75)
        cb.set_label("slowness  (1 = full speed, low = costly/unsafe)")
        ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_title("Slowness cost-map  "
                     f"(econ {W['econ']} / air {W['air']} / ground {W['ground']}"
                     f" / traffic {W['traffic']})")
        add_map_rule(ax, extent[0], extent[2], extent[1], extent[3])
        fig.tight_layout(); fig.savefig(out_dir / "slowness_costmap.png", dpi=130)
        plt.close(fig)

        scale_note = ("scaled on the corridor network" if blend_dom is inside
                      else "scaled on the flyable area")
        panels = [
            ("(1) Economic balance  (cân bằng kinh tế)\nhigh = cost outweighs value",
             nan_out(econ), "PuOr", 0, 1, "economic penalty"),
            ("(2) Air operational safety  (an toàn trên không)\nhigh = encounter / merge / RA exposure",
             nan_out(air), "Blues", 0, 1, "air risk"),
            ("(3) Ground safety  (an toàn mặt đất)\nhigh = third-party exposure below",
             nan_out(ground), "Reds", 0, 1, "ground risk"),
            ("Economic value V  (demand + corridor throughput)",
             nan_out(econ_value), "YlGn", 0, 1, "value"),
            (f"Composite risk  (weighted blend, {scale_note})",
             nan_out(risk), "magma", 0, 1, "risk"),
            ("Slowness cost-map  (network only)",
             slowness, "RdYlGn", lo, hi, "slowness"),
        ]
        if dens_s is not None:
            panels.insert(3, ("(4) Measured traffic  (pass-1 sim)",
                              nan_out(traffic), "inferno", 0, 1, "traffic density"))
        layers_figure(panels, extent, corridor_dir, lane_gap, out_dir / "cost_layers.png")
        print("Figures       : slowness_costmap.png, cost_layers.png")

        if dens_s is not None:
            fig, ax = plt.subplots(figsize=(12, 11))
            im = ax.imshow(dens_s, origin="lower", extent=extent, cmap="inferno", zorder=1)
            draw_network(ax, corridor_dir, lane_gap)
            fig.colorbar(im, ax=ax, shrink=0.75).set_label("smoothed traffic density")
            ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
            ax.set_title("Gaussian-smoothed traffic density (pass-1 sim)")
            add_map_rule(ax, extent[0], extent[2], extent[1], extent[3])
            fig.tight_layout(); fig.savefig(out_dir / "density.png", dpi=130)
            plt.close(fig)
            print("                density.png")

    # ---- interactive viewer: swap the component maps, toggle the model nodes ----
    if bool(P["MAKE_HTML"]):
        html_layers = [
            ("risk", "composite risk", "risk",
             f"weighted blend: econ {W['econ']} / air {W['air']} / "
             f"ground {W['ground']} / traffic {W['traffic']}",
             nan_out(risk), "magma", 0.0, 1.0),
            ("econ", "economic balance", "econ",
             "cân bằng kinh tế — high where the cost of serving an area "
             "outweighs the value it returns",
             nan_out(econ), "PuOr", 0.0, 1.0),
            ("air", "air safety", "air",
             "an toàn vận hành trên không — encounter density, roundabout/"
             "junction merges, restricted-airspace proximity",
             nan_out(air), "Blues", 0.0, 1.0),
            ("ground", "ground safety", "ground",
             "an toàn mặt đất — third-party exposure below the corridor, "
             "sheltered, spread over the crash footprint, minus FLZ relief",
             nan_out(ground), "Reds", 0.0, 1.0),
            ("value", "economic value", "value",
             "V — DB/DK service demand plus expected corridor throughput",
             nan_out(econ_value), "YlGn", 0.0, 1.0),
            ("slowness", "slowness", "slowness",
             "what stage 07 flies: true velocity = slowness x base speed",
             np.where(passable, slowness, np.nan), "RdYlGn", lo, hi),
        ]
        if dens_s is not None:
            html_layers.insert(4, ("traffic", "measured traffic", "traffic",
                                   "Gaussian-smoothed density of the pass-1 "
                                   "simulated trajectories",
                                   nan_out(traffic), "inferno", 0.0, 1.0))
        n_bytes = render_costmap_html(
            out_dir / "costmap.html", html_layers, extent, res,
            nofly=nofly, ra_mask=ra_mask, inside=inside, lanes=lanes,
            legs_assess=assess, nodes=nodes, rings=rings, flz_xy=flz_lbl,
            model_dx=model_dx,
            meta={"title": f"Stage 06 cost-map — {args.out_dir}",
                  "subtitle": ("economic balance · air safety · ground safety"
                               + (" · measured traffic" if dens_s is not None else "")),
                  "w_econ": W["econ"], "w_air": W["air"], "w_ground": W["ground"],
                  "w_traffic": W["traffic"], "n_legs": len(legs)})
        print(f"                costmap.html ({n_bytes/1e6:.1f} MB, "
              f"{len(html_layers)} component maps)")
    print(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
