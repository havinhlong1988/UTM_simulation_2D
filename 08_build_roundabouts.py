#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# 08_build_roundabouts.py
# ======================================================================
# Build TRAFFIC ROUNDABOUTS at the network's high-conflict junctions so
# agents can circulate through instead of holding at a point node.
#
# Pipeline position (between 07 and the cost-map):
#     07_build_corridor_network.py -> corridor network (nodes/legs/lanes)
#     08_build_roundabouts.py      (this) -> roundabout layer
#     09_generate_costmap.py / 10_simulate_agents_2d.py / 11_fmm_route_plan.py
#
# What it does
# ------------
#   1. RANK traffic nodes by conflict density -- proxied by the number of
#      DB<->DK pair-routes that traverse the node (through-load), with node
#      degree and leg-crossings as secondary signals. No simulation needed.
#   2. SELECT high-conflict seed nodes (through-load >= MIN_THROUGH_ROUTES).
#   3. CLUSTER nearby seeds (single-linkage within MERGE_RADIUS_M) and ABSORB
#      any close-by neighbour node, so a vicinity of junctions becomes ONE
#      roundabout (e.g. MAJ012+MAJ007, MAJ006+MIN011, MAJ013+MIN002+MIN001);
#      an isolated high node becomes a single-node roundabout (e.g. MIN003).
#   4. BUILD each roundabout as a one-way ring: a centre, an inner radius that
#      encircles the members, and M CONCENTRIC LANES. Every corridor that used
#      to end at a member is re-terminated at a tangential MERGE point on the
#      ring at the bearing of its approach.
#   5. OPTIMISE the lane count M from the MERGE demand: corridors that approach
#      within MERGE_SEP_DEG of one another cannot merge into a single lane at
#      once, so M = the peak angular crowding of entries (clamped to MAX_LANES).
#
# Outputs (output/08_roundabout_network/)
#     roundabouts.csv           one row per roundabout (centre, R, M, members...)
#     roundabout_members.csv    node -> roundabout membership
#     roundabout_entries.csv    incident corridors -> merge point + lane
#     ring_lanes.csv            ring lane polylines (rbt_id, lane, seq, x, y)
#     roundabouts_summary.json  run parameters + totals
#     figures/00_roundabout_network.png
#
# Usage
#     python 08_build_roundabouts.py
#     python 08_build_roundabouts.py --param-file params/roundabouts.params
# ======================================================================
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.maprule import add_map_rule

THIS_DIR = Path(__file__).resolve().parent
VERSION = "v1"


# ----------------------------------------------------------------------
# params (same tiny loader convention as the rest of the pipeline)
# ----------------------------------------------------------------------
def parse_value(raw: str) -> Any:
    raw = raw.strip()
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    if raw.lower() == "none":
        return None
    try:
        return ast.literal_eval(raw)
    except Exception:
        return raw.strip('"').strip("'")


def load_params(param_file: str | Path) -> dict[str, Any]:
    p = Path(param_file)
    params: dict[str, Any] = {}
    if not p.exists():
        return params
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "#" in line:
            line = line.split("#", 1)[0].strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        params[k.strip()] = parse_value(v.strip())
    return params


def pget(params: dict[str, Any], key: str, default: Any) -> Any:
    v = params.get(key, default)
    return default if v is None else v


# ----------------------------------------------------------------------
# network + scoring
# ----------------------------------------------------------------------
def load_network(corridor_dir: Path):
    nodes = pd.read_csv(corridor_dir / "network_nodes.csv")
    legs = pd.read_csv(corridor_dir / "network_legs.csv")
    pair = pd.read_csv(corridor_dir / "pair_routes.csv")
    return nodes, legs, pair


def score_nodes(nodes, legs, pair) -> pd.DataFrame:
    """Per traffic-node conflict-density proxy: through-load (# pair-routes
    traversing it) + degree (incident legs) + crossings (physical corridor
    crossings on incident legs)."""
    xy = {r.net_id: (float(r.x), float(r.y)) for r in nodes.itertuples()}
    kind = {r.net_id: r.kind for r in nodes.itertuples()}

    load = {n: 0 for n in xy}
    for r in pair.itertuples():
        if not bool(r.success):
            continue
        for n in str(r.via).split("-"):
            if n in load:
                load[n] += 1

    deg = {n: 0 for n in xy}
    cross = {n: 0 for n in xy}
    for r in legs.itertuples():
        for end in (r.a_id, r.b_id):
            deg[end] += 1
            cross[end] += int(getattr(r, "n_crossings", 0) or 0)

    rows = []
    for n in xy:
        if str(kind[n]) == "objective":
            continue                       # never turn a depot/delivery into a ring
        rows.append(dict(node=n, x=xy[n][0], y=xy[n][1], kind=kind[n],
                         through_load=load[n], degree=deg[n], crossings=cross[n]))
    df = pd.DataFrame(rows)
    df = df.sort_values(["through_load", "degree", "crossings"],
                        ascending=False).reset_index(drop=True)
    return df


# ----------------------------------------------------------------------
# clustering (single-linkage within radius) + neighbour absorption
# ----------------------------------------------------------------------
class _UF:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def cluster_seeds(seeds, xy, radius):
    uf = _UF(seeds)
    for i, a in enumerate(seeds):
        for b in seeds[i + 1:]:
            if math.dist(xy[a], xy[b]) <= radius:
                uf.union(a, b)
    groups: dict[str, list] = {}
    for s in seeds:
        groups.setdefault(uf.find(s), []).append(s)
    return list(groups.values())


def absorb_neighbours(members, all_tn, xy, radius):
    """Pull any non-member traffic node within `radius` of a member into the
    roundabout (so a nearby junction is combined even if it wasn't a seed)."""
    members = set(members)
    changed = True
    while changed:
        changed = False
        for n in all_tn:
            if n in members:
                continue
            if any(math.dist(xy[n], xy[m]) <= radius for m in members):
                members.add(n)
                changed = True
    return sorted(members)


# ----------------------------------------------------------------------
# roundabout geometry + lane optimisation
# ----------------------------------------------------------------------
def optimise_lanes(angles_deg, merge_sep_deg, routes_per_lane, max_lanes):
    """M = peak angular crowding of entries: the most entries falling within
    any MERGE_SEP_DEG window (they cannot all merge into one lane at once),
    but at least ceil(n_entries / routes_per_lane). Clamped to [1, max_lanes]."""
    n = len(angles_deg)
    if n == 0:
        return 1
    a = np.sort(np.asarray(angles_deg) % 360.0)
    ext = np.concatenate([a, a + 360.0])          # wrap for circular windows
    crowd = 1
    for i in range(n):
        cnt = int(np.sum((ext >= a[i]) & (ext < a[i] + merge_sep_deg)))
        crowd = max(crowd, cnt)
    by_capacity = math.ceil(n / max(1, routes_per_lane))
    return int(max(1, min(max_lanes, max(crowd, by_capacity))))


def assign_lanes_by_cost(entries, m):
    """Rank entries by route COST (leg length); the HIGHER-cost routes take the
    OUTER ring lanes, the cheaper ones the inner lanes. Several entries may share
    a lane (the lane holds them with spacing)."""
    if not entries:
        return
    order = sorted(range(len(entries)), key=lambda i: entries[i]["cost"])
    n = len(order)
    for rank, idx in enumerate(order):
        entries[idx]["lane"] = 0 if n == 1 else int(round(rank / (n - 1) * (m - 1)))


def ring_polyline(cx, cy, radius, n=120):
    t = np.linspace(0.0, 2 * np.pi, n)
    return np.column_stack([cx + radius * np.cos(t), cy + radius * np.sin(t)])


def quad_bezier(p0, p1, p2, n=32):
    t = np.linspace(0.0, 1.0, n).reshape(-1, 1)
    p0, p1, p2 = map(np.asarray, (p0, p1, p2))
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def offset_polyline(pts, d):
    """Shift a polyline sideways by d along its local normal (for drawing the two
    parallel lanes of a couple)."""
    pts = np.asarray(pts, float)
    tang = np.gradient(pts, axis=0)
    nrm = np.column_stack([-tang[:, 1], tang[:, 0]])
    nrm /= (np.hypot(nrm[:, 0], nrm[:, 1])[:, None] + 1e-9)
    return pts + d * nrm


def draw_ring_with_removals(ax, cx, cy, radius, intervals, color, lw=1.9, z=4):
    """Draw the full ring circle but cut out the SHORT arc between each (a1, a2)
    pair -- the small opening where a corridor couple penetrates the ring. The
    ring itself is otherwise unchanged (a plain circle)."""
    t = np.linspace(0.0, 360.0, 2881)
    keep = np.ones_like(t, dtype=bool)
    for a1, a2 in intervals:
        lo, hi = sorted(((a1 % 360.0), (a2 % 360.0)))
        if hi - lo <= 180.0:
            keep &= ~((t >= lo) & (t <= hi))
        else:                                    # the short arc wraps through 0
            keep &= ~((t <= lo) | (t >= hi))
    idx = np.where(keep)[0]
    if len(idx) == 0:
        return
    for run in np.split(idx, np.where(np.diff(idx) != 1)[0] + 1):
        tt = np.radians(t[run])
        ax.plot(cx + radius * np.cos(tt), cy + radius * np.sin(tt), "-",
                color=color, lw=lw, alpha=0.9, zorder=z)


def clip_polyline_outside_circles(xy, circles):
    """Split a polyline into the sub-paths that lie OUTSIDE every circle, cutting
    it exactly on each circle. Returns (subpaths, meeting_pts): meeting_pts are the
    cut points that land on a ring (where a corridor meets a roundabout)."""
    pts = [np.asarray(p, float) for p in xy]
    if len(pts) < 2:
        return [np.asarray(xy, float)], []

    def inside(p):
        return any((p[0] - cx) ** 2 + (p[1] - cy) ** 2 < (R - 1e-6) ** 2
                   for cx, cy, R in circles)

    subpaths, cur, meets = [], [], []
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        d = b - a
        ts = {0.0, 1.0}
        for cx, cy, R in circles:                         # segment x circle roots
            f = a - np.array([cx, cy])
            A = float(d @ d); B = float(2 * f @ d); C = float(f @ f - R * R)
            disc = B * B - 4 * A * C
            if disc > 0 and A > 1e-9:
                sq = math.sqrt(disc)
                for t in ((-B - sq) / (2 * A), (-B + sq) / (2 * A)):
                    if 0.0 < t < 1.0:
                        ts.add(t)
        ts = sorted(ts)
        for t0, t1 in zip(ts[:-1], ts[1:]):
            p0, p1 = a + t0 * d, a + t1 * d
            if inside(a + 0.5 * (t0 + t1) * d):
                continue
            if cur and np.allclose(cur[-1], p0):
                cur.append(p1)
            else:
                if cur:
                    subpaths.append(np.array(cur))
                cur = [p0, p1]
    if cur:
        subpaths.append(np.array(cur))
    for s in subpaths:                                    # endpoints landing on a ring
        for endp in (s[0], s[-1]):
            if any(abs(math.hypot(endp[0] - cx, endp[1] - cy) - R) < 1.0
                   for cx, cy, R in circles):
                meets.append(endp)
    return subpaths, meets


def _dmin(cx, cy, pts):
    if len(pts) == 0:
        return 1e9
    return float(np.min(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)))


def nudge_center(cx, cy, blocked, max_nudge):
    """Shift the centre a little (<= max_nudge) toward free space to maximise
    clearance from blocked (obstacle / no-fly) cells, preferring small moves."""
    best = (cx, cy, _dmin(cx, cy, blocked))
    if max_nudge <= 0 or len(blocked) == 0:
        return best
    best_score = best[2]
    for r in np.linspace(max_nudge / 6, max_nudge, 6):
        for a in np.linspace(0, 2 * np.pi, 24, endpoint=False):
            px, py = cx + r * math.cos(a), cy + r * math.sin(a)
            d = _dmin(px, py, blocked)
            score = d - 0.25 * r            # reward clearance, penalise moving far
            if score > best_score:
                best_score, best = score, (px, py, d)
    return best


def build_partial(rbt_id, members, nodes_xy, legs, leg_len, params):
    """Cluster -> centre, spread, desired lane count M, and incident corridor
    ENTRIES (external node + route cost). Radii are finalised later by
    fit_geometry so rings clear obstacles and one another."""
    seeds_xy = np.array([nodes_xy[m] for m in members], float)
    cx0, cy0 = seeds_xy.mean(axis=0)
    spread = float(np.max(np.hypot(seeds_xy[:, 0] - cx0, seeds_xy[:, 1] - cy0))) \
        if len(members) > 1 else 0.0
    mset = set(members)
    entries, internal = [], []
    for r in legs.itertuples():
        a_in, b_in = r.a_id in mset, r.b_id in mset
        if a_in and b_in:
            internal.append(r.leg_id)                       # in-ring route -> removed
            continue
        if not (a_in or b_in):
            continue
        ext = r.b_id if a_in else r.a_id
        entries.append(dict(from_node=ext, leg_id=r.leg_id,
                            ex=nodes_xy[ext][0], ey=nodes_xy[ext][1],
                            cost=float(leg_len.get(r.leg_id, 0.0))))
    return dict(rbt_id=rbt_id, members=list(members), cx0=float(cx0), cy0=float(cy0),
                spread=spread, entries=entries, internal_legs=internal)


def fit_geometry(rbs, blocked, params):
    """Finalise every ring's centre, radius and lane count under two hard
    constraints: no ring overlaps an obstacle / no-fly cell, and no two rings
    overlap. Diameters (and, if needed, lane counts) are reduced to fit."""
    lane_gap0 = float(pget(params, "LANE_GAP_M", 55.0))
    min_gap = float(pget(params, "MIN_LANE_GAP_M", 30.0))
    min_inner = float(pget(params, "RING_MIN_INNER_M", 45.0))
    inner_des0 = float(pget(params, "RING_MIN_RADIUS_M", 110.0))
    clr = float(pget(params, "OBSTACLE_CLEARANCE_M", 20.0))
    rr_gap = float(pget(params, "RING_RING_GAP_M", 40.0))
    hard_min = float(pget(params, "HARD_MIN_OUTER_M", 22.0))
    max_lanes = int(pget(params, "MAX_LANES", 4))
    routes_per_lane = int(pget(params, "ROUTES_PER_LANE", 4))
    merge_sep = float(pget(params, "MERGE_SEP_DEG", 45.0))
    nudge = float(pget(params, "CENTER_NUDGE_MAX_M", 150.0))
    overrides = pget(params, "LANE_OVERRIDES", {}) or {}

    for rb in rbs:
        rb["cx"], rb["cy"], rb["d_block"] = nudge_center(rb["cx0"], rb["cy0"], blocked, nudge)
        for e in rb["entries"]:                             # angles from FINAL centre
            e["angle_deg"] = math.degrees(math.atan2(e["ey"] - rb["cy"],
                                                     e["ex"] - rb["cx"])) % 360.0
        rb["M_want"] = optimise_lanes([e["angle_deg"] for e in rb["entries"]],
                                      merge_sep, routes_per_lane, max_lanes)
        # per-roundabout lane override, keyed by any member node label
        for mem in rb["members"]:
            if mem in overrides:
                rb["M_want"] = int(max(1, min(max_lanes, overrides[mem])))
                break
        # size the ring to ENCIRCLE its member nodes (so corridors connect to a
        # ring, not to a node inside it), then cap on obstacle clearance
        margin = float(pget(params, "RING_MARGIN_M", 60.0))
        desired = max(inner_des0 + (rb["M_want"] - 1) * lane_gap0,
                      rb["spread"] + margin)
        rb["R_out"] = min(desired, max(hard_min, rb["d_block"] - clr))

    # pairwise shrink so rings (plus a gap) do not overlap; re-cap on obstacles
    for _ in range(300):
        changed = False
        for i in range(len(rbs)):
            for j in range(i + 1, len(rbs)):
                a, b = rbs[i], rbs[j]
                d = math.dist((a["cx"], a["cy"]), (b["cx"], b["cy"]))
                over = (a["R_out"] + b["R_out"] + rr_gap) - d
                if over > 0.5:
                    tot = a["R_out"] + b["R_out"]
                    a["R_out"] -= over * a["R_out"] / tot
                    b["R_out"] -= over * b["R_out"] / tot
                    changed = True
        for rb in rbs:
            rb["R_out"] = max(hard_min, min(rb["R_out"], rb["d_block"] - clr))
        if not changed:
            break

    # SINGLE-RING roundabout (one circulating lane, turbine style): every leg
    # connects to the one ring; shortcuts between close legs are added at draw time.
    single = bool(pget(params, "SINGLE_RING", True))
    for rb in rbs:
        if single:
            rb["m"], rb["lane_gap"], rb["R"] = 1, 0.0, rb["R_out"]
        else:
            m = rb["M_want"]
            while m > 1 and (rb["R_out"] - min_inner) < (m - 1) * min_gap:
                m -= 1
            gap = lane_gap0 if m <= 1 else \
                max(min_gap, min(lane_gap0, (rb["R_out"] - min_inner) / (m - 1)))
            rb["m"], rb["lane_gap"] = m, gap
            rb["R"] = max(rb["R_out"] - (m - 1) * gap, min(min_inner, rb["R_out"] * 0.5))
        assign_lanes_by_cost(rb["entries"], rb["m"])
        for e in rb["entries"]:                             # meeting point on its lane
            rad = rb["R"] + e["lane"] * rb["lane_gap"]
            rr = math.radians(e["angle_deg"])
            e["merge_x"] = rb["cx"] + rad * math.cos(rr)
            e["merge_y"] = rb["cy"] + rad * math.sin(rr)
    return rbs


# ----------------------------------------------------------------------
# obstacle / no-fly grid -> blocked-cell centres (for clearance) + overlay
# ----------------------------------------------------------------------
def load_blocked_cells(model_file, avoid_ra=True):
    """(x,y) centres of obstacle (and, if avoid_ra, no-fly RA) grid cells that a
    ring must not overlap. Empty array if the model is missing."""
    p = THIS_DIR / str(model_file)
    if not p.exists():
        return np.zeros((0, 2))
    df = pd.read_csv(p, sep=r"\s+")
    if "obstacle_flag" not in df.columns:
        return np.zeros((0, 2))
    m = df["obstacle_flag"] > 0
    if avoid_ra and "ra_flag" in df.columns:
        m = m | (df["ra_flag"] > 0)
    return df.loc[m, ["x", "y"]].to_numpy(float)


def draw_obstacles(ax, model_file, zorder=0.5):
    from matplotlib.colors import ListedColormap
    import numpy.ma as ma
    p = THIS_DIR / str(model_file)
    if not p.exists():
        return []
    df = pd.read_csv(p, sep=r"\s+")
    if not {"x", "y", "obstacle_flag"}.issubset(df.columns):
        return []
    xs = np.sort(df["x"].unique()); ys = np.sort(df["y"].unique())
    if len(xs) < 2 or len(ys) < 2:
        return []
    dx = xs[1] - xs[0]; dy = ys[1] - ys[0]
    ix = np.rint((df["x"].to_numpy() - xs[0]) / dx).astype(int)
    iy = np.rint((df["y"].to_numpy() - ys[0]) / dy).astype(int)
    obst = np.zeros((len(ys), len(xs))); obst[iy, ix] = df["obstacle_flag"].to_numpy()
    ra = np.zeros_like(obst)
    if "ra_flag" in df.columns:
        ra[iy, ix] = df["ra_flag"].to_numpy()
    extent = (xs[0] - dx / 2, xs[-1] + dx / 2, ys[0] - dy / 2, ys[-1] + dy / 2)
    handles = []
    ra_only = (ra > 0) & (obst == 0)
    if ra_only.any():
        ax.imshow(ma.masked_where(~ra_only, ra_only), extent=extent, origin="lower",
                  cmap=ListedColormap(["#e67e22"]), alpha=0.25, zorder=zorder,
                  interpolation="nearest", aspect="auto")
        handles.append(Patch(facecolor="#e67e22", alpha=0.4, label="restricted / no-fly (RA)"))
    if (obst > 0).any():
        ax.imshow(ma.masked_where(obst == 0, obst), extent=extent, origin="lower",
                  cmap=ListedColormap(["#4d4d4d"]), alpha=0.38, zorder=zorder + 0.05,
                  interpolation="nearest", aspect="auto")
        handles.append(Patch(facecolor="#4d4d4d", alpha=0.5, label="obstacle"))
    return handles


# ----------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------
def plot_roundabouts(nodes, legs, lane_nodes, roundabouts, out_png, obstacle_file,
                     params=None):
    params = params or {}
    subconn_max = float(pget(params, "SUBCONNECTOR_MAX_DEG", 45.0))
    subconn_bulge = float(pget(params, "SUBCONNECTOR_BULGE_M", 55.0))
    subconn_off = float(pget(params, "SUBCONNECTOR_OFFSET_M", 35.0))
    subconn_couple = float(pget(params, "SUBCONNECTOR_COUPLE_M", 16.0))
    LEG_COL, RING_COL, SUB_COL = "#2166ac", "#555555", "#e6550d"
    fig, ax = plt.subplots(figsize=(13, 11))
    obs = draw_obstacles(ax, obstacle_file) if obstacle_file else []

    member_set = {m for rb in roundabouts for m in rb["members"]}
    outer_circles = [(rb["cx"], rb["cy"], rb["R_out"]) for rb in roundabouts]
    incident_legs = {e["leg_id"] for rb in roundabouts for e in rb["entries"]}
    lane_cols = ["#1a9850", "#2166ac", "#762a83", "#d73027"]
    removed = {lg for rb in roundabouts for lg in rb["internal_legs"]}

    def lane_xy(leg_id, lane):
        g = lane_nodes[(lane_nodes["leg_id"] == leg_id) &
                       (lane_nodes["lane"] == lane)].sort_values("seq")
        return g[["x", "y"]].to_numpy(float) if len(g) >= 2 else None

    # (1) faded background: every non-incident, non-removed leg, clipped so no
    #     corridor is drawn inside any ring.
    for leg_id in lane_nodes["leg_id"].unique():
        if leg_id in removed or leg_id in incident_legs:
            continue
        for lane in ("A", "B"):
            xy = lane_xy(leg_id, lane)
            if xy is None:
                continue
            subs, _ = clip_polyline_outside_circles(xy, outer_circles)
            for s in subs:
                ax.plot(s[:, 0], s[:, 1], "-", color="0.82", lw=1.0, zorder=1)

    # (2) incident corridors. The IN/OUT couple PENETRATES radially from outside,
    #     through the outer rings, to its cost-assigned lane (a small shift lets it
    #     aim at the centre). On every ring it crosses, the short arc BETWEEN the
    #     two penetrating lanes is removed. Colour = assigned lane.
    remove_intervals = {(ri, j): [] for ri, rb in enumerate(roundabouts)
                        for j in range(rb["m"])}
    meeting_pts, backup_pts, in_arrows, out_arrows = [], [], [], []
    leg_taps = {ri: [] for ri in range(len(roundabouts))}   # (angle, point) on each leg

    def penetrate(xy, cx, cy, r_k, R_out):
        """Like a road meeting a roundabout: the corridor approaches to the OUTER
        ring, then penetrates RADIALLY inward to its assigned lane r_k and stops
        there -- it never continues to the junction node inside the ring.
        Returns (route_polyline, end_point_on_r_k)."""
        subs, meets = clip_polyline_outside_circles(xy, [(cx, cy, R_out)])
        q = next((mm for mm in meets
                  if abs(math.hypot(mm[0] - cx, mm[1] - cy) - R_out) < 1.5), None)
        if q is not None and subs:                   # meets the outer ring at q
            approach = max(subs, key=len)
            if math.hypot(approach[0, 0] - q[0], approach[0, 1] - q[1]) < \
               math.hypot(approach[-1, 0] - q[0], approach[-1, 1] - q[1]):
                approach = approach[::-1]            # orient external -> q
            q = np.asarray(q)
        else:                                        # ring is small & off to a side
            approach = xy if math.hypot(xy[0, 0] - cx, xy[0, 1] - cy) > \
                math.hypot(xy[-1, 0] - cx, xy[-1, 1] - cy) else xy[::-1]
            u0 = approach[-1] - np.array([cx, cy]); u0 = u0 / (np.hypot(*u0) + 1e-9)
            q = np.array([cx + R_out * u0[0], cy + R_out * u0[1]])
            approach = np.vstack([approach, q])
        ang = math.atan2(q[1] - cy, q[0] - cx)
        u = np.array([math.cos(ang), math.sin(ang)])
        rr = np.linspace(R_out, r_k, max(2, int((R_out - r_k) / 10) + 2))
        pen = np.column_stack([cx + rr * u[0], cy + rr * u[1]])
        return np.vstack([approach, pen]), pen[-1]

    def cross_angle(route, cx, cy, r):
        _, meets = clip_polyline_outside_circles(route, [(cx, cy, r)])
        mm = next((q for q in meets if abs(math.hypot(q[0] - cx, q[1] - cy) - r) < 2.0), None)
        return None if mm is None else math.degrees(math.atan2(mm[1] - cy, mm[0] - cx)) % 360.0

    for ri, rb in enumerate(roundabouts):
        cx, cy, R, gap, m, R_out = rb["cx"], rb["cy"], rb["R"], rb["lane_gap"], rb["m"], rb["R_out"]
        for e in rb["entries"]:
            k = e["lane"]
            r_k = R + k * gap
            col = LEG_COL
            routes = {}
            for role, lane in (("in", "A"), ("out", "B")):
                xy = lane_xy(e["leg_id"], lane)
                if xy is None:
                    continue
                route, endp = penetrate(xy, cx, cy, r_k, R_out)
                routes[role] = route
                ax.plot(route[:, 0], route[:, 1], "-", color=col, lw=1.7, alpha=0.92, zorder=3)
                meeting_pts.append(endp)
                rad = np.array([endp[0] - cx, endp[1] - cy]); rad = rad / (np.hypot(*rad) + 1e-9)
                (in_arrows if role == "in" else out_arrows).append(
                    (endp, -rad if role == "in" else rad))
                if k < m - 1:                        # backup connection on the outer ring
                    ca = cross_angle(route, cx, cy, R_out)
                    if ca is not None:
                        backup_pts.append([cx + R_out * math.cos(math.radians(ca)),
                                           cy + R_out * math.sin(math.radians(ca))])
            # remove the short arc between the two penetrating lanes on each ring crossed
            if "in" in routes and "out" in routes:
                for j in range(k, m):
                    r_j = R + j * gap
                    aA = cross_angle(routes["in"], cx, cy, r_j)
                    aB = cross_angle(routes["out"], cx, cy, r_j)
                    if aA is not None and aB is not None:
                        remove_intervals[(ri, j)].append((aA, aB))
            # tap this leg's MAIN route just before the ring, for sub-connectors
            if "in" in routes:
                _, tm = clip_polyline_outside_circles(routes["in"], [(cx, cy, R + subconn_off)])
                tp = next((q for q in tm
                           if abs(math.hypot(q[0] - cx, q[1] - cy) - (R + subconn_off)) < 3.0), None)
                if tp is not None:
                    leg_taps[ri].append((e["angle_deg"], np.asarray(tp)))
    # traffic + objective nodes
    for r in nodes.itertuples():
        is_obj = str(r.kind) == "objective"
        if is_obj:
            is_db = str(r.net_id).startswith("DB")
            ax.scatter([r.x], [r.y], s=80, marker="s" if is_db else "^",
                       c="#c0392b" if is_db else "#1f6f3f", edgecolors="k",
                       linewidths=0.6, zorder=6)
            ax.annotate(str(r.net_id), (r.x, r.y), textcoords="offset points",
                        xytext=(6, 5), fontsize=8, weight="bold", zorder=7)
        else:
            inr = str(r.net_id) in member_set
            if not inr:                                   # ring replaces the junction:
                ax.scatter([r.x], [r.y], s=8, c="0.6", zorder=2)  # no centre node
            col = "#c0392b" if str(r.net_id).startswith("MAJ") else "#7f8c8d"
            ax.annotate(str(r.net_id), (r.x, r.y), textcoords="offset points",
                        xytext=(4, 3), fontsize=6.5, color=col, weight="bold", zorder=7)

    # the single circulating ring (one lane), opened only where a leg penetrates
    for ri, rb in enumerate(roundabouts):
        cx, cy = rb["cx"], rb["cy"]
        draw_ring_with_removals(ax, cx, cy, rb["R"], remove_intervals[(ri, 0)],
                                RING_COL, lw=2.4)
        # SUB-CONNECTORS: bypass shortcuts that branch off the MAIN leg routes
        # (tapped just before the ring) so an agent travelling between two nearby
        # legs skips the ring. Drawn as a mirrored two-lane COUPLE, like the legs.
        R = rb["R"]
        taps = sorted(leg_taps[ri], key=lambda t: t[0])
        for i in range(len(taps)):
            a1, p1 = taps[i]
            a2, p2 = taps[(i + 1) % len(taps)]
            gap = (a2 - a1) % 360.0
            if gap < 8.0 or gap > subconn_max:
                continue
            rm = math.radians(a1 + gap / 2)
            rad_ctrl = R + subconn_off + subconn_bulge
            ctrl = (cx + rad_ctrl * math.cos(rm), cy + rad_ctrl * math.sin(rm))
            bez = quad_bezier(p1, ctrl, p2, n=40)
            for s in (+subconn_couple, -subconn_couple):    # mirrored couple
                lane = offset_polyline(bez, s)
                ax.plot(lane[:, 0], lane[:, 1], "-", color=SUB_COL, lw=1.6,
                        alpha=0.9, zorder=3.5)
        # one-way circulation arrow on the ring (kept clear of gaps)
        r0 = rb["R"]
        ax.annotate("", xy=(cx + r0 * math.cos(math.radians(115)),
                            cy + r0 * math.sin(math.radians(115))),
                    xytext=(cx + r0 * math.cos(math.radians(80)),
                            cy + r0 * math.sin(math.radians(80))),
                    arrowprops=dict(arrowstyle="-|>", color=RING_COL, lw=1.6), zorder=8)
        ax.annotate(f"{rb['rbt_id']}  ({len(rb['entries'])} legs)",
                    (cx, cy + rb["R_out"] + 35), ha="center",
                    fontsize=8, weight="bold", color="#4a235a", zorder=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#8e44ad", alpha=0.85))

    # IN lanes (arrowhead INTO the ring) vs OUT lanes (arrowhead OUT of the ring):
    # direction is the differentiator (colour still encodes the cost lane)
    for mp, rad in in_arrows:
        ax.annotate("", xy=(mp[0] - 30 * rad[0], mp[1] - 30 * rad[1]), xytext=tuple(mp),
                    arrowprops=dict(arrowstyle="-|>", color="#222", lw=1.2), zorder=8)
    for mp, rad in out_arrows:
        ax.annotate("", xy=(mp[0] + 42 * rad[0], mp[1] + 42 * rad[1]), xytext=tuple(mp),
                    arrowprops=dict(arrowstyle="-|>", color="#222", lw=1.2), zorder=8)
    if meeting_pts:                                  # where each leg meets the ring
        mp = np.array(meeting_pts)
        ax.scatter(mp[:, 0], mp[:, 1], s=14, c="#e08214", edgecolors="k",
                   linewidths=0.3, zorder=7)

    handles = [
        Line2D([0], [0], color=RING_COL, lw=2.6, label="single circulating ring"),
        Line2D([0], [0], color=LEG_COL, lw=2.2, label="leg (corridor in/out couple)"),
        Line2D([0], [0], color=SUB_COL, lw=2.2, label="sub-connector (bypass between close legs)"),
        Line2D([0], [0], marker=r"$\rightarrow$", color="#222", markersize=11, lw=0,
               label="IN / OUT lane (arrow = merge in / diverge out)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e08214",
               markeredgecolor="k", markersize=7, label="leg meets ring"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#c0392b",
               markeredgecolor="k", markersize=9, label="DB depot"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#1f6f3f",
               markeredgecolor="k", markersize=9, label="DK delivery"),
    ] + obs
    ax.legend(handles=handles, loc="upper right", framealpha=0.9, fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"Single-ring roundabouts + bypass sub-connectors "
                 f"({len(roundabouts)} rings)")
    add_map_rule(ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=int(pget(params, "FIG_DPI", 130)))
    plt.close(fig)


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Build traffic roundabouts at high-conflict junctions.")
    p.add_argument("--param-file", default="params/roundabouts.params")
    p.add_argument("--corridor-dir", default=None)
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    params = load_params(args.param_file)
    corridor_dir = THIS_DIR / str(args.corridor_dir or
                                  pget(params, "CORRIDOR_DIR", "output/07_corridor_network"))
    out_dir = THIS_DIR / str(args.out_dir or
                             pget(params, "OUT_DIR", "output/08_roundabout_network"))
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    obstacle_file = pget(params, "OBSTACLE_MODEL_FILE",
                         "output/02_thetastar_master_plan/planning_model_with_flz_support.xyz")

    min_load = int(pget(params, "MIN_THROUGH_ROUTES", 4))
    radius = float(pget(params, "MERGE_RADIUS_M", 320.0))

    print("=" * 66)
    print(f"08_build_roundabouts.py  {VERSION}")
    print(f"Corridor dir  : {corridor_dir}")
    print(f"Output dir    : {out_dir}")
    print("=" * 66)

    nodes, legs, pair = load_network(corridor_dir)
    scores = score_nodes(nodes, legs, pair)
    nodes_xy = {r.net_id: (float(r.x), float(r.y)) for r in nodes.itertuples()}
    leg_len = {r.leg_id: float(r.length_m) for r in legs.itertuples()}
    all_tn = list(scores["node"])
    load_of = dict(zip(scores["node"], scores["through_load"]))
    blocked = load_blocked_cells(obstacle_file,
                                 bool(pget(params, "AVOID_RA", True)))

    seeds = [n for n in all_tn if load_of[n] >= min_load]
    print(f"Nodes         : {len(all_tn)} traffic nodes; {len(seeds)} seeds "
          f"(through-load >= {min_load})")

    clusters = cluster_seeds(seeds, nodes_xy, radius)
    clusters = [absorb_neighbours(c, all_tn, nodes_xy, radius) for c in clusters]
    # de-duplicate members that got pulled into two clusters (keep first)
    seen: set = set()
    uniq = []
    for c in sorted(clusters, key=lambda c: -len(c)):
        c2 = [n for n in c if n not in seen]
        if c2:
            seen.update(c2)
            uniq.append(sorted(c2))
    clusters = sorted(uniq, key=lambda c: (-max(load_of[n] for n in c),
                                           -len(c), c[0]))

    roundabouts = [build_partial(f"RBT{i:02d}", members, nodes_xy, legs, leg_len, params)
                   for i, members in enumerate(clusters, 1)]
    fit_geometry(roundabouts, blocked, params)   # clear obstacles + each other
    print(f"Obstacle clr  : {len(blocked)} blocked cells; rings shrunk to clear "
          f"obstacles + neighbours")
    for rb in roundabouts:
        nudged = math.dist((rb["cx"], rb["cy"]), (rb["cx0"], rb["cy0"]))
        print(f"  {rb['rbt_id']}: {'+'.join(rb['members']):26s} "
              f"R={rb['R']:.0f}/{rb['R_out']:.0f}m  M={rb['m']}"
              f"(want {rb['M_want']})  entries={len(rb['entries'])}  "
              f"clr={rb['d_block'] - rb['R_out']:.0f}m  nudge={nudged:.0f}m")

    # ---- write outputs ------------------------------------------------
    rb_rows, mem_rows, ent_rows, ring_rows = [], [], [], []
    for rb in roundabouts:
        rb_rows.append(dict(rbt_id=rb["rbt_id"], center_x=round(rb["cx"], 2),
                            center_y=round(rb["cy"], 2), radius_m=round(rb["R"], 2),
                            outer_radius_m=round(rb["R_out"], 2), n_lanes=rb["m"],
                            n_members=len(rb["members"]), members="+".join(rb["members"]),
                            n_entries=len(rb["entries"]),
                            n_internal_legs=len(rb["internal_legs"])))
        for n in rb["members"]:
            mem_rows.append(dict(rbt_id=rb["rbt_id"], node=n,
                                 through_load=load_of.get(n, 0),
                                 is_seed=bool(load_of.get(n, 0) >= min_load)))
        # entries sorted by cost so the lane ordering (inner->outer) is legible
        for e in sorted(rb["entries"], key=lambda e: e["cost"]):
            ent_rows.append(dict(rbt_id=rb["rbt_id"], from_node=e["from_node"],
                                 source_leg=e["leg_id"], route_cost_m=round(e["cost"], 1),
                                 angle_deg=round(e["angle_deg"], 1), lane=e["lane"],
                                 merge_x=round(e["merge_x"], 2), merge_y=round(e["merge_y"], 2)))
        for i in range(rb["m"]):
            poly = ring_polyline(rb["cx"], rb["cy"], rb["R"] + i * rb["lane_gap"])
            for seq, (x, y) in enumerate(poly):
                ring_rows.append(dict(rbt_id=rb["rbt_id"], lane=i, seq=seq,
                                      x=round(float(x), 3), y=round(float(y), 3)))

    removed_rows = [dict(rbt_id=rb["rbt_id"], removed_leg=lg)
                    for rb in roundabouts for lg in rb["internal_legs"]]
    pd.DataFrame(rb_rows).to_csv(out_dir / "roundabouts.csv", index=False)
    pd.DataFrame(mem_rows).to_csv(out_dir / "roundabout_members.csv", index=False)
    pd.DataFrame(ent_rows).to_csv(out_dir / "roundabout_entries.csv", index=False)
    pd.DataFrame(ring_rows).to_csv(out_dir / "ring_lanes.csv", index=False)
    pd.DataFrame(removed_rows or [{"rbt_id": None, "removed_leg": None}]).to_csv(
        out_dir / "removed_internal_legs.csv", index=False)
    # augmented corridor legs = original network minus the in-ring routes
    removed_set = {lg for rb in roundabouts for lg in rb["internal_legs"]}
    legs[~legs["leg_id"].isin(removed_set)].to_csv(
        out_dir / "network_legs_pruned.csv", index=False)

    summary = dict(version=VERSION, corridor_dir=str(corridor_dir),
                   min_through_routes=min_load, merge_radius_m=radius,
                   n_roundabouts=len(roundabouts),
                   n_nodes_absorbed=int(sum(len(rb["members"]) for rb in roundabouts)),
                   n_internal_legs_removed=len(removed_set),
                   total_entries=int(sum(len(rb["entries"]) for rb in roundabouts)),
                   min_obstacle_clearance_m=round(float(min(
                       rb["d_block"] - rb["R_out"] for rb in roundabouts)), 1),
                   lane_counts={rb["rbt_id"]: rb["m"] for rb in roundabouts})
    (out_dir / "roundabouts_summary.json").write_text(json.dumps(summary, indent=2))

    lane_nodes = pd.read_csv(corridor_dir / "lane_nodes.csv")
    plot_roundabouts(nodes, legs, lane_nodes, roundabouts,
                     fig_dir / "00_roundabout_network.png", obstacle_file, params)

    print("-" * 66)
    print(f"Roundabouts   : {len(roundabouts)} rings; "
          f"{summary['n_nodes_absorbed']} junctions combined; "
          f"{summary['n_internal_legs_removed']} in-ring legs removed; "
          f"min obstacle clearance {summary['min_obstacle_clearance_m']:.0f} m")
    print(f"Figure        : {fig_dir / '00_roundabout_network.png'}")
    print(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
