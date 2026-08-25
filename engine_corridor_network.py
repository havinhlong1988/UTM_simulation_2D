#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
engine_corridor_network.py

Version: v1

Build the shared UAV corridor NETWORK: infrastructure only.

  - network nodes: DB/DK objectives + major (optionally minor) traffic
    nodes (TN) from output/03_clustering_hitcount/master_plan_input_nodes.csv.
    Every node is a CIRCLE of NODE_CIRCLE_DIAMETER_M (50 m). Additional
    BACKUP TNs (BAKxx, drawn purple) are suggested for the low-density
    areas inside the network hull; their extra legs back up the network
    when a primary corridor is unavailable
  - MAXIMIZED connections: every node pair within LEG_MAX_LENGTH_M whose
    straight line is obstacle-free becomes a connection (objective-TN,
    objective-objective and TN-TN alike, e.g. DK05-DK03 directly without
    an intermediate TN). Blocked K-nearest neighbors additionally get
    one Theta* search, so any-angle turns appear only where an obstacle
    forces them
  - every connection carries 2 PARALLEL corridors of CORRIDOR_DIAMETER_M
    (50 m): the two lane centerlines are the parallel EXTERNAL TANGENTS
    of the end node circles -- symmetric +/-25 m offsets of the
    connection centerline that touch the outside of each circle. No
    perpendicular connector stubs anywhere; a vehicle flows from the
    tangent line onto the node circle and out onto the next leg.
    Where an obstacle blocks one symmetric offset side (e.g. the
    DK02-DK05 no-fly gap), the whole pair is shifted sideways off the
    centerline by the smallest clearing shift, separation preserved,
    and the end node circles MOVE WITH their shifted corridors (average
    demanded displacement per node) so the lanes stay tangent to the
    circles; the network is then rebuilt from the moved nodes
  - route drawings: for every objective pair A-B the shortest sequence
    of legs over the network is drawn as one map
    (figures/route_with_legs/A_to_B.png), showing how a path from A to
    B rides through the intermediate legs of the overall network

Planning actual UAV flights on the network is outside this work; the
pair routes here only demonstrate that every A-B is reachable through
the built legs.

Run
---
    python engine_corridor_network.py --param-file params/corridor_network.params

Main outputs
------------
    output/06_corridor_network/
        network_nodes.csv       one row per network node (objective / TN)
        network_legs.csv        one row per connection (type, length, lane report)
        lane_nodes.csv          polyline points of both lanes of every connection
        pair_routes.csv         leg sequence per objective pair (for the drawings)
        figures/00_corridor_network.png          whole-network overview
        figures/route_with_legs/<A>_to_<B>.png   route through intermediate legs
"""

from __future__ import annotations

import argparse
import heapq
import importlib.util
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.maprule import add_map_rule
from matplotlib.patches import Circle

THIS_DIR = Path(__file__).resolve().parent

VERSION = "v1"


def _load_master06():
    """Reuse 06's solver machinery (Theta* wrappers, soft-buffer lane
    planner, model loading) without duplicating it."""
    spec = importlib.util.spec_from_file_location("master06", THIS_DIR / "04b_master_corridor_thetastar.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["master06"] = module
    spec.loader.exec_module(module)
    return module


M6 = _load_master06()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the shared UAV corridor network.")
    parser.add_argument("--param-file", default="params/corridor_network.params")
    return parser.parse_args()


# ======================================================================
# Network nodes
# ======================================================================

def build_network_nodes(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Objectives (DB/DK) + TN candidates as one table with model indices."""
    obj_prefixes = [str(p) for p in M6.pget(params, "NETWORK_OBJECTIVE_PREFIXES", ["DB", "DK"])]
    include_minor = bool(M6.pget(params, "NETWORK_INCLUDE_MINOR_TN", True))

    rows = []
    is_obj = df["label_prefix"].astype(str).isin(obj_prefixes)
    for idx in np.flatnonzero(is_obj.to_numpy()):
        r = df.iloc[int(idx)]
        rows.append({
            "net_id": str(r["label"]),
            "kind": "objective",
            "label_prefix": str(r["label_prefix"]),
            "model_idx": int(idx),
            "x": float(r["x"]),
            "y": float(r["y"]),
        })

    if "is_candidate" in df.columns:
        cand_mask = df["is_candidate"].astype(bool)
        for idx in np.flatnonzero(cand_mask.to_numpy()):
            r = df.iloc[int(idx)]
            level = str(r.get("candidate_type", "major"))
            if level == "minor" and not include_minor:
                continue
            rows.append({
                "net_id": str(r.get("candidate_id", f"TN{idx}")),
                "kind": f"tn_{level}",
                "label_prefix": "TN",
                "model_idx": int(idx),
                "x": float(r["x"]),
                "y": float(r["y"]),
            })

    nodes = pd.DataFrame(rows).drop_duplicates(subset=["model_idx"]).reset_index(drop=True)
    nodes.insert(0, "net_node", np.arange(len(nodes), dtype=int))
    return nodes


# ======================================================================
# Geometry helpers
# ======================================================================

def make_nofly_tree(df: pd.DataFrame, params: dict[str, Any]):
    nofly_thr = float(M6.pget(params, "NOFLY_SLOWNESS_THRESHOLD", 10.0))
    nofly_xy = df.loc[df["slowness"].to_numpy(float) >= nofly_thr, ["x", "y"]].to_numpy(float)
    if not len(nofly_xy):
        return None
    try:
        from scipy.spatial import cKDTree
        return cKDTree(nofly_xy)
    except Exception:
        return None


def sample_min_clearance(xy: np.ndarray, nofly_tree, sample_m: float = 10.0) -> float:
    """Minimum distance of a polyline to any no-fly node."""
    if nofly_tree is None or len(xy) < 2:
        return float("inf")
    pts = M6._resample_polyline_m(xy, sample_m)
    d, _ = nofly_tree.query(pts, k=1)
    return float(np.min(d))


def straight_line_clear(a_xy: np.ndarray, b_xy: np.ndarray, nofly_tree, clearance_m: float) -> bool:
    return sample_min_clearance(np.vstack([a_xy, b_xy]), nofly_tree, sample_m=25.0) >= clearance_m


def smooth_polyline_los(xy: np.ndarray, nofly_tree, clearance_m: float) -> np.ndarray:
    """Greedy line-of-sight shortcutting: replace any staircase run of
    grid waypoints with one straight segment as long as the straight
    segment keeps clearance_m to every no-fly node. Kills the grid
    zig-zag on Theta* detour legs so the offset lanes come out straight."""
    if len(xy) <= 2:
        return xy
    out = [xy[0]]
    i = 0
    while i < len(xy) - 1:
        j = len(xy) - 1
        while j > i + 1:
            if straight_line_clear(xy[i], xy[j], nofly_tree, clearance_m):
                break
            j -= 1
        out.append(xy[j])
        i = j
    return np.asarray(out, dtype=float)


def offset_polyline(xy: np.ndarray, signed_offset_m: float) -> np.ndarray | None:
    """Parallel offset of a polyline (positive = left of travel direction)."""
    try:
        from shapely.geometry import LineString, MultiLineString
    except Exception:
        return None
    if len(xy) < 2:
        return None
    line = LineString(xy)
    off = None
    try:
        off = line.offset_curve(signed_offset_m, join_style="round")
    except Exception:
        try:
            side = "left" if signed_offset_m >= 0 else "right"
            off = line.parallel_offset(abs(signed_offset_m), side, join_style=2)
        except Exception:
            return None
    if off is None or off.is_empty:
        return None
    if isinstance(off, MultiLineString) or off.geom_type == "MultiLineString":
        off = max(off.geoms, key=lambda g: g.length)
    oxy = np.asarray(off.coords, dtype=float)
    if len(oxy) < 2:
        return None
    if np.hypot(*(oxy[0] - xy[0])) > np.hypot(*(oxy[-1] - xy[0])):
        oxy = oxy[::-1]
    return oxy


# ======================================================================
# Connections: maximized straight links, Theta* only against obstacles
# ======================================================================

def candidate_edges(nodes: pd.DataFrame, nofly_tree, params: dict[str, Any]) -> list[tuple[int, int, float, bool]]:
    """(i, j, straight_m, straight_clear) candidate connections.

    MAXIMIZED connectivity: EVERY node pair within LEG_MAX_LENGTH_M with
    an obstacle-free straight line is a connection -- direct links like
    DK05-DK03 or MAJ001-DK01 are never dropped just because the nodes
    are not each other's nearest neighbors. Blocked pairs are only kept
    for the LEG_KNN_K nearest neighbors (those get a Theta* detour), so
    the detour count stays bounded.
    """
    k = int(M6.pget(params, "LEG_KNN_K", 5))
    max_len = float(M6.pget(params, "LEG_MAX_LENGTH_M", 2500.0))
    clearance_m = float(M6.pget(params, "LEG_CLEARANCE_M", 50.0))
    xy = nodes[["x", "y"]].to_numpy(float)
    n = len(xy)

    d_all = np.hypot(xy[:, 0:1] - xy[None, :, 0], xy[:, 1:2] - xy[None, :, 1])

    knn: set[tuple[int, int]] = set()
    for i in range(n):
        order = np.argsort(d_all[i])
        picked = 0
        for j in order:
            j = int(j)
            if j == i:
                continue
            if max_len > 0 and d_all[i, j] > max_len:
                break
            knn.add((min(i, j), max(i, j)))
            picked += 1
            if picked >= k:
                break

    out: list[tuple[int, int, float, bool]] = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(d_all[i, j])
            if max_len > 0 and dist > max_len:
                continue
            clear = straight_line_clear(xy[i], xy[j], nofly_tree, clearance_m)
            if clear or (i, j) in knn:
                out.append((i, j, dist, clear))
    return out


def build_legs(
    df: pd.DataFrame,
    cell_to_idx: dict,
    base_allowed: np.ndarray,
    nodes: pd.DataFrame,
    nofly_tree,
    params: dict[str, Any],
) -> pd.DataFrame:
    """Materialize the connections.

    Straight-clear connections become 2-waypoint straight legs (zero
    turns). Blocked K-nearest ones get one Theta* search -- the
    any-angle turns appear only where the obstacle forces them.
    Detours longer than LEG_MAX_DETOUR_RATIO x straight are dropped.
    """
    detour_cap = float(M6.pget(params, "LEG_MAX_DETOUR_RATIO", 2.0))

    rows = []
    edges = candidate_edges(nodes, nofly_tree, params)
    n_clear = sum(1 for e in edges if e[3])
    print(f"Connections     : {len(edges)} candidate ({n_clear} straight-clear, {len(edges) - n_clear} need Theta*)")
    for i, j, straight_m, clear in edges:
        a = nodes.iloc[i]
        b = nodes.iloc[j]
        a_idx = int(a["model_idx"])
        b_idx = int(b["model_idx"])
        leg_id = f"{a['net_id']}__{b['net_id']}"

        if clear:
            rows.append({
                "leg_id": leg_id,
                "net_a": int(i), "net_b": int(j),
                "a_id": str(a["net_id"]), "b_id": str(b["net_id"]),
                "leg_type": "straight",
                "length_m": straight_m,
                "straight_m": straight_m,
                "path_indices": [a_idx, b_idx],
                "path_xy": np.array([[float(a["x"]), float(a["y"])], [float(b["x"]), float(b["y"])]]),
            })
            continue

        result = M6.run_project_theta(
            base_model=df, cell_to_idx=cell_to_idx, base_allowed_mask=base_allowed,
            start_idx=a_idx, end_idx=b_idx, params=params, route_name=f"leg_{leg_id}",
        )
        if not result.get("success", False):
            print(f"  drop {leg_id}: Theta* failed ({result.get('message','')})")
            continue
        path = [int(v) for v in result.get("path_indices", [])]
        # Local optimization: shortcut the raw grid staircase so the leg
        # centerline (and its offset lanes) run straight between the
        # genuinely forced turns.
        xy_path = np.array(M6._route_xy(df, path), dtype=float, copy=True)
        # Theta* starts/ends on the grid cell of model_idx; a node moved
        # off its grid cell (node shift pass) still anchors the leg.
        xy_path[0] = [float(a["x"]), float(a["y"])]
        xy_path[-1] = [float(b["x"]), float(b["y"])]
        smooth_clear = float(M6.pget(params, "LEG_SMOOTH_CLEARANCE_M", clearance_m := float(M6.pget(params, "LEG_CLEARANCE_M", 50.0))))
        if bool(M6.pget(params, "LEG_SMOOTH_ENABLE", True)):
            xy_path = smooth_polyline_los(xy_path, nofly_tree, smooth_clear)
        length_m = float(np.hypot(np.diff(xy_path[:, 0]), np.diff(xy_path[:, 1])).sum())
        if detour_cap > 0 and length_m > detour_cap * straight_m:
            print(f"  drop {leg_id}: detour {length_m:.0f} m > {detour_cap:.1f} x straight {straight_m:.0f} m")
            continue
        rows.append({
            "leg_id": leg_id,
            "net_a": int(i), "net_b": int(j),
            "a_id": str(a["net_id"]), "b_id": str(b["net_id"]),
            "leg_type": "theta",
            "length_m": float(length_m),
            "straight_m": straight_m,
            "path_indices": path,
            "path_xy": xy_path,
        })

    legs = pd.DataFrame(rows)
    if len(legs):
        n_straight = int((legs["leg_type"] == "straight").sum())
        print(f"Built           : {len(legs)} connections ({n_straight} straight, {len(legs) - n_straight} theta)")
    return legs


# ======================================================================
# Crossing filter: corridors may only meet at network nodes
# ======================================================================
#
# The maximized all-pairs connections produce direct lines that CROSS
# other connections in mid-air, away from any node -- an unmanaged
# corridor intersection. Two rules restore network hygiene:
#
#   1. a connection whose centerline crosses (or collinearly overlaps)
#      an already-kept connection is skipped;
#   2. a connection that flies straight over the circle of an
#      intermediate network node WITHOUT stopping there is skipped --
#      the traffic belongs on the two shorter legs via that node.
#
# Shorter connections are kept first, so crossings always resolve in
# favor of the local legs and the dropped ones are the long redundant
# diagonals. Crossings that happen AT a shared node (touching
# endpoints) are of course allowed -- that is what the nodes are for.

def filter_crossing_legs(legs: pd.DataFrame, df: pd.DataFrame, nodes: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    if not bool(M6.pget(params, "LEG_SKIP_CROSSING", True)) or not len(legs):
        return legs
    try:
        from shapely.geometry import LineString, Point
    except Exception:
        print("[WARN] shapely unavailable -- crossing filter skipped")
        return legs

    node_r = 0.5 * float(M6.pget(params, "NODE_CIRCLE_DIAMETER_M", 50.0))

    geoms: dict[int, Any] = {}
    for idx, leg in legs.iterrows():
        xy = np.asarray(leg["path_xy"], dtype=float) if "path_xy" in leg.index else M6._route_xy(df, [int(v) for v in leg["path_indices"]])
        geoms[int(idx)] = LineString(xy)

    n_flyover = 0
    n_crossing = 0
    kept_idx: list[int] = []
    kept_geoms: list[Any] = []
    for idx in legs.sort_values("length_m").index:
        leg = legs.loc[idx]
        geom = geoms[int(idx)]

        # Rule 2: no flying over an intermediate node's circle.
        endpoint_ids = {int(leg["net_a"]), int(leg["net_b"])}
        flyover = False
        for _, nd in nodes.iterrows():
            if int(nd["net_node"]) in endpoint_ids:
                continue
            if geom.distance(Point(float(nd["x"]), float(nd["y"]))) < node_r:
                print(f"  skip {leg['leg_id']}: flies over intermediate node {nd['net_id']}")
                flyover = True
                break
        if flyover:
            n_flyover += 1
            continue

        # Rule 1: no crossing / collinear overlap with kept connections.
        conflict = None
        for kept_i, kg in zip(kept_idx, kept_geoms):
            if geom.crosses(kg) or geom.overlaps(kg):
                conflict = str(legs.loc[kept_i, "leg_id"])
                break
        if conflict is not None:
            print(f"  skip {leg['leg_id']}: crosses {conflict}")
            n_crossing += 1
            continue

        kept_idx.append(int(idx))
        kept_geoms.append(geom)

    print(f"Crossing filter : kept {len(kept_idx)}, skipped {n_crossing} crossing + {n_flyover} node-flyover connections")
    return legs.loc[sorted(kept_idx)].reset_index(drop=True)


# ======================================================================
# Two parallel tangent lanes per connection
# ======================================================================
#
# Every node is a circle of NODE_CIRCLE_DIAMETER_M (50 m). The two lane
# centerlines of a leg are the parallel EXTERNAL TANGENTS of the end
# circles: symmetric +/- (diameter/2 = 25 m) offsets of the connection
# centerline. An offset at exactly the circle radius touches the circle
# tangentially at the perpendicular foot -- so the lanes flow onto the
# node circles with NO perpendicular connector stubs, and the two
# centerlines sit 50 m apart along the whole leg.

def tangent_lane_pair(xy_c: np.ndarray, half_sep_m: float, half_w: float, nofly_tree, shift_m: float = 0.0) -> tuple[np.ndarray | None, np.ndarray | None]:
    xy_a = offset_polyline(xy_c, +half_sep_m + shift_m)
    xy_b = offset_polyline(xy_c, -half_sep_m + shift_m)
    if xy_a is None or xy_b is None:
        return None, None
    if sample_min_clearance(xy_a, nofly_tree) < half_w or sample_min_clearance(xy_b, nofly_tree) < half_w:
        return None, None
    return xy_a, xy_b


def shifted_lane_pair(
    xy_c: np.ndarray, half_sep_m: float, half_w: float, nofly_tree, params: dict[str, Any],
) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    """Slide the whole lane pair sideways when the symmetric tangent
    pair is blocked (e.g. DK02-DK05: the centerline threads a narrow
    no-fly gap off-center, so one +/- offset side has no room). Both
    lanes keep the full 2*half_sep_m separation; the pair is just
    shifted off the centerline by the smallest shift that clears the
    obstacles on both sides."""
    max_shift = float(M6.pget(params, "LANE_SHIFT_MAX_M", 100.0))
    step = float(M6.pget(params, "LANE_SHIFT_STEP_M", 5.0))
    n = int(max_shift / step)
    for k in range(1, n + 1):
        for s in (k * step, -k * step):
            xy_a, xy_b = tangent_lane_pair(xy_c, half_sep_m, half_w, nofly_tree, shift_m=s)
            if xy_a is not None:
                return xy_a, xy_b, s
    return None, None, float("nan")


def compute_node_shifts(
    nodes: pd.DataFrame,
    legs: pd.DataFrame,
    lane_report: pd.DataFrame,
    nofly_tree,
    params: dict[str, Any],
) -> pd.DataFrame | None:
    """Move node circles along with their shifted corridors.

    A shifted lane pair is no longer tangent to the end node circles --
    one lane floats off the circle, the other cuts through it. Each
    shifted leg therefore demands a lateral displacement of
    lane_shift_m * left-normal at both of its end nodes; a node moves
    by the AVERAGE demand of its shifted legs (unshifted legs keep
    their tangency through the rebuild that follows). Moves are capped
    at NODE_SHIFT_MAX_M and skipped when the moved circle would touch a
    no-fly node. Returns the moved nodes table, or None if nothing
    moved."""
    max_shift = float(M6.pget(params, "NODE_SHIFT_MAX_M", 50.0))
    node_r = 0.5 * float(M6.pget(params, "NODE_CIRCLE_DIAMETER_M", 50.0))

    shift_by_leg: dict[str, float] = {}
    for r in lane_report.itertuples():
        s = float(getattr(r, "lane_shift_m", 0.0))
        if str(getattr(r, "lane_b_method", "")) == "shifted_pair" and math.isfinite(s) and s != 0.0:
            shift_by_leg[str(r.leg_id)] = s
    if not shift_by_leg:
        return None

    demands: dict[int, list[np.ndarray]] = {}
    for _, leg in legs.iterrows():
        s = shift_by_leg.get(str(leg["leg_id"]))
        if s is None:
            continue
        xy = np.asarray(leg["path_xy"], dtype=float)
        for node_i, t in ((int(leg["net_a"]), xy[1] - xy[0]), (int(leg["net_b"]), xy[-1] - xy[-2])):
            norm_t = float(np.hypot(*t))
            if norm_t < 1.0e-9:
                continue
            left_n = np.array([-t[1], t[0]]) / norm_t
            demands.setdefault(node_i, []).append(s * left_n)

    out = nodes.copy()
    n_moved = 0
    for node_i, vecs in demands.items():
        v = np.mean(vecs, axis=0)
        norm = float(np.hypot(*v))
        if norm < 1.0:
            continue
        if norm > max_shift:
            v *= max_shift / norm
        row = out.index[out["net_node"] == node_i][0]
        new_c = np.array([float(out.at[row, "x"]), float(out.at[row, "y"])]) + v
        if nofly_tree is not None:
            d, _ = nofly_tree.query(new_c, k=1)
            if float(d) < node_r:
                print(f"  keep {out.at[row, 'net_id']}: moved circle would touch a no-fly node")
                continue
        out.at[row, "x"] = float(new_c[0])
        out.at[row, "y"] = float(new_c[1])
        out.at[row, "node_shift_m"] = float(np.hypot(
            float(new_c[0]) - float(out.at[row, "x_orig"]),
            float(new_c[1]) - float(out.at[row, "y_orig"]),
        ))
        print(f"  move {out.at[row, 'net_id']}: node circle follows its shifted corridors by {float(np.hypot(*v)):.1f} m")
        n_moved += 1
    return out if n_moved else None


def measure_lane_separation(xy_a: np.ndarray, xy_b: np.ndarray, params: dict[str, Any]) -> dict[str, Any]:
    """Clearance profile between the two lane centerlines."""
    step_m = float(M6.pget(params, "BUFFER_SAMPLE_STEP_M", 10.0))
    min_sep = float(M6.pget(params, "MIN_CENTERLINE_SEPARATION_M", 50.0))

    ref = M6._resample_polyline_m(xy_a, step_m)
    _, dist, side = M6.compute_clearance_profile(ref, xy_b)
    crossings = M6.detect_crossing_points(ref, dist, side)
    return {
        "centerline_min_separation_m": float(dist.min()) if len(dist) else float("nan"),
        "separation_ok": bool(not (dist < min_sep - 1.0).any() and len(crossings) == 0),
        "n_narrow_samples": int((dist < min_sep - 1.0).sum()),
        "n_crossings": int(len(crossings)),
    }


def build_leg_lanes(
    df: pd.DataFrame,
    cell_to_idx: dict,
    base_allowed: np.ndarray,
    legs: pd.DataFrame,
    nofly_tree,
    params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two parallel tangent lanes per connection.

    Fallbacks when one symmetric offset side is blocked by an obstacle:
    first the whole pair is shifted sideways off the centerline
    (separation preserved) by the smallest clearing shift; if no shift
    within LANE_SHIFT_MAX_M works, lane A stays on the centerline and
    lane B comes from the dual-parallel soft-buffer search (06)."""
    min_sep = float(M6.pget(params, "MIN_CENTERLINE_SEPARATION_M", 50.0))
    half_sep = 0.5 * min_sep
    half_w = 0.5 * float(M6.pget(params, "CORRIDOR_DIAMETER_M", 50.0))
    params = dict(params)
    params["BUFFER_TARGET_M"] = min_sep

    lane_rows = []
    report_rows = []
    for _, leg in legs.iterrows():
        path1 = [int(v) for v in leg["path_indices"]]
        xy_c = np.asarray(leg["path_xy"], dtype=float) if "path_xy" in leg.index else M6._route_xy(df, path1)

        xy_a, xy_b = tangent_lane_pair(xy_c, half_sep, half_w, nofly_tree)
        method = "tangent_pair" if xy_a is not None else ""
        shift_m = 0.0

        if xy_a is None:
            # One tangent side blocked: shift the whole pair sideways.
            xy_a, xy_b, shift_m = shifted_lane_pair(xy_c, half_sep, half_w, nofly_tree, params)
            if xy_a is not None:
                method = "shifted_pair"
                print(f"  shift {leg['leg_id']}: lane pair moved {shift_m:+.0f} m off the centerline")

        if xy_a is None:
            # No clearing shift either: centerline + soft-buffer parallel.
            shift_m = float("nan")
            xy_a = xy_c
            result = M6.make_parallel_route_soft_buffer(
                df=df, cell_to_idx=cell_to_idx, base_allowed=base_allowed,
                start_idx=path1[0], end_idx=path1[-1], main_path=path1,
                params=params, route_name=f"laneB_{leg['leg_id']}",
            )
            ok = bool(result.get("success", False)) and not bool(result.get("duplicated_from_main", False))
            xy_b = M6._route_xy(df, [int(v) for v in result.get("path_indices", [])]) if ok else None
            method = "centerline+soft_buffer" if ok else "single_lane"

        for seq, (x, y) in enumerate(xy_a):
            lane_rows.append({"leg_id": leg["leg_id"], "lane": "A", "seq": seq, "x": float(x), "y": float(y)})
        if xy_b is not None:
            for seq, (x, y) in enumerate(xy_b):
                lane_rows.append({"leg_id": leg["leg_id"], "lane": "B", "seq": seq, "x": float(x), "y": float(y)})
            sep_stats = measure_lane_separation(xy_a, xy_b, params)
            lane_b_len = float(np.hypot(np.diff(xy_b[:, 0]), np.diff(xy_b[:, 1])).sum())
        else:
            sep_stats = {
                "centerline_min_separation_m": float("nan"),
                "separation_ok": False, "n_narrow_samples": np.nan, "n_crossings": np.nan,
            }
            lane_b_len = float("nan")

        report_rows.append({
            "leg_id": leg["leg_id"],
            "two_lanes": xy_b is not None,
            "lane_b_method": method,
            "lane_shift_m": shift_m,
            "lane_b_length_m": lane_b_len,
            **sep_stats,
        })
    return pd.DataFrame(lane_rows), pd.DataFrame(report_rows)


# ======================================================================
# Backup TN suggestion: purple TNs for the low-density areas
# ======================================================================

def suggest_backup_tn(
    df: pd.DataFrame,
    nodes: pd.DataFrame,
    legs: pd.DataFrame,
    nofly_tree,
    params: dict[str, Any],
) -> pd.DataFrame:
    """Suggest ADDITIONAL TN nodes (net_id BAKxx, drawn PURPLE) in the
    low-density areas of the built network: flyable cells inside the
    network's convex hull that are far from every node and corridor
    centerline, with enough no-fly clearance for a node circle and its
    corridors. Connected like any other node on the rebuild, they open
    extra legs that back up the network when a primary corridor is
    unavailable."""
    n_max = int(M6.pget(params, "BACKUP_TN_MAX", 6))
    min_net = float(M6.pget(params, "BACKUP_TN_MIN_NETWORK_DIST_M", 600.0))
    min_clear = float(M6.pget(params, "BACKUP_TN_CLEARANCE_M", 100.0))
    spacing = float(M6.pget(params, "BACKUP_TN_SPACING_M", 800.0))
    nofly_thr = float(M6.pget(params, "NOFLY_SLOWNESS_THRESHOLD", 10.0))
    if n_max <= 0:
        return nodes

    try:
        from scipy.spatial import cKDTree
    except Exception:
        print("[WARN] scipy unavailable -- backup TN suggestion skipped")
        return nodes

    net_pts = [nodes[["x", "y"]].to_numpy(float)]
    for _, leg in legs.iterrows():
        net_pts.append(M6._resample_polyline_m(np.asarray(leg["path_xy"], dtype=float), 50.0))
    net_tree = cKDTree(np.vstack(net_pts))

    fly_pos = np.flatnonzero(df["slowness"].to_numpy(float) < nofly_thr)
    xy = df.iloc[fly_pos][["x", "y"]].to_numpy(float)
    d_net, _ = net_tree.query(xy, k=1)
    d_nofly = nofly_tree.query(xy, k=1)[0] if nofly_tree is not None else np.full(len(xy), np.inf)

    # Only INSIDE the network's hull: the goal is redundancy between the
    # existing corridors, not expansion beyond them.
    try:
        from shapely.geometry import MultiPoint, Point
        hull = MultiPoint(nodes[["x", "y"]].to_numpy(float).tolist()).convex_hull
        inside = np.array([hull.contains(Point(float(px), float(py))) for px, py in xy])
    except Exception:
        nx0, nx1 = float(nodes["x"].min()), float(nodes["x"].max())
        ny0, ny1 = float(nodes["y"].min()), float(nodes["y"].max())
        inside = (xy[:, 0] >= nx0) & (xy[:, 0] <= nx1) & (xy[:, 1] >= ny0) & (xy[:, 1] <= ny1)

    ok = (d_net >= min_net) & (d_nofly >= min_clear) & inside
    rows: list[dict[str, Any]] = []
    for k in np.argsort(-d_net):
        if not ok[k]:
            continue
        p = xy[k]
        if any(float(np.hypot(p[0] - r["x"], p[1] - r["y"])) < spacing for r in rows):
            continue
        rows.append({
            "net_id": f"BAK{len(rows) + 1:02d}",
            "kind": "tn_backup",
            "label_prefix": "TN",
            "model_idx": int(fly_pos[k]),
            "x": float(p[0]),
            "y": float(p[1]),
            "x_orig": float(p[0]),
            "y_orig": float(p[1]),
            "node_shift_m": 0.0,
            "gap_to_network_m": float(d_net[k]),
        })
        if len(rows) >= n_max:
            break
    if not rows:
        return nodes
    for r in rows:
        print(f"  suggest {r['net_id']}: ({r['x']:.0f}, {r['y']:.0f}), {r['gap_to_network_m']:.0f} m from the nearest node/corridor")
    out = pd.concat([nodes, pd.DataFrame(rows)], ignore_index=True)
    out["net_node"] = np.arange(len(out), dtype=int)
    return out


def suggest_expansion_tn(
    df: pd.DataFrame,
    nodes: pd.DataFrame,
    legs: pd.DataFrame,
    nofly_tree,
    params: dict[str, Any],
) -> pd.DataFrame:
    """Open up the unused flyable space OUTSIDE the network hull.

    Farthest-point sampling: repeatedly take the flyable cell farthest
    from the growing network (nodes, corridor centerlines and already
    suggested TNs) that still has EXPANSION_TN_CLEARANCE_M to the
    no-fly fields and lies within EXPANSION_TN_MAX_LINK_M of a node, so
    a leg can actually reach it. Sampling stops when every outside
    pocket is within EXPANSION_TN_MIN_DIST_M of the network. The picks
    (net_id EXTxx, drawn MAGENTA) join the rebuild like any other node
    and pull corridors into the previously unused space; chains of EXT
    TNs can walk far out because each pick only needs to link to the
    network grown so far."""
    n_max = int(M6.pget(params, "EXPANSION_TN_MAX", 8))
    min_dist = float(M6.pget(params, "EXPANSION_TN_MIN_DIST_M", 700.0))
    min_clear = float(M6.pget(params, "EXPANSION_TN_CLEARANCE_M", 100.0))
    max_link = float(M6.pget(params, "EXPANSION_TN_MAX_LINK_M", float(M6.pget(params, "LEG_MAX_LENGTH_M", 2500.0))))
    nofly_thr = float(M6.pget(params, "NOFLY_SLOWNESS_THRESHOLD", 10.0))
    if n_max <= 0:
        return nodes

    try:
        from scipy.spatial import cKDTree
    except Exception:
        print("[WARN] scipy unavailable -- expansion TN suggestion skipped")
        return nodes

    net_pts = [nodes[["x", "y"]].to_numpy(float)]
    for _, leg in legs.iterrows():
        net_pts.append(M6._resample_polyline_m(np.asarray(leg["path_xy"], dtype=float), 50.0))
    net_xy = np.vstack(net_pts)
    link_xy = nodes[["x", "y"]].to_numpy(float)

    fly_pos = np.flatnonzero(df["slowness"].to_numpy(float) < nofly_thr)
    xy = df.iloc[fly_pos][["x", "y"]].to_numpy(float)
    d_nofly = nofly_tree.query(xy, k=1)[0] if nofly_tree is not None else np.full(len(xy), np.inf)

    try:
        from shapely.geometry import MultiPoint, Point
        hull = MultiPoint(nodes[["x", "y"]].to_numpy(float).tolist()).convex_hull
        outside = ~np.array([hull.contains(Point(float(px), float(py))) for px, py in xy])
    except Exception:
        print("[WARN] shapely unavailable -- expansion TN suggestion skipped")
        return nodes

    base_ok = (d_nofly >= min_clear) & outside
    rows: list[dict[str, Any]] = []
    while len(rows) < n_max:
        d_net, _ = cKDTree(net_xy).query(xy, k=1)
        d_link, _ = cKDTree(link_xy).query(xy, k=1)
        ok = base_ok & (d_net >= min_dist) & (d_link <= max_link)
        if not ok.any():
            break
        k = int(np.argmax(np.where(ok, d_net, -1.0)))
        p = xy[k]
        rows.append({
            "net_id": f"EXT{len(rows) + 1:02d}",
            "kind": "tn_ext",
            "label_prefix": "TN",
            "model_idx": int(fly_pos[k]),
            "x": float(p[0]),
            "y": float(p[1]),
            "x_orig": float(p[0]),
            "y_orig": float(p[1]),
            "node_shift_m": 0.0,
            "gap_to_network_m": float(d_net[k]),
        })
        net_xy = np.vstack([net_xy, p[None, :]])
        link_xy = np.vstack([link_xy, p[None, :]])
    if not rows:
        return nodes
    for r in rows:
        print(f"  suggest {r['net_id']}: ({r['x']:.0f}, {r['y']:.0f}), opens space {r['gap_to_network_m']:.0f} m outside the network")
    out = pd.concat([nodes, pd.DataFrame(rows)], ignore_index=True)
    out["net_node"] = np.arange(len(out), dtype=int)
    return out


# ======================================================================
# Pair routes over the network (for the route_with_legs drawings)
# ======================================================================

def dijkstra(adjacency: dict[int, list[tuple[int, float, int]]], src: int, dst: int) -> tuple[list[int], list[int]]:
    dist = {src: 0.0}
    prev: dict[int, tuple[int, int]] = {}
    pq = [(0.0, src)]
    seen: set[int] = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in seen:
            continue
        seen.add(u)
        if u == dst:
            break
        for v, w, leg_row in adjacency.get(u, []):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = (u, leg_row)
                heapq.heappush(pq, (nd, v))
    if dst not in seen:
        return [], []
    node_seq = [dst]
    leg_seq: list[int] = []
    u = dst
    while u != src:
        p, leg_row = prev[u]
        leg_seq.append(leg_row)
        node_seq.append(p)
        u = p
    return node_seq[::-1], leg_seq[::-1]


def assign_pair_routes(nodes: pd.DataFrame, legs: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Shortest leg sequence for every objective pair (drawing only --
    it demonstrates A-to-B reachability through the intermediate legs)."""
    adjacency: dict[int, list[tuple[int, float, int]]] = {}
    for row_i, leg in legs.iterrows():
        a, b, w = int(leg["net_a"]), int(leg["net_b"]), float(leg["length_m"])
        adjacency.setdefault(a, []).append((b, w, int(row_i)))
        adjacency.setdefault(b, []).append((a, w, int(row_i)))

    objectives = nodes[nodes["kind"] == "objective"].reset_index(drop=True)
    skip_same_prefix = bool(M6.pget(params, "SKIP_SAME_PREFIX", False))

    rows = []
    for i in range(len(objectives)):
        for j in range(i + 1, len(objectives)):
            a = objectives.iloc[i]
            b = objectives.iloc[j]
            if skip_same_prefix and str(a["label_prefix"]) == str(b["label_prefix"]):
                continue
            pair_name = f"{a['net_id']}_to_{b['net_id']}"
            node_seq, leg_seq = dijkstra(adjacency, int(a["net_node"]), int(b["net_node"]))
            if not node_seq:
                rows.append({"pair": pair_name, "success": False, "n_legs": 0, "length_m": np.nan, "via": "", "leg_rows": [], "leg_forward": []})
                continue
            # Traversal direction per leg: lane sides are stored relative
            # to the leg's net_a -> net_b orientation, so a leg entered at
            # net_b is ridden in reverse and its left/right lanes swap.
            leg_forward = [bool(int(legs.iloc[k]["net_a"]) == int(node_seq[i])) for i, k in enumerate(leg_seq)]
            rows.append({
                "pair": pair_name,
                "success": True,
                "n_legs": len(leg_seq),
                "length_m": float(sum(float(legs.iloc[k]["length_m"]) for k in leg_seq)),
                "via": "-".join(str(nodes.iloc[n]["net_id"]) for n in node_seq),
                "leg_rows": leg_seq,
                "leg_forward": leg_forward,
            })
    return pd.DataFrame(rows)


# ======================================================================
# Plotting
# ======================================================================

def _corridor_patches(ax, xy: np.ndarray, half_width_m: float, color: str) -> None:
    """Filled transparent corridor + gray boundary around one centerline."""
    if xy is None or len(xy) < 2:
        return
    try:
        from shapely.geometry import LineString
        poly = LineString(xy).buffer(half_width_m, cap_style="round", join_style="round")
        geoms = getattr(poly, "geoms", [poly])
        for g in geoms:
            bx, by = g.exterior.xy
            ax.fill(bx, by, color=color, alpha=0.18, linewidth=0, zorder=8)
            ax.plot(bx, by, color="gray", linewidth=0.9, alpha=0.9, zorder=9)
    except Exception:
        ax.plot(xy[:, 0], xy[:, 1], color=color, alpha=0.18, linewidth=8.0, zorder=8)


def _node_circle_patches(ax, nodes: pd.DataFrame, radius_m: float, emphasize: set[str] | None = None) -> None:
    """Node circles (Ø 2*radius_m): the lanes are tangent to these."""
    for _, r in nodes.iterrows():
        strong = emphasize is not None and str(r["net_id"]) in emphasize
        ax.add_patch(Circle(
            (float(r["x"]), float(r["y"])), radius_m,
            facecolor="purple", alpha=0.22 if strong else 0.10,
            edgecolor="black" if strong else "gray",
            linewidth=1.2 if strong else 0.7, zorder=13,
        ))


def _lane_xy(lanes: pd.DataFrame, leg_id: str, lane: str) -> np.ndarray:
    g = lanes[(lanes["leg_id"] == leg_id) & (lanes["lane"] == lane)].sort_values("seq")
    return g[["x", "y"]].to_numpy(float)


def _plot_nodes(ax, nodes: pd.DataFrame, labels: bool = True, fontsize: float = 7.0) -> None:
    for kind, marker, color, size, label in [
        ("objective", None, None, 110, None),
        ("tn_major", "D", "teal", 70, "TN major"),
        ("tn_minor", "d", "turquoise", 40, "TN minor"),
        ("tn_backup", "D", "purple", 90, "TN backup (suggested)"),
        ("tn_ext", "D", "magenta", 90, "TN extension (suggested)"),
    ]:
        sub = nodes[nodes["kind"] == kind]
        if not len(sub):
            continue
        if kind == "objective":
            db = sub[sub["label_prefix"] == "DB"]
            dk = sub[sub["label_prefix"] == "DK"]
            ax.scatter(db["x"], db["y"], marker="^", s=size, c="blue", edgecolors="white", linewidths=0.8, label="DB", zorder=20)
            ax.scatter(dk["x"], dk["y"], marker="s", s=size, c="green", edgecolors="white", linewidths=0.8, label="DK", zorder=20)
        else:
            ax.scatter(sub["x"], sub["y"], marker=marker, s=size, c=color, edgecolors="white", linewidths=0.6, label=label, zorder=19)
        if labels:
            for _, r in sub.iterrows():
                ax.text(r["x"], r["y"], str(r["net_id"]), fontsize=fontsize, weight="bold", zorder=21)


def _resample_n(xy: np.ndarray, n: int = 60) -> np.ndarray:
    seg = np.hypot(*np.diff(xy, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0.0, float(s[-1]), n)
    return np.column_stack([np.interp(t, s, xy[:, 0]), np.interp(t, s, xy[:, 1])])


def travel_left_right_lanes(lanes: pd.DataFrame, leg_id: str, forward: bool) -> tuple[np.ndarray, np.ndarray]:
    """(left_lane, right_lane) of the TRAVEL direction, decided
    GEOMETRICALLY -- not by the stored A/B label. Tangent-pair legs have
    lane A on the left of their stored orientation, but soft-buffer
    fallback legs keep lane A on the centerline with lane B on an
    arbitrary side, so the label alone mis-colors those legs (and the
    lane would appear to jump sides mid-route). The sign of the cross
    product at mid-leg is authoritative for every lane construction."""
    xy_a = _lane_xy(lanes, leg_id, "A")
    xy_b = _lane_xy(lanes, leg_id, "B")
    if not forward:
        xy_a = xy_a[::-1] if len(xy_a) else xy_a
        xy_b = xy_b[::-1] if len(xy_b) else xy_b
    if len(xy_a) < 2 or len(xy_b) < 2:
        return xy_a, xy_b
    ra = _resample_n(xy_a)
    rb = _resample_n(xy_b)
    center = 0.5 * (ra + rb)
    m = len(center) // 2
    t = center[min(m + 1, len(center) - 1)] - center[max(m - 1, 0)]
    side_a = float(t[0] * (ra[m] - center[m])[1] - t[1] * (ra[m] - center[m])[0])
    return (xy_a, xy_b) if side_a >= 0.0 else (xy_b, xy_a)


def _arc_between(center: np.ndarray, p1: np.ndarray, p2: np.ndarray, step_deg: float = 6.0) -> np.ndarray:
    """Arc around `center` from p1 to p2 (shorter angular direction).

    Radii may differ slightly (e.g. a soft-buffer lane ending at the node
    center meets a tangent lane at 25 m): the radius is blended linearly
    along the arc, giving a short spiral instead of a jump."""
    v1 = p1 - center
    v2 = p2 - center
    r1 = float(np.hypot(*v1))
    r2 = float(np.hypot(*v2))
    if r1 < 1.0e-6 or r2 < 1.0e-6:
        return np.empty((0, 2), dtype=float)
    a1 = math.atan2(v1[1], v1[0])
    a2 = math.atan2(v2[1], v2[0])
    da = (a2 - a1 + math.pi) % (2.0 * math.pi) - math.pi
    n = max(int(abs(da) / math.radians(step_deg)) + 2, 2)
    ang = a1 + np.linspace(0.0, da, n)
    rad = np.linspace(r1, r2, n)
    return np.column_stack([center[0] + rad * np.cos(ang), center[1] + rad * np.sin(ang)])


def _miter_point(p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray, max_ext_m: float = 60.0) -> np.ndarray | None:
    """Intersection of segment p0-p1 with q0-q1, allowing a short
    extension (p beyond p1 / q before q0, up to max_ext_m) so slightly
    gapped inner corners still miter."""
    r = p1 - p0
    s = q1 - q0
    denom = float(r[0] * s[1] - r[1] * s[0])
    lr = float(np.hypot(*r))
    ls = float(np.hypot(*s))
    if abs(denom) < 1.0e-9 or lr < 1.0e-9 or ls < 1.0e-9:
        return None
    qp = q0 - p0
    t = float(qp[0] * s[1] - qp[1] * s[0]) / denom
    u = float(qp[0] * r[1] - qp[1] * r[0]) / denom
    if 0.0 <= t <= 1.0 + max_ext_m / lr and -max_ext_m / ls <= u <= 1.0:
        return p0 + t * r
    return None


def _join_lane_chains(cur: np.ndarray, nxt: np.ndarray, node_c: np.ndarray) -> np.ndarray:
    """Join two consecutive same-side lanes at a node.

    INNER side of a turn: the incoming and outgoing lane lines cross
    near the node -- trimming both to their intersection (miter/corner
    cut) is the continuous path; arcing between the tangent feet there
    would loop backwards through the node circle. OUTER side (and
    straight-through): the lines do not cross, so an arc fillet around
    the node circle bridges them."""
    if len(cur) >= 2 and len(nxt) >= 2:
        pt = _miter_point(cur[-2], cur[-1], nxt[0], nxt[1])
        if pt is not None:
            return np.vstack([cur[:-1], pt[None, :], nxt[1:]])
    arc = _arc_between(node_c, cur[-1], nxt[0])
    return np.vstack([cur, arc, nxt])


def build_route_lane_chains(
    lanes: pd.DataFrame,
    leg_ids: list[str],
    leg_forward: list[bool],
    via_xy: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """One CONTINUOUS polyline per travel side for a whole route.

    Consecutive legs' same-side lanes are joined at the intermediate
    node (via_xy[i] = the node between leg i-1 and leg i): the inner
    lane of a turn cuts the corner at the lane-line intersection, the
    outer lane wraps the node circle with an arc fillet. The vehicle
    never visits the TN center and never doubles back -- it just keeps
    moving along its side. A missing lane (single-lane leg) breaks that
    side's chain into separate pieces."""
    left_parts: list[np.ndarray] = []
    right_parts: list[np.ndarray] = []
    left_cur: np.ndarray | None = None
    right_cur: np.ndarray | None = None

    for i, (leg_id, fwd) in enumerate(zip(leg_ids, leg_forward)):
        left_xy, right_xy = travel_left_right_lanes(lanes, leg_id, fwd)

        if left_cur is not None and len(left_xy):
            left_cur = _join_lane_chains(left_cur, left_xy, via_xy[i])
        elif len(left_xy):
            left_cur = left_xy

        if len(right_xy) < 2:
            if right_cur is not None:
                right_parts.append(right_cur)
                right_cur = None
        elif right_cur is not None:
            right_cur = _join_lane_chains(right_cur, right_xy, via_xy[i])
        else:
            right_cur = right_xy

    if left_cur is not None:
        left_parts.append(left_cur)
    if right_cur is not None:
        right_parts.append(right_cur)
    return left_parts, right_parts


def _draw_leg_corridors(ax, lanes: pd.DataFrame, leg_id: str, half_w: float, lw: float = 1.6, forward: bool = True) -> None:
    """Draw one leg's corridors: red = geometric LEFT lane of the travel
    direction, blue = right lane, so the sides stay stable along a whole
    multi-leg route regardless of stored leg orientation or lane
    construction method."""
    left_xy, right_xy = travel_left_right_lanes(lanes, leg_id, forward)
    _corridor_patches(ax, left_xy, half_w, "red")
    if len(left_xy):
        ax.plot(left_xy[:, 0], left_xy[:, 1], "-", color="red", linewidth=lw, zorder=12)
    if len(right_xy):
        _corridor_patches(ax, right_xy, half_w, "blue")
        ax.plot(right_xy[:, 0], right_xy[:, 1], "-", color="blue", linewidth=lw, zorder=12)


def plot_route_with_legs(
    df: pd.DataFrame,
    nodes: pd.DataFrame,
    legs: pd.DataFrame,
    lanes: pd.DataFrame,
    route: pd.Series,
    out: Path,
    params: dict[str, Any],
) -> None:
    """One objective pair: the path from A to B drawn through all its
    intermediate legs of the overall network."""
    half_w = 0.5 * float(M6.pget(params, "CORRIDOR_DIAMETER_M", 50.0))
    node_r = 0.5 * float(M6.pget(params, "NODE_CIRCLE_DIAMETER_M", 50.0))
    nofly_thr = float(M6.pget(params, "NOFLY_SLOWNESS_THRESHOLD", 10.0))

    leg_rows = [int(k) for k in route["leg_rows"]]
    leg_forward = [bool(v) for v in route.get("leg_forward", [True] * len(leg_rows))]
    route_legs = legs.iloc[leg_rows]

    pts = []
    for _, leg in route_legs.iterrows():
        for lane in ("A", "B"):
            xy = _lane_xy(lanes, leg["leg_id"], lane)
            if len(xy):
                pts.append(xy)
    if not pts:
        return
    all_xy = np.vstack(pts)
    margin = 350.0
    x0, x1 = all_xy[:, 0].min() - margin, all_xy[:, 0].max() + margin
    y0, y1 = all_xy[:, 1].min() - margin, all_xy[:, 1].max() + margin

    fig, ax = plt.subplots(figsize=(10, 9))

    in_box = (df["x"] >= x0) & (df["x"] <= x1) & (df["y"] >= y0) & (df["y"] <= y1)
    sub = df[in_box]
    nofly = sub["slowness"].to_numpy(float) >= nofly_thr
    ax.scatter(sub.loc[~nofly, "x"], sub.loc[~nofly, "y"], s=3, c="lightgray", alpha=0.5, linewidths=0, label="flyable", zorder=1)
    ax.scatter(sub.loc[nofly, "x"], sub.loc[nofly, "y"], s=6, c="black", alpha=0.85, linewidths=0, label="no-fly", zorder=2)

    # Other network legs in the window, faint, for context.
    in_window = set()
    for _, leg in legs.iterrows():
        if leg["leg_id"] in set(route_legs["leg_id"]):
            continue
        xy = _lane_xy(lanes, leg["leg_id"], "A")
        if len(xy) and (xy[:, 0] > x0).any() and (xy[:, 0] < x1).any() and (xy[:, 1] > y0).any() and (xy[:, 1] < y1).any():
            ax.plot(xy[:, 0], xy[:, 1], "-", color="gray", linewidth=0.6, alpha=0.35, zorder=4)
            in_window.add(leg["leg_id"])

    # Continuous rails: consecutive same-side lanes joined by arcs around
    # the intermediate node circles -- no center visit, no doubling back.
    via_id_seq = [s for s in str(route["via"]).split("-")]
    node_by_id = {str(r["net_id"]): np.array([float(r["x"]), float(r["y"])]) for _, r in nodes.iterrows()}
    via_xy = [node_by_id.get(v, np.zeros(2)) for v in via_id_seq]
    left_parts, right_parts = build_route_lane_chains(
        lanes, [str(l) for l in route_legs["leg_id"]], leg_forward, via_xy,
    )
    for part in left_parts:
        _corridor_patches(ax, part, half_w, "red")
        ax.plot(part[:, 0], part[:, 1], "-", color="red", linewidth=1.8, zorder=12)
    for part in right_parts:
        _corridor_patches(ax, part, half_w, "blue")
        ax.plot(part[:, 0], part[:, 1], "-", color="blue", linewidth=1.8, zorder=12)

    via_ids = set(str(route["via"]).split("-"))
    node_box = nodes[(nodes["x"] >= x0) & (nodes["x"] <= x1) & (nodes["y"] >= y0) & (nodes["y"] <= y1)]
    _node_circle_patches(ax, node_box, node_r, emphasize=via_ids)
    _plot_nodes(ax, node_box, labels=True, fontsize=8.0)

    ax.plot([], [], "-", color="red", linewidth=1.8, label="left lane (A->B travel)")
    ax.plot([], [], "-", color="blue", linewidth=1.8, label="right lane (A->B travel)")
    ax.plot([], [], "-", color="gray", linewidth=0.9, label="corridor boundary")
    ax.plot([], [], "-", color="gray", linewidth=0.6, alpha=0.5, label="other legs")

    ax.set_title(
        f"{route['pair']}  ({route['n_legs']} legs, {float(route['length_m']):.0f} m)\n"
        f"via {route['via']}",
        fontweight="bold", fontsize=11,
    )
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel("X coordinate (m)")
    ax.set_ylabel("Y coordinate (m)")
    ax.set_aspect("equal", adjustable="box")
    add_map_rule(ax)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=7.5, frameon=True)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(M6.pget(params, "LEG_FIGURE_DPI", 160)))
    plt.close(fig)


def plot_network(df: pd.DataFrame, nodes: pd.DataFrame, legs: pd.DataFrame, lanes: pd.DataFrame, out: Path, params: dict[str, Any], roundabouts: pd.DataFrame | None = None) -> None:
    """Whole-network overview with the same corridor styling."""
    half_w = 0.5 * float(M6.pget(params, "CORRIDOR_DIAMETER_M", 50.0))
    node_r = 0.5 * float(M6.pget(params, "NODE_CIRCLE_DIAMETER_M", 50.0))
    nofly_thr = float(M6.pget(params, "NOFLY_SLOWNESS_THRESHOLD", 10.0))
    fig, ax = plt.subplots(figsize=(13, 12))

    slow = df["slowness"].to_numpy(float)
    nofly = slow >= nofly_thr
    ax.scatter(df.loc[~nofly, "x"], df.loc[~nofly, "y"], s=2, c="lightgray", alpha=0.4, linewidths=0, label="flyable", zorder=1)
    ax.scatter(df.loc[nofly, "x"], df.loc[nofly, "y"], s=4, c="black", alpha=0.8, linewidths=0, label="no-fly", zorder=2)

    for _, leg in legs.iterrows():
        _draw_leg_corridors(ax, lanes, leg["leg_id"], half_w, lw=1.0)

    # roundabout rings: two concentric CIRCULAR CORRIDORS (outer red = lane A,
    # inner blue = lane B), each with the corridor buffer band, + the entry
    # meeting points on the ring edges
    if roundabouts is not None and len(roundabouts):
        lane_gap = float(M6.pget(params, "ROUNDABOUT_LANE_GAP_M", 50.0))
        ringmap = {}
        for _, r in roundabouts.iterrows():
            c = (float(r["center_x"]), float(r["center_y"]))
            r_out = float(r["radius_m"])
            r_in = max(0.3 * r_out, r_out - lane_gap)
            for rr, col in ((r_out, "red"), (r_in, "blue")):
                ax.add_patch(Circle(c, rr + half_w, fill=False, ec="0.6", lw=0.7, zorder=17))
                ax.add_patch(Circle(c, max(1.0, rr - half_w), fill=False, ec="0.6", lw=0.7, zorder=17))
                ax.add_patch(Circle(c, rr, fill=False, ec=col, lw=1.8, zorder=18))
            ax.text(c[0], c[1], str(r["rbt_id"]), color="#d35400", fontsize=7.5,
                    weight="bold", ha="center", va="center", zorder=21)
            ringmap[str(r["rbt_id"])] = np.array(c, float)
        # ring ENTRY nodes: the ACTUAL lane endpoints on each ring. Lane A meets
        # the OUTER (red) ring, lane B penetrates to the INNER (blue) ring (see
        # clip_legs_to_rings), so every incident leg contributes TWO entry nodes
        # -- one red (outer), one blue (inner) -- radially + laterally separated.
        ent = {"A": ([], [], "red", "#e8352a", "ring entry (outer / lane A)"),
               "B": ([], [], "blue", "#1f4fd6", "ring entry (inner / lane B)")}
        for _, leg in legs.iterrows():
            for nid in (str(leg["a_id"]), str(leg["b_id"])):
                if nid not in ringmap:
                    continue
                cc = ringmap[nid]
                for lane in ("A", "B"):
                    lxy = _lane_xy(lanes, str(leg["leg_id"]), lane)
                    if len(lxy) < 1:
                        continue
                    e = (lxy[0] if np.hypot(*(lxy[0] - cc)) < np.hypot(*(lxy[-1] - cc))
                         else lxy[-1])
                    ent[lane][0].append(e[0]); ent[lane][1].append(e[1])
        for lane in ("A", "B"):
            xs, ys, _ecol, fcol, lab = ent[lane]
            if xs:
                ax.scatter(xs, ys, s=24, c=fcol, edgecolors="k", linewidths=0.5,
                           zorder=23, label=lab)
        ax.plot([], [], "-", color="red", lw=2.0, label="roundabout outer lane")
        ax.plot([], [], "-", color="blue", lw=2.0, label="roundabout inner lane")

    node_circle_nodes = nodes[nodes["kind"] != "roundabout"] if "kind" in nodes.columns else nodes
    _node_circle_patches(ax, node_circle_nodes, node_r)
    _plot_nodes(ax, node_circle_nodes, labels=True, fontsize=6.5)
    if "node_shift_m" in nodes.columns:
        moved = nodes[nodes["node_shift_m"] > 0]
        for _, r in moved.iterrows():
            ax.plot([r["x_orig"], r["x"]], [r["y_orig"], r["y"]], "-", color="purple", linewidth=0.8, zorder=14)
            ax.scatter([r["x_orig"]], [r["y_orig"]], marker="x", s=30, c="purple", linewidths=1.0, zorder=14)
        if len(moved):
            ax.scatter([], [], marker="x", s=30, c="purple", label="node before shift")
    ax.plot([], [], "-", color="red", linewidth=1.6, label="lane A centerline")
    ax.plot([], [], "-", color="blue", linewidth=1.6, label="lane B centerline")
    ax.plot([], [], "-", color="gray", linewidth=0.9, label="corridor boundary")

    ax.set_title(
        f"UAV corridor network {VERSION}: maximized objective/TN connections, "
        f"2 tangent lanes Ø{2 * half_w:.0f} m per connection",
        fontweight="bold",
    )
    ax.set_xlabel("X coordinate (m)")
    ax.set_ylabel("Y coordinate (m)")
    ax.set_aspect("equal", adjustable="box")
    add_map_rule(ax)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(M6.pget(params, "PLOT_DPI", 250)))
    plt.close(fig)


# ======================================================================
# Roundabouts: fat ring-nodes at high-degree junction areas
# ======================================================================
#
# Where many corridors meet at one traffic node (or a tight cluster of
# them) the point-junction serialises all the crossing/merging traffic
# -> holding (the core bottleneck). BEFORE the routes settle, such areas
# are REPLACED by a ROUNDABOUT: a single "fat" ring-node placed at the
# cluster centre with an optimally-sized radius. It carries the shared
# cost identity (model_idx) of the member nodes it absorbs; the members
# (and their internal member-member legs) are removed, so on the network
# rebuild the surrounding legs re-connect straight to the ring node and
# are then CLIPPED to its boundary -- "the legs connect to the ring".

def _clip_polyline_to_circle(xy: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """Trim a polyline oriented OUTSIDE(0) -> ring-centre(-1) so it stops
    ON the ring boundary. Returns the outside part plus the boundary hit."""
    xy = np.asarray(xy, float)
    if len(xy) < 2:
        return xy
    d = np.hypot(xy[:, 0] - center[0], xy[:, 1] - center[1])
    inside = d <= radius
    if not inside.any():
        return xy                              # never reaches the ring
    k = int(np.argmax(inside))                 # first point inside the circle
    if k == 0:
        return xy[:1]                          # whole polyline inside (degenerate)
    p0, p1 = xy[k - 1], xy[k]
    seg = p1 - p0
    a = float(seg @ seg)
    b = float(2.0 * seg @ (p0 - center))
    c = float((p0 - center) @ (p0 - center) - radius * radius)
    t = 1.0
    if a > 1e-9:
        disc = b * b - 4 * a * c
        if disc >= 0:
            # p0 is OUTSIDE, p1 INSIDE -> the boundary crossing is the NEAR
            # (entry) root, i.e. the smaller t (use -sqrt, not +sqrt)
            t = min(1.0, max(0.0, (-b - math.sqrt(disc)) / (2.0 * a)))
    hit = p0 + t * seg
    return np.vstack([xy[:k], hit])


def _clip_end_at_ring(xy: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """Clip whichever END of a polyline lands inside the ring to its boundary."""
    xy = np.asarray(xy, float)
    if len(xy) < 2:
        return xy
    d0 = np.hypot(*(xy[0] - center))
    d1 = np.hypot(*(xy[-1] - center))
    if d0 < d1:                                # ring end is index 0 -> flip, clip, flip
        return _clip_polyline_to_circle(xy[::-1], center, radius)[::-1]
    return _clip_polyline_to_circle(xy, center, radius)


def _fit_ring(center: np.ndarray, radius: float, nofly_tree,
              ring_geo: list, obst_clr: float, ring_gap: float,
              nudge_max: float, nudge_passes: int,
              half_w: float = 0.0) -> tuple[np.ndarray, float]:
    """Fit a ring of the DESIRED radius by first NUDGING the centre away
    from the nearest obstacle / other rings (estimated from centre+radius),
    only shrinking the radius as a last resort. Returns (centre, radius).

    The ring is a BUFFERED circular corridor (half_w each side), so the
    obstacle clearance is applied to the OUTER BUFFER EDGE (radius + half_w),
    not the centerline -- otherwise the buffer band falls into the no-fly
    cell it is nominally `obst_clr` away from."""
    center = np.asarray(center, float).copy()
    for _ in range(max(1, nudge_passes)):
        moved = False
        if nofly_tree is not None:                      # push off the nearest no-fly cell
            d, idx = nofly_tree.query(center, k=1)
            need = radius + half_w + obst_clr - float(d)
            if need > 1.0:
                obs = np.asarray(nofly_tree.data[int(idx)], float)
                v = center - obs
                n = float(np.hypot(*v)) or 1.0
                center = center + (v / n) * min(need, nudge_max)
                moved = True
        for (_id, oc, orad) in ring_geo:                # push off overlapping rings
            v = center - oc
            dd = float(np.hypot(*v)) or 1.0
            need = radius + orad + ring_gap - dd
            if need > 1.0:
                center = center + (v / dd) * min(need, nudge_max)
                moved = True
        if not moved:
            break
    # whatever overlap survives the nudging is removed by shrinking (buffer edge)
    if nofly_tree is not None:
        d, _ = nofly_tree.query(center, k=1)
        radius = min(radius, float(d) - obst_clr - half_w)
    for (_id, oc, orad) in ring_geo:
        radius = min(radius, float(np.hypot(*(center - oc))) - orad - ring_gap)
    return center, radius


def build_roundabouts(nodes: pd.DataFrame, legs: pd.DataFrame, nofly_tree,
                      params: dict[str, Any]):
    """Replace high-degree TN junction areas with roundabout ring-nodes.
    Returns (new_nodes, roundabouts_df, members_df). Empty frames when the
    feature is off or no qualifying junction exists."""
    empty = (pd.DataFrame(), pd.DataFrame())
    if not bool(M6.pget(params, "ROUNDABOUT_ENABLE", True)) or not len(legs):
        return nodes, *empty
    min_deg   = int(M6.pget(params, "ROUNDABOUT_MIN_DEGREE", 5))
    merge_r   = float(M6.pget(params, "ROUNDABOUT_MERGE_RADIUS_M", 320.0))
    min_rad   = float(M6.pget(params, "ROUNDABOUT_MIN_RADIUS_M", 120.0))
    max_rad   = float(M6.pget(params, "ROUNDABOUT_MAX_RADIUS_M", 450.0))
    margin    = float(M6.pget(params, "ROUNDABOUT_MEMBER_MARGIN_M", 40.0))
    obst_clr  = float(M6.pget(params, "ROUNDABOUT_OBSTACLE_CLEARANCE_M", 40.0))
    ring_gap  = float(M6.pget(params, "ROUNDABOUT_RING_GAP_M", 40.0))
    dens_gain = float(M6.pget(params, "ROUNDABOUT_DENSITY_GAIN_M", 18.0))
    nudge_max = float(M6.pget(params, "ROUNDABOUT_NUDGE_MAX_M", 200.0))
    nudge_n   = int(M6.pget(params, "ROUNDABOUT_NUDGE_PASSES", 4))
    half_w    = 0.5 * float(M6.pget(params, "CORRIDOR_DIAMETER_M", 50.0))  # ring is a buffered corridor
    # per-ring RADIUS CAP, keyed by any member node label (stable across ring
    # renumbering) -> shrink an over-large ring, e.g. {"MAJ013": 370.0}
    rad_caps  = M6.pget(params, "ROUNDABOUT_RADIUS_OVERRIDES", {}) or {}

    # degree = number of incident legs per node ("multiple connect legs")
    deg = Counter()
    for a, b in zip(legs["a_id"].astype(str), legs["b_id"].astype(str)):
        deg[a] += 1
        deg[b] += 1
    la = legs["a_id"].astype(str).to_numpy()
    lb = legs["b_id"].astype(str).to_numpy()

    kind = nodes["kind"].astype(str)
    is_tn = kind.str.startswith("tn_") & ~nodes["kind"].isin(["tn_backup", "tn_ext"])
    nid = nodes["net_id"].astype(str)
    seed_mask = is_tn & nid.map(lambda n: deg.get(n, 0) >= min_deg)
    seeds = nodes[seed_mask].reset_index(drop=True)
    if not len(seeds):
        return nodes, *empty

    # single-linkage clustering of seeds within merge_r
    sxy = seeds[["x", "y"]].to_numpy(float)
    parent = list(range(len(seeds)))
    def find(a):
        r = a
        while parent[r] != r:
            r = parent[r]
        while parent[a] != r:
            parent[a], a = r, parent[a]
        return r
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            if np.hypot(*(sxy[i] - sxy[j])) <= merge_r:
                parent[find(i)] = find(j)
    clusters = defaultdict(list)
    for i in range(len(seeds)):
        clusters[find(i)].append(i)

    tn_all = nodes[kind.str.startswith("tn_")].copy()
    used: set[str] = set()
    rbt_rows: list[dict] = []
    member_rows: list[dict] = []
    ring_geo: list[tuple[str, np.ndarray, float]] = []
    rid = 0
    for _, idxs in sorted(clusters.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        seed_ids = [str(seeds.iloc[i]["net_id"]) for i in idxs]
        members = set(seed_ids)
        # absorb nearby TN nodes within merge_r of any seed in the cluster
        for _, tr in tn_all.iterrows():
            m = str(tr["net_id"])
            if m in used or m in members:
                continue
            if any(np.hypot(tr["x"] - seeds.iloc[i]["x"],
                            tr["y"] - seeds.iloc[i]["y"]) <= merge_r for i in idxs):
                members.add(m)
        mem_df = nodes[nid.isin(members)]
        mxy = mem_df[["x", "y"]].to_numpy(float)
        center = mxy.mean(0)
        spread = (float(np.max(np.hypot(mxy[:, 0] - center[0], mxy[:, 1] - center[1])))
                  if len(mxy) > 1 else 0.0)
        # traffic DENSITY = # legs entering the cluster (exactly one endpoint a
        # member). Busier junctions get a bigger ring.
        mem_arr = np.array(list(members))
        entries = int(np.sum(np.isin(la, mem_arr) ^ np.isin(lb, mem_arr)))
        # the ring is a BUFFERED circular corridor (half_w each side), so members
        # within R_out + half_w are already covered -> the ring can be half_w smaller
        desired = min(max_rad, max(min_rad, spread + margin - half_w) + dens_gain * entries)
        for m in members:                                   # per-ring radius cap
            if m in rad_caps:
                desired = min(desired, float(rad_caps[m]))
                break
        # fit by NUDGING the centre off obstacles/rings first, shrink only if needed
        center, radius = _fit_ring(center, desired, nofly_tree, ring_geo,
                                   obst_clr, ring_gap, nudge_max, nudge_n, half_w)
        if radius + half_w < 0.9 * spread or radius < 0.5 * min_rad:
            continue                                        # no room -> leave point nodes
        rid += 1
        rbt_id = f"RBT{rid:02d}"
        ring_geo.append((rbt_id, center, radius))
        used |= members
        seed_member = max(members, key=lambda n: deg.get(n, 0))     # shared cost identity
        smi = int(nodes.loc[nid == seed_member, "model_idx"].iloc[0])
        rbt_rows.append({
            "rbt_id": rbt_id, "center_x": float(center[0]), "center_y": float(center[1]),
            "radius_m": float(radius), "n_members": len(members), "n_entries": entries,
            "members": "+".join(sorted(members)), "cost_from": seed_member, "model_idx": smi,
        })
        for m in sorted(members):
            member_rows.append({"rbt_id": rbt_id, "node": m,
                                "degree": deg.get(m, 0), "is_seed": m in seed_ids})

    if not rbt_rows:
        return nodes, *empty
    roundabouts = pd.DataFrame(rbt_rows)

    # rebuild node table: drop absorbed members, append ring nodes
    keep = nodes[~nid.isin(used)].copy()
    ring_nodes = pd.DataFrame([{
        "net_id": r["rbt_id"], "kind": "roundabout", "label_prefix": "RBT",
        "model_idx": int(r["model_idx"]), "x": r["center_x"], "y": r["center_y"],
        "x_orig": r["center_x"], "y_orig": r["center_y"], "node_shift_m": 0.0,
        "radius_m": r["radius_m"], "members": r["members"],
    } for _, r in roundabouts.iterrows()])
    new_nodes = pd.concat([keep, ring_nodes], ignore_index=True)
    for col, fill in (("radius_m", 0.0), ("members", "")):
        if col not in new_nodes.columns:
            new_nodes[col] = fill
        new_nodes[col] = new_nodes[col].fillna(fill)
    new_nodes["net_node"] = np.arange(len(new_nodes), dtype=int)
    return new_nodes, roundabouts, pd.DataFrame(member_rows)


def remove_legs_through_rings(legs: pd.DataFrame, lanes: pd.DataFrame,
                              roundabouts: pd.DataFrame):
    """Drop any leg whose centerline crosses a ring disk it does NOT connect
    to -- that traffic must route THROUGH the roundabout (e.g. an A-B leg that
    happens to pass straight over ring C is removed; A-C-B is used instead)."""
    if not len(roundabouts) or not len(legs):
        return legs, lanes, set()
    ring = {str(r["rbt_id"]): (np.array([r["center_x"], r["center_y"]], float),
                               float(r["radius_m"])) for _, r in roundabouts.iterrows()}

    def seg_dist(P, A, B):
        AB = B - A
        L2 = float(AB @ AB) or 1.0
        t = min(1.0, max(0.0, float((P - A) @ AB) / L2))
        return float(np.hypot(*(A + t * AB - P)))

    drop = set()
    for _, leg in legs.iterrows():
        xy = np.asarray(leg["path_xy"], float)
        if len(xy) < 2:
            continue
        for rid, (c, R) in ring.items():
            if str(leg["a_id"]) == rid or str(leg["b_id"]) == rid:
                continue
            if min(seg_dist(c, xy[i], xy[i + 1]) for i in range(len(xy) - 1)) < R - 1.0:
                drop.add(str(leg["leg_id"]))
                break
    if drop:
        legs = legs[~legs["leg_id"].astype(str).isin(drop)].reset_index(drop=True)
        if len(lanes):
            lanes = lanes[~lanes["leg_id"].astype(str).isin(drop)].reset_index(drop=True)
    return legs, lanes, drop


def clip_legs_to_rings(legs: pd.DataFrame, lanes: pd.DataFrame,
                       roundabouts: pd.DataFrame, params: dict[str, Any]):
    """Cut every ring-incident leg at the ring, REMOVING the interior segment.
    A two-lane roundabout has an OUTER (red, lane A) and INNER (blue, lane B)
    circulating lane: the red corridor lane meets the outer ring, the blue lane
    penetrates to the inner ring (crossing the outer one)."""
    if not len(roundabouts) or not len(legs):
        return legs, lanes
    lane_gap = float(M6.pget(params, "ROUNDABOUT_LANE_GAP_M", 50.0))
    # (centre, R_outer, R_inner) per ring
    ring = {str(r["rbt_id"]): (np.array([r["center_x"], r["center_y"]], float),
                               float(r["radius_m"]),
                               max(0.3 * float(r["radius_m"]), float(r["radius_m"]) - lane_gap))
            for _, r in roundabouts.iterrows()}
    legs = legs.reset_index(drop=True).copy()
    new_paths, new_len = [], []
    for _, leg in legs.iterrows():
        pxy = np.asarray(leg["path_xy"], float).copy()
        for nid in (str(leg["a_id"]), str(leg["b_id"])):
            if nid in ring:
                c, r_out, _r_in = ring[nid]
                pxy = _clip_end_at_ring(pxy, c, r_out)          # centerline -> outer
        new_paths.append(pxy)
        new_len.append(float(np.hypot(np.diff(pxy[:, 0]), np.diff(pxy[:, 1])).sum())
                       if len(pxy) >= 2 else 0.0)
    legs["path_xy"] = new_paths          # whole-column assign (reliable for arrays)
    legs["length_m"] = new_len

    if len(lanes):
        ends = {str(r["leg_id"]): (str(r["a_id"]), str(r["b_id"])) for _, r in legs.iterrows()}
        out = []
        for (lid, lane), g in lanes.groupby(["leg_id", "lane"]):
            lxy = g.sort_values("seq")[["x", "y"]].to_numpy(float)
            for nid in ends.get(str(lid), ()):
                if nid in ring:
                    c, r_out, r_in = ring[nid]
                    rad = r_out if str(lane) == "A" else r_in   # red->outer, blue->inner
                    lxy = _clip_end_at_ring(lxy, c, rad)
            for seq, (x, y) in enumerate(lxy):
                out.append({"leg_id": lid, "lane": lane, "seq": int(seq),
                            "x": float(x), "y": float(y)})
        lanes = pd.DataFrame(out)
    return legs, lanes


def _min_dist_point_polyline(c: np.ndarray, xy: np.ndarray) -> float:
    """Shortest distance from point c to a polyline (segment-accurate, not just
    to the vertices -- a coarse lane can graze a ring between two vertices)."""
    if len(xy) < 2:
        return float(np.min(np.hypot(xy[:, 0] - c[0], xy[:, 1] - c[1]))) if len(xy) else math.inf
    a, b = xy[:-1], xy[1:]
    ab = b - a
    l2 = np.einsum("ij,ij->i", ab, ab)
    l2 = np.where(l2 == 0.0, 1.0, l2)
    t = np.clip(np.einsum("ij,ij->i", c - a, ab) / l2, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return float(np.min(np.hypot(proj[:, 0] - c[0], proj[:, 1] - c[1])))


def _lane_ring_gaps(lane_xy: dict, ends_of: dict, ring: dict, _half_sep: float = 0.0):
    """For every (leg, ring) the leg does NOT connect to, the min gap between the
    NEAR corridor lane centerline and the ring's outer lane centerline (radial
    distance - R_out), measured segment-accurately. Yields
    (leg_id, rbt_id, gap_m, near_radial_m)."""
    for lid, ends in ends_of.items():
        for rid, (c, r_out, _r_in) in ring.items():
            if rid in ends:
                continue                               # this leg connects to the ring
            near = math.inf
            for lane in ("A", "B"):
                lxy = lane_xy.get((lid, lane))
                if lxy is None or len(lxy) == 0:
                    continue
                near = min(near, _min_dist_point_polyline(c, lxy))
            if math.isfinite(near):
                yield lid, rid, near - r_out, near


def separate_legs_from_ring_buffers(legs: pd.DataFrame, lanes: pd.DataFrame,
                                    roundabouts: pd.DataFrame, nofly_tree,
                                    params: dict[str, Any]):
    """Keep every corridor's BUFFER clear of every ring's BUFFER.

    ``remove_legs_through_rings`` only tests leg CENTERLINES, so a corridor
    whose centerline clears a ring it does not connect to can still have one of
    its +/-half_sep offset lanes (and that lane's buffer band) dip into the
    ring's buffer. For each offending (leg, ring) pair this pass SHIFTS the
    corridor outward around the ring: the two lanes are re-sampled and the
    stretch that grazes the ring is pushed radially out onto two concentric arcs
    -- the near lane onto ``R_out + RING_CORRIDOR_SEPARATION_M`` and the far lane
    one full ``MIN_CENTERLINE_SEPARATION_M`` beyond it. This preserves the
    lane-to-lane separation and the fixed leg endpoints, and leaves the rest of
    the corridor (already cleared of obstacles) untouched. A leg whose ENDPOINT
    node itself lies inside the ring buffer cannot be shifted and is reported as
    blocked. Returns (legs, lanes, report_df)."""
    if not bool(M6.pget(params, "RING_SEPARATION_ENABLE", True)) \
            or not len(roundabouts) or not len(legs):
        return legs, lanes, pd.DataFrame()

    sep       = float(M6.pget(params, "RING_CORRIDOR_SEPARATION_M",
                              M6.pget(params, "MIN_CENTERLINE_SEPARATION_M", 50.0)))
    lane_sep  = float(M6.pget(params, "MIN_CENTERLINE_SEPARATION_M", 50.0))   # A<->B spacing
    half_w    = 0.5 * float(M6.pget(params, "CORRIDOR_DIAMETER_M", 50.0))     # lane buffer
    lane_gap  = float(M6.pget(params, "ROUNDABOUT_LANE_GAP_M", 50.0))
    dens_step = float(M6.pget(params, "RING_DETOUR_SAMPLE_M", 15.0))
    max_pass  = int(M6.pget(params, "RING_DETOUR_MAX_PASSES", 3))

    # (centre, R_outer, R_inner) per ring -- corridors approach from OUTSIDE, so
    # the OUTER ring lane is the binding one for the separation.
    ring = {str(r["rbt_id"]): (np.array([r["center_x"], r["center_y"]], float),
                               float(r["radius_m"]),
                               max(0.3 * float(r["radius_m"]), float(r["radius_m"]) - lane_gap))
            for _, r in roundabouts.iterrows()}

    legs = legs.reset_index(drop=True).copy()
    ends_of = {str(r["leg_id"]): (str(r["a_id"]), str(r["b_id"])) for _, r in legs.iterrows()}

    lane_xy: dict = {}                                  # (leg_id, lane) -> polyline
    if len(lanes):
        for (lid, ln), g in lanes.groupby(["leg_id", "lane"]):
            lane_xy[(str(lid), str(ln))] = g.sort_values("seq")[["x", "y"]].to_numpy(float)

    # violations BEFORE: near lane centerline closer than R_out + sep to a ring
    before = {(lid, rid): gap for lid, rid, gap, _ in
              _lane_ring_gaps(lane_xy, ends_of, ring, 0.0) if gap < sep - 1.0}
    if not before:
        return legs, lanes, pd.DataFrame()

    def clamp_out(arr: np.ndarray, c: np.ndarray, r_target: float) -> bool:
        """Push INTERIOR vertices inside r_target radially out onto the r_target
        circle (endpoints stay fixed). Returns True if anything moved. Obstacle
        clearance is enforced by the caller (revert-on-violation)."""
        d = np.hypot(arr[:, 0] - c[0], arr[:, 1] - c[1])
        moved = False
        for k in range(1, len(arr) - 1):
            if d[k] < r_target - 1e-6:
                u = (arr[k] - c) / (d[k] if d[k] > 1e-9 else 1.0)
                arr[k] = c + u * r_target
                moved = True
        return moved

    blocked: set = set()
    obstacle_blocked: set = set()
    changed_legs: set = set()
    for _ in range(max(1, max_pass)):
        changed = False
        for lid in {l for (l, _r) in before}:
            ends = ends_of[lid]
            A, B = lane_xy.get((lid, "A")), lane_xy.get((lid, "B"))
            if A is None or B is None or len(A) < 2 or len(B) < 2:
                continue
            Af = M6._resample_polyline_m(A, dens_step)   # interior vertices to bend
            Bf = M6._resample_polyline_m(B, dens_step)

            def obs_clr(*arrs) -> float:                  # min lane clearance to a no-fly cell
                if nofly_tree is None:
                    return math.inf
                return min(float(nofly_tree.query(a, k=1)[0].min()) for a in arrs)

            moved = False
            for rid, (c, r_out, _r_in) in ring.items():
                if rid in ends:
                    continue
                dA = np.hypot(Af[:, 0] - c[0], Af[:, 1] - c[1])
                dB = np.hypot(Bf[:, 0] - c[0], Bf[:, 1] - c[1])
                # near lane -> R_out+sep, far lane one lane_sep beyond, so the two
                # stay lane_sep apart on concentric arcs and both clear the ring.
                inner, dI, outer, dO = ((Af, dA, Bf, dB) if dA.min() <= dB.min()
                                        else (Bf, dB, Af, dA))
                r_in_t, r_out_t = r_out + sep, r_out + sep + lane_sep
                if dI.min() >= r_in_t - 1e-6 and dO.min() >= r_out_t - 1e-6:
                    continue                              # this ring already clear
                # an endpoint node sitting inside the buffer cannot be shifted
                if dI[0] < r_in_t or dI[-1] < r_in_t or dO[0] < r_out_t or dO[-1] < r_out_t:
                    blocked.add((lid, rid))
                # try the shift, but OBSTACLE CLEARANCE WINS: if bending this leg
                # around the ring would push a lane buffer into a no-fly cell (a
                # corridor squeezed between the ring and an obstacle), revert this
                # ring's shift and leave the leg on its obstacle-clear path.
                snapA, snapB = Af.copy(), Bf.copy()
                clr0 = obs_clr(Af, Bf)
                mv = clamp_out(inner, c, r_in_t) | clamp_out(outer, c, r_out_t)
                if mv and obs_clr(Af, Bf) < min(half_w, clr0) - 1.0:
                    Af[:], Bf[:] = snapA, snapB           # undo -- would hit an obstacle
                    obstacle_blocked.add((lid, rid))
                    continue
                moved |= mv
            if moved:
                lane_xy[(lid, "A")], lane_xy[(lid, "B")] = Af, Bf
                idx = legs.index[legs["leg_id"].astype(str) == lid]
                if len(idx):
                    la = float(np.hypot(np.diff(Af[:, 0]), np.diff(Af[:, 1])).sum())
                    lb = float(np.hypot(np.diff(Bf[:, 0]), np.diff(Bf[:, 1])).sum())
                    legs.at[idx[0], "length_m"] = 0.5 * (la + lb)
                changed_legs.add(lid)
                changed = True
        if not changed:
            break

    # rebuild only the changed legs' rows in the lanes table
    if changed_legs and len(lanes):
        keep = lanes[~lanes["leg_id"].astype(str).isin(changed_legs)]
        rows = []
        for lid in changed_legs:
            for ln in ("A", "B"):
                for seq, (x, y) in enumerate(lane_xy[(lid, ln)]):
                    rows.append({"leg_id": lid, "lane": ln, "seq": int(seq),
                                 "x": float(x), "y": float(y)})
        lanes = pd.concat([keep, pd.DataFrame(rows)], ignore_index=True)

    after = {(lid, rid): gap for lid, rid, gap, _ in
             _lane_ring_gaps(lane_xy, ends_of, ring, 0.0)}
    report_rows = []
    for (lid, rid), gap0 in sorted(before.items(), key=lambda kv: kv[1]):
        gap1 = after.get((lid, rid), gap0)
        if (lid, rid) in blocked:
            method = "endpoint_blocked"
        elif (lid, rid) in obstacle_blocked:
            method = "obstacle_blocked"
        else:
            method = "shifted"
        report_rows.append({
            "leg_id": lid, "ring": rid,
            "gap_before_m": round(gap0, 1), "gap_after_m": round(gap1, 1),
            "required_m": round(sep, 1),
            "resolved": bool(gap1 >= sep - 1.0),
            "method": method,
        })
    return legs, lanes, pd.DataFrame(report_rows)


def apply_leg_lateral_shifts(legs: pd.DataFrame, lanes: pd.DataFrame, nofly_tree,
                             params: dict[str, Any]):
    """Manual per-leg lateral BOW (LEG_LATERAL_SHIFTS = {leg_id: metres}). The
    leg's corridor is bowed sideways toward +x (east; sign of the value flips it
    west) by up to the given amount in the middle, tapered back to its fixed
    endpoints so the leg still meets its nodes, with the A<->B lane separation
    preserved. Used to pull a corridor off a neighbour it runs too close to. The
    bow is never pushed into an obstacle (per-vertex clamped to keep half_w
    clearance). Returns (legs, lanes, applied) where applied = [(leg_id, m)]."""
    shifts = M6.pget(params, "LEG_LATERAL_SHIFTS", {}) or {}
    if not shifts or not len(lanes):
        return legs, lanes, []
    half_w   = 0.5 * float(M6.pget(params, "CORRIDOR_DIAMETER_M", 50.0))
    dens     = float(M6.pget(params, "RING_DETOUR_SAMPLE_M", 15.0))
    ramp     = 0.2                                       # taper fraction at each end
    lane_xy  = {}
    for (lid, ln), g in lanes.groupby(["leg_id", "lane"]):
        lane_xy[(str(lid), str(ln))] = g.sort_values("seq")[["x", "y"]].to_numpy(float)

    def rfrac(xy: np.ndarray, n: int) -> np.ndarray:     # resample to n points by arclength
        seg = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
        L = np.concatenate([[0.0], np.cumsum(seg)])
        s = np.linspace(0.0, L[-1], n)
        return np.column_stack([np.interp(s, L, xy[:, 0]), np.interp(s, L, xy[:, 1])])

    _DIRS = {"east": (1., 0.), "right": (1., 0.), "west": (-1., 0.), "left": (-1., 0.),
             "north": (0., 1.), "up": (0., 1.), "south": (0., -1.), "down": (0., -1.)}
    applied, changed = [], set()
    for lid, spec in shifts.items():
        lid = str(lid)
        # spec is a bare magnitude (toward +x, sign flips it) OR a (magnitude,
        # direction) pair where direction is "up"/"down"/"east"/"west"/... or a
        # (dx, dy) vector -- the leg is bowed along the perpendicular nearest that
        # direction.
        if isinstance(spec, (list, tuple)) and len(spec) == 2:
            off = float(spec[0])
            ds = spec[1]
            refdir = np.array(_DIRS.get(str(ds).lower(), (1., 0.))
                              if isinstance(ds, str) else ds, float)
        else:
            off = float(spec); refdir = np.array([1., 0.])
        if off < 0:
            refdir, off = -refdir, -off
        A0, B0 = lane_xy.get((lid, "A")), lane_xy.get((lid, "B"))
        if A0 is None or len(A0) < 2:
            continue
        # Re-sample both lanes to the SAME point count, take the centerline, and
        # apply ONE shared displacement to both -> the A<->B separation is
        # preserved exactly. The bow is a trapezoidal taper (0 at the fixed
        # endpoints, full in the middle) along +x (east).
        n = max(len(A0), (len(B0) if B0 is not None else 0),
                int(np.hypot(*(A0[-1] - A0[0])) / dens) + 2)
        A1 = rfrac(A0, n)
        B1 = rfrac(B0, n) if (B0 is not None and len(B0) >= 2) else None
        mid = A1 if B1 is None else 0.5 * (A1 + B1)
        d = mid[-1] - mid[0]
        clen = float(np.hypot(*d)) or 1.0
        u = d / clen
        p = np.array([-u[1], u[0]])                      # leg perpendicular
        en = p if float(p @ refdir) >= 0 else -p         # ... on the requested side
        t = np.clip(((mid - mid[0]) @ u) / clen, 0.0, 1.0)
        taper = np.clip(np.minimum(t / ramp, (1.0 - t) / ramp), 0.0, 1.0)
        disp = (off * taper)[:, None] * en
        lanes_k = [A1] + ([B1] if B1 is not None else [])
        if nofly_tree is not None:                      # never bow a lane into an obstacle
            for k in range(len(mid)):                   # shared fraction keeps A<->B intact
                if any(float(nofly_tree.query(a[k] + disp[k], k=1)[0]) < half_w for a in lanes_k):
                    lo, hi = 0.0, 1.0
                    for _ in range(14):
                        m = 0.5 * (lo + hi)
                        if all(float(nofly_tree.query(a[k] + m * disp[k], k=1)[0]) >= half_w
                               for a in lanes_k):
                            lo = m
                        else:
                            hi = m
                    disp[k] *= lo
        # Regenerate the lanes as true PARALLEL offsets of the bowed centerline:
        # a sheared displacement field compresses the inside of the bend, so
        # rebuild them concentric to hold the full A<->B separation.
        mid_bowed = mid + disp
        half_sep = 0.5 * float(M6.pget(params, "MIN_CENTERLINE_SEPARATION_M", 50.0))
        leftn = np.array([-u[1], u[0]])
        sA = 1.0 if (B1 is None or float(((A1 - mid) @ leftn).mean()) >= 0) else -1.0
        An = offset_polyline(mid_bowed, sA * half_sep)
        Bn = offset_polyline(mid_bowed, -sA * half_sep) if B1 is not None else None
        if An is not None and (B1 is None or Bn is not None):
            lane_xy[(lid, "A")] = An
            if Bn is not None:
                lane_xy[(lid, "B")] = Bn
        else:                                           # offset failed -> shared shift
            lane_xy[(lid, "A")] = A1 + disp
            if B1 is not None:
                lane_xy[(lid, "B")] = B1 + disp
        lbl = (("east" if en[0] > 0 else "west") if abs(en[0]) >= abs(en[1])
               else ("north" if en[1] > 0 else "south"))
        changed.add(lid); applied.append((lid, off, lbl))

    if changed:
        keep = lanes[~lanes["leg_id"].astype(str).isin(changed)]
        rows = [{"leg_id": lid, "lane": ln, "seq": int(i), "x": float(x), "y": float(y)}
                for (lid, ln), poly in lane_xy.items() if lid in changed
                for i, (x, y) in enumerate(poly)]
        lanes = pd.concat([keep, pd.DataFrame(rows)], ignore_index=True)
        legs = legs.reset_index(drop=True).copy()
        for lid in changed:
            idx = legs.index[legs["leg_id"].astype(str) == lid]
            if len(idx) and (lid, "A") in lane_xy and (lid, "B") in lane_xy:
                la = float(np.hypot(*np.diff(lane_xy[(lid, "A")], axis=0).T).sum())
                lb = float(np.hypot(*np.diff(lane_xy[(lid, "B")], axis=0).T).sum())
                legs.at[idx[0], "length_m"] = 0.5 * (la + lb)
    return legs, lanes, applied


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    args = parse_args()
    params = M6.load_params(args.param_file)

    model_file = Path(str(M6.pget(params, "MODEL_FILE", "output/03_clustering_hitcount/master_plan_input_nodes.csv")))
    output_dir = Path(str(M6.pget(params, "OUTPUT_DIR", "output/06_corridor_network")))
    fig_dir = output_dir / "figures"
    route_fig_dir = fig_dir / "route_with_legs"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"UAV CORRIDOR NETWORK BUILDER {VERSION}")
    print("=" * 80)
    print(f"Param file      : {args.param_file}")
    print(f"Model file      : {model_file}")
    print(f"Output directory: {output_dir}")

    nofly_threshold = float(M6.pget(params, "NOFLY_SLOWNESS_THRESHOLD", 10.0))
    df = M6.read_node_model(model_file)
    df, cell_to_idx, grid_m = M6.add_grid_index(df)
    base_allowed = df["slowness"].to_numpy(float) < nofly_threshold
    print(f"Model nodes     : {len(df):,} (grid {grid_m:.1f} m, {int((~base_allowed).sum()):,} no-fly)")

    nodes = build_network_nodes(df, params)
    nodes["x_orig"] = nodes["x"]
    nodes["y_orig"] = nodes["y"]
    nodes["node_shift_m"] = 0.0
    n_obj = int((nodes["kind"] == "objective").sum())
    print(f"Network nodes   : {len(nodes)} ({n_obj} objectives, {len(nodes) - n_obj} TN)")

    nofly_tree = make_nofly_tree(df, params)

    def build_network(nodes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        legs = build_legs(df, cell_to_idx, base_allowed, nodes, nofly_tree, params)
        if not len(legs):
            raise RuntimeError("No connections could be built -- check LEG_KNN_K / LEG_MAX_LENGTH_M.")
        legs = filter_crossing_legs(legs, df, nodes, params)
        lanes, lane_report = build_leg_lanes(df, cell_to_idx, base_allowed, legs, nofly_tree, params)
        return legs, lanes, lane_report

    legs, lanes, lane_report = build_network(nodes)

    # Suggested TNs: purple backup TNs fill the low-density areas
    # INSIDE the network hull (redundancy when a primary corridor is
    # unavailable); magenta extension TNs open up the unused flyable
    # space OUTSIDE the hull. Both join the rebuild like normal nodes.
    n_before = len(nodes)
    if bool(M6.pget(params, "BACKUP_TN_ENABLE", True)):
        nodes = suggest_backup_tn(df, nodes, legs, nofly_tree, params)
        n_bak = len(nodes) - n_before
        if n_bak:
            print(f"Backup TN       : {n_bak} purple TNs suggested in low-density areas")
    if bool(M6.pget(params, "EXPANSION_TN_ENABLE", True)):
        n_mid = len(nodes)
        nodes = suggest_expansion_tn(df, nodes, legs, nofly_tree, params)
        n_ext = len(nodes) - n_mid
        if n_ext:
            print(f"Expansion TN    : {n_ext} magenta TNs suggested outside the network hull")
    if len(nodes) > n_before:
        print("Suggested TN    : rebuilding network with the suggested nodes")
        legs, lanes, lane_report = build_network(nodes)
        # A suggestion no leg could reach (blocked straight lines, Theta*
        # detour too long) stays useless: drop it and rebuild.
        used = set(legs["a_id"].astype(str)) | set(legs["b_id"].astype(str))
        orphan = nodes[nodes["kind"].isin(["tn_backup", "tn_ext"]) & ~nodes["net_id"].isin(used)]
        if len(orphan):
            print(f"  drop {', '.join(orphan['net_id'])}: no connection could be built")
            nodes = nodes[~nodes["net_id"].isin(set(orphan["net_id"]))].reset_index(drop=True)
            nodes["net_node"] = np.arange(len(nodes), dtype=int)
            legs, lanes, lane_report = build_network(nodes)

    # Shifted corridors take their end nodes with them: the node circle
    # moves by the average demanded displacement so the lanes stay
    # tangent to it, then the whole network is rebuilt from the moved
    # nodes (which usually turns the shifted pairs back into symmetric
    # tangent pairs).
    if bool(M6.pget(params, "NODE_SHIFT_ENABLE", True)):
        for _ in range(int(M6.pget(params, "NODE_SHIFT_MAX_PASSES", 2))):
            moved = compute_node_shifts(nodes, legs, lane_report, nofly_tree, params)
            if moved is None:
                break
            nodes = moved
            n_m = int((nodes["node_shift_m"] > 0).sum())
            print(f"Node shift      : {n_m} node circles moved with their corridors -- rebuilding network")
            legs, lanes, lane_report = build_network(nodes)

    # Roundabouts: replace high-degree junction areas with fat ring-nodes,
    # rebuild the legs onto the rings, then clip them to the ring boundary.
    roundabouts = pd.DataFrame()
    nodes, roundabouts, rbt_members = build_roundabouts(nodes, legs, nofly_tree, params)
    if len(roundabouts):
        print(f"Roundabouts     : {len(roundabouts)} rings replace "
              f"{len(rbt_members)} junction nodes -- rebuilding legs onto the rings")
        legs, lanes, lane_report = build_network(nodes)
        legs, lanes, dropped = remove_legs_through_rings(legs, lanes, roundabouts)
        if dropped:
            print(f"  drop {len(dropped)} leg(s) crossing a ring disk (route through it): "
                  f"{', '.join(sorted(dropped))}")
        legs, lanes = clip_legs_to_rings(legs, lanes, roundabouts, params)
        # Corridor <-> ring buffer separation: bend legs whose lane buffer dips
        # into a ring buffer they do not connect to, back out to the required
        # separation (endpoints + lane-to-lane separation preserved).
        legs, lanes, ring_sep = separate_legs_from_ring_buffers(
            legs, lanes, roundabouts, nofly_tree, params)
        if len(ring_sep):
            ring_sep.to_csv(output_dir / "ring_corridor_separation.csv", index=False)
            n_fixed = int(ring_sep["resolved"].sum())
            n_blocked = int((ring_sep["method"] == "endpoint_blocked").sum())
            print(f"Ring separation : {len(ring_sep)} corridor/ring buffer conflicts "
                  f"({n_fixed} shifted clear"
                  f"{f', {n_blocked} endpoint-blocked' if n_blocked else ''}) "
                  f"-> ring_corridor_separation.csv")
            for _, r in ring_sep[~ring_sep["resolved"]].iterrows():
                print(f"  UNRESOLVED {r['leg_id']} vs {r['ring']}: "
                      f"gap {r['gap_before_m']:.0f}->{r['gap_after_m']:.0f} m "
                      f"(need {r['required_m']:.0f}; {r['method']})")
        # Manual per-leg lateral bow (LEG_LATERAL_SHIFTS) to pull a corridor off
        # a neighbour it runs too close to.
        legs, lanes, lat = apply_leg_lateral_shifts(legs, lanes, nofly_tree, params)
        for lid, off, lbl in lat:
            print(f"Lateral shift   : {lid} bowed {off:.0f} m {lbl} into open space")
        for _, r in roundabouts.iterrows():
            print(f"  {r['rbt_id']}: R={r['radius_m']:.0f} m  <- {r['members']}")

    two = int(lane_report["two_lanes"].sum())
    sep_ok = int(lane_report["separation_ok"].fillna(False).astype(bool).sum())
    n_tangent = int((lane_report["lane_b_method"] == "tangent_pair").sum())
    n_shifted = int((lane_report["lane_b_method"] == "shifted_pair").sum())
    print(
        f"Corridor lanes  : {two}/{len(lane_report)} with 2 parallel corridors "
        f"({n_tangent} tangent pairs, {n_shifted} shifted pairs, "
        f"{two - n_tangent - n_shifted} soft-buffer), {sep_ok} meet full separation"
    )

    for prefix, what in (("BAK", "backup"), ("EXT", "extension")):
        n_sugg = int(legs["a_id"].astype(str).str.startswith(prefix).sum()
                     + legs["b_id"].astype(str).str.startswith(prefix).sum())
        if n_sugg:
            print(f"{what.capitalize():<8} legs   : {n_sugg} connections ride through the suggested {what} TNs")

    pair_routes = assign_pair_routes(nodes, legs, params)
    n_routed = int(pair_routes["success"].sum())
    print(f"Pair routes     : {n_routed}/{len(pair_routes)} objective pairs reachable through the legs")

    legs_out = legs.drop(columns=["path_indices", "path_xy"]).merge(lane_report, on="leg_id", how="left")
    nodes.to_csv(output_dir / "network_nodes.csv", index=False)
    legs_out.to_csv(output_dir / "network_legs.csv", index=False)
    lanes.to_csv(output_dir / "lane_nodes.csv", index=False)
    pair_routes_out = pair_routes.copy()
    pair_routes_out["legs"] = pair_routes_out["leg_rows"].apply(lambda ks: ";".join(str(legs.iloc[int(k)]["leg_id"]) for k in ks))
    pair_routes_out["leg_directions"] = pair_routes_out["leg_forward"].apply(lambda fs: ";".join("fwd" if f else "rev" for f in fs))
    pair_routes_out.drop(columns=["leg_rows", "leg_forward"]).to_csv(output_dir / "pair_routes.csv", index=False)
    if len(roundabouts):
        roundabouts.to_csv(output_dir / "roundabouts.csv", index=False)
        rbt_members.to_csv(output_dir / "roundabout_members.csv", index=False)

    plot_network(df, nodes, legs, lanes, fig_dir / "00_corridor_network.png", params, roundabouts)

    if bool(M6.pget(params, "SAVE_ROUTE_FIGURES", True)):
        route_fig_dir.mkdir(parents=True, exist_ok=True)
        n_drawn = 0
        for _, route in pair_routes.iterrows():
            if not bool(route["success"]):
                continue
            plot_route_with_legs(df, nodes, legs, lanes, route, route_fig_dir / f"{route['pair']}.png", params)
            n_drawn += 1
        print(f"Route figures   : {n_drawn} saved to {route_fig_dir}")

    print("-" * 80)
    for name in ["network_nodes.csv", "network_legs.csv", "lane_nodes.csv", "pair_routes.csv"]:
        print(f"Saved: {output_dir / name}")
    print(f"Figure: {fig_dir / '00_corridor_network.png'}")
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
