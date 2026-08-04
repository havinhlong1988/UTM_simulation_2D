#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_cluster_theta_routes.py

Version: v1

Hit-count density + DBSCAN clustering for LAE-UTM Theta* route outputs.

    density(node) = number of generated routes that pass through that node

The hit-count map is used to classify/select candidate intermediate
connection nodes -- major/minor traffic nodes (TN) -- for the master
backbone stage (05/06):

  - major TN: an area with high route density AND strong crosscut
    (routes branch or cross there, MAJOR_CROSSCUT_MIN_GRAPH_DEGREE).
    High-density nodes carrying only a single one-directional corridor
    are demoted to minor TN.
  - minor TN: lower-threshold secondary connection nodes, selected with
    area-balanced quotas and spatial separation from the majors.
  - candidates near a DB/DK/FLZ objective are merged into that
    objective (CANDIDATE_OBJECTIVE_MERGE_*): their traffic is just the
    objective's own fan-in/fan-out.
  - candidates on an obstacle-free straight line between two objectives
    are pruned (CANDIDATE_STRAIGHT_FLOW_*): they do no obstacle
    avoidance, the through-flow passes with or without them.
  - one global minimum separation is enforced on the final pools
    (CANDIDATE_FINAL_MIN_SEPARATION_*).
  - candidates closer than CANDIDATE_MIN_OBSTACLE_CLEARANCE_M (75 m =
    node radius 25 + corridor diameter 50) to a no-fly node are shifted
    to the nearest safe flyable node (CANDIDATE_OBSTACLE_SHIFT_*), so
    the downstream corridor geometry always fits around the TN.

Run
---
    python 03_cluster_theta_routes.py

Default parameter file
----------------------
    params/dbscan.params

Main outputs
------------
    output/03_clustering_hitcount/
        node_hit_count.csv
        edge_hit_count.csv
        dbscan_hit_clusters.csv
        dbscan_hit_cluster_summary.csv
        candidate_intermediate_nodes.csv
        route_hit_cluster_membership.csv
        figures/00_node_hit_count_density.png
        figures/01_edge_hit_count_density.png
        figures/02_dbscan_hit_count_clusters.png
        figures/03_candidate_intermediate_nodes.png
"""

from __future__ import annotations

import argparse
import ast
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.maprule import add_map_rule
import matplotlib.colors as mcolors
import matplotlib.collections as mcoll

VERSION = "v1"


# ======================================================================
# Parameters
# ======================================================================

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
        return raw.strip("\"").strip("'")


def load_params(param_file: str | Path) -> dict[str, Any]:
    param_file = Path(param_file)
    if not param_file.exists():
        raise FileNotFoundError(f"Parameter file not found: {param_file}")

    params: dict[str, Any] = {}
    with open(param_file, "r", encoding="utf-8") as f:
        for line in f:
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
    return params.get(key, default)


# ======================================================================
# FLZ traffic-node handling
# ======================================================================
#
# FLZ (Forced Landing Zone) nodes are emergency-support points, not normal
# route endpoints. But 02_run_theta_plan.py's FLZ buffer requirement
# (FLZ_BUFFER_ENABLE) forces every route to pass close to at least one FLZ,
# so FLZ nodes routinely end up with very high hit_count -- often the
# highest on the map -- purely because of that structural requirement, not
# because they are organically busy corridors. Left unhandled, DBSCAN and
# the candidate selectors below will happily promote those nodes into
# major/minor "traffic node" (TN) candidates for the master backbone in
# 05_run_master_plan.py.
#
# CONSIDER_FLZ_AS_TRAFFIC_NODE controls whether that is allowed:
#   True  (default) -- FLZ nodes compete for candidate selection like any
#           other node. Appropriate when the FLZ-forced corridor is a
#           genuinely good backbone hub and reusing it is desired.
#   False -- FLZ nodes are excluded from major/minor candidate pools (same
#           mechanism as CANDIDATE_EXCLUDE_LABEL_PREFIXES for DB/DK/RA).
#           They still appear in node_hit_count / full_density / DBSCAN
#           cluster outputs (and are flagged via the is_flz column) so the
#           analysis stays visible, but they will not be exported as TN
#           candidates. Use this when FLZ should stay reserved for
#           emergency branches only (src/routeplan_PSO_ACO.py adds those separately).
def flz_prefixes(params: dict[str, Any]) -> set[str]:
    return {str(x) for x in pget(params, "FLZ_PREFIXES", ["FLZ"])}


def effective_exclude_prefixes(params: dict[str, Any]) -> set[str]:
    exclude = set(str(x) for x in pget(params, "CANDIDATE_EXCLUDE_LABEL_PREFIXES", ["DB", "DK", "RA"]))
    if not bool(pget(params, "CONSIDER_FLZ_AS_TRAFFIC_NODE", True)):
        exclude |= flz_prefixes(params)
    return exclude


def mark_is_flz(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    if "label_prefix" not in df.columns:
        df["is_flz"] = False
        return df
    df["is_flz"] = df["label_prefix"].astype(str).isin(flz_prefixes(params))
    return df


# ======================================================================
# Objective-proximity merge (candidates too close to DB/DK/FLZ)
# ======================================================================
#
# A candidate node sitting right next to a DB/DK/FLZ objective isn't a
# genuinely separate traffic hub -- it's already inside that objective's
# own driveway/buffer zone (ENDPOINT_BUFFER_M in 02, FLZ_BUFFER_RADIUS_M
# for FLZ), so any route reaching the objective already passes right by
# it. Promoting it into its own TN candidate would just duplicate the
# objective under a different name and give it a redundant TN buffer
# requirement downstream. When CANDIDATE_OBJECTIVE_MERGE_ENABLE is on,
# CANDIDATE_MIN_DISTANCE_TO_OBJECTIVE_M excludes (merges) any candidate
# within that radius of an objective from the major/minor pools -- it
# stays visible in node_hit_count / full_density like any excluded
# prefix, just never exported as a TN candidate.
def objective_node_indices(df: pd.DataFrame, params: dict[str, Any]) -> list[int]:
    prefixes = {str(p) for p in pget(params, "CANDIDATE_OBJECTIVE_MERGE_PREFIXES", ["DB", "DK", "FLZ"])}
    if "label_prefix" not in df.columns:
        return []
    pref = df["label_prefix"].astype(str)
    return [int(i) for i in np.flatnonzero(pref.isin(prefixes).to_numpy())]


def candidate_near_objective_mask(df: pd.DataFrame, params: dict[str, Any]) -> np.ndarray:
    """True for nodes within CANDIDATE_MIN_DISTANCE_TO_OBJECTIVE_M of a
    DB/DK/FLZ objective node -- see module note above. False (no merging)
    when CANDIDATE_OBJECTIVE_MERGE_ENABLE is off, the threshold is <= 0,
    or the model has no objective nodes."""
    if not bool(pget(params, "CANDIDATE_OBJECTIVE_MERGE_ENABLE", True)):
        return np.zeros(len(df), dtype=bool)
    threshold = float(pget(params, "CANDIDATE_MIN_DISTANCE_TO_OBJECTIVE_M", 100.0))
    if threshold <= 0.0:
        return np.zeros(len(df), dtype=bool)
    obj_idx = objective_node_indices(df, params)
    if not obj_idx:
        return np.zeros(len(df), dtype=bool)
    xy = df[["x", "y"]].to_numpy(float)
    obj_xy = xy[obj_idx]
    try:
        from scipy.spatial import cKDTree
        dist, _ = cKDTree(obj_xy).query(xy, k=1)
    except Exception:
        dist = np.min(np.hypot(xy[:, 0:1] - obj_xy[:, 0][None, :], xy[:, 1:2] - obj_xy[:, 1][None, :]), axis=1)
    return dist <= threshold


def objective_proximity_node_keys(full_density: pd.DataFrame, params: dict[str, Any]) -> set[str]:
    """node_key values within CANDIDATE_MIN_DISTANCE_TO_OBJECTIVE_M of a
    DB/DK/FLZ objective. MUST be computed against full_density (every
    model node, including objectives) rather than clustered/pool -- those
    are hit-nodes-only and, by the time candidate filtering runs, have
    already had DB/DK/FLZ rows themselves stripped out by
    CANDIDATE_EXCLUDE_LABEL_PREFIXES, so objective_node_indices() would
    find nothing to measure distance from."""
    if "node_key" not in full_density.columns:
        return set()
    mask = candidate_near_objective_mask(full_density, params)
    if not mask.any():
        return set()
    return set(full_density.loc[mask, "node_key"].astype(str))


# ======================================================================
# Straight-flow pruning (candidates not doing obstacle avoidance)
# ======================================================================
#
# High hit-count between a pair of objectives does not always mark a
# genuine turning hub: when the straight segment between two objectives
# is fully flyable, the dense traffic along it is just the through-flow
# of routes that fly that segment directly. A candidate sitting on such
# a segment is not part of any obstacle-avoidance detour -- corridors
# pass through that space with or without it -- so it adds no routing
# value as a TN. When CANDIDATE_STRAIGHT_FLOW_PRUNE_ENABLE is on, any
# candidate within CANDIDATE_STRAIGHT_FLOW_CORRIDOR_M lateral of an
# objective-objective segment that is clear of no-fly nodes (none within
# CANDIDATE_STRAIGHT_FLOW_CLEARANCE_M of the segment) is removed from
# the candidate pools. Candidates at genuine detour corners survive,
# because the straight segment they shortcut is blocked by an obstacle.

def prune_straight_flow_candidates(cands: pd.DataFrame, full_density: pd.DataFrame, params: dict[str, Any], pool_name: str) -> pd.DataFrame:
    if not bool(pget(params, "CANDIDATE_STRAIGHT_FLOW_PRUNE_ENABLE", True)):
        return cands
    if cands is None or len(cands) == 0 or full_density is None or len(full_density) == 0:
        return cands
    if "label_prefix" not in full_density.columns:
        return cands
    corridor_m = float(pget(params, "CANDIDATE_STRAIGHT_FLOW_CORRIDOR_M", 150.0))
    clearance_m = float(pget(params, "CANDIDATE_STRAIGHT_FLOW_CLEARANCE_M", 100.0))
    prefixes = {str(p) for p in pget(params, "CANDIDATE_STRAIGHT_FLOW_OBJECTIVE_PREFIXES", ["DB", "DK"])}

    obj = full_density[full_density["label_prefix"].astype(str).isin(prefixes)]
    if len(obj) < 2:
        return cands
    obj_xy = obj[["x", "y"]].to_numpy(float)
    cand_xy = cands[["x", "y"]].to_numpy(float)

    nofly_xy = np.empty((0, 2), dtype=float)
    if "is_nofly" in full_density.columns:
        nofly_xy = full_density.loc[full_density["is_nofly"].astype(bool), ["x", "y"]].to_numpy(float)
    tree = None
    if len(nofly_xy):
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(nofly_xy)
        except Exception:
            tree = None

    def segment_min_nofly_distance(a: np.ndarray, b: np.ndarray) -> float:
        if not len(nofly_xy):
            return float("inf")
        seg_len = float(np.hypot(*(b - a)))
        n_samples = max(int(math.ceil(seg_len / max(clearance_m * 0.5, 1.0))) + 1, 2)
        s = np.linspace(0.0, 1.0, n_samples)
        pts = a[None, :] + s[:, None] * (b - a)[None, :]
        if tree is not None:
            return float(tree.query(pts, k=1)[0].min())
        d = np.hypot(pts[:, 0:1] - nofly_xy[None, :, 0], pts[:, 1:2] - nofly_xy[None, :, 1])
        return float(d.min())

    drop = np.zeros(len(cands), dtype=bool)
    for i in range(len(obj_xy)):
        for j in range(i + 1, len(obj_xy)):
            a, b = obj_xy[i], obj_xy[j]
            ab = b - a
            seg_len2 = float(ab @ ab)
            if seg_len2 <= 1.0e-9:
                continue
            t = ((cand_xy - a) @ ab) / seg_len2
            proj = a[None, :] + np.clip(t, 0.0, 1.0)[:, None] * ab[None, :]
            lat = np.hypot(cand_xy[:, 0] - proj[:, 0], cand_xy[:, 1] - proj[:, 1])
            near = (t > 0.0) & (t < 1.0) & (lat <= corridor_m)
            if not near.any():
                continue
            if segment_min_nofly_distance(a, b) < clearance_m:
                continue  # segment blocked -> these candidates are avoidance corners, keep
            drop |= near

    if drop.any():
        print(f"Straight-flow prune ({pool_name}): removed {int(drop.sum())} candidate(s) on clear objective-objective lines")
    return cands[~drop].copy().reset_index(drop=True)


def enforce_final_candidate_separation(major: pd.DataFrame, minor: pd.DataFrame, params: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One global minimum-separation pass over the FINAL pools.

    The area-balanced selectors allow relaxed separations (down to the
    RELAXED_* values) so sparse areas still get coverage, which can leave
    two final candidates uncomfortably close together. This safety net
    enforces a single distance over everything that survived the other
    filters: majors claim their spots first (higher score first), then
    minors must clear both the kept majors and each other.
    """
    if not bool(pget(params, "CANDIDATE_FINAL_MIN_SEPARATION_ENABLE", True)):
        return major, minor
    sep = float(pget(params, "CANDIDATE_FINAL_MIN_SEPARATION_M", 250.0))
    if sep <= 0.0:
        return major, minor

    kept_xy: list[tuple[float, float]] = []

    def filter_pool(pool: pd.DataFrame, pool_name: str) -> pd.DataFrame:
        if pool is None or len(pool) == 0:
            return pool
        ordered = pool.sort_values("candidate_score", ascending=False) if "candidate_score" in pool.columns else pool
        keep_idx = []
        for idx, r in ordered.iterrows():
            if candidate_distance_ok(float(r["x"]), float(r["y"]), kept_xy, sep):
                keep_idx.append(idx)
                kept_xy.append((float(r["x"]), float(r["y"])))
        dropped = len(pool) - len(keep_idx)
        if dropped:
            print(f"Final separation ({pool_name}): removed {dropped} candidate(s) closer than {sep:.0f} m")
        return pool.loc[keep_idx].copy().reset_index(drop=True)

    return filter_pool(major, "major"), filter_pool(minor, "minor")


# ======================================================================
# Obstacle clearance shift (TN too close to no-fly)
# ======================================================================
#
# Downstream (07) every TN is a circle of ~25 m radius whose two
# parallel corridors (diameter 50 m) run tangent to it, so the corridor
# edge reaches node radius + corridor diameter = 75 m from the TN
# center. A candidate closer than CANDIDATE_MIN_OBSTACLE_CLEARANCE_M to
# a no-fly node cannot host that geometry safely. When
# CANDIDATE_OBSTACLE_SHIFT_ENABLE is on, such a candidate is SHIFTED to
# the nearest flyable model node (within CANDIDATE_OBSTACLE_SHIFT_MAX_M)
# that satisfies the clearance -- the traffic statistics stay those of
# the original density hotspot, only the anchor point moves. If no safe
# node exists nearby, the candidate stays in place and is flagged
# obstacle_clearance_ok = False for manual review.

def shift_candidates_away_from_obstacles(cands: pd.DataFrame, full_density: pd.DataFrame, params: dict[str, Any], pool_name: str) -> pd.DataFrame:
    if cands is None or len(cands) == 0:
        return cands
    cands = cands.copy()
    clearance_m = float(pget(params, "CANDIDATE_MIN_OBSTACLE_CLEARANCE_M", 75.0))
    enable = bool(pget(params, "CANDIDATE_OBSTACLE_SHIFT_ENABLE", True))
    shift_max_m = float(pget(params, "CANDIDATE_OBSTACLE_SHIFT_MAX_M", 150.0))

    if "is_nofly" not in full_density.columns or clearance_m <= 0.0:
        cands["obstacle_clearance_m"] = np.inf
        cands["obstacle_clearance_ok"] = True
        cands["shifted_from_node_key"] = ""
        cands["shift_distance_m"] = 0.0
        return cands

    nofly = full_density[full_density["is_nofly"].astype(bool)]
    flyable = full_density[~full_density["is_nofly"].astype(bool)].copy()
    exclude_prefixes = {str(p) for p in pget(params, "CANDIDATE_OBJECTIVE_MERGE_PREFIXES", ["DB", "DK", "FLZ"])} | {"RA"}
    if "label_prefix" in flyable.columns:
        flyable = flyable[~flyable["label_prefix"].astype(str).isin(exclude_prefixes)]

    if len(nofly) == 0:
        cands["obstacle_clearance_m"] = np.inf
        cands["obstacle_clearance_ok"] = True
        cands["shifted_from_node_key"] = ""
        cands["shift_distance_m"] = 0.0
        return cands

    nofly_xy = nofly[["x", "y"]].to_numpy(float)
    fly_xy = flyable[["x", "y"]].to_numpy(float)

    def dist_to_nofly(points: np.ndarray) -> np.ndarray:
        try:
            from scipy.spatial import cKDTree
            d, _ = cKDTree(nofly_xy).query(points, k=1)
            return np.asarray(d, dtype=float)
        except Exception:
            return np.min(np.hypot(points[:, 0:1] - nofly_xy[None, :, 0], points[:, 1:2] - nofly_xy[None, :, 1]), axis=1)

    fly_clear = dist_to_nofly(fly_xy)
    taken_keys = set(cands["node_key"].astype(str))

    clearance_col = np.zeros(len(cands), dtype=float)
    ok_col = np.zeros(len(cands), dtype=bool)
    from_col = [""] * len(cands)
    shift_col = np.zeros(len(cands), dtype=float)

    n_shifted = 0
    n_stuck = 0
    cand_xy = cands[["x", "y"]].to_numpy(float)
    cand_clear = dist_to_nofly(cand_xy)
    for i in range(len(cands)):
        d0 = float(cand_clear[i])
        if d0 >= clearance_m or not enable:
            clearance_col[i] = d0
            ok_col[i] = d0 >= clearance_m
            if d0 < clearance_m:
                n_stuck += 1
            continue
        # Nearest safe flyable node within the shift budget.
        d_move = np.hypot(fly_xy[:, 0] - cand_xy[i, 0], fly_xy[:, 1] - cand_xy[i, 1])
        candidates_mask = (d_move <= shift_max_m) & (fly_clear >= clearance_m)
        best = None
        if candidates_mask.any():
            order = np.flatnonzero(candidates_mask)
            order = order[np.argsort(d_move[order])]
            for j in order:
                nk = str(flyable.iloc[int(j)]["node_key"])
                if nk not in taken_keys:
                    best = int(j)
                    break
        if best is None:
            clearance_col[i] = d0
            ok_col[i] = False
            n_stuck += 1
            print(f"  [WARN] {pool_name} candidate {cands.iloc[i].get('candidate_id', cands.iloc[i]['node_key'])}: "
                  f"only {d0:.0f} m from obstacle, no safe node within {shift_max_m:.0f} m")
            continue
        new_row = flyable.iloc[best]
        old_key = str(cands.iloc[i]["node_key"])
        taken_keys.discard(old_key)
        taken_keys.add(str(new_row["node_key"]))
        from_col[i] = old_key
        shift_col[i] = float(d_move[best])
        for col in ["node_key", "node_id", "x", "y", "z"]:
            if col in cands.columns and col in new_row.index:
                cands.iloc[i, cands.columns.get_loc(col)] = new_row[col]
        clearance_col[i] = float(fly_clear[best])
        ok_col[i] = True
        n_shifted += 1

    cands["obstacle_clearance_m"] = clearance_col
    cands["obstacle_clearance_ok"] = ok_col
    cands["shifted_from_node_key"] = from_col
    cands["shift_distance_m"] = shift_col
    if n_shifted or n_stuck:
        print(f"Obstacle shift ({pool_name}): {n_shifted} candidate(s) shifted to >= {clearance_m:.0f} m clearance, {n_stuck} flagged unresolvable")
    return cands


# ======================================================================
# Major-TN crosscut requirement
# ======================================================================
#
# A major TN marks an area with high traffic density AND strong
# crosscut: routes must actually cross or branch there. A node can
# collect a very high hit_count while carrying nothing but a single
# one-directional bundle of routes (one corridor, no crossing) -- that
# is a busy road segment, not a hub, and it does not deserve a major
# TN's stronger buffer requirement downstream. When
# MAJOR_REQUIRE_CROSSCUT_ENABLE is on, majors whose route-graph degree
# is below MAJOR_CROSSCUT_MIN_GRAPH_DEGREE (degree 2 = one line passing
# straight through; >= 3 = routes branch or cross) or that serve fewer
# than MAJOR_CROSSCUT_MIN_PAIR_COUNT distinct OD pairs are demoted to
# the minor pool instead of dropped -- they still matter as secondary
# connection points.

def demote_non_crosscut_majors(major: pd.DataFrame, params: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    empty = major.iloc[0:0].copy() if major is not None else pd.DataFrame()
    if not bool(pget(params, "MAJOR_REQUIRE_CROSSCUT_ENABLE", True)):
        return major, empty
    if major is None or len(major) == 0:
        return major, empty
    min_degree = int(pget(params, "MAJOR_CROSSCUT_MIN_GRAPH_DEGREE", 3))
    min_pairs = int(pget(params, "MAJOR_CROSSCUT_MIN_PAIR_COUNT", 2))
    degree = major["route_graph_degree"].astype(float) if "route_graph_degree" in major.columns else pd.Series(0.0, index=major.index)
    pairs = major["pair_hit_count"].astype(float) if "pair_hit_count" in major.columns else pd.Series(0.0, index=major.index)
    crosscut = (degree >= float(min_degree)) & (pairs >= float(min_pairs))
    demoted = major[~crosscut].copy().reset_index(drop=True)
    kept = major[crosscut].copy().reset_index(drop=True)
    if len(demoted):
        demoted["candidate_level"] = "minor"
        demoted["candidate_type"] = "demoted_no_crosscut"
        print(f"Crosscut demotion: {len(demoted)} major candidate(s) with one-directional flow demoted to minor")
    return kept, demoted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hit-count DBSCAN clustering of Theta* routes.")
    parser.add_argument("--param-file", type=str, default="params/dbscan.params")
    return parser.parse_args()


# ======================================================================
# Input readers
# ======================================================================

def infer_pair_from_stem(stem: str) -> str:
    m = re.match(r"(.+?)_(forward|backward)_(main|backup)(?:_\d+)?$", stem)
    if m:
        return m.group(1)
    return stem


def infer_direction_from_stem(stem: str) -> str:
    if "_forward_" in stem:
        return "forward"
    if "_backward_" in stem:
        return "backward"
    return "unknown"


def infer_route_type_from_stem(stem: str) -> str:
    if "_main" in stem:
        return "main"
    if "_backup" in stem:
        return "backup"
    return "unknown"


def infer_route_rank_from_stem(stem: str) -> int:
    m = re.search(r"_(\d+)$", stem)
    if m:
        return int(m.group(1))
    return 1


def read_planning_model(model_file: str | Path) -> pd.DataFrame | None:
    model_file = Path(model_file)
    if not model_file.exists():
        return None

    df = pd.read_csv(model_file, sep=r"\s+", engine="python")
    if "node_id" not in df.columns:
        df.insert(0, "node_id", np.arange(len(df), dtype=int))
    if "z" not in df.columns:
        df["z"] = 0.0
    if "label" not in df.columns:
        df["label"] = "NONE"
    if "label_prefix" not in df.columns:
        df["label_prefix"] = "NONE"
    df["label"] = df["label"].fillna("NONE").astype(str)
    df["label_prefix"] = df["label_prefix"].fillna("NONE").astype(str)
    df["node_key"] = df["node_id"].astype(str)
    return df


def read_route_nodes(route_nodes_dir: str | Path) -> pd.DataFrame:
    route_nodes_dir = Path(route_nodes_dir)
    files = sorted(route_nodes_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No route node CSV files found in: {route_nodes_dir}")

    chunks = []
    for f in files:
        df = pd.read_csv(f)
        if len(df) == 0:
            continue

        stem = f.stem
        route_id = stem
        pair = infer_pair_from_stem(stem)
        route_key = stem.replace(pair + "_", "") if stem.startswith(pair + "_") else stem

        if "seq" not in df.columns:
            df.insert(0, "seq", np.arange(len(df), dtype=int))
        if "node_id" not in df.columns:
            df["node_id"] = np.arange(len(df), dtype=int)
        if "z" not in df.columns:
            df["z"] = 0.0
        if "label" not in df.columns:
            df["label"] = "NONE"
        if "label_prefix" not in df.columns:
            df["label_prefix"] = "NONE"

        df["route_id"] = route_id
        df["pair"] = df["pair"] if "pair" in df.columns else pair
        df["route_key"] = route_key
        df["direction"] = df["direction"] if "direction" in df.columns else infer_direction_from_stem(stem)
        df["route_type"] = df["route_type"] if "route_type" in df.columns else infer_route_type_from_stem(stem)
        df["route_rank"] = df["route_rank"] if "route_rank" in df.columns else infer_route_rank_from_stem(stem)

        required = ["x", "y", "node_id"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"[WARN] skip {f}, missing {missing}")
            continue

        df["node_key"] = df["node_id"].astype(str)
        df["label"] = df["label"].fillna("NONE").astype(str)
        df["label_prefix"] = df["label_prefix"].fillna("NONE").astype(str)
        chunks.append(df)

    if not chunks:
        raise RuntimeError("No valid route node files were loaded.")

    out = pd.concat(chunks, ignore_index=True)
    out["route_rank"] = out["route_rank"].astype(int)
    return out


# ======================================================================
# Hit-count computation
# ======================================================================

def route_length_m(g: pd.DataFrame) -> float:
    g = g.sort_values("seq")
    xy = g[["x", "y"]].to_numpy(float)
    if len(xy) < 2:
        return 0.0
    return float(np.sqrt(np.diff(xy[:, 0]) ** 2 + np.diff(xy[:, 1]) ** 2).sum())


def compute_node_hit_count(route_nodes: pd.DataFrame, model_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    hit_count = number of unique routes passing through a node.
    visit_count = raw number of visits from route rows.
    """
    rows = []
    for node_key, g in route_nodes.groupby("node_key"):
        route_ids = sorted(g["route_id"].astype(str).unique())
        pairs = sorted(g["pair"].astype(str).unique())
        directions = sorted(g["direction"].astype(str).unique())
        route_types = sorted(g["route_type"].astype(str).unique())
        r0 = g.iloc[0]

        row = {
            "node_key": str(node_key),
            "node_id": r0.get("node_id", node_key),
            "x": float(g["x"].mean()),
            "y": float(g["y"].mean()),
            "z": float(g["z"].mean()) if "z" in g.columns else 0.0,
            "hit_count": int(len(route_ids)),
            "visit_count": int(len(g)),
            "pair_hit_count": int(len(pairs)),
            "direction_hit_count": int(len(directions)),
            "route_type_hit_count": int(len(route_types)),
            "route_ids": ";".join(route_ids),
            "pairs": ";".join(pairs),
            "directions": ";".join(directions),
            "route_types": ";".join(route_types),
            "label": str(r0.get("label", "NONE")),
            "label_prefix": str(r0.get("label_prefix", "NONE")),
        }

        for c in [
            "slowness", "risk_obstacle", "risk_ra", "risk_total", "flz_support",
            "emergency_risk", "other_terminal_avoidance", "traffic_penalty_factor",
        ]:
            if c in g.columns and pd.api.types.is_numeric_dtype(g[c]):
                row[c] = float(g[c].mean())

        rows.append(row)

    hit_df = pd.DataFrame(rows)

    # Merge model metadata if route CSV lacks it or contains old values.
    if model_df is not None and len(model_df):
        meta_cols = [
            "node_key", "slowness", "risk_obstacle", "risk_ra", "risk_total", "flz_support",
            "emergency_risk", "other_terminal_avoidance", "traffic_penalty_factor", "label", "label_prefix",
        ]
        meta_cols = [c for c in meta_cols if c in model_df.columns]
        meta = model_df[meta_cols].copy()
        hit_df = hit_df.merge(meta, on="node_key", how="left", suffixes=("", "_model"))
        for c in ["label", "label_prefix"]:
            cm = c + "_model"
            if cm in hit_df.columns:
                hit_df[c] = hit_df[cm].where(hit_df[cm].notna(), hit_df[c])
                hit_df = hit_df.drop(columns=[cm])
        for c in [
            "slowness", "risk_obstacle", "risk_ra", "risk_total", "flz_support",
            "emergency_risk", "other_terminal_avoidance", "traffic_penalty_factor",
        ]:
            cm = c + "_model"
            if cm in hit_df.columns:
                hit_df[c] = hit_df[cm].where(hit_df[cm].notna(), hit_df.get(c))
                hit_df = hit_df.drop(columns=[cm])

    hit_df = hit_df.sort_values(["hit_count", "visit_count"], ascending=[False, False]).reset_index(drop=True)
    return hit_df


def edge_key(a: str, b: str) -> str:
    return f"{a}|{b}" if str(a) <= str(b) else f"{b}|{a}"


def compute_edge_hit_count(route_nodes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    edge_rows = []
    for route_id, g in route_nodes.groupby("route_id"):
        g = g.sort_values("seq").reset_index(drop=True)
        if len(g) < 2:
            continue
        for i in range(len(g) - 1):
            a = str(g.at[i, "node_key"])
            b = str(g.at[i + 1, "node_key"])
            if a == b:
                continue
            edge_rows.append({
                "route_id": str(route_id),
                "pair": str(g.at[i, "pair"]),
                "route_key": str(g.at[i, "route_key"]),
                "direction": str(g.at[i, "direction"]),
                "route_type": str(g.at[i, "route_type"]),
                "route_rank": int(g.at[i, "route_rank"]),
                "from_node_key": a,
                "to_node_key": b,
                "edge_key": edge_key(a, b),
                "x1": float(g.at[i, "x"]),
                "y1": float(g.at[i, "y"]),
                "x2": float(g.at[i + 1, "x"]),
                "y2": float(g.at[i + 1, "y"]),
                "edge_distance_m": math.hypot(float(g.at[i + 1, "x"]) - float(g.at[i, "x"]), float(g.at[i + 1, "y"]) - float(g.at[i, "y"])),
            })

    edges = pd.DataFrame(edge_rows)
    if len(edges) == 0:
        return edges, pd.DataFrame()

    rows = []
    for ek, g in edges.groupby("edge_key"):
        route_ids = sorted(g["route_id"].astype(str).unique())
        pairs = sorted(g["pair"].astype(str).unique())
        directions = sorted(g["direction"].astype(str).unique())
        rows.append({
            "edge_key": ek,
            "x1": float(g["x1"].mean()),
            "y1": float(g["y1"].mean()),
            "x2": float(g["x2"].mean()),
            "y2": float(g["y2"].mean()),
            "edge_distance_m": float(g["edge_distance_m"].mean()),
            "edge_hit_count": int(len(route_ids)),
            "edge_visit_count": int(len(g)),
            "pair_hit_count": int(len(pairs)),
            "direction_hit_count": int(len(directions)),
            "route_ids": ";".join(route_ids),
            "pairs": ";".join(pairs),
            "directions": ";".join(directions),
        })
    edge_hit = pd.DataFrame(rows).sort_values(["edge_hit_count", "edge_visit_count"], ascending=[False, False]).reset_index(drop=True)
    return edges, edge_hit


def compute_graph_degree(node_hit: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    deg: dict[str, set[str]] = {str(k): set() for k in node_hit["node_key"].astype(str)}
    if len(edges):
        for _, r in edges.iterrows():
            a = str(r["from_node_key"])
            b = str(r["to_node_key"])
            deg.setdefault(a, set()).add(b)
            deg.setdefault(b, set()).add(a)
    out = node_hit.copy()
    out["route_graph_degree"] = out["node_key"].astype(str).map(lambda k: len(deg.get(k, set()))).astype(int)
    return out


# ======================================================================
# DBSCAN + candidates
# ======================================================================

def run_dbscan(xy: np.ndarray, eps_m: float, min_samples: int, sample_weight: np.ndarray | None = None) -> np.ndarray:
    try:
        from sklearn.cluster import DBSCAN
        db = DBSCAN(eps=float(eps_m), min_samples=int(min_samples))
        if sample_weight is not None:
            return db.fit_predict(xy, sample_weight=sample_weight)
        return db.fit_predict(xy)
    except Exception as exc:
        print(f"[WARN] sklearn DBSCAN unavailable/failed: {exc}")
        print("[WARN] Fallback DBSCAN O(N^2) will be used.")

    n = len(xy)
    labels = np.full(n, -99, dtype=int)
    cluster_id = 0

    def neighbors(i: int) -> np.ndarray:
        d = np.sqrt(((xy - xy[i]) ** 2).sum(axis=1))
        return np.flatnonzero(d <= float(eps_m))

    for i in range(n):
        if labels[i] != -99:
            continue
        nb = neighbors(i)
        if len(nb) < int(min_samples):
            labels[i] = -1
            continue
        labels[i] = cluster_id
        seeds = list(nb)
        j = 0
        while j < len(seeds):
            p = seeds[j]
            if labels[p] == -1:
                labels[p] = cluster_id
            if labels[p] != -99:
                j += 1
                continue
            labels[p] = cluster_id
            nb2 = neighbors(p)
            if len(nb2) >= int(min_samples):
                for q in nb2:
                    if labels[q] in (-99, -1):
                        seeds.append(int(q))
            j += 1
        cluster_id += 1
    labels[labels == -99] = -1
    return labels


def cluster_hit_nodes(node_hit: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    min_hit = int(pget(params, "MIN_HIT_COUNT_FOR_DBSCAN", 2))
    eps_m = float(pget(params, "DBSCAN_EPS_M", 100.0))
    min_samples = int(pget(params, "DBSCAN_MIN_SAMPLES", 4))
    use_weight = bool(pget(params, "DBSCAN_USE_HIT_COUNT_WEIGHT", True))

    out = node_hit.copy()
    out["dbscan_cluster"] = -1
    out["dbscan_input"] = out["hit_count"] >= min_hit

    cand = out[out["dbscan_input"]].copy()
    if len(cand) == 0:
        return out

    xy = cand[["x", "y"]].to_numpy(float)
    weights = cand["hit_count"].to_numpy(float) if use_weight else None
    labels = run_dbscan(xy, eps_m=eps_m, min_samples=min_samples, sample_weight=weights)

    out.loc[cand.index, "dbscan_cluster"] = labels.astype(int)
    return out


def summarize_hit_clusters(clustered: pd.DataFrame) -> pd.DataFrame:
    rows = []
    valid = clustered[clustered["dbscan_cluster"] >= 0].copy()
    for cid, g in valid.groupby("dbscan_cluster"):
        weights = np.maximum(g["hit_count"].to_numpy(float), 1.0)
        cx = float(np.average(g["x"].to_numpy(float), weights=weights))
        cy = float(np.average(g["y"].to_numpy(float), weights=weights))
        route_ids = sorted(set(";".join(g["route_ids"].astype(str)).split(";")) - {""})
        pairs = sorted(set(";".join(g["pairs"].astype(str)).split(";")) - {""})
        directions = sorted(set(";".join(g["directions"].astype(str)).split(";")) - {""})
        rows.append({
            "dbscan_cluster": int(cid),
            "cluster_node_count": int(len(g)),
            "center_x": cx,
            "center_y": cy,
            "max_hit_count": int(g["hit_count"].max()),
            "mean_hit_count": float(g["hit_count"].mean()),
            "sum_hit_count": int(g["hit_count"].sum()),
            "route_count": int(len(route_ids)),
            "pair_count": int(len(pairs)),
            "direction_count": int(len(directions)),
            "max_graph_degree": int(g["route_graph_degree"].max()) if "route_graph_degree" in g.columns else 0,
            "route_ids": ";".join(route_ids),
            "pairs": ";".join(pairs),
            "directions": ";".join(directions),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["max_hit_count", "sum_hit_count", "route_count"], ascending=[False, False, False]).reset_index(drop=True)
    return out


def build_full_area_density_map(model_df: pd.DataFrame | None, node_hit: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """
    Build a whole-area density table.

    If planning model exists, all model nodes are kept. Nodes without route pass-through
    get hit_count = 0. No-fly nodes are explicitly forced to density 0.

    If model is unavailable, fallback to route-hit nodes only.
    """
    nofly_thr = float(pget(params, "NOFLY_SLOWNESS_THRESHOLD", 10.0))

    if model_df is None or len(model_df) == 0:
        full = node_hit.copy()
        full["density_hit_count"] = full["hit_count"].astype(float)
        full["is_nofly"] = False
        return full

    keep_cols = [c for c in [
        "node_key", "node_id", "x", "y", "z", "slowness", "label", "label_prefix",
        "risk_total", "flz_support", "emergency_risk", "other_terminal_avoidance",
        "traffic_penalty_factor",
    ] if c in model_df.columns]
    full = model_df[keep_cols].copy()

    merge_cols = [c for c in [
        "node_key", "hit_count", "visit_count", "pair_hit_count", "direction_hit_count",
        "route_type_hit_count", "route_ids", "pairs", "directions", "route_types",
        "route_graph_degree",
    ] if c in node_hit.columns]
    full = full.merge(node_hit[merge_cols], on="node_key", how="left")

    for c in ["hit_count", "visit_count", "pair_hit_count", "direction_hit_count", "route_type_hit_count", "route_graph_degree"]:
        if c in full.columns:
            full[c] = full[c].fillna(0)
    for c in ["route_ids", "pairs", "directions", "route_types"]:
        if c in full.columns:
            full[c] = full[c].fillna("")

    if "slowness" in full.columns:
        full["is_nofly"] = full["slowness"].to_numpy(float) >= nofly_thr
    else:
        full["is_nofly"] = False

    full["density_hit_count"] = full["hit_count"].astype(float)
    full.loc[full["is_nofly"], "density_hit_count"] = 0.0
    return full


def assign_area_grid(df: pd.DataFrame, grid_size_m: float, xmin: float | None = None, ymin: float | None = None) -> tuple[pd.DataFrame, float, float]:
    out = df.copy()
    if len(out) == 0:
        out["area_ix"] = []
        out["area_iy"] = []
        out["area_id"] = []
        return out, 0.0, 0.0
    if xmin is None:
        xmin = float(out["x"].min())
    if ymin is None:
        ymin = float(out["y"].min())
    g = max(float(grid_size_m), 1.0)
    out["area_ix"] = np.floor((out["x"].to_numpy(float) - xmin) / g).astype(int)
    out["area_iy"] = np.floor((out["y"].to_numpy(float) - ymin) / g).astype(int)
    out["area_id"] = out["area_ix"].astype(str) + "_" + out["area_iy"].astype(str)
    return out, xmin, ymin


def build_candidate_pool(clustered: pd.DataFrame, params: dict[str, Any], near_objective_keys: set[str] = frozenset()) -> pd.DataFrame:
    min_hit = int(pget(params, "CANDIDATE_MIN_HIT_COUNT", 2))
    min_degree = int(pget(params, "CANDIDATE_MIN_GRAPH_DEGREE", 2))
    exclude_prefixes = effective_exclude_prefixes(params)
    include_noise_top = bool(pget(params, "CANDIDATE_INCLUDE_HIGH_HIT_NOISE", True))

    pool = clustered.copy()
    pool = pool[pool["hit_count"] >= min_hit].copy()
    if "route_graph_degree" in pool.columns:
        pool = pool[pool["route_graph_degree"] >= min_degree].copy()
    if "label_prefix" in pool.columns:
        pool = pool[~pool["label_prefix"].astype(str).isin(exclude_prefixes)].copy()
    if near_objective_keys and "node_key" in pool.columns:
        pool = pool[~pool["node_key"].astype(str).isin(near_objective_keys)].copy()

    clustered_part = pool[pool["dbscan_cluster"] >= 0].copy()
    parts = [clustered_part]

    if include_noise_top:
        noise = pool[pool["dbscan_cluster"] < 0].copy()
        if len(noise):
            q = float(pget(params, "HIGH_HIT_NOISE_PERCENTILE", 90.0))
            thr = float(np.percentile(noise["hit_count"].to_numpy(float), q))
            noise = noise[noise["hit_count"] >= thr].copy()
            noise = noise.sort_values(["hit_count", "route_graph_degree", "pair_hit_count"], ascending=[False, False, False])
            noise = noise.head(int(pget(params, "MAX_HIGH_HIT_NOISE_CANDIDATES", 20))).copy()
            parts.append(noise)

    pool = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=clustered.columns)
    if len(pool) == 0:
        return pool

    # score for balanced candidate selection
    hit_norm = pool["hit_count"].to_numpy(float)
    hit_norm = hit_norm / max(hit_norm.max(), 1.0)
    deg_norm = pool["route_graph_degree"].to_numpy(float) if "route_graph_degree" in pool.columns else np.zeros(len(pool), dtype=float)
    deg_norm = deg_norm / max(deg_norm.max(), 1.0) if len(deg_norm) else deg_norm
    pair_norm = pool["pair_hit_count"].to_numpy(float) if "pair_hit_count" in pool.columns else np.zeros(len(pool), dtype=float)
    pair_norm = pair_norm / max(pair_norm.max(), 1.0) if len(pair_norm) else pair_norm
    cluster_bonus = (pool["dbscan_cluster"] >= 0).astype(float).to_numpy()

    w_hit = float(pget(params, "CANDIDATE_SCORE_HIT_WEIGHT", 0.60))
    w_deg = float(pget(params, "CANDIDATE_SCORE_DEGREE_WEIGHT", 0.20))
    w_pair = float(pget(params, "CANDIDATE_SCORE_PAIR_WEIGHT", 0.15))
    w_cluster = float(pget(params, "CANDIDATE_SCORE_CLUSTER_WEIGHT", 0.05))
    pool["candidate_score"] = w_hit * hit_norm + w_deg * deg_norm + w_pair * pair_norm + w_cluster * cluster_bonus
    pool = pool.sort_values(["candidate_score", "hit_count", "route_graph_degree", "pair_hit_count"], ascending=[False, False, False, False]).drop_duplicates(subset=["node_key"]).reset_index(drop=True)
    return pool


def build_area_summary(full_density: pd.DataFrame, candidate_pool: pd.DataFrame, params: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    grid_size = float(pget(params, "AREA_GRID_SIZE_M", 1000.0))
    full_density, xmin, ymin = assign_area_grid(full_density, grid_size)
    candidate_pool, _, _ = assign_area_grid(candidate_pool, grid_size, xmin=xmin, ymin=ymin)

    rows = []
    grouped_full = full_density.groupby("area_id") if len(full_density) else []
    full_map = {aid: g for aid, g in grouped_full}
    pool_map = {aid: g for aid, g in candidate_pool.groupby("area_id")} if len(candidate_pool) else {}
    all_area_ids = sorted(set(list(full_map.keys()) + list(pool_map.keys())))
    for aid in all_area_ids:
        gf = full_map.get(aid, pd.DataFrame())
        gp = pool_map.get(aid, pd.DataFrame())
        area_ix = int(gf["area_ix"].iloc[0]) if len(gf) else int(gp["area_ix"].iloc[0])
        area_iy = int(gf["area_iy"].iloc[0]) if len(gf) else int(gp["area_iy"].iloc[0])
        rows.append({
            "area_id": aid,
            "area_ix": area_ix,
            "area_iy": area_iy,
            "n_model_nodes": int(len(gf)),
            "n_candidate_pool_nodes": int(len(gp)),
            "area_density_sum": float(gf["density_hit_count"].sum()) if len(gf) else 0.0,
            "area_density_mean": float(gf["density_hit_count"].mean()) if len(gf) else 0.0,
            "area_density_max": float(gf["density_hit_count"].max()) if len(gf) else 0.0,
            "area_candidate_hit_max": float(gp["hit_count"].max()) if len(gp) else 0.0,
        })
    area_summary = pd.DataFrame(rows)
    if len(area_summary):
        area_summary["is_active_area"] = (area_summary["n_candidate_pool_nodes"] > 0).astype(int)
        max_sum = max(float(area_summary["area_density_sum"].max()), 1.0)
        area_summary["area_density_norm"] = area_summary["area_density_sum"] / max_sum
    else:
        area_summary["is_active_area"] = []
        area_summary["area_density_norm"] = []
    return full_density, area_summary, xmin, ymin


def candidate_distance_ok(x: float, y: float, selected_xy: list[tuple[float, float]], min_sep_m: float) -> bool:
    if min_sep_m <= 0.0:
        return True
    for sx, sy in selected_xy:
        if math.hypot(float(x) - float(sx), float(y) - float(sy)) < float(min_sep_m):
            return False
    return True


def select_balanced_candidates(full_density: pd.DataFrame, clustered: pd.DataFrame, cluster_summary: pd.DataFrame, params: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Balanced candidate selection by area.

    Goal:
    - produce whole-area density map (0 density for no-fly and unused cells)
    - distribute candidates more evenly across active areas
    - still allow denser candidate allocation in crowded central areas
    """
    near_objective_keys = objective_proximity_node_keys(full_density, params)
    candidate_pool = build_candidate_pool(clustered, params, near_objective_keys)
    if len(candidate_pool) == 0:
        empty = pd.DataFrame()
        return empty, full_density, empty

    full_density, area_summary, xmin, ymin = build_area_summary(full_density, candidate_pool, params)
    candidate_pool, _, _ = assign_area_grid(candidate_pool, float(pget(params, "AREA_GRID_SIZE_M", 1000.0)), xmin=xmin, ymin=ymin)

    base_quota = int(pget(params, "AREA_BASE_CANDIDATES_PER_ACTIVE_AREA", 1))
    extra_max = int(pget(params, "AREA_EXTRA_CANDIDATES_MAX", 2))
    max_per_area = int(pget(params, "AREA_MAX_CANDIDATES_PER_AREA", 4))
    min_sep = float(pget(params, "CANDIDATE_MIN_SEPARATION_M", 300.0))
    relaxed_sep = float(pget(params, "CANDIDATE_RELAXED_MIN_SEPARATION_M", max(min_sep * 0.6, 0.0)))
    allow_relax = bool(pget(params, "ALLOW_RELAXED_AREA_SELECTION", True))

    active = area_summary[area_summary["is_active_area"] > 0].copy()
    if len(active) == 0:
        empty = pd.DataFrame()
        return empty, full_density, area_summary

    active["area_quota"] = (base_quota + np.floor(active["area_density_norm"] * extra_max + 1.0e-9)).astype(int)
    active["area_quota"] = active[["area_quota"]].clip(lower=1).iloc[:,0]
    active["area_quota"] = np.minimum(active["area_quota"], max_per_area)
    active["area_quota"] = np.minimum(active["area_quota"], active["n_candidate_pool_nodes"])
    area_summary = area_summary.merge(active[["area_id", "area_quota"]], on="area_id", how="left")
    area_summary["area_quota"] = area_summary["area_quota"].fillna(0).astype(int)

    # candidate selection in two passes: one guaranteed pass per active area, then extras.
    selected_rows = []
    selected_keys = set()
    selected_xy: list[tuple[float, float]] = []
    selected_count: dict[str, int] = {str(aid): 0 for aid in active["area_id"].astype(str)}

    pool_groups = {aid: g.sort_values(["candidate_score", "hit_count", "route_graph_degree", "pair_hit_count"], ascending=[False, False, False, False]).copy() for aid, g in candidate_pool.groupby("area_id")}

    # pass 1: at least one per active area where possible
    for _, ar in active.sort_values(["area_density_sum", "area_candidate_hit_max"], ascending=[False, False]).iterrows():
        aid = str(ar["area_id"])
        quota = int(ar["area_quota"])
        if quota <= 0 or aid not in pool_groups:
            continue
        g = pool_groups[aid]
        chosen_idx = None
        for idx, r in g.iterrows():
            if str(r["node_key"]) in selected_keys:
                continue
            if candidate_distance_ok(float(r["x"]), float(r["y"]), selected_xy, min_sep):
                chosen_idx = idx
                break
        if chosen_idx is None and allow_relax:
            for idx, r in g.iterrows():
                if str(r["node_key"]) in selected_keys:
                    continue
                if candidate_distance_ok(float(r["x"]), float(r["y"]), selected_xy, relaxed_sep):
                    chosen_idx = idx
                    break
        if chosen_idx is None:
            continue
        r = g.loc[chosen_idx].copy()
        r["candidate_type"] = "area_balanced_primary"
        r["area_quota"] = quota
        r["area_selected_rank"] = 1
        for _col in ["area_density_sum", "area_density_mean", "area_density_max", "area_density_norm", "n_candidate_pool_nodes"]:
            if _col in ar.index:
                r[_col] = ar[_col]
        selected_rows.append(r)
        selected_keys.add(str(r["node_key"]))
        selected_xy.append((float(r["x"]), float(r["y"])))
        selected_count[aid] += 1

    # pass 2: fill extra quotas in dense areas first
    made_progress = True
    while made_progress:
        made_progress = False
        for _, ar in active.sort_values(["area_density_sum", "area_candidate_hit_max"], ascending=[False, False]).iterrows():
            aid = str(ar["area_id"])
            quota = int(ar["area_quota"])
            if selected_count.get(aid, 0) >= quota or aid not in pool_groups:
                continue
            g = pool_groups[aid]
            chosen_idx = None
            for idx, r in g.iterrows():
                if str(r["node_key"]) in selected_keys:
                    continue
                if candidate_distance_ok(float(r["x"]), float(r["y"]), selected_xy, min_sep):
                    chosen_idx = idx
                    break
            if chosen_idx is None and allow_relax:
                for idx, r in g.iterrows():
                    if str(r["node_key"]) in selected_keys:
                        continue
                    if candidate_distance_ok(float(r["x"]), float(r["y"]), selected_xy, relaxed_sep):
                        chosen_idx = idx
                        break
            if chosen_idx is None:
                continue
            r = g.loc[chosen_idx].copy()
            r["candidate_type"] = "area_balanced_extra"
            r["area_quota"] = quota
            r["area_selected_rank"] = int(selected_count.get(aid, 0) + 1)
            for _col in ["area_density_sum", "area_density_mean", "area_density_max", "area_density_norm", "n_candidate_pool_nodes"]:
                if _col in ar.index:
                    r[_col] = ar[_col]
            selected_rows.append(r)
            selected_keys.add(str(r["node_key"]))
            selected_xy.append((float(r["x"]), float(r["y"])))
            selected_count[aid] = int(selected_count.get(aid, 0) + 1)
            made_progress = True

    candidates = pd.DataFrame(selected_rows)
    if len(candidates):
        candidates["density_hit_count"] = candidates["hit_count"].astype(float)
        # Robust fallback for older selected rows: merge area-level fields if missing.
        needed_area_cols = ["area_density_sum", "area_density_mean", "area_density_max", "area_density_norm", "n_candidate_pool_nodes"]
        missing_area_cols = [c for c in needed_area_cols if c not in candidates.columns]
        if missing_area_cols and "area_id" in candidates.columns and len(area_summary):
            merge_cols = ["area_id"] + [c for c in needed_area_cols if c in area_summary.columns]
            candidates = candidates.merge(area_summary[merge_cols].drop_duplicates("area_id"), on="area_id", how="left")
        for _col in needed_area_cols:
            if _col not in candidates.columns:
                candidates[_col] = 0.0
            candidates[_col] = candidates[_col].fillna(0.0)
        candidates = candidates.sort_values(["area_density_sum", "candidate_score", "hit_count"], ascending=[False, False, False]).reset_index(drop=True)
        candidates.insert(0, "candidate_id", [f"IP{i+1:03d}" for i in range(len(candidates))])

    area_selected = pd.DataFrame({"area_id": list(selected_count.keys()), "area_selected_count": list(selected_count.values())})
    area_summary = area_summary.merge(area_selected, on="area_id", how="left")
    area_summary["area_selected_count"] = area_summary["area_selected_count"].fillna(0).astype(int)
    return candidates, full_density, area_summary


def select_minor_candidates(full_density: pd.DataFrame, clustered: pd.DataFrame, major_candidates: pd.DataFrame, params: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select lower-threshold minor candidates.

    Major candidates are the current balanced candidates. Minor candidates are
    secondary connection nodes selected with a lower hit-count threshold, while
    keeping spatial separation from the major candidates and from each other.
    """
    if not bool(pget(params, "ENABLE_MINOR_CANDIDATES", True)):
        return pd.DataFrame(), pd.DataFrame()

    major_min_hit = int(pget(params, "CANDIDATE_MIN_HIT_COUNT", 10))
    min_hit = int(pget(params, "MINOR_CANDIDATE_MIN_HIT_COUNT", max(1, major_min_hit // 2)))
    max_hit_raw = pget(params, "MINOR_CANDIDATE_MAX_HIT_COUNT", None)
    if max_hit_raw is None:
        max_hit = major_min_hit - 1
    else:
        max_hit = int(max_hit_raw)
    min_degree = int(pget(params, "MINOR_CANDIDATE_MIN_GRAPH_DEGREE", pget(params, "CANDIDATE_MIN_GRAPH_DEGREE", 2)))
    exclude_prefixes = effective_exclude_prefixes(params)

    pool = clustered.copy()
    pool = pool[pool["hit_count"].astype(float) >= float(min_hit)].copy()
    if max_hit is not None and max_hit >= min_hit:
        pool = pool[pool["hit_count"].astype(float) <= float(max_hit)].copy()
    if "route_graph_degree" in pool.columns:
        pool = pool[pool["route_graph_degree"].astype(float) >= float(min_degree)].copy()
    if "label_prefix" in pool.columns:
        pool = pool[~pool["label_prefix"].astype(str).isin(exclude_prefixes)].copy()
    near_objective_keys = objective_proximity_node_keys(full_density, params)
    if near_objective_keys and "node_key" in pool.columns:
        pool = pool[~pool["node_key"].astype(str).isin(near_objective_keys)].copy()
    if len(pool) == 0:
        return pd.DataFrame(), pd.DataFrame()

    # Exclude major candidates from minor pool.
    major_keys = set()
    major_xy: list[tuple[float, float]] = []
    if major_candidates is not None and len(major_candidates):
        if "node_key" in major_candidates.columns:
            major_keys = set(major_candidates["node_key"].astype(str))
        for _, r in major_candidates.iterrows():
            major_xy.append((float(r["x"]), float(r["y"])))
    if major_keys and "node_key" in pool.columns:
        pool = pool[~pool["node_key"].astype(str).isin(major_keys)].copy()
    if len(pool) == 0:
        return pd.DataFrame(), pd.DataFrame()

    # Score minor nodes. The score still favors local hit-count, but the lower
    # threshold allows sparse/outer zones to produce candidates.
    hit_norm = pool["hit_count"].to_numpy(float)
    hit_norm = hit_norm / max(hit_norm.max(), 1.0)
    deg_norm = pool["route_graph_degree"].to_numpy(float) if "route_graph_degree" in pool.columns else np.zeros(len(pool), dtype=float)
    deg_norm = deg_norm / max(deg_norm.max(), 1.0) if len(deg_norm) else deg_norm
    pair_norm = pool["pair_hit_count"].to_numpy(float) if "pair_hit_count" in pool.columns else np.zeros(len(pool), dtype=float)
    pair_norm = pair_norm / max(pair_norm.max(), 1.0) if len(pair_norm) else pair_norm
    cluster_bonus = (pool["dbscan_cluster"] >= 0).astype(float).to_numpy() if "dbscan_cluster" in pool.columns else np.zeros(len(pool), dtype=float)

    w_hit = float(pget(params, "MINOR_CANDIDATE_SCORE_HIT_WEIGHT", 0.50))
    w_deg = float(pget(params, "MINOR_CANDIDATE_SCORE_DEGREE_WEIGHT", 0.25))
    w_pair = float(pget(params, "MINOR_CANDIDATE_SCORE_PAIR_WEIGHT", 0.20))
    w_cluster = float(pget(params, "MINOR_CANDIDATE_SCORE_CLUSTER_WEIGHT", 0.05))
    pool["candidate_score"] = w_hit * hit_norm + w_deg * deg_norm + w_pair * pair_norm + w_cluster * cluster_bonus

    # Use the same area grid origin as the full density map.
    grid_size = float(pget(params, "AREA_GRID_SIZE_M", 1000.0))
    xmin = float(full_density["x"].min()) if len(full_density) else float(pool["x"].min())
    ymin = float(full_density["y"].min()) if len(full_density) else float(pool["y"].min())
    pool, _, _ = assign_area_grid(pool, grid_size, xmin=xmin, ymin=ymin)
    full_area, area_summary, _, _ = build_area_summary(full_density, pool, params)

    active = area_summary[area_summary["n_candidate_pool_nodes"] > 0].copy()
    if len(active) == 0:
        return pd.DataFrame(), area_summary

    base_quota = int(pget(params, "MINOR_AREA_BASE_CANDIDATES_PER_ACTIVE_AREA", 1))
    extra_max = int(pget(params, "MINOR_AREA_EXTRA_CANDIDATES_MAX", 1))
    max_per_area = int(pget(params, "MINOR_AREA_MAX_CANDIDATES_PER_AREA", 2))
    min_sep = float(pget(params, "MINOR_CANDIDATE_MIN_SEPARATION_M", 250.0))
    min_sep_major = float(pget(params, "MINOR_CANDIDATE_MIN_SEPARATION_FROM_MAJOR_M", 250.0))
    relaxed_sep = float(pget(params, "MINOR_CANDIDATE_RELAXED_MIN_SEPARATION_M", max(min_sep * 0.6, 0.0)))
    relaxed_sep_major = float(pget(params, "MINOR_CANDIDATE_RELAXED_MIN_SEPARATION_FROM_MAJOR_M", max(min_sep_major * 0.6, 0.0)))
    allow_relax = bool(pget(params, "ALLOW_RELAXED_MINOR_SELECTION", True))

    active["area_quota"] = (base_quota + np.floor(active["area_density_norm"] * extra_max + 1.0e-9)).astype(int)
    active["area_quota"] = active[["area_quota"]].clip(lower=1).iloc[:, 0]
    active["area_quota"] = np.minimum(active["area_quota"], max_per_area)
    active["area_quota"] = np.minimum(active["area_quota"], active["n_candidate_pool_nodes"])
    area_summary = area_summary.merge(active[["area_id", "area_quota"]], on="area_id", how="left")
    area_summary["area_quota"] = area_summary["area_quota"].fillna(0).astype(int)

    selected_rows = []
    selected_keys: set[str] = set()
    selected_xy: list[tuple[float, float]] = []
    selected_count: dict[str, int] = {str(aid): 0 for aid in active["area_id"].astype(str)}
    pool_groups = {
        aid: g.sort_values(["candidate_score", "hit_count", "route_graph_degree", "pair_hit_count"], ascending=[False, False, False, False]).copy()
        for aid, g in pool.groupby("area_id")
    }

    def pick_from_area(g: pd.DataFrame, sep: float, sep_major: float):
        for idx, r in g.iterrows():
            nk = str(r["node_key"])
            if nk in selected_keys:
                continue
            if not candidate_distance_ok(float(r["x"]), float(r["y"]), major_xy, sep_major):
                continue
            if not candidate_distance_ok(float(r["x"]), float(r["y"]), selected_xy, sep):
                continue
            return idx
        return None

    made_progress = True
    while made_progress:
        made_progress = False
        for _, ar in active.sort_values(["area_density_sum", "area_candidate_hit_max"], ascending=[False, False]).iterrows():
            aid = str(ar["area_id"])
            quota = int(ar["area_quota"])
            if selected_count.get(aid, 0) >= quota or aid not in pool_groups:
                continue
            g = pool_groups[aid]
            chosen_idx = pick_from_area(g, min_sep, min_sep_major)
            if chosen_idx is None and allow_relax:
                chosen_idx = pick_from_area(g, relaxed_sep, relaxed_sep_major)
            if chosen_idx is None:
                continue
            r = g.loc[chosen_idx].copy()
            r["candidate_type"] = "minor_area_balanced"
            r["candidate_level"] = "minor"
            r["area_quota"] = quota
            r["area_selected_rank"] = int(selected_count.get(aid, 0) + 1)
            for _col in ["area_density_sum", "area_density_mean", "area_density_max", "area_density_norm", "n_candidate_pool_nodes"]:
                if _col in ar.index:
                    r[_col] = ar[_col]
            selected_rows.append(r)
            selected_keys.add(str(r["node_key"]))
            selected_xy.append((float(r["x"]), float(r["y"])))
            selected_count[aid] = int(selected_count.get(aid, 0) + 1)
            made_progress = True

    minor = pd.DataFrame(selected_rows)
    if len(minor):
        minor["density_hit_count"] = minor["hit_count"].astype(float)
        for _col in ["area_density_sum", "area_density_mean", "area_density_max", "area_density_norm", "n_candidate_pool_nodes"]:
            if _col not in minor.columns:
                minor[_col] = 0.0
            minor[_col] = minor[_col].fillna(0.0)
        minor = minor.sort_values(["area_density_sum", "candidate_score", "hit_count"], ascending=[False, False, False]).reset_index(drop=True)
        minor.insert(0, "candidate_id", [f"MIN{i+1:03d}" for i in range(len(minor))])

    area_selected = pd.DataFrame({"area_id": list(selected_count.keys()), "minor_area_selected_count": list(selected_count.values())})
    area_summary = area_summary.merge(area_selected, on="area_id", how="left")
    area_summary["minor_area_selected_count"] = area_summary["minor_area_selected_count"].fillna(0).astype(int)
    return minor, area_summary


def combine_major_minor_candidates(major: pd.DataFrame, minor: pd.DataFrame) -> pd.DataFrame:
    major = major.copy() if major is not None else pd.DataFrame()
    minor = minor.copy() if minor is not None else pd.DataFrame()
    if len(major):
        if "candidate_level" not in major.columns:
            major["candidate_level"] = "major"
        major["candidate_type"] = major.get("candidate_type", "area_balanced_major")
        major["old_candidate_id"] = major.get("candidate_id", "")
        major["candidate_id"] = [f"MAJ{i+1:03d}" for i in range(len(major))]
    if len(minor):
        if "candidate_level" not in minor.columns:
            minor["candidate_level"] = "minor"
    out = pd.concat([major, minor], ignore_index=True, sort=False)
    if len(out):
        out = out.drop_duplicates(subset=["node_key"], keep="first").reset_index(drop=True)
    return out


def select_candidates(clustered: pd.DataFrame, cluster_summary: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    top_k_per_cluster = int(pget(params, "CANDIDATE_TOP_K_PER_CLUSTER", 1))
    min_hit = int(pget(params, "CANDIDATE_MIN_HIT_COUNT", 2))
    min_degree = int(pget(params, "CANDIDATE_MIN_GRAPH_DEGREE", 2))
    exclude_prefixes = effective_exclude_prefixes(params)
    include_noise_top = bool(pget(params, "CANDIDATE_INCLUDE_HIGH_HIT_NOISE", True))

    rows = []
    valid = clustered[(clustered["dbscan_cluster"] >= 0) & (clustered["hit_count"] >= min_hit)].copy()

    if len(valid):
        for cid, g in valid.groupby("dbscan_cluster"):
            g = g.copy()
            if "label_prefix" in g.columns:
                g = g[~g["label_prefix"].astype(str).isin(exclude_prefixes)]
            if "route_graph_degree" in g.columns:
                g = g[g["route_graph_degree"] >= min_degree]
            if len(g) == 0:
                # fallback: keep strongest non-terminal even if degree is low
                g = valid[valid["dbscan_cluster"] == cid].copy()
                if "label_prefix" in g.columns:
                    g = g[~g["label_prefix"].astype(str).isin(exclude_prefixes)]
            if len(g) == 0:
                continue
            g = g.sort_values(["hit_count", "route_graph_degree", "pair_hit_count"], ascending=[False, False, False]).head(top_k_per_cluster)
            for _, r in g.iterrows():
                rr = r.to_dict()
                rr["candidate_type"] = "cluster_peak_hit_count"
                rows.append(rr)

    if include_noise_top:
        # Add high-hit nodes that DBSCAN labels as noise, useful for isolated bottlenecks.
        q = float(pget(params, "HIGH_HIT_NOISE_PERCENTILE", 90.0))
        noise = clustered[(clustered["dbscan_cluster"] < 0) & (clustered["hit_count"] >= min_hit)].copy()
        if len(noise):
            threshold = float(np.percentile(noise["hit_count"].to_numpy(float), q))
            noise = noise[noise["hit_count"] >= threshold]
            if "label_prefix" in noise.columns:
                noise = noise[~noise["label_prefix"].astype(str).isin(exclude_prefixes)]
            for _, r in noise.sort_values(["hit_count", "route_graph_degree"], ascending=[False, False]).head(int(pget(params, "MAX_HIGH_HIT_NOISE_CANDIDATES", 20))).iterrows():
                rr = r.to_dict()
                rr["candidate_type"] = "isolated_high_hit_bottleneck"
                rows.append(rr)

    out = pd.DataFrame(rows)
    if len(out):
        # Drop duplicates if a node was selected twice.
        out = out.drop_duplicates(subset=["node_key"]).copy()
        out = out.sort_values(["hit_count", "route_graph_degree", "pair_hit_count"], ascending=[False, False, False]).reset_index(drop=True)
        out.insert(0, "candidate_id", [f"IP{i+1:03d}" for i in range(len(out))])
    return out


def compute_route_cluster_membership(route_nodes: pd.DataFrame, clustered: pd.DataFrame) -> pd.DataFrame:
    mapping = clustered[["node_key", "dbscan_cluster", "hit_count"]].copy()
    rn = route_nodes.merge(mapping, on="node_key", how="left")
    rn["dbscan_cluster"] = rn["dbscan_cluster"].fillna(-1).astype(int)

    rows = []
    for route_id, g in rn.groupby("route_id"):
        g = g.sort_values("seq")
        total = max(len(g), 1)
        clusters = []
        for cid, cg in g[g["dbscan_cluster"] >= 0].groupby("dbscan_cluster"):
            clusters.append((int(cg["seq"].median()), int(cid), len(cg) / total))
        clusters_sorted = sorted(clusters)
        rows.append({
            "route_id": route_id,
            "pair": str(g["pair"].iloc[0]),
            "route_key": str(g["route_key"].iloc[0]),
            "direction": str(g["direction"].iloc[0]),
            "route_type": str(g["route_type"].iloc[0]),
            "route_rank": int(g["route_rank"].iloc[0]),
            "route_length_m": route_length_m(g),
            "n_nodes": int(len(g)),
            "n_clusters_touched": int(len(set(c for _, c, _ in clusters_sorted))),
            "cluster_sequence": ";".join(str(c) for _, c, _ in clusters_sorted),
            "cluster_coverage_sequence": ";".join(f"{c}:{cov:.3f}" for _, c, cov in clusters_sorted),
        })
    return pd.DataFrame(rows)


# ======================================================================
# Master-plan input table (feeds 05_run_master_plan.py directly)
# ======================================================================

def build_master_node_table(full_density: pd.DataFrame, candidates: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Whole-map node table for 05_run_master_plan.py's INPUT_NODE_FILE.

    src/routeplan_PSO_ACO.py needs every raw model node, not just the selected
    candidates: DB/DK/FLZ objective nodes are matched by label_prefix over
    the whole table, and build_master_edges() builds a KD-tree of ALL
    no-fly nodes (model.nofly_mask over the full CSV) to keep master edges
    clear of obstacles/RA -- a candidates-only file would have almost no
    no-fly nodes in it and silently break that clearance check. full_density
    already is that whole-map table (every model node, flyable or not), so
    this just layers the candidate flags on top of it.

    candidate_type here must be the literal "major"/"minor" string --
    that's what build_master_nodes() in src/routeplan_PSO_ACO.py switches on to assign
    TN_major/TN_minor roles. It is NOT the same as this script's own
    candidate_type column (the selection-method label, e.g.
    "area_balanced_primary"), which is kept here as
    candidate_selection_method for traceability.
    """
    out = full_density.copy()
    out["model_index"] = np.arange(len(out), dtype=int)
    out["route_hit_count"] = out["hit_count"] if "hit_count" in out.columns else out.get("density_hit_count", 0.0)

    if candidates is not None and len(candidates) and "node_key" in candidates.columns:
        cand = candidates[["node_key", "candidate_id"]].copy()
        cand["candidate_type"] = candidates["candidate_level"].astype(str) if "candidate_level" in candidates.columns else "major"
        cand["candidate_selection_method"] = candidates["candidate_type"].astype(str) if "candidate_type" in candidates.columns else ""
        cand = cand.drop_duplicates(subset=["node_key"])
        out = out.merge(cand, on="node_key", how="left")
    else:
        out["candidate_id"] = np.nan
        out["candidate_type"] = np.nan
        out["candidate_selection_method"] = np.nan

    out["is_candidate"] = out["candidate_id"].notna()
    out["candidate_id"] = out["candidate_id"].fillna("")
    out["candidate_type"] = out["candidate_type"].fillna("")
    out["candidate_selection_method"] = out["candidate_selection_method"].fillna("")
    return out


# ======================================================================
# Plots
# ======================================================================

def setup_plot_style(params: dict[str, Any]) -> None:
    plt.rcParams["font.family"] = str(pget(params, "PLOT_FONT_FAMILY", "DejaVu Serif"))
    plt.rcParams["axes.titlesize"] = int(pget(params, "PLOT_TITLE_SIZE", 16))
    plt.rcParams["axes.labelsize"] = int(pget(params, "PLOT_LABEL_SIZE", 12))
    plt.rcParams["legend.fontsize"] = int(pget(params, "PLOT_LEGEND_SIZE", 9))


def plot_base(ax, model_df: pd.DataFrame | None, params: dict[str, Any]) -> None:
    if not bool(pget(params, "PLOT_SHOW_BASE_MODEL", False)):
        return
    if model_df is None or len(model_df) == 0:
        return
    nofly_thr = float(pget(params, "NOFLY_SLOWNESS_THRESHOLD", 10.0))
    base_size = float(pget(params, "BASE_NODE_SIZE", 2.0))
    if "slowness" in model_df.columns:
        nofly = model_df["slowness"].to_numpy(float) >= nofly_thr
        ax.scatter(model_df.loc[~nofly, "x"], model_df.loc[~nofly, "y"], s=base_size, c="lightgray", alpha=0.25, linewidths=0, label="flyable", zorder=1)
        ax.scatter(model_df.loc[nofly, "x"], model_df.loc[nofly, "y"], s=base_size * 1.5, c="black", alpha=0.8, linewidths=0, label="no-fly", zorder=2)

    markers = {
        "DB": ("^", "blue", 90),
        "DK": ("s", "green", 80),
        "FLZ": ("*", "orange", 150),
        "RA": ("X", "purple", 100),
    }
    for prefix, (marker, color, size) in markers.items():
        sub = model_df[model_df["label_prefix"].astype(str) == prefix]
        if len(sub) == 0:
            continue
        ax.scatter(sub["x"], sub["y"], marker=marker, s=size, c=color, edgecolors="white", linewidths=0.8, label=prefix, zorder=20)
        for _, r in sub.iterrows():
            ax.text(r["x"], r["y"], str(r.get("label", prefix)), fontsize=8, color=color, weight="bold", zorder=21)


def overlay_flz_markers(ax, model_df: pd.DataFrame | None, params: dict[str, Any]) -> None:
    """Draw FLZ node positions as reference rings, independent of PLOT_SHOW_BASE_MODEL.

    FLZ nodes are frequently forced into every route (02's FLZ buffer
    requirement), so they can become the map's top traffic-density hotspots
    whether or not CONSIDER_FLZ_AS_TRAFFIC_NODE lets them become candidates.
    This lightweight overlay makes that visible on the density/candidate
    figures even when PLOT_SHOW_BASE_MODEL is off (its default).
    """
    if not bool(pget(params, "PLOT_MARK_FLZ_NODES", True)):
        return
    if model_df is None or len(model_df) == 0 or "label_prefix" not in model_df.columns:
        return
    sub = model_df[model_df["label_prefix"].astype(str).isin(flz_prefixes(params))]
    if len(sub) == 0:
        return
    ax.scatter(
        sub["x"], sub["y"],
        marker="o",
        s=float(pget(params, "FLZ_MARKER_RING_SIZE", 260.0)),
        facecolors="none",
        edgecolors=str(pget(params, "FLZ_MARKER_RING_COLOR", "orange")),
        linewidths=float(pget(params, "FLZ_MARKER_RING_LINEWIDTH", 1.4)),
        label="FLZ node",
        zorder=25,
    )


def overlay_db_dk_markers(ax, model_df: pd.DataFrame | None, params: dict[str, Any]) -> None:
    """Draw DB/DK objective node positions, independent of PLOT_SHOW_BASE_MODEL.

    DB/DK nodes are the route endpoints every Theta* route starts or ends
    at, so the hit-count densities and clusters form around and between
    them. Showing them on the density/cluster/candidate figures gives
    that spatial context even when the full base model overlay is off
    (its default).
    """
    if not bool(pget(params, "PLOT_MARK_DB_DK_NODES", True)):
        return
    if model_df is None or len(model_df) == 0 or "label_prefix" not in model_df.columns:
        return
    markers = {
        "DB": ("^", "blue", 90),
        "DK": ("s", "green", 80),
    }
    for prefix, (marker, color, size) in markers.items():
        sub = model_df[model_df["label_prefix"].astype(str) == prefix]
        if len(sub) == 0:
            continue
        ax.scatter(sub["x"], sub["y"], marker=marker, s=size, c=color, edgecolors="white", linewidths=0.8, label=prefix, zorder=26)
        for _, r in sub.iterrows():
            ax.text(r["x"], r["y"], str(r.get("label", prefix)), fontsize=8, color=color, weight="bold", zorder=27)


def finalize_map(ax, title: str) -> None:
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("X coordinate (m)")
    ax.set_ylabel("Y coordinate (m)")
    ax.set_aspect("equal", adjustable="box")
    add_map_rule(ax)
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    valid = [(h, l) for h, l in zip(handles, labels) if str(l).strip() and str(l) != "_nolegend_"]
    if valid:
        ax.legend([h for h, _ in valid], [l for _, l in valid], loc="upper right", frameon=True)


def hit_sizes(hit: pd.Series, params: dict[str, Any]) -> np.ndarray:
    base = float(pget(params, "HIT_MARKER_BASE_SIZE", 16.0))
    scale = float(pget(params, "HIT_MARKER_SCALE", 18.0))
    max_size = float(pget(params, "HIT_MARKER_MAX_SIZE", 180.0))
    return np.clip(base + scale * np.sqrt(hit.to_numpy(float)), base, max_size)


def density_points_for_plot(full_density: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Filter density points for plotting only."""
    out = full_density.copy()
    if len(out) == 0:
        return out
    if bool(pget(params, "DENSITY_PLOT_REMOVE_NOFLY", True)) and "is_nofly" in out.columns:
        out = out[~out["is_nofly"].astype(bool)].copy()
    if bool(pget(params, "DENSITY_PLOT_REMOVE_ZERO", True)) and "density_hit_count" in out.columns:
        out = out[out["density_hit_count"].astype(float) > 0.0].copy()
    return out


def make_density_cmap_norm(values: pd.Series | np.ndarray, params: dict[str, Any]):
    """
    Balanced color scaling for route-density maps.

    A few extremely high values can compress the colorbar and make most points look
    identical. The default uses a percentile vmax and colors values above that
    threshold with COLORBAR_OVER_COLOR, e.g. purple.
    """
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        vals = np.array([0.0, 1.0])

    cmap_name = str(pget(params, "HEATMAP_CMAP", "magma"))
    if bool(pget(params, "COLORBAR_REVERSE", False)) and not cmap_name.endswith("_r"):
        cmap_name = cmap_name + "_r"
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_over(str(pget(params, "COLORBAR_OVER_COLOR", "purple")))
    cmap.set_under(str(pget(params, "COLORBAR_UNDER_COLOR", "black")))

    vmin_mode = str(pget(params, "COLORBAR_VMIN_MODE", "min_positive"))
    if vmin_mode == "zero":
        vmin = 0.0
    elif vmin_mode == "percentile":
        vmin = float(np.percentile(vals, float(pget(params, "COLORBAR_PERCENTILE_MIN", 1.0))))
    else:
        positive = vals[vals > 0.0]
        vmin = float(positive.min()) if len(positive) else 0.0

    mode = str(pget(params, "COLORBAR_BALANCE_MODE", "percentile"))
    if mode == "fixed":
        vmax = float(pget(params, "COLORBAR_FIXED_MAX", max(vals.max(), vmin + 1.0)))
    elif mode == "max":
        vmax = float(vals.max())
    else:
        pct = float(pget(params, "COLORBAR_PERCENTILE_MAX", 95.0))
        vmax = float(np.percentile(vals, pct))

    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=False)
    return cmap, norm, vmin, vmax


def plot_candidate_symbols(ax, candidates: pd.DataFrame, cmap, norm, params: dict[str, Any], *, add_colorbar: bool = False):
    """Plot major and minor candidates using different symbols."""
    if candidates is None or len(candidates) == 0:
        return None
    plot_df = candidates.copy()
    if bool(pget(params, "DENSITY_PLOT_REMOVE_ZERO", True)) and "density_hit_count" in plot_df.columns:
        plot_df = plot_df[plot_df["density_hit_count"].astype(float) > 0.0].copy()
    if len(plot_df) == 0:
        return None

    if "candidate_level" not in plot_df.columns:
        plot_df["candidate_level"] = "major"
    if "is_flz" not in plot_df.columns:
        plot_df["is_flz"] = False

    flz_edge_color = str(pget(params, "FLZ_CANDIDATE_EDGE_COLOR", "orange"))
    flz_edge_lw_bonus = float(pget(params, "FLZ_CANDIDATE_EDGE_LINEWIDTH_BONUS", 1.2))

    sc_ref = None
    major = plot_df[plot_df["candidate_level"].astype(str).str.lower() == "major"].copy()
    minor = plot_df[plot_df["candidate_level"].astype(str).str.lower() == "minor"].copy()

    def _split_flz(g: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        is_flz = g["is_flz"].astype(bool)
        return g[~is_flz], g[is_flz]

    if len(major):
        major_normal, major_flz = _split_flz(major)
        if len(major_normal):
            sc_ref = ax.scatter(
                major_normal["x"], major_normal["y"],
                s=float(pget(params, "CANDIDATE_STAR_SIZE", 220.0)),
                c=major_normal["density_hit_count"],
                marker=str(pget(params, "MAJOR_CANDIDATE_MARKER", "*")),
                cmap=cmap,
                norm=norm,
                edgecolors="black",
                linewidths=float(pget(params, "CANDIDATE_EDGE_LINEWIDTH", 1.0)),
                label="major candidate",
                zorder=32,
            )
        if len(major_flz):
            sc_major_flz = ax.scatter(
                major_flz["x"], major_flz["y"],
                s=float(pget(params, "CANDIDATE_STAR_SIZE", 220.0)),
                c=major_flz["density_hit_count"],
                marker=str(pget(params, "MAJOR_CANDIDATE_MARKER", "*")),
                cmap=cmap,
                norm=norm,
                edgecolors=flz_edge_color,
                linewidths=float(pget(params, "CANDIDATE_EDGE_LINEWIDTH", 1.0)) + flz_edge_lw_bonus,
                label="major candidate (FLZ)",
                zorder=33,
            )
            if sc_ref is None:
                sc_ref = sc_major_flz
    if len(minor):
        minor_normal, minor_flz = _split_flz(minor)
        if len(minor_normal):
            sc_minor = ax.scatter(
                minor_normal["x"], minor_normal["y"],
                s=float(pget(params, "MINOR_CANDIDATE_MARKER_SIZE", 95.0)),
                c=minor_normal["density_hit_count"],
                marker=str(pget(params, "MINOR_CANDIDATE_MARKER", "D")),
                cmap=cmap,
                norm=norm,
                edgecolors="black",
                linewidths=float(pget(params, "MINOR_CANDIDATE_EDGE_LINEWIDTH", 0.85)),
                label="minor candidate",
                zorder=31,
            )
            if sc_ref is None:
                sc_ref = sc_minor
        if len(minor_flz):
            sc_minor_flz = ax.scatter(
                minor_flz["x"], minor_flz["y"],
                s=float(pget(params, "MINOR_CANDIDATE_MARKER_SIZE", 95.0)),
                c=minor_flz["density_hit_count"],
                marker=str(pget(params, "MINOR_CANDIDATE_MARKER", "D")),
                cmap=cmap,
                norm=norm,
                edgecolors=flz_edge_color,
                linewidths=float(pget(params, "MINOR_CANDIDATE_EDGE_LINEWIDTH", 0.85)) + flz_edge_lw_bonus,
                label="minor candidate (FLZ)",
                zorder=32,
            )
            if sc_ref is None:
                sc_ref = sc_minor_flz

    if bool(pget(params, "PLOT_CANDIDATE_LABELS", False)):
        for _, r in plot_df.iterrows():
            ax.text(
                r["x"], r["y"],
                str(r.get("candidate_id", "")),
                color="black",
                fontsize=7,
                weight="bold",
                ha="center",
                va="bottom",
                zorder=35,
            )
    return sc_ref


def plot_node_hit_density(model_df: pd.DataFrame | None, full_density: pd.DataFrame, candidates: pd.DataFrame, out: Path, params: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(11, 10))

    plot_density = density_points_for_plot(full_density, params)
    cmap, norm, cmin, cmax = make_density_cmap_norm(plot_density["density_hit_count"] if len(plot_density) else full_density.get("density_hit_count", pd.Series([0.0])), params)

    normal_size = float(pget(params, "NORMAL_NODE_SIZE", 8.0))
    candidate_size = float(pget(params, "CANDIDATE_STAR_SIZE", 220.0))
    edge_lw = float(pget(params, "SYMBOL_EDGE_LINEWIDTH", 0.5))
    candidate_edge_lw = float(pget(params, "CANDIDATE_EDGE_LINEWIDTH", 1.0))
    alpha = float(pget(params, "HEATMAP_ALPHA", 1.0))

    raw_max = float(full_density["density_hit_count"].max()) if len(full_density) else 0.0
    raw_mean = float(full_density["density_hit_count"].mean()) if len(full_density) else 0.0

    # Candidate nodes should not be plotted twice as normal circles.
    candidate_keys = set()
    if candidates is not None and len(candidates):
        if "node_key" in candidates.columns:
            candidate_keys = set(candidates["node_key"].astype(str))
        elif "node_id" in candidates.columns:
            candidate_keys = set(candidates["node_id"].astype(str))

    plot_nodes = plot_density.copy()
    if candidate_keys:
        if "node_key" in plot_nodes.columns:
            plot_nodes = plot_nodes[~plot_nodes["node_key"].astype(str).isin(candidate_keys)]
        elif "node_id" in plot_nodes.columns:
            plot_nodes = plot_nodes[~plot_nodes["node_id"].astype(str).isin(candidate_keys)]

    sc = None
    if len(plot_nodes):
        sc = ax.scatter(
            plot_nodes["x"], plot_nodes["y"],
            s=normal_size,
            c=plot_nodes["density_hit_count"],
            marker="o",
            cmap=cmap,
            norm=norm,
            edgecolors="black",
            linewidths=edge_lw,
            alpha=alpha,
            label="route-density node",
            zorder=10,
        )

    overlay_flz_markers(ax, model_df, params)
    overlay_db_dk_markers(ax, model_df, params)

    sc_candidates = plot_candidate_symbols(ax, candidates, cmap, norm, params)
    if sc is None and sc_candidates is not None:
        sc = sc_candidates

    if sc is not None:
        cb = fig.colorbar(sc, ax=ax, shrink=0.74, pad=0.02, extend="max")
        cb.set_label("Route-density hit-count; high outliers shown as purple")

    title = f"Whole-area route density map (plotted positive flyable nodes; mean={raw_mean:.2f}, raw max={raw_max:.0f}, color max={cmax:.1f})"
    finalize_map(ax, title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(pget(params, "PLOT_DPI", 250)))
    plt.close(fig)


def plot_edge_hit_density(model_df: pd.DataFrame | None, edge_hit: pd.DataFrame, out: Path, params: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(11, 10))
    # Edge-density map only. No flyable/obstacle background is plotted for speed.
    if len(edge_hit):
        plot_edges = edge_hit.copy()
        if bool(pget(params, "EDGE_DENSITY_PLOT_REMOVE_ZERO", True)):
            plot_edges = plot_edges[plot_edges["edge_hit_count"].astype(float) > 0.0].copy()

        if len(plot_edges):
            values = plot_edges["edge_hit_count"].astype(float).to_numpy()
            cmap, norm, cmin, cmax = make_density_cmap_norm(values, params)

            segments = [
                [(float(r["x1"]), float(r["y1"])), (float(r["x2"]), float(r["y2"]))]
                for _, r in plot_edges.iterrows()
            ]

            raw_max = max(float(values.max()), 1.0)
            lw_min = float(pget(params, "EDGE_LINEWIDTH_MIN", 2.50))
            lw_max = float(pget(params, "EDGE_LINEWIDTH_MAX", 9.00))
            edge_alpha = float(pget(params, "EDGE_ALPHA", 1.00))
            lw_power = float(pget(params, "EDGE_LINEWIDTH_POWER", 0.50))
            linewidths = lw_min + (lw_max - lw_min) * np.power(values / raw_max, lw_power)

            lc = mcoll.LineCollection(
                segments,
                cmap=cmap,
                norm=norm,
                linewidths=linewidths,
                alpha=edge_alpha,
                linestyles="solid",
                zorder=12,
            )
            # Force continuous-looking solid edges. Rounded caps/joins remove the
            # small visual gaps that can make grid-edge collections look dashed.
            try:
                lc.set_capstyle(str(pget(params, "EDGE_CAPSTYLE", "round")))
                lc.set_joinstyle(str(pget(params, "EDGE_JOINSTYLE", "round")))
                lc.set_antialiased(True)
            except Exception:
                pass
            lc.set_array(values)
            ax.add_collection(lc)

            cb = fig.colorbar(lc, ax=ax, shrink=0.74, pad=0.02, extend="max")
            cb.set_label("Edge hit-count: number of routes using edge")

            # Make sure map limits include all edges.
            xs = np.r_[plot_edges["x1"].to_numpy(float), plot_edges["x2"].to_numpy(float)]
            ys = np.r_[plot_edges["y1"].to_numpy(float), plot_edges["y2"].to_numpy(float)]
            if len(xs) and len(ys):
                pad_x = max((xs.max() - xs.min()) * 0.03, 1.0)
                pad_y = max((ys.max() - ys.min()) * 0.03, 1.0)
                ax.set_xlim(xs.min() - pad_x, xs.max() + pad_x)
                ax.set_ylim(ys.min() - pad_y, ys.max() + pad_y)

            title = f"Edge hit-count density map (raw max={raw_max:.0f}, color max={cmax:.1f})"
        else:
            title = "Edge hit-count density map (no positive edge hit-count)"
    else:
        title = "Edge hit-count density map (no edge data)"

    overlay_db_dk_markers(ax, model_df, params)
    finalize_map(ax, title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(pget(params, "PLOT_DPI", 250)))
    plt.close(fig)


def plot_dbscan_clusters(model_df: pd.DataFrame | None, clustered: pd.DataFrame, summary: pd.DataFrame, out: Path, params: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(11, 10))
    plot_base(ax, model_df, params)
    overlay_flz_markers(ax, model_df, params)
    overlay_db_dk_markers(ax, model_df, params)
    noise = clustered[clustered["dbscan_cluster"] < 0]
    valid = clustered[clustered["dbscan_cluster"] >= 0]
    if len(noise):
        ax.scatter(noise["x"], noise["y"], s=10, c="gray", alpha=0.20, linewidths=0, label="low/noise hit nodes", zorder=5)
    if len(valid):
        sc = ax.scatter(
            valid["x"], valid["y"],
            s=hit_sizes(valid["hit_count"], params),
            c=valid["dbscan_cluster"],
            alpha=0.9,
            linewidths=0,
            label="DBSCAN hit cluster",
            zorder=11,
        )
        cb = fig.colorbar(sc, ax=ax, shrink=0.74, pad=0.02)
        cb.set_label("DBSCAN cluster ID")
    if summary is not None and len(summary):
        for _, r in summary.iterrows():
            ax.text(r["center_x"], r["center_y"], f"C{int(r['dbscan_cluster'])}\nH={int(r['max_hit_count'])}", ha="center", va="center", fontsize=8, weight="bold", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", alpha=0.75), zorder=30)
    finalize_map(ax, "DBSCAN clusters of high hit-count route nodes")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(pget(params, "PLOT_DPI", 250)))
    plt.close(fig)


def plot_candidates(model_df: pd.DataFrame | None, full_density: pd.DataFrame, candidates: pd.DataFrame, area_summary: pd.DataFrame, out: Path, params: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(11, 10))
    plot_density = density_points_for_plot(full_density, params)
    cmap, norm, _, _ = make_density_cmap_norm(plot_density["density_hit_count"] if len(plot_density) else full_density.get("density_hit_count", pd.Series([0.0])), params)

    if len(plot_density):
        ax.scatter(
            plot_density["x"], plot_density["y"],
            s=max(float(pget(params, "NORMAL_NODE_SIZE", 8.0)) * 0.7, 2.0),
            c=plot_density["density_hit_count"],
            cmap=cmap,
            norm=norm,
            alpha=0.30,
            linewidths=0,
            zorder=3,
        )

    grid_size = float(pget(params, "AREA_GRID_SIZE_M", 1000.0))
    if len(area_summary) and len(full_density):
        xmin = float(full_density["x"].min())
        ymin = float(full_density["y"].min())
        for _, ar in area_summary[area_summary["is_active_area"] > 0].iterrows():
            x0 = xmin + float(ar["area_ix"]) * grid_size
            y0 = ymin + float(ar["area_iy"]) * grid_size
            rect = plt.Rectangle((x0, y0), grid_size, grid_size, fill=False, edgecolor="black", linewidth=0.6, alpha=0.35, zorder=5)
            ax.add_patch(rect)
            if bool(pget(params, "PLOT_AREA_QUOTA_LABELS", False)):
                ax.text(
                    x0 + 0.03 * grid_size,
                    y0 + 0.92 * grid_size,
                    f"q={int(ar['area_quota'])}/s={int(ar['area_selected_count'])}",
                    fontsize=7,
                    color="black",
                    zorder=6,
                )

    overlay_flz_markers(ax, model_df, params)
    overlay_db_dk_markers(ax, model_df, params)

    sc = plot_candidate_symbols(ax, candidates, cmap, norm, params)
    if sc is not None:
        cb = fig.colorbar(sc, ax=ax, shrink=0.74, pad=0.02, extend="max")
        cb.set_label("Candidate route-density hit-count")
    finalize_map(ax, "Major and minor intermediate candidates by adaptive area quotas")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(pget(params, "PLOT_DPI", 250)))
    plt.close(fig)


def plot_area_candidate_quota(full_density: pd.DataFrame, area_summary: pd.DataFrame, out: Path, params: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(11, 10))
    plot_density = density_points_for_plot(full_density, params)
    cmap, norm, _, _ = make_density_cmap_norm(plot_density["density_hit_count"] if len(plot_density) else full_density.get("density_hit_count", pd.Series([0.0])), params)
    if len(plot_density):
        sc = ax.scatter(
            plot_density["x"], plot_density["y"],
            s=max(float(pget(params, "NORMAL_NODE_SIZE", 8.0)) * 0.7, 2.0),
            c=plot_density["density_hit_count"],
            cmap=cmap,
            norm=norm,
            alpha=0.35,
            linewidths=0,
            zorder=3,
        )
        cb = fig.colorbar(sc, ax=ax, shrink=0.74, pad=0.02, extend="max")
        cb.set_label("Route-density hit-count")
    if len(area_summary) and len(full_density):
        grid_size = float(pget(params, "AREA_GRID_SIZE_M", 1000.0))
        xmin = float(full_density["x"].min())
        ymin = float(full_density["y"].min())
        for _, ar in area_summary.iterrows():
            x0 = xmin + float(ar["area_ix"]) * grid_size
            y0 = ymin + float(ar["area_iy"]) * grid_size
            is_active = int(ar.get("is_active_area", 0)) > 0
            rect = plt.Rectangle((x0, y0), grid_size, grid_size, fill=False, edgecolor="black", linewidth=0.8 if is_active else 0.3, alpha=0.5 if is_active else 0.10, zorder=10)
            ax.add_patch(rect)
            if is_active and bool(pget(params, "PLOT_AREA_QUOTA_LABELS", False)):
                ax.text(
                    x0 + 0.05 * grid_size,
                    y0 + 0.88 * grid_size,
                    f"A {ar['area_id']}\nD={ar['area_density_sum']:.0f}\nq={int(ar['area_quota'])}",
                    fontsize=7,
                    color="black",
                    zorder=11,
                )
    finalize_map(ax, "Adaptive area quotas for balanced intermediate candidates")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(pget(params, "PLOT_DPI", 250)))
    plt.close(fig)


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    args = parse_args()
    params = load_params(args.param_file)
    setup_plot_style(params)

    theta_dir = Path(str(pget(params, "THETA_OUTPUT_DIR", "output/02_thetastar_master_plan")))
    route_nodes_dir = Path(str(pget(params, "ROUTE_NODES_DIR", theta_dir / "route_nodes")))
    model_file = Path(str(pget(params, "MODEL_FILE", theta_dir / "planning_model_with_flz_support.xyz")))
    output_dir = Path(str(pget(params, "OUTPUT_DIR", "output/03_clustering_hitcount")))
    fig_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"LAE-UTM ROUTE HIT-COUNT DBSCAN CLUSTERING {VERSION}")
    print("=" * 80)
    print(f"Param file      : {args.param_file}")
    print(f"Theta output dir: {theta_dir}")
    print(f"Route nodes dir : {route_nodes_dir}")
    print(f"Model file      : {model_file}")
    print(f"Output dir      : {output_dir}")
    print("-" * 80)

    model_df = read_planning_model(model_file)
    route_nodes = read_route_nodes(route_nodes_dir)

    print(f"Route rows      : {len(route_nodes):,}")
    print(f"Routes          : {route_nodes['route_id'].nunique():,}")
    print(f"Pairs           : {route_nodes['pair'].nunique():,}")

    consider_flz = bool(pget(params, "CONSIDER_FLZ_AS_TRAFFIC_NODE", True))
    print(f"FLZ as traffic node candidates: {consider_flz}")

    raw_edges, edge_hit = compute_edge_hit_count(route_nodes)
    node_hit = compute_node_hit_count(route_nodes, model_df=model_df)
    node_hit = compute_graph_degree(node_hit, raw_edges)
    node_hit = mark_is_flz(node_hit, params)
    full_density = build_full_area_density_map(model_df=model_df, node_hit=node_hit, params=params)
    full_density = mark_is_flz(full_density, params)

    clustered = cluster_hit_nodes(node_hit, params)
    cluster_summary = summarize_hit_clusters(clustered)
    major_candidates, full_density, area_summary = select_balanced_candidates(full_density, clustered, cluster_summary, params)
    major_candidates = prune_straight_flow_candidates(major_candidates, full_density, params, "major")
    major_candidates, demoted_minor_candidates = demote_non_crosscut_majors(major_candidates, params)
    minor_candidates, minor_area_summary = select_minor_candidates(full_density, clustered, major_candidates, params)
    minor_candidates = prune_straight_flow_candidates(minor_candidates, full_density, params, "minor")
    if len(demoted_minor_candidates):
        minor_candidates = pd.concat([demoted_minor_candidates, minor_candidates], ignore_index=True, sort=False)
    major_candidates = shift_candidates_away_from_obstacles(major_candidates, full_density, params, "major")
    minor_candidates = shift_candidates_away_from_obstacles(minor_candidates, full_density, params, "minor")
    major_candidates, minor_candidates = enforce_final_candidate_separation(major_candidates, minor_candidates, params)
    if len(minor_candidates):
        minor_candidates["candidate_id"] = [f"MIN{i+1:03d}" for i in range(len(minor_candidates))]
    candidates = combine_major_minor_candidates(major_candidates, minor_candidates)
    route_cluster_membership = compute_route_cluster_membership(route_nodes, clustered)
    master_plan_input_nodes = build_master_node_table(full_density, candidates, params)

    node_hit_file = output_dir / "node_hit_count.csv"
    full_density_file = output_dir / "full_area_density_map.csv"
    positive_density_file = output_dir / "full_area_density_map_positive_flyable.csv"
    area_summary_file = output_dir / "area_candidate_summary.csv"
    edge_hit_file = output_dir / "edge_hit_count.csv"
    clustered_file = output_dir / "dbscan_hit_clusters.csv"
    summary_file = output_dir / "dbscan_hit_cluster_summary.csv"
    major_candidates_file = output_dir / "major_candidate_nodes.csv"
    minor_candidates_file = output_dir / "minor_candidate_nodes.csv"
    minor_area_summary_file = output_dir / "minor_area_candidate_summary.csv"
    candidates_file = output_dir / "candidate_intermediate_nodes.csv"
    route_membership_file = output_dir / "route_hit_cluster_membership.csv"
    master_plan_input_file = output_dir / "master_plan_input_nodes.csv"

    node_hit.to_csv(node_hit_file, index=False)
    full_density.to_csv(full_density_file, index=False)
    density_points_for_plot(full_density, params).to_csv(positive_density_file, index=False)
    area_summary.to_csv(area_summary_file, index=False)
    edge_hit.to_csv(edge_hit_file, index=False)
    clustered.to_csv(clustered_file, index=False)
    cluster_summary.to_csv(summary_file, index=False)
    major_candidates.to_csv(major_candidates_file, index=False)
    minor_candidates.to_csv(minor_candidates_file, index=False)
    minor_area_summary.to_csv(minor_area_summary_file, index=False)
    candidates.to_csv(candidates_file, index=False)
    route_cluster_membership.to_csv(route_membership_file, index=False)
    master_plan_input_nodes.to_csv(master_plan_input_file, index=False)

    print(f"Unique hit nodes: {len(node_hit):,}")
    print(f"Whole-area nodes: {len(full_density):,}")
    print(f"Unique hit edges: {len(edge_hit):,}")
    print(f"DBSCAN clusters : {clustered.loc[clustered['dbscan_cluster'] >= 0, 'dbscan_cluster'].nunique():,}")
    print(f"Active areas    : {int((area_summary['is_active_area'] > 0).sum()) if len(area_summary) else 0:,}")
    print(f"Major candidates: {len(major_candidates):,}")
    print(f"Minor candidates: {len(minor_candidates):,}")
    print(f"Candidates total: {len(candidates):,}")
    n_flz_hit = int(node_hit["is_flz"].sum()) if "is_flz" in node_hit.columns else 0
    n_flz_candidates = int(candidates["is_flz"].sum()) if len(candidates) and "is_flz" in candidates.columns else 0
    print(f"FLZ nodes seen in routes  : {n_flz_hit:,}")
    print(f"FLZ nodes among candidates: {n_flz_candidates:,} (CONSIDER_FLZ_AS_TRAFFIC_NODE={consider_flz})")
    print("-" * 80)
    print(f"Saved: {node_hit_file}")
    print(f"Saved: {full_density_file}")
    print(f"Saved: {positive_density_file}")
    print(f"Saved: {area_summary_file}")
    print(f"Saved: {edge_hit_file}")
    print(f"Saved: {clustered_file}")
    print(f"Saved: {summary_file}")
    print(f"Saved: {major_candidates_file}")
    print(f"Saved: {minor_candidates_file}")
    print(f"Saved: {minor_area_summary_file}")
    print(f"Saved: {candidates_file}")
    print(f"Saved: {route_membership_file}")
    print(f"Saved: {master_plan_input_file}  (point 05_run_master_plan.py's INPUT_NODE_FILE here)")

    plot_node_hit_density(model_df, full_density, candidates, fig_dir / "00_node_hit_count_density.png", params)
    plot_edge_hit_density(model_df, edge_hit, fig_dir / "01_edge_hit_count_density.png", params)
    plot_dbscan_clusters(model_df, clustered, cluster_summary, fig_dir / "02_dbscan_hit_count_clusters.png", params)
    plot_candidates(model_df, full_density, candidates, area_summary, fig_dir / "03_candidate_intermediate_nodes.png", params)
    plot_area_candidate_quota(full_density, area_summary, fig_dir / "04_area_candidate_quota.png", params)

    print("-" * 80)
    print(f"Figures: {fig_dir}")
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
