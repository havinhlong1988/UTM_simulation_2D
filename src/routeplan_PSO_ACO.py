#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/routeplan_PSO_ACO.py

Master-route planner for LAE-UTM using PSO + ACO.

Workflow
--------
1. Read master_plan_input_nodes.csv from 03_cluster_theta_routes.py (the
   DBSCAN hit-count clustering stage) -- every raw model node plus DB/DK/FLZ
   objective flags and TN candidate flags. (04_kmean_route_hit_density.py's
   all_model_nodes_objective_TNcandidates.csv is an older, independent
   TN-detection path with the same column contract and still works as a
   drop-in INPUT_NODE_FILE if you want to compare the two candidate sources,
   but is not part of the active pipeline.)
2. Build a reduced master graph from DB/DK objectives, TN candidates, and FLZ.
3. Treat RA and no-fly/obstacle nodes as blocked.
4. Use PSO to tune/seed ACO parameters and edge-cost weights.
5. Use ACO to produce one forward route and one backward route for each DK-DB
   pair, preferably connected through TN nodes.
6. Add optional outer-zone backup route around the AOI/model boundary for backup-of-backup use.
7. Add TA/TN coverage scenarios so traffic-anchor candidates are used as many
   independent route options as possible.
8. Add one emergency FLZ branch for every successful operational route.
9. Save route summaries, route edges, graph diagnostics, PSO history, and maps.

Called from 05_run_master_plan.py, which imports run_master_plan/load_params
from this module.
"""
from __future__ import annotations

import heapq
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

try:
    from src.PSO import PSOOptimizer, bounds_from_params
    from src.ACO import ACOPlanner, ACOResult
except Exception:  # pragma: no cover - fallback for direct local execution
    from PSO import PSOOptimizer, bounds_from_params
    from ACO import ACOPlanner, ACOResult


MODULE_VERSION = "v8_aco_progress_bars"


# ----------------------------------------------------------------------
# Parameter handling
# ----------------------------------------------------------------------


def load_params(params_file: str | Path) -> SimpleNamespace:
    params_file = Path(params_file)
    if not params_file.exists():
        raise FileNotFoundError(f"Parameter file not found: {params_file}")
    namespace: dict[str, object] = {}
    code = params_file.read_text(encoding="utf-8")
    exec(compile(code, str(params_file), "exec"), {}, namespace)
    clean = {k: v for k, v in namespace.items() if not k.startswith("__")}
    clean["PARAMS_FILE"] = str(params_file)
    return SimpleNamespace(**clean)


def pget(params: SimpleNamespace, name: str, default=None):
    return getattr(params, name, default)


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _norm_col(c: str) -> str:
    return str(c).strip().lower().replace(" ", "_")


def _find_column(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    lookup = {_norm_col(c): c for c in df.columns}
    for cand in candidates:
        key = _norm_col(cand)
        if key in lookup:
            return lookup[key]
    return None


def _coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _truthy_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    text = s.fillna("").astype(str).str.strip().str.lower()
    return text.isin(["1", "true", "yes", "y", "t"])


# ----------------------------------------------------------------------
# Coordinate and geometry helpers
# ----------------------------------------------------------------------


def _looks_like_lonlat(xy: np.ndarray) -> bool:
    if xy.size == 0:
        return False
    x = xy[:, 0]; y = xy[:, 1]
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() == 0:
        return False
    x = x[finite]; y = y[finite]
    return bool(np.nanmin(x) >= -180 and np.nanmax(x) <= 180 and np.nanmin(y) >= -90 and np.nanmax(y) <= 90 and (np.nanmax(x)-np.nanmin(x)) < 5 and (np.nanmax(y)-np.nanmin(y)) < 5)


def _lonlat_to_local_m(xy: np.ndarray, origin: tuple[float, float]) -> np.ndarray:
    lon0, lat0 = origin
    earth_r = 6371000.0
    lat0_rad = math.radians(lat0)
    lon = xy[:, 0].astype(float)
    lat = xy[:, 1].astype(float)
    x = np.radians(lon - lon0) * earth_r * math.cos(lat0_rad)
    y = np.radians(lat - lat0) * earth_r
    return np.column_stack([x, y])


def _metric_xy(xy: np.ndarray, mode: str, origin: tuple[float, float] | None) -> np.ndarray:
    if mode == "meter":
        return xy.astype(float)
    if origin is None:
        raise ValueError("origin required for lonlat mode")
    return _lonlat_to_local_m(xy.astype(float), origin)


def _segment_sample_points(a: np.ndarray, b: np.ndarray, step_m: float) -> np.ndarray:
    d = float(np.linalg.norm(b - a))
    if not np.isfinite(d) or d <= 0:
        return a.reshape(1, 2)
    n = max(2, int(math.ceil(d / max(float(step_m), 1.0))) + 1)
    t = np.linspace(0.0, 1.0, n)
    return a[None, :] * (1.0 - t[:, None]) + b[None, :] * t[:, None]


def _edge_key(a: int, b: int) -> tuple[int, int]:
    a = int(a); b = int(b)
    return (a, b) if a <= b else (b, a)


def _point_to_segment_distance_m(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """2-D point-to-segment distance in metric coordinates."""
    p = np.asarray(p, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 0.0:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    q = a + t * ab
    return float(np.linalg.norm(p - q))


def _distance_to_polyline_m(points: np.ndarray, line: np.ndarray) -> np.ndarray:
    """Distance from each point to a closed/open polyline in metric coordinates."""
    points = np.asarray(points, dtype=float)
    line = np.asarray(line, dtype=float)
    if len(points) == 0 or len(line) < 2:
        return np.full(len(points), np.inf, dtype=float)
    out = np.full(len(points), np.inf, dtype=float)
    for a, b in zip(line[:-1], line[1:]):
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 0.0:
            d = np.linalg.norm(points - a[None, :], axis=1)
        else:
            t = np.sum((points - a[None, :]) * ab[None, :], axis=1) / denom
            t = np.clip(t, 0.0, 1.0)
            q = a[None, :] + t[:, None] * ab[None, :]
            d = np.linalg.norm(points - q, axis=1)
        out = np.minimum(out, d)
    return out


def _aoi_polygon_metric(params: SimpleNamespace, model: "ModelData") -> np.ndarray | None:
    """Return AOI polygon in metric coordinates, or None if not configured."""
    polygon = pget(params, "AOI_POLYGON", None)
    if polygon is None:
        return None
    pts = []
    for item in polygon:
        if item is None or len(item) < 2:
            continue
        pts.append((float(item[0]), float(item[1])))
    if len(pts) < 3:
        return None
    poly = np.asarray(pts, dtype=float)
    # Close polygon if needed.
    if not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])
    return _metric_xy(poly, model.xy_mode, model.origin_lonlat)


def _bbox_boundary_metric(model: "ModelData") -> np.ndarray:
    xy = model.xy_metric[np.isfinite(model.xy_metric[:, 0]) & np.isfinite(model.xy_metric[:, 1])]
    xmin, ymin = np.nanmin(xy, axis=0)
    xmax, ymax = np.nanmax(xy, axis=0)
    return np.asarray([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]], dtype=float)


# ----------------------------------------------------------------------
# Data containers
# ----------------------------------------------------------------------


@dataclass
class ModelData:
    df: pd.DataFrame
    x_col: str
    y_col: str
    label_col: str
    prefix_col: str
    slowness_col: str | None
    xy_original: np.ndarray
    xy_metric: np.ndarray
    xy_mode: str
    origin_lonlat: tuple[float, float] | None
    nofly_mask: np.ndarray
    ra_mask: np.ndarray
    flyable_mask: np.ndarray


# ----------------------------------------------------------------------
# Loading and classification
# ----------------------------------------------------------------------


def load_node_model(params: SimpleNamespace) -> ModelData:
    path = Path(pget(params, "INPUT_NODE_FILE", "output/04_kmean_route_hit_density/all_model_nodes_objective_TNcandidates.csv"))
    if not path.exists():
        raise FileNotFoundError(f"INPUT_NODE_FILE not found: {path}")

    df = pd.read_csv(path)
    x_col = _find_column(df, ["x_original", "x", "lon", "longitude"])
    y_col = _find_column(df, ["y_original", "y", "lat", "latitude"])
    if x_col is None or y_col is None:
        raise ValueError(f"Could not detect coordinate columns in {path}. Columns: {list(df.columns)}")

    label_col = _find_column(df, ["label", "objective_label", "candidate_id"])
    if label_col is None:
        label_col = "label"
        df[label_col] = ""

    prefix_col = _find_column(df, ["label_prefix", "objective_prefix"])
    if prefix_col is None:
        prefix_col = "label_prefix"
        labels = df[label_col].fillna("").astype(str)
        df[prefix_col] = labels.str.extract(r"^([A-Za-z_]+)", expand=False).fillna("").str.upper()
    else:
        df[prefix_col] = df[prefix_col].fillna("").astype(str).str.upper()

    slowness_col = _find_column(df, ["slowness", "slow", "cost"])
    df[x_col] = _coerce_numeric(df[x_col])
    df[y_col] = _coerce_numeric(df[y_col])
    if slowness_col is not None:
        df[slowness_col] = _coerce_numeric(df[slowness_col])

    if "model_index" not in df.columns:
        df["model_index"] = np.arange(len(df), dtype=int)
    else:
        # pandas.Series.fillna() cannot receive a NumPy array.
        # Use a same-index fallback Series so missing model_index values are
        # replaced by the row number while preserving valid existing IDs.
        fallback_index = pd.Series(np.arange(len(df), dtype=int), index=df.index)
        df["model_index"] = pd.to_numeric(df["model_index"], errors="coerce").fillna(fallback_index).astype(int)

    xy_original = df[[x_col, y_col]].to_numpy(dtype=float)
    coord_mode = str(pget(params, "XY_MODE", "auto")).lower()
    if coord_mode == "auto":
        xy_mode = "lonlat" if _looks_like_lonlat(xy_original) else "meter"
    elif coord_mode in ("lonlat", "meter"):
        xy_mode = coord_mode
    else:
        raise ValueError("XY_MODE must be auto, lonlat, or meter")

    finite = np.isfinite(xy_original[:, 0]) & np.isfinite(xy_original[:, 1])
    origin = (float(np.nanmean(xy_original[finite, 0])), float(np.nanmean(xy_original[finite, 1]))) if xy_mode == "lonlat" else None
    xy_metric = _metric_xy(xy_original, xy_mode, origin)

    prefixes = df[prefix_col].fillna("").astype(str).str.upper()
    ra_prefixes = {str(v).upper() for v in _as_list(pget(params, "RA_PREFIXES", ["RA"]))}
    ra_mask = prefixes.isin(ra_prefixes).to_numpy()

    nofly_mask = np.zeros(len(df), dtype=bool)
    if slowness_col is not None:
        slow = df[slowness_col].to_numpy(dtype=float)
        thr = float(pget(params, "NOFLY_SLOWNESS_THRESHOLD", 10.0))
        mode = str(pget(params, "NOFLY_THRESHOLD_MODE", "greater_equal")).lower()
        if mode == "greater_equal":
            nofly_mask |= slow >= thr
        elif mode == "greater":
            nofly_mask |= slow > thr
        elif mode == "equal":
            nofly_mask |= np.isclose(slow, thr)
        else:
            raise ValueError(f"Unknown NOFLY_THRESHOLD_MODE={mode}")

    nofly_mask |= ra_mask
    nofly_mask |= ~finite
    flyable_mask = ~nofly_mask

    return ModelData(df, x_col, y_col, label_col, prefix_col, slowness_col, xy_original, xy_metric, xy_mode, origin, nofly_mask, ra_mask, flyable_mask)


def _candidate_mask(df: pd.DataFrame) -> np.ndarray:
    if "is_candidate" in df.columns:
        return _truthy_series(df["is_candidate"]).to_numpy()
    if "candidate_id" in df.columns:
        return df["candidate_id"].fillna("").astype(str).str.len().to_numpy() > 0
    return np.zeros(len(df), dtype=bool)


def build_master_nodes(model: ModelData, params: SimpleNamespace) -> pd.DataFrame:
    df = model.df
    prefixes = df[model.prefix_col].fillna("").astype(str).str.upper()
    labels = df[model.label_col].fillna("").astype(str)
    cand_mask = _candidate_mask(df)

    db_prefixes = {str(v).upper() for v in _as_list(pget(params, "DB_PREFIXES", ["DB"]))}
    dk_prefixes = {str(v).upper() for v in _as_list(pget(params, "DK_PREFIXES", ["DK"]))}
    flz_prefixes = {str(v).upper() for v in _as_list(pget(params, "FLZ_PREFIXES", ["FLZ"]))}

    use_flz = bool(pget(params, "INCLUDE_FLZ_AS_GRAPH_NODE", True))
    rows: list[dict] = []
    seen_model_idx: set[int] = set()

    for i, row in df.iterrows():
        prefix = str(prefixes.iloc[i]).upper()
        role = None
        if prefix in db_prefixes:
            role = "DB"
        elif prefix in dk_prefixes:
            role = "DK"
        elif use_flz and prefix in flz_prefixes:
            role = "FLZ"
        elif cand_mask[i] and model.flyable_mask[i]:
            ctype = str(row.get("candidate_type", "TN")).strip().lower()
            role = "TN_major" if ctype == "major" else "TN_minor"

        if role is None:
            continue
        if model.ra_mask[i] or (role.startswith("TN") and not model.flyable_mask[i]):
            continue

        midx = int(row.get("model_index", i))
        if midx in seen_model_idx:
            continue
        seen_model_idx.add(midx)

        if role.startswith("TN"):
            name = str(row.get("candidate_id", "")).strip() or f"TN_{midx}"
        else:
            name = str(row.get("objective_label", "")).strip() or str(labels.iloc[i]).strip() or f"{role}_{midx}"

        rows.append({
            "node_id": midx,
            "model_row": int(i),
            "name": name,
            "role": role,
            "label": str(labels.iloc[i]),
            "label_prefix": prefix,
            "x": float(model.xy_original[i, 0]),
            "y": float(model.xy_original[i, 1]),
            "x_m": float(model.xy_metric[i, 0]),
            "y_m": float(model.xy_metric[i, 1]),
            "route_hit_count": float(row.get("route_hit_count", 0.0)) if pd.notna(row.get("route_hit_count", np.nan)) else 0.0,
            "nofly": bool(model.nofly_mask[i]),
            "ra": bool(model.ra_mask[i]),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No master graph nodes found. Check DB/DK/TN/FLZ columns in input CSV.")
    return out.sort_values(["role", "name", "node_id"]).reset_index(drop=True)


def annotate_outer_zone_nodes(model: ModelData, nodes: pd.DataFrame, params: SimpleNamespace) -> pd.DataFrame:
    """Mark candidate nodes that can support the outer-zone backup route.

    The outer-zone route is intended as a backup-of-backup corridor.  Normal
    inner DK-DB routes are planned first.  This annotation identifies TN nodes
    near the configured AOI boundary or the inferred model boundary so the later ACO pass can
    intentionally route through the outer band when inner corridors are busy.
    """
    out = nodes.copy()
    out["is_outer_zone_node"] = False
    out["outer_boundary_distance_m"] = np.nan
    out["outer_radial_score"] = np.nan

    if not bool(pget(params, "ENABLE_OUTER_ZONE_BACKUP_ROUTE", True)):
        return out

    xy = out[["x_m", "y_m"]].to_numpy(dtype=float)
    if len(xy) == 0:
        return out

    method = str(pget(params, "OUTER_ZONE_METHOD", "aoi_boundary")).lower()
    if method == "aoi_boundary":
        boundary = _aoi_polygon_metric(params, model)
        if boundary is None:
            boundary = _bbox_boundary_metric(model)
    elif method == "bbox_boundary":
        boundary = _bbox_boundary_metric(model)
    else:
        boundary = None

    if boundary is not None:
        boundary_dist = _distance_to_polyline_m(xy, boundary)
    else:
        boundary_dist = np.full(len(out), np.nan, dtype=float)

    center = np.nanmean(model.xy_metric[np.isfinite(model.xy_metric[:, 0]) & np.isfinite(model.xy_metric[:, 1])], axis=0)
    radial = np.linalg.norm(xy - center[None, :], axis=1)
    rmax = max(float(np.nanmax(radial)), 1.0)
    radial_score = radial / rmax

    roles_allowed = {str(v) for v in _as_list(pget(params, "OUTER_ZONE_NODE_ROLES", ["TN_major", "TN_minor"]))}
    role_ok = out["role"].astype(str).isin(roles_allowed).to_numpy()

    band_m = float(pget(params, "OUTER_ZONE_BAND_M", 700.0))
    radial_pct = float(pget(params, "OUTER_ZONE_RADIAL_PERCENTILE", 85.0))
    radial_thr = np.nanpercentile(radial_score, radial_pct) if len(radial_score) else 1.0

    if method in ("aoi_boundary", "bbox_boundary"):
        outer = role_ok & np.isfinite(boundary_dist) & (boundary_dist <= band_m)
    elif method == "radial_percentile":
        outer = role_ok & np.isfinite(radial_score) & (radial_score >= radial_thr)
    else:
        # Conservative hybrid fallback: boundary band OR far radial percentile.
        outer = role_ok & ((np.isfinite(boundary_dist) & (boundary_dist <= band_m)) | (radial_score >= radial_thr))

    # If too few outer TN nodes are found, fall back to the farthest radial TN nodes.
    min_outer = int(pget(params, "OUTER_ZONE_MIN_CANDIDATE_NODES", 3))
    if int(outer.sum()) < min_outer and role_ok.any():
        candidate_idx = np.flatnonzero(role_ok)
        order = candidate_idx[np.argsort(-radial_score[candidate_idx])]
        take = order[: min(len(order), max(min_outer, int(outer.sum())))]
        outer[take] = True

    out["is_outer_zone_node"] = outer
    out["outer_boundary_distance_m"] = boundary_dist
    out["outer_radial_score"] = radial_score
    return out


def annotate_outer_zone_edges(edges: pd.DataFrame, nodes: pd.DataFrame, params: SimpleNamespace) -> pd.DataFrame:
    """Add outer-zone edge flags used by the backup-of-backup ACO pass."""
    if edges.empty:
        return edges
    out = edges.copy()
    outer_lookup = dict(zip(nodes["node_id"].astype(int), nodes.get("is_outer_zone_node", pd.Series(False, index=nodes.index)).astype(bool)))
    u_outer = out["u"].astype(int).map(outer_lookup).fillna(False).astype(bool).to_numpy()
    v_outer = out["v"].astype(int).map(outer_lookup).fillna(False).astype(bool).to_numpy()
    mode = str(pget(params, "OUTER_ZONE_EDGE_MODE", "any_outer_endpoint")).lower()
    if mode == "both_outer_endpoints":
        is_outer = u_outer & v_outer
    else:
        is_outer = u_outer | v_outer
    out["u_is_outer_zone"] = u_outer
    out["v_is_outer_zone"] = v_outer
    out["is_outer_zone_edge"] = is_outer.astype(int)
    return out


# ----------------------------------------------------------------------
# Graph construction
# ----------------------------------------------------------------------


def _segment_min_distance_to_tree(samples: np.ndarray, tree: cKDTree | None) -> float:
    if tree is None or samples.size == 0:
        return float("inf")
    d, _ = tree.query(samples, k=1)
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    return float(np.min(d)) if len(d) else float("inf")


def _segment_mean_nearest_value(samples: np.ndarray, tree: cKDTree, values: np.ndarray) -> float:
    if samples.size == 0 or len(values) == 0:
        return 0.0
    _, idx = tree.query(samples, k=1)
    idx = np.asarray(idx, dtype=int)
    vals = values[idx]
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if len(vals) else 0.0


def build_master_edges(model: ModelData, nodes: pd.DataFrame, params: SimpleNamespace) -> pd.DataFrame:
    xy = nodes[["x_m", "y_m"]].to_numpy(dtype=float)
    max_dist = float(pget(params, "MAX_MASTER_EDGE_DISTANCE_M", 2500.0))
    min_clear = float(pget(params, "EDGE_NOFLY_CLEARANCE_M", 80.0))
    sample_step = float(pget(params, "EDGE_SAMPLE_STEP_M", 50.0))
    allow_direct = bool(pget(params, "ALLOW_DIRECT_DB_DK_EDGE", True))

    nofly_xy = model.xy_metric[model.nofly_mask]
    nofly_tree = cKDTree(nofly_xy) if len(nofly_xy) else None

    all_tree = cKDTree(model.xy_metric[np.isfinite(model.xy_metric[:, 0]) & np.isfinite(model.xy_metric[:, 1])])
    all_valid_mask = np.isfinite(model.xy_metric[:, 0]) & np.isfinite(model.xy_metric[:, 1])
    all_hit = model.df.get("route_hit_count", pd.Series(0.0, index=model.df.index)).fillna(0.0).astype(float).to_numpy()[all_valid_mask]

    flz_nodes = nodes[nodes["role"] == "FLZ"]
    flz_tree = cKDTree(flz_nodes[["x_m", "y_m"]].to_numpy(dtype=float)) if len(flz_nodes) else None

    node_tree = cKDTree(xy)
    pairs = node_tree.query_pairs(r=max_dist, output_type="set")
    rows: list[dict] = []

    for i, j in sorted(pairs):
        a = xy[int(i)]; b = xy[int(j)]
        u = int(nodes.loc[int(i), "node_id"]); v = int(nodes.loc[int(j), "node_id"])
        role_u = str(nodes.loc[int(i), "role"]); role_v = str(nodes.loc[int(j), "role"])

        # Avoid meaningless direct DB-DK if the user wants all traffic through TN.
        if not allow_direct and {role_u, role_v} == {"DB", "DK"}:
            continue

        # Normally FLZ is available as safe parking, but not forced.  Still keep
        # FLZ edges unless the user disables them.
        if not bool(pget(params, "ALLOW_FLZ_NORMAL_GRAPH_EDGES", True)) and (role_u == "FLZ" or role_v == "FLZ"):
            continue

        dist = float(np.linalg.norm(b - a))
        if not math.isfinite(dist) or dist <= 0:
            continue
        samples = _segment_sample_points(a, b, sample_step)
        nofly_clear = _segment_min_distance_to_tree(samples, nofly_tree)
        if nofly_clear < min_clear:
            continue

        density = _segment_mean_nearest_value(samples, all_tree, all_hit)
        flz_dist = _segment_min_distance_to_tree(samples, flz_tree)
        emergency_support = 0.0 if not math.isfinite(flz_dist) else math.exp(-0.5 * (flz_dist / max(float(pget(params, "FLZ_SUPPORT_SIGMA_M", 800.0)), 1.0)) ** 2)

        is_tn_edge = int(role_u.startswith("TN") or role_v.startswith("TN"))
        is_flz_edge = int(role_u == "FLZ" or role_v == "FLZ")
        rows.append({
            "edge_id": len(rows),
            "u": u,
            "v": v,
            "u_name": nodes.loc[int(i), "name"],
            "v_name": nodes.loc[int(j), "name"],
            "u_role": role_u,
            "v_role": role_v,
            "x1": float(nodes.loc[int(i), "x"]),
            "y1": float(nodes.loc[int(i), "y"]),
            "x2": float(nodes.loc[int(j), "x"]),
            "y2": float(nodes.loc[int(j), "y"]),
            "distance_m": dist,
            "nofly_clearance_m": nofly_clear,
            "route_hit_density": density,
            "nearest_flz_distance_m": flz_dist,
            "emergency_support": emergency_support,
            "is_tn_edge": is_tn_edge,
            "is_flz_edge": is_flz_edge,
        })

    edges = pd.DataFrame(rows)
    if edges.empty:
        raise RuntimeError("No master graph edges remain after no-fly/clearance filtering. Increase MAX_MASTER_EDGE_DISTANCE_M or reduce EDGE_NOFLY_CLEARANCE_M.")
    return edges


def apply_edge_weights(edges: pd.DataFrame, weights: dict[str, float], params: SimpleNamespace) -> pd.DataFrame:
    out = edges.copy()
    d = out["distance_m"].astype(float).to_numpy()
    density = out["route_hit_density"].astype(float).to_numpy()
    clearance = out["nofly_clearance_m"].astype(float).to_numpy()
    support = out["emergency_support"].astype(float).to_numpy()
    is_tn = out["is_tn_edge"].astype(float).to_numpy()
    is_flz = out["is_flz_edge"].astype(float).to_numpy()
    is_outer = out.get("is_outer_zone_edge", pd.Series(0, index=out.index)).astype(float).to_numpy()

    # Robust normalization.
    d_norm = d / max(float(np.nanpercentile(d, 95)), 1.0)
    den_norm = density / max(float(np.nanpercentile(density, 95)), 1.0)
    clear_norm = np.clip(clearance / max(float(pget(params, "CLEARANCE_NORMALIZATION_M", 600.0)), 1.0), 0.0, 1.0)
    clearance_risk = 1.0 - clear_norm

    cost = (
        float(weights.get("distance_weight", 1.0)) * d_norm
        - float(weights.get("density_weight", 0.5)) * den_norm
        + float(weights.get("clearance_weight", 1.0)) * clearance_risk
        - float(weights.get("emergency_weight", 0.3)) * support
        - float(weights.get("tn_bonus", 0.5)) * is_tn
        + float(weights.get("flz_penalty", 0.7)) * is_flz
        + float(pget(params, "OUTER_ZONE_NORMAL_EDGE_PENALTY", 0.0)) * is_outer
    )

    # Keep cost positive for ACO probability and Dijkstra.
    cost = cost - np.nanmin(cost) + float(pget(params, "EDGE_COST_EPSILON", 0.05))
    out["edge_cost"] = cost

    # Pheromone seed: short + dense + clear + emergency support + TN edges.
    desirability = 1.0 / np.maximum(cost, 1.0e-9)
    out["aco_initial_desirability"] = desirability
    out["edge_cost_distance_component"] = d_norm
    out["edge_cost_density_component"] = den_norm
    out["edge_cost_clearance_risk_component"] = clearance_risk
    out["edge_cost_outer_zone_component"] = is_outer
    return out


# ----------------------------------------------------------------------
# Route pairs and fitness
# ----------------------------------------------------------------------


def build_route_pairs(nodes: pd.DataFrame, params: SimpleNamespace) -> list[tuple[int, int, str, str]]:
    db = nodes[nodes["role"] == "DB"].copy()
    dk = nodes[nodes["role"] == "DK"].copy()
    if db.empty or dk.empty:
        raise RuntimeError(f"Need at least one DB and one DK. Found DB={len(db)}, DK={len(dk)}")

    mode = str(pget(params, "ROUTE_PAIR_MODE", "all_dk_to_all_db")).lower()
    pairs: list[tuple[int, int, str, str]] = []
    if mode == "nearest_db_per_dk":
        db_xy = db[["x_m", "y_m"]].to_numpy(dtype=float)
        tree = cKDTree(db_xy)
        for _, row in dk.iterrows():
            d, ii = tree.query([[float(row["x_m"]), float(row["y_m"])]], k=1)
            db_row = db.iloc[int(ii[0])]
            pairs.append((int(row["node_id"]), int(db_row["node_id"]), str(row["name"]), str(db_row["name"])))
    elif mode == "all_dk_to_all_db":
        for _, dkrow in dk.iterrows():
            for _, dbrow in db.iterrows():
                pairs.append((int(dkrow["node_id"]), int(dbrow["node_id"]), str(dkrow["name"]), str(dbrow["name"])))
    else:
        raise ValueError("ROUTE_PAIR_MODE must be all_dk_to_all_db or nearest_db_per_dk")

    max_pairs = int(pget(params, "MAX_ROUTE_PAIRS", 0))
    if max_pairs > 0:
        pairs = pairs[:max_pairs]
    return pairs


def _dijkstra_path(nodes: pd.DataFrame, edges: pd.DataFrame, start: int, goal: int) -> list[int]:
    adj: dict[int, list[tuple[int, float]]] = {}
    for _, row in edges.iterrows():
        u = int(row["u"]); v = int(row["v"]); c = float(row["edge_cost"])
        adj.setdefault(u, []).append((v, c)); adj.setdefault(v, []).append((u, c))
    pq = [(0.0, int(start))]
    dist = {int(start): 0.0}
    parent = {int(start): int(start)}
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, math.inf):
            continue
        if u == int(goal):
            break
        for v, c in adj.get(u, []):
            nd = d + c
            if nd < dist.get(v, math.inf):
                dist[v] = nd; parent[v] = u; heapq.heappush(pq, (nd, v))
    if int(goal) not in parent:
        return []
    path = [int(goal)]
    cur = int(goal)
    while parent[cur] != cur:
        cur = parent[cur]
        path.append(cur)
    path.reverse()
    return path


def evaluate_pso_particle(nodes: pd.DataFrame, base_edges: pd.DataFrame, pairs: list[tuple[int, int, str, str]], params: SimpleNamespace, weights: dict[str, float]) -> float:
    edges = apply_edge_weights(base_edges, weights, params)
    node_role = dict(zip(nodes["node_id"].astype(int), nodes["role"].astype(str)))
    max_eval_pairs = int(pget(params, "PSO_EVALUATION_MAX_PAIRS", 20))
    eval_pairs = pairs[:max_eval_pairs] if max_eval_pairs > 0 else pairs
    if not eval_pairs:
        return -1.0e9

    connected = 0
    total_cost = 0.0
    total_tn = 0
    total_clear = 0.0
    for dk_id, db_id, _dk_name, _db_name in eval_pairs:
        path = _dijkstra_path(nodes, edges, dk_id, db_id)
        if not path:
            total_cost += 1.0e6
            continue
        connected += 1
        tn_count = sum(1 for n in path[1:-1] if node_role.get(int(n), "").startswith("TN"))
        total_tn += tn_count
        # path edge stats
        edge_map = {_edge_key(row.u, row.v): row for row in edges.itertuples(index=False)}
        pcost = 0.0
        pclear = []
        for a, b in zip(path[:-1], path[1:]):
            r = edge_map.get(_edge_key(a, b))
            if r is not None:
                pcost += float(r.edge_cost)
                pclear.append(float(r.nofly_clearance_m))
        if bool(pget(params, "REQUIRE_TN_IN_ROUTE", True)) and tn_count < int(pget(params, "MIN_TN_PER_ROUTE", 1)):
            pcost += float(pget(params, "PSO_MISSING_TN_PENALTY", 500.0))
        total_cost += pcost
        total_clear += min(pclear) if pclear else 0.0

    connected_ratio = connected / max(len(eval_pairs), 1)
    mean_cost = total_cost / max(len(eval_pairs), 1)
    mean_tn = total_tn / max(connected, 1)
    mean_clear = total_clear / max(connected, 1)

    fitness = (
        10000.0 * connected_ratio
        - 120.0 * mean_cost
        + 50.0 * mean_tn
        + 0.05 * mean_clear
    )
    return float(fitness)


# ----------------------------------------------------------------------
# ACO routing and outputs
# ----------------------------------------------------------------------


def _used_edges_from_path(path: Sequence[int]) -> set[tuple[int, int]]:
    return {_edge_key(a, b) for a, b in zip(path[:-1], path[1:])}


def _route_flz_min_distance(result, edge_lookup: dict) -> float:
    """min(nearest_flz_distance_m) over result.path's consecutive edges, or
    inf if the path has no edges / none are found in edge_lookup.

    edge_lookup maps _edge_key(u, v) -> edge row (as produced by
    {_edge_key(row.u, row.v): row for row in edges.itertuples(index=False)}),
    the same shape _route_summary() already builds -- share one lookup
    rather than rebuilding it per call."""
    if not result.success or len(result.path) < 2:
        return float("inf")
    dists = []
    for a, b in zip(result.path[:-1], result.path[1:]):
        row = edge_lookup.get(_edge_key(a, b))
        if row is not None:
            dists.append(float(row.nearest_flz_distance_m))
    return float(np.min(dists)) if dists else float("inf")


def _route_tn_count(result, node_role: dict[int, str]) -> int:
    """Count interior-path nodes (endpoints excluded) whose role starts with
    'TN'. Mirrors ACOPlanner._path_stats' own tn_count convention
    (src/ACO.py) -- endpoints are always DB/DK, never TN, so excluding them
    is a no-op in practice but keeps the two counts consistent by
    construction."""
    if not result.success or len(result.path) < 3:
        return 0
    return sum(1 for n in result.path[1:-1] if str(node_role.get(int(n), "")).upper().startswith("TN"))


def _route_summary(result, nodes: pd.DataFrame, edges: pd.DataFrame, direction: str, dk_name: str, db_name: str) -> dict:
    node_lookup = nodes.set_index("node_id").to_dict("index")
    edge_lookup = {_edge_key(row.u, row.v): row for row in edges.itertuples(index=False)}
    names = [str(node_lookup.get(int(n), {}).get("name", n)) for n in result.path]
    roles = [str(node_lookup.get(int(n), {}).get("role", "")) for n in result.path]
    tn_nodes = [names[i] for i, r in enumerate(roles) if r.startswith("TN")]
    flz_nodes = [names[i] for i, r in enumerate(roles) if r == "FLZ"]
    outer_nodes = [names[i] for i, n in enumerate(result.path) if bool(node_lookup.get(int(n), {}).get("is_outer_zone_node", False))]
    clearances = []
    for a, b in zip(result.path[:-1], result.path[1:]):
        row = edge_lookup.get(_edge_key(a, b))
        if row is not None:
            clearances.append(float(row.nofly_clearance_m))
    flz_min_dist = _route_flz_min_distance(result, edge_lookup)
    return {
        "route_key": result.route_key,
        "direction": direction,
        "dk_name": dk_name,
        "db_name": db_name,
        "success": bool(result.success),
        "message": result.message,
        "path_node_ids": ";".join(str(int(v)) for v in result.path),
        "path_node_names": ";".join(names),
        "path_node_roles": ";".join(roles),
        "tn_nodes": ";".join(tn_nodes),
        "flz_nodes_on_route": ";".join(flz_nodes),
        "n_nodes": len(result.path),
        "n_edges": max(0, len(result.path) - 1),
        "n_tn_nodes": len(tn_nodes),
        "n_flz_nodes_on_route": len(flz_nodes),
        "outer_zone_nodes_on_route": ";".join(outer_nodes),
        "n_outer_zone_nodes": len(outer_nodes),
        "total_cost": float(result.total_cost),
        "total_distance_m": float(result.total_distance_m),
        "min_nofly_clearance_m": float(np.min(clearances)) if clearances else np.nan,
        "nearest_flz_distance_m": flz_min_dist if math.isfinite(flz_min_dist) else np.nan,
    }


def _route_edges_table(result, nodes: pd.DataFrame, edges: pd.DataFrame, direction: str, dk_name: str, db_name: str) -> pd.DataFrame:
    node_lookup = nodes.set_index("node_id").to_dict("index")
    edge_lookup = {_edge_key(row.u, row.v): row for row in edges.itertuples(index=False)}
    rows = []
    for seq, (a, b) in enumerate(zip(result.path[:-1], result.path[1:])):
        na = node_lookup[int(a)]; nb = node_lookup[int(b)]
        er = edge_lookup.get(_edge_key(a, b))
        rows.append({
            "route_key": result.route_key,
            "direction": direction,
            "dk_name": dk_name,
            "db_name": db_name,
            "edge_seq": int(seq),
            "from_node_id": int(a),
            "to_node_id": int(b),
            "from_name": na.get("name", str(a)),
            "to_name": nb.get("name", str(b)),
            "from_role": na.get("role", ""),
            "to_role": nb.get("role", ""),
            "x1": float(na.get("x", np.nan)),
            "y1": float(na.get("y", np.nan)),
            "x2": float(nb.get("x", np.nan)),
            "y2": float(nb.get("y", np.nan)),
            "edge_distance_m": float(getattr(er, "distance_m", np.nan)) if er is not None else np.nan,
            "edge_cost": float(getattr(er, "edge_cost", np.nan)) if er is not None else np.nan,
            "nofly_clearance_m": float(getattr(er, "nofly_clearance_m", np.nan)) if er is not None else np.nan,
            "nearest_flz_distance_m": float(getattr(er, "nearest_flz_distance_m", np.nan)) if er is not None else np.nan,
            "route_hit_density": float(getattr(er, "route_hit_density", np.nan)) if er is not None else np.nan,
            "is_outer_zone_edge": int(getattr(er, "is_outer_zone_edge", 0)) if er is not None else 0,
        })
    return pd.DataFrame(rows)


def _outer_biased_edges(edges: pd.DataFrame, used_edges: set[tuple[int, int]], params: SimpleNamespace) -> pd.DataFrame:
    """Return edge table reweighted for outer-zone backup routing."""
    out = edges.copy()
    if "edge_cost" not in out.columns:
        out["edge_cost"] = out.get("distance_m", pd.Series(1.0, index=out.index)).astype(float)
    is_outer = out.get("is_outer_zone_edge", pd.Series(0, index=out.index)).astype(int).to_numpy() > 0
    bonus = float(pget(params, "OUTER_ZONE_EDGE_BONUS_FACTOR", 0.65))
    non_outer_penalty = float(pget(params, "OUTER_ZONE_NON_OUTER_EDGE_PENALTY", 2.0))
    avoid_penalty = float(pget(params, "OUTER_ZONE_AVOID_INNER_EDGE_PENALTY", 5.0))

    cost = out["edge_cost"].astype(float).to_numpy().copy()
    cost[is_outer] *= max(0.05, 1.0 - bonus)
    cost[~is_outer] *= max(1.0, 1.0 + non_outer_penalty)

    if used_edges:
        used = np.asarray([_edge_key(u, v) in used_edges for u, v in zip(out["u"], out["v"])], dtype=bool)
        cost[used] *= max(1.0, avoid_penalty)

    out["edge_cost"] = cost
    # Increase initial desirability on outer-zone edges for ACO.
    desirability = 1.0 / np.maximum(cost, 1.0e-9)
    if "aco_initial_desirability" in out.columns:
        out["aco_initial_desirability"] = np.maximum(out["aco_initial_desirability"].astype(float).to_numpy(), desirability)
    else:
        out["aco_initial_desirability"] = desirability
    return out


def _merge_two_route_results(r1, r2, route_key: str) -> ACOResult:
    """Merge start->waypoint and waypoint->goal ACO results."""
    if not r1.success or not r2.success or not r1.path or not r2.path:
        return ACOResult(False, route_key, [], math.inf, math.inf, [], int(max(getattr(r1, "iterations", 0), getattr(r2, "iterations", 0))), int(max(getattr(r1, "ants", 0), getattr(r2, "ants", 0))), "No valid outer waypoint route.")
    path = [int(v) for v in r1.path] + [int(v) for v in r2.path[1:]]
    edge_indices = list(getattr(r1, "edge_indices", [])) + list(getattr(r2, "edge_indices", []))
    return ACOResult(
        True,
        route_key,
        path,
        float(getattr(r1, "total_cost", 0.0)) + float(getattr(r2, "total_cost", 0.0)),
        float(getattr(r1, "total_distance_m", 0.0)) + float(getattr(r2, "total_distance_m", 0.0)),
        edge_indices,
        int(max(getattr(r1, "iterations", 0), getattr(r2, "iterations", 0))),
        int(max(getattr(r1, "ants", 0), getattr(r2, "ants", 0))),
        "Path found by ACO through an outer-zone waypoint.",
    )


def plan_outer_backup_route(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    base_planner: ACOPlanner,
    start_id: int,
    goal_id: int,
    route_key: str,
    used_edges: set[tuple[int, int]],
    params: SimpleNamespace,
) -> ACOResult:
    """Plan a boundary/outer-zone backup route for a DK-DB pair.

    The route is intentionally biased to use at least one outer-zone TN node.
    This represents the future operational option: when all inner-zone routes
    are busy, the system still has a longer outer fallback corridor.
    """
    outer = nodes[nodes.get("is_outer_zone_node", pd.Series(False, index=nodes.index)).astype(bool)].copy()
    if not bool(pget(params, "OUTER_ZONE_ALLOW_FLZ_WAYPOINTS", False)):
        outer = outer[outer["role"].astype(str).str.startswith("TN")]
    if outer.empty:
        return base_planner.plan_route(start_id, goal_id, route_key=route_key, avoid_edges=used_edges)

    # Prefer outer nodes that are far from the map center/boundary band, have
    # strong route density, and are not too close to the endpoints.
    sx = nodes.loc[nodes["node_id"].astype(int) == int(start_id), ["x_m", "y_m"]].to_numpy(dtype=float)[0]
    gx = nodes.loc[nodes["node_id"].astype(int) == int(goal_id), ["x_m", "y_m"]].to_numpy(dtype=float)[0]
    ox = outer[["x_m", "y_m"]].to_numpy(dtype=float)
    d_endpoint = np.minimum(np.linalg.norm(ox - sx[None, :], axis=1), np.linalg.norm(ox - gx[None, :], axis=1))
    hit = outer.get("route_hit_count", pd.Series(0.0, index=outer.index)).fillna(0.0).astype(float).to_numpy()
    radial = outer.get("outer_radial_score", pd.Series(0.0, index=outer.index)).fillna(0.0).astype(float).to_numpy()
    score = radial + 0.25 * (hit / max(float(np.nanmax(hit)), 1.0)) + 0.10 * (d_endpoint / max(float(np.nanmax(d_endpoint)), 1.0))
    outer = outer.assign(_outer_waypoint_score=score).sort_values("_outer_waypoint_score", ascending=False)
    top_n = int(pget(params, "OUTER_ZONE_WAYPOINT_TOP_N", 30))
    waypoints = [int(v) for v in outer.head(max(1, top_n))["node_id"].tolist() if int(v) not in (int(start_id), int(goal_id))]

    best = None
    best_cost = math.inf
    for wp in waypoints:
        r1 = base_planner.plan_route(start_id, wp, route_key=f"{route_key}_leg1", avoid_edges=used_edges)
        if not r1.success:
            continue
        avoid2 = set(used_edges) | _used_edges_from_path(r1.path)
        r2 = base_planner.plan_route(wp, goal_id, route_key=f"{route_key}_leg2", avoid_edges=avoid2)
        if not r2.success:
            continue
        merged = _merge_two_route_results(r1, r2, route_key)
        if merged.success and merged.total_cost < best_cost:
            best = merged
            best_cost = float(merged.total_cost)

    if best is not None:
        return best

    # Fallback: let the outer-biased graph try any route, even if it cannot
    # force a waypoint. This keeps the workflow from failing when the outer
    # ring is locally disconnected.
    return base_planner.plan_route(start_id, goal_id, route_key=route_key, avoid_edges=used_edges)



def _make_aco_planner(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    best_params: dict[str, float],
    params: SimpleNamespace,
    *,
    random_offset: int = 0,
    require_tn: bool | None = None,
    min_tn_nodes: int | None = None,
    ants_param: str = "ACO_ANTS",
    iterations_param: str = "ACO_ITERATIONS",
    max_steps_param: str = "ACO_MAX_STEPS",
) -> ACOPlanner:
    """Build an ACO planner with the PSO-selected early conditions."""
    return ACOPlanner(
        nodes,
        edges,
        alpha=float(best_params.get("aco_alpha", pget(params, "ACO_ALPHA", 1.2))),
        beta=float(best_params.get("aco_beta", pget(params, "ACO_BETA", 3.0))),
        evaporation=float(best_params.get("aco_evaporation", pget(params, "ACO_EVAPORATION", 0.25))),
        pheromone_q=float(best_params.get("aco_pheromone_q", pget(params, "ACO_PHEROMONE_Q", 1.0))),
        n_ants=int(pget(params, ants_param, pget(params, "ACO_ANTS", 70))),
        n_iterations=int(pget(params, iterations_param, pget(params, "ACO_ITERATIONS", 100))),
        max_steps=int(pget(params, max_steps_param, pget(params, "ACO_MAX_STEPS", 80))),
        random_state=int(pget(params, "ACO_RANDOM_STATE", 42)) + int(random_offset),
        require_tn=bool(pget(params, "REQUIRE_TN_IN_ROUTE", True)) if require_tn is None else bool(require_tn),
        min_tn_nodes=int(pget(params, "MIN_TN_PER_ROUTE", 1)) if min_tn_nodes is None else int(min_tn_nodes),
        missing_tn_penalty=float(pget(params, "ACO_MISSING_TN_PENALTY", 5000.0)),
        avoid_edge_penalty=float(pget(params, "BACKWARD_AVOID_FORWARD_EDGE_PENALTY", 2.5)),
        initial_pheromone_scale=float(pget(params, "ACO_INITIAL_PHEROMONE_SCALE", 1.0)),
        verbose=bool(pget(params, "ACO_VERBOSE", False)),
    )


def _incident_edges_for_nodes(edges: pd.DataFrame, node_ids: Iterable[int]) -> set[tuple[int, int]]:
    node_set = {int(v) for v in node_ids}
    if not node_set:
        return set()
    out: set[tuple[int, int]] = set()
    for row in edges.itertuples(index=False):
        u = int(row.u); v = int(row.v)
        if u in node_set or v in node_set:
            out.add(_edge_key(u, v))
    return out


def _avoid_edges_for_swarm_separation(
    edges: pd.DataFrame,
    used_edges: set[tuple[int, int]],
    used_nodes: set[int],
    params: SimpleNamespace,
) -> set[tuple[int, int]]:
    """Avoid already used corridors/nodes to reduce collision risk in swarm operation."""
    avoid = set(used_edges)
    if bool(pget(params, "SWARM_AVOID_USED_ROUTE_NODES", True)):
        avoid |= _incident_edges_for_nodes(edges, used_nodes)
    return avoid


def _traffic_anchor_nodes(nodes: pd.DataFrame, params: SimpleNamespace) -> pd.DataFrame:
    """Return TA/TN candidate nodes used for scenario maximization.

    The code uses TA as an operational alias for the traffic-node candidates
    produced by KMeans: TN_major and TN_minor.
    """
    roles = {str(v) for v in _as_list(pget(params, "TA_NODE_ROLES", ["TN_major", "TN_minor"]))}
    ta = nodes[nodes["role"].astype(str).isin(roles)].copy()
    # Highest-density major TN first, then lower-density TN.  This makes the
    # first scenarios use the most important traffic anchors, but the loop still
    # tries to cover every TA node when enabled.
    if "route_hit_count" in ta.columns:
        ta = ta.sort_values(["role", "route_hit_count", "name"], ascending=[True, False, True])
    return ta.reset_index(drop=True)


def _nearest_endpoint_pair_for_ta(nodes: pd.DataFrame, ta_row: pd.Series) -> tuple[int, int, str, str]:
    """Choose the nearest DK and DB endpoints for a TA coverage scenario."""
    dk = nodes[nodes["role"] == "DK"].copy()
    db = nodes[nodes["role"] == "DB"].copy()
    if dk.empty or db.empty:
        raise RuntimeError("TA coverage needs at least one DK and one DB node.")
    p = np.asarray([float(ta_row["x_m"]), float(ta_row["y_m"])], dtype=float)
    dk_xy = dk[["x_m", "y_m"]].to_numpy(dtype=float)
    db_xy = db[["x_m", "y_m"]].to_numpy(dtype=float)
    dk_i = int(np.argmin(np.linalg.norm(dk_xy - p[None, :], axis=1)))
    db_i = int(np.argmin(np.linalg.norm(db_xy - p[None, :], axis=1)))
    dk_row = dk.iloc[dk_i]
    db_row = db.iloc[db_i]
    return int(dk_row["node_id"]), int(db_row["node_id"]), str(dk_row["name"]), str(db_row["name"])


def _detour_sorted_candidates(
    nodes: pd.DataFrame,
    role_predicate,
    start_id: int,
    end_id: int,
) -> list[int]:
    """node_ids whose role matches role_predicate(role), sorted ascending by
    dist(start, cand) + dist(cand, end) in metric coordinates -- least
    straight-line detour first. Generalizes 02_run_theta_plan.py's
    replan_main_via_flz() candidate ordering to any node-role predicate, so
    it serves both FLZ and TN candidate pools."""
    cand = nodes[nodes["role"].astype(str).map(role_predicate)]
    cand = cand[~cand["node_id"].astype(int).isin({int(start_id), int(end_id)})]
    if cand.empty:
        return []
    node_lookup = nodes.set_index("node_id")
    sx, sy = float(node_lookup.loc[int(start_id), "x_m"]), float(node_lookup.loc[int(start_id), "y_m"])
    ex, ey = float(node_lookup.loc[int(end_id), "x_m"]), float(node_lookup.loc[int(end_id), "y_m"])
    xy = cand[["x_m", "y_m"]].to_numpy(dtype=float)
    detour = np.hypot(xy[:, 0] - sx, xy[:, 1] - sy) + np.hypot(xy[:, 0] - ex, xy[:, 1] - ey)
    order = np.argsort(detour)
    return [int(v) for v in cand["node_id"].to_numpy(dtype=int)[order]]


def plan_route_via_waypoints(
    planner: ACOPlanner,
    start_id: int,
    waypoint_ids: list[int],
    goal_id: int,
    route_key: str,
    avoid_edges: set[tuple[int, int]],
) -> ACOResult:
    """Plan start -> waypoint_ids[0] -> ... -> waypoint_ids[-1] -> goal.

    Chains independent planner.plan_route() legs, accumulating avoid_edges
    from each successful leg so later legs don't double back over earlier
    ones, and merges all legs into one ACOResult via _merge_two_route_results().
    An empty waypoint_ids list degenerates to a direct start->goal plan_route().
    """
    waypoint_ids = [int(w) for w in waypoint_ids]
    if not waypoint_ids:
        return planner.plan_route(start_id, goal_id, route_key=route_key, avoid_edges=avoid_edges)

    stops = [int(start_id)] + waypoint_ids + [int(goal_id)]
    running_avoid = set(avoid_edges)
    merged: ACOResult | None = None
    for leg_i, (a, b) in enumerate(zip(stops[:-1], stops[1:]), start=1):
        leg = planner.plan_route(a, b, route_key=f"{route_key}_leg{leg_i}", avoid_edges=running_avoid)
        if not leg.success:
            return ACOResult(False, route_key, [], math.inf, math.inf, [], leg.iterations, leg.ants, f"Failed leg{leg_i} ({a}->{b}): {leg.message}")
        running_avoid |= _used_edges_from_path(leg.path)
        merged = leg if merged is None else _merge_two_route_results(merged, leg, route_key)
    return merged


def plan_route_via_waypoint(
    planner: ACOPlanner,
    start_id: int,
    waypoint_id: int,
    goal_id: int,
    route_key: str,
    avoid_edges: set[tuple[int, int]],
) -> ACOResult:
    """Plan start -> waypoint -> goal. Thin wrapper over plan_route_via_waypoints()."""
    return plan_route_via_waypoints(planner, start_id, [waypoint_id], goal_id, route_key, avoid_edges)


def enforce_hard_route_requirements(
    planner: ACOPlanner,
    nodes: pd.DataFrame,
    edge_lookup: dict,
    node_role: dict[int, str],
    result: ACOResult,
    start_id: int,
    goal_id: int,
    route_key: str,
    avoid_edges: set[tuple[int, int]],
    params: SimpleNamespace,
) -> tuple[ACOResult, dict]:
    """Post-process one forward/backward/outer_backup ACO result against the
    hard FLZ-buffer and hard TN-waypoint requirements (mirrors
    02_run_theta_plan.py's FLZ_BUFFER_ENABLE hard requirement).

    Checked AFTER the normal ACO search returns a route. If the route
    already complies, it is returned unchanged. If not, this forces a
    reroute through the nearest (least-detour) compliant node(s) via
    plan_route_via_waypoints() and re-validates BOTH requirements on the
    forced candidate (forcing one must not silently break the other).

    ALWAYS returns a usable ACOResult: if forcing cannot produce a
    compliant route within the retry budget, the ORIGINAL best-effort
    result is kept unchanged -- a route is never dropped -- and the
    returned audit dict reports the failure honestly instead of hiding it.
    """
    flz_on = bool(pget(params, "MASTER_FLZ_BUFFER_ENABLE", False))
    tn_on = bool(pget(params, "REQUIRE_TN_IN_ROUTE_HARD", False))
    audit: dict = {
        "flz_buffer_required": flz_on,
        "flz_buffer_ok": True,
        "flz_buffer_min_distance_m": np.nan,
        "flz_buffer_forced": False,
        "flz_buffer_forced_ok": np.nan,
        "flz_buffer_via": "",
        "tn_hard_required": tn_on,
        "tn_hard_ok": True,
        "tn_hard_count": 0,
        "tn_hard_forced": False,
        "tn_hard_forced_ok": np.nan,
        "tn_hard_via": "",
        "hard_requirement_message": "",
    }
    if not result.success or not (flz_on or tn_on):
        return result, audit

    radius_m = float(pget(params, "MASTER_FLZ_BUFFER_RADIUS_M", 250.0))
    min_tn = int(pget(params, "MIN_TN_PER_ROUTE", 1))

    def _measure(res: ACOResult) -> tuple[bool, float, bool, int]:
        f_dist = _route_flz_min_distance(res, edge_lookup) if flz_on else float("nan")
        f_ok = (not flz_on) or (math.isfinite(f_dist) and f_dist <= radius_m)
        t_count = _route_tn_count(res, node_role) if tn_on else 0
        t_ok = (not tn_on) or (t_count >= min_tn)
        return f_ok, f_dist, t_ok, t_count

    flz_ok, flz_dist, tn_ok, tn_count = _measure(result)
    audit["flz_buffer_ok"], audit["flz_buffer_min_distance_m"] = flz_ok, flz_dist
    audit["tn_hard_ok"], audit["tn_hard_count"] = tn_ok, tn_count
    if flz_ok and tn_ok:
        return result, audit

    tn_missing = tn_on and not tn_ok
    flz_missing = flz_on and not flz_ok
    max_tries = int(pget(params, "MASTER_HARD_REQUIREMENT_MAX_TRIES", 12))
    max_tries_both = int(pget(params, "MASTER_HARD_REQUIREMENT_MAX_TRIES_BOTH_MISSING", 4))

    tn_pool = _detour_sorted_candidates(nodes, lambda r: r.startswith("TN"), start_id, goal_id) if tn_missing else []
    flz_pool = _detour_sorted_candidates(nodes, lambda r: r == "FLZ", start_id, goal_id) if flz_missing else []

    if tn_missing and flz_missing:
        # TN before FLZ: TN is the harder topological constraint, and
        # visiting it first keeps the FLZ-penalized final leg short
        # (apply_edge_weights already adds flz_penalty to FLZ-touching edges).
        waypoint_trials = [(t, f) for t in tn_pool[:max_tries_both] for f in flz_pool[:max_tries_both]]
    elif tn_missing:
        waypoint_trials = [(t,) for t in tn_pool[:max_tries]]
    else:
        waypoint_trials = [(f,) for f in flz_pool[:max_tries]]

    node_name = nodes.set_index("node_id")["name"].astype(str).to_dict()
    forced_result: ACOResult | None = None
    forced_tn_name = ""
    forced_flz_name = ""
    last_message = "No forcing candidate produced a compliant route."

    for waypoints in waypoint_trials:
        cand = plan_route_via_waypoints(planner, start_id, list(waypoints), goal_id, f"{route_key}_forced", avoid_edges)
        if not cand.success:
            last_message = cand.message
            continue
        c_flz_ok, c_flz_dist, c_tn_ok, c_tn_count = _measure(cand)
        if c_flz_ok and c_tn_ok:
            forced_result = cand
            for w in waypoints:
                role = str(node_role.get(int(w), ""))
                if role.startswith("TN"):
                    forced_tn_name = node_name.get(int(w), "")
                elif role == "FLZ":
                    forced_flz_name = node_name.get(int(w), "")
            audit["flz_buffer_ok"], audit["flz_buffer_min_distance_m"] = c_flz_ok, c_flz_dist
            audit["tn_hard_ok"], audit["tn_hard_count"] = c_tn_ok, c_tn_count
            break
        last_message = "Forced candidate still failed the other requirement."

    audit["flz_buffer_forced"] = flz_missing
    audit["tn_hard_forced"] = tn_missing
    audit["flz_buffer_via"] = forced_flz_name
    audit["tn_hard_via"] = forced_tn_name

    if forced_result is not None:
        audit["flz_buffer_forced_ok"] = True if flz_missing else np.nan
        audit["tn_hard_forced_ok"] = True if tn_missing else np.nan
        audit["hard_requirement_message"] = f"Forced via TN={forced_tn_name or '-'} FLZ={forced_flz_name or '-'}."
        forced_result.message = f"{forced_result.message} {audit['hard_requirement_message']}"
        return forced_result, audit

    audit["flz_buffer_forced_ok"] = False if flz_missing else np.nan
    audit["tn_hard_forced_ok"] = False if tn_missing else np.nan
    audit["hard_requirement_message"] = f"Hard requirement forcing FAILED, kept best-effort route. {last_message}"
    return result, audit


def _nearest_flz_branch(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    branch_planner: ACOPlanner,
    parent_result: ACOResult,
    parent_direction: str,
    used_edges: set[tuple[int, int]],
    params: SimpleNamespace,
) -> ACOResult | None:
    """Create one emergency branch from the route to a nearby FLZ.

    The branch is attached to the nearest reachable route node / FLZ pair.  It
    is not forced through TN because this is an emergency safe-parking branch.
    """
    if not bool(pget(params, "ENABLE_FLZ_BRANCH_PER_ROUTE", True)):
        return None
    if not parent_result.success or len(parent_result.path) == 0:
        return None
    flz = nodes[nodes["role"] == "FLZ"].copy()
    if flz.empty:
        return None

    node_lookup = nodes.set_index("node_id")
    candidates: list[tuple[float, int, int, str]] = []
    for n in parent_result.path:
        n = int(n)
        if n not in node_lookup.index:
            continue
        if str(node_lookup.loc[n, "role"]) == "FLZ":
            continue
        p = node_lookup.loc[n, ["x_m", "y_m"]].to_numpy(dtype=float)
        for _, fr in flz.iterrows():
            f_id = int(fr["node_id"])
            q = np.asarray([float(fr["x_m"]), float(fr["y_m"])], dtype=float)
            d = float(np.linalg.norm(p - q))
            candidates.append((d, n, f_id, str(fr["name"])))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    max_try = int(pget(params, "FLZ_BRANCH_MAX_ATTACH_TRIES", 12))
    max_dist = float(pget(params, "FLZ_BRANCH_MAX_DISTANCE_M", 2500.0))
    for d, start_id, flz_id, flz_name in candidates[:max(1, max_try)]:
        if math.isfinite(max_dist) and max_dist > 0 and d > max_dist:
            continue
        key = f"{parent_result.route_key}_to_{flz_name}_FLZ_branch".replace(" ", "_")
        r = branch_planner.plan_route(start_id, flz_id, route_key=key, avoid_edges=used_edges)
        if r.success:
            r.message = f"FLZ branch for {parent_result.route_key}; attach_node={start_id}; parent_direction={parent_direction}."
            return r
    return ACOResult(False, f"{parent_result.route_key}_FLZ_branch", [], math.inf, math.inf, [], 0, 0, "No reachable FLZ branch found.")


def run_aco_routes(nodes: pd.DataFrame, edges: pd.DataFrame, pairs: list[tuple[int, int, str, str]], best_params: dict[str, float], params: SimpleNamespace):
    """Run all master-route scenarios.

    v8 adds terminal progress bars for the long ACO section:
        1. base pair routes: forward, backward, and optional outer backup
        2. TA/TN coverage scenarios
        3. FLZ emergency branches

    The progress backend uses tqdm when available.  If tqdm is not installed,
    the code falls back to a simple text percentage printer.
    """

    class _NullProgress:
        def update(self, n: int = 1, label: str = "") -> None:
            return
        def close(self) -> None:
            return

    class _TextProgress:
        def __init__(self, total: int, desc: str, every: int = 1) -> None:
            self.total = max(1, int(total))
            self.desc = str(desc)
            self.count = 0
            self.every = max(1, int(every))
            print(f"[{self.desc}] 0/{self.total} (0.0%)")
        def update(self, n: int = 1, label: str = "") -> None:
            self.count += int(n)
            if self.count < self.total and (self.count % self.every) != 0:
                return
            pct = 100.0 * min(self.count, self.total) / self.total
            msg = f"[{self.desc}] {min(self.count, self.total)}/{self.total} ({pct:5.1f}%)"
            if label:
                msg += f" | {label}"
            print(msg)
        def close(self) -> None:
            if self.count < self.total:
                self.count = self.total
                self.update(0, "done")

    class _TqdmProgress:
        def __init__(self, total: int, desc: str) -> None:
            from tqdm.auto import tqdm
            self.bar = tqdm(total=max(1, int(total)), desc=str(desc), unit="route", dynamic_ncols=True)
        def update(self, n: int = 1, label: str = "") -> None:
            if label:
                self.bar.set_postfix_str(str(label)[:80])
            self.bar.update(int(n))
        def close(self) -> None:
            self.bar.close()

    def _make_progress(total: int, desc: str):
        if not bool(pget(params, "ACO_PROGRESS_BAR", True)) or int(total) <= 0:
            return _NullProgress()
        backend = str(pget(params, "ACO_PROGRESS_BACKEND", "auto")).lower().strip()
        if backend in ("auto", "tqdm"):
            try:
                return _TqdmProgress(total, desc)
            except Exception:
                if backend == "tqdm":
                    print(f"[routerplan] tqdm progress backend failed for {desc}; using text progress.")
        every = int(pget(params, "ACO_PROGRESS_TEXT_EVERY_N", 1))
        return _TextProgress(total, desc, every=every)

    planner = _make_aco_planner(nodes, edges, best_params, params, random_offset=0)
    node_role_hard = dict(zip(nodes["node_id"].astype(int), nodes["role"].astype(str)))
    edge_lookup_hard = {_edge_key(row.u, row.v): row for row in edges.itertuples(index=False)}
    branch_planner = _make_aco_planner(
        nodes,
        edges,
        best_params,
        params,
        random_offset=5000,
        require_tn=False,
        min_tn_nodes=0,
        ants_param="FLZ_BRANCH_ACO_ANTS",
        iterations_param="FLZ_BRANCH_ACO_ITERATIONS",
        max_steps_param="FLZ_BRANCH_ACO_MAX_STEPS",
    )

    summaries: list[dict] = []
    edge_tables: list[pd.DataFrame] = []
    aco_histories: dict[str, pd.DataFrame] = {}
    used_edges: set[tuple[int, int]] = set()
    used_nodes: set[int] = set()
    operational_results: list[tuple[ACOResult, str, str, str]] = []  # result, direction, dk_name, db_name

    def _append_route(result: ACOResult, direction: str, dk_name: str, db_name: str, edge_source: pd.DataFrame | None = None, extra_summary_fields: dict | None = None):
        nonlocal used_edges, used_nodes
        edge_source = edges if edge_source is None else edge_source
        summaries.append(_route_summary(result, nodes, edge_source, direction, dk_name, db_name))
        if extra_summary_fields:
            summaries[-1].update(extra_summary_fields)
        edge_tables.append(_route_edges_table(result, nodes, edge_source, direction, dk_name, db_name))
        if getattr(result, "history_rows", None):
            aco_histories[str(result.route_key)] = pd.DataFrame(list(result.history_rows))
        if result.success:
            used_edges |= _used_edges_from_path(result.path)
            used_nodes |= {int(v) for v in result.path}
            if direction != "flz_branch":
                operational_results.append((result, direction, dk_name, db_name))

    enable_outer = bool(pget(params, "ENABLE_OUTER_ZONE_BACKUP_ROUTE", True))
    base_total = len(pairs) * (2 + int(enable_outer))
    base_bar = _make_progress(base_total, "ACO base pair routes")

    # ------------------------------------------------------------------
    # 1) Normal forward/backward/outer route layers for every DK-DB pair.
    # ------------------------------------------------------------------
    try:
        for dk_id, db_id, dk_name, db_name in pairs:
            pair_name = f"{dk_name}_to_{db_name}".replace(" ", "_")
            fwd_key = f"{pair_name}_forward"
            bwd_key = f"{db_name}_to_{dk_name}_backward".replace(" ", "_")

            avoid_fwd = _avoid_edges_for_swarm_separation(edges, used_edges, used_nodes, params) if bool(pget(params, "SWARM_SEPARATE_ALL_ROUTES", True)) else set()
            fwd = planner.plan_route(dk_id, db_id, route_key=fwd_key, avoid_edges=avoid_fwd)
            fwd, fwd_audit = enforce_hard_route_requirements(planner, nodes, edge_lookup_hard, node_role_hard, fwd, dk_id, db_id, fwd_key, avoid_fwd, params)
            _append_route(fwd, "forward", dk_name, db_name, extra_summary_fields=fwd_audit)
            base_bar.update(1, f"{dk_name}->{db_name} forward")

            avoid_bwd = set(_used_edges_from_path(fwd.path)) if bool(pget(params, "ACO_BACKWARD_AVOID_FORWARD_EDGES", True)) else set()
            if bool(pget(params, "SWARM_SEPARATE_ALL_ROUTES", True)):
                avoid_bwd |= _avoid_edges_for_swarm_separation(edges, used_edges, used_nodes, params)
            bwd = planner.plan_route(db_id, dk_id, route_key=bwd_key, avoid_edges=avoid_bwd)
            bwd, bwd_audit = enforce_hard_route_requirements(planner, nodes, edge_lookup_hard, node_role_hard, bwd, db_id, dk_id, bwd_key, avoid_bwd, params)
            _append_route(bwd, "backward", dk_name, db_name, extra_summary_fields=bwd_audit)
            base_bar.update(1, f"{db_name}->{dk_name} backward")

            if enable_outer:
                used = set()
                if bool(pget(params, "OUTER_ZONE_AVOID_INNER_ROUTES", True)):
                    used |= _used_edges_from_path(fwd.path)
                    used |= _used_edges_from_path(bwd.path)
                if bool(pget(params, "SWARM_SEPARATE_ALL_ROUTES", True)):
                    used |= _avoid_edges_for_swarm_separation(edges, used_edges, used_nodes, params)
                outer_edges = _outer_biased_edges(edges, used, params)
                outer_planner = _make_aco_planner(
                    nodes,
                    outer_edges,
                    best_params,
                    params,
                    random_offset=1000,
                    require_tn=True,
                    min_tn_nodes=max(1, int(pget(params, "OUTER_ZONE_MIN_TN_WAYPOINTS", 1))),
                    ants_param="OUTER_ZONE_ACO_ANTS",
                    iterations_param="OUTER_ZONE_ACO_ITERATIONS",
                    max_steps_param="OUTER_ZONE_ACO_MAX_STEPS",
                )
                outer_key = f"{pair_name}_outer_backup"
                outer = plan_outer_backup_route(nodes, outer_edges, outer_planner, dk_id, db_id, outer_key, used, params)
                outer, outer_audit = enforce_hard_route_requirements(outer_planner, nodes, edge_lookup_hard, node_role_hard, outer, dk_id, db_id, outer_key, used, params)
                _append_route(outer, "outer_backup", dk_name, db_name, outer_edges, extra_summary_fields=outer_audit)
                base_bar.update(1, f"{dk_name}->{db_name} outer_backup")
    finally:
        base_bar.close()

    # ------------------------------------------------------------------
    # 2) Maximize TA/TN use: create a route scenario through every TA node.
    # ------------------------------------------------------------------
    if bool(pget(params, "ENABLE_TA_COVERAGE_SCENARIOS", True)):
        ta_nodes = _traffic_anchor_nodes(nodes, params)
        max_ta = int(pget(params, "MAX_TA_COVERAGE_SCENARIOS", 0))
        if max_ta > 0:
            ta_nodes = ta_nodes.head(max_ta)
        ta_planner = _make_aco_planner(
            nodes,
            edges,
            best_params,
            params,
            random_offset=2000,
            require_tn=False,
            min_tn_nodes=0,
            ants_param="TA_COVERAGE_ACO_ANTS",
            iterations_param="TA_COVERAGE_ACO_ITERATIONS",
            max_steps_param="TA_COVERAGE_ACO_MAX_STEPS",
        )
        print(f"[routerplan] TA/TN coverage scenarios requested: {len(ta_nodes):,}")
        ta_bar = _make_progress(len(ta_nodes), "ACO TA/TN coverage")
        try:
            for k, (_, ta) in enumerate(ta_nodes.iterrows(), start=1):
                ta_id = int(ta["node_id"])
                ta_name = str(ta["name"])
                dk_id, db_id, dk_name, db_name = _nearest_endpoint_pair_for_ta(nodes, ta)
                avoid = set()
                if bool(pget(params, "TA_COVERAGE_AVOID_USED_ROUTES", True)):
                    avoid |= _avoid_edges_for_swarm_separation(edges, used_edges, used_nodes, params)
                route_key = f"TA{k:03d}_{ta_name}_{dk_name}_to_{db_name}".replace(" ", "_")
                result = plan_route_via_waypoint(ta_planner, dk_id, ta_id, db_id, route_key, avoid)
                if result.success:
                    result.message = f"TA coverage scenario through {ta_name}. " + str(result.message)
                _append_route(result, "ta_coverage", dk_name, db_name)
                ta_bar.update(1, f"{ta_name}: {dk_name}->{db_name}")
        finally:
            ta_bar.close()

    # ------------------------------------------------------------------
    # 3) Add one FLZ emergency branch for every successful operational route.
    # ------------------------------------------------------------------
    if bool(pget(params, "ENABLE_FLZ_BRANCH_PER_ROUTE", True)):
        print(f"[routerplan] Adding FLZ branch for successful operational routes: {len(operational_results):,}")
        flz_bar = _make_progress(len(operational_results), "ACO FLZ branches")
        try:
            # Iterate over a copy because _append_route changes route tables.
            for result, direction, dk_name, db_name in list(operational_results):
                branch_avoid = set()
                if bool(pget(params, "FLZ_BRANCH_AVOID_USED_ROUTES", False)):
                    branch_avoid |= _avoid_edges_for_swarm_separation(edges, used_edges, used_nodes, params)
                branch = _nearest_flz_branch(nodes, edges, branch_planner, result, direction, branch_avoid, params)
                if branch is not None:
                    _append_route(branch, "flz_branch", dk_name, db_name)
                flz_bar.update(1, f"{result.route_key}")
        finally:
            flz_bar.close()

    route_summary = pd.DataFrame(summaries)
    route_edges = pd.concat(edge_tables, ignore_index=True) if edge_tables else pd.DataFrame()
    return route_summary, route_edges, aco_histories



# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------




def _safe_name(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("_") or "item"


def _parse_path_ids(text: str) -> list[int]:
    if text is None or (isinstance(text, float) and not np.isfinite(text)):
        return []
    s = str(text).strip()
    if not s:
        return []
    out = []
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(float(part)))
        except Exception:
            continue
    return out


def _pair_mask(df: pd.DataFrame, db_label: str, dk_label: str) -> pd.Series:
    db_col = df.get("db_name", pd.Series("", index=df.index)).astype(str).str.upper()
    dk_col = df.get("dk_name", pd.Series("", index=df.index)).astype(str).str.upper()
    return (db_col == str(db_label).upper()) & (dk_col == str(dk_label).upper())


def make_pso_snapshots(pso_history: pd.DataFrame, output_dir: Path, params: SimpleNamespace) -> None:
    if pso_history is None or pso_history.empty or not bool(pget(params, "PLOT_PSO_SNAPSHOTS", True)):
        return
    import matplotlib.pyplot as plt
    snap_dir = output_dir / "figures" / "PSO_snap"
    snap_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(pget(params, "FIGURE_DPI", 220))
    every = max(1, int(pget(params, "PSO_SNAPSHOT_EVERY_N_ITER", 1)))
    cols = [c for c in pso_history.columns if c.startswith("best_") and c not in {"best_fitness"}]
    for i in range(len(pso_history)):
        if i != len(pso_history)-1 and (i % every) != 0:
            continue
        sub = pso_history.iloc[:i+1]
        last = sub.iloc[-1]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(sub["iteration"] + 1, sub["best_fitness"], marker="o", lw=1.5)
        ax.set_title(f"PSO snapshot iteration {int(last['iteration'])+1}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best fitness")
        text_lines = [f"{c.replace('best_','')}: {float(last[c]):.3f}" for c in cols[:8]]
        ax.text(1.02, 0.98, "\n".join(text_lines), transform=ax.transAxes, va="top", ha="left", fontsize=8)
        fig.tight_layout()
        fig.savefig(snap_dir / f"PSO_snap_{int(last['iteration'])+1:03d}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def make_aco_snapshots(model: ModelData, nodes: pd.DataFrame, output_dir: Path, params: SimpleNamespace) -> None:
    if not bool(pget(params, "PLOT_ACO_SNAPSHOTS", True)):
        return
    hist_dir = output_dir / "aco_history"
    if not hist_dir.exists():
        return
    import matplotlib.pyplot as plt
    snap_dir = output_dir / "figures" / "ACO_snap"
    snap_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(pget(params, "FIGURE_DPI", 220))
    every = max(1, int(pget(params, "ACO_SNAPSHOT_EVERY_N_ITER", 5)))
    db_label = str(pget(params, "SINGLE_PAIR_DB_LABEL", "DB01"))
    dk_label = str(pget(params, "SINGLE_PAIR_DK_LABEL", "DK04"))
    allowed_keys = None
    route_summary_file = output_dir / "master_routes.csv"
    if route_summary_file.exists():
        rs = pd.read_csv(route_summary_file)
        allowed_keys = set(rs.loc[_pair_mask(rs, db_label, dk_label), "route_key"].astype(str))
    node_lookup = nodes.set_index("node_id").to_dict("index")
    for csv_file in sorted(hist_dir.glob("*.csv")):
        h = pd.read_csv(csv_file)
        route_key = str(h.get("route_key", pd.Series([csv_file.stem])).iloc[0])
        if allowed_keys is not None and len(allowed_keys) > 0 and route_key not in allowed_keys:
            continue
        for i, row in h.iterrows():
            if i != len(h)-1 and (i % every) != 0:
                continue
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.scatter(model.xy_original[model.flyable_mask, 0], model.xy_original[model.flyable_mask, 1], s=2, c="lightgray", alpha=0.20, linewidths=0)
            ax.scatter(model.xy_original[model.nofly_mask, 0], model.xy_original[model.nofly_mask, 1], s=3, c="black", alpha=0.22, linewidths=0)
            path = _parse_path_ids(row.get("best_path_node_ids", ""))
            if len(path) >= 2:
                xs=[]; ys=[]
                for nid in path:
                    nd = node_lookup.get(int(nid), {})
                    xs.append(float(nd.get("x", np.nan))); ys.append(float(nd.get("y", np.nan)))
                ax.plot(xs, ys, lw=2.2, c="tab:blue", alpha=0.95, label="current best path", zorder=6)
            for role, marker, size in [("DB", "s", 90), ("DK", "^", 90), ("FLZ", "P", 80), ("TN_major", "*", 110), ("TN_minor", "o", 60)]:
                sub = nodes[nodes["role"].astype(str) == role]
                if not sub.empty:
                    ax.scatter(sub["x"], sub["y"], marker=marker, s=size, edgecolors="white", linewidths=0.5, label=role, zorder=5)
            ax.set_title(f"ACO snapshot: {route_key} iter {int(row.get('iteration', i))+1}")
            ax.set_xlabel(model.x_col); ax.set_ylabel(model.y_col); ax.set_aspect("equal", adjustable="box")
            ax.legend(loc="best", fontsize=8, frameon=True)
            fig.tight_layout()
            fig.savefig(snap_dir / f"{_safe_name(route_key)}_iter{int(row.get('iteration', i))+1:03d}.png", dpi=dpi)
            plt.close(fig)


def make_single_pair_corridor_figures(model: ModelData, nodes: pd.DataFrame, route_edges: pd.DataFrame, output_dir: Path, params: SimpleNamespace) -> None:
    """Create corridor figures grouped by DK-DB pair.

    v6 behavior:
        - Default is all pairs: one figure per DK-DB pair.
        - Each pair figure contains all routes for that pair: forward, backward,
          outer_backup, ta_coverage, and flz_branch.
        - Optional single-pair mode remains available for debugging only.
    """
    enabled = bool(pget(params, "PLOT_PAIR_CORRIDORS", pget(params, "PLOT_SINGLE_PAIR_CORRIDORS", True)))
    if route_edges is None or route_edges.empty or not enabled:
        return

    import matplotlib.pyplot as plt

    corr_dir = output_dir / "figures" / "corridors"
    corr_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(pget(params, "FIGURE_DPI", 220))

    pair_mode = str(pget(params, "CORRIDOR_PAIR_MODE", "all")).strip().lower()
    db_label = str(pget(params, "SINGLE_PAIR_DB_LABEL", "DB01"))
    dk_label = str(pget(params, "SINGLE_PAIR_DK_LABEL", "DK04"))

    work = route_edges.copy()
    if "dk_name" not in work.columns or "db_name" not in work.columns:
        print("[routerplan] Cannot make pair corridor figures: route_edges lacks dk_name/db_name columns.")
        return

    if pair_mode == "single":
        work = work.loc[_pair_mask(work, db_label, dk_label)].copy()
        if work.empty:
            print(f"[routerplan] No routes found for single pair figure: {dk_label} <-> {db_label}")
            return
        pair_groups = [(dk_label, db_label, work)]
    else:
        pair_groups = []
        pair_table = work[["dk_name", "db_name"]].drop_duplicates().sort_values(["dk_name", "db_name"])
        for pr in pair_table.itertuples(index=False):
            dk = str(pr.dk_name)
            db = str(pr.db_name)
            sub = work.loc[_pair_mask(work, db, dk)].copy()
            if not sub.empty:
                pair_groups.append((dk, db, sub))

    if not pair_groups:
        print("[routerplan] No DK-DB pair route groups found for corridor figures.")
        return

    style_map = {
        "forward": dict(lw=2.0, ls="-", alpha=0.90, c="tab:blue"),
        "backward": dict(lw=1.5, ls="--", alpha=0.85, c="tab:orange"),
        "outer_backup": dict(lw=2.0, ls=":", alpha=0.90, c="tab:purple"),
        "ta_coverage": dict(lw=1.0, ls="-", alpha=0.55, c="tab:green"),
        "flz_branch": dict(lw=1.2, ls="-.", alpha=0.80, c="tab:red"),
    }

    def _plot_nodes(ax):
        for role, marker, size in [("DB", "s", 90), ("DK", "^", 90), ("FLZ", "P", 80), ("TN_major", "*", 110), ("TN_minor", "o", 60)]:
            nsub = nodes[nodes["role"].astype(str) == role]
            if not nsub.empty:
                ax.scatter(nsub["x"], nsub["y"], marker=marker, s=size, edgecolors="white", linewidths=0.5, label=role, zorder=5)
                if bool(pget(params, "PLOT_PAIR_CORRIDOR_NODE_LABELS", False)):
                    for r in nsub.itertuples(index=False):
                        ax.text(float(r.x), float(r.y), str(r.name), fontsize=6, ha="left", va="bottom")

    max_pairs = int(pget(params, "MAX_PAIR_CORRIDOR_FIGURES", 0))
    if max_pairs > 0:
        pair_groups = pair_groups[:max_pairs]

    print(f"[routerplan] Pair corridor figures: {len(pair_groups):,} pair(s), mode={pair_mode}")
    for dk, db, sub in pair_groups:
        fig, ax = plt.subplots(figsize=(10, 8))
        if bool(pget(params, "PLOT_NOFLY_IN_PAIR_CORRIDORS", True)):
            ax.scatter(model.xy_original[model.nofly_mask, 0], model.xy_original[model.nofly_mask, 1], s=3, c="black", alpha=0.22, linewidths=0, label="no-fly / RA / obstacle")
        if bool(pget(params, "PLOT_FLYABLE_IN_PAIR_CORRIDORS", False)):
            ax.scatter(model.xy_original[model.flyable_mask, 0], model.xy_original[model.flyable_mask, 1], s=2, c="lightgray", alpha=0.18, linewidths=0, label="flyable raw nodes")

        # Plot all routes belonging to this pair in the same figure.
        for row in sub.itertuples(index=False):
            st = style_map.get(str(row.direction), dict(lw=1.0, ls="-", alpha=0.6, c="gray"))
            ax.plot([row.x1, row.x2], [row.y1, row.y2], **st)

        _plot_nodes(ax)
        route_count = int(sub["route_key"].nunique()) if "route_key" in sub.columns else 0
        ax.set_title(f"Pair corridor plan: {dk} ↔ {db} ({route_count} routes)")
        ax.set_xlabel(model.x_col)
        ax.set_ylabel(model.y_col)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="best", fontsize=8, frameon=True)
        fig.tight_layout()
        fig.savefig(corr_dir / f"{_safe_name(dk)}_to_{_safe_name(db)}_all_routes.png", dpi=dpi)
        plt.close(fig)

        # Optional: also save one figure per route key inside this pair.
        if bool(pget(params, "PLOT_EACH_ROUTE_IN_PAIR_CORRIDORS", False)):
            route_dir = corr_dir / f"{_safe_name(dk)}_to_{_safe_name(db)}_routes"
            route_dir.mkdir(parents=True, exist_ok=True)
            for route_key, grp in sub.groupby("route_key"):
                fig, ax = plt.subplots(figsize=(10, 8))
                if bool(pget(params, "PLOT_NOFLY_IN_PAIR_CORRIDORS", True)):
                    ax.scatter(model.xy_original[model.nofly_mask, 0], model.xy_original[model.nofly_mask, 1], s=3, c="black", alpha=0.22, linewidths=0, label="no-fly / RA / obstacle")
                for row in grp.itertuples(index=False):
                    st = style_map.get(str(row.direction), dict(lw=1.0, ls="-", alpha=0.6, c="gray"))
                    ax.plot([row.x1, row.x2], [row.y1, row.y2], **st)
                _plot_nodes(ax)
                ax.set_title(f"Route corridor: {route_key}")
                ax.set_xlabel(model.x_col)
                ax.set_ylabel(model.y_col)
                ax.set_aspect("equal", adjustable="box")
                ax.legend(loc="best", fontsize=8, frameon=True)
                fig.tight_layout()
                fig.savefig(route_dir / f"{_safe_name(route_key)}.png", dpi=dpi)
                plt.close(fig)


def make_figures(model: ModelData, nodes: pd.DataFrame, edges: pd.DataFrame, route_edges: pd.DataFrame, output_dir: Path, params: SimpleNamespace) -> None:
    if not bool(pget(params, "MAKE_FIGURES", True)):
        return
    import matplotlib.pyplot as plt

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(pget(params, "FIGURE_DPI", 220))

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(model.xy_original[model.flyable_mask, 0], model.xy_original[model.flyable_mask, 1], s=2, c="lightgray", alpha=0.35, linewidths=0, label="flyable raw nodes")
    ax.scatter(model.xy_original[model.nofly_mask, 0], model.xy_original[model.nofly_mask, 1], s=3, c="black", alpha=0.25, linewidths=0, label="no-fly / RA / obstacle")

    # Reduced graph edges in background.
    if bool(pget(params, "PLOT_MASTER_GRAPH_EDGES", False)):
        max_edges = int(pget(params, "PLOT_MAX_GRAPH_EDGES", 1000))
        for row in edges.head(max_edges).itertuples(index=False):
            ax.plot([row.x1, row.x2], [row.y1, row.y2], lw=0.25, alpha=0.15, c="gray")

    # Planned route edges.  The style map keeps all route layers in one plan:
    # forward/backward lanes, outer backup, TA coverage scenarios, and FLZ branches.
    style_map = {
        "forward": dict(lw=1.8, ls="-", alpha=0.85, c="tab:blue"),
        "backward": dict(lw=1.3, ls="--", alpha=0.80, c="tab:orange"),
        "outer_backup": dict(lw=2.0, ls=":", alpha=0.90, c="tab:purple"),
        "ta_coverage": dict(lw=0.9, ls="-", alpha=0.45, c="tab:green"),
        "flz_branch": dict(lw=1.1, ls="-.", alpha=0.80, c="tab:red"),
    }
    for row in route_edges.itertuples(index=False):
        st = style_map.get(str(row.direction), dict(lw=1.0, ls="-", alpha=0.60, c="gray"))
        ax.plot([row.x1, row.x2], [row.y1, row.y2], **st)

    roles = nodes["role"].astype(str)
    for role, marker, size in [("DB", "s", 90), ("DK", "^", 90), ("FLZ", "P", 80), ("TN_major", "*", 110), ("TN_minor", "o", 60)]:
        sub = nodes[roles == role]
        if not sub.empty:
            ax.scatter(sub["x"], sub["y"], marker=marker, s=size, edgecolors="white", linewidths=0.5, label=role, zorder=5)
            if bool(pget(params, "PLOT_NODE_LABELS", True)):
                for r in sub.itertuples(index=False):
                    ax.text(float(r.x), float(r.y), str(r.name), fontsize=6, ha="left", va="bottom")

    ax.set_title("PSO-seeded ACO master route scenario network with TA coverage and FLZ branches")
    ax.set_xlabel(model.x_col)
    ax.set_ylabel(model.y_col)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(fig_dir / "01_master_routes_pso_aco.png", dpi=dpi)
    plt.close(fig)

    # Cleaner single-plan figure: route network + graph nodes, without raw-node cloud.
    fig, ax = plt.subplots(figsize=(10, 8))
    for row in route_edges.itertuples(index=False):
        st = style_map.get(str(row.direction), dict(lw=1.0, ls="-", alpha=0.60, c="gray"))
        ax.plot([row.x1, row.x2], [row.y1, row.y2], **st)

    for role, marker, size in [("DB", "s", 90), ("DK", "^", 90), ("FLZ", "P", 85), ("TN_major", "*", 110), ("TN_minor", "o", 55)]:
        sub = nodes[roles == role]
        if not sub.empty:
            ax.scatter(sub["x"], sub["y"], marker=marker, s=size, edgecolors="white", linewidths=0.5, label=role, zorder=5)
            if bool(pget(params, "PLOT_NODE_LABELS_SINGLE_PLAN", False)):
                for r in sub.itertuples(index=False):
                    ax.text(float(r.x), float(r.y), str(r.name), fontsize=6, ha="left", va="bottom")

    ax.set_title("Single master plan: TA/TN scenarios and FLZ emergency branches")
    ax.set_xlabel(model.x_col)
    ax.set_ylabel(model.y_col)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(fig_dir / "02_single_master_plan_TA_FLZ.png", dpi=dpi)
    plt.close(fig)


# ----------------------------------------------------------------------
# Main workflow
# ----------------------------------------------------------------------




def run_master_plan(params_file: str | Path | SimpleNamespace) -> None:
    params = load_params(params_file) if not isinstance(params_file, SimpleNamespace) else params_file
    output_dir = Path(pget(params, "OUTPUT_DIR", "output/04_router_master_plan"))
    output_dir.mkdir(parents=True, exist_ok=True)
    if Path(str(pget(params, "PARAMS_FILE", ""))).exists():
        shutil.copy2(str(pget(params, "PARAMS_FILE")), output_dir / "routeplan_PSO_ACO.params.snapshot")

    run_mode = str(pget(params, "RUN_MODE", "all")).strip().lower()
    print(f"[routerplan] src.routeplan_PSO_ACO version: {MODULE_VERSION}")
    print("[routerplan] Loading KMeans node/objective/TN candidate table...")
    model = load_node_model(params)
    print(f"[routerplan] Raw nodes: {len(model.df):,}; no-fly/RA/obstacle: {int(model.nofly_mask.sum()):,}; coordinate mode: {model.xy_mode}")

    if run_mode == "plot_only":
        print("[routerplan] RUN_MODE=plot_only: skip PSO+ACO computation and regenerate figures from saved outputs.")
        nodes = pd.read_csv(output_dir / "master_graph_nodes.csv")
        edges = pd.read_csv(output_dir / "master_graph_edges.csv")
        route_edges = pd.read_csv(output_dir / "master_route_edges.csv")
        pso_history = pd.read_csv(output_dir / "pso_history.csv") if (output_dir / "pso_history.csv").exists() else pd.DataFrame()
        make_figures(model, nodes, edges, route_edges, output_dir, params)
        make_pso_snapshots(pso_history, output_dir, params)
        make_aco_snapshots(model, nodes, output_dir, params)
        make_single_pair_corridor_figures(model, nodes, route_edges, output_dir, params)
        print(f"[routerplan] Plot-only figures saved under: {output_dir / 'figures'}")
        return

    print("[routerplan] Building master graph nodes...")
    nodes = build_master_nodes(model, params)
    nodes = annotate_outer_zone_nodes(model, nodes, params)
    print(nodes["role"].value_counts().to_string())
    if "is_outer_zone_node" in nodes.columns:
        print(f"[routerplan] Outer-zone TN/backup nodes: {int(nodes['is_outer_zone_node'].sum()):,}")

    print("[routerplan] Building master graph edges with no-fly clearance checks...")
    base_edges = build_master_edges(model, nodes, params)
    base_edges = annotate_outer_zone_edges(base_edges, nodes, params)
    print(f"[routerplan] Master graph edges: {len(base_edges):,}")
    if "is_outer_zone_edge" in base_edges.columns:
        print(f"[routerplan] Outer-zone backup edges: {int(base_edges['is_outer_zone_edge'].sum()):,}")

    pairs = build_route_pairs(nodes, params)
    print(f"[routerplan] DK-DB route pairs: {len(pairs):,}; base routes before TA/FLZ expansion: {2 * len(pairs):,}")

    print("[routerplan] Running PSO to tune/seed ACO variables...")
    variables = bounds_from_params(params)
    fitness_fn = lambda w: evaluate_pso_particle(nodes, base_edges, pairs, params, w)
    pso = PSOOptimizer(
        variables,
        fitness_fn,
        n_particles=int(pget(params, "PSO_PARTICLES", 24)),
        n_iterations=int(pget(params, "PSO_ITERATIONS", 40)),
        inertia=float(pget(params, "PSO_INERTIA", 0.72)),
        cognitive=float(pget(params, "PSO_COGNITIVE", 1.45)),
        social=float(pget(params, "PSO_SOCIAL", 1.45)),
        random_state=int(pget(params, "PSO_RANDOM_STATE", 42)),
        verbose=bool(pget(params, "PSO_VERBOSE", True)),
    )
    best, pso_history = pso.optimize()
    print("[routerplan] Best PSO/ACO variables:")
    for k, v in best.items():
        print(f"    {k}: {v}")

    edges = apply_edge_weights(base_edges, best, params)

    print("[routerplan] Running ACO for forward/backward, outer backup, TA coverage, and FLZ branch routes...")
    route_summary, route_edges, aco_histories = run_aco_routes(nodes, edges, pairs, best, params)

    nodes.to_csv(output_dir / "master_graph_nodes.csv", index=False)
    edges.to_csv(output_dir / "master_graph_edges.csv", index=False)
    route_summary.to_csv(output_dir / "master_routes.csv", index=False)
    route_edges.to_csv(output_dir / "master_route_edges.csv", index=False)
    pso_history.to_csv(output_dir / "pso_history.csv", index=False)
    pd.DataFrame([best]).to_csv(output_dir / "pso_best_aco_params.csv", index=False)
    hist_dir = output_dir / "aco_history"
    hist_dir.mkdir(parents=True, exist_ok=True)
    for route_key, dfh in aco_histories.items():
        dfh.to_csv(hist_dir / f"{_safe_name(route_key)}.csv", index=False)

    make_figures(model, nodes, edges, route_edges, output_dir, params)
    make_pso_snapshots(pso_history, output_dir, params)
    make_aco_snapshots(model, nodes, output_dir, params)
    make_single_pair_corridor_figures(model, nodes, route_edges, output_dir, params)

    op_rows = route_summary[route_summary["direction"].isin(["forward", "backward", "outer_backup"])]
    if len(op_rows) and "tn_hard_required" in op_rows.columns:
        tn_req = op_rows[op_rows["tn_hard_required"].astype(bool)]
        if len(tn_req):
            tn_ok = int(tn_req["tn_hard_ok"].astype(bool).sum())
            tn_forced = int(tn_req["tn_hard_forced"].astype(bool).sum())
            tn_forced_failed = int((tn_req["tn_hard_forced"].astype(bool) & ~tn_req["tn_hard_forced_ok"].fillna(False).astype(bool)).sum())
            print(f"[routerplan] Hard TN requirement: {tn_ok}/{len(tn_req)} forward+backward+outer routes compliant ({tn_forced} forced, {tn_forced_failed} forcing FAILED)")
        flz_req = op_rows[op_rows["flz_buffer_required"].astype(bool)]
        if len(flz_req):
            flz_ok = int(flz_req["flz_buffer_ok"].astype(bool).sum())
            flz_forced = int(flz_req["flz_buffer_forced"].astype(bool).sum())
            flz_forced_failed = int((flz_req["flz_buffer_forced"].astype(bool) & ~flz_req["flz_buffer_forced_ok"].fillna(False).astype(bool)).sum())
            print(f"[routerplan] Hard FLZ buffer requirement: {flz_ok}/{len(flz_req)} forward+backward+outer routes compliant ({flz_forced} forced, {flz_forced_failed} forcing FAILED)")

    print(f"[routerplan] Successful routes: {int(route_summary['success'].sum())}/{len(route_summary)}")
    print(f"[routerplan] Done. Output directory: {output_dir}")
