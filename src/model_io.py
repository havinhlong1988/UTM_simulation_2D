#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model I/O and graph construction utilities.

Expected model format:
  x y z slowness ... label

The parser is flexible. It reads whitespace-separated XYZ-like files.
The first four columns are assumed to be:
  x, y, z, slowness

The last column is assumed to be the node label:
  N01, DB01, DK01, FLZ01, RA01, etc.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional, Sequence
from scipy.spatial import cKDTree

import numpy as np
import pandas as pd


def load_labelled_model(model_file: Path) -> pd.DataFrame:
    rows = []

    with open(model_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) < 5:
                continue

            try:
                x = float(parts[0])
                y = float(parts[1])
                z = float(parts[2])
                slowness = float(parts[3])
            except ValueError:
                # Skip header or malformed line
                continue

            label = str(parts[-1])
            label_prefix = get_label_prefix(label)

            rows.append(
                {
                    "x": x,
                    "y": y,
                    "z": z,
                    "slowness": slowness,
                    "label": label,
                    "label_prefix": label_prefix,
                    "raw": " ".join(parts),
                }
            )

    if not rows:
        raise ValueError(f"No valid model rows found in: {model_file}")

    df = pd.DataFrame(rows)
    df.index.name = "node_index"

    # Remove bad slowness values
    df["slowness"] = pd.to_numeric(df["slowness"], errors="coerce")
    df = df.dropna(subset=["x", "y", "z", "slowness"]).copy()

    # Keep original integer node index
    df = df.reset_index(drop=True)
    df.index.name = "node_index"

    return df


def get_label_prefix(label: str) -> str:
    """
    Convert:
      DB01 -> DB
      DK02 -> DK
      FLZ01 -> FLZ
      RA03 -> RA
      N100 -> N
    """
    label = str(label).strip()
    m = re.match(r"([A-Za-z_]+)", label)
    if m:
        return m.group(1).upper()
    return label.upper()


def find_start_end_indices(
    model: pd.DataFrame,
    start_label: Optional[str] = None,
    end_label: Optional[str] = None,
    start_coord: Optional[Sequence[float]] = None,
    end_coord: Optional[Sequence[float]] = None,
) -> tuple[int, int]:
    start_idx = find_node_index(
        model,
        label=start_label,
        coord=start_coord,
        default_prefix="DB",
        role="start",
    )

    end_idx = find_node_index(
        model,
        label=end_label,
        coord=end_coord,
        default_prefix="DK",
        role="end",
    )

    if start_idx == end_idx:
        raise ValueError("Start node and end node are the same.")

    return start_idx, end_idx


def find_node_index(
    model: pd.DataFrame,
    label: Optional[str],
    coord: Optional[Sequence[float]],
    default_prefix: str,
    role: str,
) -> int:
    if label is not None:
        exact = model.index[model["label"].astype(str) == str(label)].tolist()
        if exact:
            return int(exact[0])

        # Also allow prefix match if user gives DB but file has DB01
        prefix = str(label).upper()
        prefix_match = model.index[
            model["label"].astype(str).str.upper().str.startswith(prefix)
        ].tolist()
        if prefix_match:
            return int(prefix_match[0])

        print(f"[WARN] {role} label not found: {label}")

    if coord is not None:
        return nearest_node_index(model, coord)

    prefix_match = model.index[model["label_prefix"] == default_prefix].tolist()
    if prefix_match:
        print(f"[INFO] Using first {default_prefix} node as {role}: {model.loc[prefix_match[0], 'label']}")
        return int(prefix_match[0])

    raise ValueError(
        f"Cannot find {role} node. "
        f"Set {role.upper()}_LABEL or {role.upper()}_COORD in parameter.py"
    )


def nearest_node_index(model: pd.DataFrame, coord: Sequence[float]) -> int:
    if len(coord) != 3:
        raise ValueError("Coordinate must be (x, y, z).")

    x0, y0, z0 = map(float, coord)
    d2 = (
        (model["x"].values - x0) ** 2
        + (model["y"].values - y0) ** 2
        + (model["z"].values - z0) ** 2
    )
    return int(model.index[np.argmin(d2)])


def build_grid_graph(
    model: pd.DataFrame,
    block_label_prefixes: Sequence[str] = ("RA",),
    high_cost_label_prefixes: Sequence[str] = ("FLZ",),
    high_cost_factor: float = 5.0,
    connectivity_2d: int = 8,
    connectivity_3d: int = 26,
    always_flyable_prefixes: Sequence[str] = ("DB", "DK"),
    graph_neighbor_mode: str = "kdtree",
    kdtree_radius_factor: float = 1.60,
    kdtree_max_neighbors_2d: int = 8,
    kdtree_max_neighbors_3d: int = 26,
) -> dict:
    """
    Build graph metadata.

    Important:
    The old exact-grid neighbor method can fail for lon/lat models because
    coordinates are not always perfectly regular after clipping/interpolation.

    This version uses KDTree in local meter coordinates.
    """

    block_label_prefixes = tuple(p.upper() for p in block_label_prefixes)
    high_cost_label_prefixes = tuple(p.upper() for p in high_cost_label_prefixes)
    always_flyable_prefixes = tuple(p.upper() for p in always_flyable_prefixes)

    prefix = model["label_prefix"].astype(str).str.upper()

    blocked_mask = prefix.isin(block_label_prefixes)
    always_flyable_mask = prefix.isin(always_flyable_prefixes)

    # RA blocked, but DB/DK are flyable exceptions.
    valid_mask = (~blocked_mask) | always_flyable_mask
    valid_indices = set(model.index[valid_mask].astype(int).tolist())

    xs = np.sort(model["x"].unique())
    ys = np.sort(model["y"].unique())
    zs = np.sort(model["z"].unique())

    dimension = 2 if len(zs) <= 1 else 3
    connectivity = connectivity_2d if dimension == 2 else connectivity_3d

    is_lonlat = detect_lonlat_from_model(model)

    metric_xyz = make_metric_coordinates(model, is_lonlat=is_lonlat)

    # Estimate grid spacing from KDTree nearest-neighbor distance.
    valid_list = np.array(sorted(valid_indices), dtype=int)
    valid_metric = metric_xyz[valid_list]

    if len(valid_metric) < 2:
        raise ValueError("Not enough valid nodes to build graph.")

    tree_valid = cKDTree(valid_metric)

    # Query nearest 2 nodes: self + nearest neighbor.
    dd, _ = tree_valid.query(valid_metric, k=2)
    nearest_dist = dd[:, 1]
    nearest_dist = nearest_dist[np.isfinite(nearest_dist)]
    nearest_dist = nearest_dist[nearest_dist > 0]

    if len(nearest_dist) == 0:
        raise ValueError("Cannot estimate grid spacing from model nodes.")

    grid_spacing_m = float(np.median(nearest_dist))
    neighbor_radius_m = grid_spacing_m * float(kdtree_radius_factor)

    max_neighbors = (
        int(kdtree_max_neighbors_2d)
        if dimension == 2
        else int(kdtree_max_neighbors_3d)
    )

    # KDTree over all nodes, but neighbors will be filtered by valid_indices.
    tree_all = cKDTree(metric_xyz)

    # Slowness for heuristic.
    valid_slowness = pd.to_numeric(model.loc[valid_mask, "slowness"], errors="coerce")
    valid_slowness = valid_slowness[np.isfinite(valid_slowness)]
    valid_slowness = valid_slowness[valid_slowness > 0]

    if len(valid_slowness) == 0:
        min_slowness = 1.0
    else:
        min_slowness = max(float(valid_slowness.min()), 1e-12)

    graph = {
        "valid_indices": valid_indices,
        "metric_xyz": metric_xyz,
        "tree_all": tree_all,
        "is_lonlat": is_lonlat,
        "dimension": dimension,
        "connectivity": connectivity,
        "graph_neighbor_mode": graph_neighbor_mode,
        "grid_spacing_m": grid_spacing_m,
        "neighbor_radius_m": neighbor_radius_m,
        "max_neighbors": max_neighbors,
        "block_label_prefixes": block_label_prefixes,
        "high_cost_label_prefixes": high_cost_label_prefixes,
        "always_flyable_prefixes": always_flyable_prefixes,
        "high_cost_factor": high_cost_factor,
        "min_slowness": min_slowness,
        "neighbor_cache": {},
    }

    return graph

def make_metric_coordinates(model: pd.DataFrame, is_lonlat: bool = True) -> np.ndarray:
    """
    Convert coordinates to approximate local meters.

    If model is lon/lat:
      x = longitude converted to meters
      y = latitude converted to meters
      z = meters

    If model is already projected:
      x, y, z are used directly.
    """

    x = model["x"].astype(float).values
    y = model["y"].astype(float).values
    z = model["z"].astype(float).values

    if not is_lonlat:
        return np.column_stack([x, y, z])

    lon0 = float(np.nanmean(x))
    lat0 = float(np.nanmean(y))
    lat0_rad = np.deg2rad(lat0)

    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * np.cos(lat0_rad)

    xm = (x - lon0) * meters_per_deg_lon
    ym = (y - lat0) * meters_per_deg_lat
    zm = z

    return np.column_stack([xm, ym, zm])


def detect_lonlat_from_model(model: pd.DataFrame) -> bool:
    xmin, xmax = model["x"].min(), model["x"].max()
    ymin, ymax = model["y"].min(), model["y"].max()

    return (
        -180 <= xmin <= 180
        and -180 <= xmax <= 180
        and -90 <= ymin <= 90
        and -90 <= ymax <= 90
    )

def estimate_spacing(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return 1.0

    diffs = np.diff(np.sort(values))
    diffs = diffs[diffs > 0]

    if len(diffs) == 0:
        return 1.0

    return float(np.median(diffs))


def make_key(x: float, y: float, z: float, dx: float, dy: float, dz: float) -> tuple[int, int, int]:
    return (
        int(round(float(x) / dx)),
        int(round(float(y) / dy)),
        int(round(float(z) / dz)),
    )


def make_neighbor_offsets(dimension: int, connectivity: int) -> list[tuple[int, int, int]]:
    offsets = []

    if dimension == 2:
        for ix in (-1, 0, 1):
            for iy in (-1, 0, 1):
                if ix == 0 and iy == 0:
                    continue

                if connectivity == 4 and abs(ix) + abs(iy) != 1:
                    continue

                offsets.append((ix, iy, 0))

    else:
        for ix in (-1, 0, 1):
            for iy in (-1, 0, 1):
                for iz in (-1, 0, 1):
                    if ix == 0 and iy == 0 and iz == 0:
                        continue

                    manhattan = abs(ix) + abs(iy) + abs(iz)

                    if connectivity == 6 and manhattan != 1:
                        continue

                    if connectivity == 18 and manhattan > 2:
                        continue

                    offsets.append((ix, iy, iz))

    return offsets


def iter_neighbors(model: pd.DataFrame, graph: dict, idx: int):
    """
    Return valid neighbors using KDTree radius search.

    This is much more robust than exact grid-key matching.
    """

    idx = int(idx)

    if idx in graph["neighbor_cache"]:
        for nidx in graph["neighbor_cache"][idx]:
            yield nidx
        return

    xyz = graph["metric_xyz"]
    tree_all = graph["tree_all"]
    radius = float(graph["neighbor_radius_m"])
    valid_indices = graph["valid_indices"]
    max_neighbors = int(graph["max_neighbors"])

    candidate_indices = tree_all.query_ball_point(
        xyz[idx],
        r=radius,
    )

    neighbors = []

    for nidx in candidate_indices:
        nidx = int(nidx)

        if nidx == idx:
            continue

        if nidx not in valid_indices:
            continue

        dist = float(np.linalg.norm(xyz[nidx] - xyz[idx]))

        if dist <= 0:
            continue

        neighbors.append((dist, nidx))

    # Keep nearest neighbors only.
    neighbors.sort(key=lambda t: t[0])
    neighbors = neighbors[:max_neighbors]

    out = [nidx for _, nidx in neighbors]
    graph["neighbor_cache"][idx] = out

    for nidx in out:
        yield nidx


def edge_cost(model: pd.DataFrame, graph: dict, idx1: int, idx2: int) -> float:
    """
    Edge cost = metric distance * average slowness * penalty.
    """

    idx1 = int(idx1)
    idx2 = int(idx2)

    xyz = graph["metric_xyz"]

    dist_m = float(np.linalg.norm(xyz[idx2] - xyz[idx1]))

    a = model.loc[idx1]
    b = model.loc[idx2]

    s1 = float(a["slowness"])
    s2 = float(b["slowness"])

    # Avoid zero slowness for DB/DK/special points.
    if s1 <= 0:
        s1 = graph["min_slowness"]
    if s2 <= 0:
        s2 = graph["min_slowness"]

    slow = 0.5 * (s1 + s2)

    factor = 1.0

    if str(a["label_prefix"]).upper() in graph["high_cost_label_prefixes"]:
        factor *= float(graph["high_cost_factor"])

    if str(b["label_prefix"]).upper() in graph["high_cost_label_prefixes"]:
        factor *= float(graph["high_cost_factor"])

    return dist_m * slow * factor


def heuristic_cost(model: pd.DataFrame, graph: dict, idx: int, end_idx: int) -> float:
    """
    A* heuristic using local meter distance.
    """

    idx = int(idx)
    end_idx = int(end_idx)

    xyz = graph["metric_xyz"]

    dist_m = float(np.linalg.norm(xyz[end_idx] - xyz[idx]))

    return dist_m * float(graph["min_slowness"])

def snap_index_to_nearest_traversable_node(
    model: pd.DataFrame,
    graph: dict,
    idx: int,
    target_prefixes=("N", "FLZ"),
) -> int:
    """
    Snap a point such as DB01/DK01 to nearest traversable regular grid node.

    This is needed because DB/DK/KML points may not lie exactly on the
    regular model grid, so they may have no graph neighbors.
    """

    target_prefixes = tuple(str(p).upper() for p in target_prefixes)

    src = model.loc[idx]

    candidate_mask = (
        model.index.isin(graph["valid_indices"])
        & model["label_prefix"].isin(target_prefixes)
    )

    candidates = model[candidate_mask].copy()

    if candidates.empty:
        raise ValueError(
            f"No candidate nodes found for snapping. "
            f"Check SNAP_TARGET_PREFIXES={target_prefixes}"
        )

    dx = candidates["x"].values - float(src["x"])
    dy = candidates["y"].values - float(src["y"])
    dz = candidates["z"].values - float(src["z"])

    d2 = dx * dx + dy * dy + dz * dz

    nearest_pos = int(np.argmin(d2))
    nearest_idx = int(candidates.index[nearest_pos])

    return nearest_idx


def snap_start_end_to_grid_if_needed(
    model: pd.DataFrame,
    graph: dict,
    start_idx: int,
    end_idx: int,
    snap: bool = True,
    target_prefixes=("N", "FLZ"),
) -> tuple[int, int]:
    """
    Return algorithm start/end indices.

    If snap=False:
      use original start/end.

    If snap=True:
      snap DB/DK or off-grid points to nearest traversable N/FLZ node.
    """

    if not snap:
        return start_idx, end_idx

    start_search_idx = start_idx
    end_search_idx = end_idx

    start_prefix = str(model.loc[start_idx, "label_prefix"]).upper()
    end_prefix = str(model.loc[end_idx, "label_prefix"]).upper()

    if start_prefix not in target_prefixes:
        start_search_idx = snap_index_to_nearest_traversable_node(
            model=model,
            graph=graph,
            idx=start_idx,
            target_prefixes=target_prefixes,
        )

    if end_prefix not in target_prefixes:
        end_search_idx = snap_index_to_nearest_traversable_node(
            model=model,
            graph=graph,
            idx=end_idx,
            target_prefixes=target_prefixes,
        )

    return start_search_idx, end_search_idx


def add_real_start_end_to_path(
    path_indices: list[int],
    real_start_idx: int,
    real_end_idx: int,
    include: bool = True,
) -> list[int]:
    """
    Add original DB/DK points to exported path.

    Algorithm path:
      nearest_grid_start -> ... -> nearest_grid_end

    Exported footprint:
      DB01 -> nearest_grid_start -> ... -> nearest_grid_end -> DK01
    """

    if not include:
        return list(path_indices)

    out = list(path_indices)

    if len(out) == 0:
        return out

    if out[0] != real_start_idx:
        out = [real_start_idx] + out

    if out[-1] != real_end_idx:
        out = out + [real_end_idx]

    return out

def add_endpoint_flyable_buffer(
    model: pd.DataFrame,
    graph: dict,
    endpoint_indices: list[int],
    radius_m: float = 100.0,
) -> dict:
    """
    Force nodes within radius_m around endpoint indices to be traversable.

    This is useful when DB/DK or the snapped start/end node is located inside
    a restricted/no-fly zone. It opens only a local takeoff/landing buffer.

    Notes
    -----
    If x/y are lon/lat, the radius is approximately converted to degrees
    using local latitude.
    """

    radius_m = float(radius_m)

    if radius_m <= 0:
        return graph

    endpoint_indices = [int(i) for i in endpoint_indices]

    is_lonlat = detect_lonlat_from_model(model)

    added_indices = set()

    for endpoint_idx in endpoint_indices:
        if endpoint_idx not in model.index:
            continue

        endpoint = model.loc[endpoint_idx]

        if is_lonlat:
            mask = distance_mask_lonlat_m(
                model=model,
                lon0=float(endpoint["x"]),
                lat0=float(endpoint["y"]),
                z0=float(endpoint["z"]),
                radius_m=radius_m,
            )
        else:
            dx = model["x"].values - float(endpoint["x"])
            dy = model["y"].values - float(endpoint["y"])
            dz = model["z"].values - float(endpoint["z"])

            dist = np.sqrt(dx * dx + dy * dy + dz * dz)
            mask = dist <= radius_m

        buffer_indices = set(model.index[mask].astype(int).tolist())
        added_indices.update(buffer_indices)

    graph["valid_indices"].update(added_indices)

    graph["endpoint_flyable_buffer_radius_m"] = radius_m
    graph["endpoint_flyable_buffer_indices"] = sorted(added_indices)

    return graph


def detect_lonlat_from_model(model: pd.DataFrame) -> bool:
    xmin, xmax = model["x"].min(), model["x"].max()
    ymin, ymax = model["y"].min(), model["y"].max()

    return (
        -180 <= xmin <= 180
        and -180 <= xmax <= 180
        and -90 <= ymin <= 90
        and -90 <= ymax <= 90
    )


def distance_mask_lonlat_m(
    model: pd.DataFrame,
    lon0: float,
    lat0: float,
    z0: float,
    radius_m: float,
):
    """
    Approximate lon/lat/z distance mask in meters.

    Good for local buffer such as 100 m.
    """

    lat0_rad = np.deg2rad(lat0)

    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * np.cos(lat0_rad)

    dx_m = (model["x"].values - lon0) * meters_per_deg_lon
    dy_m = (model["y"].values - lat0) * meters_per_deg_lat
    dz_m = model["z"].values - float(z0)

    dist_m = np.sqrt(dx_m * dx_m + dy_m * dy_m + dz_m * dz_m)

    return dist_m <= float(radius_m)

def count_valid_neighbors(model: pd.DataFrame, graph: dict, idx: int) -> int:
    """
    Count valid neighbors of one node.

    Used only for diagnostic printing in main.py.
    """
    return sum(1 for _ in iter_neighbors(model, graph, idx))

def compute_path_metrics(
    model: pd.DataFrame,
    graph: dict,
    path_indices: list[int],
) -> dict:
    """
    Compute total path distance and estimated travel time.

    Distance:
      meter distance from graph["metric_xyz"]

    Traveltime:
      sum(edge_cost)

    If slowness unit is s/m:
      traveltime unit = seconds
    """

    if path_indices is None or len(path_indices) < 2:
        return {
            "distance_traveled_m": 0.0,
            "distance_traveled_km": 0.0,
            "estimated_traveltime_s": 0.0,
            "estimated_traveltime_min": 0.0,
            "n_segments": 0,
        }

    distance_m = 0.0
    traveltime_s = 0.0

    xyz = graph["metric_xyz"]

    for i in range(len(path_indices) - 1):
        idx1 = int(path_indices[i])
        idx2 = int(path_indices[i + 1])

        segment_distance_m = float(np.linalg.norm(xyz[idx2] - xyz[idx1]))
        distance_m += segment_distance_m

        # Use the same cost rule as A*
        traveltime_s += float(edge_cost(model, graph, idx1, idx2))

    return {
        "distance_traveled_m": float(distance_m),
        "distance_traveled_km": float(distance_m / 1000.0),
        "estimated_traveltime_s": float(traveltime_s),
        "estimated_traveltime_min": float(traveltime_s / 60.0),
        "n_segments": int(len(path_indices) - 1),
    }

def cap_slowness_values(
    model: pd.DataFrame,
    cap_value: float,
    inplace: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """
    Cap slowness values larger than cap_value.
    Due to current model setup give supper high slowness to no-fly zones, capping can help reduce numerical issues
    Example:
      slowness > 1e5 -> 1e5

    Returns
    -------
    model_out, summary
    """

    cap_value = float(cap_value)

    if inplace:
        out = model
    else:
        out = model.copy()

    slow = pd.to_numeric(out["slowness"], errors="coerce")

    mask = slow > cap_value
    n_capped = int(mask.sum())

    old_max = float(np.nanmax(slow.values)) if len(slow) > 0 else float("nan")

    out.loc[mask, "slowness"] = cap_value

    new_slow = pd.to_numeric(out["slowness"], errors="coerce")
    new_max = float(np.nanmax(new_slow.values)) if len(new_slow) > 0 else float("nan")

    summary = {
        "cap_value": cap_value,
        "n_capped": n_capped,
        "old_max_slowness": old_max,
        "new_max_slowness": new_max,
    }

    return out, summary

'''
All the functions in this file are related to loading the model, building the graph, and computing path metrics. 
The main.py file is responsible for orchestrating the overall workflow, including plotting and cleanup. 
The parameters.py file contains configuration settings that can be adjusted based on the specific model and requirements.
'''
__all__ = [
    "load_labelled_model",
    "find_start_end_indices",
    "build_grid_graph",
    "snap_start_end_to_grid_if_needed",
    "add_real_start_end_to_path",
    "add_endpoint_flyable_buffer",
    "count_valid_neighbors",
    "compute_path_metrics",
    "cap_slowness_values",
]
