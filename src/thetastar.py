# LAE-UTM Theta* backend v16
# Packaged as thetastar_v1.py for the node-riskmap master planner.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/thetastar.py

Theta* any-angle path-finding algorithm for the LAE-UTM main.py protocol.

Interface, output fields, and parameter style are intentionally kept close to
astar.py:

    result = run(model=model, graph=graph, start_idx=i, end_idx=j, **kwargs)

Important model rule is inherited from main.py/build_grid_graph():
    slowness < 10.0   -> flyable
    slowness >= 10.0  -> no-fly / blocked

Theta* difference from A*:
    A* expands graph edges and sets parent(neighbor)=current.
    Theta* also tests whether parent(current) has line-of-sight to neighbor.
    If that straight segment crosses only valid graph nodes, it sets
    parent(neighbor)=parent(current), producing an any-angle path.

The line-of-sight check samples along the straight segment and requires every
sampled node to belong to graph["valid_indices"]. Therefore hard no-fly cells
are not crossed when build_grid_graph() has removed them from valid_indices.

Route corridor model (v16)
---------------------------
Every route is modeled as a circular prism (a cylinder): a central line --
the straight LOS segment between two waypoints -- swept by a circle of
radius THETASTAR_ROUTE_WIDTH_M / 2 in the map plane. On the flat 2D map this
looks like a band of width THETASTAR_ROUTE_WIDTH_M straddling the line; that
width is the diameter of the cylinder once you consider the corridor as a
solid safety envelope rather than a flat line. A shortcut is accepted only
when this whole cylinder -- not just the centerline -- stays clear of no-fly
node centers (see _route_corridor_segment_indices()).
"""

# =====================================================================
# FUNCTION GUIDE / COMMENT MAP
# =====================================================================
# This file is intentionally verbose.  The goal is not only to run Theta*,
# but also to make the logic easy to audit and modify in the LAE-UTM project.
#
# High-level execution modes
# --------------------------
# 1) Recommended mode: A* first, then LOS smoothing
#       THETASTAR_ASTAR_FIRST_LOS_SMOOTH = True
#
#       run()
#         -> _astar_then_los_smooth()
#            -> _call_project_astar()
#               runs src/astar.py, so reachability is controlled by your
#               already-working A* implementation.
#            -> _smooth_path_by_line_of_sight()
#               removes unnecessary intermediate waypoints only when a
#               straight segment is safe.
#
#       This is the safest mode for your current model because line-of-sight
#       can improve the path, but it cannot make the whole search fail.
#
# 2) Debug mode: pure A* delegate
#       THETASTAR_ALLOW_ANY_ANGLE = False
#       THETASTAR_PURE_ASTAR_DELEGATE = True
#
#       run()
#         -> _call_project_astar()
#
#       This should behave like astar.py while still exporting under the
#       algorithm name "thetastar".  Use this mode to confirm labels, graph,
#       plotting, and output folders.
#
# 3) Full in-search Theta*
#       THETASTAR_ASTAR_FIRST_LOS_SMOOTH = False
#       THETASTAR_ALLOW_ANY_ANGLE = True
#
#       run()
#         -> internal Theta* loop
#            -> _line_of_sight() inside neighbor relaxation
#
#       This is the classical Theta* search, but it is more sensitive to
#       line-of-sight sampling.  Keep THETASTAR_EDGE_FALLBACK_TO_ASTAR=True
#       so normal graph-neighbor moves use A*-style edge cost.
#
# Important hard no-fly rule
# --------------------------
# The graph builder decides which model nodes are traversable.  In your current
# LAE-UTM model:
#       slowness < 10.0   -> flyable
#       slowness >= 10.0  -> no-fly / blocked
#
# Theta* never directly checks this threshold here.  Instead, it respects
# graph["valid_indices"].  That keeps the algorithm consistent with astar.py
# and main.py.
#
# Function groups
# ---------------
# Parameter helpers:
#   _parameters_module()
#       Import parameters.py if available.
#   _param()
#       Read a setting from kwargs first, then parameters.py, then default.
#   _as_bool(), _as_float(), _as_int()
#       Safe type converters for configuration values.
#
# Model / geometry helpers:
#   _get_row()
#       Read one model row by node index.
#   _xy_columns()
#       Decide whether coordinates are x/y or lon/lat.
#   _coord_tuple()
#       Return x, y, z for one node.
#   _coordinates_look_lonlat()
#       Detect whether coordinate values look like geographic lon/lat.
#   _distance_m()
#       Compute distance in meters; converts lon/lat degrees if needed.
#   _slowness()
#       Read positive slowness from a node.
#   _valid_index_set()
#       Get traversable nodes from graph["valid_indices"].
#
# Line-of-sight grid cache:
#   GridCache
#       Stores coordinate vectors, coordinate-to-node lookup, valid nodes,
#       LOS sample step, and lon/lat mode.
#   _median_positive_spacing()
#       Estimate grid spacing.
#   _make_grid_cache()
#       Prepare fast lookup for line-of-sight sampling.
#   _nearest_value(), _nearest_model_index()
#       Snap a sampled coordinate back to the nearest model node.
#   _sample_segment_indices()
#       Old continuous-sampling LOS; kept as fallback/debug.
#   _bresenham_segment_indices()
#       Integer-grid LOS using Bresenham cells. Good for true rectangular grids,
#       but can be too strict for lon/lat point-cloud or KDTree grids.
#   _route_corridor_segment_indices()
#       Geometry-based LOS: models the route as a circular prism (cylinder)
#       around the straight segment -- radius THETASTAR_ROUTE_WIDTH_M / 2 --
#       and rejects a shortcut when that corridor passes too close to a
#       no-fly node center. This is better for the LAE-UTM lon/lat/KDTree
#       graph because it does not assume perfect row/column cells.
#   _line_of_sight()
#       Return True only when configured LOS method says the shortcut is safe.
#
# Cost / heuristic helpers:
#   _direct_segment_cost()
#       Cost of an accepted straight LOS segment using sampled slowness.
#   _graph_edge_cost()
#       Cost of one normal graph edge, A*-compatible and less strict.
#   _minimum_positive_slowness()
#       Fallback heuristic multiplier.
#   _safe_heuristic()
#       Use project heuristic_cost() if possible, otherwise use distance*min_slow.
#
# A* delegate and LOS smoothing:
#   _call_project_astar()
#       Runs your src/astar.py implementation.
#   _path_direct_segment_cost()
#       Recomputes total cost for a smoothed waypoint path.
#   _smooth_path_by_line_of_sight()
#       Greedily removes waypoints from a valid A* path where LOS allows.
#   _astar_then_los_smooth()
#       Recommended workflow: A* result + LOS smoothing.
#
# Main search and output helpers:
#   run()
#       Main entry point called by main.py.
#   reconstruct_path()
#       Backtrack parent pointers to produce node index path.
#   expand_theta_path_to_sampled_nodes()
#       Optional plotting/debug mode that expands any-angle segments back to
#       sampled grid nodes.
# =====================================================================


from __future__ import annotations

__version__ = "v16"

import heapq
import math
import time
import importlib
import inspect
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.model_io import iter_neighbors, heuristic_cost


# ============================================================
# Parameter helpers
# ============================================================


def _parameters_module():
    """Return the project parameters module if it can be imported.

    main.py normally loads algorithm parameters from params/*.params into
    kwargs, but this helper is a backup.  It lets the algorithm also read
    values directly from parameters.py when kwargs does not contain them.

    Returns
    -------
    module or None
        The imported parameters module, or None if it is unavailable.
    """
    try:
        import parameters as P  # type: ignore
        return P
    except Exception:
        return None


def _param(kwargs: dict[str, Any], name: str, default: Any = None, aliases=()):
    """Read parameter from kwargs first, then parameters.py, then default."""
    keys = (name, *tuple(aliases or ()))
    for key in keys:
        if key in kwargs:
            return kwargs[key]

    P = _parameters_module()
    if P is not None:
        for key in keys:
            if hasattr(P, key):
                return getattr(P, key)

    return default


def _as_bool(value: Any, default: bool = False) -> bool:
    """Convert many user-friendly values into True/False.

    Accepted True strings include: true, yes, on, enabled.
    Accepted False strings include: false, no, off, disabled.
    This makes params files tolerant of different writing styles.
    """
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "on", "enable", "enabled"):
        return True
    if text in ("0", "false", "no", "n", "off", "disable", "disabled"):
        return False
    return bool(default)


def _as_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float.

    If the value is missing, invalid, NaN, or infinite, return default.
    """
    try:
        v = float(value)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)


def _as_int(value: Any, default: int = 0) -> int:
    """Safely convert a value to int.

    The conversion allows strings like '300.0' by first casting to float.
    """
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _progress_bar_text(current: int, total: int, label: str = "", width: int = 24, suffix: str = "") -> str:
    """Build a single in-place progress line: [####----] 42.0% label suffix.

    Enable with THETASTAR_SHOW_PROGRESS=True. total<=0 falls back to a
    plain counter (no meaningful percentage exists in that case).
    """
    current = max(0, int(current))
    if total is None or total <= 0:
        prefix = f"{label} " if label else ""
        return f"\r[thetastar] {prefix}{current:,} expanded{(' ' + suffix) if suffix else ''}"
    total = int(total)
    current_clamped = min(current, total)
    pct = 100.0 * current_clamped / total
    filled = int(round(width * current_clamped / total))
    bar = "#" * filled + "-" * (width - filled)
    prefix = f"{label} " if label else ""
    return f"\r[thetastar] {prefix}[{bar}] {current:,}/{total:,} ({pct:5.1f}%){(' ' + suffix) if suffix else ''}"


# ============================================================
# Model / geometry helpers
# ============================================================


def _get_row(model, idx: int):
    """Return model row by index label first, then integer position."""
    idx = int(idx)
    if idx in model.index:
        return model.loc[idx]
    return model.iloc[idx]


def _xy_columns(model) -> tuple[str, str]:
    """Return the coordinate column names used by the model.

    The LAE-UTM scripts usually store projected or geographic coordinates in
    x/y.  Some exported tables may use lon/lat.  This helper keeps the rest
    of the algorithm independent from that naming detail.
    """
    if "x" in model.columns and "y" in model.columns:
        return "x", "y"
    if "lon" in model.columns and "lat" in model.columns:
        return "lon", "lat"
    raise ValueError("Theta* requires model columns x/y or lon/lat.")


def _coord_tuple(model, idx: int) -> tuple[float, float, float]:
    """Return the numeric coordinate tuple (x, y, z) for one node.

    z is optional.  For a 2D grid, z is returned as 0.0 so the same distance
    functions can work for both 2D and 3D.
    """
    row = _get_row(model, int(idx))
    xcol, ycol = _xy_columns(model)
    x = _as_float(row[xcol])
    y = _as_float(row[ycol])
    z = _as_float(row["z"], 0.0) if "z" in model.columns else 0.0
    return x, y, z


def _coordinates_look_lonlat(model) -> bool:
    """Heuristically detect whether x/y coordinates are lon/lat degrees.

    If all x values are within [-180, 180] and all y values are within
    [-90, 90], they probably represent longitude/latitude.  This is important
    because degree differences must be converted to meters before computing
    path costs.
    """
    try:
        xcol, ycol = _xy_columns(model)
        x = model[xcol].astype(float)
        y = model[ycol].astype(float)
        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]
        if len(x) == 0 or len(y) == 0:
            return False
        return bool((x.between(-180.0, 180.0).all()) and (y.between(-90.0, 90.0).all()))
    except Exception:
        return False


def _distance_m(model, idx1: int, idx2: int, lonlat_as_meters: bool = True, coordinates_are_lonlat: bool | None = None) -> float:
    """Distance between two nodes in meters.

    If x/y look like lon/lat degrees, use an equirectangular approximation.
    Otherwise treat x/y/z as metric coordinates.
    """
    x1, y1, z1 = _coord_tuple(model, idx1)
    x2, y2, z2 = _coord_tuple(model, idx2)

    if coordinates_are_lonlat is None:
        coordinates_are_lonlat = bool(lonlat_as_meters and _coordinates_look_lonlat(model))

    if bool(coordinates_are_lonlat):
        lat0 = math.radians(0.5 * (y1 + y2))
        dx = (x2 - x1) * 111_320.0 * math.cos(lat0)
        dy = (y2 - y1) * 110_540.0
    else:
        dx = x2 - x1
        dy = y2 - y1

    dz = z2 - z1
    return float(math.sqrt(dx * dx + dy * dy + dz * dz))


def _slowness(model, idx: int) -> float:
    row = _get_row(model, int(idx))
    s = _as_float(row.get("slowness", math.inf), math.inf)
    if not math.isfinite(s) or s <= 0.0:
        return math.inf
    return float(s)


def _valid_index_set(model, graph) -> set[int]:
    valid = graph.get("valid_indices", None) if isinstance(graph, dict) else None
    if valid is None:
        if "is_flyable" in model.columns:
            return {int(i) for i in model.index[model["is_flyable"].astype(bool)]}
        return {int(i) for i in model.index}
    return {int(v) for v in valid}


# ============================================================
# Grid cache for line-of-sight sampling
# ============================================================


@dataclass
class GridCache:
    x_values: np.ndarray
    y_values: np.ndarray
    z_values: np.ndarray
    xyz_to_idx: dict[tuple[float, float, float], int]
    valid_indices: set[int]
    step: float
    ndims: int
    lonlat_as_meters: bool
    coordinates_are_lonlat: bool

    # LOS method used by _line_of_sight().
    #   "bresenham" : integer grid cells between two nodes. Best for regular grids.
    #   "sample"    : old continuous sample + nearest-node snapping method.
    line_of_sight_method: str = "bresenham"

    # Extra safety around every Bresenham cell.
    # 0 = check only cells on the line.
    # 1 = also check 8 neighboring cells around each line cell.
    # Larger values are safer but can make Theta* look like A*.
    bresenham_clearance_cells: int = 0

    # Route corridor for non-perfect grids: a circular prism (cylinder) swept
    # along the straight segment. route_width_m is the corridor's width on
    # the 2D map -- the cylinder's diameter in 3D -- and corridor_radius_m
    # (= route_width_m / 2) is the actual radius used by the geometry check.
    # A Theta* shortcut is rejected if the segment's corridor passes within
    # corridor_radius_m of any blocked/no-fly model node center.
    route_width_m: float = 50.0
    corridor_radius_m: float = 25.0

    # Fast arrays for route-corridor LOS. They are prepared once in
    # _make_grid_cache() and reused for every LOS test.
    metric_ref_lat_deg: float = 0.0
    node_xy_m: dict[int, tuple[float, float]] | None = None
    blocked_node_indices: np.ndarray | None = None
    blocked_x_m: np.ndarray | None = None
    blocked_y_m: np.ndarray | None = None


def _median_positive_spacing(values: np.ndarray) -> float | None:
    """Estimate coordinate spacing from a sorted coordinate vector.

    Only positive finite differences are used.  The median is more robust
    than the minimum when there are duplicate or slightly irregular values.
    """
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None
    diffs = np.diff(np.sort(values))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if len(diffs) == 0:
        return None
    return float(np.median(diffs))


def _make_grid_cache(
    model,
    graph,
    *,
    line_of_sight_step_factor: float = 0.5,
    lonlat_as_meters: bool = True,
    line_of_sight_method: str = "point_clearance",
    bresenham_clearance_cells: int = 0,
    route_width_m: float = 50.0,
) -> GridCache:
    xcol, ycol = _xy_columns(model)
    x_values = np.sort(model[xcol].astype(float).unique())
    y_values = np.sort(model[ycol].astype(float).unique())

    if "z" in model.columns:
        z_values = np.sort(model["z"].astype(float).unique())
    else:
        z_values = np.array([0.0], dtype=float)

    ndims = 2 if len(z_values) <= 1 else 3

    dx = _median_positive_spacing(x_values)
    dy = _median_positive_spacing(y_values)
    dz = _median_positive_spacing(z_values) if ndims == 3 else None

    spacings = [v for v in (dx, dy, dz) if v is not None and math.isfinite(v) and v > 0.0]
    if spacings:
        step = max(1.0e-12, float(line_of_sight_step_factor) * min(spacings))
    else:
        step = 1.0

    xyz_to_idx: dict[tuple[float, float, float], int] = {}
    has_z = "z" in model.columns

    # Prepare a local metric coordinate cache.
    # Why: the LAE-UTM model often stores x/y as lon/lat and the graph is
    # built by KDTree.  Integer row/column Bresenham can be too strict on this
    # kind of point cloud.  The route-corridor LOS uses metric x/y so the
    # cylinder's radius can be checked against no-fly nodes robustly.
    coordinates_are_lonlat = bool(lonlat_as_meters and _coordinates_look_lonlat(model))
    try:
        metric_ref_lat_deg = float(model[ycol].astype(float).mean()) if coordinates_are_lonlat else 0.0
    except Exception:
        metric_ref_lat_deg = 0.0

    valid_indices = _valid_index_set(model, graph)
    node_xy_m: dict[int, tuple[float, float]] = {}
    blocked_indices: list[int] = []
    blocked_x: list[float] = []
    blocked_y: list[float] = []

    lat0_rad = math.radians(metric_ref_lat_deg)
    meters_per_lon = 111_320.0 * max(abs(math.cos(lat0_rad)), 1.0e-8)
    meters_per_lat = 110_540.0

    for idx, row in model.iterrows():
        idx_i = int(idx)
        x = round(float(row[xcol]), 8)
        y = round(float(row[ycol]), 8)
        z = round(float(row["z"]), 8) if has_z else 0.0
        xyz_to_idx[(x, y, z)] = idx_i

        if coordinates_are_lonlat:
            xm = float(row[xcol]) * meters_per_lon
            ym = float(row[ycol]) * meters_per_lat
        else:
            xm = float(row[xcol])
            ym = float(row[ycol])
        node_xy_m[idx_i] = (float(xm), float(ym))

        if idx_i not in valid_indices:
            blocked_indices.append(idx_i)
            blocked_x.append(float(xm))
            blocked_y.append(float(ym))

    return GridCache(
        x_values=x_values,
        y_values=y_values,
        z_values=z_values,
        xyz_to_idx=xyz_to_idx,
        valid_indices=valid_indices,
        step=float(step),
        ndims=int(ndims),
        lonlat_as_meters=bool(lonlat_as_meters),
        coordinates_are_lonlat=coordinates_are_lonlat,
        line_of_sight_method=str(line_of_sight_method or "point_clearance").strip().lower(),
        bresenham_clearance_cells=max(0, int(bresenham_clearance_cells or 0)),
        route_width_m=float(route_width_m),
        corridor_radius_m=max(0.0, float(route_width_m)) / 2.0,
        metric_ref_lat_deg=float(metric_ref_lat_deg),
        node_xy_m=node_xy_m,
        blocked_node_indices=np.asarray(blocked_indices, dtype=int),
        blocked_x_m=np.asarray(blocked_x, dtype=float),
        blocked_y_m=np.asarray(blocked_y, dtype=float),
    )


def _nearest_value(values: np.ndarray, value: float) -> float:
    """Snap one coordinate value to the nearest coordinate in the grid.

    Line-of-sight samples are continuous points.  The model is a discrete grid.
    This function maps a continuous x/y/z sample back to the nearest grid value.
    """
    if len(values) == 1:
        return float(values[0])
    pos = int(np.searchsorted(values, value))
    if pos <= 0:
        return float(values[0])
    if pos >= len(values):
        return float(values[-1])
    before = float(values[pos - 1])
    after = float(values[pos])
    return before if abs(value - before) <= abs(value - after) else after


def _nearest_model_index(cache: GridCache, x: float, y: float, z: float) -> int | None:
    """Return the model node nearest to a sampled coordinate.

    The coordinate is snapped independently in x, y, and z.  The rounded
    snapped coordinate is then looked up in cache.xyz_to_idx.  None means the
    sampled point could not be mapped to a model node.
    """
    xn = round(_nearest_value(cache.x_values, x), 8)
    yn = round(_nearest_value(cache.y_values, y), 8)
    if cache.ndims == 3:
        zn = round(_nearest_value(cache.z_values, z), 8)
    else:
        zn = round(float(cache.z_values[0]), 8)
    return cache.xyz_to_idx.get((xn, yn, zn), None)



def _nearest_grid_position(values: np.ndarray, value: float) -> int:
    """Return integer grid coordinate nearest to a continuous coordinate.

    Why this exists
    ---------------
    The model stores physical coordinates (lon/lat or x/y), but Bresenham works
    on integer grid coordinates.  For each path node, we snap its coordinate to
    the closest entry in the sorted unique coordinate vector.

    Example
    -------
    If x_values = [105.54, 105.5405, 105.5410], a node x near 105.5405
    becomes grid x index 1.
    """
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return 0
    pos = int(np.searchsorted(values, float(value)))
    if pos <= 0:
        return 0
    if pos >= len(values):
        return int(len(values) - 1)
    before = float(values[pos - 1])
    after = float(values[pos])
    return int(pos - 1 if abs(float(value) - before) <= abs(float(value) - after) else pos)


def _node_grid_position(model, cache: GridCache, idx: int) -> tuple[int, int, int]:
    """Return integer grid position (gx, gy, gz) for a model node index.

    gx and gy are positions inside cache.x_values/cache.y_values, not physical
    coordinates.  gz is usually 0 for your current 2D model.
    """
    x, y, z = _coord_tuple(model, int(idx))
    gx = _nearest_grid_position(cache.x_values, x)
    gy = _nearest_grid_position(cache.y_values, y)
    gz = _nearest_grid_position(cache.z_values, z) if cache.ndims == 3 else 0
    return int(gx), int(gy), int(gz)


def _model_index_from_grid_position(cache: GridCache, gx: int, gy: int, gz: int = 0) -> int | None:
    """Map integer grid position back to a model node index.

    If the grid cell does not exist in the model table, return None.  In LOS
    checking, missing cells are treated as blocked for safety.
    """
    gx = int(gx)
    gy = int(gy)
    gz = int(gz)
    if gx < 0 or gx >= len(cache.x_values):
        return None
    if gy < 0 or gy >= len(cache.y_values):
        return None
    if gz < 0 or gz >= len(cache.z_values):
        return None
    x = round(float(cache.x_values[gx]), 8)
    y = round(float(cache.y_values[gy]), 8)
    z = round(float(cache.z_values[gz]), 8) if cache.ndims == 3 else round(float(cache.z_values[0]), 8)
    return cache.xyz_to_idx.get((x, y, z), None)


def _bresenham_line_cells_2d(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Classic Bresenham line cells from (x0, y0) to (x1, y1).

    The returned list includes both endpoints.  Each pair is an integer grid
    cell coordinate, not a model node index.

    This is intentionally simple and deterministic.  It follows the same idea
    as the MATLAB line_of_sight() code you showed: walk along the dominant axis
    and step the other axis when the accumulated error requires it.
    """
    x0 = int(x0); y0 = int(y0); x1 = int(x1); y1 = int(y1)
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else (-1 if x0 > x1 else 0)
    sy = 1 if y0 < y1 else (-1 if y0 > y1 else 0)

    x = x0
    y = y0
    cells: list[tuple[int, int]] = [(x, y)]

    if dx >= dy:
        err = dx / 2.0
        while x != x1:
            x += sx
            err -= dy
            if err < 0:
                y += sy
                err += dx
            cells.append((x, y))
    else:
        err = dy / 2.0
        while y != y1:
            y += sy
            err -= dx
            if err < 0:
                x += sx
                err += dy
            cells.append((x, y))
    return cells


def _bresenham_segment_indices(model, cache: GridCache, idx1: int, idx2: int) -> list[int] | None:
    """Return only the ordered Bresenham line-cell model indices.

    Important v1 plot fix
    ---------------------
    ``bresenham_clearance_cells`` is used only as a *safety check* around each
    line cell.  Clearance cells must NOT be appended to the exported path.
    Otherwise ``THETASTAR_OUTPUT_SAMPLED_PATH=True`` plots a saw-tooth/triangle
    route, because the path connects all neighboring clearance cells in order.

    Return value:
      - ordered center cells along the Bresenham line
      - None if any center or clearance cell is missing / no-fly
    """
    gx0, gy0, gz0 = _node_grid_position(model, cache, int(idx1))
    gx1, gy1, gz1 = _node_grid_position(model, cache, int(idx2))

    # Current project is 2D.  If 3D appears later, only allow Bresenham inside
    # the same z layer; otherwise fall back to continuous sampling.
    if cache.ndims == 3 and gz0 != gz1:
        return _sample_segment_indices(model, cache, idx1, idx2)

    line_cells = _bresenham_line_cells_2d(gx0, gy0, gx1, gy1)
    clearance = max(0, int(getattr(cache, "bresenham_clearance_cells", 0)))

    sampled: list[int] = []
    seen: set[int] = set()

    for gx, gy in line_cells:
        # 1) Validate the center line cell.
        center_idx = _model_index_from_grid_position(cache, gx, gy, gz0)
        if center_idx is None:
            return None
        center_idx = int(center_idx)

        if center_idx not in cache.valid_indices:
            return None

        # 2) Validate optional clearance cells around the center line cell.
        #    These cells are checked for safety but are not returned/exported.
        for dx in range(-clearance, clearance + 1):
            for dy in range(-clearance, clearance + 1):
                mi = _model_index_from_grid_position(cache, gx + dx, gy + dy, gz0)
                if mi is None:
                    return None
                if int(mi) not in cache.valid_indices:
                    return None

        # 3) Export only the ordered center line cells.
        if center_idx not in seen:
            sampled.append(center_idx)
            seen.add(center_idx)

    # Keep exact endpoints.
    if not sampled or sampled[0] != int(idx1):
        sampled.insert(0, int(idx1))
    if sampled[-1] != int(idx2):
        sampled.append(int(idx2))

    # Remove repeated consecutive nodes.
    cleaned: list[int] = []
    for v in sampled:
        v = int(v)
        if not cleaned or cleaned[-1] != v:
            cleaned.append(v)

    return cleaned



def _route_corridor_segment_indices(model, cache: GridCache, idx1: int, idx2: int) -> list[int] | None:
    """Geometry-based LOS for the LAE-UTM lon/lat/KDTree graph.

    Why this exists
    ---------------
    Bresenham assumes every node can be mapped to a clean integer row/column
    grid.  Your current graph is built from lon/lat points with KDTree
    neighbors, so the model is closer to a point cloud than a perfect raster.
    In that situation Bresenham can reject every shortcut because the implied
    row/column cell does not exist, even when the visual path section is a
    straight safe corridor.

    Corridor model
    --------------
    The route is a circular prism (cylinder): the straight segment idx1->idx2
    is the central line, swept by a circle of radius corridor_radius_m in the
    map plane. THETASTAR_ROUTE_WIDTH_M is that circle's diameter -- the width
    of the corridor band you would see on a 2D map, and the diameter of the
    cylindrical safety envelope in 3D. A shortcut is accepted only if:
      1. both endpoints are traversable; and
      2. the ENTIRE corridor -- every point within corridor_radius_m of the
         straight segment, not just the centerline -- stays clear of every
         blocked/no-fly node center.

    For a 50 m grid spacing, THETASTAR_ROUTE_WIDTH_M = 50.0 (radius 25 m) is
    approximately half a cell width; it rejects a line that would cut through
    the footprint of a no-fly cell, while still allowing useful straight
    shortcuts in open flyable areas.

    Return value
    ------------
    This returns [idx1, idx2] when the corridor is clear.  The direct segment
    cost then uses the true metric endpoint distance.  If blocked, it returns
    None and the smoother keeps the next A* waypoint.
    """
    idx1 = int(idx1)
    idx2 = int(idx2)

    if idx1 not in cache.valid_indices or idx2 not in cache.valid_indices:
        return None

    node_xy = cache.node_xy_m or {}
    if idx1 not in node_xy or idx2 not in node_xy:
        return None

    ax, ay = node_xy[idx1]
    bx, by = node_xy[idx2]
    radius = float(getattr(cache, "corridor_radius_m", 25.0))
    if not math.isfinite(radius) or radius < 0.0:
        radius = 25.0

    blocked_x = getattr(cache, "blocked_x_m", None)
    blocked_y = getattr(cache, "blocked_y_m", None)
    if blocked_x is None or blocked_y is None or len(blocked_x) == 0:
        return [idx1, idx2]

    # Fast bounding-box prefilter: only no-fly nodes near the segment could
    # be inside this corridor.  This keeps thousands of LOS checks cheap.
    xmin = min(ax, bx) - radius
    xmax = max(ax, bx) + radius
    ymin = min(ay, by) - radius
    ymax = max(ay, by) + radius

    bx_arr = np.asarray(blocked_x, dtype=float)
    by_arr = np.asarray(blocked_y, dtype=float)
    mask = (bx_arr >= xmin) & (bx_arr <= xmax) & (by_arr >= ymin) & (by_arr <= ymax)
    if not np.any(mask):
        return [idx1, idx2]

    px = bx_arr[mask]
    py = by_arr[mask]

    vx = bx - ax
    vy = by - ay
    seg2 = vx * vx + vy * vy
    if seg2 <= 0.0:
        dist = np.sqrt((px - ax) ** 2 + (py - ay) ** 2)
    else:
        t = ((px - ax) * vx + (py - ay) * vy) / seg2
        t = np.clip(t, 0.0, 1.0)
        cx = ax + t * vx
        cy = ay + t * vy
        dist = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)

    # If any blocked/no-fly node center falls inside the corridor's radius,
    # the cylinder is not clear and the shortcut must be rejected.
    if np.any(dist <= radius):
        return None

    return [idx1, idx2]


def _los_segment_indices(model, cache: GridCache, idx1: int, idx2: int) -> list[int] | None:
    """Return crossed model indices for the configured LOS method.

    This wrapper makes LOS method selection explicit and easy to debug.
    Set in params/thetastar.params:

        THETASTAR_LINE_OF_SIGHT_METHOD = "point_clearance"  # recommended: route-as-cylinder corridor check, benchmarked faster on this model (aliases: "corridor", "route_corridor")
        THETASTAR_LINE_OF_SIGHT_METHOD = "bresenham"         # only faster on a true rectangular grid; over-rejects and is slower here
        THETASTAR_LINE_OF_SIGHT_METHOD = "hybrid"            # bresenham first, falls back to the corridor check if the cell is missing
        THETASTAR_LINE_OF_SIGHT_METHOD = "sample"            # old continuous-sampling method
    """
    method = str(getattr(cache, "line_of_sight_method", "point_clearance") or "point_clearance").strip().lower()

    # Recommended for this LAE-UTM model: no row/column assumption, but still
    # blocks any shortcut whose route corridor (circular prism, see
    # _route_corridor_segment_indices()) passes too close to no-fly nodes.
    if method in ("point_clearance", "clearance", "geometric", "point", "corridor", "route_corridor"):
        return _route_corridor_segment_indices(model, cache, idx1, idx2)

    # Hybrid: try strict Bresenham first.  If the implied row/column cell is
    # missing, fall back to the route-corridor check instead of rejecting
    # everything.
    if method in ("hybrid", "bresenham_then_clearance"):
        seg = _bresenham_segment_indices(model, cache, idx1, idx2)
        if seg is not None:
            return seg
        return _route_corridor_segment_indices(model, cache, idx1, idx2)

    if method in ("bresenham", "grid", "grid_bresenham"):
        return _bresenham_segment_indices(model, cache, idx1, idx2)

    return _sample_segment_indices(model, cache, idx1, idx2)


def _sample_segment_indices(model, cache: GridCache, idx1: int, idx2: int) -> list[int] | None:
    x1, y1, z1 = _coord_tuple(model, int(idx1))
    x2, y2, z2 = _coord_tuple(model, int(idx2))

    # Use coordinate-space length only to decide how many grid samples to take.
    # Cost uses true metric distance below.
    length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    if not math.isfinite(length):
        return None
    if length <= 0.0:
        return [int(idx1)]

    n_steps = max(1, int(math.ceil(length / max(cache.step, 1.0e-12))))

    sampled: list[int] = []
    last_idx: int | None = None
    for k in range(n_steps + 1):
        t = k / n_steps
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        z = z1 + t * (z2 - z1)
        nearest_idx = _nearest_model_index(cache, x, y, z)
        if nearest_idx is None:
            return None
        nearest_idx = int(nearest_idx)
        if nearest_idx != last_idx:
            sampled.append(nearest_idx)
            last_idx = nearest_idx

    if sampled and sampled[0] != int(idx1):
        sampled.insert(0, int(idx1))
    if sampled and sampled[-1] != int(idx2):
        sampled.append(int(idx2))
    return sampled


def _line_of_sight(model, cache: GridCache, idx1: int, idx2: int, los_cache: dict[tuple[int, int], bool]) -> bool:
    idx1 = int(idx1)
    idx2 = int(idx2)
    key = (idx1, idx2) if idx1 <= idx2 else (idx2, idx1)
    if key in los_cache:
        return los_cache[key]

    if idx1 not in cache.valid_indices or idx2 not in cache.valid_indices:
        los_cache[key] = False
        return False

    sampled = _los_segment_indices(model, cache, idx1, idx2)
    if sampled is None:
        los_cache[key] = False
        return False

    ok = all(int(i) in cache.valid_indices for i in sampled)
    los_cache[key] = bool(ok)
    return bool(ok)


def _direct_segment_cost(model, cache: GridCache, idx1: int, idx2: int, cost_cache: dict[tuple[int, int], float]) -> float:
    idx1 = int(idx1)
    idx2 = int(idx2)
    key = (idx1, idx2) if idx1 <= idx2 else (idx2, idx1)
    if key in cost_cache:
        return cost_cache[key]

    sampled = _los_segment_indices(model, cache, idx1, idx2)
    if sampled is None or len(sampled) == 0:
        cost_cache[key] = math.inf
        return math.inf
    if len(sampled) == 1:
        cost_cache[key] = 0.0
        return 0.0

    total = 0.0
    for a, b in zip(sampled[:-1], sampled[1:]):
        d_m = _distance_m(model, int(a), int(b), lonlat_as_meters=cache.lonlat_as_meters, coordinates_are_lonlat=cache.coordinates_are_lonlat)
        s1 = _slowness(model, int(a))
        s2 = _slowness(model, int(b))
        if not (math.isfinite(d_m) and math.isfinite(s1) and math.isfinite(s2)):
            total = math.inf
            break
        total += d_m * 0.5 * (s1 + s2)

    cost_cache[key] = float(total)
    return float(total)


def _graph_edge_cost(model, graph, idx1: int, idx2: int, cache: GridCache | None = None) -> float:
    """Cost for one normal graph edge, A*-style.

    This intentionally does NOT sample the straight segment through the grid.
    For a graph neighbor, build_grid_graph()/iter_neighbors() already decided
    that the edge is valid.  Sampling even one normal edge can be too strict on
    lon/lat grids or forced DB/DK/FLZ endpoint cells and can make Theta* fail
    where A* succeeds.

    Cost = distance_m * average endpoint slowness.
    If the graph stores an explicit numeric edge cost, use it when available.
    """
    idx1 = int(idx1)
    idx2 = int(idx2)

    # Try common adjacency formats with explicit weights first.
    if isinstance(graph, dict):
        for key in ("neighbors", "adjacency", "adj", "edges"):
            adj = graph.get(key, None)
            if adj is None:
                continue
            try:
                items = adj.get(idx1, [])
            except AttributeError:
                try:
                    items = adj[idx1]
                except Exception:
                    items = []
            except Exception:
                items = []

            for item in items or []:
                nb = None
                cost = None
                if isinstance(item, dict):
                    for k in ("to", "target", "node", "idx", "index", "neighbor"):
                        if k in item:
                            try:
                                nb = int(item[k])
                            except Exception:
                                nb = None
                            break
                    for k in ("cost", "weight", "travel_cost", "time", "distance_cost"):
                        if k in item:
                            cost = item[k]
                            break
                elif isinstance(item, (tuple, list)) and len(item) >= 1:
                    try:
                        nb = int(item[0])
                    except Exception:
                        nb = None
                    if len(item) >= 2:
                        cost = item[1]
                else:
                    try:
                        nb = int(item)
                    except Exception:
                        nb = None

                if nb == idx2 and cost is not None:
                    try:
                        c = float(cost)
                        if math.isfinite(c) and c >= 0.0:
                            return c
                    except Exception:
                        pass

    d_m = _distance_m(
        model,
        idx1,
        idx2,
        lonlat_as_meters=True if cache is None else cache.lonlat_as_meters,
        coordinates_are_lonlat=None if cache is None else cache.coordinates_are_lonlat,
    )
    s1 = _slowness(model, idx1)
    s2 = _slowness(model, idx2)
    if not (math.isfinite(d_m) and math.isfinite(s1) and math.isfinite(s2)):
        return math.inf
    return float(d_m * 0.5 * (s1 + s2))


def _minimum_positive_slowness(model) -> float:
    """Return the lowest positive slowness in the model.

    This is used only for the fallback heuristic.  Multiplying straight-line
    distance by the minimum slowness gives a conservative travel-time estimate.
    """
    try:
        vals = model["slowness"].astype(float).to_numpy()
        vals = vals[np.isfinite(vals) & (vals > 0.0)]
        return float(np.min(vals)) if len(vals) else 0.0
    except Exception:
        return 0.0


def _safe_heuristic(model, graph, idx: int, end_idx: int, *, heuristic_weight: float = 1.0, lonlat_as_meters: bool = True) -> float:
    """Use project heuristic_cost() when available, otherwise metric fallback."""
    h = None
    try:
        h = heuristic_cost(model=model, graph=graph, idx=int(idx), end_idx=int(end_idx))
    except TypeError:
        try:
            h = heuristic_cost(model, graph, int(idx), int(end_idx))
        except Exception:
            h = None
    except Exception:
        h = None

    try:
        h = float(h)
        if math.isfinite(h):
            return float(max(0.0, heuristic_weight) * h)
    except Exception:
        pass

    distance = _distance_m(model, int(idx), int(end_idx), lonlat_as_meters=lonlat_as_meters)
    return float(max(0.0, heuristic_weight) * distance * _minimum_positive_slowness(model))


def _call_project_astar(model, graph, start_idx: int, end_idx: int, kwargs: dict[str, Any]) -> dict | None:
    """Run the project astar.py implementation and wrap its result as thetastar.

    This is used for debug mode when THETASTAR_ALLOW_ANY_ANGLE=False.
    It guarantees the same behavior as the working A* module instead of
    re-implementing edge-cost details inside Theta*.
    """
    try:
        astar_mod = importlib.import_module("src.astar")
    except Exception:
        try:
            astar_mod = importlib.import_module("astar")
        except Exception:
            return None

    if not hasattr(astar_mod, "run"):
        return None

    run_fn = astar_mod.run

    # Filter kwargs if astar.run does not accept **kwargs.
    call_kwargs = dict(kwargs or {})
    try:
        sig = inspect.signature(run_fn)
        accepts_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
        if not accepts_var_kwargs:
            allowed = set(sig.parameters.keys())
            call_kwargs = {k: v for k, v in call_kwargs.items() if k in allowed}
    except Exception:
        call_kwargs = {}

    try:
        result = run_fn(
            model=model,
            graph=graph,
            start_idx=int(start_idx),
            end_idx=int(end_idx),
            **call_kwargs,
        )
    except TypeError:
        try:
            result = run_fn(model, graph, int(start_idx), int(end_idx))
        except Exception:
            return None
    except Exception:
        return None

    if not isinstance(result, dict):
        return None

    out = dict(result)

    # -----------------------------------------------------------------
    # IMPORTANT BUG FIX
    # -----------------------------------------------------------------
    # Some project A* implementations return path_indices but do not include
    # an explicit success=True field.  The LOS-smoothing wrapper checks
    # result["success"] before smoothing.  Without this inference, Theta* can
    # silently return the raw A* path unchanged, which is exactly the symptom
    # "Theta* still has the same 47 steps as A*".
    if "success" not in out:
        out["success"] = bool(out.get("path_indices"))

    out["algorithm"] = "thetastar"
    out["thetastar_any_angle"] = False
    out["thetastar_pure_astar_mode"] = True
    out["thetastar_project_astar_delegate"] = True
    out["message"] = str(out.get("message", "Path search completed.")) + " [Theta* base path from project A*]"
    if "expanded_states" not in out and "expanded_nodes" in out:
        out["expanded_states"] = out.get("expanded_nodes")
    if "k_paths_found" not in out:
        out["k_paths_found"] = 1 if out.get("path_indices") else 0
    return out


# ============================================================
# Theta* main algorithm
# ============================================================



def _path_direct_segment_cost(model, cache: GridCache, path: list[int]) -> float:
    """Cost of a waypoint path whose segments are accepted by line-of-sight."""
    if not path or len(path) <= 1:
        return 0.0
    cost_cache: dict[tuple[int, int], float] = {}
    total = 0.0
    for a, b in zip(path[:-1], path[1:]):
        c = _direct_segment_cost(model, cache, int(a), int(b), cost_cache)
        if not math.isfinite(c):
            return math.inf
        total += float(c)
    return float(total)




def _segment_reference_lat_deg(model, cache: GridCache, idx1: int, idx2: int) -> float:
    """Reference latitude for local lon/lat-to-meter conversion."""
    if not cache.coordinates_are_lonlat:
        return 0.0
    _, y1, _ = _coord_tuple(model, int(idx1))
    _, y2, _ = _coord_tuple(model, int(idx2))
    return float(0.5 * (y1 + y2))


def _node_xy_m_for_segment(model, cache: GridCache, idx: int, ref_lat_deg: float) -> tuple[float, float]:
    """Return local metric x/y for a node for straightness tests.

    For projected grids, x/y are already meters.  For lon/lat grids, this
    converts degrees to local meters using one reference latitude for the
    segment.  The value is only used for geometry diagnostics/smoothing, not
    for changing the model coordinates.
    """
    x, y, _ = _coord_tuple(model, int(idx))
    if cache.coordinates_are_lonlat:
        lat0 = math.radians(float(ref_lat_deg))
        return float(x * 111_320.0 * math.cos(lat0)), float(y * 110_540.0)
    return float(x), float(y)


def _point_to_segment_distance_m(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    """Shortest 2D distance from point P to line segment AB, in meters."""
    vx = bx - ax
    vy = by - ay
    wx = px - ax
    wy = py - ay
    denom = vx * vx + vy * vy
    if denom <= 0.0:
        return float(math.sqrt((px - ax) ** 2 + (py - ay) ** 2))
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / denom))
    cx = ax + t * vx
    cy = ay + t * vy
    return float(math.sqrt((px - cx) ** 2 + (py - cy) ** 2))


def _path_segment_is_straight_enough(
    model,
    cache: GridCache,
    path: list[int],
    i: int,
    j: int,
    *,
    tolerance_m: float = 75.0,
) -> tuple[bool, float]:
    """Check whether A* path nodes i..j already form a near-straight corridor.

    Why this helper is needed
    -------------------------
    The strict LOS checker samples the continuous straight line and snaps the
    samples to nearby grid nodes.  On a coarse 50 m grid, that can sometimes
    touch a neighboring blocked/no-fly node even though the original A* path
    clearly follows a safe nearly-straight corridor.  When that happens, the
    smoother keeps every A* step, so the zoom figure still shows 1,2,3,...

    This helper is a safe practical fallback for visualization/routing:
      - it only considers nodes that are already on the valid A* path;
      - it allows a shortcut only when all intermediate A* nodes lie close to
        the straight segment from path[i] to path[j].

    Return
    ------
    ok : bool
        True when the path section is straight enough to replace by one segment.
    max_deviation_m : float
        Maximum intermediate-node distance from the segment.
    """
    i = int(i)
    j = int(j)
    if j <= i + 1:
        return True, 0.0

    # All original A* nodes in this section must be traversable.
    # This preserves the A* safety corridor even when strict sampled LOS fails.
    for node in path[i : j + 1]:
        if int(node) not in cache.valid_indices:
            return False, math.inf

    ref_lat = _segment_reference_lat_deg(model, cache, path[i], path[j])
    ax, ay = _node_xy_m_for_segment(model, cache, path[i], ref_lat)
    bx, by = _node_xy_m_for_segment(model, cache, path[j], ref_lat)

    max_dev = 0.0
    for node in path[i + 1 : j]:
        px, py = _node_xy_m_for_segment(model, cache, int(node), ref_lat)
        d = _point_to_segment_distance_m(px, py, ax, ay, bx, by)
        if not math.isfinite(d):
            return False, math.inf
        max_dev = max(max_dev, float(d))
        if max_dev > float(tolerance_m):
            return False, float(max_dev)

    return True, float(max_dev)

def _smooth_path_by_line_of_sight(
    model,
    cache: GridCache,
    raw_path: list[int],
    *,
    max_lookahead_nodes: int | None = 300,
    straight_path_fallback: bool = True,
    straight_path_tolerance_m: float = 75.0,
) -> tuple[list[int], dict[str, int | float]]:
    """Post-process a valid A* path using Theta* line-of-sight shortcuts.

    This is the safer fix for this project:
      1. A* first finds a guaranteed graph-valid path.
      2. LOS is used only to remove intermediate waypoints.
      3. If one LOS shortcut is blocked, we keep the next normal A* waypoint.

    Therefore the algorithm cannot fail only because LOS is too strict.
    """
    path = [int(v) for v in (raw_path or [])]
    if len(path) <= 2:
        return path, {
            "raw_nodes": int(len(path)),
            "smoothed_nodes": int(len(path)),
            "los_checks": 0,
            "los_shortcuts": 0,
            "straight_fallback_checks": 0,
            "straight_fallback_shortcuts": 0,
            "straight_fallback_max_deviation_m": 0.0,
        }

    # None or <=0 means unlimited lookahead.  A finite value is much faster
    # on long paths and still gives useful any-angle smoothing.
    if max_lookahead_nodes is not None:
        try:
            max_lookahead_nodes = int(max_lookahead_nodes)
            if max_lookahead_nodes <= 0:
                max_lookahead_nodes = None
        except Exception:
            max_lookahead_nodes = 300

    los_cache: dict[tuple[int, int], bool] = {}
    out: list[int] = [path[0]]

    i = 0
    los_checks = 0
    los_shortcuts = 0
    straight_fallback_checks = 0
    straight_fallback_shortcuts = 0
    straight_fallback_max_deviation_m = 0.0

    n = len(path)
    while i < n - 1:
        if max_lookahead_nodes is None:
            j_start = n - 1
        else:
            j_start = min(n - 1, i + int(max_lookahead_nodes))

        # Default: keep the next A* waypoint.  This preserves the valid path.
        best_j = i + 1

        # Try the farthest valid true LOS shortcut first.
        # If strict LOS is too conservative for this coarse grid, optionally
        # allow a near-straight A* corridor shortcut.  This is what lets the
        # zoom figure show jumps such as 1 -> 8 or 38 -> 45 when those A*
        # nodes already lie on a safe, nearly straight corridor.
        for j in range(j_start, i + 1, -1):
            los_checks += 1
            if _line_of_sight(model, cache, path[i], path[j], los_cache):
                best_j = j
                if j > i + 1:
                    los_shortcuts += 1
                break

            if straight_path_fallback and j > i + 1:
                straight_fallback_checks += 1
                ok_straight, max_dev = _path_segment_is_straight_enough(
                    model,
                    cache,
                    path,
                    i,
                    j,
                    tolerance_m=straight_path_tolerance_m,
                )
                if math.isfinite(max_dev):
                    straight_fallback_max_deviation_m = max(
                        straight_fallback_max_deviation_m,
                        float(max_dev),
                    )
                if ok_straight:
                    best_j = j
                    straight_fallback_shortcuts += 1
                    break

        out.append(path[best_j])
        i = best_j

    return out, {
        "raw_nodes": int(len(path)),
        "smoothed_nodes": int(len(out)),
        "los_checks": int(los_checks),
        "los_shortcuts": int(los_shortcuts),
        "straight_fallback_checks": int(straight_fallback_checks),
        "straight_fallback_shortcuts": int(straight_fallback_shortcuts),
        "straight_fallback_max_deviation_m": float(straight_fallback_max_deviation_m),
    }


def _astar_then_los_smooth(
    model,
    graph,
    start_idx: int,
    end_idx: int,
    kwargs: dict[str, Any],
    *,
    los_step_factor: float,
    lonlat_as_meters: bool,
    output_sampled_path: bool,
    max_lookahead_nodes: int | None,
    straight_path_fallback: bool,
    straight_path_tolerance_m: float,
    line_of_sight_method: str,
    bresenham_clearance_cells: int,
    route_width_m: float,
) -> dict | None:
    """Run project A*, then smooth the found path using LOS.

    This fixes the current LOS issue by moving LOS out of the search loop.
    A* controls reachability; LOS only improves the geometry after success.
    """
    delegated = _call_project_astar(model, graph, start_idx, end_idx, kwargs)
    if delegated is None:
        return None

    result = dict(delegated)
    result["algorithm"] = "thetastar"
    result["thetastar_mode"] = "astar_then_los_smooth"

    if not bool(result.get("success", False)):
        result["message"] = "A* delegate failed before LOS smoothing. " + str(result.get("message", ""))
        return result

    raw_path = [int(v) for v in result.get("path_indices", [])]
    if len(raw_path) <= 2:
        result["thetastar_raw_astar_nodes"] = int(len(raw_path))
        result["thetastar_smoothed_nodes"] = int(len(raw_path))
        result["thetastar_los_checks"] = 0
        result["thetastar_los_shortcuts"] = 0
        return result

    cache = _make_grid_cache(
        model,
        graph,
        line_of_sight_step_factor=los_step_factor,
        lonlat_as_meters=lonlat_as_meters,
        line_of_sight_method=line_of_sight_method,
        bresenham_clearance_cells=bresenham_clearance_cells,
        route_width_m=route_width_m,
    )

    smoothed, stats = _smooth_path_by_line_of_sight(
        model,
        cache,
        raw_path,
        max_lookahead_nodes=max_lookahead_nodes,
        straight_path_fallback=straight_path_fallback,
        straight_path_tolerance_m=straight_path_tolerance_m,
    )

    # Optional: expand accepted straight segments back to sampled grid nodes.
    # Keep False if you want the report figure to show the any-angle segments.
    export_path = expand_theta_path_to_sampled_nodes(model, cache, smoothed) if output_sampled_path else smoothed

    direct_cost = _path_direct_segment_cost(model, cache, smoothed)
    if math.isfinite(direct_cost):
        result["total_cost"] = float(direct_cost)
        result["travel_cost"] = float(direct_cost)
        result["cost"] = float(direct_cost)

    result["path_indices"] = [int(v) for v in export_path]
    result["thetastar_raw_astar_nodes"] = int(stats.get("raw_nodes", len(raw_path)))
    result["thetastar_smoothed_nodes"] = int(stats.get("smoothed_nodes", len(smoothed)))
    result["thetastar_export_nodes"] = int(len(export_path))
    result["thetastar_los_checks"] = int(stats.get("los_checks", 0))
    result["thetastar_los_shortcuts"] = int(stats.get("los_shortcuts", 0))
    result["thetastar_straight_fallback_checks"] = int(stats.get("straight_fallback_checks", 0))
    result["thetastar_straight_fallback_shortcuts"] = int(stats.get("straight_fallback_shortcuts", 0))
    result["thetastar_straight_fallback_max_deviation_m"] = float(stats.get("straight_fallback_max_deviation_m", 0.0))
    result["thetastar_straight_path_fallback"] = bool(straight_path_fallback)
    result["thetastar_straight_path_tolerance_m"] = float(straight_path_tolerance_m)
    result["thetastar_line_of_sight_step_factor"] = float(los_step_factor)
    result["thetastar_line_of_sight_method"] = str(line_of_sight_method)
    result["thetastar_bresenham_clearance_cells"] = int(bresenham_clearance_cells)
    result["thetastar_route_width_m"] = float(route_width_m)
    result["thetastar_corridor_radius_m"] = max(0.0, float(route_width_m)) / 2.0
    result["thetastar_los_smooth_max_lookahead_nodes"] = (
        None if max_lookahead_nodes is None else int(max_lookahead_nodes)
    )
    result["message"] = (
        "Path found by A* and post-smoothed by Theta* line-of-sight. "
        f"nodes: {len(raw_path)} -> {len(smoothed)}"
    )

    # Print a compact debug line when THETASTAR_VERBOSE=True.
    # This is the fastest way to confirm that this file is loaded and that
    # smoothing actually ran.  If this line does not appear, main.py is not
    # using this thetastar.py file or the params are not being loaded.
    verbose = _as_bool(_param(kwargs, "THETASTAR_VERBOSE", _param(kwargs, "verbose", False)), False)
    show_progress = _as_bool(_param(kwargs, "THETASTAR_SHOW_PROGRESS", False), False)
    progress_label = str(_param(kwargs, "THETASTAR_PROGRESS_LABEL", "") or "")
    if show_progress:
        # The A* search itself runs inside src/astar.py, which this file does
        # not control, so there is no live percentage while it is running.
        # What we CAN report honestly is the smoothing pass that follows it.
        suffix = f"raw={len(raw_path):,}->smoothed={len(smoothed):,} nodes, LOS={int(stats.get('los_checks', 0)):,} checks"
        print(_progress_bar_text(1, 1, progress_label, suffix=f"done ({suffix})"))
    if verbose:
        print(
            "[thetastar] A* -> LOS smoothing: "
            f"raw={len(raw_path):,}, smoothed={len(smoothed):,}, "
            f"export={len(export_path):,}, "
            f"LOS checks={int(stats.get('los_checks', 0)):,}, "
            f"LOS shortcuts={int(stats.get('los_shortcuts', 0)):,}, "
            f"method={line_of_sight_method}, "
            f"bresenham_clearance={int(bresenham_clearance_cells)}, "
            f"route_width_m={float(route_width_m):.1f} (corridor_radius_m={float(route_width_m) / 2.0:.1f})"
        )

    return result

def run(model, graph, start_idx: int, end_idx: int, **kwargs) -> dict:
    """Main entry point called by main.py.

    Parameters
    ----------
    model : pandas.DataFrame
        Grid node table with coordinates, slowness, labels, and other fields.
    graph : dict
        Graph built by build_grid_graph(); must contain traversable nodes in
        graph["valid_indices"] and neighbor information used by iter_neighbors().
    start_idx, end_idx : int
        Actual node indices selected by main.py from START_LABEL/END_LABEL or
        start/end coordinates.
    **kwargs
        Algorithm parameters loaded from params/thetastar.params and/or
        parameters.py.

    Returns
    -------
    dict
        main.py-compatible result dictionary.  Important fields include
        success, path_indices, total_cost, expanded_nodes, runtime_seconds,
        and Theta*-specific diagnostic fields.
    """
    t0 = time.time()

    start_idx = int(start_idx)
    end_idx = int(end_idx)

    # ------------------------------------------------------------
    # Read parameters.  All of these can come from params/thetastar.params,
    # from parameters.py, or from kwargs passed by main.py.
    # ------------------------------------------------------------
    heuristic_weight = float(_param(kwargs, "THETASTAR_HEURISTIC_WEIGHT", _param(kwargs, "heuristic_weight", 1.0)))
    max_expansions = _param(kwargs, "THETASTAR_MAX_EXPANSIONS", _param(kwargs, "max_expansions", None))
    max_expansions = None if max_expansions is None else int(max_expansions)
    allow_any_angle = _as_bool(_param(kwargs, "THETASTAR_ALLOW_ANY_ANGLE", True), True)
    los_step_factor = float(_param(kwargs, "THETASTAR_LINE_OF_SIGHT_STEP_FACTOR", 0.5))
    line_of_sight_method = str(_param(kwargs, "THETASTAR_LINE_OF_SIGHT_METHOD", "point_clearance") or "point_clearance").strip().lower()
    bresenham_clearance_cells = max(0, _as_int(_param(kwargs, "THETASTAR_BRESENHAM_CLEARANCE_CELLS", 0), 0))
    # Route corridor: modeled as a circular prism (cylinder) around the
    # centerline. THETASTAR_ROUTE_WIDTH_M is the corridor's width on the 2D
    # map, which is that cylinder's diameter in 3D; the geometry checks use
    # half of it as the actual radius (see _route_corridor_segment_indices).
    route_width_m = float(_param(kwargs, "THETASTAR_ROUTE_WIDTH_M", 50.0))
    lonlat_as_meters = _as_bool(_param(kwargs, "THETASTAR_LONLAT_AS_METERS", True), True)
    verbose = _as_bool(_param(kwargs, "THETASTAR_VERBOSE", _param(kwargs, "verbose", False)), False)
    show_progress = _as_bool(_param(kwargs, "THETASTAR_SHOW_PROGRESS", False), False)
    progress_interval_s = float(_param(kwargs, "THETASTAR_PROGRESS_INTERVAL_S", 0.15))
    progress_label = str(_param(kwargs, "THETASTAR_PROGRESS_LABEL", "") or "")
    if show_progress:
        print(f"\r[thetastar] {progress_label} starting search...", end="", flush=True)
    output_sampled_path = _as_bool(_param(kwargs, "THETASTAR_OUTPUT_SAMPLED_PATH", False), False)
    edge_fallback_to_astar = _as_bool(_param(kwargs, "THETASTAR_EDGE_FALLBACK_TO_ASTAR", True), True)
    pure_astar_delegate = _as_bool(_param(kwargs, "THETASTAR_PURE_ASTAR_DELEGATE", True), True)
    astar_first_los_smooth = _as_bool(_param(kwargs, "THETASTAR_ASTAR_FIRST_LOS_SMOOTH", True), True)
    los_smooth_max_lookahead = _param(kwargs, "THETASTAR_LOS_SMOOTH_MAX_LOOKAHEAD_NODES", 300)
    if los_smooth_max_lookahead is not None:
        try:
            los_smooth_max_lookahead = int(los_smooth_max_lookahead)
            if los_smooth_max_lookahead <= 0:
                los_smooth_max_lookahead = None
        except Exception:
            los_smooth_max_lookahead = 300

    # If strict LOS rejects shortcuts on a coarse grid, allow shortcuts along
    # A* sections that are already nearly straight.  This makes the exported
    # Theta* path show useful jumps such as 1 -> 8 or 38 -> 45.
    straight_path_fallback = _as_bool(
        _param(kwargs, "THETASTAR_LOS_STRAIGHT_PATH_FALLBACK", False),
        False,
    )
    straight_path_tolerance_m = float(
        _param(kwargs, "THETASTAR_LOS_STRAIGHT_PATH_TOLERANCE_M", 75.0)
    )

    # Recommended safe mode for this project:
    # First use the working A* algorithm to guarantee reachability, then use
    # line-of-sight only as a post-smoothing step.  This is the key LOS fix.
    if astar_first_los_smooth:
        smoothed_result = _astar_then_los_smooth(
            model,
            graph,
            start_idx,
            end_idx,
            kwargs,
            los_step_factor=los_step_factor,
            lonlat_as_meters=lonlat_as_meters,
            output_sampled_path=output_sampled_path,
            max_lookahead_nodes=los_smooth_max_lookahead,
            straight_path_fallback=straight_path_fallback,
            straight_path_tolerance_m=straight_path_tolerance_m,
            line_of_sight_method=line_of_sight_method,
            bresenham_clearance_cells=bresenham_clearance_cells,
            route_width_m=route_width_m,
        )
        if smoothed_result is not None:
            return smoothed_result
        # If project A* cannot be imported/called, continue to internal search.

    # Debug/safe mode: when any-angle is off, run the same project A* module.
    if (not allow_any_angle) and pure_astar_delegate:
        delegated = _call_project_astar(model, graph, start_idx, end_idx, kwargs)
        if delegated is not None:
            return delegated
        # If project A* cannot be imported/called, continue with internal
        # graph-only fallback below.

    valid_indices = _valid_index_set(model, graph)

    def _fail(message: str, expanded: int = 0, visited: int = 0) -> dict:
        if show_progress:
            shown = expanded or expanded_nodes
            print(_progress_bar_text(shown, max_expansions or max(shown, 1), progress_label, suffix="failed"))
        return {
            "success": False,
            "algorithm": "thetastar",
            "message": str(message),
            "path_indices": [],
            "total_cost": None,
            "expanded_nodes": int(expanded),
            "expanded_states": int(expanded),
            "visited_nodes": int(visited),
            "runtime_seconds": float(time.time() - t0),
            "k_paths_found": 0,
        }

    if start_idx not in valid_indices:
        return _fail("Start node is blocked or not traversable.")
    if end_idx not in valid_indices:
        return _fail("End node is blocked or not traversable.")

    cache = _make_grid_cache(
        model,
        graph,
        line_of_sight_step_factor=los_step_factor,
        lonlat_as_meters=lonlat_as_meters,
        line_of_sight_method=line_of_sight_method,
        bresenham_clearance_cells=bresenham_clearance_cells,
        route_width_m=route_width_m,
    )

    open_heap: list[tuple[float, int, int]] = []
    heap_counter = 0
    g_score: dict[int, float] = {start_idx: 0.0}
    came_from: dict[int, int] = {start_idx: start_idx}
    visited: set[int] = set()
    expanded_nodes = 0
    last_progress_t = time.time()

    los_cache: dict[tuple[int, int], bool] = {}
    cost_cache: dict[tuple[int, int], float] = {}
    neighbors_considered = 0
    edge_relaxations = 0
    parent_shortcuts = 0
    skipped_nonfinite_edges = 0

    f_start = _safe_heuristic(
        model,
        graph,
        start_idx,
        end_idx,
        heuristic_weight=heuristic_weight,
        lonlat_as_meters=lonlat_as_meters,
    )
    heapq.heappush(open_heap, (float(f_start), heap_counter, start_idx))

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        current = int(current)

        if current in visited:
            continue

        visited.add(current)
        expanded_nodes += 1

        if show_progress:
            now = time.time()
            if (now - last_progress_t) >= progress_interval_s:
                print(_progress_bar_text(expanded_nodes, max_expansions, progress_label), end="", flush=True)
                last_progress_t = now

        if max_expansions is not None and expanded_nodes > max_expansions:
            return _fail(
                f"Maximum expansions reached: {max_expansions}",
                expanded=expanded_nodes,
                visited=len(visited),
            )

        if current == end_idx:
            path = reconstruct_path(came_from, current)
            if output_sampled_path:
                path = expand_theta_path_to_sampled_nodes(model, cache, path)
            total_cost = float(g_score[current])
            if show_progress:
                print(_progress_bar_text(expanded_nodes, max_expansions or expanded_nodes, progress_label,
                                          suffix=f"done, {len(path):,} waypoints"))
            if verbose:
                print(f"[thetastar] Path found: nodes={len(path):,}, cost={total_cost:.6g}, expanded={expanded_nodes:,}")
            return {
                "success": True,
                "algorithm": "thetastar",
                "message": "Path found.",
                "path_indices": [int(v) for v in path],
                "total_cost": total_cost,
                "travel_cost": total_cost,
                "cost": total_cost,
                "expanded_nodes": int(expanded_nodes),
                "expanded_states": int(expanded_nodes),
                "visited_nodes": int(len(visited)),
                "runtime_seconds": float(time.time() - t0),
                "k_paths_found": 1,
                "thetastar_neighbors_considered": int(neighbors_considered),
                "thetastar_edge_relaxations": int(edge_relaxations),
                "thetastar_parent_shortcuts": int(parent_shortcuts),
                "thetastar_skipped_nonfinite_edges": int(skipped_nonfinite_edges),
                "thetastar_edge_fallback_to_astar": bool(edge_fallback_to_astar),
                "heuristic_weight": float(heuristic_weight),
                "thetastar_any_angle": bool(allow_any_angle),
                "thetastar_line_of_sight_checks": int(len(los_cache)),
                "thetastar_line_of_sight_method": str(line_of_sight_method),
                "thetastar_bresenham_clearance_cells": int(bresenham_clearance_cells),
                "thetastar_route_width_m": float(route_width_m),
                "thetastar_corridor_radius_m": max(0.0, float(route_width_m)) / 2.0,
                "thetastar_segment_cost_cache": int(len(cost_cache)),
            }

        current_parent = int(came_from.get(current, current))

        for neighbor in iter_neighbors(model, graph, current):
            neighbor = int(neighbor)
            neighbors_considered += 1
            if neighbor in visited:
                continue
            if neighbor not in valid_indices:
                continue

            use_parent_shortcut = False
            if allow_any_angle and current_parent != current:
                use_parent_shortcut = _line_of_sight(
                    model=model,
                    cache=cache,
                    idx1=current_parent,
                    idx2=neighbor,
                    los_cache=los_cache,
                )

            if use_parent_shortcut:
                candidate_parent = current_parent
                base_g = g_score.get(current_parent, math.inf)
                segment_cost = _direct_segment_cost(model, cache, current_parent, neighbor, cost_cache)
                if not math.isfinite(segment_cost):
                    # Safety fallback: never let a bad sampled any-angle segment
                    # block the normal graph-edge update.
                    candidate_parent = current
                    base_g = g_score.get(current, math.inf)
                    segment_cost = _graph_edge_cost(model, graph, current, neighbor, cache)
                else:
                    parent_shortcuts += 1
            else:
                candidate_parent = current
                base_g = g_score.get(current, math.inf)
                if edge_fallback_to_astar:
                    # A*-compatible fallback.  This is the key fix: normal
                    # neighbor moves use the graph edge directly and do not
                    # require line-of-sight sampling.
                    segment_cost = _graph_edge_cost(model, graph, current, neighbor, cache)
                else:
                    segment_cost = _direct_segment_cost(model, cache, current, neighbor, cost_cache)

            tentative_g = base_g + segment_cost
            if not math.isfinite(tentative_g):
                skipped_nonfinite_edges += 1
                continue

            if tentative_g < g_score.get(neighbor, math.inf):
                edge_relaxations += 1
                came_from[neighbor] = int(candidate_parent)
                g_score[neighbor] = float(tentative_g)
                f = tentative_g + _safe_heuristic(
                    model,
                    graph,
                    neighbor,
                    end_idx,
                    heuristic_weight=heuristic_weight,
                    lonlat_as_meters=lonlat_as_meters,
                )
                heap_counter += 1
                heapq.heappush(open_heap, (float(f), heap_counter, neighbor))

    out = _fail("No path found.", expanded=expanded_nodes, visited=len(visited))
    out.update({
        "thetastar_neighbors_considered": int(neighbors_considered),
        "thetastar_edge_relaxations": int(edge_relaxations),
        "thetastar_parent_shortcuts": int(parent_shortcuts),
        "thetastar_skipped_nonfinite_edges": int(skipped_nonfinite_edges),
        "thetastar_edge_fallback_to_astar": bool(edge_fallback_to_astar),
        "thetastar_line_of_sight_checks": int(len(los_cache)),
        "thetastar_line_of_sight_method": str(line_of_sight_method),
        "thetastar_bresenham_clearance_cells": int(bresenham_clearance_cells),
        "thetastar_route_width_m": float(route_width_m),
        "thetastar_corridor_radius_m": max(0.0, float(route_width_m)) / 2.0,
        "thetastar_segment_cost_cache": int(len(cost_cache)),
    })
    return out


# ============================================================
# Path reconstruction helpers
# ============================================================


def reconstruct_path(came_from: dict[int, int], current: int) -> list[int]:
    """Reconstruct a path from a parent-pointer dictionary.

    came_from[child] = parent.  Starting from the goal node, this function
    walks backward through parents until the start node, then reverses the
    list so the output goes from start to end.
    """
    current = int(current)
    path = [current]
    while current in came_from:
        parent = int(came_from[current])
        if parent == current:
            break
        current = parent
        path.append(current)
    path.reverse()
    return [int(v) for v in path]


def expand_theta_path_to_sampled_nodes(model, cache: GridCache, path: list[int]) -> list[int]:
    """Optional: expand any-angle waypoints into sampled grid nodes for plotting/debug."""
    if not path:
        return []
    expanded: list[int] = []
    for a, b in zip(path[:-1], path[1:]):
        seg = _los_segment_indices(model, cache, int(a), int(b))
        if not seg:
            seg = [int(a), int(b)]
        if expanded and seg[0] == expanded[-1]:
            expanded.extend(int(v) for v in seg[1:])
        else:
            expanded.extend(int(v) for v in seg)
    if len(path) == 1:
        expanded = [int(path[0])]
    return expanded
