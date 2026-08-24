#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_v15.py

LAE-UTM node-based Theta* master route planner, v15.

Changes in v15 (from v14):
  - FIGURE HONESTY FIX: rejected routes carry their best-failed path for
    diagnostics, and the plot loop drew any non-empty path as if
    accepted -- a failed backward main rendered as a lone solid line
    with no backup beside it, making figures contradict the v9
    main+backup atomicity (which was actually holding). Figures and the
    overview now draw success=True routes only; PLOT_INCLUDE_FAILED=True
    shows failures as thin gray dotted "failed (not accepted)" lines.
  - BALANCED DIRECTION SCHEDULING (PAIR_DIRECTION_ORDER, default
    "interleaved"): fwd couple 1 -> bwd couple 1 -> fwd couple 2 -> bwd
    couple 2 ... The old sequential order (kept as "sequential") let
    forward claim every good corridor -- including multiple FLZ
    approaches -- before backward started, structurally starving
    backward on constrained maps. Cross-direction locking stays
    mains-only (matching the original semantics; locking the other
    direction's backups too was tested and over-constrains), while each
    direction's own locks accumulate mains+backups as before.

Changes in v14 (from v13):
  - Backup separation requirement (BACKUP_SEPARATION_ENABLE, on by
    default): a successful backup is REJECTED unless genuinely separated
    from its main. The block mask only keeps backup NODES off the main;
    smoothed SEGMENTS could still cut across between allowed nodes --
    node-level blocking is not polyline-level separation. Now the backup
    polyline is resampled at 25 m and measured point-to-segment against
    the main (driveways excluded); at most
    BACKUP_SEPARATION_MAX_VIOLATION_PCT_LONG (10%) of a long route's
    length -- or _SHORT (5%) below BACKUP_SEPARATION_LONG_ROUTE_M -- may
    run closer than BACKUP_MIN_SEPARATION_M. Rejection behaves like the
    other gates (wider corridors, then v9 retry/skip). New summary
    columns: backup_min_offset_m, backup_separation_violation_pct,
    backup_separation_ok. Verified: catches real segment-cutting cases
    (nodes >= 30 m, segments dipping to 16.8 m) and correctly brands a
    duplicated backup as 100% violation / 0 m offset.

Changes in v13 (from v12):
  - All planner geometry is now METRIC regardless of input units. The
    search stack (model_io/thetastar/output_io) was already lon/lat-aware,
    but every mask and geometry check in THIS file (corridors, block-main,
    endpoint/FLZ buffers, lock radius, bbox, offsets, ring candidates)
    compared RAW x/y against meter thresholds -- correct on projected
    maps, silently catastrophic on lon/lat maps (corridor <= 50 'degrees'
    passes everything; block <= 30 'degrees' blocks nearly everything).
    read_node_model() now derives xm/ym via model_io's own
    make_metric_coordinates (single source of truth, same equirectangular
    convention), and all geometry uses path_to_xy_metric/all_metric_xy.
    Raw x/y remain the plotting and export coordinates. Verified: lon/lat
    twin of a metric map yields identical meter geometry (max per-node
    discrepancy 2.7 cm, boundary ties only); metric-map routes are
    byte-identical before/after (regression).

Changes in v12 (from v11):
  - BUG FIX (introduced v10): in make_backup_route, the FLZ buffer
    exemption was OR'd into endpoint_mask, which serves two jobs -- lock
    -overlap accounting exemption (intended) AND carving holes in the
    block-main mask (not intended). Consequence: inside FLZ buffers a
    backup could legally ride exactly ON its own main, rendering as
    invisible dashes over the solid main line ("route with no backup
    beside it" in figures). Fixed by splitting into overlap_exempt_mask
    (endpoints + FLZ buffers, overlap accounting only) and the block
    carve (endpoints only) -- the side-by-side guarantee now holds at
    the FLZ too. Verified: via-ring backups now select off-main ring
    nodes instead of the FLZ node itself.

Changes in v11 (from v10):
  - Live per-pair progress (LIVE_PAIR_PROGRESS): a percentage line inside
    each pair, ticking as each rank's main+backup couple resolves, e.g.
    "DB01_to_DK01:  50%|###...| 4/8  forward rank 2: ok". The percentage
    is over route slots (2 per rank per direction) -- the only denominator
    known in advance, since unlimited retries make total search count
    unknowable upfront. Retry attempts emit heartbeat lines ("attempt 3
    (searching a new main+backup couple)") so long grinds are visibly
    alive. Lines bypass the worker stdout capture (live_line ->
    sys.__stdout__) so they appear in real time in parallel mode too,
    interleaving only by whole line.

Changes in v10 (from v9):
  - FLZ buffer requirement: FLZ_BUFFER_ENABLE / FLZ_BUFFER_RADIUS_M.
    When on, every route must pass within the buffer radius of at least
    one FLZ node (checked segment-wise against the route polyline, since
    a smoothed leg can pass a FLZ between waypoints). Mains that miss the
    buffer are re-planned start -> best FLZ -> end (least straight-line
    detour first, still subject to lock-overlap limits); if no via-route
    works, the main is locked and a new couple is searched, consistent
    with the v9 lock-as-memory semantics. Backups that miss the buffer
    are forced through it via a ring node inside the buffer under the
    same corridor/block masks (backup_via_flz_ring) -- a cost-minimizing
    backup never takes the main's FLZ dip voluntarily, it cuts the
    corner, so forcing is required, not optional. FLZ buffer zones are
    exempt from lock-overlap accounting (flz_buffer_exempt_mask), same
    principle as the endpoint driveway exemption: overlap that the hard
    requirement itself forces cannot be counted against a route.
  - Progress report restyled to stage bars in the requested format:
    "Theta* pair routes:  45%|##########..." plus "Route tables export"
    and "Overview figures" stages. (batch_progress_bar now delegates to
    the new stage_progress_bar.)

Changes in v9 (from v8):
  - Retry semantics corrected per design intent: locking a main whose
    backup failed is memory to avoid re-finding the same pathway, not
    rejection of the rank. The rank now keeps searching for new
    main+backup couples until the main search itself fails -- i.e. no
    new distinct main exists under the lock/overlap limits (termination
    guaranteed: every rejected main locks its nodes, so the candidate
    space strictly shrinks). Only then is the rank skipped.
    MAX_MAIN_BACKUP_RETRY is now an optional time-budget cap only,
    default 0 = unlimited.
  - Parallel execution across pairs: N_CORES in thetastar.params
    (0 = auto: cpu_count - 1). Pairs are independent by design (no
    cross-pair locking in this stage), so each worker plans one pair's
    full forward+backward set; per-pair work extracted into top-level
    process_single_pair() with captured logs so parallel output never
    interleaves mid-line. Figures render on the Agg backend in workers.
    Result order is restored to input order regardless of completion
    order, so summary CSVs are stable run to run.

Changes in v8 (from v7):
  - generate_direction_route_sets() now treats main+backup as one unit:
    if a main's backup search fails (duplicated_from_main / no success),
    that main is rejected and locked (added to the avoid-list) instead of
    accepted with a degenerate backup, and a new main is searched for.
    Up to MAX_MAIN_BACKUP_RETRY candidates per rank; if none get a real
    backup, that rank is skipped (fewer route-sets for that direction,
    never a duplicate). Fixed a related gap in make_main_route(): the
    rank<=1 fast path didn't check previous_main_paths, so a rank-1 retry
    could have silently rediscovered the very main it just rejected.
  - Confirmed by testing: each retry is expensive (a full new main search
    plus a fully-exhausted backup search), since it stacks on the
    pre-existing masks-to-try/safe-fallback/corridor-list retry layers in
    make_backup_route() and run_project_theta(). Default
    MAX_MAIN_BACKUP_RETRY lowered to 2 accordingly -- see thetastar.params
    v8 comment for the numbers behind this.

Changes in v7 (from v6):
  - Added TEST_SINGLE_PAIR: set to "A_LABEL,B_LABEL" in thetastar.params
    to run just one pair instead of the full batch, for fast iteration.
    Empty (default) runs everything as before.
  - Added backup_vs_main_geometry(): every accepted backup route now
    reports distance_diff_m (backup length minus main length) and
    backup_mean_offset_m / backup_max_offset_m (how far the backup sits
    from the nearest point on main, averaged and at worst). Confirmed the
    existing BACKUP_CORRIDOR_M_LIST mechanism -- already present before
    this session -- is doing real work: on a synthetic test pair it held
    the backup within 50m of main the entire way for a 29m (0.4%) length
    penalty. See conversation notes; no algorithm change needed there,
    only visibility into what it's already doing.

Changes in v6 (from v5):
  - Added run_theta_k_paths(): penalty/plateau k-shortest-paths table.
    Finds the shortest A-B path, penalizes the nodes it used (same
    capped-multiplicative-slowness mechanism as
    apply_pair_terminal_traffic_avoidance, just retargeted), searches
    again, repeats up to TABLE_MAX_K times. Returns a ranked table
    (cheapest first, re-priced against the TRUE unpenalized cost so rows
    are actually comparable) instead of one route -- caller picks how
    many rows to keep. See TABLE_MAX_K / TABLE_PENALTY_WEIGHT /
    TABLE_MAX_PENALTY_FACTOR / TABLE_DUPLICATE_OVERLAP_PERCENT in
    thetastar.params v6.

Changes in v5 (from v4):
  - THETASTAR_LINE_OF_SIGHT_METHOD default switched to point_clearance
    (see thetastar.params v5 / thetastar.py v2 for the benchmark behind
    this: ~4.85x faster and far fewer redundant waypoints than bresenham
    on this nodal model).
  - Added batch_progress_bar(): the per-pair loop now prints a real
    percentage bar ([####----] 36.7% | pair 12/45  ...) instead of a
    plain [12/45] counter.
  - route_name is now threaded through to thetastar.run() as
    THETASTAR_PROGRESS_LABEL, so each search's own live progress line
    (when THETASTAR_SHOW_PROGRESS=True) is tagged with which pair/route
    it belongs to.

Purpose
-------
This script is the master planning wrapper. It does NOT implement Theta*.
It reads a node-based 2D riskmap, prepares route constraints, builds the graph,
then calls the existing Theta* backend:

    src/thetastar.py  -> run(model, graph, start_idx, end_idx, **kwargs)

Route concept in v1
-------------------
- DB and DK are normal route endpoints.
- FLZ is not a normal endpoint by default. It is used as emergency-safety
  attraction, so routes prefer to stay closer to FLZ when possible.
- RA and obstacle nodes remain hard no-fly.
- Each DB/DK pair gets configurable route sets per direction:
    forward_main_01, forward_backup_01, ...
    backward_main_01, backward_backup_01, ...

Important safety fix in v1
--------------------------
Theta* may export any-angle segments. This script validates every segment using
Bresenham grid traversal. If a segment crosses obstacle/RA/no-fly cells, the
route is rejected and retried in safe A*-delegate mode.

Run
---
    python main_v15.py --param-file params/thetastar.params

Expected input model columns
----------------------------
Required:
    x y slowness label label_prefix
Optional:
    node_id z risk_obstacle risk_ra risk_total obstacle_flag ra_flag objective_flag

Outputs
-------
    OUTPUT_DIR/
        objective_table.csv
        planning_model_with_flz_support.xyz
        route_summary.csv
        all_route_edges.csv
        route_nodes/*.csv
        figures/*.png
        00_all_theta_routes_overview_v15.png
"""

from __future__ import annotations

import argparse
import ast
import inspect
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.maprule import add_map_rule
from shapely.geometry import LineString


# ======================================================================
# Import project backend
# ======================================================================

try:
    from src.thetastar import run as theta_run
except Exception:
    try:
        from thetastar import run as theta_run
    except Exception as exc:
        raise ImportError(
            "Cannot import Theta* backend. Put thetastar.py in src/thetastar.py "
            "or in the same directory as this main_v15.py."
        ) from exc

try:
    from src.model_io import build_grid_graph, edge_cost as _mio_edge_cost, detect_lonlat_from_model, make_metric_coordinates
except Exception as exc:
    raise ImportError(
        "Cannot import src.model_io.build_grid_graph. This planner expects the "
        "existing LAE-UTM project structure with src/model_io.py."
    ) from exc


VERSION = "v15"


# ======================================================================
# Parameter loading
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
        return raw.strip('"').strip("'")


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

            key, value = line.split("=", 1)
            params[key.strip()] = parse_value(value.strip())

    return params


def pget(params: dict[str, Any], key: str, default: Any) -> Any:
    return params.get(key, default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LAE-UTM Theta* master route planner v2")
    parser.add_argument(
        "--param-file",
        default="params/thetastar.params",
        help="Path to params/thetastar.params",
    )
    return parser.parse_args()


# ======================================================================
# Model utilities
# ======================================================================

def read_node_model(model_file: str | Path) -> pd.DataFrame:
    model_file = Path(model_file)

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    df = pd.read_csv(model_file, sep=r"\s+", engine="python")
    df = df.reset_index(drop=True)

    if "node_id" not in df.columns:
        df.insert(0, "node_id", np.arange(len(df), dtype=int))

    if "z" not in df.columns:
        df["z"] = 0.0

    if "label" not in df.columns:
        df["label"] = "NONE"

    if "label_prefix" not in df.columns:
        df["label_prefix"] = "NONE"

    if "objective_flag" not in df.columns:
        df["objective_flag"] = (df["label_prefix"].astype(str) != "NONE").astype(int)

    required = ["x", "y", "z", "slowness", "label", "label_prefix"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Input model is missing required column: {col}")

    df["label"] = df["label"].fillna("NONE").astype(str)
    df["label_prefix"] = df["label_prefix"].fillna("NONE").astype(str)

    # Metric coordinate columns (added v13). ALL planner geometry -- masks,
    # corridors, buffers, offsets -- must measure in meters. On a projected
    # map xm/ym == x/y; on a lon/lat map they are the local equirectangular
    # meters from model_io.make_metric_coordinates (the same convention the
    # search stack already uses), so meter-thresholds stay meaningful.
    # Raw x/y remain the plotting/export coordinates.
    metric = make_metric_coordinates(df, is_lonlat=detect_lonlat_from_model(df))
    df["xm"] = metric[:, 0]
    df["ym"] = metric[:, 1]

    return df


def infer_grid_spacing(values: np.ndarray) -> float:
    vals = np.unique(np.round(values.astype(float), 6))
    if len(vals) < 2:
        return 1.0

    diffs = np.diff(vals)
    diffs = diffs[diffs > 1.0e-9]

    if len(diffs) == 0:
        return 1.0

    return float(np.median(diffs))


def add_grid_index(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[tuple[int, int], int], float]:
    dx = infer_grid_spacing(df["x"].to_numpy(float))
    dy = infer_grid_spacing(df["y"].to_numpy(float))
    grid_m = float(np.median([dx, dy]))

    xmin = float(df["x"].min())
    ymin = float(df["y"].min())

    work = df.copy()
    work["_ix"] = np.rint((work["x"].to_numpy(float) - xmin) / grid_m).astype(int)
    work["_iy"] = np.rint((work["y"].to_numpy(float) - ymin) / grid_m).astype(int)

    cell_to_idx: dict[tuple[int, int], int] = {}
    for idx, row in work.iterrows():
        cell_to_idx[(int(row["_ix"]), int(row["_iy"]))] = int(idx)

    return work, cell_to_idx, grid_m


def build_graph_for_model(model: pd.DataFrame, params: dict[str, Any]):
    """Build graph using the existing project build_grid_graph()."""
    graph_kwargs: dict[str, Any] = {}

    possible_keys = [
        "CONNECTIVITY",
        "NEIGHBOR_MODE",
        "NEIGHBOR_RADIUS_M",
        "NOFLY_SLOWNESS_THRESHOLD",
        "SLOWNESS_NOFLY_THRESHOLD",
    ]

    for key in possible_keys:
        if key in params:
            graph_kwargs[key] = params[key]
            graph_kwargs[key.lower()] = params[key]

    try:
        sig = inspect.signature(build_grid_graph)
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )

        if accepts_kwargs:
            return build_grid_graph(model, **graph_kwargs)

        allowed = set(sig.parameters.keys())
        filtered = {k: v for k, v in graph_kwargs.items() if k in allowed}

        if "model" in allowed:
            return build_grid_graph(model=model, **filtered)

        return build_grid_graph(model, **filtered)

    except TypeError:
        return build_grid_graph(model)


def apply_allowed_mask_to_model(
    model: pd.DataFrame,
    allowed_mask: np.ndarray,
    nofly_value: float,
    start_idx: int | None = None,
    end_idx: int | None = None,
) -> pd.DataFrame:
    work = model.copy()
    allowed = allowed_mask.copy()

    if start_idx is not None:
        allowed[int(start_idx)] = True
    if end_idx is not None:
        allowed[int(end_idx)] = True

    work.loc[~allowed, "slowness"] = float(nofly_value)

    return work


# ======================================================================
# FLZ emergency support
# ======================================================================

def compute_flz_support(df: pd.DataFrame, sigma_m: float) -> np.ndarray:
    flz = df[df["label_prefix"] == "FLZ"]

    if len(flz) == 0:
        return np.zeros(len(df), dtype=float)

    x = df["x"].to_numpy(float)
    y = df["y"].to_numpy(float)

    support = np.zeros(len(df), dtype=float)
    sigma_m = max(float(sigma_m), 1.0)

    for _, row in flz.iterrows():
        dx = x - float(row["x"])
        dy = y - float(row["y"])
        d2 = dx * dx + dy * dy
        s = np.exp(-d2 / (2.0 * sigma_m * sigma_m))
        support = np.maximum(support, s)

    return support


def apply_flz_safety_attraction(
    df: pd.DataFrame,
    params: dict[str, Any],
    nofly_threshold: float,
) -> pd.DataFrame:
    """
    Reduce flyable-node slowness near FLZ.

    This makes routes prefer emergency-support corridors without making no-fly
    cells flyable.
    """
    use_flz = bool(pget(params, "USE_FLZ_SAFETY_ATTRACTION", True))
    if not use_flz:
        return df

    work = df.copy()

    sigma_m = float(pget(params, "FLZ_SUPPORT_SIGMA_M", 700.0))
    weight = float(pget(params, "FLZ_SAFETY_WEIGHT", 0.30))
    min_factor = float(pget(params, "MIN_FLZ_SLOWNESS_FACTOR", 0.70))

    support = compute_flz_support(work, sigma_m=sigma_m)
    emergency_risk = 1.0 - support

    factor = 1.0 - weight * support
    factor = np.maximum(factor, min_factor)

    original_slowness = work["slowness"].to_numpy(float)
    modified_slowness = original_slowness.copy()

    flyable = original_slowness < nofly_threshold
    modified_slowness[flyable] = original_slowness[flyable] * factor[flyable]
    modified_slowness[~flyable] = original_slowness[~flyable]

    work["flz_support"] = support
    work["emergency_risk"] = emergency_risk
    work["slowness_raw"] = original_slowness
    work["slowness"] = modified_slowness

    return work


# ======================================================================
# Pair-aware swarm traffic avoidance
# ======================================================================

def compute_other_terminal_avoidance(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    prefixes: list[str],
    sigma_m: float,
) -> np.ndarray:
    """
    Return a 0..1 traffic-attraction/avoidance field around DB/DK terminals
    that are NOT part of the current route pair.

    1.0 = very close to another DB/DK terminal
    0.0 = far from other DB/DK terminals
    """
    if "label_prefix" not in df.columns:
        return np.zeros(len(df), dtype=float)

    prefixes = [str(v) for v in prefixes]
    terminal_mask = df["label_prefix"].astype(str).isin(prefixes)
    terminal_mask &= df["label"].astype(str) != "NONE"

    # Exclude current pair terminals. For DB01->DK02, DB01 and DK02 are allowed;
    # all other DB/DK nodes become traffic-density areas to avoid.
    terminal_mask.iloc[int(start_idx)] = False
    terminal_mask.iloc[int(end_idx)] = False

    terminals = df.loc[terminal_mask]
    if len(terminals) == 0:
        return np.zeros(len(df), dtype=float)

    x = df["x"].to_numpy(float)
    y = df["y"].to_numpy(float)
    sigma_m = max(float(sigma_m), 1.0)

    avoidance = np.zeros(len(df), dtype=float)
    for _, row in terminals.iterrows():
        dx = x - float(row["x"])
        dy = y - float(row["y"])
        d2 = dx * dx + dy * dy
        score = np.exp(-d2 / (2.0 * sigma_m * sigma_m))
        avoidance = np.maximum(avoidance, score)

    return avoidance


def apply_pair_terminal_traffic_avoidance(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    params: dict[str, Any],
    nofly_threshold: float,
) -> pd.DataFrame:
    """
    Increase flyable-node slowness near DB/DK terminals that are not part of
    the current pair. This discourages paths from passing near unrelated
    bases/docking stations, reducing future swarm-traffic density.
    """
    use_avoid = bool(pget(params, "USE_OTHER_DB_DK_TRAFFIC_AVOIDANCE", True))
    if not use_avoid:
        return df

    work = df.copy()
    prefixes = list(pget(params, "OTHER_TERMINAL_AVOID_PREFIXES", ["DB", "DK"]))
    sigma_m = float(pget(params, "OTHER_TERMINAL_AVOID_SIGMA_M", 450.0))
    weight = float(pget(params, "OTHER_TERMINAL_AVOID_WEIGHT", 0.80))
    max_factor = float(pget(params, "MAX_OTHER_TERMINAL_SLOWNESS_FACTOR", 2.50))

    avoidance = compute_other_terminal_avoidance(
        work,
        start_idx=int(start_idx),
        end_idx=int(end_idx),
        prefixes=prefixes,
        sigma_m=sigma_m,
    )

    original_slowness = work["slowness"].to_numpy(float)
    modified_slowness = original_slowness.copy()
    flyable = original_slowness < nofly_threshold

    traffic_factor = 1.0 + weight * avoidance
    traffic_factor = np.minimum(traffic_factor, max_factor)

    modified_slowness[flyable] = original_slowness[flyable] * traffic_factor[flyable]
    modified_slowness[~flyable] = original_slowness[~flyable]

    work["other_terminal_avoidance"] = avoidance
    work["traffic_penalty_factor"] = traffic_factor
    work["slowness"] = modified_slowness

    return work


def path_mean_column(df: pd.DataFrame, path_indices: list[int], column: str) -> float:
    if not path_indices or column not in df.columns:
        return np.nan
    try:
        vals = df.loc[[int(v) for v in path_indices], column].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        return float(np.mean(vals)) if len(vals) else np.nan
    except Exception:
        return np.nan


# ======================================================================
# Objective and pair utilities
# ======================================================================

def clean_route_prefixes(params: dict[str, Any]) -> tuple[list[str], list[str]]:
    """
    v1 default:
      - DB/DK are route endpoints.
      - FLZ is emergency support, not route endpoint, unless FLZ_AS_ROUTE_ENDPOINT=True.
    """
    raw_prefixes = list(pget(params, "ROUTE_OBJECTIVE_PREFIXES", ["DB", "DK"]))
    exclude = list(pget(params, "EXCLUDE_ROUTE_OBJECTIVE_PREFIXES", ["RA"]))

    flz_as_endpoint = bool(pget(params, "FLZ_AS_ROUTE_ENDPOINT", False))

    prefixes: list[str] = []
    for p in raw_prefixes:
        p = str(p)
        if p == "FLZ" and not flz_as_endpoint:
            if "FLZ" not in exclude:
                exclude.append("FLZ")
            continue
        if p not in prefixes:
            prefixes.append(p)

    # If user supplied only FLZ accidentally, recover the intended master-route endpoints.
    if not prefixes:
        prefixes = ["DB", "DK"]

    return prefixes, exclude


def objective_table(
    df: pd.DataFrame,
    route_prefixes: list[str],
    exclude_prefixes: list[str],
) -> pd.DataFrame:
    mask = df["label_prefix"].isin(route_prefixes)
    mask &= ~df["label_prefix"].isin(exclude_prefixes)
    mask &= df["label"].astype(str) != "NONE"

    obj = df.loc[mask, ["node_id", "x", "y", "z", "label", "label_prefix"]].copy()
    obj = obj.reset_index().rename(columns={"index": "idx"})
    obj = obj.sort_values(["label_prefix", "label"]).reset_index(drop=True)

    return obj


def make_pairs(
    obj: pd.DataFrame,
    pair_mode: str,
    skip_same_prefix: bool,
    max_pair_distance_m: float,
) -> list[dict[str, Any]]:
    pair_mode = pair_mode.lower().strip()
    pairs: list[dict[str, Any]] = []

    for i in range(len(obj)):
        for j in range(len(obj)):
            if i == j:
                continue
            if pair_mode == "unordered" and j <= i:
                continue

            a = obj.iloc[i]
            b = obj.iloc[j]

            if skip_same_prefix and a["label_prefix"] == b["label_prefix"]:
                continue

            dist = math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))

            if max_pair_distance_m > 0 and dist > max_pair_distance_m:
                continue

            pairs.append({
                "a_idx": int(a["idx"]),
                "b_idx": int(b["idx"]),
                "a_label": str(a["label"]),
                "b_label": str(b["label"]),
                "a_prefix": str(a["label_prefix"]),
                "b_prefix": str(b["label_prefix"]),
                "straight_distance_m": float(dist),
            })

    return pairs


def safe_name(text: str) -> str:
    text = str(text)
    for ch in [" ", "/", "\\", ":", ";", ",", "(", ")", "[", "]"]:
        text = text.replace(ch, "_")
    return text


def batch_progress_bar(current: int, total: int, label: str = "", width: int = 30) -> str:
    """Stage-style progress line: Theta* pair routes:  40%|████████░░░░| 12/30  DB03 ↔ DK07"""
    return stage_progress_bar("Theta* pair routes", current, total, extra=label, width=width)


def live_line(text: str) -> None:
    """Print a whole line to the REAL stdout, bypassing any redirect.

    Workers capture their stdout so parallel logs never interleave
    mid-line; live progress must escape that capture or it would only
    appear after the pair finishes -- defeating its purpose. Whole-line
    prints from multiple processes stay intact (they interleave only by
    line), so this is safe in parallel mode too.
    """
    print(text, file=sys.__stdout__, flush=True)


def stage_progress_bar(stage: str, current: int, total: int, extra: str = "", width: int = 30) -> str:
    """Format one stage line like: ACO base pair routes: 45%|████████████░░░░| 9/20"""
    total = max(1, int(total))
    current = max(0, min(int(current), total))
    pct = 100.0 * current / total
    filled = int(round(width * current / total))
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    tail = f"  {extra}" if extra else ""
    return f"{stage}: {pct:3.0f}%|{bar}| {current}/{total}{tail}"


# ======================================================================
# Geometry, masks, and validation
# ======================================================================

def path_distance_m(df: pd.DataFrame, path_indices: list[int]) -> float:
    if not path_indices or len(path_indices) < 2:
        return 0.0

    xy = df.loc[path_indices, _metric_cols(df)].to_numpy(float)
    d = np.sqrt(np.diff(xy[:, 0]) ** 2 + np.diff(xy[:, 1]) ** 2)
    return float(np.sum(d))


def path_to_xy(df: pd.DataFrame, path_indices: list[int]) -> np.ndarray:
    """RAW coordinates -- correct for plotting and export only."""
    if not path_indices:
        return np.empty((0, 2), dtype=float)
    return df.loc[path_indices, ["x", "y"]].to_numpy(float)


def route_corridor_polygon_xy(xy: np.ndarray, radius_m: float) -> np.ndarray | None:
    """2D map footprint of the route's circular-prism corridor.

    src/thetastar.py models a route as a cylinder: the centerline swept by a
    disk of radius THETASTAR_ROUTE_WIDTH_M / 2. This is that same disk swept
    along the plotted polyline (a capsule on a straight route; capsules
    unioned with rounded joins at interior waypoints for multi-segment
    routes) -- i.e. the cylinder's projection onto the map.
    """
    if radius_m <= 0.0 or len(xy) < 2:
        return None
    line = LineString(xy)
    if line.length <= 0.0:
        return None
    poly = line.buffer(float(radius_m), cap_style="round", join_style="round")
    if poly.is_empty:
        return None
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    return np.asarray(poly.exterior.coords)


def draw_route_corridor(
    ax,
    corridor_xy: np.ndarray,
    fill_color: str,
    fill_alpha: float,
    edge_alpha: float,
    zorder: float,
    label: str = "_nolegend_",
) -> None:
    """Fill a route's corridor footprint and trace its boundary in gray.

    Two corridors sitting close together (main + backup, or adjacent ranks)
    otherwise blend into one translucent blob once their fills overlap --
    the gray boundary line is what actually lets you see where one
    corridor ends and the next begins.
    """
    ax.fill(corridor_xy[:, 0], corridor_xy[:, 1], color=fill_color, alpha=fill_alpha,
            linewidth=0, zorder=zorder, label=label)
    ax.plot(corridor_xy[:, 0], corridor_xy[:, 1], color="dimgray", linewidth=0.7,
            alpha=edge_alpha, zorder=zorder + 0.5, solid_capstyle="round")


def _metric_cols(df: pd.DataFrame) -> list[str]:
    return ["xm", "ym"] if ("xm" in df.columns and "ym" in df.columns) else ["x", "y"]


def path_to_xy_metric(df: pd.DataFrame, path_indices: list[int]) -> np.ndarray:
    """METRIC coordinates -- what every mask/geometry computation must use."""
    if not path_indices:
        return np.empty((0, 2), dtype=float)
    return df.loc[path_indices, _metric_cols(df)].to_numpy(float)


def all_metric_xy(df: pd.DataFrame) -> np.ndarray:
    return df[_metric_cols(df)].to_numpy(float)


def distance_to_path(df: pd.DataFrame, path_indices: list[int]) -> np.ndarray:
    pts = all_metric_xy(df)
    path_xy = path_to_xy_metric(df, path_indices)

    if len(path_xy) == 0:
        return np.full(len(df), np.inf)

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(path_xy)
        dist, _ = tree.query(pts, k=1)
        return dist.astype(float)
    except Exception:
        dist = np.full(len(df), np.inf)
        for px, py in path_xy:
            d = np.sqrt((pts[:, 0] - px) ** 2 + (pts[:, 1] - py) ** 2)
            dist = np.minimum(dist, d)
        return dist


def backup_vs_main_geometry(
    df: pd.DataFrame, main_path: list[int], backup_path: list[int]
) -> dict[str, float]:
    """How much longer is the backup, and how far does it sit from main.

    backup_mean_offset_m / backup_max_offset_m: for every backup waypoint,
    distance to the NEAREST point on main, then averaged / maxed. Low mean
    and low max = a tight parallel corridor the whole way. Low mean with a
    much higher max = mostly parallel with one real detour (e.g. around an
    obstacle main didn't have to avoid at that spot).
    """
    if not main_path or not backup_path:
        return {"distance_diff_m": float("nan"), "backup_mean_offset_m": float("nan"), "backup_max_offset_m": float("nan")}

    main_dist = path_distance_m(df, main_path)
    backup_dist = path_distance_m(df, backup_path)

    main_xy = path_to_xy_metric(df, main_path)
    backup_xy = path_to_xy_metric(df, backup_path)

    if len(main_xy) == 0 or len(backup_xy) == 0:
        return {"distance_diff_m": float("nan"), "backup_mean_offset_m": float("nan"), "backup_max_offset_m": float("nan")}

    try:
        from scipy.spatial import cKDTree
        offsets, _ = cKDTree(main_xy).query(backup_xy, k=1)
    except Exception:
        offsets = np.array([
            np.min(np.hypot(main_xy[:, 0] - bx, main_xy[:, 1] - by))
            for bx, by in backup_xy
        ])

    return {
        "distance_diff_m": float(backup_dist - main_dist),
        "backup_mean_offset_m": float(np.mean(offsets)),
        "backup_max_offset_m": float(np.max(offsets)),
    }


def backup_separation_stats(
    df: pd.DataFrame,
    main_path: list[int],
    backup_path: list[int],
    start_idx: int,
    end_idx: int,
    endpoint_buffer_m: float,
    threshold_m: float,
    sample_step_m: float = 25.0,
) -> dict[str, float]:
    """Fraction of the backup's length that runs closer than threshold_m
    to the main, excluding the endpoint driveways (where main and backup
    necessarily converge on the same hubs).

    Measured properly, not approximately:
    - the backup POLYLINE is resampled at sample_step_m, so the result
      does not depend on how sparse/dense the waypoints happen to be;
    - distance is point-to-SEGMENT against the main polyline -- point-to-
      waypoint would err by up to half the grid spacing, the same order
      as the threshold itself.
    """
    empty = {"backup_separation_violation_pct": float("nan"), "backup_min_offset_m": float("nan")}
    if not main_path or not backup_path or len(backup_path) < 2:
        return empty

    main_xy = path_to_xy_metric(df, main_path)
    backup_xy = path_to_xy_metric(df, backup_path)
    if len(main_xy) < 2:
        return empty

    # Resample the backup polyline at uniform arc-length steps.
    seg = np.diff(backup_xy, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    total = float(seg_len.sum())
    if total <= 0.0:
        return empty
    n_samples = max(2, int(np.ceil(total / max(1.0, float(sample_step_m)))) + 1)
    s_targets = np.linspace(0.0, total, n_samples)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    seg_of = np.clip(np.searchsorted(cum, s_targets, side="right") - 1, 0, len(seg_len) - 1)
    local = (s_targets - cum[seg_of]) / np.maximum(seg_len[seg_of], 1e-12)
    samples = backup_xy[seg_of] + seg[seg_of] * local[:, None]

    # Exclude driveway samples (near either endpoint).
    mxy_s = path_to_xy_metric(df, [int(start_idx)])[0]
    mxy_e = path_to_xy_metric(df, [int(end_idx)])[0]
    d_start = np.hypot(samples[:, 0] - mxy_s[0], samples[:, 1] - mxy_s[1])
    d_end = np.hypot(samples[:, 0] - mxy_e[0], samples[:, 1] - mxy_e[1])
    keep = (d_start > float(endpoint_buffer_m)) & (d_end > float(endpoint_buffer_m))
    if not np.any(keep):
        return {"backup_separation_violation_pct": 0.0, "backup_min_offset_m": float("nan")}
    pts = samples[keep]

    # Point-to-segment distance against the main polyline.
    best = np.full(len(pts), np.inf)
    for k in range(len(main_xy) - 1):
        a = main_xy[k]
        ab = main_xy[k + 1] - a
        denom = float(ab @ ab)
        if denom <= 0.0:
            proj = np.zeros(len(pts))
        else:
            proj = np.clip(((pts - a) @ ab) / denom, 0.0, 1.0)
        closest = a + proj[:, None] * ab
        d = np.hypot(pts[:, 0] - closest[:, 0], pts[:, 1] - closest[:, 1])
        best = np.minimum(best, d)

    violation_pct = 100.0 * float(np.count_nonzero(best < float(threshold_m))) / len(pts)
    return {
        "backup_separation_violation_pct": violation_pct,
        "backup_min_offset_m": float(np.min(best)),
    }


def backup_separation_verdict(
    df: pd.DataFrame,
    main_path: list[int],
    backup_path: list[int],
    start_idx: int,
    end_idx: int,
    params: dict[str, Any],
) -> tuple[bool, dict[str, float]]:
    """Apply the length-dependent separation rule.

    Long routes (length >= BACKUP_SEPARATION_LONG_ROUTE_M) may have up to
    BACKUP_SEPARATION_MAX_VIOLATION_PCT_LONG percent of their (non-
    driveway) length closer than BACKUP_MIN_SEPARATION_M to the main;
    short routes get the stricter _SHORT allowance.
    """
    if not bool(pget(params, "BACKUP_SEPARATION_ENABLE", True)):
        return True, {}
    threshold_m = float(pget(params, "BACKUP_MIN_SEPARATION_M",
                             float(pget(params, "BACKUP_BLOCK_MAIN_RADIUS_M", 30.0))))
    long_route_m = float(pget(params, "BACKUP_SEPARATION_LONG_ROUTE_M", 3000.0))
    pct_long = float(pget(params, "BACKUP_SEPARATION_MAX_VIOLATION_PCT_LONG", 10.0))
    pct_short = float(pget(params, "BACKUP_SEPARATION_MAX_VIOLATION_PCT_SHORT", 5.0))
    endpoint_buffer_m = float(pget(params, "ENDPOINT_BUFFER_M", 150.0))

    stats = backup_separation_stats(
        df, main_path, backup_path, start_idx, end_idx, endpoint_buffer_m, threshold_m,
    )
    backup_len = path_distance_m(df, backup_path)
    allowed = pct_long if backup_len >= long_route_m else pct_short
    v = stats.get("backup_separation_violation_pct", float("nan"))
    stats["backup_separation_allowed_pct"] = allowed
    stats["backup_separation_ok"] = bool(not np.isfinite(v) or v <= allowed)
    return bool(stats["backup_separation_ok"]), stats


def flz_node_indices(df: pd.DataFrame) -> list[int]:
    """All node indices whose label prefix is FLZ."""
    pref = df["label_prefix"].astype(str).str.upper()
    return [int(i) for i in np.flatnonzero((pref == "FLZ").to_numpy())]


def route_min_distance_to_flz(df: pd.DataFrame, path_indices: list[int]) -> float:
    """Minimum distance from the route polyline to any FLZ node.

    Segment-based, not waypoint-based: smoothed any-angle routes can have
    long straight legs, and a leg can pass right beside an FLZ between two
    waypoints. Checking only waypoints would miss that.
    """
    flz_idx = flz_node_indices(df)
    if not flz_idx or len(path_indices) == 0:
        return float("inf")

    flz_xy = df.iloc[flz_idx][_metric_cols(df)].to_numpy(float)
    path_xy = path_to_xy_metric(df, path_indices)
    if len(path_xy) == 1:
        d = np.hypot(flz_xy[:, 0] - path_xy[0, 0], flz_xy[:, 1] - path_xy[0, 1])
        return float(np.min(d))

    best = float("inf")
    for k in range(len(path_xy) - 1):
        a = path_xy[k]
        b = path_xy[k + 1]
        ab = b - a
        denom = float(ab @ ab)
        if denom <= 0.0:
            proj = np.zeros(len(flz_xy))
        else:
            proj = np.clip(((flz_xy - a) @ ab) / denom, 0.0, 1.0)
        closest = a + proj[:, None] * ab
        d = np.hypot(flz_xy[:, 0] - closest[:, 0], flz_xy[:, 1] - closest[:, 1])
        seg_min = float(np.min(d))
        if seg_min < best:
            best = seg_min
    return best


def flz_buffer_settings(params: dict[str, Any]) -> tuple[bool, float]:
    return (
        bool(pget(params, "FLZ_BUFFER_ENABLE", False)),
        float(pget(params, "FLZ_BUFFER_RADIUS_M", 100.0)),
    )


def flz_buffer_exempt_mask(df: pd.DataFrame, params: dict[str, Any]) -> np.ndarray | None:
    """Nodes inside any FLZ buffer, exempt from lock-overlap accounting.

    Same principle as the endpoint 'driveway' exemption: when the FLZ
    buffer requirement is ON, every compliant route is FORCED through an
    FLZ buffer, so overlap there is unavoidable by design and must not
    count against the route-overlap limit -- otherwise the hard FLZ
    requirement and the overlap limit contradict each other at the FLZ.
    Returns None when the buffer requirement is off.
    """
    flz_on, radius = flz_buffer_settings(params)
    if not flz_on:
        return None
    mask = np.zeros(len(df), dtype=bool)
    mxy = all_metric_xy(df)
    xs, ys = mxy[:, 0], mxy[:, 1]
    for i in flz_node_indices(df):
        fx, fy = float(mxy[i, 0]), float(mxy[i, 1])
        mask |= np.hypot(xs - fx, ys - fy) <= float(radius)
    return mask


def replan_main_via_flz(
    df: pd.DataFrame,
    cell_to_idx: dict[tuple[int, int], int],
    base_allowed: np.ndarray,
    start_idx: int,
    end_idx: int,
    params: dict[str, Any],
    route_name: str,
    locked_paths: list[list[int]],
) -> dict[str, Any]:
    """Force a route through an FLZ buffer by planning start->FLZ->end.

    FLZ candidates are tried in order of least straight-line detour. The
    merged route is guaranteed to touch the chosen FLZ's buffer (it passes
    through the FLZ node itself), but it must still pass the same
    lock-overlap test as any other main, so it cannot silently duplicate
    an already-locked route.
    """
    flz_idx = flz_node_indices(df)
    if not flz_idx:
        return {"success": False, "message": "FLZ buffer required but the model has no FLZ nodes.", "path_indices": []}

    mxy = all_metric_xy(df)
    sx, sy = float(mxy[start_idx, 0]), float(mxy[start_idx, 1])
    ex, ey = float(mxy[end_idx, 0]), float(mxy[end_idx, 1])
    fxy = mxy[flz_idx]
    detour = np.hypot(fxy[:, 0] - sx, fxy[:, 1] - sy) + np.hypot(fxy[:, 0] - ex, fxy[:, 1] - ey)
    order = [flz_idx[i] for i in np.argsort(detour)]

    lock_enabled, max_overlap_pct, lock_radius_m, strict_lock_first = lock_settings(params)
    endpoint_mask = endpoint_buffer_mask(df, start_idx, end_idx, float(pget(params, "ENDPOINT_BUFFER_M", 150.0)))
    _flz_exempt = flz_buffer_exempt_mask(df, params)
    if _flz_exempt is not None:
        endpoint_mask = endpoint_mask | _flz_exempt
    locked_mask = build_locked_node_mask(df, locked_paths, endpoint_mask, lock_radius_m)

    last_message = "No FLZ candidate produced a valid via-route."
    for flz_i in order:
        flz_label = str(df.iloc[flz_i]["label"])
        leg1 = run_theta_once(
            base_model=df, cell_to_idx=cell_to_idx, mask=base_allowed,
            start_idx=start_idx, end_idx=int(flz_i), params=params,
            route_name=f"{route_name}_via_{flz_label}_leg1", safe_fallback=True,
        )
        if not leg1.get("success", False):
            last_message = f"Leg to {flz_label} failed: {leg1.get('message', '')}"
            continue
        leg2 = run_theta_once(
            base_model=df, cell_to_idx=cell_to_idx, mask=base_allowed,
            start_idx=int(flz_i), end_idx=end_idx, params=params,
            route_name=f"{route_name}_via_{flz_label}_leg2", safe_fallback=True,
        )
        if not leg2.get("success", False):
            last_message = f"Leg from {flz_label} failed: {leg2.get('message', '')}"
            continue

        p1 = [int(v) for v in leg1.get("path_indices", [])]
        p2 = [int(v) for v in leg2.get("path_indices", [])]
        merged = p1 + p2[1:]

        result: dict[str, Any] = {
            "success": True,
            "message": f"Route forced through FLZ buffer via {flz_label}.",
            "route_name": route_name,
            "path_indices": merged,
            "distance_m": path_distance_m(df, merged),
            "total_cost": float(leg1.get("total_cost", np.nan)) + float(leg2.get("total_cost", np.nan)),
            "runtime_sec": float(leg1.get("runtime_sec", 0.0) or 0.0) + float(leg2.get("runtime_sec", 0.0) or 0.0),
            "expanded_nodes": int(leg1.get("expanded_nodes", 0) or 0) + int(leg2.get("expanded_nodes", 0) or 0),
            "safe_fallback": bool(leg1.get("safe_fallback", False) or leg2.get("safe_fallback", False)),
            "flz_buffer_via": flz_label,
            "mean_flz_support": path_mean_column(df, merged, "flz_support"),
            "mean_emergency_risk": path_mean_column(df, merged, "emergency_risk"),
            "mean_other_terminal_avoidance": path_mean_column(df, merged, "other_terminal_avoidance"),
            "mean_traffic_penalty_factor": path_mean_column(df, merged, "traffic_penalty_factor"),
        }

        if lock_enabled and np.any(locked_mask):
            ok_overlap, result = route_respects_lock_limit(result, locked_mask, endpoint_mask, max_overlap_pct)
            if not ok_overlap:
                last_message = f"Via-{flz_label} route exceeded the lock overlap limit."
                continue
        return result

    return {"success": False, "message": last_message, "path_indices": []}


def backup_via_flz_ring(
    df: pd.DataFrame,
    cell_to_idx: dict[tuple[int, int], int],
    base_allowed: np.ndarray,
    start_idx: int,
    end_idx: int,
    params: dict[str, Any],
    route_name: str,
    corridor_mask: np.ndarray,
    block_mask: np.ndarray | None,
    main_path: list[int],
    flz_radius: float,
) -> dict[str, Any]:
    """Force a backup through the FLZ buffer under the same corridor/block
    masks as a normal backup.

    Candidate 'ring' nodes: allowed, inside the FLZ buffer, inside the
    corridor, not blocked (i.e. at least the block radius off the main).
    Tried nearest-to-FLZ first; the backup is planned as two legs through
    the candidate and merged.
    """
    flz_idx = flz_node_indices(df)
    if not flz_idx:
        return {"success": False, "message": "No FLZ nodes.", "path_indices": []}

    mxy = all_metric_xy(df)
    xs, ys = mxy[:, 0], mxy[:, 1]
    in_buffer = np.zeros(len(df), dtype=bool)
    dist_to_flz = np.full(len(df), np.inf)
    for i in flz_idx:
        fx, fy = float(mxy[i, 0]), float(mxy[i, 1])
        d = np.hypot(xs - fx, ys - fy)
        in_buffer |= d <= float(flz_radius)
        dist_to_flz = np.minimum(dist_to_flz, d)

    cand = in_buffer & corridor_mask & base_allowed
    if block_mask is not None:
        cand &= ~block_mask
    cand_idx = np.flatnonzero(cand)
    if len(cand_idx) == 0:
        return {"success": False, "message": "No allowed node inside FLZ buffer, corridor, and off-main block.", "path_indices": []}
    cand_idx = cand_idx[np.argsort(dist_to_flz[cand_idx])][:6]

    last_message = "All FLZ ring candidates failed."
    for ci in cand_idx:
        ci = int(ci)
        leg1 = run_project_theta(
            base_model=df, cell_to_idx=cell_to_idx, base_allowed_mask=base_allowed,
            start_idx=start_idx, end_idx=ci, params=params,
            extra_allowed_mask=corridor_mask, block_mask=block_mask,
            route_name=f"{route_name}_ring_leg1",
        )
        if not leg1.get("success", False):
            last_message = f"Ring leg1 failed: {leg1.get('message', '')}"
            continue
        leg2 = run_project_theta(
            base_model=df, cell_to_idx=cell_to_idx, base_allowed_mask=base_allowed,
            start_idx=ci, end_idx=end_idx, params=params,
            extra_allowed_mask=corridor_mask, block_mask=block_mask,
            route_name=f"{route_name}_ring_leg2",
        )
        if not leg2.get("success", False):
            last_message = f"Ring leg2 failed: {leg2.get('message', '')}"
            continue

        p1 = [int(v) for v in leg1.get("path_indices", [])]
        p2 = [int(v) for v in leg2.get("path_indices", [])]
        merged = p1 + p2[1:]
        min_d = route_min_distance_to_flz(df, merged)
        return {
            "success": True,
            "message": f"Backup forced through FLZ buffer via ring node {str(df.iloc[ci]['label'])}.",
            "route_name": route_name,
            "path_indices": merged,
            "distance_m": path_distance_m(df, merged),
            "total_cost": float(leg1.get("total_cost", np.nan)) + float(leg2.get("total_cost", np.nan)),
            "runtime_sec": float(leg1.get("runtime_sec", 0.0) or 0.0) + float(leg2.get("runtime_sec", 0.0) or 0.0),
            "expanded_nodes": int(leg1.get("expanded_nodes", 0) or 0) + int(leg2.get("expanded_nodes", 0) or 0),
            "safe_fallback": bool(leg1.get("safe_fallback", False) or leg2.get("safe_fallback", False)),
            "flz_buffer_via": str(df.iloc[ci]["label"]),
            "flz_buffer_min_distance_m": float(min_d),
            "flz_buffer_ok": bool(min_d <= flz_radius),
        }

    return {"success": False, "message": last_message, "path_indices": []}


def bbox_mask(df: pd.DataFrame, start_idx: int, end_idx: int, buffer_m: float) -> np.ndarray:
    if buffer_m <= 0:
        return np.ones(len(df), dtype=bool)

    mxy = all_metric_xy(df)
    sx = float(mxy[start_idx, 0])
    sy = float(mxy[start_idx, 1])
    ex = float(mxy[end_idx, 0])
    ey = float(mxy[end_idx, 1])

    xmin = min(sx, ex) - buffer_m
    xmax = max(sx, ex) + buffer_m
    ymin = min(sy, ey) - buffer_m
    ymax = max(sy, ey) + buffer_m

    x = mxy[:, 0]
    y = mxy[:, 1]

    return (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)


def endpoint_buffer_mask(df: pd.DataFrame, start_idx: int, end_idx: int, radius_m: float) -> np.ndarray:
    mxy = all_metric_xy(df)
    x = mxy[:, 0]
    y = mxy[:, 1]

    sx = float(mxy[start_idx, 0])
    sy = float(mxy[start_idx, 1])
    ex = float(mxy[end_idx, 0])
    ey = float(mxy[end_idx, 1])

    ds = np.sqrt((x - sx) ** 2 + (y - sy) ** 2)
    de = np.sqrt((x - ex) ** 2 + (y - ey) ** 2)

    return (ds <= radius_m) | (de <= radius_m)


def bresenham_cells(ix0: int, iy0: int, ix1: int, iy1: int) -> list[tuple[int, int]]:
    ix0 = int(ix0); iy0 = int(iy0); ix1 = int(ix1); iy1 = int(iy1)

    dx = abs(ix1 - ix0)
    dy = abs(iy1 - iy0)

    sx = 1 if ix0 < ix1 else (-1 if ix0 > ix1 else 0)
    sy = 1 if iy0 < iy1 else (-1 if iy0 > iy1 else 0)

    x = ix0
    y = iy0
    cells = [(x, y)]

    if dx >= dy:
        err = dx / 2.0
        while x != ix1:
            x += sx
            err -= dy
            if err < 0:
                y += sy
                err += dx
            cells.append((x, y))
    else:
        err = dy / 2.0
        while y != iy1:
            y += sy
            err -= dx
            if err < 0:
                x += sx
                err += dy
            cells.append((x, y))

    return cells


def expand_and_validate_path_by_grid_los(
    df: pd.DataFrame,
    cell_to_idx: dict[tuple[int, int], int],
    path_indices: list[int],
    allowed_mask: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    """
    Expand every segment by Bresenham cells and reject any obstacle crossing.
    """
    if not path_indices:
        return [], []

    if len(path_indices) == 1:
        return [int(path_indices[0])], []

    expanded: list[int] = []
    bad_segments: list[dict[str, Any]] = []

    for a, b in zip(path_indices[:-1], path_indices[1:]):
        a = int(a)
        b = int(b)

        ix0 = int(df.at[a, "_ix"])
        iy0 = int(df.at[a, "_iy"])
        ix1 = int(df.at[b, "_ix"])
        iy1 = int(df.at[b, "_iy"])

        cells = bresenham_cells(ix0, iy0, ix1, iy1)

        segment_indices: list[int] = []
        blocked_hits: list[tuple[int, int]] = []

        for cell in cells:
            idx = cell_to_idx.get(cell, None)

            if idx is None:
                blocked_hits.append(cell)
                continue

            idx = int(idx)

            # Allow exact terminal cells because objective cells may be forced flyable.
            if idx not in (int(start_idx), int(end_idx)) and not bool(allowed_mask[idx]):
                blocked_hits.append(cell)
                continue

            segment_indices.append(idx)

        if blocked_hits:
            bad_segments.append({
                "from_idx": a,
                "to_idx": b,
                "from_node_id": int(df.at[a, "node_id"]) if "node_id" in df.columns else a,
                "to_node_id": int(df.at[b, "node_id"]) if "node_id" in df.columns else b,
                "n_blocked_cells": len(blocked_hits),
                "blocked_cells_preview": blocked_hits[:10],
            })

        if not segment_indices:
            continue

        if expanded and segment_indices[0] == expanded[-1]:
            expanded.extend(segment_indices[1:])
        else:
            expanded.extend(segment_indices)

    cleaned: list[int] = []
    for idx in expanded:
        idx = int(idx)
        if not cleaned or cleaned[-1] != idx:
            cleaned.append(idx)

    return cleaned, bad_segments


def closeness_to_reference_m(df: pd.DataFrame, path: list[int], ref_path: list[int]) -> float:
    if not path or not ref_path:
        return np.nan

    path_xy = path_to_xy(df, path)
    ref_xy = path_to_xy(df, ref_path)

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(ref_xy)
        dist, _ = tree.query(path_xy, k=1)
        return float(np.mean(dist))
    except Exception:
        out = []
        for x, y in path_xy:
            d = np.sqrt((ref_xy[:, 0] - x) ** 2 + (ref_xy[:, 1] - y) ** 2)
            out.append(float(np.min(d)))
        return float(np.mean(out))


# ======================================================================
# Theta* wrapper
# ======================================================================

def theta_kwargs_from_params(params: dict[str, Any], safe_fallback: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {}

    for key, value in params.items():
        if key.startswith("THETASTAR_"):
            out[key] = value

    if "THETASTAR_HEURISTIC_WEIGHT" in params:
        out["heuristic_weight"] = params["THETASTAR_HEURISTIC_WEIGHT"]

    if "THETASTAR_MAX_EXPANSIONS" in params:
        out["max_expansions"] = params["THETASTAR_MAX_EXPANSIONS"]

    if safe_fallback:
        # Pure A* delegate through the Theta* module. This avoids any-angle shortcuts.
        out["THETASTAR_ASTAR_FIRST_LOS_SMOOTH"] = False
        out["THETASTAR_ALLOW_ANY_ANGLE"] = False
        out["THETASTAR_PURE_ASTAR_DELEGATE"] = True
        out["THETASTAR_OUTPUT_SAMPLED_PATH"] = True
        out["THETASTAR_LOS_STRAIGHT_PATH_FALLBACK"] = False

    return out


def result_runtime(result: dict[str, Any]) -> float:
    for key in ["runtime_seconds", "runtime_sec", "runtime"]:
        if key in result and result[key] is not None:
            try:
                return float(result[key])
            except Exception:
                pass
    return np.nan


def result_expanded(result: dict[str, Any]) -> int:
    for key in ["expanded_nodes", "expanded_states", "visited_nodes"]:
        if key in result and result[key] is not None:
            try:
                return int(result[key])
            except Exception:
                pass
    return -1


def run_theta_once(
    base_model: pd.DataFrame,
    cell_to_idx: dict[tuple[int, int], int],
    mask: np.ndarray,
    start_idx: int,
    end_idx: int,
    params: dict[str, Any],
    route_name: str,
    safe_fallback: bool,
) -> dict[str, Any]:
    nofly_value = float(pget(params, "NOFLY_SLOWNESS_VALUE", 10.0))

    work_model = apply_allowed_mask_to_model(
        model=base_model,
        allowed_mask=mask,
        nofly_value=nofly_value,
        start_idx=start_idx,
        end_idx=end_idx,
    )

    # Pair-aware traffic criterion:
    # keep the current pair terminals usable, but penalize corridors close to
    # all other DB/DK terminals so the final swarm network does not overload
    # unrelated bases/docking stations.
    work_model = apply_pair_terminal_traffic_avoidance(
        df=work_model,
        start_idx=start_idx,
        end_idx=end_idx,
        params=params,
        nofly_threshold=float(pget(params, "NOFLY_SLOWNESS_THRESHOLD", 10.0)),
    )

    graph = build_graph_for_model(work_model, params)

    theta_kwargs = theta_kwargs_from_params(params, safe_fallback=safe_fallback)
    theta_kwargs["THETASTAR_PROGRESS_LABEL"] = route_name

    result = theta_run(
        model=work_model,
        graph=graph,
        start_idx=int(start_idx),
        end_idx=int(end_idx),
        **theta_kwargs,
    )

    if not isinstance(result, dict):
        result = {
            "success": False,
            "message": "src.thetastar.run() did not return a dict.",
            "path_indices": [],
        }

    raw_path = [int(v) for v in result.get("path_indices", [])]

    invalid_nodes = [idx for idx in raw_path if idx < 0 or idx >= len(mask) or not bool(mask[idx])]
    if invalid_nodes:
        result["success"] = False
        result["message"] = (
            str(result.get("message", "")) +
            f" Path contains {len(invalid_nodes)} blocked path nodes."
        )
        result["path_indices"] = []
        raw_path = []
    else:
        expanded_path, bad_segments = expand_and_validate_path_by_grid_los(
            df=base_model,
            cell_to_idx=cell_to_idx,
            path_indices=raw_path,
            allowed_mask=mask,
            start_idx=start_idx,
            end_idx=end_idx,
        )

        if bad_segments:
            result["success"] = False
            result["message"] = (
                str(result.get("message", "")) +
                f" Theta* segment crossed blocked cells. "
                f"Bad segments={len(bad_segments)}; first={bad_segments[0]}"
            )
            result["path_indices"] = []
            raw_path = []
        else:
            raw_path = expanded_path
            result["path_indices"] = raw_path
            result["grid_los_validated"] = True
            result["grid_los_expanded_nodes"] = len(expanded_path)

    success = bool(result.get("success", False)) and len(raw_path) > 0
    result["success"] = success
    result["route_name"] = route_name
    result["path_indices"] = raw_path
    result["distance_m"] = path_distance_m(base_model, raw_path) if raw_path else np.nan
    result["mean_flz_support"] = path_mean_column(work_model, raw_path, "flz_support")
    result["mean_emergency_risk"] = path_mean_column(work_model, raw_path, "emergency_risk")
    result["mean_other_terminal_avoidance"] = path_mean_column(work_model, raw_path, "other_terminal_avoidance")
    result["mean_traffic_penalty_factor"] = path_mean_column(work_model, raw_path, "traffic_penalty_factor")
    result["runtime_sec"] = result_runtime(result)
    result["expanded_nodes"] = result_expanded(result)
    result["safe_fallback"] = bool(safe_fallback)

    return result


def run_theta_k_paths(
    base_model: pd.DataFrame,
    cell_to_idx: dict[tuple[int, int], int],
    mask: np.ndarray,
    start_idx: int,
    end_idx: int,
    params: dict[str, Any],
    route_name: str,
) -> list[dict[str, Any]]:
    """
    Penalty/plateau k-shortest-paths table (added v6).

    Runs Theta*, penalizes the nodes the winning path used, runs again,
    repeat up to TABLE_MAX_K times. Returns a ranked table, cheapest
    first -- the caller decides how many rows to actually keep.

    This does not guarantee the mathematically exact k-th shortest path the
    way Yen's algorithm does. What it guarantees is that every accepted
    path was found under a cost field actively discouraging reuse of every
    node accepted into an earlier row, so the table trends toward
    genuinely different corridors rather than near-duplicates -- and it
    gets there in max_k Theta* calls, not the larger candidate-and-filter
    search Yen's needs for the same diversity.

    Node penalties reuse the exact mechanism already in this file for
    apply_pair_terminal_traffic_avoidance: a capped multiplicative
    slowness factor on flyable nodes. Same knob, different target (nodes
    from prior accepted paths for this pair, instead of foreign
    terminals).
    """
    nofly_value = float(pget(params, "NOFLY_SLOWNESS_VALUE", 10.0))
    nofly_threshold = float(pget(params, "NOFLY_SLOWNESS_THRESHOLD", 10.0))

    max_k = max(1, int(pget(params, "TABLE_MAX_K", 6)))
    penalty_weight = float(pget(params, "TABLE_PENALTY_WEIGHT", 1.5))
    max_penalty_factor = float(pget(params, "TABLE_MAX_PENALTY_FACTOR", 4.0))
    dup_overlap_pct = float(pget(params, "TABLE_DUPLICATE_OVERLAP_PERCENT", 95.0))

    base_work_model = apply_allowed_mask_to_model(
        model=base_model,
        allowed_mask=mask,
        nofly_value=nofly_value,
        start_idx=start_idx,
        end_idx=end_idx,
    )
    base_work_model = apply_pair_terminal_traffic_avoidance(
        df=base_work_model,
        start_idx=start_idx,
        end_idx=end_idx,
        params=params,
        nofly_threshold=nofly_threshold,
    )
    base_slowness = base_work_model["slowness"].to_numpy(float).copy()
    flyable = base_slowness < nofly_threshold

    node_use_count = np.zeros(len(base_work_model), dtype=int)
    accepted_interior_sets: list[set[int]] = []
    table: list[dict[str, Any]] = []
    theta_kwargs = theta_kwargs_from_params(params, safe_fallback=False)

    # Built once from the UNPENALIZED model, used only to report each
    # accepted path's real cost -- the search itself runs on penalized
    # copies below, so raw total_cost from theta_run is not comparable
    # across rows unless re-priced against this same fixed baseline.
    true_graph = build_graph_for_model(base_work_model, params)

    def _true_path_cost(path_indices: list[int]) -> float:
        return float(sum(
            _mio_edge_cost(base_work_model, true_graph, path_indices[i], path_indices[i + 1])
            for i in range(len(path_indices) - 1)
        ))

    for k in range(1, max_k + 1):
        penalty_factor = np.minimum(max_penalty_factor, 1.0 + penalty_weight * node_use_count)
        penalized_slowness = base_slowness.copy()
        penalized_slowness[flyable] = base_slowness[flyable] * penalty_factor[flyable]

        iter_model = base_work_model.copy()
        iter_model["slowness"] = penalized_slowness
        graph = build_graph_for_model(iter_model, params)

        iter_kwargs = dict(theta_kwargs)
        iter_kwargs["THETASTAR_PROGRESS_LABEL"] = f"{route_name} table[{k}/{max_k}]"

        result = theta_run(
            model=iter_model,
            graph=graph,
            start_idx=int(start_idx),
            end_idx=int(end_idx),
            **iter_kwargs,
        )

        if not isinstance(result, dict) or not result.get("success"):
            break

        raw_path = [int(v) for v in result.get("path_indices", [])]
        if len(raw_path) < 2:
            break

        # Same legality check run_theta_once uses: a table row is never one
        # a real drone couldn't fly, even though it searched a penalized
        # (not the true) cost field.
        invalid_nodes = [idx for idx in raw_path if idx < 0 or idx >= len(mask) or not bool(mask[idx])]
        if invalid_nodes:
            break

        expanded_path, bad_segments = expand_and_validate_path_by_grid_los(
            df=base_model,
            cell_to_idx=cell_to_idx,
            path_indices=raw_path,
            allowed_mask=mask,
            start_idx=start_idx,
            end_idx=end_idx,
        )
        if bad_segments:
            break
        raw_path = expanded_path

        interior = set(raw_path[1:-1])

        if accepted_interior_sets:
            best_interior = accepted_interior_sets[0]
            union = interior | best_interior
            overlap_pct = 100.0 * len(interior & best_interior) / max(1, len(union))
            if overlap_pct >= dup_overlap_pct:
                break  # penalty wasn't enough to move this off the best path; stop rather than pad with near-copies
        else:
            overlap_pct = 0.0

        table.append({
            "rank": len(table) + 1,
            "route_name": route_name,
            "path_indices": raw_path,
            "total_cost": _true_path_cost(raw_path),
            "search_cost_this_iteration": result.get("total_cost"),
            "distance_m": path_distance_m(base_model, raw_path),
            "waypoints": len(raw_path),
            "expanded_nodes": result_expanded(result),
            "runtime_sec": result_runtime(result),
            "overlap_with_shortest_pct": round(overlap_pct, 1),
        })
        accepted_interior_sets.append(interior)
        for idx in interior:
            node_use_count[idx] += 1

    table.sort(key=lambda row: row["total_cost"])
    for i, row in enumerate(table, start=1):
        row["rank"] = i

    return table


def run_project_theta(
    base_model: pd.DataFrame,
    cell_to_idx: dict[tuple[int, int], int],
    base_allowed_mask: np.ndarray,
    start_idx: int,
    end_idx: int,
    params: dict[str, Any],
    extra_allowed_mask: np.ndarray | None = None,
    block_mask: np.ndarray | None = None,
    route_name: str = "",
) -> dict[str, Any]:
    bbox_buffer = float(pget(params, "SEARCH_BBOX_BUFFER_M", 0.0))
    retry_full = bool(pget(params, "RETRY_FULL_MAP_IF_FAILED", True))

    allowed = base_allowed_mask.copy()

    if extra_allowed_mask is not None:
        allowed &= extra_allowed_mask

    if block_mask is not None:
        allowed &= ~block_mask

    allowed[int(start_idx)] = True
    allowed[int(end_idx)] = True

    masks_to_try: list[tuple[np.ndarray, bool]] = []
    if bbox_buffer > 0:
        mask_bbox = allowed & bbox_mask(base_model, start_idx, end_idx, bbox_buffer)
        mask_bbox[int(start_idx)] = True
        mask_bbox[int(end_idx)] = True
        masks_to_try.append((mask_bbox, False))
        if retry_full:
            masks_to_try.append((allowed, True))
    else:
        masks_to_try.append((allowed, False))

    last_result: dict[str, Any] | None = None

    for mask, used_full_retry in masks_to_try:
        # First, try normal configured Theta*.
        result = run_theta_once(
            base_model=base_model,
            cell_to_idx=cell_to_idx,
            mask=mask,
            start_idx=start_idx,
            end_idx=end_idx,
            params=params,
            route_name=route_name,
            safe_fallback=False,
        )
        result["used_full_map_retry"] = bool(used_full_retry)

        if result.get("success", False):
            return result

        last_result = result

        # If any-angle route is invalid, retry pure A* delegate through Theta*.
        result_safe = run_theta_once(
            base_model=base_model,
            cell_to_idx=cell_to_idx,
            mask=mask,
            start_idx=start_idx,
            end_idx=end_idx,
            params=params,
            route_name=route_name,
            safe_fallback=True,
        )
        result_safe["used_full_map_retry"] = bool(used_full_retry)
        result_safe["message"] = str(result_safe.get("message", "")) + " [safe fallback attempted]"

        if result_safe.get("success", False):
            return result_safe

        last_result = result_safe

    if last_result is None:
        return {
            "success": False,
            "message": "No search attempted.",
            "path_indices": [],
            "route_name": route_name,
        }

    return last_result


# ======================================================================
# Route generation
# ======================================================================

def sorted_corridor_list(params: dict[str, Any]) -> list[float]:
    vals = [float(v) for v in list(pget(params, "BACKUP_CORRIDOR_M_LIST", [150.0, 250.0, 400.0, 700.0]))]
    vals = sorted(set(v for v in vals if v >= 0.0))
    return vals or [150.0, 250.0, 400.0, 700.0]


def sorted_avoid_list(params: dict[str, Any]) -> list[float]:
    vals = [float(v) for v in list(pget(params, "BACKWARD_AVOID_FORWARD_RADIUS_M_LIST", [150.0, 100.0, 50.0, 0.0]))]
    vals = sorted(set(v for v in vals if v >= 0.0), reverse=True)
    if 0.0 not in vals:
        vals.append(0.0)
    return vals


def route_sets_per_direction(params: dict[str, Any]) -> int:
    """Number of main/backup route sets to generate per direction.

    Example:
        ROUTE_SETS_PER_DIRECTION = 2

    gives:
        forward_main_01, forward_backup_01
        forward_main_02, forward_backup_02
        backward_main_01, backward_backup_01
        backward_main_02, backward_backup_02
    """
    value = pget(
        params,
        "ROUTE_SETS_PER_DIRECTION",
        pget(params, "N_ROUTES_EACH_DIRECTION", pget(params, "ROUTE_COUNT_EACH_DIRECTION", 1)),
    )
    try:
        return max(1, int(float(value)))
    except Exception:
        return 1


def sorted_main_avoid_list(params: dict[str, Any]) -> list[float]:
    """Avoid-radius list for secondary main routes.

    main_01 is the fastest route.
    main_02, main_03, ... are generated by blocking previous main routes
    with these radii.  Larger values force stronger separation and usually
    make the secondary route longer.
    """
    default = pget(params, "BACKWARD_AVOID_FORWARD_RADIUS_M_LIST", [200.0, 150.0, 100.0, 50.0, 0.0])
    vals = [float(v) for v in list(pget(params, "MAIN_ROUTE_AVOID_PREVIOUS_RADIUS_M_LIST", default))]
    vals = sorted(set(v for v in vals if v >= 0.0), reverse=True)
    if 0.0 not in vals:
        vals.append(0.0)
    return vals


def combined_distance_to_paths(df: pd.DataFrame, paths: list[list[int]]) -> np.ndarray:
    """Minimum distance from each node to any node on any reference path."""
    valid_paths = [p for p in paths if p]
    if not valid_paths:
        return np.full(len(df), np.inf, dtype=float)

    dist = np.full(len(df), np.inf, dtype=float)
    for path in valid_paths:
        dist = np.minimum(dist, distance_to_path(df, path))
    return dist


def unique_path_indices(path: list[int]) -> list[int]:
    seen = set()
    out: list[int] = []
    for idx in path:
        idx = int(idx)
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out


def lock_settings(params: dict[str, Any]) -> tuple[bool, float, float, bool]:
    enabled = bool(pget(params, "ENABLE_ROUTE_NODE_LOCK", True))
    max_overlap_pct = float(pget(params, "MAX_ROUTE_NODE_OVERLAP_PERCENT", 10.0))
    lock_radius_m = float(pget(params, "LOCK_NODE_RADIUS_M", 0.0))
    strict_first = bool(pget(params, "STRICT_LOCK_BLOCK_FIRST", True))
    return enabled, max_overlap_pct, lock_radius_m, strict_first


def build_locked_node_mask(
    df: pd.DataFrame,
    locked_paths: list[list[int]],
    endpoint_mask: np.ndarray,
    lock_radius_m: float,
) -> np.ndarray:
    if not locked_paths:
        return np.zeros(len(df), dtype=bool)
    dist_to_locked = combined_distance_to_paths(df, locked_paths)
    if float(lock_radius_m) <= 0.0:
        mask = np.isfinite(dist_to_locked) & (dist_to_locked <= 1.0e-9)
    else:
        mask = dist_to_locked <= float(lock_radius_m)
    return mask & (~endpoint_mask)


def route_overlap_stats(
    path_indices: list[int],
    locked_mask: np.ndarray,
    endpoint_mask: np.ndarray,
) -> tuple[int, int, float]:
    if not path_indices:
        return 0, 0, 0.0
    idxs = np.array(unique_path_indices(path_indices), dtype=int)
    if idxs.size == 0:
        return 0, 0, 0.0
    valid = ~endpoint_mask[idxs]
    idxs = idxs[valid]
    total = int(idxs.size)
    if total <= 0:
        return 0, 0, 0.0
    overlap = int(np.count_nonzero(locked_mask[idxs]))
    pct = 100.0 * overlap / float(total)
    return overlap, total, pct


def route_respects_lock_limit(
    result: dict[str, Any],
    locked_mask: np.ndarray,
    endpoint_mask: np.ndarray,
    max_overlap_pct: float,
) -> tuple[bool, dict[str, Any]]:
    path = [int(v) for v in result.get("path_indices", [])]
    overlap_nodes, checked_nodes, overlap_pct = route_overlap_stats(path, locked_mask, endpoint_mask)
    result["locked_overlap_nodes"] = int(overlap_nodes)
    result["locked_overlap_checked_nodes"] = int(checked_nodes)
    result["locked_overlap_percent"] = float(overlap_pct)
    ok = overlap_pct <= float(max_overlap_pct) + 1.0e-9
    if not ok:
        result["message"] = (
            str(result.get("message", ""))
            + f" Overlap with locked routes = {overlap_nodes}/{checked_nodes} nodes ({overlap_pct:.2f}%),"
            + f" exceeds limit {float(max_overlap_pct):.2f}%."
        ).strip()
    return ok, result


def duplicate_route_result(
    df: pd.DataFrame,
    source_result: dict[str, Any],
    route_name: str,
    message: str,
) -> dict[str, Any]:
    path = [int(v) for v in source_result.get("path_indices", [])]
    return {
        "success": bool(path),
        "message": message,
        "route_name": route_name,
        "path_indices": path,
        "distance_m": path_distance_m(df, path) if path else np.nan,
        "total_cost": np.nan,
        "runtime_sec": 0.0,
        "expanded_nodes": 0,
        "duplicated_from_main": True,
        "duplicated_from_previous_main": True,
        "safe_fallback": False,
    }


def make_main_route(
    df: pd.DataFrame,
    cell_to_idx: dict[tuple[int, int], int],
    base_allowed: np.ndarray,
    start_idx: int,
    end_idx: int,
    params: dict[str, Any],
    route_name: str,
    previous_main_paths: list[list[int]] | None = None,
    opposite_direction_paths: list[list[int]] | None = None,
    locked_paths: list[list[int]] | None = None,
    rank: int = 1,
) -> dict[str, Any]:
    """Generate one main route.

    rank=1: fastest route with only the base constraints.
    rank>1: alternative main route. It avoids previous main routes and,
            when provided, opposite-direction routes. The alternative does
            not need to be fastest; with ALTERNATIVE_MAIN_SELECTION_MODE =
            "longest", the script keeps the longest successful candidate.
    """
    previous_main_paths = previous_main_paths or []
    opposite_direction_paths = opposite_direction_paths or []
    locked_paths = locked_paths or []
    endpoint_mask = endpoint_buffer_mask(df, start_idx, end_idx, float(pget(params, "ENDPOINT_BUFFER_M", 150.0)))
    _flz_exempt = flz_buffer_exempt_mask(df, params)
    if _flz_exempt is not None:
        endpoint_mask = endpoint_mask | _flz_exempt
    lock_enabled, max_overlap_pct, lock_radius_m, strict_lock_first = lock_settings(params)
    locked_mask = build_locked_node_mask(df, locked_paths, endpoint_mask, lock_radius_m)

    # First main route is the fastest normal route -- but only when there is
    # truly nothing yet to avoid. If an earlier attempt in this same rank
    # was already rejected (no viable backup) and is sitting in
    # previous_main_paths, this must NOT take the bypass, or a retry would
    # silently rediscover the exact same rejected path.
    if int(rank) <= 1 and not opposite_direction_paths and not locked_paths and not previous_main_paths:
        result = run_project_theta(
            base_model=df,
            cell_to_idx=cell_to_idx,
            base_allowed_mask=base_allowed,
            start_idx=start_idx,
            end_idx=end_idx,
            params=params,
            route_name=route_name,
        )
        result["main_rank"] = int(rank)
        result["alternative_main"] = False
        return result

    avoid_paths = list(previous_main_paths) + list(opposite_direction_paths)
    if not avoid_paths and not locked_paths:
        result = run_project_theta(
            base_model=df,
            cell_to_idx=cell_to_idx,
            base_allowed_mask=base_allowed,
            start_idx=start_idx,
            end_idx=end_idx,
            params=params,
            route_name=route_name,
        )
        result["main_rank"] = int(rank)
        result["alternative_main"] = False
        return result

    avoid_list = sorted_main_avoid_list(params)
    selection_mode = str(pget(params, "ALTERNATIVE_MAIN_SELECTION_MODE", "longest")).strip().lower()
    allow_duplicate = bool(pget(params, "ALLOW_MAIN_DUPLICATE_IF_FAILED", False))

    dist_to_avoid = combined_distance_to_paths(df, avoid_paths)

    candidates: list[dict[str, Any]] = []
    best_failed: dict[str, Any] | None = None

    for avoid_m in avoid_list:
        base_block_mask = None if float(avoid_m) <= 0.0 else ((dist_to_avoid <= float(avoid_m)) & (~endpoint_mask))

        candidate_attempts = []
        if lock_enabled and np.any(locked_mask):
            if strict_lock_first:
                candidate_attempts.append(("strict_lock", locked_mask if base_block_mask is None else (base_block_mask | locked_mask)))
            candidate_attempts.append(("soft_lock", base_block_mask))
        else:
            candidate_attempts.append(("no_lock", base_block_mask))

        for lock_mode, block_mask in candidate_attempts:
            result = run_project_theta(
                base_model=df,
                cell_to_idx=cell_to_idx,
                base_allowed_mask=base_allowed,
                start_idx=start_idx,
                end_idx=end_idx,
                params=params,
                block_mask=block_mask,
                route_name=route_name,
            )
            result["main_rank"] = int(rank)
            result["alternative_main"] = True
            result["main_avoid_previous_radius_m"] = float(avoid_m)
            result["lock_mode"] = lock_mode

            if result.get("success", False):
                ok_overlap, result = route_respects_lock_limit(result, locked_mask, endpoint_mask, max_overlap_pct)
                if ok_overlap:
                    candidates.append(result)
                    if selection_mode not in ("longest", "max_distance", "long"):
                        return result
                else:
                    result["success"] = False

            best_failed = result

    if candidates:
        if selection_mode in ("longest", "max_distance", "long"):
            chosen = max(candidates, key=lambda r: float(r.get("distance_m", -np.inf)))
        else:
            chosen = candidates[0]
        chosen["alternative_main_selection_mode"] = selection_mode
        return chosen

    if allow_duplicate and previous_main_paths:
        fake_source = {"path_indices": previous_main_paths[-1]}
        return duplicate_route_result(
            df=df,
            source_result=fake_source,
            route_name=route_name,
            message="Alternative main failed; duplicated previous main route because ALLOW_MAIN_DUPLICATE_IF_FAILED=True.",
        )

    if best_failed is None:
        best_failed = {"success": False, "message": "Alternative main not attempted.", "path_indices": []}

    best_failed["main_rank"] = int(rank)
    best_failed["alternative_main"] = True
    return best_failed


def generate_direction_route_sets(
    df: pd.DataFrame,
    cell_to_idx: dict[tuple[int, int], int],
    base_allowed: np.ndarray,
    start_idx: int,
    end_idx: int,
    params: dict[str, Any],
    pair_name: str,
    direction: str,
    n_sets: int,
    opposite_direction_paths: list[list[int]] | None = None,
    prelocked_paths: list[list[int]] | None = None,
    slot_offset: int = 0,
    slot_total: int = 0,
    ranks: list[int] | None = None,
    shared_previous_main_paths: list[list[int]] | None = None,
    shared_locked_paths: list[list[int]] | None = None,
    extra_locked_paths: list[list[int]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Generate K main + K backup routes for one direction.

    slot_offset/slot_total (added v11): when slot_total > 0 and
    LIVE_PAIR_PROGRESS is on, emit live per-pair progress. The percentage
    is over route SLOTS (2 per rank per direction) because that is the
    only denominator known in advance -- with unlimited retries the number
    of searches a rank will need is unknowable until it finishes. Retry
    attempts emit heartbeat lines instead, so long grinds are visibly
    alive rather than silent.
    """
    routes: dict[str, dict[str, Any]] = {}
    live_progress = bool(slot_total > 0 and bool(pget(params, "LIVE_PAIR_PROGRESS", True)))
    # shared_* lists (added v15) are MUTATED in place so interleaved
    # per-rank calls across both directions accumulate one common state:
    # every accepted or rejected route immediately constrains whichever
    # direction plans next.
    if shared_previous_main_paths is not None:
        previous_main_paths = shared_previous_main_paths
    else:
        previous_main_paths = []
    opposite_direction_paths = opposite_direction_paths if opposite_direction_paths is not None else []
    if shared_locked_paths is not None:
        locked_paths = shared_locked_paths
    else:
        locked_paths = list(prelocked_paths or [])
    couples_done = 0
    # Read-only cross-direction locks (v15): the OTHER direction's mains.
    # Own-direction locks (mains + backups + rejected mains) live in
    # locked_paths and are mutated; cross-direction locking stays
    # mains-only, matching the original sequential semantics -- locking
    # the other direction's backups too was tested and over-constrains
    # (backward ranks that succeed under sequential order start skipping).
    extra_locked = [list(pp) for pp in (extra_locked_paths or [])]

    # 0 (or negative) = unlimited: retry until no new main path exists.
    max_retry = int(pget(params, "MAX_MAIN_BACKUP_RETRY", 0))

    rank_list = [int(r) for r in ranks] if ranks else list(range(1, int(n_sets) + 1))
    for rank in rank_list:
        main_key = f"{direction}_main_{rank:02d}"
        backup_key = f"{direction}_backup_{rank:02d}"

        outcome_main: dict[str, Any] | None = None
        outcome_backup: dict[str, Any] | None = None
        found_any_main = False
        last_main_result: dict[str, Any] | None = None
        rejected_count = 0
        attempt = 0

        # Retry until either a main+backup couple is found, or the main
        # search itself fails -- meaning no new distinct main exists under
        # the (growing) lock list. Termination is guaranteed: every rejected
        # main locks its nodes, and each new candidate must stay under
        # MAX_ROUTE_NODE_OVERLAP_PERCENT against ALL locked nodes, so the
        # candidate space strictly shrinks each cycle. MAX_MAIN_BACKUP_RETRY
        # is only an optional time-budget cap (0 = unlimited).
        while True:
            attempt += 1
            if max_retry > 0 and attempt > max_retry:
                break
            suffix = "" if attempt == 1 else f"_retry{attempt:02d}"
            attempt_main_name = f"{pair_name}_{main_key}{suffix}"
            if live_progress and attempt >= 2:
                live_line(f"    {pair_name} {direction} rank {rank}: attempt {attempt} (searching a new main+backup couple)")

            main_result = make_main_route(
                df=df,
                cell_to_idx=cell_to_idx,
                base_allowed=base_allowed,
                start_idx=start_idx,
                end_idx=end_idx,
                params=params,
                route_name=attempt_main_name,
                previous_main_paths=previous_main_paths,
                opposite_direction_paths=opposite_direction_paths,
                locked_paths=locked_paths + extra_locked,
                rank=rank,
            )
            last_main_result = main_result

            if not main_result.get("success", False):
                # Couldn't even find *a* main under current avoidance --
                # widening the search further won't help; stop retrying.
                break

            found_any_main = True
            main_path = [int(v) for v in main_result.get("path_indices", [])]

            # ---- FLZ buffer enforcement (v10) ----
            flz_on, flz_radius = flz_buffer_settings(params)
            if flz_on:
                min_d = route_min_distance_to_flz(df, main_path)
                main_result["flz_buffer_min_distance_m"] = float(min_d)
                main_result["flz_buffer_ok"] = bool(min_d <= flz_radius)
                if min_d > flz_radius:
                    via_result = replan_main_via_flz(
                        df=df, cell_to_idx=cell_to_idx, base_allowed=base_allowed,
                        start_idx=start_idx, end_idx=end_idx, params=params,
                        route_name=attempt_main_name, locked_paths=locked_paths + extra_locked,
                    )
                    if via_result.get("success", False):
                        via_path = [int(v) for v in via_result.get("path_indices", [])]
                        via_d = route_min_distance_to_flz(df, via_path)
                        via_result["flz_buffer_min_distance_m"] = float(via_d)
                        via_result["flz_buffer_ok"] = bool(via_d <= flz_radius)
                        main_result = via_result
                        main_path = via_path
                    else:
                        # Same lock-as-memory treatment as a failed backup:
                        # lock this main so it is never re-found, and search
                        # for another main+backup couple.
                        previous_main_paths.append(main_path)
                        locked_paths.append(main_path)
                        rejected_count += 1
                        cap_note = f"/{max_retry}" if max_retry > 0 else ""
                        print(
                            f"  reject: {attempt_main_name} misses FLZ buffer "
                            f"(min dist {min_d:.1f} m > {flz_radius:.1f} m) and via-FLZ replan failed "
                            f"(attempt {attempt}{cap_note}) -- locking it, searching for another main"
                        )
                        continue

            backup_result = make_backup_route(
                df=df,
                cell_to_idx=cell_to_idx,
                base_allowed=base_allowed,
                start_idx=start_idx,
                end_idx=end_idx,
                main_path=main_path,
                params=params,
                route_name=f"{pair_name}_{backup_key}{suffix}",
                locked_paths=locked_paths + extra_locked,
            )
            backup_ok = bool(backup_result.get("success", False)) and not backup_result.get("duplicated_from_main", False)

            # Whichever main was just tried -- accepted or about to be
            # rejected -- goes on the avoid-list now, so neither a further
            # retry in this rank nor a later rank can rediscover or
            # duplicate it.
            previous_main_paths.append(main_path)
            locked_paths.append(main_path)

            if backup_ok:
                locked_paths.append([int(v) for v in backup_result.get("path_indices", [])])
                outcome_main, outcome_backup = main_result, backup_result
                break

            rejected_count += 1
            cap_note = f"/{max_retry}" if max_retry > 0 else ""
            print(
                f"  reject: {attempt_main_name} has no viable backup "
                f"(attempt {attempt}{cap_note}) -- locking it, searching for another main"
            )

        if outcome_main is not None:
            main_result, backup_result = outcome_main, outcome_backup
        elif not found_any_main:
            main_result = last_main_result or {"success": False, "message": "No main route found.", "path_indices": []}
            backup_result = {
                "success": False,
                "message": f"Skipped because {main_key} failed.",
                "path_indices": [],
                "route_name": f"{pair_name}_{backup_key}",
            }
        else:
            reason = (
                "no new distinct main exists under the lock/overlap limits"
                if max_retry <= 0 or attempt <= max_retry
                else f"time-budget cap MAX_MAIN_BACKUP_RETRY={max_retry} reached"
            )
            print(
                f"  SKIP: {pair_name} {direction} rank {rank} -- {rejected_count} candidate "
                f"main(s) tried, none had a viable backup; {reason}"
            )
            main_result = {
                "success": False,
                "message": f"Rank skipped: {rejected_count} candidate main(s) tried, none had a "
                           f"viable backup; {reason}.",
                "path_indices": [],
                "route_name": f"{pair_name}_{main_key}",
            }
            backup_result = {
                "success": False,
                "message": "Rank skipped: no main+backup couple found.",
                "path_indices": [],
                "route_name": f"{pair_name}_{backup_key}",
            }

        main_result["direction"] = direction
        main_result["route_type"] = "main"
        main_result["route_rank"] = int(rank)
        routes[main_key] = main_result

        backup_result["direction"] = direction
        backup_result["route_type"] = "backup"
        backup_result["route_rank"] = int(rank)
        routes[backup_key] = backup_result

        couples_done += 1
        if live_progress:
            done = int(slot_offset) + 2 * couples_done
            outcome = "ok" if main_result.get("success", False) else "skip"
            live_line("  " + stage_progress_bar(
                pair_name, done, int(slot_total),
                extra=f"{direction} rank {rank}: {outcome}",
            ))

    return routes


def make_backup_route(
    df: pd.DataFrame,
    cell_to_idx: dict[tuple[int, int], int],
    base_allowed: np.ndarray,
    start_idx: int,
    end_idx: int,
    main_path: list[int],
    params: dict[str, Any],
    route_name: str,
    locked_paths: list[list[int]] | None = None,
) -> dict[str, Any]:
    corridor_list = sorted_corridor_list(params)
    block_radius = float(pget(params, "BACKUP_BLOCK_MAIN_RADIUS_M", 30.0))
    endpoint_buffer_m = float(pget(params, "ENDPOINT_BUFFER_M", 150.0))
    allow_duplicate = bool(pget(params, "ALLOW_BACKUP_DUPLICATE_IF_FAILED", True))
    locked_paths = locked_paths or []
    lock_enabled, max_overlap_pct, lock_radius_m, strict_lock_first = lock_settings(params)

    dist_to_main = distance_to_path(df, main_path)
    endpoint_mask = endpoint_buffer_mask(df, start_idx, end_idx, endpoint_buffer_m)

    # Two masks with two different jobs (split in v12; v10/v11 conflated
    # them, which let a backup ride exactly ON its own main inside FLZ
    # buffers, hiding dashed-under-solid in the figures):
    #   overlap_exempt_mask -- zones where sharing locked nodes is
    #     unavoidable by design (endpoint driveways + FLZ buffers); used
    #     ONLY for lock-overlap accounting.
    #   block carve -- endpoint driveways ONLY. The block-main mask is
    #     what guarantees the backup stays side-by-side rather than
    #     coincident, and that guarantee must hold at the FLZ too.
    _flz_exempt = flz_buffer_exempt_mask(df, params)
    overlap_exempt_mask = (endpoint_mask | _flz_exempt) if _flz_exempt is not None else endpoint_mask
    block_mask = (dist_to_main <= block_radius) & (~endpoint_mask)
    locked_mask = build_locked_node_mask(df, locked_paths, overlap_exempt_mask, lock_radius_m)

    best_failed: dict[str, Any] | None = None

    for corridor_m in corridor_list:
        corridor_mask = dist_to_main <= float(corridor_m)

        candidate_attempts = []
        if lock_enabled and np.any(locked_mask):
            if strict_lock_first:
                candidate_attempts.append(("strict_lock", block_mask | locked_mask))
            candidate_attempts.append(("soft_lock", block_mask))
        else:
            candidate_attempts.append(("no_lock", block_mask))

        for lock_mode, this_block_mask in candidate_attempts:
            result = run_project_theta(
                base_model=df,
                cell_to_idx=cell_to_idx,
                base_allowed_mask=base_allowed,
                start_idx=start_idx,
                end_idx=end_idx,
                params=params,
                extra_allowed_mask=corridor_mask,
                block_mask=this_block_mask,
                route_name=route_name,
            )

            result["backup_corridor_m"] = float(corridor_m)
            result["backup_block_main_radius_m"] = block_radius
            result["lock_mode"] = lock_mode

            if result.get("success", False):
                ok_overlap, result = route_respects_lock_limit(result, locked_mask, overlap_exempt_mask, max_overlap_pct)
                if ok_overlap:
                    flz_on, flz_radius = flz_buffer_settings(params)
                    if flz_on:
                        min_d = route_min_distance_to_flz(df, [int(v) for v in result.get("path_indices", [])])
                        result["flz_buffer_min_distance_m"] = float(min_d)
                        result["flz_buffer_ok"] = bool(min_d <= flz_radius)
                        if min_d > flz_radius:
                            # A cost-minimizing backup will never take the
                            # main's dip to the FLZ voluntarily -- it cuts the
                            # corner. Force it through the buffer with the
                            # same corridor/block masks, exactly as mains are
                            # forced via replan_main_via_flz.
                            via_backup = backup_via_flz_ring(
                                df=df, cell_to_idx=cell_to_idx, base_allowed=base_allowed,
                                start_idx=start_idx, end_idx=end_idx, params=params,
                                route_name=route_name, corridor_mask=corridor_mask,
                                block_mask=this_block_mask, main_path=main_path,
                                flz_radius=flz_radius,
                            )
                            if via_backup.get("success", False):
                                ok2, via_backup = route_respects_lock_limit(via_backup, locked_mask, overlap_exempt_mask, max_overlap_pct)
                                if ok2:
                                    sep_ok2, sep_stats2 = backup_separation_verdict(
                                        df, main_path, [int(v) for v in via_backup.get("path_indices", [])],
                                        start_idx, end_idx, params,
                                    )
                                    via_backup.update(sep_stats2)
                                    if sep_ok2:
                                        via_backup["backup_corridor_m"] = float(corridor_m)
                                        via_backup["backup_block_main_radius_m"] = block_radius
                                        via_backup["lock_mode"] = lock_mode
                                        via_backup["duplicated_from_main"] = False
                                        via_backup.update(backup_vs_main_geometry(df, main_path, via_backup.get("path_indices", [])))
                                        return via_backup
                            result["success"] = False
                            result["message"] = (
                                str(result.get("message", ""))
                                + f" Backup misses FLZ buffer (min {min_d:.1f} m > {flz_radius:.1f} m) and via-ring replan failed."
                            ).strip()
                            best_failed = result
                            continue
                    sep_ok, sep_stats = backup_separation_verdict(
                        df, main_path, [int(v) for v in result.get("path_indices", [])],
                        start_idx, end_idx, params,
                    )
                    result.update(sep_stats)
                    if not sep_ok:
                        result["success"] = False
                        result["message"] = (
                            str(result.get("message", ""))
                            + f" Backup not genuinely separated: {sep_stats.get('backup_separation_violation_pct', float('nan')):.1f}%"
                            + f" of length closer than threshold (allowed {sep_stats.get('backup_separation_allowed_pct', float('nan')):.1f}%)."
                        ).strip()
                        best_failed = result
                        continue
                    result["duplicated_from_main"] = False
                    result.update(backup_vs_main_geometry(df, main_path, result.get("path_indices", [])))
                    return result
                result["success"] = False

            best_failed = result

    if allow_duplicate:
        return {
            "success": True,
            "message": "Backup failed; duplicated main route because ALLOW_BACKUP_DUPLICATE_IF_FAILED=True.",
            "route_name": route_name,
            "path_indices": list(main_path),
            "distance_m": path_distance_m(df, main_path),
            "total_cost": np.nan,
            "runtime_sec": 0.0,
            "expanded_nodes": 0,
            "duplicated_from_main": True,
            "safe_fallback": False,
            "distance_diff_m": 0.0,
            "backup_mean_offset_m": 0.0,
            "backup_max_offset_m": 0.0,
        }

    if best_failed is None:
        best_failed = {"success": False, "message": "Backup not attempted.", "path_indices": []}

    best_failed["duplicated_from_main"] = False
    best_failed.setdefault("distance_diff_m", float("nan"))
    best_failed.setdefault("backup_mean_offset_m", float("nan"))
    best_failed.setdefault("backup_max_offset_m", float("nan"))
    return best_failed


def make_backward_main_route(
    df: pd.DataFrame,
    cell_to_idx: dict[tuple[int, int], int],
    base_allowed: np.ndarray,
    start_idx: int,
    end_idx: int,
    forward_main_path: list[int],
    params: dict[str, Any],
    route_name: str,
) -> dict[str, Any]:
    avoid_list = sorted_avoid_list(params)
    endpoint_buffer_m = float(pget(params, "ENDPOINT_BUFFER_M", 150.0))

    dist_to_forward = distance_to_path(df, forward_main_path)
    endpoint_mask = endpoint_buffer_mask(df, start_idx, end_idx, endpoint_buffer_m)
    _flz_exempt = flz_buffer_exempt_mask(df, params)
    if _flz_exempt is not None:
        endpoint_mask = endpoint_mask | _flz_exempt

    best_failed: dict[str, Any] | None = None

    for avoid_m in avoid_list:
        if float(avoid_m) <= 0:
            block_mask = None
        else:
            block_mask = (dist_to_forward <= float(avoid_m)) & (~endpoint_mask)

        result = run_project_theta(
            base_model=df,
            cell_to_idx=cell_to_idx,
            base_allowed_mask=base_allowed,
            start_idx=start_idx,
            end_idx=end_idx,
            params=params,
            block_mask=block_mask,
            route_name=route_name,
        )

        result["backward_avoid_forward_radius_m"] = float(avoid_m)

        if result.get("success", False):
            return result

        best_failed = result

    if best_failed is None:
        best_failed = {"success": False, "message": "Backward route not attempted.", "path_indices": []}

    return best_failed


# ======================================================================
# Output and plotting
# ======================================================================

def save_route_nodes(
    df: pd.DataFrame,
    path_indices: list[int],
    output_file: Path,
    pair_name: str,
    direction: str,
    route_type: str,
) -> None:
    if not path_indices:
        return

    cols = [
        "node_id", "x", "y", "z", "slowness", "slowness_raw",
        "flz_support", "emergency_risk",
        "risk_obstacle", "risk_ra", "risk_total",
        "obstacle_flag", "ra_flag", "objective_flag", "label", "label_prefix",
    ]
    cols = [c for c in cols if c in df.columns]

    out = df.loc[path_indices, cols].copy()
    out.insert(0, "seq", np.arange(len(out), dtype=int))
    out.insert(1, "pair", pair_name)
    out.insert(2, "direction", direction)
    out.insert(3, "route_type", route_type)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_file, index=False)


def route_edges_dataframe(
    df: pd.DataFrame,
    path_indices: list[int],
    pair_name: str,
    direction: str,
    route_type: str,
) -> pd.DataFrame:
    rows = []

    for k in range(len(path_indices) - 1):
        i = int(path_indices[k])
        j = int(path_indices[k + 1])

        x1 = float(df.at[i, "x"])
        y1 = float(df.at[i, "y"])
        x2 = float(df.at[j, "x"])
        y2 = float(df.at[j, "y"])

        rows.append({
            "pair": pair_name,
            "direction": direction,
            "route_type": route_type,
            "edge_seq": k,
            "from_node_id": int(df.at[i, "node_id"]),
            "to_node_id": int(df.at[j, "node_id"]),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "edge_distance_m": math.hypot(x2 - x1, y2 - y1),
        })

    return pd.DataFrame(rows)


def plot_pair_routes(
    df: pd.DataFrame,
    routes: dict[str, dict[str, Any]],
    pair_name: str,
    output_file: Path,
    nofly_threshold: float,
    plot_node_size: float,
    line_width: float,
    params: dict[str, Any],
) -> None:
    fig, ax = plt.subplots(figsize=(10, 9))
    font_family = str(pget(params, "PLOT_FONT_FAMILY", "DejaVu Serif"))
    title_size = float(pget(params, "PLOT_TITLE_FONT_SIZE", 16))
    label_size = float(pget(params, "PLOT_LABEL_FONT_SIZE", 13))
    legend_size = float(pget(params, "PLOT_LEGEND_FONT_SIZE", 10))
    text_size = float(pget(params, "PLOT_TEXT_FONT_SIZE", 8))
    route_width_m = float(pget(params, "THETASTAR_ROUTE_WIDTH_M", 50.0))
    show_corridor = bool(pget(params, "PLOT_SHOW_ROUTE_CORRIDOR", True))
    corridor_alpha_base = float(pget(params, "PLOT_ROUTE_CORRIDOR_ALPHA", 0.15))
    corridor_edge_alpha = float(pget(params, "PLOT_ROUTE_CORRIDOR_EDGE_ALPHA", 0.5))

    nofly = df["slowness"].to_numpy(float) >= nofly_threshold

    ax.scatter(df.loc[~nofly, "x"], df.loc[~nofly, "y"], s=plot_node_size, c="lightgray", linewidths=0, label="flyable")
    ax.scatter(df.loc[nofly, "x"], df.loc[nofly, "y"], s=plot_node_size, c="black", linewidths=0, label="no-fly")

    marker_info = {
        "DB": ("^", "blue", 90),
        "DK": ("s", "green", 80),
        "FLZ": ("*", "orange", 140),
        "RA": ("X", "purple", 100),
    }

    for prefix, info in marker_info.items():
        marker, color, size = info
        sub = df[df["label_prefix"] == prefix]
        if len(sub) == 0:
            continue
        ax.scatter(sub["x"], sub["y"], marker=marker, s=size, c=color, edgecolors="white", linewidths=0.8, zorder=10, label=prefix)
        for _, row in sub.iterrows():
            ax.text(row["x"], row["y"], str(row["label"]), fontsize=text_size, color=color, weight="bold", zorder=11, fontfamily=font_family)

    seen_route_labels: set[str] = set()
    for key, result in routes.items():
        path = result.get("path_indices", [])
        if not path:
            continue
        # A rejected route can still carry its best-failed path; drawing it
        # like an accepted route makes figures lie (a failed backward main
        # renders as a lone solid line with no backup beside it). Skip
        # unless PLOT_INCLUDE_FAILED asks for diagnostic display.
        if not result.get("success", False):
            include_failed = bool(pget(params, "PLOT_INCLUDE_FAILED", False)) if params is not None else False
            if not include_failed:
                continue
            xy_f = path_to_xy(df, path)
            ax.plot(xy_f[:, 0], xy_f[:, 1], color="gray", linestyle=":", linewidth=1.0, alpha=0.6,
                    label="failed (not accepted)" if "failed (not accepted)" not in seen_route_labels else "_nolegend_",
                    zorder=15)
            seen_route_labels.add("failed (not accepted)")
            continue

        direction = str(result.get("direction", "forward" if key.startswith("forward") else "backward"))
        route_type = str(result.get("route_type", "backup" if "backup" in key else "main"))
        route_rank = int(result.get("route_rank", 1))

        xy = path_to_xy(df, path)
        color = "red" if direction == "forward" else "blue"
        linestyle = "-" if route_type == "main" else "--"
        alpha = max(0.30, 0.95 - 0.12 * (route_rank - 1))
        lw = line_width if route_rank == 1 else max(1.0, line_width * 0.75)
        label = f"{direction} {route_type}"
        if label in seen_route_labels:
            label = "_nolegend_"
        else:
            seen_route_labels.add(label)

        if show_corridor:
            corridor_xy = route_corridor_polygon_xy(xy, route_width_m / 2.0)
            if corridor_xy is not None:
                corridor_label = f"route corridor (Ø{route_width_m:.0f} m)"
                if corridor_label in seen_route_labels:
                    corridor_label = "_nolegend_"
                else:
                    seen_route_labels.add(corridor_label)
                draw_route_corridor(
                    ax, corridor_xy,
                    fill_color=color,
                    fill_alpha=corridor_alpha_base * (alpha / 0.95),
                    edge_alpha=corridor_edge_alpha,
                    zorder=5,
                    label=corridor_label,
                )

        ax.plot(
            xy[:, 0],
            xy[:, 1],
            color=color,
            linestyle=linestyle,
            linewidth=lw,
            alpha=alpha,
            label=label,
            zorder=20,
        )

    ax.set_title(pair_name, fontfamily=font_family, fontsize=title_size, fontweight="bold")
    ax.set_xlabel("X coordinate (m)", fontfamily=font_family, fontsize=label_size)
    ax.set_ylabel("Y coordinate (m)", fontfamily=font_family, fontsize=label_size)
    ax.set_aspect("equal", adjustable="box")
    add_map_rule(ax)
    ax.grid(True, alpha=0.25)
    legend = ax.legend(fontsize=legend_size, loc="upper right", frameon=True)
    for text_obj in legend.get_texts():
        text_obj.set_fontfamily(font_family)

    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=250)
    plt.close(fig)


def plot_overview(
    df: pd.DataFrame,
    route_records: list[dict[str, Any]],
    output_file: Path,
    nofly_threshold: float,
    plot_node_size: float,
    params: dict[str, Any],
) -> None:
    fig, ax = plt.subplots(figsize=(11, 10))
    font_family = str(pget(params, "PLOT_FONT_FAMILY", "DejaVu Serif"))
    title_size = float(pget(params, "PLOT_TITLE_FONT_SIZE", 16))
    label_size = float(pget(params, "PLOT_LABEL_FONT_SIZE", 13))
    legend_size = float(pget(params, "PLOT_LEGEND_FONT_SIZE", 10))
    text_size = float(pget(params, "PLOT_TEXT_FONT_SIZE", 8))
    route_width_m = float(pget(params, "THETASTAR_ROUTE_WIDTH_M", 50.0))
    show_corridor = bool(pget(params, "PLOT_SHOW_ROUTE_CORRIDOR", True))
    corridor_alpha_base = float(pget(params, "PLOT_ROUTE_CORRIDOR_ALPHA", 0.15))
    corridor_edge_alpha = float(pget(params, "PLOT_ROUTE_CORRIDOR_EDGE_ALPHA", 0.5))

    nofly = df["slowness"].to_numpy(float) >= nofly_threshold

    ax.scatter(df.loc[~nofly, "x"], df.loc[~nofly, "y"], s=plot_node_size, c="lightgray", linewidths=0, label="flyable")
    ax.scatter(df.loc[nofly, "x"], df.loc[nofly, "y"], s=plot_node_size, c="black", linewidths=0, label="no-fly")

    for rec in route_records:
        path = rec.get("path_indices", [])
        if not path:
            continue
        xy = path_to_xy(df, path)
        direction = rec.get("direction", "")
        route_type = rec.get("route_type", "")
        color = "red" if direction == "forward" else "blue"
        linestyle = "-" if route_type == "main" else "--"
        alpha = 0.75 if route_type == "main" else 0.45
        if show_corridor:
            corridor_xy = route_corridor_polygon_xy(xy, route_width_m / 2.0)
            if corridor_xy is not None:
                draw_route_corridor(
                    ax, corridor_xy,
                    fill_color=color,
                    fill_alpha=corridor_alpha_base * alpha,
                    edge_alpha=corridor_edge_alpha,
                    zorder=5,
                )
        ax.plot(xy[:, 0], xy[:, 1], color=color, linestyle=linestyle, linewidth=1.2, alpha=alpha, zorder=20)

    marker_info = {
        "DB": ("^", "blue", 90),
        "DK": ("s", "green", 80),
        "FLZ": ("*", "orange", 140),
        "RA": ("X", "purple", 100),
    }

    for prefix, info in marker_info.items():
        marker, color, size = info
        sub = df[df["label_prefix"] == prefix]
        if len(sub) == 0:
            continue
        ax.scatter(sub["x"], sub["y"], marker=marker, s=size, c=color, edgecolors="white", linewidths=0.8, zorder=10, label=prefix)
        for _, row in sub.iterrows():
            ax.text(row["x"], row["y"], str(row["label"]), fontsize=text_size, color=color, weight="bold", zorder=11, fontfamily=font_family)

    ax.set_title("All Theta* master routes v15", fontfamily=font_family, fontsize=title_size, fontweight="bold")
    ax.set_xlabel("X coordinate (m)", fontfamily=font_family, fontsize=label_size)
    ax.set_ylabel("Y coordinate (m)", fontfamily=font_family, fontsize=label_size)
    ax.set_aspect("equal", adjustable="box")
    add_map_rule(ax)
    ax.grid(True, alpha=0.25)
    legend = ax.legend(fontsize=legend_size, loc="upper right", frameon=True)
    for text_obj in legend.get_texts():
        text_obj.set_fontfamily(font_family)

    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=250)
    plt.close(fig)


# ======================================================================
# Main workflow
# ======================================================================

def _pool_worker_init():
    """Initializer for parallel workers: force the non-interactive
    Agg matplotlib backend so figure saving works in subprocesses
    with no display."""
    import matplotlib
    matplotlib.use("Agg", force=True)


def process_single_pair(
    pair_id: int,
    pair: dict[str, Any],
    df: pd.DataFrame,
    cell_to_idx: dict[tuple[int, int], int],
    base_allowed: np.ndarray,
    params: dict[str, Any],
    route_dir: Path,
    figure_dir: Path,
    nofly_threshold: float,
    n_route_sets: int,
) -> dict[str, Any]:
    """One pair's complete forward+backward planning: route search,
    per-route CSVs, summary rows, edge tables, and the pair figure.

    Top-level (not nested) so multiprocessing can pickle it. All prints
    are captured and returned as log_text so parallel workers never
    interleave partial lines on the console."""
    import io
    import contextlib

    summary_rows: list[dict[str, Any]] = []
    route_records: list[dict[str, Any]] = []
    edge_tables: list[pd.DataFrame] = []

    log_buf = io.StringIO()
    with contextlib.redirect_stdout(log_buf):
        a_idx = int(pair["a_idx"])
        b_idx = int(pair["b_idx"])
        a_label = str(pair["a_label"])
        b_label = str(pair["b_label"])
        pair_name = f"{safe_name(a_label)}_to_{safe_name(b_label)}"


        direction_order = str(pget(params, "PAIR_DIRECTION_ORDER", "interleaved")).strip().lower()

        if direction_order == "interleaved":
            # Balanced rank order (v15): fwd couple 1 -> bwd couple 1 ->
            # fwd couple 2 -> bwd couple 2 -> ... All state is shared, so
            # each new couple avoids EVERY route claimed so far in BOTH
            # directions. This removes the structural asymmetry of the old
            # sequential order, where forward planned all its ranks first
            # and monopolized the good corridors, leaving backward to
            # exhaust against a fully locked map.
            forward_routes: dict[str, Any] = {}
            backward_routes: dict[str, Any] = {}
            fwd_prev_mains: list[list[int]] = []
            bwd_prev_mains: list[list[int]] = []
            fwd_own_locked: list[list[int]] = []
            bwd_own_locked: list[list[int]] = []
            slot_total = 4 * int(n_route_sets)
            for rank in range(1, int(n_route_sets) + 1):
                forward_routes.update(generate_direction_route_sets(
                    df=df, cell_to_idx=cell_to_idx, base_allowed=base_allowed,
                    start_idx=a_idx, end_idx=b_idx, params=params,
                    pair_name=pair_name, direction="forward", n_sets=n_route_sets,
                    opposite_direction_paths=bwd_prev_mains,
                    ranks=[rank],
                    shared_previous_main_paths=fwd_prev_mains,
                    shared_locked_paths=fwd_own_locked,
                    extra_locked_paths=bwd_prev_mains,
                    slot_offset=4 * (rank - 1), slot_total=slot_total,
                ))
                backward_routes.update(generate_direction_route_sets(
                    df=df, cell_to_idx=cell_to_idx, base_allowed=base_allowed,
                    start_idx=b_idx, end_idx=a_idx, params=params,
                    pair_name=pair_name, direction="backward", n_sets=n_route_sets,
                    opposite_direction_paths=fwd_prev_mains,
                    ranks=[rank],
                    shared_previous_main_paths=bwd_prev_mains,
                    shared_locked_paths=bwd_own_locked,
                    extra_locked_paths=fwd_prev_mains,
                    slot_offset=4 * (rank - 1) + 2, slot_total=slot_total,
                ))
        else:
            # Legacy sequential order: all forward ranks, then all backward.
            forward_routes = generate_direction_route_sets(
                df=df,
                cell_to_idx=cell_to_idx,
                base_allowed=base_allowed,
                start_idx=a_idx,
                end_idx=b_idx,
                params=params,
                pair_name=pair_name,
                direction="forward",
                n_sets=n_route_sets,
                opposite_direction_paths=[],
                prelocked_paths=[],
                slot_offset=0,
                slot_total=4 * int(n_route_sets),
            )

            forward_main_paths = [
                [int(v) for v in r.get("path_indices", [])]
                for k, r in forward_routes.items()
                if r.get("route_type") == "main" and r.get("success", False)
            ]

            backward_routes = generate_direction_route_sets(
                df=df,
                cell_to_idx=cell_to_idx,
                base_allowed=base_allowed,
                start_idx=b_idx,
                end_idx=a_idx,
                params=params,
                pair_name=pair_name,
                direction="backward",
                n_sets=n_route_sets,
                opposite_direction_paths=forward_main_paths,
                prelocked_paths=forward_main_paths,
                slot_offset=2 * int(n_route_sets),
                slot_total=4 * int(n_route_sets),
            )

        routes = {}
        routes.update(forward_routes)
        routes.update(backward_routes)

        for route_key, result in routes.items():
            direction = str(result.get("direction", "forward" if route_key.startswith("forward") else "backward"))
            route_type = str(result.get("route_type", "backup" if "backup" in route_key else "main"))
            route_rank = int(result.get("route_rank", 1))
            path = [int(v) for v in result.get("path_indices", [])]
            route_file = route_dir / f"{pair_name}_{route_key}.csv"

            if path:
                save_route_nodes(
                    df=df,
                    path_indices=path,
                    output_file=route_file,
                    pair_name=pair_name,
                    direction=direction,
                    route_type=route_type,
                )
                edge_table = route_edges_dataframe(
                    df=df,
                    path_indices=path,
                    pair_name=pair_name,
                    direction=direction,
                    route_type=route_type,
                )
                if len(edge_table):
                    edge_table["route_rank"] = route_rank
                    edge_table["route_key"] = route_key
                edge_tables.append(edge_table)

            if route_type == "backup":
                ref_key = f"{direction}_main_{route_rank:02d}"
                ref_path = routes.get(ref_key, {}).get("path_indices", [])
                mean_distance_to_main = closeness_to_reference_m(df, path, ref_path)
            else:
                mean_distance_to_main = np.nan

            total_cost = result.get("total_cost", result.get("cost", np.nan))

            row = {
                "version": VERSION,
                "pair": pair_name,
                "route_key": route_key,
                "start_label": a_label if direction == "forward" else b_label,
                "end_label": b_label if direction == "forward" else a_label,
                "direction": direction,
                "route_type": route_type,
                "route_rank": route_rank,
                "success": bool(result.get("success", False)),
                "message": result.get("message", result.get("status", "")),
                "distance_m": float(result.get("distance_m", np.nan)),
                "distance_km": float(result.get("distance_m", np.nan)) / 1000.0
                    if not pd.isna(result.get("distance_m", np.nan)) else np.nan,
                "total_cost": total_cost,
                "n_path_nodes": len(path),
                "expanded_nodes": int(result.get("expanded_nodes", -1)),
                "runtime_sec": float(result.get("runtime_sec", np.nan)),
                "mean_distance_to_main_m": mean_distance_to_main,
                "duplicated_from_main": bool(result.get("duplicated_from_main", False)),
                "duplicated_from_previous_main": bool(result.get("duplicated_from_previous_main", False)),
                "used_full_map_retry": bool(result.get("used_full_map_retry", False)),
                "safe_fallback": bool(result.get("safe_fallback", False)),
                "main_avoid_previous_radius_m": result.get("main_avoid_previous_radius_m", np.nan),
                "backup_corridor_m": result.get("backup_corridor_m", np.nan),
                "backup_mean_offset_m": result.get("backup_mean_offset_m", np.nan),
                "backup_min_offset_m": result.get("backup_min_offset_m", np.nan),
                "backup_separation_violation_pct": result.get("backup_separation_violation_pct", np.nan),
                "backup_separation_ok": result.get("backup_separation_ok", np.nan),
                "backup_max_offset_m": result.get("backup_max_offset_m", np.nan),
                "distance_diff_m": result.get("distance_diff_m", np.nan),
                "duplicated_from_main": result.get("duplicated_from_main", np.nan),
                "flz_buffer_ok": result.get("flz_buffer_ok", np.nan),
                "flz_buffer_min_distance_m": result.get("flz_buffer_min_distance_m", np.nan),
                "flz_buffer_via": result.get("flz_buffer_via", ""),
                "lock_mode": result.get("lock_mode", ""),
                "locked_overlap_nodes": int(result.get("locked_overlap_nodes", 0)),
                "locked_overlap_checked_nodes": int(result.get("locked_overlap_checked_nodes", 0)),
                "locked_overlap_percent": float(result.get("locked_overlap_percent", 0.0)),
                "mean_flz_support": float(result.get("mean_flz_support", np.nan)),
                "mean_emergency_risk": float(result.get("mean_emergency_risk", np.nan)),
                "mean_other_terminal_avoidance": float(result.get("mean_other_terminal_avoidance", np.nan)),
                "mean_traffic_penalty_factor": float(result.get("mean_traffic_penalty_factor", np.nan)),
                "route_file": str(route_file) if path else "",
            }

            summary_rows.append(row)

            # Summary rows keep failures (they must stay auditable); the
            # overview figure must not draw rejected routes as accepted.
            if bool(result.get("success", False)) or bool(pget(params, "PLOT_INCLUDE_FAILED", False)):
                record = dict(row)
                record["path_indices"] = path
                route_records.append(record)

            print(
                f"  {route_key:20s} | "
                f"success={str(row['success']):5s} | "
                f"dist={row['distance_m']:9.1f} m | "
                f"nodes={row['n_path_nodes']:5d} | "
                f"expanded={row['expanded_nodes']:8d} | "
                f"fallback={str(row['safe_fallback']):5s} | "
                f"time={row['runtime_sec']:7.3f} s"
            )

        if bool(pget(params, "SAVE_PAIR_FIGURES", True)):
            plot_pair_routes(
                df=df,
                routes=routes,
                pair_name=pair_name,
                output_file=figure_dir / f"{pair_name}_theta_routes_v15.png",
                nofly_threshold=nofly_threshold,
                plot_node_size=float(pget(params, "PLOT_NODE_SIZE", 4.0)),
                line_width=float(pget(params, "PLOT_ROUTE_LINEWIDTH", 2.2)),
                params=params,
            )



    return {
        "pair_id": int(pair_id),
        "pair_name": pair_name,
        "pair_label": f"{a_label} \u2194 {b_label}",
        "summary_rows": summary_rows,
        "route_records": route_records,
        "edge_tables": edge_tables,
        "log_text": log_buf.getvalue(),
    }


def main() -> None:
    args = parse_args()
    params = load_params(args.param_file)

    model_file = Path(str(pget(params, "MODEL_FILE", "")))
    output_dir = Path(str(pget(params, "OUTPUT_DIR", "output/02_thetastar_master_plan")))

    route_dir = output_dir / "route_nodes"
    figure_dir = output_dir / "figures"

    output_dir.mkdir(parents=True, exist_ok=True)
    route_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"THETA* MASTER OBJECTIVE-PAIR PLANNER {VERSION}")
    print("=" * 80)
    print(f"Param file      : {args.param_file}")
    print(f"Model file      : {model_file}")
    print(f"Output directory: {output_dir}")

    nofly_threshold = float(pget(params, "NOFLY_SLOWNESS_THRESHOLD", 10.0))

    df = read_node_model(model_file)
    df, cell_to_idx, grid_m = add_grid_index(df)

    # FLZ support modifies slowness only on flyable cells.
    df = apply_flz_safety_attraction(df, params, nofly_threshold=nofly_threshold)

    planning_model_file = output_dir / "planning_model_with_flz_support.xyz"
    save_cols = [c for c in df.columns if not c.startswith("_")]
    df[save_cols].to_csv(planning_model_file, sep=" ", index=False, float_format="%.6f")

    print(f"Nodes           : {len(df):,}")
    print(f"Inferred grid    : {grid_m:.3f} m")
    print(f"Planning model   : {planning_model_file}")

    base_allowed = df["slowness"].to_numpy(float) < nofly_threshold

    route_prefixes, exclude_prefixes = clean_route_prefixes(params)
    print(f"Route prefixes  : {route_prefixes}")
    print(f"Excluded prefix : {exclude_prefixes}")

    obj = objective_table(df=df, route_prefixes=route_prefixes, exclude_prefixes=exclude_prefixes)

    if len(obj) < 2:
        raise RuntimeError("Need at least two DB/DK route objectives.")

    if bool(pget(params, "FORCE_ROUTE_OBJECTIVES_FLYABLE", False)):
        for idx in obj["idx"].to_numpy(int):
            base_allowed[int(idx)] = True

    obj.to_csv(output_dir / "objective_table.csv", index=False)

    pair_mode = str(pget(params, "PAIR_MODE", "unordered"))
    skip_same_prefix = bool(pget(params, "SKIP_SAME_PREFIX", False))
    max_pair_distance_m = float(pget(params, "MAX_PAIR_DISTANCE_M", 0.0))

    pairs = make_pairs(
        obj=obj,
        pair_mode=pair_mode,
        skip_same_prefix=skip_same_prefix,
        max_pair_distance_m=max_pair_distance_m,
    )

    test_single_pair = str(pget(params, "TEST_SINGLE_PAIR", "") or "").strip()
    if test_single_pair:
        wanted = [tok.strip().upper() for tok in test_single_pair.replace("-", ",").split(",") if tok.strip()]
        if len(wanted) != 2:
            raise ValueError(
                f"TEST_SINGLE_PAIR='{test_single_pair}' is not two labels. "
                "Expected something like 'DB01,DK01'."
            )
        wanted_set = set(wanted)
        matched = [
            p for p in pairs
            if {str(p["a_label"]).upper(), str(p["b_label"]).upper()} == wanted_set
        ]
        if not matched:
            raise ValueError(
                f"TEST_SINGLE_PAIR='{test_single_pair}' did not match any pair in "
                f"the {len(pairs)} generated pairs. Check the labels and PAIR_MODE/"
                "SKIP_SAME_PREFIX/MAX_PAIR_DISTANCE_M filters above."
            )
        pairs = matched
        print(f"TEST MODE       : running 1 pair only ({wanted[0]} <-> {wanted[1]})")

    print(f"Route objectives: {len(obj)}")
    print(obj[["label", "label_prefix", "x", "y"]].to_string(index=False))
    print(f"Route pairs     : {len(pairs)}")
    print(f"Backup corridor : {sorted_corridor_list(params)}")
    print(f"Backward avoid  : {sorted_avoid_list(params)}")
    print("-" * 80)

    summary_rows: list[dict[str, Any]] = []
    route_records: list[dict[str, Any]] = []
    edge_tables: list[pd.DataFrame] = []

    n_route_sets = route_sets_per_direction(params)
    print(f"Route sets/dir  : {n_route_sets}")
    print(f"Main avoid list : {sorted_main_avoid_list(params)}")
    print(f"Node lock       : {lock_settings(params)}")
    print(f"FLZ attraction  : {bool(pget(params, 'USE_FLZ_SAFETY_ATTRACTION', True))}, sigma={float(pget(params, 'FLZ_SUPPORT_SIGMA_M', 700.0)):.1f} m, weight={float(pget(params, 'FLZ_SAFETY_WEIGHT', 0.30)):.2f}")
    print(f"DB/DK avoidance : {bool(pget(params, 'USE_OTHER_DB_DK_TRAFFIC_AVOIDANCE', True))}, sigma={float(pget(params, 'OTHER_TERMINAL_AVOID_SIGMA_M', 450.0)):.1f} m, weight={float(pget(params, 'OTHER_TERMINAL_AVOID_WEIGHT', 0.80)):.2f}")

    n_cores_param = int(pget(params, "N_CORES", 0))
    # Cap at cpu_count-1 so one core always stays free for the OS: N_CORES <= 0
    # means "use that cap", and a larger request is clamped down to it.
    n_cap = max(1, (os.cpu_count() or 2) - 1)
    n_cores = n_cap if n_cores_param <= 0 else min(n_cores_param, n_cap)
    clamped = n_cores_param > n_cap
    n_cores = max(1, min(n_cores, len(pairs))) if pairs else 1
    note = (" (auto: max-1)" if n_cores_param <= 0
            else (f" (N_CORES={n_cores_param} clamped to max-1={n_cap})" if clamped
                  else " (from N_CORES)"))
    print(f"Cores           : {n_cores}" + note)
    print("-" * 80)

    worker_args = [
        (
            pair_id, pair, df, cell_to_idx, base_allowed, params,
            route_dir, figure_dir, nofly_threshold, n_route_sets,
        )
        for pair_id, pair in enumerate(pairs, start=1)
    ]

    pair_results: list[dict[str, Any]] = []

    if n_cores <= 1:
        for args in worker_args:
            res = process_single_pair(*args)
            print(batch_progress_bar(len(pair_results) + 1, len(pairs), label=res["pair_label"]))
            print(res["log_text"], end="")
            print("-" * 80)
            pair_results.append(res)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        done_count = 0
        with ProcessPoolExecutor(max_workers=n_cores, initializer=_pool_worker_init) as pool:
            futures = [pool.submit(process_single_pair, *args) for args in worker_args]
            for fut in as_completed(futures):
                res = fut.result()
                done_count += 1
                print(batch_progress_bar(done_count, len(pairs), label=res["pair_label"]))
                print(res["log_text"], end="")
                print("-" * 80)
                pair_results.append(res)
        # Completion order is nondeterministic; restore input order so the
        # summary CSV and overview are stable run to run.
        pair_results.sort(key=lambda r: r["pair_id"])

    for res in pair_results:
        summary_rows.extend(res["summary_rows"])
        route_records.extend(res["route_records"])
        edge_tables.extend(res["edge_tables"])
    summary_df = pd.DataFrame(summary_rows)
    summary_file = output_dir / "route_summary_v15.csv"
    summary_df.to_csv(summary_file, index=False)
    print(stage_progress_bar("Route tables export", 1, 2))

    if edge_tables:
        edge_df = pd.concat(edge_tables, ignore_index=True)
    else:
        edge_df = pd.DataFrame()

    edge_file = output_dir / "all_route_edges_v15.csv"
    edge_df.to_csv(edge_file, index=False)
    print(stage_progress_bar("Route tables export", 2, 2))

    if bool(pget(params, "SAVE_OVERVIEW_FIGURE", True)):
        # Save one overview inside figures/ and one directly in OUTPUT_DIR.
        # The parent-directory copy is easier to find after batch processing.
        overview_figures = figure_dir / "00_all_theta_routes_overview_v15.png"
        overview_parent = output_dir / "00_all_theta_routes_overview_v15.png"

        plot_overview(
            df=df,
            route_records=route_records,
            output_file=overview_figures,
            nofly_threshold=nofly_threshold,
            plot_node_size=float(pget(params, "PLOT_NODE_SIZE", 4.0)),
            params=params,
        )

        print(stage_progress_bar("Overview figures", 1, 2))

        plot_overview(
            df=df,
            route_records=route_records,
            output_file=overview_parent,
            nofly_threshold=nofly_threshold,
            plot_node_size=float(pget(params, "PLOT_NODE_SIZE", 4.0)),
            params=params,
        )
        print(stage_progress_bar("Overview figures", 2, 2))

    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Objective table : {output_dir / 'objective_table.csv'}")
    print(f"Planning model  : {planning_model_file}")
    print(f"Route summary   : {summary_file}")
    print(f"All route edges : {edge_file}")
    print(f"Route nodes     : {route_dir}")
    print(f"Figures         : {figure_dir}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[FAILED] {exc}", file=sys.stderr)
        raise
