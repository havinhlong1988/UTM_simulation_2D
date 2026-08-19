#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Random 2D node-based riskmap generator for LAE-UTM.

Purpose
-------
Generate a synthetic 2D node map in meter coordinates with:

    - flyable nodes
    - random obstacles (kept clear of a border ring, --border-margin-m, so a
      routable outer strip is preserved for corridors to spread dense flows
      around the boundary)
    - DB  : drone base
    - DK  : docking station
    - FLZ : emergency landing zone
    - RA  : restricted airspace / no-fly area

The output is saved as an XYZ-like node table.

Coordinate system
-----------------
The map is generated in local Cartesian meter coordinates:

    x = 0 ... MAP_WIDTH_M
    y = 0 ... MAP_HEIGHT_M
    z = 0

Output columns
--------------
    node_id
    x
    y
    z
    slowness
    risk_obstacle
    risk_ra
    risk_total
    obstacle_flag
    ra_flag
    objective_flag
    label
    label_prefix

Default logic
-------------
    Flyable node:
        slowness = 0.085 s/m

    Obstacle / RA no-fly node:
        slowness = 10.0 s/m

    No-fly rule:
        slowness >= 10.0

Example
-------
python 01_generate_random_2d_node_riskmap.py \\
    --width-m 5000 \\
    --height-m 5000 \\
    --dx-m 50 \\
    --obstacle-rate 0.20 \\
    --n-db 2 \\
    --n-dk 6 \\
    --n-flz 4 \\
    --n-ra 3 \\
    --seed 12 \\
    --output-dir output/01_random_node_map_seed12

"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.maprule import add_map_rule


# ======================================================================
# Default model values
# ======================================================================

DEFAULT_FLYABLE_SLOWNESS = 0.085
DEFAULT_NOFLY_SLOWNESS = 10.0

DEFAULT_Z_VALUE = 0.0

# Keep obstacles / RA clear of a border ring of this width (m) so a routable
# strip is always left along all four map edges. Downstream (02/06/07) can then
# lay corridors AROUND the boundary, giving high-density interior flows an
# alternative outer path -- spreading traffic instead of forcing it through the
# congested centre. 0 disables the ring (obstacles may touch the border).
DEFAULT_BORDER_MARGIN_M = 400.0

LABEL_NONE = "NONE"
PREFIX_NONE = "NONE"


# ======================================================================
# Utility functions
# ======================================================================

def make_grid(width_m: float, height_m: float, dx_m: float) -> pd.DataFrame:
    """
    Create regular 2D node grid in meter coordinates.
    """
    xs = np.arange(0.0, width_m + 0.5 * dx_m, dx_m)
    ys = np.arange(0.0, height_m + 0.5 * dx_m, dx_m)

    xx, yy = np.meshgrid(xs, ys)

    df = pd.DataFrame({
        "x": xx.ravel(),
        "y": yy.ravel(),
    })

    df.insert(0, "node_id", np.arange(len(df), dtype=int))
    df["z"] = DEFAULT_Z_VALUE

    return df


def distance_to_center(df: pd.DataFrame, cx: float, cy: float) -> np.ndarray:
    """
    Distance from all nodes to one center.
    """
    return np.sqrt((df["x"].to_numpy() - cx) ** 2 + (df["y"].to_numpy() - cy) ** 2)


def random_circle_obstacles(
    df: pd.DataFrame,
    width_m: float,
    height_m: float,
    dx_m: float,
    obstacle_rate: float,
    rng: np.random.Generator,
    min_radius_m: float,
    max_radius_m: float,
    border_margin_m: float = 0.0,
    max_trials: int = 10000,
) -> tuple[np.ndarray, list[dict]]:
    """
    Generate random circular obstacle blobs until target obstacle rate is reached.

    border_margin_m keeps every blob fully inside the map by at least this margin,
    leaving a clear routable ring along all four edges.
    """
    n_nodes = len(df)
    target_count = int(round(obstacle_rate * n_nodes))

    obstacle_mask = np.zeros(n_nodes, dtype=bool)
    obstacle_objects: list[dict] = []

    if target_count <= 0:
        return obstacle_mask, obstacle_objects

    for trial in range(max_trials):
        current_count = int(obstacle_mask.sum())
        if current_count >= target_count:
            break

        radius = rng.uniform(min_radius_m, max_radius_m)
        # keep the whole blob clear of the border ring so a routable edge strip
        # is preserved: centre in [margin + radius, extent - margin - radius].
        lo_x = border_margin_m + radius
        hi_x = width_m - border_margin_m - radius
        lo_y = border_margin_m + radius
        hi_y = height_m - border_margin_m - radius
        if hi_x <= lo_x or hi_y <= lo_y:
            continue                      # margin too large for this radius; retry
        cx = rng.uniform(lo_x, hi_x)
        cy = rng.uniform(lo_y, hi_y)

        d = distance_to_center(df, cx, cy)
        new_mask = d <= radius

        before = obstacle_mask.sum()
        obstacle_mask |= new_mask
        after = obstacle_mask.sum()

        if after > before:
            obstacle_objects.append({
                "type": "circle_obstacle",
                "cx": float(cx),
                "cy": float(cy),
                "radius_m": float(radius),
                "added_nodes": int(after - before),
            })

    return obstacle_mask, obstacle_objects


def choose_free_center_node(
    df: pd.DataFrame,
    available_mask: np.ndarray,
    rng: np.random.Generator,
    min_dist_m: float,
    chosen_centers: list[tuple[float, float]],
    max_trials: int = 10000,
) -> int:
    """
    Randomly choose one available node, separated from previously chosen centers.
    """
    available_indices = np.flatnonzero(available_mask)

    if len(available_indices) == 0:
        raise RuntimeError("No available free node remains for objective placement.")

    for _ in range(max_trials):
        idx = int(rng.choice(available_indices))
        x = float(df.at[idx, "x"])
        y = float(df.at[idx, "y"])

        ok = True
        for px, py in chosen_centers:
            if math.hypot(x - px, y - py) < min_dist_m:
                ok = False
                break

        if ok:
            return idx

    raise RuntimeError(
        "Could not place objective with the requested minimum separation. "
        "Reduce --objective-min-dist-m or reduce number of objectives."
    )


def add_point_objectives(
    df: pd.DataFrame,
    available_mask: np.ndarray,
    rng: np.random.Generator,
    prefix: str,
    count: int,
    chosen_centers: list[tuple[float, float]],
    min_dist_m: float,
) -> list[dict]:
    """
    Add point objectives such as DB, DK, FLZ.
    """
    objects: list[dict] = []

    for i in range(1, count + 1):
        idx = choose_free_center_node(
            df=df,
            available_mask=available_mask,
            rng=rng,
            min_dist_m=min_dist_m,
            chosen_centers=chosen_centers,
        )

        label = f"{prefix}{i:02d}"

        df.at[idx, "label"] = label
        df.at[idx, "label_prefix"] = prefix
        df.at[idx, "objective_flag"] = 1

        x = float(df.at[idx, "x"])
        y = float(df.at[idx, "y"])
        chosen_centers.append((x, y))

        objects.append({
            "type": prefix,
            "label": label,
            "node_id": int(df.at[idx, "node_id"]),
            "x": x,
            "y": y,
            "z": float(df.at[idx, "z"]),
        })

        # Do not place another objective exactly on this node.
        available_mask[idx] = False

    return objects


def add_ra_objects(
    df: pd.DataFrame,
    width_m: float,
    height_m: float,
    rng: np.random.Generator,
    n_ra: int,
    min_radius_m: float,
    max_radius_m: float,
    chosen_centers: list[tuple[float, float]],
    min_dist_m: float,
    border_margin_m: float = 0.0,
) -> tuple[np.ndarray, list[dict]]:
    """
    Add circular restricted airspace objects.

    RA is treated as hard no-fly. border_margin_m keeps every RA disc clear of the
    border ring (its centre stays >= margin + max_radius from each edge).
    """
    n_nodes = len(df)
    ra_mask = np.zeros(n_nodes, dtype=bool)
    ra_objects: list[dict] = []

    # restrict candidate centres to the inner region so any RA radius up to
    # max_radius_m still leaves the border ring clear; fall back to all nodes if
    # the margin is so large that no inner node remains.
    xs = df["x"].to_numpy()
    ys = df["y"].to_numpy()
    clr = border_margin_m + max_radius_m
    inner = (xs >= clr) & (xs <= width_m - clr) & (ys >= clr) & (ys <= height_m - clr)
    dummy_available = inner.copy() if inner.any() else np.ones(n_nodes, dtype=bool)

    for i in range(1, n_ra + 1):
        idx = choose_free_center_node(
            df=df,
            available_mask=dummy_available,
            rng=rng,
            min_dist_m=min_dist_m,
            chosen_centers=chosen_centers,
        )

        cx = float(df.at[idx, "x"])
        cy = float(df.at[idx, "y"])
        radius = float(rng.uniform(min_radius_m, max_radius_m))

        d = distance_to_center(df, cx, cy)
        this_ra_mask = d <= radius

        ra_mask |= this_ra_mask

        label = f"RA{i:02d}"

        # Label only the center node as the RA objective point.
        df.at[idx, "label"] = label
        df.at[idx, "label_prefix"] = "RA"
        df.at[idx, "objective_flag"] = 1

        chosen_centers.append((cx, cy))
        dummy_available[idx] = False

        ra_objects.append({
            "type": "RA",
            "label": label,
            "center_node_id": int(df.at[idx, "node_id"]),
            "cx": cx,
            "cy": cy,
            "radius_m": radius,
            "affected_nodes": int(this_ra_mask.sum()),
        })

    return ra_mask, ra_objects


def compute_risk_and_slowness(
    df: pd.DataFrame,
    obstacle_mask: np.ndarray,
    ra_mask: np.ndarray,
    flyable_slowness: float,
    nofly_slowness: float,
) -> pd.DataFrame:
    """
    Update flags, risk, and slowness.
    """
    nofly_mask = obstacle_mask | ra_mask

    df["obstacle_flag"] = obstacle_mask.astype(int)
    df["ra_flag"] = ra_mask.astype(int)

    df["risk_obstacle"] = obstacle_mask.astype(float)
    df["risk_ra"] = ra_mask.astype(float)

    # Hard risk combination.
    # RA and obstacle are both treated as hard no-fly.
    df["risk_total"] = np.maximum(df["risk_obstacle"], df["risk_ra"])

    df["slowness"] = flyable_slowness
    df.loc[nofly_mask, "slowness"] = nofly_slowness

    # Force DB / DK / FLZ objective center nodes to remain flyable.
    # RA center remains no-fly because it is a restricted-area objective.
    force_flyable_prefixes = {"DB", "DK", "FLZ"}
    force_mask = df["label_prefix"].isin(force_flyable_prefixes)

    df.loc[force_mask, "obstacle_flag"] = 0
    df.loc[force_mask, "risk_obstacle"] = 0.0
    df.loc[force_mask, "risk_total"] = 0.0
    df.loc[force_mask, "slowness"] = flyable_slowness

    return df


def plot_map(
    df: pd.DataFrame,
    output_png: Path,
    width_m: float,
    height_m: float,
    title: str,
) -> None:
    """
    Plot generated 2D node map.
    """
    fig, ax = plt.subplots(figsize=(10, 9))

    fly_mask = df["slowness"].to_numpy() < DEFAULT_NOFLY_SLOWNESS
    obs_mask = df["obstacle_flag"].to_numpy() == 1
    ra_mask = df["ra_flag"].to_numpy() == 1

    ax.scatter(
        df.loc[fly_mask, "x"],
        df.loc[fly_mask, "y"],
        s=4,
        c="lightgray",
        label="Flyable nodes",
        linewidths=0,
    )

    ax.scatter(
        df.loc[obs_mask, "x"],
        df.loc[obs_mask, "y"],
        s=7,
        c="black",
        label="Obstacle",
        linewidths=0,
    )

    ax.scatter(
        df.loc[ra_mask, "x"],
        df.loc[ra_mask, "y"],
        s=7,
        c="red",
        label="RA / no-fly",
        linewidths=0,
    )

    marker_style = {
        "DB": ("^", "blue", 90),
        "DK": ("s", "green", 80),
        "FLZ": ("*", "orange", 140),
        "RA": ("X", "purple", 100),
    }

    for prefix, (marker, color, size) in marker_style.items():
        sub = df[df["label_prefix"] == prefix]
        if len(sub) == 0:
            continue

        ax.scatter(
            sub["x"],
            sub["y"],
            marker=marker,
            s=size,
            c=color,
            edgecolors="white",
            linewidths=0.8,
            label=prefix,
            zorder=10,
        )

        for _, row in sub.iterrows():
            ax.text(
                row["x"] + 0.01 * width_m,
                row["y"] + 0.01 * height_m,
                str(row["label"]),
                fontsize=8,
                color=color,
                weight="bold",
                zorder=11,
            )

    ax.set_xlim(-0.02 * width_m, width_m * 1.02)
    ax.set_ylim(-0.02 * height_m, height_m * 1.02)
    ax.set_aspect("equal", adjustable="box")
    add_map_rule(ax)
    ax.set_xlabel("X coordinate (m)")
    ax.set_ylabel("Y coordinate (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_png, dpi=250)
    plt.close(fig)


# ======================================================================
# Main
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate random 2D node-based LAE-UTM riskmap."
    )

    parser.add_argument("--width-m", type=float, default=5000.0)
    parser.add_argument("--height-m", type=float, default=5000.0)
    parser.add_argument("--dx-m", type=float, default=50.0)

    parser.add_argument(
        "--obstacle-rate",
        type=float,
        default=0.20,
        help="Target fraction of obstacle nodes, from 0.0 to 1.0.",
    )

    parser.add_argument(
        "--border-margin-m",
        type=float,
        default=DEFAULT_BORDER_MARGIN_M,
        help="Keep obstacles/RA clear of a ring this wide (m) along every map "
             "edge, leaving a routable outer strip so corridors can spread "
             "high-density flows around the boundary. 0 disables the ring.",
    )

    parser.add_argument("--n-db", type=int, default=2)
    parser.add_argument("--n-dk", type=int, default=6)
    parser.add_argument("--n-flz", type=int, default=4)
    parser.add_argument("--n-ra", type=int, default=3)

    parser.add_argument(
        "--objective-min-dist-m",
        type=float,
        default=400.0,
        help="Minimum distance between objective centers.",
    )

    parser.add_argument(
        "--obstacle-min-radius-m",
        type=float,
        default=80.0,
    )

    parser.add_argument(
        "--obstacle-max-radius-m",
        type=float,
        default=350.0,
    )

    parser.add_argument(
        "--ra-min-radius-m",
        type=float,
        default=250.0,
    )

    parser.add_argument(
        "--ra-max-radius-m",
        type=float,
        default=600.0,
    )

    parser.add_argument("--flyable-slowness", type=float, default=DEFAULT_FLYABLE_SLOWNESS)
    parser.add_argument("--nofly-slowness", type=float, default=DEFAULT_NOFLY_SLOWNESS)

    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/01_random_node_map",
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default="random_2d_node_riskmap",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not (0.0 <= args.obstacle_rate <= 1.0):
        raise ValueError("--obstacle-rate must be between 0.0 and 1.0.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    xyz_file = output_dir / f"{args.output_name}_seed{args.seed}.xyz"
    json_file = output_dir / f"{args.output_name}_seed{args.seed}_metadata.json"
    fig_file = output_dir / f"{args.output_name}_seed{args.seed}.png"

    rng = np.random.default_rng(args.seed)

    print("=" * 70)
    print("GENERATING RANDOM 2D NODE RISKMAP")
    print("=" * 70)
    print(f"Map size          : {args.width_m:.1f} m x {args.height_m:.1f} m")
    print(f"Grid spacing      : {args.dx_m:.1f} m")
    print(f"Obstacle rate     : {args.obstacle_rate:.3f}")
    print(f"Border margin     : {args.border_margin_m:.0f} m "
          f"(clear routable ring along every edge)")
    print(f"Random seed       : {args.seed}")
    print(f"Output directory  : {output_dir}")

    # ------------------------------------------------------------------
    # 1. Build base grid
    # ------------------------------------------------------------------
    df = make_grid(
        width_m=args.width_m,
        height_m=args.height_m,
        dx_m=args.dx_m,
    )

    df["label"] = LABEL_NONE
    df["label_prefix"] = PREFIX_NONE
    df["objective_flag"] = 0

    # ------------------------------------------------------------------
    # 2. Generate random obstacle mask
    # ------------------------------------------------------------------
    obstacle_mask, obstacle_objects = random_circle_obstacles(
        df=df,
        width_m=args.width_m,
        height_m=args.height_m,
        dx_m=args.dx_m,
        obstacle_rate=args.obstacle_rate,
        rng=rng,
        min_radius_m=args.obstacle_min_radius_m,
        max_radius_m=args.obstacle_max_radius_m,
        border_margin_m=args.border_margin_m,
    )

    # Free nodes for DB/DK/FLZ placement.
    available_mask = ~obstacle_mask.copy()

    # ------------------------------------------------------------------
    # 3. Add DB / DK / FLZ objectives
    # ------------------------------------------------------------------
    chosen_centers: list[tuple[float, float]] = []

    db_objects = add_point_objectives(
        df=df,
        available_mask=available_mask,
        rng=rng,
        prefix="DB",
        count=args.n_db,
        chosen_centers=chosen_centers,
        min_dist_m=args.objective_min_dist_m,
    )

    dk_objects = add_point_objectives(
        df=df,
        available_mask=available_mask,
        rng=rng,
        prefix="DK",
        count=args.n_dk,
        chosen_centers=chosen_centers,
        min_dist_m=args.objective_min_dist_m,
    )

    flz_objects = add_point_objectives(
        df=df,
        available_mask=available_mask,
        rng=rng,
        prefix="FLZ",
        count=args.n_flz,
        chosen_centers=chosen_centers,
        min_dist_m=args.objective_min_dist_m,
    )

    # ------------------------------------------------------------------
    # 4. Add RA objects
    # ------------------------------------------------------------------
    ra_mask, ra_objects = add_ra_objects(
        df=df,
        width_m=args.width_m,
        height_m=args.height_m,
        rng=rng,
        n_ra=args.n_ra,
        min_radius_m=args.ra_min_radius_m,
        max_radius_m=args.ra_max_radius_m,
        chosen_centers=chosen_centers,
        min_dist_m=args.objective_min_dist_m,
        border_margin_m=args.border_margin_m,
    )

    # ------------------------------------------------------------------
    # 5. Compute risk and slowness
    # ------------------------------------------------------------------
    df = compute_risk_and_slowness(
        df=df,
        obstacle_mask=obstacle_mask,
        ra_mask=ra_mask,
        flyable_slowness=args.flyable_slowness,
        nofly_slowness=args.nofly_slowness,
    )

    # Reorder columns.
    output_columns = [
        "node_id",
        "x",
        "y",
        "z",
        "slowness",
        "risk_obstacle",
        "risk_ra",
        "risk_total",
        "obstacle_flag",
        "ra_flag",
        "objective_flag",
        "label",
        "label_prefix",
    ]

    df = df[output_columns]

    # ------------------------------------------------------------------
    # 6. Save XYZ-like node file
    # ------------------------------------------------------------------
    df.to_csv(
        xyz_file,
        sep=" ",
        index=False,
        float_format="%.6f",
    )

    # ------------------------------------------------------------------
    # 7. Save metadata
    # ------------------------------------------------------------------
    metadata = {
        "description": "Random 2D node-based LAE-UTM riskmap",
        "seed": args.seed,
        "width_m": args.width_m,
        "height_m": args.height_m,
        "dx_m": args.dx_m,
        "n_nodes": int(len(df)),
        "obstacle_rate_target": float(args.obstacle_rate),
        "obstacle_rate_actual": float(df["obstacle_flag"].mean()),
        "ra_rate_actual": float(df["ra_flag"].mean()),
        "nofly_rate_actual": float((df["slowness"] >= args.nofly_slowness).mean()),
        "flyable_slowness": args.flyable_slowness,
        "nofly_slowness": args.nofly_slowness,
        "nofly_rule": f"slowness >= {args.nofly_slowness}",
        "objects": {
            "DB": db_objects,
            "DK": dk_objects,
            "FLZ": flz_objects,
            "RA": ra_objects,
            "obstacles": obstacle_objects,
        },
        "output_xyz": str(xyz_file),
        "output_figure": str(fig_file),
    }

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 8. Save quick-look figure
    # ------------------------------------------------------------------
    title = (
        f"Random 2D Node Riskmap | seed={args.seed} | "
        f"obstacle={df['obstacle_flag'].mean():.2f} | "
        f"RA={df['ra_flag'].mean():.2f}"
    )

    plot_map(
        df=df,
        output_png=fig_file,
        width_m=args.width_m,
        height_m=args.height_m,
        title=title,
    )

    # ------------------------------------------------------------------
    # 9. Report
    # ------------------------------------------------------------------
    print("-" * 70)
    print(f"Total nodes        : {len(df):,}")
    print(f"Obstacle nodes     : {int(df['obstacle_flag'].sum()):,}")
    print(f"RA nodes           : {int(df['ra_flag'].sum()):,}")
    print(f"No-fly nodes       : {int((df['slowness'] >= args.nofly_slowness).sum()):,}")
    print(f"DB count           : {args.n_db}")
    print(f"DK count           : {args.n_dk}")
    print(f"FLZ count          : {args.n_flz}")
    print(f"RA count           : {args.n_ra}")
    print("-" * 70)
    print(f"Saved XYZ          : {xyz_file}")
    print(f"Saved metadata     : {json_file}")
    print(f"Saved figure       : {fig_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()