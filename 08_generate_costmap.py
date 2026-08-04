#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_generate_costmap.py

Auto-generate a SLOWNESS cost-map from the traffic density produced by the
agent simulation (09_simulate_agents_2d.py). The map is INDEPENDENT of the
routes -- it does not change where agents fly, only how fast they go:

    true velocity = slowness(x, y) * base drone velocity

Higher traffic density -> lower slowness (slower). The density is smoothed
with a GAUSSIAN kernel, then mapped to a slowness field clamped to
[MIN_SLOWNESS, MAX_SLOWNESS] = [1e-2, 1].

Corridor confinement
--------------------
Only cells INSIDE a corridor keep their density-derived slowness. A cell is
"inside" when it lies within CORRIDOR_HALF_WIDTH metres of any lane centre-
line (from 07's lane_nodes.csv). Everything OUTSIDE the corridor network is
forced to OUTSIDE_SLOWNESS = 1e-2 (a crawl), so agents that stray off the
corridors are slowed to a near halt and effectively cannot fly out of the
map. The corridors themselves are unaffected.

Pipeline
--------
    09_simulate_agents_2d.py   (run once, base speeds)  -> traffic samples
    08_generate_costmap.py     (this)                   -> slowness cost-map
    09_simulate_agents_2d.py   (re-run)                 -> speeds modulated

Run
---
    python 08_generate_costmap.py
    python 08_generate_costmap.py --traffic output/09_agent_sim_2d/trajectories.csv \\
        --res 40 --sigma 150 --min-slowness 0.01 --corridor-half-width 60

Outputs (output/08_costmap/)
----------------------------
    slowness_costmap.npz   grid: slowness[ny,nx], x0, y0, res
    slowness_costmap.png   the field with the corridor network overlaid
    density.png            the smoothed traffic density it was built from
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from src.maprule import add_map_rule

THIS_DIR = Path(__file__).resolve().parent
VERSION = "v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Traffic-density slowness cost-map.")
    p.add_argument("--traffic", default="output/09_agent_sim_2d/trajectories.csv",
                   help="agent position samples (t_s,agent_id,x,y,...) from the sim")
    p.add_argument("--corridor-dir", default="output/07_corridor_network")
    p.add_argument("--out-dir", default="output/08_costmap")
    p.add_argument("--res", type=float, default=40.0, help="grid cell size (m)")
    p.add_argument("--sigma", type=float, default=150.0, help="Gaussian smoothing sigma (m)")
    p.add_argument("--min-slowness", type=float, default=1e-2)
    p.add_argument("--max-slowness", type=float, default=1.0)
    p.add_argument("--hold-weight", type=float, default=3.0,
                   help="extra weight for holding (jammed) samples")
    p.add_argument("--corridor-half-width", type=float, default=60.0,
                   help="cells within this distance (m) of a lane centreline are "
                        "'inside' a corridor and keep their density slowness")
    p.add_argument("--outside-slowness", type=float, default=1e-2,
                   help="slowness forced on cells outside every corridor "
                        "(low = agents crawl and cannot leave the network)")
    return p.parse_args()


def corridor_mask(lanes: pd.DataFrame, xc: np.ndarray, yc: np.ndarray,
                  half_width: float) -> np.ndarray:
    """Boolean grid[ny,nx]: True where a cell centre lies within `half_width`
    metres of any lane centreline. Lane polylines are densely resampled so a
    nearest-point KD-tree query approximates the true distance-to-polyline."""
    step = max(2.0, half_width / 3.0)
    pts = []
    for (leg_id, lane), g in lanes.groupby(["leg_id", "lane"]):
        xy = g.sort_values("seq")[["x", "y"]].to_numpy(float)
        if len(xy) < 1:
            continue
        if len(xy) == 1:
            pts.append(xy)
            continue
        seg = np.hypot(*np.diff(xy, axis=0).T)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        n = max(2, int(np.ceil(s[-1] / step)) + 1)
        t = np.linspace(0.0, float(s[-1]), n)
        pts.append(np.column_stack([np.interp(t, s, xy[:, 0]),
                                    np.interp(t, s, xy[:, 1])]))
    lane_pts = np.vstack(pts)
    tree = cKDTree(lane_pts)
    gx, gy = np.meshgrid(xc, yc)                       # [ny,nx]
    grid_pts = np.column_stack([gx.ravel(), gy.ravel()])
    dist, _ = tree.query(grid_pts, k=1)
    return (dist <= half_width).reshape(gx.shape)


def draw_network(ax, corridor_dir: Path):
    """Light overlay of lane centrelines + objectives, for context."""
    lanes = pd.read_csv(corridor_dir / "lane_nodes.csv")
    for leg_id in lanes["leg_id"].unique():
        for lane in ("A", "B"):
            g = lanes[(lanes["leg_id"] == leg_id) & (lanes["lane"] == lane)].sort_values("seq")
            xy = g[["x", "y"]].to_numpy(float)
            if len(xy) >= 2:
                ax.plot(xy[:, 0], xy[:, 1], "-", color="0.5", lw=0.6, alpha=0.7, zorder=5)
    nodes = pd.read_csv(corridor_dir / "network_nodes.csv")
    obj = nodes[nodes["kind"] == "objective"]
    for r in obj.itertuples():
        is_db = str(r.net_id).startswith("DB")
        ax.scatter([r.x], [r.y], s=70, marker="s" if is_db else "^",
                   c="#c0392b" if is_db else "#1f6f3f", edgecolors="k",
                   linewidths=0.5, zorder=7)
        ax.annotate(str(r.net_id), (r.x, r.y), textcoords="offset points",
                    xytext=(5, 4), fontsize=7, weight="bold", zorder=8)


def main() -> None:
    args = parse_args()
    corridor_dir = THIS_DIR / args.corridor_dir
    out_dir = THIS_DIR / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 66)
    print(f"08_generate_costmap.py  {VERSION}")
    print(f"Traffic       : {args.traffic}")
    print(f"Output dir    : {out_dir}")
    print("=" * 66)

    # ---- bounding box from the corridor network ----
    nodes = pd.read_csv(corridor_dir / "network_nodes.csv")
    pad = 5 * args.res
    x0 = float(nodes["x"].min()) - pad
    y0 = float(nodes["y"].min()) - pad
    x1 = float(nodes["x"].max()) + pad
    y1 = float(nodes["y"].max()) + pad
    nx = int(np.ceil((x1 - x0) / args.res))
    ny = int(np.ceil((y1 - y0) / args.res))
    xedges = x0 + args.res * np.arange(nx + 1)
    yedges = y0 + args.res * np.arange(ny + 1)

    # ---- traffic density from the agent position samples ----
    traf = pd.read_csv(THIS_DIR / args.traffic)
    w = np.ones(len(traf))
    if "holding" in traf.columns and args.hold_weight != 1.0:
        w = np.where(traf["holding"].astype(bool), args.hold_weight, 1.0)
    H, _, _ = np.histogram2d(traf["x"].to_numpy(float), traf["y"].to_numpy(float),
                             bins=[xedges, yedges], weights=w)
    density = H.T   # -> density[iy, ix]
    print(f"Grid          : {nx} x {ny} cells @ {args.res:.0f} m "
          f"({len(traf)} traffic samples)")

    # ---- Gaussian smoothing -> smooth density field ----
    sigma_cells = args.sigma / args.res
    dens_s = gaussian_filter(density, sigma=sigma_cells)

    # ---- map density -> slowness in [min, max]; higher density = slower ----
    dmax = float(dens_s.max()) or 1.0
    dn = dens_s / dmax                       # normalised 0..1
    slowness = args.max_slowness - dn * (args.max_slowness - args.min_slowness)
    slowness = np.clip(slowness, args.min_slowness, args.max_slowness)

    # ---- confine to corridors: keep density slowness INSIDE, crawl OUTSIDE ----
    lanes = pd.read_csv(corridor_dir / "lane_nodes.csv")
    xc = x0 + args.res * (np.arange(nx) + 0.5)
    yc = y0 + args.res * (np.arange(ny) + 0.5)
    inside = corridor_mask(lanes, xc, yc, args.corridor_half_width)   # [ny,nx]
    slowness = np.where(inside, slowness, args.outside_slowness)
    frac_in = float(inside.mean())
    print(f"Corridor mask : half-width {args.corridor_half_width:.0f} m, "
          f"{frac_in*100:.1f}% of cells inside; outside slowness "
          f"{args.outside_slowness:.2f}")

    np.savez(out_dir / "slowness_costmap.npz",
             slowness=slowness.astype(np.float32),
             x0=x0, y0=y0, res=args.res)
    print(f"Slowness      : range [{slowness.min():.3f}, {slowness.max():.3f}] "
          f"(1.0 = full speed, {args.min_slowness:.2f} = slowest); "
          f"Gaussian sigma {args.sigma:.0f} m")
    print(f"Saved         : {out_dir/'slowness_costmap.npz'}")

    extent = [x0, x0 + nx * args.res, y0, y0 + ny * args.res]
    # ---- slowness map figure ----
    fig, ax = plt.subplots(figsize=(12, 11))
    im = ax.imshow(slowness, origin="lower", extent=extent, cmap="RdYlGn",
                   vmin=args.min_slowness, vmax=args.max_slowness, zorder=1)
    ax.contour(xc, yc, inside.astype(float), levels=[0.5],
               colors="k", linewidths=0.6, linestyles="--", zorder=6)
    draw_network(ax, corridor_dir)
    cb = fig.colorbar(im, ax=ax, shrink=0.75)
    cb.set_label("slowness  (1 = full speed, low = congested/slow)")
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("Traffic-density slowness cost-map  (true velocity = slowness x base)")
    add_map_rule(ax, extent[0], extent[2], extent[1], extent[3])
    fig.tight_layout(); fig.savefig(out_dir / "slowness_costmap.png", dpi=130)
    plt.close(fig)

    # ---- smoothed density figure ----
    fig, ax = plt.subplots(figsize=(12, 11))
    im = ax.imshow(dens_s, origin="lower", extent=extent, cmap="inferno", zorder=1)
    draw_network(ax, corridor_dir)
    fig.colorbar(im, ax=ax, shrink=0.75).set_label("smoothed traffic density")
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("Gaussian-smoothed traffic density")
    add_map_rule(ax, extent[0], extent[2], extent[1], extent[3])
    fig.tight_layout(); fig.savefig(out_dir / "density.png", dpi=130)
    plt.close(fig)
    print(f"Done. Figures in {out_dir}")


if __name__ == "__main__":
    main()
