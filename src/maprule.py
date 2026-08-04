#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/maprule.py -- a shared "map rule" for every figure in the pipeline.

add_map_rule(ax) stamps two things onto a metric (x/y in metres) matplotlib
axis so every plot is self-describing at a glance:

    * a SCALE BAR (ruler) of a round length (1/2/5 x 10^k) ~ 1/5 of the map
      width, labelled in m or km;
    * a MAP-SIZE label ("map W x H m") in a corner.

Import from any of the 01..10 scripts:

    from src.maprule import add_map_rule
    ...
    add_map_rule(ax)                      # infers extent from ax limits
    add_map_rule(ax, 0, 0, 5000, 5000)   # or pass the extent explicitly
"""
from __future__ import annotations

import math

import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


def _nice_length(v: float) -> float:
    """Largest 1/2/5 x 10^k not exceeding v (a tidy scale-bar length)."""
    if v <= 0:
        return 1.0
    k = math.floor(math.log10(v))
    base = v / (10 ** k)
    mult = 5 if base >= 5 else 2 if base >= 2 else 1
    return mult * (10 ** k)


def add_map_rule(ax, x0=None, y0=None, x1=None, y1=None, *,
                 loc="lower right", color="black", show_size=True,
                 frac=0.2):
    """Add a scale bar + map-size label to ``ax`` (data units = metres).

    Extent defaults to the current axis limits; pass x0,y0,x1,y1 to force it.
    Returns the AnchoredSizeBar artist.
    """
    if x0 is None or x1 is None:
        x0, x1 = ax.get_xlim()
    if y0 is None or y1 is None:
        y0, y1 = ax.get_ylim()
    w = abs(x1 - x0)
    h = abs(y1 - y0)
    L = _nice_length(w * frac)
    label = f"{L / 1000:g} km" if L >= 1000 else f"{L:g} m"

    bar = AnchoredSizeBar(
        ax.transData, L, label, loc,
        pad=0.4, borderpad=0.5, sep=4, frameon=True,
        size_vertical=max(h * 0.005, w * 0.0015),
        color=color, fontproperties=fm.FontProperties(size=9),
    )
    bar.patch.set(alpha=0.75, edgecolor="none")
    ax.add_artist(bar)

    if show_size:
        ax.text(0.01, 0.99, f"map {w:.0f} × {h:.0f} m",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=8, color=color,
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="none", alpha=0.7))
    return bar
