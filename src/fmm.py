#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/fmm.py -- Fast Marching Method (FMM) on a 2D cost grid.

A dependency-free (numpy + heapq) first-order Eikonal solver used by
10_fmm_route_plan.py.  Given a per-cell IMPEDANCE field ``cost`` (the
"slowness" in the Eikonal sense -- higher = more expensive to cross) and a
set of source cells, it computes the minimum accumulated cost T(x) from the
nearest source to every cell.  Routes are then recovered by steepest descent
on T (``backtrack``), which yields the cost-minimising path back to a source.

The accumulated field solves   |grad T| = cost ,   T(source) = 0.
With ``cost`` a weighted blend of travel-time, risk and traffic-conflict, the
descent path minimises that weighted blend -- i.e. a risk/conflict-aware route.

skfmm is not required (and is not installed in this project's env); this is a
compact standalone implementation.
"""
from __future__ import annotations

import heapq

import numpy as np

_FAR, _TRIAL, _KNOWN = 0, 1, 2


def _eikonal_update(T, cost, i, j, dx, ny, nx):
    """First-order Godunov solution of the Eikonal equation at cell (i, j).
    ``cost[i, j]`` is the local slowness/impedance; f = cost*dx is the price
    of crossing one cell."""
    f = cost[i, j] * dx
    if not np.isfinite(f):
        return np.inf
    tx = min(T[i, j - 1] if j > 0 else np.inf,
             T[i, j + 1] if j < nx - 1 else np.inf)
    ty = min(T[i - 1, j] if i > 0 else np.inf,
             T[i + 1, j] if i < ny - 1 else np.inf)
    vals = sorted(v for v in (tx, ty) if np.isfinite(v))
    if not vals:
        return np.inf
    if len(vals) == 1:
        return vals[0] + f
    a, b = vals            # a <= b
    if (b - a) >= f:       # 1-D update wins (upwind only from the smaller)
        return a + f
    # 2-D update: solve (T-a)^2 + (T-b)^2 = f^2
    disc = 2.0 * f * f - (b - a) ** 2
    return 0.5 * (a + b + np.sqrt(max(disc, 0.0)))


def eikonal_fmm(cost: np.ndarray, source_mask: np.ndarray, dx: float) -> np.ndarray:
    """Accumulated-cost field T[ny, nx] from the sources.

    Parameters
    ----------
    cost : (ny, nx) float
        Per-cell impedance (>0). Use ``np.inf`` for impassable cells.
    source_mask : (ny, nx) bool
        True at the source cell(s); these get T = 0.
    dx : float
        Grid spacing (metres). T is returned in the same weighted units as
        ``cost * dx`` (i.e. a cost-distance, not necessarily seconds).
    """
    cost = np.asarray(cost, float)
    ny, nx = cost.shape
    T = np.full((ny, nx), np.inf)
    state = np.zeros((ny, nx), np.uint8)
    passable = np.isfinite(cost) & (cost > 0)
    heap: list[tuple[float, int, int]] = []

    for i, j in np.argwhere(source_mask & passable):
        T[i, j] = 0.0
        state[i, j] = _KNOWN

    def _push_neighbours(i, j):
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < ny and 0 <= b < nx and passable[a, b] and state[a, b] != _KNOWN:
                t = _eikonal_update(T, cost, a, b, dx, ny, nx)
                if t < T[a, b]:
                    T[a, b] = t
                    state[a, b] = _TRIAL
                    heapq.heappush(heap, (t, a, b))

    for i, j in np.argwhere(source_mask & passable):
        _push_neighbours(i, j)

    while heap:
        t, i, j = heapq.heappop(heap)
        if state[i, j] == _KNOWN:
            continue
        state[i, j] = _KNOWN
        T[i, j] = t
        _push_neighbours(i, j)
    return T


def backtrack(T: np.ndarray, start_ij: tuple[int, int],
              max_steps: int | None = None) -> list[tuple[int, int]]:
    """Steepest-descent route (list of (i, j) cells) from ``start_ij`` down T
    to a source (T == 0). 8-connected greedy descent: robust because an FMM
    field has no interior local minima away from the sources. Returns [] if
    the start is unreachable (T not finite)."""
    ny, nx = T.shape
    i, j = start_ij
    if not np.isfinite(T[i, j]):
        return []
    if max_steps is None:
        max_steps = 4 * ny * nx
    path = [(i, j)]
    seen = {(i, j)}
    for _ in range(max_steps):
        if T[i, j] <= 0.0:
            break
        best, best_t = None, T[i, j]
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                a, b = i + di, j + dj
                if 0 <= a < ny and 0 <= b < nx and np.isfinite(T[a, b]) and T[a, b] < best_t:
                    best, best_t = (a, b), T[a, b]
        if best is None or best in seen:
            break
        i, j = best
        seen.add(best)
        path.append(best)
    return path
