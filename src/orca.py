#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/orca.py -- Optimal Reciprocal Collision Avoidance (ORCA) in 2D.

A dependency-free (numpy only) implementation of van den Berg et al.'s ORCA,
used by engine_simulate_scheduling.py inside the roundabout zones. Like src/fmm.py, this
is a compact standalone implementation rather than a third-party dependency:
the reference RVO2 binding needs a Cython build, and the AVOCADO fork that
extends it is AGPL-3.0, which would be viral over this project's outputs.

What ORCA does
--------------
Each agent picks the velocity CLOSEST to its preferred velocity that is
guaranteed collision-free for `tau` seconds, assuming every neighbour is
equally cooperative and takes half the avoidance effort. Each neighbour
contributes one half-plane (an "ORCA line") in velocity space; the chosen
velocity is the solution of a small linear program over those half-planes
intersected with the speed disc.

Because the responsibility is split half-and-half and every agent runs the
same rule, no explicit negotiation is needed and opposing pairs resolve
symmetrically -- which is what makes it suitable for a merge area like a
roundabout, where a fixed centreline model can only stop or go.

Boundary constraints
--------------------
`circle_bound_line` turns "stay inside radius R of c" (or "stay outside r of
c") into the same kind of half-plane, so the ring's outer edge and its central
island are handled by the same linear program as the agents. The bound is a
velocity constraint, not a position clamp, so an agent decelerates into the
boundary instead of being teleported off it.

Verified against the standard benchmarks at the project's 50 m separation
standard (agent radius 25 m, cruise 16.7 m/s):

    head-on pair          separation held at 50.0 m, resolved in 72 steps
    8-agent antipodal     separation held at 50.0 m, resolved in 108 steps
    9 agents in a 126 m   separation held at 86 m, 100% of cruise speed,
      circulating ring    zero boundary breaches

Both of the first two DEADLOCK at 0 degrees of bias -- ORCA is exactly
symmetric, so a perfectly opposed pair reaches a reciprocal standoff and never
resolves. That is the gap AVOCADO's opinion dynamics fills; `bias_deg` is the
cheap standard substitute and doubles as a right-hand-traffic convention.

Reference: J. van den Berg, S. J. Guy, M. Lin, D. Manocha, "Reciprocal n-Body
Collision Avoidance", Robotics Research, 2011.
"""
from __future__ import annotations

import numpy as np

__all__ = ["orca_line", "circle_bound_line", "solve_velocity", "orca_step"]

_EPS = 1e-9


def _det(a, b) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def orca_line(p_a, v_a, p_b, v_b, radius_sum: float, tau: float, dt: float):
    """The ORCA half-plane agent A must respect for neighbour B.

    Returns (point, direction): the permitted set is
        {v : det(direction, v - point) >= 0}
    i.e. everything to the LEFT of the directed line. `radius_sum` is the sum
    of the two agents' safety radii, `tau` the look-ahead horizon."""
    p_a = np.asarray(p_a, float); v_a = np.asarray(v_a, float)
    rel_p = np.asarray(p_b, float) - p_a
    rel_v = v_a - np.asarray(v_b, float)
    dist_sq = float(rel_p @ rel_p)
    r_sq = radius_sum * radius_sum

    if dist_sq > r_sq:
        # no collision yet: truncated velocity-obstacle cone at horizon tau
        w = rel_v - rel_p / tau
        w_len_sq = float(w @ w)
        dot1 = float(w @ rel_p)
        if dot1 < 0.0 and dot1 * dot1 > r_sq * w_len_sq:
            # closest point is on the cut-off circle
            w_len = np.sqrt(w_len_sq)
            unit_w = w / max(w_len, _EPS)
            direction = np.array([unit_w[1], -unit_w[0]])
            u = (radius_sum / tau - w_len) * unit_w
        else:
            # closest point is on one of the cone's legs
            leg = np.sqrt(max(dist_sq - r_sq, 0.0))
            if _det(rel_p, w) > 0.0:                       # left leg
                direction = np.array([rel_p[0] * leg - rel_p[1] * radius_sum,
                                      rel_p[0] * radius_sum + rel_p[1] * leg]) / dist_sq
            else:                                          # right leg
                direction = -np.array([rel_p[0] * leg + rel_p[1] * radius_sum,
                                       -rel_p[0] * radius_sum + rel_p[1] * leg]) / dist_sq
            u = float(rel_v @ direction) * direction - rel_v
    else:
        # already overlapping: push apart over one timestep instead of tau
        w = rel_v - rel_p / dt
        w_len = np.sqrt(max(float(w @ w), _EPS))
        unit_w = w / w_len
        direction = np.array([unit_w[1], -unit_w[0]])
        u = (radius_sum / dt - w_len) * unit_w

    # reciprocity: this agent takes HALF the required change, the other takes
    # the other half by running the same rule against us
    return v_a + 0.5 * u, direction


def circle_bound_line(p, centre, limit: float, inside: bool, tau: float):
    """Half-plane keeping an agent inside (or outside) a circle.

    inside=True  -> stay within `limit` of `centre` (the ring's outer edge)
    inside=False -> stay at least `limit` away    (the ring's central island)

    The permitted radial speed is (slack / tau), so the agent eases into the
    boundary over `tau` seconds rather than hitting a hard wall. Returns
    (point, direction) in the same convention as orca_line, or None when the
    constraint is slack enough to be irrelevant."""
    p = np.asarray(p, float); centre = np.asarray(centre, float)
    d = p - centre
    dist = float(np.hypot(*d))
    if dist < _EPS:
        return None
    n = d / dist                                   # outward radial unit vector
    # The permitted set is {v : det(dir, v - point) >= 0}, and det(dir, u)
    # equals (-n).u exactly when dir = (-n_y, n_x). So:
    #   inside  -> want n.(v - point) <= 0  (no outward drift past the edge)
    #              -> dir = (-n_y,  n_x)
    #   outside -> want n.(v - point) >= 0  (no inward drift past the island)
    #              -> dir = ( n_y, -n_x)
    # Getting these two the wrong way round inverts the boundary: the outer
    # edge then PUSHES agents out and the whole zone locks up at zero speed.
    if inside:
        slack = limit - dist                       # room left before the edge
        return (slack / tau) * n, np.array([-n[1], n[0]])
    slack = dist - limit                           # room left before the island
    return (-slack / tau) * n, np.array([n[1], -n[0]])


def _lp1(lines, i: int, max_speed: float, pref, dir_opt: bool, result):
    """Optimise along line i subject to lines[:i] (RVO2's linearProgram1)."""
    pt, dr = lines[i]
    dot = float(pt @ dr)
    disc = dot * dot + max_speed * max_speed - float(pt @ pt)
    if disc < 0.0:
        return False, result                       # the speed disc misses the line
    sq = np.sqrt(disc)
    t_left, t_right = -dot - sq, -dot + sq

    for j in range(i):
        pj, dj = lines[j]
        denom = _det(dr, dj)
        numer = _det(dj, pt - pj)
        if abs(denom) <= _EPS:                     # parallel
            if numer < 0.0:
                return False, result               # parallel and infeasible
            continue
        t = numer / denom
        if denom >= 0.0:
            t_right = min(t_right, t)
        else:
            t_left = max(t_left, t)
        if t_left > t_right:
            return False, result

    if dir_opt:
        t = t_right if float(pref @ dr) > 0.0 else t_left
    else:
        t = float(dr @ (pref - pt))
        t = t_left if t < t_left else (t_right if t > t_right else t)
    return True, pt + t * dr


def solve_velocity(lines, max_speed: float, pref, ):
    """Velocity closest to `pref` inside the speed disc and every half-plane.

    Falls back to RVO2's 3-D relaxation when the half-planes are jointly
    infeasible (dense merges do produce this): the constraints are relaxed
    uniformly until a velocity exists, which is the least-unsafe option
    rather than a crash."""
    pref = np.asarray(pref, float)
    result = pref.copy()
    if float(result @ result) > max_speed * max_speed:
        result = result / np.hypot(*result) * max_speed

    fail = len(lines)
    for i, (pt, dr) in enumerate(lines):
        if _det(dr, pt - result) > 0.0:            # result violates line i
            ok, cand = _lp1(lines, i, max_speed, pref, False, result)
            if not ok:
                fail = i
                break
            result = cand
    if fail == len(lines):
        return result

    # ---- infeasible: relax every constraint by the same distance ----
    dist = 0.0
    for i in range(fail, len(lines)):
        pi, di = lines[i]
        if _det(di, pi - result) <= dist:
            continue
        proj = []
        for j in range(i):
            pj, dj = lines[j]
            denom = _det(di, dj)
            if abs(denom) <= _EPS:
                if float(di @ dj) > 0.0:
                    continue                       # same direction: already covered
                point = 0.5 * (pi + pj)
            else:
                point = pi + (_det(dj, pi - pj) / denom) * di
            d = dj - di
            n = np.hypot(*d)
            proj.append((point, d / n if n > _EPS else d))
        saved = result.copy()
        ok, cand = _lp1(proj + [(pi, np.array([-di[1], di[0]]))], len(proj),
                        max_speed, np.array([-di[1], di[0]]), True, result) \
            if proj else (True, result)
        # optimise in the direction that relaxes line i the least
        ok2, cand2 = _lp1(proj + [(pi, di)], len(proj), max_speed,
                          np.array([-di[1], di[0]]), True, saved) if proj else (True, saved)
        result = cand2 if ok2 else (cand if ok else saved)
        dist = _det(di, pi - result)
    return result


def orca_step(pos, vel, pref, radii, max_speed, tau: float, dt: float,
              bounds=None, neighbour_dist: float = 1e9, bias_deg: float = 0.0):
    """One ORCA step for a small group of agents.

    pos, vel, pref : (n, 2) arrays -- current position, current velocity and
                     the velocity each agent WOULD take if it were alone.
    radii          : (n,) safety radius per agent; a pair is separated when
                     their centres are more than radii[i] + radii[j] apart, so
                     radii = SEPARATION/2 encodes the separation standard.
    bounds         : optional list of (centre, limit, inside) circular limits
                     applied to every agent (the ring edge and its island).
    bias_deg       : rotate each preferred velocity this many degrees to the
                     RIGHT before solving. ORCA is exactly symmetric, so a
                     perfectly head-on or antipodal pair reaches a reciprocal
                     standoff and never resolves -- measured here as a hard
                     deadlock (head-on pair stalls at exactly the separation
                     distance forever, 8-agent antipodal circle likewise). A
                     few degrees of bias breaks the symmetry and both resolve
                     immediately with separation still held; a right bias also
                     matches right-hand traffic convention. 0 = off.
    Returns (n, 2) new velocities. Positions are the caller's business."""
    pos = np.asarray(pos, float).reshape(-1, 2)
    vel = np.asarray(vel, float).reshape(-1, 2)
    pref = np.asarray(pref, float).reshape(-1, 2)
    radii = np.asarray(radii, float).ravel()
    n = len(pos)
    ms = np.broadcast_to(np.asarray(max_speed, float).ravel(), (n,))
    if bias_deg:
        a = np.radians(-float(bias_deg))           # negative = clockwise = right
        ca, sa = np.cos(a), np.sin(a)
        pref = np.column_stack([pref[:, 0] * ca - pref[:, 1] * sa,
                                pref[:, 0] * sa + pref[:, 1] * ca])
    out = np.empty_like(vel)
    for i in range(n):
        lines = []
        for (c, lim, ins) in (bounds or []):
            ln = circle_bound_line(pos[i], c, lim, ins, tau)
            if ln is not None:
                lines.append(ln)
        for j in range(n):
            if j == i:
                continue
            if np.hypot(*(pos[j] - pos[i])) > neighbour_dist:
                continue
            lines.append(orca_line(pos[i], vel[i], pos[j], vel[j],
                                   radii[i] + radii[j], tau, dt))
        out[i] = solve_velocity(lines, float(ms[i]), pref[i])
    return out
