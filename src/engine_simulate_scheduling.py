#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
engine_simulate_scheduling.py

Multi-agent 2D corridor traffic simulation on the shared UAV network
built by 07_build_corridor_network.py.

A fleet of N agents fly missions between the DB (depot) and DK
(delivery) objectives over a working shift (default 100 agents, 8 h).
Each mission randomly picks one DB and one DK, is randomly a round trip
(DB->DK->DB) or a one-way hop, and departs at a random time inside the
launch window. Every mission rides the precomputed corridor route
(pair_routes.csv) leg by leg, following the two-lane centreline geometry
(lane_nodes.csv).

Separation standard
-------------------
Two halves, declared in the params and measured separately:

  HORIZONTAL   SEPARATION_M metres between any two agents at the same
               flight level, whatever corridor each is on (50 m). This
               is also the rule stage 05 builds to -- the two lane
               centrelines of every leg are kept >= 50 m apart -- so a
               compliant pair passes on its own geometry.

  LONGITUDINAL TIME_HEADWAY_S seconds in trail on a shared lane, i.e.
               between agents travelling the SAME corridor (30 s). The
               distance gap therefore scales with speed (250 m at
               30 km/h, 500 m at 60 km/h) and never drops below the
               horizontal minimum.

Compliance is reported, not assumed: metrics.json carries every
same-level pair-sample checked and every one that came closer than the
standard, and separation_violations.csv says WHERE each loss happened.

Coordination -- scheduling first, tactical second
-------------------------------------------------
  STRATEGIC  schedule_departures() assigns every mission a CTOT before
             the run: departures on one first leg-lane are spaced by the
             in-trail headway, predicted airborne stays under
             MAX_CONCURRENT, and no origin exceeds its fair share. The
             fleet is deconflicted on paper at the one point the
             operator controls -- the departure time (SCHEDULE_MODE).

  SPATIAL    Every leg carries two parallel lanes and each travel
             direction takes its own, so opposing traffic occupies
             physically separated lanes and disjoint resources. Legs are
             additionally split into flight levels by heading, so
             crossing traffic is separated vertically.

  ROUNDABOUTS A route through a ring node gets a circulating ARC spliced
             in (RING_TRAVEL). Circulation is one-way COUNTER-CLOCKWISE
             (RING_RIGHT_HAND), which keeps the island on the left and
             therefore puts every exit on the RIGHT of the direction of
             travel. Legs are clipped at the ring boundary, so an agent
             joins and leaves the circulation on the ring itself and
             never crosses the enclosed disk.

  DOCK STOP  A round trip PARKS at the destination dock for MIN_DEST_IDLE_S
             and recharges to CHARGE_TARGET_PCT before flying home. A parked
             drone is not airborne traffic: it holds no corridor, draws no
             hover power, is exempt from the separation and conflict checks
             (it is outside the airborne set by construction), and does not
             consume the MAX_CONCURRENT airborne budget. metrics.json reports
             how many agent-samples that exemption covered.

  TACTICAL   On a shared lane a follower never closes inside the headway
             (car-following). To enter the next leg an agent must find
             its entry slot clear -- otherwise it HOLDS at the upstream
             node. This is the safety net under the schedule, not the
             primary coordination.

The clock, positions and the live minimum pairwise separation are shared
by every output.

Run
---
    python engine_simulate_scheduling.py
    python engine_simulate_scheduling.py --param-file params/simulate_agents_2d.params
    python engine_simulate_scheduling.py --agents 60 --hours 4 --seed 7 --no-animation

Explicit CLI flags override the params file.

Outputs (output/08_agent_sim_2d/)
---------------------------------
    agent_missions.csv    one row per mission (endpoints, timings, holds) plus
                          the ACHIEVED velocity: velocity_kmh = distance flown /
                          total mission time (launch -> complete, so it carries
                          holds, cost-map slow-downs and the dock stop), and
                          air_velocity_kmh = the same over airborne time only
    separation_violations.csv  every loss of the horizontal standard:
                          time, both agents, gap, location, both corridors
    sim_timeline.csv      per-sample airborne / holding counts, min sep
    trajectories.csv      downsampled agent positions over time
    metrics.json          fleet-level summary + deconfliction check
    figures/00_network_traffic.png    network + all mission routes
    figures/01_density.png            traffic density heatmap
    figures/02_timeline.png           airborne / holding / separation vs time
    agents_animation.gif              time-compressed fleet animation
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from collections import defaultdict
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D

# This engine lives in src/ but is anchored on the PROJECT ROOT: every path it
# resolves -- params files, output trees, and the root-level stage scripts -- is
# written relative to the root, not to src/. Put the root on sys.path too, so
# `from src.x import y` works whether this file is run through its launcher
# (runpy from the root) or directly as `python src/engine_*.py`.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.maprule import add_map_rule
from src.orca import orca_step

THIS_DIR = _ROOT          # project root: all params/output paths hang off it
VERSION = "v1"


# ======================================================================
# Parameter file handling (matches the rest of the pipeline)
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


# ======================================================================
# Lane geometry helpers (mirrors 07_build_corridor_network.py)
# ======================================================================
def _lane_xy(lanes: pd.DataFrame, leg_id: str, lane: str) -> np.ndarray:
    g = lanes[(lanes["leg_id"] == leg_id) & (lanes["lane"] == lane)].sort_values("seq")
    return g[["x", "y"]].to_numpy(float)


def _resample_n(xy: np.ndarray, n: int = 60) -> np.ndarray:
    seg = np.hypot(*np.diff(xy, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0.0, float(s[-1]), n)
    return np.column_stack([np.interp(t, s, xy[:, 0]), np.interp(t, s, xy[:, 1])])


def travel_color_lane(lanes: pd.DataFrame, leg_id: str, forward: bool) -> tuple[str, np.ndarray]:
    """Direction-coded lane: FORWARD rides the RED lane A, BACKWARD rides the
    BLUE lane B (07 draws A red, B blue; on a roundabout A meets the outer ring,
    B the inner ring). The stored lanes are oriented a->b, so backward is B
    reversed. The two directions use different physical lanes -> spatial
    deconfliction, same as travel_right_lane but colour-explicit."""
    xy_a = _lane_xy(lanes, leg_id, "A")
    xy_b = _lane_xy(lanes, leg_id, "B")
    if forward:
        if len(xy_a) >= 2:
            return "A", xy_a
        return "B", xy_b
    if len(xy_b) >= 2:
        return "B", xy_b[::-1]
    return "A", xy_a[::-1] if len(xy_a) else xy_a


def travel_right_lane(lanes: pd.DataFrame, leg_id: str, forward: bool) -> tuple[str, np.ndarray]:
    """Physical label and travel-oriented polyline of the lane on the
    RIGHT of the travel direction. The right/left side is decided
    GEOMETRICALLY (sign of the cross product at mid-leg), exactly as in
    07's travel_left_right_lanes, so soft-buffer legs are not mis-sided.
    Because 'right' flips when the leg is ridden in reverse, the two
    travel directions of a leg return different physical labels and hence
    disjoint traffic resources -- that is the spatial deconfliction."""
    xy_a = _lane_xy(lanes, leg_id, "A")
    xy_b = _lane_xy(lanes, leg_id, "B")
    if not forward:
        xy_a = xy_a[::-1] if len(xy_a) else xy_a
        xy_b = xy_b[::-1] if len(xy_b) else xy_b
    if len(xy_a) < 2 or len(xy_b) < 2:
        if len(xy_a) >= 2:
            return "A", xy_a
        return "B", xy_b
    ra = _resample_n(xy_a)
    rb = _resample_n(xy_b)
    center = 0.5 * (ra + rb)
    m = len(center) // 2
    t = center[min(m + 1, len(center) - 1)] - center[max(m - 1, 0)]
    d = ra[m] - center[m]
    side_a = float(t[0] * d[1] - t[1] * d[0])
    # 07 returns (left, right) = (A, B) when side_a >= 0, else (B, A).
    if side_a >= 0.0:
        return "B", xy_b
    return "A", xy_a


def polyline_cumlen(xy: np.ndarray) -> np.ndarray:
    seg = np.hypot(*np.diff(xy, axis=0).T)
    return np.concatenate([[0.0], np.cumsum(seg)])


def interp_xy(xy: np.ndarray, cs: np.ndarray, s: float) -> np.ndarray:
    s = min(max(s, 0.0), float(cs[-1]))
    return np.array([np.interp(s, cs, xy[:, 0]), np.interp(s, cs, xy[:, 1])])


def load_costmap(path):
    """Load the slowness cost-map produced by 08_generate_costmap.py.
    Returns (grid[ny,nx], x0, y0, res, nx, ny) or None if absent."""
    if not str(path or "").strip():        # blank -> THIS_DIR, which exists
        return None
    p = THIS_DIR / str(path)
    if not p.is_file():
        return None
    z = np.load(p)
    g = z["slowness"].astype(float)
    return (g, float(z["x0"]), float(z["y0"]), float(z["res"]),
            int(g.shape[1]), int(g.shape[0]))


# ======================================================================
# Network / route model
# ======================================================================
class LegSeg:
    """Read-only travel-oriented geometry of one directed leg-lane.
    Shared by every agent that rides it (state lives on the agent).
    src_node/dst_node are the network nodes this lane leaves / arrives at,
    used for node-level (junction) deconfliction.

    s_offset makes the along-lane spacing coordinate GLOBAL: for a shared ring
    lane every agent enters at a different angle, so its private arc `s_local`
    is not comparable to another's. s_offset is the ring-lane arc position of
    this seg's start (measured in the lane's circulation direction from a fixed
    reference), so `s_local + s_offset` is a common progress coordinate for all
    agents on the same ring resource. Straight legs use s_offset = 0.

    ring_circ is the circumference (2*pi*r) of the ring lane this seg rides, or
    0.0 for straight legs. The global progress `s_local + s_offset` is NOT reduced
    modulo the circumference, so an agent that has circulated past the 2*pi*r wrap
    reads a global coordinate larger than every un-wrapped occupant near angle 0 --
    it then sees no leader ahead and can close inside the headway at the wrap seam.
    ring_circ lets the RING_METER logic compare positions on the circle (mod
    ring_circ) so spacing holds across that seam."""
    __slots__ = ("res", "xy", "cs", "length", "src_node", "dst_node",
                 "s_offset", "ring_circ")

    def __init__(self, res: str, xy: np.ndarray, src_node: str, dst_node: str,
                 s_offset: float = 0.0, ring_circ: float = 0.0):
        self.res = res
        self.xy = xy
        self.cs = polyline_cumlen(xy)
        self.length = float(self.cs[-1])
        self.src_node = src_node
        self.dst_node = dst_node
        self.s_offset = float(s_offset)
        self.ring_circ = float(ring_circ)


class Network:
    def __init__(self, corridor_dir: Path, ring_travel: bool = False,
                 lane_gap: float = 50.0, ring_right_hand: bool = True):
        self.nodes = pd.read_csv(corridor_dir / "network_nodes.csv")
        self.lanes = pd.read_csv(corridor_dir / "lane_nodes.csv")
        self.legs = pd.read_csv(corridor_dir / "network_legs.csv")
        self.pair_routes = pd.read_csv(corridor_dir / "pair_routes.csv")

        # roundabout ring geometry -> ring circulation between entry legs.
        # A route that passes THROUGH a ring node gets a RING ARC seg spliced
        # in (see route_segs / ring_seg): agents circulate on the outer RED
        # ring (forward, CCW) or inner BLUE ring (backward, CW) rather than
        # jumping across the ring centre.
        self.ring_travel = bool(ring_travel)
        self.ring_right_hand = bool(ring_right_hand)  # one-way CCW circulation
        self._ring_ctr = 0                    # unique-id source for ring-arc segs
        self.rings: dict[str, tuple[np.ndarray, float, float]] = {}
        rf = corridor_dir / "roundabouts.csv"
        if rf.exists():
            for r in pd.read_csv(rf).itertuples():
                c = np.array([float(r.center_x), float(r.center_y)], float)
                r_out = float(r.radius_m)
                r_in = max(0.3 * r_out, r_out - lane_gap)   # mirrors 07/08
                self.rings[str(r.rbt_id)] = (c, r_out, r_in)

        obj = self.nodes[self.nodes["kind"] == "objective"]
        self.obj_xy = {r.net_id: np.array([r.x, r.y], float) for r in obj.itertuples()}
        self.node_xy = {str(r.net_id): np.array([r.x, r.y], float)
                        for r in self.nodes.itertuples()}
        # map centre + each leg's "outerness" (0 = centre .. 1 = edge), used
        # to steer fast agents toward outer corridors
        cx = float(self.nodes["x"].mean()); cy = float(self.nodes["y"].mean())
        self.centroid = np.array([cx, cy], float)

        # pair -> (leg_ids, forward flags, node sequence) in stored orientation
        self._routes: dict[str, tuple[list[str], list[bool], list[str]]] = {}
        for r in self.pair_routes.itertuples():
            if not bool(r.success):
                continue
            legs = str(r.legs).split(";")
            dirs = [d == "fwd" for d in str(r.leg_directions).split(";")]
            via = str(r.via).split("-")
            self._routes[str(r.pair)] = (legs, dirs, via)

        # directed graph over network legs (for backup rerouting): each
        # undirected leg net_a<->net_b gives two directed edges carrying
        # (neighbour, leg_id, forward, length)
        # a_id/b_id hold the string net_ids (net_a/net_b are numeric indices)
        self._adj: dict[str, list[tuple[str, str, bool, float]]] = defaultdict(list)
        self._leg_ends: dict[str, tuple[str, str]] = {}   # leg_id -> (a_id, b_id)
        self.leg_outerness: dict[str, float] = {}         # leg_id -> 0..1
        for r in self.legs.itertuples():
            a, b, lg, L = str(r.a_id), str(r.b_id), str(r.leg_id), float(r.length_m)
            self._adj[a].append((b, lg, True, L))
            self._adj[b].append((a, lg, False, L))
            self._leg_ends[lg] = (a, b)
        _diff = self.nodes[["x", "y"]].to_numpy(float) - self.centroid
        span = max(float(np.hypot(_diff[:, 0], _diff[:, 1]).max()), 1.0)
        for lg, (a, b) in self._leg_ends.items():
            mid = 0.5 * (self.node_xy[a] + self.node_xy[b])
            self.leg_outerness[lg] = float(np.hypot(*(mid - self.centroid))) / span

        self._seg_cache: dict[tuple[str, bool], LegSeg] = {}

    def objectives(self, prefix: str) -> list[str]:
        obj = self.nodes[(self.nodes["kind"] == "objective") &
                         (self.nodes["net_id"].str.startswith(prefix))]
        return sorted(obj["net_id"].tolist())

    def leg_seg(self, leg_id: str, forward: bool) -> LegSeg:
        key = (leg_id, forward)
        seg = self._seg_cache.get(key)
        if seg is None:
            pick = travel_color_lane if self.ring_travel else travel_right_lane
            label, xy = pick(self.lanes, leg_id, forward)
            res = f"{leg_id}#{label}"
            a, b = self._leg_ends[leg_id]        # (net_a, net_b)
            src, dst = (a, b) if forward else (b, a)
            seg = LegSeg(res, np.ascontiguousarray(xy, dtype=float), src, dst)
            self._seg_cache[key] = seg
        return seg

    def route_pairs(self, origin: str, dest: str):
        """(pairs, nodes) for origin -> dest from the precomputed pair
        route: pairs = [(leg_id, forward), ...], nodes = [origin, ..., dest].
        Returns None if the pair is not stored."""
        fwd = f"{origin}_to_{dest}"
        rev = f"{dest}_to_{origin}"
        if fwd in self._routes:
            legs, dirs, via = self._routes[fwd]
            return list(zip(legs, dirs)), list(via)
        if rev in self._routes:
            legs, dirs, via = self._routes[rev]
            pairs = [(lg, not d) for lg, d in zip(legs, dirs)][::-1]
            return pairs, list(via)[::-1]
        return None

    def dijkstra(self, src: str, dst: str, blocked: set[str] | None = None):
        """Shortest leg sequence src -> dst over the network graph,
        optionally excluding blocked leg_ids. Returns [(leg_id, forward)]
        or None if unreachable. Used for backup reroutes."""
        blocked = blocked or set()
        import heapq
        dist = {src: 0.0}
        prev: dict[str, tuple[str, str, bool]] = {}
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == dst:
                break
            if d > dist.get(u, np.inf):
                continue
            for (v, lg, fwd, L) in self._adj[u]:
                if lg in blocked:
                    continue
                nd = d + L
                if nd < dist.get(v, np.inf):
                    dist[v] = nd
                    prev[v] = (u, lg, fwd)
                    heapq.heappush(pq, (nd, v))
        if dst not in prev and dst != src:
            return None
        pairs = []
        cur = dst
        while cur != src:
            u, lg, fwd = prev[cur]
            pairs.append((lg, fwd))
            cur = u
        return pairs[::-1]

    def pairs_to_segs(self, pairs) -> list[LegSeg]:
        return [self.leg_seg(lg, d) for lg, d in pairs]

    def nodes_of(self, pairs) -> list[str]:
        """Network-node sequence [src0, dst0, dst1, ...] visited by a directed
        leg path, reconstructed from the leg endpoints (dst_i == src_{i+1})."""
        seq: list[str] = []
        for lg, fwd in pairs:
            a, b = self._leg_ends[lg]
            s, d = (a, b) if fwd else (b, a)
            if not seq:
                seq.append(s)
            seq.append(d)
        return seq

    def ring_seg(self, rbt: str, p_in: np.ndarray, p_out: np.ndarray,
                 forward: bool = True, step_deg: float = 6.0) -> LegSeg | None:
        """Circulating RING ARC across roundabout `rbt`, from entry point p_in
        to exit point p_out.

        RIGHT-HAND TRAFFIC (`ring_right_hand`, default): every agent circulates
        COUNTER-CLOCKWISE (a right-hand-traffic roundabout is one-way CCW), so no
        agent ever takes a clockwise short-cut across the ring the "wrong way".
        The two ring lanes stay co-directional: forward travel rides the outer
        RED ring, backward travel the inner BLUE ring, but BOTH go CCW. This can
        add up to a full extra turn versus the shorter arc -- that is the price of
        a consistent, head-on-free circulation.

        LEGACY (`ring_right_hand` off): the SHORTER way round is taken --
        counter-clockwise rides the outer RED ring, clockwise the inner BLUE ring.

        Endpoints are stitched on so the arc is continuous with the entry/exit
        legs. Returns None when entry and exit coincide."""
        c, r_out, r_in = self.rings[rbt]
        th_in = math.atan2(float(p_in[1] - c[1]), float(p_in[0] - c[0]))
        th_out = math.atan2(float(p_out[1] - c[1]), float(p_out[0] - c[0]))
        d_ccw = (th_out - th_in) % (2.0 * math.pi)
        if d_ccw < 1e-2 or (2.0 * math.pi - d_ccw) < 1e-2:
            return None
        # ONE shared resource per (roundabout, ring lane): RINGA = outer red,
        # RINGB = inner blue. Each lane is a fixed one-way circle, so every
        # agent's along-lane progress is comparable via s_offset (arc position of
        # the entry angle in the circulation direction) -> real headway spacing on
        # the ring, no phantom leaders. `rbt#` prefix keeps conflict-grouping by
        # roundabout (res.split('#')[0]).
        two_pi = 2.0 * math.pi
        if self.ring_right_hand:
            # right-hand rule: ALWAYS circulate CCW; lane picked by travel dir.
            direction, sweep = +1.0, d_ccw
            if abs(r_out - r_in) < 1e-6:
                # SINGLE ring (05 built it with ROUNDABOUT_LANE_GAP_M = 0): both
                # travel directions ride the SAME physical circle, so they must
                # also share ONE traffic resource -- otherwise two agents at the
                # same point are metered separately and the headway is fiction.
                r, res = r_out, f"{rbt}#RING"
            else:
                r, res = (r_out, f"{rbt}#RINGA") if forward else (r_in, f"{rbt}#RINGB")
        elif d_ccw <= math.pi:                    # CCW shorter -> outer red, forward
            direction, r, res, sweep = +1.0, r_out, f"{rbt}#RINGA", d_ccw
        else:                                     # CW shorter -> inner blue, backward
            direction, r, res, sweep = -1.0, r_in, f"{rbt}#RINGB", two_pi - d_ccw
        s_offset = r * ((direction * th_in) % two_pi)   # global entry progress
        n = max(2, int(math.ceil(sweep / math.radians(step_deg))) + 1)
        ang = th_in + direction * np.linspace(0.0, sweep, n)
        arc = np.column_stack([c[0] + r * np.cos(ang), c[1] + r * np.sin(ang)])
        xy = np.vstack([np.asarray(p_in, float), arc, np.asarray(p_out, float)])
        return LegSeg(res, np.ascontiguousarray(xy, float), rbt, rbt,
                      s_offset=s_offset, ring_circ=two_pi * r)

    @staticmethod
    def _circle_cross(p_out: np.ndarray, p_in: np.ndarray, c: np.ndarray,
                      r: float) -> np.ndarray:
        """Point where the segment p_out->p_in (outside->inside) crosses the
        circle (c, r). Solves |f + t*a|^2 = r^2 for the first root in [0, 1]."""
        a = np.asarray(p_in, float) - np.asarray(p_out, float)
        f = np.asarray(p_out, float) - np.asarray(c, float)
        aa = float(a @ a)
        if aa < 1e-12:
            return np.asarray(p_out, float)
        bb = 2.0 * float(f @ a)
        cc = float(f @ f) - r * r
        disc = bb * bb - 4.0 * aa * cc
        if disc < 0.0:
            return np.asarray(p_out, float)
        rt = math.sqrt(disc)
        for t in sorted(((-bb - rt) / (2.0 * aa), (-bb + rt) / (2.0 * aa))):
            if -1e-9 <= t <= 1.0 + 1e-9:
                return np.asarray(p_out, float) + min(max(t, 0.0), 1.0) * a
        return np.asarray(p_out, float)

    def clip_at_ring(self, xy: np.ndarray, rbt: str, at_end: bool) -> np.ndarray:
        """Trim a leg polyline where it meets a roundabout's ring circle.

        Legs are stored running all the way to the roundabout's NODE, which sits
        at the ring CENTRE -- so splicing an arc onto the raw leg end sends the
        agent down a radial spoke, across the island, and back out. Clipping the
        leg at the ring boundary makes the entry/exit points lie ON the circle:
        the arc is then tangent-continuous with the legs, the enclosed disk is
        never entered, and the exit peels off to the RIGHT of CCW travel."""
        c, r_out, _r_in = self.rings[rbt]
        d = np.hypot(xy[:, 0] - c[0], xy[:, 1] - c[1])
        # Anchor points sit EXACTLY on the circle, so a strict `d > r` test can
        # call them outside on a rounding hair and leave the dip that follows
        # them un-clipped (05's lane fit overshoots a few metres inside its own
        # ring right after the anchor). Cut past the LAST inside-or-on point
        # instead, which is insensitive to both.
        eps = 1e-6 * max(r_out, 1.0)
        inside = np.nonzero(d <= r_out + eps)[0]
        if len(inside) == 0:
            return xy                      # leg never touches the ring
        if at_end:
            k = int(inside[0]) - 1         # last point before it first goes inside
            if k < 0:
                return xy                  # starts inside: nothing to keep
            return np.vstack([xy[:k + 1], self._circle_cross(xy[k], xy[k + 1], c, r_out)])
        j = int(inside[-1]) + 1            # first point after it is last inside
        if j >= len(xy):
            return xy                      # ends inside: nothing to keep
        return np.vstack([self._circle_cross(xy[j], xy[j - 1], c, r_out), xy[j:]])

    def route_segs(self, pairs) -> list[LegSeg]:
        """Legs of a route as directed segs, with a RING ARC spliced in wherever
        the route passes through a roundabout node (ring_travel only). Legs that
        touch a ring are CLIPPED at its boundary first (see clip_at_ring), so the
        agent joins and leaves the circulation on the ring itself."""
        if not self.ring_travel or not self.rings:
            return self.pairs_to_segs(pairs)
        pairs = list(pairs)
        nodes = self.nodes_of(pairs)
        segs: list[LegSeg] = []
        for i, (lg, fwd) in enumerate(pairs):
            base = self.leg_seg(lg, fwd)
            xy = base.xy
            # clip at EVERY ring endpoint, including the route's own first and
            # last node -- a diverted agent can start or finish at a roundabout,
            # and an unclipped end there is a radial spoke across the island.
            clip_end = nodes[i + 1] in self.rings
            clip_start = nodes[i] in self.rings
            if clip_end:
                xy = self.clip_at_ring(xy, nodes[i + 1], at_end=True)
            if clip_start:
                xy = self.clip_at_ring(xy, nodes[i], at_end=False)
            if clip_end or clip_start:
                if len(xy) < 2:
                    xy = base.xy           # degenerate clip: keep the raw leg
                seg = LegSeg(base.res, np.ascontiguousarray(xy, float),
                             base.src_node, base.dst_node)
            else:
                seg = base
            segs.append(seg)
            if i + 1 < len(pairs):
                mid = nodes[i + 1]
                if mid in self.rings:
                    nb = self.leg_seg(pairs[i + 1][0], pairs[i + 1][1])
                    nxt_xy = self.clip_at_ring(nb.xy, mid, at_end=False)
                    # travel direction through the ring = direction of the leg
                    # ARRIVING at it, so the outer/inner lane matches fwd/back.
                    arc = self.ring_seg(mid, seg.xy[-1], nxt_xy[0],
                                        forward=bool(pairs[i][1]))
                    if arc is not None:
                        segs.append(arc)
        return segs


# ======================================================================
# Agent
# ======================================================================
class Agent:
    __slots__ = ("aid", "origin", "dest", "round_trip", "arrival_t", "depart_t",
                 "priority", "path", "contingency", "segs", "kinds", "dwell_dur",
                 "outbound_upto", "seg_idx", "s_local", "dwell_left", "status",
                 "dist_m", "air_s", "hold_s", "n_holds", "holding", "was_holding",
                 "launch_t", "complete_t", "route_len", "held_node",
                 "pad_t0", "pad_t1", "pad_booked",
                 "energy_req_wh", "energy_short", "speed_promoted",
                 "speed", "speed_kmh", "is_patrol", "laps", "stuck_s", "n_reroutes",
                 "cont_frac", "battery_wh", "charge_s", "n_charges", "is_priority",
                 "sched_t", "orca_xy", "orca_v")

    def __init__(self, aid, origin, dest, round_trip, priority,
                 segs, kinds, dwell_dur, outbound_upto, route_len,
                 speed_mps, speed_kmh, contingency=None, is_patrol=False):
        self.aid = aid
        self.origin = origin
        self.dest = dest
        self.round_trip = round_trip
        self.arrival_t = 0.0            # demand time (mission becomes releasable)
        self.priority = priority        # release order (lower launches first)
        self.path = (origin, dest)      # nominal route key (reporting)
        self.contingency = contingency  # None | "backup" | "return"
        self.speed = speed_mps          # cruise speed (m/s), by UAV class
        self.speed_kmh = speed_kmh
        self.is_patrol = is_patrol      # continuous city-monitoring loop agent
        self.is_priority = False        # route-aware priority mission (fast + spine access)
        self.laps = 0                   # completed patrol laps
        self.stuck_s = 0.0              # continuous time blocked (for reroute)
        self.n_reroutes = 0             # diversions taken around congestion
        self.cont_frac = 0.0            # outbound fraction at which a contingency hits
        self.battery_wh = 0.0          # live battery energy (Wh); set at build
        self.charge_s = 0.0            # total time spent charging at docks
        self.n_charges = 0             # dock charge sessions
        self.depart_t = None            # set when released by the launcher
        self.segs = segs                # list[LegSeg | None]  (None marks a dwell)
        self.kinds = kinds              # list[str] "leg" | "dwell"
        self.dwell_dur = dwell_dur
        self.outbound_upto = outbound_upto  # seg indices < this are outbound
        self.route_len = route_len

        self.seg_idx = -1           # -1 = at origin, not yet launched
        self.s_local = 0.0
        self.dwell_left = 0.0
        self.status = "pre"         # pre|queued|flying|hold|dwell|done
        self.dist_m = 0.0
        self.air_s = 0.0
        self.hold_s = 0.0
        self.n_holds = 0
        self.holding = False
        self.was_holding = False
        self.launch_t = None
        self.complete_t = None
        self.held_node = None       # network node currently reserved for transit
        self.pad_t0 = None          # dock pad reservation: window start (s)
        self.pad_t1 = None          # ... and end; None = no reservation held
        self.pad_booked = False
        self.energy_req_wh = 0.0    # predicted Wh for the binding leg
        self.energy_short = False   # in flight: will not make it on what is left
        self.speed_promoted = False # class raised by the energy check
        self.sched_t = None         # CTOT from the departure scheduler (None = unscheduled)
        # free 2-D state, live ONLY while the agent is inside a roundabout's ORCA
        # zone. Everywhere else the agent is a 1-D point on a lane centreline.
        self.orca_xy = None
        self.orca_v = None

    @property
    def done(self) -> bool:
        return self.status == "done"

    @property
    def launched(self) -> bool:
        return self.depart_t is not None

    def on_leg(self) -> bool:
        return (self.status not in ("done", "dead") and 0 <= self.seg_idx < len(self.segs)
                and self.kinds[self.seg_idx] == "leg")

    def cur_seg(self) -> LegSeg:
        return self.segs[self.seg_idx]

    def outbound(self) -> bool:
        return self.seg_idx < self.outbound_upto

    def position(self, obj_xy) -> np.ndarray:
        if self.orca_xy is not None:      # inside a roundabout: real 2-D position
            return self.orca_xy
        if self.seg_idx == -1:
            return self.segs[0].xy[0] if self.segs else obj_xy[self.origin]
        if self.kinds[self.seg_idx] == "dwell":
            return obj_xy[self.dest]
        seg = self.segs[self.seg_idx]
        return interp_xy(seg.xy, seg.cs, self.s_local)


# ======================================================================
# Fleet generation
# ======================================================================
def _speed_classes(params):
    """[(kmh, m/s), ...] UAV classes and their selection weights."""
    kmh = list(pget(params, "SPEED_CLASSES_KMH", [60.0, 50.0, 30.0]))
    w = list(pget(params, "SPEED_CLASS_WEIGHTS", [1.0] * len(kmh)))
    w = (w + [1.0] * len(kmh))[:len(kmh)]
    tot = sum(w) or 1.0
    return kmh, [x / tot for x in w]


def build_patrol_loop(net: Network, start_prefix: str = "DB") -> list[tuple[str, bool]]:
    """A closed circuit around the map for city-monitoring patrols: visit
    the objectives ordered by bearing around the map centre, joining each
    to the next by the shortest corridor path, then close the loop. The loop
    is rotated to START at the drone base (first `start_prefix` objective) so
    the single patrol unit launches from and returns to the depot."""
    objs = list(net.obj_xy.items())
    cx = float(np.mean([p[0] for _i, p in objs]))
    cy = float(np.mean([p[1] for _i, p in objs]))
    order = sorted(objs, key=lambda ip: math.atan2(ip[1][1] - cy, ip[1][0] - cx))
    ids = [i for i, _p in order]
    # rotate the bearing order so the circuit begins at the drone base
    starts = [k for k, i in enumerate(ids) if str(i).startswith(start_prefix)]
    if starts:
        k = starts[0]
        ids = ids[k:] + ids[:k]
    loop: list[tuple[str, bool]] = []
    for i in range(len(ids)):
        a, b = ids[i], ids[(i + 1) % len(ids)]
        seg = net.dijkstra(a, b)
        if seg:
            loop.extend(seg)
    return loop


def build_fleet(net: Network, params, rng: np.random.Generator):
    n = int(pget(params, "N_AGENTS", 400))
    rt_prob = float(pget(params, "ROUND_TRIP_PROB", 0.5))
    cont_prob = float(pget(params, "CONTINGENCY_PROB", 0.10))
    backup_frac = float(pget(params, "BACKUP_FRACTION", 0.5))
    service_s = float(pget(params, "SERVICE_TIME_S", 60.0))
    battery_wh = float(pget(params, "BATTERY_WH", 200.0))
    start_pct = float(pget(params, "START_BATTERY_PCT", 1.0))
    db_pref = str(pget(params, "DB_PREFIX", "DB"))
    dk_pref = str(pget(params, "DK_PREFIX", "DK"))
    # arrival window: demand appears over this span (0 -> all at t=0)
    arr_win_s = float(pget(params, "ARRIVAL_WINDOW_H",
                            pget(params, "SHIFT_HOURS", 1.0))) * 3600.0
    balance = bool(pget(params, "BALANCE_ROUTES", True))
    kmh, weights = _speed_classes(params)

    # ---- route-aware priority ------------------------------------------
    # Missions whose ORIGIN or DEST is one of these objective labels are
    # prioritised: forced to PRIORITY_SPEED_KMH (fastest class by default),
    # exempt from the inner/outer zone bias so they may use the congested
    # inner spine directly (see zone_cost), and given launch/merge precedence
    # via a priority bonus. Empty list -> feature OFF (uniform behaviour).
    prio_labels = set(pget(params, "PRIORITY_ORIGIN_LABELS", []) or [])
    prio_speed_kmh = float(pget(params, "PRIORITY_SPEED_KMH", max(kmh)))
    # NOTE: separation is a TIME headway, so faster agents need MORE lane spacing
    # (~500 m at 60 vs ~250 m at 30 km/h) -- forcing priority traffic fast can
    # REDUCE capacity on the very spine we want to clear. Gate speed-forcing so the
    # queue-precedence + direct-spine-access levers can be used without it.
    prio_force_speed = bool(pget(params, "PRIORITY_FORCE_SPEED", True))

    dbs = net.objectives(db_pref)
    dks = net.objectives(dk_pref)
    if not dbs or not dks:
        raise RuntimeError(f"No objectives found for prefixes {db_pref!r}/{dk_pref!r}.")

    # every valid directed DB<->DK route (both directions) -- balancing
    # cycles missions round-robin over these so ALL routes are used at once
    route_keys = []
    for db in dbs:
        for dk in dks:
            if net.route_pairs(db, dk):
                route_keys.append((db, dk))
            if net.route_pairs(dk, db):
                route_keys.append((dk, db))

    def rlen(segs):
        return sum(s.length for s in segs if s is not None)

    def pick_speed():
        c = int(rng.choice(len(kmh), p=weights))
        return kmh[c] / 3.6, kmh[c]

    agents: list[Agent] = []
    for aid in range(n):
        if balance:
            origin, dest = route_keys[aid % len(route_keys)]
        else:
            db = str(rng.choice(dbs))
            dk = str(rng.choice(dks))
            origin, dest = (db, dk) if rng.random() < 0.5 else (dk, db)

        # mission SPEC only -- the actual route is planned dynamically at
        # launch (cost-aware, avoiding captured legs); see simulate()
        contingency = None
        round_trip = bool(rng.random() < rt_prob)
        if rng.random() < cont_prob:
            contingency = "backup" if rng.random() < backup_frac else "return"
            round_trip = False
        sp_mps, sp_kmh = pick_speed()
        # priority missions (touching a PRIORITY_ORIGIN_LABELS hub) are forced
        # to the priority speed class so they clear the shared spine faster
        is_prio = bool(prio_labels) and (origin in prio_labels or dest in prio_labels)
        if is_prio and prio_force_speed:
            sp_kmh = prio_speed_kmh
            sp_mps = prio_speed_kmh / 3.6
        a = Agent(aid, origin, dest, round_trip, aid, [], [], service_s,
                  0, 0.0, sp_mps, sp_kmh, contingency=contingency)
        a.is_priority = is_prio
        a.cont_frac = float(rng.uniform(0.3, 0.7))
        a.battery_wh = battery_wh * start_pct
        agents.append(a)
    # demand arrives uniformly over the arrival window (0 -> all at t=0)
    for a in agents:
        a.arrival_t = float(rng.uniform(0.0, arr_win_s)) if arr_win_s > 0 else 0.0
    agents.sort(key=lambda a: a.arrival_t)
    # priority = release/tie-break rank (lower goes first). Priority missions get
    # an n-agent bonus so they win launch/merge TIES (arrival time and remaining
    # leg distance still dominate, so this nudges rather than starves others).
    for p, a in enumerate(agents):
        a.priority = (p - n) if a.is_priority else p

    # ---- city-monitoring patrol: ONE unit launched from the drone base, that
    # circulates the map perimeter and RE-LAUNCHES every PATROL_INTERVAL_MIN.
    # A single physical drone is modelled (never two airborne at once); the
    # relaunch scheduling is handled in simulate(). The patrol has the HIGHEST
    # priority and is treated by deliveries as a moving (dynamic) obstacle.
    patrol_kmh = float(pget(params, "PATROL_SPEED_KMH", 50.0))
    interval_s = float(pget(params, "PATROL_INTERVAL_MIN", 30.0)) * 60.0
    patrols: list[Agent] = []
    loop = build_patrol_loop(net, start_prefix=db_pref)
    if interval_s > 0 and loop:
        loop_segs = net.route_segs(loop)
        loop_len = sum(s.length for s in loop_segs)
        a = Agent(1_000_000, "PATROL", "PATROL", False, -(n + 1),  # top priority
                  list(loop_segs), ["leg"] * len(loop_segs), 0.0,
                  len(loop_segs), loop_len,
                  patrol_kmh / 3.6, patrol_kmh, is_patrol=True)
        a.arrival_t = 0.0                        # first sortie at shift start
        a.battery_wh = battery_wh
        patrols.append(a)

    return agents, patrols


# ======================================================================
# Simulation
# ======================================================================
class PadBook:
    """Live dock-pad reservations, re-planned from the drone's ACTUAL progress.

    The CTOT scheduler books a pad once, before the mission launches, from a
    predicted arrival. Over a 15 h schedule that prediction drifts -- holds,
    cost-map slow-downs, ORCA detours -- and once the pads run at 100% any
    drift becomes a booking conflict, which is why the static plan degraded
    from 0 reactive dock-full holds at 100 agents to 3353 at 1000.

    This keeps the reservations honest while the drone is in the air: every
    RESERVATION_UPDATE_S the remaining route is re-priced through the cost-map
    and, if the arrival has moved by more than the tolerance, the pad is
    re-booked. A drone may only land on a pad it holds a STARTED reservation
    for, so the capacity is guaranteed by construction rather than checked on
    arrival, and a drone that has lost its slot learns about it EN ROUTE
    instead of on top of a full dock."""

    def __init__(self, capacity: int, hold_s: float):
        self.cap = int(capacity)
        self.hold = float(hold_s)
        self.res: dict = defaultdict(dict)      # dock -> {aid: (t0, t1)}
        self.n_book = 0
        self.n_rebook = 0
        self.n_update = 0
        self.drift_s = 0.0

    def _earliest(self, dock: str, t_from: float, skip_aid=None) -> float:
        """Earliest start >= t_from with a free pad for the whole hold window.
        Only reservation START times can become free slots, so scanning the
        existing starts is exact, not a discretisation."""
        if self.cap <= 0:
            return t_from
        windows = [(t0, t1) for aid, (t0, t1) in self.res[dock].items()
                   if aid != skip_aid]
        cand = t_from
        for _ in range(len(windows) + 1):
            overlap = [(t0, t1) for (t0, t1) in windows
                       if t0 < cand + self.hold and t1 > cand]
            if len(overlap) < self.cap:
                return cand
            cand = min(t1 for (_t0, t1) in overlap)   # wait for the first to free
        return cand

    def book(self, agent, dock: str, earliest: float) -> float:
        t0 = self._earliest(dock, earliest, skip_aid=agent.aid)
        self.res[dock][agent.aid] = (t0, t0 + self.hold)
        agent.pad_t0, agent.pad_t1, agent.pad_booked = t0, t0 + self.hold, True
        self.n_book += 1
        return t0

    def rebook(self, agent, dock: str, earliest: float) -> float:
        self.n_update += 1
        old = agent.pad_t0
        t0 = self._earliest(dock, earliest, skip_aid=agent.aid)
        if old is not None:
            self.drift_s += abs(t0 - old)
            if abs(t0 - old) > 1e-6:
                self.n_rebook += 1
        self.res[dock][agent.aid] = (t0, t0 + self.hold)
        agent.pad_t0, agent.pad_t1 = t0, t0 + self.hold
        return t0

    def release(self, agent, dock: str):
        self.res[dock].pop(agent.aid, None)
        agent.pad_t0 = agent.pad_t1 = None
        agent.pad_booked = False

    def peak(self, dock: str, t: float) -> int:
        return sum(1 for (t0, t1) in self.res[dock].values() if t0 <= t < t1)


def route_energy_wh(segs, v_cruise: float, p0: float, cd: float,
                    slowness_at=None, sample_m: float = 100.0,
                    s_from: float = 0.0, first_idx: int = 0) -> float:
    """Energy (Wh) to fly `segs` at `v_cruise`, priced through the cost-map.

    The multirotor power curve is P(v) = p0 + cd*v^3, and the cost-map makes the
    true velocity v*slowness, so a metre of route costs

        dE/dL = P(v*s) / (v*s) = p0/(v*s) + cd*(v*s)^2

    The hover floor is DIVIDED by the speed, so flying slower costs MORE energy
    per metre, not less -- and a slow drone in slow airspace pays twice. That
    term is why the 30 km/h class is the one that runs flat on the long routes:
    energy per metre is minimised at v = (p0/(2*cd))^(1/3), about 47 km/h here,
    and 30 km/h sits well below it.

    `s_from`/`first_idx` let this be re-evaluated from where a drone actually
    is, for the in-flight check."""
    v = max(float(v_cruise), 1e-3)
    total = 0.0
    for k, sg in enumerate(segs):
        if sg is None:
            continue
        L = float(sg.length)
        start = s_from if k == first_idx else 0.0
        if k < first_idx or L - start <= 0:
            continue
        n = max(2, int((L - start) / sample_m) + 1)
        acc = 0.0
        for x in np.linspace(start, L, n):
            sl = 1.0
            if slowness_at is not None:
                px, py = interp_xy(sg.xy, sg.cs, float(x))
                sl = min(max(slowness_at(px, py), 0.05), 1.0)
            ve = v * sl
            acc += p0 / ve + cd * ve * ve
        total += (L - start) * (acc / n)
    return total / 3600.0


def energy_optimal_speed(p0: float, cd: float) -> float:
    """Cruise speed (m/s) minimising energy per metre: d/dv [p0/v + cd*v^2] = 0."""
    return float((p0 / (2.0 * max(cd, 1e-12))) ** (1.0 / 3.0))


def route_time_factor(segs, slowness_at=None, sample_m: float = 100.0) -> float:
    """Metre-seconds per (metre / (m/s)) along a route -- i.e. how much longer
    the route takes than length/cruise would suggest.

    The cost-map makes true velocity = slowness(x, y) * cruise, so travel time
    is the integral of ds / (v * slowness). The legs are therefore sampled and
    the MEAN OF 1/slowness taken -- not 1/mean, which would flatter exactly the
    slow stretches that dominate the time. Returns the weighted factor f such
    that eta = (route_length / cruise) * f; f = 1.0 with no cost-map."""
    if slowness_at is None:
        return 1.0
    total_len = 0.0
    total_w = 0.0
    for sg in segs:
        L = float(sg.length)
        if L <= 0.0:
            continue
        n = max(2, int(L / sample_m) + 1)
        inv = 0.0
        for x in np.linspace(0.0, L, n):
            px, py = interp_xy(sg.xy, sg.cs, float(x))
            inv += 1.0 / min(max(slowness_at(px, py), 0.05), 1.0)
        total_len += L
        total_w += L * (inv / n)
    return (total_w / total_len) if total_len > 0 else 1.0


def _earliest_window(arr: np.ndarray, start: int, cap: int, width: int) -> int:
    """First index >= start where `width` consecutive bins are all under `cap`.
    Jumps over blocked runs rather than stepping, so a long horizon is cheap."""
    n = len(arr)
    width = max(1, width)
    x = max(0, min(start, n - 1))
    while x + width <= n:
        blocked = np.nonzero(arr[x:x + width] >= cap)[0]
        if len(blocked) == 0:
            return x
        x += int(blocked[-1]) + 1          # first index past the last blocker
    return max(0, n - width)


def schedule_departures(net: Network, agents: list[Agent], params,
                        slowness_at=None) -> dict:
    """STRATEGIC DEPARTURE SCHEDULING -- assign every mission a CTOT before the
    run, instead of metering departures reactively as the queue drains.

    This is the pre-tactical half of UTM coordination: the fleet is deconflicted
    on paper first, so the separation standard is met BY CONSTRUCTION at the one
    place the operator actually controls -- the departure time. Three constraints,
    all applied at the earliest slot that satisfies them:

      1. LONGITUDINAL, per departure corridor. Two missions leaving on the SAME
         first leg-lane are spaced >= SCHEDULE_HEADWAY_S (default = the
         TIME_HEADWAY_S in-trail standard, 30 s). Missions leaving the same
         origin on DIFFERENT first legs are laterally separated by the corridor
         geometry, so they may depart together -- the same rule the block model
         already used, now enforced ahead of time.
      2. CONCURRENCY. The predicted airborne count at the departure instant stays
         under MAX_CONCURRENT; each mission books its predicted flight interval
         (route length / cruise speed, plus dwell) into an occupancy profile.
      3. ORIGIN FAIR SHARE. No origin may hold more than SCHEDULE_ORIGIN_CAP
         airborne at once (default: the DCB fair share), so one hub cannot eat
         the whole airborne budget while the other corridors sit idle.
      4. DOCK PARKING. A dock holds DOCK_CAPACITY drones. A round trip parks
         there for DOCK_PARK_BUFFER_S + MIN_DEST_IDLE_S (manoeuvring onto the
         pad, then charging), so before a mission is cleared to launch its
         arrival is predicted -- eta_seconds(), which prices the route through
         the cost-map's slowness rather than assuming cruise speed -- and a pad
         is booked for that window. If the dock would be full when it lands,
         the DEPARTURE is pushed back until a pad frees: a drone is held on the
         ground rather than sent to hover over a full dock burning battery.

    Missions are served FIRST-COME-FIRST-SERVED: they are processed in arrival
    order, so an earlier mission gets first pick of the slots. That is NOT the
    same as a monotone departure order -- each mission contends for its own
    departure lane and its own destination dock, so one bound for a quiet lane
    and an empty dock can be cleared well before one queued ahead of it that
    needs a busy one. (This is why simulate() sorts its release queue by CTOT,
    not by arrival: gating on arrival order would serialise the whole fleet
    behind the worst-scheduled mission.) Returns a summary dict. The reactive gates in simulate() stay in place as a
    safety net -- a CTOT is a plan, and a plan can still meet a busy lane."""
    dely = [a for a in agents if not a.is_patrol]
    if not dely:
        return {"scheduled": 0}
    headway = float(pget(params, "SCHEDULE_HEADWAY_S",
                         pget(params, "TIME_HEADWAY_S", 30.0)))
    bin_s = max(1.0, float(pget(params, "SCHEDULE_BIN_S", 5.0)))
    max_conc = int(pget(params, "MAX_CONCURRENT", 10 ** 9))
    n_org = max(1, len({a.origin for a in dely}))
    org_cap = int(pget(params, "SCHEDULE_ORIGIN_CAP", 0)) or \
        max(1, math.ceil(float(pget(params, "DCB_CORRIDOR_SHARE", 1.5))
                         * max_conc / n_org))
    # only the SERVICE stop is booked: MIN_DEST_IDLE_S is spent parked at a dock,
    # which is not airborne time and does not consume the concurrency budget.
    dwell = float(pget(params, "SERVICE_TIME_S", 60.0))
    dock_cap = int(pget(params, "DOCK_CAPACITY", 0))
    dock_hold = float(pget(params, "DOCK_PARK_BUFFER_S", 600.0)) + \
        float(pget(params, "MIN_DEST_IDLE_S", 1800.0))

    # occupancy profile over a generous horizon, in bin_s buckets
    span = float(pget(params, "SHIFT_HOURS", 1.0)) * 3600.0 * \
        float(pget(params, "HORIZON_FACTOR", 12.0))
    nb = int(math.ceil(max(span, 3600.0) / bin_s)) + 8
    occ = np.zeros(nb, int)                              # airborne, all origins
    occ_org: dict = {o: np.zeros(nb, int) for o in {a.origin for a in dely}}
    occ_dock: dict = {d: np.zeros(nb, int) for d in {a.dest for a in dely}}
    last_dep: dict = {}                                  # first-leg res -> last CTOT
    hold_bins = int(math.ceil(dock_hold / bin_s))
    n_dock_pushed = 0
    dock_push_s = 0.0
    eta_sum = 0.0
    # ---- energy feasibility ----
    # A mission is only cleared if the battery can actually fly it. Energy per
    # metre is p0/(v*s) + cd*(v*s)^2, so the hover floor is divided by speed:
    # the SLOW classes are the expensive ones on long routes, and they are the
    # ones that run flat. Candidate classes are therefore tried cheapest-energy
    # first, which generally means raising the speed, not lowering it.
    e_p0, e_cd, e_bat = energy_params(params)
    e_reserve = float(pget(params, "ENERGY_RESERVE_PCT", 0.20))
    e_hold_allow = float(pget(params, "ENERGY_HOLD_ALLOWANCE_PCT", 0.25))
    e_usable = e_bat / (1.0 + e_reserve)
    _classes = [float(k) for k in pget(params, "SPEED_CLASSES_KMH", [60.0, 50.0, 30.0])]
    # order by energy per metre at that class -- cheapest first
    _classes.sort(key=lambda k: e_p0 / (k / 3.6) + e_cd * (k / 3.6) ** 2)
    n_promoted = 0
    n_infeasible = 0
    rows: list = []                     # the schedule itself, for audit + plotting
    dock_series: dict = {d: None for d in occ_dock}

    # Delivery agents are built with EMPTY segs -- their route is planned
    # lazily at launch -- so the nominal corridor route has to be resolved here
    # or every mission would look like a zero-length flight from a single
    # pseudo-corridor. Cached per (origin, dest): the geometry is shared, only
    # the cruise speed differs per agent.
    _route_cache: dict = {}

    def _nominal(a):
        """(first leg-lane resource, route length m, cost-map time factor)."""
        key = (a.origin, a.dest)
        got = _route_cache.get(key)
        if got is None:
            pr = net.route_pairs(a.origin, a.dest)
            if pr is None:
                got = (f"__origin__{a.origin}", 0.0, 1.0)
            else:
                segs = net.route_segs(pr[0])
                got = (segs[0].res if segs else f"__origin__{a.origin}",
                       float(sum(sg.length for sg in segs)),
                       route_time_factor(segs, slowness_at), segs)
            _route_cache[key] = got
        return got

    _energy_cache: dict = {}

    def _energy_need(a, segs, kmh):
        """Wh for the binding leg at `kmh`, with an allowance for holding.
        A round trip recharges to full at the dock, so the requirement is ONE
        leg, not the sum of both."""
        key = (a.origin, a.dest, round(kmh, 3))
        got = _energy_cache.get(key)
        if got is None:
            got = route_energy_wh(segs, kmh / 3.6, e_p0, e_cd, slowness_at) \
                * (1.0 + e_hold_allow)
            _energy_cache[key] = got
        return got

    def _earliest_free(arr, b0, cap):
        """First bin >= b0 whose count is under cap."""
        room = arr[b0:] < cap
        if room.all():
            return b0
        if not room.any():
            return nb - 1
        return b0 + int(np.argmax(room))

    n_delayed = 0
    total_delay = 0.0
    for a in sorted(dely, key=lambda x: (x.arrival_t, x.priority)):
        res, route_len, tfac, segs = _nominal(a)
        # ---- energy gate: can this drone actually fly this leg? ----
        need = _energy_need(a, segs, a.speed_kmh) if segs else 0.0
        if segs and need > e_usable:
            for kmh in _classes:                    # cheapest-energy class first
                cand = _energy_need(a, segs, kmh)
                if cand <= e_usable:
                    a.speed_kmh = kmh
                    a.speed = kmh / 3.6
                    a.speed_promoted = True
                    need = cand
                    n_promoted += 1
                    break
            else:
                n_infeasible += 1        # not flyable on one battery at any class
        a.energy_req_wh = need
        v = max(a.speed, 1e-3)
        # (1) corridor headway
        t0 = max(float(a.arrival_t), last_dep.get(res, -1e18) + headway)
        b = min(max(int(math.ceil(t0 / bin_s)), 0), nb - 1)
        og = occ_org[a.origin]
        # a round trip parks; predict WHEN it lands so a pad can be booked
        parks = bool(a.round_trip) and dock_cap > 0 and a.dest in occ_dock
        # arrival = outbound length / cruise, priced through the cost-map
        eta = (route_len / v) * tfac if parks else 0.0
        eta_sum += eta
        eta_bins = int(math.ceil(eta / bin_s))
        dk = occ_dock[a.dest] if parks else None
        b_first = b
        # WHY this mission ends up where it does. FCFS order is preserved --
        # scheduling only ever delays -- so the binding constraint is the whole
        # story of the schedule, and it is recorded per mission.
        reason = "ready" if t0 <= float(a.arrival_t) + 1e-9 else "corridor_headway"
        # (2)(3)(4) concurrency, origin fair share and dock parking -- advance
        # until all of them hold at the same slot
        for _ in range(nb):
            b2 = _earliest_free(occ, b, max_conc)
            if b2 > b:
                reason = "airborne_cap"
            b3 = _earliest_free(og, b2, org_cap)
            if b3 > b2:
                reason = "origin_cap"
            b4 = b3
            if dk is not None:
                # the pad is needed from eta after departure, for the whole stop
                w = _earliest_window(dk, b3 + eta_bins, dock_cap, hold_bins)
                b4 = max(b3, w - eta_bins)
                if b4 > b3:
                    reason = "dock_pad"
            if b4 == b3 == b2:
                b = b2
                break
            b = b4
        if parks and b > b_first:
            n_dock_pushed += 1
            dock_push_s += (b - b_first) * bin_s
        t = b * bin_s
        # airborne duration: out (+ back for a round trip), cost-map priced.
        # The dock stop is NOT airborne, so it is not part of this interval.
        dur = (route_len / v) * tfac * (2.0 if a.round_trip else 1.0) \
            + (dwell if a.round_trip else 0.0)
        b_end = min(nb, b + int(math.ceil(dur / bin_s)) + 1)
        occ[b:b_end] += 1
        og[b:b_end] += 1
        if dk is not None:                       # book the pad for the whole stop
            d0 = min(nb - 1, b + eta_bins)
            dk[d0:min(nb, d0 + hold_bins)] += 1
        last_dep[res] = t
        a.sched_t = t
        rows.append({
            "fcfs_rank": len(rows), "agent_id": a.aid,
            "origin": a.origin, "dest": a.dest,
            "round_trip": bool(a.round_trip), "speed_kmh": a.speed_kmh,
            "arrival_s": round(float(a.arrival_t), 1),
            "ctot_s": round(t, 1),
            "delay_s": round(t - float(a.arrival_t), 1),
            "binding": reason,
            "first_leg": res,
            "route_len_m": round(route_len, 1),
            "eta_s": round((route_len / v) * tfac, 1),
            "slowness_factor": round(tfac, 3),
            "energy_req_wh": round(need, 1),
            "energy_margin_pct": round(100.0 * (1.0 - need / max(e_bat, 1e-9)), 1),
            "speed_promoted": bool(a.speed_promoted),
            "dock_hold_s": round(dock_hold, 1) if parks else 0.0,
        })
        if t > a.arrival_t + 1e-9:
            n_delayed += 1
            total_delay += t - a.arrival_t
    ctots = [a.sched_t for a in dely]
    n_park = sum(1 for a in dely if a.round_trip)
    return {
        "scheduled": len(dely),
        "headway_s": headway,
        "origin_cap": org_cap,
        "dock_capacity": dock_cap,
        "dock_hold_s": dock_hold,
        "n_parking_missions": n_park,
        "mean_eta_to_dock_s": round(eta_sum / max(n_park, 1), 1),
        "energy": {
            "battery_wh": e_bat, "reserve_pct": e_reserve,
            "hold_allowance_pct": e_hold_allow, "usable_wh": round(e_usable, 1),
            "optimal_cruise_kmh": round(3.6 * energy_optimal_speed(e_p0, e_cd), 1),
            "n_speed_promoted": n_promoted,
            "n_infeasible_any_class": n_infeasible,
            "mean_required_wh": round(
                sum(r["energy_req_wh"] for r in rows) / max(len(rows), 1), 1),
            "max_required_wh": round(max((r["energy_req_wh"] for r in rows),
                                         default=0.0), 1),
        },
        "n_pushed_for_dock": n_dock_pushed,
        "mean_dock_push_s": round(dock_push_s / max(n_dock_pushed, 1), 1),
        "rows": rows,
        "bin_s": bin_s,
        "dock_cap_for_plot": dock_cap,
        "dock_profile": {d: occ_dock[d].tolist() for d in occ_dock},
        "n_delayed": n_delayed,
        "mean_delay_s": (total_delay / n_delayed) if n_delayed else 0.0,
        "last_ctot_s": max(ctots),
        "span_h": max(ctots) / 3600.0,
        "n_departure_lanes": len(last_dep),
    }


def simulate(net: Network, agents: list[Agent], patrols: list[Agent], params):
    """Per-LEG block simulation. A directed leg-lane is a block that holds
    at most ONE agent at a time ("same leg cannot be captured by two
    agents"). An origin may launch many agents at once as long as they
    take different first legs. No global concurrency cap, so the observed
    peak reflects the network's true simultaneous-agent capacity."""
    from collections import deque
    dt = float(pget(params, "DT_S", 1.0))
    # separation is a TIME headway (>= TIME_HEADWAY_S seconds), so the
    # distance gap scales with each agent's speed; SEPARATION_M is a floor
    headway_s = float(pget(params, "TIME_HEADWAY_S", 30.0))
    sep_floor = float(pget(params, "SEPARATION_M", 80.0))
    # representative separation THRESHOLD for metrics/plots: the actual binding
    # same-lane gap is the time headway (30 s) scaled by speed, floored at
    # sep_floor. The tightest such gap -- the slowest class -- is the real
    # lower bound every same-lane pair must respect, so use that, not the floor.
    _cls_kmh = list(pget(params, "SPEED_CLASSES_KMH", [60.0, 50.0, 30.0]))
    sep = max(headway_s * (min(_cls_kmh) / 3.6), sep_floor)

    def sep_of(a):
        return max(headway_s * a.speed, sep_floor)

    patrol_laps = int(pget(params, "PATROL_LAPS", 1))          # loops per sortie
    # the single patrol unit re-launches from base every PATROL_INTERVAL_MIN;
    # deliveries treat its live position as a moving (dynamic) obstacle.
    patrol_interval_s = float(pget(params, "PATROL_INTERVAL_MIN", 30.0)) * 60.0
    patrol_obstacle = bool(pget(params, "PATROL_AS_OBSTACLE", True))
    patrol_relaunch: dict[int, float] = {}     # aid -> next launch time
    patrol_total_sorties = 0                    # completed monitoring sorties
    patrol_total_laps = 0                       # loops flown across all sorties
    conflict_time = float(pget(params, "CONFLICT_TIME_S", 5.0))   # near-miss window

    # battery + dock charging
    battery_cap = float(pget(params, "BATTERY_WH", 200.0))
    charge_power = float(pget(params, "CHARGE_POWER_W", 600.0))
    charge_target = float(pget(params, "CHARGE_TARGET_PCT", 0.9)) * battery_cap
    # after arriving at the destination (round trips), an agent must idle at the
    # dock at least this long (charge / reload) before flying the return leg.
    min_dest_idle = float(pget(params, "MIN_DEST_IDLE_S", 300.0))
    e_p0 = float(pget(params, "HOVER_POWER_W", 220.0))
    e_cd = float(pget(params, "DRAG_POWER_COEF", 0.050))
    # slowness cost-map (from step 09): true velocity = slowness * base speed.
    # Routes are unaffected (fixed) -- the map only modulates travel speed.
    cost_map = load_costmap(pget(params, "COST_MAP_FILE",
                                 "output/07_costmap/slowness_costmap.npz"))
    if cost_map is not None:
        cm_grid, cm_x0, cm_y0, cm_res, cm_nx, cm_ny = cost_map

        def slowness_at(x, y):
            ix = min(max(int((x - cm_x0) / cm_res), 0), cm_nx - 1)
            iy = min(max(int((y - cm_y0) / cm_res), 0), cm_ny - 1)
            return float(cm_grid[iy, ix])
    else:
        def slowness_at(x, y):
            return 1.0
    shift_s = float(pget(params, "SHIFT_HOURS", 1.0)) * 3600.0
    horizon = shift_s * float(pget(params, "HORIZON_FACTOR", 12.0))
    sample_every = float(pget(params, "SAMPLE_EVERY_S", 20.0))
    max_concurrent = int(pget(params, "MAX_CONCURRENT", 10 ** 9))
    # concurrency metering: minimum seconds between successive delivery launches
    # so departures are spread in time (fewer simultaneous crossings/conflicts).
    launch_spacing = float(pget(params, "LAUNCH_SPACING_S", 2.0))
    # A6 DEMAND-CAPACITY BALANCING (**default ON** -- the KEPT config baked in; the
    # frozen baselines set DCB_MODE=False explicitly to reproduce the old default).
    # The launch gate above is a single GLOBAL concurrency cap (n_active <
    # max_concurrent); with 1000 missions dumped at t=0 it lets whichever corridor
    # is at the queue head hog the airborne budget, piling agents onto a few origin
    # corridors (peak_backlog=999 stays bound; A7/A5 could not move it). DCB meters
    # each ORIGIN CORRIDOR to a fair-share airborne cap and round-robins deferred
    # candidates to the back of the queue, so launches spread across corridors
    # (scheduling-level deconfliction, NASA-UTM style) -> fewer hot-node pileups.
    # A/B KEEP: completed +5.4%, holds -13%, battery_dead -38% vs the pre-DCB base.
    dcb_mode = bool(pget(params, "DCB_MODE", True))
    # SCHEDULE_MODE: departures come from schedule_departures()'s CTOTs instead of
    # the reactive meters. The schedule already enforces the corridor headway, the
    # concurrency cap and the origin fair share at PLAN time, so the global launch
    # spacing and the DCB rotation are switched off -- leaving them on would meter
    # the same traffic twice and fight the plan. enter_leg() still has the last
    # word: a CTOT is a plan, and the lane may nonetheless be occupied on the day.
    schedule_mode = bool(pget(params, "SCHEDULE_MODE", False))
    if schedule_mode:
        launch_spacing = 0.0
        dcb_mode = False
    # each corridor may hold at most dcb_share * (max_concurrent / n_origins)
    # airborne (slack > 1 lets demand imbalance still fill capacity); or set an
    # absolute DCB_CORRIDOR_CAP to override the derived value.
    dcb_share = float(pget(params, "DCB_CORRIDOR_SHARE", 1.5))
    dcb_cap_param = int(pget(params, "DCB_CORRIDOR_CAP", 0))
    dcb_n_origins = max(1, len({a.origin for a in agents if not a.is_patrol}))
    dcb_cap = dcb_cap_param if dcb_cap_param > 0 else \
        max(1, math.ceil(dcb_share * max_concurrent / dcb_n_origins))
    # A4 SPEED CONTROL (**default ON** -- the KEPT config baked in; the frozen
    # baselines that predate A4 set SPEED_CONTROL=False explicitly). Baseline
    # car-following is bang-bang: run at full speed up to the hard (leader - sep)
    # cap, then STOP (stop-and-go, and a stopped agent hovers). Speed control
    # instead ramps the cruise speed DOWN over a band above the separation floor,
    # so an agent decelerates early and keeps creeping instead of fully stopping --
    # smoother flow, fewer hard holds. Battery then drains at the agent's ACTUAL
    # velocity (slow cruise costs less than full cruise), which the bang-bang model
    # could not represent. A/B KEEP: makespan -4.2%, holds -38% on top of DCB.
    speed_control = bool(pget(params, "SPEED_CONTROL", True))
    # width of the deceleration band, as a multiple of the required gap sep_of(a):
    # speed ramps 0 (at the floor) -> full (at floor + band). 0.5 is the sweep knee
    # (1.0 over-packs creeping agents -> conflicts up, G3 fails; 2.0 far worse).
    speed_ctrl_band = float(pget(params, "SPEED_CTRL_BAND_FACTOR", 0.5))
    node_mutex = bool(pget(params, "NODE_MUTEX_ENABLE", False))
    node_approach = float(pget(params, "NODE_APPROACH_M", 50.0))
    obj_approach = float(pget(params, "OBJECTIVE_APPROACH_M", 120.0))
    # Flight-level (altitude) separation: each leg is assigned a level from its
    # travel HEADING (semicircular-rule style), so crossing / opposing traffic
    # is vertically separated. Two agents only CONFLICT when horizontally close
    # AND on the same level; the node mutex also only serialises same-level
    # agents. More levels -> more separation.
    n_levels = max(1, int(pget(params, "FLIGHT_LEVELS", 4)))
    level_sep = float(pget(params, "LEVEL_SEP_M", 30.0))
    base_level_z = float(pget(params, "BASE_LEVEL_M", 60.0))
    gridlock_s = float(pget(params, "GRIDLOCK_TIMEOUT_S", 300.0))
    # FLOW_MODE: "spacing" (many agents per leg, kept >= SEPARATION_M apart --
    # deadlock-free, so the fleet always drains) or "block" (strict one agent
    # per leg-lane -- can gridlock under heavy load).
    flow_block = str(pget(params, "FLOW_MODE", "spacing")).lower() == "block"
    # MERGE rule: an agent may merge onto a lane only if the nearest agent
    # already on it is at least this fraction of the LEG length ahead (i.e.
    # that much of the leg is free), never closer than SEPARATION_M. A lane
    # is effectively "locked" to new entries until its occupant clears that
    # much of it, which spreads traffic across more lanes/routes.
    merge_frac = float(pget(params, "MERGE_FREE_FRAC", 0.5))
    # A7 RING METERING (default OFF -> baseline unchanged). The ring global
    # progress s_local+s_offset is not wrapped to the circumference, so spacing
    # can break at the 2*pi*r seam (an agent past the wrap sees no leader near
    # angle 0). Two INDEPENDENT halves, so their cost can be isolated:
    #   RING_WRAP_FOLLOW -- ring car-following compares positions on the circle
    #        (mod ring_circ) so the headway holds across the seam (the cheap fix).
    #   RING_MERGE_METER -- a ring merge is additionally metered to keep
    #        >= ring_meter_gap clear on BOTH sides of the entry angle (adds holds).
    # RING_METER is a master switch turning BOTH on (the original A7 = KILL: the
    # meter's entry holds drove hover-drain battery deaths past gate G3). A7b runs
    # RING_WRAP_FOLLOW alone.
    # ---- ORCA inside the roundabouts ------------------------------------
    # A 40 m ring has a 251 m circumference: at a 50 m separation standard it
    # holds FIVE agents, and the first 1000-drone run peaked at NINE on the
    # busiest ring -- 1.8x oversubscribed, which no coordination rule can fix,
    # only geometry can. Stage 05 now sizes rings from the predicted density
    # (77-126 m here), which turns each one into a real 2-D manoeuvring area,
    # and ORCA is what uses that area: inside the zone agents choose a free 2-D
    # velocity by reciprocal collision avoidance instead of queueing on an arc.
    orca_rings = bool(pget(params, "ORCA_RINGS", False))
    # half the horizontal separation standard: a pair is clear when their
    # centres are more than 2*orca_radius apart
    orca_radius = float(pget(params, "ORCA_RADIUS_M", 0.5 * sep_floor))
    orca_tau = float(pget(params, "ORCA_TAU_S", 12.0))       # look-ahead horizon
    orca_half_w = float(pget(params, "ORCA_ZONE_MARGIN_M", 25.0))  # ring buffer half-width
    orca_island = float(pget(params, "ORCA_ISLAND_FRAC", 0.35))    # kept-clear centre
    # ORCA is exactly symmetric, so head-on / antipodal pairs deadlock at the
    # separation distance forever. A few degrees of RIGHT bias breaks it (and
    # matches right-hand traffic); measured: head-on pair unresolved after 1500
    # steps at 0 deg, resolved in 72 at 3 deg, separation held either way.
    orca_bias_deg = float(pget(params, "ORCA_BIAS_DEG", 3.0))
    # start lining up on the exit once the remaining CCW sweep is under this
    orca_exit_rad = math.radians(float(pget(params, "ORCA_EXIT_ANGLE_DEG", 25.0)))
    orca_exit_tol = float(pget(params, "ORCA_EXIT_TOL_M", 30.0))
    orca_ticks = [0]                                          # agent-steps run under ORCA

    ring_meter = bool(pget(params, "RING_METER", False))
    ring_wrap_follow = ring_meter or bool(pget(params, "RING_WRAP_FOLLOW", False))
    ring_merge_meter = ring_meter or bool(pget(params, "RING_MERGE_METER", False))
    # meter gap floor at ring entry; defaults to the same distance floor as legs.
    ring_meter_gap = float(pget(params, "RING_METER_GAP_M", sep_floor))
    # Dynamic (ACO-style) congestion cost: capturing a leg raises its routing
    # cost; the penalty of an occupant decays to 0 as it passes the half-way
    # point of the leg (a gradient), so a freshly-taken leg is expensive and
    # one whose occupant is leaving is cheap. Agents plan least-cost routes,
    # so the "next agent" avoids captured legs and takes longer detours.
    capture_cost = float(pget(params, "CAPTURE_COST_M", 3000.0))
    zone_bias = float(pget(params, "ZONE_BIAS_M", 2000.0))
    # HIGH penalty added to a leg while an occupant is HOLDING/WAITING on it, so
    # the router routes the next agents around a jam instead of queueing behind it.
    hold_penalty = float(pget(params, "HOLD_PENALTY_M", 20000.0))
    # A5 SYSTEM-OPTIMUM TOLLING (default OFF -> baseline unchanged). The capture
    # penalty above is a congestion cost LINEAR in leg occupancy that each agent
    # minimises for ITSELF -> a user equilibrium (each takes its own cheapest
    # route, ignoring the delay it imposes on others). For a link cost linear in
    # flow t(x)=t0+b*x, the system-optimum marginal-cost toll internalises that
    # externality: SO edge cost = b*x + x*b = 2*b*x = UE * (1+beta), beta=1. So
    # tolling simply scales the congestion term by toll_marginal_mult (default 2),
    # steering agents to accept longer detours off busy legs -> load spreads and
    # network utilisation rises. Static zone_cost is a speed/zone bias, not a
    # congestion externality, so it is left untolled.
    toll_mode = bool(pget(params, "TOLL_MODE", False))
    toll_marginal_mult = float(pget(params, "TOLL_MARGINAL_MULT", 2.0))
    kmh_list, _w = _speed_classes(params)
    fast_kmh, slow_kmh = max(kmh_list), min(kmh_list)

    obj_xy = net.obj_xy
    obj_set = set(obj_xy.keys())
    leg_busy: dict[str, Agent] = {}    # block mode: res -> occupying agent
    node_busy: dict[str, Agent] = {}   # node -> transiting agent (optional mutex)
    occ: dict = {}                     # spacing mode: res -> [(s_local, agent), ...]
    leg_penalty: dict = {}             # res -> extra routing cost from capture
    # A1 SPACE-TIME RESERVATION ROUTING (default OFF -> baseline unchanged). The
    # ACO leg_penalty above is MYOPIC: it charges the CURRENT occupancy of a leg,
    # so an agent routes around legs busy NOW, not legs that will be busy WHEN IT
    # ARRIVES. A1 replaces it with a reservation table: each planned agent books
    # the future time window it expects to occupy each leg (from its speed), and a
    # new plan pays resv_penalty for booking a leg beyond its capacity during its
    # own transit window -- so routes deconflict in space-TIME (SIPP-lite).
    # Reservations past the current time are pruned. Static zone_cost still applies.
    resv_routing = bool(pget(params, "RESV_ROUTING", False))
    resv_penalty = float(pget(params, "RESV_PENALTY_M", 8000.0))
    # only book/consider reservations up to this far ahead: near-term congestion is
    # what routing can act on, and this bounds each leg's window list (and cost).
    resv_horizon = float(pget(params, "RESV_HORIZON_S", 1800.0))
    leg_resv: dict = {}                # res -> [(t_enter, t_exit), ...] booked windows
    leg_cap_cache: dict = {}           # res -> concurrent-agent capacity (cached)

    def leg_capacity(res, L):
        c = leg_cap_cache.get(res)
        if c is None:
            c = max(1, int(L / sep)); leg_cap_cache[res] = c   # sep ~ nominal headway gap
        return c

    def _resv_overlap(res, a0, a1, t0):
        # count booked windows overlapping [a0, a1], and prune THIS leg's stale
        # (already-past) windows in the same pass so lists stay bounded. A global
        # prune every call is far too slow; past windows can't overlap a future
        # transit anyway (their exit < t0 <= a0), so dropping them lazily is exact.
        lst = leg_resv.get(res)
        if not lst:
            return 0
        keep = []
        n = 0
        for iv in lst:
            if iv[1] < t0:
                continue
            keep.append(iv)
            if iv[0] < a1 and a0 < iv[1]:
                n += 1
        leg_resv[res] = keep
        return n

    outerness = net.leg_outerness

    def zone_cost(lg, a):
        """Fast agents pay to use inner legs (steered to the OUTER zone);
        slow agents pay to use outer legs (kept to short inner routes)."""
        # priority missions are exempt: they take the DIRECT (often inner) spine
        # rather than being steered onto longer outer detours
        if getattr(a, "is_priority", False):
            return 0.0
        o = outerness.get(lg, 0.5)
        if a.speed_kmh >= fast_kmh:
            return zone_bias * (1.0 - o)
        if a.speed_kmh <= slow_kmh:
            return zone_bias * o
        return 0.0

    def leg_level(seg) -> int:
        """Flight level (0..n_levels-1) from a leg's travel HEADING. Opposing /
        crossing legs have different headings -> different levels -> vertical
        separation, so they are not counted as (2D) conflicts and the junction
        mutex only serialises SAME-level traffic."""
        if seg is None or len(seg.xy) < 2:
            return 0
        d = seg.xy[-1] - seg.xy[0]
        brg = math.degrees(math.atan2(float(d[1]), float(d[0]))) % 360.0
        return int(brg / (360.0 / n_levels)) % n_levels

    def agent_level(a) -> int:
        return leg_level(a.cur_seg())

    def plan_route(src, dst, a, blocked=None, book=False):
        """Least-cost directed leg path src->dst by Dijkstra over dynamic
        edge cost = length + capture penalty + speed/zone bias. Returns
        (pairs, nodes) or None. This is the ACO-style stigmergic routing:
        captured legs carry a cost 'pheromone' that others route around.

        A1 (resv_routing): a time-aware variant that plans over a space-time
        reservation table and books the chosen route's predicted occupancy."""
        import heapq
        blocked = blocked or set()
        if resv_routing:
            t0 = t                                 # closure: current sim time
            v_a = max(a.speed, 1e-3)               # m/s used to predict transit times
            dist = {src: 0.0}
            tarr = {src: t0}                       # predicted arrival time along best path
            prev: dict = {}
            pq = [(0.0, src)]
            while pq:
                d, u = heapq.heappop(pq)
                if u == dst:
                    break
                if d > dist.get(u, np.inf):
                    continue
                for (v, lg, fwd, L) in net._adj[u]:
                    if lg in blocked:
                        continue
                    res = net.leg_seg(lg, fwd).res
                    enter = tarr[u]; exitt = enter + L / v_a
                    over = _resv_overlap(res, enter, exitt, t0) - leg_capacity(res, L) + 1
                    pen = resv_penalty * over if over > 0 else 0.0
                    nd = d + L + zone_cost(lg, a) + pen
                    if nd < dist.get(v, np.inf):
                        dist[v] = nd; tarr[v] = exitt; prev[v] = (u, lg, fwd)
                        heapq.heappush(pq, (nd, v))
            if dst != src and dst not in prev:
                return None
            pairs, nodes = [], [dst]; cur = dst
            while cur != src:
                u, lg, fwd = prev[cur]; pairs.append((lg, fwd)); nodes.append(u); cur = u
            pairs.reverse(); nodes.reverse()
            # book this route's predicted occupancy (only within the horizon, so
            # each leg's window list stays short) so later plans deconflict from it.
            # Only the mission's primary out/return plans book -- reroutes/backups
            # query the table but do NOT add to it, else replans compound bookings
            # into a runaway loop (table -> penalty -> more reroutes -> more books).
            if book:
                tacc = t0
                t_cap = t0 + resv_horizon
                for (lg, fwd) in pairs:
                    if tacc >= t_cap:
                        break
                    seg = net.leg_seg(lg, fwd)
                    nxt = tacc + seg.length / v_a
                    leg_resv.setdefault(seg.res, []).append((tacc, nxt)); tacc = nxt
            return pairs, nodes
        dist = {src: 0.0}
        prev: dict = {}
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == dst:
                break
            if d > dist.get(u, np.inf):
                continue
            for (v, lg, fwd, L) in net._adj[u]:
                if lg in blocked:
                    continue
                res = net.leg_seg(lg, fwd).res
                nd = d + L + leg_penalty.get(res, 0.0) + zone_cost(lg, a)
                if nd < dist.get(v, np.inf):
                    dist[v] = nd
                    prev[v] = (u, lg, fwd)
                    heapq.heappush(pq, (nd, v))
        if dst != src and dst not in prev:
            return None
        pairs, nodes = [], [dst]
        cur = dst
        while cur != src:
            u, lg, fwd = prev[cur]
            pairs.append((lg, fwd)); nodes.append(u); cur = u
        return pairs[::-1], nodes[::-1]

    def plan_mission(a) -> bool:
        """Build a's route now, cost-aware. Handles round trips (out + dwell
        + return) and contingencies (backup reroute / return-to-base)."""
        rp = plan_route(a.origin, a.dest, a, book=True)
        if rp is None:
            return False
        pairs, nodes = rp
        if a.contingency and len(pairs) >= 2:
            k = max(1, min(len(pairs) - 1, int(a.cont_frac * len(pairs))))
            turn = nodes[k]
            flown = pairs[:k]
            # outbound_upto is a SEG index; ring arcs make seg count != pair
            # count, so measure segs, not pairs.
            if a.contingency == "backup":
                alt = plan_route(turn, a.dest, a, blocked={pairs[k][0]})
                if alt:
                    a.segs = list(net.route_segs(flown + alt[0]))
                    a.outbound_upto = len(a.segs)     # backup diversion is one-way
                else:
                    a.contingency = "return"
            if a.contingency == "return":
                back = plan_route(turn, a.origin, a)
                if back is None:
                    return False
                out_segs = list(net.route_segs(flown))
                a.segs = out_segs + list(net.route_segs(back[0]))
                a.outbound_upto = len(out_segs)       # return starts after flown
            a.kinds = ["leg"] * len(a.segs)
        else:
            a.segs = list(net.route_segs(pairs))
            a.kinds = ["leg"] * len(a.segs)
            a.outbound_upto = len(a.segs)
            if a.round_trip:
                ret = plan_route(a.dest, a.origin, a, book=True)
                if ret is not None:
                    ret_segs = list(net.route_segs(ret[0]))
                    a.segs.append(None); a.kinds.append("dwell")
                    a.segs.extend(ret_segs)
                    a.kinds.extend(["leg"] * len(ret_segs))
        a.route_len = sum(s.length for s in a.segs if s is not None)
        return True

    def node_zone(node, L):
        z = obj_approach if node in obj_set else node_approach
        return min(z, 0.45 * L)

    def enter_leg(a: Agent, idx: int) -> bool:
        """Move agent onto leg `idx`. Patrols fly a separate monitoring
        layer -- they neither capture nor are blocked by delivery legs.
        Deliveries: block mode needs the lane empty; spacing mode needs the
        entry zone [0, sep) clear so a new agent keeps its safe headway."""
        seg = a.segs[idx]
        if a.is_patrol:
            a.seg_idx = idx; a.s_local = 0.0; a.status = "flying"
            a.orca_xy = None; a.orca_v = None
            if a.launch_t is None:
                a.launch_t = t
            return True
        if flow_block:
            if seg.res in leg_busy:
                return False
        else:
            # merge only if the entry neighbourhood on this lane is free. Global
            # coords: a shared ring lane is ENTERED at s_offset (its entry angle),
            # not at 0, so check for occupants just behind..ahead of that point.
            entry_g = seg.s_offset
            occg = occ.get(seg.res, ())
            if ring_merge_meter and seg.ring_circ > 0.0:
                # A7: meter the ring merge on the CIRCLE. Keep >= gap clear on both
                # sides of the entry angle (wrap-aware), so no agent is injected
                # inside the headway of one that has circulated past the seam.
                circ = seg.ring_circ
                gap = max(ring_meter_gap, sep_of(a))
                p0 = entry_g % circ
                for (g, _o) in occg:
                    d = abs((g % circ) - p0)
                    if min(d, circ - d) < gap:
                        return False
            else:
                merge_gap = max(merge_frac * seg.length, sep_of(a))
                if any((entry_g - sep_of(a)) < g < (entry_g + merge_gap) for (g, _o) in occg):
                    return False
        if node_mutex:
            key = (seg.src_node, leg_level(seg))     # (node, level): same-level only
            if a.held_node != key:
                h = node_busy.get(key)
                if h is not None and h is not a:
                    return False
                node_busy[key] = a
                a.held_node = key
        if flow_block:
            leg_busy[seg.res] = a
        else:
            occ.setdefault(seg.res, []).append((0.0, a))
        a.seg_idx = idx
        a.s_local = 0.0
        a.orca_xy = None            # a fresh leg is always 1-D until the ORCA
        a.orca_v = None             # pass re-seeds it inside a ring zone
        a.status = "flying"
        if a.launch_t is None:
            a.launch_t = t
        return True

    def release_leg(res):
        leg_busy.pop(res, None)

    def leave_orca(a):
        """Drop the free 2-D state when an agent leaves a roundabout zone, so
        it is a point on a lane centreline again everywhere else."""
        a.orca_xy = None
        a.orca_v = None

    def release_node(a):
        if a.held_node is not None and node_busy.get(a.held_node) is a:
            del node_busy[a.held_node]
        a.held_node = None

    stuck_timeout = float(pget(params, "STUCK_TIMEOUT_S", 180.0))

    def try_reroute(a: Agent) -> bool:
        """Agent blocked too long at a node: divert the rest of the current
        phase via an alternate corridor (Dijkstra avoiding the jammed next
        leg). Breaks spillback deadlocks and spreads traffic onto more of
        the network. Returns True if a detour was taken."""
        i = a.seg_idx
        nxt = i + 1
        if nxt >= len(a.segs) or a.kinds[nxt] != "leg":
            return False
        cur_node = a.segs[i].dst_node
        j = nxt
        while j < len(a.segs) and a.kinds[j] == "leg":
            j += 1
        phase_dest = a.segs[j - 1].dst_node
        blocked_leg = a.segs[nxt].res.split("#")[0]
        alt = plan_route(cur_node, phase_dest, a, blocked={blocked_leg})
        if not alt:
            return False
        new_segs = net.route_segs(alt[0])
        delta = len(new_segs) - (j - nxt)
        a.segs = a.segs[:nxt] + list(new_segs) + a.segs[j:]
        a.kinds = a.kinds[:nxt] + ["leg"] * len(new_segs) + a.kinds[j:]
        if a.outbound_upto >= j:
            a.outbound_upto += delta
        a.route_len = sum(s.length for s in a.segs if s is not None)
        a.n_reroutes += 1
        a.stuck_s = 0.0
        return True

    frames, timeline = [], []
    # ---- hold-cause attribution: count agent-ticks blocked by each cause ----
    hold_cause = {"leader": 0, "node": 0, "block": 0, "launch_queue": 0}
    min_approach_global = np.inf
    min_lane_gap_global = np.inf
    peak_concurrent = peak_backlog = max_agents_per_leg = 0
    # ---- compliance against the SEPARATION STANDARD ------------------------
    # The standard has two halves and they are measured separately:
    #   HORIZONTAL  sep_floor metres between any two agents at the same flight
    #               level, whatever corridor each is on (the 50 m lateral minimum);
    #   LONGITUDINAL headway_s seconds in trail on a shared lane (the 30 s
    #               in-corridor minimum), already enforced by the car-following
    #               cap and reported as min_lane_gap_m.
    sep_std = sep_floor
    sep_violation_samples = 0          # same-level pairs closer than sep_std
    sep_violation_frames = 0
    peak_sep_violations = 0
    worst_sep_m = np.inf
    n_pair_samples = 0
    # centreline / roundabout compliance: an agent must never be inside a ring's
    # enclosed disk (that area is not a lane -- 06 raises its cost for the same
    # reason). With ring travel on this must stay at 0.
    _ring_c = np.array([net.rings[k][0] for k in net.rings], float) \
        if net.rings else np.zeros((0, 2))
    # With ORCA on, the ring is a 2-D manoeuvring AREA: using the disc is the
    # intended behaviour, and only the kept-clear ISLAND at its centre is a
    # violation. Without ORCA the ring is a 1-D circle and any incursion counts.
    _ring_ri = np.array([(orca_island * net.rings[k][1]) if orca_rings
                         else net.rings[k][2] for k in net.rings], float) \
        if net.rings else np.zeros(0)
    ring_cut_samples = 0               # agent positions inside a ring interior
    ring_pos_samples = 0
    sep_violation_log: list = []       # (t, aid_a, aid_b, gap_m, x, y, res_a, res_b, level)
    # A drone PARKED at a dock is not traffic: it sits on the pad recharging, it
    # is not airborne, and the separation standard is an AIRBORNE standard. Such
    # agents are already outside `air` (on_leg() is False while dwelling), so
    # they never enter the pairwise checks -- counted here so the exemption is
    # auditable instead of incidental, and so the docked population is visible.
    dock_exempt_samples = 0
    peak_docked = 0
    # Physical dock capacity. The scheduler books a pad before clearing a
    # mission to launch, but the plan can drift, so the pads are enforced here
    # too: an arriving drone that finds its dock full HOLDS on the last leg
    # instead of landing on top of someone. dock_full_holds counts how often
    # the strategic booking failed to protect the pad -- if it stays near zero
    # the scheduling is doing its job and this is only a safety net.
    dock_cap_rt = int(pget(params, "DOCK_CAPACITY", 0))
    dock_now: dict = defaultdict(int)          # dock -> drones parked right now
    dock_peak: dict = defaultdict(int)
    dock_full_holds = 0
    dock_hold_s = float(pget(params, "DOCK_PARK_BUFFER_S", 600.0)) + min_dest_idle
    padbook = PadBook(dock_cap_rt, dock_hold_s) if dock_cap_rt else None
    res_update_s = float(pget(params, "RESERVATION_UPDATE_S", 60.0))
    res_tol_s = float(pget(params, "RESERVATION_TOLERANCE_S", 120.0))
    next_res_update = 0.0
    n_launch_deferred_pad = 0
    pad_launch_slip = float(pget(params, "PAD_LAUNCH_SLIP_S", 300.0))
    deferred_aids: set = set()
    e_reserve_rt = float(pget(params, "ENERGY_RESERVE_PCT", 0.20))
    n_energy_short = 0
    energy_short_aids: set = set()
    n_launch_held_energy = 0

    def _energy_left_needed(a) -> float:
        """Wh still required to reach the destination dock from where the drone
        is now, priced through the cost-map at its actual cruise class."""
        upto = a.outbound_upto if a.round_trip else len(a.segs)
        if a.seg_idx >= upto:
            return 0.0
        return route_energy_wh(a.segs[:upto], a.speed, e_p0, e_cd, slowness_at,
                               s_from=a.s_local, first_idx=max(a.seg_idx, 0))

    def _live_eta(a) -> float:
        """Seconds from now to the destination dock, from where the drone
        ACTUALLY is: the remaining outbound legs, re-priced through the
        cost-map (true velocity = slowness * cruise)."""
        v = max(a.speed, 1e-3)
        upto = a.outbound_upto if a.round_trip else len(a.segs)
        total = 0.0
        for k in range(max(a.seg_idx, 0), min(upto, len(a.segs))):
            sg = a.segs[k]
            if sg is None or a.kinds[k] != "leg":
                continue
            L = float(sg.length) - (a.s_local if k == a.seg_idx else 0.0)
            if L <= 0:
                continue
            if cost_map is None:
                total += L / v
                continue
            n = max(2, int(L / 150.0) + 1)
            s0 = (a.s_local if k == a.seg_idx else 0.0)
            inv = 0.0
            for x in np.linspace(s0, float(sg.length), n):
                px, py = interp_xy(sg.xy, sg.cs, float(x))
                inv += 1.0 / min(max(slowness_at(px, py), 0.05), 1.0)
            total += (L / v) * (inv / n)
        return total

    conflict_pts_all: list = []        # every conflict location (red stars)
    total_conflict_samples = 0
    peak_conflicts = 0
    n_conflict_frames = 0
    next_sample = 0.0
    n_active = 0                       # airborne deliveries (docked ones excluded)
    active_agents: list[Agent] = []    # launched, not yet done
    all_agents = agents + patrols
    # Release order. Without a schedule this is demand order (arrival, then
    # priority). WITH a schedule it must be CTOT order: the launch gate below
    # stops at the first agent that is not due yet, which is only sound if
    # nothing behind it is due earlier. The scheduler assigns each mission the
    # earliest slot on ITS OWN departure lane, so CTOTs are NOT monotone in
    # demand order -- a mission on a quiet lane can be cleared before one queued
    # ahead of it on a busy lane. Patrols carry no CTOT and keep arrival order.
    def _release_key(a):
        return (a.sched_t if (schedule_mode and a.sched_t is not None)
                else a.arrival_t, a.priority)
    arr = sorted(all_agents, key=_release_key)
    max_del_arr = max((a.arrival_t for a in agents), default=0.0)
    arr_i = 0
    waiting: deque = deque()           # arrived, not yet launched (FIFO by arrival)
    n_waiting = 0
    last_move_t = 0.0
    gridlock = False
    last_launch_t = -1e18              # launch-rate metering (concurrency)

    t = 0.0
    while t <= horizon:
        moved = False
        # ---- spacing mode: snapshot per-lane positions + dynamic penalties ----
        if not flow_block:
            occ.clear()
            for a in active_agents:
                # the patrol is included as a DYNAMIC OBSTACLE (patrol_obstacle):
                # deliveries sharing its leg keep their time-headway behind it and
                # will not merge in front of it, but the patrol itself never yields
                # (highest priority -- see the move loop, which skips leaders for it)
                if a.on_leg() and (not a.is_patrol or patrol_obstacle):
                    seg = a.cur_seg()
                    # store GLOBAL progress (s_local + s_offset) so ring-lane
                    # occupants share a comparable coordinate for spacing
                    occ.setdefault(seg.res, []).append((a.s_local + seg.s_offset, a))
            leg_penalty.clear()
            for res, lst in occ.items():
                pen = 0.0
                held = False
                for (_g, ag) in lst:
                    L = ag.cur_seg().length
                    sl = ag.s_local                 # LOCAL position for the pheromone
                    # full cost in the first half, decays to 0 at the exit
                    pen += capture_cost if sl < 0.5 * L else \
                        capture_cost * max(0.0, (L - sl) / (0.5 * L))
                    if ag.holding:
                        held = True
                # A5: marginal-cost tolling scales the congestion (capture) term
                # so each agent internalises the delay it imposes on the leg's
                # other users (system optimum), before the jam-avoidance penalty.
                if toll_mode:
                    pen *= toll_marginal_mult
                # a leg with a holding/waiting occupant gets a HIGH penalty so
                # following agents route around the jam rather than queue on it
                if held:
                    pen += hold_penalty
                leg_penalty[res] = pen
        # ---- re-launch the single patrol unit at each scheduled interval ----
        # (one physical drone: it only relaunches once its previous sortie has
        # landed, so two patrols are never airborne at the same time)
        for a in patrols:
            due = patrol_relaunch.get(a.aid)
            if due is not None and t >= due and a not in active_agents:
                a.seg_idx = -1; a.s_local = 0.0; a.laps = 0
                a.status = "pre"; a.holding = False; a.was_holding = False
                a.battery_wh = battery_cap        # recharged at the base
                if enter_leg(a, 0):
                    a.depart_t = t; active_agents.append(a)
                    del patrol_relaunch[a.aid]
        # ---- demand: patrols launch immediately at their scheduled time
        # (separate layer, exempt from the delivery concurrency cap so they
        # never pile up and fly out together); deliveries join the queue ----
        while arr_i < len(arr) and _release_key(arr[arr_i])[0] <= t:
            a = arr[arr_i]; arr_i += 1
            if a.is_patrol:
                if enter_leg(a, 0):
                    a.depart_t = t; active_agents.append(a)
            else:
                waiting.append(a); n_waiting += 1
        # ---- launch deliveries: plan each candidate's route NOW (cost-aware,
        # routing around captured legs) and launch if its first leg is mergeable
        # (n_active counts DELIVERIES only) ----
        # metered: at most `per_tick` new launches, and none until LAUNCH_SPACING_S
        # has elapsed since the last, so departures don't all peak together
        if n_waiting and n_active < max_concurrent and (t - last_launch_t) >= launch_spacing:
            slots = max_concurrent - n_active
            per_tick = max(1, int(dt / launch_spacing)) if launch_spacing > 0 else 10 ** 9
            examined, requeue, launched = 0, [], 0
            dcb_deferred: list = []
            if dcb_mode:
                # current airborne count per origin corridor (recomputed each tick
                # so no cross-code bookkeeping is needed on completion/death)
                active_per_key: dict = {}
                for ag in active_agents:
                    if not ag.is_patrol:
                        active_per_key[ag.origin] = active_per_key.get(ag.origin, 0) + 1
            while (waiting and n_active < max_concurrent and examined < slots * 3 + 40
                   and launched < per_tick):
                if schedule_mode and waiting[0].sched_t is not None \
                        and waiting[0].sched_t > t:
                    break          # queue is in CTOT order: nothing else is due yet
                a = waiting.popleft(); n_waiting -= 1; examined += 1
                if dcb_mode and not a.is_patrol and \
                        active_per_key.get(a.origin, 0) >= dcb_cap:
                    dcb_deferred.append(a)         # corridor at capacity: rotate to back
                    continue
                if not a.is_patrol and not plan_mission(a):
                    continue                       # unreachable: drop (shouldn't happen)
                # a mission that parks needs a pad it can actually land on.
                # Book it from the LIVE route now that the route is planned; if
                # the earliest pad is far beyond its arrival the drone would
                # only hover over a full dock, so hold it on the ground instead
                # -- rescheduling driven by the live pad state, not the plan.
                # never launch into a leg the battery cannot finish -- the
                # scheduler already raised the speed class where that helped,
                # so anything caught here waits on the pad and charges
                if not a.is_patrol and a.energy_req_wh > 0.0 \
                        and a.battery_wh < a.energy_req_wh:
                    requeue.append(a); n_launch_held_energy += 1
                    continue
                wants_pad = (padbook is not None and a.round_trip
                             and not a.is_patrol)
                eta0 = 0.0
                if wants_pad:
                    eta0 = _live_eta(a)
                    t0 = padbook._earliest(a.dest, t + eta0, skip_aid=a.aid)
                    if t0 - (t + eta0) > pad_launch_slip:
                        requeue.append(a)
                        n_launch_deferred_pad += 1
                        deferred_aids.add(a.aid)
                        continue
                if enter_leg(a, 0):
                    a.depart_t = t
                    if wants_pad:                  # book only once airborne
                        padbook.book(a, a.dest, t + eta0)
                    active_agents.append(a); n_active += 1; moved = True
                    launched += 1; last_launch_t = t
                    if dcb_mode:
                        active_per_key[a.origin] = active_per_key.get(a.origin, 0) + 1
                else:
                    requeue.append(a)              # first leg busy: try again later
            for a in reversed(requeue):
                waiting.appendleft(a); n_waiting += 1
            for a in dcb_deferred:                 # capped corridors to the BACK (fair rotation)
                waiting.append(a); n_waiting += 1

        for a in active_agents:
            a.holding = False

        # ---- move each active agent along its leg ----
        # nearest-to-end first, so a leg frees up before the follower asks
        movers = [a for a in active_agents if a.on_leg()]
        movers.sort(key=lambda a: (a.cur_seg().length - a.s_local, a.priority))

        # ---- ORCA inside the roundabout zones -------------------------------
        # A widened ring is a 2-D manoeuvring area, not a 1-D circle, so agents
        # inside one are advanced by reciprocal collision avoidance rather than
        # by car-following on an arc. Each agent's PREFERRED velocity is the
        # roundabout rule -- circulate CCW at cruise until its exit bearing
        # comes up, then peel off toward the exit -- and ORCA returns the
        # closest velocity that keeps every pair >= 2*orca_radius apart and
        # everyone between the island and the ring's outer edge. The 1-D
        # movement loop below skips these agents (`orca_done`).
        orca_done: dict = {}          # id(agent) -> metres advanced this tick
        if orca_rings and net.rings:
            by_ring: dict = defaultdict(list)
            for a in movers:
                sg = a.cur_seg()
                if sg.ring_circ > 0.0 and not a.is_patrol:
                    by_ring[sg.res.split("#")[0]].append(a)
            for rbt, grp in by_ring.items():
                if rbt not in net.rings:
                    continue
                c, r_out, _r_in = net.rings[rbt]
                P, V, PR, RD, MS = [], [], [], [], []
                for a in grp:
                    sg = a.cur_seg()
                    if a.orca_xy is None:                  # entering the zone
                        a.orca_xy = np.asarray(sg.xy[0], float).copy()
                        d0 = sg.xy[min(1, len(sg.xy) - 1)] - sg.xy[0]
                        n0 = float(np.hypot(*d0)) or 1.0
                        a.orca_v = (d0 / n0) * a.speed
                    pos = a.orca_xy
                    exit_xy = np.asarray(sg.xy[-1], float)
                    rel = pos - c
                    rad = float(np.hypot(*rel)) or 1e-9
                    th = math.atan2(rel[1], rel[0])
                    th_x = math.atan2(exit_xy[1] - c[1], exit_xy[0] - c[0])
                    sweep = (th_x - th) % (2.0 * math.pi)   # CCW sweep still to fly
                    v_cruise = a.speed * (slowness_at(pos[0], pos[1])
                                          if cost_map is not None else 1.0)
                    v_cruise = max(v_cruise, 0.5)
                    if sweep <= orca_exit_rad:
                        d = exit_xy - pos                   # line up on the exit
                        nd = float(np.hypot(*d)) or 1e-9
                        pref = d / nd * v_cruise
                    else:
                        tang = np.array([-math.sin(th), math.cos(th)])   # CCW
                        radial = (c + (r_out / rad) * rel - pos)         # hold the ring
                        nr = float(np.hypot(*radial))
                        pref = tang * v_cruise
                        if nr > 1e-6:
                            pref = pref + (radial / nr) * min(nr / max(orca_tau, 1e-6),
                                                              0.35 * v_cruise)
                        npf = float(np.hypot(*pref)) or 1e-9
                        pref = pref / npf * v_cruise
                    P.append(pos); V.append(a.orca_v); PR.append(pref)
                    RD.append(orca_radius); MS.append(max(a.speed, v_cruise))
                if not P:
                    continue
                bounds = [(c, r_out + orca_half_w - orca_radius, True),
                          (c, orca_island * r_out + orca_radius, False)]
                NV = orca_step(np.array(P), np.array(V), np.array(PR), RD,
                               np.array(MS), orca_tau, dt, bounds=bounds,
                               bias_deg=orca_bias_deg)
                for a, nv in zip(grp, NV):
                    sg = a.cur_seg()
                    step = nv * dt
                    a.orca_xy = a.orca_xy + step
                    a.orca_v = nv
                    adv = float(np.hypot(*step))
                    if adv > 1e-9:
                        moved = True
                    # s_local mirrors the CCW sweep completed, so lane occupancy,
                    # the metrics and the HTML keep working unchanged. Kept
                    # MONOTONE and capped just below the arc length: the raw
                    # sweep is an angle mod 2*pi, so a hair of backward drift at
                    # the entry wraps it to ~2*pi and would read as "arrived".
                    # Only the explicit exit-proximity test below ends the arc.
                    rel = a.orca_xy - c
                    th = math.atan2(rel[1], rel[0])
                    x0 = np.asarray(sg.xy[0], float) - c
                    done = (th - math.atan2(x0[1], x0[0])) % (2.0 * math.pi)
                    full = max(sg.length, 1e-6)
                    prog = done / (2.0 * math.pi) * (2.0 * math.pi * r_out)
                    a.s_local = min(max(a.s_local, prog), 0.999 * full)
                    if adv < 0.2 * a.speed * dt:
                        a.holding = True
                        hold_cause["leader"] += 1
                    orca_ticks[0] += 1
                    # reached the exit -> hand back to the 1-D corridor model
                    if float(np.hypot(*(a.orca_xy - np.asarray(sg.xy[-1], float)))) \
                            <= orca_exit_tol:
                        a.s_local = full
                    orca_done[id(a)] = adv

        for a in movers:
            seg = a.cur_seg()
            L = seg.length
            # true velocity = base speed * slowness(here); slowness in [1e-2,1]
            if cost_map is not None and not a.is_patrol:
                px, py = interp_xy(seg.xy, seg.cs, a.s_local)
                eff_speed = a.speed * slowness_at(px, py)
            else:
                eff_speed = a.speed
            desired = a.s_local + eff_speed * dt
            cap = L
            leader_cap = node_cap = None   # for hold-cause attribution
            # spacing mode: keep the time headway behind the leader on this lane.
            # Compare GLOBAL progress (handles shared ring lanes); cap is a local
            # bound, so subtract this seg's s_offset back out.
            if not flow_block and not a.is_patrol:
                my_g = a.s_local + seg.s_offset
                if ring_wrap_follow and seg.ring_circ > 0.0:
                    # A7: find the leader by smallest FORWARD gap around the circle,
                    # so an agent past the 2*pi*r seam still yields to the occupant
                    # near angle 0 ahead of it (fixes the wrap under-separation).
                    circ = seg.ring_circ
                    fgap = min(((g - my_g) % circ for (g, o) in occ.get(seg.res, ())
                                if o is not a), default=None)
                    if fgap is not None and fgap > 1e-6:
                        leader_cap = a.s_local + (fgap - sep_of(a))
                        cap = min(cap, leader_cap)
                else:
                    leader = min((g for (g, o) in occ.get(seg.res, ())
                                  if o is not a and g > my_g + 1e-6), default=None)
                    if leader is not None:
                        leader_cap = leader - seg.s_offset - sep_of(a)
                        cap = min(cap, leader_cap)
            if node_mutex:
                lev = leg_level(seg)
                zone_src = node_zone(seg.src_node, L)
                zone_dst = node_zone(seg.dst_node, L)
                if a.held_node is not None and a.held_node[0] == seg.src_node and \
                        max(a.s_local, min(desired, L)) > zone_src:
                    release_node(a)
                approach_s = L - zone_dst
                dst_key = (seg.dst_node, lev)
                if desired > approach_s and a.held_node != dst_key:
                    h = node_busy.get(dst_key)
                    if h is None or h is a:
                        node_busy[dst_key] = a; a.held_node = dst_key
                    else:
                        node_cap = approach_s
                        cap = min(cap, approach_s)
            # A4: smooth the approach to the leader -- ramp cruise speed down over
            # a band above the separation floor instead of running full-tilt into
            # the hard cap and stopping. `room` is the metres of headroom before
            # the sep floor (= leader_cap - s_local, same for straight and ring
            # legs). The hard caps above still bound new_s, so this only ever slows
            # an agent, never speeds it past a leader.
            if speed_control and leader_cap is not None:
                room = leader_cap - a.s_local
                band = speed_ctrl_band * sep_of(a)
                if band > 1e-6:
                    frac = room / band
                    frac = 0.0 if frac < 0.0 else (1.0 if frac > 1.0 else frac)
                    desired = a.s_local + eff_speed * frac * dt
            _oadv = orca_done.get(id(a))
            if _oadv is not None:
                # the ORCA pass already flew this agent in 2-D and set s_local.
                # Fall through with no further 1-D motion so the battery drain
                # and the leg hand-off below run exactly as for everyone else.
                new_s, adv = a.s_local, _oadv
                desired = L if a.s_local >= L - 1e-6 else new_s
                eff_speed = adv / dt if dt > 0 else 0.0
            else:
                new_s = max(a.s_local, min(desired, cap))
                adv = new_s - a.s_local
            if adv > 1e-9 and not a.is_patrol:
                moved = True   # only delivery progress resets the gridlock timer
            a.dist_m += adv
            a.air_s += dt
            if desired > new_s + 1e-6:
                a.holding = True
                # attribute the hold to whichever cap actually bound this tick
                if not a.is_patrol:
                    if node_cap is not None and node_cap <= new_s + 1e-6 and \
                            (leader_cap is None or node_cap <= leader_cap + 1e-6):
                        hold_cause["node"] += 1
                    elif leader_cap is not None:
                        hold_cause["leader"] += 1
            a.s_local = new_s
            # drain battery: cruise power at the TRUE velocity, hover when held.
            # A4: when speed control is on the agent may cruise BELOW eff_speed, so
            # charge cube-law power at the ACTUAL velocity (adv/dt); baseline runs
            # full-speed-or-hover, so eff_speed keeps it byte-identical when off.
            if not a.is_patrol:
                v_pw = (adv / dt) if (speed_control and dt > 0.0) else eff_speed
                pw = (e_p0 + e_cd * v_pw ** 3) if adv > 1e-3 else e_p0
                a.battery_wh -= pw * dt / 3600.0
                if a.battery_wh <= 0.0:            # emergency landing: battery flat
                    if flow_block:
                        release_leg(seg.res)
                    if node_mutex:
                        release_node(a)
                    a.status = "dead"; a.complete_t = t; moved = True
                    continue

            if a.s_local >= L - 1e-6 and desired >= L - 1e-6:
                cur_res = seg.res
                nxt = a.seg_idx + 1
                # patrols never captured a block, so they never release one
                if a.is_patrol and nxt >= len(a.segs):
                    a.laps += 1
                    if a.laps >= patrol_laps:  # sortie done: land, relaunch in 30 min
                        a.status = "done"; a.complete_t = t
                        patrol_total_sorties += 1; patrol_total_laps += a.laps
                        # schedule the next sortie of this SAME unit on the fixed
                        # 30-min cadence (next interval boundary strictly after t)
                        if patrol_interval_s > 0:
                            k = math.floor(t / patrol_interval_s) + 1
                            patrol_relaunch[a.aid] = k * patrol_interval_s
                    elif not enter_leg(a, 0):  # loop again
                        a.laps -= 1; a.holding = True
                elif nxt >= len(a.segs):
                    if not a.is_patrol:
                        release_leg(cur_res)
                    if node_mutex:
                        release_node(a)
                    a.status = "done"; a.complete_t = t; moved = True
                elif a.kinds[nxt] == "dwell":
                    # PHYSICAL occupancy is the authority for landing: a pad
                    # that is empty now is usable now, even if someone still
                    # 20 minutes out holds a reservation for it. Making the
                    # reservation itself the gate was over-conservative -- it
                    # left drones holding over demonstrably empty pads. The
                    # reservations shape LAUNCH decisions and give warning en
                    # route; they do not ration a pad that is already free.
                    if padbook is not None:
                        if dock_now[a.dest] >= dock_cap_rt:
                            a.holding = True
                            dock_full_holds += 1
                            if not a.is_patrol:
                                hold_cause["block"] += 1
                            continue
                        padbook.book(a, a.dest, t)     # sync the books to reality
                    elif dock_cap_rt and dock_now[a.dest] >= dock_cap_rt:
                        a.holding = True           # dock full: wait on the leg
                        dock_full_holds += 1
                        if not a.is_patrol:
                            hold_cause["block"] += 1
                        continue
                    if not a.is_patrol:
                        release_leg(cur_res)
                    if node_mutex:
                        release_node(a)
                    a.seg_idx = nxt; a.status = "dwell"; a.n_charges += 1
                    dock_now[a.dest] += 1
                    dock_peak[a.dest] = max(dock_peak[a.dest], dock_now[a.dest])
                    a.dwell_left = min_dest_idle   # mandatory idle before return
                else:
                    if enter_leg(a, nxt):
                        if not a.is_patrol:
                            release_leg(cur_res)
                        moved = True
                    else:
                        a.holding = True   # block ahead busy: hold at leg end
                        if not a.is_patrol:
                            hold_cause["block"] += 1

        # ---- dock charging: recharge to CHARGE_TARGET, then depart ----
        n_air_now = n_active                    # airborne right now (docked excluded)
        for a in active_agents:
            if a.status == "dwell":
                if a.dwell_left > 0.0:                       # mandatory idle
                    a.dwell_left -= dt
                charging = a.battery_wh < charge_target - 1e-6
                if charging:                                 # still charging
                    a.battery_wh = min(battery_cap, a.battery_wh + charge_power * dt / 3600.0)
                    a.charge_s += dt
                # depart only once the idle has elapsed AND the battery is full
                # AND a return lane is free; else parked (no hover drain).
                # Leaving the pad makes it AIRBORNE again, so it has to take an
                # airborne slot -- only a drone that is still parked is exempt
                # from the cap. Without this the return legs bypass the gate and
                # the observed peak drifts above MAX_CONCURRENT.
                elif a.dwell_left <= 0.0 and n_air_now < max_concurrent \
                        and enter_leg(a, a.seg_idx + 1):
                    if dock_cap_rt:
                        dock_now[a.dest] = max(0, dock_now[a.dest] - 1)
                    if padbook is not None:
                        padbook.release(a, a.dest)     # pad free for the next drone
                    n_air_now += 1
                    moved = True

        # ---- re-plan the pad reservations from where the drones ACTUALLY are ----
        # A prediction made at launch drifts over a 15 h schedule. Re-pricing the
        # remaining route keeps every reservation honest, so a drone that has
        # lost its slot finds out en route -- and the pads it no longer needs go
        # back to the pool for someone else.
        if padbook is not None and t >= next_res_update:
            next_res_update = t + res_update_s
            for a in active_agents:
                if a.is_patrol or not a.pad_booked or a.status == "dwell":
                    continue
                if a.seg_idx >= a.outbound_upto:      # already past the dock
                    continue
                want = t + _live_eta(a)
                # ENERGY: will it still reach the dock on what is left? Priced
                # from the drone's real position and remaining charge, so a
                # slow-down or a detour that eats the margin shows up here
                # rather than as a dead battery. A drone that is short is given
                # the EARLIEST pad available -- it needs to land and charge, and
                # making it wait for a tidier slot is what kills it.
                short = a.battery_wh < _energy_left_needed(a) * (1.0 + e_reserve_rt)
                if short and not a.energy_short:
                    a.energy_short = True
                    n_energy_short += 1
                    energy_short_aids.add(a.aid)
                if short:
                    padbook.rebook(a, a.dest, t)
                elif a.pad_t0 is None or abs(want - a.pad_t0) > res_tol_s:
                    padbook.rebook(a, a.dest, want)

        # arrived-but-not-launched deliveries are blocked on the launch queue
        hold_cause["launch_queue"] += n_waiting
        # ---- hold bookkeeping + retire completed ----
        for a in active_agents:
            if a.holding:
                a.hold_s += dt
                a.stuck_s += dt
                if not a.was_holding:
                    a.n_holds += 1
            else:
                a.stuck_s = 0.0
            a.was_holding = a.holding
        # ---- reroute agents jammed at a node for too long (deadlock break) ----
        for a in active_agents:
            if (not a.is_patrol and a.stuck_s >= stuck_timeout and a.on_leg()
                    and a.s_local >= a.cur_seg().length - 1.0):
                if try_reroute(a):
                    moved = True
        done_now = [a for a in active_agents if a.status in ("done", "dead")]
        if done_now:
            active_agents = [a for a in active_agents if a.status not in ("done", "dead")]

        # MAX_CONCURRENT is an AIRBORNE cap, so recompute it from who is actually
        # flying. A drone parked at a dock is not traffic -- it holds no corridor,
        # meets no one, and is exempt from the separation checks -- so it must not
        # hold an airborne slot either. With a 30-minute dock stop and ~400 round
        # trips this is not a detail: counting parked drones against the cap would
        # idle most of the airborne budget on drones sitting on pads.
        n_active = sum(1 for a in active_agents
                       if not a.is_patrol and a.status != "dwell")
        peak_concurrent = max(peak_concurrent, n_active)
        peak_backlog = max(peak_backlog, n_waiting)
        if moved or n_active == 0:
            last_move_t = t
        elif t - last_move_t > gridlock_s:
            gridlock = True
            break

        # ---- sampling ----
        if t >= next_sample - 1e-9:
            air = [a for a in active_agents if a.on_leg()]
            n_docked = sum(1 for a in active_agents if a.status == "dwell")
            dock_exempt_samples += n_docked          # excluded from every check below
            peak_docked = max(peak_docked, n_docked)
            if air:
                xy = np.array([a.position(obj_xy) for a in air])
                outb = np.array([(a.is_patrol and 2) or (1 if a.outbound() else 0) for a in air])
                hold = np.array([a.holding for a in air])
                aids = np.array([a.aid for a in air])
                spd = np.array([a.speed_kmh for a in air])
                # flight level per agent (patrols get their own top band)
                lev = np.array([n_levels if a.is_patrol else agent_level(a) for a in air], int)
            else:
                xy = np.zeros((0, 2)); outb = np.zeros(0, int)
                hold = np.zeros(0, bool); aids = np.zeros(0, int); spd = np.zeros(0)
                lev = np.zeros(0, int)
            # metrics + conflicts use DELIVERY agents only (patrols fly a
            # separate altitude layer). A CONFLICT is a loss of the TIME
            # separation: two agents within conflict_time seconds of each
            # other (distance < conflict_time * the faster agent's speed).
            dair = [a for a in air if not a.is_patrol]
            dxy = np.array([a.position(obj_xy) for a in dair]) if dair else np.zeros((0, 2))
            dv = np.array([a.speed for a in dair])                 # m/s
            dleg = np.array([a.cur_seg().res.split("#")[0] for a in dair])  # leg id
            dlev = np.array([agent_level(a) for a in dair], int)   # flight level
            approach = np.inf
            conflict_pts = np.zeros((0, 2))
            if len(dxy) >= 2:
                diff = dxy[:, None, :] - dxy[None, :, :]
                d = np.hypot(diff[..., 0], diff[..., 1])
                iu = np.triu_indices(len(dxy), 1)
                dij = d[iu]
                # closest approach is measured among SAME-level pairs only,
                # since different levels are vertically separated
                same_level = (dlev[iu[0]] == dlev[iu[1]])
                approach = float(dij[same_level].min()) if same_level.any() else np.inf
                min_approach_global = min(min_approach_global, approach)
                # HORIZONTAL separation standard: every same-level pair, no
                # same-leg exemption -- the two lanes of a leg are built >= 50 m
                # apart, so a compliant pair passes this test on its own merit.
                n_pair_samples += int(same_level.sum())
                bad_m = (dij < sep_std) & same_level
                bad = int(bad_m.sum())
                if bad:
                    sep_violation_samples += bad
                    sep_violation_frames += 1
                    peak_sep_violations = max(peak_sep_violations, bad)
                    worst_sep_m = min(worst_sep_m, float(dij[bad_m].min()))
                    # log WHERE each loss of separation happened, so the hot
                    # spots can be read off instead of guessed at
                    if len(sep_violation_log) < 20000:
                        dres = [a.cur_seg().res for a in dair]
                        bi, bj = iu[0][bad_m], iu[1][bad_m]
                        for u in range(len(bi)):
                            ia, ib = int(bi[u]), int(bj[u])
                            mx, my = 0.5 * (dxy[ia] + dxy[ib])
                            sep_violation_log.append(
                                (t, dair[ia].aid, dair[ib].aid,
                                 float(d[ia, ib]), float(mx), float(my),
                                 dres[ia], dres[ib], int(dlev[ia])))
                thr = (conflict_time * np.maximum(dv[:, None], dv[None, :]))[iu]
                # NOT a conflict if: same leg (spatially separated lanes) OR on a
                # different flight level (vertically separated)
                same_leg = (dleg[iu[0]] == dleg[iu[1]])
                cm = (dij < thr) & (~same_leg) & same_level
                if cm.any():
                    ii, jj = iu[0][cm], iu[1][cm]
                    conflict_pts = 0.5 * (dxy[ii] + dxy[jj])
            # centreline compliance: no agent may be inside a roundabout's
            # enclosed disk (it is not a lane -- agents must circulate the ring)
            if len(_ring_c) and len(dxy):
                ring_pos_samples += len(dxy)
                dcen = np.hypot(dxy[:, None, 0] - _ring_c[None, :, 0],
                                dxy[:, None, 1] - _ring_c[None, :, 1])
                ring_cut_samples += int((dcen < (_ring_ri[None, :] - 1.0)).any(axis=1).sum())
            nconf = len(conflict_pts)
            total_conflict_samples += nconf
            peak_conflicts = max(peak_conflicts, nconf)
            if nconf:
                n_conflict_frames += 1
                if len(conflict_pts_all) < 200000:
                    conflict_pts_all.extend(conflict_pts.tolist())
            # per-lane occupancy (deliveries): peak count and min along-track
            # gap between successive agents on the same lane. Use GLOBAL progress
            # (s_local + s_offset) so occupants of a shared ring lane -- each on
            # its own arc that starts at s_local=0 at ITS entry angle -- are
            # compared on one common coordinate, mirroring the move-loop spacing;
            # without s_offset two agents far apart on the ring falsely read a 0 m
            # gap. s_offset is 0 for ordinary legs, so their metric is unchanged.
            lane_s: dict = defaultdict(list)
            for a in air:
                if not a.is_patrol:
                    seg = a.cur_seg()
                    # Agents in an ORCA zone are separated in 2-D, not in trail:
                    # their s_local is a sweep ANGLE, so two agents at the same
                    # bearing but different radii read a 0 m in-trail gap while
                    # being properly clear. Excluded here and measured by the
                    # horizontal standard instead.
                    if orca_rings and seg.ring_circ > 0.0:
                        continue
                    lane_s[seg.res].append(a.s_local + seg.s_offset)
            for ss in lane_s.values():
                max_agents_per_leg = max(max_agents_per_leg, len(ss))
                if len(ss) >= 2:
                    ss.sort()
                    min_lane_gap_global = min(min_lane_gap_global,
                                              float(np.min(np.diff(ss))))
            n_hold = int(hold.sum()) if len(hold) else 0
            n_done = int(sum(1 for a in agents if a.status == "done"))
            frames.append((t, xy, outb, hold, aids, spd, conflict_pts, lev))
            timeline.append((t, len(air), n_hold, approach, n_done, n_waiting, n_active))
            next_sample += sample_every

        # finish once every DELIVERY is resolved (patrol sorties, which are
        # scheduled across the whole horizon, do not keep the run alive)
        if t >= max_del_arr and \
                all(a.status in ("done", "dead") for a in agents):
            break
        t += dt

    n_incomplete = sum(1 for a in agents if a.status != "done")
    stats = {
        "sim_end_t": t,
        "gridlock": gridlock,
        "flow_mode": "block" if flow_block else "spacing",
        "n_incomplete": n_incomplete,
        "peak_concurrent": peak_concurrent,
        "peak_backlog": peak_backlog,
        "max_agents_per_leg": max_agents_per_leg,
        "min_lane_gap_m": None if not np.isfinite(min_lane_gap_global) else min_lane_gap_global,
        "min_approach_m": None if not np.isfinite(min_approach_global) else min_approach_global,
        "effective_separation_m": sep,
        "sep_standard_h_m": sep_std,
        "sep_standard_v_s": headway_s,
        "sep_violation_samples": sep_violation_samples,
        "sep_violation_frames": sep_violation_frames,
        "peak_sep_violations": peak_sep_violations,
        "worst_sep_m": None if not np.isfinite(worst_sep_m) else worst_sep_m,
        "n_pair_samples": n_pair_samples,
        "ring_cut_samples": ring_cut_samples,
        "ring_pos_samples": ring_pos_samples,
        "sep_violation_log": sep_violation_log,
        "dock_exempt_samples": dock_exempt_samples,
        "peak_docked": peak_docked,
        "dock_capacity": dock_cap_rt,
        "dock_peak_per_dock": dict(dock_peak),
        "dock_full_holds": dock_full_holds,
        "pad_reservation": ({
            "updates": padbook.n_update, "rebookings": padbook.n_rebook,
            "initial_bookings": padbook.n_book,
            "mean_drift_s": round(padbook.drift_s / max(padbook.n_update, 1), 1),
            "launch_deferral_events": n_launch_deferred_pad,
            "energy_short_in_flight": len(energy_short_aids),
            "launches_held_low_battery": n_launch_held_energy,
            "missions_deferred_for_pad": len(deferred_aids),
            "update_period_s": res_update_s, "tolerance_s": res_tol_s,
        } if padbook is not None else None),
        "hold_cause": dict(hold_cause),
        "schedule_mode": schedule_mode,
        "orca_rings": orca_rings,
        "orca_agent_steps": orca_ticks[0],
        "orca_radius_m": orca_radius,
        "orca_tau_s": orca_tau,
        "patrol_laps": patrol_total_laps + sum(a.laps for a in patrols if not a.done),
        "patrol_sorties": patrol_total_sorties,
        "n_legs_dir": 2 * int(net.lanes["leg_id"].nunique()),
        "conflict_time_s": conflict_time,
        "conflict_pts": conflict_pts_all,
        "total_conflict_samples": total_conflict_samples,
        "peak_conflicts": peak_conflicts,
        "n_conflict_frames": n_conflict_frames,
        "cost_map_loaded": cost_map is not None,
        "cost_map_min_slowness": None if cost_map is None else float(cm_grid.min()),
    }
    return frames, timeline, stats


# ======================================================================
# Drone energy model
# ======================================================================
def energy_params(params):
    """(P_hover W, drag coef, battery Wh). The multirotor power curve is
    P(v) = P_hover + c*v^3 (hover/induced floor + parasitic drag), so
    energy-per-metre P(v)/v = P_hover/v + c*v^2 has a minimum at an
    energy-optimal cruise speed -- flying faster or slower costs more."""
    p0 = float(pget(params, "HOVER_POWER_W", 220.0))
    cd = float(pget(params, "DRAG_POWER_COEF", 0.050))
    bat = float(pget(params, "BATTERY_WH", 200.0))
    return p0, cd, bat


def drone_power_w(v, p0, cd):
    return p0 + cd * v ** 3


def agent_energy_wh(a, p0, cd):
    """Energy for one agent: cruise power over its moving time plus hover
    power over any node-holding (hover) time."""
    move_t = a.dist_m / a.speed if a.speed > 0 else 0.0
    e_j = drone_power_w(a.speed, p0, cd) * move_t + p0 * a.hold_s
    return e_j / 3600.0


def deadline_search(complete_times, n_total, start_h=1.0, step_h=0.5):
    """Stepped search: smallest window T (from start_h, +step_h each step)
    within which ALL n_total missions finish. Returns (required_h, table)
    where table = [(T_h, n_done_by_T, pct), ...]. required_h is None if
    not all finished within the simulated horizon."""
    comp = sorted(t for t in complete_times if t is not None)
    table, required = [], None
    T = start_h * 3600.0
    max_c = comp[-1] if comp else 0.0
    while True:
        done = int(np.searchsorted(comp, T + 1e-6))
        table.append((T / 3600.0, done, 100.0 * done / n_total))
        if done >= n_total:
            required = T / 3600.0
            break
        if T > max_c + step_h * 3600.0:      # even the last finish is past T
            break
        T += step_h * 3600.0
    return required, table


# ======================================================================
# Plotting
# ======================================================================
def draw_network(ax, net: Network):
    """Light background: both lane centrelines, node circles, objectives."""
    for leg_id in net.lanes["leg_id"].unique():
        for lane in ("A", "B"):
            xy = _lane_xy(net.lanes, leg_id, lane)
            if len(xy) >= 2:
                ax.plot(xy[:, 0], xy[:, 1], "-", color="0.78", lw=1.0, zorder=1)
    obj = net.nodes[net.nodes["kind"] == "objective"]
    tn = net.nodes[net.nodes["kind"] != "objective"]
    ax.scatter(tn["x"], tn["y"], s=8, c="0.6", zorder=2)
    for r in obj.itertuples():
        is_db = str(r.net_id).startswith("DB")
        ax.scatter([r.x], [r.y], s=90, marker="s" if is_db else "^",
                   c="#c0392b" if is_db else "#1f6f3f",
                   edgecolors="k", linewidths=0.6, zorder=6)
        ax.annotate(str(r.net_id), (r.x, r.y), textcoords="offset points",
                    xytext=(6, 5), fontsize=8, weight="bold", zorder=7)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def draw_obstacles(ax, model_file, zorder=0.5):
    """Overlay step-01 obstacles (gray) and restricted / no-fly RA cells
    (orange) from the planning-model .xyz grid. Returns legend handles for
    the categories actually drawn (empty if the file is missing/unusable)."""
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    import numpy.ma as ma
    p = THIS_DIR / str(model_file)
    if not p.exists():
        return []
    df = pd.read_csv(p, sep=r"\s+")
    if not {"x", "y", "obstacle_flag"}.issubset(df.columns):
        return []
    xs = np.sort(df["x"].unique()); ys = np.sort(df["y"].unique())
    if len(xs) < 2 or len(ys) < 2:
        return []
    dx = xs[1] - xs[0]; dy = ys[1] - ys[0]
    ix = np.rint((df["x"].to_numpy() - xs[0]) / dx).astype(int)
    iy = np.rint((df["y"].to_numpy() - ys[0]) / dy).astype(int)
    obst = np.zeros((len(ys), len(xs))); obst[iy, ix] = df["obstacle_flag"].to_numpy()
    ra = np.zeros_like(obst)
    if "ra_flag" in df.columns:
        ra[iy, ix] = df["ra_flag"].to_numpy()
    extent = (xs[0] - dx / 2, xs[-1] + dx / 2, ys[0] - dy / 2, ys[-1] + dy / 2)
    handles = []
    ra_only = (ra > 0) & (obst == 0)
    if ra_only.any():
        ax.imshow(ma.masked_where(~ra_only, ra_only), extent=extent, origin="lower",
                  cmap=ListedColormap(["#e67e22"]), alpha=0.28, zorder=zorder,
                  interpolation="nearest", aspect="auto")
        handles.append(Patch(facecolor="#e67e22", alpha=0.45, label="restricted / no-fly (RA)"))
    if (obst > 0).any():
        ax.imshow(ma.masked_where(obst == 0, obst), extent=extent, origin="lower",
                  cmap=ListedColormap(["#4d4d4d"]), alpha=0.42, zorder=zorder + 0.05,
                  interpolation="nearest", aspect="auto")
        handles.append(Patch(facecolor="#4d4d4d", alpha=0.55, label="obstacle"))
    return handles


def plot_network_traffic(net: Network, agents, out_png: Path, obstacle_file=None):
    fig, ax = plt.subplots(figsize=(13, 11))
    obs_handles = draw_obstacles(ax, obstacle_file) if obstacle_file else []
    draw_network(ax, net)
    # label the traffic nodes: MAJ* (major) in red, MIN* (minor) in gray
    nid = net.nodes["net_id"].astype(str)
    for prefix, color in (("MAJ", "#c0392b"), ("MIN", "#7f8c8d")):
        for r in net.nodes[nid.str.startswith(prefix)].itertuples():
            ax.annotate(str(r.net_id), (r.x, r.y), textcoords="offset points",
                        xytext=(4, 3), fontsize=6.5, color=color, weight="bold",
                        zorder=7)
    for a in agents:
        for seg in a.segs:
            if seg is None:
                continue
            col = "#2166ac" if True else "#b2182b"
            ax.plot(seg.xy[:, 0], seg.xy[:, 1], "-", color="#3a7bd5",
                    lw=1.2, alpha=0.10, zorder=3)
    handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#c0392b",
               markeredgecolor="k", markersize=10, label="DB depot"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#1f6f3f",
               markeredgecolor="k", markersize=10, label="DK delivery"),
        Line2D([0], [0], color="#3a7bd5", lw=3, alpha=0.5, label="flown routes"),
    ] + obs_handles
    ax.legend(handles=handles, loc="upper right", framealpha=0.9)
    ax.set_title(f"Corridor traffic - {len(agents)} agent missions (flown route overlay)")
    add_map_rule(ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def plot_density(net: Network, frames, out_png: Path, conflict_pts=None):
    pts = np.vstack([xy for (_t, xy, *_r) in frames if len(xy)]) if frames else np.zeros((0, 2))
    fig, ax = plt.subplots(figsize=(13, 11))
    draw_network(ax, net)
    if len(pts):
        hb = ax.hexbin(pts[:, 0], pts[:, 1], gridsize=60, cmap="inferno",
                       mincnt=1, alpha=0.85, zorder=4)
        cb = fig.colorbar(hb, ax=ax, shrink=0.7)
        cb.set_label("agent-samples per cell")
    cp = np.array(conflict_pts) if conflict_pts is not None and len(conflict_pts) else None
    if cp is not None:
        ax.scatter(cp[:, 0], cp[:, 1], marker="*", s=90, c="red",
                   edgecolors="k", linewidths=0.4, zorder=10,
                   label=f"conflicts ({len(cp)})")
        ax.legend(loc="upper right", framealpha=0.9)
    ax.set_title("Traffic density + conflicts (loss of time separation, red stars)")
    add_map_rule(ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def plot_timeline(timeline, sep_eff, out_png: Path, peak_concurrent=None):
    # timeline rows: (t, n_air, n_hold, approach, n_done, n_waiting, n_active)
    t = np.array([r[0] for r in timeline]) / 3600.0
    n_air = np.array([r[1] for r in timeline])
    n_hold = np.array([r[2] for r in timeline])
    approach = np.array([r[3] for r in timeline], float)
    n_wait = np.array([r[5] for r in timeline])
    n_active = np.array([r[6] for r in timeline])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.plot(t, n_wait, color="#b2182b", lw=1.2, label="backlog (waiting for a leg)")
    ax1.plot(t, n_active, color="#2166ac", label="airborne (concurrent, docked excluded)")
    ax1.plot(t, n_hold, color="#d95f02", lw=1, alpha=0.8, label="holding at a node")
    if peak_concurrent is not None:
        ax1.axhline(peak_concurrent, color="#2166ac", ls="--", lw=1,
                    label=f"peak concurrent = {peak_concurrent}")
    ax1.set_ylabel("agents")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)
    ax1.set_title("Fleet activity (one agent per leg-block)")
    ax2.plot(t, np.where(np.isfinite(approach), approach, np.nan),
             color="#7570b3", lw=1, label="closest approach between agents")
    ax2.axhline(sep_eff, color="k", ls="--", lw=1, label=f"reference gap {sep_eff:.0f} m")
    ax2.set_ylabel("separation (m)")
    ax2.set_xlabel("time (h)")
    ax2.set_ylim(bottom=0)
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


# binding-constraint palette, shared by the Gantt and the rank plot
_BIND_COLOR = {
    "ready":            "#7fbf7b",   # cleared at its arrival time
    "corridor_headway": "#4a6fa5",   # 30 s in-trail spacing on its departure lane
    "airborne_cap":     "#f0a860",   # MAX_CONCURRENT airborne
    "origin_cap":       "#c07de0",   # per-origin fair share
    "dock_pad":         "#d7503a",   # no free pad at the predicted arrival
}
_BIND_LABEL = {
    "ready":            "ready (no delay)",
    "corridor_headway": "corridor headway (30 s on its departure lane)",
    "airborne_cap":     "airborne cap (MAX_CONCURRENT)",
    "origin_cap":       "origin fair share",
    "dock_pad":         "dock pad unavailable on arrival",
}


def plot_schedule(sched, params, out_png: Path):
    """FIRST-COME-FIRST-SERVED departure schedule.

    Missions are ranked by arrival (all at t=0 here, so rank = queue position)
    and served in that order: FCFS decides who gets FIRST PICK of a slot. It
    does NOT produce a monotone departure order -- each mission contends for
    its own departure lane and its own destination dock, so one bound for a
    quiet lane and an empty dock is cleared long before one ahead of it in the
    queue that is bound for a busy one. The CTOT scatter shows that spread
    directly. Colour says WHICH constraint set each slot, which is the whole
    explanation of the schedule's shape."""
    from matplotlib.collections import LineCollection
    rows = sched.get("rows") or []
    if not rows:
        return
    H = 3600.0
    rank = np.array([r["fcfs_rank"] for r in rows], float)
    arr = np.array([r["arrival_s"] for r in rows]) / H
    ctot = np.array([r["ctot_s"] for r in rows]) / H
    eta = np.array([r["eta_s"] for r in rows]) / H
    hold = np.array([r["dock_hold_s"] for r in rows]) / H
    bind = [r["binding"] for r in rows]
    cols = [_BIND_COLOR.get(b, "#999") for b in bind]

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.4, 1, 1], hspace=0.34, wspace=0.22)

    # ---- (1) Gantt: one row per mission, in FCFS order ----
    ax = fig.add_subplot(gs[0, :])
    ax.add_collection(LineCollection(
        [[(a, k), (c, k)] for a, c, k in zip(arr, ctot, rank)],
        colors="#c8ced8", linewidths=0.8, label="_"))
    ax.add_collection(LineCollection(
        [[(c, k), (c + e, k)] for c, e, k in zip(ctot, eta, rank)],
        colors="#2166ac", linewidths=0.8))
    rt = hold > 0
    if rt.any():
        ax.add_collection(LineCollection(
            [[(c + e, k), (c + e + h, k)]
             for c, e, h, k in zip(ctot[rt], eta[rt], hold[rt], rank[rt])],
            colors="#e08214", linewidths=0.8))
        ax.add_collection(LineCollection(
            [[(c + e + h, k), (c + 2 * e + h, k)]
             for c, e, h, k in zip(ctot[rt], eta[rt], hold[rt], rank[rt])],
            colors="#1f8a4c", linewidths=0.8))
    ax.scatter(ctot, rank, s=3, c="#111", linewidths=0, zorder=5)
    ax.set_xlim(0, float((ctot + 2 * eta + hold).max()) * 1.02)
    ax.set_ylim(-5, len(rows) + 5)
    ax.set_xlabel("time (h)"); ax.set_ylabel("mission, in FCFS arrival order")
    ax.set_title("First-come-first-served departure schedule "
                 f"({len(rows)} missions; black dots are the assigned CTOTs)")
    ax.grid(alpha=0.25)
    ax.legend(handles=[
        Line2D([], [], color="#c8ced8", lw=3, label="waiting on the ground (arrival → CTOT)"),
        Line2D([], [], color="#2166ac", lw=3, label="outbound flight (ETA priced by slowness)"),
        Line2D([], [], color="#e08214", lw=3, label="parked at the dock (10 min pad + 30 min charge)"),
        Line2D([], [], color="#1f8a4c", lw=3, label="return flight"),
        Line2D([], [], marker="o", ls="", ms=3, color="#111",
               label="CTOT (FCFS = first pick of a slot, not a monotone order)"),
    ], loc="lower right", fontsize=8, framealpha=0.92)

    # ---- (2) what set each slot ----
    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(ctot, rank, s=2, c=cols, linewidths=0)
    ax.set_xlabel("CTOT (h)"); ax.set_ylabel("FCFS rank")
    ax.set_title("Binding constraint per mission", fontsize=10)
    ax.grid(alpha=0.25)
    seen = [b for b in _BIND_COLOR if b in set(bind)]
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=5,
                              color=_BIND_COLOR[b], label=_BIND_LABEL[b]) for b in seen],
              fontsize=7.5, loc="upper left", framealpha=0.92)

    # ---- (3) delay distribution by cause ----
    ax = fig.add_subplot(gs[1, 1])
    delay = np.array([r["delay_s"] for r in rows]) / 60.0
    for b in seen:
        d = delay[np.array([x == b for x in bind])]
        if len(d):
            ax.hist(d, bins=40, alpha=0.75, color=_BIND_COLOR[b], label=f"{_BIND_LABEL[b]} ({len(d)})")
    ax.set_xlabel("ground delay (min)"); ax.set_ylabel("missions")
    ax.set_title(f"Ground delay by cause (mean {delay.mean():.0f} min)", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.25)

    # ---- (4) booked pad occupancy per dock ----
    ax = fig.add_subplot(gs[2, :])
    prof = sched.get("dock_profile") or {}
    cap = int(sched.get("dock_capacity", 0))
    bin_s = float(sched.get("bin_s", 5.0))
    for d, series in sorted(prof.items()):
        y = np.asarray(series, float)
        nz = np.nonzero(y)[0]
        if not len(nz):
            continue
        hi = min(len(y), int(nz[-1]) + 20)
        ax.plot(np.arange(hi) * bin_s / H, y[:hi], lw=1.0, label=d)
    if cap:
        ax.axhline(cap, color="#d7503a", ls="--", lw=1.4, label=f"capacity {cap} pads")
    ax.set_xlabel("time (h)"); ax.set_ylabel("pads booked")
    ax.set_title("Dock pad occupancy as BOOKED by the scheduler "
                 "(a mission only launches once its pad is reserved)", fontsize=10)
    ax.legend(fontsize=7, ncol=5); ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def plot_energy(agents, params, out_png: Path):
    p0, cd, bat = energy_params(params)
    kmh, _w = _speed_classes(params)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    # energy per km vs speed, with the classes and optimum marked
    vv = np.linspace(3, 25, 200)
    epm = (drone_power_w(vv, p0, cd) / vv) / 3600.0 * 1000.0   # Wh/km
    ax1.plot(vv * 3.6, epm, color="#333", lw=2)
    v_opt = (p0 / (2 * cd)) ** (1 / 3)
    ax1.axvline(v_opt * 3.6, color="#1b7837", ls="--", lw=1,
                label=f"optimum {v_opt*3.6:.0f} km/h")
    for k in kmh:
        v = k / 3.6
        ax1.scatter([k], [(drone_power_w(v, p0, cd) / v) / 3600 * 1000],
                    s=70, zorder=5, label=f"{int(k)} km/h class")
    ax1.set_xlabel("cruise speed (km/h)")
    ax1.set_ylabel("energy per km (Wh/km)")
    ax1.set_title("Drone energy vs speed (P = P_hover + c·v³)")
    ax1.grid(alpha=0.3); ax1.legend()
    # per-mission energy distribution by class
    by_cls = defaultdict(list)
    for a in agents:
        if a.status == "done":
            by_cls[a.speed_kmh].append(agent_energy_wh(a, p0, cd))
    for k in sorted(by_cls, reverse=True):
        ax2.hist(by_cls[k], bins=40, alpha=0.55, label=f"{int(k)} km/h")
    ax2.axvline(bat, color="k", ls="--", lw=1, label=f"battery {bat:.0f} Wh")
    ax2.set_xlabel("energy per completed mission (Wh)")
    ax2.set_ylabel("missions")
    ax2.set_title("Per-mission energy by UAV class")
    ax2.grid(alpha=0.3); ax2.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def make_animation(net: Network, frames, params, sep_eff, out_gif: Path):
    max_frames = int(pget(params, "MAX_GIF_FRAMES", 360))
    fps = int(pget(params, "GIF_FPS", 20))
    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        frames = [frames[i] for i in idx]

    # colour = direction (forward/back), marker = speed class (3 styles)
    cls_desc = sorted(_speed_classes(params)[0], reverse=True)  # [60,50,30]
    markers = ["o", "^", "s"]                                   # fast, mid, slow
    col_out, col_in = "#2166ac", "#b2182b"                      # forward / backward

    def sidx_of(spd):
        s = np.full(len(spd), len(cls_desc) - 1, int)
        for i, c in enumerate(cls_desc):
            s[np.isclose(spd, c)] = i
        return s

    fig, ax = plt.subplots(figsize=(12, 10))
    draw_network(ax, net)
    sc = {}                                    # (dir, class) -> scatter
    for d, col in ((1, col_out), (0, col_in)):
        for c in range(len(cls_desc)):
            sc[(d, c)] = ax.scatter([], [], s=24, c=col, marker=markers[c],
                                    edgecolors="none", zorder=8)
    sc_pat = ax.scatter([], [], s=46, c="#7b3fbf", marker="P", edgecolors="k",
                        linewidths=0.4, zorder=10)
    sc_hold = ax.scatter([], [], s=70, facecolors="none", edgecolors="#ff7f0e",
                         linewidths=1.2, zorder=9)
    sc_conf = ax.scatter([], [], s=150, marker="*", c="red", edgecolors="k",
                         linewidths=0.5, zorder=12)
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=col_out, markersize=9,
               label="forward (to DK)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=col_in, markersize=9,
               label="backward (to DB)"),
    ] + [Line2D([0], [0], marker=markers[c], color="w", markerfacecolor="0.4",
                markersize=9, label=f"{int(cls_desc[c])} km/h")
         for c in range(len(cls_desc))] + [
        Line2D([0], [0], marker="P", color="w", markerfacecolor="#7b3fbf", markersize=10,
               label="patrol"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="red", markersize=13,
               label="conflict"),
    ]
    ax.legend(handles=handles, loc="upper right", framealpha=0.9, fontsize=9)
    txt = ax.text(0.01, 0.99, "", transform=ax.transAxes, va="top", ha="left",
                  fontsize=10, family="monospace",
                  bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    ax.set_title("2D corridor multi-agent simulation")

    def update(k):
        t, xy, outb, hold, _aids, spd, cf, *_ = frames[k]
        if len(xy):
            si = sidx_of(spd)
            for d in (1, 0):
                for c in range(len(cls_desc)):
                    m = (outb == d) & (si == c)
                    sc[(d, c)].set_offsets(xy[m] if m.any() else np.zeros((0, 2)))
            sc_pat.set_offsets(xy[outb == 2] if (outb == 2).any() else np.zeros((0, 2)))
            sc_hold.set_offsets(xy[hold] if hold.any() else np.zeros((0, 2)))
        else:
            for s in list(sc.values()) + [sc_pat, sc_hold]:
                s.set_offsets(np.zeros((0, 2)))
        sc_conf.set_offsets(np.asarray(cf) if len(cf) else np.zeros((0, 2)))
        hh = int(t // 3600); mm = int((t % 3600) // 60); ss = int(t % 60)
        txt.set_text(f"t = {hh:02d}:{mm:02d}:{ss:02d}\n"
                     f"airborne = {len(xy):4d}\n"
                     f"holding  = {int(hold.sum()) if len(hold) else 0:4d}\n"
                     f"conflicts= {len(cf):4d}")
        return (*sc.values(), sc_pat, sc_hold, sc_conf, txt)

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)
    anim.save(out_gif, writer=PillowWriter(fps=fps))
    plt.close(fig)


def write_html(net: Network, frames, params, sep_eff, out_html: Path,
               velocity: dict | None = None):
    """Self-contained interactive HTML animation: the corridor network on
    a <canvas> with the agents moving along it. Play/pause, speed and a
    time scrubber; all data embedded inline so the file opens offline."""
    # network background geometry. Each lane carries its side flag (0 = A/red/
    # forward, 1 = B/blue/backward) so the browser colour-codes by direction.
    lanes = []
    for leg_id in net.lanes["leg_id"].unique():
        for lane in ("A", "B"):
            xy = _lane_xy(net.lanes, leg_id, lane)
            if len(xy) >= 2:
                lanes.append([0 if lane == "A" else 1,
                              [[round(float(x), 1), round(float(y), 1)] for x, y in xy]])
    # roundabout rings: [cx, cy, r_outer, r_inner] -> outer RED / inner BLUE
    rings = [[round(float(c[0]), 1), round(float(c[1]), 1),
              round(float(ro), 1), round(float(ri), 1)]
             for (c, ro, ri) in net.rings.values()]
    tn = net.nodes[net.nodes["kind"] != "objective"]
    tn_pts = [[round(float(r.x), 1), round(float(r.y), 1)] for r in tn.itertuples()]
    objs = []
    for r in net.nodes[net.nodes["kind"] == "objective"].itertuples():
        objs.append({"id": str(r.net_id), "x": round(float(r.x), 1),
                     "y": round(float(r.y), 1),
                     "kind": "DB" if str(r.net_id).startswith("DB") else "DK"})

    # frames: per sample, [x, y, cat(0=to DB,1=to DK,2=patrol), hold, aid, spdclass]
    cls_desc = sorted(_speed_classes(params)[0], reverse=True)   # [60,50,30]
    cls_idx = {int(round(c)): i for i, c in enumerate(cls_desc)}

    def scls_of(v):
        return cls_idx.get(int(round(v)), len(cls_desc) - 1)
    times = [round(float(t), 1) for (t, *_r) in frames]
    fdata, cdata = [], []
    for (_t, xy, outb, hold, aids, spd, cf, lev) in frames:
        rows = []
        for i in range(len(xy)):
            # per-agent row: [x, y, cat, hold, aid, spdclass, level]. `level` lets
            # the viewer draw inter-agent reference links only between agents that
            # actually interact (same flight level -- different levels are
            # vertically separated and never conflict).
            rows.append([round(float(xy[i, 0]), 1), round(float(xy[i, 1]), 1),
                         int(outb[i]), int(bool(hold[i])), int(aids[i]),
                         scls_of(spd[i]), int(lev[i])])
        fdata.append(rows)
        cdata.append([[round(float(p[0]), 1), round(float(p[1]), 1)] for p in cf])

    allx = [p[0] for _s, pts in lanes for p in pts] + [o["x"] for o in objs]
    ally = [p[1] for _s, pts in lanes for p in pts] + [o["y"] for o in objs]
    bbox = [min(allx), min(ally), max(allx), max(ally)]

    # slowness cost-map (step 09) as a background layer, if present. Stored as
    # a flat row-major grid of slowness*1000 ints to keep the file compact.
    costmap = None
    cm_path = THIS_DIR / str(pget(params, "COST_MAP_FILE",
                                  "output/07_costmap/slowness_costmap.npz"))
    if cm_path.exists():
        z = np.load(cm_path)
        g = z["slowness"].astype(float)
        costmap = {
            "x0": round(float(z["x0"]), 1), "y0": round(float(z["y0"]), 1),
            "res": round(float(z["res"]), 3),
            "nx": int(g.shape[1]), "ny": int(g.shape[0]),
            "vmin": round(float(g.min()), 4), "vmax": round(float(g.max()), 4),
            "vals": [int(round(v * 1000)) for v in g.ravel()],
        }

    # real same-lane separation: >= TIME_HEADWAY_S seconds, i.e. a distance gap
    # that scales with speed (floored at SEPARATION_M). Report the headway and
    # the resulting min..max distance across the speed classes, not the floor.
    _headway_s = float(pget(params, "TIME_HEADWAY_S", 30.0))
    _sep_floor = float(pget(params, "SEPARATION_M", 80.0))
    _gaps = [max(_headway_s * (c / 3.6), _sep_floor) for c in cls_desc]

    data = {
        "meta": {
            "n_agents": len(set(a for fr in fdata for a in [r[4] for r in fr])),
            "sep_m": sep_eff,
            "headway_s": round(_headway_s, 1),
            "gap_min_m": int(round(min(_gaps))),
            "gap_max_m": int(round(max(_gaps))),
            # inter-agent reference links: required same-lane gap per speed class
            # (index matches spdclass in each frame row), the absolute floor, and
            # the proximity radius within which a "reference" link is drawn.
            "gap_by_class": [int(round(g)) for g in _gaps],
            "sep_floor_m": int(round(_sep_floor)),
            "link_watch_m": int(round(1.6 * max(_gaps))),
            "map_w_m": int(round(bbox[2] - bbox[0])),
            "map_h_m": int(round(bbox[3] - bbox[1])),
            "shift_h": float(pget(params, "SHIFT_HOURS", 1.0)),
            "speed_classes": [int(c) for c in cls_desc],
            # ACHIEVED velocity over the completed missions: distance flown
            # divided by time taken. door_to_door spans launch -> complete so it
            # carries holds, cost-map slow-downs and the dock stop; airborne
            # divides by airborne time only. The gap between them is the time
            # the fleet spent parked.
            "velocity": velocity or {},
        },
        "bbox": bbox, "lanes": lanes, "rings": rings, "tn": tn_pts, "objs": objs,
        "times": times, "frames": fdata, "conflicts": cdata,
        "costmap": costmap,
    }
    blob = json.dumps(data, separators=(",", ":"))

    html = _HTML_TEMPLATE.replace("__DATA__", blob)
    out_html.write_text(html, encoding="utf-8")


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>2D corridor multi-agent simulation</title>
<style>
  :root { --bg:#0f1420; --panel:#1a2233; --ink:#e8edf5; --muted:#9fb0c8; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
  header { padding:14px 18px 6px; }
  h1 { font-size:18px; margin:0 0 2px; font-weight:600; }
  .sub { color:var(--muted); font-size:13px; }
  .wrap { display:flex; gap:14px; padding:10px 18px 18px; flex-wrap:wrap; }
  .stage { background:#fff; border-radius:10px; padding:8px; box-shadow:0 4px 18px rgba(0,0,0,.4); }
  canvas { display:block; width:100%; height:auto; border-radius:6px; }
  .side { min-width:220px; flex:1 1 220px; max-width:340px; }
  .card { background:var(--panel); border-radius:10px; padding:12px 14px; margin-bottom:12px; }
  .card h2 { font-size:12px; text-transform:uppercase; letter-spacing:.06em;
             color:var(--muted); margin:0 0 8px; }
  .stat { display:flex; justify-content:space-between; font-variant-numeric:tabular-nums;
          padding:3px 0; font-size:14px; }
  .stat b { font-weight:600; }
  .clock { font-size:26px; font-weight:700; font-variant-numeric:tabular-nums; letter-spacing:.03em; }
  .controls { display:flex; align-items:center; gap:10px; margin-top:8px; flex-wrap:wrap; }
  button { background:#2f6fed; color:#fff; border:0; border-radius:7px; padding:8px 14px;
           font-size:14px; cursor:pointer; }
  button.sec { background:#33405a; }
  input[type=range] { width:100%; }
  .legend { display:flex; gap:14px; flex-wrap:wrap; font-size:13px; color:var(--muted); align-items:center; }
  .dot { display:inline-block; width:11px; height:11px; border-radius:50%; margin-right:5px;
         vertical-align:-1px; border:1px solid rgba(0,0,0,.35); }
  .ring { display:inline-block; width:12px; height:12px; border-radius:50%; margin-right:5px;
          vertical-align:-1px; border:2px solid #ff8c1a; }
  label.sp { font-size:13px; color:var(--muted); }
</style>
</head>
<body>
<header>
  <h1>2D corridor network &mdash; multi-agent simulation</h1>
  <div class="sub">Agents fly DB&nbsp;(depot)&nbsp;&harr;&nbsp;DK&nbsp;(delivery); purple agents patrol every 30&nbsp;min.
    Colour = direction, shape = speed class. &ge;30&nbsp;s time separation on each lane; red stars mark conflicts.</div>
</header>
<div class="wrap">
  <div class="stage"><canvas id="cv" width="1000" height="1000"></canvas></div>
  <div class="side">
    <div class="card">
      <h2>Clock</h2>
      <div class="clock" id="clock">00:00:00</div>
      <div class="controls">
        <button id="play">Pause</button>
        <button id="restart" class="sec">Restart</button>
        <label class="sp">speed
          <select id="speed">
            <option value="0.1">0.1&times;</option>
            <option value="0.25">0.25&times;</option>
            <option value="0.5">0.5&times;</option>
            <option value="1" selected>1&times;</option>
            <option value="2">2&times;</option>
            <option value="4">4&times;</option>
            <option value="8">8&times;</option>
          </select>
        </label>
        <label class="sp"><input type="checkbox" id="cmtoggle" checked> costmap</label>
        <label class="sp"><input type="checkbox" id="linktoggle" checked> interactions</label>
        <label class="sp"><input type="checkbox" id="idtoggle" checked> agent id</label>
      </div>
      <div class="controls"><input type="range" id="scrub" min="0" value="0" step="1"></div>
    </div>
    <div class="card">
      <h2>Live</h2>
      <div class="stat"><span>Airborne</span><b id="s_air">0</b></div>
      <div class="stat"><span>Holding</span><b id="s_hold">0</b></div>
      <div class="stat"><span>Outbound &rarr; DK</span><b id="s_out">0</b></div>
      <div class="stat"><span>Inbound &rarr; DB</span><b id="s_in">0</b></div>
      <div class="stat"><span>Patrol</span><b id="s_pat">0</b></div>
      <div class="stat"><span>Conflicts</span><b id="s_conf" style="color:red">0</b></div>
      <div class="stat"><span>Too-close pairs</span><b id="s_link" style="color:#e23b3b">0</b></div>
    </div>
    <div class="card">
      <h2>Legend</h2>
      <div class="legend">
        <span><b>colour = direction:</b></span>
        <span><span class="dot" style="background:#e23b3b"></span>forward (to DK, red lane)</span>
        <span><span class="dot" style="background:#2f6fed"></span>backward (to DB, blue lane)</span>
        <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;border:2px solid #d62728;margin-right:5px;vertical-align:-1px"></span>roundabout: outer red / inner blue rings</span>
      </div>
      <div class="legend" id="speedleg" style="margin-top:6px"><span><b>shape = speed:</b></span></div>
      <div class="legend" style="margin-top:6px">
        <span><span class="dot" style="background:#7b3fbf"></span>patrol</span>
        <span><span class="ring"></span>holding</span>
        <span><span style="color:red">&#9733;</span> conflict</span>
      </div>
      <div class="legend" style="margin-top:6px">
        <span><b>interactions (same level):</b></span>
        <span><span style="display:inline-block;width:16px;height:0;border-top:2px solid #e23b3b;margin-right:5px;vertical-align:3px"></span>too close (&lt; required gap)</span>
        <span><span style="display:inline-block;width:16px;height:0;border-top:1px solid #6f93c8;margin-right:5px;vertical-align:3px"></span>nearby / keeping separation</span>
      </div>
      <div class="legend" style="margin-top:6px">
        <span><b>background = slowness:</b></span>
        <span><span class="dot" style="background:rgb(165,0,38)"></span>slow / outside corridor</span>
        <span><span class="dot" style="background:rgb(255,255,191)"></span>mid</span>
        <span><span class="dot" style="background:rgb(0,104,55)"></span>full speed</span>
      </div>
    </div>
    <div class="card" id="meta"></div>
  </div>
</div>
<script>
const DATA = __DATA__;
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const W = cv.width, H = cv.height, M = 26;
const [x0,y0,x1,y1] = DATA.bbox;
const sc = Math.min((W-2*M)/(x1-x0), (H-2*M)/(y1-y0));
const ox = (W-(x1-x0)*sc)/2, oy = (H-(y1-y0)*sc)/2;
const TX = x => ox + (x-x0)*sc;
const TY = y => H - (oy + (y-y0)*sc);   // flip Y so north is up

// ---- slowness cost-map background layer (step 09) ----
// RdYlGn-style ramp: low slowness (congested / outside corridor) = red,
// high slowness (full speed) = green. Matches slowness_costmap.png.
function slowColor(t){
  t = Math.max(0, Math.min(1, t));
  const stops=[[0.0,[165,0,38]],[0.25,[244,109,67]],[0.5,[255,255,191]],
               [0.75,[102,189,99]],[1.0,[0,104,55]]];
  let a=stops[0], b=stops[stops.length-1];
  for(let i=0;i<stops.length-1;i++){ if(t>=stops[i][0] && t<=stops[i+1][0]){ a=stops[i]; b=stops[i+1]; break; } }
  const f=(t-a[0])/((b[0]-a[0])||1);
  const c=i=>Math.round(a[1][i]+(b[1][i]-a[1][i])*f);
  return `rgb(${c(0)},${c(1)},${c(2)})`;
}
let cmCanvas=null;
function buildCostmap(){
  const CM=DATA.costmap; if(!CM) return;
  const off=document.createElement('canvas'); off.width=W; off.height=H;
  const c=off.getContext('2d');
  const span=(CM.vmax-CM.vmin)||1;
  for(let iy=0; iy<CM.ny; iy++){
    for(let ix=0; ix<CM.nx; ix++){
      const s=CM.vals[iy*CM.nx+ix]/1000;
      c.fillStyle=slowColor((s-CM.vmin)/span);
      const wx=CM.x0+ix*CM.res, wy=CM.y0+iy*CM.res;
      const X0=TX(wx), X1=TX(wx+CM.res), Y0=TY(wy), Y1=TY(wy+CM.res);
      c.fillRect(Math.min(X0,X1)-0.5, Math.min(Y0,Y1)-0.5,
                 Math.abs(X1-X0)+1, Math.abs(Y1-Y0)+1);
    }
  }
  cmCanvas=off;
}
buildCostmap();
let showCostmap = !!DATA.costmap;
let showLinks = true;     // draw inter-agent reference links (same-level proximity)
let showIds = true;       // agent index labels ON by default; the declutter below
                          // keeps them readable even with 150 airborne

function drawNetwork(){
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle = '#ffffff'; ctx.fillRect(0,0,W,H);
  if(showCostmap && cmCanvas) ctx.drawImage(cmCanvas,0,0);
  // lane centrelines, colour-coded by direction: A = red (forward),
  // B = blue (backward), matching the ring circulation convention
  ctx.lineWidth = 1.4;
  for(const pl of DATA.lanes){
    const side = pl[0], pts = pl[1];
    ctx.strokeStyle = side ? '#9fb4ee' : '#eaa6a6';   // B blue / A red (light)
    ctx.beginPath();
    for(let i=0;i<pts.length;i++){ const p=pts[i]; i?ctx.lineTo(TX(p[0]),TY(p[1])):ctx.moveTo(TX(p[0]),TY(p[1])); }
    ctx.stroke();
  }
  // roundabout rings: two concentric circulating lanes -- outer RED (forward,
  // CCW), inner BLUE (backward, CW). Agents circulate these; the centre is a
  // no-fly hole.
  if(DATA.rings){
    for(const r of DATA.rings){
      const cx=TX(r[0]), cy=TY(r[1]);
      ctx.lineWidth=1.8;
      ctx.strokeStyle='#d62728'; ctx.beginPath(); ctx.arc(cx,cy,r[2]*sc,0,7); ctx.stroke();
      ctx.strokeStyle='#2f5fe0'; ctx.beginPath(); ctx.arc(cx,cy,r[3]*sc,0,7); ctx.stroke();
    }
  }
  // traffic nodes
  ctx.fillStyle = '#aab6c6';
  for(const p of DATA.tn){ ctx.beginPath(); ctx.arc(TX(p[0]),TY(p[1]),2.5,0,7); ctx.fill(); }
  // objectives
  ctx.font = 'bold 13px system-ui'; ctx.textBaseline='middle';
  for(const o of DATA.objs){
    const X=TX(o.x), Y=TY(o.y);
    if(o.kind==='DB'){ ctx.fillStyle='#c0392b'; ctx.fillRect(X-6,Y-6,12,12);
      ctx.strokeStyle='#000'; ctx.lineWidth=1; ctx.strokeRect(X-6,Y-6,12,12); }
    else { ctx.fillStyle='#1f8f4f'; ctx.beginPath();
      ctx.moveTo(X,Y-7); ctx.lineTo(X+7,Y+6); ctx.lineTo(X-7,Y+6); ctx.closePath();
      ctx.fill(); ctx.strokeStyle='#000'; ctx.lineWidth=1; ctx.stroke(); }
    ctx.fillStyle='#111'; ctx.fillText(o.id, X+9, Y-8);
  }
}

// short trails: remember recent positions per agent id
const TRAIL = 7;
function frameAt(k){ return DATA.frames[k]; }

const COL = ['#2f6fed','#e23b3b','#7b3fbf'];        // 0 backward=blue, 1 forward=red, 2 patrol
const COLT = ['47,111,237','226,59,59','123,63,191'];

function draw(k, f){
  f = f || 0;
  drawNetwork();
  // trails
  for(let b=TRAIL;b>=1;b--){
    const kk=k-b; if(kk<0) continue;
    const a=(1-b/(TRAIL+1))*0.5;
    for(const r of frameAt(kk)){
      ctx.fillStyle = `rgba(${COLT[r[2]]},${a})`;
      ctx.beginPath(); ctx.arc(TX(r[0]),TY(r[1]),2.0,0,7); ctx.fill();
    }
  }
  // current agents: colour = direction (fwd/back), shape = speed class.
  // Smooth motion: lerp each agent toward its position in the next frame
  // (matched by agent id r[4]) by the fractional time f in [0,1).
  let air=0,hold=0,out=0,inb=0,pat=0;
  let nxt=null;
  if(f>0 && k+1<N){ nxt=new Map(); for(const q of frameAt(k+1)) nxt.set(q[4], q); }
  // pass 1: interpolated world position for every airborne agent this frame
  const AN=[];
  for(const r of frameAt(k)){
    let px=r[0], py=r[1];
    if(nxt){ const q=nxt.get(r[4]); if(q){ px=r[0]+(q[0]-r[0])*f; py=r[1]+(q[1]-r[1])*f; } }
    AN.push({px, py, cat:r[2], hold:r[3], cls:r[5], lev:(r[6]|0), aid:r[4]});
  }
  // inter-agent reference links (drawn UNDER the glyphs): every agent related to
  // the same-level neighbours whose separation constrains its motion.
  if(showLinks) drawLinks(AN); else { const el=document.getElementById('s_link'); if(el) el.textContent='-'; }
  // pass 2: draw the agent glyphs on top of the links
  for(const a of AN){
    const X=TX(a.px), Y=TY(a.py); air++;
    if(a.cat===1) out++; else if(a.cat===0) inb++; else pat++;
    ctx.fillStyle = COL[a.cat];
    if(a.cat===2){ ctx.beginPath(); ctx.arc(X,Y,5.5,0,7); ctx.fill();
      ctx.lineWidth=0.9; ctx.strokeStyle='rgba(0,0,0,.6)'; ctx.stroke(); }
    else shape(X,Y,a.cls,4.2);          // a.cls = speed class 0/1/2
    if(a.hold){ hold++; ctx.strokeStyle='#ff8c1a'; ctx.lineWidth=2.0;
      ctx.beginPath(); ctx.arc(X,Y,7.5,0,7); ctx.stroke(); }
  }
  // agent index next to each glyph. DECLUTTERED: a label is skipped when one is
  // already drawn within ID_MIN_PX on screen, so the map stays readable at full
  // extent and more ids appear as you zoom in, instead of 150 overlapping
  // numbers. Patrols are labelled P<n> -- their raw ids start at 1,000,000.
  if(showIds){
    const placed=[]; const ID_MIN_PX=26;
    ctx.font='10px ui-monospace,Menlo,monospace'; ctx.textAlign='left';
    ctx.lineWidth=2.6; ctx.strokeStyle='rgba(255,255,255,.92)';
    for(const a of AN){
      const X=TX(a.px), Y=TY(a.py);
      if(X<-20||Y<-20||X>W+20||Y>H+20) continue;
      let clash=false;
      for(const q of placed){
        if(Math.abs(q[0]-X)<ID_MIN_PX && Math.abs(q[1]-Y)<ID_MIN_PX){ clash=true; break; }
      }
      if(clash) continue;
      placed.push([X,Y]);
      const lbl = a.cat===2 ? ('P'+(a.aid-1000000)) : String(a.aid);
      ctx.strokeText(lbl, X+7, Y-6);
      ctx.fillStyle = a.cat===2 ? '#6b4fa8' : '#111';
      ctx.fillText(lbl, X+7, Y-6);
    }
  }
  // conflicts: red stars (loss of time separation)
  const cf=(DATA.conflicts&&DATA.conflicts[k])||[];
  for(const c of cf){ star(TX(c[0]),TY(c[1]),8); }
  let tt=DATA.times[k]; if(f>0 && k+1<N) tt+=(DATA.times[k+1]-DATA.times[k])*f;
  const t=tt|0;
  const hh=String((t/3600|0)).padStart(2,'0');
  const mm=String((t%3600/60|0)).padStart(2,'0');
  const ss=String(t%60).padStart(2,'0');
  document.getElementById('clock').textContent=`${hh}:${mm}:${ss}`;
  document.getElementById('s_air').textContent=air;
  document.getElementById('s_hold').textContent=hold;
  document.getElementById('s_out').textContent=out;
  document.getElementById('s_in').textContent=inb;
  document.getElementById('s_pat').textContent=pat;
  document.getElementById('s_conf').textContent=cf.length;
}

// inter-agent reference links: for each pair of SAME-level agents within the
// proximity radius, draw a line -- red (and thicker) when they are closer than
// the required same-lane gap (the larger of the two speed-based gaps), else a
// faint blue "keeping separation" link. Distances are in world metres (agent
// coords are metres), so thresholds compare directly. Also updates the
// too-close-pairs counter.
function drawLinks(AN){
  const watch=DATA.meta.link_watch_m||500, floor=DATA.meta.sep_floor_m||80;
  const gaps=DATA.meta.gap_by_class||[];
  let viol=0;
  for(let i=0;i<AN.length;i++){
    const a=AN[i];
    for(let j=i+1;j<AN.length;j++){
      const b=AN[j];
      if(a.lev!==b.lev) continue;                 // different level: vertically separated
      const dx=a.px-b.px, dy=a.py-b.py;
      const d=Math.hypot(dx,dy);
      if(d>watch) continue;
      const req=Math.max(gaps[a.cls]||floor, gaps[b.cls]||floor);
      const fade=Math.max(0.05, 1-d/watch);
      if(d<req){ viol++; ctx.strokeStyle=`rgba(226,59,59,${Math.min(0.95,0.45+0.5*fade)})`; ctx.lineWidth=2.0; }
      else { ctx.strokeStyle=`rgba(74,128,214,${0.18+0.42*fade})`; ctx.lineWidth=1.1; }
      ctx.beginPath(); ctx.moveTo(TX(a.px),TY(a.py)); ctx.lineTo(TX(b.px),TY(b.py)); ctx.stroke();
    }
  }
  const el=document.getElementById('s_link'); if(el) el.textContent=viol;
}

function star(cx,cy,R){
  ctx.beginPath();
  for(let i=0;i<10;i++){ const r=(i%2===0)?R:R*0.45; const a=Math.PI*i/5-Math.PI/2;
    const x=cx+r*Math.cos(a), y=cy+r*Math.sin(a); i?ctx.lineTo(x,y):ctx.moveTo(x,y); }
  ctx.closePath(); ctx.fillStyle='red'; ctx.fill();
  ctx.lineWidth=0.8; ctx.strokeStyle='#000'; ctx.stroke();
}
// speed-class glyph: 0=circle (fast), 1=triangle (mid), 2=square (slow)
function shape(x,y,c,r){
  ctx.beginPath();
  if(c===1){ ctx.moveTo(x,y-r); ctx.lineTo(x+r,y+r*0.8); ctx.lineTo(x-r,y+r*0.8); ctx.closePath(); }
  else if(c===2){ ctx.rect(x-r*0.9,y-r*0.9,r*1.8,r*1.8); }
  else { ctx.arc(x,y,r,0,7); }
  ctx.fill();
}

const N=DATA.frames.length;
const scrub=document.getElementById('scrub'); scrub.max=N-1;
// tpos is a CONTINUOUS frame position; base frames/s advance is interpolated
// between samples for smooth motion regardless of the sampling interval.
let tpos=0, playing=true, base=8, mult=1, last=performance.now();
function redraw(){ let k=Math.floor(tpos)%N; if(k<0) k+=N; draw(k, tpos-Math.floor(tpos)); scrub.value=k; }

function loop(now){
  requestAnimationFrame(loop);                   // keep the chain alive no matter what
  const dt=Math.max(0,(now-last)/1000); last=now; // clamp: a rAF timestamp can precede
  if(playing){ tpos += dt*base*mult; while(tpos>=N) tpos-=N; if(tpos<0) tpos=0; }
  redraw();
}
document.getElementById('play').onclick=e=>{ playing=!playing; e.target.textContent=playing?'Pause':'Play'; last=performance.now(); };
document.getElementById('restart').onclick=()=>{ tpos=0; redraw(); };
document.getElementById('speed').onchange=e=>{ mult=parseFloat(e.target.value); };
scrub.oninput=e=>{ tpos=parseInt(e.target.value); playing=false;
  document.getElementById('play').textContent='Play'; redraw(); };
const cmtoggle=document.getElementById('cmtoggle');
if(cmtoggle){ cmtoggle.disabled=!DATA.costmap; cmtoggle.checked=showCostmap;
  cmtoggle.onchange=e=>{ showCostmap=e.target.checked; redraw(); }; }
const linktoggle=document.getElementById('linktoggle');
if(linktoggle){ linktoggle.checked=showLinks;
  linktoggle.onchange=e=>{ showLinks=e.target.checked; redraw(); }; }
const idtoggle=document.getElementById('idtoggle');
if(idtoggle){ idtoggle.checked=showIds;
  idtoggle.onchange=e=>{ showIds=e.target.checked; redraw(); }; }

document.getElementById('meta').innerHTML =
  `<h2>Scenario</h2>`+
  `<div class="stat"><span>Map size</span><b>${DATA.meta.map_w_m}&times;${DATA.meta.map_h_m} m</b></div>`+
  `<div class="stat"><span>Arrival window</span><b>${DATA.meta.shift_h} h</b></div>`+
  `<div class="stat"><span>Separation</span><b>&ge;${DATA.meta.headway_s}s (${DATA.meta.gap_min_m}&ndash;${DATA.meta.gap_max_m} m)</b></div>`+
  `<div class="stat"><span>Frames</span><b>${N}</b></div>`+
  (DATA.meta.velocity && DATA.meta.velocity.door_to_door_kmh ?
    `<div class="stat"><span>Velocity (door&#8209;to&#8209;door)</span><b>${DATA.meta.velocity.door_to_door_kmh.toFixed(1)} km/h</b></div>`+
    `<div class="stat"><span>Velocity (airborne)</span><b>${DATA.meta.velocity.airborne_kmh.toFixed(1)} km/h</b></div>`+
    `<div class="stat"><span>Nominal cruise</span><b>${DATA.meta.velocity.nominal_cruise_kmh.toFixed(1)} km/h</b></div>`+
    `<div class="stat"><span>Flown / time</span><b>${DATA.meta.velocity.total_distance_km.toFixed(0)} km / ${DATA.meta.velocity.total_time_h.toFixed(1)} h</b></div>`
    : '');

// speed-class shape legend (circle / triangle / square = fast .. slow)
const SHAPES=['&#9679;','&#9650;','&#9632;'];
document.getElementById('speedleg').innerHTML =
  `<span><b>shape = speed:</b></span>` +
  (DATA.meta.speed_classes||[]).map((v,i)=>`<span>${SHAPES[i]||'&#9679;'} ${v} km/h</span>`).join('');

redraw(); requestAnimationFrame(loop);
</script>
</body>
</html>
"""


# ======================================================================
# Per-agent route HTML (output/09.../agent_route/)
# ======================================================================
def write_agent_routes(net: Network, frames, params, out_dir: Path, missions_csv: Path):
    """One interactive HTML per agent under out_dir/'agent_route/', to inspect
    the travel of that single agent: its flown path, a smooth moving marker
    with a time scrubber, and its mission stats. A shared network.js holds the
    map so each per-agent file stays small; an index.html links them all."""
    ar = out_dir / "agent_route"
    ar.mkdir(parents=True, exist_ok=True)

    # ---- shared network geometry (mirrors write_html) ----
    # each lane carries its side flag (0 = A/red/forward, 1 = B/blue/backward)
    lanes = []
    for leg_id in net.lanes["leg_id"].unique():
        for lane in ("A", "B"):
            xy = _lane_xy(net.lanes, leg_id, lane)
            if len(xy) >= 2:
                lanes.append([0 if lane == "A" else 1,
                              [[round(float(x), 1), round(float(y), 1)] for x, y in xy]])
    rings = [[round(float(c[0]), 1), round(float(c[1]), 1),
              round(float(ro), 1), round(float(ri), 1)]
             for (c, ro, ri) in net.rings.values()]
    tn = net.nodes[net.nodes["kind"] != "objective"]
    tn_pts = [[round(float(r.x), 1), round(float(r.y), 1)] for r in tn.itertuples()]
    objs = []
    for r in net.nodes[net.nodes["kind"] == "objective"].itertuples():
        objs.append({"id": str(r.net_id), "x": round(float(r.x), 1),
                     "y": round(float(r.y), 1),
                     "kind": "DB" if str(r.net_id).startswith("DB") else "DK"})
    allx = [p[0] for _s, pts in lanes for p in pts] + [o["x"] for o in objs]
    ally = [p[1] for _s, pts in lanes for p in pts] + [o["y"] for o in objs]
    bbox = [min(allx), min(ally), max(allx), max(ally)]
    # ---- global timeline shared by every agent page: the slowness costmap,
    #      ALL agents per frame, and conflicts, so each replay can show the
    #      traffic AROUND the focal agent on the same map as the main view ----
    cls_desc = sorted(_speed_classes(params)[0], reverse=True)
    cls_idx = {int(round(c)): i for i, c in enumerate(cls_desc)}

    def scls(v):
        return cls_idx.get(int(round(v)), len(cls_desc) - 1)

    times, fdata, cdata = [], [], []
    focal: dict[int, list] = defaultdict(list)   # aid -> [[kg, x, y, cat, hold, level], ...]
    for kg, (t, xy, outb, hold, aids_f, spd, cf, lev) in enumerate(frames):
        times.append(round(float(t), 1))
        rows = []
        for i in range(len(xy)):
            aid = int(aids_f[i])
            x = round(float(xy[i, 0]), 1)
            y = round(float(xy[i, 1]), 1)
            cat = int(outb[i])
            h = int(bool(hold[i]))
            # row: [x, y, cat, hold, aid, spdclass, level]. `level` lets a focal
            # page draw reference links only to SAME-level neighbours (the ones it
            # can actually conflict with).
            rows.append([x, y, cat, h, aid, scls(spd[i]), int(lev[i])])
            focal[aid].append([kg, x, y, cat, h, int(lev[i])])
        fdata.append(rows)
        cdata.append([[round(float(p[0]), 1), round(float(p[1]), 1)] for p in cf])

    costmap = None
    cm_path = THIS_DIR / str(pget(params, "COST_MAP_FILE",
                                  "output/07_costmap/slowness_costmap.npz"))
    if cm_path.exists():
        z = np.load(cm_path)
        g = z["slowness"].astype(float)
        costmap = {"x0": round(float(z["x0"]), 1), "y0": round(float(z["y0"]), 1),
                   "res": round(float(z["res"]), 3), "nx": int(g.shape[1]),
                   "ny": int(g.shape[0]), "vmin": round(float(g.min()), 4),
                   "vmax": round(float(g.max()), 4),
                   "vals": [int(round(v * 1000)) for v in g.ravel()]}

    levels = {"n": int(pget(params, "FLIGHT_LEVELS", 4)),
              "base_z": float(pget(params, "BASE_LEVEL_M", 60.0)),
              "sep": float(pget(params, "LEVEL_SEP_M", 30.0))}
    # inter-agent reference links (focal page): required same-lane gap per speed
    # class (index matches spdclass in each frame row), the floor, and the
    # proximity radius within which a "reference" link to the focal agent is drawn.
    _hw = float(pget(params, "TIME_HEADWAY_S", 30.0))
    _flr = float(pget(params, "SEPARATION_M", 80.0))
    _gp = [max(_hw * (c / 3.6), _flr) for c in cls_desc]
    link_meta = {"gap_by_class": [int(round(g)) for g in _gp],
                 "sep_floor_m": int(round(_flr)),
                 "link_watch_m": int(round(1.6 * max(_gp)))}
    net_json = json.dumps({"bbox": bbox, "lanes": lanes, "rings": rings,
                           "tn": tn_pts, "objs": objs,
                           "costmap": costmap, "times": times, "frames": fdata,
                           "conflicts": cdata, "levels": levels,
                           "meta": link_meta}, separators=(",", ":"))
    (ar / "network.js").write_text("window.NETWORK=" + net_json + ";\n", encoding="utf-8")

    mrec = {}
    if missions_csv.exists():
        for r in pd.read_csv(missions_csv).itertuples():
            mrec[int(r.agent_id)] = r

    include_patrol = bool(pget(params, "AGENT_ROUTE_INCLUDE_PATROL", False))
    cap = int(pget(params, "AGENT_ROUTE_MAX", -1))
    patrol_kmh = float(pget(params, "PATROL_SPEED_KMH", 50.0))
    aids = sorted(focal)
    if not include_patrol:
        aids = [a for a in aids if a < 1_000_000]
    if cap and cap > 0:
        aids = aids[:cap]

    idx = []
    for aid in aids:
        fo = focal[aid]                       # [[kg, x, y, cat, hold], ...]
        m = mrec.get(aid)
        is_patrol = aid >= 1_000_000
        if m is not None:
            s = {"agent": int(aid), "kind": "patrol" if is_patrol else str(m.mission),
                 "origin": str(m.origin), "dest": str(m.dest),
                 "speed_kmh": float(m.speed_kmh), "launch_s": float(m.launch_s),
                 "complete_s": float(m.complete_s), "flight_time_s": float(m.flight_time_s),
                 "route_len_m": float(m.route_len_m), "flown_m": float(m.flown_m),
                 # achieved velocity: flown / mission time, and flown / airborne
                 # time. The difference between them is the dock stop.
                 "velocity_kmh": None if pd.isna(getattr(m, "velocity_kmh", None))
                                 else float(m.velocity_kmh),
                 "air_velocity_kmh": None if pd.isna(getattr(m, "air_velocity_kmh", None))
                                     else float(m.air_velocity_kmh),
                 "dock_idle_s": 0.0 if pd.isna(getattr(m, "dock_idle_s", None))
                                else float(m.dock_idle_s),
                 "n_holds": int(m.n_holds), "hold_s": float(m.hold_s),
                 "energy_wh": float(m.energy_wh), "battery_end_pct": float(m.battery_end_pct),
                 "completed": bool(m.completed), "battery_dead": bool(m.battery_dead)}
        else:
            k0 = fo[0][0] if fo else 0
            k1 = fo[-1][0] if fo else 0
            s = {"agent": int(aid), "kind": "patrol" if is_patrol else "?",
                 "origin": "-", "dest": "-",
                 "speed_kmh": patrol_kmh if is_patrol else 0.0,
                 "launch_s": times[k0] if fo else 0.0, "complete_s": times[k1] if fo else 0.0,
                 "flight_time_s": (times[k1] - times[k0]) if fo else 0.0, "route_len_m": 0.0,
                 "flown_m": 0.0, "velocity_kmh": None, "air_velocity_kmh": None,
                 "dock_idle_s": 0.0,
                 "n_holds": 0, "hold_s": 0.0, "energy_wh": 0.0,
                 "battery_end_pct": 0.0, "completed": True, "battery_dead": False}
        F = {"aid": int(aid), "focal": fo, "stats": s,
             "fcls": scls(s["speed_kmh"])}   # focal speed class -> its required gap
        html = (_AGENT_HTML_TEMPLATE
                .replace("__AID__", str(aid))
                .replace("__F__", json.dumps(F, separators=(",", ":"))))
        (ar / f"agent_{aid}.html").write_text(html, encoding="utf-8")
        idx.append(s)

    rows = "".join(
        f"<tr><td><a href='agent_{s['agent']}.html'>agent {s['agent']}</a></td>"
        f"<td>{s['kind']}</td><td>{s['origin']}&rarr;{s['dest']}</td>"
        f"<td>{s['speed_kmh']:.0f}</td><td>{s['flight_time_s']/60:.1f}</td>"
        f"<td>{s['route_len_m']:.0f}</td><td>{s['n_holds']}</td>"
        f"<td>{'yes' if s['completed'] else ('dead' if s['battery_dead'] else 'no')}</td></tr>"
        for s in idx)
    index = _AGENT_INDEX_TEMPLATE.replace("__N__", str(len(idx))).replace("__ROWS__", rows)
    (ar / "index.html").write_text(index, encoding="utf-8")
    return len(idx)


_AGENT_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent __AID__ route</title>
<style>
  :root{--bg:#0f1420;--panel:#1a2233;--ink:#e8edf5;--muted:#9fb0c8;}
  *{box-sizing:border-box;} body{margin:0;background:var(--bg);color:var(--ink);
    font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;}
  header{padding:12px 16px 4px;} h1{font-size:17px;margin:0;}
  a{color:#6ab0ff;} .sub{color:var(--muted);font-size:13px;margin-top:2px;}
  .wrap{display:flex;gap:14px;padding:10px 16px 18px;flex-wrap:wrap;}
  .stage{background:#fff;border-radius:10px;padding:8px;box-shadow:0 4px 18px rgba(0,0,0,.4);}
  canvas{display:block;width:100%;height:auto;border-radius:6px;}
  .side{min-width:230px;flex:1 1 230px;max-width:340px;}
  .card{background:var(--panel);border-radius:10px;padding:12px 14px;margin-bottom:12px;}
  .card h2{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:0 0 8px;}
  .stat{display:flex;justify-content:space-between;font-variant-numeric:tabular-nums;padding:3px 0;font-size:14px;}
  .clock{font-size:24px;font-weight:700;font-variant-numeric:tabular-nums;}
  .controls{display:flex;align-items:center;gap:10px;margin-top:8px;flex-wrap:wrap;}
  button{background:#2f6fed;color:#fff;border:0;border-radius:7px;padding:8px 14px;font-size:14px;cursor:pointer;}
  button.sec{background:#33405a;} input[type=range]{width:100%;} label.sp{font-size:13px;color:var(--muted);}
</style></head><body>
<header><h1>Agent __AID__ &mdash; route replay</h1>
  <div class="sub"><a href="index.html">&larr; all agents</a> &nbsp; <b style="color:#e11">&#9670; big red diamond</b> = this agent; small dots = other agents sharing the air; lines = this agent's <b style="color:#4a80d6">reference links</b> to same-level neighbours (<b style="color:#e23b3b">red = too close</b>); <b style="color:#8e2fb8">route forward = purple</b>, <b style="color:#12246b">backward = navy</b>; green/red = start/end; amber circle = holding/waiting; red star = conflict.</div>
</header>
<div class="wrap">
  <div class="stage"><canvas id="cv" width="900" height="900"></canvas></div>
  <div class="side">
    <div class="card"><h2>Clock</h2><div class="clock" id="clock">00:00:00</div>
      <div class="stat"><span>Status</span><b id="status">-</b></div>
      <div class="stat"><span>Flight level</span><b id="flvl">-</b></div>
      <div class="controls"><button id="play">Pause</button><button id="restart" class="sec">Restart</button>
        <label class="sp">speed<select id="speed">
          <option value="0.25">0.25&times;</option><option value="0.5">0.5&times;</option>
          <option value="1" selected>1&times;</option><option value="2">2&times;</option>
          <option value="4">4&times;</option><option value="8">8&times;</option></select></label>
        <label class="sp"><input type="checkbox" id="cmtoggle" checked> costmap</label>
        <label class="sp"><input type="checkbox" id="linktoggle" checked> interactions</label>
        <label class="sp"><input type="checkbox" id="idtoggle" checked> agent id</label></div>
      <div class="stat"><span>Other agents now</span><b id="near">0</b></div>
      <div class="stat"><span>Too-close now</span><b id="tclose" style="color:#e23b3b">0</b></div>
      <div class="controls"><input type="range" id="scrub" min="0" value="0" step="1"></div></div>
    <div class="card" id="info"></div>
  </div>
</div>
<script src="network.js"></script>
<script>
const F=__F__, NET=window.NETWORK;
const FO=F.focal;                 // [[kg, x, y, cat, hold], ...]
const FR=NET.frames||[], TIMES=NET.times||[], CONF=NET.conflicts||[];
const cv=document.getElementById('cv'), ctx=cv.getContext('2d');
const W=cv.width,H=cv.height,M=22;
const [x0,y0,x1,y1]=NET.bbox;
const sc=Math.min((W-2*M)/(x1-x0),(H-2*M)/(y1-y0));
const ox=(W-(x1-x0)*sc)/2, oy=(H-(y1-y0)*sc)/2;
const TX=x=>ox+(x-x0)*sc, TY=y=>H-(oy+(y-y0)*sc);
const COL=['#e23b3b','#2f6fed','#7b3fbf'];
// route-line colours: forward (outbound) = purple, backward (return) = navy
const PCOL=['#12246b','#8e2fb8','#7b3fbf'];   // [0 back=navy, 1 fwd=purple, 2 patrol]

// slowness cost-map background (shared via network.js), same ramp as main view
function slowColor(t){ t=Math.max(0,Math.min(1,t));
  const st=[[0.0,[165,0,38]],[0.25,[244,109,67]],[0.5,[255,255,191]],[0.75,[102,189,99]],[1.0,[0,104,55]]];
  let a=st[0],b=st[st.length-1];
  for(let i=0;i<st.length-1;i++){ if(t>=st[i][0]&&t<=st[i+1][0]){a=st[i];b=st[i+1];break;} }
  const f=(t-a[0])/((b[0]-a[0])||1), c=i=>Math.round(a[1][i]+(b[1][i]-a[1][i])*f);
  return `rgb(${c(0)},${c(1)},${c(2)})`; }
let cmCanvas=null;
function buildCostmap(){ const CM=NET.costmap; if(!CM) return;
  const off=document.createElement('canvas'); off.width=W; off.height=H; const c=off.getContext('2d');
  const span=(CM.vmax-CM.vmin)||1;
  for(let iy=0;iy<CM.ny;iy++) for(let ix=0;ix<CM.nx;ix++){
    const s=CM.vals[iy*CM.nx+ix]/1000; c.fillStyle=slowColor((s-CM.vmin)/span);
    const wx=CM.x0+ix*CM.res, wy=CM.y0+iy*CM.res;
    const X0=TX(wx),X1=TX(wx+CM.res),Y0=TY(wy),Y1=TY(wy+CM.res);
    c.fillRect(Math.min(X0,X1)-0.5,Math.min(Y0,Y1)-0.5,Math.abs(X1-X0)+1,Math.abs(Y1-Y0)+1); }
  cmCanvas=off; }
buildCostmap();
let showCostmap=!!NET.costmap;
let showFLinks=true;    // draw this agent's reference links to same-level neighbours
let showIds = true;       // neighbour agent ids: ON here -- few enough to stay legible

function bg(){
  ctx.clearRect(0,0,W,H); ctx.fillStyle='#fff'; ctx.fillRect(0,0,W,H);
  if(showCostmap&&cmCanvas) ctx.drawImage(cmCanvas,0,0);
  // lane centrelines colour-coded by direction: A=red(forward), B=blue(backward)
  ctx.lineWidth=1.1;
  for(const pl of NET.lanes){ const side=pl[0], pts=pl[1];
    ctx.strokeStyle = showCostmap
      ? (side?'rgba(120,150,240,.7)':'rgba(220,120,120,.7)')
      : (side?'#9fb4ee':'#eaa6a6');
    ctx.beginPath();
    for(let i=0;i<pts.length;i++){const p=pts[i]; i?ctx.lineTo(TX(p[0]),TY(p[1])):ctx.moveTo(TX(p[0]),TY(p[1]));}
    ctx.stroke(); }
  // roundabout rings: outer red (forward) / inner blue (backward)
  if(NET.rings){ ctx.lineWidth=1.6;
    for(const r of NET.rings){ const cx=TX(r[0]),cy=TY(r[1]);
      ctx.strokeStyle='#d62728'; ctx.beginPath(); ctx.arc(cx,cy,r[2]*sc,0,7); ctx.stroke();
      ctx.strokeStyle='#2f5fe0'; ctx.beginPath(); ctx.arc(cx,cy,r[3]*sc,0,7); ctx.stroke(); } }
  ctx.font='bold 12px system-ui'; ctx.textBaseline='middle';
  for(const o of NET.objs){ const X=TX(o.x),Y=TY(o.y);
    if(o.kind==='DB'){ctx.fillStyle='#c0392b';ctx.fillRect(X-5,Y-5,10,10);}
    else{ctx.fillStyle='#1f8f4f';ctx.beginPath();ctx.moveTo(X,Y-6);ctx.lineTo(X+6,Y+5);ctx.lineTo(X-6,Y+5);ctx.closePath();ctx.fill();}
    ctx.fillStyle=showCostmap?'#111':'#444';ctx.fillText(o.id,X+8,Y-7); }
}
function drawPath(){          // focal agent's flown path: fwd=purple, back=navy
  for(let i=1;i<FO.length;i++){ const a=FO[i-1],b=FO[i];
    ctx.strokeStyle=PCOL[b[3]]; ctx.lineWidth=2.6; ctx.globalAlpha=0.85;
    ctx.beginPath(); ctx.moveTo(TX(a[1]),TY(a[2])); ctx.lineTo(TX(b[1]),TY(b[2])); ctx.stroke(); }
  ctx.globalAlpha=1;
  if(FO.length){ const s=FO[0], e=FO[FO.length-1];
    ctx.fillStyle='#1f8f4f'; ctx.beginPath(); ctx.arc(TX(s[1]),TY(s[2]),6,0,7); ctx.fill();
    ctx.strokeStyle='#000'; ctx.lineWidth=1; ctx.stroke();
    ctx.fillStyle='#c0392b'; ctx.beginPath(); ctx.arc(TX(e[1]),TY(e[2]),6,0,7); ctx.fill(); ctx.stroke(); }
}
function star(cx,cy,R){ ctx.beginPath();
  for(let i=0;i<10;i++){ const r=(i%2===0)?R:R*0.45,a=Math.PI*i/5-Math.PI/2;
    const x=cx+r*Math.cos(a),y=cy+r*Math.sin(a); i?ctx.lineTo(x,y):ctx.moveTo(x,y); }
  ctx.closePath(); ctx.fillStyle='red'; ctx.fill(); ctx.lineWidth=0.7; ctx.strokeStyle='#000'; ctx.stroke(); }
function diamond(X,Y,R){ ctx.beginPath();
  ctx.moveTo(X,Y-R); ctx.lineTo(X+R,Y); ctx.lineTo(X,Y+R); ctx.lineTo(X-R,Y); ctx.closePath();
  ctx.fillStyle='#e11111'; ctx.fill(); ctx.lineWidth=2; ctx.strokeStyle='#000'; ctx.stroke(); }
// other agents sharing the airspace at global frame kg (interpolated by f)
function others(kg,f){ const cur=FR[kg]||[]; let nxt=null, n=0;
  if(f>0 && kg+1<FR.length){ nxt=new Map(); for(const q of FR[kg+1]) nxt.set(q[4],q); }
  const idPlaced=[];   // declutter: drop a label that would sit on another
  for(const r of cur){ if(r[4]===F.aid) continue; n++;
    let px=r[0],py=r[1];
    if(nxt){ const q=nxt.get(r[4]); if(q){ px=r[0]+(q[0]-r[0])*f; py=r[1]+(q[1]-r[1])*f; } }
    const X=TX(px),Y=TY(py);
    if(r[3]){ // HOLDING status: amber circle covering the symbol
      ctx.fillStyle='#ff8c1a'; ctx.beginPath(); ctx.arc(X,Y,3.6,0,7); ctx.fill();
      ctx.lineWidth=0.8; ctx.strokeStyle='#5a2d00'; ctx.stroke();
    } else {
      ctx.globalAlpha=0.55; ctx.fillStyle=COL[r[2]];
      ctx.beginPath(); ctx.arc(X,Y,2.7,0,7); ctx.fill(); ctx.globalAlpha=1;
    }
    // label WHO the neighbour is -- on this page the whole point of the other
    // agents is identifying the one this drone is interacting with
    if(showIds){
      let clash=false;
      for(const q of idPlaced){
        if(Math.abs(q[0]-X)<18 && Math.abs(q[1]-Y)<12){ clash=true; break; } }
      if(!clash){
        idPlaced.push([X,Y]);
        const lbl = r[2]===2 ? ('P'+(r[4]-1000000)) : String(r[4]);
        ctx.font='10px ui-monospace,Menlo,monospace'; ctx.textAlign='left';
        ctx.lineWidth=2.6; ctx.strokeStyle='rgba(255,255,255,.92)';
        ctx.strokeText(lbl, X+5, Y-5);
        ctx.fillStyle = r[2]===2 ? '#6b4fa8' : '#333';
        ctx.fillText(lbl, X+5, Y-5);
      }
    } }
  for(const c of (CONF[kg]||[])) star(TX(c[0]),TY(c[1]),6);
  return n; }
// this agent's reference links: a line from the focal agent (fx,fy at level flev)
// to every SAME-level neighbour within the proximity radius -- red + thick when
// closer than the required same-lane gap (the larger of the two speed-based
// gaps), else a faint blue "keeping separation" link. Returns the too-close count.
function focalLinks(fx,fy,flev,kg,f){
  const M=NET.meta||{}, watch=M.link_watch_m||800, floor=M.sep_floor_m||80, gaps=M.gap_by_class||[];
  const cur=FR[kg]||[]; let nxt=null, viol=0;
  if(f>0 && kg+1<FR.length){ nxt=new Map(); for(const q of FR[kg+1]) nxt.set(q[4],q); }
  for(const r of cur){
    if(r[4]===F.aid) continue;
    const rlev = (r.length>6)?(r[6]|0):0;
    if(rlev!==flev) continue;                 // different level: vertically separated
    let px=r[0],py=r[1];
    if(nxt){ const q=nxt.get(r[4]); if(q){ px=r[0]+(q[0]-r[0])*f; py=r[1]+(q[1]-r[1])*f; } }
    const d=Math.hypot(fx-px,fy-py);
    if(d>watch) continue;
    const req=Math.max(gaps[F.fcls]||floor, gaps[r[5]]||floor);
    const fade=Math.max(0.05,1-d/watch);
    if(d<req){ viol++; ctx.strokeStyle=`rgba(226,59,59,${Math.min(0.95,0.5+0.45*fade)})`; ctx.lineWidth=2.4; }
    else { ctx.strokeStyle=`rgba(74,128,214,${0.22+0.45*fade})`; ctx.lineWidth=1.4; }
    ctx.beginPath(); ctx.moveTo(TX(fx),TY(fy)); ctx.lineTo(TX(px),TY(py)); ctx.stroke();
  }
  return viol; }
function draw(p){
  bg(); drawPath();
  const i=Math.min(Math.floor(p),FO.length-1), f=p-Math.floor(p);
  const a=FO[i], b=FO[Math.min(i+1,FO.length-1)];
  const kg=a[0], cont=(b[0]===kg+1), nf=cont?f:0;
  const n=others(kg,nf);
  const x=a[1]+(b[1]-a[1])*f, y=a[2]+(b[2]-a[2])*f, X=TX(x), Y=TY(y);
  const flev=a[5]|0;
  const tc = showFLinks ? focalLinks(x,y,flev,kg,nf) : 0;
  const tcl=document.getElementById('tclose'); if(tcl) tcl.textContent = showFLinks ? tc : '-';
  const holding=!!a[4];
  if(holding){ // HOLD/WAIT status: an amber circle COVERING the current agent
    ctx.fillStyle='#ff8c1a'; ctx.beginPath(); ctx.arc(X,Y,15,0,7); ctx.fill();
    ctx.lineWidth=1.8; ctx.strokeStyle='#5a2d00'; ctx.stroke();
  }
  diamond(X,Y,11);                 // red diamond marks the current agent
  const t=(TIMES[kg]+((TIMES[Math.min(kg+1,TIMES.length-1)]-TIMES[kg])*nf))|0;
  const hh=String(t/3600|0).padStart(2,'0'),mm=String(t%3600/60|0).padStart(2,'0'),ss=String(t%60).padStart(2,'0');
  document.getElementById('clock').textContent=`${hh}:${mm}:${ss}`;
  document.getElementById('near').textContent=n;
  const st=document.getElementById('status');
  if(st){ st.textContent=holding?'HOLDING / WAIT':'flying'; st.style.color=holding?'#ff8c1a':'#6ab0ff'; }
  const fl=document.getElementById('flvl'), LV=NET.levels;
  if(fl && LV){ const lv=a[5]|0; fl.textContent=`FL${lv} (${(LV.base_z+lv*LV.sep)|0} m)`; }
}
const NP=Math.max(FO.length,1);
const scrub=document.getElementById('scrub'); scrub.max=NP-1;
let pos=0,playing=true,base=3,mult=1,last=performance.now();
function redraw(){ draw(pos); scrub.value=Math.floor(pos); }
function loop(now){ requestAnimationFrame(loop);
  const dt=Math.max(0,(now-last)/1000); last=now;
  if(playing){ pos+=dt*base*mult; if(pos<0)pos=0;
    if(pos>=NP-1){ pos=NP-1; playing=false; document.getElementById('play').textContent='Play'; } }
  redraw(); }
document.getElementById('play').onclick=e=>{ if(pos>=NP-1)pos=0; playing=!playing; e.target.textContent=playing?'Pause':'Play'; last=performance.now(); };
document.getElementById('restart').onclick=()=>{ pos=0; playing=true; document.getElementById('play').textContent='Pause'; };
document.getElementById('speed').onchange=e=>{ mult=parseFloat(e.target.value); };
scrub.oninput=e=>{ pos=parseInt(e.target.value); playing=false; document.getElementById('play').textContent='Play'; redraw(); };
const cmtoggle=document.getElementById('cmtoggle');
if(cmtoggle){ cmtoggle.disabled=!NET.costmap; cmtoggle.checked=showCostmap;
  cmtoggle.onchange=e=>{ showCostmap=e.target.checked; redraw(); }; }
const linktoggle=document.getElementById('linktoggle');
if(linktoggle){ linktoggle.checked=showFLinks;
  linktoggle.onchange=e=>{ showFLinks=e.target.checked; redraw(); }; }
const idtoggle=document.getElementById('idtoggle');
if(idtoggle){ idtoggle.checked=showIds;
  idtoggle.onchange=e=>{ showIds=e.target.checked; redraw(); }; }
const S=F.stats;
const row=(k,v)=>`<div class="stat"><span>${k}</span><b>${v}</b></div>`;
document.getElementById('info').innerHTML='<h2>Mission</h2>'+
  row('type',S.kind)+row('route',S.origin+' &rarr; '+S.dest)+row('speed',S.speed_kmh.toFixed(0)+' km/h')+
  row('launch',(S.launch_s/60).toFixed(1)+' min')+row('flight time',(S.flight_time_s/60).toFixed(1)+' min')+
  row('route length',S.route_len_m.toFixed(0)+' m')+row('flown',S.flown_m.toFixed(0)+' m')+
  (S.velocity_kmh!=null?row('velocity (door-to-door)',(+S.velocity_kmh).toFixed(1)+' km/h'):'')+
  (S.air_velocity_kmh!=null?row('velocity (airborne)',(+S.air_velocity_kmh).toFixed(1)+' km/h'):'')+
  (S.dock_idle_s?row('dock stop',(S.dock_idle_s/60).toFixed(1)+' min'):'')+
  row('holds',S.n_holds+' ('+(S.hold_s/60).toFixed(1)+' min)')+row('energy',S.energy_wh.toFixed(0)+' Wh')+
  row('battery end',(S.battery_end_pct).toFixed(0)+' %')+
  row('outcome', S.completed?'completed':(S.battery_dead?'battery dead':'unfinished'));
redraw(); requestAnimationFrame(loop);
</script></body></html>
"""


_AGENT_INDEX_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent routes</title>
<style>
  body{margin:0;background:#0f1420;color:#e8edf5;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;padding:16px;}
  h1{font-size:18px;} .sub{color:#9fb0c8;font-size:13px;margin-bottom:12px;}
  input{padding:7px 10px;border-radius:7px;border:1px solid #33405a;background:#1a2233;color:#e8edf5;width:220px;margin-bottom:12px;}
  table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums;}
  th,td{text-align:left;padding:6px 10px;border-bottom:1px solid #222c40;}
  th{color:#9fb0c8;position:sticky;top:0;background:#0f1420;cursor:pointer;}
  a{color:#6ab0ff;text-decoration:none;} tr:hover td{background:#161d2c;}
</style></head><body>
<h1>Per-agent route replays</h1>
<div class="sub">__N__ agents &mdash; click one to replay its travel. <a href="../agents_animation.html">&larr; full simulation</a></div>
<input id="q" placeholder="filter (agent id, DB/DK, type)..." oninput="filter()">
<table id="t"><thead><tr>
  <th>agent</th><th>type</th><th>route</th><th>km/h</th><th>flight (min)</th><th>route (m)</th><th>holds</th><th>done</th>
</tr></thead><tbody>__ROWS__</tbody></table>
<script>
function filter(){ const q=document.getElementById('q').value.toLowerCase();
  for(const tr of document.querySelectorAll('#t tbody tr'))
    tr.style.display = tr.textContent.toLowerCase().includes(q)?'':'none'; }
</script></body></html>
"""


# ======================================================================
# PyVista 3D view
# ======================================================================
def _network_line_polydata(net: Network, z: float = 0.0):
    """One PolyData holding every lane centreline as poly-lines at z."""
    import pyvista as pv
    pts, lines = [], []
    for leg_id in net.lanes["leg_id"].unique():
        for lane in ("A", "B"):
            xy = _lane_xy(net.lanes, leg_id, lane)
            if len(xy) < 2:
                continue
            start = len(pts)
            for x, y in xy:
                pts.append((float(x), float(y), z))
            lines.append(len(xy))
            lines.extend(range(start, start + len(xy)))
    poly = pv.PolyData()
    poly.points = np.array(pts, float) if pts else np.zeros((0, 3))
    poly.lines = np.array(lines, np.int64) if lines else np.zeros(0, np.int64)
    return poly


def _pv_base_scene(net: Network, params, pl):
    """Draw the persistent network + objectives into a PyVista plotter."""
    import pyvista as pv
    n_levels = max(1, int(pget(params, "FLIGHT_LEVELS", 4)))
    base_z = float(pget(params, "BASE_LEVEL_M", 60.0))
    level_sep = float(pget(params, "LEVEL_SEP_M", 30.0))
    x0, y0 = net.nodes["x"].min(), net.nodes["y"].min()
    x1, y1 = net.nodes["x"].max(), net.nodes["y"].max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    # faint ground plane
    plane = pv.Plane(center=(cx, cy, -2.0),
                     i_size=(x1 - x0) * 1.15, j_size=(y1 - y0) * 1.15)
    pl.add_mesh(plane, color="#eef1f5", opacity=0.6, show_edges=False)
    # lane centrelines
    pl.add_mesh(_network_line_polydata(net, 0.0), color="#9fb0c8", line_width=2)
    # one translucent sheet per FLIGHT LEVEL (heading band) + the patrol band on
    # top, so the vertical separation is visible; label each with its altitude
    lvl_pal = ["#2f6fed", "#1f8f4f", "#e23b3b", "#e6a010", "#7b3fbf",
               "#0aa3a3", "#c0398b", "#5a6b12"]
    lp, ll = [], []
    for k in range(n_levels + 1):
        z = base_z + k * level_sep
        c = "#7b3fbf" if k == n_levels else lvl_pal[k % len(lvl_pal)]
        sheet = pv.Plane(center=(cx, cy, z),
                         i_size=(x1 - x0) * 1.05, j_size=(y1 - y0) * 1.05)
        pl.add_mesh(sheet, color=c, opacity=0.06, show_edges=False)
        lp.append((x0, y0, z))
        ll.append(f"patrol {z:.0f} m" if k == n_levels else f"FL{k} {z:.0f} m")
    pl.add_point_labels(np.array(lp, float), ll, font_size=11,
                        text_color="black", shape=None, always_visible=True)
    # objectives as glyphs + labels
    obj = net.nodes[net.nodes["kind"] == "objective"]
    dep_pts, del_pts, labels, lpts = [], [], [], []
    for r in obj.itertuples():
        if str(r.net_id).startswith("DB"):
            dep_pts.append((r.x, r.y, 0.0))
        else:
            del_pts.append((r.x, r.y, 0.0))
        labels.append(str(r.net_id)); lpts.append((r.x, r.y, 120.0))
    if dep_pts:
        pl.add_mesh(pv.PolyData(np.array(dep_pts, float)).glyph(
            geom=pv.Cube(x_length=90, y_length=90, z_length=90), scale=False),
            color="#c0392b")
    if del_pts:
        pl.add_mesh(pv.PolyData(np.array(del_pts, float)).glyph(
            geom=pv.Cone(direction=(0, 0, 1), height=140, radius=60), scale=False),
            color="#1f8f4f")
    pl.add_point_labels(np.array(lpts, float), labels, font_size=14,
                        text_color="black", shape=None, always_visible=True)


def _agent_points(frame, base_z, level_sep):
    """(out_xyz, in_xyz, patrol_xyz, hold_xyz) for one frame, lifted to the
    agent's FLIGHT LEVEL altitude (z = base + level*sep). Colour still encodes
    direction; height now encodes the heading-based level, so the vertical
    separation that resolves crossing conflicts is visible in 3D."""
    _t, xy, outb, hold, _a, _spd, _cf, lev = frame
    out_p, in_p, pat_p, hold_p = [], [], [], []
    for i in range(len(xy)):
        cat = int(outb[i])
        z = base_z + int(lev[i]) * level_sep
        p = (float(xy[i, 0]), float(xy[i, 1]), z)
        (pat_p if cat == 2 else out_p if cat == 1 else in_p).append(p)
        if hold[i]:
            hold_p.append(p)
    a = lambda L: np.array(L, float) if L else np.zeros((0, 3))
    return a(out_p), a(in_p), a(pat_p), a(hold_p)


def render_pyvista(net: Network, frames, params, sep_eff, out_dir: Path):
    """3D corridor scene rendered with PyVista: an animated movie of the
    fleet plus a self-contained interactive HTML snapshot and a PNG.
    Outbound (to-DK) agents cruise on a lower plane than inbound
    (to-DB), so the two directions are also vertically separated."""
    import pyvista as pv
    pv.OFF_SCREEN = True
    base_z = float(pget(params, "BASE_LEVEL_M", 60.0))
    level_sep = float(pget(params, "LEVEL_SEP_M", 30.0))
    max_frames = int(pget(params, "PYVISTA_MAX_FRAMES", 300))
    fps = int(pget(params, "GIF_FPS", 20))
    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        frames = [frames[i] for i in idx]

    def draw_agents(pl, frame):
        out_p, in_p, pat_p, hold_p = _agent_points(frame, base_z, level_sep)
        for name, pts, col, sz in (("out", out_p, "#2f6fed", 12),
                                   ("in", in_p, "#e23b3b", 12),
                                   ("pat", pat_p, "#7b3fbf", 22)):
            if len(pts):
                pl.add_mesh(pv.PolyData(pts), name=name, color=col,
                            render_points_as_spheres=True, point_size=sz)
            else:
                pl.remove_actor(name, render=False)
        if len(hold_p):
            pl.add_mesh(pv.PolyData(hold_p), name="hold", color="#ff8c1a",
                        render_points_as_spheres=True, point_size=26, opacity=0.5)
        else:
            pl.remove_actor("hold", render=False)
        t = frame[0]
        pl.add_text(f"t = {int(t//3600):02d}:{int(t%3600//60):02d}:{int(t%60):02d}   "
                    f"airborne {len(frame[1])}", name="clock", font_size=12)

    # ---- movie ----
    pl = pv.Plotter(off_screen=True, window_size=(1280, 960))
    pl.set_background("white")
    _pv_base_scene(net, params, pl)
    pl.camera_position = "iso"
    pl.camera.elevation -= 12
    movie_path = out_dir / "agents_pyvista.mp4"
    try:
        pl.open_movie(str(movie_path), framerate=fps)
    except Exception:
        movie_path = out_dir / "agents_pyvista.gif"
        pl.open_gif(str(movie_path))
    for fr in frames:
        draw_agents(pl, fr)
        pl.write_frame()
    pl.close()

    # ---- interactive HTML snapshot at the busiest frame ----
    busy = max(range(len(frames)), key=lambda i: len(frames[i][1])) if frames else 0
    html_path = out_dir / "agents_pyvista.html"
    png_path = out_dir / "figures" / "03_pyvista_3d.png"
    ph = pv.Plotter(off_screen=True, window_size=(1280, 960))
    ph.set_background("white")
    _pv_base_scene(net, params, ph)
    if frames:
        draw_agents(ph, frames[busy])
    ph.camera_position = "iso"
    ph.camera.elevation -= 12
    try:
        ph.export_html(str(html_path))
    except Exception as e:
        html_path = None
        print(f"    (export_html unavailable: {e})")
    ph.screenshot(str(png_path))
    ph.close()
    return movie_path, html_path


# ======================================================================
# Output writers
# ======================================================================
def write_missions(agents, params, out_csv: Path):
    p0, cd, bat = energy_params(params)
    rows = []
    for a in agents:
        e_wh = agent_energy_wh(a, p0, cd)
        rows.append({
            "agent_id": a.aid,
            "origin": a.origin,
            "dest": a.dest,
            "mission": a.contingency if a.contingency else
                       ("round_trip" if a.round_trip else "one_way"),
            "round_trip": a.round_trip,
            "arrival_s": round(a.arrival_t, 1),
            "depart_s": None if a.depart_t is None else round(a.depart_t, 1),
            "wait_s": None if a.depart_t is None else round(a.depart_t - a.arrival_t, 1),
            "launch_s": None if a.launch_t is None else round(a.launch_t, 1),
            "complete_s": None if a.complete_t is None else round(a.complete_t, 1),
            "flight_time_s": None if (a.complete_t is None or a.depart_t is None)
                             else round(a.complete_t - a.depart_t, 1),
            "speed_kmh": a.speed_kmh,                # nominal cruise of its class
            # ACHIEVED velocity = total distance flown / total mission time
            # (launch -> complete). This is the door-to-door figure: it carries
            # every hold, every slow-down under the cost-map and the dock stop,
            # so it is always below the class cruise speed. `air_velocity_kmh`
            # divides by airborne time only, which isolates how much of the loss
            # is flight-phase slow-down rather than time parked at a dock.
            "velocity_kmh": None if (a.complete_t is None or a.depart_t is None
                                     or a.complete_t <= a.depart_t)
                            else round(3.6 * a.dist_m / (a.complete_t - a.depart_t), 2),
            "air_velocity_kmh": None if a.air_s <= 0
                                else round(3.6 * a.dist_m / a.air_s, 2),
            "air_time_s": round(a.air_s, 1),
            "dock_idle_s": None if (a.complete_t is None or a.depart_t is None)
                           else round(max(0.0, (a.complete_t - a.depart_t) - a.air_s), 1),
            "route_len_m": round(a.route_len, 1),
            "flown_m": round(a.dist_m, 1),
            "hold_s": round(a.hold_s, 1),
            "n_holds": a.n_holds,
            "energy_wh": round(e_wh, 2),
            "charge_s": round(a.charge_s, 1),
            "n_charges": a.n_charges,
            "battery_end_pct": round(100.0 * a.battery_wh / bat, 1),
            "completed": a.status == "done",
            "battery_dead": a.status == "dead",
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def write_timeline(timeline, out_csv: Path):
    df = pd.DataFrame(timeline, columns=["t_s", "n_airborne", "n_holding",
                                         "min_approach_m", "n_done",
                                         "n_backlog", "n_active"])
    df["min_approach_m"] = df["min_approach_m"].replace(np.inf, np.nan)
    df.to_csv(out_csv, index=False)


def write_trajectories(frames, out_csv: Path):
    cat = {0: "inbound", 1: "outbound", 2: "patrol"}
    rows = []
    for (t, xy, outb, hold, aids, spd, _cf, lev) in frames:
        for i in range(len(xy)):
            rows.append({"t_s": round(t, 1), "agent_id": int(aids[i]),
                         "x": round(float(xy[i, 0]), 2), "y": round(float(xy[i, 1]), 2),
                         "kind": cat.get(int(outb[i]), "?"),
                         "speed_kmh": float(spd[i]), "holding": bool(hold[i]),
                         "level": int(lev[i])})
    pd.DataFrame(rows).to_csv(out_csv, index=False)


# ======================================================================
# CLI / main
# ======================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="2D corridor multi-agent traffic simulation.")
    p.add_argument("--param-file", default="params/simulate_agents_2d.params")
    p.add_argument("--agents", type=int, default=None, help="override N_AGENTS")
    p.add_argument("--hours", type=float, default=None, help="override SHIFT_HOURS")
    p.add_argument("--seed", type=int, default=None, help="override SIM_SEED")
    p.add_argument("--no-animation", action="store_true", help="skip the GIF")
    p.add_argument("--no-html", action="store_true", help="skip the interactive HTML")
    p.add_argument("--no-pyvista", action="store_true", help="skip the 3D PyVista view")
    p.add_argument("--concurrent", type=int, default=None, help="override MAX_CONCURRENT")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    params = load_params(args.param_file)
    if args.agents is not None:
        params["N_AGENTS"] = args.agents
    if args.hours is not None:
        params["SHIFT_HOURS"] = args.hours
    if args.seed is not None:
        params["SIM_SEED"] = args.seed
    if args.no_animation:
        params["MAKE_ANIMATION"] = False
    if args.no_html:
        params["MAKE_HTML"] = False
    if args.no_pyvista:
        params["MAKE_PYVISTA"] = False
    if args.concurrent is not None:
        params["MAX_CONCURRENT"] = args.concurrent

    corridor_dir = THIS_DIR / str(pget(params, "CORRIDOR_DIR", "output/06_corridor_network"))
    output_dir = THIS_DIR / str(pget(params, "OUTPUT_DIR", "output/08_agent_sim_2d"))
    fig_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 66)
    print(f"engine_simulate_scheduling.py  {VERSION}")
    print(f"Param file    : {args.param_file}")
    print(f"Corridor dir  : {corridor_dir}")
    print(f"Output dir    : {output_dir}")
    print("=" * 66)

    ring_travel = bool(pget(params, "RING_TRAVEL", False))
    # the ring lane gap must match the geometry stage 05 BUILT, not a local
    # default: 05 with ROUNDABOUT_LANE_GAP_M = 0 makes a single ring, and
    # modelling a phantom inner lane here would put agents on a circle the
    # network does not have. Read 05's own params when they are reachable.
    lane_gap = pget(params, "ROUNDABOUT_LANE_GAP_M", None)
    if lane_gap is None:
        cpf = THIS_DIR / str(pget(params, "CORRIDOR_PARAM_FILE", ""))
        lane_gap = 50.0
        if cpf.is_file():
            ns: dict = {}
            try:
                exec(compile(cpf.read_text(encoding="utf-8"), str(cpf), "exec"), {}, ns)
                lane_gap = float(ns.get("ROUNDABOUT_LANE_GAP_M", 50.0))
            except Exception:
                pass
    lane_gap = float(lane_gap)
    ring_right_hand = bool(pget(params, "RING_RIGHT_HAND", True))
    net = Network(corridor_dir, ring_travel=ring_travel, lane_gap=lane_gap,
                  ring_right_hand=ring_right_hand)
    print(f"Network       : {len(net.obj_xy)} objectives, "
          f"{net.lanes['leg_id'].nunique()} legs, "
          f"{len(net._routes)} pair routes")
    if ring_travel:
        rule = ("right-hand (one-way CCW, exit right)" if ring_right_hand
                else "shorter-arc")
        lanes_desc = ("ONE shared ring lane (lane gap 0)" if lane_gap <= 0
                      else f"outer ring (forward) / inner ring (backward), "
                           f"lane gap {lane_gap:.0f} m")
        print(f"Ring travel   : ON -- {len(net.rings)} roundabouts, {rule}; "
              f"{lanes_desc}; legs clipped at the ring boundary")
    else:
        print("Ring travel   : OFF -- legs meet ring nodes directly (no ring arcs)")

    rng = np.random.default_rng(int(pget(params, "SIM_SEED", 12345)))
    agents, patrols = build_fleet(net, params, rng)
    sched_info = None
    _slw = None
    if bool(pget(params, "SCHEDULE_MODE", False)):
        # the ETA that books a dock pad must price the route the way the agent
        # will actually fly it: true velocity = slowness * cruise
        _cm = load_costmap(pget(params, "COST_MAP_FILE", ""))
        if _cm is not None:
            _g, _x0, _y0, _rs, _nx, _ny = _cm

            def _slw(x, y):
                return float(_g[min(max(int((y - _y0) / _rs), 0), _ny - 1),
                                min(max(int((x - _x0) / _rs), 0), _nx - 1)])
        sched_info = schedule_departures(net, agents, params, slowness_at=_slw)
        print(f"Scheduling    : ON -- {sched_info['scheduled']} CTOTs over "
              f"{sched_info['n_departure_lanes']} departure lanes; "
              f"{sched_info['headway_s']:.0f} s corridor headway, origin cap "
              f"{sched_info['origin_cap']}; {sched_info['n_delayed']} missions "
              f"delayed (mean {sched_info['mean_delay_s']/60.0:.1f} min), "
              f"last CTOT {sched_info['span_h']:.2f} h")
        _en = sched_info.get("energy")
        if _en:
            print(f"Energy gate   : battery {_en['battery_wh']:.0f} Wh, "
                  f"{_en['reserve_pct']*100:.0f}% reserve + "
                  f"{_en['hold_allowance_pct']*100:.0f}% hold allowance -> "
                  f"{_en['usable_wh']:.0f} Wh usable; need mean "
                  f"{_en['mean_required_wh']:.0f} / max {_en['max_required_wh']:.0f} Wh; "
                  f"energy-optimal cruise {_en['optimal_cruise_kmh']:.0f} km/h -> "
                  f"{_en['n_speed_promoted']} missions had their speed class raised"
                  + (f", {_en['n_infeasible_any_class']} INFEASIBLE at any class"
                     if _en['n_infeasible_any_class'] else ""))
        if sched_info["dock_capacity"]:
            print(f"Dock booking  : {sched_info['dock_capacity']} pads/dock x "
                  f"{sched_info['dock_hold_s']/60:.0f} min stop; "
                  f"{sched_info['n_parking_missions']} parking missions, mean ETA "
                  f"{sched_info['mean_eta_to_dock_s']/60:.1f} min"
                  + (" (slowness-priced)" if _slw is not None else " (no cost-map)")
                  + f"; {sched_info['n_pushed_for_dock']} pushed back for a pad "
                    f"(mean {sched_info['mean_dock_push_s']/60:.1f} min)")
    else:
        print("Scheduling    : OFF -- reactive launch metering "
              "(LAUNCH_SPACING_S + DCB)")
    n_rt = sum(a.round_trip for a in agents)
    n_backup = sum(a.contingency == "backup" for a in agents)
    n_return = sum(a.contingency == "return" for a in agents)
    n_cont = n_backup + n_return
    n_normal = len(agents) - n_cont
    cls = defaultdict(int)
    for a in agents:
        cls[a.speed_kmh] += 1
    cls_str = ", ".join(f"{int(k)}km/h:{v}" for k, v in sorted(cls.items(), reverse=True))
    print(f"Fleet         : {len(agents)} deliveries "
          f"({n_rt} round trips, {n_normal - n_rt} one-way, "
          f"{n_cont} contingency: {n_backup} backup / {n_return} return-to-base)")
    print(f"Speed classes : {cls_str}")
    print(f"Patrol        : {len(patrols)} unit from base, re-launches every "
          f"{pget(params,'PATROL_INTERVAL_MIN',30.0)} min @ "
          f"{pget(params,'PATROL_SPEED_KMH',50.0)} km/h "
          f"(highest priority, dynamic obstacle="
          f"{bool(pget(params,'PATROL_AS_OBSTACLE',True))})")
    fm = str(pget(params, "FLOW_MODE", "spacing")).lower()
    aw = float(pget(params, "ARRIVAL_WINDOW_H", pget(params, "SHIFT_HOURS", 1.0)))
    flow_desc = (f">={float(pget(params,'SEPARATION_M',80.0)):.0f}m car-following (deadlock-free)"
                 if fm == "spacing" else "one agent per leg-block")
    print(f"Flow mode     : {fm} -- {flow_desc}; balance routes "
          f"{bool(pget(params,'BALANCE_ROUTES',True))}")
    print(f"Demand        : {len(agents)} ready over {aw:.1f} h "
          f"({'all at t=0' if aw == 0 else 'spread'}); cap {pget(params,'MAX_CONCURRENT',10**9)}")

    print("Simulating ...")
    frames, timeline, stats = simulate(net, agents, patrols, params)

    completed = [a for a in agents if a.status == "done"]
    tot_hold = sum(a.hold_s for a in agents)
    tot_holds = sum(a.n_holds for a in agents)
    peak_air = stats["peak_concurrent"]
    peak_hold = max((r[2] for r in timeline), default=0)
    sep_eff = stats["effective_separation_m"]
    approach = stats["min_approach_m"]
    lane_gap = stats["min_lane_gap_m"]
    shift_s = float(pget(params, "SHIFT_HOURS", 1.0)) * 3600.0
    done_in_shift = sum(1 for a in completed if a.complete_t is not None and a.complete_t <= shift_s)
    waits = [a.depart_t - a.arrival_t for a in agents if a.depart_t is not None]
    mean_wait = float(np.mean(waits)) if waits else 0.0
    max_wait = float(np.max(waits)) if waits else 0.0
    n_legs_dir = stats["n_legs_dir"]
    n_dead = sum(1 for a in agents if a.status == "dead")
    n_unfinished = sum(1 for a in agents if a.status not in ("done", "dead"))
    headway = float(pget(params, "TIME_HEADWAY_S", 30.0))
    chg = [a.charge_s for a in agents if a.n_charges > 0]
    mean_charge = float(np.mean(chg)) if chg else 0.0

    print("-" * 66)
    if stats["gridlock"]:
        print(f"GRIDLOCK      : no movement for >{pget(params,'GRIDLOCK_TIMEOUT_S',300)}s "
              f"-> stopped at t={stats['sim_end_t']/3600:.2f} h")
    print(f"Completed     : {len(completed)}/{len(agents)} deliveries by "
          f"t={stats['sim_end_t']/3600:.2f} h  ({n_dead} battery-dead, {n_unfinished} unfinished)")
    print(f">>> MAX SIMULTANEOUS AGENTS (peak concurrent) = {stats['peak_concurrent']}")
    _h_std = stats["sep_standard_h_m"]
    _nviol, _npair = stats["sep_violation_samples"], stats["n_pair_samples"]
    print(f"Separation std: {_h_std:.0f} m horizontal (any corridor, same level) + "
          f"{headway:.0f} s longitudinal (in trail, same corridor)")
    print(f"  longitudinal: min same-lane gap "
          f"{lane_gap if lane_gap is None else round(lane_gap,1)} m "
          f"(required {headway*8.3:.0f} m at 30km/h .. {headway*16.7:.0f} m at 60km/h)")
    print(f"  horizontal  : min observed {stats['min_approach_m']:.1f} m over "
          f"{_npair} same-level pair-samples -> "
          + (f"OK, no pair under {_h_std:.0f} m"
             if _nviol == 0 else
             f"{_nviol} violating pair-samples "
             f"({100.0*_nviol/max(_npair,1):.3f}%), worst {stats['worst_sep_m']:.1f} m, "
             f"peak {stats['peak_sep_violations']} at once"))
    if stats["orca_rings"]:
        print(f"  ORCA rings  : {stats['orca_agent_steps']} agent-steps flown under "
              f"reciprocal avoidance inside the roundabout zones "
              f"(pair clearance {2*stats['orca_radius_m']:.0f} m, tau {stats['orca_tau_s']:.0f} s)")
    if ring_travel:
        _cut, _tot = stats["ring_cut_samples"], stats["ring_pos_samples"]
        print(f"  centreline  : {_tot} position-samples, {_cut} inside a roundabout "
              f"interior -> " + ("OK (all agents circulated the ring)" if _cut == 0
                                 else f"VIOLATION ({100.0*_cut/max(_tot,1):.3f}% cut across)"))
    if completed:
        _km = sum(a.dist_m for a in completed) / 1000.0
        _h = sum(a.complete_t - a.depart_t for a in completed) / 3600.0
        _ah = sum(a.air_s for a in completed) / 3600.0
        print(f"Velocity      : {_km/max(_h,1e-9):.1f} km/h door-to-door "
              f"({_km:.0f} km flown / {_h:.1f} h from launch to complete); "
              f"{_km/max(_ah,1e-9):.1f} km/h airborne-only; "
              f"nominal cruise {float(np.mean([a.speed_kmh for a in completed])):.1f} km/h")
    _idle_min = float(pget(params, 'MIN_DEST_IDLE_S', 300.0)) / 60.0
    print(f"Battery/dock  : park {_idle_min:.0f} min at the dock and recharge to "
          f"{float(pget(params,'CHARGE_TARGET_PCT',0.9))*100:.0f}% before the return leg; "
          f"mean charge {mean_charge/60:.1f} min; {n_dead} ran flat mid-air")
    print(f"                docked drones are NOT airborne traffic -> exempt from the "
          f"separation/conflict checks ({stats['dock_exempt_samples']} agent-samples, "
          f"peak {stats['peak_docked']} parked at once)")
    if stats["dock_capacity"]:
        _pk = stats["dock_peak_per_dock"]
        _over = {k: v for k, v in _pk.items() if v > stats["dock_capacity"]}
        _pr = stats.get("pad_reservation")
        if _pr:
            print(f"Pad booking   : live re-plan every {_pr['update_period_s']:.0f} s "
                  f"(tolerance {_pr['tolerance_s']:.0f} s) -- {_pr['initial_bookings']} "
                  f"pads booked, {_pr['updates']} re-planned of which "
                  f"{_pr['rebookings']} moved (mean shift {_pr['mean_drift_s']/60:.1f} min); "
                  f"{_pr['missions_deferred_for_pad']} missions held on the ground "
                  f"for want of a pad ({_pr['launch_deferral_events']} retries)")
            print(f"Energy watch  : {_pr['energy_short_in_flight']} drones found short "
                  f"of charge in flight and given the earliest pad; "
                  f"{_pr['launches_held_low_battery']} launches held to charge; "
                  f"{n_dead} ran flat")
        print(f"Dock capacity : {stats['dock_capacity']} pads/dock; peak use "
              + ", ".join(f"{k} {v}" for k, v in sorted(_pk.items()))
              + f"  -> {'OK, none over capacity' if not _over else 'OVER: ' + str(_over)}"
              + (f"; {stats['dock_full_holds']} arrivals held for a free pad"
                 if stats["dock_full_holds"] else "; no arrival ever waited for a pad"))
    if stats["cost_map_loaded"]:
        print(f"Cost-map      : slowness field applied (min slowness "
              f"{stats['cost_map_min_slowness']:.2f}); true velocity = slowness x base speed")
    else:
        print(f"Cost-map      : none (run 08_generate_costmap.py first); base speeds used")
    if stats["flow_mode"] != "spacing":
        print(f"Leg rule      : max agents on any one leg = {stats['max_agents_per_leg']} -> "
              f"{'OK (<=1)' if stats['max_agents_per_leg'] <= 1 else 'VIOLATION'}")
    print(f"Patrols       : {stats['patrol_sorties']} sorties launched "
          f"(every {pget(params,'PATROL_INTERVAL_MIN',30.0):.0f} min), {stats['patrol_laps']} loops flown")
    _hc = stats["hold_cause"]
    _hct = sum(v for k, v in _hc.items() if k != "launch_queue") or 1
    print(f"Hold causes   : leader {100*_hc['leader']/_hct:.0f}%  "
          f"node-mutex {100*_hc['node']/_hct:.0f}%  block {100*_hc['block']/_hct:.0f}%  "
          f"({_hct} airborne hold-samples; launch-queue {_hc['launch_queue']} separately)")
    print(f"Conflicts     : {stats['total_conflict_samples']} conflict-samples in "
          f"{stats['n_conflict_frames']} frames (peak {stats['peak_conflicts']} at once); "
          f"threshold < {stats['conflict_time_s']:.0f}s time-separation  [red stars]")
    routes_total = len({a.path for a in agents})
    routes_flown = len({a.path for a in agents if a.launch_t is not None})
    n_reroutes = sum(a.n_reroutes for a in agents)
    n_diverted = sum(1 for a in agents if a.n_reroutes > 0)
    print(f"Routes used   : {routes_flown}/{routes_total} distinct routes carried traffic")
    print(f"Reroutes      : {n_reroutes} diversions by {n_diverted} agents (congestion detours)")

    def route_outerness(a):
        vals = [net.leg_outerness.get(s.res.split("#")[0], 0.5)
                for s in a.segs if s is not None]
        return float(np.mean(vals)) if vals else 0.0
    outer_by_cls = defaultdict(list)
    for a in completed:
        outer_by_cls[a.speed_kmh].append(route_outerness(a))
    zone_str = ", ".join(f"{int(k)}km/h:{np.mean(v):.2f}"
                         for k, v in sorted(outer_by_cls.items(), reverse=True))
    print(f"Zone (outerness 0=centre..1=edge): {zone_str}  <- faster should be more outer")
    print(f"Wait to launch: mean {mean_wait/60:.1f} min, max {max_wait/60:.1f} min")

    # ---- deadline search: min window T for ALL deliveries to finish ----
    start_h = float(pget(params, "DEADLINE_START_H", 1.0))
    step_h = float(pget(params, "DEADLINE_STEP_H", 0.5))
    req_h, table = deadline_search(
        [a.complete_t if a.status == "done" else None for a in agents],
        len(agents), start_h, step_h)
    print("-" * 66)
    print(f"DEADLINE SEARCH (all {len(agents)} must finish; start {start_h:.1f}h, "
          f"+{step_h*60:.0f}min/step):")
    shown = 0
    for (Th, done, pct) in table:
        if Th in (start_h,) or pct >= 100.0 or shown % 4 == 0 or Th == table[-1][0]:
            flag = "  <-- ALL DONE" if done >= len(agents) else ""
            print(f"    within {Th:5.1f} h : {done:6d}/{len(agents)} ({pct:5.1f}%){flag}")
        shown += 1
    if req_h is not None:
        print(f">>> ALL {len(agents)} deliveries finish within {req_h:.1f} h "
              f"({req_h/start_h:.0f}x the {start_h:.0f}h target)")
    else:
        print(f">>> NOT all finished within simulated horizon "
              f"({stats['n_incomplete']} unfinished at {stats['sim_end_t']/3600:.1f}h)")

    # ---- energy ----
    p0, cd, bat = energy_params(params)
    e_all = [agent_energy_wh(a, p0, cd) for a in completed]
    e_patrol = sum(agent_energy_wh(a, p0, cd) for a in patrols)
    tot_e = sum(e_all) + e_patrol
    v_opt = (p0 / (2 * cd)) ** (1 / 3)
    over_bat = sum(1 for e in e_all if e > bat)
    e_by_cls = defaultdict(list)
    for a in completed:
        e_by_cls[a.speed_kmh].append(agent_energy_wh(a, p0, cd))
    print("-" * 66)
    print(f"ENERGY (P={p0:.0f}W hover + {cd:.3f}*v^3; battery {bat:.0f}Wh; "
          f"energy-optimal cruise {v_opt*3.6:.0f} km/h):")
    for k in sorted(e_by_cls, reverse=True):
        ev = e_by_cls[k]
        print(f"    {int(k):2d} km/h : mean {np.mean(ev):5.1f} Wh/mission "
              f"({np.mean(ev)/bat*100:4.1f}% battery), {len(ev)} missions")
    print(f"    fleet total {tot_e/1000:.1f} kWh "
          f"(deliveries {sum(e_all)/1000:.1f} + patrols {e_patrol/1000:.1f}); "
          f"{over_bat} missions exceed one battery")

    # ---- outputs ----
    write_missions(agents, params, output_dir / "agent_missions.csv")
    write_timeline(timeline, output_dir / "sim_timeline.csv")
    # every loss of the horizontal separation standard, with where it happened
    pd.DataFrame(stats["sep_violation_log"],
                 columns=["t_s", "agent_a", "agent_b", "gap_m", "x", "y",
                          "res_a", "res_b", "flight_level"]) \
      .round({"t_s": 1, "gap_m": 2, "x": 1, "y": 1}) \
      .to_csv(output_dir / "separation_violations.csv", index=False)
    if bool(pget(params, "WRITE_TRAJECTORY", True)):
        write_trajectories(frames, output_dir / "trajectories.csv")

    metrics = {
        "version": VERSION,
        "n_deliveries": len(agents),
        "n_patrols": len(patrols),
        "n_round_trips": int(n_rt),
        "n_one_way": int(n_normal - n_rt),
        "n_contingency_backup": int(n_backup),
        "n_contingency_return": int(n_return),
        "speed_classes_kmh": {int(k): int(v) for k, v in sorted(cls.items(), reverse=True)},
        "n_completed": len(completed),
        "n_battery_dead": int(n_dead),
        "n_unfinished": int(n_unfinished),
        "time_headway_s": headway,
        "charge_power_w": float(pget(params, "CHARGE_POWER_W", 600.0)),
        "charge_target_pct": float(pget(params, "CHARGE_TARGET_PCT", 0.9)),
        "min_dest_idle_s": float(pget(params, "MIN_DEST_IDLE_S", 300.0)),
        "hold_penalty_m": float(pget(params, "HOLD_PENALTY_M", 20000.0)),
        "mean_dock_charge_min": mean_charge / 60.0,
        "n_dock_charges": int(sum(a.n_charges for a in agents)),
        "gridlock": bool(stats["gridlock"]),
        "flow_mode": stats["flow_mode"],
        "max_agents_on_a_lane": int(stats["max_agents_per_leg"]),
        "min_same_lane_gap_m": lane_gap,
        "separation_respected": bool((lane_gap is None) or (lane_gap >= sep_eff - 1e-6)),
        "n_directed_leg_lanes": int(n_legs_dir),
        "routes_used": int(routes_flown),
        "routes_total": int(routes_total),
        "merge_free_frac": float(pget(params, "MERGE_FREE_FRAC", 0.5)),
        "n_reroutes": int(n_reroutes),
        "n_agents_diverted": int(n_diverted),
        "max_simultaneous_agents": int(stats["peak_concurrent"]),
        "peak_backlog": int(stats["peak_backlog"]),
        "patrol_sorties": int(stats["patrol_sorties"]),
        "patrol_interval_min": float(pget(params, "PATROL_INTERVAL_MIN", 30.0)),
        "conflict_time_s": float(stats["conflict_time_s"]),
        "conflict_samples": int(stats["total_conflict_samples"]),
        "conflict_frames": int(stats["n_conflict_frames"]),
        "peak_simultaneous_conflicts": int(stats["peak_conflicts"]),
        "cost_map_loaded": bool(stats["cost_map_loaded"]),
        "cost_map_min_slowness": stats["cost_map_min_slowness"],
        "mean_wait_for_leg_s": mean_wait,
        "max_wait_for_leg_s": max_wait,
        "shift_hours": float(pget(params, "SHIFT_HOURS", 1.0)),
        "sim_end_hours": stats["sim_end_t"] / 3600.0,
        "node_mutex_enabled": bool(pget(params, "NODE_MUTEX_ENABLE", False)),
        "hold_cause": stats["hold_cause"],
        "flight_levels": int(pget(params, "FLIGHT_LEVELS", 4)),
        "launch_spacing_s": float(pget(params, "LAUNCH_SPACING_S", 2.0)),
        "reference_separation_m": sep_eff,
        "min_closest_approach_m": approach,
        # ---- compliance against the declared separation standard ----
        "separation_standard": {
            "horizontal_m": stats["sep_standard_h_m"],
            "longitudinal_s": stats["sep_standard_v_s"],
            "min_horizontal_observed_m": stats["min_approach_m"],
            "min_in_trail_gap_observed_m": stats["min_lane_gap_m"],
            "pair_samples_checked": int(stats["n_pair_samples"]),
            "violation_samples": int(stats["sep_violation_samples"]),
            "violation_frames": int(stats["sep_violation_frames"]),
            "peak_simultaneous_violations": int(stats["peak_sep_violations"]),
            "worst_horizontal_m": stats["worst_sep_m"],
            "violation_rate": (stats["sep_violation_samples"] / stats["n_pair_samples"])
            if stats["n_pair_samples"] else 0.0,
            "horizontal_respected": bool(stats["sep_violation_samples"] == 0),
        },
        "centreline_compliance": {
            "ring_travel": bool(ring_travel),
            "roundabout_rule": ("one-way CCW, exit right" if ring_right_hand
                                else "shorter-arc"),
            "position_samples": int(stats["ring_pos_samples"]),
            "zone_model": ("2-D ORCA area (violation = inside the central island)"
                           if stats["orca_rings"] else
                           "1-D circulating ring (violation = inside the ring)"),
            "inside_ring_interior_samples": int(stats["ring_cut_samples"]),
            "respected": bool(stats["ring_cut_samples"] == 0),
        },
        "scheduling": ({"mode": "ctot"}
                       | {k: v for k, v in sched_info.items()
                          if k not in ("rows", "dock_profile")}
                       ) if sched_info else {"mode": "reactive"},
        "orca": {
            "enabled": bool(stats["orca_rings"]),
            "agent_steps_in_zones": int(stats["orca_agent_steps"]),
            "agent_radius_m": stats["orca_radius_m"],
            "pair_clearance_m": 2.0 * stats["orca_radius_m"],
            "tau_s": stats["orca_tau_s"],
        },
        "peak_holding": int(peak_hold),
        "total_hold_events": int(tot_holds),
        "total_hold_minutes": tot_hold / 60.0,
        "mean_flight_time_s": float(np.mean([a.complete_t - a.depart_t for a in completed]))
        if completed else None,
        # ACHIEVED velocity = distance flown / time taken, over completed missions.
        # "door_to_door" divides by the whole launch->complete span, so it carries
        # holds, cost-map slow-downs and the dock stop; "airborne" divides by
        # airborne time only, isolating flight-phase loss from time parked.
        "velocity": ({
            "door_to_door_kmh": round(3.6 * sum(a.dist_m for a in completed)
                                      / max(sum(a.complete_t - a.depart_t
                                                for a in completed), 1e-9), 2),
            "airborne_kmh": round(3.6 * sum(a.dist_m for a in completed)
                                  / max(sum(a.air_s for a in completed), 1e-9), 2),
            "nominal_cruise_kmh": round(float(np.mean([a.speed_kmh for a in completed])), 2),
            "total_distance_km": round(sum(a.dist_m for a in completed) / 1000.0, 1),
            "total_time_h": round(sum(a.complete_t - a.depart_t for a in completed)
                                  / 3600.0, 2),
        } if completed else None),
        "dock_stop": {
            "min_idle_s": float(pget(params, "MIN_DEST_IDLE_S", 300.0)),
            "charge_target_pct": float(pget(params, "CHARGE_TARGET_PCT", 0.9)),
            "conflict_exempt": True,
            "note": "a drone parked at a dock is not airborne traffic, so it is "
                    "excluded from the separation and conflict checks",
            "exempt_agent_samples": int(stats["dock_exempt_samples"]),
            "peak_docked": int(stats["peak_docked"]),
            "park_buffer_s": float(pget(params, "DOCK_PARK_BUFFER_S", 600.0)),
            "capacity_per_dock": int(stats["dock_capacity"]),
            "peak_per_dock": stats["dock_peak_per_dock"],
            "dock_full_holds": int(stats["dock_full_holds"]),
            "pad_reservation": stats["pad_reservation"],
            "capacity_respected": bool(
                stats["dock_capacity"] == 0
                or all(v <= stats["dock_capacity"]
                       for v in stats["dock_peak_per_dock"].values())),
        },
        "total_flown_km": (sum(a.dist_m for a in agents) + sum(a.dist_m for a in patrols)) / 1000.0,
        "balance_routes": bool(pget(params, "BALANCE_ROUTES", True)),
        "arrival_window_h": float(pget(params, "ARRIVAL_WINDOW_H", pget(params, "SHIFT_HOURS", 1.0))),
        "deadline_all_finish_h": req_h,
        "deadline_target_h": start_h,
        "deadline_table": [{"within_h": round(Th, 2), "done": int(d), "pct": round(p, 2)}
                           for (Th, d, p) in table],
        "energy_hover_w": p0,
        "energy_drag_coef": cd,
        "battery_wh": bat,
        "energy_optimal_kmh": round(v_opt * 3.6, 1),
        "energy_total_kwh": tot_e / 1000.0,
        "energy_deliveries_kwh": sum(e_all) / 1000.0,
        "energy_patrol_kwh": e_patrol / 1000.0,
        "energy_mean_wh_per_mission": float(np.mean(e_all)) if e_all else None,
        "energy_wh_by_class": {int(k): round(float(np.mean(v)), 2) for k, v in e_by_cls.items()},
        "missions_over_battery": int(over_bat),
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Plotting ...")
    plot_network_traffic(net, agents, fig_dir / "00_network_traffic.png",
                         obstacle_file=pget(params, "OBSTACLE_MODEL_FILE",
                                            "output/02_thetastar_master_plan/"
                                            "planning_model_with_flz_support.xyz"))
    plot_density(net, frames, fig_dir / "01_density.png", stats.get("conflict_pts"))
    plot_timeline(timeline, sep_eff, fig_dir / "02_timeline.png", stats["peak_concurrent"])
    plot_energy(agents, params, fig_dir / "03_energy.png")
    if sched_info and sched_info.get("rows"):
        # the schedule as data, then as a picture
        pd.DataFrame(sched_info["rows"]).to_csv(
            output_dir / "departure_schedule.csv", index=False)
        sched_info["dock_capacity"] = sched_info.get("dock_cap_for_plot",
                                                     sched_info.get("dock_capacity", 0))
        plot_schedule(sched_info, params, fig_dir / "04_schedule.png")
        print(f"  schedule      : departure_schedule.csv + figures/04_schedule.png")
    if bool(pget(params, "MAKE_HTML", True)):
        print("  writing interactive HTML ...")
        write_html(net, frames, params, sep_eff, output_dir / "agents_animation.html",
                   velocity=metrics.get("velocity"))
    if bool(pget(params, "MAKE_AGENT_ROUTE_HTML", True)):
        print("  writing per-agent route HTML (agent_route/) ...")
        n_ar = write_agent_routes(net, frames, params, output_dir,
                                  output_dir / "agent_missions.csv")
        print(f"    {n_ar} per-agent route files in {output_dir/'agent_route'}")
    if bool(pget(params, "MAKE_ANIMATION", True)):
        print("  rendering animation (GIF) ...")
        make_animation(net, frames, params, sep_eff, output_dir / "agents_animation.gif")
    if bool(pget(params, "MAKE_PYVISTA", True)):
        print("  rendering PyVista 3D view (movie + interactive HTML) ...")
        try:
            render_pyvista(net, frames, params, sep_eff, output_dir)
        except Exception as e:
            print(f"    WARNING: PyVista render failed ({type(e).__name__}: {e})")

    print(f"Done. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
