#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/ACO.py

Ant Colony Optimization route planner for the reduced LAE-UTM master graph.

The graph is built by src/routeplan_PSO_ACO.py from objective/TN/FLZ nodes.  ACO then
searches one route at a time, for example DK01 -> DB01 and DB01 -> DK01.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd


MODULE_VERSION = "v8_aco_progress_bars"


def _edge_key(a: int, b: int) -> tuple[int, int]:
    a = int(a); b = int(b)
    return (a, b) if a <= b else (b, a)


@dataclass
class ACOResult:
    success: bool
    route_key: str
    path: list[int]
    total_cost: float
    total_distance_m: float
    edge_indices: list[int]
    iterations: int
    ants: int
    message: str
    history_rows: list[dict] | None = None


class ACOPlanner:
    """ACO planner on an undirected reduced master graph."""

    def __init__(
        self,
        nodes_df: pd.DataFrame,
        edges_df: pd.DataFrame,
        *,
        alpha: float = 1.2,
        beta: float = 3.0,
        evaporation: float = 0.25,
        pheromone_q: float = 1.0,
        n_ants: int = 60,
        n_iterations: int = 80,
        max_steps: int = 80,
        random_state: int = 42,
        require_tn: bool = True,
        min_tn_nodes: int = 1,
        missing_tn_penalty: float = 5000.0,
        avoid_edge_penalty: float = 2.5,
        initial_pheromone_scale: float = 1.0,
        verbose: bool = False,
    ) -> None:
        self.nodes_df = nodes_df.copy()
        self.edges_df = edges_df.copy()
        if self.edges_df.empty:
            raise ValueError("ACOPlanner received an empty edge table.")

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.evaporation = float(np.clip(evaporation, 0.001, 0.95))
        self.pheromone_q = float(pheromone_q)
        self.n_ants = max(1, int(n_ants))
        self.n_iterations = max(1, int(n_iterations))
        self.max_steps = max(2, int(max_steps))
        self.rng = np.random.default_rng(int(random_state))
        self.require_tn = bool(require_tn)
        self.min_tn_nodes = max(0, int(min_tn_nodes))
        self.missing_tn_penalty = float(missing_tn_penalty)
        self.avoid_edge_penalty = float(avoid_edge_penalty)
        self.initial_pheromone_scale = float(initial_pheromone_scale)
        self.verbose = bool(verbose)

        self.node_role = dict(zip(self.nodes_df["node_id"].astype(int), self.nodes_df["role"].astype(str)))
        self._build_adjacency()
        self._init_pheromone()

    def _build_adjacency(self) -> None:
        self.adj: dict[int, list[tuple[int, int]]] = {}
        self.edge_lookup: dict[tuple[int, int], int] = {}
        for idx, row in self.edges_df.iterrows():
            u = int(row["u"]); v = int(row["v"])
            self.adj.setdefault(u, []).append((v, int(idx)))
            self.adj.setdefault(v, []).append((u, int(idx)))
            self.edge_lookup[_edge_key(u, v)] = int(idx)

    def _init_pheromone(self) -> None:
        desirability = self.edges_df.get("aco_initial_desirability", pd.Series(1.0, index=self.edges_df.index)).astype(float).to_numpy()
        desirability = np.where(np.isfinite(desirability) & (desirability > 0), desirability, 1.0)
        dmin, dmax = float(np.min(desirability)), float(np.max(desirability))
        if dmax > dmin:
            norm = (desirability - dmin) / (dmax - dmin)
        else:
            norm = np.ones_like(desirability)
        self.pheromone = {int(idx): float(1.0 + self.initial_pheromone_scale * norm_i) for idx, norm_i in zip(self.edges_df.index, norm)}

    def _edge_cost(self, edge_idx: int, avoid_edges: set[tuple[int, int]] | None = None, u: int | None = None, v: int | None = None) -> float:
        row = self.edges_df.loc[int(edge_idx)]
        cost = float(row.get("edge_cost", row.get("distance_m", 1.0)))
        if avoid_edges and u is not None and v is not None and _edge_key(int(u), int(v)) in avoid_edges:
            cost *= max(1.0, self.avoid_edge_penalty)
        if not math.isfinite(cost) or cost <= 0:
            cost = 1.0e9
        return cost

    def _path_stats(self, path: Sequence[int], avoid_edges: set[tuple[int, int]] | None = None) -> tuple[float, float, list[int], int]:
        total_cost = 0.0
        total_dist = 0.0
        edge_indices: list[int] = []
        for a, b in zip(path[:-1], path[1:]):
            key = _edge_key(a, b)
            if key not in self.edge_lookup:
                return math.inf, math.inf, [], 0
            ei = int(self.edge_lookup[key])
            edge_indices.append(ei)
            total_cost += self._edge_cost(ei, avoid_edges, int(a), int(b))
            total_dist += float(self.edges_df.loc[ei].get("distance_m", 0.0))

        tn_count = sum(1 for n in path[1:-1] if str(self.node_role.get(int(n), "")).upper().startswith("TN"))
        if self.require_tn and tn_count < self.min_tn_nodes:
            total_cost += self.missing_tn_penalty * float(self.min_tn_nodes - tn_count)
        return float(total_cost), float(total_dist), edge_indices, int(tn_count)

    def _construct_ant_path(self, start: int, goal: int, avoid_edges: set[tuple[int, int]] | None = None) -> list[int]:
        current = int(start)
        goal = int(goal)
        path = [current]
        visited = {current}

        for _ in range(self.max_steps):
            if current == goal:
                return path
            neighbors = self.adj.get(current, [])
            if not neighbors:
                return []

            candidates: list[tuple[int, int, float]] = []
            for nb, ei in neighbors:
                nb = int(nb)
                if nb in visited and nb != goal:
                    continue
                cost = self._edge_cost(ei, avoid_edges, current, nb)
                tau = max(float(self.pheromone.get(int(ei), 1.0)), 1.0e-12)
                eta = 1.0 / max(cost, 1.0e-12)
                score = (tau ** self.alpha) * (eta ** self.beta)
                if math.isfinite(score) and score > 0:
                    candidates.append((nb, int(ei), score))

            if not candidates:
                return []

            scores = np.asarray([c[2] for c in candidates], dtype=float)
            total = float(np.sum(scores))
            if not math.isfinite(total) or total <= 0:
                chosen = candidates[int(self.rng.integers(0, len(candidates)))]
            else:
                probs = scores / total
                chosen = candidates[int(self.rng.choice(len(candidates), p=probs))]

            current = int(chosen[0])
            path.append(current)
            visited.add(current)

            if current == goal:
                return path

        return []

    def _dijkstra(self, start: int, goal: int, avoid_edges: set[tuple[int, int]] | None = None) -> list[int]:
        start = int(start); goal = int(goal)
        pq: list[tuple[float, int]] = [(0.0, start)]
        dist = {start: 0.0}
        parent: dict[int, int] = {start: start}
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, math.inf):
                continue
            if u == goal:
                break
            for v, ei in self.adj.get(u, []):
                c = self._edge_cost(ei, avoid_edges, u, v)
                nd = d + c
                if nd < dist.get(int(v), math.inf):
                    dist[int(v)] = nd
                    parent[int(v)] = int(u)
                    heapq.heappush(pq, (nd, int(v)))
        if goal not in parent:
            return []
        path = [goal]
        cur = goal
        while parent[cur] != cur:
            cur = parent[cur]
            path.append(cur)
        path.reverse()
        return path


    def plan_route(self, start: int, goal: int, *, route_key: str = "route", avoid_edges: Iterable[tuple[int, int]] | None = None) -> ACOResult:
        start = int(start); goal = int(goal)
        avoid_set = {_edge_key(a, b) for a, b in (avoid_edges or [])}

        best_path: list[int] = []
        best_cost = math.inf
        best_dist = math.inf
        best_edges: list[int] = []
        history_rows: list[dict] = []

        for it in range(self.n_iterations):
            iteration_paths: list[tuple[list[int], float, list[int]]] = []
            for _ant in range(self.n_ants):
                path = self._construct_ant_path(start, goal, avoid_set)
                if not path or path[-1] != goal:
                    continue
                cost, dist_m, edge_indices, _tn = self._path_stats(path, avoid_set)
                if not math.isfinite(cost):
                    continue
                iteration_paths.append((path, cost, edge_indices))
                if cost < best_cost:
                    best_path = [int(v) for v in path]
                    best_cost = float(cost)
                    best_dist = float(dist_m)
                    best_edges = [int(e) for e in edge_indices]

            for key in list(self.pheromone.keys()):
                self.pheromone[key] = max(1.0e-9, (1.0 - self.evaporation) * self.pheromone[key])

            iteration_paths.sort(key=lambda x: x[1])
            for path, cost, edge_indices in iteration_paths[: max(1, min(8, len(iteration_paths)))]:
                deposit = self.pheromone_q / max(cost, 1.0e-12)
                for ei in edge_indices:
                    self.pheromone[int(ei)] = self.pheromone.get(int(ei), 1.0) + deposit

            history_rows.append({
                "route_key": route_key,
                "iteration": int(it),
                "n_feasible_ant_paths": int(len(iteration_paths)),
                "best_cost": float(best_cost) if math.isfinite(best_cost) else np.nan,
                "best_distance_m": float(best_dist) if math.isfinite(best_dist) else np.nan,
                "best_path_node_ids": ";".join(str(int(v)) for v in best_path) if best_path else "",
            })

            if self.verbose and (it + 1) % max(1, self.n_iterations // 10) == 0:
                print(f"[ACO] {route_key}: iter={it + 1}/{self.n_iterations}, best_cost={best_cost:.6g}")

        if not best_path:
            path = self._dijkstra(start, goal, avoid_set)
            if path:
                cost, dist_m, edge_indices, _tn = self._path_stats(path, avoid_set)
                best_path, best_cost, best_dist, best_edges = path, cost, dist_m, edge_indices
                history_rows.append({
                    "route_key": route_key,
                    "iteration": int(self.n_iterations),
                    "n_feasible_ant_paths": 0,
                    "best_cost": float(best_cost) if math.isfinite(best_cost) else np.nan,
                    "best_distance_m": float(best_dist) if math.isfinite(best_dist) else np.nan,
                    "best_path_node_ids": ";".join(str(int(v)) for v in best_path) if best_path else "",
                    "message": "fallback_dijkstra",
                })

        if not best_path:
            return ACOResult(False, route_key, [], math.inf, math.inf, [], self.n_iterations, self.n_ants, "No ACO/Dijkstra path found.", history_rows=history_rows)

        return ACOResult(True, route_key, best_path, float(best_cost), float(best_dist), best_edges, self.n_iterations, self.n_ants, "Path found by ACO.", history_rows=history_rows)
