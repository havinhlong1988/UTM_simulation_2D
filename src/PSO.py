#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/PSO.py

Particle Swarm Optimization helper for LAE-UTM master route planning.

Role in this workflow
---------------------
PSO does not directly build the final route.  It searches for a good early
condition / parameter set for ACO, including:
    - ACO alpha / beta / evaporation / pheromone Q
    - edge-cost weights for distance, route density, clearance, emergency FLZ
    - TN preference and FLZ normal-route penalty

The routeplan_PSO_ACO.py module provides the fitness function because it knows the
map, graph, DK/DB route pairs, TN candidates, FLZ nodes, RA/no-fly nodes, and
edge features.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable

import numpy as np
import pandas as pd


MODULE_VERSION = "v8_aco_progress_bars"


@dataclass(frozen=True)
class VariableSpec:
    name: str
    low: float
    high: float


def default_pso_bounds() -> list[VariableSpec]:
    """Default PSO search variables.

    The first four variables are ACO parameters.  The remaining variables are
    edge-cost weights later used by ACO.
    """
    return [
        VariableSpec("aco_alpha", 0.6, 2.5),
        VariableSpec("aco_beta", 1.0, 6.0),
        VariableSpec("aco_evaporation", 0.05, 0.65),
        VariableSpec("aco_pheromone_q", 0.5, 5.0),
        VariableSpec("distance_weight", 0.25, 3.0),
        VariableSpec("density_weight", 0.0, 2.5),
        VariableSpec("clearance_weight", 0.0, 3.0),
        VariableSpec("emergency_weight", 0.0, 2.0),
        VariableSpec("tn_bonus", 0.0, 2.0),
        VariableSpec("flz_penalty", 0.0, 3.0),
    ]


def bounds_from_params(params) -> list[VariableSpec]:
    """Read PSO_BOUNDS from params if present, otherwise use defaults.

    Expected format in params/routeplan_PSO_ACO.params:
        PSO_BOUNDS = {
            "aco_alpha": (0.6, 2.5),
            ...
        }
    """
    default = default_pso_bounds()
    user_bounds = getattr(params, "PSO_BOUNDS", None)
    if not isinstance(user_bounds, dict):
        return default

    out: list[VariableSpec] = []
    lookup = {v.name: v for v in default}
    for name, spec in lookup.items():
        if name in user_bounds:
            lo_hi = user_bounds[name]
            try:
                lo, hi = float(lo_hi[0]), float(lo_hi[1])
                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                    raise ValueError
                out.append(VariableSpec(name, lo, hi))
            except Exception:
                out.append(spec)
        else:
            out.append(spec)
    return out


class PSOOptimizer:
    """Small, deterministic PSO implementation for parameter tuning."""

    def __init__(
        self,
        variables: Iterable[VariableSpec],
        fitness_fn: Callable[[Dict[str, float]], float],
        *,
        n_particles: int = 24,
        n_iterations: int = 40,
        inertia: float = 0.72,
        cognitive: float = 1.45,
        social: float = 1.45,
        random_state: int = 42,
        verbose: bool = True,
    ) -> None:
        self.variables = list(variables)
        if not self.variables:
            raise ValueError("PSO requires at least one variable.")
        self.fitness_fn = fitness_fn
        self.n_particles = max(2, int(n_particles))
        self.n_iterations = max(1, int(n_iterations))
        self.inertia = float(inertia)
        self.cognitive = float(cognitive)
        self.social = float(social)
        self.rng = np.random.default_rng(int(random_state))
        self.verbose = bool(verbose)

        self.names = [v.name for v in self.variables]
        self.low = np.asarray([v.low for v in self.variables], dtype=float)
        self.high = np.asarray([v.high for v in self.variables], dtype=float)
        self.span = self.high - self.low

    def _to_dict(self, vector: np.ndarray) -> Dict[str, float]:
        return {name: float(value) for name, value in zip(self.names, vector)}

    def optimize(self) -> tuple[Dict[str, float], pd.DataFrame]:
        dim = len(self.variables)
        pos = self.low + self.rng.random((self.n_particles, dim)) * self.span
        vel = self.rng.normal(0.0, 0.15, size=(self.n_particles, dim)) * self.span

        pbest_pos = pos.copy()
        pbest_fit = np.full(self.n_particles, -np.inf, dtype=float)
        gbest_pos = pos[0].copy()
        gbest_fit = -np.inf

        history: list[dict] = []

        for it in range(self.n_iterations):
            for i in range(self.n_particles):
                x = self._to_dict(pos[i])
                try:
                    fit = float(self.fitness_fn(x))
                    if not np.isfinite(fit):
                        fit = -np.inf
                except Exception:
                    fit = -np.inf

                if fit > pbest_fit[i]:
                    pbest_fit[i] = fit
                    pbest_pos[i] = pos[i].copy()

                if fit > gbest_fit:
                    gbest_fit = fit
                    gbest_pos = pos[i].copy()

            row = {
                "iteration": int(it),
                "best_fitness": float(gbest_fit),
                "mean_particle_best_fitness": float(np.nanmean(pbest_fit[np.isfinite(pbest_fit)])) if np.any(np.isfinite(pbest_fit)) else -np.inf,
            }
            row.update({f"best_{k}": v for k, v in self._to_dict(gbest_pos).items()})
            history.append(row)

            if self.verbose:
                print(f"[PSO] iter={it + 1:03d}/{self.n_iterations:03d} best_fitness={gbest_fit:.6g}")

            r1 = self.rng.random((self.n_particles, dim))
            r2 = self.rng.random((self.n_particles, dim))
            vel = (
                self.inertia * vel
                + self.cognitive * r1 * (pbest_pos - pos)
                + self.social * r2 * (gbest_pos - pos)
            )
            max_vel = 0.50 * self.span
            vel = np.clip(vel, -max_vel, max_vel)
            pos = pos + vel

            # Reflective boundary handling.
            below = pos < self.low
            above = pos > self.high
            vel[below | above] *= -0.5
            pos = np.clip(pos, self.low, self.high)

        best = self._to_dict(gbest_pos)
        best["fitness"] = float(gbest_fit)
        return best, pd.DataFrame(history)
