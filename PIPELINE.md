# Pipeline — two parallel branches

After the shared random map the study splits into two branches that differ only
in the path-finding method, so FMM and Theta* can be compared end to end.

Every stage exists as `NN a` and `NN b`: **a = FMM**, **b = Theta***.

| Stage | branch (a) FMM | branch (b) Theta* | what it does |
|---|---|---|---|
| 01 | `01_generate_random_2d_node_riskmap.py` | *(shared)* | random map + riskmap |
| 02 | `02a_route_plan_fmm.py` | `02b_route_plan_thetastar.py` | plan the routes |
| 03 | `03a_route_density_fmm.py` | `03b_route_density_thetastar.py` | density → traffic nodes (TN) + relief nodes (RN) |
| 04 | `04a_master_corridor_fmm.py` | `04b_master_corridor_thetastar.py` | re-plan DB↔DK through the TN network |
| 05 | `05a_corridor_network_fmm.py` | `05b_corridor_network_thetastar.py` | node circles, 2 lanes per leg, roundabouts |
| 06 | `06a_costmap_fmm.py` | `06b_costmap_thetastar.py` | price the network → slowness cost-map |
| 07 | `07a_simulate_fmm_scheduling.py` | `07b_simulate_thetastar_scheduling.py` | coordination: scheduling + ORCA |

## Running

```bash
python 01_generate_random_2d_node_riskmap.py   # once, shared by both branches
./run_branch_a_fmm.sh                          # branch (a)
./run_branch_b_theta.sh                        # branch (b)
```

Stage 06 prices the stage-05 network with four layers — **economic balance**
(cân bằng kinh tế), **air operational safety**, **ground safety**, and
optionally **measured traffic** — and writes the slowness map every later stage
reads (`true velocity = slowness × base`, `conflict = 1 − slowness`). The first
three need nothing but stage 01 + stage 05, so 06 runs straight after 05; the
traffic layer is folded in on a second 06 run once a pass-1 sim exists. The
runners do 06 → 07 → 06 → 07 for exactly that reason. Weights and every layer
knob live in `params/<branch>/06_costmap.params`.

`output/<branch>/06_costmap/costmap.html` is the interactive view of it: one
radio per component map, an opacity slider, a toggle per model-node family
(DB / DK / TN / roundabouts / FLZ / RA zones / the step-01 model grid), lanes
coloured by each leg's composite risk, and a cursor readout that reports every
layer at once. `MAKE_HTML = False` (or `--no-html`) turns it off; it is
independent of `MAKE_FIGURES`.

## Why stage 07 carries a `_scheduling` suffix

`07[ab]_simulate_*_scheduling.py`, `src/engine_simulate_scheduling.py`,
`params/<branch>/07_simulate_scheduling.params` and
`output/<branch>/07_agent_sim_scheduling/` are named for the coordination method
they implement: departures are planned ahead as CTOTs (with dock-pad and energy
feasibility booked before launch) and separation is kept tactically in flight,
with ORCA inside the roundabouts. The suffix is deliberate — a different
coordination model (reservation-based 4-D contracts, market/priority auctions,
free-flight self-separation) belongs in a sibling `07x_simulate_<method>_*`
rather than silently replacing this one, so the two can be compared on the same
network. Nothing else in the pipeline hard-codes the stage-07 name; the runners
and stage 06's `--traffic` path are the only references.

## Coordination (stages 05 → 07)

The separation standard is **50 m horizontal** (any corridor, same flight
level) and **30 s longitudinal** (in trail, same corridor). Three mechanisms
hold it, each where it actually applies:

- **Strategic — departure scheduling.** `SCHEDULE_MODE` gives every mission a
  CTOT before the run: per-departure-lane headway, a predicted-airborne cap and
  a per-origin fair share. It replaces the reactive launch metering rather than
  stacking on top of it.
- **Structural — corridor geometry.** Two lanes per leg kept ≥ 50 m apart, one
  per travel direction, split into flight levels by heading. On a leg an agent
  is a point on its lane centreline.
- **Tactical — ORCA in the roundabouts.** Stage 05 sizes each ring from the
  predicted density in `03_route_density/route_density.npz`, subject to the
  ring **buffer** clearing every obstacle; busier junctions get more room
  (77–126 m here, from a fixed 40 m). That turns a ring from a 1-D circle —
  which at 50 m separation holds only five agents, against an observed peak of
  nine — into a 2-D manoeuvring area, and `src/orca.py` flies it: agents pick a
  free 2-D velocity by reciprocal collision avoidance, circulating CCW until
  their exit bearing comes up and then peeling off right.

`src/orca.py` is a standalone numpy implementation, matching `src/fmm.py`'s
no-dependency style. ORCA is exactly symmetric and therefore deadlocks on
head-on and antipodal pairs, so `ORCA_BIAS_DEG` rotates each preferred velocity
a few degrees right — which also matches right-hand traffic.

Compliance is measured, not assumed: `metrics.json` carries every same-level
pair-sample checked against the 50 m standard, and `separation_violations.csv`
records where each loss happened.

## Layout

```
<root>/NN[ab]_*.py     the ONLY .py in the root: thin branch launchers that
                     call an engine with the branch's parameters. 04a/04b are
                     the exception -- the two master-corridor solvers are
                     genuinely different code, so they are real scripts.
src/engine_*.py      shared implementations (route plan, density, corridor
                     network, cost-map, simulation). They live in src/ but are
                     ANCHORED ON THE PROJECT ROOT: params, output trees and the
                     root-level 04a/04b scripts are all addressed relative to
                     the root, so each engine sets THIS_DIR to its parent's
                     parent and puts the root on sys.path. That means an engine
                     runs the same through its launcher, directly as
                     `python src/engine_x.py`, or from any working directory.
src/*.py             libraries the engines import (fmm, orca, thetastar,
                     maprule, route_html, costmap_html, model_io, ...)
params/a_fmm/        parameters for branch (a)
params/b_theta/      parameters for branch (b)
output/a_fmm/        results of branch (a)
output/b_theta/      results of branch (b)
output/legacy/       results from before the fork, kept for reference; no stage
                     reads or writes them any more
```

Stages 02, 03, 05, 06 and 07 run the SAME engine in both branches -- only the
paths differ, which is what makes the comparison fair.

## Why the branches diverge

They are not two renderings of one plan. FMM and Theta* lay their routes
differently, so the density field differs, so the traffic nodes differ, so the
corridor network and the simulated traffic differ. One run gave branch (a)
40 TN + 11 RN against branch (b) 15 TN + 13 RN.

## Legacy — kept deliberately, as comparison baselines

No active stage imports any of these, but **do not delete them**. They are the
earlier generation of the study, retained so the current pipeline can be
measured against a genuinely different approach rather than only against its own
earlier parameters.

| file | what it is | what it is a baseline for |
|---|---|---|
| `02_run_theta_plan.py` | the original node-based Theta* master planner (v15) | the pre-fork route plan, before stages 02/04 split into launcher + engine |
| `03_cluster_theta_routes.py` | DBSCAN clustering of route hits | the current density-field route to traffic nodes in stage 03 |
| `src/kmean.py` | k-means route-hit density | same — a different clustering family for picking TN/RN |
| `04_run_master_plan_ACO_legacy.py`, `src/ACO.py`, `src/PSO.py`, `src/routeplan_PSO_ACO.py` | ant-colony and particle-swarm master-route planning | **the most valuable of the set**: metaheuristic planners against the deterministic FMM / Theta* the branches use now. A different family, not a different tuning. |
| `src/output_io.py` | the pre-fork CSV writer | reading the legacy outputs below |
| `output/legacy/` | 202 files: `02_route_plan`, `03_route_density`, `04_master_corridor_plan_FMM`, `05_master_corridor_theta`, and `compare_fmm_vs_theta.html` | the actual pre-fork numbers to compare against |

They also pin two dependencies (`scikit-learn`, `tqdm`) that nothing else needs —
see the README. That is the price of keeping them, and it is intentional.
