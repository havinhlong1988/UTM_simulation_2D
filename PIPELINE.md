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
| 06 | `06a_costmap_fmm.py` | `06b_costmap_thetastar.py` | traffic → slowness cost-map |
| 07 | `07a_simulate_fmm.py` | `07b_simulate_thetastar.py` | coordination model |

## Running

```bash
python 01_generate_random_2d_node_riskmap.py   # once, shared by both branches
./run_branch_a_fmm.sh                          # branch (a)
./run_branch_b_theta.sh                        # branch (b)
```

Stage 07 runs twice: once before the cost-map exists, then again after stage 06
has built it from that first pass. The runners already do this.

## Layout

```
engine_*.py          shared implementations (route plan, density, corridor
                     network, cost-map, simulation)
NN[ab]_*.py          branch launchers -- thin, they call an engine with the
                     branch's parameters. 04a/04b are the exception: the two
                     master-corridor solvers are genuinely different code.
params/a_fmm/        parameters for branch (a)
params/b_theta/      parameters for branch (b)
output/a_fmm/        results of branch (a)
output/b_theta/      results of branch (b)
```

Stages 02, 03, 05, 06 and 07 run the SAME engine in both branches -- only the
paths differ, which is what makes the comparison fair.

## Why the branches diverge

They are not two renderings of one plan. FMM and Theta* lay their routes
differently, so the density field differs, so the traffic nodes differ, so the
corridor network and the simulated traffic differ. One run gave branch (a)
40 TN + 11 RN against branch (b) 15 TN + 13 RN.

## Legacy (in neither branch)

`02_run_theta_plan.py`, `03_cluster_theta_routes.py`,
`04_run_master_plan_ACO_legacy.py`, and in `src/`: `kmean.py`, `PSO.py`,
`ACO.py`, `routeplan_PSO_ACO.py`, `output_io.py` — superseded, imported by
nothing in the branches above.
