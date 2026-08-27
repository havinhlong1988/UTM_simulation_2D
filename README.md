# UTM_simulation_2D

A 2-D simulation of a low-altitude UAV traffic-management (UTM) network: it
generates a random city-like map, plans routes across it, condenses those routes
into a corridor network with roundabouts, prices that network by economics and
safety, and finally flies a fleet of drones through it under a coordination
model.

Everything is reproducible from the scripts — no data files need to be supplied.

---

## 1. Environment

**Python 3.11.** The pipeline is numerical + matplotlib; there is no build step
and no compiled extension.

### Required

```bash
pip install numpy pandas matplotlib scipy
```

| package | used for |
|---|---|
| `numpy` | every stage |
| `pandas` | the CSV interchange between stages |
| `matplotlib` | all figures (uses the headless `Agg` backend, no display needed) |
| `scipy` | KD-trees, Gaussian filters, distance transforms |

### Optional

Each of these is imported lazily and has a working fallback, so the pipeline
runs without them — you lose a feature, not the run.

```bash
pip install shapely pillow pyvista
```

| package | if missing |
|---|---|
| `shapely` | stage 05 falls back to its own polyline offset; corridor geometry is slightly less exact |
| `pillow` | HTML field images are encoded via matplotlib instead (larger files) |
| `pyvista` | the 3-D flight replay is skipped; set `MAKE_PYVISTA = False` or pass `--no-pyvista` |

### Legacy only

`scikit-learn` and `tqdm` are needed only by the superseded scripts listed in
[PIPELINE.md](PIPELINE.md) (`02_run_theta_plan.py`, `03_cluster_theta_routes.py`,
`04_run_master_plan_ACO_legacy.py`). No active stage imports them.

### Verify the install

```bash
python -c "import numpy, pandas, matplotlib, scipy; print('ok')"
```

### A note on the interpreter

Use the interpreter that has these packages. If you keep them in a conda env,
call it explicitly rather than relying on `python` resolving to it:

```bash
~/anaconda3/envs/utm/bin/python 01_generate_random_2d_node_riskmap.py
```

---

## 2. Running it

Stage 01 is shared by both branches — run it once, then run a branch:

```bash
python 01_generate_random_2d_node_riskmap.py
./run_branch_a_fmm.sh
```

`run_branch_a_fmm.sh` executes stages 02 → 07 in order (including the
06 → 07 → 06 → 07 cost-map/simulation loop) and writes everything under
`output/a_fmm/`. `run_branch_b_theta.sh` does the same for the Theta* branch.

Any single stage can also be run on its own:

```bash
python 05a_corridor_network_fmm.py          # through its launcher
python src/engine_costmap.py --help         # or the engine directly
```

> **Careful:** running one stage alone writes into the real `output/` tree and
> can desync it — the committed cost-map is built from pass-1 traffic while the
> committed simulation consumed it. When you are only checking something, point
> the stage at a scratch directory (`--out-dir /tmp/...`, or `OUTPUT_DIR` in the
> params file).

---

## 3. How the directory works

```
01_*.py                shared stage 01: the random map
NN[ab]_*.py            the ONLY .py files in the root — thin launchers that call
                       an engine with one branch's parameters.
                       a = FMM branch, b = Theta* branch.
                       04a/04b are the exception: the two master-corridor
                       solvers are genuinely different algorithms, not launchers.
run_branch_*.sh        run a whole branch end to end

src/engine_*.py        the actual implementations, shared by both branches
src/*.py               libraries the engines use (fmm, orca, thetastar, maprule,
                       route_html, costmap_html, model_io, ...)

params/<branch>/       one params file per stage, per branch — this is where you
                       change anything. Plain `KEY = value` text.
output/<branch>/       one directory per stage, named for it
docs, PIPELINE.md      what each stage does and why
roadmap.md             the development history and what was learned
```

### The two-branch design

After the shared map the study **splits into two branches that differ only in
the path-finding method**, so FMM and Theta* can be compared end to end rather
than only at the route stage:

```
01 map ─┬─ a: 02a → 03a → 04a → 05a → 06a → 07a   → output/a_fmm/
        └─ b: 02b → 03b → 04b → 05b → 06b → 07b   → output/b_theta/
```

Stages 02, 03, 05, 06 and 07 run the **same engine** in both branches — only the
params and the output tree differ, which is what makes the comparison fair.

### Launcher → engine → params

A launcher is deliberately tiny. It exists to name one branch's params file:

```python
sys.argv = [str(HERE / "src" / "engine_costmap.py"),
            "--param-file", "params/a_fmm/06_costmap.params", ...]
runpy.run_path(str(HERE / "src" / "engine_costmap.py"), run_name="__main__")
```

So: **to change behaviour, edit a params file; to change logic, edit an engine.**

### Engines are anchored on the project root, not on `src/`

Every path an engine resolves — params files, output trees, and the root-level
`04b` script that `engine_corridor_network` loads by path — is written relative
to the **project root**. Each engine therefore sets `THIS_DIR` to its parent's
parent and puts the root on `sys.path`. An engine runs identically through its
launcher, directly as `python src/engine_x.py`, or from any working directory.

### Stage names carry their method

`07*_scheduling` says stage 07 coordinates by **planned departures (CTOT) plus
tactical separation**, with ORCA inside the roundabouts. A different
coordination model belongs in a sibling `07x_simulate_<method>_*` rather than
replacing this one, so the two can be compared on the same network.

---

## 4. Outputs

Each stage writes CSVs (the interchange format between stages), PNG figures, a
`metrics.json`, and for most stages a self-contained interactive HTML map — pan,
zoom, and per-layer toggles, no server and no external assets.

Worth opening first:

| file | shows |
|---|---|
| `output/<b>/05_corridor_network/corridor_network.html` | the network: lanes, roundabouts, nodes |
| `output/<b>/06_costmap/costmap.html` | the cost layers; switch between them, hover to read all at once |
| `output/<b>/07_agent_sim_scheduling/agents_animation.html` | the fleet flying, with agent ids |
| `output/<b>/07_agent_sim_scheduling/figures/04_schedule.png` | the FCFS departure schedule |

See [PIPELINE.md](PIPELINE.md) for what each stage does, and [roadmap.md](roadmap.md)
for how the current design was arrived at — including its **known limits** and the
**next steps**. The active track is the move to **3-D**; VO-MPC, multi-agent RL and
the stress-test programme are specified there but postponed.
