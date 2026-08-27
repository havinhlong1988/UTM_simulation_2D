# Roadmap

How the current design was arrived at, in the order it happened. Each section
names the commit, what changed, and — where it matters — what turned out to be
wrong on the way. The dead ends are kept deliberately: several of them are the
reason a parameter has the value it does.

Baseline for all of this is `0041d7c` (the two-branch fork).

---

## 1. Stage 03 — density HTML and lane-clearance repair · `7fb5aeb`

Already in the working tree when this work started; committed first so it stays
separable.

* stage 03 renders `route_density.html` — coverage, through-traffic and
  high-density fields with the picked TN/RN nodes;
* lane-clearance repair: a leg centreline is planned with `LEG_CLEARANCE_M`, but
  the two lane centrelines sit a node radius either side of it, so a lane *band*
  could still end up over an obstacle. Clearance 50 → 70 m, node circle
  50 → 75 m, node-shift passes 2 → 4.

---

## 2. Stage 05 — roundabouts sized by predicted density · `2abbd73`

Rings were pinned at a fixed 40 m radius.

**The finding that forced the change was arithmetic, not tuning.** A 40 m ring
has a 251 m circumference; at the 50 m separation standard that holds **five**
aircraft. The stage-07 run peaked at **nine** on the busiest ring. 15 of 19 rings
were oversubscribed, the worst by 1.8×. No coordination algorithm could have held
the standard there.

`load_density_field()` now samples stage 03's predicted route-density field at
each ring centre, and the radius gains `PREDICTED_DENSITY_GAIN_M · d^POW` on top
of the existing entries term.

| | before | after |
|---|---|---|
| radius | 40 m fixed | 77–126 m |
| correlation density ↔ radius | — | 0.87 |
| rings violating the obstacle buffer | — | **0 / 19** |

Obstacle clearance was already applied to the ring's **outer buffer edge**, not
its centreline, so the buffer cannot fall into a no-fly cell it is nominally
clear of. Verified independently: six rings sit exactly on the limit, tightest
ring-to-ring gap +283 m, and all 19 are limited by the desired radius rather
than by the fit. `roundabouts.csv` now records `pred_density`,
`radius_desired_m` and `radius_limited_by`, so a ring that had to shrink is
visible instead of silent.

---

## 3. Stage 06 — pricing the network · `e483919`

The cost-map was a single traffic-density heuristic, and it **crashed** without
a pass-1 simulation — so it could only ever run *after* stage 07.

It is now a four-layer assessment of the network stage 05 actually built:

| layer | what it prices |
|---|---|
| `econ` | *cân bằng kinh tế* — value (DB/DK demand + expected corridor throughput) minus cost (deadhead energy, distance from built infrastructure), plus an equity term penalising saturated cores |
| `air` | *an toàn vận hành trên không* — encounter density per leg, roundabout/junction merge hotspots, restricted-airspace margin |
| `ground` | *an toàn mặt đất* — built-up exposure below, discounted by sheltering, spread over the ballistic footprint, inflated where no FLZ is in reach |
| `traffic` | measured pass-1 density — now **optional**, dropped when absent |

**The subtle bug worth remembering:** every layer must be renormalised over the
same blend domain before mixing. Without it the weights are not what they say —
the economic balance varies far less across a built network than the safety
terms do, and spanned 0.17–0.51 in-corridor against 0–1 for air/ground, diluting
`econ` to about a third of its stated weight.

Because the traffic layer became optional, the runners were reordered to
**06 → 07 → 06 → 07**: the assessed map now exists before the first simulation
instead of after it.

FMM A/B on the new map: route length **+6.0 %**, ground exposure **−13.9 %**,
air **−9.0 %**, composite risk **−7.5 %**. The `econ` term moves **+4.2 %** the
other way — the equity term working as designed, since leaving a saturated core
costs economic centrality.

Also added `costmap.html`: one radio per component map, a toggle per model-node
family, lanes coloured by leg risk, and a cursor readout of every layer at once.

---

## 4. Stage 07 — ORCA rings, dock model, scheduling · `649e53a`

### The node mutex is off, and why

`NODE_MUTEX_ENABLE` deadlocks. An agent reserves a hub *while still approaching
it* and only releases after entering the next leg, so a blocked agent holds a
hub indefinitely and the waits close into a cycle.

Evidence, from an A/B differing in one flag: with the mutex on, **95 %** of
airborne hold-samples are node-mutex holds and there is **11× more holding**
overall. The `hold_cause` counter that proves this already existed in the engine
but was never exported.

### ORCA — `src/orca.py`

Written standalone (numpy only), matching `src/fmm.py`'s dependency-free style:
the RVO2 binding needs a Cython build, and the AVOCADO fork that extends it is
AGPL-3.0, which would be viral here.

Validated at the 50 m standard: head-on pair and 8-agent antipodal circle both
hold 50.0 m; 9 agents in a 126 m ring hold 86 m at full cruise with no boundary
escapes.

**Both symmetric cases deadlock at zero bias.** ORCA is perfectly symmetric, so
an exactly opposed pair reaches a reciprocal standoff and never resolves. A fixed
3° right bias breaks the tie — which also matches right-hand traffic. That is the
gap AVOCADO's opinion dynamics fills; the fixed bias is the cheap substitute.

Ring merge/exit losses fell from **60 % of all violations to 0.3 %** — 1 of the
353 remaining, in the final run.

### Two silent failures found on the way

* `circle_bound_line` had its half-plane **sign inverted**, so the ring's outer
  edge pushed agents *outward*. Every zone froze at ~0 % speed while the
  separation metric still looked healthy (86 m) — the metric revealed nothing.
* ORCA agents were skipped entirely by the 1-D loop, so they never reached the
  leg hand-off and circled forever.

---

## 5. Dock model and the achieved-velocity metric

Round trips park `MIN_DEST_IDLE_S = 1800 s` (30 min) at the destination dock and
charge to 100 % before the return leg.

**A parked drone is not traffic.** It holds no corridor, draws no hover power,
and is exempt from the separation checks — that part came free, since it was
already outside the airborne set. But `n_active` counted "launched-not-done", so
a docked drone held an **airborne slot** for the whole 30 minutes. With ~400
round trips against a 150 cap that would idle most of the budget on drones
sitting on pads. It is now recomputed each tick from who is actually flying.

`agent_missions.csv` gained `velocity_kmh` (distance ÷ total mission time,
door-to-door) and `air_velocity_kmh` (÷ airborne time only). The gap between
them *is* the dock stop: one-way missions show them identical, round trips show
≈8 vs ≈20 km/h.

---

## 6. Dock capacity, and the bug it exposed

`DOCK_CAPACITY = 5` pads per dock, each held `DOCK_PARK_BUFFER_S (600 s) +
MIN_DEST_IDLE_S` = 40 min. The scheduler predicts arrival with a cost-map-priced
ETA and **books a pad before clearing a launch**; if the dock would be full it
pushes the departure back.

Travel time is the integral of `ds/(v·slowness)`, so `route_time_factor()` uses
the **mean of 1/slowness**, not 1/mean — the latter flatters exactly the slow
stretches that dominate the time.

**The bug this exposed had been quietly crippling the scheduler.** Delivery
agents come out of `build_fleet` with `segs = []` and `route_len = 0` — their
route is planned lazily at launch. So in `schedule_departures` every mission
looked like a zero-length flight:

* the airborne booking `dur = route_len/speed` reserved **one bin** instead of a
  real interval, so the concurrency constraint was nearly inert;
* `_first_res` fell through to `__origin__<origin>`, so the "per departure
  corridor" headway was really a per-**origin** headway.

Fixed by resolving the nominal route in the scheduler itself. At 100 agents:
departure lanes 8 → 16, mean ETA 0.0 → 11.7 min, runtime dock-full holds
**1754 → 0**.

### The FCFS schedule chart · `04_schedule.png`

The scheduler now records the **binding constraint per mission** (ready /
corridor headway / airborne cap / origin cap / dock pad), which is what makes
the chart readable.

**It also corrected a claim of mine.** I had written in docstrings and params
that scheduling "only delays, never reorders". The CTOT curve is a sawtooth.
FCFS decides who gets **first pick** of a slot; each mission contends for its own
departure lane and its own destination dock, so one bound for a quiet lane clears
well before one queued ahead of it that needs a busy one.

### The real bottleneck

Neither resource is saturated — pads 45 %, airspace 41 % — yet makespan is
15.45 h. It is the **drone bases**:

| | parking missions | pads | pad-time needed |
|---|---|---|---|
| **DB** (2 bases) | 208 | 10 | **13.9 h** |
| DK (6 docks) | 207 | 30 | 4.6 h |

The last 100 CTOTs are 100 % DB-bound; DK pads sit empty after 6.1 h. The bases
carry **half** the parking traffic on a **quarter** of the pads. The lever is
pads allocated by demand, not by dock count.

---

## 7. Live pad reservations · `7e1fb5f`

A prediction made once at launch drifts over a 15 h schedule, and once the pads
run at 100 % every drift becomes a booking conflict — the static plan degraded
from 0 reactive dock-full holds at 100 agents to **3353** at 1000.

`PadBook` re-prices each airborne drone's remaining route every
`RESERVATION_UPDATE_S` (60 s) and re-books its pad when the arrival moves more
than `RESERVATION_TOLERANCE_S` (120 s). `PAD_LAUNCH_SLIP_S` keeps a drone on the
ground when the earliest free pad is that far past its predicted arrival.

At 1000 drones: reactive dock-full holds **3353 → 0**, horizontal violations
0.217 % → 0.184 %. Waiting moved off the dock and onto the apron, where it is
cheap.

**Design mistake worth not repeating.** I first made the *reservation* the gate
for landing — a drone could only touch down on a pad it held a started
reservation for. That produced 867 holds at 100 agents against a baseline of 0,
because a physically empty pad was being withheld for a drone still 20 minutes
away. **A reservation must not ration a resource that is already free.**
Physical occupancy is the authority for landing; reservations shape *launch*
decisions and give warning en route.

---

## 8. Energy feasibility · `3c63710`

Three missions ran the battery flat. All three were the 30 km/h class on the
longest route — which looked like bad luck and is structural.

With `P(v) = p0 + cd·v³` and true velocity `v·s`, energy per metre is

```
dE/dL = p0/(v·s) + cd·(v·s)²
```

**The hover floor is divided by speed.** Flying slower costs *more* energy per
metre, and a slow drone in slow airspace pays twice. The minimum is at
`v = (p0/2cd)^⅓` = **46.8 km/h**, so of the three classes the 30 km/h one is the
**expensive** one, not the frugal one: 108 Wh over 13 km against 92 Wh at
50 km/h.

`route_energy_wh()` integrates that through the cost-map, applied in both places:

* **scheduling** — price the binding leg (a round trip recharges at the dock, so
  the requirement is *one* leg, not both) and **raise the speed class** if the
  pack cannot cover it, candidates tried cheapest-energy first;
* **reservation** — re-price from the drone's actual position and remaining
  charge; one that comes up short is given the **earliest** pad, because holding
  it for a tidier slot is what kills it.

1000 drones: **0 battery-dead** (was 3), 29 speed promotions, 0 infeasible at any
class, peak actual draw 164.9 Wh of 200.

**The trade this buys is real**: more fast traffic raises horizontal separation
violations 0.184 % → 0.222 %. Velocity improves (10.5 → 10.8 km/h door-to-door)
for the same reason. Makespan is unchanged — the dock pads still set it.

> This conclusion depends on `DRAG_POWER_COEF` and `HOVER_POWER_W`. The estimator
> derives the optimum from those parameters, so it follows them; but "29 missions
> promoted" is specific to the current values.

---

## 9. Presentation · `083416d`, `c7ebaea`, `07b174d`, `617f75f`

* **Agent ids on both HTML maps**, decluttered in screen space (a label is
  skipped when another sits within 26 px), so the map stays readable at full
  extent and reveals more ids as you zoom. Default on.
* **Stage 07 renamed** with a `_scheduling` suffix — scripts, engine, params and
  output — so the coordination method is named rather than assumed.
* **Engines moved to `src/`**, leaving only run scripts in the root. Each engine
  is anchored on the project root (`THIS_DIR = parent.parent`, root on
  `sys.path`), so it runs the same through its launcher, directly, or from any
  cwd.

---

## Where it stands

1000 drones, branch (a):

| | |
|---|---|
| delivered | **1000 / 1000**, 0 battery-dead |
| makespan | 15.45 h |
| velocity | 10.8 km/h door-to-door, 17.8 airborne, 47.5 nominal cruise |
| horizontal separation | 0.222 % of pair-samples below 50 m |
| longitudinal | 250 m in-trail held |
| roundabout interior | 0 incursions |
| dock capacity | never exceeded; 0 arrivals waited for a pad |

---

## Known limits

* **"Conflict" is not "collision", and collisions are not modelled at all.** A
  conflict is a loss of *time* separation (`CONFLICT_TIME_S = 5 s`, i.e. 42–83 m
  depending on speed class). There is no airframe radius and no contact test —
  the run recorded a minimum gap of **0.49 m**, with 5 pairs under 1 m, which
  would physically be collisions logged as mere separation losses. The headline
  "1000/1000, 0 battery-dead" counts no losses from drones hitting each other.
* **The separation check is sampled every 20 s** while the sim steps at 1 s, so
  19 of every 20 steps go unchecked and brief encounters can slip between
  samples. The reported violation rate is a **lower bound**.
* **Cross-leg junction conflicts are unhandled** — 58 % of the remaining
  violations (205 of 353). That was the node mutex's job, and it is off because it deadlocks.
  A lock-ordering scheme would fix both.
* **The 1-D ↔ 2-D hand-off at the ORCA zone boundary** is the second-largest
  group (42 %, 147 of 353): nothing constrains the two models against each other at the seam.
* **The launch-retry loop is hot** — 112 k retries for 139 deferred missions,
  since a blocked mission retries every tick rather than sleeping until its pad
  is due.
* **No VO-MPC.** Coordination is planned scheduling plus single-step ORCA. ORCA
  is a velocity-obstacle method, but it solves one LP for the next velocity — no
  control sequence, no cost function, no receding horizon.

---

# Next steps

**The next move is 3-D.** VO-MPC, MARL and the stress-test programme below are
postponed — they are kept because each is specified against a defect this
document records, and none of that analysis expires.

Anything new goes in a sibling `07x_simulate_<method>_*` per the convention in
[PIPELINE.md](PIPELINE.md), so it can be compared against the scheduling build on
the same network rather than replacing it.

---

## 0. Moving to 3-D — the active track

### What "2-D" currently means here

The model is better described as **2.5-D than 2-D**, and the difference decides
how much work the move actually is:

| already present | but |
|---|---|
| `FLIGHT_LEVELS = 6` altitude bands, `LEVEL_SEP_M = 30` | a level is a property of the **leg** (`agent_level(a) = leg_level(a.cur_seg())`). It is assigned by the leg and **never changes in flight** — there is no climb or descent |
| the map carries a `z` column | every value is **0.0**. Obstacles are 2-D discs with no height, so nothing can be flown *over* |
| separation is checked per level | different levels are assumed **perfectly separated** — `same_level` gates the test, and no vertical geometry is ever evaluated |
| pyvista renders the levels stacked | display only; the levels are not a planning dimension |

So the vertical axis exists as a **label**, not as geometry. That is the gap to
close.

### What the move touches, stage by stage

1. **Stage 01 — give obstacles height.** A `z_max` per obstacle turns the map
   from discs into a true volume, which is what makes "fly over it" a decision
   rather than an impossibility. This is the change everything else depends on.
2. **Stage 02/04 — plan in 3-D.** `src/fmm.py` solves the Eikonal equation on a
   2-D grid; the method itself is dimension-agnostic, so the solver generalises
   to a voxel grid, but memory goes as the third power — a 101×101 grid becomes
   101×101×N_z. Expect to need a coarser vertical resolution than horizontal.
3. **Stage 06 — the cost layers become volumes.** Ground risk genuinely depends
   on altitude (higher = wider ballistic footprint but more glide time to an
   FLZ), so the ground layer stops being a 2-D field and gains real structure.
   The economic and air layers extend more simply.
4. **Stage 07 — the real work.**
   * **Continuous altitude with climb/descent.** The power model
     `P(v) = p0 + cd·v³` has **no vertical term**; climbing is currently free.
     A climb power term is required before any 3-D energy result means anything,
     and note that the energy gate is already tuned against the 2-D model.
   * **A true 3-D separation standard** — a cylinder (50 m horizontal AND 30 m
     vertical) replacing "different level ⇒ safe". This will *raise* the
     reported violation rate, because pairs currently exempted by the
     `same_level` gate start being checked.
   * **3-D ORCA.** ORCA generalises cleanly — half-**spaces** instead of
     half-planes, and the linear program becomes 3-D. `src/orca.py` is 2-D
     throughout (`_det`, the leg construction, `circle_bound_line`), so this is
     a rewrite of the geometry, not a parameter change. The validation suite in
     its docstring should be extended alongside.
   * **Level change as a coordination resource.** The thing the current model
     structurally cannot express: climb to overtake, or to resolve a merge. This
     is the payoff — it directly attacks the two largest remaining violation
     groups, cross-leg junctions (58 %) and the ring-boundary seam (42 %), both
     of which are conflicts between agents that simply have nowhere to go.

### Carry these forward

* **Every result to date is one map seed and one fleet seed** (see the
  methodological gap below). Do not use the 2-D numbers as a baseline for 3-D
  without first establishing their spread.
* **Collisions are still not modelled**, and in 3-D the `same_level` exemption
  that hid some encounters disappears — so the violation counts will move for
  two different reasons at once. Add the collision metric *before* the move, or
  the two effects cannot be told apart.
* Keep 2-D runnable. The branch structure already supports a parallel build; a
  3-D stage set should sit beside the 2-D one, not replace it.

---

## A. VO-MPC — postponed

Today's tactical layer is **VO but not MPC**: ORCA solves one linear program for
the next velocity. `ORCA_TAU_S = 4.0` is the geometric look-ahead of the velocity
cone, not a prediction horizon — there is no control sequence, no cost function
and no receding horizon.

The VO half is already built and validated, so the step is contained: keep the
ORCA half-planes from `src/orca.py` as **constraints**, and solve over an N-step
horizon with an objective instead of a single projection.

```
minimise  Σ ‖p_k − centreline‖²  +  λ_u‖Δv_k‖²  +  λ_e·energy(v_k)
s.t.      ORCA half-planes at each step k
          ‖v_k‖ ≤ v_max, boundary constraints (ring edge, island)
```

What it should buy — each of these is a defect this document already records:

* **Removes the `ORCA_BIAS_DEG = 3.0` hack.** ORCA is exactly symmetric, so an
  opposed pair deadlocks; a fixed 3° right bias currently breaks the tie. An
  asymmetric cost term resolves it on principle rather than by nudge.
* **A candidate fix for the 1-D ↔ 2-D seam** (42 % of remaining violations). An
  MPC horizon spans the hand-off, so the drone plans *through* the zone boundary
  instead of switching models at it.
* **Replaces the reactive leg speed law** with one that anticipates the leader
  rather than ramping down on present clearance.

Cost and risks: roughly **N× the per-agent compute** — the current run does 224 k
single-step ORCA solves. A QP solver becomes a dependency unless the problem is
kept linear. And the horizon must be short enough that the constant-neighbour-
velocity assumption still holds; ORCA's guarantee does not automatically extend
to a multi-step plan.

**Measure against** the current run, same network and seed: separation violations
by category, achieved velocity, makespan, and solve time per agent-step.

---

## B. Multi-agent reinforcement learning — postponed

The simulator is already close to an environment — state, action and reward all
exist as quantities the engine computes.

| | candidate |
|---|---|
| observation | own leg progress + remaining route, k nearest same-level neighbours (relative position/velocity), local cost-map slowness, battery, time to the booked pad |
| action | speed along the lane, plus lateral offset inside a ring zone — the same two degrees of freedom the current model exposes |
| reward | delivered on time − separation violations − energy − holding: the terms already in `metrics.json` |

**The blocker is speed, and it is a hard one.** A 1000-drone run takes ~15
minutes; MARL needs millions of episodes. Before any training is meaningful the
simulator needs a headless vectorised fast path — no HTML, no figures, no
per-agent trajectory logging, many environments stepped in parallel. That
refactor is a prerequisite, not a detail, and it is worth doing anyway because it
also makes the stress-test sweeps below cheap.

Two risks worth naming before investing:

* **Reward shaping decides the answer.** The weights between "delivered" and
  "separation kept" *are* the policy; a learned agent will exploit whatever the
  reward under-specifies. The cost-map layers (econ / air / ground) are a
  reasonable starting basis precisely because they were designed as an explicit
  trade-off rather than a single scalar.
* **Non-stationarity.** Every agent learning at once means each one's environment
  is moving. Self-play against a frozen opponent pool, or centralised training
  with decentralised execution — not naive independent learners.

A fair comparison also needs the **scripted baseline in the same environment**
(the current scheduling + ORCA build), or an improvement cannot be attributed to
learning rather than to the environment having changed.

---

## C. Stress tests — postponed

### Methodological gap to close first

Every result in this document — 1000/1000 delivered, 0.222 % violations, the DB
pad bottleneck — comes from **`SIM_SEED = 12345` on map `seed2298177982`**. One
map, one fleet realisation. None of it has an error bar, and the ranking of two
designs could plausibly flip on another map.

`--seed` is already a CLI override, so this is cheap:

```bash
for s in 1 2 3 4 5 6 7 8 9 10; do
  python 07a_simulate_fmm_scheduling.py --seed $s --no-pyvista
done
```

**Report medians and spread, not single runs**, and re-check the headline claims
against them.

### Load and capacity sweeps

| sweep | knob | what it answers |
|---|---|---|
| demand | `N_AGENTS` 250 → 4000 | where throughput saturates, and which constraint binds at each level |
| dock pads | `DOCK_CAPACITY` 2 → 20 | confirms the DB-pad bottleneck in the full sim, not only in the scheduler-only sweep already done |
| pads by demand | per-dock capacities | the actual proposal — more pads at DB01/DB02 — which today's single global number cannot express |
| airborne cap | `MAX_CONCURRENT` 50 → 400 | at what point airspace replaces pads as the binding constraint |
| separation | `SEPARATION_M` 30 → 100 m | the cost of the safety standard, in throughput |
| flight levels | `FLIGHT_LEVELS` 2 → 10 | vertical stacking as the alternative to widening rings |
| fleet mix | `SPEED_CLASSES_KMH` | the energy gate promotes 29 missions today; how that scales |

### Degraded modes

None are currently modelled, and each targets a specific known weakness:

* **Dock outage** — take DB01 offline mid-shift. The pad bottleneck predicts this
  is severe; the live reservation system should re-plan around it.
* **Corridor closure** — close a trunk leg and see whether the reroute logic
  finds capacity or gridlocks.
* **Battery derate** — 200 → 150 Wh, an aged fleet. The energy gate should promote
  far more missions; past some derate no class is feasible and the gate should
  say so rather than launch into failure.
* **Demand burst** — compress `ARRIVAL_WINDOW_H` so all demand arrives at once,
  against a scheduler that assumes it can spread departures.
* **Wind field** — a directional bias on true velocity. The cost-map already
  carries a slowness field so the mechanism exists; the asymmetry (cheap
  downwind, expensive upwind) is what the energy gate has never been tested on.

### Fidelity checks

* **Sampling rate.** The separation check runs every `SAMPLE_EVERY_S = 20 s`
  against `DT_S = 1 s` — 19 of every 20 steps go unchecked. Re-run at
  `SAMPLE_EVERY_S = 1` and measure how far the violation rate rises. Everything
  reported here is a lower bound until this is quantified.
* **Collision metric.** Add `AIRFRAME_RADIUS_M` and count contacts separately from
  separation losses. The run recorded a 0.49 m minimum gap with 5 pairs under
  1 m — physically collisions, currently logged as separation losses.
* **Time-step convergence.** Halve `DT_S` and confirm the results are a property
  of the model rather than of the integrator.
