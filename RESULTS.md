# UTM Step-09 — Coordination Improvement: Results (A1–A9 campaign)

**Scope:** improve the deconfliction / coordination algorithm in
`09_simulate_agents_2d.py` (reduce conflicts, raise route utilization, cut holds)
**without changing** three invariants — the battery-dead failure model, mandatory
return-to-base, and the one-way (CCW) roundabout. Branch:
`roundabout-corridor-tuning`. Method: each candidate = one param flag, default
**OFF** (baseline byte-identical), A/B'd one-at-a-time over 5 seeds
(`12345, 54321, 11111, 22222, 33333`), judged on the §5 acceptance gates in
`progress.md`. Full detail and chronology: `progress.md`, `ledger/LEDGER.md`.

---

## 1. Headline

The binding constraint was **launch scheduling**, not spacing or routing: 1000
missions arrive at `t=0` against ~150 concurrent capacity (`peak_backlog=999`).
The two interventions that touch it or its downstream cost were kept; ring-spacing
and routing interventions were killed or neutral.

**Final config `baseline_p2` = A6 (demand-capacity balancing) + A4 (speed
control), both ON.** Cumulative 5-seed median vs the original frozen baseline
(`baseline_p0`):

| Metric | baseline_p0 | **baseline_p2 (A6+A4)** | Change |
|---|---|---|---|
| Missions completed | 877 | **925** | **+5.5%** |
| Total hold (min) | 12772 | **6911** | **−46%** |
| Battery-dead (lost drones) | 123 | **75** | **−39%** |
| Conflict samples | 4006 | 4049 | flat |
| Makespan `sim_end` (h) | 5.12 | 5.96 | +16% |
| Mean wait / leg (s) | 6119 | 6658 | +8.8% |
| Gridlock | False | False | — |

More delivered, far less holding, far fewer drones lost — at a contained latency
cost (the demand-smoothing trade, partly bought back by A4).

**Baked into the script.** The kept config is now the **default in
`09_simulate_agents_2d.py`**: `DCB_MODE` and `SPEED_CONTROL` default ON
(`DCB_CORRIDOR_SHARE=1.5`, `SPEED_CTRL_BAND_FACTOR=0.5`); the killed/neutral flags
(`RING_METER`, `TOLL_MODE`, `RESV_ROUTING`) default OFF. A run with **no flags set
reproduces `baseline_p2`**. The frozen baselines pin the flags explicitly
(`baseline_p0`/experiment configs set them OFF) so they — and the env-parity
handshake — still reproduce exactly.

---

## 2. Frozen baselines

| Baseline | Config | Role |
|---|---|---|
| `baseline_p0` | mutex OFF, ring one-way ON, no coordination extras | original reference **+ env-parity handshake ref** (unchanged) |
| `baseline_p1` | `baseline_p0` + **A6** (DCB, share 1.5) | reference after the first KEEP |
| `baseline_p2` | `baseline_p1` + **A4** (speed control, band 0.5) | **current best; phase-4+ A/Bs measure against this** |

Each `baseline_pN/` holds `run_seeds.sh`, `aggregate.py`, `baseline_summary.json`,
and per-seed `metrics/`. Regenerate: `bash baseline_pN/run_seeds.sh && python
baseline_pN/aggregate.py`.

---

## 3. Verdicts (doc A1–A9)

| Item | Flag | Verdict | Result (5-seed median vs its baseline) |
|---|---|---|---|
| **A6** demand-capacity balancing | `DCB_MODE` | **KEEP ★** | vs p0: completed +5.4%, holds −13%, deaths −38%; the lever |
| **A4** speed control | `SPEED_CONTROL` | **KEEP** | vs p1: makespan −4.2% (recovers ~24% of DCB's add), holds −38%, deaths 76→75 |
| **A5** system-optimum tolling | `TOLL_MODE` | NEUTRAL | all gates pass but nothing moves (routing ≠ bottleneck) |
| **A7** ring-entry metering | `RING_METER` | KILL | gap 65.8→94.8 m (G1′ pass) **but battery-dead +24% (G3 fail)** |
| **A7b** ring wrap-follow only | `RING_WRAP_FOLLOW` | KILL | worse than A7 (deaths +95%), gap still < 80 m |
| **A1** space-time reservation (SIPP) | `RESV_ROUTING` | NOT-VIABLE | doesn't scale (N=200 > 90 s vs 10 s); reactive-vs-planned mismatch |
| A2/A3 CBS/PBS + LNS | — | not attempted | same plan-first mismatch as A1 |
| A8 energy-aware charging | — | not attempted | out of scope this campaign |
| A9 conflict-prob + RTA | — | not attempted | out of scope this campaign |

All flags remain in the code, **default OFF**, so every baseline stays
byte-identical.

---

## 4. Why each landed where it did

- **A6 (KEEP, the winner).** The launch gate was a single global concurrency cap,
  letting the queue-head corridor hog the airborne budget. DCB meters each origin
  corridor to a fair-share cap and round-robins the rest, spreading the fleet.
  This is the only intervention that touched the binding constraint. A share
  sweep `{0.75, 1.0, 1.25, 1.5}` traced a clean throughput↔latency curve; **1.5**
  is the efficiency knee (0.75 is safety-max but its 8.67 h makespan overruns an
  ~8 h shift).

- **A4 (KEEP).** Baseline car-following is bang-bang (full speed to the leader−sep
  cap, then stop). A4 ramps speed down over a band above the floor and drains
  battery at the *actual* velocity. Band matters sharply: **0.5** recovers
  makespan cleanly; 1.0 over-packs creeping agents (conflicts +13%, G3 fails); 2.0
  far worse. It recovers *makespan* but **not mean_wait** — wait is launch-queue
  deferral, which air control cannot touch.

- **A5 (NEUTRAL).** Marginal-cost tolling is theoretically sound but ineffective
  here: throughput is launch-bound, so redistributing route choice can't help
  (higher toll multipliers were also flat). First proof the bottleneck isn't
  routing.

- **A7 / A7b (KILL).** The real sub-80 m ring gap comes from an un-wrapped ring
  coordinate at the 2·π·r seam. Enforcing ≥80 m circulation spacing forces
  entry/creep holds → hover-drain → **more battery deaths**, failing the hard INV-1
  gate. Neither the merge meter nor the wrap-follow alone escapes the trade.

- **A1 (NOT-VIABLE).** A reservation *plan* on a *reactive* time-stepped executor:
  the sim never executes the schedule, so execution diverges → agents stall →
  STUCK-timeout reroute → each reroute is a full time-aware replan → the reroute
  rate feeds back and wall time explodes super-linearly with N. Correct at small
  N, infeasible at N=1000.

**Structural takeaways.** (1) The lever for this workload is **demand/launch
scheduling** (A6), not spacing (A7) or routing (A5/A1). (2) **mean_wait is
launch-queue-bound** — only demand-side levers move it. (3) **Plan-first methods
(A1/A2/A3) need a schedule-following executor** to work here; bolting them onto the
reactive loop causes a reroute-feedback blowup.

---

## 5. Reproduce

```bash
conda activate utm   # numpy pandas matplotlib scipy
# any single A/B (example: the winner)
python 09_simulate_agents_2d.py --param-file params/phase2_A6_dcb.params \
       --seed 12345 --no-animation --no-html --no-pyvista
# frozen baselines
bash baseline_p2/run_seeds.sh && python baseline_p2/aggregate.py
# interactive viz (agents + inter-agent reference links; also per-agent pages)
python 09_simulate_agents_2d.py --param-file params/viz_a6.params \
       --seed 12345 --no-animation --no-pyvista   # -> output/09_viz_a6/agents_animation.html + agent_route/
```

Per-task A/B scripts live in `ledger/tasks/` (`phase2-A6-demand.sh`,
`phase3-A4-speed.sh`, `phase1-A5-tolling.sh`, `phase1-A7-ring.sh`, …); their
metrics are under `phase1/`, `phase2/`, `phase3/`.

---

## 6. Future work (needs a new shift)

- **Schedule-following executor** — make agents honor reservation windows
  (speed/launch timing), which would unlock A1 (SIPP) and A2/A3 (CBS/PBS).
- **A8 energy-aware routing + charge scheduling** — orthogonal to the above;
  could push battery-dead lower still.
- The launch backlog (`peak_backlog=999`) is inherent to the `t=0` demand dump;
  A6 mitigates its *effects* but a smoother arrival model would attack it directly.
