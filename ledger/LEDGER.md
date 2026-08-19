# Relay ledger — append-only journal

Shared brain for the Mac↔Linux Claude Code relay (local Claude memory does NOT
sync across machines; this file, in git, is the source of truth). Newest entry at
the bottom. Each shift appends one block; do not edit past blocks.

Format:
```
## <ISO time> · host=<mac|linux> · task=<id> · <OK|FAIL|NOTE>
- what ran / key numbers
- NEXT: <next_task> (baton released)
```

---

## 2026-08-18T23:30+07 · host=mac · task=bootstrap · NOTE

Phase −1 and Phase 0 complete on **mac**. State handed to the ledger.

- **Baseline frozen** (5 seeds, mutex OFF, ring one-way ON): median n_completed 877,
  gridlock False, @5h 87.6%, peak_backlog 999 (binding), min_same_lane_gap 65.8m.
  Full numbers in `phase0/baseline_summary.json`; narrative in `progress.md`.
- **Code fix applied** to `09_simulate_agents_2d.py` (ring same-lane gap uses
  s_offset; measurement-only). **Commit was blocked by a transient harness issue —
  verify it is committed before relying on it.**
- **Revised gate G1′**: `min_same_lane_gap_m ≥ 80m` (replaces the 250m
  `separation_respected`).

**BOOTSTRAP — do once, on mac, before the relay starts (needs push auth):**
1. Commit the metric fix (`09_simulate_agents_2d.py`) — see `progress.md` §8.
2. `git add params/baseline_p0.params phase0/ ledger/ .claude/skills/ progress.md docs/`
   then commit + `git push`. (gitignore `__pycache__/`, `*.pyc`, `ledger/relay.log`;
   never commit `.claude/settings.local.json`.)
3. On **linux**: `git pull`, recreate conda env `utm` (see `progress.md` §0),
   then run the first relay task.

- NEXT: **handshake-baseline** (run on linux first to confirm env parity)

## 2026-08-19T08:37:05+07:00 · host=linux · task=handshake-baseline · OK

First linux shift. Env parity confirmed against frozen Phase-0 baseline (seed 12345):
- n_completed 869 (ref 869) OK · gridlock False OK
- min_same_lane_gap_m 62.875 (ref 62.88, ±1m) OK · conflict_samples 3999 (ref 3999) OK
- [handshake] MATCH — env parity confirmed (utm conda env, ~/anaconda3/envs/utm).
- Cross-machine numbers can now be trusted on linux.
- NEXT: none `todo`. Phase-1 tasks (phase1-A5-tolling, -A4-speed, -A7-ring) are
  `blocked` pending implementation (one param flag each, default off, vs baseline).
  Baton released (idle).

## 2026-08-19T09:02:01+07:00 · host=linux · task=phase1-A7-ring · KILL

Implemented A7 ring-entry metering behind `RING_METER` (default OFF -> baseline
byte-identical). Root cause of the sub-80m ring floor: ring global progress
`s_local+s_offset` is NOT wrapped to the circumference, so an agent past the
2*pi*r seam sees no leader near angle 0 and closes inside the headway. A7 makes
ring car-following + ring merges wrap-aware (compare mod ring_circ) and meters the
merge to keep >= RING_METER_GAP_M (80m) clear both sides.

Code: `09_simulate_agents_2d.py` — LegSeg.ring_circ slot; ring_seg sets 2*pi*r;
`ring_meter`/`ring_meter_gap` params; wrap-aware branch in enter_leg merge check
and in the move-loop leader cap. Config: `params/phase1_A7_ring.params` (baseline
+ RING_METER=True). A/B: `ledger/tasks/phase1-A7-ring.sh`, 5 D2 seeds,
`phase1/A7/metrics/`.

A/B result (median of 5 seeds, A7 vs frozen baseline):
- min_same_lane_gap_m  65.8 -> 94.8  (min 81.1 all seeds) .... G1' PASS (target hit)
- gridlock             False ......................... G2 PASS
- n_battery_dead       123  -> 153  (+24%) ............ G3 FAIL  <-- kills it (INV-1)
- total_hold_minutes   12772 -> 16532 (+29%) .......... ★ wanted -15%, went +29%
- conflict_samples     4006 -> 4883  (+22%)
- n_completed          877  -> 847  (-3.4%)
- mean_wait / reroutes 6119->6654 (+9%) / 1063->1727 (+63%)
- peak_backlog         999 (unchanged; still the binding launch-backlog constraint)

VERDICT KILL (keep/kill rule §5): metering the ring entry forces entry holds ->
hover drain -> more battery deaths (hard-gate G3) and a hold/conflict cascade.
The *wrap fix* is correct and the gap target is reachable; the *cost* is the
merge meter + eager cross-seam yielding. Flag stays in, default OFF; baseline
untouched.

- NEXT: **phase1-A7b-ring** (blocked) — retry with ONLY the wrap-aware ring
  car-following (drop the merge meter) or a smaller RING_METER_GAP_M, to isolate
  which half costs G3 and get the gap up without the battery cascade.
  Baton released (idle).

## 2026-08-19T09:09:37+07:00 · host=linux · task=phase1-A7b-ring · KILL

Isolation follow-up to A7: split RING_METER into two independent flags
(RING_WRAP_FOLLOW = wrap-aware ring car-following; RING_MERGE_METER = metered ring
merge; RING_METER now a master enabling both). A7b runs RING_WRAP_FOLLOW ALONE to
test the hypothesis that the merge meter was the costly half.

Hypothesis was WRONG. A7b (5-seed median) vs baseline / vs A7:
- min_same_lane_gap_m  76.1  (min 65.8) ....... G1' FAIL (one seed never improved)
- n_battery_dead       240   (vs A7 153, base 123) ... G3 FAIL, MUCH worse
- total_hold_minutes   19948 (vs A7 16532, base 12772) ... worse
- n_completed          760   (vs A7 847, base 877) ... worse
- conflict_samples     5954 · n_reroutes 2016 ... worse

Finding: the WRAP-AWARE CAR-FOLLOWING is the expensive half -- agents braking for
cross-seam leaders around the whole ring cause far more holding (hover drain ->
240 deaths, throughput collapse) than the merge meter did. And without the meter
the gap target is NOT met. So neither half alone works; full A7 was the best ring
variant and still KILL.

CONCLUSION: enforcing >=80m ring circulation spacing is a DEAD END under this
demand -- every variant fails a hard gate on battery (holds -> hover -> death).
The binding constraint is peak_backlog=999 (launch backlog, progress.md §1),
unchanged by all ring work. Pivot to demand-side: A5 tolling / A6 demand-capacity
scheduling. Ring flags stay in, all default OFF (baseline untouched).

- NEXT: **phase1-A5-tolling** (blocked; needs implementation) -- or promote
  **phase2-A6-demand** (launch-slot scheduling) which attacks peak_backlog head-on.
  Baton released (idle).

## 2026-08-19T09:27:42+07:00 · host=linux · task=phase1-A5-tolling · NEUTRAL

Implemented A5 system-optimum marginal-cost tolling behind `TOLL_MODE` (default
OFF -> baseline byte-identical). The capture penalty is a congestion cost linear
in leg occupancy that each agent minimises selfishly (user equilibrium); for a
link cost linear in flow, the SO marginal-cost toll internalises the externality
by scaling that term (SO = UE*(1+beta) = 2x for beta=1). Code: `toll_mode` /
`toll_marginal_mult` params; one multiply in the leg_penalty rebuild (~1063).
Config: `params/phase1_A5_tolling.params` (TOLL_MODE=True, mult 2.0). A/B:
`ledger/tasks/phase1-A5-tolling.sh`, 5 D2 seeds, `phase1/A5/metrics/`.

A/B result (median of 5 seeds, A5 vs frozen baseline):
- ALL HARD GATES PASS: gridlock False (G2); n_battery_dead 121 <= 123 (G3);
  min_same_lane_gap 65.8 not-worse (G1').
- But NO objective moves: total_hold_minutes +3.2% (★ wanted -15%),
  n_completed +2 (877->879, noise), max_agents_on_a_lane 6->6 (load evenness
  flat), conflict_samples +1.9%, mean_wait +0.5%, n_reroutes -3%.
- Robustness: mult=6x (2 seeds) is also flat / slightly worse battery; peak_backlog
  stays 999, max lane stays 6 -> not a weak-multiplier artifact.

VERDICT NEUTRAL (keep/kill §5: all hard gates pass but NO ★ goal clears -> not
kept). SAFE, unlike A7. Root cause: peak_backlog=999 is unchanged -- throughput
is bound by the LAUNCH BACKLOG, not by route choice, so routing tolling cannot
help. Flag stays in, default OFF.

CUMULATIVE FINDING (3 experiments): A7/A7b (ring spacing) and A5 (routing) all
leave peak_backlog=999 untouched. The bottleneck is demand/launch scheduling.
-> A6 Demand-Capacity Balancing is the demonstrated real lever.

- NEXT: **phase2-A6-demand** (blocked; needs implementation) -- replace the
  n_active<max_concurrent launch gate with a per-corridor slot/capacity check.
  Baton released (idle).

## 2026-08-19T09:48:24+07:00 · host=linux · task=phase2-A6-demand · KEEP ★

Implemented A6 demand-capacity balancing behind `DCB_MODE` (default OFF ->
baseline byte-identical). The launch gate was a single GLOBAL cap
(n_active<max_concurrent); with 1000 missions at t=0 it lets the queue-head
corridor hog the airborne budget. DCB meters each ORIGIN corridor to a fair-share
airborne cap (cap = dcb_share * max_concurrent / n_origins) and round-robins
corridor-capped candidates to the BACK of the queue, spreading launches across
corridors. Code: `dcb_mode`/`dcb_share`/`dcb_cap` params + a per-origin gate in
the launch selection loop (active_per_key recomputed each tick from
active_agents, no cross-code bookkeeping). Config: `params/phase2_A6_dcb.params`
(DCB_MODE=True, share=1.0 = equal fair share). A/B: `ledger/tasks/phase2-A6-demand.sh`,
5 D2 seeds, `phase2/A6/metrics/`.

A/B result (median of 5 seeds, A6 share=1.0 vs frozen baseline):
- n_completed        877 -> 947   (+8.0%)  ★ throughput up
- total_hold_minutes 12772 -> 9093 (-28.8%) ★ clears the -15% goal
- n_battery_dead     123 -> 53    (-56.9%) G3 improves massively
- conflict_samples   4006 -> 3497 (-12.7%)
- n_reroutes         1063 -> 742  (-30.2%)
- min_same_lane_gap  65.8 -> 65.75 (flat, not-worse) · max_agents_on_a_lane 6 (flat)
- gridlock False (G2) · peak_backlog 999 (inherent t=0 dump, unchanged)
TRADEOFF (launch deferral, same cause both):
- mean_wait_for_leg_s 6119 -> 7664 (+25.3%)
- sim_end_hours (makespan) 5.12 -> 7.43 (+45%)  <- allowed by §5 (throughput up)

VERDICT KEEP: all hard gates pass; BOTH ★ goals clear (completions +8%, holds
-29%); every safety/flow metric improves. Cost is latency: DCB throttles launches
to sustainable network capacity, so the operation runs longer but completes more
and loses far fewer drones. This is the first KEEP after A7/A7b (KILL) and A5
(NEUTRAL), and confirms the whole diagnosis -- the bottleneck was LAUNCH
SCHEDULING, not ring spacing or route choice.

Sensitivity (DCB_CORRIDOR_SHARE, 2 seeds): share=1.0 (equal fair share) is the
sweet spot; share=1.5 is gentler (completed +5.4%, holds -12.6%, dead 76,
mean_wait +5.5%) -- a lower-latency alternative; share=2.5 reverts to baseline
(loose cap = no metering), confirming the mechanism.

- NEXT: **phase1-A4-speed** (blocked) -- speed control to smooth stop-and-go;
  best tested STACKED on A6 to try to recover some of the makespan/wait cost.
  Also: consider promoting the A6 config to a new frozen baseline for phase-3+.
  Baton released (idle).

## 2026-08-19T10:15:40+07:00 · host=linux · task=promote-baseline-p1 · NOTE

A6 share sweep + baseline promotion (user-chosen operating point).

DCB_CORRIDOR_SHARE sweep (5 seeds each, median vs baseline_p0):
  share  completed  hold_min  dead  conflict  mwait  makespan
  0.75      970(+11%)  6873(-46%)  30(-76%)  3246   8989   8.67h(+69%)
  1.0       947(+8%)   9093(-29%)  53(-57%)  3497   7664   7.43h(+45%)
  1.25      919(+5%)  10462(-18%)  81(-34%)  3811   6907   6.68h(+30%)
  1.5       924(+5%)  11162(-13%)  76(-38%)  4082   6455   6.22h(+21%)
Monotone throughput<->latency tradeoff. 1.25 is dominated by 1.5 (seed noise).
0.75 is safety-max (-76% deaths!) but makespan 8.67h EXCEEDS an ~8h shift.

CHOSEN: share=1.5 (efficiency knee) -- most benefit per latency cost, makespan
6.22h safely within shift. Locked into params/phase2_A6_dcb.params.

FROZEN new baseline **baseline_p1** = baseline_p0 + DCB_MODE (share=1.5):
  params/baseline_p1.params ; baseline_p1/run_seeds.sh + aggregate.py +
  baseline_summary.json + metrics/. 5-seed frozen medians: completed 924,
  hold 11162min, battery_dead 76, conflict 4082, mean_wait 6455s, makespan 6.22h,
  @5h 88.2%, gridlock False, routes 24/24. Phase-3+ improvements A/B against
  baseline_p1. baseline_p0 remains the env-parity handshake reference (unchanged).

- NEXT: **phase1-A4-speed** (blocked) -- speed control vs baseline_p1 (DCB already
  on), to smooth stop-and-go and claw back some makespan/wait. Baton released (idle).

## 2026-08-19T10:31:00+07:00 · host=linux · task=viz-interaction-links · NOTE

HTML animation (write_html / agents_animation.html) now draws INTER-AGENT
REFERENCE LINKS: per frame, every pair of SAME-flight-level agents within a
proximity radius (meta.link_watch_m = 1.6x max gap = 800m) gets a line -- red +
thick when closer than the required same-lane gap (max of the two speed-based
gaps, meta.gap_by_class), else a faint blue "keeping separation" link (alpha
fades with distance). Same-level only (different levels are vertically separated
-> never conflict), mirroring the sim's conflict model. New "interactions"
checkbox toggles them; a "Too-close pairs" live counter shows red-link count.
Per-agent row gained a 7th field (flight level). Verified in-browser: busiest
frame (144 airborne) = 213 links, 85 too-close.

Generate: params with MAKE_HTML=True (baseline_* have it OFF). Helper
params/viz_a6.params (baseline_p1 + MAKE_HTML). Not a sim-dynamics change;
baseline metrics unaffected. Static proof rendered from the embedded DATA.

- NEXT: unchanged (phase1-A4-speed). Baton idle.
