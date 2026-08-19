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
