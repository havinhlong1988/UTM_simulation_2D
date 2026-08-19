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
