# UTM Step-09 — Coordination Improvement: Progress, Fix Log & Roadmap

**Scope:** improve the deconfliction / coordination algorithm in
`09_simulate_agents_2d.py` — reduce flight-route conflicts and raise route
utilization — **without changing** three invariants:

- **INV-1** failure model (battery-dead; `CONTINGENCY_PROB=0.15`, `BACKUP_FRACTION=0.667`)
- **INV-2** mandatory return-to-base (round-trip + contingency-return legs)
- **INV-3** one-way roundabout (`RING_TRAVEL=True`, `RING_RIGHT_HAND=True`, CCW)

Branch: `roundabout-corridor-tuning`. Last updated: 2026-08-18.

---

## 0. How to resume (Linux) — quick start

```bash
# 1. clone / pull this repo (make sure the code fix + baseline files are pushed, see §8)
git checkout roundabout-corridor-tuning

# 2. recreate the conda env `utm` (env does NOT travel via git) with the sim deps:
#    numpy pandas matplotlib scipy   (+ pyvista imageio only if you want 3D/GIF)
conda create -n utm python=3.11 numpy pandas matplotlib scipy -y
conda activate utm

# 3. run ONE baseline seed (~30 s) to sanity-check:
python 09_simulate_agents_2d.py --param-file params/baseline_p0.params \
       --seed 12345 --no-animation --no-html --no-pyvista

# 4. run the full 5-seed baseline + aggregate:
bash phase0/run_seeds.sh          # ~2.5 min, writes phase0/metrics/*.json
python phase0/aggregate.py        # prints table, writes phase0/baseline_summary.json
```

`python` must be the **utm** interpreter. Do **not** use the macOS path
`/Users/vinhlongha/miniforge3/envs/utm/bin/python` on Linux — the scripts use a
bare `python` (override with `PYTHON=... bash phase0/run_seeds.sh` if needed).

---

## 1. Decisions locked (D1–D4)

| ID | Decision | Value |
|----|----------|-------|
| **D1** | Baseline config | `NODE_MUTEX_ENABLE=False` **+** `RING_TRAVEL=True` **+** `RING_RIGHT_HAND=True`. (Canonical params has **no** `RING_TRAVEL` line → defaults `False` → one-way roundabout was inactive; baseline must add it.) |
| **D2** | Seed sweep | `12345, 54321, 11111, 22222, 33333` (report median + range) |
| **D3** | Soft-goal thresholds | hold ↓≥15% · wait ↓≥10% · @5h ↑≥3 pts (a change is "kept" only if a ★ goal clears this and no hard gate fails) |
| **D4** | Ring separation issue | Root-caused: **part metric bug (fixed), part real** ring-entry gap → see §4/§5. G1 gate revised (§6). |

**Why mutex OFF:** the on-disk canonical run `output/09_agent_sim_2d/metrics.json`
is **gridlocked** (7/1000 done, 318 dead, `gridlock=true`) because
`params/simulate_agents_2d.params` has `NODE_MUTEX_ENABLE=True`, contradicting its
own comment (lines 183–185: *"left OFF … it GRIDLOCKS"*). That is commit
`45545d6` "freeze condition issue". Mutex OFF drains cleanly (≈877/1000).

**Binding constraint (author-confirmed + reproduced):** launch backlog — 1000
missions dumped at `t=0` (`ARRIVAL_WINDOW_H=0`) vs ~150 concurrent capacity →
`peak_backlog=999`, ~102 min mean wait. Named fixes: **demand smoothing** (A6) +
**risk-aware routing** (A1/A5).

---

## 2. Improvement roadmap (order matters)

Each improvement = one param flag, default OFF, A/B one-at-a-time on the 5 seeds.

| Phase | Item(s) | Rationale / order | Code touchpoint |
|-------|---------|-------------------|-----------------|
| **−1** | Strategic plan | done (`docs/strategic_plan_phase_minus1.docx`) | — |
| **0** | Baseline + A/B harness | **done** (this doc, `phase0/`) | — |
| **1** | **A5** system-optimum tolling · **A4** speed control · **A7** ring metering | drop-in, low risk; **A7 fixes the real ring-entry gap** (§5) | `leg_penalty:~1016` / `zone_cost:~788` · `move loop:~1094-1123` · `enter_leg:~914-918` (RING res, `s_offset` from `ring_seg:~377`) |
| **2** | **A6** demand-capacity balancing (launch slot scheduling) | attacks `peak_backlog=999` binding constraint | `launch:~1060` (replace `n_active < max_concurrent`) |
| **3** | **A1** space-time reservation (SIPP) — replaces ACO `plan_route` | foundational for phase 4 | `plan_route:~816` |
| **4** | **A2/A3** CBS / PBS + windowed LNS repair | CBS low-level planner = A1's space-time A* | wrap the sim loop; `try_reroute:~948` as repair op |
| **5** | **A9** conflict-probability + RTA decision layer · **A8** energy-aware routing + charge scheduling | A9 needs multiple actions (A4/A2) to choose among; A8 after congestion (hover-drain) reduced | `try_reroute`/`STUCK_TIMEOUT_S` · `plan_route` + `charge:~1190-1202` |

Full write-up with 17 citations: `docs/coordination_algorithm_improvements.docx`.

---

## 3. Frozen baseline (Phase 0, median of 5 seeds)

Config: N=1000, mutex OFF, ring one-way ON, `ARRIVAL_WINDOW_H=0`, `MAX_CONCURRENT=150`,
FLOW_MODE=spacing. Machine copy: `phase0/baseline_summary.json`.

| Family | Metric | Baseline (median · min..max) |
|--------|--------|------------------------------|
| Safety | `min_same_lane_gap_m` (fixed) | **65.8** (57.5..78.2) — sub-80m floor at ring entry |
| | `conflict_samples` | 4006 (3826..4214) |
| | `min_closest_approach_m` | 0.0 — real cross-leg junction coincidence (see §5) |
| Failure inv. | `n_battery_dead` | 123 (86..132) |
| | `n_contingency_backup / return` | 104 / 51 |
| Throughput | `n_completed` | 877 (868..914) |
| | `n_unfinished` | 0 (all seeds) |
| | @1h / @3h / @5h | 17.8 / 59.9 / 87.6 % |
| | `sim_end_hours` | 5.12 (4.80..5.19) |
| Congestion | `gridlock` | False (all seeds) |
| | `total_hold_minutes` | 12772 (11989..12917) |
| | `peak_backlog` | 999 (binding) |
| | `mean_wait_for_leg_s` | 6119 (~102 min) |
| | `n_reroutes` | 1063 |
| Utilization | `routes_used/total` | 24/24 |
| | `max_agents_on_a_lane` | 6 (6..7) |

---

## 4. Fix log (chronological journal)

1. **Found baseline regression** — canonical on-disk run gridlocked; cause =
   `NODE_MUTEX_ENABLE=True` (contradicts params' own comment; commit `45545d6`).
2. **Built Phase-0 baseline config** `params/baseline_p0.params` — copy of
   canonical + mutex OFF + `RING_TRAVEL=True` + heavy outputs off +
   `OUTPUT_DIR=output/09_phase0_baseline`. (Regenerate: `python phase0/make_baseline_params.py`.)
3. **Ran 5-seed baseline** — stable, reproducible, drains (n_unfinished=0),
   matches author's documented ~902/98 reference. §3.
4. **Investigated `separation_respected=False` / `min_same_lane_gap=0.0`** (ring only):
   - **Part METRIC BUG (fixed):** the same-lane-gap metric grouped shared ring-lane
     occupants by `res` and diffed raw `s_local`; each ring arc starts `s_local=0`
     at *its own* entry angle, so two agents far apart on the ring falsely read a
     0 m gap. **Fix:** add `seg.s_offset` (matches move-loop spacing at `~1101`).
     File `09_simulate_agents_2d.py`, metric block near line ~1291. After fix the
     real ring-entry min gap = **57.5–78.2 m** (median 65.8), not 0.
   - **Part REAL:** that 58–78 m is still below the `SEPARATION_M=80 m` floor — a
     genuine, mild, reproducible tight-spacing at ring **entry** → this is A7's job.
5. **Corrected a mischaracterization:** `min_closest_approach_m` (metrics.json,
   line ~2863) is **not** a last-sample artifact — it is `stats["min_approach_m"]`
   = the true global min. `0.0` is real (same-level cross-leg agents coincide at a
   junction with node-mutex OFF; also captured by `conflict_samples`). **Not
   changed** — massaging a real safety number would be dishonest. Caveat: it is
   sampled every 20 s, so crossing traffic ~10 s apart can look coincident; this
   affects ring and no-ring equally and is not a code bug.

6. **A7 ring-entry metering — implemented + A/B'd, VERDICT KILL** (behind
   `RING_METER`, default OFF). Root cause of the sub-80 m ring floor: the ring
   global coordinate `s_local + s_offset` is **not wrapped** to the circumference,
   so an agent past the 2·π·r seam sees no leader near angle 0 and closes inside
   the headway. A7 makes ring car-following + ring merges wrap-aware (compare mod
   `ring_circ`) and meters the merge to keep ≥ `RING_METER_GAP_M` (80 m) clear on
   both sides. Result (5-seed median): gap **65.8 → 94.8 m** (G1′ PASS), but
   `n_battery_dead` **123 → 153** (**G3 FAIL** — metering holds → hover drain),
   `total_hold_minutes` +29 %, conflicts +22 %, reroutes +63 %, n_completed −3.4 %.
   **Killed** by the keep/kill rule. Flag stays in, default OFF (baseline
   untouched). Follow-up queued: `phase1-A7b-ring` (wrap car-following only, no
   merge meter). Files: `params/phase1_A7_ring.params`, `ledger/tasks/phase1-A7-ring.sh`,
   `phase1/A7/metrics/`.

7. **A7b isolation — VERDICT KILL (worse than A7).** Split `RING_METER` into
   `RING_WRAP_FOLLOW` (ring car-following) + `RING_MERGE_METER` (metered merge);
   ran wrap-follow alone to test whether the meter was the costly half. It was
   **not**: A7b gap 76.1 m median but **min 65.8 m (G1′ FAIL)**, `n_battery_dead`
   **240** (G3 FAIL, far worse than A7's 153), completed 760, holds 19948.
   Braking for cross-seam leaders around the whole ring holds agents more than the
   merge meter did → hover-drain collapse. **Conclusion: enforcing ≥80 m ring
   circulation spacing is a dead end under this demand — every variant fails a hard
   gate on battery.** The binding constraint is unchanged: `peak_backlog=999`
   (launch backlog, §1). **Pivot to demand-side (A5 tolling / A6 demand-capacity).**
   Ring flags stay in, all default OFF. Files: `params/phase1_A7b_ring.params`,
   `ledger/tasks/phase1-A7b-ring.sh`, `phase1/A7b/metrics/`.

8. **A5 system-optimum tolling — VERDICT NEUTRAL (safe, no benefit).** Behind
   `TOLL_MODE` (default OFF). The capture penalty is congestion cost linear in leg
   occupancy that each agent minimises selfishly (user equilibrium); the SO
   marginal-cost toll internalises the externality by scaling that term
   (SO = UE·(1+β) = 2× for β=1) — one multiply in the `leg_penalty` rebuild.
   Result (5-seed median): **all hard gates PASS** (gridlock False, battery 121 ≤
   123, gap not-worse) but **no ★ goal moves** — holds +3.2%, completed +2 (noise),
   `max_agents_on_a_lane` flat 6→6, conflicts +1.9%. Higher toll (6×) also flat /
   slightly worse battery. **Root cause: `peak_backlog=999` is unchanged —
   throughput is launch-backlog-bound, not routing-bound, so tolling can't help.**
   Not kept; flag stays in, default OFF. Files: `params/phase1_A5_tolling.params`,
   `ledger/tasks/phase1-A5-tolling.sh`, `phase1/A5/metrics/`.
   **Cumulative:** A7/A7b/A5 all leave `peak_backlog=999` untouched → the
   bottleneck is demand/launch scheduling → **A6 (DCB) is the demonstrated lever.**

**Code change made:** exactly one — the same-lane-gap `s_offset` fix (measurement

**Code change made:** exactly one — the same-lane-gap `s_offset` fix (measurement
only; **no change to simulation dynamics**; `s_offset=0` for ordinary legs so their
metric is unchanged). Verified: n_completed / conflict_samples / gridlock identical
before/after; only `min_same_lane_gap_m` changed 0.0 → ~63 m.

---

## 5. Revised acceptance gates

**Hard gates (a violation kills the change):**

- **G1′** `min_same_lane_gap_m ≥ 80 m` (`SEPARATION_M` floor, absolute) **and**
  not worse than baseline (~58 m). *(Replaces the old `separation_respected`,
  whose 250 m reference is too strict — even no-ring only reaches exactly 250.)*
- **G2** `gridlock == false`
- **G3** `n_battery_dead ≤ baseline` (INV-1 preserved)
- **G4** round-trip / return completion share ≥ baseline (INV-2 preserved)

**Soft goals (★ = the two stated objectives):** conflict_samples ↓ · ★
total_hold_minutes ↓≥15% · peak_backlog ↓ · mean_wait ↓≥10% · ★ route-load
evenness ↑ (keep 24/24; a CoV-of-per-leg-load metric is a small TODO) · n_completed
↑ · @5h ↑≥3 pts · n_battery_dead ↓ · n_reroutes ↓.

**Keep/kill rule:** keep iff all hard gates pass, ≥1 ★ goal clears its threshold,
and no soft metric regresses >5% (makespan not worse >10% unless throughput up).

---

## 6. Files for this work (in-repo)

| Path | What | Tracked? |
|------|------|----------|
| `09_simulate_agents_2d.py` | the sim; **contains the metric fix** (§4.4) | tracked — **commit pending, see §8** |
| `params/baseline_p0.params` | Phase-0 baseline config (D1) | **untracked — commit it** |
| `phase0/run_seeds.sh` | run the 5-seed baseline (portable) | untracked — commit it |
| `phase0/aggregate.py` | aggregate metrics → table + summary (portable) | untracked — commit it |
| `phase0/make_baseline_params.py` | regenerate baseline_p0.params | untracked — commit it |
| `phase0/baseline_summary.json` | frozen baseline reference | untracked — commit it |
| `phase0/metrics/` | per-seed raw metrics (regenerable) | (regenerated by run_seeds.sh) |
| `docs/*.docx` | roadmap + strategic plan write-ups | untracked — commit if wanted |
| `progress.md` | this file | untracked — commit it |

Not in git and **not needed on Linux**: the macOS scratchpad (`.../scratchpad/`,
incl. a `python-docx` venv used only to regenerate the `.docx`).

---

## 7. Pending action + next step

- **DONE (pushed):** metric fix (`b063284`) + Phase-0 harness/baseline (`5b082f2`)
  are committed and on `origin/roundabout-corridor-tuning`. The old §7 "pending
  commit" is resolved.
- **DONE — handshake:** linux reproduces the frozen baseline (seed 12345 MATCH);
  cross-machine numbers trusted. Ledger: `ledger/LEDGER.md`.
- **DONE — A7 ring metering (VERDICT: KILL):** implemented behind `RING_METER`
  (default OFF). Lifts ring gap **65.8 → 94.8 m** (G1′ PASS) but **G3 fails**
  (`n_battery_dead` 123 → 153: metering holds → hover drain) and holds +29%. Flag
  kept, default OFF; baseline unchanged. See §4.6.
- **DONE — A7b isolation (VERDICT: KILL, worse than A7):** wrap-follow alone is
  the expensive half and still misses G1′. **Ring-spacing enforcement is a dead
  end here** (see §4.7). All ring flags default OFF.
- **DONE — A5 tolling (VERDICT: NEUTRAL):** safe (all hard gates pass) but no
  benefit — `peak_backlog=999` untouched, so routing tolling can't help (§4.8).
- **Next → `phase2-A6-demand` (promoted).** Three interventions (A7, A7b, A5) now
  all leave `peak_backlog=999` fixed, empirically confirming the binding
  constraint is the **launch backlog**, not ring spacing or route choice.
  Implement demand-capacity balancing: replace the `n_active < max_concurrent`
  launch gate ([:1099](09_simulate_agents_2d.py:1099)) with a per-corridor
  slot/capacity check (doc A6, §2 Phase 2). `A4` (speed) is deprioritised — also a
  tactical lever unlikely to move the launch wall.

---

## 8. GitHub → Linux — will it "just run"? (checklist)

`progress.md` alone transfers fine, but to actually continue **you must also push
the code fix + baseline files** (they are not all committed yet), and **recreate
the env on Linux**. Steps:

```bash
# on THIS machine — commit the fix + the phase-0 artifacts, then push:
git commit -F - <<'MSG'
Step 09: ring same-lane gap metric uses global progress (s_offset)

min_same_lane_gap grouped shared ring-lane occupants by res and diffed raw
s_local; each ring arc starts s_local=0 at its OWN entry angle, so two agents
far apart on the ring falsely read a 0 m gap (flipping separation_respected to
False on ring runs). Add seg.s_offset to share one coordinate (mirrors the
move-loop spacing). s_offset is 0 for ordinary legs -> unchanged. Real ring
gap now ~58-78 m; measurement fix only, no dynamics change.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
MSG

git add params/baseline_p0.params phase0/ progress.md docs/
git commit -m "Phase-0 baseline harness, frozen baseline, progress log"
git push
```

Caveats:
- The conda env `utm` does **not** travel via git — recreate it (§0). Consider
  adding an `environment.yml` so Linux gets identical deps.
- Use the env's `python`; the scripts are path-portable (no hard-coded macOS path).
- `output/09_agent_sim_2d/` shows as modified in git — that is the **old gridlocked
  run from before this work**, not part of it; leave it or `git checkout` it.
- Sim runtime ≈30 s/seed for N=1000 (CPU-bound, single-thread) — comparable on Linux.
