---
name: relay-resume
description: Resume the Mac<->Linux ledger relay for the UTM step-09 work. Pull the repo, read ledger/STATE.json + the tail of ledger/LEDGER.md, report where the work stands, run (or implement then run) the next queued task, then hand off (journal + release baton + push). Use at the start of a shift on either machine, or when the user types /relay-resume.
---

# relay-resume — cross-machine shift handoff

The Mac and the Linux box take turns on this repo (Linux in working hours, Mac
outside). Claude Code's **local memory does not sync between them** — the git repo
is the only shared brain. `ledger/LEDGER.md` (journal) + `ledger/STATE.json`
(baton + task queue) are that brain. Follow these steps exactly.

## 1. Sync and orient
1. Identify this host: run `uname -s` → `Darwin`=**mac**, `Linux`=**linux**.
2. `git pull --ff-only origin <branch>` (branch is in `STATE.json`, default `roundabout-corridor-tuning`).
3. Read `ledger/STATE.json` and the last ~40 lines of `ledger/LEDGER.md`.
4. Tell the user, in 3–4 lines: current phase, `next_task`, `baton` holder,
   `last_host`, and the newest journal entry's outcome.

## 2. Baton check (guarantees no double-work)
- If `baton` is this host or `idle` → you may proceed.
- If `baton` is the **other** host → the other machine (or its cron) may be mid-shift.
  Warn the user and ask before proceeding; if they confirm it is stale, continue.

## 3. Do the next task
Take the first queue item with status `todo`.
- If `ledger/tasks/<id>.sh` **exists**: claim the baton (set `baton`=this host,
  the item's status=`running`, `last_host`, `updated`; commit `relay: <host> claims <id>`),
  run the script, and read its output.
- If the task is a not-yet-built improvement (status `blocked`, e.g.
  `phase1-A5-tolling`): this is real work — implement it per `progress.md` §2 and
  `docs/coordination_algorithm_improvements.docx` (one param flag, default off),
  run the A/B against the frozen baseline in `phase0/baseline_summary.json`, and
  judge it with gates G1′–G4 (`progress.md` §5). Add a `ledger/tasks/<id>.sh` that
  reproduces the run so the relay can re-run it, and flip the item to `todo`→`done`.

Always run the sim with the **utm** env's `python` and the baseline params:
`python 09_simulate_agents_2d.py --param-file params/baseline_p0.params --seed N --no-animation --no-html --no-pyvista`.

## 4. Hand off (end of shift)
1. Append one block to `ledger/LEDGER.md`:
   `## <ISO time> · host=<host> · task=<id> · <OK|FAIL|NOTE>` + key numbers + `NEXT: <next_task>`.
2. Update `ledger/STATE.json`: task status `done` (or back to `todo` if it failed),
   `baton`=`idle`, set `next_task` to the next `todo`, refresh `updated`.
3. Commit everything. **Ask the user before `git push`** (push is an outward action);
   on their OK, `git push origin <branch>`. Pushing is what lets the other machine
   pick up the baton next shift.

## Notes
- Never edit past `LEDGER.md` blocks — it is append-only (the audit chain).
- The frozen baseline is `phase0/baseline_summary.json`; never overwrite it without
  re-running the 5-seed sweep and saying so in the journal.
- Unattended runs use `ledger/relay.py` (cron/launchd); it follows the same
  STATE/LEDGER contract, so interactive and automated shifts interleave cleanly.
