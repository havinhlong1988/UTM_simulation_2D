# ledger/ — Mac↔Linux relay for continuous, interleaved work

One person, two machines, **taking turns** (not running in parallel): **Linux during
working hours, Mac outside**. GitHub is the relay hub; this `ledger/` is the shared,
append-only "sổ cái". Because Claude Code's local memory does **not** sync across
machines, everything needed to continue lives here, in git.

## Files

| File | Role |
|------|------|
| `STATE.json` | baton (`idle`/`mac`/`linux`) + task queue + `next_task`. Machine-readable. |
| `LEDGER.md` | append-only journal — one block per shift. The audit chain. |
| `tasks/<id>.sh` | a runnable task (portable; uses `python` from the active env). |
| `results/` | per-run outputs committed so the other machine sees them (created on demand). |
| `relay.py` | standalone one-shift driver for cron/launchd (headless automation). |
| `../.claude/skills/relay-resume/` | the interactive ritual a Claude session runs (`/relay-resume`). |

Relationship to other docs: `progress.md` = the static handoff/onboarding doc
(read once to understand the project); `LEDGER.md` = the live running journal
(append every shift). `phase0/baseline_summary.json` = the frozen baseline every
A/B compares against.

## Two ways to take a shift

**Interactive (recommended while designing/deciding):** on the machine you're at,
`git pull`, then in Claude Code type `/relay-resume`. Claude reads the ledger, tells
you where things stand, does the next task (implementing it if needed), journals,
and hands off (asks before pushing).

**Automated (unattended batch, e.g. Mac overnight):** a timer runs
`python ledger/relay.py --push`. It runs the next ready `tasks/<id>.sh`, journals,
and pushes. Only works on tasks that already have a script (status `todo`);
`blocked` tasks wait for an interactive session to implement them.

## Scheduling the windows (this is the "interleaved" part)

The baton prevents double-work; the **schedule windows** make the two machines
alternate. macOS also has `cron`, so cron works on both:

```cron
# --- Linux (in working hours: weekdays 08:00–18:00, every 20 min) ---
*/20 8-18 * * 1-5  cd /path/to/repo && /path/to/envs/utm/bin/python ledger/relay.py --push >> ledger/relay.log 2>&1

# --- Mac (outside hours: evenings/nights + weekends, every 20 min) ---
*/20 19-23,0-7 * * 1-5  cd /path/to/repo && /path/to/envs/utm/bin/python ledger/relay.py --push >> ledger/relay.log 2>&1
*/20 * * * 6,0          cd /path/to/repo && /path/to/envs/utm/bin/python ledger/relay.py --push >> ledger/relay.log 2>&1
```

macOS launchd alternative (if you prefer): a `~/Library/LaunchAgents/utm.relay.plist`
with `ProgramArguments` = `/bin/zsh -lc "cd /repo && /envs/utm/bin/python ledger/relay.py --push"`
and `StartInterval` 1200. Enforce the evening window via `StartCalendarInterval`
entries, or just let the baton + the Linux daytime cron keep them from overlapping.

Put `ledger/relay.log` in `.gitignore` (it's local noise).

## Reproducibility (so both machines agree)

- Fixed seeds (`12345, 54321, 11111, 22222, 33333`) + a pinned env → identical runs.
  Add an `environment.yml` so Linux and Mac install the same numpy/pandas.
- The first task, **`handshake-baseline`**, re-runs seed 12345 and compares to the
  frozen reference; if a machine disagrees, it fails loudly — fix env drift before
  trusting any cross-machine number.

## First-time bootstrap (once, on mac, needs push auth)

1. Ensure the metric-fix commit landed (`progress.md` §8).
2. `git add params/baseline_p0.params phase0/ ledger/ .claude/skills/ progress.md docs/`
   → commit → `git push`. (Add `__pycache__/`, `*.pyc`, `ledger/relay.log` to
   `.gitignore` first; do **not** commit `.claude/settings.local.json`.)
3. On Linux: `git pull`, build the `utm` env (`progress.md` §0), then either
   `/relay-resume` in Claude Code or `python ledger/relay.py` to run `handshake-baseline`.
