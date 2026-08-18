#!/usr/bin/env python3
"""Standalone one-shift relay driver (for cron/launchd, headless).

Interactive alternative: the `/relay-resume` Claude Code skill, which does the same
handoff but with Claude interpreting results and implementing not-yet-built tasks.

One shift = pull -> (if baton held by the OTHER host: stand by) -> claim the next
`todo` task -> run ledger/tasks/<id>.sh -> journal the result -> release baton ->
commit -> optionally push. Mutual exclusion comes from git push atomicity: a losing
claimant's push is rejected, and it backs off.

Usage:
    python ledger/relay.py            # local only (no push) -- safe dry-ish run
    python ledger/relay.py --push     # commit AND push (real relay; needs auth)
Env: RELAY_BRANCH overrides the branch (default: value in STATE.json).
"""
import json, subprocess, platform, datetime, sys, os, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
LDIR = ROOT / "ledger"
STATE = LDIR / "STATE.json"
JOURNAL = LDIR / "LEDGER.md"
HOST = {"Darwin": "mac", "Linux": "linux"}.get(platform.system(), platform.system().lower())
PUSH = "--push" in sys.argv


def sh(*args, check=False):
    return subprocess.run(list(args), cwd=ROOT, text=True, capture_output=True, check=check)


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def load():
    return json.loads(STATE.read_text())


def save(s):
    s["updated"] = now()
    STATE.write_text(json.dumps(s, indent=2) + "\n")


def main():
    s = load()
    branch = os.environ.get("RELAY_BRANCH", s.get("branch", "main"))
    sh("git", "pull", "--ff-only", "origin", branch)
    s = load()

    if s.get("baton") not in ("idle", HOST):
        print(f"[relay] baton held by {s['baton']} — {HOST} stands by")
        return 0

    todo = next((t for t in s["queue"] if t["status"] == "todo"), None)
    if not todo:
        print("[relay] no 'todo' task (queue empty or all blocked) — nothing to do")
        return 0

    tid = todo["id"]
    task = LDIR / "tasks" / f"{tid}.sh"
    if not task.exists():
        print(f"[relay] task script missing: {task} — mark it 'blocked' or add the script")
        return 1

    # ---- claim the baton (atomic via push-reject) ----
    s["baton"] = HOST
    s["last_host"] = HOST
    todo["status"] = "running"
    save(s)
    # curated add only (never blanket `-A`: the pre-existing dirty output/09_agent_sim_2d
    # gridlocked run and volatile per-run outputs must not be swept in)
    sh("git", "add", "ledger/", "phase0/")
    sh("git", "commit", "-m", f"relay: {HOST} claims {tid}")
    if PUSH:
        if sh("git", "push", "origin", branch).returncode != 0:
            print("[relay] push rejected on claim — lost the race, backing off")
            sh("git", "reset", "--hard", f"origin/{branch}")
            return 0

    # ---- run the task ----
    print(f"[relay] {HOST} running {tid} ...")
    r = subprocess.run(["bash", str(task)], cwd=ROOT, text=True, capture_output=True)
    ok = r.returncode == 0

    with JOURNAL.open("a") as f:
        f.write(f"\n## {now()} · host={HOST} · task={tid} · {'OK' if ok else 'FAIL'}\n")
        tail = (r.stdout or "")[-1500:] + (("\n[stderr]\n" + r.stderr[-500:]) if r.stderr.strip() else "")
        f.write("```\n" + tail.strip() + "\n```\n")

    # ---- release the baton, advance the queue ----
    s = load()
    for t in s["queue"]:
        if t["id"] == tid:
            t["status"] = "done" if ok else "todo"
    s["baton"] = "idle"
    s["next_task"] = next((t["id"] for t in s["queue"] if t["status"] == "todo"), None)
    save(s)
    sh("git", "add", "ledger/", "phase0/")
    sh("git", "commit", "-m", f"relay: {HOST} {'done' if ok else 'FAILED'} {tid}")
    if PUSH:
        sh("git", "push", "origin", branch)
    print(f"[relay] {tid} {'done' if ok else 'FAILED'}; next_task={s['next_task']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
