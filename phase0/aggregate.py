# -*- coding: utf-8 -*-
"""Aggregate Phase-0 baseline metrics across seeds into the 5 metric families.
Prints a table (median + min..max) and writes baseline_summary.json.

Portable: reads metrics from ./metrics next to this script. Run from anywhere:
    python phase0/aggregate.py
"""
import json, statistics, pathlib, sys

MDIR = pathlib.Path(__file__).resolve().parent / "metrics"
files = sorted(MDIR.glob("metrics_seed*.json"))
runs = []
for f in files:
    d = json.loads(f.read_text())
    d["_seed"] = f.stem.replace("metrics_seed", "")
    runs.append(d)

if not runs:
    print(f"no metrics files in {MDIR} -- run `bash phase0/run_seeds.sh` first")
    sys.exit(1)

def dl_pct(d, h):
    # deadline_table stores pct already as a PERCENTAGE (e.g. 18.2), not a fraction
    for row in d.get("deadline_table", []):
        if abs(row.get("within_h", -1) - h) < 1e-6:
            p = row.get("pct", None)
            if p is None:
                return None
            return p * 100.0 if p <= 1.0 else p   # tolerate either encoding
    return None

for d in runs:
    d["@1h_%"] = dl_pct(d, 1.0)
    d["@3h_%"] = dl_pct(d, 3.0)
    d["@5h_%"] = dl_pct(d, 5.0)
    d["routes_used/total"] = f'{d.get("routes_used")}/{d.get("routes_total")}'

FAMILIES = [
    ("— Safety —", None, None),
    ("separation_respected", "separation_respected", "bool"),
    ("min_same_lane_gap_m (fixed)", "min_same_lane_gap_m", "f1"),
    ("min_closest_approach_m [real]", "min_closest_approach_m", "f1"),
    ("conflict_samples", "conflict_samples", "i"),
    ("conflict_frames", "conflict_frames", "i"),
    ("peak_simultaneous_conflicts", "peak_simultaneous_conflicts", "i"),
    ("— Failure/return invariants —", None, None),
    ("n_battery_dead", "n_battery_dead", "i"),
    ("missions_over_battery", "missions_over_battery", "i"),
    ("n_contingency_backup", "n_contingency_backup", "i"),
    ("n_contingency_return", "n_contingency_return", "i"),
    ("n_round_trips", "n_round_trips", "i"),
    ("— Throughput —", None, None),
    ("n_completed", "n_completed", "i"),
    ("n_unfinished", "n_unfinished", "i"),
    ("@1h_%", "@1h_%", "f1"),
    ("@3h_%", "@3h_%", "f1"),
    ("@5h_%", "@5h_%", "f1"),
    ("sim_end_hours", "sim_end_hours", "f2"),
    ("mean_flight_time_s", "mean_flight_time_s", "f0"),
    ("— Congestion —", None, None),
    ("gridlock", "gridlock", "bool"),
    ("total_hold_minutes", "total_hold_minutes", "f0"),
    ("peak_backlog", "peak_backlog", "i"),
    ("mean_wait_for_leg_s", "mean_wait_for_leg_s", "f0"),
    ("max_wait_for_leg_s", "max_wait_for_leg_s", "f0"),
    ("n_reroutes", "n_reroutes", "i"),
    ("— Route utilization —", None, None),
    ("routes_used/total", "routes_used/total", "str"),
    ("max_agents_on_a_lane", "max_agents_on_a_lane", "i"),
    ("max_simultaneous_agents", "max_simultaneous_agents", "i"),
]

def fmt(v, kind):
    if v is None: return "-"
    if kind in ("bool", "str"): return str(v)
    if kind == "i": return f"{int(round(v))}"
    if kind == "f0": return f"{v:.0f}"
    if kind == "f1": return f"{v:.1f}"
    if kind == "f2": return f"{v:.2f}"
    return str(v)

seeds = [d["_seed"] for d in runs]
hdr = f'{"metric":<32} ' + " ".join(f"{s:>9}" for s in seeds) + f'  {"median":>10} {"min..max":>16}'
print(hdr); print("-" * len(hdr))

summary = {"seeds": seeds, "n_runs": len(runs), "config": {
    "N_AGENTS": runs[0].get("n_deliveries"), "node_mutex": runs[0].get("node_mutex_enabled"),
    "flow_mode": runs[0].get("flow_mode"), "shift_hours": runs[0].get("shift_hours"),
}, "metrics": {}}

for label, field, kind in FAMILIES:
    if field is None:
        print(f"\n{label}"); continue
    vals = [d.get(field) for d in runs]
    per_seed = "  ".join(f"{fmt(v, kind):>7}" for v in vals)
    numeric = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if numeric and kind not in ("bool", "str"):
        med = statistics.median(numeric)
        med_s = fmt(med, kind); rng = f"{fmt(min(numeric),kind)}..{fmt(max(numeric),kind)}"
        summary["metrics"][field] = {"per_seed": dict(zip(seeds, vals)),
                                     "median": med, "min": min(numeric), "max": max(numeric)}
    else:
        alleq = len(set(map(str, vals))) == 1
        med_s = str(vals[0]) if alleq else "MIXED!"
        rng = "(all equal)" if alleq else str(sorted(set(map(str, vals))))
        summary["metrics"][field] = {"per_seed": dict(zip(seeds, vals)), "all_equal": alleq}
    print(f'{label:<32} {per_seed}  {med_s:>10} {rng:>16}')

out = pathlib.Path(__file__).resolve().parent / "baseline_summary.json"
out.write_text(json.dumps(summary, indent=2, default=str))
print(f"\nwrote {out}")
