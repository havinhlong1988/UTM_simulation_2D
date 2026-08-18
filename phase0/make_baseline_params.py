# -*- coding: utf-8 -*-
"""Regenerate params/baseline_p0.params from the canonical params by applying the
D1 baseline overrides. Does NOT touch params/simulate_agents_2d.params.

Run from the repo root:  python phase0/make_baseline_params.py
(You normally do NOT need this -- params/baseline_p0.params is committed. Use this
only to regenerate it after the canonical params change.)
"""
import re, pathlib

SRC = pathlib.Path("params/simulate_agents_2d.params")
DST = pathlib.Path("params/baseline_p0.params")

# D1 baseline: mutex OFF (avoid the gridlock regression), ring one-way ON,
# heavy per-run artefacts off (we only need metrics.json), separate output dir.
OVERRIDES = {
    "NODE_MUTEX_ENABLE": "False",
    "OUTPUT_DIR":        '"output/09_phase0_baseline"',
    "MAKE_HTML":               "False",
    "MAKE_ANIMATION":          "False",
    "MAKE_PYVISTA":            "False",
    "MAKE_AGENT_ROUTE_HTML":   "False",
    "WRITE_TRAJECTORY":        "False",
}
APPEND = {"RING_TRAVEL": "True"}   # absent in canonical -> defaults False; needed for INV-3

text = SRC.read_text()
seen, out = set(), []
for ln in text.splitlines():
    m = re.match(r"^(\s*)([A-Z_0-9]+)(\s*)=(\s*)(.*)$", ln)
    if m and m.group(2) in OVERRIDES:
        key = m.group(2); seen.add(key)
        rhs = m.group(5); comment = ""
        if "#" in rhs:
            _, comment = rhs.split("#", 1); comment = "  #" + comment
        out.append(f"{key:<15} = {OVERRIDES[key]}{comment}")
    else:
        out.append(ln)

extra = []
for key, val in APPEND.items():
    if not re.search(rf"^\s*{key}\s*=", text, re.M):
        extra.append(f"{key:<15} = {val}")
for key, val in OVERRIDES.items():
    if key not in seen:
        extra.append(f"{key:<15} = {val}")
if extra:
    out += ["", "# ---- Phase-0 baseline overrides (D1): mutex OFF, ring one-way ON ----", *extra]

DST.write_text("\n".join(out) + "\n")
print("wrote", DST)
