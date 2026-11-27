#!/usr/bin/env python3
"""Warp stall reasons (% of warp-active cycles) per launch from an ncu raw CSV made with ncu-stall.sh."""
import csv, io, sys
lines = [l for l in open(sys.argv[1], errors="ignore") if not l.startswith("==")]
rows = list(csv.reader(io.StringIO("".join(lines)))); h = rows[0]; ix = {c: i for i, c in enumerate(h)}
cols = [c for c in h if c.startswith("smsp__warp_issue_stalled_")]
for r in rows[2:]:
    if len(r) < len(h) - 2: continue
    name = r[ix["Kernel Name"]].replace("void <unnamed>::", "")[:34]
    vals = sorted(((float(r[ix[c]].replace(",", "") or 0), c.replace("smsp__warp_issue_stalled_", "").replace("_per_warp_active.pct", "")) for c in cols), reverse=True)
    print(f"{name:36s} grid {r[ix['launch__grid_size']] if 'launch__grid_size' in ix else ''}  " + "  ".join(f"{n} {v:.1f}%" for v, n in vals[:6]))
