#!/usr/bin/env python3
"""Aggregate an ncu raw CSV of decode kernels: per kernel name, calls, total / mean time, DRAM read, DRAM utilization, warps active, launch shape."""
import csv, re, sys, collections
rows = list(csv.reader(open(sys.argv[1], errors="ignore")))
hi = next(i for i, r in enumerate(rows) if "Kernel Name" in r); h = rows[hi]; units = rows[hi + 1]; data = rows[hi + 2:]
ix = {c: i for i, c in enumerate(h)}
def col(r, name): return r[ix[name]]
tu = units[ix["gpu__time_duration.sum"]]; scale = {"ns": 1e-3, "nsecond": 1e-3, "us": 1.0, "usecond": 1.0, "ms": 1e3, "msecond": 1e3, "s": 1e6, "second": 1e6}.get(tu, 1.0)
bu = units[ix["dram__bytes_read.sum"]]; bscale = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "KB": 1e3, "MB": 1e6, "GB": 1e9}.get(bu, 1)
def f(x):
    try: return float(x.replace(",", ""))
    except Exception: return 0.0
def short(n):
    n = re.sub(r"^(void )?<unnamed>::", "", n); n = re.sub(r"\(.*", "", n); return n[:58]
only_decode = len(sys.argv) < 3 or sys.argv[2] != "all"
agg = collections.OrderedDict(); order = []
if only_decode:
    nm = [short(col(r, "Kernel Name")) for r in data if len(r) >= len(h)]
    last = max((i for i, n in enumerate(nm) if n.startswith(("kb_", "kbm_"))), default=-1)
    data = [r for r in data if len(r) >= len(h)][last + 1:]
for r in data:
    if len(r) < len(h): continue
    name = short(col(r, "Kernel Name"))
    if only_decode and (name.startswith(("kb_", "kbm_"))): continue
    a = agg.setdefault(name, dict(n=0, us=0.0, byt=0.0, dram=0.0, warps=0.0, grid=set(), block=set()))
    a["n"] += 1; a["us"] += f(col(r, "gpu__time_duration.sum")) * scale; a["byt"] += f(col(r, "dram__bytes_read.sum")) * bscale
    a["dram"] += f(col(r, "dram__throughput.avg.pct_of_peak_sustained_elapsed")); a["warps"] += f(col(r, "sm__warps_active.avg.pct_of_peak_sustained_active"))
    a["grid"].add(int(f(col(r, "launch__grid_size")))); a["block"].add(int(f(col(r, "launch__block_size"))))
tot = sum(a["us"] for a in agg.values()); totb = sum(a["byt"] for a in agg.values())
print(f"{len(data)} kernels captured; {sum(a['n'] for a in agg.values())} counted, {tot / 1e3:.2f} ms GPU time, {totb / 1e6:.0f} MB read from DRAM (ncu serialised, one stream)")
print(f"{'%time':>6} {'ms':>8} {'calls':>6} {'avg us':>8} {'MB read':>9} {'GB/s':>7} {'dram%':>6} {'warps%':>7}  grid x block   kernel")
for name, a in sorted(agg.items(), key=lambda kv: -kv[1]["us"])[:int(sys.argv[3]) if len(sys.argv) > 3 else 22]:
    gbs = a["byt"] / (a["us"] * 1e-6) / 1e9 if a["us"] else 0
    print(f"{100 * a['us'] / tot:6.1f} {a['us'] / 1e3:8.3f} {a['n']:6d} {a['us'] / a['n']:8.1f} {a['byt'] / 1e6:9.1f} {gbs:7.0f} {a['dram'] / a['n']:6.1f} {a['warps'] / a['n']:7.1f}  {','.join(str(g) for g in sorted(a['grid'])[:2])}x{','.join(str(b) for b in sorted(a['block'])[:2])}   {name}")
