#!/usr/bin/env python3
"""The kernel sequence of the decode region of an ncu raw CSV, with durations: seq.py file.csv [first_index] [count]"""
import csv, re, sys
rows = list(csv.reader(open(sys.argv[1], errors="ignore")))
hi = next(i for i, r in enumerate(rows) if "Kernel Name" in r); h = rows[hi]; ix = {c: i for i, c in enumerate(h)}
f = lambda x: float(x.replace(",", "") or 0)
data = [r for r in rows[hi + 2:] if len(r) >= len(h)]
nm = [re.sub(r"^(void )?<unnamed>::", "", r[ix["Kernel Name"]]).split("(")[0] for r in data]
last = max(i for i, n in enumerate(nm) if n.startswith("kb"))
dec, names = data[last + 1:], nm[last + 1:]
seq = [(n, f(r[ix["gpu__time_duration.sum"]]) / 1e3, r[ix["launch__grid_size"]], r[ix["launch__block_size"]]) for n, r in zip(names, dec)]
start = int(sys.argv[2]) if len(sys.argv) > 2 else next(i for i, s in enumerate(seq) if s[0] == "k_rms")
cnt = int(sys.argv[3]) if len(sys.argv) > 3 else 60
print("kernels in region:", len(seq), "| total us", round(sum(s[1] for s in seq)))
for i, (n, us, g, b) in enumerate(seq[start:start + cnt]):
    print(f"{start + i:4d} {us:7.1f} us  {g:>6}x{b:<4} {n[:50]}")
