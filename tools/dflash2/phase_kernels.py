#!/usr/bin/env python3
"""Per-kernel time of one phase of the speculative cycle from an nsys sqlite of an INK2_SPEC_BENCH_ONLY=<phase> run.
usage: phase_kernels.py x.sqlite iterations [top]   (iterations = timed loop count + 3 warmups)"""
import sqlite3, sys, collections
db = sqlite3.connect(sys.argv[1]); it = float(sys.argv[2]); top = int(sys.argv[3]) if len(sys.argv) > 3 else 22
rows = db.execute("select k.start, k.end, s.value, k.gridX, k.gridY, k.blockX from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id = k.shortName order by k.start").fetchall()
# the timed loops are the long tail of the run: keep the last 80 % of kernels after the model load / prefill (find the largest gap)
gaps = [(rows[i + 1][0] - rows[i][1], i) for i in range(len(rows) - 1)]
cut = max(gaps)[1] + 1
rows = rows[cut:]
agg = collections.defaultdict(lambda: [0, 0, None])
for r in rows:
    a = agg[r[2][:44]]; a[0] += r[1] - r[0]; a[1] += 1; a[2] = (r[3], r[4], r[5])
tot = sum(a[0] for a in agg.values())
print(f"{len(rows)} kernels after the last gap, {tot / 1e6 / it:.2f} ms of kernel time per iteration ({it:.0f} iterations)")
for n, a in sorted(agg.items(), key=lambda kv: -kv[1][0])[:top]:
    print(f"  {100 * a[0] / tot:5.1f}%  {a[0] / 1e6 / it:7.3f} ms/it  {a[1] / it:6.1f} calls/it  {a[0] / a[1] / 1e3:8.1f} us  grid {a[2][0]}x{a[2][1]} blk {a[2][2]}  {n}")
