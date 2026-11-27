import sqlite3, sys, collections
db = sqlite3.connect(sys.argv[1]); c = db.cursor()
t0 = float(sys.argv[2]); t1 = float(sys.argv[3])
rows = c.execute("select k.start, k.end, s.value, k.gridX, k.gridY, k.blockX, k.registersPerThread from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id = k.shortName order by k.start").fetchall()
base = rows[0][0]
reg = [r for r in rows if t0 <= (r[0]-base)/1e9 <= t1]
span = reg[-1][1]-reg[0][0]; busy = sum(r[1]-r[0] for r in reg)
print(f"window {len(reg)} kernels, span {span/1e6:.1f} ms, busy {100*busy/span:.1f}%  (63 batched steps of 4 sequences: {span/63e6:.2f} ms/step)")
agg = collections.defaultdict(lambda: [0,0,None])
for r in reg:
    k = agg[r[2]]; k[0] += r[1]-r[0]; k[1] += 1; k[2] = (r[3], r[4], r[5], r[6])
for n, v in sorted(agg.items(), key=lambda kv: -kv[1][0])[:26]:
    print(f"{100*v[0]/span:5.1f}% {v[0]/63e6:7.2f} ms/step {v[1]/63:6.1f} calls/step {v[0]/v[1]/1e3:7.1f} us  grid {v[2][:3]} regs {v[2][3]}  {n[:48]}")
