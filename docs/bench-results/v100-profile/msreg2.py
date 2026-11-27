import sqlite3, sys, collections
db = sqlite3.connect(sys.argv[1]); c = db.cursor()
rows = c.execute("select k.start, k.end, s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id = k.shortName order by k.start").fetchall()
pk = [r for r in rows if r[2] == "kbm_attn_partial"]
groups = [[pk[0]]]
for r in pk[1:]:
    if r[0] - groups[-1][-1][1] > 3_000_000: groups.append([r])   # > 3 ms between attention kernels of consecutive layers = a phase break
    else: groups[-1].append(r)
for g in groups:
    if len(g) > 50: print(f"group start {(g[0][0]-rows[0][0])/1e9:8.3f} s  span {(g[-1][1]-g[0][0])/1e6:8.1f} ms  attn_partial calls {len(g)}")
