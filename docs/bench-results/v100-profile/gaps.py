import sqlite3, sys
db = sqlite3.connect(sys.argv[1]); c = db.cursor()
rows = c.execute("select k.start, k.end, s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id = k.shortName order by k.start").fetchall()
last = max(i for i, r in enumerate(rows) if r[2].startswith("kb"))
dec = rows[last + 1:]
span = dec[-1][1] - dec[0][0]; busy = sum(r[1] - r[0] for r in dec)
gaps = [dec[i + 1][0] - dec[i][1] for i in range(len(dec) - 1)]
print(f"decode kernels {len(dec)}, span {span/1e6:.1f} ms, kernel time {busy/1e6:.1f} ms ({100*busy/span:.1f}%), idle between kernels {sum(gaps)/1e6:.1f} ms ({100*sum(gaps)/span:.1f}%), median gap {sorted(gaps)[len(gaps)//2]} ns, mean {sum(gaps)/len(gaps):.0f} ns")
