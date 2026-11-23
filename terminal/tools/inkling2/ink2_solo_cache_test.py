#!/usr/bin/env python3
"""Solo path + conversation cache: conversation A, then B (A is parked to host RAM), then A's next turn (restored).  Prints the tokens' hash and the
reused count of each turn, so a pooled and an unpooled engine can be compared:
  ink2_solo_cache_test.py --exe E --model DIR --ids a.ids,b.ids --extra "--kv k8v4 --ctx 131072 --chunk 1024 --tc 1 --spec 1 --conversation-cache-mib 4096 [--kv-pool-tokens N]" """
import argparse, os, sys, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ink2_slots_test import Engine, ids_of
ap = argparse.ArgumentParser()
ap.add_argument("--exe", required=True); ap.add_argument("--model", required=True); ap.add_argument("--ids", required=True); ap.add_argument("--extra", required=True)
a = ap.parse_args()
pa, pb = [ids_of(p) for p in a.ids.split(",")]
e = Engine(a.exe, a.model, 4, a.extra, {"INK2_Q8": "15"})
def gen(ids, n):
    e.send(f"GEN {n} ignore_eos=1 {','.join(map(str, ids))}")
    toks, reused = [], None
    while True:
        l = e.q.get(timeout=900)
        if l is None: sys.exit("engine died")
        if l.startswith("T "): toks.append(int(l[2:]))
        elif l.startswith("DONE"): reused = int(l.split()[8]); return toks, reused
        elif l.startswith("ERR"): sys.exit(l)
out = {}
t1, r1 = gen(pa, 40); print(f"turn A1: {len(pa)} prompt tokens, reused {r1}, tokens hash {hashlib.sha1(str(t1).encode()).hexdigest()[:12]}")
t2, r2 = gen(pb, 40); print(f"turn B1: {len(pb)} prompt tokens, reused {r2}, tokens hash {hashlib.sha1(str(t2).encode()).hexdigest()[:12]}")
turn = pa + t1[:-1] + pb[:30]
t3, r3 = gen(turn, 40); print(f"turn A2: {len(turn)} prompt tokens, reused {r3} (A was parked while B ran), tokens hash {hashlib.sha1(str(t3).encode()).hexdigest()[:12]}")
e.send("QUIT")
