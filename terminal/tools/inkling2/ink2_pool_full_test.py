#!/usr/bin/env python3
"""--kv-pool-tokens: a prompt that does not fit beside the others is refused with `ERR BGEN: ...` (the server then tries another slot / answers 503),
the engine stays up, and the same prompt is accepted once another slot lets go (BSTOP does not free KV; a new BGEN into the slot trims it).
  ink2_pool_full_test.py --exe E --model DIR --ids a.ids,b.ids,c.ids --pool 1000 [--extra "..."]   (three different prompts of ~5000 tokens)
The pool is the tokens asked for plus one granule per array and sequence, so a pool of 1000 tokens holds two 5000-token sequences and not three."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ink2_slots_test import Engine, ids_of
ap = argparse.ArgumentParser()
ap.add_argument("--exe", required=True); ap.add_argument("--model", required=True); ap.add_argument("--ids", required=True)
ap.add_argument("--pool", type=int, default=1000); ap.add_argument("--extra", default="--kv k8v4 --ctx 131072 --chunk 1024 --tc 1 --vram-reserve-mib 800")
a = ap.parse_args()
pa, pb, pc = [ids_of(p) for p in a.ids.split(",")]
e = Engine(a.exe, a.model, 4, a.extra + f" --kv-pool-tokens {a.pool}", {"INK2_Q8": "15"})
def bgen(slot, ids, n=4):
    e.send(f"BGEN {slot} {n} ignore_eos=1 {','.join(map(str, ids))}")
    err = None
    while True:
        l = e.q.get(timeout=600)
        if l is None: return "engine died"
        if l.startswith("ERR"): return l
        if l.startswith("BADM"): return None
        if l.startswith("BT ") or l.startswith("BDONE"): continue
ok = True
def check(name, cond, detail=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name + (" - " + detail if detail else "")); ok = ok and cond
r = bgen(0, pa); check(f"first prompt ({len(pa)} tokens) fits", r is None, str(r))
r = bgen(1, pb); check(f"second prompt ({len(pb)} tokens) fits beside it", r is None, str(r))
r = bgen(2, pc); check(f"third prompt ({len(pc)} tokens) does not fit beside the other two: refused", r is not None and r.startswith("ERR BGEN:"), str(r))
r = bgen(0, pc); check("the same prompt into slot 0 (replacing the first one) is accepted: the engine is still up", r is None, str(r))
r = bgen(3, pa[:300]); check("a short prompt in a new slot is refused too: a sequence needs at least 32 granules (64 MiB) and 23 are free", r is not None and r.startswith("ERR BGEN:"), str(r))
r = bgen(1, pa[:300]); check("a short prompt replacing slot 1's long one is accepted (its granules go back to the pool)", r is None, str(r))
r = bgen(3, pa[:300]); check("and now a short prompt in the new slot fits too", r is None, str(r))
e.send("QUIT")
print("ALL PASSED" if ok else "FAILED"); sys.exit(0 if ok else 1)
