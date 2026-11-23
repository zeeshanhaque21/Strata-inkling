#!/usr/bin/env python3
"""--kv-pool-tokens: eviction and refusal.  A pool of 1000 tokens (plus the 32-granule minimum per sequence) holds two 5000-token sequences, not three.
  ink2_pool_full_test.py --exe E --model DIR --ids a.ids,b.ids,c.ids [--pool 1000] [--extra "..."]   (three different prompts of ~5000 tokens)
Checks: a third prompt evicts the least recently used idle conversation (the engine says so, the evicted conversation is read again from scratch,
the other one is still reused); when every slot is decoding nothing can be evicted and the prompt is refused with ERR BGEN:, the engine stays up."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ink2_slots_test import Engine, ids_of
ap = argparse.ArgumentParser()
ap.add_argument("--exe", required=True); ap.add_argument("--model", required=True); ap.add_argument("--ids", required=True)
ap.add_argument("--pool", type=int, default=1000); ap.add_argument("--extra", default="--kv k8v4 --ctx 131072 --chunk 1024 --tc 1 --vram-reserve-mib 800")
a = ap.parse_args()
pa, pb, pc = [ids_of(p) for p in a.ids.split(",")]
e = Engine(a.exe, a.model, 4, a.extra + f" --kv-pool-tokens {a.pool}", {"INK2_Q8": "15"})
def bgen(slot, ids, n=1):   # n=1: the slot's history is the prompt itself (the one token is never fed), so a longer prompt continues it
    """-> (error line or None, reused tokens, continues in the slot)"""
    e.send(f"BGEN {slot} {n} ignore_eos=1 {','.join(map(str, ids))}")
    reused, cont = None, False
    while True:
        l = e.q.get(timeout=600)
        if l is None: return "engine died", None, False
        if l.startswith("ERR"): return l, None, False
        if l.startswith("DONE"): reused = int(l.split()[8])
        if l.startswith("BADM"): return None, reused, l.split()[2] == "1"
def drain(slots):
    left = set(slots)
    while left:
        l = e.q.get(timeout=600)
        if l and l.startswith("BDONE"): left.discard(int(l.split()[1]))
ok = True
def check(name, cond, detail=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name + (" - " + detail if detail else "")); ok = ok and cond
r, ru, c = bgen(0, pa); check("A in slot 0", r is None, str(r)); drain([0]) if c else None
r, ru, c = bgen(1, pb); check("B in slot 1", r is None, str(r)); drain([1]) if c else None
r, ru, c = bgen(2, pc); check("C in slot 2: the pool is full, so the least recently used idle conversation (A, slot 0) is evicted and C is accepted", r is None, str(r)); drain([2]) if c else None
r, ru, c = bgen(1, pb + [11, 12]); check(f"B's next turn in slot 1 reuses its state ({ru} reused): B was not evicted", r is None and ru is not None and ru >= len(pb) - 2, f"{r} {ru}"); drain([1]) if c else None
r, ru, c = bgen(0, pa + [11, 12]); check(f"A's next turn in slot 0 reads from scratch ({ru} reused): A was evicted (and C, the least recently used now, goes)", r is None and ru == 0, f"{r} {ru}"); drain([0]) if c else None
# every slot busy: slots 0 and 1 decode 300 tokens; a prompt for slot 2 has nothing to evict
r, ru, c = bgen(0, pa, n=300); check("slot 0 starts a long answer", r is None and c, str(r))
r, ru, c = bgen(1, pb, n=300); check("slot 1 starts a long answer", r is None and c, str(r))
r, ru, c = bgen(2, pa[1000:5000]); check("a different prompt with every other slot decoding is refused: nothing is idle to evict", r is not None and r.startswith("ERR BGEN:"), str(r))
drain([0, 1])
r, ru, c = bgen(2, pc); check("once they are done the same prompt is accepted", r is None, str(r)); drain([2]) if c else None
e.send("QUIT")
print("ALL PASSED" if ok else "FAILED"); sys.exit(0 if ok else 1)
