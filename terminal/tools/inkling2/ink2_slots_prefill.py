#!/usr/bin/env python3
"""Where does the time go when several long prompts arrive together?  Drives strata-ink2 --serve --slots N directly.

  ink2_slots_prefill.py --exe E --model DIR --ids long.ids --scenario solo|slot|four [--prompt 10000] [--n 200] [--extra "..."]
solo: one GEN of a prompt (max_new 1).  slot: one BGEN into slot 0 (max_new 1).
four: BGEN of four different prompts back to back (what the server does), each continuing for --n tokens; prints per-admission times
and the wall clock until every slot is done.  INK2_INTERLEAVE=0|1 in the environment switches the decode-between-chunks share."""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ink2_slots_test import Engine, ids_of, collect

ap = argparse.ArgumentParser()
ap.add_argument("--exe", required=True); ap.add_argument("--model", required=True); ap.add_argument("--ids", required=True)
ap.add_argument("--scenario", required=True); ap.add_argument("--prompt", type=int, default=10000); ap.add_argument("--n", type=int, default=200)
ap.add_argument("--offset", type=int, default=0); ap.add_argument("--slots", type=int, default=4)
ap.add_argument("--extra", default="--kv k8v4 --ctx 131072 --chunk 1024 --tc 1 --vram-reserve-mib 800 --spec 1 --conversation-cache-mib 4096")
a = ap.parse_args()
allids = ids_of(a.ids)
def prompt(i): return allids[a.offset + i * a.prompt: a.offset + (i + 1) * a.prompt]
e = Engine(a.exe, a.model, a.slots, a.extra, {"INK2_Q8": "15", "INK2_INTERLEAVE": os.environ.get("INK2_INTERLEAVE", "1")})
ign = "ignore_eos=1"
import subprocess
dmon = None
if os.environ.get("DMON_GPU"):   # PCIe traffic of the card while the scenario runs (after the load): nvidia-smi dmon, MB/s per second
    dmon = subprocess.Popen(["nvidia-smi", "dmon", "-i", os.environ["DMON_GPU"], "-s", "t", "-d", "1"], stdout=open("dmon.txt", "w"))
    time.sleep(2.5)

def read_until(prefix_done, out=None):
    """Reads lines until one starting with prefix_done; returns (lines with time)."""
    got = []
    while True:
        l = e.q.get(timeout=900)
        if l is None: sys.exit("engine died")
        got.append((time.time(), l))
        if out is not None and l.startswith(("BT ", "BDONE ")): collect(out, l)
        if l.startswith("ERR"): sys.exit(l)
        if l.startswith(prefix_done): return got

if a.scenario in ("solo", "slot"):
    ids = prompt(0)
    t0 = time.time()
    e.send(("GEN 1 " if a.scenario == "solo" else "BGEN 0 1 ") + ign + " " + ",".join(map(str, ids)))
    got = read_until("DONE" if a.scenario == "solo" else "BADM")
    done = [l for _, l in got if l.startswith("DONE")][0].split()
    pp = [l for _, l in got if l.startswith("PP ")]
    print(f"{a.scenario}: {len(ids)} tokens read in {got[-1][0] - t0:.2f} s = {len(ids) / (got[-1][0] - t0):.0f} tok/s (engine: prompt_ms {done[3]}, last PP {pp[-1] if pp else None})")
elif a.scenario == "four":
    out = {"toks": {}, "done": {}}
    T0 = time.time()
    adm = []
    for i in range(a.slots):
        ids = prompt(i)
        t0 = time.time()
        e.send(f"BGEN {i} {a.n} {ign} {','.join(map(str, ids))}")
        got = read_until("BADM", out)
        tok = [l for _, l in got if l.startswith("T ")]
        out["toks"].setdefault(i, []).insert(0, int(tok[0][2:]))
        nbt = sum(1 for _, l in got if l.startswith("BT "))
        adm.append((i, len(ids), got[-1][0] - t0, nbt))
        print(f"admission {i}: {len(ids)} tokens in {got[-1][0] - t0:.2f} s ({len(ids) / (got[-1][0] - t0):.0f} tok/s), {nbt} tokens of other slots decoded meanwhile")
    t_adm = time.time()
    while any(s not in out["done"] for s in range(a.slots)):
        l = e.q.get(timeout=900)
        if l is None: sys.exit("engine died")
        if l.startswith(("BT ", "BDONE ")): collect(out, l)
    tot = sum(len(v) for v in out["toks"].values())
    print(f"all admitted at +{t_adm - T0:.1f} s; every slot done at +{time.time() - T0:.1f} s; {tot} tokens decoded in all; "
          f"after the last admission: {time.time() - t_adm:.1f} s for the rest")
if dmon:
    time.sleep(1.5)
    dmon.terminate()
    rx = [int(l.split()[1]) for l in open("dmon.txt") if l.strip() and not l.startswith("#")]
    print(f"PCIe received during the scenario: {sum(rx) / 1024:.1f} GiB, busy {sum(1 for x in rx if x > 0)} s, peak {max(rx)} MB/s")
e.send("QUIT")
