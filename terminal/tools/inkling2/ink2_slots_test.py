#!/usr/bin/env python3
"""strata-ink2 --serve --slots N protocol test: drives the engine over stdin / stdout like serve/server.py does.

  ink2_slots_test.py --exe build/strata-ink2 --model DIR --ids a.ids,b.ids,c.ids,d.ids [--slots 4] [--n 48] [--extra "--kv k8v4 --ctx 32768"]

Checks (every one greedy):
  1. each prompt alone in a slot (BGEN, nothing else running) gives its reference tokens
  2. the same prompts together (admissions one after another, decoding in between) give the same tokens, per slot
  3. a second turn (prompt + answer + new text) in the same slot reads only the new tokens (DONE's last field) and equals a fresh read
  4. the same second turn sent to ANOTHER slot copies the state (reads only the new tokens) and gives the same tokens
  5. BSTOP ends a slot (BDONE cancel) without disturbing the others
  6. a sampled request (temperature, seed) runs next to a greedy one, which still gives its reference tokens
  7. a solo GEN afterwards still works
Prints the aggregate decode rate of the together run.  Exit code 1 on any failure."""
import argparse, os, queue, subprocess, sys, threading, time


class Engine:
    def __init__(self, exe, model, slots, extra, env):
        cmd = [exe, "--serve", "--model", model, "--slots", str(slots)] + extra.split()
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open("ink2_slots_test.log", "w"),
                                  text=True, bufsize=1, env={**os.environ, **env})
        self.q = queue.Queue()
        threading.Thread(target=self._rd, daemon=True).start()
        self.info = {}
        while True:
            l = self.q.get(timeout=600)
            if l is None:
                sys.exit("the engine ended before READY - see ink2_slots_test.log")
            if l.startswith("INFO "):
                for kv in l.split()[1:]:
                    k, _, v = kv.partition("=")
                    self.info[k] = v
            if l.startswith("READY"):
                break

    def _rd(self):
        for l in self.p.stdout:
            self.q.put(l.rstrip("\n"))
        self.q.put(None)

    def send(self, s):
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()


def ids_of(path):
    return [int(x) for x in open(path).read().replace("\n", ",").split(",") if x.strip()]


def admit(e, slot, ids, n, keys="", out=None):
    """BGEN <slot> ...: returns (admission token or None, reused, continues).  Reads PP / T / DONE / BADM."""
    e.send(f"BGEN {slot} {n}{(' ' + keys) if keys else ''} {','.join(map(str, ids))}")
    tok, reused, cont = None, None, None
    while True:
        l = e.q.get(timeout=600)
        if l is None:
            sys.exit("the engine died during an admission")
        if l.startswith("T "):
            tok = int(l[2:])
        elif l.startswith("DONE"):
            reused = int(l.split()[8])
        elif l.startswith("BADM"):
            cont = l.split()[2] == "1"
            return tok, reused, cont
        elif l.startswith("ERR"):
            sys.exit(f"ERR on BGEN: {l}")
        elif l.startswith(("BT ", "BDONE ")) and out is not None:   # slots decoding while the prompt is read
            collect(out, l)


def collect(out, l):
    f = l.split()
    if f[0] == "BT":
        out["toks"].setdefault(int(f[1]), []).append(int(f[2]))
    elif f[0] == "BDONE":
        out["done"][int(f[1])] = (int(f[2]), f[3], float(f[4]))


def run_slots(e, slots, out, timeout=600):
    """Reads BT / BDONE until every slot in `slots` is done."""
    end = time.time() + timeout
    while any(s not in out["done"] for s in slots):
        try:
            l = e.q.get(timeout=max(1, end - time.time()))
        except queue.Empty:
            sys.exit("timeout waiting for BDONE")
        if l is None:
            sys.exit("the engine died")
        if l.startswith(("BT ", "BDONE ")):
            collect(out, l)


def gen_slot(e, slot, ids, n, keys="", fresh=True):
    out = {"toks": {}, "done": {}}
    tok, reused, cont = admit(e, slot, ids, n, keys, out)
    if tok is not None:
        out["toks"].setdefault(slot, []).insert(0, tok)
    if cont:
        run_slots(e, [slot], out)
    return out["toks"].get(slot, []), reused, out["done"].get(slot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--extra", default="--kv i8 --ctx 32768 --chunk 1024 --tc 1")
    a = ap.parse_args()
    prompts = [ids_of(p) for p in a.ids.split(",")]
    env = {"INK2_Q8": os.environ.get("INK2_Q8", "15")}
    e = Engine(a.exe, a.model, a.slots, a.extra, env)
    print("engine ready:", e.info)
    fails = []

    def check(name, ok, detail=""):
        print(("PASS " if ok else "FAIL ") + name + (" - " + detail if detail else ""))
        if not ok:
            fails.append(name)

    n = a.n
    ign = "ignore_eos=1"
    # 1. alone
    ref = []
    for i, p in enumerate(prompts):
        toks, reused, done = gen_slot(e, i, p, n, ign)
        ref.append(toks)
        check(f"alone slot {i}: {len(toks)} tokens, BDONE {done}", len(toks) == n and done and done[0] == n and done[1] == "length")
    # 2. together
    out = {"toks": {}, "done": {}}
    t0 = time.time()
    firsts = {}
    for i, p in enumerate(prompts):
        tok, reused, cont = admit(e, i, p, n, ign, out)
        firsts[i] = tok
        out["toks"].setdefault(i, []).insert(0, tok)
        if not cont:
            sys.exit("admission did not continue")
    t_adm = time.time()
    run_slots(e, list(range(len(prompts))), out)
    t1 = time.time()
    for i in range(len(prompts)):
        got = out["toks"][i]
        first = next((k for k in range(min(len(got), len(ref[i]))) if got[k] != ref[i][k]), None)
        check(f"together slot {i} == alone", got == ref[i], f"{len(got)} tokens" + (f", first difference at {first}: {got[first]} vs {ref[i][first]}" if first is not None else ""))
    tot = sum(len(v) for v in out["toks"].values())
    print(f"  together: {tot} tokens, admissions {t_adm - t0:.1f} s, then {tot} tokens in {t1 - t_adm:.2f} s")
    # 3. a second turn: cold (slot 0 wiped first), then the conversation again and its next turn in the same slot (history reused)
    p0 = prompts[0]
    turn2 = p0 + ref[0][:-1] + prompts[1][:40]
    gen_slot(e, 0, prompts[1][:300], 2, ign)                                  # slot 0 now holds something unrelated
    cold, reused_cold, _ = gen_slot(e, 0, turn2, 24, ign)
    check(f"turn 2 cold: {reused_cold} reused", reused_cold == 0)
    gen_slot(e, 0, p0, n, ign)                                               # slot 0 holds p0 + its answer again
    warm, reused_warm, _ = gen_slot(e, 0, turn2, 24, ign)
    first = next((k for k in range(min(len(warm), len(cold))) if warm[k] != cold[k]), None)
    check(f"turn 2 in the same slot reads only the new tokens ({reused_warm} of {len(turn2)} reused)", reused_warm == len(p0) + n - 1,
          f"tokens {'equal to the cold read' if first is None else 'differ from the cold read at ' + str(first) + ' (chunking changes the arithmetic)'}")
    # 4. its next turn in ANOTHER slot: the state is copied from slot 0
    turn3 = turn2 + warm[:-1] + prompts[2][:30]
    inplace_src_toks, reused_ip, _ = gen_slot(e, 2, turn3, 16, ign)           # slot 2 holds prompt 2: slot 0's state is the longer match -> copied
    check(f"turn 3 in another slot copies the state ({reused_ip} of {len(turn3)} reused)", reused_ip == len(turn2) + len(warm) - 1)
    same_slot, reused_s, _ = gen_slot(e, 0, turn3, 16, ign)                   # the same turn in place
    check(f"turn 3 copied == turn 3 in place ({reused_s} reused in place)", same_slot == inplace_src_toks, f"{inplace_src_toks[:6]} vs {same_slot[:6]}")
    # 5. BSTOP
    out = {"toks": {}, "done": {}}
    for i in (0, 1):
        tok, _, cont = admit(e, i, prompts[i], 400, ign, out)
        out["toks"].setdefault(i, []).insert(0, tok)
    time.sleep(0.4)
    e.send("BSTOP 1")
    run_slots(e, [0, 1], out)
    check(f"BSTOP: slot 1 ended with {out['done'][1]}, slot 0 ran on to {out['done'][0]}",
          out["done"][1][1] == "cancel" and out["done"][0][1] == "length" and out["done"][0][0] == 400 and out["toks"][0][:n] == ref[0])
    # 6. sampled next to greedy
    out = {"toks": {}, "done": {}}
    tok, _, cont = admit(e, 0, prompts[0], n, ign, out)
    out["toks"].setdefault(0, []).insert(0, tok)
    tok, _, cont = admit(e, 3, prompts[3], n, ign + " temperature=0.8 top_p=0.9 seed=7", out)
    out["toks"].setdefault(3, []).insert(0, tok)
    run_slots(e, [0, 3], out)
    check("greedy slot unchanged next to a sampled one", out["toks"][0] == ref[0])
    check("sampled slot produced tokens", len(out["toks"][3]) == n and out["toks"][3] != ref[3], f"{out['toks'][3][:8]}")
    # 7. solo GEN
    e.send(f"GEN 16 {ign} {','.join(map(str, prompts[0]))}")
    got, done = [], None
    while True:
        l = e.q.get(timeout=600)
        if l.startswith("T "):
            got.append(int(l[2:]))
        elif l.startswith("DONE"):
            done = l
            break
        elif l.startswith("ERR"):
            done = l
            break
    check(f"solo GEN after slot work: {len(got)} tokens ({done})", len(got) == 16 and got[:8] == ref[0][:8],
          f"{got[:8]} vs {ref[0][:8]}" if got[:8] != ref[0][:8] else "")
    e.send("QUIT")
    print("FAILED: " + ", ".join(fails) if fails else "ALL PASSED")
    sys.exit(1 if fails else 0)


main()
