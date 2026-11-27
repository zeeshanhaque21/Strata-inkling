#!/usr/bin/env python3
"""Drive upstream Strata's strata-inkling in --serve mode on the same token prompts as the ink2 benchmarks (greedy), one process per drafter setting.
usage: up_serve_bench.py ENGINE MODEL GPU_UUID draft(on|off) OUT   prompts come from the lists below (run it where the .ids files are)."""
import os, subprocess, sys, time
engine, model, uuid, draft, out = sys.argv[1:6]
R = os.path.expanduser("~/strata-inkling/runs/"); C = os.path.expanduser("~/strata-bench/corpus/")
prompts = [(R + p + ".ids", p, 300) for p in ("code", "chat-code", "chat-qa", "code700", "segA")] + \
          [(C + p + ".ids", p, 200) for p in ("00-py", "02-py", "04-cuda", "06-cuda", "08-md", "10-md", "13-cfg", "15-cfg")]
def temp():
    return int(subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits", "-i", uuid], capture_output=True, text=True).stdout.strip() or 0)
env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER="PCI_BUS_ID")
p = subprocess.Popen([engine, "--serve", "--native", model, "--max-context", "8192", "--gpu", "0", "--kv", "k8v4", "--draft", draft], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env, bufsize=1)
def read_until(prefix):
    lines = []
    for l in p.stdout:
        lines.append(l.rstrip("\n"))
        if l.startswith(prefix): return lines
    return lines
ready = read_until("READY"); print("engine:", ready[-1][:80], flush=True)
log = open(out, "a")
for f, name, n in prompts:
    ids = open(f).read().strip().replace("\n", "").replace(" ", "")
    # upstream's engine keeps the card at P0 while it is loaded (62-67 C idle on a V100), so the start threshold is 72 C; a hard stop at 78 C
    for _ in range(90):
        if temp() <= 72: break
        time.sleep(4)
    if temp() >= 78: p.kill(); sys.exit("aborted: the card reached %d C" % temp())
    p.stdin.write(f"GEN {n} temperature=0 {ids}\n"); p.stdin.flush()
    lines = read_until("DONE"); done = lines[-1].split()
    # DONE <gen> <prompt> <prompt_ms> <decode_ms> <reason> <accepted> <offered> <reused>
    gen, plen, pms, dms = int(done[1]), int(done[2]), float(done[3]), float(done[4])
    acc, off = (int(done[6]), int(done[7])) if len(done) > 7 else (0, 0)
    log.write(f"up draft={draft} {name} | decode {gen} tokens {dms:.0f} ms ({gen / dms * 1000 if dms else 0:.1f} tok/s) | prompt {plen} tokens {pms:.0f} ms ({plen / pms * 1000 if pms else 0:.0f} tok/s) | drafts accepted {acc} of {off} offered | finish {done[5]}\n"); log.flush()
p.stdin.write("QUIT\n"); p.stdin.flush()
try: p.wait(timeout=20)
except Exception: p.kill()
