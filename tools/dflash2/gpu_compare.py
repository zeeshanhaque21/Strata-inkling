#!/usr/bin/env python3
"""Side-by-side table from the xgpu.sh / xgpu-batched.sh logs of one build on several GPUs.
usage: gpu_compare.py <dir with xgpu-<label>.log and xgpu-batched-<label>.log> label1 label2 ...   (prints markdown)"""
import re, sys, os
d, labels = sys.argv[1], sys.argv[2:]
def load(label):
    out = {}
    for name in (f"xgpu-{label}.log", f"xgpu-batched-{label}.log"):
        p = os.path.join(d, name)
        if not os.path.exists(p): continue
        for l in open(p):
            l = l.rstrip("\n")
            m = re.match(r"batched (\d+) \(chunk \d+\) \| ([\d.]+) tok/s aggregate, [\d.]+ tok/s each, ([\d.]+) ms/step \| (\w+)", l)
            if m: out[f"batched {m.group(1)}"] = (float(m.group(2)), f"{m.group(3)} ms/step", m.group(4) == "EQUIVALENT"); continue
            m = re.match(r"plain decode, spec off \| decode \d+ tokens \d+ ms \(([\d.]+) tok/s\)", l)
            if m: out["plain decode"] = (float(m.group(1)), "", True); continue
            m = re.match(r"prefill 20k \(chunk (\d+)\) \| prompt \d+ tokens \d+ ms \(([\d.]+) tok/s\)", l)
            if m: out["prefill 20k"] = (float(m.group(2)), f"chunk {m.group(1)}", True); continue
            m = re.match(r"spec (\w+) (\S+)\.ids \| decode \d+ tokens \d+ ms \(([\d.]+) tok/s\) \| ([\d.]+) drafts accepted per verify", l)
            if m: out[f"spec {m.group(1)} {m.group(2)}"] = (float(m.group(3)), f"acc {m.group(4)}", True); continue
            m = re.match(r"phases (\w+): (.*)", l)
            if m:
                for part in m.group(2).split(";"):
                    pm = re.match(r"\s*(draft|verify 8 tokens|plain decode step)\s+([\d.]+) ms", part)
                    if pm: out[f"phase {m.group(1)} {pm.group(1)}"] = (float(pm.group(2)), "ms", True)
    return out
data = {l: load(l) for l in labels}
rows = []
for k in ["batched 1", "batched 2", "batched 3", "batched 4", "batched 8", "plain decode", "prefill 20k"]:
    rows.append(k)
spec_keys = sorted({k for l in labels for k in data[l] if k.startswith("spec ")}, key=lambda s: (s.split()[2][0].isdigit() is False, s.split()[2], s.split()[1]))
rows += spec_keys
rows += sorted({k for l in labels for k in data[l] if k.startswith("phase ")})
print("| experiment | " + " | ".join(labels) + " |"); print("|" + "---|" * (len(labels) + 1))
for k in rows:
    cells = []
    for l in labels:
        v = data[l].get(k)
        cells.append("n/a" if v is None else f"{v[0]:g}" + (f" ({v[1]})" if v[1] else "") + ("" if v[2] else " NOT EQUIVALENT"))
    print(f"| {k} | " + " | ".join(cells) + " |")
