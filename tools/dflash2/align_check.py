#!/usr/bin/env python3
"""From an INK2_DRAFT_TRACE log (BLOCK p= tok= drafts ...), the prompt ids and the generated ids: for each draft slot i (1..7) how often draft_i equals the true token at p + i + delta, per delta.
usage: align_check.py trace.log prompt.ids gen.txt"""
import re, sys
import numpy as np
prompt = [int(x) for x in open(sys.argv[2]).read().replace("\n", "").split(",") if x.strip()]
gen = [int(x) for x in open(sys.argv[3]).read().split()[1:]]
full = prompt + gen
blocks = []
for l in open(sys.argv[1]):
    m = re.match(r"BLOCK p=(\d+) tok=(\d+) drafts ((?:\d+ ?)+)", l)
    if m: blocks.append((int(m.group(1)), int(m.group(2)), [int(x) for x in m.group(3).split()]))
print(f"{len(blocks)} blocks; anchor token equals the true token at p for {sum(1 for p, t, d in blocks if p < len(full) and full[p] == t)} of {len(blocks)}")
for delta in (-2, -1, 0, 1, 2):
    hit = np.zeros(7); tot = np.zeros(7)
    for p, t, d in blocks:
        for i in range(1, 8):
            q = p + i + delta
            if 0 <= q < len(full): tot[i - 1] += 1; hit[i - 1] += d[i - 1] == full[q]
    print(f"delta {delta:+d}: per-slot match rate", (hit / np.maximum(tot, 1)).round(2).tolist(), f"| overall {hit.sum() / max(tot.sum(), 1):.3f}")
