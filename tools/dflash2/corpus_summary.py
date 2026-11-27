#!/usr/bin/env python3
"""Summarise corpus.sh output.  usage: corpus_summary.py corpus.log
Lines: <prompt>.ids <config> | decode N tokens T ms (X tok/s) | spec: V verifies, A drafts accepted per verify | P plain tokens, S switches | hash H"""
import re, sys, collections
import numpy as np
rows = {}
for l in open(sys.argv[1]):
    m = re.match(r"(\d+-\w+)\.ids (\S+) \| decode \d+ tokens \d+ ms \(([\d.]+) tok/s\) \| spec: (\d+) verifies, ([\d.]+) drafts accepted per verify \| (\d+) plain tokens, (\d+) switches \| hash (\w+)", l)
    if m: rows[(m.group(1), m.group(2))] = dict(tps=float(m.group(3)), acc=float(m.group(5)), ver=int(m.group(4)), plain=int(m.group(6)), hash=m.group(8))
prompts = sorted({p for p, _ in rows}); cfgs = sorted({c for _, c in rows})
kind = lambda p: p.split("-")[1]
print(f"{len(prompts)} prompts, configs {cfgs}\n")
print(f"{'config':14s} {'mean tok/s':>10s} {'mean acc/verify':>16s} {'mean plain tokens':>18s}")
for c in cfgs:
    v = [rows[(p, c)] for p in prompts if (p, c) in rows]
    print(f"{c:14s} {np.mean([x['tps'] for x in v]):10.1f} {np.mean([x['acc'] for x in v]):16.2f} {np.mean([x['plain'] for x in v]):18.0f}")
print("\nper kind (mean tok/s | mean accepted per verify):")
for k in sorted({kind(p) for p in prompts}):
    print(f"  {k:5s}", "  ".join(f"{c}: {np.mean([rows[(p, c)]['tps'] for p in prompts if kind(p) == k and (p, c) in rows]):6.1f} | {np.mean([rows[(p, c)]['acc'] for p in prompts if kind(p) == k and (p, c) in rows]):4.2f}" for c in cfgs))
def pair(a, b):
    same = [p for p in prompts if (p, a) in rows and (p, b) in rows and rows[(p, a)]['hash'] == rows[(p, b)]['hash']]
    allp = [p for p in prompts if (p, a) in rows and (p, b) in rows]
    if not allp: return
    d = np.array([rows[(p, b)]['tps'] - rows[(p, a)]['tps'] for p in allp]); wins = int((d > 0).sum())
    ds = np.array([rows[(p, b)]['tps'] - rows[(p, a)]['tps'] for p in same]) if same else np.array([])
    print(f"\n{b} vs {a}: {len(allp)} prompts, {wins} wins, mean diff {d.mean():+.1f} tok/s; identical generated text on {len(same)} prompts"
          + (f" (there: {int((ds > 0).sum())} wins, mean {ds.mean():+.1f} tok/s)" if len(ds) else ""))
pair("dspark-q15", "dflash2-q15"); pair("dspark-q0", "dflash2-q0"); pair("dspark-q15", "dspark-q0"); pair("dflash2-q15", "dflash2-q0")
best = [max(((rows[(p, c)]['tps'], c) for c in ("dspark-q15", "dflash2-q15") if (p, c) in rows))[0] for p in prompts]
print(f"\nper-prompt best of the two drafters (q15): mean {np.mean(best):.1f} tok/s")
