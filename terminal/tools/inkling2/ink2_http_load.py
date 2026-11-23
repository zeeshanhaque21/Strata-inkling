#!/usr/bin/env python3
"""Concurrent streaming load against an OpenAI-compatible Strata server: N requests with different prompts at once.

  ink2_http_load.py --url http://host:8080/v1 --key KEY --model inkling2-mini --n 4 --max-tokens 300 [--ctx-chars 24000 --files a.cu,b.cu]

Prints each request's prompt tokens, time to first token, tokens received (one per SSE chunk with text or reasoning), its decode rate,
and the aggregate rate over the time when at least one stream was decoding."""
import argparse, json, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--key", required=True)
ap.add_argument("--model", default="inkling2-mini")
ap.add_argument("--n", type=int, default=4)
ap.add_argument("--max-tokens", type=int, default=300)
ap.add_argument("--files", default="")
ap.add_argument("--ctx-chars", type=int, default=0, help="characters of each file put in the prompt (0: a short prompt)")
a = ap.parse_args()

TOPICS = ["the history of the printing press", "how a transistor works", "the water cycle", "why the sky is blue",
          "how vaccines train the immune system", "the rules of chess", "how bridges carry loads", "how GPS works"]
files = [f for f in a.files.split(",") if f]
res = [None] * a.n


def one(i):
    if files and a.ctx_chars:
        txt = open(files[i % len(files)], errors="ignore").read()
        off = (i * 9973) % max(1, len(txt) - a.ctx_chars)
        prompt = f"Here is part of a source file:\n\n{txt[off:off + a.ctx_chars]}\n\nExplain in detail what this code does and how it works."
    else:
        prompt = f"Write a long, detailed explanation of {TOPICS[i % len(TOPICS)]}."
    body = {"model": a.model, "stream": True, "max_tokens": a.max_tokens, "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(a.url + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json", "Authorization": "Bearer " + a.key})
    t0 = time.time(); first = None; n = 0; usage = None; stamps = []
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices", []):
                dl = c.get("delta", {})
                if dl.get("content") or dl.get("reasoning_content"):
                    n += 1
                    now = time.time()
                    if first is None:
                        first = now
                    stamps.append(now)
    res[i] = {"t0": t0, "first": first, "end": time.time(), "n": n, "usage": usage, "stamps": stamps}


T0 = time.time()
th = [threading.Thread(target=one, args=(i,)) for i in range(a.n)]
for t in th:
    t.start()
for t in th:
    t.join()
for i, r in enumerate(res):
    pt = (r["usage"] or {}).get("prompt_tokens")
    rate = (r["n"] - 1) / (r["end"] - r["first"]) if r["first"] and r["n"] > 1 else 0
    print(f"req {i}: prompt {pt} tokens, first token after {r['first'] - r['t0']:.1f} s, {r['n']} tokens, {rate:.1f} tok/s while decoding, done at +{r['end'] - T0:.1f} s")
allst = sorted(s for r in res for s in r["stamps"])
# the window in which every request had started to answer and none had finished
lo = max(r["first"] for r in res); hi = min(r["end"] for r in res)
inside = [s for s in allst if lo <= s <= hi]
if hi > lo and len(inside) > 1:
    print(f"all {a.n} decoding at once for {hi - lo:.1f} s: {len(inside)} tokens = {len(inside) / (hi - lo):.1f} tok/s aggregate")
tot = sum(r["n"] for r in res)
print(f"overall: {tot} tokens in {max(r['end'] for r in res) - T0:.1f} s = {tot / (max(r['end'] for r in res) - T0):.1f} tok/s")
