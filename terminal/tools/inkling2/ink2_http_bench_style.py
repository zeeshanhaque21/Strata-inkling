#!/usr/bin/env python3
"""One streaming request shaped like the strata_matrix benchmark (temperature 0, thinking off, ignore_eos, a filler prompt of random words
ending in a story task), against an OpenAI-compatible server.  Prints the prompt size, time to first token and the decode rate.
  ink2_http_bench_style.py --url http://host:8080/v1 --key KEY --model NAME [--tokens 8000] [--seeds 1,2]"""
import argparse, json, random, time, urllib.request
ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True); ap.add_argument("--key", required=True); ap.add_argument("--model", required=True)
ap.add_argument("--tokens", type=int, default=8000); ap.add_argument("--seeds", default="1,2"); ap.add_argument("--max-tokens", type=int, default=300)
a = ap.parse_args()
WORDS = "river stone cloud engine garden silver market window forest copper lantern orchard meadow harbor candle ribbon thunder pebble violet anchor basket castle desert feather glacier hammer island jungle kettle ladder mirror needle oyster pillow quartz rocket saddle tunnel umbrella velvet wagon yonder zephyr".split()
def filler(n_tokens, seed):
    rng = random.Random(seed); body = []
    for i in range(int(n_tokens / 1.3)):
        body.append(rng.choice(WORDS))
        if i % 24 == 23: body.append(f"({rng.randrange(10**6)}).\n")
    return f"[document {seed:08x}]\n" + " ".join(body) + "\n\nTask: continue writing a long, detailed story about the words above. Do not stop early."
extra = {"temperature": 0.0, "ignore_eos": True, "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}
for seed in [int(s) for s in a.seeds.split(",")]:
    body = {"model": a.model, "stream": True, "max_tokens": a.max_tokens, "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": filler(a.tokens, seed)}], **extra}
    req = urllib.request.Request(a.url + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json", "Authorization": "Bearer " + a.key})
    t0 = time.time(); stamps = []; usage = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            l = raw.decode().strip()
            if not l.startswith("data:") or l.endswith("[DONE]"): continue
            d = json.loads(l[5:]); usage = d.get("usage") or usage
            for c in d.get("choices", []):
                dl = c.get("delta", {})
                if dl.get("content") or dl.get("reasoning_content"): stamps.append(time.time())
    rate = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 else 0
    pt = usage["prompt_tokens"] if usage else 0
    print(f"seed {seed}: prompt {pt} tokens, first token after {stamps[0] - t0:.1f} s ({pt / (stamps[0] - t0):.0f} tok/s read), {len(stamps)} tokens, decode {rate:.1f} tok/s")
