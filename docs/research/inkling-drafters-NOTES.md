# Inkling-2-Mini drafters (research 2026-11-26)

## Candidates
| drafter | what it is | size | published result | status in Strata ink2 |
| --- | --- | --- | --- | --- |
| **DSpark MTP** (`mtp.safetensors`, Thinking Machines) | the model's own 3-layer semi-autoregressive block drafter: parallel backbone + lightweight sequential (Markov, rank 256) head; block size 8; confidence-scheduled verification in the paper | 636 MB BF16 (Unsloth 402 MB Q8_0 copy) | acceptance length 2.95 (MT-Bench) to 3.66 (HumanEval), 7 draft tokens, H200 / SGLang; 1.56x to 2.17x single-stream | **implemented** (`--spec 1`, `draft_load` / `draft_device`, DBLK = 8, DL = 3, DFF = 9216) |
| **DFlash 2** (`incoai/Inkling-2-Mini-DFlash2`, Inco AI, Aug 2026) | block-diffusion drafter: predicts a whole block in one pass, keeps the top-16 candidates per position, a low-rank bilinear selector over adjacent candidate pairs picks one coherent path; two-tap dynamic convolutions stop suffix decay | 0.5B BF16: 5 Qwen3-style layers, hidden 2304, FF 6912, 32 heads / 8 kv, sliding window 2048, block 8, taps on target layers [5, 23, 41, 48] of 48, selector rank 256, top-k 16 | acceptance length 3.90 (MT-Bench) to 4.97 (MATH-500) against 2.95 to 3.47 for DSpark; single-stream 2.03x to 2.77x against 1.56x to 2.17x; concurrency 8: 1.37x to 1.80x against 1.05x to 1.37x; same target output (verification unchanged) | **not implemented** (Strata's DFlash support is for Qwen drafters only) |
| EAGLE-3 style | autoregressive draft head | n/a | the comparison post (kaushalya.io) ranks it below DFlash / DSpark on acceptance per draft cost | not available for Inkling |
| NeMo AutoModel DSpark recipe | training recipe for a custom DSpark drafter | n/a | n/a | would allow a drafter fine-tuned on our own traffic (coding agent prompts) |

All numbers above are the authors' (H200, SGLang, 7 drafted tokens, temperature 1.0, MT-Bench / GSM8K / MATH-500 / HumanEval / MBPP), not ours.  None of them has been reproduced on Strata or on a V100.

## Observations
- Strata's own engine report (Gumbii, X): DSpark on Strata native, BEAM-W4 TREE-6, RTX PRO 6000: 2.04x, bit-exact; the upstream HTTP serve API did not engage it in v0.2.19.
  llama.cpp's early drafter: 0.45x to 0.58x (6 to 14% acceptance), i.e. a drafter can lose badly when acceptance is low: the engine's own switch-to-plain logic matters.
- DFlash 2 proposes a single chain of 7 verified tokens, so verification is the same T = 8 chunk our kernels already run (the QPN grouped kernel serves exactly 8 rows per expert).
- The engine's drafter only proposes; a wrong or lossy drafter port cannot change the output, it can only lower acceptance.

## Result of the port (2026-11-27, V100, Strata ink2, QPN on, greedy)
16-prompt corpus: DFlash 2 119.0 tok/s and 3.90 accepted per verify against 118.5 and 3.65 for DSpark (11 of 16 wins).  The published acceptance gap does not reproduce in greedy decoding (it was measured with temperature-1 sampling).
Per-prompt best of the two: 123.5 tok/s.  Details: strata-inkling `docs/INKLING2-VOLTA-PROFILE.md`, seventh round.
