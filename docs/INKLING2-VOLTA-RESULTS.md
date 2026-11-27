# Inkling-2-Mini on a Tesla V100: every experiment, compared

Strata `ink2` engine, V100 PCIe 32 GB (GPU 0 of the T5810 "cachyos"), Inkling-2-Mini official checkpoint, `INK2_Q8=15`, k8v4 KV.
Companion to `INKLING2-VOLTA-PROFILE.md` (the narrative, kernel by kernel) and `INKLING2-VOLTA.md` (the sm_70 bring-up).
Raw logs: `docs/bench-results/v100-profile/`.  Branches (fork `zeeshanhaque21/Strata-inkling`): `volta-opt`, `volta-fuse`, `volta-qpn`, `dflash2`, `dense-w8`.

## Test conditions

Numbers are comparable only within one condition.

| | Condition | When |
| --- | --- | --- |
| **A** | 250 W, SM up to 1380 MHz | everything until about 11:00 on 2026-11-27, and the final rerun after the clock lock was removed |
| **B** | graphics clock locked to 600-900 MHz (power limit 120 W, later 250 W) by `v100-cool.sh on` | about 11:00 until `nvidia-smi -rgc`.  All the dense-kernel runs except the final rerun, and (inferred from its 14.8 ms plain step against 11.6 ms at full clocks) the first draft / verify profile |

"Cool start": a run begun with the card at or below 55 C (60 C for the 16-prompt corpus).  Every run of the later rounds went through the guarded `v100-bench` unit (the thermal guard stops it at 80 C).

## 1. Single-stream plain decode (speculation off, 4K context, 200 tokens, condition A)

| Experiment | tok/s | Change | Tokens identical? | Outcome |
| --- | --- | --- | --- | --- |
| Native sm_70 build (baseline) | 79.4 | | | |
| + dense int8 GEMV prefetches its first load | 81.2 | +2.3 % | yes | kept |
| + NVFP4 expert decoded as half2 pairs | 82.4-82.5 | +1.5 % | yes | kept |
| Launch-shape knobs (`RPW`, `SEL_RPW`, `MINB`) | 79-82 | flat | yes | reverted |
| Expert loop unroll 3 | 80.2 / 82.0 | -2.8 % / -0.6 % | yes | reverted |
| Expert prologue prefetch | 82.2 | -0.4 % | yes | reverted |
| All loads hoisted into registers | slower | | yes | reverted |
| Padded activations (shared-memory bank conflicts 75 % to 0 %) | 82.5 | 0 % | yes | kept (kernels 23.8 to 20.4 us) |
| Fused launches (`INK2_FUSE=1`; only 6.3 % of a step is idle gaps) | 82.2-82.3 | -0.3 % | yes | off by default |
| QPN expert kernels (`INK2_QPN=1`) | **86.4** | **+4.8 %** | no (drifts after about 30 tokens) | opt-in |
| Dense int8 kernel (`INK2_DQ8=1`) | not routed for T = 1 | | | next step |

## 2. Kernel microbenchmarks (condition A)

Expert gate_up (2304 x 2304, 8 experts, 24 MB):

| Kernel | M = 1 | M = 2 | M = 4 | M = 8 |
| --- | --- | --- | --- | --- |
| Engine CUDA-core kernel | 47.5 us | 55.7 | 89.7 | 142.8 |
| My tensor-core kernel, first version (weights as A, activations in shared memory) | 148 | 148 | 148 | 148 |
| Same, shared-memory stride padded | 75 | 75 | 77 | 79 |
| Same, weights repacked warp-contiguous | 48.7 | 49.3 | 50.3 | 52.1 |
| **QPN2 port** (v100-skinny design) | **36.4** | **36.4** | **36.7** | **41.0** |

Expert-kernel floors (the L2-prefetch idea was disproved):

| Probe | Result |
| --- | --- |
| Weights only | 29.6 us (806 GB/s) |
| + scales | 33.4 us |
| + activations | 34.4 us |
| Real kernel | 43.1 us |
| Real kernel, weights L2-resident (no DRAM) | 33.4 us |
| Pre-warming 5 MB of the weights into L2 | 44.6 to 44.2 us |
| Free decode (wrong values) | 38.0 us |
| PRMT-table decode (exact outputs) | 42.9 us |
| Float-direct decode | 41.8 us (nibble mapping wrong: does not count) |

Dense int8 (error about 2e-4 against fp64):

| Shape | Old emulated kernel | New kernel | Best K split | Speed |
| --- | --- | --- | --- | --- |
| LM head 200064 x 2304 | 1,870 us | 678-818 us | 8 | 600-720 GB/s |
| qkv / gate projection 8704 x 2304 | about 83 us per call (average) | 35 us | 8 | 611 GB/s |
| `wo` 2304 x 3072 | | 21.5 to 16 us | 32 | 480 GB/s |
| Drafter down 2304 x 9216 | | 57 to 36 us | 32 | 619 GB/s |

## 3. Batched decode (`--ms-test`, short prompts, 4K context; output equal to alone in every run)

| Sequences | CUDA-core experts (A) | + QPN (A) | + QPN + dense kernel (A) | Dense kernel off, clock-locked (B) | Dense kernel on, clock-locked (B) |
| --- | --- | --- | --- | --- | --- |
| 1 | 46.4 | 52.2 | not measured | 36.1 | 46.7 |
| 2 | 75.3 | 91.0 | not measured | | |
| 3 | 97.4 | 123.4 | not measured | | |
| 4 | 115.1-115.6 | 151.6 | not measured | 108.5 | 130.8 |
| 8 | 170.5 | 241.9-242.4 | **269.6-270.5** | 177.0-177.2 | 204.4-204.9 |

- Cumulative at 8 sequences: 170.5 to 242 to 270 tok/s (1.58x).
- The dense kernel at 8 sequences: +11.8 % at full clocks, +15.5 % clock-locked.
- Excluded: an 8-sequence run at 83 C that throttled (105.6 tok/s).

## 4. Prefill (20,000 tokens, 1,024-token chunks, condition A)

| Variant | tok/s | Change |
| --- | --- | --- |
| QPN off | 1,759-1,760 | |
| QPN on, first unpack kernel (strided writes) | 1,253 | -29 % |
| QPN on, coalesced unpack | 1,573 | -10.6 % |

## 5. Speculative decode, one stream (greedy, 300 tokens, QPN on unless noted)

Built-in drafter (DSpark) against DFlash 2, condition A:

| Prompt (tokens) | Spec off | DSpark | Accepted / verify | DFlash 2 | Accepted / verify |
| --- | --- | --- | --- | --- | --- |
| chat-code (36) | 90.9 | 177.4 | 6.11 | **197.7-201.2** | 6.20 |
| code (5,083) | 86.0 | **116.2-119.8** | 3.98-4.22 | 81.0 | 1.76 |
| chat-qa (33) | 91.0 | **138.0** | 4.62 | 87.2-88.5 | 1.73-1.84 |
| code (700) | 87.5 | 85.9 | 2.15 | **146.2-148.4** | 5.28 |
| prose (2,500) | 86.5 | 81.7 | 1.83 | **136.4-138.2** | 5.26 |

16-prompt corpus (800-token slices of code, docs and logs, 200 tokens, condition A; `drafter-corpus.log`):

| Drafter, dense weights | tok/s | Accepted / verify |
| --- | --- | --- |
| DSpark, int8 | 118.5 | 3.65 |
| **DFlash 2, int8** | **119.0** | **3.90** |
| DSpark, bf16 | 105.6 | 3.77 |
| DFlash 2, bf16 | 109.0 | 4.03 |

DFlash 2 wins 11 of 16 prompts; on the 9 prompts where both generate identical text it wins 7, by +6.5 tok/s on average.  The better drafter per prompt: 123.5 tok/s.

What did and did not explain the prompts where DFlash 2 lost:

| Test | Result |
| --- | --- |
| Drafts shifted by one position | No: offset 0 is best (DFlash 2 matches 88.5 % on chat-code, 33 % on chat-qa) |
| Feature taps shifted one visit | No: chat-qa 1.12, code 2.06, chat-code 6.02 accepted per verify |
| Features from the speculative path against a clean prefill | Same difference as two prefills of different chunk sizes (median 1-13 % at the deep taps) |
| QPN on/off and int8/bf16 dense (chat-qa) | 1.84 / 4.54 / 2.32 / 4.73: the text differs, so trajectory noise, not a kernel property |
| Block math against a numpy reference | Matches to about 1e-4 on every stage, on two prompts |

Dense kernel in speculative decode:

| Test | Dense kernel off | Dense kernel on | Change |
| --- | --- | --- | --- |
| 8 corpus prompts, mean, DFlash 2, condition B | 87.7 | 100.9 | +15 % |
| Solo DFlash 2, `15-cfg`, condition B | 141.9 | 171.4 | +20.8 % |
| Solo DFlash 2, `15-cfg`, condition A | 191.7 | **223.7** | **+16.7 %** |

(accepted per verify: 6.54 both ways on `15-cfg`; the 8-prompt mean moves 3.75 to 3.50 because fp16 activations change the greedy path.)

## 6. One speculative cycle, phase timer (`INK2_SPEC_BENCH`; probably condition B, so absolute ms are inflated by about 1.3x; the proportions hold)

| Phase | Before the dense kernel | After |
| --- | --- | --- |
| Draft, DFlash 2 | 5.6 ms | 2.8 ms |
| Draft, DSpark | 4.7 ms | 2.9 ms |
| 8-token verify | 44.3-44.8 ms | 37.8-38.2 ms |
| Context update | 0.2-0.5 ms | 0.1-0.3 ms |
| Plain decode step (reference) | 14.8 ms | unchanged |

Verify per kernel (ms per verify, nsys):

| Kernel | Before | After |
| --- | --- | --- |
| Dense int8 | 13.4 (162 calls) | 3.8 (149 calls) + 2.4 still on the old kernel (20 calls; fixed since) |
| NVFP4 experts (`k_qpn_grp`) | 12.8 | 12.6 |
| Verify attention (`kb_attn_tc`) | 5.0 | 5.0 |
| MXFP8 layers (not QPN) | 3.5 | 3.5 |
| Router GEMVs | 2.4 | 2.4 |
| About 15 small kernels | about 11.5 | about 11.5 |

The 4-sequence batched step was 38.1 ms in the earlier profile: 18.2 ms the CUDA-core expert kernel, 7.5 ms the old dense kernel.

## Caveats

- Not measured: the 16-prompt corpus with the dense kernel at full clocks (8 prompts clock-locked, one prompt at full clocks); the cycle profile at full clocks.
- Nothing is deployed; production still runs the older 2080 Ti build.
- The dense kernel carries a 2.4 GiB second copy of the int8 weights; QPN is a single in-place copy plus 0.5 GiB of scratch.
- Everything new is opt-in: `INK2_QPN=1`, `INK2_DQ8=1`, `INK2_DRAFTER=<dir>`.
