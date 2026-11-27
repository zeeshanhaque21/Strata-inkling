# Where a plain decode step goes on a Tesla V100 (strata-ink2, 0222 build, sm_70 native)

Measured with Nsight Compute 2025.2 (CUDA 12.9 redistributable; the CUDA 13 profiler cannot target Volta), application replay, one decode step of Inkling-2-Mini
(24-token prompt, int8 dense weights `INK2_Q8=15`, k8v4 KV, no speculation).  `docs/bench-results/v100-profile/` has the raw CSV and `agg.py`.
CUPTI activity records come back with zero timestamps on driver 580 under this setup, so ncu is the only per-kernel timer here.

Plain decode: 79.4 tok/s (12.6 ms per token).  The window covers about 1.4 steps, 1,095 launches, 15.9 ms of GPU time: about 11.5 ms of kernels per step,
so launch gaps are only about 9 %.

| share of GPU time | kernel | avg us | DRAM GB/s (peak 900) | warps active |
| --- | --- | --- | --- | --- |
| 20.1 % + 12.0 % | `k_gemv_nvfp4_sel<2,1>` / `<2,0>` (expert gate_up / down) | 51 / 31 | 465 / 389 | 55 % / 54 % |
| 17.0 % | `k_gemv_q8<4>` (dense, incl. the 6252-block unembedding) | 73 | 644 | 34 % |
| 9.1 % | `k_gemv_q8_qkv<4>` | 45 | 472 | 34 % |
| 8.3 % | `k_gemv_q8<1>` (wo, 288 blocks) | 20 | 381 | 44 % |
| about 25 % | 15 tiny kernels: attn prep / partial / merge, router, gate mix, gate down, attnres, tails | 3 to 11 | under 25 | under 25 % |

## Reading
- The step is memory-bound, but the big GEMVs run at 43 to 60 % of the HBM2 peak.  The 2080 Ti and the V100 decode at the same speed (77.7 against 79.5 tok/s) although the V100 has 46 % more bandwidth.
- The NVFP4 expert GEMV is ALU-bound, not load-bound: about 6 operations per weight to turn e2m1 nibbles into floats, and at 900 GB/s that is about 11 T lane-ops/s against the V100's roughly 7 T.
  An unroll of 3 (three loads in flight, no remainder at K = 2304) was slightly slower (80.2 against 81.2 tok/s), which agrees.
- The dense GEMVs stage the activation vector in shared memory and synchronise before their first weight load, and the small ones (wo, qkv) launch only 270 to 290 blocks (3.6 per SM): latency, not bandwidth, sets their time.
- The launch-shape knobs (`INK2_RPW`, `INK2_SEL_RPW`, `INK2_GEMV_MINB`) are flat within noise (79 to 82 tok/s) on the V100.

## What limits the expert GEMV (measured)
Nsight Compute on `k_gemv_nvfp4_sel` (gate_up and down, 1,152 blocks of 256):
- issue slots 54 % busy, IPC 2.2 of 4, DRAM 50 to 58 %: neither the ALUs nor the memory system are saturated;
- warp stall reasons: **long_scoreboard 51 to 56 %** (waiting for loads), wait 12.7 %, not_selected 10 %, math_pipe_throttle 5.7 %.
So the earlier guess (ALU-bound at about 6 operations per weight) was wrong; it is load latency that is not hidden well enough.

`tools/volta_bw/` models the kernel one piece at a time (gate_up shape, 8 experts, 24 MB per launch, buffers larger than L2):

| variant | time | GB/s |
| --- | --- | --- |
| weights only | 29.7 us | 805 |
| + scale loads | 33.3 us | 717 |
| + activation loads | 34.2 us | 698 |
| + decode and FMA (the kernel) | 43.0 us | 555 |
| same, unroll 3 | 40.2 us | 594 |
| all loads hoisted into registers (9 trips) | 50.7 us | 471 |

A bare streaming read of the same layout reaches about 720 GB/s whether each lane loads 4, 8 or 16 bytes, so the load width is not the problem.
The decode and FMA work costs about 9 us although the ALUs are not full: loads and compute overlap only two iterations deep.

## Done (all bit-identical to the previous build, same tokens)
| change | plain decode, V100, 2.5K context |
| --- | --- |
| baseline (native sm_70 build) | 79.4 tok/s |
| dense int8 GEMV: first weight load issued before x is staged | 81.2 (+2.2 %) |
| NVFP4 expert GEMV: a word decoded as four half2 pairs {n_k, n_k+4} | 82.4 (+1.5 %) |
| together | **+3.8 %** |

## Second round
Three measurements, one kept change and one disproved estimate.

**Shared-memory bank conflicts in the dense int8 GEMVs (kept).**  ncu (`ncu-metrics.sh`, `ncu-q8-metrics*.csv`) showed that 75 % of the shared-load wavefronts of `k_gemv_q8` and `k_gemv_q8_qkv` were bank conflicts:
a lane reads 16 consecutive activations per step, so lanes sit 64 B apart and four lanes of a quarter-warp share their banks.
The staged x now has one padding float4 per eight (`xpad4`, `stage_x_padded`); the conflict share fell from 75 % to under 0.5 % and the shared-load wavefronts from 620 k / 885 k to 155 k / 222 k.
Kernel time: `k_gemv_q8<1>` 23.8 to 20.4 us, `k_gemv_q8<4>` (qkv) 49 to 46 us.  Whole-step decode is unchanged within noise (82.5 tok/s before and after) because those kernels wait on DRAM, not on shared memory; the ncu-serialised step total fell from 15.87 to 15.18 ms.
Same values in the same order, so the tokens are identical (hash `b3b55d972d`).

**Fused launches (kept behind `INK2_FUSE=1`, off by default).**  `k_attn_partial` + `k_attn_merge`, `k_gemv_bf16_rowblock` + `k_router`, and `k_attn_tail` + the following `k_attnres`, each fused with a last-block-done counter that resets itself.
Tokens are bit-identical (plain decode hash `b3b55d972d`; `--ms-test` 4 x 127 steps EQUIVALENT with both settings).  Speed: 82.2 to 82.3 tok/s fused against 82.5 unfused, so 0.3 % slower.

**The "9 % from fusion" estimate was wrong.**  Nsight Systems (`nsys`, installed in `~/nsight-systems`; `gaps.py` reads its sqlite) shows the decode kernels occupy 93.7 % of the step's wall time; only 6.3 % (median gap 0.6 us) is idle between kernels.
Launch overhead is therefore not where the time goes: the CUDA graph already hides it.  The small kernels cost their own run time (a ramp-up, a barrier, a tail), which fusion does not shorten.

**Other launch facts (ncu).**  `k_gemv_q8_qkv<4>` uses 100 registers and is limited to 2 blocks per SM (22 % achieved occupancy, 1.68 waves); the expert kernels use 48 registers, 2.88 waves and 56 % occupancy.

## Third round: the two larger changes, tested as spikes first (V100, expert gate_up GEMV, 24 MB per launch)
Both were measured in microbenchmarks (`tools/volta_bw/gemv_steps.cu`, `gemv_dec.cu`) before any engine work.

**Persistent decode kernel / prefetching the next layer into L2: falsified.**
- Touching the first 1 to 5 MB of the kernel's weights into L2 just before it runs changes it from 44.6 to 44.2 us (0.9 %), where a DRAM-bound kernel would save 1.4 us per MB.
- With the weights fully L2-resident (one expert read eight times, 3 MB, no DRAM traffic) the kernel still takes 33.4 us, against 43.1 us DRAM-fed.
  So the kernel is not limited by DRAM bandwidth; about 33 us of its time is its own issue and latency chain.
- The V100's 6 MB L2 also holds under a quarter of one expert pass (24 MB), so prefetching could only ever have helped a little.

**Cutting the per-weight decode / tensor cores: the ceiling is small.**
| variant (gate_up, 8 experts) | DRAM-fed | L2-resident |
| --- | --- | --- |
| engine: half2 decode + two F2F per pair | 42.8 us | 33.3 us |
| PRMT tables (bit-identical outputs, checked on random data) | 42.9 | 36.7 |
| float-direct shift/or + FMUL | 41.8 | 35.9 |
| decode free (wrong values, floor only) | 38.0 | 28.3 |
| weights + scales + activations loaded, no arithmetic at all | 34.4 | |
- A different decode cannot beat the engine's by more than noise: the PRMT version, which does fewer F2F conversions, is not faster.
- Even a free decode saves only 4.8 us of 42.8 (11 %).  Removing all arithmetic, which is the best case for any tensor-core path, saves 8.7 us (20 %) of this kernel, i.e. at most about 6 % of a decode step.
- A tensor-core path would also need fp16 activations (the engine keeps fp32 here), so it changes the numerics, and on Volta the m8n8k4 fragments waste 7/8 of the columns of a single-token GEMV.  Not worth building.
- The load pieces cost more than the arithmetic: scale loads alone add 3.8 us (29.6 to 33.4), x loads 1 us.  A different scale layout (e.g. one word per lane per four trips) is the one remaining bit-identical idea, worth at most 1 to 2 % of the step.

## Fourth round: batched decode (V100, `--ms-test`, 4096 context, k8v4, same 0222-based volta10 build)
**Scaling.**  Step time of the batched path (ms per step) and aggregate tok/s:

| sequences | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| ms per step | 21.6 | 26.5 | 30.7 | 34.7 |
| aggregate tok/s | 46.4 | 75.5 | 97.7 | 115.2 |

Each extra sequence costs 4.0 to 4.9 ms (its eight experts per layer application), on top of a fixed 17.6 ms.  The single-stream path takes 12.1 ms, so the multi-sequence path carries about 5.5 ms of fixed overhead the plain path does not have.

**Where a 4-sequence step goes (nsys, `msreg3.py`, 63 steps, 38.1 ms per step including some prefill in the window).**  `k_moe_gemv_grp` (CUDA-core grouped experts) 18.2 ms (48 %), `k_mma_q8_multi` (dense int8, emulated tensor core) 7.5 ms (20 %), attention partial 1.6 ms, the rest are small kernels of 0.2 to 1 ms each.  The run is 94.6 % kernel time.
At 4 sequences about 29 distinct experts are read per layer application (28 to 29 expected for 4 x top-8 of 128), 3.5x the bytes of one sequence, at about 430 GB/s: the MoE part is bound by the distinct-expert bytes.

**A native Volta tensor-core expert kernel (`tools/volta_bw/gemv_tc.cu`, gate_up, 8 experts, 24 MB).**  Weights repacked once into a warp-contiguous layout, e2m1 decoded straight into the `mma.m8n8k4` A fragment with the block scale folded in by one HMUL2 (exact in fp16), fp16 activations, fp32 accumulate in Volta's D layout:
| tokens per expert | CUDA cores (engine shape) | tensor core, native layout |
| --- | --- | --- |
| 1 | 47.5 us | 48.7 us |
| 2 | 55.7 | 49.3 |
| 4 | 89.7 | 50.3 |
| 8 | 142.8 | 52.1 |
Max relative error against a double-precision reference with fp16-rounded x: 5e-7.  Two bugs cost most of the time and are worth remembering: an eight-way shared-memory bank conflict from a power-of-two token row stride (148 to 75 us), and one-lane-per-row global loads at half-sector efficiency (75 to 50 us, fixed by the repack).
- The kernel pays off when many tokens share an expert: prefill, speculative verification of long drafts, and batches well past the 8-slot limit.  At 4 sequences the average expert is chosen by about 1.1 tokens, where the two kernels tie, so it does **not** speed up batched decode.
- Batched decode on this card is therefore limited by distinct-expert bytes (about 4 ms per sequence) and, beyond that, by the 5.5 ms fixed overhead.

**What is left for batched decode.**
- The fixed overhead: `k_mma_q8_multi` (7.5 ms against about 5 ms for the single-stream int8 GEMVs) and the multi-sequence small kernels.  A native-layout dense int8 kernel built like the one above is the candidate: about 3 ms of a 35 ms step (about 9 %) at 4 sequences, more at 2 to 3 where the fixed part dominates.
- Nothing cheap on the expert side: the bytes are what they are.

## Fifth round: what other V100 projects do 
v100-skinny (MIT) and its MoE fork volta-nvfp4 run NVFP4 models on V100 with hand-written `mma.m8n8k4` kernels that reach 680 GB/s at M=1 (77 % of a measured 879 GB/s ceiling); anyei/llamacpp-v100 ported the same mapping to the GGUF quantizations.
The mapping: weights on the N side of the MMA, all four quad-pairs sharing one activation fragment, weights prepacked at load into fragment order (nibble-interleaved so the decoder's half2 output is the B fragment), no shared memory in the main loop, K split over warps.
My own tensor-core kernel put weights on the A side with x staged in shared memory and stalled at 48.7 us for exactly that reason.

`tools/volta_bw/gemv_qpn.cu` ports their design for an expert-indirected launch and checks it against a double-precision reference (error 2e-7):
| tokens per expert | 1 | 2 | 4 | 8 |
| --- | --- | --- | --- | --- |
| CUDA cores, engine shape | 47.5 us | 55.7 | 89.7 | 142.8 |
| QPN2 design | **36.4 us** | **36.4** | **36.7** | **41.0** |
That is 23 % faster at one token per expert, which is the decode regime, and flat up to four tokens.  36 us is about the 34 us floor of loading all weights, scales and activations with no arithmetic.
Expected effect if the engine adopts it (expert kernels only, from the profiles above): plain decode 12.1 ms to about 11 ms (82 to about 90 tok/s), 4-sequence batched step 38 ms to about 34 ms (about 11 %).
It uses fp16 activations (x rounded to fp16) with fp32 accumulation, like the Turing tensor-core path, so tokens will not be bit-identical to the CUDA-core path.

## Sixth round: QPN2 experts in the engine (`INK2_QPN=1`, sm_70 only)
The NVFP4 expert layers (0 to 19) are repacked once at load, in place, into the fragment order of `mma.m8n8k4` (a layer is copied to a 0.53 GiB scratch and repacked back from it, so there is one copy: 17.21 GiB in VRAM against 16.67 GiB without).
Kernels: `k_qpn_sel` (one token per expert, plain decode), `k_qpn_grp` (the plan tiles of `kb_moe_plan`, up to 8 rows per expert in one pass over the weights; batched decode and speculative verification), `k_qpn_repack` / `k_qpn_unpack`.
Chunks larger than `SMALL_T` unpack the layer into the scratch (checkpoint order) and run the existing prefill GEMM on it, with the layer's tables swapped for the call.
Design: v100-skinny (MIT, Dennis Vertzyas); see the v100-skinny repository.  The expert factor goes into the half2 scale (the e4m3 block scale alone times 2^14 overflows fp16 for scales above 4; the checkpoint's factors are 3.8e-5 to 3.2e-4).

V100, 4096 context, k8v4, speculation off, every run started from a card at or below 55 C (an 8-sequence run that did not hit 83 C and throttled, which made an earlier comparison invalid):
| | `INK2_QPN=0` | `INK2_QPN=1` |
| --- | --- | --- |
| plain decode, one stream | 82.5 tok/s | 86.4 (+4.8 %) |
| 1 sequence, batched path | 46.4 | 52.1 (+12 %) |
| 2 sequences | 75.3 | 91.6 (+22 %) |
| 3 sequences | 97.4 | 124.3 (+28 %) |
| 4 sequences | 115.1 | 151.6 to 152.1 (+32 %) |
| 8 short sequences | 170.4 to 170.5 | 241.9 to 242.4 (+42 %) |
Batched equals alone (`--ms-test`) for every sequence with QPN on, at 1, 2, 3, 4 and 8 sequences.
Tokens differ from the CUDA-core path after about 30 tokens (fp16 activations, fp32 accumulation, like the Turing tensor-core path).
Engine kernel against the microbenchmark: a code-generation cliff cost the engine kernel 5 us (41 against 36 us) until the activation load and the launch bounds were both changed; any single change recovered it.

**Prefill cost of the single copy.**  Chunks above 8 rows unpack the layer into the scratch before the existing GEMM.  20,000-token prefill, 1024-token chunks: without QPN 1,759 tok/s; with QPN and the first unpack kernel (8-byte strided writes) 1,253 (-29 %); with the shared-memory-staged unpack (coalesced both ways) 1,573 (-10.6 %), tokens identical (hash `ac582fefb5`).
The remaining 1.3 s is 800 layer unpacks of 573 MB read and written at about 700 GB/s: it can only shrink with larger chunks or a prefill GEMM that reads the QPN layout directly.

### Speculative decoding with QPN (`--spec 1`, the model's own DSpark drafter, 300 tokens, greedy, 8192 context, V100 from a cool start)
| prompt (tokens) | QPN, spec off | QPN, spec on | drafts accepted per verify | no QPN, spec on | no QPN, spec off |
| --- | --- | --- | --- | --- | --- |
| code, 5083 | 86.0 | **116.2** | 3.98 | 90.5 | 81.9 |
| chat-code, 36 | 90.9 | **177.4** | 6.11 | 139.7 | 86.4 |
| chat-qa, 33 | 91.0 | **138.0** | 4.62 | 71.5 (0.60 accepted) | 86.5 |
| code, 700 | 87.5 | 85.9 | 2.15 | 79.2 | 83.3 |
| prose, 2500 | 86.5 | 81.7 | 1.83 | 71.6 | 82.4 |
- With QPN a verify (8 rows per expert) is nearly as cheap as a plain step, so speculation pays whenever acceptance is high: +27 to +28 % over the same prompts without QPN where the acceptance is the same (code, chat-code), and up to 177 tok/s.
- Below about 2 accepted drafts per verify it does not pay at all (85.9 against 87.5, 81.7 against 86.5): the acceptance of the built-in drafter is now the limit, not the expert kernels.
- The same prompt can accept very differently with and without QPN (chat-qa: 4.62 against 0.60 drafts per verify), because the fp16 activations change a near-tie in the greedy path; do not read a per-prompt acceptance as a property of the kernels.

## Seventh round: the DFlash 2 drafter (`INK2_DRAFTER=<dir>`)
`incoai/Inkling-2-Mini-DFlash2` (0.5B, 5 layers, grouped dynamic convs, top-16 candidate selector) is implemented in the engine next to the built-in DSpark drafter; design and sources in the research notes (not included).
Correctness: `tools/dflash2/ref_check.py` compares the engine's dump (`INK2_DF2_DUMP`) with an independent numpy implementation of the reference (SGLang `dflash.py`): context K/V, every layer, the final hidden state, candidates and the greedy walk all agree to about 1e-4 on two prompts.
Bugs found on the way (all mine): the engine's attention kernels had the main model's 24 query heads compiled in (the drafter has 32); the output-side convolution used the input side's base kernel; the drafter needs the target's embed norm (0 of 7 drafts accepted without it, 6.2 with it).
Checks that ruled out other explanations: the drafts align with the true tokens at offset 0 (anchor token correct in every block); shifting the three intermediate feature taps by one visit does not help (`INK2_TAPSHIFT`); the features captured on the one-token and verify paths are as close to a teacher-forced prefill as two prefills of different chunk sizes are to each other (median 1 to 13 % at the deep taps, the target's own chunk-size sensitivity, not a drafter-side error).

**Evaluation on 16 varied prompts** (800-token slices of real code, markdown and logs from this machine, 200 greedy tokens, QPN on, V100, every run from a card at or below 60 C; `tools/dflash2/corpus_summary.py`, raw log `docs/bench-results/v100-profile/drafter-corpus.log`):
| drafter, dense weights | mean tok/s | accepted drafts per verify |
| --- | --- | --- |
| DSpark, int8 dense (production) | 118.5 | 3.65 |
| **DFlash 2**, int8 dense | **119.0** | **3.90** |
| DSpark, bf16 dense | 105.6 | 3.77 |
| DFlash 2, bf16 dense | 109.0 | 4.03 |
- DFlash 2 wins 11 of 16 prompts; on the 9 prompts where both generate identical text, 7 wins and +6.5 tok/s on average.  Per kind (tok/s, int8): code 122.4 against 119.4 (Python), 116.2 against 116.8 (CUDA), markdown 102.8 against 103.5, logs 145.1 against 144.4.
- The published gap (acceptance length 4.8 to 5.0 against 3.5) is not reproduced: in greedy decoding the built-in drafter already accepts 3.65 per verify here, and DFlash 2's edge is about +7 % in acceptance.  The published figures are for sampling at temperature 1.
- The "losers" of the first A/B (5,083-token code, chat-qa) were single-trajectory noise: the same prompt with QPN off (a different greedy continuation) gives DFlash 2 4.54 accepted per verify on chat-qa, and DSpark 0.60 on the same text; with QPN on it is 1.84 against 4.62.  Compare drafters only on identical text and over many prompts.
- bf16 dense weights raise acceptance by about 3 % for either drafter but cost 10 tok/s: not worth it.
- Taking the better of the two drafters per prompt gives 123.5 tok/s (+3.8 % over either alone).

## Eighth round: where a speculative cycle goes (V100, QPN on, `INK2_SPEC_BENCH`, nsys, `tools/dflash2/phase_kernels.py`)
`INK2_SPEC_BENCH=N` times the phases of one cycle in isolation (and `INK2_SPEC_BENCH_ONLY=draft|verify|ctx|plain` runs just one so nsys attributes kernels cleanly).
| phase | ms | note |
| --- | --- | --- |
| draft, DSpark | 4.7 to 4.9 | graph + copy of the 7 drafts |
| draft, DFlash 2 | 5.6 to 5.9 | |
| 8-token verify | 42 to 48 | the same at a 36-token and an 800-token prompt: not context-bound |
| drafter context update | 0.2 (1 row), 0.5 (8 rows) | negligible |
| plain decode step, for reference | 11.6 to 14.8 | |
So a cycle is about 90 % verify.  The batched-decode path does the same 8-token workload (8 sequences) in 33 ms; the verify is slower because it runs the prefill-style kernels.

**Verify, per kernel (ms per 8-token verify, steady-state kernels only; the load / prompt one-offs removed):**
| kernel | ms | calls | |
| --- | --- | --- | --- |
| `k_mma_q8_multi` dense int8, emulated tensor core | 13.4 | 162 (83 us each) | about 3x its bandwidth floor (2.9 GB of dense weights, 4 ms at 700 GB/s) |
| `k_qpn_grp` NVFP4 experts | 12.8 | 88 | near the memory floor |
| `kb_attn_tc` verify attention | 5.0 | 50 | the batched-decode attention kernels take about 30 us a call |
| `k_moe_gemv_grp` the 4 MXFP8 layers (not repacked) | 3.5 | 8 (438 us each) | CUDA cores |
| about 15 small kernels (conv, rel logits, gate mix, merge, plan, swiglu, router GEMV ...) | about 11.5 | about 1,500 | 3 to 26 us each |

**Draft step.**  DFlash 2: 4.4 ms of its 5.9 are `k_gemv_bf16_multi` (the drafter's dense layers run in bf16; `INK2_DF2_Q8=1` makes them int8) and 1.9 ms is the 7-row LM head (`k_mma_q8_multi`, 6252 blocks, about 3x its floor).  DSpark: 3.6 ms in `k_mma_q8_multi` calls (the LM head and its dense layers).

**What follows.**  The dense int8 path (`k_mma_q8_multi`, Volta m16n8k8 emulated through m8n8k4) is the largest single cost of a verify (13.4 ms), of the draft step, and of batched decode (7.5 ms of a 4-sequence step).  A QPN8-style kernel (weights on the MMA N side, one shared fp16 activation fragment, weights repacked into fragment order, K split over warps; v100-skinny reaches 82 % of the read ceiling at M <= 8 for 8-bit weights) would take it to about the bandwidth floor: about 9 ms off a verify (about 20 % of a cycle), about 3.5 ms off the DFlash 2 draft step and about 4 ms off a 4-sequence batched step, with fp16 instead of int8 activations (more accurate).
Then, in order of size: the verify attention through the batched-decode kernels (about 3 to 4 ms), the MXFP8 layers (about 2 ms), small-kernel fusion (about 3 ms at best).

## Ninth round: dense int8 GEMVs on the tensor cores (`INK2_DQ8=1`, sm_70)
`k_dq8` (tools/volta_bw/gemv_dq8.cu has the microbenchmark): the QPN2 mapping for int8 weights.  Weights on the MMA N side, one shared fp16 activation fragment (T <= 8 rows), weights repacked into fragment order, int8 -> fp16 exactly by a bias trick, the block scale folded in by one HMUL2 (its rounding to fp16 is 5e-4 relative per weight, below the int8 quantization), K split over 8 to 32 warps by the tile count.
Microbenchmark, V100, 250 W: LM head 678 to 818 us (the emulated `k_mma_q8_multi`: 1,870 us), qkv / gate projection 35 us at 611 GB/s, `wo` 16 us, drafter down 36 us, 600 to 720 GB/s on the large shapes, error 2e-4 against fp64.
In the engine it is a second, fragment-order copy of the int8 weights (about 2.4 GiB, built after `quantize_q8`, segment starts keep the projection's q / k / v slices addressable); T = 1 plain decode, the prefill dequant and every other reader still use the row-major copy.

**Measured at a 120 W power limit** (the card's limit was found at 120 W instead of 250 W partway through the session, after the corpus runs; all runs below are interleaved from cool starts under that one cap, so they compare with each other, not with the 250 W numbers above; batched output equals alone in every run):
| | DQ8 off | DQ8 on | |
| --- | --- | --- | --- |
| 1 sequence, batched path | 36.1 tok/s | 46.7 | +29 % |
| 4 sequences | 108.5 | 130.8 | +21 % |
| 8 sequences | 177.1 | 204.4 | +15 % |
| DFlash 2 draft step | 5.6 ms | 2.8 ms | |
| DSpark draft step | 4.7 ms | 2.9 ms | |
| 8-token verify | 44.3 ms | 37.8 ms | |
| speculative decode, 8 corpus prompts, mean | 87.7 tok/s | 100.9 | +15 % |
The verify gained 6.5 ms of the 9 ms the profile promised: 3.8 ms of dense kernel time remain (148 calls), the verify attention (5 ms), the MXFP8 experts (3.5 ms), the router GEMVs (2.4 ms) and the small kernels are untouched.  Mean acceptance moves from 3.75 to 3.50 accepted per verify (fp16 instead of int8 activations changes the greedy path; the text differs on 5 of 8 prompts).
Next: drop the row-major copy (T = 1 qkv, the prefill dequant), route T = 1 through the same kernel, then the verify attention.

## Tried, no gain (reverted)
- launch-shape knobs `INK2_RPW`, `INK2_SEL_RPW`, `INK2_GEMV_MINB`: flat within noise (79 to 82);
- unroll 3 of the expert GEMV loop: 80.2 (K = 1152 has 4.5 trips), unroll 3 for gate_up only: 82.0 against 82.5 although the microbenchmark gained 6.5 %;
- expert GEMV prologue (expert pointer, first weight / scale / x loads before the scale table and barrier): 82.2 against 82.5;
- all weight and scale loads of a lane hoisted into registers: slower (occupancy drops).

## Where the rest is
One decode step reads about 4.1 GB from DRAM (5,601 MB over 1.4 steps in the ncu window).  At the 720 to 800 GB/s that streaming reads reach on this card that is 5.1 to 5.7 ms, about 175 to 195 tok/s;
the step takes 12.1 ms.  The gap is about 25 % tiny kernels (3 to 11 us each, latency-bound, about 1,050 launches per step) and GEMVs that stream at 50 to 60 % of that.
- Fusing the small kernels: measured, see the second round (no gain; the step is 93.7 % kernel time).
- The larger structural step is a persistent decode kernel that streams the next layer's weights while the current one computes; that is days of work, not a tuning pass.
- A tensor-core NVFP4 / int8 expert path with weights loaded in Volta's own fragment layout (no lane shuffles) would remove the per-weight decode and FMAs from the CUDA cores; the shuffle-based emulation measured 27 % slower, the native layout is untested.
