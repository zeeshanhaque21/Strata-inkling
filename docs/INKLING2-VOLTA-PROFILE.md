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

## Second round (branch `volta-fuse`)
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
