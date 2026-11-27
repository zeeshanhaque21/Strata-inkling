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

## Tried, no gain (reverted)
- launch-shape knobs `INK2_RPW`, `INK2_SEL_RPW`, `INK2_GEMV_MINB`: flat within noise (79 to 82);
- unroll 3 of the expert GEMV loop: 80.2 (K = 1152 has 4.5 trips), unroll 3 for gate_up only: 82.0 against 82.5 although the microbenchmark gained 6.5 %;
- expert GEMV prologue (expert pointer, first weight / scale / x loads before the scale table and barrier): 82.2 against 82.5;
- all weight and scale loads of a lane hoisted into registers: slower (occupancy drops).

## Where the rest is
One decode step reads about 4.1 GB from DRAM (5,601 MB over 1.4 steps in the ncu window).  At the 720 to 800 GB/s that streaming reads reach on this card that is 5.1 to 5.7 ms, about 175 to 195 tok/s;
the step takes 12.1 ms.  The gap is about 25 % tiny kernels (3 to 11 us each, latency-bound, about 1,050 launches per step) and GEMVs that stream at 50 to 60 % of that.
- Fusing the small kernels (attn_partial + merge via a last-block-done counter, router into its neighbours, gate chains): about 1 to 2 % each, about 9 % in all.
- The larger structural step is a persistent decode kernel that streams the next layer's weights while the current one computes; that is days of work, not a tuning pass.
- A tensor-core NVFP4 / int8 expert path with weights loaded in Volta's own fragment layout (no lane shuffles) would remove the per-weight decode and FMAs from the CUDA cores; the shuffle-based emulation measured 27 % slower, the native layout is untested.
