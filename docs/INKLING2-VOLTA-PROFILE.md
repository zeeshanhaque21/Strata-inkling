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

## Done
- Dense int8 GEMV issues each lane's first 16-byte weight load before staging x (`q8_prefetch`): bit-identical output, plain decode 79.4 to 81.2 tok/s (+2.2 %, three runs each).

## Next, by expected gain (estimates, not measurements)
1. NVFP4 expert GEMV with cheaper decoding (pair nibbles n and n+4 across the two halves of a half2 word: about 25 % fewer ALU operations bit-exactly; or half2 math / tensor cores with a Volta-native fragment layout: the larger win but not bit-exact).  Up to about 12 % of the step.
2. Fuse the small kernels (attention prep, partial, merge, router, gates): up to about 9 %.
3. Split-K or more warps for the small dense GEMVs: about 4 %.
