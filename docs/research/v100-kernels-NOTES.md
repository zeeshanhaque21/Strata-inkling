# V100 optimized-kernel research (2026-11-26)

Question: other people run LLMs on V100 now; is there an optimized kernel we should be using instead of our CUDA-core / emulated paths?

## Found, in order of relevance
1. **dnv2003/v100-skinny** (MIT, kernels/skinny_kernels.cu).
   Hand-written NVFP4 (W4A16) and FP8 (W8A16) GEMV/GEMM kernels around `mma.sync.m8n8k4` ("QPN2" / "QPN8").
   Measured by the authors on V100: NVFP4 679 GB/s at M=1 (77% of a measured 879 GB/s read ceiling), 620 GB/s at M=8, FP8 719 GB/s (82%).
   Design: weights on the MMA N side, all four quad-pairs share ONE activation fragment, weights prepacked at load into fragment order (nibble-interleaved so the decoder's half2 output is the B fragment), no shared memory in the main loop, split-K over warps, one reduce at the end.
2. **gilby/volta-nvfp4** (fork of 1): the same kernels for fused MoE experts (`qpn2_moe_kernel.cu`, 174 lines): slots sorted by expert into row-blocks of 8, A-fragment rows via indirection, per-(expert, tile) global scale.
   Reports Qwen3.6-35B-A3B (3B active) NVFP4 at 99.4 tok/s single-stream on one V100 (vLLM, TP2 numbers for batches: N=4 297.7, N=8 509.4 aggregate on two cards).
   Also: a cudagraph-capture bug (capture list `[1,2]`) collapsed concurrency at N>=3 to eager mode.  Not our problem (our engine always graphs), but a reminder to check graph coverage per batch size.
3. **anyei/llamacpp-v100**, `docs/volta-smallt-gemm-plan.md`: a llama.cpp V100 fork that ported the same quadpair-split-N mapping to Q8_0, Q4_K, Q5_K, Q6_K, IQ4_XS and others and reached the same conclusions: lane-per-row streaming of unpacked weights gives scattered HBM streams (slow), coalesced layouts win, T=1 on dp4a is already at the floor.
4. Not yet read: 1Cat-vLLM (SM70 vLLM fork, FlashAttention for SM70), WyvernTKC/llama.cpp-4xV100, jackjusko/jusko-llama-volta-qwen3flash, ruizette/llama.cpp-volta-gpu-optimizer, proprietarylegal.github.io/volta-v100-research.

Cloned read-only into this directory (v100-skinny/, volta-nvfp4/, llamacpp-v100/); nothing is pushed anywhere.

## Result of porting their design into our microbenchmark
`strata-inkling/tools/volta_bw/gemv_qpn.cu` (attribution in the file header), V100, gate_up 2304 x 2304, 8 experts, 24 MB per launch:
| kernel | M=1 | M=2 | M=4 | M=8 |
| --- | --- | --- | --- | --- |
| our CUDA-core kernel (engine shape) | 47.5 us | 55.7 | 89.7 | 142.8 |
| my native-layout tensor-core kernel (weights as A, x via smem) | 48.7 | 49.3 | 50.3 | 52.1 |
| QPN2 design (weights as B, shared A, prepacked, split-K 8) | **36.4** | **36.4** | **36.7** | **41.0** |
Correct against a double-precision reference (max relative error 2.3e-7).  36 us is essentially the 34 us no-arithmetic floor measured earlier.
