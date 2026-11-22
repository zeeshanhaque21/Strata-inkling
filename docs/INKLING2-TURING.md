# Inkling-2-Mini on Turing (RTX 2080 Ti): tensor-core kernels, int8 dense weights, a multi-GPU server

Measured on a 22 GB RTX 2080 Ti (sm_75, 616 GB/s) in a 4-core Xeon box, `--kv i8 --chunk 1024 --tc 1`, the official checkpoint.
Everything here is opt-out through an environment variable and was checked against the CUDA-core kernel it replaces.

## What was added

| Switch | Default | What it does |
|---|---|---|
| `INK2_Q8` (mask) | 1 | Dense BF16 weights quantised to int8 at load (`DMat::q` / `qs`, the layout of the GGUF Q8_0 path, fp16 scale per 32 columns). 1 = unembedding, 2 = `wo`, 4 = q/k/v projections, 8 = the DSpark drafter. 15 = all. The BF16 copy is freed. |
| `INK2_MMA` | 1 | Tensor-core int8 GEMV for 1-8 vectors (`k_mma_q8_multi`): a verify block reads the weights once for all tokens. |
| `INK2_MMA_MOE` | 1 | Tensor-core grouped NVFP4 and MXFP8 experts for a verify block (`k_moe_mma_grp`). |
| `INK2_MMA_PLAIN` | 1 | The same kernel for a single decoded token. |
| `INK2_ATTN16` | 1 | A 16-row variant of the verify attention (one warp per head). The K/V tiles are loaded one tile ahead (16 B per thread) for int8 caches. |
| `INK2_MOECHECK=<layer>` | off | Runs the CUDA-core expert kernel next to the tensor-core one and prints the relative error (needs `--graph 0`). |
| `INK2_PLAIN_RUN`, `INK2_SPEC_MIN`, `INK2_SPEC_MARGIN`, `INK2_SPEC_EST=old` | 32, 3, 1.0 | Tunables of the speculative controller (defaults unchanged). |
| `ignore_eos=1` on `GEN` | off | The end token does not stop the generation (fixed-length benchmarks). |

The weights of the tensor-core expert kernels are exact in fp16 (the e2m1 value times the e4m3 scale, folded as a half2 product);
the activations are rounded to fp16, as in the prompt kernels. Against the CUDA-core kernels the relative error is 1.6e-4 to 2.1e-4 in every layer.

## Measured effect

Plain and forced-speculation decode, 5K-token prompt, one card (tok/s):

| | upstream head | with these changes |
|---|---|---|
| plain decode | 65.6 | 77.7 |
| speculation forced on, predictable text | 140.4 | 214.9 |
| speculation forced on, hard text | 90.7 | 120.1 |

At 100K tokens of context a speculative verify cycle is 72 ms before the attention change and 58 ms after (tokens bit-identical).
Prefill chunk 1024 reads a long prompt 41 % faster than 512 at 128K context; 2048 reads 66 % faster but costs 12 % decode speed (a smaller expert cache).

## Quality

Next-token loss on 4 text segments (2,200 positions each), paired against BF16 (nats, positive = worse):

| | decode path | prefill path |
|---|---|---|
| int8 unembedding | | -0.0002 +- 0.0001 |
| int8 dense weights (all) | +0.0086 +- 0.0041 | -0.0116 +- 0.0039 |
| tensor-core experts, plain decode | -0.0177 +- 0.0041 | |

One segment (repository text with long headers) is volatile: MoE routing flips dominate it, and it carries most of every difference.
On the other three segments each change is within +-0.004 nats. KL against the BF16 logits is not a useful measure here: changing the prefill chunk
size alone gives a KL of 0.12 (a different cuBLAS kernel flips near-tied routing decisions).

## Serving

`terminal/tools/inkling2/ink2-serve start|stop|status|logs` runs one engine per GPU (`ink2_shim.py`, an OpenAI endpoint) behind `ink2_pool.py`,
which sends each request to an idle engine and queues the rest (two engines: two streams at once, any more wait their turn).
Environment: `GPUS="0 1" CTX=131072 KV=i8 CHUNK=1024 RESERVE=800 SPEC=1 Q8=15 PORT=8080 HOST=127.0.0.1`.

## Profiling without Nsight

`tools/ink2prof/kprof.cpp` is a CUPTI activity profiler loaded with `LD_PRELOAD`: GPU time per kernel, including kernels inside CUDA graphs.
Nsight Compute needs `NVreg_RestrictProfilingToAdminUsers=0` in the driver options.
