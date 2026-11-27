# Upstream Strata 0.2.23 against this branch, same V100, same 13 token prompts

Tesla V100-PCIE-32GB (PCIe, 250 W, full clocks), greedy decoding, k8v4 KV, one request at a time, decode tokens per second.

- **Upstream:** `strata-inkling` 0.1.70 (Strata 0.2.23, commit 4560d0379) built for sm_70 with CUDA 12.9 (`STRATA_EXPERIMENTAL_SM60=ON`), model `Inkling-2-Mini-SQv2-20.strata` (SQ-20 v2, 19.4 GB), `--draft off` and `--draft on` (loads the DFlash 2 sidecar, confirmed in its startup log). Driven through its `--serve` protocol: one `GEN <n> temperature=0 <prompt ids>` line per prompt, `--max-context 8192 --kv k8v4`, timings from the `DONE` line, the card cooled to 72 C or less before each prompt; raw log `up-0223-v100.log`.
- **This branch:** `strata-ink2`, official checkpoint (NVFP4 and MXFP8 experts, 16.7 GiB experts), `INK2_QPN=1 INK2_DQ8=1`, DSpark and DFlash 2 drafters (`INK2_DRAFTER`), logs `xgpu-v100-on.log`.
- The two sides run different model files (different quantisations of the same network), so their token streams and generated lengths differ (for example code700: 274 against 300 tokens). The comparison is speed on the same prompt text, not output quality, and quality was not compared.
- The upstream run was at 72-75 C with the engine loaded (the engine holds the card at P0); clocks were not logged during that run.

| prompt | upstream plain | upstream DFlash 2 | ours DSpark | ours DFlash 2 |
|---|---|---|---|---|
| code | 55.3 | 88.6 | 136.0 | 106.0 |
| chat-code | 59.1 | 147.0 | 225.9 | 232.6 |
| chat-qa | 59.5 | 74.1 | 85.8 | 100.9 |
| code700 | 56.8 | 113.8 | 91.7 | 167.6 |
| segA | 55.9 | 88.4 | 84.5 | 141.2 |
| 00-py | 56.7 | 89.9 | 105.8 | 104.7 |
| 02-py | 56.7 | 110.1 | 138.9 | 161.4 |
| 04-cuda | 56.8 | 84.2 | 113.7 | 124.7 |
| 06-cuda | 56.8 | 81.9 | 103.5 | 87.7 |
| 08-md | 56.9 | 79.3 | 113.8 | 126.1 |
| 10-md | 56.7 | 85.5 | 81.5 | 92.6 |
| 13-cfg | 56.7 | 115.6 | 160.1 | 133.6 |
| 15-cfg | 56.7 | 126.2 | 201.0 | 222.8 |
| **mean** | 57.0 | 98.8 | 126.3 | 138.6 |

Plain decode (no drafter) on this branch: 86.2 tok/s on the same card with the same settings (`xgpu-v100-on.log`), 1.51x upstream's 57.0.
