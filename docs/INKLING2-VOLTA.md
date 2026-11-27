# strata-ink2 on Volta (Tesla V100, sm_70)

Status: works, measured on one Tesla V100-PCIE-32GB (driver 580.196.07, CachyOS).
Not part of upstream Strata; the CMake guard `STRATA_EXPERIMENTAL_SM60=ON` is needed.

## Build
- CUDA 13 cannot target sm_70. Use CUDA 12.9 (nvcc, cudart, cuBLAS, cupti: the redistributable archives from NVIDIA are enough).
- CUDA 12.9's front end cannot read GCC 16's libstdc++ or glibc 2.43's `cospi` / `sinpi` / `rsqrt` declarations.
  Use `g++-14` and the shim header from volta-forever (`scripts/nvcc-host/bits/mathcalls.h`) on the include path (`-isystem`).
- `cmake -S engine -B build-v70 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=<cuda-12.9>/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER=g++-14 -DCUDAToolkit_ROOT=<cuda-12.9> -DCMAKE_CUDA_ARCHITECTURES="70;75" -DCMAKE_CUDA_FLAGS="-isystem <shim dir> -Wno-deprecated-gpu-targets" -DSTRATA_EXPERIMENTAL_SM60=ON -DSTRATA_BUILD_INKLING2=ON`
- The driver must be a branch that supports Volta (580.x; the open kernel modules do not).

## What the engine does on Volta
- `mma.sync.m16n8k8` (Turing) does not exist on Volta.  `ink2_volta_mma.cuh` runs it as `mma.m8n8k4` with lane shuffles; a chain of products keeps its accumulator in Volta's layout and converts once (`MAcc`).
  `tools/volta_mma/volta_mma_test.cu` checks it against a CPU reference (chains of 1 to 16 products: 0 wrong, relative error about 1e-5).
- Measured on the V100 (4 sequences x 128 steps, batched decode): the dense int8 tensor-core GEMV gains 22 %, the tensor-core MoE kernels lose 27 % (and 24 % in plain decode):
  a chain of 8 products pays six shuffles each.  So on sm_70 the MoE kernels stay on CUDA cores by default; `INK2_MMA_MOE=1` and `INK2_MMA_PLAIN=1` turn them on.
- Batched equals alone for every sequence (`--ms-test`), which the same sm_75 binary run through volta-forever does not give.

## Numbers (dspark on, one stream, int8 KV, `INK2_Q8=15`, 250 W, extra case airflow; the card overheats at 80 C within about a minute of 64K prefill)
| | V100 volta-forever | V100 native | RTX 2080 Ti |
| --- | --- | --- | --- |
| 8K decode / prefill | 67.8 / 2,012 | 69.0 / 1,978 | 73.8 / 1,720 |
| 32K decode / prefill | 58.1 / 1,873 | 62.0 / 1,845 | 66.4 / 1,614 |

## Open
- The tensor-core MoE path would win if its kernels loaded weights and activations in Volta's own fragment layout instead of shuffling Turing's into it.
- The fat sm_70+sm_75 binary has not been run on a 2080 Ti.
