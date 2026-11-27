// What turns a 720 GB/s weight stream into the real NVFP4 expert GEMV?  Adds the kernel's work back one piece at a time (S scale loads, X activation loads, C decode + FMA).
// nvcc -arch=sm_70 -ccbin g++-14 -isystem <nvcc-host> -O3 gemv_steps.cu -o gemv_steps
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
__device__ __forceinline__ float2 e2m1_k4(uint32_t t) { const uint32_t h = ((t & 0x00080008u) << 12) | ((t & 0x00070007u) << 9); return __half22float2(*reinterpret_cast<const __half2*>(&h)); }
template <bool S, bool X, bool C, int UNR>
__global__ void k_step(const uint8_t* __restrict__ base, size_t expert_bytes, const float* __restrict__ x, int rows, int K, float* __restrict__ out) {
    constexpr int R = 2;
    __shared__ float e4[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) e4[i] = 1.0f + i * 1e-3f;
    __syncthreads();
    const int row0 = (blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32) * R, lane = threadIdx.x & 31;
    if (row0 >= rows) return;
    const uint8_t* W = base + blockIdx.y * expert_bytes;
    const uint8_t* Sc = W + (size_t) rows * (K / 2);
    const uint32_t* w[R]; const uint8_t* s[R];
    for (int r = 0; r < R; ++r) { const size_t row = min(row0 + r, rows - 1); w[r] = reinterpret_cast<const uint32_t*>(W + row * (K / 2)); s[r] = Sc + row * (K / 16); }
    const float4* x4 = reinterpret_cast<const float4*>(x + blockIdx.y * K);
    float acc[R] = {0.f, 0.f};
#pragma unroll UNR
    for (int i = lane; i < K / 8; i += 32) {
        uint32_t v[R]; uint8_t sc[R] = {1, 1};
#pragma unroll
        for (int r = 0; r < R; ++r) { v[r] = __ldg(w[r] + i); if (S) sc[r] = __ldg(s[r] + (i >> 1)); }
        float4 a = make_float4(1, 1, 1, 1), b = a;
        if (X) { a = __ldg(x4 + 2 * i); b = __ldg(x4 + 2 * i + 1); }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (C) {
                const float2 q0 = e2m1_k4(v[r]), q1 = e2m1_k4(v[r] >> 4), q2 = e2m1_k4(v[r] >> 8), q3 = e2m1_k4(v[r] >> 12);
                acc[r] += (q0.x * a.x + q1.x * a.y + q2.x * a.z + q3.x * a.w + q0.y * b.x + q1.y * b.y + q2.y * b.z + q3.y * b.w) * e4[sc[r]];
            } else acc[r] += __uint_as_float(v[r]) * a.x + (float) sc[r] * b.x;
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) { float t = acc[r]; for (int o = 16; o; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o); if (!lane) out[blockIdx.y * rows + row0 + r] = t; }
}
// every weight word and scale of the lane loaded first (NT trips: 9 at K = 2304), then decoded in order
template <int NT>
__global__ void k_hoist(const uint8_t* __restrict__ base, size_t expert_bytes, const float* __restrict__ x, int rows, int K, float* __restrict__ out) {
    constexpr int R = 2;
    __shared__ float e4[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) e4[i] = 1.0f + i * 1e-3f;
    __syncthreads();
    const int row0 = (blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32) * R, lane = threadIdx.x & 31;
    if (row0 >= rows) return;
    const uint8_t* W = base + blockIdx.y * expert_bytes;
    const uint8_t* Sc = W + (size_t) rows * (K / 2);
    const uint32_t* w[R]; const uint8_t* s[R];
    for (int r = 0; r < R; ++r) { const size_t row = min(row0 + r, rows - 1); w[r] = reinterpret_cast<const uint32_t*>(W + row * (K / 2)); s[r] = Sc + row * (K / 16); }
    const float4* x4 = reinterpret_cast<const float4*>(x + blockIdx.y * K);
    uint32_t v[NT][R]; uint8_t sc[NT][R];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        const int i = lane + 32 * j;
#pragma unroll
        for (int r = 0; r < R; ++r) { v[j][r] = 0; sc[j][r] = 0; if (i < K / 8) { v[j][r] = __ldg(w[r] + i); sc[j][r] = __ldg(s[r] + (i >> 1)); } }
    }
    float acc[R] = {0.f, 0.f};
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        const int i = lane + 32 * j;
        if (i >= K / 8) break;
        const float4 a = __ldg(x4 + 2 * i), b = __ldg(x4 + 2 * i + 1);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float2 q0 = e2m1_k4(v[j][r]), q1 = e2m1_k4(v[j][r] >> 4), q2 = e2m1_k4(v[j][r] >> 8), q3 = e2m1_k4(v[j][r] >> 12);
            acc[r] += (q0.x * a.x + q1.x * a.y + q2.x * a.z + q3.x * a.w + q0.y * b.x + q1.y * b.y + q2.y * b.z + q3.y * b.w) * e4[sc[j][r]];
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) { float t = acc[r]; for (int o = 16; o; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o); if (!lane) out[blockIdx.y * rows + row0 + r] = t; }
}
template <int NT>
void run_hoist(const char* tag, uint8_t* buf, size_t eb, float* x, int rows, int K, float* out, int nbuf) {
    const dim3 grid((rows + 15) / 16, 8);
    for (int w = 0; w < 6; ++w) k_hoist<NT><<<grid, 256>>>(buf + (size_t) (w % nbuf) * 8 * eb, eb, x, rows, K, out);
    cudaDeviceSynchronize(); cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); const int it = 60; cudaEventRecord(a);
    for (int i = 0; i < it; ++i) k_hoist<NT><<<grid, 256>>>(buf + (size_t) (i % nbuf) * 8 * eb, eb, x, rows, K, out);
    cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b);
    const double us = ms * 1000.0 / it, bytes = 8.0 * rows * (K / 2 + K / 16);
    printf("%-34s %6.1f us  %6.0f GB/s\n", tag, us, bytes / (us * 1e-6) / 1e9);
}
template <bool S, bool X, bool C, int UNR>
void run(const char* tag, uint8_t* buf, size_t eb, float* x, int rows, int K, float* out, int nbuf) {
    const dim3 grid((rows + 15) / 16, 8);
    for (int w = 0; w < 6; ++w) k_step<S, X, C, UNR><<<grid, 256>>>(buf + (size_t) (w % nbuf) * 8 * eb, eb, x, rows, K, out);
    cudaDeviceSynchronize(); cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); const int it = 60; cudaEventRecord(a);
    for (int i = 0; i < it; ++i) k_step<S, X, C, UNR><<<grid, 256>>>(buf + (size_t) (i % nbuf) * 8 * eb, eb, x, rows, K, out);
    cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b);
    const double us = ms * 1000.0 / it, bytes = 8.0 * rows * (K / 2 + K / 16);
    printf("%-34s %6.1f us  %6.0f GB/s\n", tag, us, bytes / (us * 1e-6) / 1e9);
}
int main() {
    const int rows = 2304, K = 2304; const size_t eb = (size_t) rows * (K / 2 + K / 16); const int nbuf = 4;
    uint8_t* buf; float *x, *out; cudaMalloc(&buf, eb * 8 * nbuf); cudaMalloc(&x, 8 * K * 4); cudaMalloc(&out, 8 * rows * 4);
    cudaMemset(buf, 0x31, eb * 8 * nbuf); cudaMemset(x, 0, 8 * K * 4);
    printf("expert GEMV pieces, gate_up shape (%d rows x K %d), 8 experts, %.1f MB per launch\n", rows, K, 8.0 * eb / 1e6);
    run<false, false, false, 2>("weights only", buf, eb, x, rows, K, out, nbuf);
    run<true, false, false, 2>("+ scale loads", buf, eb, x, rows, K, out, nbuf);
    run<true, true, false, 2>("+ activation loads", buf, eb, x, rows, K, out, nbuf);
    run<true, true, true, 2>("+ decode and FMA (the real kernel)", buf, eb, x, rows, K, out, nbuf);
    run<true, true, true, 3>("  same, unroll 3", buf, eb, x, rows, K, out, nbuf);
    run<false, true, true, 2>("  no scale loads (decode + x)", buf, eb, x, rows, K, out, nbuf);
    run<true, false, true, 2>("  no activation loads", buf, eb, x, rows, K, out, nbuf);
    run_hoist<9>("all loads hoisted (9 trips)", buf, eb, x, rows, K, out, nbuf);
    { const int K2 = 1152; const int rows2 = 2304; const size_t eb2 = (size_t) rows2 * (K2 / 2 + K2 / 16);
      printf("down shape (%d rows x K %d)\n", rows2, K2);
      run<true, true, true, 2>("  current (unroll 2)", buf, eb2, x, rows2, K2, out, nbuf);
      run_hoist<5>("  all loads hoisted (5 trips)", buf, eb2, x, rows2, K2, out, nbuf); }
    return 0;
}
