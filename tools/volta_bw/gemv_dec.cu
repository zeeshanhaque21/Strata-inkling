// NVFP4 expert GEMV, decode variants.  The previous spike (gemv_steps) showed the kernel is issue-bound: with the weights L2-resident it still takes 33 us (DRAM-fed 43).
// Variants decode the same e2m1 values exactly and are checked against variant 0 (the engine's half2 + F2F path) on random data.
// nvcc -arch=sm_70 -ccbin g++-14 -isystem <nvcc-host> -O3 gemv_dec.cu -o gemv_dec
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
__device__ __forceinline__ float2 e2m1_k4(uint32_t t) { const uint32_t h = ((t & 0x00080008u) << 12) | ((t & 0x00070007u) << 9); return __half22float2(*reinterpret_cast<const __half2*>(&h)); }
// float with value * 2^-126: bits (e1 e0 m) of a nibble at float bits 24..22, sign at 31
__device__ __forceinline__ float nib_f(uint32_t t, int j) { return __uint_as_float((((t << (22 - 4 * j + 0)) & 0x01C00000u) | ((t << (28 - 4 * j)) & 0x80000000u))); }
// DEC 0: engine (half2 trick + 2 F2F per pair)       -> q * 2^-14 ; scale table includes 2^14
// DEC 1: float-direct (shift/and/or) + FMUL 2^126    -> exact value
// DEC 2: PRMT: 8-entry float tables in registers, two PRMT per 4 weights give the high 16 bits of 4 floats, sign OR-ed in
// DEC 3: dummy (1 op per weight, wrong values): the floor if decode were free
template <int DEC>
__global__ void k_dec(const uint8_t* __restrict__ base, size_t expert_bytes, const float* __restrict__ x, int rows, int K, float* __restrict__ out) {
    constexpr int R = 2;
    const int row0 = (blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32) * R, lane = threadIdx.x & 31;
    if (row0 >= rows) return;
    const uint8_t* W = base + blockIdx.y * expert_bytes;
    const uint8_t* Sc = W + (size_t) rows * (K / 2);
    const uint32_t* w[R]; const uint8_t* s[R];
    for (int r = 0; r < R; ++r) { const size_t row = min(row0 + r, rows - 1); w[r] = reinterpret_cast<const uint32_t*>(W + row * (K / 2)); s[r] = Sc + row * (K / 16); }
    const float4* x4 = reinterpret_cast<const float4*>(x + blockIdx.y * K);
    float acc[R] = {0.f, 0.f};
    // PRMT tables: high byte / second byte of the float bits of {0, .5, 1, 1.5, 2, 3, 4, 6}
    const uint32_t H0 = 0x3F3F3F00u, H1 = 0x40404040u, M0 = 0xC0800000u, M1 = 0xC0804000u;
#pragma unroll 2
    for (int i = lane; i < K / 8; i += 32) {
        uint32_t v[R]; uint8_t sc[R];
#pragma unroll
        for (int r = 0; r < R; ++r) { v[r] = __ldg(w[r] + i); sc[r] = __ldg(s[r] + (i >> 1)); }
        const float4 a = __ldg(x4 + 2 * i), b = __ldg(x4 + 2 * i + 1);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint32_t t = v[r];
            // nibble j of t multiplies activation element j (a.x a.y a.z a.w b.x b.y b.z b.w)
            float q[8];
            if (DEC == 0) {
                const float2 q0 = e2m1_k4(t), q1 = e2m1_k4(t >> 4), q2 = e2m1_k4(t >> 8), q3 = e2m1_k4(t >> 12);   // q0 = nibbles 0, 4; q1 = 1, 5; ...
                q[0] = q0.x; q[4] = q0.y; q[1] = q1.x; q[5] = q1.y; q[2] = q2.x; q[6] = q2.y; q[3] = q3.x; q[7] = q3.y;
            } else if (DEC == 1) {
#pragma unroll
                for (int j = 0; j < 8; ++j) q[j] = nib_f(t, j) * 8.50705917e37f;   // 2^126: exact
            } else if (DEC == 2) {
                const uint32_t sel0 = t & 0x7777u, sel1 = (t >> 16) & 0x7777u;
                const uint32_t sg0 = t & 0x8888u, sg1 = (t >> 16) & 0x8888u;
#pragma unroll
                for (int g = 0; g < 2; ++g) {
                    const uint32_t sel = g ? sel1 : sel0, sg = g ? sg1 : sg0;
                    const uint32_t hi = __byte_perm(H0, H1, sel), mid = __byte_perm(M0, M1, sel);   // bytes j: weight j
                    const uint32_t p01 = __byte_perm(mid, hi, 0x5140u), p23 = __byte_perm(mid, hi, 0x7362u);   // bf16 pairs {w0,w1}, {w2,w3}
                    q[4 * g + 0] = __uint_as_float((p01 << 16) | ((sg & 0x8u) << 28));
                    q[4 * g + 1] = __uint_as_float((p01 & 0xFFFF0000u) | ((sg & 0x80u) << 24));
                    q[4 * g + 2] = __uint_as_float((p23 << 16) | ((sg & 0x800u) << 20));
                    q[4 * g + 3] = __uint_as_float((p23 & 0xFFFF0000u) | ((sg & 0x8000u) << 16));
                }
            } else {
#pragma unroll
                for (int j = 0; j < 8; ++j) q[j] = __uint_as_float(t >> (4 * j));
            }
            const float f = (DEC == 0) ? 16384.0f : 1.0f;
            acc[r] += (q[0] * a.x + q[1] * a.y + q[2] * a.z + q[3] * a.w + q[4] * b.x + q[5] * b.y + q[6] * b.z + q[7] * b.w) * (f * (1.0f + sc[r] * 1e-2f));
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) { float t = acc[r]; for (int o = 16; o; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o); if (!lane) out[blockIdx.y * rows + row0 + r] = t; }
}
template <int DEC>
float time_it(uint8_t* buf, size_t eb, float* x, int rows, int K, float* out, int nbuf, bool resident) {
    const dim3 grid((rows + 15) / 16, 8);
    for (int w = 0; w < 6; ++w) k_dec<DEC><<<grid, 256>>>(buf + (size_t) (w % nbuf) * 8 * eb, resident ? 0 : eb, x, rows, K, out);
    cudaDeviceSynchronize(); cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); const int it = 60; cudaEventRecord(a);
    for (int i = 0; i < it; ++i) k_dec<DEC><<<grid, 256>>>(buf + (size_t) (i % nbuf) * 8 * eb, resident ? 0 : eb, x, rows, K, out);
    cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); return ms * 1000.0f / it;
}
int main() {
    const int rows = 2304, K = 2304; const size_t eb = (size_t) rows * (K / 2 + K / 16); const int nbuf = 4;
    uint8_t* buf; float *x, *out; cudaMalloc(&buf, eb * 8 * nbuf); cudaMalloc(&x, 8 * K * 4); cudaMalloc(&out, 2 * 8 * rows * 4);
    std::vector<uint8_t> hb(eb * 8 * nbuf); for (auto& c : hb) c = (uint8_t) rand(); cudaMemcpy(buf, hb.data(), hb.size(), cudaMemcpyHostToDevice);
    std::vector<float> hx(8 * K); for (auto& f : hx) f = (rand() / (float) RAND_MAX) - 0.5f; cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice);
    const dim3 grid((rows + 15) / 16, 8);
    std::vector<float> o0(8 * rows), o1(8 * rows);
    k_dec<0><<<grid, 256>>>(buf, eb, x, rows, K, out); cudaMemcpy(o0.data(), out, 8 * rows * 4, cudaMemcpyDeviceToHost);
    for (int d = 1; d <= 2; ++d) {
        if (d == 1) k_dec<1><<<grid, 256>>>(buf, eb, x, rows, K, out); else k_dec<2><<<grid, 256>>>(buf, eb, x, rows, K, out);
        cudaMemcpy(o1.data(), out, 8 * rows * 4, cudaMemcpyDeviceToHost);
        int bad = 0; for (int i = 0; i < 8 * rows; ++i) bad += (o0[i] != o1[i]);
        printf("DEC %d vs DEC 0: %d of %d outputs differ bitwise (sample %g vs %g)\n", d, bad, 8 * rows, o0[7], o1[7]);
    }
    printf("gate_up shape %d x %d, 8 experts, %.1f MB per launch\n", rows, K, 8.0 * eb / 1e6);
    printf("%-34s %8s %12s\n", "decode", "DRAM-fed", "L2-resident");
    printf("%-34s %6.1f us %9.1f us\n", "0 engine (half2 + F2F)", time_it<0>(buf, eb, x, rows, K, out, nbuf, false), time_it<0>(buf, eb, x, rows, K, out, nbuf, true));
    printf("%-34s %6.1f us %9.1f us\n", "1 float-direct + FMUL", time_it<1>(buf, eb, x, rows, K, out, nbuf, false), time_it<1>(buf, eb, x, rows, K, out, nbuf, true));
    printf("%-34s %6.1f us %9.1f us\n", "2 PRMT tables", time_it<2>(buf, eb, x, rows, K, out, nbuf, false), time_it<2>(buf, eb, x, rows, K, out, nbuf, true));
    printf("%-34s %6.1f us %9.1f us\n", "3 dummy (free decode, wrong values)", time_it<3>(buf, eb, x, rows, K, out, nbuf, false), time_it<3>(buf, eb, x, rows, K, out, nbuf, true));
    return 0;
}
