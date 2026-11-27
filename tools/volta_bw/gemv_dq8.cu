// Dense int8 GEMV for Volta, QPN-style (design: v100-skinny, MIT, Dennis Vertzyas; see gemv_qpn.cu): y[m][n] = sum_k W[n][k] * s[n][k / 32] * x[m][k], M <= 8 rows.
// W int8 [N][K] with an fp16 scale per 32 columns (the engine's DMat q / qs layout).  Weights on the MMA N side (quad-pair p of a warp owns 8 columns of a 32-row tile), one shared fp16
// activation fragment (rows 0..M-1 of x), weights repacked into fragment order [tile][16-k group][lane] x 16 B and scales [tile][32-k group][lane] x 2 B, int8 -> fp16 exactly by a
// bias trick (0x6400 | (q + 128) - 1152), the block scale folded in with one HMUL2 (the product is rounded to fp16: 5e-4 relative per weight, below the int8 quantization itself),
// K split over SPLITK warps, one cross-warp reduce.  Compared here against the ideal read time and checked against an fp64 reference.
// nvcc -arch=sm_70 -ccbin g++-14 -isystem <nvcc-host> -O3 gemv_dq8.cu -o gemv_dq8
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#define MMA_8N8K4(c, a0, a1, b0, b1) asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n" \
    : "+f"((c)[0]), "+f"((c)[1]), "+f"((c)[2]), "+f"((c)[3]), "+f"((c)[4]), "+f"((c)[5]), "+f"((c)[6]), "+f"((c)[7]) : "r"(a0), "r"(a1), "r"(b0), "r"(b1))
// 4 int8 (little endian bytes of w) -> two half2 {q0, q1}, {q2, q3}, times the scale
__device__ __forceinline__ void dq4(unsigned w, half2 sc, unsigned& o0, unsigned& o1) {
    const unsigned x = w ^ 0x80808080u;   // q + 128 as unsigned bytes
    const unsigned h0 = __byte_perm(x, 0x64646464u, 0x4140), h1 = __byte_perm(x, 0x64646464u, 0x4342);   // half 0x6400 | byte = 1024 + byte
    const half2 bias = __float2half2_rn(1152.0f);
    const half2 a = __hmul2(__hsub2(*reinterpret_cast<const half2*>(&h0), bias), sc), b = __hmul2(__hsub2(*reinterpret_cast<const half2*>(&h1), bias), sc);
    o0 = *reinterpret_cast<const unsigned*>(&a); o1 = *reinterpret_cast<const unsigned*>(&b);
}
// x16 [M][ldx] fp16; y [M][ldy] fp32; grid N / 32, block 32 * SPLITK
template <int SPLITK>
__global__ void k_dq8(const uint8_t* __restrict__ codes, const __half* __restrict__ scales, const __half* __restrict__ x, int ldx, float* __restrict__ y, int ldy, int N, int K, int M) {
    __shared__ float cs[SPLITK][256];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, tile = blockIdx.x;
    const int qp = (lane >> 2) & 3, r = (lane & 3) + ((lane & 16) ? 4 : 0), G = K >> 4, Gq = G / SPLITK, g0 = warp * Gq;
    const uint4* cb = reinterpret_cast<const uint4*>(codes) + (size_t) tile * G * 32 + lane;
    const __half* sb = scales + (size_t) tile * (G / 2) * 32 + lane;
    const __half* xr = x + (size_t) (r < M ? r : 0) * ldx;
    float c[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) c[i] = 0.0f;
#pragma unroll 2
    for (int g = g0; g < g0 + Gq; g += 2) {   // a pair of 16-k groups shares one scale
        const half2 sc = __half2half2(__ldg(sb + (size_t) (g >> 1) * 32));
        const uint4 w0 = __ldcs(cb + (size_t) g * 32), w1 = __ldcs(cb + (size_t) (g + 1) * 32);
        uint4 a0 = make_uint4(0, 0, 0, 0), a1 = a0, a2 = a0, a3 = a0;
        if (r < M) {
            const uint4* xp = reinterpret_cast<const uint4*>(xr + g * 16);
            a0 = xp[0]; a1 = xp[1]; a2 = xp[2]; a3 = xp[3];
        }
        unsigned b[8];
        dq4(w0.x, sc, b[0], b[1]); dq4(w0.y, sc, b[2], b[3]); dq4(w0.z, sc, b[4], b[5]); dq4(w0.w, sc, b[6], b[7]);
        MMA_8N8K4(c, a0.x, a0.y, b[0], b[1]); MMA_8N8K4(c, a0.z, a0.w, b[2], b[3]); MMA_8N8K4(c, a1.x, a1.y, b[4], b[5]); MMA_8N8K4(c, a1.z, a1.w, b[6], b[7]);
        dq4(w1.x, sc, b[0], b[1]); dq4(w1.y, sc, b[2], b[3]); dq4(w1.z, sc, b[4], b[5]); dq4(w1.w, sc, b[6], b[7]);
        MMA_8N8K4(c, a2.x, a2.y, b[0], b[1]); MMA_8N8K4(c, a2.z, a2.w, b[2], b[3]); MMA_8N8K4(c, a3.x, a3.y, b[4], b[5]); MMA_8N8K4(c, a3.z, a3.w, b[6], b[7]);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1), col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
        cs[warp][row * 32 + qp * 8 + col] = c[i];
    }
    __syncthreads();
    for (int t = threadIdx.x; t < 256; t += blockDim.x) {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < SPLITK; ++w) v += cs[w][t];
        const int row = t >> 5, col = t & 31;
        if (row < M) y[(size_t) row * ldy + (size_t) tile * 32 + col] = v;
    }
}
// repack (host): W [N][K] int8, S [N][K/32] fp16 bits -> fragment order
static void prepack(const int8_t* W, const uint16_t* S, uint8_t* qc, uint16_t* qs, int N, int K) {
    const int G = K / 16;
    for (int t = 0; t < N / 32; ++t) for (int lane = 0; lane < 32; ++lane) {
        const int row = t * 32 + ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
        for (int g = 0; g < G; ++g) memcpy(qc + (((size_t) t * G + g) * 32 + lane) * 16, W + (size_t) row * K + g * 16, 16);
        for (int g = 0; g < G / 2; ++g) qs[((size_t) t * (G / 2) + g) * 32 + lane] = S[(size_t) row * (K / 32) + g];
    }
}
template <typename F> float timeit(F f, int it = 100) { for (int w = 0; w < 6; ++w) f(w); cudaDeviceSynchronize(); cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); cudaEventRecord(a); for (int i = 0; i < it; ++i) f(i); cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); return ms * 1000.f / it; }
struct Shape { const char* name; int N, K; };
int main() {
    const Shape shapes[] = {{"proj (qkv + gates, N 8704?)", 8704, 2304}, {"wo", 2304, 3072}, {"lm head", 200064, 2304}, {"drafter gate_up (DSpark)", 18432, 2304}, {"drafter down", 2304, 9216}};
    printf("%-30s %3s %-12s %9s %8s %10s %s\n", "shape (N x K)", "M", "kernel", "time", "GB/s", "vs floor", "max rel err");
    for (const Shape& sh : shapes) {
        const int N = sh.N, K = sh.K, G = K / 16, nbuf = std::max(2, (int) (300e6 / ((double) N * K)) + 1);   // enough copies to defeat the 6 MB L2
        std::vector<int8_t> W((size_t) N * K); std::vector<uint16_t> S((size_t) N * (K / 32));
        for (auto& v : W) v = (int8_t) (rand() % 255 - 127);
        for (auto& v : S) { const float f = (0.5f + (rand() % 100) / 100.f) * 0.01f; __half h = __float2half(f); v = *reinterpret_cast<uint16_t*>(&h); }
        std::vector<uint8_t> qc((size_t) N * K); std::vector<uint16_t> qs((size_t) N * (K / 32));
        prepack(W.data(), S.data(), qc.data(), qs.data(), N, K);
        uint8_t* dc; __half* ds; __half* dx; float* dy;
        cudaMalloc(&dc, qc.size() * nbuf); cudaMalloc(&ds, qs.size() * 2 * nbuf); cudaMalloc(&dx, 8 * K * 2); cudaMalloc(&dy, (size_t) 8 * N * 4);
        for (int b = 0; b < nbuf; ++b) { cudaMemcpy(dc + (size_t) b * qc.size(), qc.data(), qc.size(), cudaMemcpyHostToDevice); cudaMemcpy(ds + (size_t) b * qs.size(), qs.data(), qs.size() * 2, cudaMemcpyHostToDevice); }
        std::vector<float> hx(8 * K); std::vector<__half> hx16(8 * K);
        for (int i = 0; i < 8 * K; ++i) { hx[i] = (rand() / (float) RAND_MAX) - 0.5f; hx16[i] = __float2half(hx[i]); }
        cudaMemcpy(dx, hx16.data(), hx16.size() * 2, cudaMemcpyHostToDevice);
        const double bytes = (double) N * K + (double) N * (K / 32) * 2;
        for (int M : {1, 8}) {
            auto run = [&](const char* tag, auto kern, int splitk) {
                if (G % (2 * splitk)) return;   // groups must split evenly in pairs
                auto launch2 = [&](int i) { kern<<<N / 32, 32 * splitk>>>(dc + (size_t) (i % nbuf) * qc.size(), ds + (size_t) (i % nbuf) * qs.size(), dx, K, dy, N, N, K, M); };   // ldx = K, ldy = N
                const float us = timeit(launch2);
                launch2(0); cudaDeviceSynchronize();
                std::vector<float> o((size_t) M * N); cudaMemcpy(o.data(), dy, o.size() * 4, cudaMemcpyDeviceToHost);
                double me = 0, mx = 0;
                for (int row = 0; row < N; row += 41) for (int m = 0; m < M; ++m) {
                    double a = 0; for (int k = 0; k < K; ++k) { const float s = __half2float(*reinterpret_cast<const __half*>(&S[(size_t) row * (K / 32) + k / 32])); a += (double) W[(size_t) row * K + k] * s * (double) __half2float(hx16[(size_t) m * K + k]); }
                    me = fmax(me, fabs(o[(size_t) m * N + row] - a)); mx = fmax(mx, fabs(a));
                }
                printf("%-30s %3d %-12s %7.1f us %7.0f %9.2fx   %.2e\n", sh.name, M, tag, us, bytes / (us * 1e-6) / 1e9, us / (bytes / 720e9 * 1e6), me / mx);
            };
            run("split 8", k_dq8<8>, 8); run("split 12", k_dq8<12>, 12); run("split 16", k_dq8<16>, 16); run("split 18", k_dq8<18>, 18); run("split 24", k_dq8<24>, 24); run("split 32", k_dq8<32>, 32);
        }
        cudaFree(dc); cudaFree(ds); cudaFree(dx); cudaFree(dy);
    }
    cudaError_t e = cudaGetLastError(); if (e) printf("cuda error: %s\n", cudaGetErrorString(e));
    return 0;
}
