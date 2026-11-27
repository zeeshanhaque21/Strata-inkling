// QPN2-style NVFP4 expert GEMV for Volta, adapted for Strata.
// The design (weights on the N side, all four quad-pairs sharing one activation fragment, fragment-order prepacked weights, nibble-interleaved so the decoder's
// half2 output is the B fragment, no shared memory in the main loop, split-K over warps) is from v100-skinny by Dennis Vertzyas (MIT, github.com/dnv2003/v100-skinny,
// kernels/skinny_kernels.cu, skinny_nvfp4_qpn2 and dequant8_tm; decoder derived from TurboMind, Apache-2.0).  The kernel body below is that design re-expressed for an
// expert-indirected launch (blockIdx.y = expert); the prepack is the same permutation.  Checked here against a double-precision reference.
// nvcc -arch=sm_70 -ccbin g++-14 -isystem <nvcc-host> -O3 gemv_qpn.cu -o gemv_qpn
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
__device__ __forceinline__ half2 fp8e4m3_to_half2(unsigned char b) {
    const unsigned short hb = (((unsigned short) b & 0x80u) << 8) | (((unsigned short) b & 0x7Fu) << 7);
    const half hs = __hmul(__ushort_as_half(hb), __ushort_as_half(0x5C00));   // * 256
    return __halves2half2(hs, hs);
}
// 8 nibbles -> four half2 {code i, code i+4}, scaled; values carry 2^-14 (folded into the scale by the caller)
__device__ __forceinline__ void dequant8_tm(unsigned q, half2 sc2p, half2 out[4]) {
    constexpr unsigned S = 0x80008000u, EM = 0x0E000E00u;
    unsigned v0 = ((q << 12) & S) | ((q << 9) & EM), v1 = ((q << 8) & S) | ((q << 5) & EM), v2 = ((q << 4) & S) | ((q << 1) & EM), v3 = (q & S) | ((q >> 3) & EM);
    out[0] = __hmul2(*reinterpret_cast<half2*>(&v0), sc2p); out[1] = __hmul2(*reinterpret_cast<half2*>(&v1), sc2p);
    out[2] = __hmul2(*reinterpret_cast<half2*>(&v2), sc2p); out[3] = __hmul2(*reinterpret_cast<half2*>(&v3), sc2p);
}
// grid (N/32 tiles, experts); block 32*SPLITK; x: [M][K] fp16 shared by all experts (gate_up); y: [expert][M][N] fp32
template <int SPLITK, int NACC>
__global__ void k_qpn2(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales, size_t cbytes, size_t sbytes, const half* __restrict__ x, float* __restrict__ y, int N, int K, int M) {
    __shared__ float cs[SPLITK > 1 ? SPLITK : 1][SPLITK > 1 ? 256 : 1];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, tile = blockIdx.x, e = blockIdx.y;
    const int qp = (lane >> 2) & 3, r = (lane & 3) + ((lane & 16) ? 4 : 0), G = K >> 4, Gq = G / SPLITK, g0 = warp * Gq;
    const uint2* cb = reinterpret_cast<const uint2*>(codes + e * cbytes) + (size_t) tile * G * 32 + lane;
    const uint8_t* sb = scales + e * sbytes + (size_t) tile * G * 32 + lane;
    const half2 gm2 = __float2half2_rn(16384.f);
    float c[NACC][8];
#pragma unroll
    for (int a = 0; a < NACC; ++a)
#pragma unroll
        for (int i = 0; i < 8; ++i) c[a][i] = 0.f;
#pragma unroll 4
    for (int g = g0; g < g0 + Gq; ++g) {
        const uint2 q2 = __ldcs(cb + (size_t) g * 32);
        const half2 sc2 = __hmul2(fp8e4m3_to_half2(__ldg(sb + (size_t) g * 32)), gm2);
        half2 b[8];
        dequant8_tm(q2.x, sc2, b + 0); dequant8_tm(q2.y, sc2, b + 4);
        const unsigned* B = reinterpret_cast<const unsigned*>(b);
        uint4 a01 = make_uint4(0, 0, 0, 0), a23 = make_uint4(0, 0, 0, 0);
        if (r < M) { const half* xrow = x + (size_t) r * K; a01 = *reinterpret_cast<const uint4*>(xrow + g * 16); a23 = *reinterpret_cast<const uint4*>(xrow + g * 16 + 8); }
        const unsigned *A0 = reinterpret_cast<const unsigned*>(&a01), *A1 = reinterpret_cast<const unsigned*>(&a23);
        MMA_8N8K4(c[0], A0[0], A0[1], B[0], B[1]);
        MMA_8N8K4(c[1 % NACC], A0[2], A0[3], B[2], B[3]);
        MMA_8N8K4(c[2 % NACC], A1[0], A1[1], B[4], B[5]);
        MMA_8N8K4(c[3 % NACC], A1[2], A1[3], B[6], B[7]);
    }
#pragma unroll
    for (int a = 1; a < NACC; ++a)
#pragma unroll
        for (int i = 0; i < 8; ++i) c[0][i] += c[a][i];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1), col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
        cs[warp][row * 32 + qp * 8 + col] = c[0][i];
    }
    __syncthreads();
    for (int t = threadIdx.x; t < 256; t += blockDim.x) {
        float v = 0.f;
#pragma unroll
        for (int w = 0; w < SPLITK; ++w) v += cs[w][t];
        const int row = t >> 5, col = t & 31;
        if (row < M) y[((size_t) e * M + row) * N + (size_t) tile * 32 + col] = v;
    }
}
// prepack of one matrix [N][K/2] bytes (low nibble = even k) + scales [N][K/16]: fragment order [tile][group][lane] x 8 B / 1 B
static void prepack(const uint8_t* codes, const uint8_t* scales, uint8_t* qc, uint8_t* qs, int N, int K) {
    const int G = K / 16, korder[16] = {0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15};
    for (int t = 0; t < N / 32; ++t) for (int g = 0; g < G; ++g) for (int lane = 0; lane < 32; ++lane) {
        const int col = t * 32 + ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
        int nb[16]; for (int j = 0; j < 16; ++j) { const int k = g * 16 + korder[j]; const uint8_t by = codes[(size_t) col * (K / 2) + k / 2]; nb[j] = (k & 1) ? by >> 4 : by & 15; }
        for (int b = 0; b < 8; ++b) qc[(((size_t) t * G + g) * 32 + lane) * 8 + b] = (uint8_t) (nb[2 * b] | (nb[2 * b + 1] << 4));
        qs[((size_t) t * G + g) * 32 + lane] = scales[(size_t) col * G + g];
    }
}
static float e4m3(int b) { const int e = (b >> 3) & 15, m = b & 7; return (e ? ldexpf(1.0f + m / 8.0f, e - 7) : ldexpf(m / 8.0f, -6)) * ((b & 128) ? -1.f : 1.f); }

// same kernel, expert chosen through a pointer table (as the engine does): codes at tab[kb], scales after the codes
template <int SPLITK, int NACC>
__global__ void k_qpn2_tab(const uint8_t* const* __restrict__ tab, const half* __restrict__ x, float* __restrict__ y, int N, int K, int M) {
    __shared__ float cs[SPLITK][256];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, tile = blockIdx.x, e = blockIdx.y;
    const uint8_t* W = tab[e];
    const int qp = (lane >> 2) & 3, r = (lane & 3) + ((lane & 16) ? 4 : 0), G = K >> 4, Gq = G / SPLITK, g0 = warp * Gq;
    const uint2* cb = reinterpret_cast<const uint2*>(W) + (size_t) tile * G * 32 + lane;
    const uint8_t* sb = W + (size_t) N * (K / 2) + (size_t) tile * G * 32 + lane;
    const half2 gm2 = __float2half2_rn(16384.f);
    float c[NACC][8];
#pragma unroll
    for (int a = 0; a < NACC; ++a)
#pragma unroll
        for (int i = 0; i < 8; ++i) c[a][i] = 0.f;
#pragma unroll 4
    for (int g = g0; g < g0 + Gq; ++g) {
        const uint2 q2 = __ldcs(cb + (size_t) g * 32);
        const half2 sc2 = __hmul2(fp8e4m3_to_half2(__ldg(sb + (size_t) g * 32)), gm2);
        half2 b[8];
        dequant8_tm(q2.x, sc2, b + 0); dequant8_tm(q2.y, sc2, b + 4);
        const unsigned* B = reinterpret_cast<const unsigned*>(b);
        uint4 a01 = make_uint4(0, 0, 0, 0), a23 = make_uint4(0, 0, 0, 0);
        if (r < M) { const half* xrow = x + (size_t) r * K; a01 = *reinterpret_cast<const uint4*>(xrow + g * 16); a23 = *reinterpret_cast<const uint4*>(xrow + g * 16 + 8); }
        const unsigned *A0 = reinterpret_cast<const unsigned*>(&a01), *A1 = reinterpret_cast<const unsigned*>(&a23);
        MMA_8N8K4(c[0], A0[0], A0[1], B[0], B[1]); MMA_8N8K4(c[1 % NACC], A0[2], A0[3], B[2], B[3]); MMA_8N8K4(c[2 % NACC], A1[0], A1[1], B[4], B[5]); MMA_8N8K4(c[3 % NACC], A1[2], A1[3], B[6], B[7]);
    }
#pragma unroll
    for (int a = 1; a < NACC; ++a)
#pragma unroll
        for (int i = 0; i < 8; ++i) c[0][i] += c[a][i];
#pragma unroll
    for (int i = 0; i < 8; ++i) { const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1), col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2); cs[warp][row * 32 + qp * 8 + col] = c[0][i]; }
    __syncthreads();
    for (int t = threadIdx.x; t < 256; t += blockDim.x) { float v = 0.f;
#pragma unroll
        for (int w = 0; w < SPLITK; ++w) v += cs[w][t];
        const int row = t >> 5, col = t & 31; if (row < M) y[((size_t) e * M + row) * N + (size_t) tile * 32 + col] = v; }
}

__device__ __forceinline__ float swiglu1(float g, float u) { return (tanhf(g / 7.0f) * 7.0f) * (1.0f / (1.0f + expf(-g))) * (tanhf(u / 7.0f) * 7.0f); }
__device__ __forceinline__ half2 qpn_e4m3_half2(unsigned char b) { return fp8e4m3_to_half2(b); }
__device__ __forceinline__ void qpn_dequant8(unsigned q, half2 sc2, half2 out[4]) { dequant8_tm(q, sc2, out); }
template <int SPLITK, bool GU>
#ifdef NOLB
__global__ void k_qpn_sel(
#else
__global__ void __launch_bounds__(32 * SPLITK) k_qpn_sel(
#endif
    const uint8_t* const* __restrict__ tab, const float* __restrict__ factors, const int* __restrict__ ids,
                                                         const __half* __restrict__ x, int x_stride, float* __restrict__ y, int N, int K, float* __restrict__ act,
                                                         __half* __restrict__ act16) {
    __shared__ float cs[SPLITK][32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, tile = blockIdx.x, kk = blockIdx.y;
    const int e = ids[kk];
    const uint8_t* W = tab[e];
    if (!W) return;
    const int qp = (lane >> 2) & 3, r = (lane & 3) + ((lane & 16) ? 4 : 0), G = K >> 4, Gq = G / SPLITK, g0 = warp * Gq;
    const uint2* cb = reinterpret_cast<const uint2*>(W) + (size_t) tile * G * 32 + lane;
    const uint8_t* sb = W + (size_t) N * (K / 2) + (size_t) tile * G * 32 + lane;
    const __half* xk = x + (size_t) kk * x_stride;
    #ifdef FIXG
    const half2 gm2 = __float2half2_rn(16384.0f);
#else
    const half2 gm2 = __float2half2_rn(factors[e] * 16384.0f);
#endif

    float c[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) c[i] = 0.0f;
#pragma unroll 4
    for (int g = g0; g < g0 + Gq; ++g) {
        const uint2 q2 = __ldcs(cb + (size_t) g * 32);
        const half2 sc2 = __hmul2(qpn_e4m3_half2(__ldg(sb + (size_t) g * 32)), gm2);
        half2 b[8];
        qpn_dequant8(q2.x, sc2, b + 0);
        qpn_dequant8(q2.y, sc2, b + 4);
        const unsigned* B = reinterpret_cast<const unsigned*>(b);
        uint4 a01 = make_uint4(0, 0, 0, 0), a23 = make_uint4(0, 0, 0, 0);
        if (r == 0) {   // one token: only matrix row 0 of the A fragment is real
            #ifdef XPLAIN
            a01 = *reinterpret_cast<const uint4*>(xk + g * 16);
            a23 = *reinterpret_cast<const uint4*>(xk + g * 16 + 8);
#else
            a01 = __ldg(reinterpret_cast<const uint4*>(xk + g * 16));
            a23 = __ldg(reinterpret_cast<const uint4*>(xk + g * 16 + 8));
#endif
        }
        asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7]) : "r"(a01.x), "r"(a01.y), "r"(B[0]), "r"(B[1]));
        asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7]) : "r"(a01.z), "r"(a01.w), "r"(B[2]), "r"(B[3]));
        asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7]) : "r"(a23.x), "r"(a23.y), "r"(B[4]), "r"(B[5]));
        asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7]) : "r"(a23.z), "r"(a23.w), "r"(B[6]), "r"(B[7]));
    }
    // accumulator i of this lane: matrix row (i & 2) | (lane & 16 ? 4 : 0) | (lane & 1), column (i & 1) | ((lane >> 1 & 1) << 1) | ((i >> 2) << 2) of quad-pair qp's 8 columns
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
        if (row == 0) cs[warp][qp * 8 + ((i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2))] = c[i];
    }
    __syncthreads();
    if (warp == 0) {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < SPLITK; ++w) v += cs[w][lane];
        const int col = tile * 32 + lane;
        y[(size_t) kk * N + col] = v;
        if (GU) {
            const float u = __shfl_down_sync(0xffffffffu, v, 1);
            if (!(lane & 1)) { const float a = swiglu1(v, u); act[(size_t) kk * (N / 2) + col / 2] = a; act16[(size_t) kk * (N / 2) + col / 2] = __float2half(a); }
        }
    }
}

template <typename F> float timeit(F f, int it = 80) { for (int w = 0; w < 6; ++w) f(w); cudaDeviceSynchronize(); cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); cudaEventRecord(a); for (int i = 0; i < it; ++i) f(i); cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); return ms * 1000.f / it; }
int main(int argc, char** argv) {
    const int N = 2304, K = argc > 1 ? atoi(argv[1]) : 2304, NE = 8, nbuf = 4, G = K / 16; const size_t cb = (size_t) N * K / 2, sb = (size_t) N * G;
    std::vector<uint8_t> hc(cb * NE * nbuf), hs(sb * NE * nbuf), pc(cb * NE * nbuf), ps(sb * NE * nbuf);
    for (auto& v : hc) v = (uint8_t) rand();
    for (auto& v : hs) v = (uint8_t) (((3 + rand() % 6) << 3) | (rand() & 7));
    for (int e = 0; e < NE * nbuf; ++e) prepack(hc.data() + e * cb, hs.data() + e * sb, pc.data() + e * cb, ps.data() + e * sb, N, K);
    uint8_t *dc, *ds; half* dx; float* dy; cudaMalloc(&dc, pc.size()); cudaMalloc(&ds, ps.size()); cudaMalloc(&dx, 8 * K * 2); cudaMalloc(&dy, (size_t) NE * 8 * N * 4);
    cudaMemcpy(dc, pc.data(), pc.size(), cudaMemcpyHostToDevice); cudaMemcpy(ds, ps.data(), ps.size(), cudaMemcpyHostToDevice);
    std::vector<float> hx(8 * K); for (auto& f : hx) f = (rand() / (float) RAND_MAX) - 0.5f;
    std::vector<half> hx16(8 * K); for (int i = 0; i < 8 * K; ++i) hx16[i] = __float2half(hx[i]); cudaMemcpy(dx, hx16.data(), hx16.size() * 2, cudaMemcpyHostToDevice);
    printf("gate_up %d x %d, %d experts, %.1f MB per launch\n%-3s %-28s %9s %9s %s\n", N, K, NE, NE * (cb + sb) / 1e6, "M", "kernel", "DRAM-fed", "GB/s", "max rel err");
    auto run = [&](const char* tag, int M, auto kern, int splitk) {
        const dim3 grid(N / 32, NE);
        auto launch = [&](int i) { kern<<<grid, 32 * splitk>>>(dc + (size_t) (i % nbuf) * NE * cb, ds + (size_t) (i % nbuf) * NE * sb, cb, sb, dx, dy, N, K, M); };
        const float us = timeit(launch);
        launch(0); cudaDeviceSynchronize(); std::vector<float> o((size_t) NE * M * N); cudaMemcpy(o.data(), dy, o.size() * 4, cudaMemcpyDeviceToHost);
        double me = 0, mx = 0;
        for (int e = 0; e < NE; e += 3) for (int row = 0; row < N; row += 37) for (int t = 0; t < M; ++t) {   // sampled reference (every 3rd expert, every 37th row)
            double a = 0; const uint8_t* W = hc.data() + e * cb; const uint8_t* S = hs.data() + e * sb;
            static const double mag[8] = {0, .5, 1, 1.5, 2, 3, 4, 6};
            for (int k = 0; k < K; ++k) { const uint8_t by = W[(size_t) row * (K / 2) + k / 2]; const int n = (k & 1) ? by >> 4 : by & 15; a += (n & 8 ? -1 : 1) * mag[n & 7] * e4m3(S[(size_t) row * G + k / 16]) * (double) __half2float(hx16[(size_t) t * K + k]); }
            const double g = o[((size_t) e * M + t) * N + row]; me = fmax(me, fabs(g - a)); mx = fmax(mx, fabs(a));
        }
        printf("%-3d %-28s %7.1f us %7.0f   %.2e\n", M, tag, us, NE * (cb + sb) / (us * 1e-6) / 1e9, me / mx);
    };
    if (K != 2304) for (int M : {1}) {
        run("qpn2 split 2, 1 acc", M, k_qpn2<2, 1>, 2); run("qpn2 split 3, 1 acc", M, k_qpn2<3, 1>, 3); run("qpn2 split 4, 1 acc", M, k_qpn2<4, 1>, 4); run("qpn2 split 6, 1 acc", M, k_qpn2<6, 1>, 6);
        run("qpn2 split 8, 1 acc", M, k_qpn2<8, 1>, 8); run("qpn2 split 9, 1 acc", M, k_qpn2<9, 1>, 9); run("qpn2 split 12, 1 acc", M, k_qpn2<12, 1>, 12); run("qpn2 split 18, 1 acc", M, k_qpn2<18, 1>, 18);
    } else for (int M : {1, 4}) {
        run("qpn2 split 2, 1 acc", M, k_qpn2<2, 1>, 2); run("qpn2 split 3, 1 acc", M, k_qpn2<3, 1>, 3); run("qpn2 split 4, 1 acc", M, k_qpn2<4, 1>, 4); run("qpn2 split 6, 1 acc", M, k_qpn2<6, 1>, 6);
        run("qpn2 split 8, 1 acc", M, k_qpn2<8, 1>, 8); run("qpn2 split 9, 1 acc", M, k_qpn2<9, 1>, 9); run("qpn2 split 16, 1 acc", M, k_qpn2<16, 1>, 16); run("qpn2 split 4, 2 acc", M, k_qpn2<4, 2>, 4);
    }
    {   // scattered experts: a 3 GB pool of 640 expert blocks, 8 random ones per launch through a pointer table
        const int NP = 640; uint8_t* pool; cudaMalloc(&pool, (size_t) NP * (cb + sb)); for (int i = 0; i < NP; ++i) cudaMemcpy(pool + (size_t) i * (cb + sb), pc.data() + (size_t) (i % (NE * nbuf)) * cb, cb, cudaMemcpyHostToDevice);
        const uint8_t** tabs; cudaMalloc(&tabs, 64 * NE * sizeof(void*)); std::vector<const uint8_t*> ht(64 * NE); for (auto& p : ht) p = pool + (size_t) (rand() % NP) * (cb + sb); cudaMemcpy(tabs, ht.data(), ht.size() * sizeof(void*), cudaMemcpyHostToDevice);
        for (int M : {1}) {
            auto t1 = timeit([&](int i) { k_qpn2_tab<8, 1><<<dim3(N / 32, NE), 256>>>(tabs + (i % 64) * NE, dx, dy, N, K, M); });
            auto t2 = timeit([&](int i) { k_qpn2_tab<4, 1><<<dim3(N / 32, NE), 128>>>(tabs + (i % 64) * NE, dx, dy, N, K, M); });
            printf("scattered experts (3 GB pool, pointer table): split 8: %.1f us   split 4: %.1f us\n", t1, t2);
        }
    }
    {   // the engine's kernel verbatim (ids, factors, swiglu epilogue, act16)
        const int NP = 640; uint8_t* pool; cudaMalloc(&pool, (size_t) NP * (cb + sb)); for (int i = 0; i < NP; ++i) cudaMemcpy(pool + (size_t) i * (cb + sb), pc.data() + (size_t) (i % (NE * nbuf)) * cb, cb, cudaMemcpyHostToDevice);
        const uint8_t** tabs; cudaMalloc(&tabs, NP * sizeof(void*)); std::vector<const uint8_t*> ht(NP); for (int i = 0; i < NP; ++i) ht[i] = pool + (size_t) i * (cb + sb); cudaMemcpy(tabs, ht.data(), ht.size() * sizeof(void*), cudaMemcpyHostToDevice);
        int* ids; cudaMalloc(&ids, 64 * NE * 4); std::vector<int> hi(64 * NE); for (auto& v : hi) v = rand() % NP; cudaMemcpy(ids, hi.data(), hi.size() * 4, cudaMemcpyHostToDevice);
        float* fac; cudaMalloc(&fac, NP * 4); std::vector<float> hf(NP, 1e-4f); cudaMemcpy(fac, hf.data(), NP * 4, cudaMemcpyHostToDevice);
        float *act; cudaMalloc(&act, NE * N * 4); __half* act16; cudaMalloc(&act16, NE * N * 2);
        auto t1 = timeit([&](int i) { k_qpn_sel<8, true><<<dim3(N / 32, NE), 256>>>(tabs, fac, ids + (i % 64) * NE, dx, 0, dy, N, K, act, act16); });
        auto t2 = timeit([&](int i) { k_qpn_sel<8, false><<<dim3(N / 32, NE), 256>>>(tabs, fac, ids + (i % 64) * NE, dx, 0, dy, N, K, act, act16); });
        printf("engine kernel verbatim: GU %.1f us   no-GU %.1f us\n", t1, t2);
    }
    cudaError_t er = cudaGetLastError(); if (er) printf("cuda error: %s\n", cudaGetErrorString(er));
    return 0;
}
