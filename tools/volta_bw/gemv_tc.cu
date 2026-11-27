// Batched NVFP4 expert GEMV on Volta: CUDA-core kernel (the engine's k_moe_gemv_grp shape) against a tensor-core kernel written for Volta's own mma.m8n8k4 layout.
// Both read the same 24 MB (8 experts, gate_up 2304 x 2304) and serve T tokens.  The tensor-core kernel decodes e2m1 straight into the A fragment (no float conversion,
// scale folded in by one HMUL2: v * s is exact in fp16) and keeps its accumulators in Volta's D layout; activations are fp16 (as on the Turing tensor-core path).
// nvcc -arch=sm_70 -ccbin g++-14 -isystem <nvcc-host> -O3 gemv_tc.cu -o gemv_tc
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
__device__ __forceinline__ uint32_t e2m1_h2(uint32_t t) { return ((t & 0x00080008u) << 12) | ((t & 0x00070007u) << 9); }   // (n0, n4) as half2, value * 2^-14
__device__ __forceinline__ float2 e2m1_k4(uint32_t t) { const uint32_t h = e2m1_h2(t); return __half22float2(*reinterpret_cast<const __half2*>(&h)); }
__device__ __forceinline__ void mma884(float (&d)[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]) : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}
// scale byte (e4m3) -> half2 (s * 2^14, s * 2^14); s in the file stays below 4 so s * 2^14 fits fp16
__device__ __forceinline__ float e4m3_f(uint32_t b) { const int e = (b >> 3) & 15, m = b & 7; return (e ? ldexpf(1.0f + m / 8.0f, e - 7) : ldexpf(m / 8.0f, -6)) * ((b & 128) ? -1.f : 1.f); }

// ---- CUDA-core, T tokens: warp per 2 rows, lanes along K, weights decoded once, T FMAs per weight
template <int T>
__global__ void k_cc(const uint8_t* __restrict__ base, size_t eb, const float* __restrict__ x, int rows, int K, float* __restrict__ out) {
    constexpr int R = 2;
    __shared__ float e4[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) e4[i] = e4m3_f(i) * 16384.0f;
    __syncthreads();
    const int row0 = (blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32) * R, lane = threadIdx.x & 31;
    if (row0 >= rows) return;
    const uint8_t* W = base + blockIdx.y * eb; const uint8_t* Sc = W + (size_t) rows * (K / 2);
    const uint32_t* w[R]; const uint8_t* s[R];
    for (int r = 0; r < R; ++r) { const size_t row = min(row0 + r, rows - 1); w[r] = reinterpret_cast<const uint32_t*>(W + row * (K / 2)); s[r] = Sc + row * (K / 16); }
    float acc[R][T];
    for (int r = 0; r < R; ++r) for (int t = 0; t < T; ++t) acc[r][t] = 0.f;
#pragma unroll 2
    for (int i = lane; i < K / 8; i += 32) {
        uint32_t v[R]; uint8_t sc[R];
#pragma unroll
        for (int r = 0; r < R; ++r) { v[r] = __ldg(w[r] + i); sc[r] = __ldg(s[r] + (i >> 1)); }
        float4 a[T], b[T];
#pragma unroll
        for (int t = 0; t < T; ++t) { a[t] = __ldg(reinterpret_cast<const float4*>(x + t * K) + 2 * i); b[t] = __ldg(reinterpret_cast<const float4*>(x + t * K) + 2 * i + 1); }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float2 q0 = e2m1_k4(v[r]), q1 = e2m1_k4(v[r] >> 4), q2 = e2m1_k4(v[r] >> 8), q3 = e2m1_k4(v[r] >> 12);
            const float f = e4[sc[r]];
#pragma unroll
            for (int t = 0; t < T; ++t) acc[r][t] += (q0.x * a[t].x + q1.x * a[t].y + q2.x * a[t].z + q3.x * a[t].w + q0.y * b[t].x + q1.y * b[t].y + q2.y * b[t].z + q3.y * b[t].w) * f;
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) for (int t = 0; t < T; ++t) { float v = acc[r][t]; for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o); if (!lane) out[((size_t) blockIdx.y * T + t) * rows + row0 + r] = v; }
}

// ---- tensor core.  Block = 8 warps, RB rows.  Warp tile = 8 rows; octet q of the warp (lanes 4q..4q+3 and 16+4q..) takes quarter q of K for those 8 rows; lane p = (lane&3)+((lane&16)>>2)
// holds weights of row p and activations of token p (zero past T).  Each octet's four k-quarters are summed at the end (two shuffles).
// x16: [8 tokens][K] halves, k permuted inside every 8: 0 4 1 5 2 6 3 7 (so the half2 pairs the decode yields are B's k pairs)
template <int RB, int NW, int NA>
__global__ void k_tc(const uint8_t* __restrict__ base, size_t eb, const __half* __restrict__ x16, int rows, int K, int T, float* __restrict__ out) {
    extern __shared__ __align__(16) uint8_t smem[];
    __half* xs = reinterpret_cast<__half*>(smem);   // [8][K]
    __shared__ uint32_t sct[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) { const __half h = __float2half(e4m3_f(i) * 16384.0f); sct[i] = ((uint32_t) __half_as_ushort(h) << 16) | __half_as_ushort(h); }
    constexpr int XPAD = 4;   // halves: row stride K + 4 spreads the eight token rows over banks (stride K = 2304 halves put all eight on one bank: 8-way conflict)
    const int KP = K + XPAD;
    for (int i = threadIdx.x; i < 8 * K / 4; i += blockDim.x) { const int t = i / (K / 4), c = i % (K / 4); reinterpret_cast<uint2*>(xs + (size_t) t * KP)[c] = reinterpret_cast<const uint2*>(x16)[i]; }
    __syncthreads();
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5, q = (lane >> 2) & 3, p = (lane & 3) + ((lane & 16) >> 2);
    const uint8_t* W = base + blockIdx.y * eb; const uint8_t* Sc = W + (size_t) rows * (K / 2);
    const int KQ = K / 4;   // k per octet
    for (int tile = wp; tile < RB / 8; tile += NW) {
        const int rowbase = blockIdx.x * RB + tile * 8;
        const int row = min(rowbase + p, rows - 1);
        const uint4* wv = reinterpret_cast<const uint4*>(W + (size_t) row * (K / 2) + (size_t) q * (KQ / 2));
        const uint32_t* sv = reinterpret_cast<const uint32_t*>(Sc + (size_t) row * (K / 16) + q * (KQ / 16));
        const uint2* xv = reinterpret_cast<const uint2*>(xs + (size_t) p * KP + q * KQ);   // token p, this quarter: 4 halves per uint2
        float da[NA][8];
#pragma unroll
        for (int a = 0; a < NA; ++a)
#pragma unroll
            for (int i = 0; i < 8; ++i) da[a][i] = 0.f;
#pragma unroll 3
        for (int j = 0; j < KQ / 64; ++j) {   // 64 k per trip: two uint4 of weights, one word of four scales
            const uint4 w0 = __ldg(wv + 2 * j), w1 = __ldg(wv + 2 * j + 1);
            const uint32_t sw = __ldg(sv + j);
            const uint32_t ww[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
            for (int u = 0; u < 8; ++u) {   // word u = 8 k; its scale: byte (u >> 1) of sw
                const uint32_t s2 = sct[(sw >> (8 * (u >> 1))) & 255];
                uint32_t h[4];
#pragma unroll
                for (int g = 0; g < 4; ++g) { const uint32_t e = e2m1_h2(ww[u] >> (4 * g)); h[g] = *reinterpret_cast<const uint32_t*>(&(const __half2&) __hmul2(*reinterpret_cast<const __half2*>(&e), *reinterpret_cast<const __half2*>(&s2))); }
                const uint2 b0 = xv[(8 * j + u) * 2], b1 = xv[(8 * j + u) * 2 + 1];   // k permuted x: [0 4 1 5] then [2 6 3 7]
                mma884(da[u % NA], h[0], h[1], b0.x, b0.y);
                mma884(da[(u + 1) % NA], h[2], h[3], b1.x, b1.y);
            }
        }
        float d[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) { d[i] = da[0][i]; if (NA > 1) d[i] += da[NA - 1][i]; }
        // sum the four quarters (octets differ in lane bits 2 and 3)
#pragma unroll
        for (int i = 0; i < 8; ++i) { d[i] += __shfl_xor_sync(0xffffffffu, d[i], 4); d[i] += __shfl_xor_sync(0xffffffffu, d[i], 8); }
        if (q == 0) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int r = (lane & 1) + (i & 2) + ((lane & 16) >> 2), c = (i & 4) + (lane & 2) + (i & 1);
                if (c < T && rowbase + r < rows) out[((size_t) blockIdx.y * T + c) * rows + rowbase + r] = d[i];
            }
        }
    }
}

// k_tc2: weights repacked (host, once) so that each warp load is 512 contiguous bytes, x staged with a per-quarter offset so the four octets read different banks.
// weight layout per expert: [tile of 8 rows][trip j (K/4/64)][half h][octet q][p] x 16 B (the lane's 32 B of trip j, half h); scales: [tile][j][q][p] x 4 B
constexpr int KQPAD = 32;   // halves between k-quarters in smem: quarter stride (K/4 + 32) halves = 16 mod 32 words: octets 0 / 2 and 1 / 3 share banks (the 2-wavefront minimum for 32 distinct 8-byte reads)
template <int RB, int NW>
__global__ void k_tc2(const uint8_t* __restrict__ base, size_t eb, const __half* __restrict__ x16, int rows, int K, int T, float* __restrict__ out) {
    extern __shared__ __align__(16) uint8_t smem[];
    __half* xs = reinterpret_cast<__half*>(smem);
    __shared__ uint32_t sct[256];
    const int KQ = K / 4, QS = KQ + KQPAD, KP = 4 * QS + 4;   // row stride 2 words past a multiple of 32: the eight token rows take eight different bank pairs
    for (int i = threadIdx.x; i < 256; i += blockDim.x) { const __half h = __float2half(e4m3_f(i) * 16384.0f); sct[i] = ((uint32_t) __half_as_ushort(h) << 16) | __half_as_ushort(h); }
    for (int i = threadIdx.x; i < 8 * K / 4; i += blockDim.x) {
        const int t = i / (K / 4), c = i % (K / 4), k = c * 4, qq = k / KQ;
        reinterpret_cast<uint2*>(xs + (size_t) t * KP + qq * QS + (k - qq * KQ))[0] = reinterpret_cast<const uint2*>(x16)[i];
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5, q = (lane >> 2) & 3, p = (lane & 3) + ((lane & 16) >> 2), lq = q * 8 + p;
    const uint8_t* W = base + blockIdx.y * eb; const uint8_t* Sc = W + (size_t) rows * (K / 2);
    const int NJ = KQ / 64;
    for (int tile = wp; tile < RB / 8; tile += NW) {
        const int gt = blockIdx.x * (RB / 8) + tile, rowbase = gt * 8;
        const uint4* wt = reinterpret_cast<const uint4*>(W + (size_t) gt * 8 * (K / 2));
        const uint32_t* st = reinterpret_cast<const uint32_t*>(Sc + (size_t) gt * 8 * (K / 16));
        const uint2* xv = reinterpret_cast<const uint2*>(xs + (size_t) p * KP + q * QS);
        float d[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) d[i] = 0.f;
#pragma unroll 3
        for (int j = 0; j < NJ; ++j) {
            const uint4 w0 = __ldg(wt + (j * 2 + 0) * 32 + lq), w1 = __ldg(wt + (j * 2 + 1) * 32 + lq);
            const uint32_t sw = __ldg(st + j * 32 + lq);
            const uint32_t ww[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                const uint32_t s2 = sct[(sw >> (8 * (u >> 1))) & 255];
                uint32_t h[4];
#pragma unroll
                for (int g = 0; g < 4; ++g) { const uint32_t e = e2m1_h2(ww[u] >> (4 * g)); h[g] = *reinterpret_cast<const uint32_t*>(&(const __half2&) __hmul2(*reinterpret_cast<const __half2*>(&e), *reinterpret_cast<const __half2*>(&s2))); }
                const uint2 b0 = xv[(8 * j + u) * 2], b1 = xv[(8 * j + u) * 2 + 1];
                mma884(d, h[0], h[1], b0.x, b0.y);
                mma884(d, h[2], h[3], b1.x, b1.y);
            }
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) { d[i] += __shfl_xor_sync(0xffffffffu, d[i], 4); d[i] += __shfl_xor_sync(0xffffffffu, d[i], 8); }
        if (q == 0) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int r = (lane & 1) + (i & 2) + ((lane & 16) >> 2), c = (i & 4) + (lane & 2) + (i & 1);
                if (c < T && rowbase + r < rows) out[((size_t) blockIdx.y * T + c) * rows + rowbase + r] = d[i];
            }
        }
    }
}
// host repack of one expert (src, dst: eb bytes)
static void repack(const uint8_t* src, uint8_t* dst, int rows, int K) {
    const int KQ = K / 4, NJ = KQ / 64;
    const uint8_t* Ssrc = src + (size_t) rows * (K / 2); uint8_t* Sdst = dst + (size_t) rows * (K / 2);
    for (int tile = 0; tile < rows / 8; ++tile) for (int j = 0; j < NJ; ++j) for (int q = 0; q < 4; ++q) for (int p = 0; p < 8; ++p) {
        const int row = tile * 8 + p;
        for (int h = 0; h < 2; ++h) memcpy(dst + (size_t) tile * 8 * (K / 2) + ((size_t) ((j * 2 + h) * 32 + q * 8 + p)) * 16, src + (size_t) row * (K / 2) + (size_t) q * (KQ / 2) + 32 * j + 16 * h, 16);
        memcpy(Sdst + (size_t) tile * 8 * (K / 16) + ((size_t) (j * 32 + q * 8 + p)) * 4, Ssrc + (size_t) row * (K / 16) + q * (KQ / 16) + 4 * j, 4);
    }
}
template <typename F> float timeit(F f, int it = 60) {
    for (int w = 0; w < 6; ++w) f(w); cudaDeviceSynchronize(); cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); cudaEventRecord(a);
    for (int i = 0; i < it; ++i) f(i); cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); return ms * 1000.f / it;
}
int main() {
    const int rows = 2304, K = 2304, nbuf = 4; const size_t eb = (size_t) rows * (K / 2 + K / 16);
    uint8_t* buf; float *x, *out, *out2; __half* x16; cudaMalloc(&buf, eb * 8 * nbuf); cudaMalloc(&x, 8 * K * 4); cudaMalloc(&x16, 8 * K * 2); cudaMalloc(&out, 8 * 8 * rows * 4); cudaMalloc(&out2, 8 * 8 * rows * 4);
    std::vector<uint8_t> hb(eb * 8 * nbuf); for (size_t i = 0; i < hb.size(); ++i) hb[i] = (uint8_t) rand();
    // valid scales: bytes of the scale blocks restricted to small positive e4m3 (exponent 3..8: 2^-4 .. 2^1)
    for (int e = 0; e < 8 * nbuf; ++e) for (size_t i = (size_t) rows * (K / 2); i < eb; ++i) hb[e * eb + i] = (uint8_t) (((3 + rand() % 6) << 3) | (rand() & 7));
    cudaMemcpy(buf, hb.data(), hb.size(), cudaMemcpyHostToDevice);
    uint8_t* rbuf; cudaMalloc(&rbuf, hb.size()); { std::vector<uint8_t> rb(hb.size()); for (int e = 0; e < 8 * nbuf; ++e) repack(hb.data() + e * eb, rb.data() + e * eb, rows, K); cudaMemcpy(rbuf, rb.data(), rb.size(), cudaMemcpyHostToDevice); }
    std::vector<float> hx(8 * K); for (auto& f : hx) f = (rand() / (float) RAND_MAX) - 0.5f;
    cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice);
    std::vector<__half> hx16(8 * K, __float2half(0.f));   // permuted, token-major, T = 4 filled below
    const int perm[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    auto fill16 = [&](int T) { for (int t = 0; t < 8; ++t) for (int k = 0; k < K; ++k) { const int kp = (k / 8) * 8 + perm[k % 8]; (void) kp; }
        for (int t = 0; t < 8; ++t) for (int g = 0; g < K / 8; ++g) for (int j = 0; j < 8; ++j) hx16[(size_t) t * K + g * 8 + j] = t < T ? __float2half(hx[(size_t) t * K + g * 8 + perm[j]]) : __float2half(0.f);
        cudaMemcpy(x16, hx16.data(), hx16.size() * 2, cudaMemcpyHostToDevice); };
    // reference for correctness: double precision, x rounded to fp16, weights per the engine's format (nibble j of word i = k 8i + j; scale byte = k / 16)
    auto ref = [&](int T, std::vector<double>& r) {
        r.assign((size_t) 8 * T * rows, 0.0);
        for (int e = 0; e < 8; ++e) for (int row = 0; row < rows; ++row) for (int t = 0; t < T; ++t) {
            const uint8_t* W = hb.data() + e * eb; const uint8_t* Sc = W + (size_t) rows * (K / 2); double a = 0;
            for (int k = 0; k < K; ++k) {
                const uint8_t byte = W[(size_t) row * (K / 2) + k / 2]; const int n = (k & 1) ? byte >> 4 : byte & 15;
                static const double mag[8] = {0, .5, 1, 1.5, 2, 3, 4, 6}; const double wv = (n & 8 ? -1 : 1) * mag[n & 7];
                const int sb = Sc[(size_t) row * (K / 16) + k / 16], se = (sb >> 3) & 15, sm = sb & 7;
                const double s = (se ? ldexp(1.0 + sm / 8.0, se - 7) : ldexp(sm / 8.0, -6)) * (sb & 128 ? -1 : 1);
                a += wv * s * (double) __half2float(__float2half(hx[(size_t) t * K + k]));
            }
            r[((size_t) e * T + t) * rows + row] = a;
        }
    };
    printf("gate_up %d x %d, 8 experts, %.1f MB per launch\n%-4s %-34s %9s %9s\n", rows, K, 8.0 * eb / 1e6, "T", "kernel", "DRAM-fed", "max rel err");
    auto bench = [&](int T) {
        fill16(T);
        const dim3 gcc((rows + 15) / 16, 8), gtc(rows / 128, 8); const size_t sm = (size_t) 8 * (K + 4) * 2;
        cudaFuncSetAttribute(k_tc<128, 8, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sm);
        float t_cc = 0;
        if (T == 1) t_cc = timeit([&](int i) { k_cc<1><<<gcc, 256>>>(buf + (size_t) (i % nbuf) * 8 * eb, eb, x, rows, K, out); });
        if (T == 2) t_cc = timeit([&](int i) { k_cc<2><<<gcc, 256>>>(buf + (size_t) (i % nbuf) * 8 * eb, eb, x, rows, K, out); });
        if (T == 4) t_cc = timeit([&](int i) { k_cc<4><<<gcc, 256>>>(buf + (size_t) (i % nbuf) * 8 * eb, eb, x, rows, K, out); });
        if (T == 8) t_cc = timeit([&](int i) { k_cc<8><<<gcc, 256>>>(buf + (size_t) (i % nbuf) * 8 * eb, eb, x, rows, K, out); });
        float t_tc = timeit([&](int i) { k_tc<128, 8, 1><<<gtc, 256, sm>>>(buf + (size_t) (i % nbuf) * 8 * eb, eb, x16, rows, K, T, out2); });
        {
            const size_t sm2 = (size_t) 8 * (4 * (K / 4 + 32) + 4) * 2; cudaFuncSetAttribute(k_tc2<128, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sm2); cudaFuncSetAttribute(k_tc2<64, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sm2); cudaFuncSetAttribute(k_tc2<256, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sm2);
            const float t2 = timeit([&](int i) { k_tc2<128, 8><<<gtc, 256, sm2>>>(rbuf + (size_t) (i % nbuf) * 8 * eb, eb, x16, rows, K, T, out); });
            k_tc2<128, 8><<<gtc, 256, sm2>>>(rbuf, eb, x16, rows, K, T, out); cudaDeviceSynchronize();
            std::vector<float> o((size_t) 8 * T * rows); cudaMemcpy(o.data(), out, o.size() * 4, cudaMemcpyDeviceToHost); std::vector<double> r; ref(T, r); double me = 0, mx = 0; for (size_t i = 0; i < o.size(); ++i) { me = fmax(me, fabs(o[i] - r[i])); mx = fmax(mx, fabs(r[i])); }
            const dim3 g64(rows / 64, 8), g256(rows / 256, 8);
            printf("%-4d %-34s %7.1f us   %.2e   (64 rows/block %.1f, 256 rows/block %.1f)\n", T, "tensor core, repacked + padded", t2, me / mx,
                   timeit([&](int i) { k_tc2<64, 8><<<g64, 256, sm2>>>(rbuf + (size_t) (i % nbuf) * 8 * eb, eb, x16, rows, K, T, out); }), timeit([&](int i) { k_tc2<256, 8><<<g256, 256, sm2>>>(rbuf + (size_t) (i % nbuf) * 8 * eb, eb, x16, rows, K, T, out); }));
        }
        if (false) {
            cudaFuncSetAttribute(k_tc<128, 8, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sm); cudaFuncSetAttribute(k_tc<32, 4, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sm);
            cudaFuncSetAttribute(k_tc<32, 4, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sm); cudaFuncSetAttribute(k_tc<64, 8, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sm);
            const dim3 g32(rows / 32, 8), g64(rows / 64, 8);
            printf("     (T=4 variants)  128 rows/8 warps 2 acc sets: %.1f us | 32 rows/4 warps 1 acc: %.1f | 32 rows/4 warps 2 acc: %.1f | 64 rows/8 warps 2 acc: %.1f\n",
                   timeit([&](int i) { k_tc<128, 8, 2><<<gtc, 256, sm>>>(buf + (size_t) (i % nbuf) * 8 * eb, eb, x16, rows, K, T, out2); }),
                   timeit([&](int i) { k_tc<32, 4, 1><<<g32, 128, sm>>>(buf + (size_t) (i % nbuf) * 8 * eb, eb, x16, rows, K, T, out2); }),
                   timeit([&](int i) { k_tc<32, 4, 2><<<g32, 128, sm>>>(buf + (size_t) (i % nbuf) * 8 * eb, eb, x16, rows, K, T, out2); }),
                   timeit([&](int i) { k_tc<64, 8, 2><<<g64, 256, sm>>>(buf + (size_t) (i % nbuf) * 8 * eb, eb, x16, rows, K, T, out2); }));
        }
        // correctness on buffer 0 (the first launch of each sequence above used buffers 0..): run once on buffer 0
        k_tc<128, 8, 1><<<gtc, 256, sm>>>(buf, eb, x16, rows, K, T, out2); cudaDeviceSynchronize();
        std::vector<float> o((size_t) 8 * T * rows); cudaMemcpy(o.data(), out2, o.size() * 4, cudaMemcpyDeviceToHost);
        std::vector<double> r; ref(T, r); double me = 0, mx = 0; for (size_t i = 0; i < o.size(); ++i) { me = fmax(me, fabs(o[i] - r[i])); mx = fmax(mx, fabs(r[i])); }
        printf("%-4d %-34s %7.1f us\n%-4d %-34s %7.1f us   %.2e (max |err| %.3g of max |out| %.3g)\n", T, "CUDA cores (engine shape)", t_cc, T, "tensor core m8n8k4 native", t_tc, me / mx, me, mx);
    };
    for (int T : {1, 2, 4, 8}) bench(T);
    cudaError_t e = cudaGetLastError(); if (e) printf("cuda error: %s\n", cudaGetErrorString(e));
    return 0;
}
