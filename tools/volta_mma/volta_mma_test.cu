// nvcc -arch=sm_70 -ccbin g++-14 -isystem <nvcc-host> volta_mma_test.cu -o volta_mma_test     (run on a V100)
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_fp16.h>
#include "../../engine/src/inkling2/ink2_volta_mma.cuh"
static uint32_t pk(__half lo, __half hi) { return (uint32_t) __half_as_ushort(lo) | ((uint32_t) __half_as_ushort(hi) << 16); }
__global__ void k_test(const uint32_t* a, const uint32_t* b, const float* c, float* d, int iters) {   // per lane, per step: a0 a1 b0; c0..c3 once; a chain of `iters` products
    const int lane = threadIdx.x;
    float c0 = c[lane * 4], c1 = c[lane * 4 + 1], c2 = c[lane * 4 + 2], c3 = c[lane * 4 + 3];
    float v[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int it = 0; it < iters; ++it) volta_step(v, a[(it * 32 + lane) * 2], a[(it * 32 + lane) * 2 + 1], b[it * 32 + lane]);
    volta_finish(v, c0, c1, c2, c3);
    d[lane * 4] = c0; d[lane * 4 + 1] = c1; d[lane * 4 + 2] = c2; d[lane * 4 + 3] = c3;
}
int main(int argc, char** argv) {
    const int STEPS = argc > 1 ? atoi(argv[1]) : 8;   // products in one chain, as in the engine's kernels
    srand(7);
    double worst = 0, worst_rel = 0; long bad = 0, total = 0;
    for (int trial = 0; trial < 200; ++trial) {
        std::vector<std::vector<std::vector<float>>> A(STEPS, std::vector<std::vector<float>>(16, std::vector<float>(8))), B(STEPS, std::vector<std::vector<float>>(8, std::vector<float>(8)));
        float C[16][8];
        auto rnd = [] { return __half2float(__float2half((rand() / (float) RAND_MAX - 0.5f) * 4.f)); };
        for (int k = 0; k < STEPS; ++k) { for (auto& r : A[k]) for (float& x : r) x = rnd(); for (auto& r : B[k]) for (float& x : r) x = rnd(); }
        for (auto& r : C) for (float& x : r) x = (rand() / (float) RAND_MAX - 0.5f) * 8.f;
        std::vector<uint32_t> ha(64 * STEPS), hb(32 * STEPS); std::vector<float> hc(128), hd(128);
        for (int k = 0; k < STEPS; ++k)
            for (int l = 0; l < 32; ++l) {
                const int g = l >> 2, t = l & 3;
                ha[(k * 32 + l) * 2] = pk(__float2half(A[k][g][2 * t]), __float2half(A[k][g][2 * t + 1]));
                ha[(k * 32 + l) * 2 + 1] = pk(__float2half(A[k][g + 8][2 * t]), __float2half(A[k][g + 8][2 * t + 1]));
                hb[k * 32 + l] = pk(__float2half(B[k][2 * t][g]), __float2half(B[k][2 * t + 1][g]));
            }
        for (int l = 0; l < 32; ++l) {
            const int g = l >> 2, t = l & 3;
            hc[l * 4] = C[g][2 * t]; hc[l * 4 + 1] = C[g][2 * t + 1]; hc[l * 4 + 2] = C[g + 8][2 * t]; hc[l * 4 + 3] = C[g + 8][2 * t + 1];
        }
        uint32_t *da, *db; float *dc, *dd;
        cudaMalloc(&da, ha.size() * 4); cudaMalloc(&db, hb.size() * 4); cudaMalloc(&dc, 512); cudaMalloc(&dd, 512);
        cudaMemcpy(da, ha.data(), ha.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(db, hb.data(), hb.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(dc, hc.data(), 512, cudaMemcpyHostToDevice);
        k_test<<<1, 32>>>(da, db, dc, dd, STEPS);
        if (cudaDeviceSynchronize() != cudaSuccess) { printf("kernel failed: %s\n", cudaGetErrorString(cudaGetLastError())); return 1; }
        cudaMemcpy(hd.data(), dd, 512, cudaMemcpyDeviceToHost);
        cudaFree(da); cudaFree(db); cudaFree(dc); cudaFree(dd);
        for (int l = 0; l < 32; ++l) {
            const int g = l >> 2, t = l & 3;
            for (int j = 0; j < 4; ++j) {
                const int r = g + (j >> 1) * 8, cc = 2 * t + (j & 1);
                double ref = C[r][cc];
                for (int k = 0; k < STEPS; ++k) for (int kk = 0; kk < 8; ++kk) ref += (double) A[k][r][kk] * (double) B[k][kk][cc];
                const double err = fabs(hd[l * 4 + j] - ref); ++total;
                worst = fmax(worst, err); worst_rel = fmax(worst_rel, err / fmax(1.0, fabs(ref)));
                if (err > 1e-3 * fmax(1.0, fabs(ref))) { if (bad < 6) printf("MISMATCH trial %d lane %d reg %d (D[%d][%d]): got %g want %g\n", trial, l, j, r, cc, hd[l * 4 + j], ref); ++bad; }
            }
        }
    }
    printf("chain of %d products: %ld results, %ld wrong; worst abs error %.3g, worst relative %.3g\n", STEPS, total, bad, worst, worst_rel);
    return bad ? 2 : 0;
}
