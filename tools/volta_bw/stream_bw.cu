// Achievable DRAM read bandwidth for the expert-GEMV access pattern: warps stream R rows of ROWB bytes each, W bytes per lane per load, UNR loads in flight, almost no ALU.
// nvcc -arch=sm_70 -ccbin g++-14 -isystem <nvcc-host> -O3 stream_bw.cu -o stream_bw
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
template <int W, int R, int UNR>
__global__ void k_stream(const uint8_t* __restrict__ base, size_t expert_bytes, int rows, int rowb, float* __restrict__ out) {
    const int row0 = (blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32) * R, lane = threadIdx.x & 31;
    if (row0 >= rows) return;
    const uint8_t* e = base + blockIdx.y * expert_bytes;
    float acc[R] = {};
    const int words = rowb / W;
#pragma unroll UNR
    for (int i = lane; i < words; i += 32)
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint8_t* p = e + (size_t) min(row0 + r, rows - 1) * rowb + (size_t) i * W;
            if (W == 4) acc[r] += __uint_as_float(__ldg(reinterpret_cast<const uint32_t*>(p)));
            else if (W == 8) { const uint2 v = __ldg(reinterpret_cast<const uint2*>(p)); acc[r] += __uint_as_float(v.x ^ v.y); }
            else { const uint4 v = __ldg(reinterpret_cast<const uint4*>(p)); acc[r] += __uint_as_float(v.x ^ v.y ^ v.z ^ v.w); }
        }
#pragma unroll
    for (int r = 0; r < R; ++r) { float t = acc[r]; for (int o = 16; o; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o); if (!lane) out[(blockIdx.y * rows) + row0 + r] = t; }
}
template <int W, int R, int UNR>
void run(const char* tag, uint8_t* buf, size_t expert_bytes, int rows, int rowb, float* out, int nbuf) {
    const dim3 grid((rows + 8 * R - 1) / (8 * R), 8);
    for (int w = 0; w < 6; ++w) k_stream<W, R, UNR><<<grid, 256>>>(buf + (size_t) (w % nbuf) * 8 * expert_bytes, expert_bytes, rows, rowb, out);
    cudaDeviceSynchronize();
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    const int iters = 60; cudaEventRecord(a);
    for (int i = 0; i < iters; ++i) k_stream<W, R, UNR><<<grid, 256>>>(buf + (size_t) (i % nbuf) * 8 * expert_bytes, expert_bytes, rows, rowb, out);
    cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b);
    const double us = ms * 1000.0 / iters, bytes = 8.0 * rows * rowb;
    printf("%-26s grid %4dx8  %6.1f us  %6.0f GB/s\n", tag, grid.x, us, bytes / (us * 1e-6) / 1e9);
}
int main() {
    const int rows = 2304, rowb = 1152;            // gate_up of one expert, NVFP4: 2 FF rows x 1152 B (K = 2304 nibbles)
    const size_t expert_bytes = (size_t) rows * rowb; const int nbuf = 4;
    uint8_t* buf; float* out; cudaMalloc(&buf, expert_bytes * 8 * nbuf); cudaMalloc(&out, 8 * rows * 4); cudaMemset(buf, 0x31, expert_bytes * 8 * nbuf);
    printf("each launch reads %.1f MB; %d rotating buffers (%.0f MB, larger than L2)\n", 8.0 * rows * rowb / 1e6, nbuf, expert_bytes * 8 * nbuf / 1e6);
    run<4, 2, 2>("4 B/lane  R=2 unroll 2", buf, expert_bytes, rows, rowb, out, nbuf);
    run<4, 2, 4>("4 B/lane  R=2 unroll 4", buf, expert_bytes, rows, rowb, out, nbuf);
    run<4, 4, 2>("4 B/lane  R=4 unroll 2", buf, expert_bytes, rows, rowb, out, nbuf);
    run<8, 2, 2>("8 B/lane  R=2 unroll 2", buf, expert_bytes, rows, rowb, out, nbuf);
    run<16, 2, 2>("16 B/lane R=2 unroll 2", buf, expert_bytes, rows, rowb, out, nbuf);
    run<16, 1, 2>("16 B/lane R=1 unroll 2", buf, expert_bytes, rows, rowb, out, nbuf);
    return 0;
}
