// mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 on Volta (sm_70), built from Volta's one tensor-core instruction, mma.m8n8k4.row.col.
// Turing fragment layout (lane = 4 g + t): a0 = A[g][2t..2t+1], a1 = A[g+8][2t..2t+1], b0 = B[2t..2t+1][g], c0,c1 = C[g][2t..2t+1], c2,c3 = C[g+8][2t..2t+1].
// Volta m8n8k4.row.col, per octet (lanes 4q..4q+3 and 16+4q..16+4q+3; p = (lane & 3) + ((lane & 16) >> 2)): A row p, B column p, 4 halves each;
// D (f32, 8 values i): row (lane & 1) + (i & 2) + ((lane & 16) >> 2), column (i & 4) + (lane & 2) + (i & 1).
// Octet q = (lane >> 2) & 3 does row tile rt = q >> 1 and k-half kh = q & 1.  A chain of products into the same C keeps its accumulator in Volta's layout
// (volta_step: 6 shuffles and one mma.m8n8k4); volta_finish adds the two k-halves (lanes 4 apart) and moves each result to the lane Turing's layout has it in.
// tools/volta_mma/volta_mma_test.cu checks it against a CPU reference.  Every lane of the warp must call these together.
#pragma once
#include <cstdint>
__device__ __forceinline__ uint32_t volta_lane() { uint32_t l; asm volatile("mov.u32 %0, %%laneid;" : "=r"(l)); return l; }
__device__ __forceinline__ void volta_mma8(float (&d)[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void volta_step(float (&v)[8], uint32_t a0, uint32_t a1, uint32_t b0) {
    const uint32_t lane = volta_lane(), q = (lane >> 2) & 3, rt = q >> 1, kh = q & 1, p = (lane & 3) + ((lane & 16) >> 2);
    const uint32_t s0 = 4 * p + 2 * kh;
    const uint32_t a00 = __shfl_sync(0xffffffffu, a0, s0), a01 = __shfl_sync(0xffffffffu, a0, s0 + 1);
    const uint32_t a10 = __shfl_sync(0xffffffffu, a1, s0), a11 = __shfl_sync(0xffffffffu, a1, s0 + 1);
    const uint32_t b_lo = __shfl_sync(0xffffffffu, b0, s0), b_hi = __shfl_sync(0xffffffffu, b0, s0 + 1);
    volta_mma8(v, rt ? a10 : a00, rt ? a11 : a01, b_lo, b_hi);
}
// d0..d3 += the chain's sum, in Turing's layout
__device__ __forceinline__ void volta_finish(const float (&v)[8], float& d0, float& d1, float& d2, float& d3) {
    const uint32_t lane = volta_lane(), g = lane >> 2, t = lane & 3;
    float s[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) s[i] = v[i] + __shfl_xor_sync(0xffffffffu, v[i], 4);
    const uint32_t i_base = ((g >> 1) & 1) * 2 + (t >> 1) * 4;
    const uint32_t src_lo = (g & 1) | (((g >> 2) & 1) << 4) | ((t & 1) << 1);   // octet 0 holds rows 0..7; octet 2 (lane | 8) rows 8..15
    float r0[8], r1[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) { r0[i] = __shfl_sync(0xffffffffu, s[i], src_lo); r1[i] = __shfl_sync(0xffffffffu, s[i], src_lo | 8); }
    float o0 = r0[0], o1 = r0[0], o2 = r1[0], o3 = r1[0];
#pragma unroll
    for (int b = 0; b < 8; ++b) {
        o0 = (b == i_base) ? r0[b] : o0;     o1 = (b == i_base + 1) ? r0[b] : o1;
        o2 = (b == i_base) ? r1[b] : o2;     o3 = (b == i_base + 1) ? r1[b] : o3;
    }
    d0 += o0; d1 += o1; d2 += o2; d3 += o3;
}
// one product on its own (the test)
__device__ __forceinline__ void mma_m16n8k8_volta(float& d0, float& d1, float& d2, float& d3, uint32_t a0, uint32_t a1, uint32_t b0) {
    float v[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    volta_step(v, a0, a1, b0);
    volta_finish(v, d0, d1, d2, d3);
}
