// Q4 (W4) weight x int8 (A8) activation fused gate/up SwiGLU, group-64 scaled.
//
// The activation is materialized per-token int8 with a per-token absmax scale (q4_a8.cu).  The Q4
// weight codes stay packed (32 bytes per (row, group)); they are sign-extended to int8 in the
// shared stage with two 16-byte stores per stream.  Q4 scales vary per (row, K-group-64), so each
// k32 step folds its int accumulator into the float total with `w_group_scale * a_token_scale`
// (valid by linearity).  A block computes kBN gate output rows and their kBN matching up rows;
// the epilogue pairs them as silu(gate) * up.
#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ops/linear/q4/q4_a8_plan.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_kernels.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/token_slices.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

__device__ __forceinline__ unsigned q4a8_smem(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void q4a8_cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(q4a8_smem(dst)), "l"(src));
}
__device__ __forceinline__ void q4a8_cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N> __device__ __forceinline__ void q4a8_cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void q4a8_ldm_x2(unsigned& a, unsigned& b, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(a), "=r"(b) : "r"(addr));
}
__device__ __forceinline__ void q4a8_ldm_x4(unsigned& a, unsigned& b, unsigned& c, unsigned& d,
                                            unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(addr));
}
__device__ __forceinline__ void q4a8_mma(int& c0, int& c1, int& c2, int& c3, unsigned a0,
                                         unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

constexpr int kBM      = 128;
constexpr int kBN      = 64;
constexpr int kG       = 64;
constexpr int kThreads = 512;
constexpr int kStages  = 2;
constexpr int kSeg     = kG / 16;
constexpr int kWarps   = kThreads / 32;
constexpr int kNT      = kBM / 16;
constexpr int kNR      = kWarps / kNT;
constexpr int kMR      = (kBN / kNR) / 8;

__device__ __forceinline__ int q4a8_swz(int row, int logical_byte) {
    return (((logical_byte >> 4) ^ (row & (kSeg - 1))) * 16) + (logical_byte & 15);
}
__device__ __forceinline__ int q4a8_s4(int x) {
    return static_cast<int>(static_cast<std::int8_t>(x << 4)) >> 4;
}
__device__ __forceinline__ unsigned q4a8_pack4(std::uint8_t b0, std::uint8_t b1) {
    return static_cast<unsigned>(static_cast<std::uint8_t>(q4a8_s4(b0 & 0xF))) |
           (static_cast<unsigned>(static_cast<std::uint8_t>(q4a8_s4(b0 >> 4))) << 8) |
           (static_cast<unsigned>(static_cast<std::uint8_t>(q4a8_s4(b1 & 0xF))) << 16) |
           (static_cast<unsigned>(static_cast<std::uint8_t>(q4a8_s4(b1 >> 4))) << 24);
}

__global__ __launch_bounds__(kThreads, 2) void q4_linear_swiglu_a8_kernel(
    const std::int8_t* __restrict__ xq, const float* __restrict__ xs,
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ out, std::int32_t intermediate, std::int32_t k, std::int32_t t,
    std::int32_t padded_k) {
    __shared__ __align__(16) std::int8_t As[kStages][kBM * kG];
    __shared__ __align__(16) std::int8_t Bgs[kStages][kBN * kG];
    __shared__ __align__(16) std::int8_t Bus[kStages][kBN * kG];
    __shared__ float Wsg[kStages][kBN], Wsu[kStages][kBN], xss[kBM];

    const int tid   = static_cast<int>(threadIdx.x);
    const int warp  = tid >> 5;
    const int lane  = tid & 31;
    // Token block is the fast grid axis: consecutive CTAs share the same weight slice, so the
    // active Q4 rows stay L2-resident instead of streaming the whole weight once per token block.
    const int tok0  = static_cast<int>(blockIdx.x) * kBM;
    const int orow0 = static_cast<int>(blockIdx.y) * kBN;
    const int kg    = padded_k >> 6;
    const int groups = kg;

    auto stage = [&](int s, int g) {
        const int k0 = g * kG;
        for (int item = tid; item < kBM * (kG / 16); item += kThreads) {
            const int r = item / (kG / 16);
            const int c = (item % (kG / 16)) * 16;
            std::int8_t* dst = &As[s][r * kG + q4a8_swz(r, c)];
            const int tok = tok0 + r;
            if (tok < t && k0 + c + 16 <= k) {
                q4a8_cp16(dst, &xq[static_cast<std::int64_t>(tok) * k + k0 + c]);
            } else {
                *reinterpret_cast<int4*>(dst) = make_int4(0, 0, 0, 0);
            }
        }
        for (int item = tid; item < kBN * (kG / 32); item += kThreads) {
            const int r   = item / (kG / 32);
            const int c16 = (item % (kG / 32)) * 16;
            const int p0  = (((c16 << 1) >> 4) ^ (r & 3)) * 16;
            const int p1  = p0 ^ 16;
            const std::uint8_t* pg =
                codes + (static_cast<std::int64_t>(orow0 + r) * kg + g) * 32 + c16;
            const std::uint8_t* pu =
                codes + (static_cast<std::int64_t>(intermediate + orow0 + r) * kg + g) * 32 + c16;
            const uint4 vg = *reinterpret_cast<const uint4*>(pg);
            const uint4 vu = *reinterpret_cast<const uint4*>(pu);
            const std::uint8_t* bg = reinterpret_cast<const std::uint8_t*>(&vg);
            const std::uint8_t* bu = reinterpret_cast<const std::uint8_t*>(&vu);
            unsigned wg[8], wu[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                wg[j] = q4a8_pack4(bg[2 * j], bg[2 * j + 1]);
                wu[j] = q4a8_pack4(bu[2 * j], bu[2 * j + 1]);
            }
            *reinterpret_cast<uint4*>(&Bgs[s][r * kG + p0]) = make_uint4(wg[0], wg[1], wg[2], wg[3]);
            *reinterpret_cast<uint4*>(&Bgs[s][r * kG + p1]) = make_uint4(wg[4], wg[5], wg[6], wg[7]);
            *reinterpret_cast<uint4*>(&Bus[s][r * kG + p0]) = make_uint4(wu[0], wu[1], wu[2], wu[3]);
            *reinterpret_cast<uint4*>(&Bus[s][r * kG + p1]) = make_uint4(wu[4], wu[5], wu[6], wu[7]);
        }
        for (int r = tid; r < kBN; r += kThreads) {
            Wsg[s][r] = __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(
                scales + (static_cast<std::int64_t>(orow0 + r) * kg + g) * 2)));
            Wsu[s][r] = __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(
                scales + (static_cast<std::int64_t>(intermediate + orow0 + r) * kg + g) * 2)));
        }
        q4a8_cp_commit();
    };

    const int a_matrix = lane >> 3;
    const int a_rowoff = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_col    = (a_matrix >> 1) * 16;
    const int b_rowoff = lane & 7;
    const int b_col    = ((lane >> 3) & 1) * 16;
    const int warpt    = warp % kNT;
    const int warpr    = warp / kNT;

    float tg[kMR][4] = {}, tu[kMR][4] = {};
    for (int tt = tid; tt < kBM; tt += kThreads) {
        xss[tt] = (tok0 + tt) < t ? xs[tok0 + tt] : 0.0F;
    }

#pragma unroll
    for (int s = 0; s < kStages; ++s) { stage(s, s); }

#pragma unroll 1
    for (int g = 0; g < groups; ++g) {
        const int s = g % kStages;
        if (g + kStages < groups) { q4a8_cp_wait<kStages - 1>(); } else { q4a8_cp_wait<0>(); }
        __syncthreads();

        const float xl = xss[warpt * 16 + (lane >> 2)];
        const float xh = xss[warpt * 16 + (lane >> 2) + 8];
#pragma unroll
        for (int ks = 0; ks < kG / 32; ++ks) {
            unsigned a0, a1, a2, a3;
            const int ar = warpt * 16 + a_rowoff;
            q4a8_ldm_x4(a0, a1, a2, a3,
                        q4a8_smem(&As[s][ar * kG + q4a8_swz(ar, ks * 32 + a_col)]));
#pragma unroll
            for (int mr = 0; mr < kMR; ++mr) {
                const int br = warpr * (kBN / kNR) + mr * 8 + b_rowoff;
                unsigned bg0, bg1, bu0, bu1;
                q4a8_ldm_x2(bg0, bg1,
                            q4a8_smem(&Bgs[s][br * kG + q4a8_swz(br, ks * 32 + b_col)]));
                q4a8_ldm_x2(bu0, bu1,
                            q4a8_smem(&Bus[s][br * kG + q4a8_swz(br, ks * 32 + b_col)]));
                int gt0 = 0, gt1 = 0, gt2 = 0, gt3 = 0, ut0 = 0, ut1 = 0, ut2 = 0, ut3 = 0;
                q4a8_mma(gt0, gt1, gt2, gt3, a0, a1, a2, a3, bg0, bg1);
                q4a8_mma(ut0, ut1, ut2, ut3, a0, a1, a2, a3, bu0, bu1);
                const int r = warpr * (kBN / kNR) + mr * 8 + (lane & 3) * 2;
                const float wg0 = Wsg[s][r], wg1 = Wsg[s][r + 1];
                const float wu0 = Wsu[s][r], wu1 = Wsu[s][r + 1];
                tg[mr][0] += gt0 * wg0 * xl;
                tg[mr][1] += gt1 * wg1 * xl;
                tg[mr][2] += gt2 * wg0 * xh;
                tg[mr][3] += gt3 * wg1 * xh;
                tu[mr][0] += ut0 * wu0 * xl;
                tu[mr][1] += ut1 * wu1 * xl;
                tu[mr][2] += ut2 * wu0 * xh;
                tu[mr][3] += ut3 * wu1 * xh;
            }
        }
        __syncthreads();
        if (g + kStages < groups) { stage(s, g + kStages); }
    }

#pragma unroll
    for (int mr = 0; mr < kMR; ++mr) {
        const int row = orow0 + warpr * (kBN / kNR) + mr * 8 + (lane & 3) * 2;
        const int tlo = tok0 + warpt * 16 + (lane >> 2);
        const int thi = tlo + 8;
        auto store = [&](int tok, int r, float gv, float uv) {
            if (tok < t) {
                out[static_cast<std::int64_t>(tok) * intermediate + r] =
                    __float2bfloat16_rn(silu(gv) * uv);
            }
        };
        store(tlo, row, tg[mr][0], tu[mr][0]);
        store(tlo, row + 1, tg[mr][1], tu[mr][1]);
        store(thi, row, tg[mr][2], tu[mr][2]);
        store(thi, row + 1, tg[mr][3], tu[mr][3]);
    }
}

} // namespace

void q4_linear_swiglu_a8_launch(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                                cudaStream_t stream) {
    auto scope = ws.scope();
    const Q4A8Workspace scratch =
        allocate_q4_a8_workspace(ws, x.ne[1], static_cast<std::int32_t>(x.ne[0]));
    launch_q4_a8_quantize(x, scratch, stream);
    const std::int32_t intermediate = out.ne[0];
    const std::int32_t k           = static_cast<std::int32_t>(x.ne[0]);
    const std::int32_t t           = x.ne[1];
    const std::int32_t padded_k    = w.padded_shape[1];
    const dim3 grid(static_cast<unsigned>(div_up(t, kBM)),
                    static_cast<unsigned>(div_up(intermediate, kBN)));
    q4_linear_swiglu_a8_kernel<<<grid, kThreads, 0, stream>>>(
        scratch.codes, scratch.scales, static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data),
        intermediate, k, t, padded_k);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
