#include "core/weight.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/linear/nvfp4/nvfp4_a16_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

using Geometry              = Nvfp4N34816K5120;
constexpr int kIntermediate = Geometry::kOutputRows / 2;

// Each m16 tile holds eight gate rows in its upper half and their up rows in its lower half, so
// the lane that owns a gate accumulator also owns the matching up accumulator and SwiGLU needs no
// exchange. The 8 * RowTiles gate rows of a block share scale lines, and so do their up rows.
template <class Schedule>
struct SwiGluTile {
    static constexpr int kGateRows = Schedule::kRowTiles * 8;
    static constexpr int kBlocks   = kIntermediate / kGateRows;
    static_assert((kIntermediate % 128) == 0);

    __nv_bfloat16* out;

    __device__ __forceinline__ int row(int block, int mma_row) const {
        const int tile  = mma_row / 16;
        const int inner = mma_row % 16;
        const int gate =
            nvfp4_a16_mma_scale_line_row<kGateRows>(block, tile * 8 + (inner & 7));
        return inner < 8 ? gate : gate + kIntermediate;
    }

    __device__ __forceinline__ void
    finish(const int (&rows)[Schedule::kRowTiles][2],
           const float (&values)[Schedule::kRowTiles][Schedule::kTokenTiles][4], int lane_token,
           int tokens) const {
#pragma unroll
        for (int mt = 0; mt < Schedule::kRowTiles; ++mt) {
#pragma unroll
            for (int nt = 0; nt < Schedule::kTokenTiles; ++nt) {
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int token = nt * 8 + lane_token + e;
                    if (token < tokens) {
                        out[static_cast<std::int64_t>(token) * kIntermediate + rows[mt][0]] =
                            __float2bfloat16_rn(silu(values[mt][nt][e]) * values[mt][nt][e + 2]);
                    }
                }
            }
        }
    }
};

template <class Schedule>
void launch(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Tile = SwiGluTile<Schedule>;
    const Tile tile{static_cast<__nv_bfloat16*>(out.data)};
    nvfp4_a16_mma_kernel<Geometry, Schedule, Tile><<<Tile::kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), 1.0F / weight.weight_scale_divisor, tile,
        x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void nvfp4_linear_swiglu_small_t_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream) {
    if (x.ne[1] <= 8) return launch<Nvfp4A16MmaSchedule<1, 1, 16>>(x, weight, out, stream);
    launch<Nvfp4A16MmaSchedule<4, 2, 2>>(x, weight, out, stream);
}

} // namespace ninfer::ops::detail
