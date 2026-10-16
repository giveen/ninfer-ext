// The Nemotron-H latent round trip: `fc1_latent` [1024, 4096] and `fc2_latent` [4096, 1024], the
// two projections that bracket a MoE bank's routed experts, at the model's hidden width (4096) and
// its latent width (1024). Both used to reach the dense fallback (bf16_generic.cuh), which is 253
// us for the 8 MB of weights either one reads: at one token it is four blocks of 256 threads, one
// scalar BF16 load per FMA.
//
// The parallelism to take is over output rows, which is the shape's own axis: 1024 and 4096 rows
// are enough blocks for the device once a block is one warp, and 32 rows per warp make every
// 32-byte weight load a whole sector, so the fallback's over-fetch (one 2-byte element per lane,
// every lane in a different sector) disappears.
//
// What is deliberately *not* taken from the gemv and SIMT schedules beside this file is their
// reduction. Both hand one row's K axis to the lanes of a warp and sum the lane partials with
// `warp_reduce_sum`, which reassociates. This family's forward walk holds the dense fallback's own
// arithmetic as its reference -- one FP32 accumulator per output element, K ascending, one BF16
// rounding at the store -- so a warp-reduced row disagrees with the route it replaces wherever a
// partial sum lands the other side of a BF16 rounding boundary: host-measured on these two shapes,
// ~0.05% and ~0.02% of output elements move by one BF16 ulp. Instead every lane owns a whole row
// and walks it in ascending order through `accumulate_bf16_packs<..., 1>`: with a single
// accumulator chain, gemv.cuh's own accumulate loop writes `accumulators[0]` in ascending column
// order, which is exactly the fallback's `fmaf(w, x, acc)` chain. The rest of what this file takes
// from the same header -- `Bf16GemvPack<16>`, `load_bf16_pack`, `load_bf16_weight_pack` -- is what
// moves 32 bytes per instruction instead of two.
//
// `TokensPerThread` is the per-token choice: a weight pack is loaded once and reused across the
// batch, which is what keeps a wide pass from re-reading the matrix per token. It does not touch
// the order -- every (token, row) keeps its own single accumulator and its own ascending K walk --
// so both routes are bit-identical to the fallback and to each other. One token per thread is the
// route that asks for the device's SMs at single-token decode, where there are only `N` output
// elements to go around; four is worth it as soon as there is a second token to share a weight pack
// with.

#include "core/device.h"
#include "ops/linear/bf16/bf16_config.h"
#include "ops/linear/bf16/bf16_gemv.cuh"
#include "ops/linear/bf16/bf16_launch.h"
#include "ops/linear/bf16/bf16_shapes.h"

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace ninfer::ops::detail {
namespace {

// One 16-wide pack is 32 bytes: one lane covers a whole memory sector per weight load, and
// `Bf16GemvPack<16>` is the widest pack gemv.cuh's loaders know.
constexpr int kValuesPerStep = 16;

// One warp per block. The whole point of this route at single-token decode is that the 1024 or 4096
// output rows it has to spread over the device become 1024 or 4096 *lanes*, and the more blocks the
// scheduler can place on distinct SMs, the more outstanding sector requests the partition can hold.
// A block wider than a warp would trade that for occupancy this kernel cannot use: it is one
// dependent FMA chain per lane, so there is nothing to switch to.
constexpr int kThreadsPerCta = 32;

// The two schedule knobs are template parameters rather than literals so a development sweep can
// instantiate the candidates around them with one compilation; production selects the defaults and
// nothing else. `ThreadsPerCta` is how many output rows a block covers, which is the device's share
// of the work at a fixed row count, and `Unroll` is how many 32-byte steps are in flight per lane
// before the FMA chain consumes them.
constexpr int kUnrollSteps = 8;

// One lane owns one output row of `TokensPerThread` consecutive tokens and walks K ascending with a
// single FP32 accumulator per (token, row). The tail token group reads the last token again rather
// than predicating the inner loop: the repeated chain is computed and discarded, where a guard
// would sit in the body of every step.
template <class Geometry, int TokensPerThread, int ThreadsPerCta = kThreadsPerCta,
          int Unroll = kUnrollSteps>
__global__ __launch_bounds__(ThreadsPerCta) void bf16_latent_row_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ out, const std::int32_t tokens) {
    static_assert((Geometry::kInputRows % kValuesPerStep) == 0);
    static_assert((Geometry::kOutputRows % ThreadsPerCta) == 0);
    constexpr int kSteps = Geometry::kInputRows / kValuesPerStep;

    const std::int32_t row    = static_cast<std::int32_t>(blockIdx.y) * ThreadsPerCta + threadIdx.x;
    const std::int32_t token0 = static_cast<std::int32_t>(blockIdx.x) * TokensPerThread;
    const std::int32_t last   = tokens - 1;

    const __nv_bfloat16* token_base[TokensPerThread];
#pragma unroll
    for (int slot = 0; slot < TokensPerThread; ++slot) {
        token_base[slot] =
            x + static_cast<std::int64_t>(min(token0 + slot, last)) * Geometry::kInputRows;
    }
    const __nv_bfloat16* weight_row =
        weight + static_cast<std::int64_t>(row) * Geometry::kInputRows;

    float accumulator[TokensPerThread][1] = {};
#pragma unroll(Unroll)
    for (int step = 0; step < kSteps; ++step) {
        const int offset = step * kValuesPerStep;
        const Bf16GemvPack<kValuesPerStep> weight_pack =
            load_bf16_weight_pack<Bf16WeightCache::Default, kValuesPerStep>(weight_row + offset);
#pragma unroll
        for (int slot = 0; slot < TokensPerThread; ++slot) {
            const Bf16GemvPack<kValuesPerStep> activation_pack =
                load_bf16_pack<kValuesPerStep>(token_base[slot] + offset);
            accumulate_bf16_packs<kValuesPerStep, 1>(weight_pack, activation_pack,
                                                     accumulator[slot]);
        }
    }

#pragma unroll
    for (int slot = 0; slot < TokensPerThread; ++slot) {
        const std::int32_t token = token0 + slot;
        if (token < tokens) {
            out[static_cast<std::int64_t>(token) * Geometry::kOutputRows + row] =
                __float2bfloat16_rn(accumulator[slot][0]);
        }
    }
}

template <class Geometry, int TokensPerThread, int ThreadsPerCta = kThreadsPerCta>
void launch_bf16_latent_row(const Tensor& x, const Weight& weight, Tensor& out,
                            cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    const dim3 grid((tokens + TokensPerThread - 1) / TokensPerThread,
                    Geometry::kOutputRows / ThreadsPerCta);
    bf16_latent_row_kernel<Geometry, TokensPerThread, ThreadsPerCta>
        <<<grid, ThreadsPerCta, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                             static_cast<const __nv_bfloat16*>(weight.qdata),
                                             static_cast<__nv_bfloat16*>(out.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry>
Bf16Launch select_latent(std::int32_t tokens) {
    if (tokens == 1) return launch_bf16_latent_row<Geometry, 1>;
    return launch_bf16_latent_row<Geometry, 4>;
}

} // namespace

Bf16Launch select_bf16_n1024_k4096(std::int32_t tokens) {
    return select_latent<Bf16Geometry<1024, 4096>>(tokens);
}

Bf16Launch select_bf16_n4096_k1024(std::int32_t tokens) {
    return select_latent<Bf16Geometry<4096, 1024>>(tokens);
}

} // namespace ninfer::ops::detail
