#pragma once

// One thread per output element: the Op is a handful of BF16 reads per write over a block's width,
// so the grid is (channels x window x batch) threads and nothing is tiled. See the contract in
// include/ninfer/ops/grouped_tap_apply.h for the formula.

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace ninfer::ops::detail {

struct GroupedTapParams {
    const __nv_bfloat16* x;     // [H, W, B]
    const __nv_bfloat16* base;  // [2, taps*H], side-major taps
    const __nv_bfloat16* delta; // [2*taps*G, W, B]
    __nv_bfloat16* out;         // [H, W, B]
    std::int32_t channels;
    std::int32_t window;
    std::int32_t batch;
    std::int32_t groups;
    std::int32_t taps;
    std::int32_t block_size;
    std::int32_t group_width;
    std::int32_t side;
    std::int32_t rows; // 2*taps*groups, delta's leading extent
};

__global__ void grouped_tap_apply_kernel(GroupedTapParams p, std::int64_t owners) {
    const std::int64_t at = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at >= owners) { return; }
    // The engine's tensor order is dim0-fastest: `at` is already `h + H*(t + W*b)`, so `x`/`out`
    // are read at `at` and a tap is exactly `k * H` earlier. `delta` and `base` keep their own
    // extents: `delta` is indexed over its row, `base` is the artifact's payload, whose flat order
    // is side-major taps.
    const std::int32_t h     = static_cast<std::int32_t>(at % p.channels);
    const std::int64_t rest  = at / p.channels;
    const std::int32_t t     = static_cast<std::int32_t>(rest % p.window);
    const std::int32_t b     = static_cast<std::int32_t>(rest / p.window);
    const std::int32_t g     = h / p.group_width;
    const std::int32_t reach = min(p.taps - 1, t % p.block_size);
    float sum                = 0.0F;
    for (std::int32_t k = 0; k <= reach; ++k) {
        const std::int32_t row  = (p.side * p.taps + k) * p.groups + g;
        const std::int32_t slot = p.side * p.taps * p.channels + k * p.channels + h;
        const float base        = __bfloat162float(p.base[slot]);
        const float delta =
            __bfloat162float(p.delta[static_cast<std::int64_t>(row) + p.rows * (t + p.window * b)]);
        const float previous =
            __bfloat162float(p.x[at - static_cast<std::int64_t>(k) * p.channels]);
        sum += (base + delta) * previous;
    }
    p.out[at] = __float2bfloat16_rn(sum);
}

} // namespace ninfer::ops::detail
