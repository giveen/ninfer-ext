#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_launch.cuh"

namespace ninfer::ops::detail {
// Gemma 4's MLP down projection under the G0 layout, kept at FP8 in the sensitive layers. The
// schedule notes below were written for the geometry this was copied from; only the extents
// differ, and 5376 and 21504 divide every row tile in use.
namespace {
// Gemma 4's sliding-attention output projection: n = 5376, k = 32 heads x 256.
//
// Same route derivation as n8192_k5376, with the row and K roles exchanged: k = 8192 is a power of
// two, so any ValuesPerLane admits it and 8 is carried for consistency; n = 5376 = 2^8 * 21 admits
// the power-of-two row tiles. The sliced-K and SIMT routes are dropped and uses_a8 reports false.
// Untuned.
using Geometry = Fp8Geometry<5376, 21504>;
using Gemv     = Fp8A16GemvSchedule<4, 4, 8, 4, Fp8CodeCache::Default, 1, 1>;

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens == 1) return fp8_linear_a16_gemv<Geometry, Gemv>(x, weight, out, stream);
    if (tokens <= 64)
        return fp8_linear_a16_mma<Geometry, Fp8A16MmaSchedule<32, 64, 128, 32, 16, 2, 2>>(
            x, weight, out, stream);
    if (tokens <= 96)
        return fp8_linear_a16_mma<Geometry, Fp8A16MmaSchedule<64, 96, 128, 64, 16, 1, 2>>(
            x, weight, out, stream);
    fp8_linear_a16_mma<Geometry, Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>(x, weight, out,
                                                                               stream);
}

void (*launch_a8())(const Tensor&, const Weight&, Tensor&, Fp8A8Workspace, cudaStream_t) {
    return nullptr;
}

bool uses_a8(std::int32_t, std::int32_t) { return false; }
} // namespace

const Fp8LinearShape kFp8N5376K21504{5376, 21504, launch_a16, launch_a8(), uses_a8, nullptr};
} // namespace ninfer::ops::detail
