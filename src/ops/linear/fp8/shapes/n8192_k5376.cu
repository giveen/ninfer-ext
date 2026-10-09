#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_launch.cuh"

namespace ninfer::ops::detail {
namespace {
// Gemma 4's sliding-attention query projection: n = 32 heads x 256, k = 5376.
//
// The schedules are the nearest registered geometry's, with the one value this shape's dimensions
// force changed and the routes whose preconditions cannot be met by copying dropped:
//
//  - K divisibility. Every A16 route requires k % (32 * ValuesPerLane) == 0, so 5376 = 2^8 * 21
//    admits ValuesPerLane = 8 (256) and not 16 (512) or 32 (1024). The GEMV schedule therefore
//    carries 8 where the reference geometry carries 16.
//  - Row divisibility. 8192 is a power of two, so every row tile in use divides it.
//  - Dropped routes. The sliced-K route takes a token-capacity precondition that the copied
//    instances do not satisfy here, and the SIMT route's remaining arguments were not verified for
//    this geometry. Neither is needed: the A16 MMA route has no token precondition, so it covers
//    every token count above one.
//
// These schedules are correctness routes, not tuned ones; the shape is untuned.
using Geometry = Fp8Geometry<8192, 5376>;
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

// The A8 routes are deliberately absent: uses_a8 returns false, which the dispatcher consults before
// any A8 launch, so a caller asking for AllowA8 or AllowA4 gets this qualified A16 route rather than
// an unqualified quantized one.
void (*launch_a8())(const Tensor&, const Weight&, Tensor&, Fp8A8Workspace, cudaStream_t) {
    return nullptr;
}

bool uses_a8(std::int32_t, std::int32_t) { return false; }
} // namespace

const Fp8LinearShape kFp8N8192K5376{8192, 5376, launch_a16, launch_a8(), uses_a8, nullptr};
} // namespace ninfer::ops::detail
