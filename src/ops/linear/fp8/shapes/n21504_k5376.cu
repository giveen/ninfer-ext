#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_launch.cuh"

namespace ninfer::ops::detail {
namespace {
// Gemma 4's MLP up/gate projection under the G0 layout: the halves are separate objects, and the
// sensitive layers keep them at FP8 rather than NVFP4.
//
// The schedules are the nearest registered geometry's, with the one value this shape's dimensions
// force changed and the routes whose preconditions cannot be met by copying dropped:
//
//  - K divisibility. 5376 = 2^8 * 21 admits ValuesPerLane = 8 and not 16 or 32, as for the query
//    projection.
//  - Row divisibility. 21504 = 2^9 * 42 is a multiple of every row tile in use.
//  - Dropped routes. As for the query projection: the A16 MMA route has no token precondition, so it
//    covers every token count above one.
//
// These schedules are correctness routes, not tuned ones; the shape is untuned.
using Geometry = Fp8Geometry<21504, 5376>;
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

const Fp8LinearShape kFp8N21504K5376{21504, 5376, launch_a16, launch_a8(), uses_a8, nullptr};
} // namespace ninfer::ops::detail
