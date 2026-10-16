#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_launch.cuh"

namespace ninfer::ops::detail {
namespace {
// Nemotron-H's attention key and value projection: n = 2 key/value heads x 128, k = 4096.
//
// Same route derivation as n8192_k5376, which is the registered shape this one is copied from:
//
//  - K divisibility. Every A16 route requires k % (32 * ValuesPerLane) == 0. 4096 = 2^12 admits
//    ValuesPerLane = 8 (256) and would also admit 16 or 32; 8 is carried for consistency with the
//    sibling shapes. k % 128 == 0 as well, so the MMA route's K tile divides.
//  - Row divisibility. n = 256 and every row tile in use (32 and 64) divide it.
//  - Dropped routes. The sliced-K route takes a token-capacity precondition that the copied
//    instances do not satisfy at this row count, and the SIMT route's remaining arguments were not
//    verified for this geometry. Neither is needed: the A16 MMA route has no token precondition, so
//    it covers every token count above one.
//
// These schedules are correctness routes, not tuned ones; the shape is untuned.
using Geometry = Fp8Geometry<256, 4096>;
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

const Fp8LinearShape kFp8N256K4096{256, 4096, launch_a16, launch_a8(), uses_a8, nullptr};
} // namespace ninfer::ops::detail
