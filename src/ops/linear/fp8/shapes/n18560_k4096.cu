#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_launch.cuh"

namespace ninfer::ops::detail {
namespace {
// Nemotron-H's mamba in-projection: n = 18560, k = 4096. This is the widest row count of the family's
// dense projections and the one shape here whose n is not a power of two;
// 18560 = 2^7 * 145, so 18560 % 64 == 0 and 18560 % 32 == 0, which is what the row tiles require.
//
// Same route derivation as n8192_k5376, which is the registered shape this one is copied from:
// k = 4096 = 2^12 admits ValuesPerLane 8 (256 divides it) and the MMA K tile's 128 divides it; the
// sliced-K and SIMT routes are dropped and uses_a8 reports false. Untuned.
using Geometry = Fp8Geometry<18560, 4096>;
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

const Fp8LinearShape kFp8N18560K4096{18560, 4096, launch_a16, launch_a8(), uses_a8, nullptr};
} // namespace ninfer::ops::detail
