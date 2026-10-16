#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_launch.cuh"

namespace ninfer::ops::detail {
namespace {
// Nemotron-H's output head: n = 131072 (the vocabulary), k = 4096 (the hidden width). It is the
// family's FP8 head, and the closed vocabulary makes its output the pass's widest tensor.
//
// Same route derivation as the existing vocabulary shape n262144_k5376, whose n is likewise a power
// of two: every row tile in use divides 131072, and k = 4096 = 2^12 admits the GEMV ValuesPerLane's
// 256 and the MMA K tile's 128. The sliced-K and SIMT routes are dropped; the A16 MMA route has no
// token precondition, so it covers every token count above one. Untuned.
using Geometry = Fp8Geometry<131072, 4096>;
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

const Fp8LinearShape kFp8N131072K4096{131072, 4096, launch_a16, launch_a8(), uses_a8, nullptr};
} // namespace ninfer::ops::detail
