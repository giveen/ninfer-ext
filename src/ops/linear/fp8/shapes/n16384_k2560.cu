// Qwen3.8-Flash-Next (Qwen4Exp) GDN input projection, n = 16384 (query/key/value and the z gate
// stacked), k = 2560 (the hidden width). It is the largest dense GEMM in the model: 50.3% of the
// prefill's dense FLOPs at T = 16384, in every one of the 36 GDN layers.
//
// Route derivation: k = 2560 = 20 x 128, so the MMA k tile, the split-K slices and a GEMV
// ValuesPerLane of 8 (32 x 8 = 256, 2560 / 256 = 10) all divide; n = 16384 divides every row tile
// in use. The A16 ladder is the mid-K one the 5120-wide shapes settled on, and the A8 entry is the
// two-term TMA route: the pair kernel stages both E4M3 terms of one weight tile per pass, and it is
// only reachable above kFp8A8PairMinTokens, which is what `uses_a8` admits (below it the
// two-term route costs two sequential passes and A16 is cheaper). Single-term A8 is not offered:
// this shape is stored for a two-term activation policy.
#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Geometry  = Fp8Geometry<16384, 2560>;
using Gemv      = Fp8A16GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
using Tma64x128 = Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 3, 1>;

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens == 1) return fp8_linear_a16_gemv<Geometry, Gemv>(x, weight, out, stream);
    if (tokens <= 16)
        return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<16, 4, 1>>(x, weight, out,
                                                                              stream);
    if (tokens <= 32)
        return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<32, 4, 1>>(x, weight, out,
                                                                              stream);
    if (tokens <= 64)
        return fp8_linear_a16_mma<Geometry, Fp8A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>(
            x, weight, out, stream);
    if (tokens <= 128)
        return fp8_linear_a16_mma<Geometry, Fp8A16MmaSchedule<64, 64, 64, 32, 16, 2, 2>>(
            x, weight, out, stream);
    fp8_linear_a16_mma<Geometry, Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>(x, weight, out,
                                                                               stream);
}

void launch_a8(const Tensor& x, const Weight& weight, Tensor& out, Fp8A8Workspace scratch,
               cudaStream_t stream) {
    // Above kFp8A8PairMinTokens the two-term workspace makes this the fused pair kernel.
    launch_fp8_a8_tma<Geometry, Tma64x128>(x, weight, out, scratch, stream);
}

bool uses_a8(std::int32_t, std::int32_t max_tokens) { return max_tokens > kFp8A8PairMinTokens; }
} // namespace

const Fp8LinearShape kFp8N16384K2560{16384, 2560, launch_a16, launch_a8, uses_a8, nullptr};
} // namespace ninfer::ops::detail
