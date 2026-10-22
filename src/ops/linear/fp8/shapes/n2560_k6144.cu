// Qwen3.8-Flash-Next (Qwen4Exp) the GDN and QSA output projection (k = the 6144-wide mixer output); 6.4% of the prefill's dense FLOPs at T = 16384.
//
// Same route derivation as n16384_k2560: k = 6144 and n = 2560 both divide the MMA k tile, the
// split-K slices, a GEMV ValuesPerLane of 8 and every row tile in use. The A8 entry is the two-term
// TMA route and is admitted only above kFp8A8PairMinTokens, where the pair kernel fuses both
// activation terms; below it A16 is cheaper than two sequential passes. Single-term A8 is not
// offered: these shapes are stored for a two-term activation policy.
#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Geometry  = Fp8Geometry<2560, 6144>;
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
    launch_fp8_a8_tma<Geometry, Tma64x128>(x, weight, out, scratch, stream);
}

bool uses_a8(std::int32_t, std::int32_t max_tokens) { return max_tokens > kFp8A8PairMinTokens; }
} // namespace

const Fp8LinearShape kFp8N2560K6144{2560, 6144, launch_a16, launch_a8, uses_a8, nullptr};
} // namespace ninfer::ops::detail
