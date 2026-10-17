// The Lightning 30B's FP8 projections: hidden width 2688 and the widths around it. Same route
// derivation as n4096_k4096 (the registered shape these are copied from): every output width is a
// multiple of the MMA tile in use and K = 2688 or 4096 divides into whole 128-wide steps, the
// sliced-K and SIMT routes are dropped, and uses_a8 reports false, so each is W8A16 -- the
// checkpoint's own assignment for the two mamba mixer projections, and the same route the 74B
// artifact uses for its attention, mixer and head.
#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_launch.cuh"

namespace ninfer::ops::detail {
namespace {

// One lane owns an output row and walks K in 16-value steps; the MMA schedules take over as soon as
// there is more than one token.
using Gemv = Fp8A16GemvSchedule<4, 4, 8, 4, Fp8CodeCache::Default, 1, 1>;

template <class Geometry>
void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    // The GEMV route walks K in `32 * ValuesPerLane` steps plus one guarded tail phase, so it needs
    // only K to divide by ValuesPerLane. Every width in this family does -- 2688, 3712 and 4096 are
    // all multiples of 32 -- so a one-row call takes the GEMV route for all of them. It used to be
    // gated on K dividing by 256, which sent 2688 and 3712 into the MMA schedule at one row: 31 of
    // that schedule's 32 rows empty, and 390-875 GB/s against this route's 1.3 TB/s.
    if constexpr (Geometry::kInputRows % Gemv::kValuesPerLane == 0) {
        if (tokens == 1) return fp8_linear_a16_gemv<Geometry, Gemv>(x, weight, out, stream);
    }
    // A verify pass is 8-9 rows, and the 64-token tile spends 8x the activation staging on 8 rows
    // that exist. A 16-token tile with 8-token warps and 1024 blocks instead of 512 measured
    // 100.4 -> 77.8 us on the (16384, 5376) probe (49% -> 63% of DRAM peak), and the two are the
    // same kernel, so the wide-token route is untouched above this bound.
    if (tokens <= 16)
        return fp8_linear_a16_mma<Geometry, Fp8A16MmaSchedule<16, 16, 128, 16, 8, 2, 2>>(
            x, weight, out, stream);
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

const Fp8LinearShape kFp8N10304K2688{10304,       2688,    launch_a16<Fp8Geometry<10304, 2688>>,
                                     launch_a8(), uses_a8, nullptr};
const Fp8LinearShape kFp8N2688K4096{2688,        4096,    launch_a16<Fp8Geometry<2688, 4096>>,
                                    launch_a8(), uses_a8, nullptr};
const Fp8LinearShape kFp8N4096K2688{4096,        2688,    launch_a16<Fp8Geometry<4096, 2688>>,
                                    launch_a8(), uses_a8, nullptr};
const Fp8LinearShape kFp8N256K2688{256,         2688,    launch_a16<Fp8Geometry<256, 2688>>,
                                   launch_a8(), uses_a8, nullptr};
const Fp8LinearShape kFp8N3712K2688{3712,        2688,    launch_a16<Fp8Geometry<3712, 2688>>,
                                    launch_a8(), uses_a8, nullptr};
const Fp8LinearShape kFp8N2688K3712{2688,        3712,    launch_a16<Fp8Geometry<2688, 3712>>,
                                    launch_a8(), uses_a8, nullptr};
const Fp8LinearShape kFp8N131072K2688{131072,      2688,    launch_a16<Fp8Geometry<131072, 2688>>,
                                      launch_a8(), uses_a8, nullptr};

} // namespace ninfer::ops::detail
