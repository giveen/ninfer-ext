#include "ops/linear/nvfp4/nvfp4_shapes.h"
#include "ops/linear/nvfp4/nvfp4_launch.cuh"

namespace ninfer::ops::detail {
namespace {
// Gemma 4's fused gate/up MLP parent: n = 2 x 21504, k = 5376.
//
// Route derivation. Every ValuesPerLane on this op is the K tile divided into 32-value lanes, so
// k = 5376 = 2^8 * 21 admits 8 (256) and not 16 (512). The reference geometry's per-token exact SIMT
// schedules are replaced by one capacity-32 schedule, which the chunked launcher drives for any
// token count, because those exact intervals are tuning results for the reference geometry.
//
// The A4 routes are deliberately absent: uses_a4 reports false, which is the shape's own field, so a
// quantized-activation policy gets the A16 route rather than an unqualified one, and the TMA route's
// geometry enumeration is not needed. Untuned.
using Geometry = Nvfp4Geometry<43008, 5376>;
using Gemv =
    Nvfp4GemvSchedule<8, 2, 8, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>;
using C32 = Nvfp4SimtSchedule<4, 1, 2, 8, 8, 1, Nvfp4SimtActivationAccess::TokenPacked,
                              Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                              Nvfp4SimtBlockOrder::TokenTilesContiguous, 4>;

Nvfp4Launch select_a16(std::int32_t tokens) {
    if (tokens == 1) return launch_nvfp4_gemv<Geometry, Gemv>;
    return launch_nvfp4_simt<Geometry, 32, C32, false>;
}

bool uses_a4(std::int32_t, std::int32_t) { return false; }
} // namespace

const Nvfp4LinearShape kNvfp4N43008K5376{43008, 5376, launch_nvfp4_a16_chunks<32, select_a16>,
                                         nullptr, uses_a4};
} // namespace ninfer::ops::detail
