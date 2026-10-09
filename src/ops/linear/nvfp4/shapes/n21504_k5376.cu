#include "ops/linear/nvfp4/nvfp4_shapes.h"
#include "ops/linear/nvfp4/nvfp4_launch.cuh"

namespace ninfer::ops::detail {
namespace {
// Gemma 4's MLP gate and up halves: n = 21504, k = 5376, one shape serving both.
//
// The artifact binds the MLP as two halves rather than one fused 43008-row parent, so this is the
// shape the unfused path needs: a linear per half, then gelu_mul. The fused parent (n43008_k5376) is
// registered too, for the recipe that fuses them.
//
// Same route derivation as n43008_k5376: ValuesPerLane 8 because 32 * 8 divides k5376, one
// capacity-32 SIMT schedule behind the chunked launcher, and no A4 routes. Untuned.
using Geometry = Nvfp4Geometry<21504, 5376>;
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

const Nvfp4LinearShape kNvfp4N21504K5376{21504, 5376, launch_nvfp4_a16_chunks<32, select_a16>,
                                         nullptr, uses_a4};
} // namespace ninfer::ops::detail
