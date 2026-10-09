#include "ops/linear/nvfp4/nvfp4_shapes.h"
#include "ops/linear/nvfp4/nvfp4_launch.cuh"

namespace ninfer::ops::detail {
namespace {
// Gemma 4's MLP down projection: n = 5376, k = 21504.
//
// Same route derivation as n43008_k5376. Here the K tile is the wide side: 21504 = 2^9 * 42 admits
// both 8 (256) and 16 (512), and 8 is carried for consistency with the other Gemma shapes. Untuned.
using Geometry = Nvfp4Geometry<5376, 21504>;
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

const Nvfp4LinearShape kNvfp4N5376K21504{5376, 21504, launch_nvfp4_a16_chunks<32, select_a16>,
                                         nullptr, uses_a4};
} // namespace ninfer::ops::detail
