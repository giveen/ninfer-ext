#include "ops/linear/nvfp4/nvfp4_shapes.h"
#include "ops/linear/nvfp4/nvfp4_launch.cuh"

namespace ninfer::ops::detail {
namespace {
// The Lightning 30B DFlash companion's tap fusion: n = 2688 = the tower's width, k = 16128 = six
// taps' residual stream. The companion stores it as ModelOpt NVFP4 (packed E2M1 with per-16 E4M3
// scales and one FP32 divisor), and a draft round is the round's largest single read, so the codes
// are preserved rather than decoded to BF16.
//
// Route derivation. k % 256 == 0, so the chunked SIMT launcher's phase tiling is exact (the same
// constraint the registered Gemma and Nemotron shapes satisfy); 2688 % 128 == 0 satisfies its row
// constraint. The A4 routes are absent: the draft's projection takes the A16 route, and a
// quantized-activation policy falls back to it instead of an unqualified one. Untuned.
using Geometry = Nvfp4Geometry<2688, 16128>;
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

const Nvfp4LinearShape kNvfp4N2688K16128{2688, 16128, launch_nvfp4_a16_chunks<32, select_a16>,
                                         nullptr, uses_a4};
} // namespace ninfer::ops::detail
