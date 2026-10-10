#include "ops/linear/bf16/bf16_launch.cuh"
#include "ops/linear/bf16/bf16_shapes.h"

// The Gemma 4 vision tower's two BF16 projections: the patch embedding (1152 x 768) over every patch of
// an image (9 to 2520), and the soft-token projection into the text width (5376 x 1152) over its pooled
// tokens (1 to 280). Images are encoded whole, so the MMA route carries nearly every call; a single
// soft token or a handful of patches takes the GEMV or SIMT route. Untuned beyond that.

namespace ninfer::ops::detail {
namespace {
using Gemv = Bf16GemvSchedule<4, 1, 8, 4, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                              Bf16PhaseOrder::RowSwizzled, 1, 1, 1, 2>;
using C8   = Bf16SimtSchedule<4, 1, 4, 4, 1, 4, Bf16SimtActivationAccess::WarpPacked,
                              Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 1, 2>;
using Mma  = Bf16MmaSchedule<64, 128, 64, 32, 32, 2, 2, Cache::cg, Cache::cg,
                             Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;

template <class Geometry>
Bf16Launch select(std::int32_t tokens) {
    if (tokens == 1) return launch_bf16_gemv<Geometry, Gemv>;
    if (tokens <= 8) return launch_bf16_simt<Geometry, 8, C8>;
    return launch_bf16_mma<Geometry, Mma>;
}
} // namespace

Bf16Launch select_bf16_n1152_k768(std::int32_t t) { return select<Bf16Geometry<1152, 768>>(t); }
Bf16Launch select_bf16_n5376_k1152(std::int32_t t) { return select<Bf16Geometry<5376, 1152>>(t); }

} // namespace ninfer::ops::detail
