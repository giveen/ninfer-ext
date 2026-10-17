#include "ops/linear/nvfp4/nvfp4_shapes.h"
#include "ops/linear/nvfp4/nvfp4_launch.cuh"

namespace ninfer::ops::detail {
namespace {
// Nemotron-H's shared expert, down half: n = 4096 = the hidden width, k = 5376 = the shared expert's
// intermediate width. This is the wide-K sibling of n5376_k4096, and the second of the two shapes
// the family's shared MLP needs.
//
// Route derivation. k = 5376 = 2^8 * 21 admits ValuesPerLane 8 (256 divides it) and not 16 (512),
// which is the same K divisibility the registered Gemma shapes carry; its SIMT precondition
// `kInputRows % kValuesPerPhase` is satisfied for the same reason, and 4096 % 128 == 0 satisfies the
// row constraint. The capacity-32 SIMT schedule is driven by the chunked launcher for any token
// count, and uses_a4 reports false so a quantized-activation policy gets this A16 route. Untuned.
using Geometry = Nvfp4Geometry<4096, 5376>;
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

const Nvfp4LinearShape kNvfp4N4096K5376{4096, 5376, launch_nvfp4_a16_chunks<32, select_a16>,
                                        nullptr, uses_a4};
} // namespace ninfer::ops::detail
