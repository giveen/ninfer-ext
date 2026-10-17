#include "ops/linear/nvfp4/nvfp4_shapes.h"
#include "ops/linear/nvfp4/nvfp4_launch.cuh"

namespace ninfer::ops::detail {
namespace {
// Nemotron-H's shared expert, gate and up half: n = 5376 = the shared expert's intermediate width,
// k = 4096 = the hidden width. The artifact binds the shared MLP as one up matrix (the family's
// relu2 MLP is not gated) plus its down matrix, so this is the shape the unfused path needs.
//
// Route derivation. Every ValuesPerLane on this op is the K tile divided into 32-value lanes, so
// k = 4096 = 2^12 admits 8 (256) and would also admit 16 or 32; 8 is carried from the registered
// Gemma shapes this is copied from. k % 256 == 0 also satisfies the SIMT kernel's own
// `kInputRows % kValuesPerPhase` constraint, and 5376 % 128 == 0 satisfies its row constraint.
// The capacity-32 SIMT schedule is driven by the chunked launcher for any token count.
//
// The A4 routes are deliberately absent: uses_a4 reports false, so a quantized-activation policy
// gets the A16 route rather than an unqualified one. Untuned.
using Geometry = Nvfp4Geometry<5376, 4096>;
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

const Nvfp4LinearShape kNvfp4N5376K4096{5376, 4096, launch_nvfp4_a16_chunks<32, select_a16>,
                                        nullptr, uses_a4};
} // namespace ninfer::ops::detail
