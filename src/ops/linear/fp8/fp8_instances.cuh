#pragma once
#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear/fp8/fp8_schedule.cuh"

namespace ninfer::ops::detail {
// The two-term E4M3 activation route's fused schedule: a 128x128 tile holding one accumulator set
// per term, so both terms' MMAs consume a single weight load and the split-K plan holds one partial
// plane per term.
using Fp8A8TmaPair = Fp8A8TmaSplitKSchedule<
    Fp8A8TmaMmaSchedule<kFp8A8PairBlockTokens, kFp8A8PairBlockRows, 128, 2, 4, 2, 1>,
    kFp8A8PairWaveCtas, 4, 8>;
static_assert(kFp8A8PairPartialBytes == std::size_t(kFp8A8PairTerms) * Fp8A8TmaPair::kPartialBytes,
              "the pair route's declared partial planes must match its split-K schedule");

using Fp8A8T64R128K128 =
    Fp8A8MmaSchedule<64, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                     Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
using Fp8A8T16R64K128 = Fp8A8MmaSchedule<16, 64, 128, 1, 2, 2, 4, Cache::cg, Cache::cg,
                                         Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
using Fp8A8T16R128K128 =
    Fp8A8MmaSchedule<16, 128, 128, 1, 4, 2, 2, Cache::cg, Cache::cg,
                     Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
using Fp8A8T32R32K128 = Fp8A8MmaSchedule<32, 32, 128, 1, 2, 3, 3, Cache::cg, Cache::cg,
                                         Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
using Fp8A8T32R64K128 = Fp8A8MmaSchedule<32, 64, 128, 1, 2, 3, 2, Cache::cg, Cache::cg,
                                         Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
using Fp8A8T32R128K128 =
    Fp8A8MmaSchedule<32, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                     Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
using Fp8A8T64R64K128 = Fp8A8MmaSchedule<64, 64, 128, 2, 2, 3, 2, Cache::cg, Cache::cg,
                                         Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
using Fp8A8T64R64K128S2 =
    Fp8A8MmaSchedule<64, 64, 128, 2, 2, 2, 3, Cache::cg, Cache::cg,
                     Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
using Fp8A8T64R128K256 =
    Fp8A8MmaSchedule<64, 128, 256, 2, 4, 2, 1, Cache::cg, Cache::cg,
                     Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
template <int Tokens, int Warps, int Stages>
using Fp8SlicedInstance = Fp8A16SlicedKMmaSchedule<
    Warps, Tokens, (Stages * (16 * Warps * 64 + Warps * Tokens * 128) > 48 * 1024 ? 1 : 2),
    Cache::ca, Cache::cg, Fp8ActivationStage::PaddedZero, Stages>;
} // namespace ninfer::ops::detail
