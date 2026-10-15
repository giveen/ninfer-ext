#pragma once

#include "ops/softmax_attention/dense/causal_cache/fp8/schedule.cuh"

namespace ninfer::ops::detail {

template <class G, int Tokens>
struct Fp8KvGroupedInstance {
    static_assert(Tokens > 0 && Tokens * G::GroupSize <= 64);
    static constexpr int kRowTiles = (Tokens * G::GroupSize + 15) / 16;
    using Schedule                 = Fp8KvGroupedMmaSchedule<Tokens,
                                             kRowTiles == 4   ? 16
                                                             : kRowTiles == 3 ? 12
                                                                              : 8,
                                             Tokens == 1 ? 32 : 64, Tokens == 1 ? 2 : 1>;
    // The merge covers the head dimension with grid.y = div_up(kHeadDim, DChunk) blocks of DChunk
    // columns (one column per thread, DChunk <= 256) and grid.x = QHeads, so the 16-head
    // geometries take four 64-wide blocks where the 24-head one keeps a single 256-wide block.
    using Merge                    = Fp8KvMergeSchedule<G::QHeads == 24 ? 256 : 64>;
};

using Fp8KvTiledInstance = Fp8KvTiledMmaSchedule<kMxfp8TiledQueryRows, 64>;

} // namespace ninfer::ops::detail
