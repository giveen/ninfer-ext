#pragma once
#include "ops/softmax_attention/dense/causal_cache/nvfp4/schedule.cuh"

namespace ninfer::ops::detail {
template <class G, int Tokens>
struct Nvfp4KvGroupedInstance {
    static_assert(Tokens > 0 && Tokens * G::GroupSize <= 64);
    // Must stay in sync with plan.cpp's row_tiles/key_shift/split budget and grouped_mma.cuh's
    // RowTiles: Bc must divide (1 << key_shift), and MinBlocks must equal the planner's split
    // budget for this row-tile count.
    static constexpr int kRowTiles = (Tokens * G::GroupSize + 15) / 16;
    using Schedule =
        Nvfp4KvGroupedMmaSchedule<Tokens, kRowTiles <= 2 ? 8 : 4 * kRowTiles,
                                  kRowTiles <= 2 ? 32 : 64, kRowTiles <= 2 ? 2 : 1, true>;
    using Merge = Nvfp4KvMergeSchedule;
};

using Nvfp4KvTiledInstance = Nvfp4KvTiledMmaSchedule<64, 12>;
} // namespace ninfer::ops::detail
