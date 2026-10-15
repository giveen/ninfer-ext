#pragma once
#include "models/qwen3_5/program/program.h"
#include "ninfer/types.h"

#include <cstdint>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

[[nodiscard]] std::vector<GraphExecutionProfile> ordinary_graph_profiles(std::uint32_t capacity);
[[nodiscard]] std::vector<GraphExecutionProfile> mtp_graph_profiles(std::uint32_t capacity,
                                                                    std::uint32_t draft_window);
// The masked-draft backend's graph profiles for a `capacity`-token cache and `draft_window`
// drafts. Throws std::invalid_argument when `capacity` or `draft_window` is 0, or `draft_window`
// exceeds 15; `backend` picks the DFlash2 or the DFlash topology classes.
[[nodiscard]] std::vector<GraphExecutionProfile> dflash_graph_profiles(SpeculativeBackend backend,
                                                                       std::uint32_t capacity,
                                                                       std::uint32_t draft_window);

} // namespace ninfer::models::qwen3_5::detail
