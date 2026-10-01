#pragma once

#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/program/program.h"

#include <cstdint>

namespace ninfer::models::qwen3_5 {

inline constexpr std::uint32_t kPrefillChunkAlignment    = 128;
inline constexpr std::uint32_t kMaximumMtpDraftTokens    = 7;
// Widest column count one MTP forward/verify call may carry. The MTP layer drafts at most
// kMaximumMtpDraftTokens tokens, but a chained round verifies MTP drafts plus a suffix-drafter
// continuation, so the alignment forward runs over the whole committed prefix.
inline constexpr std::uint32_t kMaximumMtpForwardDrafts  = 15;
inline constexpr std::uint32_t kMaximumDFlashDraftTokens = 15;
// Abort salvage publishes the live state only when it covers enough committed work that the
// saved rebuild outweighs the checkpoint's retention cost.
inline constexpr std::uint32_t kSalvageMinFrontier       = 1024;

} // namespace ninfer::models::qwen3_5

namespace ninfer::models::qwen3_5::detail {
using ContractAccess = RuntimeContractAccess;

[[nodiscard]] inline std::uint32_t backend_frontier_at(SpeculativeBackend backend,
                                                       std::uint32_t main_frontier) noexcept {
    if (backend == SpeculativeBackend::Mtp) { return main_frontier == 0 ? 0U : main_frontier - 1U; }
    return backend == SpeculativeBackend::DFlash ? main_frontier : 0U;
}

} // namespace ninfer::models::qwen3_5::detail
