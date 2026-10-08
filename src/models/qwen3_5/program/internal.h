#pragma once

#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/program/program.h"

#include <cstdint>

namespace ninfer::models::qwen3_5 {

inline constexpr std::uint32_t kPrefillChunkAlignment    = 128;
inline constexpr std::uint32_t kMaximumMtpDraftTokens    = 7;
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
    // EAGLE3 keeps the full draft KV like DFlash: the next round's alignment forward recomputes
    // its anchor seam from fresh target features, so no one-behind bridge lag is retained.
    return backend == SpeculativeBackend::DFlash || backend == SpeculativeBackend::Eagle3
               ? main_frontier
               : 0U;
}

} // namespace ninfer::models::qwen3_5::detail
