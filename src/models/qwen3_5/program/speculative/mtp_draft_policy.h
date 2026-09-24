#pragma once

// Draft-length policy for the concurrent MTP round.
//
// A round drafts K tokens and verifies the previous round's drafts at width K+1. Each K is a
// separate captured graph, so the policy chooses among a short ladder of K values before every
// round. The choice maximizes expected committed tokens per second: the tokens a lane commits per
// round follow from its measured acceptance, and the round time of each rung is measured at
// startup on the real graphs. Selection is host-only and never changes what is committed: the
// target still verifies every draft, so any K samples the same distribution.

#include "models/qwen3_5/program/round_buffers.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_5 {

// Draft lengths captured for a configured window, ascending. A fixed policy has one rung. The
// adaptive ladder is the subset of {2, 3, 4, 7} below the window, plus the window itself: on a
// dense and an MoE model these reach within a few percent of the best per-workload choice among K
// in [2, 7], and every further rung costs another graph family.
std::vector<std::uint32_t> mtp_draft_ladder(std::uint32_t draft_window, bool adaptive);

// Draft length a round uses whenever more than one request decodes together, and the length a
// request starts on. Measured on the dense 27B graphs with mixed workloads, no longer draft raised
// aggregate throughput at concurrency 2, 4 or 8: the wider verify costs more than the extra
// accepted tokens repay, and one lane's rejections slow every other lane in the batch.
inline constexpr std::uint32_t kBatchDraft = 3;

// Index of that rung on a ladder: the longest rung not above `kBatchDraft`, or the first. Only this
// rung is ever replayed for more than one request, so only it needs graphs for every batch size.
[[nodiscard]] std::size_t mtp_batch_rung(std::span<const std::uint32_t> ladder) noexcept;

// Per-lane estimate of how far a round's drafts get accepted. Position j is tested only when the
// drafts before it were accepted, so its acceptance is a conditional probability. Counts decay each
// round so the estimate follows phase changes within a request, such as a prose explanation
// followed by a tool call.
class MtpAcceptanceEstimate {
public:
    void reset() noexcept;
    // `extent` drafts were verified this round and the first `accepted` of them were accepted.
    void observe(std::uint32_t extent, std::uint32_t accepted) noexcept;
    // Expected tokens committed by a round that verifies `k` drafts: one target token plus each
    // draft whose whole prefix is accepted. Positions never tested inherit the deepest measured
    // conditional acceptance.
    [[nodiscard]] double expected_tokens(std::uint32_t k) const noexcept;

private:
    std::array<double, kMtpDecodeMaximumDrafts> tested_{};
    std::array<double, kMtpDecodeMaximumDrafts> accepted_{};
};

class MtpDraftPolicy {
public:
    // `round_seconds[r]` is the measured time of a single-request round on ladder rung r.
    MtpDraftPolicy() = default;
    MtpDraftPolicy(std::vector<std::uint32_t> ladder, std::vector<double> round_seconds);

    [[nodiscard]] const std::vector<std::uint32_t>& ladder() const noexcept { return ladder_; }

    [[nodiscard]] std::size_t rung_count() const noexcept { return ladder_.size(); }

    // Rung a request starts on, before any acceptance has been measured: the longest rung that is
    // not above `kBatchDraft`.
    [[nodiscard]] std::size_t initial_rung() const noexcept { return batch_rung_; }

    // The rung to run next for lanes with the given acceptance estimates. A single lane keeps its
    // current rung unless another is predicted to be at least 2% faster; several lanes always run
    // the initial rung.
    [[nodiscard]] std::size_t select(std::size_t current,
                                     std::span<const MtpAcceptanceEstimate* const> lanes) const;

private:
    std::vector<std::uint32_t> ladder_;
    std::vector<double> round_seconds_;
    std::size_t batch_rung_ = 0;
};

} // namespace ninfer::models::qwen3_5
