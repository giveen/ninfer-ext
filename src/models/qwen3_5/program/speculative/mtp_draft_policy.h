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

// Whether rounds of several requests decode without drafts. A model whose routed experts live in
// Host memory pays for every distinct expert a round touches, and every draft column routes to its
// own: on Qwen3.8-Flash-Next a K=3 batch round committed 18%, 39% and 44% fewer tokens per second
// than plain decode at concurrency 2, 4 and 8. Its adaptive policy therefore runs batches as
// ordinary rounds that only append the MTP layer's KV; a fixed draft length keeps drafting, since
// it asks for that length.
[[nodiscard]] constexpr bool mtp_plain_batches(bool adaptive, bool host_resident_experts) noexcept {
    return adaptive && host_resident_experts;
}

// Draft length a request starts on, and the one a batch round uses before its cost is measured.
// Which draft length a batch should use depends on the model, the batch size and the text: on the
// decode-saturation suite K=5 beat K=3 by 14-17% at concurrency 2-8 on Qwen3.8-27B NVFP4, while
// Qwen3.6-35B-A3B lost 13-15% at concurrency 2 and 4 with it. So only two rungs serve batches,
// this one and the window, and the policy picks between them from startup-measured batch times.
inline constexpr std::uint32_t kBatchDraft = 3;

// Index of the default batch rung on a ladder: the longest rung not above `kBatchDraft`, or the
// first. A new request starts on it.
[[nodiscard]] std::size_t mtp_batch_rung(std::span<const std::uint32_t> ladder) noexcept;

// Rungs replayed for more than one request, ascending: the default batch rung, plus the window
// when it is longer. Only these need graphs for every batch size.
[[nodiscard]] std::vector<std::size_t> mtp_batch_rungs(std::span<const std::uint32_t> ladder);

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

    // The rung to run next for lanes with the given acceptance estimates. A single lane chooses
    // among every rung, several lanes among the batch rungs, by the tokens all lanes are expected
    // to commit per second of the batch's measured round. The current rung is kept unless another
    // is predicted to be at least 2% faster. Without measured times for a batch size, several
    // lanes run the initial rung.
    [[nodiscard]] std::size_t select(std::size_t current,
                                     std::span<const MtpAcceptanceEstimate* const> lanes) const;

    // The measured time of a round of `batch_size` (>= 2) requests on batch rung `rung`.
    void set_batch_round_seconds(std::size_t rung, std::uint32_t batch_size, double seconds);

    // Refine round times from single-request rounds as they run. A model whose round cost the
    // startup replays cannot see (host-resident experts: repeated replays hit the expert cache,
    // while real verifies of longer drafts miss it) enables this; others keep the startup times.
    void learn_round_times() noexcept { learning_ = true; }

    // A single-request round on `rung` took `seconds` end to end. A learned round time is the
    // startup time plus an extra (host overhead, expert-cache misses) measured per rung: its first
    // observation sets the extra, later ones move an average. A rung not yet run takes the extra
    // of the nearest measured rung, preferring a shorter one: longer drafts miss at least as much,
    // so that is a lower bound that still lets the policy try the longer rung and measure it.
    void observe_round(std::size_t rung, double seconds);

    // The round time `select` uses for `rung`.
    [[nodiscard]] double round_seconds(std::size_t rung) const;

private:
    std::vector<std::uint32_t> ladder_;
    std::vector<double> round_seconds_; // startup times
    std::vector<double> extra_;         // learned extra per rung, valid where observed_
    std::vector<bool> observed_;
    std::vector<std::size_t> batch_rungs_;
    // batch_seconds_[batch_size][rung]: measured batch round time, 0 where unmeasured.
    std::vector<std::vector<double>> batch_seconds_;
    std::size_t batch_rung_ = 0;
    bool learning_          = false;
};

} // namespace ninfer::models::qwen3_5
