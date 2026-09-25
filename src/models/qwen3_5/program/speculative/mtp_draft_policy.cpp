#include "models/qwen3_5/program/speculative/mtp_draft_policy.h"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen3_5 {
namespace {

constexpr double kDecay           = 0.9;
constexpr double kPriorAcceptance = 0.7;
constexpr double kPriorWeight     = 2.0;
constexpr double kShrinkWeight    = 3.0;
constexpr double kSwitchMargin    = 1.02;
constexpr double kRoundTimeWeight = 0.2; // weight of a new round in a learned round time

constexpr std::array<std::uint32_t, 4> kAdaptiveRungs{2, 3, 4, 7};

} // namespace

std::size_t mtp_batch_rung(std::span<const std::uint32_t> ladder) noexcept {
    std::size_t rung = 0;
    for (std::size_t i = 0; i < ladder.size(); ++i) {
        if (ladder[i] <= kBatchDraft) { rung = i; }
    }
    return rung;
}

std::vector<std::size_t> mtp_batch_rungs(std::span<const std::uint32_t> ladder) {
    std::vector<std::size_t> rungs{mtp_batch_rung(ladder)};
    if (!ladder.empty() && rungs.front() + 1U < ladder.size()) { rungs.push_back(ladder.size() - 1U); }
    return rungs;
}

std::vector<std::uint32_t> mtp_draft_ladder(std::uint32_t draft_window, bool adaptive) {
    if (draft_window == 0 || draft_window > kMtpDecodeMaximumDrafts) {
        throw std::invalid_argument("MTP draft window is outside the supported domain");
    }
    std::vector<std::uint32_t> ladder;
    if (adaptive) {
        for (const std::uint32_t rung : kAdaptiveRungs) {
            if (rung < draft_window) { ladder.push_back(rung); }
        }
    }
    ladder.push_back(draft_window);
    return ladder;
}

void MtpAcceptanceEstimate::reset() noexcept {
    tested_   = {};
    accepted_ = {};
}

void MtpAcceptanceEstimate::observe(std::uint32_t extent, std::uint32_t accepted) noexcept {
    extent   = std::min<std::uint32_t>(extent, kMtpDecodeMaximumDrafts);
    accepted = std::min(accepted, extent);
    for (std::size_t j = 0; j < tested_.size(); ++j) {
        tested_[j] *= kDecay;
        accepted_[j] *= kDecay;
    }
    // Draft j is tested when drafts 0..j-1 were accepted; it is accepted when j < accepted.
    for (std::uint32_t j = 0; j < extent && j <= accepted; ++j) {
        tested_[j] += 1.0;
        if (j < accepted) { accepted_[j] += 1.0; }
    }
}

double MtpAcceptanceEstimate::expected_tokens(std::uint32_t k) const noexcept {
    k = std::min<std::uint32_t>(k, kMtpDecodeMaximumDrafts);
    // Every position is evidence about how predictable this lane's text is, so a pooled estimate
    // converges within a round or two; each position then moves off the pool only as far as its
    // own tests justify.
    double pooled_tested   = 0.0;
    double pooled_accepted = 0.0;
    for (std::size_t j = 0; j < tested_.size(); ++j) {
        pooled_tested += tested_[j];
        pooled_accepted += accepted_[j];
    }
    const double pooled =
        (pooled_accepted + kPriorAcceptance * kPriorWeight) / (pooled_tested + kPriorWeight);
    double expected = 1.0;
    double prefix   = 1.0;
    for (std::uint32_t j = 0; j < k; ++j) {
        prefix *= (accepted_[j] + kShrinkWeight * pooled) / (tested_[j] + kShrinkWeight);
        expected += prefix;
    }
    return expected;
}

MtpDraftPolicy::MtpDraftPolicy(std::vector<std::uint32_t> ladder, std::vector<double> round_seconds)
    : ladder_(std::move(ladder)), round_seconds_(std::move(round_seconds)) {
    if (ladder_.empty() || !std::ranges::is_sorted(ladder_) ||
        std::ranges::adjacent_find(ladder_) != ladder_.end() || ladder_.front() == 0 ||
        ladder_.back() > kMtpDecodeMaximumDrafts) {
        throw std::invalid_argument("MTP draft ladder must be ascending and within the window");
    }
    if (round_seconds_.size() != ladder_.size() ||
        std::ranges::any_of(round_seconds_, [](double seconds) { return !(seconds > 0.0); })) {
        throw std::invalid_argument("MTP round times must be positive for every rung");
    }
    batch_rung_  = mtp_batch_rung(ladder_);
    batch_rungs_ = mtp_batch_rungs(ladder_);
    extra_.assign(ladder_.size(), 0.0);
    observed_.assign(ladder_.size(), false);
}

void MtpDraftPolicy::set_batch_round_seconds(std::size_t rung, std::uint32_t batch_size,
                                             double seconds) {
    if (std::ranges::find(batch_rungs_, rung) == batch_rungs_.end() || batch_size < 2 ||
        batch_size > kMaximumConcurrency || !(seconds > 0.0)) {
        throw std::invalid_argument("MTP batch round time is outside its domain");
    }
    if (batch_seconds_.size() <= batch_size) { batch_seconds_.resize(batch_size + 1U); }
    batch_seconds_[batch_size].resize(ladder_.size(), 0.0);
    batch_seconds_[batch_size][rung] = seconds;
}

void MtpDraftPolicy::observe_round(std::size_t rung, double seconds) {
    if (rung >= ladder_.size()) {
        throw std::invalid_argument("MTP round observation is outside the ladder");
    }
    if (!learning_ || !(seconds > 0.0)) { return; }
    // A round faster than its startup replay is noise, not a negative cost.
    const double extra = std::max(0.0, seconds - round_seconds_[rung]);
    extra_[rung] =
        observed_[rung] ? extra_[rung] + kRoundTimeWeight * (extra - extra_[rung]) : extra;
    observed_[rung] = true;
}

double MtpDraftPolicy::round_seconds(std::size_t rung) const {
    if (rung >= ladder_.size()) { throw std::invalid_argument("MTP rung is outside the ladder"); }
    if (!learning_) { return round_seconds_[rung]; }
    std::optional<std::size_t> source;
    for (std::size_t r = rung + 1; r-- > 0 && !source;) {
        if (observed_[r]) { source = r; }
    }
    for (std::size_t r = rung + 1; r < ladder_.size() && !source; ++r) {
        if (observed_[r]) { source = r; }
    }
    return round_seconds_[rung] + (source ? extra_[*source] : 0.0);
}

std::size_t MtpDraftPolicy::select(std::size_t current,
                                   std::span<const MtpAcceptanceEstimate* const> lanes) const {
    if (current >= ladder_.size() || lanes.empty()) {
        throw std::invalid_argument("MTP draft policy selection is outside its domain");
    }
    if (ladder_.size() == 1) { return batch_rung_; }
    if (lanes.size() == 1) {
        const auto rate = [&](std::size_t rung) {
            return lanes.front()->expected_tokens(ladder_[rung]) / round_seconds(rung);
        };
        std::size_t best    = current;
        double best_rate    = rate(current);
        const double margin = best_rate * kSwitchMargin;
        for (std::size_t rung = 0; rung < ladder_.size(); ++rung) {
            if (rung == current) { continue; }
            const double candidate = rate(rung);
            if (candidate > margin && candidate > best_rate) {
                best      = rung;
                best_rate = candidate;
            }
        }
        return best;
    }

    // Every lane commits its own accepted prefix, so a batch round's yield is the lanes' summed
    // expectation; its cost is the measured round of this batch size.
    const std::size_t batch = lanes.size();
    if (batch >= batch_seconds_.size() || batch_seconds_[batch].empty()) { return batch_rung_; }
    const std::vector<double>& seconds = batch_seconds_[batch];
    const auto rate = [&](std::size_t rung) {
        double tokens = 0.0;
        for (const MtpAcceptanceEstimate* lane : lanes) {
            tokens += lane->expected_tokens(ladder_[rung]);
        }
        return tokens / seconds[rung];
    };
    const bool current_serves_batch = std::ranges::find(batch_rungs_, current) != batch_rungs_.end() &&
                                      seconds[current] > 0.0;
    std::size_t best = current_serves_batch ? current : batch_rung_;
    if (!(seconds[best] > 0.0)) { return batch_rung_; }
    double best_rate    = rate(best);
    const double margin = current_serves_batch ? best_rate * kSwitchMargin : best_rate;
    for (const std::size_t rung : batch_rungs_) {
        if (rung == best || !(seconds[rung] > 0.0)) { continue; }
        const double candidate = rate(rung);
        if (candidate > margin && candidate > best_rate) {
            best      = rung;
            best_rate = candidate;
        }
    }
    return best;
}

} // namespace ninfer::models::qwen3_5
