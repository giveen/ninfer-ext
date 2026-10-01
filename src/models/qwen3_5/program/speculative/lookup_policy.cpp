#include "models/qwen3_5/program/speculative/lookup_policy.h"

#include <algorithm>
#include <limits>

namespace ninfer::models::qwen3_5 {
namespace {

// Priors before the engine has measured its own text: Strata's controller corner of the measured
// range, deliberately conservative so an unmeasured bucket does not out-bid a measured MTP.
constexpr std::array<double, LookupAcceptance::kBuckets> kPriorRate = {0.35, 0.60, 0.80, 0.92};

} // namespace

double LookupAcceptance::rate(std::uint32_t match) const noexcept {
    const std::size_t index = bucket(match);
    return (accepted_[index] + kPriorN * kPriorRate[index]) / (drafted_[index] + kPriorN);
}

void LookupAcceptance::observe(std::uint32_t match, std::uint32_t drafted,
                               std::uint32_t accepted) noexcept {
    if (drafted == 0) { return; }
    const std::size_t index = bucket(match);
    drafted_[index]  = kDecay * drafted_[index] + static_cast<double>(drafted);
    accepted_[index] = kDecay * accepted_[index] +
                       static_cast<double>(std::min(accepted, drafted));
    if (observations_[index] < std::numeric_limits<std::uint32_t>::max()) {
        ++observations_[index];
    }
}

double LookupAcceptance::expected_tokens(std::uint32_t drafts, double q) noexcept {
    double expected = 1.0; // the verify pass always commits at least the correction token
    double prefix   = 1.0;
    for (std::uint32_t i = 0; i < drafts; ++i) {
        prefix *= q;
        expected += prefix;
    }
    return expected;
}

std::uint32_t LookupPolicy::choose(std::uint32_t available, std::uint32_t match,
                                   double mtp_expected, bool force) const noexcept {
    if (available == 0U || match < min_match_) { return 0U; }
    const std::uint32_t extent = std::min(available, kMaxDrafts);
    if (force) { return extent; }
    const double q = acceptance_.rate(match);
    if (LookupAcceptance::expected_tokens(extent, q) > mtp_expected * (1.0 + margin_)) {
        return extent;
    }
    // A promising bucket the engine has barely measured is tried anyway: the measured rate may be
    // far better than the prior, and the only way to find out is to verify its drafts once or
    // twice. Bounded to kProbes rounds per bucket per request.
    if (acceptance_.observations(match) < kProbes && q >= kProbeRate) { return extent; }
    return 0U;
}

} // namespace ninfer::models::qwen3_5
