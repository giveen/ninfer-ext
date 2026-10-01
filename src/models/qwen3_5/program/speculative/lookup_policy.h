#pragma once

// Acceptance model and per-lane choice for the prompt-lookup (suffix) draft source.
//
// A lookup proposal is only worth verifying when it is expected to commit more tokens than the
// drafts it replaces. The MTP backend already carries that estimate per sequence
// (`MtpAcceptanceEstimate`); this type carries the lookup side and makes the comparison.
//
// Acceptance depends strongly on how long the matched suffix was: a trigram repeat is a guess, a
// twenty-token repeat is usually verbatim. Buckets by match length, with a prior until the engine
// has measured its own, are how Strata's `DraftPolicy` and `Controller` both model it, and the
// measured shape (a monotone rise from ~0.3 to ~0.9) is what we start from.

#include <array>
#include <cstdint>

namespace ninfer::models::qwen3_5 {

// Probability that a lookup draft is accepted given its whole prefix was accepted.
//
// The rate must be the *conditional* per-position probability, because `expected_tokens` sums
// q + q^2 + ... . A round that drafts `n` and commits `accepted` contributes `accepted` to the
// numerator and one "round that did not accept everything" to the denominator when
// `accepted < n`; this is the geometric estimate Strata's DraftPolicy uses, and it reproduces the
// realized mean accepted per round (`q / (1 - q) ~ accepted / rounds`). Using `accepted / drafted`
// instead understates q by the window width and would reject profitable rounds.
class LookupAcceptance {
public:
    static constexpr std::uint32_t kBuckets = 4;

    [[nodiscard]] static std::uint32_t bucket(std::uint32_t match) noexcept {
        return match < 6U ? 0U : match < 12U ? 1U : match < 24U ? 2U : 3U;
    }

    // P(accept) for one draft whose match had this length.
    [[nodiscard]] double rate(std::uint32_t match) const noexcept;

    // Rounds observed for this match bucket. Zero means the prior is still the only evidence.
    [[nodiscard]] std::uint32_t observations(std::uint32_t match) const noexcept {
        return observations_[bucket(match)];
    }

    // `drafted` drafts from a match of `match` were verified; `accepted` were committed.
    void observe(std::uint32_t match, std::uint32_t drafted, std::uint32_t accepted) noexcept;

    // One target token plus each draft whose whole prefix is accepted.
    [[nodiscard]] static double expected_tokens(std::uint32_t drafts, double q) noexcept;

private:
    static constexpr double kPriorN = 4.0; // prior worth four rounds, so a few real ones win
    static constexpr double kDecay  = 0.97;
    std::array<double, kBuckets> accepted_{};        // drafts committed in this bucket
    std::array<double, kBuckets> partial_rounds_{};  // rounds that did not accept everything
    std::array<std::uint32_t, kBuckets> observations_{};
};

class LookupPolicy {
public:
    // Largest lookup window the MTP frame can verify (its own window is at most seven).
    static constexpr std::uint32_t kMaxDrafts = 15;
    // Rounds a promising but unmeasured match bucket is tried before its measured rate may veto it.
    static constexpr std::uint32_t kProbes = 3;
    // A bucket is "promising" from its prior (or measurement) at or above this rate. Without this,
    // a conservative prior that the policy never contradicts would keep a bucket it would win
    // permanently disabled.
    static constexpr double kProbeRate = 0.80;
    // Probing costs a round, so it is only worth it when the lookup round is not much more
    // expensive than the round it replaces. A wide lookup round on host-resident experts can cost
    // several ordinary rounds, and then a probe is a loss, not a measurement.
    static constexpr double kMaxProbeCostRatio = 2.0;

    LookupPolicy() = default;
    explicit LookupPolicy(std::uint32_t min_match, double margin = 0.0)
        : min_match_(min_match), margin_(margin) {}

    // Extent of lookup drafts to verify for a lane, or 0 to keep the alternative.
    //
    // `available` is the proposal length and `match` its match length. The alternative commits
    // `alternative_tokens` per round at `cost_ratio` times the lookup round's cost (1 when the
    // alternative is an MTP draft in the same round; the measured lookup/ordinary round-time ratio
    // when the alternative is an ordinary one-token round). Lookup is taken only when its expected
    // committed tokens exceed `alternative_tokens * cost_ratio * (1 + margin)`. `force` is the
    // `always` measurement mode.
    [[nodiscard]] std::uint32_t choose(std::uint32_t available, std::uint32_t match,
                                       double alternative_tokens, double cost_ratio,
                                       bool force) const noexcept;

    void observe(std::uint32_t match, std::uint32_t drafted, std::uint32_t accepted) noexcept {
        acceptance_.observe(match, drafted, accepted);
    }

    [[nodiscard]] double rate(std::uint32_t match) const noexcept {
        return acceptance_.rate(match);
    }
    [[nodiscard]] std::uint32_t min_match() const noexcept { return min_match_; }

private:
    LookupAcceptance acceptance_;
    std::uint32_t min_match_ = 8;
    double margin_           = 0.0;
};

} // namespace ninfer::models::qwen3_5
