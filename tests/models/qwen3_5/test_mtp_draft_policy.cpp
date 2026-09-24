#include "models/qwen3_5/program/speculative/mtp_draft_policy.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

namespace q36 = ninfer::models::qwen3_5;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

template <class Fn>
bool throws(Fn&& fn) {
    try {
        fn();
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

using Ladder = std::vector<std::uint32_t>;

// A lane that has verified `rounds` rounds of `extent` drafts of which `accepted` were accepted.
q36::MtpAcceptanceEstimate lane(std::uint32_t extent, std::uint32_t accepted, int rounds = 60) {
    q36::MtpAcceptanceEstimate estimate;
    for (int i = 0; i < rounds; ++i) { estimate.observe(extent, accepted); }
    return estimate;
}

// Single-request round time of the dense 27B graphs measured on the RTX 5090: about 12.9 ms plus
// 1.9 ms per drafted token, so the ladder {2, 3, 4, 7} costs 16.7, 18.6, 20.5 and 26.2 ms.
std::vector<double> dense_round_seconds() { return {0.0167, 0.0186, 0.0205, 0.0262}; }

void test_ladder() {
    expect(q36::mtp_draft_ladder(7, true) == Ladder({2, 3, 4, 7}), "window 7 ladder");
    expect(q36::mtp_draft_ladder(5, true) == Ladder({2, 3, 4, 5}), "window 5 ladder");
    expect(q36::mtp_draft_ladder(4, true) == Ladder({2, 3, 4}), "window 4 ladder");
    expect(q36::mtp_draft_ladder(3, true) == Ladder({2, 3}), "window 3 ladder");
    expect(q36::mtp_draft_ladder(2, true) == Ladder({2}), "window 2 ladder");
    expect(q36::mtp_draft_ladder(1, true) == Ladder({1}), "window 1 ladder");
    expect(q36::mtp_draft_ladder(7, false) == Ladder({7}), "fixed window is one rung");
    expect(throws([] { (void)q36::mtp_draft_ladder(0, true); }), "window 0 is invalid");
    expect(throws([] { (void)q36::mtp_draft_ladder(q36::kMtpDecodeMaximumDrafts + 1, true); }),
           "window above the frame domain is invalid");

    expect(q36::mtp_batch_rung(Ladder({2, 3, 4, 7})) == 1, "batch rung is K=3 on the full ladder");
    expect(q36::mtp_batch_rung(Ladder({2})) == 0 && q36::mtp_batch_rung(Ladder({1})) == 0,
           "a window below K=3 is its own batch rung");
    expect(q36::mtp_batch_rung(Ladder({7})) == 0, "a fixed window above K=3 is its own batch rung");
}

void test_estimate() {
    // No evidence: the 70% prior gives 1 + .7 + .49 + ...
    const q36::MtpAcceptanceEstimate fresh;
    double expected = 1.0;
    double prefix   = 1.0;
    for (int j = 0; j < 4; ++j) {
        prefix *= 0.7;
        expected += prefix;
    }
    expect(std::abs(fresh.expected_tokens(4) - expected) < 1e-12, "prior expectation");
    expect(fresh.expected_tokens(0) == 1.0, "zero drafts commit one token");

    // Everything accepted: expectation approaches K+1 and grows with K.
    const q36::MtpAcceptanceEstimate high = lane(7, 7);
    expect(high.expected_tokens(7) > 6.5 && high.expected_tokens(7) < 8.0, "high acceptance");
    expect(high.expected_tokens(7) > high.expected_tokens(4), "monotone in K");

    // Nothing accepted: only the target token is worth having.
    const q36::MtpAcceptanceEstimate none = lane(4, 0);
    expect(none.expected_tokens(4) < 1.2, "zero acceptance");

    // A position is tested only behind an accepted prefix: rejecting draft 0 says nothing about
    // draft 3, which keeps inheriting position 0's low estimate rather than the prior.
    expect(none.expected_tokens(4) < fresh.expected_tokens(4), "rejections lower the tail");

    // Extents beyond the frame domain and accepted > extent are clamped, never out of bounds.
    q36::MtpAcceptanceEstimate clamped;
    clamped.observe(99, 99);
    expect(clamped.expected_tokens(99) < 9.0, "clamped observation");

    // A phase change is followed: long acceptance, then a run of rejections.
    q36::MtpAcceptanceEstimate phase = lane(7, 7);
    const double before              = phase.expected_tokens(7);
    for (int i = 0; i < 20; ++i) { phase.observe(7, 1); }
    expect(phase.expected_tokens(7) < before / 2.0, "estimate tracks a phase change");

    phase.reset();
    expect(std::abs(phase.expected_tokens(4) - expected) < 1e-12, "reset restores the prior");
}

void test_policy_selection() {
    const q36::MtpDraftPolicy policy(Ladder({2, 3, 4, 7}), dense_round_seconds());
    expect(policy.initial_rung() == 1, "adaptive requests start on K=3");
    expect(q36::MtpDraftPolicy(Ladder({7}), {0.02}).initial_rung() == 0,
           "a fixed policy starts on its only rung");

    const q36::MtpAcceptanceEstimate high = lane(7, 7);
    const q36::MtpAcceptanceEstimate mid  = lane(4, 3);
    const q36::MtpAcceptanceEstimate low  = lane(4, 0);
    const auto select = [&](std::size_t current, std::vector<const q36::MtpAcceptanceEstimate*> l) {
        return policy.select(current, l);
    };
    expect(select(1, {&high}) == 3, "copy-heavy output climbs to the longest draft");
    expect(select(1, {&low}) == 0, "unpredictable output falls to the shortest draft");
    expect(select(0, {&high}) == 3 && select(3, {&low}) == 0, "any rung can move to any rung");

    // Drafts accepted up to the third and never the fourth put the optimum on K=3 from both sides.
    expect(select(0, {&mid}) == 1 && select(3, {&mid}) == 1,
           "acceptance ending at three picks K=3");

    // Several lanes always run the batch rung, whatever their acceptance.
    expect(select(3, {&high, &high}) == 1 && select(0, {&low, &low, &low}) == 1, "batches run K=3");
}

void test_hysteresis() {
    // Round times proportional to expected tokens make the rungs equally fast: the policy stays
    // where it is from either side. Making the far rung 1% cheaper is still inside the margin;
    // 5% cheaper is not.
    const q36::MtpAcceptanceEstimate a = lane(4, 2);
    const std::vector<const q36::MtpAcceptanceEstimate*> lanes{&a};
    const double r2   = a.expected_tokens(2);
    const double r4   = a.expected_tokens(4);
    const auto policy = [&](double cheaper_long) {
        return q36::MtpDraftPolicy(Ladder({2, 4}), {0.005 * r2, 0.005 * r4 * cheaper_long});
    };
    expect(policy(1.0).select(0, lanes) == 0 && policy(1.0).select(1, lanes) == 1,
           "equally fast rungs do not move the policy");
    expect(policy(0.99).select(0, lanes) == 0, "a 1% gain is inside the margin");
    expect(policy(0.95).select(0, lanes) == 1, "a 5% gain moves the policy");

    // Costs decide, not just tokens: an expensive long rung loses to a cheap short one.
    const q36::MtpDraftPolicy expensive(Ladder({2, 7}), {0.010, 0.040});
    const q36::MtpAcceptanceEstimate high = lane(7, 7);
    const std::vector<const q36::MtpAcceptanceEstimate*> high_lane{&high};
    expect(expensive.select(1, high_lane) == 0, "measured round cost outweighs extra tokens");
}

void test_policy_validation() {
    using Seconds = std::vector<double>;
    expect(throws([] { (void)q36::MtpDraftPolicy(Ladder({}), Seconds{}); }), "empty ladder");
    expect(throws([] { (void)q36::MtpDraftPolicy(Ladder({4, 2}), Seconds{1.0, 1.0}); }),
           "descending ladder");
    expect(throws([] { (void)q36::MtpDraftPolicy(Ladder({2, 2}), Seconds{1.0, 1.0}); }),
           "duplicate rungs");
    expect(throws([] { (void)q36::MtpDraftPolicy(Ladder({2, 8}), Seconds{1.0, 1.0}); }),
           "rung above the frame domain");
    expect(throws([] { (void)q36::MtpDraftPolicy(Ladder({2, 4}), Seconds{1.0}); }),
           "round times must cover every rung");
    expect(throws([] { (void)q36::MtpDraftPolicy(Ladder({2, 4}), Seconds{1.0, 0.0}); }),
           "round times must be positive");

    const q36::MtpDraftPolicy policy(Ladder({2, 4}), Seconds{1.0, 1.0});
    const q36::MtpAcceptanceEstimate a;
    const std::vector<const q36::MtpAcceptanceEstimate*> one{&a};
    expect(throws([&] { (void)policy.select(2, one); }), "rung out of range");
    expect(
        throws([&] { (void)policy.select(0, std::vector<const q36::MtpAcceptanceEstimate*>{}); }),
        "empty batch");
}

} // namespace

int main() {
    test_ladder();
    test_estimate();
    test_policy_selection();
    test_hysteresis();
    test_policy_validation();
    if (failures != 0) {
        std::cerr << failures << " MTP draft policy checks failed\n";
        return 1;
    }
    std::cout << "MTP draft policy checks passed\n";
    return 0;
}
