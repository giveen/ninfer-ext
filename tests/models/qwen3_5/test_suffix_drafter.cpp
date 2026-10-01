#include "models/qwen3_5/program/speculative/lookup_policy.h"
#include "models/qwen3_5/program/speculative/suffix_drafter.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
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

void expect_near(double value, double wanted, double tolerance, std::string_view message) {
    if (std::abs(value - wanted) <= tolerance) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << " (" << value << " != " << wanted << ")\n";
}

using Tokens = std::vector<ninfer::TokenId>;

std::vector<ninfer::TokenId> propose(q36::SuffixDrafter& drafter, std::uint32_t max_drafts) {
    std::vector<ninfer::TokenId> out(max_drafts);
    const std::uint32_t count = drafter.propose(max_drafts, out.data());
    out.resize(count);
    return out;
}

void test_no_repeat() {
    q36::SuffixDrafter drafter(1024);
    drafter.sync(Tokens{1, 2, 3, 4, 5, 6, 7, 8});
    expect(drafter.propose(4, nullptr) == 0, "distinct tokens propose nothing");
    expect(drafter.match_length() == 0, "no match length without a repeat");
}

void test_short_history() {
    q36::SuffixDrafter drafter(1024);
    drafter.sync(Tokens{1, 2, 3});
    std::array<ninfer::TokenId, 4> out{};
    expect(drafter.propose(4, out.data()) == 0, "fewer than four tokens propose nothing");
}

void test_exact_repeat() {
    q36::SuffixDrafter drafter(1024);
    // A B C D E | A B C D : the second ABCD repeats the first four tokens, so the next token is E.
    drafter.sync(Tokens{1, 2, 3, 4, 5, 1, 2, 3, 4});
    const std::vector<ninfer::TokenId> proposal = propose(drafter, 4);
    expect(!proposal.empty() && proposal[0] == 5, "proposal continues the earlier occurrence");
    // The match is the re-synced ABCD: four tokens (extended past the trigram), and the proposal
    // continues past it into the current suffix.
    expect(drafter.match_length() == 4, "match length is the extended repeat");
    expect(proposal.size() == 4, "proposal fills the window");
    expect(proposal[0] == 5 && proposal[1] == 1 && proposal[2] == 2 && proposal[3] == 3,
           "proposal is the earlier occurrence's continuation");
}

void test_max_drafts() {
    q36::SuffixDrafter drafter(1024);
    drafter.sync(Tokens{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4});
    expect(propose(drafter, 2).size() == 2, "proposal honors the window");
    expect(propose(drafter, 5).size() == 5, "a larger window proposes more");
}

void test_most_recent_wins() {
    q36::SuffixDrafter drafter(1024);
    // Two occurrences of the trigram (1,2,3): the older continues with 4, the newer with 7. The
    // current suffix ends in the same trigram, so the newer continuation is the proposal.
    drafter.sync(Tokens{1, 2, 3, 4, 9, 9, 9, 1, 2, 3, 7, 1, 2, 3});
    const std::vector<ninfer::TokenId> proposal = propose(drafter, 1);
    expect(!proposal.empty() && proposal[0] == 7, "the most recent occurrence wins");
}

void test_incremental_sync_matches_at_once() {
    const Tokens history{5, 4, 3, 2, 1, 5, 4, 3, 2, 1, 5, 4, 3};
    q36::SuffixDrafter incremental(1024);
    // The round loop re-syncs the growing ledger, so pass growing prefixes.
    for (std::size_t i = 1; i <= history.size(); ++i) {
        incremental.sync(std::span<const ninfer::TokenId>(history.data(), i));
    }
    q36::SuffixDrafter once(1024);
    once.sync(history);
    const std::vector<ninfer::TokenId> a = propose(incremental, 6);
    const std::vector<ninfer::TokenId> b = propose(once, 6);
    expect(a == b, "chunked sync matches a single sync");
    expect(incremental.indexed() == once.indexed(), "indexed count is the same either way");
}

void test_reset() {
    q36::SuffixDrafter drafter(256);
    drafter.sync(Tokens{1, 2, 3, 4, 1, 2, 3, 4});
    drafter.reset();
    expect(drafter.indexed() == 0, "reset drops the index");
    expect(drafter.propose(4, nullptr) == 0, "reset drops proposals");
}

void test_sync_prefix_is_idempotent() {
    q36::SuffixDrafter drafter(256);
    const Tokens history{1, 2, 3, 4, 1, 2, 3, 4};
    drafter.sync(history);
    const std::vector<ninfer::TokenId> before = propose(drafter, 4);
    drafter.sync(history); // the round loop re-syncs the whole ledger
    const std::vector<ninfer::TokenId> after = propose(drafter, 4);
    expect(before == after && drafter.indexed() == history.size(), "re-syncing a prefix appends nothing");
}

void test_propose_after_tail() {
    // A tail stands in for MTP drafts: the search suffix is history followed by the tail, and the
    // earlier occurrence's continuation is proposed.
    q36::SuffixDrafter drafter(1024);
    drafter.sync(Tokens{1, 2, 3, 4, 5, 6, 9, 9, 1, 2, 3, 4});
    std::array<ninfer::TokenId, 8> out{};
    const std::uint32_t count =
        drafter.propose_after(Tokens{5, 6}, 8, out.data());
    expect(count >= 2 && out[0] == 9 && out[1] == 9,
           "propose_after continues the earlier occurrence of history+tail");
    // An empty tail is exactly propose().
    q36::SuffixDrafter plain(1024);
    plain.sync(Tokens{1, 2, 3, 4, 5, 1, 2, 3, 4});
    std::array<ninfer::TokenId, 8> a{};
    std::array<ninfer::TokenId, 8> b{};
    const std::uint32_t na = plain.propose(8, a.data());
    const std::uint32_t nb = plain.propose_after({}, 8, b.data());
    expect(na == nb && std::equal(a.begin(), a.begin() + na, b.begin()),
           "an empty tail matches propose()");
}

void test_policy_priors() {
    q36::LookupPolicy policy(3);
    expect_near(policy.rate(3), 0.35, 1e-12, "short-match prior");
    expect_near(policy.rate(10), 0.60, 1e-12, "mid-match prior");
    expect_near(policy.rate(20), 0.80, 1e-12, "long-match prior");
    expect_near(policy.rate(40), 0.92, 1e-12, "longest-match prior");
    expect_near(q36::LookupAcceptance::expected_tokens(0, 0.9), 1.0, 1e-12,
                "no drafts commit the correction token");
    expect_near(q36::LookupAcceptance::expected_tokens(3, 0.5), 1.0 + 0.5 + 0.25 + 0.125, 1e-12,
                "geometric expectation");
}

void test_policy_choice() {
    q36::LookupPolicy policy(8);
    // A weak match does not beat a healthy MTP.
    expect(policy.choose(4, 8, 3.0, 1.0, false) == 0, "a short match loses to a strong MTP");
    // A long match does.
    expect(policy.choose(4, 30, 2.0, 1.0, false) == 4, "a long match beats a moderate MTP");
    // Below the minimum match the proposal is not considered at all, even in measurement mode.
    expect(policy.choose(4, 4, 1.0, 1.0, false) == 0, "a match below the minimum is rejected");
    expect(policy.choose(4, 4, 1.0, 1.0, true) == 0, "the minimum applies in measurement mode too");
    // No proposal.
    expect(policy.choose(0, 30, 1.0, 1.0, true) == 0, "an empty proposal is never used");
    // Measurement mode ignores the value comparison.
    expect(policy.choose(3, 30, 100.0, 1.0, true) == 3, "force uses the proposal");
}

void test_policy_costs() {
    // The break-even is the lookup round's cost relative to the round it replaces. A wide lookup
    // round that costs eight ordinary rounds must commit more than that to be worth running.
    q36::LookupPolicy policy(3);
    expect(policy.choose(15, 30, 1.0, 8.0, false) == 15, "a deep match pays for a costly round");
    expect(policy.choose(15, 10, 1.0, 8.0, false) == 0, "a short match does not");
    // A cheap lookup round (device-resident experts) lowers the bar and may probe.
    expect(policy.choose(15, 20, 1.0, 1.5, false) == 15, "a cheap round accepts a mid match");
    // Probing is a measurement, so it is not paid for when the round is expensive.
    expect(policy.choose(15, 20, 1.0, 8.0, false) == 0, "no probing when the round is costly");
}

void test_policy_explores() {
    // A promising but unmeasured bucket is tried even when the MTP looks stronger, so its measured
    // rate can contradict the prior; a few failures then stop it.
    q36::LookupPolicy policy(8);
    expect(policy.choose(4, 20, 100.0, 1.0, false) == 4, "an unmeasured promising bucket is probed");
    for (int i = 0; i < static_cast<int>(q36::LookupPolicy::kProbes); ++i) {
        policy.observe(20, 4, 0);
    }
    expect(policy.choose(4, 20, 100.0, 1.0, false) == 0, "a measured poor bucket is rejected");
    // A bucket whose prior is below the probe rate is not explored.
    q36::LookupPolicy weak(8);
    expect(weak.choose(4, 8, 100.0, 1.0, false) == 0, "a weak unmeasured bucket is not probed");
}

void test_policy_conditional_rate() {
    // The rate is the conditional per-position probability: rounds that draft 15 and commit 3 are
    // evidence that E(15) ~= 4, i.e. q ~= 0.75 - not accepted/drafted = 0.2.
    q36::LookupPolicy policy(8);
    for (int i = 0; i < 40; ++i) { policy.observe(20, 15, 3); }
    const double q = policy.rate(20);
    expect(q > 0.6 && q < 0.85, "conditional rate is not the accepted/drafted ratio");
    const double e = q36::LookupAcceptance::expected_tokens(15, q);
    expect(e > 3.0 && e < 5.0, "expected tokens match the realized accepted per round");
}

void test_policy_learns() {
    q36::LookupPolicy policy(3);
    const double prior = policy.rate(10);
    for (int i = 0; i < 40; ++i) { policy.observe(10, 4, 4); }
    expect(policy.rate(10) > prior + 0.1, "consistent acceptance raises the estimate");
    q36::LookupPolicy rejecting(3);
    for (int i = 0; i < 40; ++i) { rejecting.observe(10, 4, 0); }
    expect(rejecting.rate(10) < prior - 0.1, "consistent rejection lowers the estimate");
}

} // namespace

int main() {
    test_no_repeat();
    test_short_history();
    test_exact_repeat();
    test_max_drafts();
    test_most_recent_wins();
    test_incremental_sync_matches_at_once();
    test_reset();
    test_sync_prefix_is_idempotent();
    test_propose_after_tail();
    test_policy_priors();
    test_policy_choice();
    test_policy_costs();
    test_policy_explores();
    test_policy_conditional_rate();
    test_policy_learns();
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "suffix drafter and lookup policy tests passed\n";
    return 0;
}
