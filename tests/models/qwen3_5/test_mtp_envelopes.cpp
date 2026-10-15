// The MTP/EAGLE3 attention envelopes carry one entry per drafted step of the widest round the
// machinery runs. The MTP layer drafts at most kMaximumMtpDraftTokens tokens, but a prompt-lookup
// round is verified at the suffix index's width and EAGLE3 drafts wider as well, and a drafting
// round walks every step of its rung -- so the array must hold kLookupDecodeMaximumDrafts - 1
// entries. Rounds above the MTP layer's own domain wrote and read past a six-entry array before
// this was enforced.
#include "models/qwen3_5/program/context.h"
#include "models/qwen3_5/program/context_work.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

namespace q36   = ninfer::models::qwen3_5;
using Envelopes = q36::execution::MtpCausalAttentionEnvelopes;

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

// The requirement, stated independently of the array's definition: every step of the widest round
// the draft ladder can build has to have an envelope of its own.
static_assert(Envelopes{}.ar.size() >= q36::kLookupDecodeMaximumDrafts - 1,
              "the envelope array must cover the widest round the draft ladder can produce");
static_assert(q36::kLookupDecodeMaximumDrafts >= q36::kMaximumMtpDraftTokens,
              "the widest round cannot be narrower than the MTP layer's own draft domain");

// `visible` clamps each bound to the context capacity, so the mapping is checked at a capacity
// above every frontier and again at one below it.
constexpr std::uint32_t kFrontier      = 64;
constexpr std::uint32_t kCapacity      = 4096;
constexpr std::uint32_t kTightCapacity = 32;

void test_step_mapping() {
    for (std::uint32_t k = 1; k <= q36::kLookupDecodeMaximumDrafts; ++k) {
        const Envelopes envelopes =
            q36::detail::mtp_causal_attention_envelopes(kFrontier, k, kCapacity);
        expect(envelopes.target_verify.min_visible_keys == 1 &&
                   envelopes.target_verify.max_visible_keys == kFrontier + k + 1,
               "target verify envelope");
        expect(envelopes.batch.min_visible_keys == envelopes.target_verify.min_visible_keys &&
                   envelopes.batch.max_visible_keys == envelopes.target_verify.max_visible_keys,
               "batch envelope mirrors target verify");
        for (std::uint32_t step = 0; step + 1 < k; ++step) {
            expect(envelopes.ar[step].min_visible_keys == 1, "AR envelope minimum");
            expect(envelopes.ar[step].max_visible_keys == kFrontier + k + step + 2,
                   "AR envelope maximum grows with the step");
        }
    }
    // The step that used to fall off the end of the array.
    const Envelopes widest = q36::detail::mtp_causal_attention_envelopes(
        kFrontier, q36::kLookupDecodeMaximumDrafts, kCapacity);
    const std::uint32_t last = q36::kLookupDecodeMaximumDrafts - 2;
    expect(widest.ar[last].max_visible_keys ==
               kFrontier + q36::kLookupDecodeMaximumDrafts + last + 2,
           "the widest round fills its last envelope");
}

void test_capacity_clamp() {
    const Envelopes envelopes = q36::detail::mtp_causal_attention_envelopes(
        kFrontier, q36::kLookupDecodeMaximumDrafts, kTightCapacity);
    expect(envelopes.target_verify.max_visible_keys == kTightCapacity,
           "target verify clamps to the context capacity");
    expect(envelopes.ar[q36::kLookupDecodeMaximumDrafts - 2].max_visible_keys == kTightCapacity,
           "the widest AR step clamps to the context capacity");
}

void test_domain() {
    expect(
        throws([] { (void)q36::detail::mtp_causal_attention_envelopes(kFrontier, 0, kCapacity); }),
        "a zero-width round is rejected");
    expect(throws([] {
               (void)q36::detail::mtp_causal_attention_envelopes(
                   kFrontier, q36::kLookupDecodeMaximumDrafts + 1, kCapacity);
           }),
           "a round wider than the ladder can build is rejected");
}

} // namespace

int main() {
    test_step_mapping();
    test_capacity_clamp();
    test_domain();
    if (failures != 0) {
        std::cerr << failures << " MTP attention envelope checks failed\n";
        return 1;
    }
    std::cout << "MTP attention envelope checks passed\n";
    return 0;
}
