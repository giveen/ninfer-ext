// ninfer::ops::nemotron_h_moe_router against its FP64 oracle, the reference dump, and the traps.
//
// The cases that are built rather than sampled are the ones a spot check cannot see:
//
//   * the correction bias selects and does not weight. It is added to the scores the selection
//   ranks
//     -- so it moves an expert up or down the top-k *and* feeds the group scores -- while the
//     emitted weight stays the plain sigmoid of that expert's own logit. Two bias layouts over the
//     same logits therefore choose different experts whose per-expert weights are unchanged, and
//     the case asserts that ratio *and* that an implementation taking the weights from the
//     corrected scores would fail it. This is the trap in the rule: the two arrays differ by the
//     bias everywhere, so using the wrong one still produces plausible-looking weights.
//   * the group mask. With groups > groups_used the mask can exclude the expert with the largest
//     score in the whole bank, and the case is built so that it does; at G = U = 1, which is what
//     this checkpoint sets, a missing mask is invisible.
//   * the tie convention, at the expert boundary and at the group boundary.
//   * the degenerate case the checkpoint uses (G = U = 1) at the real banks, 240 and 496 experts,
//     with a correction bias drawn from the range the checkpoint's own layer-1 bank carries.
//
// The reference's arithmetic for this stage is FP32 sequential, which is what makes a sequential
// oracle exact enough: the only nonlinearity is the sigmoid and every other step is a sum or a
// product. The oracle in nemotron_h_moe_router_ref.h evaluates that sequence in double from the
// same represented BF16 operands, so it bounds the reference's rounding instead of adding to it.
#include "ninfer/ops/nemotron_h_moe_router.h"

#include "ops/direct_bf16_weight.h"
#include "ops/nemotron_h_moe_router_ref.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// The Op accumulates in FP32 over the same represented BF16 operands the oracle evaluates in FP64,
// so what separates them is the FP32 sigmoid, the FP32 normalization and the FP32 scale: a few ulp.
// The measured maxima over every case below are reported with NINFER_OP_REPORT_STATS=1; the limits
// are set one order of magnitude above the largest of them so a real break is what trips them.
constexpr PointwiseCriterion kWeightCriterion = {/*absolute*/ 1e-6, /*relative*/ 1e-5};
// The router logits are a 4096-term FP32 sum, where the Op's lane-then-tree order and the oracle's
// sequential one may differ by the accumulation's own rounding; 1e-5 relative covers an FMA chain
// over 4096 terms with margin.
constexpr PointwiseCriterion kLogitCriterion = {/*absolute*/ 1e-3, /*relative*/ 1e-5};

std::vector<std::uint16_t> encode(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        bits[index] = f32_to_bf16(values[index]);
    }
    return bits;
}

std::vector<std::uint16_t> sampled_bits(std::size_t count, std::uint32_t seed, float low,
                                        float high) {
    std::vector<float> values(count);
    fill_uniform(values, seed, low, high);
    round_to_bf16(values);
    return encode(values);
}

struct Inputs {
    std::vector<std::uint16_t> x;      // BF16 [H,T], the hidden axis the fastest one
    std::vector<std::uint16_t> router; // BF16 [E,H]
    std::vector<float> bias;           // FP32 [E]
};

struct Geometry {
    std::int32_t hidden      = 0;
    std::int32_t experts     = 0;
    std::int32_t top_k       = 0;
    std::int32_t groups      = 1;
    std::int32_t groups_used = 1;
    bool normalize           = false;
    float scale              = 1.0F;
    std::int32_t tokens      = 0;
};

ops::NemotronHMoeRouterGeometry api_geometry(const Geometry& geometry) {
    return ops::NemotronHMoeRouterGeometry{
        geometry.hidden,      geometry.experts,   geometry.top_k, geometry.groups,
        geometry.groups_used, geometry.normalize, geometry.scale};
}

// The checkpoint's own correction bias is a large near-constant offset, not a small tie-breaker:
// measured over the released BF16 checkpoint, layer 1's 240-expert bank spans 4.4701 to 4.5500
// (mean 4.5131) and layer 8's 304-expert bank spans 17.5569 to 17.6381. The sampled cases use the
// first of those ranges, so a case that confused the corrected score with the emitted weight would
// be off by about 4.5 rather than by a rounding.
std::vector<float> sampled_bias(std::int32_t experts, std::uint32_t seed, float low, float high) {
    std::vector<float> bias(static_cast<std::size_t>(experts));
    fill_uniform(bias, seed, low, high);
    return bias;
}

Inputs sampled_inputs(const Geometry& geometry, std::uint32_t seed) {
    Inputs inputs;
    // The router reads the block's normalized hidden state, so the activations are O(1); the router
    // weight rows are small, which makes a logit an average rather than a sum.
    inputs.x      = sampled_bits(static_cast<std::size_t>(geometry.hidden) * geometry.tokens, seed,
                                 -3.0F, 3.0F);
    inputs.router = sampled_bits(static_cast<std::size_t>(geometry.experts) * geometry.hidden,
                                 seed + 1, -0.05F, 0.05F);
    inputs.bias   = sampled_bias(geometry.experts, seed + 2, 4.47F, 4.55F);
    return inputs;
}

// x is all ones and every weight row is zero except its first element, so logit[e] is exactly the
// value asked for and the case's expectations can be stated as numbers rather than as samples.
Inputs point_inputs(const std::vector<float>& logits, const std::vector<float>& bias,
                    std::int32_t hidden, std::int32_t tokens) {
    const std::int32_t experts = static_cast<std::int32_t>(logits.size());
    Inputs inputs;
    inputs.x.assign(static_cast<std::size_t>(hidden) * tokens, f32_to_bf16(1.0F));
    inputs.router.assign(static_cast<std::size_t>(experts) * hidden, f32_to_bf16(0.0F));
    for (std::int32_t e = 0; e < experts; ++e) {
        inputs.router[static_cast<std::size_t>(e) * hidden] = f32_to_bf16(logits[e]);
    }
    inputs.bias = bias;
    return inputs;
}

// A valid call with every buffer bound, so a refusal case has to perturb exactly one thing and a
// comparison case only has to read the outputs back.
struct Fixture {
    Fixture(const Inputs& in, const Geometry& g, std::size_t workspace_bytes = 0)
        : geometry(g), router_host{g.experts, g.hidden, in.router}, router(router_host),
          x_buffer(in.x.size() * sizeof(std::uint16_t)),
          bias_buffer(in.bias.size() * sizeof(float)),
          ids_buffer(static_cast<std::size_t>(g.top_k) * g.tokens * sizeof(std::int32_t)),
          weights_buffer(static_cast<std::size_t>(g.top_k) * g.tokens * sizeof(float)),
          workspace_buffer(workspace_bytes != 0 ? workspace_bytes
                                                : ops::nemotron_h_moe_router_workspace_bytes(
                                                      api_geometry(g), g.tokens)) {
        x_buffer.copy_from_host(in.x.data(), x_buffer.bytes());
        bias_buffer.copy_from_host(in.bias.data(), bias_buffer.bytes());
        x                       = Tensor(x_buffer.data(), DType::BF16, {g.hidden, g.tokens});
        expert_ids              = Tensor(ids_buffer.data(), DType::I32, {g.top_k, g.tokens});
        expert_weights          = Tensor(weights_buffer.data(), DType::FP32, {g.top_k, g.tokens});
        workspace               = Tensor(workspace_buffer.data(), DType::U8,
                                         {static_cast<std::int32_t>(workspace_buffer.bytes())});
        weights.router          = router.view();
        weights.correction_bias = Tensor(bias_buffer.data(), DType::FP32, {g.experts});
    }

    void call() {
        ops::nemotron_h_moe_router(x, weights, api_geometry(geometry), expert_ids, expert_weights,
                                   workspace, nullptr);
    }

    Geometry geometry;
    direct_bf16_weight::HostWeight router_host;
    direct_bf16_weight::DeviceWeight router;
    GuardedDeviceBuffer x_buffer;
    GuardedDeviceBuffer bias_buffer;
    GuardedDeviceBuffer ids_buffer;
    GuardedDeviceBuffer weights_buffer;
    GuardedDeviceBuffer workspace_buffer;
    Tensor x;
    Tensor expert_ids;
    Tensor expert_weights;
    Tensor workspace;
    ops::NemotronHMoeRouterWeights weights;
};

struct Run {
    int failures = 0;
    std::vector<std::int32_t> ids; // [K,T], rank fastest
    std::vector<double> weights;   // [K,T]
    std::vector<double> logits;    // [E,T]
};

Run run_case(const char* label, const Inputs& inputs, const Geometry& geometry,
             const std::vector<std::int32_t>* expected_ids = nullptr) {
    const std::size_t ranks = static_cast<std::size_t>(geometry.top_k) * geometry.tokens;
    const auto reference    = nemotron_h_moe_router_ref::run(
        inputs.x, geometry.hidden, geometry.tokens, inputs.router, geometry.experts, inputs.bias,
        geometry.top_k, geometry.groups, geometry.groups_used, geometry.normalize, geometry.scale);

    Fixture fixture(inputs, geometry);
    fixture.call();
    cuda_synchronize();

    Run run;
    const auto ids = from_device<std::int32_t>(fixture.ids_buffer.data(), ranks);
    run.ids        = ids;
    const std::vector<std::int32_t>& want_ids =
        expected_ids != nullptr ? *expected_ids : reference.ids;
    run.failures += verify_exact((std::string(label) + " expert ids").c_str(), ids, want_ids);

    const auto got_weights = from_device<float>(fixture.weights_buffer.data(), ranks);
    run.weights.assign(got_weights.begin(), got_weights.end());
    run.failures += verify_pointwise(std::string(label) + " expert weights", run.weights,
                                     reference.weights, kWeightCriterion);

    const auto logits =
        from_device<float>(fixture.workspace_buffer.data(),
                           static_cast<std::int64_t>(geometry.experts) * geometry.tokens);
    run.logits.assign(logits.begin(), logits.end());
    run.failures += verify_pointwise(std::string(label) + " router logits", run.logits,
                                     reference.logits, kLogitCriterion);

    run.failures += fixture.x_buffer.verify_guards(label);
    run.failures += fixture.workspace_buffer.verify_guards(label);
    run.failures += fixture.router.verify_preserved(label);
    return run;
}

// The weights an implementation that gathered from the *corrected* scores would emit: the same
// selection, the biased values in place of the sigmoids. It exists so a case can show that it
// discriminates the trap rather than only that it passes.
std::vector<double> corrected_weight_variant(const std::vector<double>& logits,
                                             const std::vector<float>& bias,
                                             const std::vector<std::int32_t>& ids,
                                             const Geometry& geometry) {
    const std::size_t ranks = static_cast<std::size_t>(geometry.top_k) * geometry.tokens;
    std::vector<double> variant(ranks);
    for (std::int32_t token = 0; token < geometry.tokens; ++token) {
        double sum = 0.0;
        std::vector<double> emitted(static_cast<std::size_t>(geometry.top_k));
        for (std::int32_t rank = 0; rank < geometry.top_k; ++rank) {
            const std::int32_t expert =
                ids[static_cast<std::size_t>(rank) * geometry.tokens + token];
            const double corrected =
                1.0 / (1.0 +
                       std::exp(
                           -logits[static_cast<std::size_t>(expert) * geometry.tokens + token])) +
                static_cast<double>(bias[static_cast<std::size_t>(expert)]);
            emitted[static_cast<std::size_t>(rank)] = corrected;
            sum += corrected;
        }
        for (std::int32_t rank = 0; rank < geometry.top_k; ++rank) {
            double weight = emitted[static_cast<std::size_t>(rank)];
            if (geometry.normalize) { weight /= sum + 1e-20; }
            variant[static_cast<std::size_t>(rank) * geometry.tokens + token] =
                weight * geometry.scale;
        }
    }
    return variant;
}

double maximum_deviation(const std::vector<double>& left, const std::vector<double>& right) {
    if (left.size() != right.size()) { throw std::invalid_argument("deviation sizes differ"); }
    double worst = 0.0;
    for (std::size_t index = 0; index < left.size(); ++index) {
        worst = std::max(worst, std::abs(left[index] - right[index]));
    }
    return worst;
}

// --- the correction bias ----------------------------------------------------

constexpr std::int32_t kBiasHidden = 8;

// sigmoid(-1), sigmoid(1), sigmoid(0), sigmoid(-3): the four logits both bias layouts below rank.
const std::vector<float> kBiasLogits = {-1.0F, 1.0F, 0.0F, -3.0F};

int bias_trap_case() {
    int failures = 0;
    const Geometry geometry{
        /*hidden*/ kBiasHidden, /*experts*/ 4,     /*top_k*/ 2,
        /*groups*/ 1,           /*groups_used*/ 1, /*normalize*/ true,
        /*scale*/ 1.0F,         /*tokens*/ 1};

    // Expert 0 has the *smallest* sigmoid of the four (0.2689) and a bias of 0.6, so the corrected
    // ranking puts it first: the bias changes which experts are selected, from {1,2} to {0,1}.
    const Inputs biased_first = point_inputs(kBiasLogits, {0.6F, 0.0F, 0.0F, 0.0F}, kBiasHidden, 1);
    const std::vector<std::int32_t> expected_first{0, 1};
    const Run first =
        run_case("bias selects, weights uncorrected", biased_first, geometry, &expected_first);
    failures += first.failures;

    // Expert 2 instead: same logits, same Op, one buffer changed, and the selected set moves.
    const Inputs biased_third = point_inputs(kBiasLogits, {0.0F, 0.0F, 0.5F, 0.0F}, kBiasHidden, 1);
    const std::vector<std::int32_t> expected_third{2, 1};
    const Run third =
        run_case("bias selects, other expert", biased_third, geometry, &expected_third);
    failures += third.failures;

    // The emitted weight is the uncorrected sigmoid of the ranked expert, so the ratio between the
    // two ranks is the ratio of those experts' sigmoids -- expert 1 over expert 2 here -- *despite*
    // the bias that put expert 2 first, and the rank order is by the corrected score, so the larger
    // weight sits in the second rank.
    const auto uncorrected = [](float logit) {
        return 1.0 / (1.0 + std::exp(-static_cast<double>(logit)));
    };
    const double expected_ratio =
        uncorrected(kBiasLogits[third.ids[1]]) / uncorrected(kBiasLogits[third.ids[0]]);
    const double ratio = third.weights[1] / third.weights[0];
    const std::vector<double> corrected_first =
        corrected_weight_variant(first.logits, biased_first.bias, first.ids, geometry);
    const std::vector<double> corrected_third =
        corrected_weight_variant(third.logits, biased_third.bias, third.ids, geometry);
    const double first_gap   = maximum_deviation(first.weights, corrected_first);
    const double third_gap   = maximum_deviation(third.weights, corrected_third);
    const bool ranks_swapped = third.weights[0] < third.weights[1];
    std::printf("bias trap: uncorrected ratio %.6f (expected %.6f), corrected-score variant off by "
                "%.6f / %.6f, rank 0 weight %s rank 1\n",
                ratio, expected_ratio, first_gap, third_gap, ranks_swapped ? "<" : ">");
    if (std::abs(ratio - expected_ratio) > 1e-4) {
        std::cerr << "bias trap: the emitted weights are not the uncorrected ratio\n";
        ++failures;
    }
    if (!ranks_swapped) {
        std::cerr << "bias trap: the ranks are ordered by the emitted weight, not by the corrected "
                     "score\n";
        ++failures;
    }
    // The gap the case would have caught: both layouts must separate the two readings by far more
    // than the criterion, or the case proves nothing.
    if (first_gap < 0.1 || third_gap < 0.1) {
        std::cerr << "bias trap: the corrected-score variant is " << first_gap << " / " << third_gap
                  << " away, which does not discriminate\n";
        ++failures;
    }
    // The selected sets must differ too, or the bias did not change the choice.
    if (first.ids == third.ids) {
        std::cerr << "bias trap: the bias did not change the selection\n";
        ++failures;
    }
    return failures;
}

// The bias also feeds the group scores, which is a second place the corrected score is used.
int bias_group_case() {
    int failures = 0;
    const std::vector<float> logits{1.0F, 0.8F, 0.6F, 0.5F, 0.4F, 0.3F, 3.0F, -2.0F};
    const std::vector<float> bias{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.9F};
    const Geometry geometry{/*hidden*/ 8,   /*experts*/ 8,     /*top_k*/ 3,
                            /*groups*/ 4,   /*groups_used*/ 2, /*normalize*/ true,
                            /*scale*/ 1.0F, /*tokens*/ 1};
    const Inputs inputs = point_inputs(logits, bias, 8, 1);

    // Group scores without the bias: 1.4210, 1.2682, 1.1731 and 0.1192+0.9526 = 1.0718, so groups
    // {0,1} are selected and expert 6 -- the largest score in the bank -- is masked out. With 0.9
    // on expert 7 the fourth group scores 1.9718, so groups {3,0} are selected and expert 7, whose
    // sigmoid is the smallest of the three selected, is ranked first because the corrected score is
    // what ranks.
    const std::vector<std::int32_t> expected{7, 6, 0};
    const Run run = run_case("bias feeds the group scores", inputs, geometry, &expected);
    failures += run.failures;
    std::printf("bias group: ids %d %d %d, weights %.6f %.6f %.6f\n", run.ids[0], run.ids[1],
                run.ids[2], run.weights[0], run.weights[1], run.weights[2]);
    return failures;
}

// --- the group mask --------------------------------------------------------

int group_mask_case() {
    int failures = 0;
    const std::vector<float> logits{1.0F, 0.8F, 0.6F, 0.5F, 0.4F, 0.3F, 3.0F, -2.0F};
    const std::vector<float> bias(8, 0.0F);
    const Geometry geometry{/*hidden*/ 8,   /*experts*/ 8,     /*top_k*/ 3,
                            /*groups*/ 4,   /*groups_used*/ 2, /*normalize*/ true,
                            /*scale*/ 1.0F, /*tokens*/ 1};
    const Inputs inputs = point_inputs(logits, bias, 8, 1);

    // Groups {0,1} carry the two largest group scores (1.4210 and 1.2682 against 1.1731
    // and 1.0718), so the eligible experts are 0..3 and the three ranks are the three largest of
    // those. Expert 6 is the largest score in the bank and is *not* selected, which is the whole
    // point of the mask; the unmasked answer would be the ranks {6,0,1}.
    const std::vector<std::int32_t> expected{0, 1, 2};
    const Run run = run_case("group mask", inputs, geometry, &expected);
    failures += run.failures;
    const std::vector<std::int32_t> unmasked{6, 0, 1};
    std::printf("group mask: ids %d %d %d, unmasked would be %d %d %d\n", run.ids[0], run.ids[1],
                run.ids[2], unmasked[0], unmasked[1], unmasked[2]);
    if (run.ids == unmasked) {
        std::cerr << "group mask: the case does not discriminate an unmasked top-k\n";
        ++failures;
    }
    return failures;
}

// --- ties ------------------------------------------------------------------

// Rows 1 and 2 are identical and positive, rows 0 and 3 their negation, so the two projections are
// bitwise equal and strictly above the others: with one rank the lower id must win. The reference
// ranks with torch.topk(sorted=False), whose order among equal values is not defined, so this is
// this Op's convention rather than something the reference settles.
int tie_case() {
    constexpr std::int32_t kHidden = 8;
    Inputs inputs;
    inputs.x.assign(kHidden, f32_to_bf16(1.0F));
    inputs.router.resize(4 * kHidden);
    for (std::int32_t h = 0; h < kHidden; ++h) {
        inputs.router[static_cast<std::size_t>(0) * kHidden + h] = f32_to_bf16(-1.0F);
        inputs.router[static_cast<std::size_t>(1) * kHidden + h] = f32_to_bf16(1.0F);
        inputs.router[static_cast<std::size_t>(2) * kHidden + h] = f32_to_bf16(1.0F);
        inputs.router[static_cast<std::size_t>(3) * kHidden + h] = f32_to_bf16(-1.0F);
    }
    inputs.bias.assign(4, 0.0F);
    const Geometry geometry{/*hidden*/ kHidden, /*experts*/ 4,     /*top_k*/ 1,
                            /*groups*/ 1,       /*groups_used*/ 1, /*normalize*/ false,
                            /*scale*/ 1.0F,     /*tokens*/ 1};
    const std::vector<std::int32_t> expected{1};
    return run_case("expert tie", inputs, geometry, &expected).failures;
}

// Every logit is equal, so all four group scores tie and the lower group id must win: the two ranks
// come from group 0, and the tie inside it resolves to the lower expert id as well.
int group_tie_case() {
    constexpr std::int32_t kHidden = 8;
    Inputs inputs;
    inputs.x.assign(kHidden, f32_to_bf16(1.0F));
    inputs.router.assign(4 * kHidden, f32_to_bf16(0.5F));
    inputs.bias.assign(4, 0.0F);
    const Geometry geometry{/*hidden*/ kHidden, /*experts*/ 4,     /*top_k*/ 2,
                            /*groups*/ 2,       /*groups_used*/ 1, /*normalize*/ true,
                            /*scale*/ 1.0F,     /*tokens*/ 1};
    const std::vector<std::int32_t> expected{0, 1};
    return run_case("group tie", inputs, geometry, &expected).failures;
}

// --- normalization and the scale -------------------------------------------

int normalize_and_scale_case() {
    int failures = 0;
    Geometry geometry{/*hidden*/ 16,  /*experts*/ 8,     /*top_k*/ 3,
                      /*groups*/ 1,   /*groups_used*/ 1, /*normalize*/ true,
                      /*scale*/ 1.0F, /*tokens*/ 3};
    const Inputs inputs = sampled_inputs(geometry, 91);

    for (const bool normalize : {false, true}) {
        for (const float scale : {1.0F, 5.0F}) {
            geometry.normalize      = normalize;
            geometry.scale          = scale;
            const std::string label = std::string("normalize=") + (normalize ? "on" : "off") +
                                      " scale=" + std::to_string(scale);
            const Run run           = run_case(label.c_str(), inputs, geometry);
            failures += run.failures;
            double sum = 0.0;
            for (const double weight : run.weights) { sum += weight; }
            std::printf("normalize=%s scale=%.1f: weight sum over all ranks %.6f\n",
                        normalize ? "on" : "off", scale, sum);
            // With normalization on the scale is the whole sum: the division redistributes one per
            // token. Without it the sum is the scale times the selected sigmoids, which is not the
            // scale -- so this is also the check that the division happened before the
            // multiplication.
            if (normalize && std::abs(sum - scale * geometry.tokens) > 1e-3) {
                std::cerr << label << ": the weight sum is " << sum << ", expected "
                          << scale * geometry.tokens << '\n';
                ++failures;
            }
            if (!normalize && std::abs(sum - scale * geometry.tokens) < 1e-3) {
                std::cerr << label << ": an unnormalized sum landed on the scale\n";
                ++failures;
            }
        }
    }
    return failures;
}

// --- the checkpoint's own configuration ------------------------------------

// The banks the checkpoint carries and the degenerate grouping it sets: n_group = topk_group = 1,
// so the mask selects the whole bank; norm_topk_prob with routed_scaling_factor 5.0.
int real_bank_case(std::int32_t experts, std::int32_t tokens, std::uint32_t seed) {
    const Geometry geometry{/*hidden*/ 4096,    experts,           /*top_k*/ 9,
                            /*groups*/ 1,       /*groups_used*/ 1,
                            /*normalize*/ true, /*scale*/ 5.0F,    tokens};
    const Inputs inputs = sampled_inputs(geometry, seed);
    const std::string label =
        "real bank E=" + std::to_string(experts) + " T=" + std::to_string(tokens);
    const Run run = run_case(label.c_str(), inputs, geometry);
    return run.failures;
}

// The bias spread against the score gaps it decides. The checkpoint's bias spans about 0.08 across
// a bank while the corrected scores of the experts competing for the ninth rank differ by less than
// that, so the bias is not decoration: this counts how many tokens select a different set with it
// than without it, at the real geometry.
int bias_changes_selection_case() {
    const Geometry geometry{
        /*hidden*/ 4096,   /*experts*/ 240,    /*top_k*/ 9,    /*groups*/ 1,
        /*groups_used*/ 1, /*normalize*/ true, /*scale*/ 5.0F, /*tokens*/ 130};
    const Inputs inputs = sampled_inputs(geometry, 4242);
    std::vector<float> flat(static_cast<std::size_t>(geometry.experts), 0.0F);
    const auto with_bias = nemotron_h_moe_router_ref::run(
        inputs.x, geometry.hidden, geometry.tokens, inputs.router, geometry.experts, inputs.bias,
        geometry.top_k, geometry.groups, geometry.groups_used, geometry.normalize, geometry.scale);
    const auto without_bias = nemotron_h_moe_router_ref::run(
        inputs.x, geometry.hidden, geometry.tokens, inputs.router, geometry.experts, flat,
        geometry.top_k, geometry.groups, geometry.groups_used, geometry.normalize, geometry.scale);
    int differing = 0;
    for (std::int32_t token = 0; token < geometry.tokens; ++token) {
        std::vector<std::int32_t> with;
        std::vector<std::int32_t> without;
        for (std::int32_t rank = 0; rank < geometry.top_k; ++rank) {
            const std::size_t at = static_cast<std::size_t>(rank) * geometry.tokens + token;
            with.push_back(with_bias.ids[at]);
            without.push_back(without_bias.ids[at]);
        }
        std::sort(with.begin(), with.end());
        std::sort(without.begin(), without.end());
        if (with != without) { ++differing; }
    }
    std::printf("bias over 240 experts: %d of %d tokens select a different expert set with the "
                "bias than without it\n",
                differing, geometry.tokens);

    const Run run = run_case("bias changes the selection", inputs, geometry);
    int failures  = run.failures;
    if (differing == 0) {
        std::cerr
            << "bias: the correction bias changed nothing, so the case does not exercise it\n";
        ++failures;
    }
    return failures;
}

// --- refusals --------------------------------------------------------------

template <typename Call>
int refuse_case(const char* label, Call&& call) {
    try {
        call();
    } catch (const std::invalid_argument&) { return 0; } catch (const std::exception& error) {
        std::cerr << label << ": wrong exception: " << error.what() << '\n';
        return 1;
    }
    std::cerr << label << ": no exception\n";
    return 1;
}

int refusal_cases() {
    int failures = 0;
    const Geometry good{/*hidden*/ 8,       /*experts*/ 8,     /*top_k*/ 2,
                        /*groups*/ 2,       /*groups_used*/ 1,
                        /*normalize*/ true, /*scale*/ 1.0F,    /*tokens*/ 2};
    const Inputs inputs = sampled_inputs(good, 7);
    const auto perturb  = [&](const Geometry& geometry) {
        Fixture fixture(inputs, geometry);
        fixture.call();
    };

    Geometry geometry = good;
    geometry.hidden   = 12;
    failures += refuse_case("hidden not a multiple of 8", [&] { perturb(geometry); });
    geometry        = good;
    geometry.hidden = 0;
    failures += refuse_case("hidden 0", [&] { perturb(geometry); });
    geometry         = good;
    geometry.experts = 0;
    failures += refuse_case("no experts", [&] { perturb(geometry); });
    geometry         = good;
    geometry.experts = 513;
    failures += refuse_case("more experts than the family allows", [&] { perturb(geometry); });
    geometry       = good;
    geometry.top_k = 0;
    failures += refuse_case("top_k 0", [&] { perturb(geometry); });
    geometry       = good;
    geometry.top_k = 9;
    failures += refuse_case("top_k above the bank", [&] { perturb(geometry); });
    geometry        = good;
    geometry.groups = 3;
    failures += refuse_case("groups does not divide the bank", [&] { perturb(geometry); });
    geometry        = good;
    geometry.groups = 0;
    failures += refuse_case("no groups", [&] { perturb(geometry); });
    geometry             = good;
    geometry.groups_used = 0;
    failures += refuse_case("no groups used", [&] { perturb(geometry); });
    geometry             = good;
    geometry.groups_used = 3;
    failures += refuse_case("more groups used than groups", [&] { perturb(geometry); });
    geometry        = good;
    geometry.groups = 8;
    failures += refuse_case("the selected groups cannot supply top_k", [&] { perturb(geometry); });
    geometry       = good;
    geometry.scale = std::numeric_limits<float>::infinity();
    failures += refuse_case("a non-finite scale", [&] { perturb(geometry); });

    // One tensor at a time off the valid call. The top_k bounds above already cover 16; the shapes
    // below are what a loader passing the wrong buffer would look like.
    failures += refuse_case("router not BF16", [&] {
        Fixture fixture(inputs, good);
        fixture.weights.router.qtype = QType::FP32;
        fixture.call();
    });
    failures += refuse_case("router unbound", [&] {
        Fixture fixture(inputs, good);
        fixture.weights.router.payload = nullptr;
        fixture.call();
    });
    failures += refuse_case("router the wrong shape", [&] {
        Fixture fixture(inputs, good);
        fixture.weights.router.shape[1] = 4;
        fixture.call();
    });
    failures += refuse_case("router misaligned", [&] {
        Fixture fixture(inputs, good);
        fixture.weights.router.payload =
            static_cast<const char*>(fixture.weights.router.payload) + 8;
        fixture.call();
    });
    failures += refuse_case("correction_bias not FP32", [&] {
        Fixture fixture(inputs, good);
        fixture.weights.correction_bias.dtype = DType::BF16;
        fixture.call();
    });
    failures += refuse_case("correction_bias the wrong length", [&] {
        Fixture fixture(inputs, good);
        fixture.weights.correction_bias.ne[0] = 4;
        fixture.call();
    });
    failures += refuse_case("x not BF16", [&] {
        Fixture fixture(inputs, good);
        fixture.x.dtype = DType::FP32;
        fixture.call();
    });
    failures += refuse_case("x misaligned", [&] {
        Fixture fixture(inputs, good);
        fixture.x.data = static_cast<char*>(fixture.x.data) + 8;
        fixture.call();
    });
    failures += refuse_case("x the wrong hidden width", [&] {
        Fixture fixture(inputs, good);
        fixture.x.ne[0] = 4;
        fixture.call();
    });
    failures += refuse_case("x not contiguous", [&] {
        Fixture fixture(inputs, good);
        fixture.x.nb[0] = 8;
        fixture.call();
    });
    failures += refuse_case("expert_ids the wrong dtype", [&] {
        Fixture fixture(inputs, good);
        fixture.expert_ids.dtype = DType::FP32;
        fixture.call();
    });
    failures += refuse_case("expert_ids the wrong shape", [&] {
        Fixture fixture(inputs, good);
        fixture.expert_ids.ne[0] = 1;
        fixture.call();
    });
    failures += refuse_case("expert_weights the wrong dtype", [&] {
        Fixture fixture(inputs, good);
        fixture.expert_weights.dtype = DType::BF16;
        fixture.call();
    });
    failures += refuse_case("expert_weights the wrong shape", [&] {
        Fixture fixture(inputs, good);
        fixture.expert_weights.ne[1] = 1;
        fixture.call();
    });
    failures += refuse_case("workspace unbound", [&] {
        Fixture fixture(inputs, good);
        fixture.workspace.data = nullptr;
        fixture.call();
    });
    failures += refuse_case("workspace too small", [&] {
        Fixture fixture(inputs, good, 16);
        fixture.call();
    });
    return failures;
}

// --- the reference dump ----------------------------------------------------

struct DumpTensor {
    std::vector<double> values;
    std::vector<std::int32_t> shape;
};

std::map<std::string, DumpTensor> read_dump(const std::string& directory) {
    std::ifstream manifest(directory + "/manifest.txt");
    if (!manifest) { throw std::runtime_error("no manifest in " + directory); }
    std::map<std::string, DumpTensor> tensors;
    std::string line;
    while (std::getline(manifest, line)) {
        if (line.empty()) { continue; }
        const std::size_t name_end  = line.find('\t');
        const std::size_t shape_end = line.find('\t', name_end + 1);
        if (name_end == std::string::npos || shape_end == std::string::npos) {
            throw std::runtime_error("malformed manifest line: " + line);
        }
        DumpTensor tensor;
        std::istringstream extents(line.substr(name_end + 1, shape_end - name_end - 1));
        std::string extent;
        while (std::getline(extents, extent, 'x')) { tensor.shape.push_back(std::stoi(extent)); }
        std::ifstream file(directory + "/" + line.substr(shape_end + 1), std::ios::binary);
        if (!file) { throw std::runtime_error("missing dump file: " + line); }
        const std::vector<char> bytes((std::istreambuf_iterator<char>(file)),
                                      std::istreambuf_iterator<char>());
        std::int32_t elements = 1;
        for (const std::int32_t extent : tensor.shape) { elements *= extent; }
        if (bytes.size() != static_cast<std::size_t>(elements) * sizeof(float)) {
            throw std::runtime_error("dump file does not match its manifest shape: " + line);
        }
        std::vector<float> raw(bytes.size() / sizeof(float));
        std::memcpy(raw.data(), bytes.data(), bytes.size());
        tensor.values.assign(raw.begin(), raw.end());
        tensors.emplace(line.substr(0, name_end), std::move(tensor));
    }
    return tensors;
}

const DumpTensor& require_dump(const std::map<std::string, DumpTensor>& tensors,
                               const std::string& name, const std::vector<std::int32_t>& shape) {
    const auto found = tensors.find(name);
    if (found == tensors.end()) {
        throw std::runtime_error("the dump has no " + name +
                                 ": regenerate it with tools/verify/nemotron_h_reference.py");
    }
    if (found->second.shape != shape) {
        std::string got;
        for (const std::int32_t extent : found->second.shape) {
            got += std::to_string(extent) + "x";
        }
        throw std::runtime_error("the dump's " + name + " is " + got +
                                 " rather than the tiny "
                                 "config's shape");
    }
    return found->second;
}

// The canonical implementation's own router outputs for the tiny config, compared to this Op on the
// same tensors. Two things are worth stating about what this does and does not cover: the tiny
// model is built in FP32, so the BF16 rounding of the dumped x and weight is a perturbation the
// Op's contract imposes and the reference does not have -- which is why a selection difference is
// measured against that perturbation rather than forbidden outright -- and the tiny config's
// correction bias is zeros and its scale 1.0, so the bias and the scale are covered by the cases
// above, not here.
int reference_dump_case(const std::string& directory) {
    const auto tensors = read_dump(directory);
    int failures       = 0;
    for (const std::int32_t layer : {1, 3}) {
        const std::string prefix = "model.layers." + std::to_string(layer) + ".mixer.gate";
        // The tiny config: 8 routed experts, top-2, n_group = topk_group = 1, norm_topk_prob, and
        // routed_scaling_factor 1.0.
        constexpr std::int32_t kTokens  = 130;
        constexpr std::int32_t kHidden  = 64;
        constexpr std::int32_t kExperts = 8;
        constexpr std::int32_t kTopK    = 2;
        const DumpTensor& logits        = require_dump(tensors, prefix + ".0", {kTokens, kExperts});
        const DumpTensor& dumped_weights = require_dump(tensors, prefix + ".1", {kTokens, kTopK});
        const DumpTensor& dumped_ids     = require_dump(tensors, prefix + ".2", {kTokens, kTopK});
        const DumpTensor& input  = require_dump(tensors, prefix + ".input", {1, kTokens, kHidden});
        const DumpTensor& weight = require_dump(tensors, prefix + ".weight", {kExperts, kHidden});
        const DumpTensor& bias =
            require_dump(tensors, prefix + ".e_score_correction_bias", {kExperts});

        // The dumped indices are the tool's float32 cast of an I32 tensor.
        std::vector<std::int32_t> reference_ids(static_cast<std::size_t>(kTokens) * kTopK);
        double worst_index_cast = 0.0;
        for (std::size_t index = 0; index < reference_ids.size(); ++index) {
            const double raw     = dumped_ids.values[index];
            reference_ids[index] = static_cast<std::int32_t>(std::llround(raw));
            worst_index_cast     = std::max(worst_index_cast, std::abs(raw - reference_ids[index]));
        }

        // The tiny config's settings, checked rather than assumed: with norm_topk_prob on and a
        // scale of 1.0 the emitted weights sum to one per token, and an uncorrected sigmoid is
        // below one.
        double worst_sum      = 0.0;
        double largest_weight = 0.0;
        double largest_bias   = 0.0;
        for (std::int32_t token = 0; token < kTokens; ++token) {
            double sum = 0.0;
            for (std::int32_t rank = 0; rank < kTopK; ++rank) {
                const double w =
                    dumped_weights.values[static_cast<std::size_t>(token) * kTopK + rank];
                sum += w;
                largest_weight = std::max(largest_weight, w);
            }
            worst_sum = std::max(worst_sum, std::abs(sum - 1.0));
        }
        for (const double value : bias.values) {
            largest_bias = std::max(largest_bias, std::abs(value));
        }

        Inputs inputs;
        inputs.x      = encode(std::vector<float>(input.values.begin(), input.values.end()));
        inputs.router = encode(std::vector<float>(weight.values.begin(), weight.values.end()));
        inputs.bias.assign(bias.values.begin(), bias.values.end());
        const Geometry geometry{
            /*hidden*/ kHidden, /*experts*/ kExperts, /*top_k*/ kTopK,
            /*groups*/ 1,       /*groups_used*/ 1,    /*normalize*/ true,
            /*scale*/ 1.0F,     /*tokens*/ kTokens};
        const Run run =
            run_case(("reference dump layer " + std::to_string(layer)).c_str(), inputs, geometry);
        failures += run.failures;

        // The Op's logits and the dumped ones are the same stage written in transposed order: the
        // reference emits [tokens, experts] and this Op's workspace is [experts, tokens]. They are
        // also computed from different operands -- the tiny reference runs in FP32 while this Op's
        // contract is BF16 -- so the difference is bounded rather than required to vanish: the
        // first-order effect of the two rounded operands, plus the slack of two different FP32
        // accumulation orders (the reference's matmul against this Op's lane-then-tree one, n*eps
        // of the summed magnitudes). The per-token maximum of the measured difference is what
        // bounds a legitimate selection difference below.
        std::vector<double> perturbation(static_cast<std::size_t>(kTokens), 0.0);
        double worst_logit_abs = 0.0;
        double worst_logit_rel = 0.0;
        double worst_excess    = 0.0;
        for (std::int32_t token = 0; token < kTokens; ++token) {
            for (std::int32_t e = 0; e < kExperts; ++e) {
                const std::size_t row = static_cast<std::size_t>(token) * kHidden;
                double bound          = 0.0;
                double products       = 0.0;
                for (std::int32_t h = 0; h < kHidden; ++h) {
                    const double xd = input.values[row + static_cast<std::size_t>(h)];
                    const double wd = weight.values[static_cast<std::size_t>(e) * kHidden + h];
                    const double dx = bf16_to_f32(inputs.x[row + static_cast<std::size_t>(h)]) - xd;
                    const double dw =
                        bf16_to_f32(inputs.router[static_cast<std::size_t>(e) * kHidden + h]) - wd;
                    bound += std::abs(wd * dx) + std::abs(xd * dw) + std::abs(dw * dx);
                    products += std::abs(wd * xd);
                }
                bound += static_cast<double>(kHidden) * 1.2e-7 * products;

                const double dumped = logits.values[static_cast<std::size_t>(token) * kExperts + e];
                const double got    = run.logits[static_cast<std::size_t>(e) * kTokens + token];
                const double error  = std::abs(got - dumped);
                worst_logit_abs     = std::max(worst_logit_abs, error);
                worst_logit_rel     = std::max(
                    worst_logit_rel, std::abs(dumped) > 0.0 ? error / std::abs(dumped) : error);
                worst_excess = std::max(worst_excess, error - bound);
                perturbation[static_cast<std::size_t>(token)] =
                    std::max(perturbation[static_cast<std::size_t>(token)], error);
            }
        }

        // The reference's top-k is `sorted=False`, whose rank order it does not define, so the
        // ranks are matched as a set per token and an order difference is counted rather than
        // failed. A *set* difference is failed unless the token's own boundary gap -- how far apart
        // its K-th and (K+1)-th corrected scores are -- is inside that token's logit perturbation,
        // which is the only way rounding the operands can move a selection. A difference outside it
        // would be this Op's rule being wrong rather than its precision, and the rule is separately
        // pinned by the oracle comparison inside run_case, on exactly these rounded operands.
        int reordered           = 0;
        int set_mismatch        = 0;
        int unexplained         = 0;
        double tightest_ratio   = 0.0;
        double worst_weight_abs = 0.0;
        std::vector<double> corrected(static_cast<std::size_t>(kExperts));
        for (std::int32_t token = 0; token < kTokens; ++token) {
            for (std::int32_t e = 0; e < kExperts; ++e) {
                const double dumped = logits.values[static_cast<std::size_t>(token) * kExperts + e];
                corrected[static_cast<std::size_t>(e)] =
                    1.0 / (1.0 + std::exp(-dumped)) + bias.values[static_cast<std::size_t>(e)];
            }
            std::sort(corrected.begin(), corrected.end());
            const double gap = corrected[static_cast<std::size_t>(kExperts - kTopK)] -
                               corrected[static_cast<std::size_t>(kExperts - kTopK - 1)];

            std::vector<std::pair<std::int32_t, double>> wanted;
            std::vector<std::pair<std::int32_t, double>> got;
            for (std::int32_t rank = 0; rank < kTopK; ++rank) {
                const std::size_t at = static_cast<std::size_t>(rank) * kTokens + token;
                wanted.emplace_back(
                    reference_ids[static_cast<std::size_t>(token) * kTopK + rank],
                    dumped_weights.values[static_cast<std::size_t>(token) * kTopK + rank]);
                got.emplace_back(run.ids[at], run.weights[at]);
            }
            if (got[0].first != wanted[0].first || got[1].first != wanted[1].first) { ++reordered; }
            std::sort(wanted.begin(), wanted.end());
            std::sort(got.begin(), got.end());
            if (wanted[0].first != got[0].first || wanted[1].first != got[1].first) {
                ++set_mismatch;
                const double limit = perturbation[static_cast<std::size_t>(token)];
                if (gap > limit) {
                    ++unexplained;
                } else if (limit > 0.0) {
                    const double ratio = gap / limit;
                    tightest_ratio =
                        tightest_ratio == 0.0 ? ratio : std::min(tightest_ratio, ratio);
                }
                continue;
            }
            for (int rank = 0; rank < kTopK; ++rank) {
                worst_weight_abs = std::max(worst_weight_abs,
                                            std::abs(wanted[static_cast<std::size_t>(rank)].second -
                                                     got[static_cast<std::size_t>(rank)].second));
            }
        }

        // The BF16 rounding the Op's contract imposes on tensors the reference held in FP32.
        double worst_input_rel  = 0.0;
        double worst_weight_rel = 0.0;
        for (std::size_t index = 0; index < input.values.size(); ++index) {
            const double rounded = bf16_to_f32(inputs.x[index]);
            worst_input_rel =
                std::max(worst_input_rel, std::abs(rounded - input.values[index]) /
                                              std::max(std::abs(input.values[index]), 1e-30));
        }
        for (std::size_t index = 0; index < weight.values.size(); ++index) {
            const double rounded = bf16_to_f32(inputs.router[index]);
            worst_weight_rel =
                std::max(worst_weight_rel, std::abs(rounded - weight.values[index]) /
                                               std::max(std::abs(weight.values[index]), 1e-30));
        }

        std::printf(
            "layer %d logits: max abs %.6e rel %.6e over %dx%d (bf16 rounding of x %.3e, of "
            "the weight %.3e; worst excess %.3e over the operand-rounding bound)\n",
            layer, worst_logit_abs, worst_logit_rel, kTokens, kExperts, worst_input_rel,
            worst_weight_rel, worst_excess);
        std::printf("layer %d top-k: %d/%d tokens differ as a set (%d of them outside that token's "
                    "logit perturbation, tightest gap/perturbation %.3f), %d/%d in rank order; "
                    "weights max abs %.6e where the sets agree\n",
                    layer, set_mismatch, kTokens, unexplained, tightest_ratio, reordered, kTokens,
                    worst_weight_abs);
        std::printf("layer %d geometry: reference weight sums within %.3e of 1, largest weight "
                    "%.6f, largest |bias| %.3e, index cast off by %.3e\n",
                    layer, worst_sum, largest_weight, largest_bias, worst_index_cast);
        if (unexplained != 0) {
            std::cerr
                << "layer " << layer << ": " << unexplained
                << " tokens select a different expert set than the reference by more than the "
                   "input rounding\n";
            ++failures;
        }
        if (worst_excess > 0.0) {
            std::cerr << "layer " << layer << ": a logit is " << worst_excess
                      << " outside the difference the operand rounding and the accumulation order "
                         "can explain\n";
            ++failures;
        }
        if (worst_index_cast > 1e-3) {
            std::cerr << "layer " << layer << ": the dumped indices are not integral\n";
            ++failures;
        }
        // The tiny config's settings are checked rather than assumed: with norm_topk_prob on and a
        // scale of 1.0 the reference's own weights sum to one per token, and an uncorrected sigmoid
        // is below one.
        if (std::abs(worst_sum) > 1e-3 || largest_weight > 1.0) {
            std::cerr << "layer " << layer
                      << ": the dump is not the normalized unit-scale geometry this case assumes\n";
            ++failures;
        }
        if (largest_bias != 0.0) {
            std::printf(
                "layer %d: the dump's correction bias is not zero (%.3e at the largest), so "
                "this comparison covers the bias as well\n",
                layer, largest_bias);
        }
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "nemotron_h_moe_router: no CUDA device\n";
        return 77;
    }
    int failures = 0;
    try {
        failures += bias_trap_case();
        failures += bias_group_case();
        failures += group_mask_case();
        failures += tie_case();
        failures += group_tie_case();
        failures += normalize_and_scale_case();
        failures += bias_changes_selection_case();

        // The banks and the token counts the checkpoint's MoE layers actually carry, at the
        // degenerate grouping it sets. 240 experts is the most common bank and 496 the widest.
        for (const std::int32_t tokens : {1, 5, 130, 300}) {
            failures += real_bank_case(240, tokens, 11U + static_cast<std::uint32_t>(tokens));
        }
        for (const std::int32_t tokens : {1, 130}) {
            failures += real_bank_case(496, tokens, 61U + static_cast<std::uint32_t>(tokens));
        }

        failures += refusal_cases();

        if (const char* dump = std::getenv("NINFER_NEMOTRON_REF")) {
            failures += reference_dump_case(dump);
        } else {
            std::cout << "nemotron_h_moe_router: set NINFER_NEMOTRON_REF to a "
                         "tools/verify/nemotron_h_reference.py dump to also compare against the "
                         "reference\n";
        }
    } catch (const std::exception& error) {
        std::cerr << "nemotron_h_moe_router: " << error.what() << '\n';
        return 1;
    }
    if (failures == 0) {
        std::cout << "OK: nemotron_h_moe_router matches its oracle, the group mask, the bias trap "
                     "and the scale\n";
    }
    return failures == 0 ? 0 : 1;
}
