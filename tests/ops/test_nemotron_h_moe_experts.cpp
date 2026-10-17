// ninfer::ops::nemotron_h_moe_experts against its FP64 oracle, the reference dump, and the traps.
//
// The cases that are built rather than sampled are the ones a spot check cannot see:
//
//   * the dispatch itself. The experts are not gated, so an implementation that reached for the
//     Gemma 4 gate/up arithmetic would be reading the wrong rows of the wrong bank, and one that
//     confused the expert-major row offsets (expert * intermediate into `up`, expert * latent into
//     `down`) would read a neighbouring expert. The one-expert-hit case fills every row of a
//     240-expert bank with NaN except one expert's, so any read of an unselected expert is not a
//     small numerical difference but a NaN in the output.
//   * a token with all its mass on one expert: the second rank's weight is zero, and that token's
//     output must be *bit-identical* to the same call with K = 1.
//   * fp32 accumulation. The Op contracts that the K contributions are summed in FP32, and the
//     case that shows it is a cancellation: two ranks whose expert outputs are equal and whose
//     weights are +1 and -1. The checkpoint's own BF16 path rounds each rank's contribution to
//     BF16 before accumulating, which is a different answer by about half a BF16 ulp of the
//     *terms* rather than of the result; the case measures both and requires the Op inside the
//     criterion while the per-rank-rounded model is outside it.
//   * the real banks. This family's MoE layers carry 240 to 496 experts at 1536 to 2560 rows around
//     a latent width of 1024, so the geometry arrives at run time and the row arithmetic has to be
//     right at both ends of it. Two real shapes are run end to end against the oracle.
//
// The Op computes each expert's pair in FP32 and rounds once, at the BF16 output, while the oracle
// evaluates the same formula in FP64 from the same represented operands. The criterion is the
// output grid's own resolution -- half a BF16 ulp is 1.95e-3 relative -- with a floor for a token
// whose ranks cancel. Measured maxima over every case are reported with NINFER_OP_REPORT_STATS=1.
#include "ninfer/ops/nemotron_h_moe_experts.h"

#include "ops/direct_bf16_weight.h"
#include "ops/nemotron_h_moe_experts_ref.h"
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
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::nemotron_h_moe_experts_ref;

namespace {

constexpr PointwiseCriterion kCriterion = {/*absolute*/ 1e-3, /*relative*/ 1e-2};
// The Op's own formula against a reference dump: the dump is FP32 arithmetic over the same
// operands, so an agreement at this level says the formula is the reference's rather than that two
// different formulas happen to be close.
constexpr ReductionCriterion kFormulaCriterion = {/*relative_l2*/ 1e-4, /*gross_absolute*/ 1e-3,
                                                  /*gross_relative_to_max*/ 1e-3};

constexpr std::int32_t kRealLatent = 1024; // the latent width every MoE layer projects into

std::vector<std::uint16_t> encode(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        bits[index] = f32_to_bf16(values[index]);
    }
    return bits;
}

// The bits of a quiet NaN, written directly: the one-expert-hit case needs a bank whose *other*
// experts are poison rather than data.
constexpr std::uint16_t kNaN = 0x7FC0;

// A bank whose every element is a distinct function of its (row, column) position, and cheap enough
// for the real sizes: 496 experts at 2560 rows of 1024 columns is 1.3e9 elements a bank. Both axes
// enter the mix, so an expert-major row offset of any size lands on different values.
direct_bf16_weight::HostWeight ramp_bank(std::int32_t rows, std::int32_t columns) {
    direct_bf16_weight::HostWeight bank;
    bank.n = rows;
    bank.k = columns;
    bank.bits.resize(static_cast<std::size_t>(rows) * columns);
    for (std::int32_t row = 0; row < rows; ++row) {
        for (std::int32_t column = 0; column < columns; ++column) {
            std::uint32_t hash = static_cast<std::uint32_t>(row) * 2654435761U ^
                                 static_cast<std::uint32_t>(column) * 40503U;
            hash ^= hash >> 13;
            const float value = (static_cast<float>(hash & 0x3FFU) - 512.0F) * (1.0F / 256.0F);
            bank.bits[static_cast<std::size_t>(row) * columns + column] = f32_to_bf16(value);
        }
    }
    return bank;
}

// Everything one call needs, so a refusal case can perturb exactly one field of a valid call. The
// geometry and the tensor extents are separate fields on purpose: a case that means to test the
// Op's own validation of `top_k` or of the latent width must still hand it *constructible* tensors,
// or the failure would come from this test's own Tensor rather than from the Op.
struct Call {
    std::int32_t latent       = 0;
    std::int32_t experts      = 0;
    std::int32_t top_k        = 0;
    std::int32_t intermediate = 0;
    std::int32_t tokens       = 0;
    std::vector<std::uint16_t> x;
    std::vector<std::int32_t> ids;
    std::vector<float> routing;
    direct_bf16_weight::HostWeight up;
    direct_bf16_weight::HostWeight down;
    DType x_dtype                 = DType::BF16;
    DType out_dtype               = DType::BF16;
    DType ids_dtype               = DType::I32;
    DType routing_dtype           = DType::FP32;
    QType up_format               = QType::BF16;
    QType down_format             = QType::BF16;
    bool x_column_major           = false;
    std::int32_t x_latent_extent  = 0; // 0: the geometry's latent width
    std::int32_t selection_extent = 0; // 0: the geometry's top_k
};

struct Outcome {
    std::vector<std::uint16_t> bits; // [latent, tokens], the latent axis the fastest
    std::vector<double> values;
    bool banks_preserved     = false;
    std::int64_t poison_left = 0;
};

// The call itself. Throws whatever the Op throws: the caller decides whether that is a failure.
Outcome execute(const Call& call) {
    direct_bf16_weight::DeviceWeight up_device(call.up);
    direct_bf16_weight::DeviceWeight down_device(call.down);
    GuardedDeviceBuffer x_device(call.x.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer ids_device(call.ids.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer routing_device(call.routing.size() * sizeof(float));
    GuardedDeviceBuffer out_device(call.x.size() * sizeof(std::uint16_t));
    x_device.copy_from_host(call.x.data(), x_device.bytes());
    ids_device.copy_from_host(call.ids.data(), ids_device.bytes());
    routing_device.copy_from_host(call.routing.data(), routing_device.bytes());
    // The output starts as NaN, which no finite operand can produce: a region the Op fails to write
    // therefore shows up as a NaN rather than passing for a small output. Counting against a finite
    // sentinel instead -- a value on the BF16 grid -- cannot work at the real bank magnitudes,
    // where the output grid is coarse enough that a legitimate result collides with the sentinel.
    const std::vector<std::uint16_t> poisoned(call.x.size(), kNaN);
    out_device.copy_from_host(poisoned.data(), out_device.bytes());

    const std::int32_t x_latent = call.x_latent_extent == 0 ? call.latent : call.x_latent_extent;
    const std::int32_t ranks    = call.selection_extent == 0 ? call.top_k : call.selection_extent;
    Tensor x_tensor(x_device.data(), call.x_dtype, {x_latent, call.tokens});
    if (call.x_column_major) {
        // The extents still match the geometry; the strides do not describe them.
        x_tensor.nb[0] = call.tokens;
    }
    Tensor out_tensor(out_device.data(), call.out_dtype, {call.latent, call.tokens});
    Tensor ids_tensor(ids_device.data(), call.ids_dtype, {ranks, call.tokens});
    Tensor routing_tensor(routing_device.data(), call.routing_dtype, {ranks, call.tokens});

    ops::NemotronHMoeExpertWeights weights;
    weights.up         = up_device.view();
    weights.down       = down_device.view();
    weights.up.qtype   = call.up_format;
    weights.down.qtype = call.down_format;
    const ops::NemotronHMoeExpertGeometry geometry{call.latent, call.experts, call.top_k,
                                                   call.intermediate};
    Tensor no_workspace;
    ops::nemotron_h_moe_experts(x_tensor, ids_tensor, routing_tensor, weights, geometry, out_tensor,
                                no_workspace, nullptr);
    cuda_synchronize();

    Outcome outcome;
    outcome.bits = from_device<std::uint16_t>(out_device.data(), call.x.size());
    outcome.values.resize(outcome.bits.size());
    for (std::size_t index = 0; index < outcome.bits.size(); ++index) {
        outcome.values[index] = static_cast<double>(bf16_to_f32(outcome.bits[index]));
    }
    outcome.poison_left =
        static_cast<std::int64_t>(std::count(outcome.bits.begin(), outcome.bits.end(), kNaN));
    outcome.banks_preserved = up_device.verify_preserved("nemotron_h_moe_experts up") == 0 &&
                              down_device.verify_preserved("nemotron_h_moe_experts down") == 0;
    return outcome;
}

// A call whose selection is the sampled pattern: several experts per token, and tokens that share
// an expert. `seed` drives the bank and the activation.
Call sampled_call(std::int32_t latent, std::int32_t experts, std::int32_t top_k,
                  std::int32_t intermediate, std::int32_t tokens, std::uint32_t seed) {
    Call call;
    call.latent       = latent;
    call.experts      = experts;
    call.top_k        = top_k;
    call.intermediate = intermediate;
    call.tokens       = tokens;
    call.up           = direct_bf16_weight::make_patterned(experts * intermediate, latent, seed);
    call.down = direct_bf16_weight::make_patterned(experts * latent, intermediate, seed + 1);
    std::vector<float> x(static_cast<std::size_t>(latent) * tokens);
    fill_uniform(x, seed + 2, -3.0F, 3.0F);
    round_to_bf16(x);
    call.x = encode(x);
    call.ids.resize(static_cast<std::size_t>(top_k) * tokens);
    call.routing.resize(static_cast<std::size_t>(top_k) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t rank = 0; rank < top_k; ++rank) {
            const std::size_t at = static_cast<std::size_t>(rank) * tokens + token;
            call.ids[at]         = (rank * 3 + token * 5) % experts;
            call.routing[at]     = 0.2F + 0.1F * static_cast<float>((rank + token) % 4);
        }
    }
    return call;
}

int compare_case(const char* label, const Call& call, Boundaries boundaries = {}) {
    const auto reference  = run(call.x, call.latent, call.tokens, call.ids, call.routing, call.up,
                                call.down, call.intermediate, call.top_k, call.experts, boundaries);
    const Outcome outcome = execute(call);
    int failures =
        verify_pointwise(std::string(label) + " out", outcome.values, reference, kCriterion);
    if (!outcome.banks_preserved) {
        std::cerr << label << ": a bank was modified\n";
        ++failures;
    }
    if (outcome.poison_left != 0) {
        std::cerr
            << label << ": " << outcome.poison_left
            << " output elements are still the NaN the buffer was poisoned with (an unwritten "
               "region, or a read of a NaN bank row)\n";
        ++failures;
    }
    return failures;
}

// A case that must be refused: the Op's own argument checking, not a numerical disagreement.
int refuse_case(const char* label, const Call& call) {
    try {
        (void)execute(call);
    } catch (const std::invalid_argument&) { return 0; }
    std::cerr << label << ": the Op accepted an invalid call\n";
    return 1;
}

// --- the dispatch ----------------------------------------------------------

// K = 2 over a bank of eight experts, with tokens sharing experts and one expert used by several
// tokens: the ordinary shape of the selection.
int dispatch_case() {
    return compare_case("nemotron_h_moe_experts dispatch", sampled_call(5, 8, 2, 6, 7, 31));
}

// A bank where only one expert is hit, and every other expert's every row is NaN. Read one of them
// and the output is not slightly wrong, it is NaN.
int one_expert_hit_case() {
    constexpr std::int32_t kLatent       = 32;
    constexpr std::int32_t kExperts      = 240;
    constexpr std::int32_t kIntermediate = 64;
    constexpr std::int32_t kTokens       = 3;
    constexpr std::int32_t kHit          = 7;
    Call call = sampled_call(kLatent, kExperts, 1, kIntermediate, kTokens, 33);
    for (std::int32_t token = 0; token < kTokens; ++token) { call.ids[token] = kHit; }
    const auto poison = [&](direct_bf16_weight::HostWeight& bank, std::int32_t first_row,
                            std::int32_t rows) {
        for (std::int32_t row = 0; row < bank.n; ++row) {
            if (row >= first_row && row < first_row + rows) { continue; }
            for (std::int32_t column = 0; column < bank.k; ++column) {
                bank.bits[static_cast<std::size_t>(row) * bank.k + column] = kNaN;
            }
        }
    };
    poison(call.up, kHit * kIntermediate, kIntermediate);
    poison(call.down, kHit * kLatent, kLatent);
    std::printf("nemotron_h_moe_experts: %d of %d experts' rows are NaN in the one-expert-hit "
                "bank\n",
                kExperts - 1, kExperts);
    return compare_case("nemotron_h_moe_experts one expert", call);
}

// A token whose second rank carries a zero weight must equal the same call at K = 1, bit for bit,
// while its neighbours must not: a zero weight contributes a zero addend, not a different path.
int single_expert_mass_case() {
    constexpr std::int32_t kLatent       = 4;
    constexpr std::int32_t kExperts      = 3;
    constexpr std::int32_t kIntermediate = 5;
    constexpr std::int32_t kTokens       = 4;
    constexpr std::int32_t kZeroToken    = 2;
    Call full = sampled_call(kLatent, kExperts, 2, kIntermediate, kTokens, 34);
    for (std::int32_t token = 0; token < kTokens; ++token) {
        full.routing[static_cast<std::size_t>(kTokens) + token] =
            token == kZeroToken ? 0.0F : full.routing[static_cast<std::size_t>(kTokens) + token];
    }
    Call single  = full;
    single.top_k = 1;
    single.ids.resize(kTokens);
    single.routing.resize(kTokens);

    int failures = compare_case("nemotron_h_moe_experts single expert mass", full);
    failures += compare_case("nemotron_h_moe_experts single rank", single);

    const Outcome both     = execute(full);
    const Outcome alone    = execute(single);
    std::int64_t differing = 0;
    for (std::int32_t token = 0; token < kTokens; ++token) {
        const std::size_t base = static_cast<std::size_t>(token) * kLatent;
        bool same              = true;
        for (std::int32_t h = 0; h < kLatent; ++h) {
            if (both.bits[base + h] != alone.bits[base + h]) { same = false; }
        }
        if (!same) { ++differing; }
        if (token == kZeroToken && !same) {
            std::cerr << "nemotron_h_moe_experts single expert mass: the token whose second rank "
                         "is weighted zero changed when that rank was removed\n";
            ++failures;
        }
    }
    if (differing == 0) {
        std::cerr << "nemotron_h_moe_experts single expert mass: the two calls agree everywhere, "
                     "so the comparison proves nothing\n";
        ++failures;
    }
    return failures;
}

// Two ranks whose expert outputs are almost equal and whose weights are +1 and -1: the result is a
// small residue, and what the case measures is the accumulator's dtype. An FP32 accumulation lands
// on that residue; an implementation that rounds each rank's contribution to BF16 -- which is what
// the checkpoint's own BF16 reference does, since its expert projection returns BF16 -- carries one
// rounding per rank instead of one at the output, and those roundings are of the *terms* rather
// than of the result.
//
// The terms have to differ for that to be visible at all: with equal terms the two roundings are
// the same rounding and cancel exactly, which is the first case this test was written with, and it
// measured a per-rank-rounded accumulator as indistinguishable from an FP32 one. The second
// expert's down matrix is therefore 1.5% away from the first's, which leaves the residue a small
// fraction of a term while the two roundings differ.
int accumulation_case() {
    constexpr std::int32_t kLatent       = 4;
    constexpr std::int32_t kExperts      = 2;
    constexpr std::int32_t kIntermediate = 5;
    constexpr std::int32_t kTokens       = 4;
    Call call = sampled_call(kLatent, kExperts, 2, kIntermediate, kTokens, 35);
    // Expert 1 becomes expert 0, so the two ranks compute the same value.
    const auto duplicate = [](direct_bf16_weight::HostWeight& bank, std::int32_t rows_per_expert) {
        for (std::int32_t row = 0; row < rows_per_expert; ++row) {
            for (std::int32_t column = 0; column < bank.k; ++column) {
                bank.bits[static_cast<std::size_t>(rows_per_expert + row) * bank.k + column] =
                    bank.bits[static_cast<std::size_t>(row) * bank.k + column];
            }
        }
    };
    duplicate(call.up, kIntermediate);
    duplicate(call.down, kLatent);
    // Expert 1's down rows then move together by 1.5%: the residue is a small fraction of a term
    // while the two terms are far enough apart (tens of BF16 ulps) that their roundings are
    // independent rather than the same rounding twice. A single flipped bit is not enough -- the
    // value it moves may be multiplied by an activation that is zero after the squared ReLU, and a
    // one-ulp difference rounds to the same BF16 value anyway.
    for (std::int32_t row = 0; row < kLatent; ++row) {
        for (std::int32_t column = 0; column < call.down.k; ++column) {
            const std::size_t at = static_cast<std::size_t>(kLatent + row) * call.down.k + column;
            call.down.bits[at]   = f32_to_bf16(bf16_to_f32(call.down.bits[at]) * 1.015F);
        }
    }
    std::vector<float> x(static_cast<std::size_t>(kLatent) * kTokens);
    fill_uniform(x, 36, -24.0F, 24.0F);
    round_to_bf16(x);
    call.x = encode(x);
    for (std::int32_t token = 0; token < kTokens; ++token) {
        const std::size_t at                                 = static_cast<std::size_t>(token);
        call.ids[at]                                         = 0;
        call.ids[static_cast<std::size_t>(kTokens) + at]     = 1;
        call.routing[at]                                     = 1.0F;
        call.routing[static_cast<std::size_t>(kTokens) + at] = -1.0F;
    }

    std::vector<double> term_magnitude;
    const auto exact =
        run(call.x, call.latent, call.tokens, call.ids, call.routing, call.up, call.down,
            call.intermediate, call.top_k, call.experts, {}, &term_magnitude);
    const auto per_rank_rounded =
        run(call.x, call.latent, call.tokens, call.ids, call.routing, call.up, call.down,
            call.intermediate, call.top_k, call.experts, Boundaries{true});
    const Outcome outcome = execute(call);
    int failures = verify_pointwise("nemotron_h_moe_experts cancellation out", outcome.values,
                                    exact, kCriterion);

    const std::int64_t count = static_cast<std::int64_t>(exact.size());
    const PointwiseStats rounded_stats =
        compute_pointwise_stats(per_rank_rounded.data(), exact.data(), count, kCriterion);
    report_pointwise_stats("nemotron_h_moe_experts per-rank BF16 accumulator", count, rounded_stats,
                           kCriterion);
    double worst_term = 0.0;
    for (const double value : term_magnitude) {
        worst_term = std::max(worst_term, std::abs(value));
    }
    std::printf("nemotron_h_moe_experts: the cancelling terms are %.6g at the largest, the FP64 "
                "result is %.6g, the Op's largest error is %.6g, and an accumulator that rounds "
                "each rank to BF16 is %.6g away (%.1f times the limit)\n",
                worst_term, exact.empty() ? 0.0 : exact.front(),
                compute_pointwise_stats(outcome.values.data(), exact.data(), count, kCriterion)
                    .maximum_absolute_error,
                rounded_stats.maximum_absolute_error, rounded_stats.maximum_criterion_ratio);
    if (rounded_stats.maximum_criterion_ratio < 2.0) {
        std::cerr << "nemotron_h_moe_experts cancellation: a per-rank BF16 accumulator lands "
                  << rounded_stats.maximum_criterion_ratio
                  << " times the limit, so this case cannot separate it from an FP32 one\n";
        ++failures;
    }
    return failures;
}

// --- the real banks --------------------------------------------------------

// The bank widths this family's MoE layers actually carry, at the latent width they all project
// into: 2048 rows for a 240-expert layer and 2560 for a 496-expert one, top-9.
int real_shape_case(const char* label, std::int32_t experts, std::int32_t intermediate) {
    constexpr std::int32_t kTopK   = 9;
    constexpr std::int32_t kTokens = 2;
    Call call;
    call.latent       = kRealLatent;
    call.experts      = experts;
    call.top_k        = kTopK;
    call.intermediate = intermediate;
    call.tokens       = kTokens;
    call.up           = ramp_bank(experts * intermediate, kRealLatent);
    call.down         = ramp_bank(experts * kRealLatent, intermediate);
    std::vector<float> x(static_cast<std::size_t>(kRealLatent) * kTokens);
    fill_uniform(x, 37, -3.0F, 3.0F);
    round_to_bf16(x);
    call.x = encode(x);
    call.ids.resize(static_cast<std::size_t>(kTopK) * kTokens);
    call.routing.resize(static_cast<std::size_t>(kTopK) * kTokens);
    for (std::int32_t token = 0; token < kTokens; ++token) {
        for (std::int32_t rank = 0; rank < kTopK; ++rank) {
            const std::size_t at = static_cast<std::size_t>(rank) * kTokens + token;
            call.ids[at]         = (rank * 7 + token * 3) % experts;
            call.routing[at]     = 0.05F + 0.01F * static_cast<float>(rank);
        }
    }
    return compare_case(label, call);
}

// --- refusals --------------------------------------------------------------

int refusal_cases() {
    int failures     = 0;
    const Call valid = sampled_call(6, 8, 2, 10, 5, 38);

    // A bank that is one expert short of what the routing can name.
    {
        Call call = valid;
        call.up   = direct_bf16_weight::make_patterned(call.experts * call.intermediate - 1,
                                                       call.latent, 39);
        failures += refuse_case("up bank one expert short", call);
    }
    // The two banks describing different expert counts.
    {
        Call call = valid;
        call.down = direct_bf16_weight::make_patterned(call.experts * call.latent + call.latent,
                                                       call.intermediate, 40);
        failures += refuse_case("down bank one expert long", call);
    }
    // A bank bound at a latent width the geometry does not name.
    {
        Call call = valid;
        call.up   = direct_bf16_weight::make_patterned(call.experts * call.intermediate,
                                                       call.latent + 1, 41);
        failures += refuse_case("up bank at another latent width", call);
    }
    {
        Call call = valid;
        call.down = direct_bf16_weight::make_patterned(call.experts * call.latent,
                                                       call.intermediate + 1, 42);
        failures += refuse_case("down bank at another intermediate width", call);
    }
    // An activation tensor whose own extent is not the geometry's latent width.
    {
        Call call            = valid;
        call.x_latent_extent = valid.latent + 1;
        failures += refuse_case("activation at another latent width", call);
    }
    // A selection whose rank extent is not the geometry's top_k, and one the geometry admits no
    // rank count for at all. The tensor stays constructible in both, so what refuses the call is
    // the Op.
    {
        Call call             = valid;
        call.selection_extent = valid.top_k + 1;
        failures += refuse_case("selection of the wrong shape", call);
    }
    {
        Call call             = valid;
        call.top_k            = valid.experts + 1;
        call.selection_extent = valid.top_k;
        failures += refuse_case("more ranks than experts", call);
    }
    {
        Call call             = valid;
        call.top_k            = 0;
        call.selection_extent = valid.top_k;
        failures += refuse_case("no ranks", call);
    }
    {
        Call call          = valid;
        call.routing_dtype = DType::BF16;
        failures += refuse_case("BF16 routing weights", call);
    }
    {
        Call call    = valid;
        call.x_dtype = DType::FP32;
        failures += refuse_case("FP32 activation", call);
    }
    {
        Call call      = valid;
        call.out_dtype = DType::FP32;
        failures += refuse_case("FP32 output", call);
    }
    {
        Call call      = valid;
        call.ids_dtype = DType::FP32;
        failures += refuse_case("FP32 expert ids", call);
    }
    // An activation whose extents match but whose strides do not.
    {
        Call call           = valid;
        call.x_column_major = true;
        failures += refuse_case("transposed activation", call);
    }
    // A bank in a format no route reads yet.
    {
        Call call      = valid;
        call.up_format = QType::NVFP4;
        failures += refuse_case("NVFP4 up bank", call);
    }
    {
        Call call        = valid;
        call.down_format = QType::Q4_G32_FP16_ROWS;
        failures += refuse_case("int4 down bank", call);
    }
    // A geometry that is not positive, through the workspace query as well as the call.
    {
        Call call         = valid;
        call.intermediate = 0;
        failures += refuse_case("zero intermediate width", call);
        const ops::NemotronHMoeExpertGeometry geometry{call.latent, call.experts, call.top_k,
                                                       call.intermediate};
        bool threw = false;
        try {
            (void)ops::nemotron_h_moe_experts_workspace_bytes(geometry, QType::BF16, call.tokens);
        } catch (const std::invalid_argument&) { threw = true; }
        if (!threw) {
            std::cerr << "workspace query: a non-positive geometry was accepted\n";
            ++failures;
        }
    }
    {
        const ops::NemotronHMoeExpertGeometry geometry{valid.latent, valid.experts, valid.top_k,
                                                       valid.intermediate};
        bool threw = false;
        try {
            (void)ops::nemotron_h_moe_experts_workspace_bytes(geometry, QType::BF16, 0);
        } catch (const std::invalid_argument&) { threw = true; }
        if (!threw) {
            std::cerr << "workspace query: zero tokens were accepted\n";
            ++failures;
        }
    }
    // The route needs no scratch, so the query reports zero and an unbound workspace is a valid
    // call rather than a violation (which every case above already exercises).
    {
        const ops::NemotronHMoeExpertGeometry geometry{valid.latent, valid.experts, valid.top_k,
                                                       valid.intermediate};
        if (ops::nemotron_h_moe_experts_workspace_bytes(geometry, QType::BF16, valid.tokens) != 0) {
            std::cerr << "workspace query: the per-token route reports scratch\n";
            ++failures;
        }
    }
    return failures;
}

// --- the reference dump ----------------------------------------------------

struct DumpTensor {
    std::vector<float> values;
    std::vector<std::int32_t> shape;
};

std::map<std::string, DumpTensor> read_dump(const std::string& directory,
                                            const std::string& manifest_name) {
    std::ifstream manifest(directory + "/" + manifest_name);
    if (!manifest) { throw std::runtime_error("no " + manifest_name + " in " + directory); }
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
        std::int64_t elements = 1;
        while (std::getline(extents, extent, 'x')) {
            const std::int32_t dimension = std::stoi(extent);
            tensor.shape.push_back(dimension);
            elements *= dimension;
        }
        std::ifstream file(directory + "/" + line.substr(shape_end + 1), std::ios::binary);
        if (!file) { throw std::runtime_error("missing dump file: " + line); }
        const std::vector<char> bytes((std::istreambuf_iterator<char>(file)),
                                      std::istreambuf_iterator<char>());
        if (static_cast<std::int64_t>(bytes.size()) != elements * sizeof(float)) {
            throw std::runtime_error("dump file does not match its manifest shape: " + line);
        }
        tensor.values.resize(bytes.size() / sizeof(float));
        std::memcpy(tensor.values.data(), bytes.data(), bytes.size());
        tensors.emplace(line.substr(0, name_end), std::move(tensor));
    }
    return tensors;
}

const DumpTensor& require_dump(const std::map<std::string, DumpTensor>& tensors,
                               const std::string& name, const std::vector<std::int32_t>& shape) {
    const auto found = tensors.find(name);
    if (found == tensors.end()) {
        throw std::runtime_error("the dump has no " + name +
                                 ": regenerate it with tools/verify/nemotron_h_reference.py and "
                                 "tools/verify/nemotron_h_expert_banks.py");
    }
    if (found->second.shape != shape) {
        std::ostringstream got;
        for (const std::int32_t extent : found->second.shape) { got << extent << 'x'; }
        throw std::runtime_error("the dump's " + name + " is " + got.str() +
                                 " rather than the tiny config's shape");
    }
    return found->second;
}

// The tiny configuration's MoE layers: eight experts, two ranks, latent 16, intermediate 32, and
// 130 tokens so the dump runs the chunked path.
constexpr std::int32_t kTinyLatent       = 16;
constexpr std::int32_t kTinyExperts      = 8;
constexpr std::int32_t kTinyIntermediate = 32;
constexpr std::int32_t kTinyTopK         = 2;
constexpr std::int32_t kTinyTokens       = 130;

// The reference's own output for this Op's boundary, in the latent space, from an independent
// implementation and an independent model construction.
//
// The tiny model runs in FP32 while this Op's contract is BF16 inputs and banks, so the comparison
// is bounded rather than required to vanish, and what it is bounded by is measured here rather than
// assumed: the oracle is evaluated three ways -- over the dump's exact operands, over the BF16
// operands the Op reads, and by the Op itself -- so the number that comes out separates the operand
// representation from the Op's own arithmetic.
int reference_dump_case(const std::string& directory) {
    const auto stages = read_dump(directory, "manifest.txt");
    std::map<std::string, DumpTensor> banks;
    try {
        banks = read_dump(directory, "expert_banks_manifest.txt");
    } catch (const std::exception& error) {
        std::cerr << "nemotron_h_moe_experts: " << error.what()
                  << "\n  run: /usr/bin/python3 tools/verify/nemotron_h_expert_banks.py --dump "
                  << directory << '\n';
        return 1;
    }

    int failures = 0;
    for (const std::int32_t layer : {1, 3}) {
        const std::string prefix = "model.layers." + std::to_string(layer) + ".mixer";
        const auto& latent_input =
            require_dump(stages, prefix + ".fc1_latent_proj", {kTinyTokens, kTinyLatent});
        const auto& expected =
            require_dump(stages, prefix + ".experts", {kTinyTokens, kTinyLatent});
        const auto& dumped_weights =
            require_dump(stages, prefix + ".gate.1", {kTinyTokens, kTinyTopK});
        const auto& dumped_ids = require_dump(stages, prefix + ".gate.2", {kTinyTokens, kTinyTopK});
        const auto& up_bank    = require_dump(banks, prefix + ".experts.up_proj",
                                              {kTinyExperts * kTinyIntermediate, kTinyLatent});
        const auto& down_bank  = require_dump(banks, prefix + ".experts.down_proj",
                                              {kTinyExperts * kTinyLatent, kTinyIntermediate});

        Call call;
        call.latent       = kTinyLatent;
        call.experts      = kTinyExperts;
        call.top_k        = kTinyTopK;
        call.intermediate = kTinyIntermediate;
        call.tokens       = kTinyTokens;
        call.x.reserve(latent_input.values.size());
        for (const float value : latent_input.values) { call.x.push_back(f32_to_bf16(value)); }
        call.up.n = kTinyExperts * kTinyIntermediate;
        call.up.k = kTinyLatent;
        call.up.bits.reserve(up_bank.values.size());
        for (const float value : up_bank.values) { call.up.bits.push_back(f32_to_bf16(value)); }
        call.down.n = kTinyExperts * kTinyLatent;
        call.down.k = kTinyIntermediate;
        call.down.bits.reserve(down_bank.values.size());
        for (const float value : down_bank.values) { call.down.bits.push_back(f32_to_bf16(value)); }
        // The dump's selection is [tokens, ranks]; the Op's is [ranks, tokens].
        call.ids.resize(static_cast<std::size_t>(kTinyTokens) * kTinyTopK);
        call.routing.resize(static_cast<std::size_t>(kTinyTokens) * kTinyTopK);
        double worst_index_cast = 0.0;
        for (std::int32_t token = 0; token < kTinyTokens; ++token) {
            for (std::int32_t rank = 0; rank < kTinyTopK; ++rank) {
                const std::size_t from = static_cast<std::size_t>(token) * kTinyTopK + rank;
                const std::size_t to   = static_cast<std::size_t>(rank) * kTinyTokens + token;
                const double raw       = dumped_ids.values[from];
                call.ids[to]           = static_cast<std::int32_t>(std::llround(raw));
                worst_index_cast =
                    std::max(worst_index_cast, std::abs(raw - static_cast<double>(call.ids[to])));
                call.routing[to] = dumped_weights.values[from];
            }
        }
        if (worst_index_cast > 1e-3) {
            std::cerr << "layer " << layer << ": the dumped indices are not integral\n";
            ++failures;
        }

        // What the Op reads, and what the reference computed with.
        const auto at_represented =
            run(call.x, call.latent, call.tokens, call.ids, call.routing, call.up, call.down,
                call.intermediate, call.top_k, call.experts);
        const auto at_exact = run_fp32(
            latent_input.values, call.latent, call.tokens, call.ids, call.routing,
            [&](int row, int column) {
                return static_cast<double>(
                    up_bank.values[static_cast<std::size_t>(row) * kTinyLatent + column]);
            },
            [&](int row, int column) {
                return static_cast<double>(
                    down_bank.values[static_cast<std::size_t>(row) * kTinyIntermediate + column]);
            },
            call.experts, call.intermediate, call.top_k);

        const Outcome outcome = execute(call);
        std::string label = "nemotron_h_moe_experts reference dump layer " + std::to_string(layer);
        failures += verify_pointwise(label + " op vs the oracle at the represented operands",
                                     outcome.values, at_represented, kCriterion);

        // The oracle's own formula against the reference's numbers: not a tolerance on this Op's
        // arithmetic but a check that the two implement the same mathematics.
        const std::vector<double> dumped(expected.values.begin(), expected.values.end());
        failures +=
            verify_reduction(label + " oracle vs the dump", at_exact, dumped, kFormulaCriterion);
        const ReductionStats op_vs_dump = compute_reduction_stats(
            outcome.values.data(), dumped.data(), static_cast<std::int64_t>(dumped.size()));
        const ReductionStats perturbation = compute_reduction_stats(
            at_represented.data(), at_exact.data(), static_cast<std::int64_t>(at_exact.size()));
        const ReductionStats exact_vs_dump = compute_reduction_stats(
            at_exact.data(), dumped.data(), static_cast<std::int64_t>(dumped.size()));
        std::printf(
            "nemotron_h_moe_experts layer %d: op vs dump rel_l2=%.6g max_abs=%.6g | operand "
            "representation (BF16 against exact) rel_l2=%.6g max_abs=%.6g | oracle vs dump "
            "rel_l2=%.6g\n",
            layer, op_vs_dump.relative_l2, op_vs_dump.maximum_absolute_error,
            perturbation.relative_l2, perturbation.maximum_absolute_error,
            exact_vs_dump.relative_l2);
        if (op_vs_dump.relative_l2 > 1.5 * perturbation.relative_l2 + 1e-3) {
            std::cerr << "layer " << layer
                      << ": the Op disagrees with the dump by more than the operand "
                         "representation explains\n";
            ++failures;
        }
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "nemotron_h_moe_experts: no CUDA device, skipping\n";
        return 77;
    }
    int failures = 0;
    try {
        failures += dispatch_case();
        failures += one_expert_hit_case();
        failures += single_expert_mass_case();
        failures += accumulation_case();
        failures += real_shape_case("nemotron_h_moe_experts 240 experts at 2048", 240, 2048);
        failures += real_shape_case("nemotron_h_moe_experts 496 experts at 2560", 496, 2560);
        failures += refusal_cases();
        if (const char* dump = std::getenv("NINFER_NEMOTRON_REF")) {
            failures += reference_dump_case(dump);
        } else {
            std::cout << "nemotron_h_moe_experts: set NINFER_NEMOTRON_REF to a "
                         "tools/verify/nemotron_h_reference.py dump (with its expert banks) to "
                         "also compare against the reference\n";
        }
    } catch (const std::exception& error) {
        std::cerr << "nemotron_h_moe_experts: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << "nemotron_h_moe_experts: " << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "nemotron_h_moe_experts: ok\n";
    return 0;
}
