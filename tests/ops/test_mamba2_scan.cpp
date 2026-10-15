// The Mamba-2 selective scan against a naive sequential FP64 oracle.
//
// What this file pins, in order of how silently the alternative would fail:
//
//   * the state's axis order. Every other linear-attention state in this tree has both extents 128,
//     so a swap is invisible there; here the key extent is 128 against a value extent of 64 (and
//     the test also runs value extents of 4, 12, 16, 20 and 128), which makes the order observable
//     -- and the test asserts the *rejected* reading is far away rather than only that the accepted
//     one is close.
//   * the B/C grouping. Heads of one group share a B and a C, which is a claim the test can check
//     without an oracle: two heads given identical operands must produce identical output rows.
//   * the unscaled readout. The gated-linear-attention Ops lock their scale to 1/sqrt(128); this Op
//     must not, and the discriminator is a variant of the oracle that does.
//   * the FP32 state. A case with a near-unit decay accumulates over 1025 tokens and the test
//     asserts that a BF16 state image would have failed the criterion the kernel passes.

#include "ninfer/ops/mamba2_scan.h"

#include "core/arena.h"
#include "core/layout.h"
#include "core/linear_attention_state.h"

#include "ops/mamba2_ref.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kStateSize     = 128;
constexpr float kTimeStepMin = 0.001F; // the checkpoint's own floor

enum class Form { SingleState, Distinct, DistinctAlias };

const char* form_label(Form form) {
    switch (form) {
    case Form::SingleState:
        return "single";
    case Form::Distinct:
        return "distinct";
    case Form::DistinctAlias:
        return "exact-alias";
    }
    return "?";
}

// The `heads / groups` heads of a group share operands, so a case can assert the sharing directly.
constexpr int kGroupedHeads  = 4;
constexpr int kGroupedGroups = 2;

struct Case {
    const char* name;
    int heads;
    int value_dim;
    int groups;
    int tokens;
    bool zero_state   = false;
    bool weak_decay   = false; // A_log 0 with a dt_bias deep under the floor: dA = 0.999 per token
    bool saturated_dt = false; // dt_bias above the softplus plateau: dA underflows the state away
    bool equal_heads  = false; // every head carries identical operands
    float state_scale = 0.1F;
};

// The kernel is FP32 SIMT evaluating the same recurrence the oracle evaluates, so its whole
// distance to the oracle is the single BF16 rounding of the output plus FP32 drift. BF16 keeps
// eight significand bits, so a correct rounding moves a value by at most half an ulp: 2^-9
// = 1.95e-3 of an element just under the top of its binade and 2^-8 = 3.9e-3 of one just above the
// bottom. The relative-L2 bound is therefore the larger of the two -- in a short vector one element
// can carry the whole norm, which is exactly what happens here (the 24-element case peaks
// at 2.21e-3) -- and the gross bound covers a full 2^-8 of the largest reference element, which is
// where the error peak lands when it sits at that element rather than a binade below it.
constexpr ReductionCriterion output_criterion() {
    return {/*relative_l2=*/4.0e-3, /*gross_absolute=*/1.0e-5,
            /*gross_relative_to_max_reference=*/5.0e-3};
}

// The state carries no output rounding, so what is left is FP32 accumulation -- a random walk,
// measured 6.2e-8 relative L2 at T=64 -- and the FP32 decay, whose 2-ulp rounding multiplies the
// whole state at every step. Under ordinary decay the state is refreshed and that bias does not
// accumulate; under a near-unit decay it is carried for the whole sequence, so the bound is the
// decay's own rounding times the token count (measured 2.2e-6 at T=33 and 4.6e-5 at T=1025, against
// the 2.5e-7 per step the derivation predicts). The reference computes the same FP32 decay, so this
// is its arithmetic rather than a kernel defect, and the two bounds are separate for that reason.
constexpr ReductionCriterion state_criterion(const Case& test_case) {
    constexpr double kDecayRounding = 2.5e-7;
    return test_case.weak_decay
               ? ReductionCriterion{kDecayRounding * test_case.tokens, 1.0e-5, 1.0e-4}
               : ReductionCriterion{3.0e-7, 1.0e-6, 1.0e-5};
}

std::string case_label(const Case& test_case) {
    return std::string(test_case.name) + " H=" + std::to_string(test_case.heads) +
           " V=" + std::to_string(test_case.value_dim) + " G=" + std::to_string(test_case.groups) +
           " T=" + std::to_string(test_case.tokens);
}

// Drawn from the case's own generator, unlike the header's seed-taking helper, so a case's numbers
// do not depend on how many tensors it happens to fill.
void fill_case(std::vector<float>& values, std::mt19937& generator, float low, float high) {
    std::uniform_real_distribution<float> distribution(low, high);
    for (float& value : values) { value = distribution(generator); }
}

mamba2_ref::Inputs make_inputs(const Case& test_case, std::uint32_t seed) {
    mamba2_ref::Inputs in;
    in.value_dim     = test_case.value_dim;
    in.heads         = test_case.heads;
    in.groups        = test_case.groups;
    in.tokens        = test_case.tokens;
    in.time_step_min = kTimeStepMin;

    const std::size_t vector_size =
        static_cast<std::size_t>(test_case.value_dim) * test_case.heads * test_case.tokens;
    const std::size_t group_size =
        static_cast<std::size_t>(kStateSize) * test_case.groups * test_case.tokens;
    in.x.resize(vector_size);
    in.b.resize(group_size);
    in.c.resize(group_size);
    in.dt.resize(static_cast<std::size_t>(test_case.heads) * test_case.tokens);
    in.dt_bias.resize(static_cast<std::size_t>(test_case.heads));
    in.a_log.resize(static_cast<std::size_t>(test_case.heads));
    in.d_skip.resize(static_cast<std::size_t>(test_case.heads));
    in.state.resize(static_cast<std::size_t>(kStateSize) * test_case.value_dim * test_case.heads);

    std::mt19937 generator(seed);
    fill_case(in.x, generator, -1.5F, 1.5F); // what the convolution leaves after its SiLU
    fill_case(in.b, generator, -1.0F, 1.0F);
    fill_case(in.c, generator, -1.0F, 1.0F);
    fill_case(in.dt, generator, -4.0F, 4.0F);
    fill_case(in.dt_bias, generator, -1.0F, 1.0F);
    fill_case(in.a_log, generator, 0.0F, 2.0F);
    fill_case(in.d_skip, generator, -1.0F, 1.0F);
    fill_case(in.state, generator, -test_case.state_scale, test_case.state_scale);

    if (test_case.zero_state) { std::fill(in.state.begin(), in.state.end(), 0.0F); }
    if (test_case.weak_decay) {
        // A = -1 with dt at the floor gives dA = exp(-0.001) = 0.999: the state keeps every step's
        // contribution instead of forgetting it, and the clamp is what sets the step because the
        // softplus argument sits deep in its tail.
        std::fill(in.a_log.begin(), in.a_log.end(), 0.0F);
        std::fill(in.dt_bias.begin(), in.dt_bias.end(), -12.0F);
    }
    if (test_case.saturated_dt) {
        // Above the plateau softplus returns its argument, so dA is exp(30*A) and the state is
        // wiped every step. The floor never engages here.
        std::fill(in.dt_bias.begin(), in.dt_bias.end(), 30.0F);
    }
    if (test_case.equal_heads) {
        // Heads inside one group get identical operands, so only the group's B/C can separate them.
        const int per_group = test_case.heads / test_case.groups;
        for (int head = 1; head < test_case.heads; ++head) {
            const int source = head - (head % per_group);
            if (source == head) { continue; }
            // Both x and dt are head-major with the token axis slower than the head axis, so the
            // two heads' rows interleave and neither copy is a contiguous run.
            for (int token = 0; token < test_case.tokens; ++token) {
                const auto row   = static_cast<std::size_t>(token) * test_case.heads;
                const auto x_row = static_cast<std::ptrdiff_t>(row) * test_case.value_dim;
                std::copy_n(in.x.begin() + x_row + source * test_case.value_dim,
                            test_case.value_dim, in.x.begin() + x_row + head * test_case.value_dim);
                in.dt[row + static_cast<std::size_t>(head)] =
                    in.dt[row + static_cast<std::size_t>(source)];
            }
            in.dt_bias[static_cast<std::size_t>(head)] =
                in.dt_bias[static_cast<std::size_t>(source)];
            in.a_log[static_cast<std::size_t>(head)]  = in.a_log[static_cast<std::size_t>(source)];
            in.d_skip[static_cast<std::size_t>(head)] = in.d_skip[static_cast<std::size_t>(source)];
            const std::size_t plane = static_cast<std::size_t>(kStateSize) * test_case.value_dim;
            std::copy_n(in.state.begin() + static_cast<std::ptrdiff_t>(source) * plane, plane,
                        in.state.begin() + static_cast<std::ptrdiff_t>(head) * plane);
        }
    }

    round_to_bf16(in.x);
    round_to_bf16(in.b);
    round_to_bf16(in.c);
    round_to_bf16(in.dt);
    return in;
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        bits[index] = f32_to_bf16(values[index]);
    }
    return bits;
}

std::vector<double> doubles(const std::vector<float>& values) {
    return std::vector<double>(values.begin(), values.end());
}

double relative_l2(const std::vector<double>& got, const std::vector<double>& rejected) {
    if (got.size() != rejected.size()) { throw std::logic_error("relative_l2: size mismatch"); }
    long double squared_error     = 0.0L;
    long double squared_reference = 0.0L;
    for (std::size_t index = 0; index < got.size(); ++index) {
        const double error = got[index] - rejected[index];
        squared_error += static_cast<long double>(error) * error;
        squared_reference += static_cast<long double>(rejected[index]) * rejected[index];
    }
    return std::sqrt(static_cast<double>(squared_error)) /
           std::max(std::sqrt(static_cast<double>(squared_reference)), 1.0e-30);
}

struct DeviceOperands {
    explicit DeviceOperands(const mamba2_ref::Inputs& in)
        : x(to_device_bf16(in.x)), b(to_device_bf16(in.b)), c(to_device_bf16(in.c)),
          dt(to_device_bf16(in.dt)), dt_bias(to_device_f32(in.dt_bias)),
          a_log(to_device_f32(in.a_log)), d_skip(to_device_f32(in.d_skip)) {}

    DeviceBuffer x;
    DeviceBuffer b;
    DeviceBuffer c;
    DeviceBuffer dt;
    DeviceBuffer dt_bias;
    DeviceBuffer a_log;
    DeviceBuffer d_skip;
};

Tensor state_view(void* data, const Case& test_case) {
    return Tensor(data, DType::FP32, {kStateSize, test_case.value_dim, test_case.heads});
}

Tensor out_view(void* data, const Case& test_case) {
    return Tensor(data, DType::BF16, {test_case.value_dim, test_case.heads, test_case.tokens});
}

struct RunResult {
    std::vector<double> out;
    std::vector<double> state;         // what the call published
    std::vector<float> state_in_after; // distinct form only: the untouched input state
    int guard_failures = 0;
};

// One call in the requested state convention.
RunResult run_scan(const Case& test_case, const mamba2_ref::Inputs& in, DeviceOperands& device,
                   Form form) {
    const std::size_t state_bytes = in.state.size() * sizeof(float);
    GuardedDeviceBuffer state_a(state_bytes);
    GuardedDeviceBuffer state_b(state_bytes);
    GuardedDeviceBuffer out(static_cast<std::size_t>(in.x.size()) * sizeof(std::uint16_t));
    state_a.copy_from_host(in.state.data(), state_bytes);
    state_b.fill(0xff);
    out.fill(0xff);

    Tensor x =
        Tensor(device.x.p, DType::BF16, {test_case.value_dim, test_case.heads, test_case.tokens});
    Tensor b  = Tensor(device.b.p, DType::BF16, {kStateSize, test_case.groups, test_case.tokens});
    Tensor c  = Tensor(device.c.p, DType::BF16, {kStateSize, test_case.groups, test_case.tokens});
    Tensor dt = Tensor(device.dt.p, DType::BF16, {test_case.heads, test_case.tokens});
    Tensor dt_bias  = Tensor(device.dt_bias.p, DType::FP32, {test_case.heads});
    Tensor a_log    = Tensor(device.a_log.p, DType::FP32, {test_case.heads});
    Tensor d_skip   = Tensor(device.d_skip.p, DType::FP32, {test_case.heads});
    Tensor out_data = out_view(out.data(), test_case);
    Tensor state_in = state_view(state_a.data(), test_case);

    switch (form) {
    case Form::SingleState:
        ops::mamba2_scan(x, b, c, dt, dt_bias, a_log, d_skip, in.time_step_min, state_in, out_data,
                         nullptr);
        break;
    case Form::Distinct: {
        Tensor state_out = state_view(state_b.data(), test_case);
        ops::mamba2_scan(x, b, c, dt, dt_bias, a_log, d_skip, in.time_step_min,
                         static_cast<const Tensor&>(state_in), state_out, out_data, nullptr);
        break;
    }
    case Form::DistinctAlias:
        ops::mamba2_scan(x, b, c, dt, dt_bias, a_log, d_skip, in.time_step_min,
                         static_cast<const Tensor&>(state_in), state_in, out_data, nullptr);
        break;
    }
    cuda_synchronize();

    RunResult result;
    result.out            = from_device_bf16(out.data(), in.x.size());
    result.state          = doubles(from_device<float>(
        form == Form::Distinct ? state_b.data() : state_a.data(), in.state.size()));
    result.guard_failures = state_a.verify_guards("state_in") + state_b.verify_guards("state_out") +
                            out.verify_guards("out");
    if (form == Form::Distinct) {
        result.state_in_after = from_device<float>(state_a.data(), in.state.size());
    }
    return result;
}

int verify_scan(const std::string& label, const Case& test_case, const mamba2_ref::Inputs& in,
                const mamba2_ref::Result& reference, const RunResult& got) {
    int failures = 0;
    failures += verify_reduction(label + " out", got.out, reference.out, output_criterion());
    failures += verify_reduction(label + " state", got.state, reference.final_state,
                                 state_criterion(test_case));
    if (got.guard_failures != 0) {
        std::cerr << label << ": device buffer guard was overwritten\n";
        ++failures;
    }
    if (!got.state_in_after.empty()) {
        failures +=
            verify_exact((label + " state-in unchanged").c_str(), got.state_in_after, in.state);
    }
    return failures;
}

int verify_operands_unchanged(const std::string& label, const mamba2_ref::Inputs& in,
                              DeviceOperands& device) {
    int failures = 0;
    failures += verify_exact((label + " x unchanged").c_str(),
                             from_device<std::uint16_t>(device.x, in.x.size()), bf16_bits(in.x));
    failures += verify_exact((label + " B unchanged").c_str(),
                             from_device<std::uint16_t>(device.b, in.b.size()), bf16_bits(in.b));
    failures += verify_exact((label + " C unchanged").c_str(),
                             from_device<std::uint16_t>(device.c, in.c.size()), bf16_bits(in.c));
    failures += verify_exact((label + " dt unchanged").c_str(),
                             from_device<std::uint16_t>(device.dt, in.dt.size()), bf16_bits(in.dt));
    failures += verify_exact((label + " dt_bias unchanged").c_str(),
                             from_device<float>(device.dt_bias, in.dt_bias.size()), in.dt_bias);
    failures += verify_exact((label + " A_log unchanged").c_str(),
                             from_device<float>(device.a_log, in.a_log.size()), in.a_log);
    failures += verify_exact((label + " D unchanged").c_str(),
                             from_device<float>(device.d_skip, in.d_skip.size()), in.d_skip);
    return failures;
}

// A discriminator only tests anything if the reading it rejects would have failed, and the margin
// that matters is relative: how much further the rejected reading is from the kernel's own result
// than the accepted one is. An absolute distance would be a statement about the operands (a state
// permutation is only observable through the initial state's share of the output, for instance),
// while this ratio is a statement about whether the case can tell the two apart at the suite's own
// accuracy.
constexpr double kDiscriminationRatio = 10.0;

int require_rejected(const std::string& label, const std::vector<double>& got,
                     const std::vector<double>& accepted, const std::vector<double>& rejected) {
    const double to_accepted = relative_l2(got, accepted);
    const double to_rejected = relative_l2(got, rejected);
    if (error_stats_enabled()) {
        std::printf("OP_REJECTED_DISTANCE case=%.*s accepted=%.17g rejected=%.17g ratio=%.17g\n",
                    static_cast<int>(label.size()), label.data(), to_accepted, to_rejected,
                    to_accepted > 0.0 ? to_rejected / to_accepted : 0.0);
    }
    if (!(to_rejected > to_accepted) || to_rejected < kDiscriminationRatio * to_accepted) {
        std::cerr << label << ": the rejected reading is only "
                  << (to_accepted > 0.0 ? to_rejected / to_accepted : 0.0)
                  << " times as far as the accepted one, so this case does not discriminate them\n";
        return 1;
    }
    return 0;
}

// Every state convention, against the same oracle result.
int scan_case(const Case& test_case, std::uint32_t seed) {
    const mamba2_ref::Inputs in        = make_inputs(test_case, seed);
    const mamba2_ref::Result reference = mamba2_ref::evaluate(in);
    DeviceOperands device(in);

    int failures = 0;
    for (const Form form : {Form::SingleState, Form::Distinct, Form::DistinctAlias}) {
        const RunResult got = run_scan(test_case, in, device, form);
        failures += verify_scan(case_label(test_case) + " " + form_label(form), test_case, in,
                                reference, got);
    }
    failures += verify_operands_unchanged(case_label(test_case), in, device);
    return failures;
}

// --- the decisions an oracle comparison alone cannot distinguish --------------------------------

// The state's axis order, the head-to-group mapping and the unscaled readout are all readings a
// wrong implementation makes silently, so each is evaluated from the same buffers here and the
// kernel's distance to the rejected reading is asserted against its distance to the accepted one.
//
// A permutation of the state's storage is only observable through the initial state's share of the
// output, so the case is built to make that share dominant: a near-unit decay puts the per-token
// update orders of magnitude below the state, which leaves the readout of the initial state as
// what the output mostly is. Under ordinary decay the same swap is still 34x the kernel's own
// error, but this case states the layout directly instead of relying on that ratio.
int discrimination_case() {
    Case test_case{"extent, grouping and scale discriminator", 4, 64, 2, 2};
    test_case.weak_decay               = true;
    test_case.state_scale              = 1.0F;
    const mamba2_ref::Inputs in        = make_inputs(test_case, 77213U);
    const mamba2_ref::Result reference = mamba2_ref::evaluate(in);
    DeviceOperands device(in);
    const RunResult got = run_scan(test_case, in, device, Form::Distinct);

    int failures       = verify_scan(case_label(test_case), test_case, in, reference, got);
    const auto swapped = mamba2_ref::evaluate(in, {.swap_state_extents = true});
    failures +=
        require_rejected("state extents swapped (out)", got.out, reference.out, swapped.out);
    failures += require_rejected("state extents swapped (state)", got.state, reference.final_state,
                                 swapped.final_state);
    const auto modulo = mamba2_ref::evaluate(in, {.group_modulo = true});
    failures += require_rejected("head mapped to h % groups", got.out, reference.out, modulo.out);
    const auto scaled = mamba2_ref::evaluate(in, {.scaled_readout = true});
    failures +=
        require_rejected("readout scaled by 1/sqrt(128)", got.out, reference.out, scaled.out);
    return failures;
}

// Heads of one group share a B and a C. Giving two of them identical operands makes that a claim
// the output has to satisfy on its own: nothing else can separate the two rows. It runs at the
// checkpoint's own 8 groups over 128 heads as well as at a small shape.
int grouping_case(Case test_case, std::uint32_t seed) {
    test_case.equal_heads              = true;
    const mamba2_ref::Inputs in        = make_inputs(test_case, seed);
    const mamba2_ref::Result reference = mamba2_ref::evaluate(in);
    DeviceOperands device(in);
    const RunResult got = run_scan(test_case, in, device, Form::SingleState);

    const std::string label = case_label(test_case);
    int failures            = verify_scan(label, test_case, in, reference, got);

    const int per_group = test_case.heads / test_case.groups;
    const auto row      = [&](int value, int head, int token) {
        return static_cast<std::size_t>((token * test_case.heads + head) * test_case.value_dim +
                                        value);
    };
    for (int head = 0; head < test_case.heads; ++head) {
        const int source = head - (head % per_group);
        if (source == head) { continue; }
        for (int token = 0; token < test_case.tokens; ++token) {
            for (int value = 0; value < test_case.value_dim; ++value) {
                if (got.out[row(value, head, token)] != got.out[row(value, source, token)]) {
                    std::cerr << label << ": head " << head << " and head " << source
                              << " share a group but do not share a result at value " << value
                              << " token " << token << '\n';
                    ++failures;
                }
            }
        }
    }

    const auto modulo = mamba2_ref::evaluate(in, {.group_modulo = true});
    failures += require_rejected("grouping h % groups", got.out, reference.out, modulo.out);
    return failures;
}

// A decay of 0.999 over 1025 tokens is the regime a lower-precision state cannot hold, so this case
// asserts that a BF16 state image would have failed the criterion the kernel passes.
int state_precision_case() {
    Case test_case{"long accumulation", 16, 64, 8, 1025};
    test_case.weak_decay               = true;
    const mamba2_ref::Inputs in        = make_inputs(test_case, 1207U);
    const mamba2_ref::Result reference = mamba2_ref::evaluate(in);
    DeviceOperands device(in);
    const RunResult got = run_scan(test_case, in, device, Form::Distinct);

    const std::string label = case_label(test_case);
    int failures            = verify_scan(label, test_case, in, reference, got);

    const auto bf16_state = mamba2_ref::evaluate(in, {.bf16_state = true});
    const auto count      = static_cast<std::int64_t>(reference.out.size());
    const ReductionStats kernel_stats =
        compute_reduction_stats(got.out.data(), reference.out.data(), count);
    const ReductionStats bf16_stats =
        compute_reduction_stats(bf16_state.out.data(), reference.out.data(), count);
    if (reduction_passes(bf16_stats, count, output_criterion())) {
        std::cerr << label << ": a BF16 state image would have passed this criterion (relative L2 "
                  << bf16_stats.relative_l2 << "), so the case does not exercise the FP32 state\n";
        ++failures;
    }
    if (!(kernel_stats.relative_l2 < bf16_stats.relative_l2)) {
        std::cerr << label << ": the kernel is no closer to the oracle than a BF16 state image\n";
        ++failures;
    }
    if (error_stats_enabled()) {
        std::printf("OP_STATE_PRECISION case=%s kernel_rel_l2=%.17g bf16_state_rel_l2=%.17g\n",
                    label.c_str(), kernel_stats.relative_l2, bf16_stats.relative_l2);
    }
    return failures;
}

// Call 2 starts from the state call 1 published, and call 3 from the state call 2 published, which
// is the handoff the engine performs between a prefill and a decode. Each oracle starts from what
// the device published, so the chain rather than a single call is what is compared.
int handoff_case() {
    int failures = 0;

    Case prefix{"handoff prefill", 128, 64, 8, 64};
    const mamba2_ref::Inputs first_in  = make_inputs(prefix, 50001U);
    const mamba2_ref::Result first_ref = mamba2_ref::evaluate(first_in);
    DeviceOperands first_device(first_in);
    const RunResult first = run_scan(prefix, first_in, first_device, Form::SingleState);
    failures += verify_scan(case_label(prefix) + " call 1", prefix, first_in, first_ref, first);
    failures += verify_operands_unchanged(case_label(prefix) + " call 1", first_in, first_device);

    Case step{"handoff step", 128, 64, 8, 5};
    mamba2_ref::Inputs second_in = make_inputs(step, 50002U);
    second_in.state.assign(first.state.begin(), first.state.end());
    const mamba2_ref::Result second_ref = mamba2_ref::evaluate(second_in);
    DeviceOperands second_device(second_in);
    const RunResult second = run_scan(step, second_in, second_device, Form::Distinct);
    failures += verify_scan(case_label(step) + " call 2", step, second_in, second_ref, second);

    Case decode{"handoff decode", 128, 64, 8, 1};
    mamba2_ref::Inputs third_in = make_inputs(decode, 50003U);
    third_in.state.assign(second.state.begin(), second.state.end());
    const mamba2_ref::Result third_ref = mamba2_ref::evaluate(third_in);
    DeviceOperands third_device(third_in);
    const RunResult third = run_scan(decode, third_in, third_device, Form::DistinctAlias);
    failures += verify_scan(case_label(decode) + " call 3", decode, third_in, third_ref, third);
    return failures;
}

// The recurrent state is a LinearAttentionStatePool plane, `[key_head_dim, value_head_dim,
// value_heads, slots]`. Mamba-2's state extent is `ssm_state_size` and its value extent is the
// mamba head dimension, and the Op refuses every other pairing (a pool planned with the two
// swapped reaches the Op as a `{64, 128, ...}` plane against a 64-wide x and is rejected by shape),
// so what is left to pin is the kernel's indexing of the plane itself: the two extents differ, so
// reading it with the value extent as the fastest axis is a *different* state, and this case runs
// the Op on the pool's own tensor and asserts the kernel is not doing that.
int pool_layout_case(std::uint32_t seed) {
    Case test_case{"pool view", 128, 64, 8, 4};
    test_case.state_scale = 0.25F;

    LayoutBuilder builder;
    const LinearAttentionStatePoolLayout layout =
        plan_linear_attention_state_pool(builder, {.layers         = 1,
                                                   .conv_channels  = 10240,
                                                   .conv_width     = 3,
                                                   .value_heads    = 128,
                                                   .value_head_dim = 64,
                                                   .key_head_dim   = 128,
                                                   .slot_count     = 1,
                                                   .conv_dtype     = DType::BF16});
    DeviceBuffer backing(builder.finish(256));
    LinearAttentionStatePool pool({backing.p, backing.bytes}, layout);
    Tensor recurrent = pool.layer_view(0).recurrent;

    const mamba2_ref::Inputs in        = make_inputs(test_case, seed);
    const mamba2_ref::Result reference = mamba2_ref::evaluate(in);
    DeviceOperands device(in);

    int failures = verify_exact("pool recurrent shape",
                                std::vector<std::int32_t>(recurrent.ne, recurrent.ne + 4),
                                std::vector<std::int32_t>{kStateSize, 64, 128, 1});
    const std::size_t state_bytes = in.state.size() * sizeof(float);
    if (recurrent.bytes() != state_bytes || !recurrent.is_contiguous()) {
        std::cerr << "pool layout: the recurrent plane is not the scan's state storage\n";
        ++failures;
    }
    cuda_check(cudaMemcpy(recurrent.data, in.state.data(), state_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy state into the pool plane");

    GuardedDeviceBuffer out(static_cast<std::size_t>(in.x.size()) * sizeof(std::uint16_t));
    out.fill(0xff);
    Tensor x        = Tensor(device.x.p, DType::BF16, {64, 128, test_case.tokens});
    Tensor b        = Tensor(device.b.p, DType::BF16, {kStateSize, 8, test_case.tokens});
    Tensor c        = Tensor(device.c.p, DType::BF16, {kStateSize, 8, test_case.tokens});
    Tensor dt       = Tensor(device.dt.p, DType::BF16, {128, test_case.tokens});
    Tensor dt_bias  = Tensor(device.dt_bias.p, DType::FP32, {128});
    Tensor a_log    = Tensor(device.a_log.p, DType::FP32, {128});
    Tensor d_skip   = Tensor(device.d_skip.p, DType::FP32, {128});
    Tensor out_data = out_view(out.data(), test_case);
    ops::mamba2_scan(x, b, c, dt, dt_bias, a_log, d_skip, in.time_step_min, recurrent, out_data,
                     nullptr);
    cuda_synchronize();

    RunResult got;
    got.out            = from_device_bf16(out.data(), in.x.size());
    got.state          = doubles(from_device<float>(recurrent.data, in.state.size()));
    got.guard_failures = out.verify_guards("pool out");
    failures += verify_scan("pool view", test_case, in, reference, got);

    // With the extents swapped the same storage describes a different plane, and 64 against 128 is
    // what makes that visible.
    const auto swapped = mamba2_ref::evaluate(in, {.swap_state_extents = true});
    failures +=
        require_rejected("pool view with swapped extents", got.out, reference.out, swapped.out);
    return failures;
}

// --- contract refusals -------------------------------------------------------------------------

int refusal_cases() {
    constexpr int heads     = 2;
    constexpr int value_dim = 8;
    constexpr int tokens    = 2;
    DeviceBuffer operands(1 << 20);
    DeviceBuffer parameters(4096);
    DeviceBuffer state_storage(1 << 20);

    const auto rejects = [](auto&& call) {
        try {
            call();
        } catch (const std::invalid_argument&) { return true; }
        return false;
    };

    // One conforming call, then one deviation from it per refusal. Every operand lives at its own
    // offset so that the conforming call is refused for the reason under test and not for an
    // overlap the harness created by accident.
    auto* operand_base = static_cast<std::uint8_t*>(operands.p);
    auto* param_base   = static_cast<std::uint8_t*>(parameters.p);
    Tensor x(operand_base, DType::BF16, {value_dim, heads, tokens});
    Tensor b(operand_base + 4096, DType::BF16, {kStateSize, 1, tokens});
    Tensor c(operand_base + 8192, DType::BF16, {kStateSize, 1, tokens});
    Tensor dt(operand_base + 12288, DType::BF16, {heads, tokens});
    Tensor out(operand_base + 16384, DType::BF16, {value_dim, heads, tokens});
    Tensor dt_bias(param_base, DType::FP32, {heads});
    Tensor a_log(param_base + 64, DType::FP32, {heads});
    Tensor d_skip(param_base + 128, DType::FP32, {heads});
    Tensor state(state_storage.p, DType::FP32, {kStateSize, value_dim, heads});

    int failures                = 0;
    const auto expect_rejection = [&](const char* what, auto&& call) {
        if (!rejects(call)) {
            std::cerr << "mamba2_scan accepted " << what << '\n';
            ++failures;
        }
    };

    // The refusals below only mean something if the conforming call is accepted.
    if (rejects([&] {
            ops::mamba2_scan(x, b, c, dt, dt_bias, a_log, d_skip, kTimeStepMin, state, out,
                             nullptr);
        })) {
        std::cerr << "mamba2_scan refused a conforming call\n";
        ++failures;
    }

    expect_rejection("a state extent other than 128", [&] {
        Tensor short_b(operand_base + 4096, DType::BF16, {64, 1, tokens});
        ops::mamba2_scan(x, short_b, c, dt, dt_bias, a_log, d_skip, kTimeStepMin, state, out,
                         nullptr);
    });
    expect_rejection("a head count its groups do not divide", [&] {
        Tensor odd_x(operand_base, DType::BF16, {value_dim, 3, tokens});
        Tensor odd_b(operand_base + 4096, DType::BF16, {kStateSize, 2, tokens});
        Tensor odd_dt(operand_base + 12288, DType::BF16, {3, tokens});
        Tensor odd_out(operand_base + 16384, DType::BF16, {value_dim, 3, tokens});
        Tensor odd_state(state_storage.p, DType::FP32, {kStateSize, value_dim, 3});
        ops::mamba2_scan(odd_x, odd_b, c, odd_dt, dt_bias, a_log, d_skip, kTimeStepMin, odd_state,
                         odd_out, nullptr);
    });
    expect_rejection("a B and a C with different group counts", [&] {
        Tensor wide_b(operand_base + 4096, DType::BF16, {kStateSize, 2, tokens});
        ops::mamba2_scan(x, wide_b, c, dt, dt_bias, a_log, d_skip, kTimeStepMin, state, out,
                         nullptr);
    });
    expect_rejection("a time step row that is not the token count", [&] {
        Tensor short_dt(operand_base + 12288, DType::BF16, {heads, tokens - 1});
        ops::mamba2_scan(x, b, c, short_dt, dt_bias, a_log, d_skip, kTimeStepMin, state, out,
                         nullptr);
    });
    expect_rejection("a state whose value extent differs from x", [&] {
        Tensor narrow_state(state_storage.p, DType::FP32, {kStateSize, value_dim + 1, heads});
        ops::mamba2_scan(x, b, c, dt, dt_bias, a_log, d_skip, kTimeStepMin, narrow_state, out,
                         nullptr);
    });
    expect_rejection("a NaN time step floor", [&] {
        ops::mamba2_scan(x, b, c, dt, dt_bias, a_log, d_skip,
                         std::numeric_limits<float>::quiet_NaN(), state, out, nullptr);
    });
    expect_rejection("a negative time step floor", [&] {
        ops::mamba2_scan(x, b, c, dt, dt_bias, a_log, d_skip, -1.0e-3F, state, out, nullptr);
    });
    expect_rejection("an FP32 time step", [&] {
        Tensor wide_dt(param_base, DType::FP32, {heads, tokens});
        ops::mamba2_scan(x, b, c, wide_dt, dt_bias, a_log, d_skip, kTimeStepMin, state, out,
                         nullptr);
    });
    expect_rejection("a BF16 state", [&] {
        Tensor narrow_state(state_storage.p, DType::BF16, {kStateSize, value_dim, heads});
        ops::mamba2_scan(x, b, c, dt, dt_bias, a_log, d_skip, kTimeStepMin,
                         static_cast<const Tensor&>(narrow_state), state, out, nullptr);
    });
    expect_rejection("a partially overlapping state pair", [&] {
        auto* shifted = static_cast<std::uint8_t*>(state_storage.p) + sizeof(float);
        Tensor state_out(shifted, DType::FP32, {kStateSize, value_dim, heads});
        ops::mamba2_scan(x, b, c, dt, dt_bias, a_log, d_skip, kTimeStepMin,
                         static_cast<const Tensor&>(state), state_out, out, nullptr);
    });
    expect_rejection("an output inside the state", [&] {
        Tensor overlapping_out(state_storage.p, DType::BF16, {value_dim, heads, tokens});
        ops::mamba2_scan(x, b, c, dt, dt_bias, a_log, d_skip, kTimeStepMin, state, overlapping_out,
                         nullptr);
    });
    expect_rejection("a null operand", [&] {
        Tensor empty(nullptr, DType::BF16, {value_dim, heads, tokens});
        ops::mamba2_scan(empty, b, c, dt, dt_bias, a_log, d_skip, kTimeStepMin, state, out,
                         nullptr);
    });
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = refusal_cases();
    failures += pool_layout_case(90210U);
    failures += discrimination_case();
    failures += grouping_case({"grouped heads", kGroupedHeads, 8, kGroupedGroups, 3}, 41001U);
    failures += grouping_case({"real grouped heads", 128, 64, 8, 2}, 41002U);
    failures += state_precision_case();
    failures += handoff_case();

    // The checkpoint's geometry: 128 value heads of 64 against a 128-wide state, 8 groups, and a
    // configured chunk of 128 -- so one chunk, a chunk and one token, and two chunks and one token
    // are the boundaries a decomposed scan would have to get right.
    failures += scan_case({"real decode", 128, 64, 8, 1}, 64001U);
    failures += scan_case({"real zero state", 128, 64, 8, 2, true}, 64002U);
    failures += scan_case({"real saturated dt", 128, 64, 8, 2, false, false, true}, 64003U);
    failures += scan_case({"real one chunk", 128, 64, 8, 128}, 64004U);
    failures += scan_case({"real chunk and one", 128, 64, 8, 129}, 64005U);
    failures += scan_case({"real two chunks and one", 128, 64, 8, 257}, 64006U);
    failures += scan_case({"real weak decay", 128, 64, 8, 33, false, true}, 64007U);

    // The value extent against the kernel's 16-value tile, at and around that boundary, plus the
    // checkpoint's own and the degenerate case where it equals the state extent, which is where a
    // swapped axis order stops being observable at all.
    for (const int value_dim : {15, 16, 17, 20, 64, 128}) {
        failures += scan_case({"value extent", 3, value_dim, 1, 2}, 70000U + value_dim);
    }

    // Group shapes: a single head, every head its own group, one group for everything, and a head
    // count that is neither a divisor nor a multiple of the tile.
    failures += scan_case({"single head", 1, 8, 1, 2}, 80001U);
    failures += scan_case({"every head its own group", 4, 8, 4, 2}, 80002U);
    failures += scan_case({"one group", 5, 8, 1, 3}, 80003U);
    failures += scan_case({"odd head count", 3, 20, 3, 2}, 80004U);

    std::cout << (failures == 0 ? "OK" : "FAIL") << " mamba2_scan correctness\n";
    return failures == 0 ? 0 : 1;
}
