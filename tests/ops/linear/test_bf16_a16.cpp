#include "core/weight.h"
#include "ninfer/ops/linear.h"

#include "core/decode_graph.h"
#include "core/device.h"
#include "ops/direct_bf16_weight.h"
#include "ops/linear/bf16/bf16_dispatch.h"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::direct_bf16_weight;

constexpr ReductionCriterion kA16Tolerance{1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0};

std::vector<std::uint16_t> make_activation_bits(std::int32_t hidden, std::int32_t tokens) {
    std::vector<std::uint16_t> result(static_cast<std::size_t>(hidden) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t column = 0; column < hidden; ++column) {
            const int centered = ((column * 29 + token * 71 + 17) & 0xff) - 128;
            result[static_cast<std::size_t>(token) * hidden + column] =
                f32_to_bf16(static_cast<float>(centered) * (1.0F / 512.0F));
        }
    }
    return result;
}

std::vector<float> materialize(std::span<const std::uint16_t> bits) {
    std::vector<float> result(bits.size());
    for (std::size_t index = 0; index < bits.size(); ++index) {
        result[index] = bf16_to_f32(bits[index]);
    }
    return result;
}

// BF16 bit patterns with one implied and seven explicit significand bits and an exponent drawn from
// a decaying distribution, so a value spans about sixteen binades and a row's partial sums round.
//
// The two generators above cannot do that: their values are integer multiples of 2^-10 over a
// narrow range, so K = 4096 terms need about twenty significand bits, every partial sum is exactly
// representable, and two accumulation orders of such a row agree *by construction*. That is
// inaudible where the comparison is an FP64 oracle within a criterion, which is what they are for,
// and fatal for the block below, which would report max_abs = 0 for a kernel that reassociated K.
//
// Host-measured sensitivity of one such block (a four-way reassociation against the ascending
// chain, run on the host over this same generator): 0.049% of the (1024, 4096) shape's outputs and
// 0.021% of the (4096, 1024) shape's move by one BF16 ulp, and at T = 1 a pass moves something in
// about half of seeds and nothing in the rest. That is arithmetic rather than data: the two orders
// differ by ~2^-20 relative where a BF16 ulp is 2^-8, so a single pass over 1024 or 4096 elements
// is a weak detector by nature. At T = 1024 the same rates are ~510 differing elements of 1.0M and
// ~900 of 4.2M, which is decisive, and that point -- not the single-token one -- is this block's
// guard on the single-token route: the wide routes use their own launcher, so a reassociation
// confined to T = 1 is caught here only through the token-count sweep and through the model walk's
// latent stages (~5200 calls over 26.6M elements per pass, asserted to stay at exactly zero).
std::vector<std::uint16_t> latent_pattern(std::size_t count, std::uint32_t seed) {
    std::vector<std::uint16_t> result(count);
    std::uint32_t state = seed * 0x9e3779b9U + 1U;
    for (std::uint16_t& bits : result) {
        state                        = state * 1664525U + 1013904223U;
        const std::uint32_t draw     = state * 22695477U + 1U;
        const int leading            = std::countl_zero(draw | 1U);
        const std::uint16_t exponent = static_cast<std::uint16_t>(124 - (leading & 15));
        const std::uint16_t mantissa = static_cast<std::uint16_t>((state >> 3) & 0x7fU);
        const std::uint16_t sign     = static_cast<std::uint16_t>((state >> 16) & 0x8000U);
        // Exponent fields 109..124: normal, never zero, so no row is trivially equal for every
        // order.
        bits = static_cast<std::uint16_t>(sign | (exponent << 7) | mantissa);
    }
    return result;
}

HostWeight make_latent_weight(std::int32_t n, std::int32_t k, std::uint32_t seed) {
    HostWeight result;
    result.n    = n;
    result.k    = k;
    result.bits = latent_pattern(static_cast<std::size_t>(n) * k, seed);
    return result;
}

// The host control the single-token point needs, because a fixture can fail the identity check in
// two ways and only the first is visible in the output. An exactly-summable fixture cannot express
// an order difference at all (argued at the generator above). A fixture that is inexact but well
// conditioned can, and the *rate* is what makes it marginal: the two orders differ by
// ~sqrt(K)*2^-24 relative where a BF16 output ulp is 2^-8, so 1024 or 4096 output elements carry
// well under one expected move, and a single-token pass would then report max_abs = 0 for a
// reassociating route in most runs.
//
// So the exact T = 1 operands are walked here, on the host, in the fallback's own order and in the
// two chain counts the gemv schedule actually uses -- `AccumulatorChains` 4 and 8, each taking
// every C-th column and summing the chains in ascending order -- and the elements that move are
// counted. The block below requires the total to be nonzero: if it ever reaches zero, the identity
// assertion has stopped being able to fail on this fixture, and the fixture is what has to change
// rather than the kernel. BF16 operands make every product exact, so `product + acc` here is the
// same single rounding `fmaf(w, x, acc)` performs in the kernel, whatever the host compiler
// contracts to.
float chained_dot(const std::uint16_t* row_bits, std::span<const std::uint16_t> activation,
                  std::int32_t k, int chains) {
    std::array<float, 8> partials{};
    for (std::int32_t column = 0; column < k; ++column) {
        const float product = bf16_to_f32(row_bits[column]) *
                              bf16_to_f32(activation[static_cast<std::size_t>(column)]);
        const auto slot     = static_cast<std::size_t>(column & (chains - 1));
        partials[slot]      = product + partials[slot];
    }
    float total = partials[0];
    for (int chain = 1; chain < chains; ++chain) {
        total = partials[static_cast<std::size_t>(chain)] + total;
    }
    return total;
}

struct OrderSensitivity {
    std::size_t four_chain  = 0;
    std::size_t eight_chain = 0;
};

OrderSensitivity measure_order_sensitivity(const HostWeight& weight,
                                           std::span<const std::uint16_t> activation) {
    OrderSensitivity counts;
    for (std::int32_t row = 0; row < weight.n; ++row) {
        const std::uint16_t* row_bits =
            weight.bits.data() + static_cast<std::size_t>(row) * weight.k;
        const std::uint16_t ascending = f32_to_bf16(chained_dot(row_bits, activation, weight.k, 1));
        if (f32_to_bf16(chained_dot(row_bits, activation, weight.k, 4)) != ascending) {
            ++counts.four_chain;
        }
        if (f32_to_bf16(chained_dot(row_bits, activation, weight.k, 8)) != ascending) {
            ++counts.eight_chain;
        }
    }
    return counts;
}

std::vector<double> oracle_all_rows(const HostWeight& weight, std::span<const float> activation) {
    std::vector<double> result(static_cast<std::size_t>(weight.n));
    const unsigned available   = std::max(1U, std::thread::hardware_concurrency());
    const std::int32_t threads = std::min(weight.n, static_cast<std::int32_t>(available));
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(threads));
    for (std::int32_t thread = 0; thread < threads; ++thread) {
        const std::int32_t begin =
            static_cast<std::int32_t>((static_cast<std::int64_t>(weight.n) * thread) / threads);
        const std::int32_t end = static_cast<std::int32_t>(
            (static_cast<std::int64_t>(weight.n) * (thread + 1)) / threads);
        workers.emplace_back([&, begin, end] {
            for (std::int32_t row = begin; row < end; ++row) {
                result[static_cast<std::size_t>(row)] = dot_fp64(weight, row, activation);
            }
        });
    }
    for (std::thread& worker : workers) { worker.join(); }
    return result;
}

std::vector<std::int32_t> sampled_rows(std::int32_t rows) {
    std::vector<std::int32_t> result{0, 1, rows / 4, rows / 2, (3 * rows) / 4, rows - 2, rows - 1};
    if (rows == 14336) {
        result.insert(result.end(), {1023, 6143, 6144, 7167, 7168, 13311, 13312});
    }
    std::ranges::sort(result);
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

std::vector<std::int32_t> sampled_tokens(std::int32_t tokens) {
    if (tokens <= 32) {
        std::vector<std::int32_t> result(static_cast<std::size_t>(tokens));
        for (std::int32_t token = 0; token < tokens; ++token) {
            result[static_cast<std::size_t>(token)] = token;
        }
        return result;
    }
    std::vector<std::int32_t> result{0, 1, tokens / 2, tokens - 2, tokens - 1};
    std::ranges::sort(result);
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

int run_bf16_linear_case(DeviceWeight& weight, std::int32_t tokens, bool replay = false) {
    const std::int32_t rows                    = weight.host.n;
    const std::int32_t hidden                  = weight.host.k;
    std::vector<std::uint16_t> activation_bits = make_activation_bits(hidden, tokens);
    std::vector<float> activation              = materialize(activation_bits);
    DeviceBuffer device_activation             = to_device(activation_bits);
    GuardedDeviceBuffer guarded_output(static_cast<std::size_t>(rows) * tokens *
                                       sizeof(std::uint16_t));
    guarded_output.fill(0xff);

    Tensor x(device_activation.p, DType::BF16, {hidden, tokens});
    Tensor output(guarded_output.data(), DType::BF16, {rows, tokens});
    DeviceArena workspace(256);
    ops::linear(x, weight.view(), output, ops::LinearPolicy::A16Only, workspace, nullptr);
    cuda_synchronize();

    if (replay) {
        DeviceContext context;
        DecodeGraphDefinition definition;
        DecodeGraphExecutable graph;
        definition.capture(context.stream, [&] {
            ops::linear(x, weight.view(), output, ops::LinearPolicy::A16Only, workspace,
                        context.stream);
        });
        graph.instantiate(definition);
        graph.launch(context.stream);
        cuda_synchronize();
        for (auto& bits : activation_bits) bits ^= 0x8000;
        activation = materialize(activation_bits);
        device_activation.copy_from_host(activation_bits.data(), device_activation.bytes);
        guarded_output.fill(0xff);
        cuda_synchronize();
        graph.launch(context.stream);
        cuda_synchronize();
    }
    const std::string suffix = " T=" + std::to_string(tokens) + (replay ? " graph" : " eager");
    int failures             = guarded_output.verify_guards("BF16_A16 Linear output" + suffix);
    const std::vector<std::uint16_t> output_bits =
        from_device<std::uint16_t>(guarded_output.data(), static_cast<std::size_t>(rows) * tokens);
    for (std::size_t index = 0; index < output_bits.size(); ++index) {
        const std::uint16_t bits = output_bits[index];
        if (!std::isfinite(bf16_to_f32(bits))) {
            std::cerr << "BF16_A16 Linear output" << suffix << " element " << index
                      << " is not finite\n";
            ++failures;
            break;
        }
    }

    std::vector<double> actual;
    std::vector<double> expected;
    if (tokens == 1) {
        const std::vector<double> complete =
            oracle_all_rows(weight.host, std::span<const float>(activation));
        actual.reserve(rows);
        for (const std::uint16_t bits : output_bits) { actual.push_back(bf16_to_f32(bits)); }
        expected = complete;
    } else {
        const std::vector<std::int32_t> sampled       = sampled_rows(rows);
        const std::vector<std::int32_t> token_samples = sampled_tokens(tokens);
        actual.reserve(sampled.size() * token_samples.size());
        expected.reserve(actual.capacity());
        for (const std::int32_t row : sampled) {
            for (const std::int32_t token : token_samples) {
                actual.push_back(
                    bf16_to_f32(output_bits[static_cast<std::size_t>(token) * rows + row]));
                expected.push_back(dot_fp64(
                    weight.host, row,
                    std::span<const float>(
                        activation.data() + static_cast<std::size_t>(token) * hidden, hidden)));
            }
        }
    }
    failures += verify_reduction("BF16_A16 Linear [" + std::to_string(rows) + "," +
                                     std::to_string(hidden) + "]" + suffix,
                                 actual, expected, kA16Tolerance);
    const std::vector<std::uint16_t> activation_after =
        from_device<std::uint16_t>(device_activation, activation_bits.size());
    if (activation_after != activation_bits) {
        std::cerr << "BF16_A16 Linear" << suffix << " modified its activation\n";
        ++failures;
    }
    failures += weight.verify_preserved("BF16_A16 Linear weight" + suffix);
    return failures;
}

int run_selector_linear() {
    constexpr int n = 256, k = 5120, max_t = 1024;
    DeviceWeight weight(make_patterned(n, k, 419U));
    const auto bits       = make_activation_bits(k, max_t);
    const auto activation = materialize(bits);
    std::vector<double> oracle(n * max_t);
    std::vector<std::thread> workers;
    const int threads = std::min(32U, std::max(1U, std::thread::hardware_concurrency()));
    for (int worker = 0; worker < threads; ++worker)
        workers.emplace_back([&, worker] {
            for (int index = worker; index < n * max_t; index += threads) {
                const int token = index / n, row = index % n;
                oracle[index] = dot_fp64(weight.host, row,
                                         std::span<const float>(activation.data() + token * k, k));
            }
        });
    for (auto& worker : workers) worker.join();
    DeviceBuffer input = to_device(bits);
    auto negative      = bits;
    for (auto& value : negative) value ^= 0x8000;
    DeviceContext context;
    int failures   = 0;
    const auto run = [&](int tokens, bool replay) {
        const auto capacity = ops::linear_workspace_capacity_bytes(
            QType::BF16, n, k, ops::LinearPolicy::A16Only, tokens, tokens);
        DeviceArena scratch(std::max<std::size_t>(capacity, 256));
        GuardedDeviceBuffer output_buffer(static_cast<std::size_t>(n) * tokens * 2);
        Tensor x(input.p, DType::BF16, {k, tokens});
        Tensor output(output_buffer.data(), DType::BF16, {n, tokens});
        const auto launch = [&] {
            ops::linear(x, weight.view(), output, ops::LinearPolicy::A16Only, scratch,
                        context.stream);
        };
        DecodeGraphDefinition definition;
        DecodeGraphExecutable graph;
        cuda_synchronize();
        if (replay) {
            definition.capture(context.stream, launch);
            graph.instantiate(definition);
        }
        const std::string label =
            "BF16 selector T=" + std::to_string(tokens) + (replay ? " graph" : " eager");
        for (int phase = 0; phase < (replay ? 2 : 1); ++phase) {
            const auto& represented = phase == 0 ? bits : negative;
            if (phase) input.copy_from_host(represented.data(), input.bytes);
            output_buffer.fill(0xff);
            cuda_synchronize();
            if (replay)
                graph.launch(context.stream);
            else
                launch();
            cuda_synchronize(context.stream);
            failures += output_buffer.verify_guards(label);
            if (scratch.peak_used() > capacity || scratch.used() != 0) {
                std::cerr << label << ": workspace query/scope mismatch\n";
                ++failures;
            }
            const auto actual_bits = from_device<std::uint16_t>(output_buffer.data(), n * tokens);
            std::vector<double> actual(actual_bits.size()),
                expected(oracle.begin(), oracle.begin() + n * tokens);
            for (std::size_t i = 0; i < actual.size(); ++i) actual[i] = bf16_to_f32(actual_bits[i]);
            if (phase)
                for (auto& value : expected) value = -value;
            failures += verify_reduction(label, actual, expected, kA16Tolerance);
            if (from_device<std::uint16_t>(input, bits.size()) != represented) {
                std::cerr << label << ": modified input\n";
                ++failures;
            }
        }
        if (replay) input.copy_from_host(bits.data(), input.bytes);
    };
    for (int tokens = 1; tokens <= 120; ++tokens) run(tokens, false);
    for (int tokens : {121, 127, 128, 129, 256, 1024}) run(tokens, false);
    for (int tokens : {1, 7, 8, 15, 16, 63, 64, 65, 76, 77, 80, 81, 119, 120, 129, 1024})
        run(tokens, true);
    failures += weight.verify_preserved("BF16 selector weight");
    return failures;
}

int run_bf16_linear() {
    int failures = 0;
    DeviceWeight attention_weight(make_patterned(14336, 5120, 401U));
    DeviceWeight output_weight(make_patterned(5120, 6144, 409U));
    for (DeviceWeight* weight : {&attention_weight, &output_weight}) {
        for (int tokens = 1; tokens <= 33; ++tokens) {
            failures += run_bf16_linear_case(*weight, tokens);
        }
        for (int tokens : {127, 128, 129, 1024, 1536}) {
            failures += run_bf16_linear_case(*weight, tokens);
        }
        for (int tokens : {3, 7, 13, 19, 23, 25, 29}) {
            failures += run_bf16_linear_case(*weight, tokens, true);
        }
    }
    failures += run_selector_linear();

    // The Gemma 4 assistant drafter's shapes, which run one token per step, eager and replayed.
    constexpr std::array<std::array<std::int32_t, 2>, 7> kDraftShapes{{
        {1024, 10752},
        {8192, 1024},
        {16384, 1024},
        {1024, 8192},
        {1024, 16384},
        {262144, 1024},
        {5376, 1024},
    }};
    std::uint32_t seed = 431U;
    for (const auto& [n, k] : kDraftShapes) {
        DeviceWeight weight(make_patterned(n, k, seed++));
        failures += run_bf16_linear_case(weight, 1);
        failures += run_bf16_linear_case(weight, 1, true);
    }

    // The Gemma 4 vision tower's projections: patches of one image (9 to 2520) and its pooled soft
    // tokens (1 to 280), across every route boundary.
    DeviceWeight patch_weight(make_patterned(1152, 768, seed++));
    for (int tokens : {1, 2, 8, 9, 36, 63, 64, 65, 129, 630, 2520}) {
        failures += run_bf16_linear_case(patch_weight, tokens);
    }
    DeviceWeight soft_weight(make_patterned(5376, 1152, seed++));
    for (int tokens : {1, 3, 8, 9, 70, 128, 129, 280}) {
        failures += run_bf16_linear_case(soft_weight, tokens);
    }
    return failures;
}

// The dense fallback, over the shapes no registered kernel covers. That is now the Nemotron-H
// family's *other* BF16 widths -- its two latent projections have their own registered route, which
// the block after this one exercises -- whose K is 4096 or 1024 (or 64 at the scaled-down reference
// config) and whose N is 18560 or 164: no registered pair carries those, so before this route
// existed not one of that family's projections computed at all.
//
// 164 is deliberate and is the case the fallback's own indexing has to survive: it is not a
// multiple of any tile, so the row tail is predicated rather than padded. K is deliberately odd in
// one case for the same reason on the inner axis. T covers the single-token decode, the scaled
// reference config's own 130, and a prefill-wide 1024.
int run_bf16_generic() {
    int failures = 0;
    constexpr std::array<std::array<std::int32_t, 2>, 8> kNemotronShapes{{
        {164, 64},
        {64, 64},
        {32, 64},
        {8, 64},
        {16, 64},
        {64, 16},
        {64, 32},
        {256, 64},
    }};
    std::uint32_t seed = 7001U;
    for (const auto& [n, k] : kNemotronShapes) {
        DeviceWeight weight(make_patterned(n, k, seed++));
        for (int tokens : {1, 130, 1024}) { failures += run_bf16_linear_case(weight, tokens); }
    }
    DeviceWeight in_projection(make_patterned(164, 64, seed++));
    failures += run_bf16_linear_case(in_projection, 130, true);

    // The family's own in_projection at the checkpoint's width, at the token counts the decode loop
    // and a prefill step use. One pass at each, not the whole sweep: this route is a correctness
    // route, and its cost is linear in T.
    DeviceWeight wide(make_patterned(18560, 4096, seed++));
    for (int tokens : {1, 7, 130}) { failures += run_bf16_linear_case(wide, tokens); }

    // An odd K and an odd N, neither of which any tile divides.
    DeviceWeight odd(make_patterned(37, 65, seed++));
    failures += run_bf16_linear_case(odd, 1);
    failures += run_bf16_linear_case(odd, 129);
    return failures;
}

// The Nemotron-H latent projections the registry now carries, (1024, 4096) = `fc1_latent` and
// (4096, 1024) = `fc2_latent`, and the contract their routes have to keep: agreement with the dense
// fallback they displace, element for element, not within a criterion.
//
// The fallback's arithmetic is one FP32 accumulator per output element, K ascending, with a single
// BF16 rounding at the store, and the reference this family's forward walk compares against holds
// those values -- which is why both routes here are called on one input and their two outputs are
// compared *as bits*. A schedule that hands a row's K axis to the lanes of a warp and reduces the
// lane partials (what the gemv and SIMT cores beside them do) produces the same values almost
// everywhere and a different BF16 value wherever a partial sum lands the other side of a rounding
// boundary, so `max_abs` and `rel_l2` below are reported at full precision and expected to be
// exactly zero; a nonzero difference is a failure rather than a tolerance.
//
// Both routes are exercised: T=1 takes the one-token-per-thread launcher and T=2, 8 and 1024 take
// the four-token one, whose tail group has to drop its extra tokens and still agree. The operands
// come from `latent_pattern` rather than the file's other generators, and why that matters is
// argued at its definition: without it this block reports 0 for a kernel that reassociates K.
int run_bf16_latent_bit_identity() {
    int failures = 0;
    constexpr std::array<std::array<std::int32_t, 2>, 5> kLatentShapes{{
        {1024, 4096},
        {4096, 1024},
        // The Lightning 30B's DFlash companion: the tap fusion, the body's MLP and its down.
        {2688, 16128},
        {6144, 2688},
        {2688, 6144},
    }};
    std::uint32_t seed = 9101U;
    for (const auto& [n, k] : kLatentShapes) {
        const std::uint32_t weight_seed = seed++;
        const std::uint32_t input_seed  = seed++;
        DeviceWeight weight(make_latent_weight(n, k, weight_seed));
        // The control runs on the exact T = 1 input, before the routes are called.
        const OrderSensitivity sensitivity =
            measure_order_sensitivity(weight.host, latent_pattern(k, input_seed));
        std::printf("BF16 latent [%d,%d] T=1 control: a four-chain order moves %zu elements and an "
                    "eight-chain one %zu, of %d\n",
                    n, k, sensitivity.four_chain, sensitivity.eight_chain, n);
        if (sensitivity.four_chain + sensitivity.eight_chain == 0) {
            std::cerr
                << "BF16 latent [" << n << "," << k
                << "]: the fixture cannot express an accumulation-order difference at T=1, so "
                   "the identity assertion below cannot fail there\n";
            ++failures;
        }
        for (const std::int32_t tokens : {1, 2, 8, 1024}) {
            const std::string label = "BF16 latent [" + std::to_string(n) + "," +
                                      std::to_string(k) + "] T=" + std::to_string(tokens);
            if (ops::detail::select_bf16_a16_launch(n, k, tokens) ==
                &ops::detail::bf16_generic_launch) {
                std::cerr << label
                          << ": the shape is not registered, so this is the fallback "
                             "against itself\n";
                ++failures;
            }
            const std::vector<std::uint16_t> activation_bits =
                latent_pattern(static_cast<std::size_t>(k) * tokens, input_seed);
            DeviceBuffer device_activation = to_device(activation_bits);
            const std::size_t elements     = static_cast<std::size_t>(n) * tokens;
            GuardedDeviceBuffer registered(elements * sizeof(std::uint16_t));
            GuardedDeviceBuffer fallback(elements * sizeof(std::uint16_t));
            registered.fill(0xff);
            fallback.fill(0xff);

            Tensor x(device_activation.p, DType::BF16, {k, tokens});
            Tensor registered_out(registered.data(), DType::BF16, {n, tokens});
            Tensor fallback_out(fallback.data(), DType::BF16, {n, tokens});
            DeviceArena workspace(256);
            ops::linear(x, weight.view(), registered_out, ops::LinearPolicy::A16Only, workspace,
                        nullptr);
            ops::detail::bf16_generic_launch(x, weight.view(), fallback_out, nullptr);
            cuda_synchronize();

            failures += registered.verify_guards(label + " registered");
            failures += fallback.verify_guards(label + " fallback");
            const std::vector<std::uint16_t> got =
                from_device<std::uint16_t>(registered.data(), elements);
            const std::vector<std::uint16_t> want =
                from_device<std::uint16_t>(fallback.data(), elements);

            std::size_t differing = 0;
            double worst          = 0.0;
            double energy_actual  = 0.0;
            double energy_error   = 0.0;
            for (std::size_t index = 0; index < elements; ++index) {
                const double actual   = bf16_to_f32(got[index]);
                const double expected = bf16_to_f32(want[index]);
                if (got[index] != want[index]) { ++differing; }
                const double difference = std::abs(actual - expected);
                worst                   = std::max(worst, difference);
                energy_actual += actual * actual;
                energy_error += difference * difference;
            }
            const double relative_l2 = std::sqrt(energy_error / std::max(energy_actual, 1e-30));
            std::printf("%s: max_abs=%.17g rel_l2=%.17g (%zu of %zu elements differ)\n",
                        label.c_str(), worst, relative_l2, differing, elements);
            if (differing != 0) { ++failures; }

            failures += weight.verify_preserved(label + " weight");
            if (from_device<std::uint16_t>(device_activation, static_cast<std::size_t>(k) *
                                                                  tokens) != activation_bits) {
                std::cerr << label << ": modified its activation\n";
                ++failures;
            }
        }
    }
    return failures;
}

} // namespace

int main() {
    if (ninfer::test::cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        const int failures =
            run_bf16_linear() + run_bf16_generic() + run_bf16_latent_bit_identity();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " BF16_A16 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "BF16_A16 Linear: " << error.what() << '\n';
        return 1;
    }
}
