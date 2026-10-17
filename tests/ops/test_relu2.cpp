// ninfer::ops::relu2 against its FP64 oracle, on the points that separate it from its neighbours.
//
// The activation is `max(z,0)^2`, and the case that matters is that it is *not* a softened one: at
// z = 2 it returns 4 where GELU(tanh) returns 1.955 and SiLU 1.762, at z = -0.5 it returns exactly
// zero where those return -0.154 and -0.188, and the asymmetry is real rather than a small
// disagreement -- the grid below includes both signs of the same magnitude. A criterion that a
// sigmoid-shaped activation could slip through is therefore not the criterion this suite relies on:
// the separating points are evaluated against both alternatives as well as against the oracle, and
// their margin over the criterion's own limit is required to be large.
//
// The input is the represented BF16 value promoted to FP32 and the result is rounded to BF16 once,
// so the only arithmetic between the oracle and the Op is one FP32 square and that rounding, whose
// relative size is at most half a BF16 ulp (1.95e-3).
#include "ninfer/ops/relu2.h"

#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// Half a BF16 ulp is 1.95e-3 relative, and a value near zero is exact (the square of a small BF16
// value is representable in FP32 well inside the output grid), so the relative limit sits two and a
// half times above the output grid and the absolute limit only catches a value that should have
// been exactly zero. Measured maxima are reported with NINFER_OP_REPORT_STATS=1.
constexpr PointwiseCriterion kCriterion = {/*absolute*/ 1e-6, /*relative*/ 5e-3};
// Every separating point lands at least four orders of magnitude outside that limit.
constexpr double kMinimumDiscrimination = 100.0;

double relu2_fp64(double value) {
    const double rectified = value > 0.0 ? value : 0.0;
    return rectified * rectified;
}

// The two activations this one is most easily confused with, for the discrimination report.
double silu_fp64(double value) { return value / (1.0 + std::exp(-value)); }

double gelu_tanh_fp64(double value) {
    constexpr double kSqrt2Pi = 0.79788456080286535588;
    return 0.5 * value * (1.0 + std::tanh(kSqrt2Pi * (value + 0.044715 * value * value * value)));
}

// Values that make the shape of the activation visible: both signs of five magnitudes over three
// binades, both signed zeros, and a seeded tail that gives the vector routes something to chew on
// rather than the same handful of points.
std::vector<float> grid_values(std::size_t tail, std::uint32_t seed) {
    std::vector<float> values;
    for (const float magnitude : {2.0F, 4.0F, 1.0F, 0.5F, 0.25F}) {
        for (const float sign : {1.0F, -1.0F}) { values.push_back(sign * magnitude); }
    }
    values.push_back(0.0F);
    values.push_back(-0.0F);
    values.push_back(8.0F);
    values.push_back(-8.0F);
    std::vector<float> tail_values(tail);
    fill_uniform(tail_values, seed, -6.0F, 6.0F);
    values.insert(values.end(), tail_values.begin(), tail_values.end());
    return values;
}

// The op is in place, so the device buffer is the input and the output at once.
int run_case(const char* label, std::vector<float> values, std::vector<std::int32_t> shape) {
    std::int64_t elements = 1;
    for (const std::int32_t extent : shape) { elements *= extent; }
    if (shape.size() < 1 || shape.size() > 4 ||
        elements != static_cast<std::int64_t>(values.size())) {
        std::cerr << label << ": the case's shape does not cover its values\n";
        return 1;
    }
    std::vector<std::int32_t> extents{1, 1, 1, 1};
    for (std::size_t index = 0; index < shape.size(); ++index) { extents[index] = shape[index]; }

    round_to_bf16(values);
    std::vector<double> reference(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        reference[index] = relu2_fp64(values[index]);
    }

    DeviceBuffer device = to_device_bf16(values);
    Tensor x(device.p, DType::BF16, {extents[0], extents[1], extents[2], extents[3]});
    ops::relu2(x, nullptr);
    cuda_synchronize();

    const auto got = from_device_bf16(device, values.size());
    int failures   = verify_pointwise(label, got, reference, kCriterion);

    // What separates this activation from its neighbours, measured rather than asserted: at every
    // separating point the two alternatives are far outside the criterion's own limit, so a suite
    // that had drifted onto either of them would fail here.
    double worst_silu = 0.0;
    double worst_gelu = 0.0;
    for (const double value : {2.0, 4.0, 1.0, -0.25, -0.5, -1.0, -2.0, -4.0}) {
        const double limit =
            kCriterion.absolute + kCriterion.relative * std::abs(relu2_fp64(value));
        worst_silu = std::max(
            worst_silu, error_limit_ratio(std::abs(silu_fp64(value) - relu2_fp64(value)), limit));
        worst_gelu =
            std::max(worst_gelu,
                     error_limit_ratio(std::abs(gelu_tanh_fp64(value) - relu2_fp64(value)), limit));
    }
    if (error_stats_enabled()) {
        std::printf(
            "OP_ERROR_STATS kind=discrimination case=%s silu_ratio=%.17g gelu_ratio=%.17g\n", label,
            worst_silu, worst_gelu);
    }
    if (std::min(worst_silu, worst_gelu) < kMinimumDiscrimination) {
        std::cerr << label << ": the activation is not separated from SiLU and GELU (" << worst_silu
                  << " and " << worst_gelu << " times the limit)\n";
        ++failures;
    }
    return failures;
}

// A refusal is part of the contract: the Op is BF16 and contiguous, and a caller that hands it
// anything else hears about it instead of getting a reinterpreted buffer.
int refuse_case(const char* label, DType dtype) {
    DeviceBuffer device(64);
    Tensor x(device.p, dtype, {4, 1, 1, 1});
    try {
        ops::relu2(x, nullptr);
    } catch (const std::invalid_argument&) { return 0; }
    std::cerr << label << ": the Op accepted an invalid dtype\n";
    return 1;
}

// A zero-element tensor is a no-op rather than a violation. Nothing is compared here: there is no
// element to compare, which is the point. The Tensor constructor rejects a zero extent, so the view
// is assembled by hand -- which is also the only way a caller can produce one.
int empty_case() {
    DeviceBuffer device(64);
    Tensor x;
    x.data  = device.p;
    x.dtype = DType::BF16;
    x.ne[0] = 0;
    x.nb[0] = static_cast<std::int64_t>(sizeof(std::uint16_t));
    try {
        ops::relu2(x, nullptr);
    } catch (const std::invalid_argument& error) {
        std::cerr << "relu2 empty: the Op refused a zero-element tensor: " << error.what() << '\n';
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "relu2: no CUDA device, skipping\n";
        return 77;
    }
    int failures = 0;
    try {
        // The aligned BF16x8 route: a whole number of 16-byte packs.
        failures += run_case("relu2 vectorized", grid_values(242, 11), {4, 64});
        // The pair route with its scalar tail: an odd element count.
        failures += run_case("relu2 paired tail", grid_values(241, 12), {255});
        // The single-element route.
        failures += run_case("relu2 scalar", {2.5F}, {1});
        // A rank the Op must not interpret.
        failures += run_case("relu2 rank three", grid_values(42, 13), {1, 4, 14});

        failures += refuse_case("relu2 fp32 input", DType::FP32);
        failures += refuse_case("relu2 i32 input", DType::I32);

        failures += empty_case();
    } catch (const std::exception& error) {
        std::cerr << "relu2: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << "relu2: " << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "relu2: ok\n";
    return 0;
}
