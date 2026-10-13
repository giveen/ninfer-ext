// ninfer::ops::residual_add_norm against an FP64 oracle that models each BF16 store, and against
// the unfused ops::rmsnorm -> residual_add -> mul_scalar sequence it replaces.
//
// The oracle evaluates the ideal in FP64 from the represented BF16 inputs and rounds at the three
// points the Op stores at. The exact-equality case is the property the Op exists for: at the widths
// the engine calls it with, the fused route must reproduce the unfused one bit for bit, because the
// pipeline's perplexity comparisons have no room for a reordered accumulation.
#include "ninfer/ops/residual_add_norm.h"

#include "ninfer/ops/mul_scalar.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"

#include "core/device.h"
#include "ops/norm_test_common.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::norm;

namespace {

// The three BF16 storages are exact in the oracle; what the bound must contain is the FP32-versus-
// FP64 difference in a row's sum of squares, which moves a value across a BF16 boundary only when
// it already sits within a relative 1e-7 of one, plus the rsqrt difference. Inputs are within one
// of zero, so the absolute term dominates wherever a sum cancels.
constexpr PointwiseCriterion kCriterion = {/*absolute*/ 1.5e-2, /*relative*/ 1.5e-2};

double round_bf16(double value) {
    return static_cast<double>(bf16_to_f32(f32_to_bf16(static_cast<float>(value))));
}

// The Op's formula in FP64, rounding exactly where the Op stores.
std::vector<double> oracle(const std::vector<float>& x, const std::vector<float>& weight,
                           const std::vector<float>& accumulator, std::int32_t d, std::int64_t rows,
                           float scale) {
    std::vector<double> reference(accumulator.size());
    for (std::int64_t row = 0; row < rows; ++row) {
        const std::size_t base = static_cast<std::size_t>(row) * static_cast<std::size_t>(d);
        double sum_squares     = 0.0;
        for (std::int32_t column = 0; column < d; ++column) {
            const double value = x[base + column];
            sum_squares += value * value;
        }
        const double inverse = 1.0 / std::sqrt(sum_squares / static_cast<double>(d) + kEps);
        for (std::int32_t column = 0; column < d; ++column) {
            const std::size_t index = base + static_cast<std::size_t>(column);
            const double normalized = round_bf16(x[index] * inverse * weight[column]);
            const double summed     = round_bf16(accumulator[index] + normalized);
            reference[index]        = round_bf16(summed * scale);
        }
    }
    return reference;
}

struct Hosts {
    std::vector<float> x, weight, accumulator;
};

Hosts make_hosts(std::int32_t d, std::int64_t rows, std::uint32_t seed) {
    Hosts hosts;
    hosts.x.resize(static_cast<std::size_t>(d) * rows);
    hosts.accumulator.resize(hosts.x.size());
    hosts.weight.resize(d);
    fill_uniform(hosts.x, seed, -1.0F, 1.0F);
    fill_uniform(hosts.accumulator, seed + 1U, -1.0F, 1.0F);
    fill_uniform(hosts.weight, seed + 2U, -1.0F, 1.0F);
    round_to_bf16(hosts.x);
    round_to_bf16(hosts.accumulator);
    round_to_bf16(hosts.weight);
    return hosts;
}

std::vector<std::uint16_t> to_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

int run_case(std::int32_t d, std::int64_t rows, float scale, std::uint32_t seed,
             bool expect_exact_match) {
    const std::string label =
        "residual_add_norm d=" + std::to_string(d) + " rows=" + std::to_string(rows);
    const std::size_t count = static_cast<std::size_t>(d) * static_cast<std::size_t>(rows);
    const Hosts hosts       = make_hosts(d, rows, seed);
    const std::size_t bytes = count * sizeof(std::uint16_t);
    const std::vector<std::uint16_t> x_bits           = to_bits(hosts.x);
    const std::vector<std::uint16_t> weight_bits      = to_bits(hosts.weight);
    const std::vector<std::uint16_t> accumulator_bits = to_bits(hosts.accumulator);

    GuardedDeviceBuffer x_device(bytes);
    GuardedDeviceBuffer weight_device(static_cast<std::size_t>(d) * sizeof(std::uint16_t));
    GuardedDeviceBuffer accumulator_device(bytes);
    x_device.copy_from_host(x_bits.data(), bytes);
    weight_device.copy_from_host(weight_bits.data(),
                                 static_cast<std::size_t>(d) * sizeof(std::uint16_t));
    accumulator_device.copy_from_host(accumulator_bits.data(), bytes);

    Tensor x_tensor(x_device.data(), DType::BF16, {d, static_cast<std::int32_t>(rows)});
    Tensor weight_tensor(weight_device.data(), DType::BF16, {d});
    Tensor accumulator_tensor(accumulator_device.data(), DType::BF16,
                              {d, static_cast<std::int32_t>(rows)});
    ops::residual_add_norm(x_tensor, weight_tensor, kEps, accumulator_tensor, scale, nullptr);
    cuda_check_last_launch("residual_add_norm");

    int failures = 0;
    failures += verify_pointwise(label, from_device_bf16(accumulator_device.data(), count),
                                 oracle(hosts.x, hosts.weight, hosts.accumulator, d, rows, scale),
                                 kCriterion);
    failures += accumulator_device.verify_guards(label);

    if (!expect_exact_match) { return failures; }

    // The unfused sequence on the same inputs, at the widths whose rmsnorm route shares this Op's
    // reduction order: the fused and unfused results must be the same BF16 words.
    GuardedDeviceBuffer normalized_device(bytes);
    GuardedDeviceBuffer unfused_device(bytes);
    normalized_device.fill(0);
    unfused_device.copy_from_host(accumulator_bits.data(), bytes);
    Tensor normalized_tensor(normalized_device.data(), DType::BF16,
                             {d, static_cast<std::int32_t>(rows)});
    Tensor unfused_tensor(unfused_device.data(), DType::BF16, {d, static_cast<std::int32_t>(rows)});
    ops::rmsnorm(x_tensor, weight_tensor, kEps, false, normalized_tensor, nullptr);
    ops::residual_add(normalized_tensor, unfused_tensor, nullptr);
    if (scale != 1.0F) { ops::mul_scalar(unfused_tensor, scale, nullptr); }
    cuda_synchronize();
    failures += unfused_device.verify_guards(label + " unfused");

    failures += verify_exact((label + " matches the unfused sequence").c_str(),
                             from_device<std::uint16_t>(accumulator_device.data(), count),
                             from_device<std::uint16_t>(unfused_device.data(), count));
    return failures;
}

int run_rejections() {
    int failures                      = 0;
    const std::vector<std::int32_t> d = {128, 5632};
    for (const std::int32_t width : d) {
        GuardedDeviceBuffer x_device(static_cast<std::size_t>(width) * sizeof(std::uint16_t));
        GuardedDeviceBuffer weight(static_cast<std::size_t>(width) * sizeof(std::uint16_t));
        GuardedDeviceBuffer accumulator(static_cast<std::size_t>(width) * sizeof(std::uint16_t));
        x_device.fill(0);
        weight.fill(0);
        accumulator.fill(0);
        Tensor x_tensor(x_device.data(), DType::BF16, {width, 1});
        Tensor weight_tensor(weight.data(), DType::BF16, {width});
        Tensor accumulator_tensor(accumulator.data(), DType::BF16, {width, 1});
        bool threw = false;
        try {
            ops::residual_add_norm(x_tensor, weight_tensor, kEps, accumulator_tensor, 1.0F,
                                   nullptr);
        } catch (const std::invalid_argument&) { threw = true; }
        if (!threw) {
            std::cerr << "residual_add_norm d=" << width << ": unadmitted width was accepted\n";
            ++failures;
        }
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no CUDA device\n";
        return 77;
    }
    int failures = 0;
    // The two widths the engine calls this Op with, where it must match the unfused sequence.
    failures += run_case(2816, 1, 0.75F, 11U, true);
    failures += run_case(2816, 3, 1.0F, 23U, true);
    failures += run_case(5376, 1, 0.5F, 37U, true);
    // A width whose unfused rmsnorm route differs, checked against the oracle only.
    failures += run_case(256, 2, 1.0F, 41U, false);
    failures += run_rejections();

    if (failures != 0) {
        std::cerr << failures << " residual_add_norm checks failed\n";
        return 1;
    }
    std::cout << "OK: residual_add_norm matches its oracle and the unfused sequence at the "
                 "Gemma 4 hidden widths\n";
    return 0;
}
