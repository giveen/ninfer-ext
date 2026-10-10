#include "ninfer/ops/mul_scalar.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// One storage rounding of the product.
constexpr PointwiseCriterion mul_scalar_criterion() {
    return {/*absolute*/ 0.0, /*relative*/ 3.95e-3};
}

std::vector<double> mul_scalar_oracle(const std::vector<float>& input, float factor) {
    std::vector<double> expected(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        expected[i] = static_cast<double>(input[i]) * static_cast<double>(factor);
    }
    return expected;
}

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

int run_case(const char* label, std::int64_t count, float factor, std::uint32_t seed,
             bool unaligned = false) {
    const std::size_t offset = unaligned ? 1 : 0;
    std::vector<float> input(count);
    fill_uniform(input, seed, -4.0f, 4.0f);
    round_to_bf16(input);

    const auto expected = mul_scalar_oracle(input, factor);
    const auto bits     = encode_bf16(input);

    GuardedDeviceBuffer device((offset + count) * sizeof(std::uint16_t));
    device.copy_from_host(bits.data(), count * sizeof(std::uint16_t), offset * sizeof(std::uint16_t));
    auto* data = static_cast<std::uint8_t*>(device.data()) + offset * sizeof(std::uint16_t);
    Tensor tensor(data, DType::BF16, {static_cast<std::int32_t>(count)});

    ops::mul_scalar(tensor, factor, nullptr);
    cuda_synchronize();

    int failures =
        verify_pointwise(label, from_device_bf16(data, count), expected, mul_scalar_criterion());
    failures += device.verify_guards("mul_scalar");
    return failures;
}

int verify_rejection(const char* label, float factor) {
    GuardedDeviceBuffer device(16 * sizeof(std::uint16_t));
    Tensor tensor(device.data(), DType::BF16, {16});
    try {
        ops::mul_scalar(tensor, factor, nullptr);
    } catch (const std::invalid_argument&) { return 0; }
    std::cerr << label << ": mul_scalar accepted an invalid factor\n";
    return 1;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;

    // Gemma's hidden width at decode and prefill sizes, with the factors a layer scalar may take:
    // exactly one for a sublayer that has none, and learned values that may be fractional, negative
    // or zero.
    for (std::int64_t count : {1, 5376, 4097, 537600}) {
        failures += run_case("mul_scalar hidden", count, -0.375F, 9101U);
    }
    failures += run_case("mul_scalar identity", 5376, 1.0F, 9102U);
    failures += run_case("mul_scalar half", 5376, 0.5F, 9103U);
    failures += run_case("mul_scalar zero", 5376, 0.0F, 9104U);
    failures += run_case("mul_scalar large", 4096, 64.0F, 9105U);
    failures += run_case("mul_scalar unaligned", 5376, 0.25F, 9106U, true);

    failures += verify_rejection("mul_scalar infinite", std::numeric_limits<float>::infinity());
    failures += verify_rejection("mul_scalar nan", std::numeric_limits<float>::quiet_NaN());

    std::cout << (failures ? "FAIL" : "OK") << " mul_scalar\n";
    return failures ? 1 : 0;
}
