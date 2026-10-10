#include "ninfer/ops/soft_cap.h"
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

// The Op rounds the transformed value once to BF16, and tanhf carries a couple of ulp of its own, so
// the bound holds a single storage rounding plus that headroom.
constexpr PointwiseCriterion soft_cap_bf16_criterion() {
    return {/*absolute*/ 0.0, /*relative*/ 4.5e-3};
}

// This is the sole oracle: the exact FP64 soft-cap of the represented BF16 logits.
std::vector<double> soft_cap_oracle(const std::vector<float>& input, float cap) {
    std::vector<double> expected(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        expected[i] = static_cast<double>(cap) *
                      std::tanh(static_cast<double>(input[i]) / static_cast<double>(cap));
    }
    return expected;
}

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

// A logits-like distribution: values are drawn well outside the cap as well as inside it, so the
// saturating part of the transform is exercised and not only its linear neighbourhood.
constexpr float kGptSoftCap = 30.0F;

int run_case(const char* label, std::int64_t count, float cap, std::uint32_t seed, float scale,
             bool unaligned = false) {
    const std::size_t offset = unaligned ? 1 : 0;
    std::vector<float> input(count);
    fill_uniform(input, seed, -scale, scale);
    round_to_bf16(input);

    const auto expected = soft_cap_oracle(input, cap);
    const auto bits     = encode_bf16(input);

    GuardedDeviceBuffer device((offset + count) * sizeof(std::uint16_t));
    device.copy_from_host(bits.data(), count * sizeof(std::uint16_t), offset * sizeof(std::uint16_t));
    auto* data = static_cast<std::uint8_t*>(device.data()) + offset * sizeof(std::uint16_t);
    Tensor tensor(data, DType::BF16, {static_cast<std::int32_t>(count)});

    ops::soft_cap(tensor, cap, nullptr);
    cuda_synchronize();

    int failures =
        verify_pointwise(label, from_device_bf16(data, count), expected, soft_cap_bf16_criterion());
    failures += device.verify_guards("soft_cap");
    return failures;
}

// A cap that breaks the transform's monotonicity or its finiteness is refused.
int verify_rejection(const char* label, float cap) {
    GuardedDeviceBuffer device(16 * sizeof(std::uint16_t));
    Tensor tensor(device.data(), DType::BF16, {16});
    try {
        ops::soft_cap(tensor, cap, nullptr);
    } catch (const std::invalid_argument&) { return 0; }
    std::cerr << label << ": soft_cap accepted an invalid cap\n";
    return 1;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;

    // Gemma's cap over decode-sized and prefill-sized logit vectors, including the saturating range.
    for (std::int64_t count : {1, 7, 262144, 1048573}) {
        failures += run_case("soft_cap gemma logits", count, kGptSoftCap, 5101U, 64.0F);
    }
    // A count that is neither eight- nor two-aligned exercises the odd tail.
    failures += run_case("soft_cap odd tail", 4097, kGptSoftCap, 5102U, 64.0F);
    failures += run_case("soft_cap unaligned", 1024, kGptSoftCap, 5103U, 64.0F, true);
    // A different cap, so the transform is not tied to the Gemma constant.
    failures += run_case("soft_cap small cap", 4096, 1.5F, 5104U, 8.0F);
    // Very small inputs, where the transform is close to the identity.
    failures += run_case("soft_cap small inputs", 4096, kGptSoftCap, 5105U, 1.0e-4F);

    failures += verify_rejection("soft_cap zero cap", 0.0F);
    failures += verify_rejection("soft_cap negative cap", -30.0F);
    failures += verify_rejection("soft_cap infinite cap", std::numeric_limits<float>::infinity());

    std::cout << (failures ? "FAIL" : "OK") << " soft_cap\n";
    return failures ? 1 : 0;
}
