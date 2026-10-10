#include "ninfer/ops/vision_pool.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// One BF16 storage rounding of an FP32 evaluation, with an absolute floor for outputs the
// standardization cancels to nearly nothing.
constexpr PointwiseCriterion kVisionPoolCriterion{/*absolute=*/2.0e-4, /*relative=*/4.0e-3};

// Gemma 4's tower: D=1152, k=3, multiplier sqrt(1152). Grids are non-square so a transposed cell
// order or a swapped axis changes the answer.
int run_case(std::int32_t width, std::int32_t height, std::uint32_t seed) {
    constexpr std::int32_t kFeatures = 1152;
    constexpr std::int32_t kKernel   = 3;
    const float multiplier           = std::sqrt(static_cast<float>(kFeatures));
    const std::int32_t patches       = width * height;
    const std::int32_t cells         = (width / kKernel) * (height / kKernel);

    std::vector<float> x(static_cast<std::size_t>(kFeatures) * patches);
    std::vector<float> bias(kFeatures), scale(kFeatures);
    fill_uniform(x, seed, -3.0f, 3.0f);
    fill_uniform(bias, seed + 1u, -20.0f, 20.0f);
    fill_uniform(scale, seed + 2u, 0.01f, 0.5f);
    round_to_bf16(x);
    // A ramp along each axis, so cells differ in a way a wrong cell order cannot reproduce.
    for (std::int32_t p = 0; p < patches; ++p) {
        const float ramp = 0.25f * static_cast<float>(p % width) - 0.5f * static_cast<float>(p / width);
        for (std::int32_t d = 0; d < 8; ++d) {
            x[static_cast<std::size_t>(p) * kFeatures + d] = bf16_to_f32(f32_to_bf16(ramp));
        }
    }

    std::vector<double> expected(static_cast<std::size_t>(kFeatures) * cells);
    for (std::int32_t cell = 0; cell < cells; ++cell) {
        const std::int32_t column0 = (cell % (width / kKernel)) * kKernel;
        const std::int32_t row0    = (cell / (width / kKernel)) * kKernel;
        for (std::int32_t d = 0; d < kFeatures; ++d) {
            double sum = 0.0;
            for (std::int32_t r = row0; r < row0 + kKernel; ++r) {
                for (std::int32_t c = column0; c < column0 + kKernel; ++c) {
                    sum += x[(static_cast<std::size_t>(r) * width + c) * kFeatures + d];
                }
            }
            const double mean = sum / (kKernel * kKernel);
            expected[static_cast<std::size_t>(cell) * kFeatures + d] =
                (static_cast<double>(multiplier) * mean - bias[d]) * scale[d];
        }
    }

    std::vector<std::uint16_t> x_bits(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) x_bits[i] = f32_to_bf16(x[i]);
    DeviceBuffer device_x    = to_device(x_bits);
    DeviceBuffer device_bias = to_device(bias);
    DeviceBuffer device_scale = to_device(scale);
    GuardedDeviceBuffer device_out(static_cast<std::size_t>(kFeatures) * cells * 2);
    device_out.fill(0x7f);
    Tensor x_tensor(device_x.p, DType::BF16, {kFeatures, patches});
    Tensor bias_tensor(device_bias.p, DType::FP32, {kFeatures});
    Tensor scale_tensor(device_scale.p, DType::FP32, {kFeatures});
    Tensor out_tensor(device_out.data(), DType::BF16, {kFeatures, cells});
    ops::vision_pool_standardize(x_tensor, width, height, kKernel, multiplier, bias_tensor,
                                 scale_tensor, out_tensor, nullptr);
    cuda_synchronize();

    const std::string label =
        "vision_pool_standardize " + std::to_string(width) + "x" + std::to_string(height);
    int failures = verify_pointwise(label.c_str(),
                                    from_device_bf16(device_out.data(), expected.size()), expected,
                                    kVisionPoolCriterion);
    failures += device_out.verify_guards(label.c_str());
    failures += verify_exact((label + " x unchanged").c_str(),
                             from_device<std::uint16_t>(device_x, x_bits.size()), x_bits);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    failures += run_case(3, 3, 11u);   // one soft token, the smallest image
    failures += run_case(48, 6, 13u);  // a wide strip
    failures += run_case(42, 60, 17u); // 280 soft tokens, portrait
    failures += run_case(60, 42, 19u); // 280 soft tokens, landscape
    try {
        Tensor none;
        ops::vision_pool_standardize(none, 4, 3, 3, 1.0f, none, none, none, nullptr);
        std::cerr << "vision_pool_standardize accepted a grid that is not a multiple of k\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    std::cout << (failures ? "FAIL" : "OK") << " vision_pool_standardize\n";
    return failures ? 1 : 0;
}
