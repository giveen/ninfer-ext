// The grouped tap application, against a naive FP64 oracle over the represented BF16 inputs.
//
// The cases cover the draft's own geometry (4096 channels in 256 groups of 16, a block of 8) and a
// small one that puts the block boundary in the middle of the window, plus the degenerate one-token
// window where only tap zero can ever be reached.

#include "ninfer/ops/grouped_tap_apply.h"

#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

DeviceExecutionView test_execution() {
    int device = 0, count = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device));
    return {nullptr, count};
}

// Four BF16 terms summed and stored once more: one storage rounding dominates the absolute side and
// four input roundings set the relative one.
constexpr PointwiseCriterion tap_criterion() { return {/*absolute*/ 2.0e-5, /*relative*/ 4.5e-3}; }

struct Case {
    const char* label;
    std::int32_t channels, group_width, window, batch, block_size;
    // 0 is the prepare half, 1 the finish half; the forward applies them to different inputs, so
    // each is checked on its own.
    std::int32_t side = 0;
};

// The tensors declare BF16 bits, so the uploads must be encoded: handing the kernel floats would
// give it 4-byte values where it reads 2-byte ones.
std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

std::size_t x_index(std::int32_t h, std::int32_t t, std::int32_t b, const Case& c) {
    return static_cast<std::size_t>(h) +
           static_cast<std::size_t>(c.channels) * static_cast<std::size_t>(t + c.window * b);
}

int run_case(const Case& c, std::uint32_t seed) {
    const std::int32_t taps   = 2;
    const std::int32_t groups = c.channels / c.group_width;
    const std::size_t x_count = static_cast<std::size_t>(c.channels) * c.window * c.batch;
    const std::size_t b_count = static_cast<std::size_t>(2) * taps * c.channels;
    const std::size_t d_count = static_cast<std::size_t>(2 * taps * groups) * c.window * c.batch;

    std::vector<float> x(x_count), base(b_count), delta(d_count);
    fill_uniform(x, seed, -0.35f, 0.35f);
    fill_uniform(base, seed + 1U, -0.5f, 0.5f);
    fill_uniform(delta, seed + 2U, -0.25f, 0.25f);
    round_to_bf16(x);
    round_to_bf16(base);
    round_to_bf16(delta);

    // The oracle: the same formula, naively, from the represented BF16 values.
    std::vector<double> expected(x_count, 0.0);
    for (std::int32_t h = 0; h < c.channels; ++h) {
        for (std::int32_t t = 0; t < c.window; ++t) {
            const std::int32_t group = h / c.group_width;
            const std::int32_t reach = std::min(taps - 1, t % c.block_size);
            for (std::int32_t b = 0; b < c.batch; ++b) {
                double sum = 0.0;
                for (std::int32_t k = 0; k <= reach; ++k) {
                    const std::size_t row =
                        static_cast<std::size_t>((c.side * taps + k) * groups + group);
                    const std::size_t slot =
                        static_cast<std::size_t>(c.side * taps * c.channels + k * c.channels + h);
                    const double base_value = base[slot];
                    const double delta_value =
                        delta[row + static_cast<std::size_t>(2 * taps * groups) *
                                        static_cast<std::size_t>(t + c.window * b)];
                    const double input = x[x_index(h, t - k, b, c)];
                    sum += (base_value + delta_value) * input;
                }
                expected[x_index(h, t, b, c)] = sum;
            }
        }
    }

    const auto x_bits     = encode_bf16(x);
    const auto base_bits  = encode_bf16(base);
    const auto delta_bits = encode_bf16(delta);
    GuardedDeviceBuffer device_x(std::max<std::size_t>(16, x_count * 2));
    GuardedDeviceBuffer device_base(std::max<std::size_t>(16, b_count * 2));
    GuardedDeviceBuffer device_delta(std::max<std::size_t>(16, d_count * 2));
    GuardedDeviceBuffer device_out(std::max<std::size_t>(16, x_count * 2));
    device_x.copy_from_host(x_bits.data(), x_bits.size() * 2);
    device_base.copy_from_host(base_bits.data(), base_bits.size() * 2);
    device_delta.copy_from_host(delta_bits.data(), delta_bits.size() * 2);
    device_out.fill(0x7d);

    Tensor input(device_x.data(), DType::BF16, {c.channels, c.window, c.batch});
    Tensor taps_tensor(device_base.data(), DType::BF16, {2, taps * c.channels});
    Tensor delta_tensor(device_delta.data(), DType::BF16, {2 * taps * groups, c.window, c.batch});
    Tensor out(device_out.data(), DType::BF16, {c.channels, c.window, c.batch});
    ops::grouped_tap_apply(input, taps_tensor, delta_tensor, out, c.side, c.group_width,
                           c.block_size, test_execution());
    cuda_synchronize();

    const auto got = from_device_bf16(device_out.data(), x_count);
    return verify_pointwise(c.label, got, expected, tap_criterion());
}

} // namespace

int main() {
    int failures = 0;
    // The draft's geometry: 4096 channels, 256 groups of 16, a block of 8 over an 8-row window.
    // Both sides are checked: they read different halves of `base` and `delta`, and in the forward
    // they are applied to different inputs, so a side that leaks into the other must fail here.
    failures += run_case({"draft block, prepare side", 4096, 16, 8, 1, 8, 0}, 8101U);
    failures += run_case({"draft block, finish side", 4096, 16, 8, 1, 8, 1}, 8105U);
    // Two blocks' worth: the window carries both the block-relative and the full-tap rows.
    failures += run_case({"two blocks", 64, 16, 16, 2, 8, 1}, 8102U);
    // One token: only tap zero is reachable, whatever the taps declare.
    failures += run_case({"single token", 32, 8, 1, 4, 8, 0}, 8103U);
    // A block of three over a window of seven, so the modulo and the wrap differ.
    failures += run_case({"short block", 48, 4, 7, 3, 3, 1}, 8104U);
    if (failures != 0) {
        std::cout << "FAIL grouped_tap_apply\n";
        return 1;
    }
    std::cout << "OK grouped_tap_apply\n";
    return 0;
}
