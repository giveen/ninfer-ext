#include "ninfer/ops/gelu_mul.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// The same criterion shape `silu_mul` uses for the same computation: one storage rounding of the
// product plus what tanhf carries, and a nonzero absolute floor, without which an element whose
// ideal is a denormal-sized product would be judged by a relative rule that cannot mean anything
// there.
constexpr PointwiseCriterion gelu_mul_bf16_criterion() {
    return {/*absolute*/ 2.0e-5, /*relative*/ 4.05e-3};
}

// This is the sole oracle. It evaluates PyTorch's gelu_pytorch_tanh on the gate and multiplies by
// the represented up value, naively in FP64.
std::vector<double> gelu_mul_oracle(const std::vector<float>& gate, const std::vector<float>& up) {
    constexpr double kSqrt2Pi = 0.7978845608028654;
    std::vector<double> expected(gate.size());
    for (std::size_t i = 0; i < gate.size(); ++i) {
        const double z = gate[i];
        const double gelu =
            0.5 * z * (1.0 + std::tanh(kSqrt2Pi * (z + 0.044715 * z * z * z)));
        expected[i] = gelu * static_cast<double>(up[i]);
    }
    return expected;
}

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

// Contiguous operands: the fast paired route.
int run_case(const char* label, std::int64_t count, std::uint32_t seed, float scale) {
    std::vector<float> gate(count), up(count);
    fill_uniform(gate, seed, -scale, scale);
    fill_uniform(up, seed + 1U, -scale, scale);
    round_to_bf16(gate);
    round_to_bf16(up);

    const auto expected = gelu_mul_oracle(gate, up);
    const auto g_bits   = encode_bf16(gate);
    const auto u_bits   = encode_bf16(up);

    GuardedDeviceBuffer device_gate(count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_up(count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_out(count * sizeof(std::uint16_t));
    device_gate.copy_from_host(g_bits.data(), device_gate.bytes());
    device_up.copy_from_host(u_bits.data(), device_up.bytes());
    device_out.fill(0x7d);

    Tensor gate_tensor(device_gate.data(), DType::BF16, {static_cast<std::int32_t>(count)});
    Tensor up_tensor(device_up.data(), DType::BF16, {static_cast<std::int32_t>(count)});
    Tensor out_tensor(device_out.data(), DType::BF16, {static_cast<std::int32_t>(count)});
    ops::gelu_mul(gate_tensor, up_tensor, out_tensor, nullptr);
    cuda_synchronize();

    int failures = verify_pointwise(label, from_device_bf16(device_out.data(), count), expected,
                                    gelu_mul_bf16_criterion());
    failures += device_out.verify_guards("gelu_mul");
    return failures;
}

// The model's layout: the two halves of one fused gate/up projection, so gate and up are strided
// views of one buffer and only out is a fresh allocation.
int run_halved_case(std::int32_t rows, std::int32_t tokens, std::uint32_t seed) {
    const std::size_t half  = static_cast<std::size_t>(rows) * tokens;
    const std::size_t total = 2 * half;
    std::vector<float> fused(total);
    fill_uniform(fused, seed, -4.0f, 4.0f);
    round_to_bf16(fused);
    const auto fused_bits = encode_bf16(fused);

    std::vector<float> gate(half), up(half);
    for (std::size_t i = 0; i < half; ++i) {
        gate[i] = fused[i];
        up[i]   = fused[half + i];
    }
    const auto expected = gelu_mul_oracle(gate, up);

    GuardedDeviceBuffer device_fused(total * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_out(half * sizeof(std::uint16_t));
    device_fused.copy_from_host(fused_bits.data(), device_fused.bytes());
    device_out.fill(0x7d);

    // gate = fused[0:rows], up = fused[rows:2*rows]: stride tokens * rows between columns.
    auto* base         = static_cast<std::uint8_t*>(device_fused.data());
    Tensor gate_tensor(base, DType::BF16, {rows, tokens});
    Tensor up_tensor(base + half * sizeof(std::uint16_t), DType::BF16, {rows, tokens});
    Tensor out_tensor(device_out.data(), DType::BF16, {rows, tokens});
    ops::gelu_mul(gate_tensor, up_tensor, out_tensor, nullptr);
    cuda_synchronize();

    int failures = verify_pointwise("gelu_mul halved projection",
                                    from_device_bf16(device_out.data(), half), expected,
                                    gelu_mul_bf16_criterion());
    failures += verify_exact("gelu_mul fused input unchanged",
                             from_device<std::uint16_t>(device_fused.data(), total), fused_bits);
    failures += device_fused.verify_guards("gelu_mul fused");
    failures += device_out.verify_guards("gelu_mul halved out");
    return failures;
}

int verify_rejection(const char* label, bool bad_shape) {
    GuardedDeviceBuffer a(8 * sizeof(std::uint16_t));
    GuardedDeviceBuffer b(8 * sizeof(std::uint16_t));
    GuardedDeviceBuffer out(8 * sizeof(std::uint16_t));
    Tensor gate_tensor(a.data(), DType::BF16, {4});
    Tensor up_tensor(b.data(), DType::BF16, {bad_shape ? 5 : 4});
    Tensor out_tensor(out.data(), DType::BF16, {4});
    try {
        ops::gelu_mul(gate_tensor, up_tensor, out_tensor, nullptr);
    } catch (const std::invalid_argument&) { return 0; }
    std::cerr << label << ": gelu_mul accepted an invalid request\n";
    return 1;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;

    // Gemma 4's MLP intermediate width at decode and short prefill, and a much smaller tensor for
    // the boundary behaviors.
    failures += run_case("gelu_mul intermediate width", 21504, 6101U, 6.0F);
    failures += run_case("gelu_mul odd count", 21505, 6102U, 6.0F);
    failures += run_case("gelu_mul one element", 1, 6103U, 6.0F);
    failures += run_case("gelu_mul large magnitude", 4096, 6104U, 64.0F);
    failures += run_case("gelu_mul tiny magnitude", 4096, 6105U, 1.0e-3F);

    // The fused projection layout the model actually produces.
    failures += run_halved_case(21504, 1, 6201U);
    failures += run_halved_case(21504, 7, 6202U);
    failures += run_halved_case(512, 33, 6203U);

    failures += verify_rejection("gelu_mul shape mismatch", true);

    std::cout << (failures ? "FAIL" : "OK") << " gelu_mul\n";
    return failures ? 1 : 0;
}
