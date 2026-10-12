// ninfer::ops::linear over a q4_g32_fp16_rows weight against an FP64 oracle.
//
// The weights are read through the fixture's own decode (code * binary16 scale). Up to eight
// tokens the route accumulates the decoded values in FP32; above that the tensor-core tile decodes
// each weight to BF16 first, which the oracle models as a declared rounding point so the criterion
// qualifies the route's arithmetic. The shapes are the Gemma 4 26B-A4B's dense projections and its
// head's width class, at the token counts a decode round, an MTP verify and a prefill pass reach.
#include "ninfer/ops/linear.h"

#include "ops/op_tester.h"
#include "ops/q4_rows_weight.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr PointwiseCriterion kCriterion = {/*absolute*/ 1e-3, /*relative*/ 1e-2};

int run_case(std::int32_t n, std::int32_t k, std::int32_t tokens) {
    const std::string label =
        "q4rows N=" + std::to_string(n) + " K=" + std::to_string(k) + " T=" + std::to_string(tokens);
    const Q4RowsBank weight = make_q4_rows_bank(n, k, static_cast<std::uint32_t>(n + 3 * k));

    std::vector<float> x(static_cast<std::size_t>(k) * tokens);
    fill_uniform(x, 17, -1.0F, 1.0F);
    round_to_bf16(x);
    std::vector<std::uint16_t> x_bits(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) { x_bits[i] = f32_to_bf16(x[i]); }

    // A route above eight tokens decodes each weight to BF16 before the tensor cores.
    const bool weights_bf16 = tokens > 8;
    std::vector<double> reference(static_cast<std::size_t>(n) * tokens);
    std::vector<double> row(static_cast<std::size_t>(k));
    for (std::int32_t r = 0; r < n; ++r) {
        for (std::int32_t c = 0; c < k; ++c) {
            const double value = weight.value(r, c);
            row[static_cast<std::size_t>(c)] =
                weights_bf16 ? static_cast<double>(bf16_to_f32(f32_to_bf16(static_cast<float>(value))))
                             : value;
        }
        for (std::int32_t t = 0; t < tokens; ++t) {
            double sum = 0.0;
            for (std::int32_t c = 0; c < k; ++c) {
                sum += row[static_cast<std::size_t>(c)] *
                       static_cast<double>(x[static_cast<std::size_t>(t) * k + c]);
            }
            reference[static_cast<std::size_t>(t) * n + r] = sum;
        }
    }

    const auto payload = weight.payload();
    GuardedDeviceBuffer weight_device(payload.size());
    weight_device.copy_from_host(payload.data(), payload.size());
    GuardedDeviceBuffer x_device(x_bits.size() * sizeof(std::uint16_t));
    x_device.copy_from_host(x_bits.data(), x_device.bytes());
    GuardedDeviceBuffer out_device(static_cast<std::size_t>(n) * tokens * sizeof(std::uint16_t));
    std::vector<std::uint16_t> poison(static_cast<std::size_t>(n) * tokens, f32_to_bf16(-12345.0F));
    out_device.copy_from_host(poison.data(), out_device.bytes());

    Tensor x_tensor(x_device.data(), DType::BF16, {k, tokens});
    Tensor out_tensor(out_device.data(), DType::BF16, {n, tokens});
    ops::linear(x_tensor, weight.device_weight(weight_device.data(), payload.size()), out_tensor,
                nullptr);
    cuda_synchronize();
    const auto got =
        from_device_bf16(out_device.data(), static_cast<std::int64_t>(n) * tokens);
    int failures = verify_pointwise(label, std::vector<double>(got.begin(), got.end()), reference,
                                    kCriterion);
    failures += verify_exact((label + " weight preserved").c_str(),
                             from_device<std::uint8_t>(weight_device.data(),
                                                       static_cast<std::int64_t>(payload.size())),
                             payload);
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    try {
        // q, k/v (sliding), k (global), o, gate/up, down: one decode token, an MTP verify width,
        // the first tile width, a ragged tile and a prefill-sized pass.
        const std::int32_t shapes[][2] = {{4096, 2816}, {2048, 2816}, {1024, 2816},
                                          {2816, 4096}, {2112, 2816}, {2816, 2112}};
        for (const auto& shape : shapes) {
            for (const std::int32_t tokens : {1, 5, 8, 9, 100}) {
                failures += run_case(shape[0], shape[1], tokens);
            }
        }
        // The head's class: many rows over the hidden width.
        failures += run_case(8192, 2816, 1);
        failures += run_case(8192, 2816, 70);
    } catch (const std::exception& error) {
        std::cerr << "linear q4rows: " << error.what() << '\n';
        return 1;
    }
    if (failures == 0) {
        std::cout << "OK: linear over q4_g32_fp16_rows matches its oracle on both routes at the "
                     "26B's dense shapes\n";
    }
    return failures == 0 ? 0 : 1;
}
