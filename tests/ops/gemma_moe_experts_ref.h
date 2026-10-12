#pragma once

// FP64 oracle for ninfer::ops::gemma_moe_experts (include/ninfer/ops/gemma_moe_experts.h).
//
// The formula, evaluated naively: per token, per selected rank, the expert's gate and up rows are
// dotted with the activation row, the tanh GELU of the gate multiplies the up, the expert's down
// rows are dotted with that product, and the weighted results are summed. Everything is double, so
// a disagreement with the Op is the Op's rounding.
#include "ops/direct_bf16_weight.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ninfer::test::gemma_moe_experts_ref {

// The checkpoint's activation: the tanh approximation of the GELU.
inline double gelu_tanh(double value) {
    constexpr double kSqrt2Pi = 0.79788456080286535588;
    return 0.5 * value * (1.0 + std::tanh(kSqrt2Pi * (value + 0.044715 * value * value * value)));
}

// gate_up is [experts * 2 * intermediate, hidden] with each expert's gate rows then its up rows;
// down is [experts * hidden, intermediate]. `ids` and `routing` are [top_k, tokens], rank fastest.
inline std::vector<double> run(const std::vector<std::uint16_t>& x_bits, std::int32_t hidden,
                               std::int32_t tokens, const std::vector<std::int32_t>& ids,
                               const std::vector<float>& routing,
                               const direct_bf16_weight::HostWeight& gate_up,
                               const direct_bf16_weight::HostWeight& down,
                               std::int32_t intermediate, std::int32_t top_k) {
    if (static_cast<std::int32_t>(x_bits.size()) != hidden * tokens ||
        static_cast<std::int32_t>(ids.size()) != top_k * tokens ||
        static_cast<std::int32_t>(routing.size()) != top_k * tokens || gate_up.k != hidden ||
        gate_up.n % (2 * intermediate) != 0 || down.n != gate_up.n / (2 * intermediate) * hidden ||
        down.k != intermediate) {
        throw std::invalid_argument("gemma_moe_experts_ref: shapes do not match the geometry");
    }

    std::vector<double> out(static_cast<std::size_t>(hidden) * tokens, 0.0);
    std::vector<double> activation(static_cast<std::size_t>(hidden));
    std::vector<double> gate(static_cast<std::size_t>(intermediate));
    std::vector<double> up(static_cast<std::size_t>(intermediate));
    std::vector<double> product(static_cast<std::size_t>(intermediate));

    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::uint16_t* row = x_bits.data() + static_cast<std::size_t>(token) * hidden;
        for (std::int32_t h = 0; h < hidden; ++h) {
            activation[static_cast<std::size_t>(h)] = static_cast<double>(bf16_to_f32(row[h]));
        }
        for (std::int32_t rank = 0; rank < top_k; ++rank) {
            const std::size_t at      = static_cast<std::size_t>(rank) * tokens + token;
            const std::int32_t expert = ids[at];
            const double weight       = static_cast<double>(routing[at]);
            const std::size_t gate_base =
                static_cast<std::size_t>(expert) * 2 * intermediate * hidden;
            const std::size_t up_base = gate_base + static_cast<std::size_t>(intermediate) * hidden;
            for (std::int32_t i = 0; i < intermediate; ++i) {
                const std::uint16_t* gate_row =
                    gate_up.bits.data() + gate_base + static_cast<std::size_t>(i) * hidden;
                const std::uint16_t* up_row =
                    gate_up.bits.data() + up_base + static_cast<std::size_t>(i) * hidden;
                double g = 0.0;
                double u = 0.0;
                for (std::int32_t h = 0; h < hidden; ++h) {
                    const double value = activation[static_cast<std::size_t>(h)];
                    g += static_cast<double>(bf16_to_f32(gate_row[h])) * value;
                    u += static_cast<double>(bf16_to_f32(up_row[h])) * value;
                }
                gate[static_cast<std::size_t>(i)] = g;
                up[static_cast<std::size_t>(i)]   = u;
            }
            for (std::int32_t i = 0; i < intermediate; ++i) {
                product[static_cast<std::size_t>(i)] =
                    gelu_tanh(gate[static_cast<std::size_t>(i)]) * up[static_cast<std::size_t>(i)];
            }
            const std::size_t down_base = static_cast<std::size_t>(expert) * hidden * intermediate;
            for (std::int32_t h = 0; h < hidden; ++h) {
                const std::uint16_t* down_row =
                    down.bits.data() + down_base + static_cast<std::size_t>(h) * intermediate;
                double acc = 0.0;
                for (std::int32_t i = 0; i < intermediate; ++i) {
                    acc += static_cast<double>(bf16_to_f32(down_row[i])) *
                           product[static_cast<std::size_t>(i)];
                }
                out[static_cast<std::size_t>(token) * hidden + h] += weight * acc;
            }
        }
    }
    return out;
}

} // namespace ninfer::test::gemma_moe_experts_ref
