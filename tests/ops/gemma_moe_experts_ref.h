#pragma once

// FP64 oracle for ninfer::ops::gemma_moe_experts (include/ninfer/ops/gemma_moe_experts.h).
//
// The formula, evaluated naively: per token, per selected rank, the expert's gate and up rows are
// dotted with the activation row, the tanh GELU of the gate multiplies the up, the expert's down
// rows are dotted with that product, and the weighted results are summed. Everything is double, so
// a disagreement with the Op is the Op's rounding.
//
// A bank value comes from a reader: the direct BF16 helper for a BF16 bank, and the
// quantized-weight fixture's own FP64 decode for an NVFP4 bank -- which reads the stored codes, the
// 16-element block scales and the per-source divisor. Neither path shares a decoder with the Op.
#include "ops/direct_bf16_weight.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

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

// `gate_up_value(row, column)` and `down_value(row, column)` read one bank element in FP64.
template <class GateUpValue, class DownValue>
inline std::vector<double> run(const GateUpValue& gate_up_value, const DownValue& down_value,
                               const std::vector<std::uint16_t>& x_bits, std::int32_t hidden,
                               std::int32_t tokens, const std::vector<std::int32_t>& ids,
                               const std::vector<float>& routing, std::int32_t experts,
                               std::int32_t intermediate, std::int32_t top_k) {
    if (static_cast<std::int32_t>(x_bits.size()) != hidden * tokens ||
        static_cast<std::int32_t>(ids.size()) != top_k * tokens ||
        static_cast<std::int32_t>(routing.size()) != top_k * tokens) {
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
            if (expert < 0 || expert >= experts) {
                throw std::invalid_argument("gemma_moe_experts_ref: expert id out of range");
            }
            const double weight          = static_cast<double>(routing[at]);
            const std::int32_t gate_base = expert * 2 * intermediate;
            const std::int32_t up_base   = gate_base + intermediate;
            for (std::int32_t i = 0; i < intermediate; ++i) {
                double g = 0.0;
                double u = 0.0;
                for (std::int32_t h = 0; h < hidden; ++h) {
                    const double value = activation[static_cast<std::size_t>(h)];
                    g += gate_up_value(gate_base + i, h) * value;
                    u += gate_up_value(up_base + i, h) * value;
                }
                gate[static_cast<std::size_t>(i)] = g;
                up[static_cast<std::size_t>(i)]   = u;
            }
            for (std::int32_t i = 0; i < intermediate; ++i) {
                product[static_cast<std::size_t>(i)] =
                    gelu_tanh(gate[static_cast<std::size_t>(i)]) * up[static_cast<std::size_t>(i)];
            }
            const std::int32_t down_base = expert * hidden;
            for (std::int32_t h = 0; h < hidden; ++h) {
                double acc = 0.0;
                for (std::int32_t i = 0; i < intermediate; ++i) {
                    acc += down_value(down_base + h, i) * product[static_cast<std::size_t>(i)];
                }
                out[static_cast<std::size_t>(token) * hidden + h] += weight * acc;
            }
        }
    }
    return out;
}

// A BF16 bank, read directly.
inline std::vector<double> run(const std::vector<std::uint16_t>& x_bits, std::int32_t hidden,
                               std::int32_t tokens, const std::vector<std::int32_t>& ids,
                               const std::vector<float>& routing,
                               const direct_bf16_weight::HostWeight& gate_up,
                               const direct_bf16_weight::HostWeight& down,
                               std::int32_t intermediate, std::int32_t top_k) {
    const std::int32_t experts = gate_up.n / (2 * intermediate);
    if (gate_up.k != hidden || down.k != intermediate || down.n != experts * hidden) {
        throw std::invalid_argument("gemma_moe_experts_ref: bank shapes do not match the geometry");
    }
    const auto value = [](const direct_bf16_weight::HostWeight& bank, int row, int column) {
        return static_cast<double>(
            bf16_to_f32(bank.bits[static_cast<std::size_t>(row) * bank.k + column]));
    };
    return run([&](int row, int column) { return value(gate_up, row, column); },
               [&](int row, int column) { return value(down, row, column); }, x_bits, hidden,
               tokens, ids, routing, experts, intermediate, top_k);
}

// An NVFP4 bank, through the fixture's decode of the stored codes, scales and per-source divisor.
inline std::vector<double> run(const std::vector<std::uint16_t>& x_bits, std::int32_t hidden,
                               std::int32_t tokens, const std::vector<std::int32_t>& ids,
                               const std::vector<float>& routing,
                               const quantized_weight::PackedWeight& gate_up,
                               const quantized_weight::PackedWeight& down,
                               std::int32_t intermediate, std::int32_t top_k) {
    const std::int32_t experts =
        static_cast<std::int32_t>(gate_up.weight.shape[0]) / (2 * intermediate);
    if (static_cast<std::int32_t>(down.weight.shape[1]) != intermediate ||
        static_cast<std::int32_t>(down.weight.shape[0]) != experts * hidden) {
        throw std::invalid_argument("gemma_moe_experts_ref: bank shapes do not match the geometry");
    }
    return run(
        [&](int row, int column) {
            return quantized_weight::logical_weight_fp64(gate_up, row, column);
        },
        [&](int row, int column) {
            return quantized_weight::logical_weight_fp64(down, row, column);
        },
        x_bits, hidden, tokens, ids, routing, experts, intermediate, top_k);
}

} // namespace ninfer::test::gemma_moe_experts_ref
