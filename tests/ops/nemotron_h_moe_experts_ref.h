#pragma once

// FP64 oracle for ninfer::ops::nemotron_h_moe_experts
// (include/ninfer/ops/nemotron_h_moe_experts.h).
//
// The formula, evaluated naively: per token, per selected rank, the expert's up rows are dotted
// with the activation row, the squared ReLU of that result is taken, the expert's down rows are
// dotted with it, and the weighted results are summed in ascending rank order. Everything is
// double, so a disagreement with the Op is the Op's rounding rather than the oracle's.
//
// The generic entry reads its three operands through value functions, which is what lets a case
// evaluate the same formula over the *represented* BF16 operands (what the Op reads) and over the
// exact FP32 ones (what a reference dump holds) with one implementation. The BF16-bank overloads
// below are the everyday form.
//
// `Boundaries` names a rounding a *different* implementation would introduce. The Op's contract has
// one rounding, at the output, and this suite's criterion is written for that; the boundary exists
// so a case can measure how far a plausible but wrong implementation lands instead of asserting
// that it would fail.
#include "ops/direct_bf16_weight.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ninfer::test::nemotron_h_moe_experts_ref {

// The activation, exactly as the Op defines it.
inline double relu2(double value) {
    const double rectified = value > 0.0 ? value : 0.0;
    return rectified * rectified;
}

struct Boundaries {
    // The checkpoint's own reference computes an expert in the activation dtype when the checkpoint
    // is BF16, so its per-token accumulator sees one rounded contribution per rank. The Op
    // contracts FP32 throughout with a single output rounding, which is what an empty Boundaries
    // models.
    bool per_rank_bf16 = false;
};

inline double round_bf16(double value) {
    return static_cast<double>(bf16_to_f32(f32_to_bf16(static_cast<float>(value))));
}

// `x_value(token, h)` is the activation element, `up_value(row, column)` and `down_value(row,
// column)` are one bank element each, all in FP64. `ids` and `routing` are the router's [K,T]
// arrays with the rank axis the fastest, and the result is [H,T] in the same order.
template <class XValue, class UpValue, class DownValue>
inline std::vector<double>
run(const XValue& x_value, const UpValue& up_value, const DownValue& down_value,
    std::int32_t latent, std::int32_t tokens, const std::vector<std::int32_t>& ids,
    const std::vector<float>& routing, std::int32_t experts, std::int32_t intermediate,
    std::int32_t top_k, Boundaries boundaries = {}, std::vector<double>* rank_magnitude = nullptr) {
    if (static_cast<std::int32_t>(ids.size()) != top_k * tokens ||
        static_cast<std::int32_t>(routing.size()) != top_k * tokens) {
        throw std::invalid_argument("nemotron_h_moe_experts_ref: shapes do not match the geometry");
    }

    std::vector<double> out(static_cast<std::size_t>(latent) * tokens, 0.0);
    // Sum over ranks of |weight * expert output|: what one ulp at the expert-output boundary is
    // relative to, where the ranks cancel.
    if (rank_magnitude != nullptr) { rank_magnitude->assign(out.size(), 0.0); }
    std::vector<double> activation(static_cast<std::size_t>(latent));
    std::vector<double> product(static_cast<std::size_t>(intermediate));

    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t h = 0; h < latent; ++h) {
            activation[static_cast<std::size_t>(h)] = x_value(token, h);
        }
        for (std::int32_t rank = 0; rank < top_k; ++rank) {
            const std::size_t at      = static_cast<std::size_t>(rank) * tokens + token;
            const std::int32_t expert = ids[at];
            if (expert < 0 || expert >= experts) {
                throw std::invalid_argument("nemotron_h_moe_experts_ref: expert id out of range");
            }
            const double weight = static_cast<double>(routing[at]);

            // Expert e's up rows are [e*I, (e+1)*I).
            const std::int32_t up_base = expert * intermediate;
            for (std::int32_t i = 0; i < intermediate; ++i) {
                double acc = 0.0;
                for (std::int32_t h = 0; h < latent; ++h) {
                    acc += up_value(up_base + i, h) * activation[static_cast<std::size_t>(h)];
                }
                product[static_cast<std::size_t>(i)] = relu2(acc);
            }

            // Expert e's down rows are [e*H, (e+1)*H).
            const std::int32_t down_base = expert * latent;
            for (std::int32_t h = 0; h < latent; ++h) {
                double acc = 0.0;
                for (std::int32_t i = 0; i < intermediate; ++i) {
                    acc += down_value(down_base + h, i) * product[static_cast<std::size_t>(i)];
                }
                double contribution = weight * acc;
                if (boundaries.per_rank_bf16) { contribution = round_bf16(contribution); }
                out[static_cast<std::size_t>(token) * latent + h] += contribution;
                if (rank_magnitude != nullptr) {
                    (*rank_magnitude)[static_cast<std::size_t>(token) * latent + h] +=
                        std::abs(contribution);
                }
            }
        }
    }
    return out;
}

// One bank element, read from a direct BF16 bank at its expert-major layout.
inline double bank_element(const direct_bf16_weight::HostWeight& bank, int row, int column) {
    return static_cast<double>(
        bf16_to_f32(bank.bits[static_cast<std::size_t>(row) * bank.k + column]));
}

// A BF16 activation and two BF16 banks: what the Op reads.
inline std::vector<double>
run(const std::vector<std::uint16_t>& x_bits, std::int32_t latent, std::int32_t tokens,
    const std::vector<std::int32_t>& ids, const std::vector<float>& routing,
    const direct_bf16_weight::HostWeight& up, const direct_bf16_weight::HostWeight& down,
    std::int32_t intermediate, std::int32_t top_k, std::int32_t experts, Boundaries boundaries = {},
    std::vector<double>* rank_magnitude = nullptr) {
    if (static_cast<std::int32_t>(x_bits.size()) != latent * tokens) {
        throw std::invalid_argument("nemotron_h_moe_experts_ref: activation shape differs");
    }
    if (up.k != latent || up.n != experts * intermediate || down.k != intermediate ||
        down.n != experts * latent) {
        throw std::invalid_argument(
            "nemotron_h_moe_experts_ref: bank shapes do not match the geometry");
    }
    const auto value = [](const direct_bf16_weight::HostWeight& bank, int row, int column) {
        return bank_element(bank, row, column);
    };
    return run(
        [&](int token, int h) {
            return static_cast<double>(
                bf16_to_f32(x_bits[static_cast<std::size_t>(token) * latent + h]));
        },
        [&](int row, int column) { return value(up, row, column); },
        [&](int row, int column) { return value(down, row, column); }, latent, tokens, ids, routing,
        experts, intermediate, top_k, boundaries, rank_magnitude);
}

// The same formula over an activation and two banks given as FP32 values, which is the form a
// reference dump holds: the operands the reference itself computed with, before any BF16 rounding
// this Op imposes.
template <class UpValue, class DownValue>
inline std::vector<double>
run_fp32(const std::vector<float>& x, std::int32_t latent, std::int32_t tokens,
         const std::vector<std::int32_t>& ids, const std::vector<float>& routing,
         const UpValue& up_value, const DownValue& down_value, std::int32_t experts,
         std::int32_t intermediate, std::int32_t top_k, Boundaries boundaries = {},
         std::vector<double>* rank_magnitude = nullptr) {
    if (static_cast<std::int32_t>(x.size()) != latent * tokens) {
        throw std::invalid_argument("nemotron_h_moe_experts_ref: activation shape differs");
    }
    return run(
        [&](int token, int h) {
            return static_cast<double>(x[static_cast<std::size_t>(token) * latent + h]);
        },
        up_value, down_value, latent, tokens, ids, routing, experts, intermediate, top_k,
        boundaries, rank_magnitude);
}

} // namespace ninfer::test::nemotron_h_moe_experts_ref
