#pragma once

// FP64 oracle for ninfer::ops::gemma_moe_router (include/ninfer/ops/gemma_moe_router.h).
//
// It follows the contract rather than the implementation: the unit-RMS row and its two scales, the
// projection, the softmax, the K-pass selection with the lower-id tie-break, the renormalization
// and then the per-expert scale. Everything is accumulated in double from the represented BF16
// values, so a disagreement with the Op is the Op's rounding, not the oracle's.
#include "ops/direct_bf16_weight.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::test::gemma_moe_router_ref {

struct Result {
    // Both are [top_k, tokens], rank fastest, like the Op's outputs.
    std::vector<std::int32_t> ids;
    std::vector<double> weights;
};

inline Result run(const std::vector<std::uint16_t>& x_bits, std::int32_t hidden,
                  std::int32_t tokens, const direct_bf16_weight::HostWeight& router,
                  const std::vector<std::uint16_t>& router_scale_bits,
                  const std::vector<std::uint16_t>& per_expert_scale_bits, double eps,
                  std::int32_t top_k) {
    const std::int32_t experts = router.n;
    if (router.k != hidden || static_cast<std::int32_t>(router_scale_bits.size()) != hidden ||
        static_cast<std::int32_t>(per_expert_scale_bits.size()) != experts ||
        static_cast<std::int32_t>(x_bits.size()) != hidden * tokens) {
        throw std::invalid_argument("gemma_moe_router_ref: shapes do not match the geometry");
    }

    Result result;
    result.ids.resize(static_cast<std::size_t>(top_k) * tokens);
    result.weights.resize(static_cast<std::size_t>(top_k) * tokens);

    std::vector<double> scaled(static_cast<std::size_t>(hidden));
    std::vector<double> probability(static_cast<std::size_t>(experts));
    std::vector<char> selected(static_cast<std::size_t>(experts));
    const double root = 1.0 / std::sqrt(static_cast<double>(hidden));

    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::uint16_t* row = x_bits.data() + static_cast<std::size_t>(token) * hidden;
        double squares           = 0.0;
        for (std::int32_t h = 0; h < hidden; ++h) {
            const double value = static_cast<double>(bf16_to_f32(row[h]));
            squares += value * value;
        }
        const double mean    = squares / static_cast<double>(hidden);
        const double inverse = 1.0 / std::sqrt(mean + eps);
        for (std::int32_t h = 0; h < hidden; ++h) {
            scaled[static_cast<std::size_t>(h)] =
                static_cast<double>(bf16_to_f32(row[h])) * inverse *
                static_cast<double>(bf16_to_f32(router_scale_bits[static_cast<std::size_t>(h)])) *
                root;
        }

        double maximum = -INFINITY;
        for (std::int32_t e = 0; e < experts; ++e) {
            const std::uint16_t* weights_row =
                router.bits.data() + static_cast<std::size_t>(e) * hidden;
            double logit = 0.0;
            for (std::int32_t h = 0; h < hidden; ++h) {
                logit += static_cast<double>(bf16_to_f32(weights_row[h])) *
                         scaled[static_cast<std::size_t>(h)];
            }
            probability[static_cast<std::size_t>(e)] = logit;
            maximum                                  = std::max(maximum, logit);
        }
        double total = 0.0;
        for (std::int32_t e = 0; e < experts; ++e) {
            const double value = std::exp(probability[static_cast<std::size_t>(e)] - maximum);
            probability[static_cast<std::size_t>(e)] = value;
            total += value;
        }
        for (std::int32_t e = 0; e < experts; ++e) {
            probability[static_cast<std::size_t>(e)] /= total;
            selected[static_cast<std::size_t>(e)] = 0;
        }

        // K passes, each taking the largest remaining probability; scanning in ascending expert id
        // with a strict comparison is the lower-id tie-break.
        std::vector<double> chosen(top_k);
        std::vector<std::int32_t> chosen_id(top_k);
        for (std::int32_t rank = 0; rank < top_k; ++rank) {
            std::int32_t best = -1;
            for (std::int32_t e = 0; e < experts; ++e) {
                if (selected[static_cast<std::size_t>(e)] != 0) { continue; }
                if (best < 0 || probability[static_cast<std::size_t>(e)] >
                                    probability[static_cast<std::size_t>(best)]) {
                    best = e;
                }
            }
            if (best < 0) { throw std::runtime_error("gemma_moe_router_ref: no expert left"); }
            selected[static_cast<std::size_t>(best)]  = 1;
            chosen[static_cast<std::size_t>(rank)]    = probability[static_cast<std::size_t>(best)];
            chosen_id[static_cast<std::size_t>(rank)] = best;
        }

        double sum = 0.0;
        for (std::int32_t rank = 0; rank < top_k; ++rank) {
            sum += chosen[static_cast<std::size_t>(rank)];
        }
        for (std::int32_t rank = 0; rank < top_k; ++rank) {
            const std::size_t at      = static_cast<std::size_t>(rank) * tokens + token;
            const std::int32_t expert = chosen_id[static_cast<std::size_t>(rank)];
            result.ids[at]            = expert;
            result.weights[at] = chosen[static_cast<std::size_t>(rank)] / sum *
                                 static_cast<double>(bf16_to_f32(
                                     per_expert_scale_bits[static_cast<std::size_t>(expert)]));
        }
    }
    return result;
}

} // namespace ninfer::test::gemma_moe_router_ref
