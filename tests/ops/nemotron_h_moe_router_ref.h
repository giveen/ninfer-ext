#pragma once

// FP64 oracle for ninfer::ops::nemotron_h_moe_router
// (include/ninfer/ops/nemotron_h_moe_router.h).
//
// It follows the contract rather than the implementation: the projection, the sigmoid, the
// correction bias, the group scores with their top-2 sum, the group mask, the K-pass selection with
// the lower-id tie-break, the gather of the *uncorrected* scores, the renormalization and then the
// scale.
//
// The reference's own arithmetic for this stage is FP32 sequential -- `F.linear` on an FP32 cast,
// then FP32 tensor ops -- so a sequential reference is exact enough by construction: the only
// nonlinearity is the sigmoid and every other step is a sum or a product. This one evaluates the
// same sequence in double from the represented BF16 operands, which bounds the reference instead of
// adding to it, so a disagreement larger than the FP32 roundings of those steps is a wrong rule
// rather than a rounding.
#include "ops/op_tester.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ninfer::test::nemotron_h_moe_router_ref {

struct Result {
    // ids and weights are [top_k, tokens] with the rank fastest, like the Op's outputs. logits is
    // [experts, tokens], like the Op's workspace -- the reference's own dump writes the transpose.
    std::vector<std::int32_t> ids;
    std::vector<double> weights;
    std::vector<double> logits;
};

inline Result run(const std::vector<std::uint16_t>& x_bits, std::int32_t hidden,
                  std::int32_t tokens, const std::vector<std::uint16_t>& router_bits,
                  std::int32_t experts, const std::vector<float>& correction_bias,
                  std::int32_t top_k, std::int32_t groups, std::int32_t groups_used, bool normalize,
                  double scaling_factor) {
    const std::int32_t width = groups > 0 ? experts / groups : 0;
    if (hidden <= 0 || tokens <= 0 || experts <= 0 || groups <= 0 || experts % groups != 0 ||
        groups_used <= 0 || groups_used > groups || top_k <= 0 || top_k > groups_used * width ||
        x_bits.size() != static_cast<std::size_t>(hidden) * tokens ||
        router_bits.size() != static_cast<std::size_t>(experts) * hidden ||
        correction_bias.size() != static_cast<std::size_t>(experts)) {
        throw std::invalid_argument("nemotron_h_moe_router_ref: shapes do not match the geometry");
    }

    Result result;
    result.ids.resize(static_cast<std::size_t>(top_k) * tokens);
    result.weights.resize(static_cast<std::size_t>(top_k) * tokens);
    result.logits.resize(static_cast<std::size_t>(experts) * tokens);

    std::vector<double> score(static_cast<std::size_t>(experts)); // corrected: the ranking key
    std::vector<double> uncorrected(
        static_cast<std::size_t>(experts)); // sigmoid: the emitted weight
    std::vector<double> group_score(static_cast<std::size_t>(groups));
    std::vector<char> selected_group(static_cast<std::size_t>(groups));
    std::vector<char> taken(static_cast<std::size_t>(experts));
    std::vector<double> emitted(static_cast<std::size_t>(top_k));
    std::vector<std::int32_t> chosen(static_cast<std::size_t>(top_k));

    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::uint16_t* row = x_bits.data() + static_cast<std::size_t>(token) * hidden;
        for (std::int32_t e = 0; e < experts; ++e) {
            const std::uint16_t* weights_row =
                router_bits.data() + static_cast<std::size_t>(e) * hidden;
            double logit = 0.0;
            for (std::int32_t h = 0; h < hidden; ++h) {
                logit += static_cast<double>(bf16_to_f32(weights_row[h])) *
                         static_cast<double>(bf16_to_f32(row[h]));
            }
            result.logits[static_cast<std::size_t>(e) * tokens + token] = logit;
            const double s                           = 1.0 / (1.0 + std::exp(-logit));
            uncorrected[static_cast<std::size_t>(e)] = s;
            score[static_cast<std::size_t>(e)] =
                s + static_cast<double>(correction_bias[static_cast<std::size_t>(e)]);
        }

        // The group scores: the two largest corrected scores of each group, summed. A group of one
        // expert has no second element and contributes its single score.
        for (std::int32_t g = 0; g < groups; ++g) {
            const std::int32_t begin = g * width;
            std::int32_t first       = begin;
            for (std::int32_t e = begin + 1; e < begin + width; ++e) {
                if (score[static_cast<std::size_t>(e)] > score[static_cast<std::size_t>(first)]) {
                    first = e;
                }
            }
            double sum = score[static_cast<std::size_t>(first)];
            if (width > 1) {
                std::int32_t second = -1;
                for (std::int32_t e = begin; e < begin + width; ++e) {
                    if (e == first) { continue; }
                    if (second < 0 || score[static_cast<std::size_t>(e)] >
                                          score[static_cast<std::size_t>(second)]) {
                        second = e;
                    }
                }
                sum += score[static_cast<std::size_t>(second)];
            }
            group_score[static_cast<std::size_t>(g)] = sum;
        }

        // The top `groups_used` groups; scanning in ascending group id with a strict comparison is
        // the lower-group tie-break.
        for (std::int32_t g = 0; g < groups; ++g) {
            selected_group[static_cast<std::size_t>(g)] = 0;
        }
        for (std::int32_t rank = 0; rank < groups_used; ++rank) {
            std::int32_t best = -1;
            for (std::int32_t g = 0; g < groups; ++g) {
                if (selected_group[static_cast<std::size_t>(g)] != 0) { continue; }
                if (best < 0 || group_score[static_cast<std::size_t>(g)] >
                                    group_score[static_cast<std::size_t>(best)]) {
                    best = g;
                }
            }
            if (best < 0) { throw std::runtime_error("nemotron_h_moe_router_ref: no group left"); }
            selected_group[static_cast<std::size_t>(best)] = 1;
        }

        // The K ranks among the selected groups' experts, likewise ascending with a strict
        // comparison, which is the lower-expert tie-break.
        for (std::int32_t e = 0; e < experts; ++e) { taken[static_cast<std::size_t>(e)] = 0; }
        for (std::int32_t rank = 0; rank < top_k; ++rank) {
            std::int32_t best = -1;
            for (std::int32_t e = 0; e < experts; ++e) {
                if (taken[static_cast<std::size_t>(e)] != 0 ||
                    selected_group[static_cast<std::size_t>(e / width)] == 0) {
                    continue;
                }
                if (best < 0 ||
                    score[static_cast<std::size_t>(e)] > score[static_cast<std::size_t>(best)]) {
                    best = e;
                }
            }
            if (best < 0) { throw std::runtime_error("nemotron_h_moe_router_ref: no expert left"); }
            taken[static_cast<std::size_t>(best)]  = 1;
            chosen[static_cast<std::size_t>(rank)] = best;
        }

        // The emitted weight is the uncorrected sigmoid at each chosen expert; the renormalization
        // divides by its sum plus 1e-20, and only then does the scale multiply.
        double sum = 0.0;
        for (std::int32_t rank = 0; rank < top_k; ++rank) {
            emitted[static_cast<std::size_t>(rank)] =
                uncorrected[static_cast<std::size_t>(chosen[static_cast<std::size_t>(rank)])];
            sum += emitted[static_cast<std::size_t>(rank)];
        }
        for (std::int32_t rank = 0; rank < top_k; ++rank) {
            const std::size_t at = static_cast<std::size_t>(rank) * tokens + token;
            double weight        = emitted[static_cast<std::size_t>(rank)];
            if (normalize) { weight /= sum + 1e-20; }
            result.ids[at]     = chosen[static_cast<std::size_t>(rank)];
            result.weights[at] = weight * scaling_factor;
        }
    }
    return result;
}

} // namespace ninfer::test::nemotron_h_moe_router_ref
