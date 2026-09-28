#include "evaluation.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace ninfer::perplexity {

std::vector<WindowPlan> plan_windows(std::size_t tokens, std::uint32_t context,
                                     std::uint32_t stride) {
    if (tokens < 2) { throw std::invalid_argument("perplexity stream must contain two tokens"); }
    if (context < 2 || stride == 0 || stride >= context) {
        throw std::invalid_argument("perplexity requires context>=2 and 1<=stride<context");
    }

    std::vector<WindowPlan> windows;
    std::size_t previous_end = std::min<std::size_t>(tokens, context);
    windows.push_back(WindowPlan{.input_begin  = 0,
                                 .input_end    = previous_end,
                                 .target_begin = 1,
                                 .target_end   = previous_end,
                                 .first_target = 1});
    while (previous_end < tokens) {
        const std::size_t end          = std::min(tokens, previous_end + stride);
        const std::size_t begin        = end > context ? end - context : 0;
        const std::size_t local_target = previous_end - begin;
        if (local_target == 0 || local_target >= end - begin ||
            local_target > std::numeric_limits<std::uint32_t>::max()) {
            throw std::logic_error("perplexity window has an invalid target suffix");
        }
        windows.push_back(WindowPlan{
            .input_begin  = begin,
            .input_end    = end,
            .target_begin = previous_end,
            .target_end   = end,
            .first_target = static_cast<std::uint32_t>(local_target),
        });
        previous_end = end;
    }
    return windows;
}

void ScoreAggregate::add(std::span<const float> logprobs) {
    for (const float logprob : logprobs) {
        if (!std::isfinite(logprob)) {
            throw std::runtime_error("causal scoring returned a non-finite logprob");
        }
        total_nll -= static_cast<double>(logprob);
    }
    if (logprobs.size() > std::numeric_limits<std::uint64_t>::max() - scored_tokens) {
        throw std::overflow_error("perplexity scored-token count overflowed");
    }
    scored_tokens += static_cast<std::uint64_t>(logprobs.size());
}

void ScoreAggregate::add(const ScoreAggregate& other) noexcept {
    scored_tokens += other.scored_tokens;
    total_nll += other.total_nll;
}

double ScoreAggregate::mean_nll() const {
    if (scored_tokens == 0) { throw std::logic_error("perplexity aggregate is empty"); }
    return total_nll / static_cast<double>(scored_tokens);
}

double ScoreAggregate::ppl() const { return std::exp(mean_nll()); }

void KlAggregate::add(double kl) noexcept {
    total_kl += kl;
    ++rows;
}

void KlAggregate::add(const KlAggregate& other) noexcept {
    total_kl += other.total_kl;
    rows += other.rows;
}

double KlAggregate::mean_kl() const {
    if (rows == 0) { throw std::logic_error("KL aggregate is empty"); }
    return total_kl / static_cast<double>(rows);
}

namespace {

float bf16_to_float(std::uint16_t value) noexcept {
    const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
    float result             = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

} // namespace

double kl_divergence_row(const std::uint16_t* reference, const std::uint16_t* model,
                         std::uint32_t vocab) {
    if (reference == nullptr || model == nullptr || vocab == 0) {
        throw std::invalid_argument("kl_divergence_row needs two non-empty rows");
    }
    float reference_max = -std::numeric_limits<float>::infinity();
    float model_max     = -std::numeric_limits<float>::infinity();
    for (std::uint32_t v = 0; v < vocab; ++v) {
        reference_max = std::max(reference_max, bf16_to_float(reference[v]));
        model_max     = std::max(model_max, bf16_to_float(model[v]));
    }
    double reference_sum = 0.0;
    double model_sum     = 0.0;
    for (std::uint32_t v = 0; v < vocab; ++v) {
        reference_sum += std::exp(static_cast<double>(bf16_to_float(reference[v]) - reference_max));
        model_sum += std::exp(static_cast<double>(bf16_to_float(model[v]) - model_max));
    }
    const double reference_lse = static_cast<double>(reference_max) + std::log(reference_sum);
    const double model_lse     = static_cast<double>(model_max) + std::log(model_sum);
    double divergence          = 0.0;
    for (std::uint32_t v = 0; v < vocab; ++v) {
        const double reference_logprob = static_cast<double>(bf16_to_float(reference[v])) - reference_lse;
        const double model_logprob     = static_cast<double>(bf16_to_float(model[v])) - model_lse;
        divergence += std::exp(reference_logprob) * (reference_logprob - model_logprob);
    }
    return divergence;
}

} // namespace ninfer::perplexity
