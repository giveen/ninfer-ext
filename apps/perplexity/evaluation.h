#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::perplexity {

struct WindowPlan {
    std::size_t input_begin    = 0;
    std::size_t input_end      = 0;
    std::size_t target_begin   = 0;
    std::size_t target_end     = 0;
    std::uint32_t first_target = 0;
};

[[nodiscard]] std::vector<WindowPlan> plan_windows(std::size_t tokens, std::uint32_t context,
                                                   std::uint32_t stride);

struct ScoreAggregate {
    std::uint64_t scored_tokens = 0;
    double total_nll            = 0.0;

    void add(std::span<const float> logprobs);
    void add(const ScoreAggregate& other) noexcept;
    [[nodiscard]] double mean_nll() const;
    [[nodiscard]] double ppl() const;
};

struct KlAggregate {
    std::uint64_t rows = 0;
    double total_kl    = 0.0;

    void add(double kl) noexcept;
    void add(const KlAggregate& other) noexcept;
    [[nodiscard]] double mean_kl() const;
};

// KL(P_reference || P_model) at one position, from two BF16 logits rows of `vocab` entries. The
// reference is the target distribution, matching exllamav3's util/measures.py::compute_kl_div.
[[nodiscard]] double kl_divergence_row(const std::uint16_t* reference, const std::uint16_t* model,
                                       std::uint32_t vocab);

} // namespace ninfer::perplexity
