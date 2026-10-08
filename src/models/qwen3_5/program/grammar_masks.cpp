#include "core/device.h"
#include "models/qwen3_5/program/program_impl.h"

namespace ninfer::models::qwen3_5::detail {

ops::SamplingMask ProgramImpl::fill_grammar_mask(runtime::TokenMaskProvider* provider,
                                                 std::size_t row,
                                                 std::span<const TokenId> drafts) {
    grammar_dead_positions[row] = 0;
    if (provider == nullptr || !provider->constrained(row)) { return {}; }
    if (drafts.size() + 1U > draft_window + 1U) {
        throw std::logic_error("constrained round verifies more positions than it can mask");
    }
    const auto words  = static_cast<std::size_t>(grammar_masks_device.ne[0]);
    const auto offset = row * (draft_window + 1) * words;
    std::span<std::uint32_t> host(static_cast<std::uint32_t*>(grammar_masks_host->data()) + offset,
                                  (drafts.size() + 1) * words);
    // The provider fills host memory; a dead position is reported, not sampled from.
    grammar_dead_positions[row] = provider->fill(row, drafts, host);
    auto* device_words = static_cast<std::uint32_t*>(grammar_masks_device.data) + offset;
    CUDA_CHECK(cudaMemcpyAsync(device_words, host.data(), host.size_bytes(), cudaMemcpyHostToDevice,
                               device.stream));
    return {device_words, static_cast<std::int32_t>(words)};
}

}  // namespace ninfer::models::qwen3_5::detail
