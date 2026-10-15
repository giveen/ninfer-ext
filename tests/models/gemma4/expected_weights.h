#pragma once
// The parameter count an artifact's own config implies, as one oracle for the tests that count
// bindings. The number is derived from the structure and never pinned to a checkpoint, so the dense
// 31B (60 layers, 833 parameters) and the MoE 26B-A4B (30 layers, 12118) are both covered.
//
// Three whole-model parameters -- token embedding, output head, final norm -- plus, per layer, the
// attention block and its dense MLP (14 parameters when the layer stores a value projection, 13
// when K is V) and, when the artifact carries the MoE block, three norms, the router with its two
// scales, and one matrix per expert per projection: 6 + 3 * num_experts.
#include "models/gemma4/config.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::test {

[[nodiscard]] inline std::uint64_t
gemma4_expected_parameter_count(const models::gemma4::TextConfig& config) {
    const std::uint64_t moe_extra =
        config.enable_moe_block ? 6 + 3 * static_cast<std::uint64_t>(config.num_experts) : 0;
    std::uint64_t expected = 3;
    for (std::size_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        expected += moe_extra + (config.sliding_attention(layer) ? 14 : 13);
    }
    return expected;
}

} // namespace ninfer::test
