#include "artifact/binder.h" // ParameterReference
#include "artifact/reader.h"
#include "models/gemma4/expected_weights.h"
#include "models/gemma4/load.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
namespace gemma = ninfer::models::gemma4;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "gemma4 load: " << message << '\n';
        ++failures;
    }
}

void check_shape(const gemma::LoadPlan& plan, gemma::WeightId id, const std::string& what,
                 std::vector<std::uint64_t> expected) {
    const auto& reference = plan.parameter(id);
    check(reference.shape == expected, what + ": expected a " + std::to_string(expected.size()) +
                                           "-D shape of " + std::to_string(expected.front()) +
                                           ", the artifact resolved something else");
}

} // namespace

int main() {
    const char* path = std::getenv("NINFER_GEMMA_ARTIFACT");
    if (path == nullptr || *path == '\0') {
        std::cout << "SKIP: NINFER_GEMMA_ARTIFACT is not set\n";
        return 77;
    }

    try {
        artifact::Reader reader(path);
        const auto plan = gemma::plan_load(reader);

        // This test does not pin the model's own numbers -- tests/models/gemma4/test_config.cpp
        // does that from the fixtures. Its subject is that the binding follows whatever the
        // artifact's own config says, so the dense 31B and the MoE 26B are both covered.
        const auto& config = plan.config();
        check(config.sliding_window == 1024, "sliding window");
        check(config.num_hidden_layers > 0, "layer count");
        std::size_t global_layers = 0;
        for (std::size_t layer = 0; layer < config.num_hidden_layers; ++layer) {
            const bool global = (layer + 1) % 6 == 0;
            check(config.sliding_attention(layer) == !global,
                  "layer pattern at " + std::to_string(layer));
            global_layers += global ? 1 : 0;
        }
        check(global_layers == config.global_layers, "global layer count matches the config");

        // Every parameter resolved, counted from the structure rather than pinned: a layer binds
        // four norms, its scalar, its attention projections and norms, and its dense MLP (14 with a
        // value projection, 13 without); a MoE layer adds three norms, the router with its two
        // scales, and one matrix per expert per projection. A shape or format the artifact does not
        // store would have thrown above. `models/gemma4/expected_weights.h` holds that structure,
        // which `test_model.cpp` counts with too.
        const auto& weights        = plan.weights();
        const std::uint64_t hidden = config.hidden_size;
        const std::uint64_t expected = ninfer::test::gemma4_expected_parameter_count(config);
        check(plan.parameter_count() == expected,
              "parameter count is " + std::to_string(plan.parameter_count()) +
                  ", the structure has " + std::to_string(expected));
        check(weights.text.layers.size() == config.num_hidden_layers, "bound layer count");

        check_shape(plan, weights.text.token_embedding, "token_embedding",
                    {config.vocab_size, hidden});
        check_shape(plan, weights.text.output_head, "output_head", {config.vocab_size, hidden});
        check_shape(plan, weights.text.final_norm, "final_norm", {hidden});

        const gemma::WeightId unset{};
        for (std::size_t layer = 0; layer < weights.text.layers.size(); ++layer) {
            const auto& bound    = weights.text.layers[layer];
            const bool sliding   = config.sliding_attention(layer);
            const auto& geometry = config.geometry(layer);
            const std::string at = "layer " + std::to_string(layer);
            const std::uint64_t query_rows =
                static_cast<std::uint64_t>(config.num_attention_heads) * geometry.head_dim;
            const std::uint64_t key_rows =
                static_cast<std::uint64_t>(geometry.num_key_value_heads) * geometry.head_dim;
            check(bound.mixer == (sliding ? gemma::MixerKind::SlidingAttention
                                          : gemma::MixerKind::FullAttention),
                  at + ": mixer kind");
            check_shape(plan, bound.input_norm, at + " input_norm", {hidden});
            check_shape(plan, bound.layer_scalar, at + " layer_scalar", {1});
            check_shape(plan, bound.attention.query, at + " query", {query_rows, hidden});
            check_shape(plan, bound.attention.key, at + " key", {key_rows, hidden});
            check_shape(plan, bound.attention.output, at + " output", {hidden, query_rows});
            check_shape(plan, bound.attention.query_norm, at + " query_norm", {geometry.head_dim});
            check_shape(plan, bound.mlp.gate, at + " mlp gate", {config.intermediate_size, hidden});
            check_shape(plan, bound.mlp.down, at + " mlp down", {hidden, config.intermediate_size});
            // The global layers bind no value at all: k is v.
            if (sliding) {
                check(bound.attention.value != unset, at + ": sliding layer binds a value");
                check_shape(plan, bound.attention.value, at + " value", {key_rows, hidden});
            } else {
                check(bound.attention.value == unset, at + ": global layer binds no value");
            }
            if (config.enable_moe_block) {
                check(bound.moe.has_value(), at + ": MoE layer binds its block");
                const auto& moe = *bound.moe;
                check_shape(plan, moe.router, at + " moe router", {config.num_experts, hidden});
                check_shape(plan, moe.router_scale, at + " moe router_scale", {hidden});
                check_shape(plan, moe.per_expert_scale, at + " per_expert_scale",
                            {config.num_experts});
                check_shape(plan, moe.post_feedforward_norm_1, at + " post_feedforward_norm_1",
                            {hidden});
                check_shape(plan, moe.pre_feedforward_norm_2, at + " pre_feedforward_norm_2",
                            {hidden});
                check_shape(plan, moe.post_feedforward_norm_2, at + " post_feedforward_norm_2",
                            {hidden});
                check(moe.experts.size() == config.num_experts, at + ": bound expert count");
                for (std::size_t expert = 0; expert < moe.experts.size(); ++expert) {
                    const std::string ex = at + " expert " + std::to_string(expert);
                    check_shape(plan, moe.experts[expert].gate, ex + " gate",
                                {config.moe_intermediate_size, hidden});
                    check_shape(plan, moe.experts[expert].up, ex + " up",
                                {config.moe_intermediate_size, hidden});
                    check_shape(plan, moe.experts[expert].down, ex + " down",
                                {hidden, config.moe_intermediate_size});
                }
            } else {
                check(!bound.moe.has_value(), at + ": dense layer binds no MoE block");
            }
        }

        std::cout << "gemma4 load: " << plan.parameter_count() << " parameters resolved over "
                  << weights.text.layers.size() << " layers, " << global_layers << " global\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "gemma4 load: " << error.what() << '\n';
        return 1;
    }
}
