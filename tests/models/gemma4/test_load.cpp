#include "artifact/binder.h" // ParameterReference
#include "artifact/reader.h"
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
    check(reference.shape == expected,
          what + ": expected a " + std::to_string(expected.size()) + "-D shape of " +
              std::to_string(expected.front()) + ", the artifact resolved something else");
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

        // The config is the artifact's own, and the layer pattern is what the mathematics needs.
        const auto& config = plan.config();
        check(config.hidden_size == 5376, "hidden size");
        check(config.intermediate_size == 21504, "intermediate size");
        check(config.vocab_size == 262144, "vocabulary");
        check(config.num_hidden_layers == 60, "layer count");
        check(config.num_attention_heads == 32, "query head count");
        check(config.sliding_window == 1024, "sliding window");
        check(config.sliding_attention(0), "layer 0 is sliding");
        check(!config.sliding_attention(5), "layer 5 is global");
        check(!config.sliding_attention(59), "layer 59 is global");
        std::size_t global_layers = 0;
        for (std::size_t layer = 0; layer < config.num_hidden_layers; ++layer) {
            if (!config.sliding_attention(layer)) { ++global_layers; }
        }
        check(global_layers == config.global_layers, "global layer count matches the config");

        // Every parameter resolved: 50 sliding layers of 14 plus 10 global layers of 13 plus three
        // text-level ones. A shape or format the artifact does not store would have thrown above.
        check(plan.parameter_count() == 833,
              "parameter count is " + std::to_string(plan.parameter_count()));

        const auto& weights = plan.weights();
        check(weights.text.layers.size() == 60, "bound layer count");

        // Text level, including the head stored as BF16 rather than the FP8 the plan expected.
        check_shape(plan, weights.text.token_embedding, "token_embedding", {262144, 5376});
        check_shape(plan, weights.text.output_head, "output_head", {262144, 5376});
        check_shape(plan, weights.text.final_norm, "final_norm", {5376});

        const gemma::WeightId unset{};
        for (std::size_t layer = 0; layer < weights.text.layers.size(); ++layer) {
            const auto& bound  = weights.text.layers[layer];
            const bool sliding = config.sliding_attention(layer);
            const std::string at = "layer " + std::to_string(layer);
            check(bound.mixer == (sliding ? gemma::MixerKind::SlidingAttention
                                          : gemma::MixerKind::FullAttention),
                  at + ": mixer kind");
            check_shape(plan, bound.input_norm, at + " input_norm", {5376});
            check_shape(plan, bound.layer_scalar, at + " layer_scalar", {1});
            check_shape(plan, bound.attention.query, at + " query",
                        {sliding ? 8192U : 16384U, 5376U});
            check_shape(plan, bound.attention.key, at + " key",
                        {sliding ? 4096U : 2048U, 5376U});
            check_shape(plan, bound.attention.output, at + " output",
                        {5376U, sliding ? 8192U : 16384U});
            check_shape(plan, bound.attention.query_norm, at + " query_norm",
                        {sliding ? 256U : 512U});
            check_shape(plan, bound.mlp.gate, at + " mlp gate", {21504, 5376});
            check_shape(plan, bound.mlp.down, at + " mlp down", {5376, 21504});
            // The global layers bind no value at all: k is v.
            if (sliding) {
                check(bound.attention.value != unset, at + ": sliding layer binds a value");
                check_shape(plan, bound.attention.value, at + " value", {4096, 5376});
            } else {
                check(bound.attention.value == unset, at + ": global layer binds no value");
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
