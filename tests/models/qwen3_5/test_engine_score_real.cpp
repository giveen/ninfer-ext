#include "ninfer/engine.h"

#include "real_test_artifact.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

// BF16 bits to float without pulling in a CUDA header.
float bf16_to_float(std::uint16_t value) {
    const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
    float result             = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    try {
    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.purpose       = ninfer::EnginePurpose::CausalScoring;
    options.max_context   = 2048;
    options.kv_cache      = ninfer::KvCacheStorage::Fp8E4M3Row256;
    ninfer::Engine engine(options);
    const auto& effective = engine.options();
    if (effective.max_concurrency != 1 || effective.prefill_chunk != 1024 ||
        effective.kv_capacity.mode != ninfer::KvCapacityMode::Explicit ||
        effective.kv_capacity.explicit_tokens != effective.max_context ||
        effective.context_cache.enabled ||
        effective.speculative.backend != ninfer::SpeculativeBackend::None ||
        effective.kv_cache != ninfer::KvCacheStorage::Fp8E4M3Row256) {
        std::cerr << "causal scoring options were not normalized correctly\n";
        return 1;
    }

    std::string text;
    const std::string paragraph =
        "NInfer scores each target token from the preceding hidden state. "
        "Every evaluation window owns fresh state and a fresh KV address space.\n";
    std::vector<ninfer::TokenId> tokens;
    while (tokens.size() < 1537) {
        text += paragraph;
        tokens = engine.tokenize_text(text);
    }
    tokens.resize(1537);

    const std::vector<float> all      = engine.score_tokens(tokens, 1);
    const std::vector<float> suffix   = engine.score_tokens(tokens, 513);
    const std::vector<float> repeated = engine.score_tokens(tokens, 513);
    if (all.size() != 1536 || suffix.size() != 1024 || repeated.size() != suffix.size()) {
        std::cerr << "causal scoring returned an invalid result shape\n";
        return 1;
    }
    for (const float value : all) {
        if (!std::isfinite(value) || value > 0.0F) {
            std::cerr << "causal scoring returned an invalid log probability\n";
            return 1;
        }
    }
    for (std::size_t i = 0; i < suffix.size(); ++i) {
        if (!std::isfinite(suffix[i]) || suffix[i] > 0.0F) {
            std::cerr << "causal scoring returned a non-finite logprob\n";
            return 1;
        }
        if (suffix[i] != repeated[i]) {
            std::cerr << "a repeated score window inherited prior State/KV\n";
            return 1;
        }
    }
    // The exported logits are the real distribution: the log softmax of a column at its target token
    // reproduces the returned log probability, and asking for them does not change the result.
    std::vector<ninfer::TokenId> export_tokens = tokens;
    double worst_export_error                  = 0.0;
    std::size_t exported_columns               = 0;
    const ninfer::LogitsSink sink              = [&](const ninfer::ScoredLogits& view) {
        for (std::uint32_t column = 0; column < view.columns; ++column) {
            const std::uint32_t target = view.first_target + column;
            const std::size_t index    = static_cast<std::size_t>(target) - 513U;
            if (index >= suffix.size()) { continue; }
            const std::uint16_t* values =
                view.values.data() + static_cast<std::size_t>(column) * view.vocab_size;
            float max_value = -std::numeric_limits<float>::infinity();
            for (std::uint32_t vocab = 0; vocab < view.vocab_size; ++vocab) {
                max_value = std::max(max_value, bf16_to_float(values[vocab]));
            }
            double sum = 0.0;
            for (std::uint32_t vocab = 0; vocab < view.vocab_size; ++vocab) {
                sum += std::exp(static_cast<double>(bf16_to_float(values[vocab]) - max_value));
            }
            const double logprob =
                static_cast<double>(
                    bf16_to_float(values[static_cast<std::uint32_t>(tokens[target])]) - max_value) -
                std::log(sum);
            worst_export_error = std::max(
                worst_export_error, std::abs(logprob - static_cast<double>(suffix[index])));
            ++exported_columns;
        }
    };
    const std::vector<float> with_logits = engine.score_tokens(export_tokens, 513, sink);
    if (exported_columns == 0) {
        std::cerr << "the logits sink received no columns\n";
        return 1;
    }
    if (with_logits.size() != suffix.size()) {
        std::cerr << "the logits sink changed the scored result shape\n";
        return 1;
    }
    if (worst_export_error > 2.0e-3) {
        std::cerr << "exported logits disagree with the scored log probability: " << worst_export_error
                  << '\n';
        return 1;
    }

    std::cout << "OK causal_score_real (logits max |delta logprob| " << worst_export_error << ")\n";
    return 0;
    } catch (const std::exception& error) {
        return ninfer::test::real_test_error(error);
    }
}
