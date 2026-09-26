// Concurrent prefill and KV streaming against a real artifact: greedy tokens of requests that run
// together, with interleaved prefill chunks and optionally with older KV pages streamed to Host,
// must equal each request run alone with its whole context resident. In a concurrent run one
// request stops at its first token, so the other decodes alone and no batched-decode arithmetic
// enters the comparison; a second run swaps the roles.

#include "ninfer/engine.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kMaxContext   = 16384;
constexpr std::uint32_t kPrefillChunk = 256;
constexpr std::uint32_t kOutputTokens = 32;

// Distinct pseudo-text prompts: a fixed LCG over ordinary vocabulary ids.
std::vector<ninfer::TokenId> make_prompt(std::uint32_t seed, std::size_t tokens) {
    std::vector<ninfer::TokenId> ids;
    ids.reserve(tokens);
    std::uint32_t state = seed;
    for (std::size_t i = 0; i < tokens; ++i) {
        state = state * 1664525U + 1013904223U;
        ids.push_back(static_cast<ninfer::TokenId>(1000U + (state >> 8) % 60000U));
    }
    return ids;
}

ninfer::EngineOptions engine_options(const char* artifact, std::uint32_t concurrency,
                                     std::uint32_t kv_capacity, bool kv_stream) {
    ninfer::EngineOptions options;
    options.artifact_path        = artifact;
    options.max_context          = kMaxContext;
    options.kv_capacity          = ninfer::KvCapacityPolicy::explicit_capacity(kv_capacity);
    options.prefill_chunk        = kPrefillChunk;
    options.max_concurrency      = concurrency;
    options.max_pending_requests = concurrency;
    options.kv_stream            = kv_stream;
    if (kv_stream) { options.context_cache.host_kv_capacity_bytes = 1ULL << 30; }
    return options;
}

ninfer::RequestOptions greedy_request(std::uint32_t output_tokens = kOutputTokens) {
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = output_tokens;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = false;
    request.stop.include_model_defaults       = false;
    return request;
}

// Runs every prompt together; request `long_request` generates kOutputTokens, the others one.
std::vector<std::vector<ninfer::TokenId>>
run_together(const ninfer::EngineOptions& options,
             const std::vector<std::vector<ninfer::TokenId>>& prompts, std::size_t long_request) {
    ninfer::Engine engine(options);
    std::vector<ninfer::GenerationHandle> handles;
    for (std::size_t i = 0; i < prompts.size(); ++i) {
        handles.push_back(engine.submit(engine.prepare_tokens(prompts[i]),
                                        greedy_request(i == long_request ? kOutputTokens : 1U)));
    }
    std::vector<std::vector<ninfer::TokenId>> outputs;
    for (auto& handle : handles) { outputs.push_back(handle.wait().generated_token_ids); }
    return outputs;
}

// Each observed output must be a prefix of the request's resident reference.
int compare(const std::string& label, const std::vector<std::vector<ninfer::TokenId>>& reference,
            const std::vector<std::vector<ninfer::TokenId>>& observed) {
    int failures = 0;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        const auto& got = observed[i];
        if (got.empty() || got.size() > reference[i].size() ||
            !std::equal(got.begin(), got.end(), reference[i].begin())) {
            std::cerr << label << ": request " << i << " differs from its resident reference\n";
            ++failures;
        }
    }
    return failures;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact) {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    try {
        const std::vector<std::vector<ninfer::TokenId>> prompts{make_prompt(1, 7000),
                                                                make_prompt(2, 6500)};
        // Reference: each request alone, its whole context resident.
        std::vector<std::vector<ninfer::TokenId>> reference;
        {
            ninfer::Engine engine(engine_options(artifact, 1, kMaxContext, false));
            for (const auto& prompt : prompts) {
                reference.push_back(
                    engine.generate(engine.prepare_tokens(prompt), greedy_request())
                        .generated_token_ids);
            }
        }
        int failures = 0;
        // Streaming alone: a ~5.5k-token Device window, so older pages move to Host during
        // prefill (read through prefill staging) and decode (read in place).
        failures += compare("kv-stream C=1", {reference[0]},
                            run_together(engine_options(artifact, 1, 5504, true), {prompts[0]}, 0));
        for (std::size_t long_request = 0; long_request < prompts.size(); ++long_request) {
            const std::string suffix = " (long request " + std::to_string(long_request) + ")";
            // Prefill chunks of both requests interleave: each must use its own KV rows.
            failures += compare("concurrent" + suffix, reference,
                                run_together(engine_options(artifact, 2, 2 * kMaxContext, false),
                                             prompts, long_request));
            failures += compare("kv-stream C=2" + suffix, reference,
                                run_together(engine_options(artifact, 2, 11008, true), prompts,
                                             long_request));
        }
        if (failures != 0) { return 1; }
    } catch (const std::exception& e) {
        std::cerr << "uncaught exception: " << e.what() << '\n';
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
