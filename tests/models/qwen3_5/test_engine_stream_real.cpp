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
                                     std::uint32_t kv_capacity, bool kv_stream, bool mtp = false) {
    ninfer::EngineOptions options;
    options.artifact_path        = artifact;
    options.max_context          = kMaxContext;
    options.kv_capacity          = ninfer::KvCapacityPolicy::explicit_capacity(kv_capacity);
    options.prefill_chunk        = kPrefillChunk;
    options.max_concurrency      = concurrency;
    options.max_pending_requests = concurrency;
    options.kv_stream            = kv_stream;
    if (kv_stream) { options.context_cache.host_kv_capacity_bytes = 1ULL << 30; }
    if (mtp) {
        // Fixed K: an adaptive draft length follows startup-timed round costs, not the model.
        options.speculative.backend      = ninfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 3;
        options.speculative.fixed_draft  = true;
    }
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

struct Turns {
    std::vector<ninfer::TokenId> first;
    std::vector<ninfer::TokenId> second;
    std::uint32_t reused = 0;
};

// Two conversation turns with prefix reuse: the second prompt extends the first turn's prompt and
// output, so it can resume from the first turn's continuation.
Turns run_turns(const ninfer::EngineOptions& options, std::vector<ninfer::TokenId> prompt,
                std::uint32_t first_output, const std::vector<ninfer::TokenId>& follow_up) {
    ninfer::Engine engine(options);
    ninfer::RequestOptions request = greedy_request(first_output);
    request.execution.allow_prefix_reuse = true;
    Turns turns;
    turns.first = engine.generate(engine.prepare_tokens(prompt), request).generated_token_ids;
    prompt.insert(prompt.end(), turns.first.begin(), turns.first.end());
    prompt.insert(prompt.end(), follow_up.begin(), follow_up.end());
    request.execution.requested_output_tokens = kOutputTokens;
    const ninfer::GenerationResult second = engine.generate(engine.prepare_tokens(prompt), request);
    turns.second = second.generated_token_ids;
    turns.reused = second.reused_prompt_tokens;
    return turns;
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
        std::vector<ninfer::TokenId> mtp_reference;
        {
            ninfer::Engine engine(engine_options(artifact, 1, kMaxContext, false, true));
            mtp_reference = engine.generate(engine.prepare_tokens(prompts[0]), greedy_request())
                                .generated_token_ids;
        }
        int failures = 0;
        // Streaming alone: a ~5.5k-token Device window, so older pages move to Host during
        // prefill and decode, and every step reads them through Device staging.
        failures += compare("kv-stream C=1", {reference[0]},
                            run_together(engine_options(artifact, 1, 5504, true), {prompts[0]}, 0));
        failures +=
            compare("kv-stream C=1 MTP", {mtp_reference},
                    run_together(engine_options(artifact, 1, 5504, true, true), {prompts[0]}, 0));
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
        // Context cache under streaming: a conversation that fits its window resumes from its
        // continuation exactly as a resident engine does.
        const std::vector<ninfer::TokenId> follow_up = make_prompt(4, 500);
        const Turns resident =
            run_turns(engine_options(artifact, 1, kMaxContext, false), make_prompt(3, 2000), 32,
                      follow_up);
        const Turns streamed =
            run_turns(engine_options(artifact, 1, 5504, true), make_prompt(3, 2000), 32, follow_up);
        if (streamed.reused == 0 || streamed.reused != resident.reused ||
            streamed.first != resident.first || streamed.second != resident.second) {
            std::cerr << "kv-stream cache: reused " << streamed.reused << " vs resident "
                      << resident.reused << ", or tokens differ\n";
            ++failures;
        }
        // A turn that spills during decode releases instead of publishing, and its longer follow-up
        // no longer fits the window: it prefills from scratch and still completes.
        const Turns spilled =
            run_turns(engine_options(artifact, 1, 5504, true), make_prompt(5, 5000), 1500,
                      follow_up);
        if (spilled.first.size() != 1500 || spilled.second.size() != kOutputTokens ||
            spilled.reused != 0) {
            std::cerr << "kv-stream spill: turn sizes " << spilled.first.size() << "/"
                      << spilled.second.size() << ", reused " << spilled.reused << "\n";
            ++failures;
        }
        if (failures != 0) { return 1; }
    } catch (const std::exception& e) {
        std::cerr << "uncaught exception: " << e.what() << '\n';
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
