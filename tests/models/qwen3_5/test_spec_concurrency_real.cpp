// Opt-in real-model Engine test: a speculative backend must complete rounds that carry more than
// one request.
//
// The failure this targets is a draft proposal that reads its logits through a buffer wider than
// its head's own output region. At one lane the trailing unit extents make that sub-range look
// contiguous, so nothing fails; at two or more lanes the same sub-range is strided, and the round
// fails (or would read the wrong region). Capture is the harsher configuration: the batch-2 body
// runs while the engine is constructed, so a missing width fails construction instead of the
// first round.
//
// The lanes run the same greedy prompt, so they must commit identical tokens: the batch dimension
// is an independent index in the verify, draft and sampling kernels, and a lane-dependent buffer
// mix is what this compares. The comparison is within one graph mode; the repository defines no
// bit parity across different shapes or routes.

#include "ninfer/engine.h"

#include "kv_cache_storage.h"
#include "real_test_artifact.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using ninfer::SpeculativeBackend;

// Stable prose rather than a repeated token, so the draft and verify paths run on real content.
constexpr std::string_view kCorpus =
    "Substrate materialization and prefix identity. The engine restores a checkpoint only when the "
    "token, position and mode identity match exactly, otherwise it recomputes the suffix from the "
    "nearest exact frontier. Generation commits tokens at stable boundaries. A captured graph "
    "replays the exact kernel sequence for one shape. ";

SpeculativeBackend parse_backend(std::string_view name) {
    if (name == "mtp") { return SpeculativeBackend::Mtp; }
    if (name == "dflash") { return SpeculativeBackend::DFlash; }
    if (name == "dflash2") { return SpeculativeBackend::DFlash2; }
    if (name == "eagle3") { return SpeculativeBackend::Eagle3; }
    throw std::invalid_argument("unknown speculative backend " + std::string(name));
}

const char* backend_name(SpeculativeBackend backend) noexcept {
    switch (backend) {
        case SpeculativeBackend::Mtp: return "mtp";
        case SpeculativeBackend::DFlash: return "dflash";
        case SpeculativeBackend::DFlash2: return "dflash2";
        case SpeculativeBackend::Eagle3: return "eagle3";
        default: return "none";
    }
}

struct Config {
    std::filesystem::path artifact;
    SpeculativeBackend backend = SpeculativeBackend::Eagle3;
    std::uint32_t draft_k      = 3;
    std::uint32_t concurrency  = 2;
    std::uint32_t outputs      = 24;
    std::uint32_t max_context  = 2048;
    std::uint32_t prompt       = 64;
    ninfer::KvCacheStorage kv  = ninfer::KvCacheStorage::BFloat16;
    bool capture               = true;
    bool eager                 = true;
};

ninfer::EngineOptions engine_options(const Config& config, bool graph) {
    ninfer::EngineOptions options;
    options.artifact_path                    = config.artifact;
    options.max_context                      = config.max_context;
    options.kv_capacity =
        ninfer::KvCapacityPolicy::explicit_capacity(config.max_context);
    options.prefill_chunk                    = 128;
    options.max_concurrency                  = config.concurrency;
    options.max_pending_requests             = config.concurrency;
    options.kv_cache                         = config.kv;
    options.use_cuda_graph                   = graph;
    options.speculative.backend              = config.backend;
    options.speculative.draft_tokens         = config.draft_k;
    options.speculative.fixed_draft          = true;
    // EAGLE3 carries its own 32k draft head, so the generic proposal head does not apply.
    options.speculative.proposal_head        = config.backend == SpeculativeBackend::Eagle3
                                                   ? ninfer::ProposalHead::Full
                                                   : ninfer::ProposalHead::Optimized;
    options.context_cache.device_state_slots = 4;
    return options;
}

void run_mode(const Config& config, bool graph, std::vector<std::string>& failures) {
    const std::string label = std::string("graph=") + (graph ? "1" : "0");
    ninfer::Engine engine(engine_options(config, graph));

    std::vector<ninfer::TokenId> prompt;
    const auto corpus = engine.tokenize_text(kCorpus);
    if (corpus.empty()) { throw std::runtime_error("tokenizer produced an empty corpus"); }
    while (prompt.size() < config.prompt) {
        const std::size_t take =
            std::min<std::size_t>(corpus.size(), config.prompt - prompt.size());
        prompt.insert(prompt.end(), corpus.begin(),
                      corpus.begin() + static_cast<std::ptrdiff_t>(take));
    }

    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = config.outputs;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = false;
    request.stop.include_model_defaults       = false;

    // Submit every lane before consuming any, so the engine's admission forms a multi-lane round.
    std::vector<ninfer::GenerationHandle> handles;
    handles.reserve(config.concurrency);
    for (std::uint32_t lane = 0; lane < config.concurrency; ++lane) {
        handles.push_back(engine.submit(engine.prepare_tokens(prompt), request));
    }

    std::vector<ninfer::GenerationResult> results;
    results.reserve(config.concurrency);
    for (auto& handle : handles) { results.push_back(handle.wait()); }

    std::uint64_t rounds = 0;
    for (std::uint32_t lane = 0; lane < config.concurrency; ++lane) {
        const std::string row = label + " lane=" + std::to_string(lane);
        const ninfer::GenerationResult& result = results[lane];
        if (result.generated_token_ids.size() != config.outputs ||
            result.finish_reason != ninfer::FinishReason::OutputLimit) {
            failures.push_back(row + ": committed " +
                               std::to_string(result.generated_token_ids.size()) + " of " +
                               std::to_string(config.outputs) + " tokens");
        }
        if (result.speculative.backend != config.backend) {
            failures.push_back(row + ": served by " + backend_name(result.speculative.backend));
        }
        if (result.speculative.rounds + result.speculative.fallback_steps == 0) {
            failures.push_back(row +
                               ": the speculative round never executed at this concurrency");
        }
        rounds += result.speculative.rounds;
        if (lane != 0 && result.generated_token_ids != results.front().generated_token_ids) {
            failures.push_back(row + ": identical prompts committed different tokens");
        }
        std::cout << "[concurrency] " << row << " rounds=" << result.speculative.rounds
                  << " fallback=" << result.speculative.fallback_steps
                  << " accepted=" << result.speculative.accepted_tokens
                  << " drafted=" << result.speculative.drafted_tokens << '\n';
    }
    if (rounds == 0) {
        failures.push_back(label +
                           ": no lane ran a speculative draft round, so the batched round path was "
                           "not exercised");
    }
}

Config parse_arguments(int argc, char** argv) {
    Config config;
    const auto value = [&](int& index) -> std::string {
        if (++index >= argc) {
            throw std::invalid_argument("missing value for " + std::string(argv[index - 1]));
        }
        return argv[index];
    };
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--artifact") {
            config.artifact = value(i);
        } else if (arg == "--backend") {
            config.backend = parse_backend(value(i));
        } else if (arg == "--draft-k") {
            config.draft_k = static_cast<std::uint32_t>(std::stoul(value(i)));
        } else if (arg == "--concurrency") {
            config.concurrency = static_cast<std::uint32_t>(std::stoul(value(i)));
        } else if (arg == "--outputs") {
            config.outputs = static_cast<std::uint32_t>(std::stoul(value(i)));
        } else if (arg == "--max-context") {
            config.max_context = static_cast<std::uint32_t>(std::stoul(value(i)));
        } else if (arg == "--prompt") {
            config.prompt = static_cast<std::uint32_t>(std::stoul(value(i)));
        } else if (arg == "--kv") {
            config.kv = ninfer::test::parse_kv_cache_storage(value(i));
        } else if (arg == "--graph") {
            const std::string mode = value(i);
            if (mode != "on" && mode != "off" && mode != "both") {
                throw std::invalid_argument("--graph expects on, off or both");
            }
            config.capture = mode != "off";
            config.eager   = mode != "on";
        } else if (arg == "--help") {
            std::cout << "--artifact PATH [--backend mtp|dflash|dflash2|eagle3] [--draft-k N]\n"
                         "  [--concurrency N] [--outputs N] [--prompt N] [--max-context N]\n"
                         "  [--kv bf16|int8|fp8|nvfp4|k8v4] [--graph on|off|both]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument " + arg);
        }
    }
    if (config.backend == SpeculativeBackend::Mtp) {
        if (config.draft_k < 1 || config.draft_k > 7) {
            throw std::invalid_argument("MTP draft-k must be 1..7");
        }
    } else if (config.draft_k < 1 || config.draft_k > 15) {
        throw std::invalid_argument("DFlash/DFlash2/Eagle3 draft-k must be 1..15");
    }
    if (config.concurrency < 2 || config.concurrency > 8) {
        throw std::invalid_argument("concurrency must be 2..8: one lane cannot exercise a batched "
                                    "round");
    }
    config.max_context = std::max(config.max_context, config.outputs + config.draft_k + 2);
    return config;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Config config = parse_arguments(argc, argv);
        if (config.artifact.empty()) {
            std::cout << "skip: supply --artifact PATH\n";
            return 77;
        }
        std::cout << "[concurrency] artifact=" << config.artifact.string() << " backend="
                  << backend_name(config.backend) << " K=" << config.draft_k
                  << " lanes=" << config.concurrency << " outputs=" << config.outputs
                  << " prompt=" << config.prompt << '\n';
        std::vector<std::string> failures;
        if (config.capture) { run_mode(config, true, failures); }
        if (config.eager) { run_mode(config, false, failures); }
        if (!failures.empty()) {
            std::cerr << "[concurrency] " << failures.size() << " failure(s):\n";
            for (const auto& failure : failures) { std::cerr << "  " << failure << '\n'; }
            return 1;
        }
        std::cout << "[concurrency] ok " << backend_name(config.backend) << " K=" << config.draft_k
                  << " lanes=" << config.concurrency << '\n';
        return 0;
    } catch (const std::exception& error) {
        return ninfer::test::real_test_error(error);
    }
}
