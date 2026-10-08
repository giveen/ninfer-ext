// Opt-in real-model residue sweep: capture transparency for speculative decode with context-cache
// (prefix) reuse.
//
// Invariant: for one artifact, backend, draft width and prompt length, CUDA-graph capture must not
// change committed tokens. The sweep runs the same request under capture and without it, on both a
// fresh computation and a request that resumes from a committed prefix checkpoint, and requires the
// two graph modes to agree.
//
// The failure this targets is a captured execution that runs the wrong graph for an
// execution shape - the class where a cached request under capture returns corrupted or truncated
// output at only some residues. Capture is semantically transparent (it replays the same kernels),
// so byte equality is a supported invariant here. It is deliberately NOT compared across the
// cache-hit and fresh routes: a restored prefix was produced by decode kernels while a fresh prompt
// is produced by prefill kernels, and the repository does not define bit parity across arbitrary
// floating-point routes. The row's `route=` field reports that divergence for information only.
//
// The draft width is startup-fixed per process (DFlash/DFlash2 1..15, MTP 1..7) and pinned with
// `fixed_draft`, so every round has the same verify shape. Sweep it by re-running with
// --draft-k 1..15. MTP's adaptive policy is deliberately excluded: it picks its per-round draft
// length from host timing, which capture changes, so it is not a shape-residue invariant.
// The binary with the wrong component (for example DFlash2 on an MTP-only artifact) skips.

#include "ninfer/engine.h"

#include "kv_cache_storage.h"
#include "real_test_artifact.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using ninfer::SpeculativeBackend;

// A deterministic corpus cycled to reach each length. Content only has to be stable, but prose
// keeps acceptance representative rather than training the MTP head on a repeated single token.
constexpr std::string_view kCorpus =
    "Substrate materialization and prefix identity. The engine restores a checkpoint only when the "
    "token, position and mode identity match exactly, otherwise it recomputes the suffix from the "
    "nearest exact frontier. Generation commits tokens at stable boundaries. A captured graph "
    "replays the exact kernel sequence for one shape. ";

std::vector<std::uint32_t> default_lengths(std::uint32_t context, std::uint32_t outputs,
                                           std::uint32_t draft_k) {
    // Lengths chosen to straddle prefill chunks (128), KV page groups (64) and the low graph
    // buckets. Each is a boundary and both of its neighbours.
    static constexpr std::uint32_t kCandidates[] = {
        63,  64,  65,   127,  128,  129,  191,  192,  193,  255,  256,  257,
        383, 384, 385,  511,  512,  513,  767,  768,  769,  1023, 1024, 1025,
    };
    const std::uint32_t ceiling =
        context > 2 * outputs + draft_k + 1 ? context - 2 * outputs - draft_k - 1 : 1;
    std::vector<std::uint32_t> lengths;
    for (const std::uint32_t candidate : kCandidates) {
        if (candidate <= ceiling) { lengths.push_back(candidate); }
    }
    if (lengths.empty()) { lengths.push_back(ceiling); }
    return lengths;
}

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

ninfer::RequestOptions greedy_request(std::uint32_t outputs, bool reuse) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = reuse;
    options.stop.include_model_defaults       = false;
    return options;
}

std::vector<ninfer::TokenId> prompt_of_length(const std::vector<ninfer::TokenId>& base,
                                              std::uint32_t length) {
    if (base.empty()) { throw std::runtime_error("tokenizer produced an empty corpus"); }
    std::vector<ninfer::TokenId> prompt;
    prompt.reserve(length);
    while (prompt.size() < length) {
        const std::size_t take = std::min<std::size_t>(base.size(), length - prompt.size());
        prompt.insert(prompt.end(), base.begin(), base.begin() + static_cast<std::ptrdiff_t>(take));
    }
    return prompt;
}

std::size_t first_difference(const std::vector<ninfer::TokenId>& lhs,
                             const std::vector<ninfer::TokenId>& rhs) {
    const std::size_t shared = std::min(lhs.size(), rhs.size());
    for (std::size_t i = 0; i < shared; ++i) {
        if (lhs[i] != rhs[i]) { return i; }
    }
    return lhs.size() == rhs.size() ? lhs.size() : shared;
}

struct Config {
    std::filesystem::path artifact;
    SpeculativeBackend backend = SpeculativeBackend::Mtp;
    std::uint32_t draft_k      = 7;
    std::uint32_t outputs      = 16;
    std::uint32_t max_context  = 2048;
    ninfer::KvCacheStorage kv  = ninfer::KvCacheStorage::BFloat16;
    std::vector<std::uint32_t> lengths;
};

ninfer::EngineOptions engine_options(const Config& config, bool graph) {
    ninfer::EngineOptions options;
    options.artifact_path                    = config.artifact;
    options.max_context                      = config.max_context;
    options.kv_capacity =
        ninfer::KvCapacityPolicy::explicit_capacity(config.max_context);
    options.prefill_chunk                    = 128;
    options.max_concurrency                  = 1;
    options.max_pending_requests             = 1;
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

struct SweepRow {
    std::uint32_t length    = 0;
    std::size_t frontier    = 0;
    std::uint32_t reuse     = 0;
    std::uint64_t rounds    = 0;
    std::uint64_t fallback  = 0;
    std::uint64_t accepted  = 0;
    std::uint64_t drafted   = 0;
    std::vector<ninfer::TokenId> fresh;
    std::vector<ninfer::TokenId> hit;
    std::size_t hit_vs_fresh_diff = 0; // informational route divergence, not a failure
};

using Sweep = std::map<std::uint32_t, SweepRow>;

// Runs one prompt-length sweep on one engine. It asserts structural progress only; the
// cross-graph token comparison is the caller's job because it needs the other graph mode.
Sweep run_sweep(ninfer::Engine& engine, const Config& config, bool graph,
                std::vector<std::string>& failures) {
    Sweep sweep;
    const auto base = engine.tokenize_text(kCorpus);
    const std::string label =
        "graph=" + std::to_string(graph ? 1 : 0) + " " + backend_name(config.backend) +
        " K=" + std::to_string(config.draft_k) + " fixed";
    const std::uint64_t graph_allowance = engine.memory_summary().cuda_graph_allowance_bytes;
    std::cout << "[residue] " << label << " graph_allowance_bytes=" << graph_allowance << '\n';
    if (graph && graph_allowance == 0) {
        failures.push_back(label +
                           ": capture requested but the engine planned no CUDA-graph allowance, so "
                           "the capture comparison would be vacuous");
    }

    for (const std::uint32_t length : config.lengths) {
        const auto origin = prompt_of_length(base, length);

        // A completed request publishes its checkpoint at the last token it processed; its final
        // committed token stays the next unprocessed anchor, so that frontier is origin + O - 1.
        // The baseline therefore runs with reuse enabled (the publishing path).
        const auto baseline = engine.generate(engine.prepare_tokens(origin),
                                              greedy_request(config.outputs, true));
        if (baseline.generated_token_ids.size() != config.outputs ||
            baseline.finish_reason != ninfer::FinishReason::OutputLimit) {
            failures.push_back(label + " L=" + std::to_string(length) + ": baseline committed " +
                               std::to_string(baseline.generated_token_ids.size()) + " of " +
                               std::to_string(config.outputs) + " tokens");
            continue;
        }
        if (baseline.speculative.backend != config.backend) {
            failures.push_back(label + " L=" + std::to_string(length) + ": baseline used backend " +
                               backend_name(baseline.speculative.backend));
            continue;
        }
        if (baseline.speculative.rounds + baseline.speculative.fallback_steps == 0) {
            failures.push_back(label + " L=" + std::to_string(length) +
                               ": speculative path executed no round, the comparison is vacuous");
            continue;
        }

        std::vector<ninfer::TokenId> frontier = origin;
        frontier.insert(frontier.end(), baseline.generated_token_ids.begin(),
                        baseline.generated_token_ids.end() - 1);

        const auto hit =
            engine.generate(engine.prepare_tokens(frontier), greedy_request(config.outputs, true));
        if (hit.reused_prompt_tokens != frontier.size()) {
            failures.push_back(label + " L=" + std::to_string(length) + ": expected a full resume of " +
                               std::to_string(frontier.size()) + " frontier tokens, got " +
                               std::to_string(hit.reused_prompt_tokens));
            continue;
        }

        const auto fresh =
            engine.generate(engine.prepare_tokens(frontier), greedy_request(config.outputs, false));
        if (fresh.reused_prompt_tokens != 0) {
            failures.push_back(label + " L=" + std::to_string(length) + ": reuse=false still reused " +
                               std::to_string(fresh.reused_prompt_tokens) + " prompt tokens");
        }
        if (fresh.generated_token_ids.size() != config.outputs ||
            fresh.finish_reason != ninfer::FinishReason::OutputLimit) {
            failures.push_back(label + " L=" + std::to_string(length) + ": fresh run committed " +
                               std::to_string(fresh.generated_token_ids.size()) + " of " +
                               std::to_string(config.outputs) + " tokens");
            continue;
        }

        SweepRow row;
        row.length    = length;
        row.frontier  = frontier.size();
        row.reuse     = hit.reused_prompt_tokens;
        row.rounds    = hit.speculative.rounds;
        row.fallback  = hit.speculative.fallback_steps;
        row.accepted  = hit.speculative.accepted_tokens;
        row.drafted   = hit.speculative.drafted_tokens;
        row.fresh     = fresh.generated_token_ids;
        row.hit        = hit.generated_token_ids;
        row.hit_vs_fresh_diff = first_difference(row.fresh, row.hit);
        std::cout << "[residue] " << label << " L=" << length << " frontier=" << row.frontier
                  << " reuse=" << row.reuse << " rounds=" << row.rounds
                  << " fallback=" << row.fallback << " accepted=" << row.accepted
                  << " drafted=" << row.drafted
                  << " route="
                  << (row.hit_vs_fresh_diff == row.fresh.size() ? "same"
                                                                : "diff@" + std::to_string(row.hit_vs_fresh_diff))
                  << '\n';
        sweep[length] = std::move(row);
    }
    return sweep;
}

std::vector<std::uint32_t> parse_lengths(std::string_view text) {
    std::vector<std::uint32_t> lengths;
    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t comma = text.find(',', start);
        const std::string_view piece =
            text.substr(start, comma == std::string_view::npos ? std::string_view::npos : comma - start);
        if (!piece.empty()) {
            lengths.push_back(static_cast<std::uint32_t>(std::stoul(std::string(piece))));
        }
        if (comma == std::string_view::npos) { break; }
        start = comma + 1;
    }
    return lengths;
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
        } else if (arg == "--outputs") {
            config.outputs = static_cast<std::uint32_t>(std::stoul(value(i)));
        } else if (arg == "--max-context") {
            config.max_context = static_cast<std::uint32_t>(std::stoul(value(i)));
        } else if (arg == "--kv") {
            config.kv = ninfer::test::parse_kv_cache_storage(value(i));
        } else if (arg == "--lengths") {
            config.lengths = parse_lengths(value(i));
        } else if (arg == "--help") {
            std::cout << "--artifact PATH [--backend mtp|dflash|dflash2|eagle3] [--draft-k N]\n"
                         "  [--outputs N] [--max-context N]\n"
                         "  [--kv bf16|int8|fp8|nvfp4|k8v4] [--lengths a,b,c]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument " + arg);
        }
    }
    if (config.backend == SpeculativeBackend::Mtp) {
        if (config.draft_k < 1 || config.draft_k > 7) {
            throw std::invalid_argument("MTP draft-k must be 1..7");
        }
    } else {
        if (config.draft_k < 1 || config.draft_k > 15) {
            throw std::invalid_argument("DFlash/DFlash2/Eagle3 draft-k must be 1..15");
        }
    }
    config.max_context = std::max(config.max_context, config.outputs + config.draft_k + 2);
    if (config.lengths.empty()) {
        config.lengths = default_lengths(config.max_context, config.outputs, config.draft_k);
    }
    return config;
}

} // namespace

int main(int argc, char** argv) {
    try {
        Config config = parse_arguments(argc, argv);
        if (config.artifact.empty()) {
            std::cout << "skip: supply --artifact PATH\n";
            return 77;
        }
        for (const std::uint32_t length : config.lengths) {
            if (length + 2 * config.outputs + config.draft_k + 1 > config.max_context) {
                throw std::runtime_error("length " + std::to_string(length) +
                                         " leaves no room under max-context " +
                                         std::to_string(config.max_context));
            }
        }

        std::vector<std::string> failures;
        std::cout << "[residue] artifact=" << config.artifact.string()
                  << " backend=" << backend_name(config.backend) << " K=" << config.draft_k
                  << " fixed lengths=" << config.lengths.size() << '\n';

        // Capture is semantically transparent, so the same request must commit the same tokens with
        // and without it - on the fresh route and, above all, on the cache-hit route where the
        // capture/cache interaction lives.
        Sweep captured;
        {
            ninfer::Engine engine(engine_options(config, true));
            captured = run_sweep(engine, config, true, failures);
        }
        Sweep eager;
        {
            ninfer::Engine engine(engine_options(config, false));
            eager = run_sweep(engine, config, false, failures);
        }

        for (const std::uint32_t length : config.lengths) {
            const auto captured_row = captured.find(length);
            const auto eager_row    = eager.find(length);
            if (captured_row == captured.end() || eager_row == eager.end()) { continue; }
            if (captured_row->second.fresh != eager_row->second.fresh) {
                failures.push_back(
                    "L=" + std::to_string(length) +
                    ": capture changed fresh tokens at index " +
                    std::to_string(first_difference(captured_row->second.fresh, eager_row->second.fresh)));
            }
            if (captured_row->second.hit != eager_row->second.hit) {
                failures.push_back(
                    "L=" + std::to_string(length) +
                    ": capture changed cache-hit tokens at index " +
                    std::to_string(first_difference(captured_row->second.hit, eager_row->second.hit)));
            }
            if (captured_row->second.reuse != eager_row->second.reuse ||
                captured_row->second.rounds != eager_row->second.rounds) {
                failures.push_back("L=" + std::to_string(length) +
                                   ": capture changed reuse/round structure");
            }
        }

        if (!failures.empty()) {
            std::cerr << "[residue] " << failures.size() << " residue failure(s):\n";
            for (const auto& failure : failures) { std::cerr << "  " << failure << '\n'; }
            return 1;
        }
        std::uint64_t rounds = 0;
        for (const auto& [length, row] : captured) {
            (void)length;
            rounds += row.rounds;
        }
        std::cout << "[residue] ok " << backend_name(config.backend) << " K=" << config.draft_k
                  << " fixed lengths=" << config.lengths.size()
                  << " captured_rounds=" << rounds << '\n';
        return 0;
    } catch (const std::exception& error) {
        return ninfer::test::real_test_error(error);
    }
}
