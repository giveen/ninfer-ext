// Scores a token sequence the way ninfer-perplexity does: the Program returns the natural
// log-probability of every target token, and this reports the mean negative log-likelihood and the
// resulting perplexity. It also writes the per-position log-probabilities so the independent reference
// can be compared entry by entry rather than only through the aggregate.
//
//   NINFER_GEMMA_ARTIFACT  the artifact
//   NINFER_GEMMA_PROMPT    a file of int32 little-endian token ids
//   NINFER_GEMMA_FIRST_TARGET  first scored position (default 1)
//   NINFER_GEMMA_SEQUENTIAL    also score token at a time and require the batched passes to agree;
//                              past the sliding window this is what catches a pass overwriting
//                              ring rows its own earlier queries still read
//   NINFER_GEMMA_SCORES_OUT    write the per-target log-probabilities (float32) to this file

#include "artifact/reader.h"
#include "core/device.h"
#include "models/gemma4/load.h"
#include "models/gemma4/program.h"
#include "ninfer/engine.h"
#include "runtime/engine/causal_score_core.h"
#include "runtime/engine/gemma_instance.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

using namespace ninfer;
namespace gemma = ninfer::models::gemma4;

std::vector<std::int32_t> read_ids(const char* path) {
    std::ifstream source(path, std::ios::binary | std::ios::ate);
    const std::streamsize bytes = source.tellg();
    source.seekg(0);
    std::vector<std::int32_t> ids(static_cast<std::size_t>(bytes) / sizeof(std::int32_t));
    source.read(reinterpret_cast<char*>(ids.data()), bytes);
    return ids;
}

} // namespace

int main() {
    const char* artifact    = std::getenv("NINFER_GEMMA_ARTIFACT");
    const char* prompt_path = std::getenv("NINFER_GEMMA_PROMPT");
    if (artifact == nullptr || *artifact == '\0' || prompt_path == nullptr) {
        std::cout << "SKIP: NINFER_GEMMA_ARTIFACT and NINFER_GEMMA_PROMPT must be set\n";
        return 77;
    }
    const char* first_text   = std::getenv("NINFER_GEMMA_FIRST_TARGET");
    const std::int32_t first = first_text != nullptr ? std::stoi(first_text) : 1;
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        // The tokenizer travels inside the artifact. A Gemma tokenizer.json is megabytes, so an empty
        // view would mean the resource binding silently found nothing.
        {
            const models::gemma4::LoadPlan plan =
                models::gemma4::plan_load(artifact::Reader(artifact));
            const models::gemma4::TextResources& resources = plan.resources();
            std::cout << "  artifact resources: tokenizer.json " << resources.tokenizer_json.size()
                      << " bytes, tokenizer_config.json " << resources.tokenizer_config_json.size()
                      << " bytes\n";
            if (resources.tokenizer_json.size() < 100000 ||
                resources.tokenizer_config_json.empty()) {
                std::cerr << "gemma4 score: the artifact's tokenizer resources are missing\n";
                return 1;
            }
        }

        DeviceContext device;
        const std::vector<std::int32_t> ids = read_ids(prompt_path);

        // The direct path, scoped so the model it needs is released before the Engine core loads its
        // own: two of these do not fit beside each other on a 32 GB device.
        std::vector<float> logprobs;
        {
            auto model = gemma::load_model(artifact, models::LoadOptions{}, device);
            gemma::Program program(*model, static_cast<std::int32_t>(ids.size()) + 8, 1, device);
            logprobs = program.causal_score(ids, first, {}, device.execution_view());

            if (std::getenv("NINFER_GEMMA_SEQUENTIAL") != nullptr) {
                const std::int32_t vocabulary = static_cast<std::int32_t>(model->config().vocab_size);
                std::vector<std::uint16_t> host(static_cast<std::size_t>(vocabulary));
                program.reset(0);
                const std::size_t window = model->config().sliding_window;
                // Per-target |batched - token-at-a-time|, split at the sliding window.
                std::array<double, 2> difference{};
                std::array<std::size_t, 2> count{};
                double worst_beyond = 0.0;
                std::vector<float> sequential_scores;
                for (std::size_t position = 0; position + 1 < ids.size(); ++position) {
                    program.decode(0, ids[position], device.execution_view());
                    if (static_cast<std::int32_t>(position + 1) < first) continue;
                    // The pass runs on the execution stream, which a legacy-stream copy does not
                    // wait for.
                    CUDA_CHECK(cudaStreamSynchronize(device.execution_view().stream));
                    CUDA_CHECK(cudaMemcpy(host.data(), program.logits().data,
                                          host.size() * sizeof(std::uint16_t),
                                          cudaMemcpyDeviceToHost));
                    double largest = -1e300;
                    std::vector<double> values(host.size());
                    for (std::size_t id = 0; id < host.size(); ++id) {
                        const std::uint32_t bits = static_cast<std::uint32_t>(host[id]) << 16;
                        float value;
                        std::memcpy(&value, &bits, sizeof(value));
                        values[id] = value;
                        largest    = std::max(largest, values[id]);
                    }
                    double total_exp = 0.0;
                    for (const double value : values) total_exp += std::exp(value - largest);
                    const double sequential =
                        values[static_cast<std::size_t>(ids[position + 1])] - largest -
                        std::log(total_exp);
                    const double batched =
                        logprobs[position + 1 - static_cast<std::size_t>(first)];
                    sequential_scores.push_back(static_cast<float>(sequential));
                    const std::size_t side = position + 1 >= window ? 1 : 0;
                    difference[side] += std::abs(sequential - batched);
                    ++count[side];
                    if (side == 1) worst_beyond = std::max(worst_beyond, std::abs(sequential - batched));
                }
                if (const char* path = std::getenv("NINFER_GEMMA_SEQUENTIAL_OUT")) {
                    std::ofstream out(path, std::ios::binary);
                    out.write(reinterpret_cast<const char*>(sequential_scores.data()),
                              static_cast<std::streamsize>(sequential_scores.size() * sizeof(float)));
                    std::ofstream batched_out(std::string(path) + ".batched", std::ios::binary);
                    batched_out.write(reinterpret_cast<const char*>(logprobs.data()),
                                      static_cast<std::streamsize>(logprobs.size() * sizeof(float)));
                }
                // The two routes run different linear kernels (a GEMV against a tiled SIMT pass), and
                // this artifact's low-probability logits are sensitive to that rounding, so inside the
                // window they already differ by a few tenths of a nat per target on average. That is
                // the baseline. Past the window a pass that overwrites ring rows its own earlier
                // queries read adds a systematic error on top of it: measured, 0.36 against 0.13.
                if (count[0] == 0 || count[1] == 0) {
                    std::cerr << "gemma4 score: the sequential check needs targets on both sides of "
                                 "the sliding window\n";
                    return 1;
                }
                const double inside = difference[0] / static_cast<double>(count[0]);
                const double beyond = difference[1] / static_cast<double>(count[1]);
                std::cout << "  token at a time against batched passes: mean |difference| " << inside
                          << " nats inside the window, " << beyond << " beyond it (worst "
                          << worst_beyond << ")\n";
                if (beyond > inside) {
                    std::cerr << "gemma4 score: past the sliding window the batched passes drift from "
                                 "token-at-a-time\n";
                    return 1;
                }
            }

        // The logits the KLD route reads: one row per scored target, each `vocabulary` wide. Dumping
        // them lets HuggingFace's own logits check that they belong to the targets they claim.
        if (std::getenv("NINFER_GEMMA_SINK_DUMP") != nullptr) {
            std::vector<float> rows;
            const LogitsSink sink = [&](const ScoredLogits& view) {
                const std::size_t width = static_cast<std::size_t>(view.vocab_size);
                for (std::size_t column = 0; column < view.columns; ++column) {
                    for (std::size_t id = 0; id < width; ++id) {
                        const std::uint16_t code = view.values[column * width + id];
                        const std::uint32_t bits   = static_cast<std::uint32_t>(code) << 16;
                        float value;
                        std::memcpy(&value, &bits, sizeof(value));
                        rows.push_back(value);
                    }
                }
                std::cout << "  sink: first_target " << view.first_target << ", columns " << view.columns
                          << ", vocabulary " << view.vocab_size << '\n';
            };
            const std::vector<float> through_sink = program.causal_score(ids, first, sink,
                                                                        device.execution_view());
            std::ofstream dump("/tmp/gemma_sink.f32", std::ios::binary);
            dump.write(reinterpret_cast<const char*>(rows.data()),
                       static_cast<std::streamsize>(rows.size() * sizeof(float)));
            std::cout << "  sink rows: " << rows.size() << " values for " << through_sink.size()
                      << " targets\n";
        }
        }

        // Per-target log-probabilities, for an entry-by-entry comparison against an oracle's.
        if (const char* scores_path = std::getenv("NINFER_GEMMA_SCORES_OUT")) {
            std::ofstream scores(scores_path, std::ios::binary);
            scores.write(reinterpret_cast<const char*>(logprobs.data()),
                         static_cast<std::streamsize>(logprobs.size() * sizeof(float)));
        }

        double total = 0.0;
        for (const float logprob : logprobs) {
            if (!std::isfinite(logprob)) {
                std::cerr << "gemma4 score: a non-finite log-probability\n";
                return 1;
            }
            total -= static_cast<double>(logprob);
        }
        const double mean_nll = total / static_cast<double>(logprobs.size());
        std::cout << "gemma4 score: " << ids.size() << " tokens, " << logprobs.size()
                  << " scored from " << first << ", mean NLL " << mean_nll << ", perplexity "
                  << std::exp(mean_nll) << '\n';
        std::cout << "  first log-probabilities:";
        for (std::size_t index = 0; index < std::min<std::size_t>(logprobs.size(), 6); ++index) {
            std::cout << ' ' << logprobs[index];
        }
        std::cout << '\n';

        // The Engine's own scoring core, instantiated on this model. It must produce exactly what the
        // Program does directly, since it is the same Program behind one adapter.
        if (std::getenv("NINFER_GEMMA_ENGINE_CORE") != nullptr) {
          {
            runtime::GemmaPreparedPrompt prompt;
            prompt.ids = ids;
            EngineOptions options;
            options.artifact_path = artifact;
            options.purpose       = EnginePurpose::CausalScoring;
            options.max_context   = static_cast<std::uint32_t>(ids.size()) + 8;
            auto instance =
                runtime::load_gemma_instance(runtime::gemma_engine_options(options), device);
            runtime::CausalScoreCore<runtime::GemmaInstance> core(*instance, device);
            const std::vector<float> through_core = core.score(std::move(prompt), first, {});
            if (through_core.size() != logprobs.size()) {
                std::cerr << "gemma4 score: the Engine core returned " << through_core.size()
                          << " scores, the Program returned " << logprobs.size() << '\n';
                return 1;
            }
            for (std::size_t index = 0; index < logprobs.size(); ++index) {
                if (through_core[index] != logprobs[index]) {
                    std::cerr << "gemma4 score: the Engine core differs at " << index << ": "
                              << through_core[index] << " against " << logprobs[index] << '\n';
                    return 1;
                }
            }
            std::cout << "  the Engine's scoring core matches the Program on " << logprobs.size()
                      << " positions\n";
          }
        }

        // The whole public route: the Engine picks this model from the artifact's architecture, loads
        // it, and scores through its own core and variant dispatch. Scoped so only one 24 GB model is
        // resident at a time.
        if (std::getenv("NINFER_GEMMA_ENGINE_API") != nullptr) {
            EngineOptions options;
            options.artifact_path = artifact;
            options.purpose       = EnginePurpose::CausalScoring;
            options.max_context   = static_cast<std::uint32_t>(ids.size()) + 8;
            Engine engine(std::move(options));
            const std::vector<float> through_api = engine.score_tokens(ids, first, {});
            if (through_api.size() != logprobs.size()) {
                std::cerr << "gemma4 score: the Engine returned " << through_api.size()
                          << " scores, the Program returned " << logprobs.size() << '\n';
                return 1;
            }
            for (std::size_t index = 0; index < logprobs.size(); ++index) {
                if (through_api[index] != logprobs[index]) {
                    std::cerr << "gemma4 score: the Engine differs at " << index << ": "
                              << through_api[index] << " against " << logprobs[index] << '\n';
                    return 1;
                }
            }
            std::cout << "  the Engine, through its public API, matches the Program on "
                      << logprobs.size() << " positions\n";
        }

        std::ofstream output("/tmp/gemma_score.out.f32", std::ios::binary);
        output.write(reinterpret_cast<const char*>(logprobs.data()),
                     static_cast<std::streamsize>(logprobs.size() * sizeof(float)));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "gemma4 score: " << error.what() << '\n';
        return 1;
    }
}
