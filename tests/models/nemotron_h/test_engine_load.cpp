// The Nemotron-H Engine's load path and its generation path, end to end.
//
// The Engine dispatches this family by the artifact's own config, binds the 45.28 GB artifact into
// the model's parameters, reports ready with the metadata it derived from that config, and then
// serves a generation: the prompt is prefilled into the lane's state, one token is drawn, and each
// later round decodes one token against the state the prefill left. The tokens the Engine returns
// are the family's own -- the last stage of a path whose first stage is the composition's forward --
// so a returned token is evidence that the whole wiring ran, not that a stub answered.
//
// NINFER_NEMOTRON_ARTIFACT unset skips (77): the artifact is 45 GB and a routine run must not pay
// for it. A machine without a CUDA device skips too, because the Engine initializes one.
#include "ninfer/engine.h"

#include "artifact/reader.h"
#include "models/nemotron_h/config.h"
#include "models/nemotron_h/program.h"

#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using namespace ninfer;

namespace {

namespace nemotron = ninfer::models::nemotron_h;

int failures = 0;

void expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

} // namespace

int main() {
    const char* artifact_path = std::getenv("NINFER_NEMOTRON_ARTIFACT");
    if (artifact_path == nullptr || *artifact_path == '\0') {
        std::cout << "SKIP: NINFER_NEMOTRON_ARTIFACT is not set\n";
        return 77;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    std::string failure;
    std::vector<std::int32_t> generated;
    try {
        // The artifact's own text config, which is what the Engine's metadata has to agree with:
        // parsed here by the family's parser rather than restated, so a metadata field that is
        // wired to the wrong config value is caught.
        const nemotron::TextConfig expected =
            nemotron::parse_config(artifact::Reader(artifact_path).directory()).text;

        EngineOptions options;
        options.artifact_path   = artifact_path;
        options.device          = 0;
        options.max_context     = 2048;
        options.max_concurrency = 1;

        const auto started = std::chrono::steady_clock::now();
        Engine engine(options);
        const double load_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::cout << "engine ready | " << engine.load_summary().model_name << " | architecture "
                  << engine.load_summary().architecture << " | load " << load_seconds << " s\n";
        expect(engine.is_available(), "the Engine reports itself unavailable after construction");

        const LoadSummary load = engine.load_summary();
        expect(load.architecture == "nemotron_h", "the load summary names another architecture");
        expect(load.model_name == std::filesystem::path(artifact_path).stem().string(),
               "the load summary's model name is not the artifact's own stem");
        expect(load.host_to_device_bytes != 0,
               "the load summary reports no device-resident weights");

        const ModelMetadata metadata = engine.model_metadata();
        std::cout << "metadata | vocab " << metadata.vocab_size << " | hidden "
                  << metadata.embedding_size << " | native context " << metadata.native_context
                  << '\n';
        expect(metadata.model_id == load.model_name,
               "the metadata's model id is not the Engine's model name");
        expect(metadata.vocab_size == expected.vocab_size && metadata.vocab_size != 0,
               "the metadata's vocabulary does not match the artifact's config");
        expect(metadata.embedding_size == expected.hidden_size,
               "the metadata's embedding width does not match the artifact's config");
        expect(metadata.native_context == expected.max_position_embeddings,
               "the metadata's native context does not match the artifact's config");

        // What this family cannot do is switched off rather than accepted and ignored, so the
        // Engine reports the options it actually runs with.
        const EngineOptions& running = engine.options();
        expect(!running.context_cache.enabled,
               "the context cache is enabled for a model that keeps none");
        expect(running.speculative.backend == SpeculativeBackend::None,
               "speculation is enabled for a family with no drafter");
        expect(!running.enable_vision, "Vision is enabled for a family with no vision component");
        expect(running.kv_cache == KvCacheStorage::BFloat16,
               "a quantized K/V storage is enabled for a family whose one K/V route is BF16");

        const MemorySummary memory = engine.memory_summary();
        std::cout << "memory | state " << (memory.sequence.capacity_bytes >> 20)
                  << " MiB | weights " << (memory.weights.used_bytes >> 20)
                  << " MiB | runtime reservation " << (memory.runtime_reservation_bytes >> 20)
                  << " MiB\n";
        expect(memory.sequence.capacity_bytes != 0,
               "the family's state was reported as no bytes at all");
        expect(memory.weights.used_bytes != 0, "no device-resident weights were reported");
        expect(memory.kv_cache == KvCacheStorage::BFloat16,
               "the memory summary reports a K/V storage this family has no route for");

        PromptInput input;
        input.messages.push_back(
            ChatMessage{.role = ChatRole::User, .parts = {MessagePart{.text = "hi"}}});
        PreparedPrompt prompt = engine.prepare(std::move(input));
        expect(prompt.summary().prompt_tokens != 0, "the frontend prepared no tokens");
        std::cout << "prepared " << prompt.summary().prompt_tokens << " prompt tokens\n";

        RequestOptions request;
        // Four tokens rather than one: a single token could come from the prefill's own draw, while
        // every token after the first is a decode round that had to carry the state forward.
        request.execution.requested_output_tokens = 4;
        try {
            const GenerationResult result = engine.generate(std::move(prompt), std::move(request));
            generated                       = result.generated_token_ids;
            for (const std::int32_t id : generated) {
                if (id < 0 || id >= static_cast<std::int32_t>(expected.vocab_size)) {
                    expect(false, "a generated token is outside the model's vocabulary");
                }
            }
        } catch (const std::exception& error) { failure = error.what(); }
        // The Engine stays usable after a request: whatever the request did, the next one may run.
        expect(engine.is_available(), "the Engine is unavailable after the request");
    } catch (const std::exception& error) {
        std::cerr << "FAIL: the Engine load path threw " << error.what() << '\n';
        return 1;
    }

    if (!failure.empty()) {
        std::cerr << "FAIL: the generation threw " << failure << '\n';
        ++failures;
    }
    std::cout << "generated " << generated.size() << " token(s):";
    for (const std::int32_t id : generated) { std::cout << ' ' << id; }
    std::cout << '\n';
    expect(generated.size() == 4, "the Engine did not return the four tokens it was asked for");

    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "OK\n";
    return 0;
}
