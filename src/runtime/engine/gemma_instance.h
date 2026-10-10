#pragma once

// The Gemma side of the Engine's instance boundary.
//
// runtime::ModelInstance holds a qwen3_5 Model, Parameters, Frontend and Program by name, and
// EngineCore and CausalScoreCore are templated on it. This is the second implementation of what those
// cores ask for: a model, a program with the signatures they call, and a KV capacity resolution. It
// deliberately provides only what the scoring core needs, which is a small surface — causal_score,
// memory_summary and reset_memory_peaks — so the boundary can be widened one piece at a time as
// generation is added.
//
// Per-position logits are not provided yet, and the sink is refused rather than ignored: those rows are
// what the KLD route reads, and silently scoring without them would look like a working KLD path.

#include "core/device.h"
#include "models/gemma4/load.h"
#include "models/gemma4/model.h"
#include "models/gemma4/program.h"
#include "ninfer/types.h"
#include "runtime/engine/kv_capacity.h"

#include <cstdint>
#include <memory>
#include <string>
#include <span>
#include <vector>

namespace ninfer::runtime {

// The Engine's prompt shape for this model: causal scoring needs the ids and nothing else, since the
// frontend's job for scoring is to have tokenized them already. The summary and preparation stats are
// what the Engine reads before handing the prompt on, so scoring must fill them.
struct GemmaPreparedPrompt {
    std::vector<TokenId> ids;
    PromptSummary descriptor;
    PromptPreparationStats preparation;

    [[nodiscard]] std::span<const TokenId> tokens() const noexcept { return ids; }
    [[nodiscard]] const PromptSummary& summary() const noexcept { return descriptor; }
    [[nodiscard]] const PromptPreparationStats& preparation_stats() const noexcept {
        return preparation;
    }
    [[nodiscard]] explicit operator bool() const noexcept { return !ids.empty(); }
};

struct GemmaRuntimeTypes {
    using PreparedPrompt = GemmaPreparedPrompt;
};

// The frontend the Engine calls for this model. Causal scoring needs only `prepare_tokens`, which
// carries the ids the caller already holds; tokenizing text, media and prompt templating do not exist
// here yet and say so rather than returning something empty.
class GemmaFrontend {
public:
    [[nodiscard]] GemmaPreparedPrompt prepare_tokens(std::vector<TokenId> token_ids,
                                                     bool allow_prefix_identity = true) const;
    [[nodiscard]] GemmaPreparedPrompt prepare(PromptInput input,
                                              const PreparationControl& control = {}) const;
    [[nodiscard]] std::uint32_t count_tokens(PromptInput input,
                                             const PreparationControl& control = {}) const;
    [[nodiscard]] std::vector<TokenId> tokenize_text(std::string_view text) const;
    [[nodiscard]] MediaCacheSummary media_cache_summary() const;
    [[nodiscard]] ModelSamplingDefaults sampling_defaults() const;
};

// models::gemma4::Program with the signatures the Engine's scoring core calls.
class GemmaEngineProgram {
public:
    GemmaEngineProgram(const models::gemma4::Model& model, std::int32_t capacity,
                       DeviceContext& device);

    [[nodiscard]] std::vector<float> causal_score(GemmaPreparedPrompt&& prompt,
                                                  std::uint32_t first_target,
                                                  const LogitsSink& logits);
    [[nodiscard]] MemorySummary memory_summary() const;
    void reset_memory_peaks() noexcept;

private:
    const DeviceExecutionView execution_;
    models::gemma4::Program program_;
};

struct GemmaInstance {
    using ModelContract = GemmaRuntimeTypes;
    std::unique_ptr<models::gemma4::Model> model;
    GemmaFrontend frontend;
    std::unique_ptr<GemmaEngineProgram> program;
    KvCapacityResolution kv_capacity_resolution;
    const std::uint32_t capacity;

    GemmaInstance(std::unique_ptr<models::gemma4::Model> model, std::uint32_t capacity,
                  DeviceContext& device);
    ~GemmaInstance();
    GemmaInstance(const GemmaInstance&)            = delete;
    GemmaInstance& operator=(const GemmaInstance&) = delete;
};

// Whether an artifact declares this model. Gemma is detected by name; anything else takes the Qwen
// path, whose own config validation rejects what it does not recognize.
[[nodiscard]] bool artifact_is_gemma(const std::string& path);

// Loads an artifact and wraps it the way the Engine's scoring core expects.
[[nodiscard]] std::unique_ptr<GemmaInstance> load_gemma_instance(const std::string& path,
                                                                 models::LoadOptions options,
                                                                 std::uint32_t capacity,
                                                                 DeviceContext& device);

} // namespace ninfer::runtime
