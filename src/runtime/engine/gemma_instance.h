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
#include <span>
#include <vector>

namespace ninfer::runtime {

// The Engine's prompt shape for this model: causal scoring needs the ids and nothing else, since the
// frontend's job for scoring is to have tokenized them already.
struct GemmaPreparedPrompt {
    std::vector<TokenId> ids;
    [[nodiscard]] std::span<const TokenId> tokens() const noexcept { return ids; }
};

struct GemmaRuntimeTypes {
    using PreparedPrompt = GemmaPreparedPrompt;
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
    std::unique_ptr<GemmaEngineProgram> program;
    KvCapacityResolution kv_capacity_resolution;
    const std::uint32_t capacity;

    GemmaInstance(std::unique_ptr<models::gemma4::Model> model, std::uint32_t capacity,
                  DeviceContext& device);
    ~GemmaInstance();
    GemmaInstance(const GemmaInstance&)            = delete;
    GemmaInstance& operator=(const GemmaInstance&) = delete;
};

// Loads an artifact and wraps it the way the Engine's scoring core expects.
[[nodiscard]] std::unique_ptr<GemmaInstance> load_gemma_instance(const std::string& path,
                                                                 models::LoadOptions options,
                                                                 std::uint32_t capacity,
                                                                 DeviceContext& device);

} // namespace ninfer::runtime
