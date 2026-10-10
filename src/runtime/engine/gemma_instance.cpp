#include "runtime/engine/gemma_instance.h"

#include "artifact/reader.h"

#include <stdexcept>

namespace ninfer::runtime {

GemmaPreparedPrompt GemmaFrontend::prepare_tokens(std::vector<TokenId> token_ids,
                                                bool allow_prefix_identity) const {
    (void)allow_prefix_identity;
    GemmaPreparedPrompt prompt;
    prompt.descriptor.prompt_tokens = static_cast<std::uint32_t>(token_ids.size());
    prompt.ids                   = std::move(token_ids);
    return prompt;
}

GemmaPreparedPrompt GemmaFrontend::prepare(PromptInput input, const PreparationControl& control) const {
    (void)input;
    (void)control;
    throw std::logic_error("Gemma prompts from text or media are not implemented");
}

std::uint32_t GemmaFrontend::count_tokens(PromptInput input, const PreparationControl& control) const {
    (void)input;
    (void)control;
    throw std::logic_error("Gemma prompt counting is not implemented");
}

std::vector<TokenId> GemmaFrontend::tokenize_text(std::string_view text) const {
    (void)text;
    throw std::logic_error("Gemma tokenization is not implemented");
}

MediaCacheSummary GemmaFrontend::media_cache_summary() const { return MediaCacheSummary{}; }

ModelSamplingDefaults GemmaFrontend::sampling_defaults() const { return ModelSamplingDefaults{}; }

GemmaEngineProgram::GemmaEngineProgram(const models::gemma4::Model& model, std::int32_t capacity,
                                       DeviceContext& device)
    : execution_(device.execution_view()), program_(model, capacity, device) {}

std::vector<float> GemmaEngineProgram::causal_score(GemmaPreparedPrompt&& prompt,
                                                    std::uint32_t first_target,
                                                    const LogitsSink& logits) {
    if (logits) {
        throw std::logic_error(
            "Gemma scoring does not provide per-position logits yet, so the KLD route cannot run");
    }
    return program_.causal_score(prompt.ids, static_cast<std::int32_t>(first_target), execution_);
}

MemorySummary GemmaEngineProgram::memory_summary() const {
    // The physical accounting the Qwen program reports through its paged KV does not exist here yet;
    // what is real is the Program's own allocation, and the capacity resolution reports the rest.
    MemorySummary summary{};
    return summary;
}

void GemmaEngineProgram::reset_memory_peaks() noexcept {}

GemmaInstance::GemmaInstance(std::unique_ptr<models::gemma4::Model> source, std::uint32_t capacity_in,
                             DeviceContext& device)
    : model(std::move(source)), program(nullptr), capacity(capacity_in) {
    program = std::make_unique<GemmaEngineProgram>(*model, static_cast<std::int32_t>(capacity),
                                                   device);
    // No paged KV here: the sliding layers hold their window and the global layers hold the tokens the
    // caller allows, so the resolution describes tokens rather than page groups.
    kv_capacity_resolution.mode                            = KvCapacityMode::Explicit;
    kv_capacity_resolution.main_page_groups                 = 1;
    kv_capacity_resolution.maximum_main_page_groups         = 1;
    kv_capacity_resolution.resolved_tokens                  = capacity;
    kv_capacity_resolution.minimum_runtime_reservation_bytes = 0;
    kv_capacity_resolution.bytes_per_additional_main_page_group = 0;
    kv_capacity_resolution.runtime_reservation_bytes        = 0;
    kv_capacity_resolution.available_after_weights_bytes    = 0;
    kv_capacity_resolution.available_after_startup_bytes    = 0;
    kv_capacity_resolution.automatic_headroom_bytes         = 0;
    kv_capacity_resolution.planned_slack_bytes              = 0;
}

GemmaInstance::~GemmaInstance() = default;

std::unique_ptr<GemmaInstance> load_gemma_instance(const std::string& path,
                                                   models::LoadOptions options,
                                                   std::uint32_t capacity, DeviceContext& device) {
    auto model = models::gemma4::load_model(path, options, device);
    return std::make_unique<GemmaInstance>(std::move(model), capacity, device);
}

bool artifact_is_gemma(const std::string& path) {
    artifact::Reader reader(path);
    const artifact::Json& config = reader.directory().component("text").config;
    if (!config.contains("architectures")) { return false; }
    const auto& architectures = config.at("architectures");
    return architectures.is_array() && !architectures.empty() &&
           architectures.at(0) == "Gemma4ForCausalLM";
}

} // namespace ninfer::runtime
