#include "runtime/engine/gemma_instance.h"

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "models/qwen3_5/frontend/tokenizer.h"

#include <stdexcept>

namespace ninfer::runtime {
namespace {

namespace text_frontend = models::qwen3_5::frontend;

// The artifact's resources are only mapped for the duration of a load, so a reader is opened, the
// bytes are copied out, and it goes away again.
std::string read_tokenizer_resource(const std::string& path, const char* role) {
    artifact::Reader reader(path);
    artifact::Binder binder(reader);
    const auto bytes = binder.host_object(binder.resource("text", role));
    return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

} // namespace


GemmaFrontend::GemmaFrontend(std::string tokenizer_json, std::string tokenizer_config_json,
                             std::string generation_config_json)
    : tokenizer_json_(std::move(tokenizer_json)),
      tokenizer_config_json_(std::move(tokenizer_config_json)),
      generation_config_json_(std::move(generation_config_json)) {
    tokenizer_ = std::make_unique<text_frontend::Tokenizer>(text_frontend::TokenizerResources{
        .tokenizer_json         = tokenizer_json_,
        .tokenizer_config_json  = tokenizer_config_json_,
        .generation_config_json = generation_config_json_,
        .family                 = text_frontend::TokenizerFamily::Gemma,
    });
    // This pipeline's config declares a beginning-of-sequence token, so the id comes from the
    // pipeline rather than from a constant.
    const std::vector<int> bos = tokenizer_->encode("<bos>", text_frontend::EncodeOptions{});
    if (bos.size() != 1) {
        throw std::runtime_error("gemma4 frontend: <bos> is not a single token in this pipeline");
    }
    bos_ = bos.front();
}

GemmaPreparedPrompt GemmaFrontend::prepare_tokens(std::vector<TokenId> token_ids,
                                                bool allow_prefix_identity) const {
    (void)allow_prefix_identity;
    // Deliberately transparent: the token count a caller handed over is the count it gets back, which
    // is what the Engine's scoring contract checks against its own window. The beginning-of-sequence
    // token is the tokenizer's business, added below.
    GemmaPreparedPrompt prompt;
    prompt.descriptor.prompt_tokens = static_cast<std::uint32_t>(token_ids.size());
    prompt.ids                      = std::move(token_ids);
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
    const std::vector<int> encoded = tokenizer_->encode(text);
    std::vector<TokenId> ids(encoded.begin(), encoded.end());
    // Text becomes the ids the model would see, and this pipeline's config declares a
    // beginning-of-sequence token: without it the model is asked about a sequence that does not start
    // the way it was trained to, which moves every probability in the sequence. Prepending here and
    // not in prepare_tokens also keeps the token count the caller planned its windows over.
    if (bos_ >= 0 && (ids.empty() || ids.front() != bos_)) {
        ids.insert(ids.begin(), bos_);
    }
    return ids;
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
                             std::string tokenizer_json, std::string tokenizer_config_json,
                             std::string generation_config_json, DeviceContext& device)
    : model(std::move(source)),
      frontend(std::move(tokenizer_json), std::move(tokenizer_config_json),
               std::move(generation_config_json)),
      program(nullptr), capacity(capacity_in) {
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
    std::string tokenizer_json = read_tokenizer_resource(path, "tokenizer.json");
    std::string tokenizer_config = read_tokenizer_resource(path, "tokenizer_config.json");
    std::string generation_config = read_tokenizer_resource(path, "generation_config.json");
    auto model = models::gemma4::load_model(path, options, device);
    return std::make_unique<GemmaInstance>(std::move(model), capacity, std::move(tokenizer_json),
                                           std::move(tokenizer_config),
                                           std::move(generation_config), device);
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
