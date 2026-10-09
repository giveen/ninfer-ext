#include "models/gemma4/load.h"

#include "artifact/binder.h"
#include "artifact/reader.h"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::models::gemma4 {
namespace {

// The formats the artifact stores, checked by the binder on every request.
constexpr QType kEmbeddingFormat = QType::FP8_E4M3FN_ROW_BF16;
constexpr QType kHeadFormat      = QType::BF16;
constexpr QType kProjectionFormat = QType::FP8_E4M3FN_ROW_BF16;
constexpr QType kMlpFormat        = QType::NVFP4;
constexpr QType kNormFormat       = QType::BF16;
constexpr QType kScalarFormat     = QType::FP32;

struct PendingWeight {
    artifact::ParameterReference reference;
};

class Bindings {
public:
    explicit Bindings(artifact::Binder& binder) : binder_(binder) {}

    WeightId parameter(const std::string& name, artifact::Shape shape, QType format) {
        PendingWeight pending;
        pending.reference = binder_.parameter(name, std::move(shape), artifact::Residency::Device,
                                              format);
        weights_.push_back(std::move(pending));
        return WeightId{weights_.size() - 1};
    }

    [[nodiscard]] const std::vector<PendingWeight>& weights() const { return weights_; }

private:
    artifact::Binder& binder_;
    std::vector<PendingWeight> weights_;
};

std::string layer_prefix(std::size_t layer) {
    return "text/layers/" + std::to_string(layer) + "/";
}

TextWeights bind_text(Bindings& bindings, const TextConfig& config) {
    const std::uint64_t hidden       = config.hidden_size;
    const std::uint64_t intermediate = config.intermediate_size;
    const std::uint64_t vocabulary   = config.vocab_size;

    TextWeights out;
    out.token_embedding = bindings.parameter("text/token_embedding", {vocabulary, hidden},
                                             kEmbeddingFormat);
    out.output_head = bindings.parameter("text/output_head", {vocabulary, hidden}, kHeadFormat);
    out.final_norm  = bindings.parameter("text/final_norm", {hidden}, kNormFormat);

    out.layers.reserve(config.num_hidden_layers);
    for (std::size_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        const bool sliding   = config.sliding_attention(layer);
        const auto& geometry = config.geometry(layer);
        const std::string prefix = layer_prefix(layer);
        const std::uint64_t query_rows =
            static_cast<std::uint64_t>(config.num_attention_heads) * geometry.head_dim;
        const std::uint64_t key_rows =
            static_cast<std::uint64_t>(geometry.num_key_value_heads) * geometry.head_dim;

        LayerWeights weights;
        weights.mixer = sliding ? MixerKind::SlidingAttention : MixerKind::FullAttention;
        weights.input_norm           = bindings.parameter(prefix + "input_norm", {hidden},
                                                          kNormFormat);
        weights.post_attention_norm  = bindings.parameter(prefix + "post_attention_norm", {hidden},
                                                          kNormFormat);
        weights.pre_feedforward_norm = bindings.parameter(prefix + "pre_feedforward_norm", {hidden},
                                                          kNormFormat);
        weights.post_feedforward_norm =
            bindings.parameter(prefix + "post_feedforward_norm", {hidden}, kNormFormat);
        weights.layer_scalar = bindings.parameter(prefix + "layer_scalar", {1}, kScalarFormat);

        // The three attention input projections are slices of one fused object; requesting the
        // slice's own shape is what makes the artifact resolve its row offset.
        weights.attention.query =
            bindings.parameter(prefix + "attention/query", {query_rows, hidden}, kProjectionFormat);
        weights.attention.key =
            bindings.parameter(prefix + "attention/key", {key_rows, hidden}, kProjectionFormat);
        if (sliding) {
            weights.attention.value = bindings.parameter(prefix + "attention/value",
                                                         {key_rows, hidden}, kProjectionFormat);
        }
        weights.attention.output = bindings.parameter(prefix + "attention/output", {hidden, query_rows},
                                                      kProjectionFormat);
        weights.attention.query_norm = bindings.parameter(prefix + "attention/query_norm",
                                                          {geometry.head_dim}, kNormFormat);
        weights.attention.key_norm = bindings.parameter(prefix + "attention/key_norm",
                                                        {geometry.head_dim}, kNormFormat);

        weights.mlp.gate = bindings.parameter(prefix + "mlp/gate", {intermediate, hidden}, kMlpFormat);
        weights.mlp.up   = bindings.parameter(prefix + "mlp/up", {intermediate, hidden}, kMlpFormat);
        weights.mlp.down = bindings.parameter(prefix + "mlp/down", {hidden, intermediate}, kMlpFormat);

        out.layers.push_back(weights);
    }
    return out;
}

} // namespace

struct LoadPlan::Impl {
    TextConfig config;
    LoadOptions options;
    ModelWeights weights;
    artifact::Binder binder;
    std::vector<artifact::ParameterReference> references;

    explicit Impl(const artifact::Reader& reader) : binder(reader) {}
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

LoadPlan::~LoadPlan()                              = default;
LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;

const TextConfig& LoadPlan::config() const { return impl_->config; }

const ModelWeights& LoadPlan::weights() const { return impl_->weights; }

std::size_t LoadPlan::parameter_count() const { return impl_->references.size(); }

const artifact::ParameterReference& LoadPlan::parameter(WeightId id) const {
    return impl_->references.at(id.index);
}

LoadPlan plan_load(const artifact::Reader& reader, LoadOptions options) {
    auto out     = std::make_unique<LoadPlan::Impl>(reader);
    out->options = options;
    if (options.purpose != EnginePurpose::Generation) {
        throw artifact::ArtifactError("Gemma 4 text loads for Generation in this build");
    }
    out->config = parse_text_config(reader.directory().component("text").config);
    Bindings bindings(out->binder);
    out->weights.text = bind_text(bindings, out->config);
    out->references.reserve(bindings.weights().size());
    for (const auto& pending : bindings.weights()) { out->references.push_back(pending.reference); }
    return LoadPlan(std::move(out));
}

} // namespace ninfer::models::gemma4
