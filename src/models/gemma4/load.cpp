#include "models/gemma4/load.h"

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "artifact/views.h" // bind_view

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::models::gemma4 {
namespace {

// The formats the artifact stores, checked by the binder on every request.
constexpr QType kNormFormat       = QType::BF16;
constexpr QType kScalarFormat     = QType::FP32;

struct PendingWeight {
    artifact::ParameterReference reference;
};

class Bindings {
public:
    explicit Bindings(artifact::Binder& binder) : binder_(binder) {}

    // A null format declares no representation constraint, which is what a parameter whose layout the
    // converter is free to choose needs: the Ops dispatch on what the weight actually is.
    WeightId parameter(const std::string& name, artifact::Shape shape,
                       std::optional<QType> format          = {},
                       artifact::Residency residency        = artifact::Residency::Device) {
        PendingWeight pending;
        pending.reference = binder_.parameter(name, std::move(shape), residency, format);
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
    // The embedding's representation is the layout's choice, and it stays in pinned host memory: a
    // pass gathers one row per token through UVA, which costs nothing measurable beside the layers,
    // and the 262K-row table then takes no device memory, so a layout can keep it exact (BF16).
    out.token_embedding = bindings.parameter("text/token_embedding", {vocabulary, hidden}, {},
                                             artifact::Residency::HostPinned);
    out.output_head = bindings.parameter("text/output_head", {vocabulary, hidden});
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
            bindings.parameter(prefix + "attention/query", {query_rows, hidden});
        weights.attention.key =
            bindings.parameter(prefix + "attention/key", {key_rows, hidden});
        if (sliding) {
            weights.attention.value = bindings.parameter(prefix + "attention/value",
                                                         {key_rows, hidden});
        }
        // Projection formats are the precision layout's choice; the linear Op dispatches on them.
        weights.attention.output =
            bindings.parameter(prefix + "attention/output", {hidden, query_rows});
        weights.attention.query_norm = bindings.parameter(prefix + "attention/query_norm",
                                                          {geometry.head_dim}, kNormFormat);
        weights.attention.key_norm = bindings.parameter(prefix + "attention/key_norm",
                                                        {geometry.head_dim}, kNormFormat);

        weights.mlp.gate = bindings.parameter(prefix + "mlp/gate", {intermediate, hidden},
                                             std::optional<QType>{});
        weights.mlp.up   = bindings.parameter(prefix + "mlp/up", {intermediate, hidden}, std::optional<QType>{});
        weights.mlp.down = bindings.parameter(prefix + "mlp/down", {hidden, intermediate}, std::optional<QType>{});

        out.layers.push_back(weights);
    }
    return out;
}

} // namespace

struct LoadPlan::Impl {
    TextConfig config;
    TextResources resources;
    LoadOptions options;
    ModelWeights weights;
    artifact::Binder binder;
    std::vector<artifact::ParameterReference> references;
    artifact::MaterializationPlan plan;

    explicit Impl(const artifact::Reader& reader) : binder(reader) {}
};

// Every Gemma parameter is Device-resident, so unlike the Qwen loader there is no file-mapped branch
// here: a plan's reference always resolves to a view over the materialized bytes.
std::vector<BoundWeight> resolve_weights(const std::vector<artifact::ParameterReference>& references,
                                        const artifact::MaterializedArtifact& materialized) {
    std::vector<BoundWeight> out;
    out.reserve(references.size());
    for (const auto& reference : references) {
        if (reference.residency != artifact::Residency::Device &&
            reference.residency != artifact::Residency::HostPinned) {
            throw artifact::ArtifactError(reference.name +
                                          ": only device or pinned host parameters are bound");
        }
        BoundWeight bound;
        bound.name = reference.name;
        bound.view = artifact::bind_view(reference, materialized);
        out.push_back(std::move(bound));
    }
    return out;
}

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

LoadPlan::~LoadPlan()                              = default;
LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;

const TextConfig& LoadPlan::config() const { return impl_->config; }

const TextResources& LoadPlan::resources() const { return impl_->resources; }

const ModelWeights& LoadPlan::weights() const { return impl_->weights; }

const artifact::MaterializationPlan& LoadPlan::materialization() const {
    return impl_->plan;
}

std::size_t LoadPlan::parameter_count() const { return impl_->references.size(); }

const artifact::ParameterReference& LoadPlan::parameter(WeightId id) const {
    return impl_->references.at(id.index);
}

LoadPlan plan_load(const artifact::Reader& reader, LoadOptions options) {
    auto out     = std::make_unique<LoadPlan::Impl>(reader);
    out->options = options;
    // Both purposes bind the same parameters: the head the scoring route reads is the head generation
    // samples from, and what differs between the two is how a Program is driven, not what is loaded.
    out->config = parse_text_config(reader.directory().component("text").config);
    // The tokenizer's files are artifact resources rather than weights, and this is where the frontend
    // will read them. The Binder must still be alive, so this comes before it is finished.
    const auto resource = [&](std::string_view role) {
        const auto bytes = out->binder.host_object(out->binder.resource("text", role));
        return std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    };
    out->resources.tokenizer_json         = resource("tokenizer.json");
    out->resources.tokenizer_config_json  = resource("tokenizer_config.json");
    out->resources.generation_config_json = resource("generation_config.json");
    out->resources.chat_template_jinja    = resource("chat_template.jinja");
    Bindings bindings(out->binder);
    out->weights.text = bind_text(bindings, out->config);
    out->references.reserve(bindings.weights().size());
    for (const auto& pending : bindings.weights()) { out->references.push_back(pending.reference); }
    // The plan's physical demand is whatever Binding produced; finishing consumes the Binder, so it
    // must come after every parameter request.
    out->plan = std::move(out->binder).finish();
    return LoadPlan(std::move(out));
}

Model::Model(TextConfig config, ModelWeights weights, std::vector<BoundWeight> bound,
             std::vector<float> layer_scalars, artifact::MaterializedArtifact backing)
    : backing_(std::move(backing)), config_(std::move(config)), weights_(std::move(weights)),
      bound_(std::move(bound)), layer_scalars_(std::move(layer_scalars)) {}

Model::~Model() = default;

Tensor Model::tensor(WeightId id) const {
    const auto& bound = weight(id);
    const auto& view  = bound.view;
    if (view.shape.empty() || view.shape.size() > 4) {
        throw artifact::ArtifactError(bound.name + ": parameter has no tensor shape");
    }
    std::array<std::int32_t, 4> axes{1, 1, 1, 1};
    for (std::size_t i = 0; i < view.shape.size(); ++i) {
        axes[i] = static_cast<std::int32_t>(view.shape[view.shape.size() - 1 - i]);
    }
    return weight_tensor(view, {axes[0], axes[1], axes[2], axes[3]});
}

ops::WeightInput Model::input(WeightId id) const {
    // The activation policy the artifact declares per input is not bound yet, so every weight is
    // requested as A16-only, which is how the registered Gemma shapes are declared.
    return ops::WeightInput{weight(id).view, ops::LinearPolicy::A16Only, std::nullopt};
}

float Model::layer_scalar(WeightId id) const { return layer_scalars_.at(id.index); }

std::unique_ptr<Model> materialize_model(LoadPlan&& plan, DeviceContext& device,
                                        const StartupObserver* observer) {
    if (!plan.impl_) { throw artifact::ArtifactError("load plan was already consumed"); }
    auto data    = std::move(plan.impl_);
    auto backing = artifact::materialize(*data->plan.source, std::move(data->plan), device, observer);
    auto bound   = resolve_weights(data->references, backing);

    // The layer scalars are consumed as host floats, so read the sixty 4-byte values once here
    // rather than syncing a device copy on every layer of every token.
    std::vector<float> layer_scalars(data->references.size(), 1.0F);
    for (const auto& layer : data->weights.text.layers) {
        const auto& reference = data->references.at(layer.layer_scalar.index);
        if (reference.shape.size() != 1 || reference.shape[0] != 1) {
            throw artifact::ArtifactError(reference.name + ": layer scalar must be [1]");
        }
        const Tensor scalar = weight_tensor(bound.at(layer.layer_scalar.index).view, {1, 1, 1, 1});
        CUDA_CHECK(cudaMemcpy(&layer_scalars[layer.layer_scalar.index], scalar.data,
                              sizeof(float), cudaMemcpyDeviceToHost));
    }
    return std::unique_ptr<Model>(new Model(std::move(data->config), std::move(data->weights),
                                            std::move(bound), std::move(layer_scalars),
                                            std::move(backing)));
}

std::unique_ptr<Model> load_model(const std::filesystem::path& path, LoadOptions options,
                                  DeviceContext& device, const StartupObserver* observer) {
    artifact::Reader reader(path);
    return materialize_model(plan_load(reader, options), device, observer);
}

} // namespace ninfer::models::gemma4
