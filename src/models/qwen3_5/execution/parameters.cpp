#include "models/qwen3_5/execution/parameters.h"

#include "core/weight_view.h"
#include "models/qwen3_5/execution/probe_registry.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::qwen3_5::execution {
namespace {

template <class Function>
auto with_context(const std::string& context, Function&& function) {
    try {
        return function();
    } catch (const std::invalid_argument& error) {
        throw std::invalid_argument(context + ": " + error.what());
    }
}

// A fused parent prepared directly (attention qkv, GDN qkv/z, MLP gate/up) is not built through
// `linear()`, so it is registered here for the sensitivity probe.
void register_projection(const std::string& name, const ops::ProjectionWeights& projection) {
    if (const auto* single = std::get_if<ops::SingleProjectionWeight>(&projection)) {
        register_probe_target(name, single->weight);
    } else if (const auto* pair = std::get_if<ops::PairedProjectionWeights>(&projection)) {
        register_probe_target(name, pair->first);
        register_probe_target(name, pair->second);
    }
}

class Prepare {
public:
    explicit Prepare(const Model& model) : model_(model) {}

    LinearParameters linear(WeightId id) const {
        return with_context(model_.weight(id).name, [&] {
            auto prepared = ops::prepare_linear_weight(model_.input(id));
            register_probe_target(model_.weight(id).name, prepared.weight);
            return prepared;
        });
    }

    LinearParameters linear(WeightUseId id) const {
        return with_context(model_.weight(id.parameter).name,
                            [&] { return ops::prepare_linear_weight(model_.input(id)); });
    }

    Tensor tensor(WeightId id) const {
        const auto& bound = model_.weight(id);
        return with_context(bound.name, [&] {
            const auto& view = bound.view;
            if (view.shape.size() > 4) {
                throw std::invalid_argument("direct parameter exceeds Tensor rank");
            }
            std::array<std::int32_t, 4> axes{1, 1, 1, 1};
            for (std::size_t i = 0; i < view.shape.size(); ++i) {
                const auto extent = view.shape[view.shape.size() - 1 - i];
                if (extent > std::uint64_t(std::numeric_limits<std::int32_t>::max())) {
                    throw std::invalid_argument("direct parameter exceeds Tensor extent");
                }
                axes[i] = static_cast<std::int32_t>(extent);
            }
            return weight_tensor(view, {axes[0], axes[1], axes[2], axes[3]});
        });
    }

    DenseParameters dense(const DenseWeights& w) const {
        return {with_context(model_.weight(w.gate).name,
                             [&] {
                                 auto prepared = ops::prepare_linear_swiglu_weight(
                                     model_.input(w.gate), model_.input(w.up));
                                 register_probe_target(model_.weight(w.gate).name, prepared.weight);
                                 return prepared;
                             }),
                linear(w.down)};
    }

    FfnParameters ffn(const BlockWeights& w) const {
        if (const auto* d = std::get_if<DenseWeights>(&w.ffn)) { return dense(*d); }
        const auto& moe = std::get<MoeWeights>(w.ffn);
        if (std::get<MoeConfig>(model_.config().text.ffn).num_experts_per_tok != 8) {
            throw std::invalid_argument("SparseMoe implements top-8 routing");
        }
        std::vector<ops::WeightInput> gate_up, down;
        gate_up.reserve(2 * moe.experts.size());
        down.reserve(moe.experts.size());
        for (const auto& expert : moe.experts) {
            gate_up.push_back(model_.input(expert.gate));
            gate_up.push_back(model_.input(expert.up));
            down.push_back(model_.input(expert.down));
        }
        return with_context(model_.weight(moe.router).name, [&] {
            return ops::prepare_sparse_moe_weights(
                model_.input(moe.router), model_.input(moe.shared_score), gate_up, down,
                model_.input(moe.shared.gate), model_.input(moe.shared.up),
                model_.input(moe.shared.down));
        });
    }

    BlockParameters block(const BlockWeights& w) const {
        BlockParameters out;
        out.input_norm          = tensor(w.input_norm);
        out.post_attention_norm = tensor(w.post_attention_norm);
        out.ffn                 = ffn(w);
        if (const auto* a = std::get_if<AttentionWeights>(&w.mixer)) {
            auto projection = ops::prepare_attn_input_proj_weights(
                model_.input(a->query), model_.input(a->key), model_.input(a->gate),
                model_.input(a->value));
            register_projection(model_.weight(a->query).name, projection);
            out.mixer = AttentionParameters{std::move(projection), tensor(a->query_norm),
                                            tensor(a->key_norm), linear(a->output)};
            out.projection_prefetch =
                prefetch(std::get<AttentionParameters>(out.mixer).projection, a->query);
        } else {
            const auto& g = std::get<GdnWeights>(w.mixer);
            auto gdn_input = ops::prepare_gdn_input_proj_weights(
                model_.input(g.query), model_.input(g.key), model_.input(g.value), model_.input(g.z));
            register_projection(model_.weight(g.query).name, gdn_input);
            out.mixer     = GdnParameters{
                std::move(gdn_input),
                ops::prepare_gdn_gating_proj_weights(model_.input(g.a_projection),
                                                         model_.input(g.b_projection)),
                tensor(g.a_log),
                tensor(g.dt_bias),
                tensor(g.convolution),
                tensor(g.norm),
                linear(g.output)};
            out.projection_prefetch =
                prefetch(std::get<GdnParameters>(out.mixer).projection, g.query);
        }
        return out;
    }

    ops::SparseMoeHints prefetch(const ops::ProjectionWeights& projection, WeightId query) const {
        const auto* single = std::get_if<LinearParameters>(&projection);
        const auto& weight =
            single ? single->weight : std::get<ops::PairedProjectionWeights>(projection).first;
        const auto& geometry = model_.weight(query).view.parts.front().parent->geometry;
        const auto row_bytes = geometry.layout == QuantLayout::Contiguous
                                   ? std::uint64_t(weight.k) * dtype_size(DType::BF16)
                                   : geometry.code_bytes_per_row;
        return {weight.qdata, static_cast<std::size_t>(row_bytes * weight.n)};
    }

    MtpParameters mtp(const MtpWeights& w) const {
        const auto& a = std::get<AttentionWeights>(w.layer.mixer);
        const std::array inputs{model_.input(a.query), model_.input(a.key), model_.input(a.gate),
                                model_.input(a.value)};
        MtpParameters out;
        out.input_projection    = linear(w.input_projection);
        out.embedding_norm      = tensor(w.embedding_norm);
        out.hidden_norm         = tensor(w.hidden_norm);
        out.input_norm          = tensor(w.layer.input_norm);
        out.post_attention_norm = tensor(w.layer.post_attention_norm);
        out.final_norm          = tensor(w.final_norm);
        out.projection.packed = ops::prepare_linear_weight(inputs);
        // The dense incremental path projects K/V and Q/gate independently through ops::linear_pair,
        // which only implements Q8. Every other format runs the complete packed parent through
        // ops::attn_input_proj instead, so it leaves `rows` empty. Ask the packed parent, not an
        // individual projection: a row slice of an EXL3 parent is not a valid native Weight.
        if (model_.config().text.architecture == Architecture::Qwen3_5 &&
            out.projection.packed.weight.qtype == QType::Q8_G32_FP16) {
            out.projection.rows = {linear(a.query), linear(a.key), linear(a.gate), linear(a.value)};
        }
        out.query_norm  = tensor(a.query_norm);
        out.key_norm    = tensor(a.key_norm);
        out.output      = linear(a.output);
        out.ffn         = ffn(w.layer);
        out.output_head = linear(w.output_head_use);
        return out;
    }

    NormParameters norm(const NormWeights& w) const { return {tensor(w.weight), tensor(w.bias)}; }

    std::optional<Tensor> joined_bias(const std::array<WeightId, 3>& ids) const {
        WeightView view;
        std::uint64_t count = 0;
        for (const auto id : ids) {
            const auto& input = model_.weight(id).view;
            count += weight_element_count(input.shape);
            for (const auto& part : input.parts) {
                if (!view.parts.empty() && (view.parts.back().parent != part.parent ||
                                            view.parts.back().end != part.begin)) {
                    return std::nullopt;
                }
                view.parts.push_back(part);
            }
        }
        if (count > std::uint64_t(std::numeric_limits<std::int32_t>::max())) {
            throw std::invalid_argument("Vision bias exceeds Tensor extent");
        }
        view.shape = {count};
        return weight_tensor(view, {static_cast<std::int32_t>(count)});
    }

    VisionParameters vision(const VisionWeights& w) const {
        VisionParameters out;
        out.patch_embedding      = linear(w.patch_embedding);
        out.patch_embedding_bias = tensor(w.patch_embedding_bias);
        out.position_embedding   = tensor(w.position_embedding);
        out.layers.reserve(w.layers.size());
        for (std::size_t i = 0; i < w.layers.size(); ++i) {
            out.layers.push_back(with_context("vision/layers/" + std::to_string(i), [&] {
                const auto& layer = w.layers[i];
                const std::array qkv{model_.input(layer.query), model_.input(layer.key),
                                     model_.input(layer.value)};
                const std::array ids{layer.query_bias, layer.key_bias, layer.value_bias};
                const auto bias = joined_bias(ids);
                if (!bias) {
                    throw std::invalid_argument(
                        "Vision QKV bias: this fixed call requires a contiguous bias bank");
                }
                return VisionBlockParameters{norm(layer.norm1),
                                             norm(layer.norm2),
                                             ops::prepare_linear_weight(qkv),
                                             *bias,
                                             linear(layer.output),
                                             linear(layer.fc1),
                                             linear(layer.fc2),
                                             tensor(layer.output_bias),
                                             tensor(layer.fc1_bias),
                                             tensor(layer.fc2_bias)};
            }));
        }
        out.merger_norm     = norm(w.merger_norm);
        out.merger_fc1      = linear(w.merger_fc1);
        out.merger_fc2      = linear(w.merger_fc2);
        out.merger_fc1_bias = tensor(w.merger_fc1_bias);
        out.merger_fc2_bias = tensor(w.merger_fc2_bias);
        return out;
    }

    DynamicConvParameters convolution(const DynamicConvWeights& w) const {
        return {tensor(w.base_kernel), linear(w.kernel_projection)};
    }

    DraftParameters draft(const DraftWeights& w) const {
        DraftParameters out;
        out.feature_projection = linear(w.feature_projection);
        out.context_norm       = tensor(w.context_norm);
        out.final_norm         = tensor(w.final_norm);
        out.output_head        = linear(w.output_head_use);
        out.layers.reserve(w.layers.size());
        for (std::size_t i = 0; i < w.layers.size(); ++i) {
            out.layers.push_back(with_context(
                std::string(model_.options().speculative_component()) + "/layers/" +
                    std::to_string(i),
                [&] {
                    const auto& layer = w.layers[i];
                    const auto& a     = layer.attention;
                    DraftBlockParameters result;
                    result.input_norm          = tensor(layer.input_norm);
                    result.post_attention_norm = tensor(layer.post_attention_norm);
                    result.query_key_value     = ops::prepare_attn_input_proj_weights(
                        model_.input(a.query), model_.input(a.key), model_.input(a.value));
                    result.context_key   = linear(a.context_key);
                    result.context_value = linear(a.context_value);
                    result.query_norm    = tensor(a.query_norm);
                    result.key_norm      = tensor(a.key_norm);
                    result.output        = linear(a.output);
                    result.mlp           = dense(layer.mlp);
                    if (layer.attention_conv) {
                        result.attention_conv = convolution(*layer.attention_conv);
                    }
                    if (layer.mlp_conv) { result.mlp_conv = convolution(*layer.mlp_conv); }
                    return result;
                }));
        }
        if (w.selector) {
            out.selector = SelectorParameters{linear(w.selector->hidden_projection),
                                              tensor(w.selector->predecessor_codebook),
                                              tensor(w.selector->successor_codebook)};
        }
        return out;
    }

    // Rows of several direct BF16 parameters that sit consecutively in one parent.
    Tensor joined_matrix(std::initializer_list<WeightId> ids) const {
        WeightView view;
        std::uint64_t rows = 0, columns = 0;
        for (const auto id : ids) {
            const auto& input = model_.weight(id).view;
            if (input.shape.size() != 2 || (columns != 0 && input.shape[1] != columns)) {
                throw std::invalid_argument("joined matrix rows must share one width");
            }
            columns = input.shape[1];
            rows += input.shape[0];
            for (const auto& part : input.parts) {
                if (!view.parts.empty() && (view.parts.back().parent != part.parent ||
                                            view.parts.back().end != part.begin)) {
                    throw std::invalid_argument("joined matrix is not one contiguous region");
                }
                view.parts.push_back(part);
            }
        }
        view.shape = {rows, columns};
        return weight_tensor(view, {dimension(columns), dimension(rows)});
    }

    HyperConnectionParameters hyper_connection(const HyperConnectionWeights& w) const {
        HyperConnectionParameters out;
        out.norm = tensor(w.norm);
        if (w.inject) {
            out.down = ops::prepare_linear_weight(
                std::array{model_.input(w.down), model_.input(*w.inject)});
            out.inject_rows = dimension(model_.weight(*w.inject).view.shape[0]);
        } else {
            out.down = linear(w.down);
        }
        out.up = linear(w.up);
        return out;
    }

    ops::ExpertWeights expert_bank(const MoeWeights& moe) const {
        const auto& config = model_.config().text;
        const auto& geo    = std::get<MoeConfig>(config.ffn);
        if (geo.num_experts != ops::kOffloadMoeExperts ||
            geo.num_experts_per_tok != ops::kOffloadMoeTopK ||
            geo.moe_intermediate_size != ops::kOffloadMoeIntermediate ||
            config.hidden_size != ops::kOffloadMoeHidden) {
            throw std::invalid_argument("offloaded MoE implements 2560/512/top-10/640 geometry");
        }
        const auto single = [&](WeightId id) -> const WeightRegion& {
            const auto& parts = model_.weight(id).view.parts;
            if (parts.size() != 1) {
                throw std::invalid_argument("a routed expert matrix must be one bank region");
            }
            return parts.front();
        };
        const WeightParent* gate_up = single(moe.experts.front().gate).parent;
        const WeightParent* down    = single(moe.experts.front().down).parent;
        const std::uint64_t h = config.hidden_size, ir = geo.moe_intermediate_size;
        for (std::uint32_t e = 0; e < geo.num_experts; ++e) {
            const auto& expert = moe.experts[e];
            const auto& g      = single(expert.gate);
            const auto& u      = single(expert.up);
            const auto& d      = single(expert.down);
            if (g.parent != gate_up || u.parent != gate_up || d.parent != down ||
                g.begin != (2 * e) * ir * h || u.begin != (2 * e + 1) * ir * h ||
                d.begin != e * h * ir) {
                throw std::invalid_argument("routed experts are not stored expert-major");
            }
        }
        for (const auto* parent : {gate_up, down}) {
            if (parent->geometry.format != QType::NVFP4 ||
                parent->geometry.layout != QuantLayout::BlockScaleK16M128x4) {
                throw std::invalid_argument("offloaded experts must be block-scaled NVFP4 banks");
            }
        }
        ops::ExpertWeights bank;
        bank.base[0]   = gate_up->data;
        bank.base[1]   = gate_up->data + gate_up->geometry.scale_offset;
        bank.base[2]   = down->data;
        bank.base[3]   = down->data + down->geometry.scale_offset;
        bank.stride[0] = ops::kExpertGateUpCodeBytes;
        bank.stride[1] = ops::kExpertGateUpScaleBytes;
        bank.stride[2] = ops::kExpertDownCodeBytes;
        bank.stride[3] = ops::kExpertDownScaleBytes;
        bank.gate_up_divisors =
            reinterpret_cast<const float*>(gate_up->data + gate_up->geometry.divisor_offset);
        bank.gate_up_divisor_rows =
            dimension(gate_up->geometry.shape[0] / gate_up->geometry.divisor_count);
        bank.down_divisors =
            reinterpret_cast<const float*>(down->data + down->geometry.divisor_offset);
        bank.down_divisor_rows = dimension(down->geometry.shape[0] / down->geometry.divisor_count);
        // The A4 route quantises a chunk once and shares it with every expert of the layer, so it
        // needs every expert to permit A4 with one common divisor per projection input; anything
        // else keeps the layer on the A16 route.
        const auto common_divisor = [&](auto projections) -> float {
            std::optional<float> common;
            for (const auto& expert : moe.experts) {
                for (const WeightId id : projections(expert)) {
                    const ops::WeightInput input = model_.input(id);
                    if (input.policy != ops::LinearPolicy::AllowA4 ||
                        !input.activation_input_divisor ||
                        (common && *common != *input.activation_input_divisor)) {
                        return 0.0F;
                    }
                    common = input.activation_input_divisor;
                }
            }
            return common.value_or(0.0F);
        };
        const float gate_up_input =
            common_divisor([](const auto& expert) { return std::array{expert.gate, expert.up}; });
        const float down_input =
            common_divisor([](const auto& expert) { return std::array{expert.down}; });
        if (gate_up_input > 0.0F && down_input > 0.0F) {
            bank.gate_up_input_divisor = gate_up_input;
            bank.down_input_divisor    = down_input;
        }
        return bank;
    }

    OffloadMoeParameters offload_moe(const MoeWeights& w) const {
        return {joined_matrix({w.router, w.shared_score}),
                ops::prepare_linear_weight(
                    std::array{model_.input(w.shared.gate), model_.input(w.shared.up)}),
                linear(w.shared.down), expert_bank(w)};
    }

    PleParameters ple(const PleWeights& w) const {
        const auto& table = model_.weight(w.table);
        if (table.mapped.empty() || table.mapped_geometry.format != QType::FP8_E4M3FN_ROW_BF16) {
            throw std::invalid_argument("PLE table must be a file-mapped row-scaled FP8 matrix");
        }
        PleParameters out;
        out.table =
            PleTable{table.mapped, table.view.shape[0], dimension(table.view.shape[1]),
                     table.mapped_geometry.code_bytes_per_row, table.mapped_geometry.scale_offset};
        out.key_value =
            ops::prepare_linear_weight(std::array{model_.input(w.key), model_.input(w.value)});
        out.key_norm    = tensor(w.key_norm);
        out.query_norm  = tensor(w.query_norm);
        out.conv_norm   = tensor(w.conv_norm);
        out.convolution = tensor(w.convolution);
        return out;
    }

    Qwen4BlockParameters qwen4_block(const Qwen4BlockWeights& w) const {
        Qwen4BlockParameters out;
        out.attention_hc = hyper_connection(w.attention_hc);
        out.ffn_hc       = hyper_connection(w.ffn_hc);
        if (const auto* q = std::get_if<QsaWeights>(&w.mixer)) {
            const auto& a = q->attention;
            out.mixer     = QsaParameters{
                ops::prepare_linear_weight(std::array{model_.input(a.query), model_.input(a.key),
                                                      model_.input(a.gate), model_.input(a.value)}),
                tensor(a.query_norm),
                tensor(a.key_norm),
                linear(a.output),
                ops::prepare_linear_weight(
                    std::array{model_.input(q->indexer_query), model_.input(q->indexer_key)}),
                tensor(q->indexer_query_norm),
                tensor(q->indexer_key_norm)};
        } else {
            const auto& g = std::get<GdnWeights>(w.mixer);
            out.mixer     = Qwen4GdnParameters{
                ops::prepare_linear_weight(std::array{model_.input(g.query), model_.input(g.key),
                                                      model_.input(g.value), model_.input(g.z)}),
                ops::prepare_linear_weight(
                    std::array{model_.input(g.a_projection), model_.input(g.b_projection)}),
                tensor(g.a_log),
                tensor(g.dt_bias),
                tensor(g.convolution),
                tensor(g.norm),
                linear(g.output)};
        }
        out.moe = offload_moe(w.moe);
        if (w.ple) { out.ple = ple(*w.ple); }
        return out;
    }

private:
    const Model& model_;
};

} // namespace

Parameters::Parameters(const Model& source) : model(source) {
    const Prepare prepare(model);
    const auto& w        = model.weights();
    text.token_embedding = native_weight(model.weight(w.text.token_embedding).view);
    text.output_head     = prepare.linear(w.text.output_head_use);
    if (w.qwen4) {
        Qwen4Parameters q;
        q.head =
            with_context("text/hc_head", [&] { return prepare.hyper_connection(w.qwen4->head); });
        q.layers.reserve(w.qwen4->layers.size());
        for (std::size_t i = 0; i < w.qwen4->layers.size(); ++i) {
            q.layers.push_back(with_context("text/layers/" + std::to_string(i), [&] {
                return prepare.qwen4_block(w.qwen4->layers[i]);
            }));
        }
        qwen4 = std::move(q);
        if (w.qwen4_mtp) {
            qwen4_mtp = with_context("mtp", [&] {
                const auto& m = *w.qwen4_mtp;
                return Qwen4MtpParameters{
                    prepare.tensor(m.embedding_norm),       prepare.tensor(m.hidden_norm),
                    prepare.linear(m.embedding_projection), prepare.linear(m.hidden_projection),
                    prepare.hyper_connection(m.head),       prepare.qwen4_block(m.layer)};
            });
        }
        if (w.vision) {
            vision = with_context("vision", [&] { return prepare.vision(*w.vision); });
        }
        return;
    }
    text.final_norm = prepare.tensor(w.text.final_norm);
    text.layers.reserve(w.text.layers.size());
    for (std::size_t i = 0; i < w.text.layers.size(); ++i) {
        text.layers.push_back(with_context("text/layers/" + std::to_string(i),
                                           [&] { return prepare.block(w.text.layers[i]); }));
    }
    if (w.mtp) {
        mtp = with_context("mtp", [&] { return prepare.mtp(*w.mtp); });
    }
    if (w.vision) {
        vision = with_context("vision", [&] { return prepare.vision(*w.vision); });
    }
    if (w.draft) {
        draft = with_context(std::string(model.options().speculative_component()),
                             [&] { return prepare.draft(*w.draft); });
    }
    if (w.proposal) {
        proposal =
            ProposalParameters{prepare.linear(w.proposal->head), std::nullopt, w.proposal->rows};
        if (w.proposal->token_ids) { proposal->token_ids = prepare.tensor(*w.proposal->token_ids); }
    }
}

} // namespace ninfer::models::qwen3_5::execution
