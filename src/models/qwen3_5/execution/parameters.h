#pragma once

#include "models/qwen3_5/model.h"
#include "ninfer/ops/offload_moe.h"
#include "ninfer/ops/weight_input.h"

#include <array>
#include <memory>
#include <limits>
#include <optional>
#include <stdexcept>
#include <variant>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

using LinearParameters = ops::SingleProjectionWeight;

[[nodiscard]] inline std::int32_t dimension(std::uint64_t value) {
    if (value > std::uint64_t(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("model dimension exceeds the Tensor integer domain");
    }
    return static_cast<std::int32_t>(value);
}

struct DenseParameters {
    LinearParameters gate_up;
    LinearParameters down;
};

using FfnParameters = std::variant<DenseParameters, ops::SparseMoeWeights>;

struct AttentionParameters {
    ops::ProjectionWeights projection;
    Tensor query_norm, key_norm;
    LinearParameters output;
};

struct GdnParameters {
    ops::ProjectionWeights projection;
    ops::ProjectionWeights control;
    Tensor a_log, dt_bias, convolution, norm;
    LinearParameters output;
};

struct BlockParameters {
    Tensor input_norm, post_attention_norm;
    std::variant<AttentionParameters, GdnParameters> mixer;
    FfnParameters ffn;
    ops::SparseMoeHints projection_prefetch;
};

struct TextParameters {
    Weight token_embedding;
    LinearParameters output_head;
    Tensor final_norm;
    std::vector<BlockParameters> layers;
};

struct MtpProjectionParameters {
    LinearParameters packed;
    // Dense MTP projects K/V and Q/gate independently in its incremental path.
    // MoE MTP uses its existing complete-parent Attention projection.
    std::optional<std::array<LinearParameters, 4>> rows;
};

struct MtpParameters {
    LinearParameters input_projection;
    Tensor embedding_norm, hidden_norm, input_norm, post_attention_norm, final_norm;
    MtpProjectionParameters projection;
    Tensor query_norm, key_norm;
    LinearParameters output;
    FfnParameters ffn;
    LinearParameters output_head;
};

struct NormParameters {
    Tensor weight, bias;
};

struct VisionBlockParameters {
    NormParameters norm1, norm2;
    LinearParameters qkv;
    Tensor qkv_bias;
    LinearParameters output, fc1, fc2;
    Tensor output_bias, fc1_bias, fc2_bias;
};

struct VisionParameters {
    LinearParameters patch_embedding;
    Tensor patch_embedding_bias, position_embedding;
    std::vector<VisionBlockParameters> layers;
    NormParameters merger_norm;
    LinearParameters merger_fc1, merger_fc2;
    Tensor merger_fc1_bias, merger_fc2_bias;
};

struct DynamicConvParameters {
    Tensor base_kernel;
    LinearParameters kernel_projection;
};

struct DraftBlockParameters {
    Tensor input_norm, post_attention_norm;
    LinearParameters query_key_value, context_key, context_value;
    Tensor query_norm, key_norm;
    LinearParameters output;
    DenseParameters mlp;
    std::optional<DynamicConvParameters> attention_conv, mlp_conv;
};

struct SelectorParameters {
    LinearParameters hidden_projection;
    Tensor predecessor_codebook, successor_codebook;
};

struct DraftParameters {
    LinearParameters feature_projection;
    Tensor context_norm, final_norm;
    std::vector<DraftBlockParameters> layers;
    std::optional<SelectorParameters> selector;
    LinearParameters output_head;
};

struct ProposalParameters {
    LinearParameters head;
    std::optional<Tensor> token_ids;
    std::uint32_t rows = 0;
};

struct HyperConnectionParameters {
    Tensor norm;
    LinearParameters down; // low-rank rows, then inject rows when inject_rows > 0
    LinearParameters up;
    std::int32_t inject_rows = 0;
};

struct QsaParameters {
    LinearParameters projection; // query, key, gate, value rows
    Tensor query_norm, key_norm;
    LinearParameters output;
    LinearParameters indexer; // indexer query rows, then the raw key row block
    Tensor indexer_query_norm, indexer_key_norm;
};

struct Qwen4GdnParameters {
    LinearParameters projection; // query, key, value, z rows
    LinearParameters control;    // a, b rows
    Tensor a_log, dt_bias, convolution, norm;
    LinearParameters output;
};

struct OffloadMoeParameters {
    Tensor router; // BF16 [513, H]: routed rows then the shared-expert score
    LinearParameters shared_gate_up;
    LinearParameters shared_down;
    ops::ExpertWeights bank; // host-pinned NVFP4 routed banks
};

// Host-mapped row-scaled FP8 n-gram table of one PLE layer.
struct PleTable {
    std::vector<artifact::MappedObjectSegment> segments;
    std::uint64_t rows        = 0;
    std::int32_t width        = 0;
    std::uint64_t scale_plane = 0; // object offset of the BF16 row multipliers
};

struct PleParameters {
    PleTable table;
    LinearParameters key_value; // key rows [S*H], then value rows [H]
    Tensor key_norm, query_norm, conv_norm, convolution;
};

struct Qwen4BlockParameters {
    HyperConnectionParameters attention_hc, ffn_hc;
    std::variant<QsaParameters, Qwen4GdnParameters> mixer;
    OffloadMoeParameters moe;
    std::optional<PleParameters> ple;
};

struct Qwen4Parameters {
    HyperConnectionParameters head;
    std::vector<Qwen4BlockParameters> layers;
};

struct Qwen4MtpParameters {
    Tensor embedding_norm, hidden_norm;
    LinearParameters embedding_projection, hidden_projection;
    HyperConnectionParameters head;
    Qwen4BlockParameters layer;
};

// Cold native preparation for the fixed model implementation. This owner is stable before
// startup sizing, execution, or Graph capture; all weight addresses borrow the source Model.
// Shape-dependent kernel selection and scratch remain with the calling implementation and Op.
class Parameters {
public:
    explicit Parameters(const Model& source);
    Parameters(const Parameters&)            = delete;
    Parameters& operator=(const Parameters&) = delete;
    Parameters(Parameters&&)                 = delete;
    Parameters& operator=(Parameters&&)      = delete;

    const Model& model;
    TextParameters text;
    std::optional<MtpParameters> mtp;
    std::optional<VisionParameters> vision;
    std::optional<DraftParameters> draft;
    std::optional<ProposalParameters> proposal;
    std::optional<Qwen4Parameters> qwen4;
    std::optional<Qwen4MtpParameters> qwen4_mtp;
};

} // namespace ninfer::models::qwen3_5::execution
