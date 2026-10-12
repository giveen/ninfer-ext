#pragma once

// Gemma 4 text parameters, in the vocabulary the artifact binds.
//
// The stored names and shapes are recorded in docs/maintainer/gemma4-model.md §9 and were read
// from the artifact itself. Two structural facts follow from what is stored:
//
//  - The query, key and value parameters are slices of one fused projection object per layer, so
//    each carries the slice's own shape here (8192 x 5376 and so on) while the artifact resolves
//    the row offset and the scale plane.
//  - Global layers bind no value at all: attention_k_eq_v makes the weightless key normalization
//    serve as the value, so `AttentionWeights::value` stays unset there, which `mixer` tells a
//    consumer how to interpret.

#include "models/gemma4/config.h"

#include <cstddef>
#include <limits>
#include <optional>
#include <vector>

namespace ninfer::models::gemma4 {

struct WeightId {
    std::size_t index                          = std::numeric_limits<std::size_t>::max();
    friend bool operator==(WeightId, WeightId) = default;
};

struct AttentionWeights {
    WeightId query, key, value, output;
    WeightId query_norm, key_norm;
};

struct MlpWeights {
    // The gate and up projections are separate objects, not slices of one packed parent: a row
    // slice of an NVFP4 parent cannot re-derive the swizzled scale plane, so each half is bound
    // whole.
    WeightId gate, up, down;
};

// The routed-expert block of a 26B-A4B layer (model reference section 16): a router over the
// pre-FFN residual and one expert MLP per expert, in parallel with the dense `mlp` above, plus the
// three FFN norms the block adds. A 31B layer carries none of it, so this stays unset there.
struct MoeWeights {
    // [num_experts, hidden], over the weightless-normalized residual scaled by `router_scale` and
    // 1/sqrt(hidden); the top-k weights are renormalized before `per_expert_scale` is applied.
    WeightId router;
    WeightId router_scale;
    WeightId per_expert_scale;
    // One entry per expert. The checkpoint stores each bank fused -- gate and up in one
    // [experts, 2 * intermediate, hidden] tensor with the gate half first -- and the artifact
    // resolves an expert's slice of it.
    std::vector<MlpWeights> experts;
    // The dense branch's own post-norm, then the MoE branch's pre- and post-norms.
    WeightId post_feedforward_norm_1, pre_feedforward_norm_2, post_feedforward_norm_2;
};

struct LayerWeights {
    MixerKind mixer = MixerKind::SlidingAttention;
    // The sandwich norms: the input norm on the way in, then one on each sublayer's branch output,
    // with the scalar applied to the MLP side only.
    WeightId input_norm, post_attention_norm, pre_feedforward_norm, post_feedforward_norm;
    WeightId layer_scalar;
    AttentionWeights attention;
    MlpWeights mlp;
    // Set only on a model whose config enables the MoE block.
    std::optional<MoeWeights> moe;
};

struct TextWeights {
    WeightId token_embedding, output_head, final_norm;
    std::vector<LayerWeights> layers;
};

// One assistant drafter layer: a target layer's body without key and value projections.
struct DraftLayerWeights {
    MixerKind mixer = MixerKind::SlidingAttention;
    WeightId input_norm, post_attention_norm, pre_feedforward_norm, post_feedforward_norm;
    WeightId layer_scalar;
    WeightId query, query_norm, output;
    MlpWeights mlp;
};

struct DraftWeights {
    // [hidden, 2 * target hidden] over [target embedding; target hidden], and back to target width.
    WeightId pre_projection, post_projection;
    // The drafter's own tied head: full vocabulary, no soft cap.
    WeightId output_head, final_norm;
    std::vector<DraftLayerWeights> layers;
};

// One vision encoder layer: sandwich norms, q/k/v with per-head q and k norms (q and k rows stored
// in the RoPE Op's layout), and a GeGLU MLP. There is no layer scalar.
struct VisionLayerWeights {
    WeightId input_norm, post_attention_norm, pre_feedforward_norm, post_feedforward_norm;
    WeightId query, key, value, query_norm, key_norm, output;
    MlpWeights mlp;
};

struct VisionWeights {
    // [hidden, 3 * 16 * 16] over a patch's pixels in (row, column, channel) order.
    WeightId patch_embedding;
    // [2 * position_embedding_size, hidden]: the x table, then the y table.
    WeightId position_embedding;
    // FP32 [hidden] standardization of the pooled tokens.
    WeightId std_bias, std_scale;
    // [text hidden, hidden] from the weightless-normed soft tokens.
    WeightId embedding_projection;
    std::vector<VisionLayerWeights> layers;
};

struct ModelWeights {
    TextWeights text;
    // Bound only when the Engine runs MTP speculation.
    std::optional<DraftWeights> draft;
    // Bound only when the Engine accepts images.
    std::optional<VisionWeights> vision;
};

} // namespace ninfer::models::gemma4
