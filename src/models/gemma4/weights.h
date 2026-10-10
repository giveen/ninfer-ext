#pragma once

// Gemma 4 text parameters, in the vocabulary the artifact binds.
//
// The stored names and shapes are recorded in docs/maintainer/gemma4-model.md §9 and were read
// from the artifact itself. Two structural facts follow from what is stored:
//
//  - The query, key and value parameters are slices of one fused projection object per layer, so
//    each carries the slice's own shape here (8192 x 5376 and so on) while the artifact resolves the
//    row offset and the scale plane.
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
    // The gate and up projections are separate objects, not slices of one packed parent: a row slice
    // of an NVFP4 parent cannot re-derive the swizzled scale plane, so each half is bound whole.
    WeightId gate, up, down;
};

struct LayerWeights {
    MixerKind mixer = MixerKind::SlidingAttention;
    // The sandwich norms: the input norm on the way in, then one on each sublayer's branch output,
    // with the scalar applied to the MLP side only.
    WeightId input_norm, post_attention_norm, pre_feedforward_norm, post_feedforward_norm;
    WeightId layer_scalar;
    AttentionWeights attention;
    MlpWeights mlp;
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

struct ModelWeights {
    TextWeights text;
    // Bound only when the Engine runs MTP speculation.
    std::optional<DraftWeights> draft;
};

} // namespace ninfer::models::gemma4
