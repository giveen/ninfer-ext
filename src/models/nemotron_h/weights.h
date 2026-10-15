#pragma once

// Nemotron-H text parameters, in the vocabulary the artifact binds.
//
// The stored names are the checkpoint's own, read from the released BF16 checkpoint (see
// docs/maintainer/nemotron-bolt-plan.md): `mixer.*` inside a block whose kind decides which mixer
// is there. Three facts shape this header:
//
//  - The MoE bank varies per layer -- 240 to 496 experts at four intermediate widths around one
//    latent width -- so the bank is described per layer rather than once for the model, and its
//    two matrices are whole banks (expert-major) rather than one entry per expert: the Op reads
//    expert `e`'s rows in place at `[e*I, (e+1)*I)`, so an expert is not a separately bindable
//    parameter.
//  - The Mamba mixer's `in_proj` is one parent whose row ranges are z, x/B/C and dt in that
//    order, and the depthwise convolution's weight is one row per channel with the bias beside
//    it. They stay whole here for the same reason Gemma 4's gate/up halves do: a row range of a
//    quantized parent cannot re-derive its own scale plane, so a consumer slices the projection's
//    *output*, not the parent.
//  - `A_log`, `D`, `dt_bias` and the router's correction bias are stored in FP32 and are consumed
//    in FP32, because that is the precision the reference's own state arithmetic and selection
//    use. A BF16 copy would lose exactly what they are for.
//
// A `WeightId` addresses one bound parameter in the model's table (`Model::tensor` /
// `Model::matrix`). The ids are per layer, so a layer's parameters are what its kind says they are
// and nothing else; `Model::bind` refuses an id twice, and `forward_block` refuses a layer whose
// bound parameter set does not match its config's kind.

#include "models/nemotron_h/config.h"

#include <cstddef>
#include <limits>
#include <optional>
#include <vector>

namespace ninfer::models::nemotron_h {

struct WeightId {
    std::size_t index                          = std::numeric_limits<std::size_t>::max();
    friend bool operator==(WeightId, WeightId) = default;
};

// One Mamba-2 block's parameters. `in_proj` is [inner + conv_channels + heads, hidden], the
// convolution's weight is declared [conv_channels, conv_kernel] (depthwise, one row per channel)
// with its bias [conv_channels], the per-head state parameters are [heads], `norm` is the gated
// grouped norm's gain over the inner width, and `out_proj` is [hidden, inner].
//
// The convolution's weight carries one requirement the declaration cannot express, so it is
// written here instead: its *payload* must be in `ops::causal_conv1d_silu`'s own order, which is
// tap-major -- the tap `j` of channel `c` at `j * conv_channels + c`. That is not the checkpoint's
// order: `mixer.conv1d.weight` is `[conv_channels, 1, conv_kernel]`, whose flat order is
// `c * conv_kernel + j`, so binding it as it is stored gives a shape-valid tensor whose every tap
// is another channel's and a silently wrong model rather than a refusal. The transpose belongs to
// whoever binds the parameter, because it is a property of this Op's kernels (which index the taps
// as the outer axis and a channel as the inner one) rather than of the checkpoint.
struct MambaWeights {
    WeightId in_proj, out_proj;
    WeightId conv, conv_bias;
    WeightId a_log, d_skip, dt_bias;
    WeightId norm;
};

// One attention block's parameters: plain GQA, no bias, no sinks, no window.
struct AttentionWeights {
    WeightId query, key, value, output;
};

// One MoE block's parameters. `router` is [experts, hidden] and `correction_bias` is FP32
// [experts]; the two latent projections are ordinary linears around the bank, which is
// expert-major with `up` [experts * intermediate, latent] and `down` [experts * latent,
// intermediate]; the shared expert is a plain relu2 MLP at the *hidden* width, because the
// reference runs it on the mixer's input rather than in the latent space.
struct MoeWeights {
    WeightId router, correction_bias;
    WeightId fc1_latent, fc2_latent;
    WeightId shared_up, shared_down;
    WeightId experts_up, experts_down;
};

// A block's parameters: the block norm (a plain whole-row RMSNorm, not the gated grouped one
// inside a Mamba mixer, which lives with the mixer), then exactly one mixer's set.
struct LayerWeights {
    BlockKind kind = BlockKind::Mamba;
    WeightId block_norm;
    std::optional<MambaWeights> mamba;
    std::optional<AttentionWeights> attention;
    std::optional<MoeWeights> moe;
};

struct TextWeights {
    WeightId token_embedding, output_head, final_norm;
    std::vector<LayerWeights> layers;
};

struct ModelWeights {
    TextWeights text;
};

} // namespace ninfer::models::nemotron_h
