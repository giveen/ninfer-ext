#include "models/gemma4/forward.h"

#include "core/device.h"
#include "models/gemma4/cache.h"
#include "ninfer/ops/causal_compact_attention.h"
#include "ninfer/ops/compact_kv_rows.h"
#include "ninfer/ops/compact_kv_rows_quantize.h"
#include "ninfer/ops/dense_kv_ring.h"
#include "ninfer/ops/gelu_mul.h"
#include "ninfer/ops/gemma_moe_experts.h"
#include "ninfer/ops/gemma_moe_router.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/mul_scalar.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/residual_add_norm.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/scale_columns.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/sliding_causal_attention.h"
#include "ninfer/ops/soft_cap.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/weight_input.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::models::gemma4 {
namespace {

// The MoE block's storage: the router's selection and weights, and one workspace the router's
// logits and then the expert apply's products use in turn -- the selection is the logits' last
// reader and runs before the experts start.
struct MoeScratch {
    Tensor ids;       // I32 [K,T]  one chosen expert per row
    Tensor weights;   // FP32 [K,T] the router's weights, renormalized and expert-scaled
    Tensor workspace; // U8
};

ops::GemmaMoeRouterGeometry router_geometry(const TextConfig& config) {
    return {static_cast<std::int32_t>(config.hidden_size),
            static_cast<std::int32_t>(config.num_experts),
            static_cast<std::int32_t>(config.top_k_experts), config.rms_norm_eps};
}

ops::GemmaMoeExpertGeometry expert_geometry(const TextConfig& config) {
    return {static_cast<std::int32_t>(config.hidden_size),
            static_cast<std::int32_t>(config.num_experts),
            static_cast<std::int32_t>(config.top_k_experts),
            static_cast<std::int32_t>(config.moe_intermediate_size)};
}

// The expert apply's share is the largest any bank format takes at this width, so the layer bound
// does not have to know the artifact: the per-token route reads none of it.
std::size_t moe_workspace_bytes(const TextConfig& config, std::int32_t tokens) {
    const auto experts = [&](QType format) {
        return ops::gemma_moe_experts_workspace_bytes(expert_geometry(config), format, tokens);
    };
    return std::max({ops::gemma_moe_router_workspace_bytes(router_geometry(config), tokens),
                     experts(QType::NVFP4), experts(QType::Q4_G32_FP16_ROWS)});
}

MoeScratch allocate_moe_scratch(const TextConfig& config, std::int32_t tokens, DeviceArena& arena) {
    const std::int32_t k = static_cast<std::int32_t>(config.top_k_experts);
    MoeScratch moe;
    moe.ids     = arena.alloc(DType::I32, {k, tokens});
    moe.weights = arena.alloc(DType::FP32, {k, tokens});
    moe.workspace =
        arena.alloc(DType::U8, {static_cast<std::int32_t>(moe_workspace_bytes(config, tokens))});
    return moe;
}

struct Scratch {
    Tensor normed;    // [H,T]     the normed layer input, reused by both sublayers
    Tensor branch;    // [H,T]     a sublayer's normalized output before the residual add
    Tensor q;         // [Hq*D,T]  raw query projection
    Tensor k;         // [Hkv*D,T]
    Tensor v;         // [Hkv*D,T]
    Tensor qkv;       // the one frame q, k and v are row views of
    Tensor q_heads;   // [D,Hq,T]  the same storage, as heads for the norms and RoPE
    Tensor k_heads;   // [D,Hkv,T]
    Tensor v_heads;   // [D,Hkv,T]
    Tensor q_normed;  // [D,Hq,T]
    Tensor k_normed;  // [D,Hkv,T]
    Tensor v_normed;  // [D,Hkv,T]
    Tensor attended;  // [Hq*D,T]
    Tensor projected; // [H,T]     the attention output projection
    Tensor gate;      // [I,T]     the MLP's gate projection
    Tensor up;        // [I,T]     the MLP's up projection
    Tensor gate_up;   // the one frame gate and up are row views of when decode fuses them
    Tensor activated; // [I,T]
    Tensor down;      // [H,T]
    // The MoE block's storage, allocated only when the model enables the block.
    MoeScratch moe;
};

// I32 [n] as BF16 [2,n], so `scatter`'s exact column copy moves whole 32-bit positions.
Tensor positions_as_pairs(const Tensor& positions) {
    return Tensor(positions.data, DType::BF16, {2, static_cast<std::int32_t>(positions.ne[0])});
}

// Writes the `kept` trailing tokens' positions (I32 [tokens]) into the cache slots `slots` names,
// which is the half of a ring write a quantized ring keeps in common with the BF16 one: the rows
// themselves take that ring's own encoder.
void write_cache_positions(const Tensor& positions, const Tensor& slots, std::int32_t kept,
                           const Tensor& cache_positions, cudaStream_t stream) {
    const std::int32_t tokens   = static_cast<std::int32_t>(positions.ne[0]);
    const std::int32_t skip     = tokens - kept;
    Tensor position_destination = positions_as_pairs(cache_positions);
    ops::scatter(positions_as_pairs(positions).slice(1, skip, kept), slots.slice(0, skip, kept),
                 position_destination, stream);
}

// Writes the `kept` trailing tokens' rows of `rows` (BF16 [width,tokens]) and their `positions`
// (I32 [tokens]) into a cache layer whose slot for each token `slots` names: rows into `cache_rows`
// (BF16 [width,slots]), positions into `cache_positions` (I32 [slots]). Every index is on the
// device.
void write_cache_rows(const Tensor& rows, const Tensor& positions, const Tensor& slots,
                      std::int32_t kept, const Tensor& cache_rows, const Tensor& cache_positions,
                      cudaStream_t stream) {
    const std::int32_t tokens = static_cast<std::int32_t>(rows.ne[1]);
    const std::int32_t skip   = tokens - kept;
    Tensor destination        = cache_rows;
    ops::scatter(rows.slice(1, skip, kept), slots.slice(0, skip, kept), destination, stream);
    write_cache_positions(positions, slots, kept, cache_positions, stream);
}

// The first byte of column `column` of a column-major tensor whose columns are `column_bytes`
// apart.
std::uint8_t* column_data(const Tensor& tensor, std::size_t column_bytes, std::int32_t column) {
    return static_cast<std::uint8_t*>(tensor.data) +
           static_cast<std::size_t>(column) * column_bytes;
}

// A pass's segments must tile its columns in order, each over one lane's cache.
void check_segments(std::span<const PassSegment> segments, std::int32_t columns,
                    const char* where) {
    std::int32_t next = 0;
    for (const PassSegment& segment : segments) {
        if (segment.cache == nullptr || segment.tokens < 1 || segment.column != next ||
            segment.first_position < 0) {
            throw std::invalid_argument(
                std::string(where) +
                ": segments must tile the columns in order, each with a cache");
        }
        next += segment.tokens;
    }
    if (segments.empty() || next != columns) {
        throw std::invalid_argument(std::string(where) + ": segments must cover every column");
    }
}

// The key-norm weight of a target global layer, which a query attending its compact rows carries.
WeightId target_layer_key_norm(const Model& model, std::size_t layer) {
    return model.weights().text.layers.at(layer).attention.key_norm;
}

// A group of projections the converter stores as one packed parent is bound as row slices of it
// (tools/convert/gemma4.py `group(...)` for the attention inputs, and for a layer's gate and up).
// When those slices are contiguous from the object's first row, one Linear over the union runs
// exactly the rows the separate launches run, through the same route: every linear route fixes an
// output row's reduction order from that row alone, never from the block that happens to cover it.
// A Q6 GEMV splits a row's K range from K and the split index alone; the int4 row-group GEMV gives
// a warp whole rows and reduces each with its own per-lane serial FMA chain and one fixed log-step
// shuffle; and the int4 row-group MMA tile accumulates each output element in one thread over the
// tile's K blocks. So the union's results are bit-identical to the separate launches' -- the
// generated token ids of a greedy decode match the unfused build exactly -- and it is free to trade
// launches for one.
//
// The check below is by plane offset and covers the two layouts a grouped parent is written in: the
// 31B's row-split Q6 and the 26B-A4B's row-grouped int4 rows. Any other format, a differing
// quantisation, or a slice that is not exactly where it should be refuses the union, so a wrong
// width here can only decline to fuse, never fuse wrongly. `prepare_linear_weight` returns a copy
// of the stored weight, so the addresses compared here are the artifact's own slice addresses.
bool row_slices_are_contiguous(const Weight* const* members, int count) {
    if (count < 2) { return false; }
    const Weight& first = *members[0];
    // Each layout's per-row plane widths, so a member's planes have to begin exactly where the
    // previous member's rows end: the row split advances a row's group index by padded_k/64 in a
    // 32-byte code plane, a 16-byte high plane and a 2-byte scale plane (q6_a16_simt.cuh), and a
    // row-grouped int4 row carries no padding at all -- K/2 code bytes and K/32 binary16 scale
    // words (docs/maintainer/tensor-formats.md section 3.4.1).
    std::uint64_t code_per_row  = 0;
    std::uint64_t high_per_row  = 0;
    std::uint64_t scale_per_row = 0;
    if (first.layout == QuantLayout::RowSplit && first.qtype == QType::Q6_G64_FP16) {
        const auto groups = static_cast<std::uint64_t>(first.padded_shape[1]) / 64;
        code_per_row      = groups * 32;
        high_per_row      = groups * 16;
        scale_per_row     = groups * 2;
    } else if (first.layout == QuantLayout::RowGroup && first.qtype == QType::Q4_G32_FP16_ROWS) {
        code_per_row  = static_cast<std::uint64_t>(first.k) / 2;
        scale_per_row = static_cast<std::uint64_t>(first.k) / 32 * 2;
    } else {
        return false;
    }
    for (int index = 0; index < count; ++index) {
        const Weight& member = *members[index];
        if (member.qtype != first.qtype || member.layout != first.layout ||
            member.group != first.group || member.scale_dtype != first.scale_dtype ||
            member.k != first.k || member.padded_shape[1] != first.padded_shape[1] ||
            member.weight_scale_divisor != first.weight_scale_divisor ||
            member.input_scale_divisor != first.input_scale_divisor) {
            return false;
        }
        if (member.qdata == nullptr || member.scales == nullptr) { return false; }
        if (high_per_row != 0 && member.qhigh == nullptr) { return false; }
    }
    const auto is_at = [](const void* base, std::uint64_t rows_ahead, std::uint64_t per_row,
                          const void* expected) {
        return static_cast<const void*>(static_cast<const std::uint8_t*>(base) +
                                        rows_ahead * per_row) == expected;
    };
    for (int index = 1; index < count; ++index) {
        const Weight& previous   = *members[index - 1];
        const Weight& member     = *members[index];
        const std::uint64_t rows = static_cast<std::uint64_t>(previous.n);
        if (!is_at(previous.qdata, rows, code_per_row, member.qdata) ||
            !is_at(previous.scales, rows, scale_per_row, member.scales)) {
            return false;
        }
        if (high_per_row != 0 && !is_at(previous.qhigh, rows, high_per_row, member.qhigh)) {
            return false;
        }
    }
    return true;
}

// The same weight with the union row count. The group's planes begin at the object's first row and
// run to the union's last once `row_slices_are_contiguous` holds, so only `n` changes.
Weight fused_row_weight(const Weight& first, std::int32_t rows) {
    Weight fused = first;
    fused.n      = rows;
    return fused;
}

} // namespace

std::size_t layer_workspace_bytes(const TextConfig& config, std::int32_t tokens,
                                  KvCacheStorage storage, std::int32_t ring_slack) {
    const std::uint64_t h = config.hidden_size;
    const std::uint64_t i = config.intermediate_size;
    const std::uint64_t slide_q =
        static_cast<std::uint64_t>(config.num_attention_heads) * config.sliding.head_dim;
    const std::uint64_t slide_kv =
        static_cast<std::uint64_t>(config.sliding.num_key_value_heads) * config.sliding.head_dim;
    const std::uint64_t global_q =
        static_cast<std::uint64_t>(config.num_attention_heads) * config.global.shared.head_dim;
    const std::uint64_t global_kv =
        static_cast<std::uint64_t>(config.global.shared.num_key_value_heads) *
        config.global.shared.head_dim;
    const std::uint64_t compact = (config.global.shared.head_dim + 2 * config.global.rope_angles) *
                                  config.global.shared.num_key_value_heads;

    // Every buffer either kind allocates, counted exactly so the bound cannot drift from the
    // scratch:
    //   hidden-sized:        normed, branch, projected, down
    //   intermediate-sized:  gate, up, activated
    //   query-sized:         q, q_normed, attended
    //   key/value-sized:     k, value, k_normed
    //   compact (global):    the R + 2P by Hkv compact row
    // plus the pass's two index vectors, and the MoE block's K-per-token router output when the
    // model enables it. Every buffer scales with the batch. Sliding layers add a v projection on
    // top of the key/value buffers. The arena aligns every allocation to 256 bytes, so the bound
    // carries that per slice, and an undercount makes the arena refuse an allocation rather than
    // overrun.
    const std::uint64_t sliding = 4 * h + 3 * slide_q + 4 * slide_kv + 3 * i;
    const std::uint64_t global  = 4 * h + 3 * global_q + 3 * global_kv + compact + 3 * i;
    if (tokens < 1) throw std::invalid_argument("layer_workspace_bytes: tokens must be positive");
    const std::uint64_t t = static_cast<std::uint64_t>(tokens);
    // The attention's key split needs its own workspace when a pass has few rows. Its size depends
    // on the pass width, not the key count once the keys are many, so the largest over every width
    // up to `tokens` against an unbounded cache bounds it.
    std::size_t split = 0;
    const ops::AttentionHeadGeometry sliding_heads{
        static_cast<std::int32_t>(config.sliding.head_dim),
        static_cast<std::int32_t>(config.num_attention_heads),
        static_cast<std::int32_t>(config.sliding.num_key_value_heads)};
    const ops::AttentionHeadGeometry global_heads{
        static_cast<std::int32_t>(config.global.shared.head_dim),
        static_cast<std::int32_t>(config.num_attention_heads),
        static_cast<std::int32_t>(config.global.shared.num_key_value_heads)};
    for (std::int32_t width = 1; width <= tokens; ++width) {
        split = std::max(
            split, ops::sliding_causal_attention_workspace_bytes(sliding_heads, width, 1 << 24, 1));
        split = std::max(split, ops::causal_compact_attention_workspace_bytes(
                                    global_heads, global_heads.head_dim, width, 1 << 24, 1));
    }
    // The MoE block's router output -- K expert ids and K FP32 weights per token -- and the expert
    // workspace, plus the alignment of its three slices.
    const std::size_t route = config.enable_moe_block
                                  ? 2 * static_cast<std::size_t>(config.top_k_experts) *
                                            static_cast<std::size_t>(t) * sizeof(float) +
                                        moe_workspace_bytes(config, tokens) + 3 * 256
                                  : 0;
    // A quantized sliding ring is decoded into BF16 scratch before the layer attends: one ring for
    // the keys and one for the values, over the window plus the slack the cache is configured with.
    // Nothing else in the frame scales with the ring rather than with the pass.
    const std::size_t decode =
        storage == KvCacheStorage::BFloat16
            ? 0
            : 2 * static_cast<std::size_t>(config.sliding.head_dim) *
                      static_cast<std::size_t>(config.sliding.num_key_value_heads) *
                      (config.sliding_window + static_cast<std::size_t>(ring_slack)) * 2 +
                  512;
    return static_cast<std::size_t>(std::max(sliding, global) * t * 2 + 18 * 256 + t * 8) + split +
           route + decode;
}

PassIndices write_pass_indices(const KvCache& cache, std::int32_t first_position,
                               std::int32_t tokens, DeviceArena& arena, cudaStream_t stream) {
    if (tokens < 1) throw std::invalid_argument("write_pass_indices: tokens must be positive");
    const PassSegment segment{const_cast<KvCache*>(&cache), first_position, 0, tokens};
    return write_pass_indices(std::span(&segment, 1), arena, stream);
}

PassIndices write_pass_indices(std::span<const PassSegment> segments, DeviceArena& arena,
                               cudaStream_t stream) {
    std::int32_t tokens = 0;
    for (const PassSegment& segment : segments) tokens += segment.tokens;
    check_segments(segments, tokens, "write_pass_indices");
    PassIndices indices;
    indices.positions  = arena.alloc(DType::I32, {tokens});
    indices.ring_slots = arena.alloc(DType::I32, {tokens});
    std::vector<std::int32_t> host(2 * static_cast<std::size_t>(tokens));
    for (const PassSegment& segment : segments) {
        const std::int32_t ring = segment.cache->ring_tokens();
        if (ring < 1) {
            throw std::invalid_argument("write_pass_indices: the cache has no ring slots");
        }
        for (std::int32_t token = 0; token < segment.tokens; ++token) {
            const std::int32_t position = segment.first_position + token;
            const std::size_t column    = static_cast<std::size_t>(segment.column + token);
            host[column]                = position;
            host[static_cast<std::size_t>(tokens) + column] = position % ring;
        }
    }
    const std::size_t bytes = static_cast<std::size_t>(tokens) * sizeof(std::int32_t);
    CUDA_CHECK(cudaMemcpyAsync(indices.positions.data, host.data(), bytes, cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(indices.ring_slots.data, host.data() + tokens, bytes,
                               cudaMemcpyHostToDevice, stream));
    return indices;
}

// The [H,T] and [I,T] buffers the feed-forward sublayer works in, whichever layer path owns them.
// A Tensor is a view, so this is a handful of pointers whichever way it is passed.
struct FfnScratch {
    Tensor normed, branch, projected, down, gate, up, activated;
    // Set when gate and up are row views of one frame, so the sublayer runs one Linear over the
    // union instead of two. `gate_up` is that frame; `gate` and `up` are its halves.
    Tensor gate_up;
    bool gate_up_fused = false;
};

ops::GemmaMoeRouterWeights router_weights(const Model& model, const MoeWeights& moe) {
    return {ops::prepare_linear_weight(model.input(moe.router)).weight,
            model.tensor(moe.router_scale), model.tensor(moe.per_expert_scale)};
}

// The two expert banks, one part per expert per projection in bank order. The checkpoint fuses an
// expert's gate and up into one tensor with the gate half first and the artifact keeps one parent
// per bank, so a bank's parts are contiguous and their union is the bank the Op reads.
ops::GemmaMoeExpertWeights expert_weights(const Model& model, const MoeWeights& moe) {
    std::vector<ops::WeightInput> gate_up, down;
    gate_up.reserve(2 * moe.experts.size());
    down.reserve(moe.experts.size());
    for (const MlpWeights& expert : moe.experts) {
        gate_up.push_back(model.input(expert.gate));
        gate_up.push_back(model.input(expert.up));
        down.push_back(model.input(expert.down));
    }
    return ops::prepare_gemma_moe_expert_weights(gate_up, down);
}

// The layer's feed-forward sublayer: the fused gate/up parent, GeGLU, down, then the sandwich and
// the layer scalar. A model whose config enables the MoE block runs the experts beside the dense
// MLP and sums the two branches, each through its own post-norm, before the shared one.
//
// Every post-norm whose only reader is the residual accumulate -- the shared one here, and each
// sublayer's own -- is folded into `ops::residual_add_norm` with the accumulate and the layer
// scalar, so the tail is one launch an accumulate instead of a norm, an add and a scale.
//
// The MoE branch reuses `normed` and `projected` in place rather than two more [H,T] buffers: by
// the time each is rewritten its last reader has run. The dense branch consumes `normed` (gate/up);
// `projected` holds the attention output, which the fused accumulate into the residual consumes;
// and the second post-norm's output is accumulated into `branch` directly rather than into `down`.
// Only the router's two outputs and the expert apply's products are new storage, and the caller
// allocates them when the block is enabled. `branch` is the h1 accumulator the MoE block needs; a
// dense model's sublayer still has it allocated, because the frame is one size for both kinds, and
// the bound counts it for both.
void feed_forward_sublayer(const Model& model, const LayerWeights& weights, FfnScratch s,
                           const MoeScratch& moe_scratch, Tensor& residual, cudaStream_t stream) {
    const TextConfig& config = model.config();
    const float eps          = config.rms_norm_eps;
    auto to_weight           = [&](WeightId id) {
        return ops::prepare_linear_weight(model.input(id)).weight;
    };

    ops::rmsnorm(residual, model.tensor(weights.pre_feedforward_norm), eps, false, s.normed,
                 stream);
    if (s.gate_up_fused) {
        ops::linear(s.normed,
                    fused_row_weight(to_weight(weights.mlp.gate),
                                     2 * static_cast<std::int32_t>(config.intermediate_size)),
                    s.gate_up, stream);
    } else {
        ops::linear(s.normed, to_weight(weights.mlp.gate), s.gate, stream);
        ops::linear(s.normed, to_weight(weights.mlp.up), s.up, stream);
    }
    ops::gelu_mul(s.gate, s.up, s.activated, stream);
    ops::linear(s.activated, to_weight(weights.mlp.down), s.down, stream);

    if (!weights.moe) {
        ops::residual_add_norm(s.down, model.tensor(weights.post_feedforward_norm), eps, residual,
                               model.layer_scalar(weights.layer_scalar), stream);
        return;
    }

    const MoeWeights& moe = *weights.moe;
    // h1: the dense branch through its own post-norm. The MoE branch reads the pre-FFN residual
    // directly -- the router weightlessly, the experts through their own second pre-norm.
    ops::rmsnorm(s.down, model.tensor(moe.post_feedforward_norm_1), eps, false, s.branch, stream);
    MoeScratch moe_out = moe_scratch;
    ops::gemma_moe_router(residual, router_weights(model, moe), router_geometry(config),
                          moe_out.ids, moe_out.weights, moe_out.workspace, stream);
    ops::rmsnorm(residual, model.tensor(moe.pre_feedforward_norm_2), eps, false, s.normed, stream);
    ops::gemma_moe_experts(s.normed, moe_out.ids, moe_out.weights, expert_weights(model, moe),
                           expert_geometry(config), s.projected, moe_out.workspace, stream);
    // h2, then h = h1 + h2.
    ops::residual_add_norm(s.projected, model.tensor(moe.post_feedforward_norm_2), eps, s.branch,
                           1.0F, stream);
    // The shared post-norm, the residual add and the scalar, exactly as the dense path ends.
    ops::residual_add_norm(s.branch, model.tensor(weights.post_feedforward_norm), eps, residual,
                           model.layer_scalar(weights.layer_scalar), stream);
}

// `tokens` is the pass's column count over every segment.
void forward_sliding_layer(const Model& model, std::size_t layer, const Tensor& hidden_in,
                           std::span<const PassSegment> segments, std::int32_t tokens,
                           const PassIndices& indices, DeviceArena& arena, Tensor& hidden_out,
                           DeviceExecutionView execution, const Tensor& position_high) {
    const cudaStream_t stream = execution.stream;
    const TextConfig& config  = model.config();
    if (layer >= model.weights().text.layers.size()) {
        throw std::invalid_argument("forward_sliding_layer: layer index out of range");
    }
    const LayerWeights& weights = model.weights().text.layers[layer];
    if (weights.mixer != MixerKind::SlidingAttention) {
        throw std::invalid_argument("forward_sliding_layer: layer is not a sliding layer");
    }
    if (tokens < 1) {
        throw std::invalid_argument("forward_sliding_layer: tokens must be positive");
    }

    // The scratch is a per-call frame: take a scope so a caller can hand the same arena to every
    // layer of a pass, and so `sliding_layer_workspace_bytes` bounds one call rather than all of
    // them.
    DeviceArena::Scope scope = arena.scope();

    const std::int32_t h        = static_cast<std::int32_t>(config.hidden_size);
    const std::int32_t i        = static_cast<std::int32_t>(config.intermediate_size);
    const std::int32_t d        = static_cast<std::int32_t>(config.sliding.head_dim);
    const std::int32_t hq       = static_cast<std::int32_t>(config.num_attention_heads);
    const std::int32_t hkv      = static_cast<std::int32_t>(config.sliding.num_key_value_heads);
    const std::int32_t q_width  = hq * d;
    const std::int32_t kv_width = hkv * d;
    const float eps             = config.rms_norm_eps;

    Scratch scratch;
    // The union is one frame the group can be viewed from only at T=1, which is why decode fuses
    // and every wider pass keeps the separate launches: tensors are column-major with dim 0
    // fastest, so a row slice of a [Q+2KV,T] frame carries that frame's token stride and is not
    // contiguous for T>1 -- which rmsnorm requires. At T=1 the frame's row 0 is byte 0, so a
    // [rows,1] slice is exactly a standalone [rows,1] tensor. See row_slices_are_contiguous for the
    // weight side.
    const Weight query_weight =
        ops::prepare_linear_weight(model.input(weights.attention.query)).weight;
    const Weight key_weight = ops::prepare_linear_weight(model.input(weights.attention.key)).weight;
    const Weight value_weight =
        ops::prepare_linear_weight(model.input(weights.attention.value)).weight;
    const Weight* qkv_weights[] = {&query_weight, &key_weight, &value_weight};
    const bool fuse_qkv         = tokens == 1 && row_slices_are_contiguous(qkv_weights, 3);
    // The same union for the layer's gate and up, which the converter groups for the layouts whose
    // parent a slice can be read from.
    const Weight gate_weight = ops::prepare_linear_weight(model.input(weights.mlp.gate)).weight;
    const Weight up_weight   = ops::prepare_linear_weight(model.input(weights.mlp.up)).weight;
    const Weight* gate_up_weights[] = {&gate_weight, &up_weight};
    const bool fuse_gate_up         = tokens == 1 && row_slices_are_contiguous(gate_up_weights, 2);
    scratch.normed                  = arena.alloc(DType::BF16, {h, tokens});
    scratch.branch                  = arena.alloc(DType::BF16, {h, tokens});
    if (fuse_qkv) {
        scratch.qkv = arena.alloc(DType::BF16, {q_width + 2 * kv_width, tokens});
        scratch.q   = scratch.qkv.slice(0, 0, q_width);
        scratch.k   = scratch.qkv.slice(0, q_width, kv_width);
        scratch.v   = scratch.qkv.slice(0, q_width + kv_width, kv_width);
    } else {
        scratch.q = arena.alloc(DType::BF16, {q_width, tokens});
        scratch.k = arena.alloc(DType::BF16, {kv_width, tokens});
        scratch.v = arena.alloc(DType::BF16, {kv_width, tokens});
    }
    scratch.q_normed  = arena.alloc(DType::BF16, {d, hq, tokens});
    scratch.k_normed  = arena.alloc(DType::BF16, {d, hkv, tokens});
    scratch.v_normed  = arena.alloc(DType::BF16, {d, hkv, tokens});
    scratch.attended  = arena.alloc(DType::BF16, {q_width, tokens});
    scratch.projected = arena.alloc(DType::BF16, {h, tokens});
    if (fuse_gate_up) {
        scratch.gate_up = arena.alloc(DType::BF16, {2 * i, tokens});
        scratch.gate    = scratch.gate_up.slice(0, 0, i);
        scratch.up      = scratch.gate_up.slice(0, i, i);
    } else {
        scratch.gate = arena.alloc(DType::BF16, {i, tokens});
        scratch.up   = arena.alloc(DType::BF16, {i, tokens});
    }
    scratch.activated = arena.alloc(DType::BF16, {i, tokens});
    scratch.down      = arena.alloc(DType::BF16, {h, tokens});
    if (weights.moe) scratch.moe = allocate_moe_scratch(config, tokens, arena);

    Tensor residual = hidden_in;
    auto to_weight  = [&](WeightId id) {
        return ops::prepare_linear_weight(model.input(id)).weight;
    };

    // Attention sublayer. The key and value sets are the token itself, so the same position vector
    // serves both sides of the causal window.
    ops::rmsnorm(residual, model.tensor(weights.input_norm), eps, false, scratch.normed, stream);
    if (fuse_qkv) {
        ops::linear(scratch.normed, fused_row_weight(query_weight, q_width + 2 * kv_width),
                    scratch.qkv, stream);
    } else {
        ops::linear(scratch.normed, query_weight, scratch.q, stream);
        ops::linear(scratch.normed, key_weight, scratch.k, stream);
        ops::linear(scratch.normed, value_weight, scratch.v, stream);
    }

    // Per-head norms: q and k carry a weight, v is normalized without one.
    scratch.q_heads = scratch.q.view({d, hq, tokens});
    scratch.k_heads = scratch.k.view({d, hkv, tokens});
    scratch.v_heads = scratch.v.view({d, hkv, tokens});
    ops::rmsnorm(scratch.q_heads, model.tensor(weights.attention.query_norm), eps, false,
                 scratch.q_normed, stream);
    ops::rmsnorm(scratch.k_heads, model.tensor(weights.attention.key_norm), eps, false,
                 scratch.k_normed, stream);
    ops::rmsnorm(scratch.v_heads, eps, scratch.v_normed, stream);

    // RoPE rotates in place over the head dimension; the sliding layers rotate the whole head, so
    // the pair count is half of it.
    ops::rope(indices.positions.view({tokens, 1, 1}), d, d / 2, config.sliding.rope_theta,
              scratch.q_normed, scratch.k_normed, execution);

    // The ring holds the last window plus slack of tokens at position modulo the ring; a slot no
    // token has written carries an invisible position. A pass's keys enter the ring as its last
    // `kept` tokens. A pass of at most slack + 1 tokens writes first and attends to the ring alone:
    // the rows it replaces are a window older than its first query, and the key set stays one. A
    // wider pass attends to the ring as it was plus its own keys as the Op's second key set, and
    // writes afterwards, so no write replaces a key one of its own queries still sees; the stale
    // rows a rolled-back speculation left at its positions are forgotten first. All of that is
    // decided per segment, against the segment's own lane: a decode round's segments are one token
    // each.
    const std::int32_t row = d * hkv;
    for (const PassSegment& segment : segments) {
        // Each segment's split workspace is released before the next one takes its own.
        DeviceArena::Scope attention_scope = arena.scope();
        KvCache& cache                     = *segment.cache;
        const std::int32_t n               = segment.tokens;
        const std::int32_t column          = segment.column;
        // A BF16 ring hands its rows to the attention directly; a quantized one is written through
        // its format's encoder and decoded back into BF16 scratch below.
        const bool quantized    = cache.quantized(layer);
        Tensor cache_keys       = quantized ? Tensor{} : cache.keys(layer);
        Tensor cache_values     = quantized ? Tensor{} : cache.values(layer);
        Tensor cache_pos        = cache.positions(layer);
        const std::int32_t ring = cache.ring_tokens();
        const bool write_first  = n <= cache.ring_slack() + 1;
        if (!write_first) cache.forget_stale(layer, segment.first_position, stream);

        // The segment's columns of the pass-wide tensors.
        const Tensor positions = indices.positions.slice(0, column, n);
        const Tensor slots     = indices.ring_slots.slice(0, column, n);
        const Tensor k_rows(column_data(scratch.k_normed, row * sizeof(std::uint16_t), column),
                            DType::BF16, {row, n});
        const Tensor v_rows(column_data(scratch.v_normed, row * sizeof(std::uint16_t), column),
                            DType::BF16, {row, n});

        // The kept rows go to the slots `indices` names, so the writes carry no host offset; the
        // last ring of a pass has distinct slots.
        const std::int32_t kept = std::min(n, ring);
        const auto write_kept   = [&] {
            if (quantized) {
                const std::int32_t skipped = n - kept;
                const Tensor key_segment   = scratch.k_normed.slice(2, column, n);
                const Tensor value_segment = scratch.v_normed.slice(2, column, n);
                Tensor key_codes           = cache.key_codes(layer);
                Tensor key_scales          = cache.key_scales(layer);
                Tensor value_codes         = cache.value_codes(layer);
                Tensor value_scales        = cache.value_scales(layer);
                ops::dense_kv_ring_encode(key_segment.slice(2, skipped, kept),
                                          value_segment.slice(2, skipped, kept),
                                          slots.slice(0, skipped, kept), cache.storage(), key_codes,
                                          key_scales, value_codes, value_scales, stream);
                write_cache_positions(positions, slots, kept, cache_pos, stream);
                return;
            }
            const Tensor key_rows(cache_keys.data, DType::BF16, {row, ring});
            const Tensor value_rows(cache_values.data, DType::BF16, {row, ring});
            write_cache_rows(k_rows, positions, slots, kept, key_rows, cache_pos, stream);
            Tensor value_destination = value_rows;
            ops::scatter(v_rows.slice(1, n - kept, kept), slots.slice(0, n - kept, kept),
                         value_destination, stream);
        };
        if (write_first) write_kept();

        // A quantized ring decodes into BF16 scratch, which is what the sliding Op takes: every
        // slot comes back in the basis its encoder read, so nothing downstream changes.
        Tensor ring_keys   = cache_keys;
        Tensor ring_values = cache_values;
        if (quantized) {
            ring_keys   = arena.alloc(DType::BF16, {d, hkv, ring});
            ring_values = arena.alloc(DType::BF16, {d, hkv, ring});
            ops::dense_kv_ring_decode(cache.key_codes(layer), cache.key_scales(layer),
                                      cache.value_codes(layer), cache.value_scales(layer),
                                      cache.storage(), ring_keys, ring_values, stream);
        }

        const ops::AttentionHeadGeometry geometry{d, hq, hkv};
        const std::size_t q_column = static_cast<std::size_t>(q_width) * sizeof(std::uint16_t);
        Tensor q_batch(column_data(scratch.q_normed, q_column, column), DType::BF16, {d, hq, n, 1});
        // A narrow pass attends to the whole ring, whose unwritten slots are invisible, so its
        // launches are the same at every position. A wide pass runs eagerly and passes only the
        // leading slots the ring has filled before it (one unwritten slot when it is empty).
        const std::int32_t filled =
            write_first ? ring : std::clamp(segment.first_position, 1, ring);
        Tensor k_ring(static_cast<std::uint8_t*>(ring_keys.data), DType::BF16, {d, hkv, filled, 1});
        Tensor v_ring(static_cast<std::uint8_t*>(ring_values.data), DType::BF16,
                      {d, hkv, filled, 1});
        Tensor position_ring(static_cast<std::uint8_t*>(cache_pos.data), DType::I32, {filled, 1});
        Tensor position_pass = positions.view({n, 1});
        Tensor attended_batch(column_data(scratch.attended, q_column, column), DType::BF16,
                              {d, hq, n, 1});
        const Tensor none;
        const Tensor k_pass = write_first ? none : Tensor(k_rows.data, DType::BF16, {d, hkv, n, 1});
        const Tensor v_pass = write_first ? none : Tensor(v_rows.data, DType::BF16, {d, hkv, n, 1});
        const Tensor position_extra = write_first ? none : position_pass;
        const std::size_t split     = ops::sliding_causal_attention_workspace_bytes(
            geometry, n, filled + (write_first ? 0 : n), 1);
        Tensor workspace;
        if (split > 0) workspace = arena.alloc(DType::U8, {static_cast<std::int32_t>(split)});
        // An image's later keys are part of this pass, in the ring (written first) or in the pass's
        // own key set, so its tokens' upper bound always finds them.
        ops::sliding_causal_attention(q_batch, k_ring, v_ring, position_pass, position_high,
                                      position_ring, k_pass, v_pass, position_extra, geometry,
                                      config.sliding_window, config.attention_scale, attended_batch,
                                      workspace, execution);
        if (!write_first) write_kept();
    }
    ops::linear(scratch.attended, to_weight(weights.attention.output), scratch.projected, stream);
    ops::residual_add_norm(scratch.projected, model.tensor(weights.post_attention_norm), eps,
                           residual, 1.0F, stream);

    // MLP sublayer: the dense branch, and the MoE block beside it when the model has one.
    feed_forward_sublayer(model, weights,
                          {scratch.normed, scratch.branch, scratch.projected, scratch.down,
                           scratch.gate, scratch.up, scratch.activated, scratch.gate_up,
                           static_cast<bool>(fuse_gate_up)},
                          scratch.moe, residual, stream);

    // The layer's output is the residual stream itself; the next layer applies its own input norm.
    CUDA_CHECK(cudaMemcpyAsync(hidden_out.data, residual.data,
                               static_cast<std::size_t>(h) * static_cast<std::size_t>(tokens) *
                                   sizeof(std::uint16_t),
                               cudaMemcpyDeviceToDevice, stream));
}

// `tokens` is the pass's column count over every segment.
void forward_global_layer(const Model& model, std::size_t layer, const Tensor& hidden_in,
                          std::span<const PassSegment> segments, std::int32_t tokens,
                          const PassIndices& indices, DeviceArena& arena, Tensor& hidden_out,
                          DeviceExecutionView execution) {
    const cudaStream_t stream = execution.stream;
    const TextConfig& config  = model.config();
    if (layer >= model.weights().text.layers.size()) {
        throw std::invalid_argument("forward_global_layer: layer index out of range");
    }
    const LayerWeights& weights = model.weights().text.layers[layer];
    if (weights.mixer != MixerKind::FullAttention) {
        throw std::invalid_argument("forward_global_layer: layer is not a global layer");
    }
    if (tokens < 1) {
        throw std::invalid_argument("forward_global_layer: tokens must be positive");
    }
    // These layers keep one row per token at the token's own position, so each lane's tokens must
    // fit.
    for (const PassSegment& segment : segments) {
        if (segment.first_position + segment.tokens > segment.cache->capacity()) {
            throw std::invalid_argument(
                "forward_global_layer: the batch exceeds the cache capacity");
        }
    }

    // The scratch is a per-call frame, exactly as in the sliding path.
    DeviceArena::Scope scope = arena.scope();

    const GlobalAttentionGeometry& geometry = config.global;
    const std::int32_t h                    = static_cast<std::int32_t>(config.hidden_size);
    const std::int32_t i                    = static_cast<std::int32_t>(config.intermediate_size);
    const std::int32_t d                    = static_cast<std::int32_t>(geometry.shared.head_dim);
    const std::int32_t hq                   = static_cast<std::int32_t>(config.num_attention_heads);
    const std::int32_t hkv   = static_cast<std::int32_t>(geometry.shared.num_key_value_heads);
    const std::int32_t pairs = static_cast<std::int32_t>(geometry.rope_angles);
    const std::int32_t width = d + 2 * pairs;
    const float eps          = config.rms_norm_eps;

    Tensor normed    = arena.alloc(DType::BF16, {h, tokens});
    Tensor branch    = arena.alloc(DType::BF16, {h, tokens});
    Tensor projected = arena.alloc(DType::BF16, {h, tokens});
    Tensor down      = arena.alloc(DType::BF16, {h, tokens});
    Tensor q         = arena.alloc(DType::BF16, {hq * d, tokens});
    Tensor q_normed  = arena.alloc(DType::BF16, {d, hq, tokens});
    Tensor attended  = arena.alloc(DType::BF16, {hq * d, tokens});
    Tensor k         = arena.alloc(DType::BF16, {hkv * d, tokens});
    Tensor value     = arena.alloc(DType::BF16, {d, hkv, tokens});
    Tensor k_normed  = arena.alloc(DType::BF16, {d, hkv, tokens});
    Tensor compact   = arena.alloc(DType::BF16, {width, hkv, tokens});
    const Weight global_gate_weight =
        ops::prepare_linear_weight(model.input(weights.mlp.gate)).weight;
    const Weight global_up_weight  = ops::prepare_linear_weight(model.input(weights.mlp.up)).weight;
    const Weight* global_gate_up[] = {&global_gate_weight, &global_up_weight};
    const bool fuse_gate_up        = tokens == 1 && row_slices_are_contiguous(global_gate_up, 2);
    Tensor gate_up;
    Tensor gate;
    Tensor up;
    if (fuse_gate_up) {
        gate_up = arena.alloc(DType::BF16, {2 * i, tokens});
        gate    = gate_up.slice(0, 0, i);
        up      = gate_up.slice(0, i, i);
    } else {
        gate = arena.alloc(DType::BF16, {i, tokens});
        up   = arena.alloc(DType::BF16, {i, tokens});
    }
    Tensor activated     = arena.alloc(DType::BF16, {i, tokens});
    const MoeScratch moe = weights.moe ? allocate_moe_scratch(config, tokens, arena) : MoeScratch{};
    const Tensor positions = indices.positions.view({tokens, 1, 1});

    Tensor residual = hidden_in;
    auto to_weight  = [&](WeightId id) {
        return ops::prepare_linear_weight(model.input(id)).weight;
    };

    ops::rmsnorm(residual, model.tensor(weights.input_norm), eps, false, normed, stream);
    ops::linear(normed, to_weight(weights.attention.query), q, stream);
    ops::linear(normed, to_weight(weights.attention.key), k, stream);

    // K is V on these layers, so the value side is the key normalized without a weight, read from
    // the raw projection; the weighted key norm is a second, independent normalization of the same
    // input.
    const Tensor k_heads = k.view({d, hkv, tokens});
    ops::rmsnorm(k_heads, eps, value, stream);
    ops::rmsnorm(q.view({d, hq, tokens}), model.tensor(weights.attention.query_norm), eps, false,
                 q_normed, stream);
    ops::rmsnorm(k_heads, model.tensor(weights.attention.key_norm), eps, false, k_normed, stream);

    // Proportional RoPE: `rope_angles` pairs taken against the full head width, so the pair partner
    // of dim i is i + d/2 and the dims outside those two runs carry no rotation.
    ops::rope(positions, d, pairs, geometry.shared.rope_theta, q_normed, execution);
    ops::rope(positions, d, pairs, geometry.shared.rope_theta, k_normed, execution);

    // On the non-rotated dims the key the score needs is the value times the key norm's weight, so
    // the query carries that weight instead: q[d] * w_kn[d] against the stored value.
    const Tensor& w_kn = model.tensor(weights.attention.key_norm);
    ops::scale_columns(w_kn, q_normed, pairs, d / 2, stream);
    ops::scale_columns(w_kn, q_normed, d / 2 + pairs, d, stream);

    ops::compact_kv_rows(value, k_normed, compact, d, pairs, stream);

    // The compact row is written at the token's own position, the slot the positions themselves
    // name, and read back causally, so a slot no token has written carries an invisible position
    // and the whole cache can be handed to the Op. Each segment writes and attends within its own
    // lane.
    const std::size_t compact_column =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(hkv) * sizeof(std::uint16_t);
    const std::size_t q_column =
        static_cast<std::size_t>(hq) * static_cast<std::size_t>(d) * sizeof(std::uint16_t);
    for (const PassSegment& segment : segments) {
        // Each segment's split workspace is released before the next one takes its own.
        DeviceArena::Scope attention_scope = arena.scope();
        KvCache& cache                     = *segment.cache;
        const std::int32_t n               = segment.tokens;
        const std::int32_t column          = segment.column;
        const bool coded                   = cache.coded_global(layer);
        Tensor cache_rows                  = coded ? Tensor{} : cache.keys(layer);
        Tensor cache_codes                 = coded ? cache.key_codes(layer) : Tensor{};
        Tensor cache_scales                = coded ? cache.key_scales(layer) : Tensor{};
        Tensor cache_pos                   = cache.positions(layer);
        const std::int32_t capacity        = cache.capacity();
        const Tensor segment_positions     = indices.positions.slice(0, column, n);
        const Tensor compact_batch =
            Tensor(column_data(compact, compact_column, column), DType::BF16, {width * hkv, n});
        if (coded) {
            // The compact rows are encoded straight into the FP8 planes; the position plane still
            // names each row's slot so the Op can make unwritten slots invisible.
            ops::compact_kv_rows_quantize(compact_batch, segment_positions, cache_codes,
                                          cache_scales, stream);
            write_cache_positions(segment_positions, segment_positions, n, cache_pos, stream);
        } else {
            write_cache_rows(compact_batch,
            segment_positions, segment_positions, n,
            Tensor(cache_rows.data, DType::BF16, {width * hkv, capacity}), cache_pos, stream);
        }

        const ops::AttentionHeadGeometry heads{d, hq, hkv};
        Tensor q_batch(column_data(q_normed, q_column, column), DType::BF16, {d, hq, n, 1});
        // The Op is handed the whole capacity: with `indexed_keys` it stops at the latest query's
        // position on the device, so the launches do not depend on how much of the cache is filled.
        Tensor attended_batch(column_data(attended, q_column, column), DType::BF16, {d, hq, n, 1});
        Tensor position_q = segment_positions.view({n, 1});
        Tensor position_k(static_cast<std::uint8_t*>(cache_pos.data), DType::I32, {capacity, 1});
        const std::size_t split =
            ops::causal_compact_attention_workspace_bytes(heads, d, n, capacity, 1);
        Tensor workspace;
        if (split > 0) workspace = arena.alloc(DType::U8, {static_cast<std::int32_t>(split)});
        // Rows sit at their own positions, which is what `indexed_keys` states.
        if (coded) {
            Tensor codes_batch(static_cast<std::uint8_t*>(cache_codes.data), DType::FP8_E4M3FN,
                               {width, hkv, capacity, 1});
            Tensor scales_batch(static_cast<std::uint8_t*>(cache_scales.data), DType::FP16,
                                {hkv, capacity, 1, 1});
            ops::causal_compact_attention_codes(q_batch, codes_batch, scales_batch, position_q,
                                                position_k, heads, d, pairs, config.attention_scale,
                                                true, attended_batch, workspace, execution);
        } else {
            Tensor kv_batch(static_cast<std::uint8_t*>(cache_rows.data), DType::BF16,
                            {width, hkv, capacity, 1});
        ops::causal_compact_attention(q_batch, kv_batch, position_q, position_k, heads, d, pairs,
                                      config.attention_scale, true, attended_batch, workspace,
                                      execution);
        }
    }
    ops::linear(attended, to_weight(weights.attention.output), projected, stream);
    ops::residual_add_norm(projected, model.tensor(weights.post_attention_norm), eps, residual,
                           1.0F, stream);

    // MLP sublayer: the dense branch, and the MoE block beside it when the model has one.
    feed_forward_sublayer(
        model, weights,
        {normed, branch, projected, down, gate, up, activated, gate_up, fuse_gate_up}, moe,
        residual, stream);

    CUDA_CHECK(cudaMemcpyAsync(hidden_out.data, residual.data,
                               static_cast<std::size_t>(h) * static_cast<std::size_t>(tokens) *
                                   sizeof(std::uint16_t),
                               cudaMemcpyDeviceToDevice, stream));
}

void forward_head(const Model& model, const Tensor& hidden_in, std::int32_t tokens,
                  DeviceArena& arena, Tensor& logits, DeviceExecutionView execution,
                  Tensor* normed_out) {
    const cudaStream_t stream = execution.stream;
    const TextConfig& config  = model.config();

    DeviceArena::Scope scope = arena.scope();
    const std::int32_t h     = static_cast<std::int32_t>(config.hidden_size);

    // The caller's buffer may be wider than this pass: its leading columns are contiguous.
    Tensor normed = normed_out != nullptr ? Tensor(normed_out->data, DType::BF16, {h, tokens})
                                          : arena.alloc(DType::BF16, {h, tokens});
    ops::rmsnorm(hidden_in, model.tensor(model.weights().text.final_norm), config.rms_norm_eps,
                 false, normed, stream);
    auto weight = ops::prepare_linear_weight(model.input(model.weights().text.output_head)).weight;
    ops::linear(normed, weight, logits, stream);
    ops::soft_cap(logits, config.final_logit_softcapping, stream);
}

void forward_draft(const Model& model, const Tensor& input, std::int32_t position,
                   const KvCache& cache, DeviceArena& arena, Tensor& logits, Tensor& hidden_out,
                   DeviceExecutionView execution) {
    const DraftLane lane{&cache, position};
    forward_draft(model, input, std::span(&lane, 1), arena, logits, hidden_out, execution);
}

void forward_draft(const Model& model, const Tensor& input, std::span<const DraftLane> lanes,
                   DeviceArena& arena, Tensor& logits, Tensor& hidden_out,
                   DeviceExecutionView execution) {
    const cudaStream_t stream = execution.stream;
    const TextConfig& target  = model.config();
    if (!model.draft_config() || !model.weights().draft) {
        throw std::logic_error("forward_draft: the drafter is not loaded");
    }
    const DraftConfig& draft    = *model.draft_config();
    const DraftWeights& weights = *model.weights().draft;
    const std::int32_t count    = static_cast<std::int32_t>(lanes.size());
    if (count < 1 || input.ne[1] != count) {
        throw std::invalid_argument("forward_draft: one input column per lane");
    }
    // The drafter reads keys strictly before its token, so a token at position 0 has none.
    for (const DraftLane& lane : lanes) {
        if (lane.cache == nullptr || lane.position < 1 || lane.position > lane.cache->capacity()) {
            throw std::invalid_argument("forward_draft: position must be in [1, capacity]");
        }
    }

    DeviceArena::Scope scope = arena.scope();
    const std::int32_t h     = static_cast<std::int32_t>(draft.hidden_size);
    const std::int32_t i     = static_cast<std::int32_t>(draft.intermediate_size);
    const std::int32_t hq    = static_cast<std::int32_t>(target.num_attention_heads);
    const float eps          = draft.rms_norm_eps;
    auto to_weight           = [&](WeightId id) {
        return ops::prepare_linear_weight(model.input(id)).weight;
    };

    // Two position vectors: RoPE rotates at the token's own position, while visibility is that of
    // the previous position, which admits exactly the keys before the token. A row the target wrote
    // at or past the token (a rejected draft's, or none at all) stays invisible either way.
    Tensor positions = arena.alloc(DType::I32, {2 * count});
    std::vector<std::int32_t> host_positions(2 * static_cast<std::size_t>(count));
    for (std::int32_t column = 0; column < count; ++column) {
        host_positions[static_cast<std::size_t>(column)]         = lanes[column].position;
        host_positions[static_cast<std::size_t>(count + column)] = lanes[column].position - 1;
    }
    CUDA_CHECK(cudaMemcpyAsync(positions.data, host_positions.data(),
                               host_positions.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                               stream));
    // The pageable source is staged before the call returns.
    const Tensor rope_position(positions.data, DType::I32, {count, 1, 1});
    const auto visible_position = [&](std::int32_t column) {
        return Tensor(static_cast<std::uint8_t*>(positions.data) +
                          static_cast<std::size_t>(count + column) * sizeof(std::int32_t),
                      DType::I32, {1, 1});
    };

    Tensor residual  = arena.alloc(DType::BF16, {h, count});
    Tensor normed    = arena.alloc(DType::BF16, {h, count});
    Tensor branch    = arena.alloc(DType::BF16, {h, count});
    Tensor projected = arena.alloc(DType::BF16, {h, count});
    Tensor gate      = arena.alloc(DType::BF16, {i, count});
    Tensor up        = arena.alloc(DType::BF16, {i, count});
    Tensor activated = arena.alloc(DType::BF16, {i, count});
    Tensor down      = arena.alloc(DType::BF16, {h, count});
    const std::int32_t widest_q =
        hq *
        static_cast<std::int32_t>(std::max(target.sliding.head_dim, target.global.shared.head_dim));
    Tensor q_storage        = arena.alloc(DType::BF16, {widest_q, count});
    Tensor q_normed_storage = arena.alloc(DType::BF16, {widest_q, count});
    Tensor attended_storage = arena.alloc(DType::BF16, {widest_q, count});

    ops::linear(input, to_weight(weights.pre_projection), residual, stream);

    for (const DraftLayerWeights& layer : weights.layers) {
        const bool sliding     = layer.mixer == MixerKind::SlidingAttention;
        const auto& geometry   = sliding ? target.sliding : target.global.shared;
        const std::int32_t d   = static_cast<std::int32_t>(geometry.head_dim);
        const std::int32_t hkv = static_cast<std::int32_t>(geometry.num_key_value_heads);
        // The buffers are sized for the wider global head; a sliding layer uses their prefix.
        Tensor q(q_storage.data, DType::BF16, {hq * d, count});
        Tensor q_normed(q_normed_storage.data, DType::BF16, {d, hq, count});
        Tensor attended(attended_storage.data, DType::BF16, {hq * d, count});
        const std::size_t q_column =
            static_cast<std::size_t>(hq) * static_cast<std::size_t>(d) * sizeof(std::uint16_t);

        ops::rmsnorm(residual, model.tensor(layer.input_norm), eps, false, normed, stream);
        ops::linear(normed, to_weight(layer.query), q, stream);
        ops::rmsnorm(q.view({d, hq, count}), model.tensor(layer.query_norm), eps, false, q_normed,
                     stream);
        const ops::AttentionHeadGeometry heads{d, hq, hkv};
        const std::int32_t pairs = static_cast<std::int32_t>(target.global.rope_angles);
        if (sliding) {
            ops::rope(rope_position, d, d / 2, geometry.rope_theta, q_normed, execution);
        } else {
            ops::rope(rope_position, d, pairs, geometry.rope_theta, q_normed, execution);
            // The compact rows hold the weightless value and the rotated key; on the dims that do
            // not rotate, the key the target scores against is the value times the source layer's
            // key-norm weight, which the query carries instead, exactly as in the target's own
            // global layer.
            const Tensor& w_kn =
                model.tensor(target_layer_key_norm(model, draft.target_global_layer));
            ops::scale_columns(w_kn, q_normed, pairs, d / 2, stream);
            ops::scale_columns(w_kn, q_normed, d / 2 + pairs, d, stream);
        }
        // Each lane attends to its own cache, in its own scope for the split workspace.
        for (std::int32_t column = 0; column < count; ++column) {
            DeviceArena::Scope attention_scope = arena.scope();
            const KvCache& cache               = *lanes[column].cache;
            const Tensor q_lane(column_data(q_normed, q_column, column), DType::BF16,
                                {d, hq, 1, 1});
            Tensor out(column_data(attended, q_column, column), DType::BF16, {d, hq, 1, 1});
            if (sliding) {
                const std::size_t source = draft.target_sliding_layer;
                const bool quantized     = cache.quantized(source);
                const Tensor keys        = quantized ? Tensor{} : cache.keys(source);
                const Tensor values      = quantized ? Tensor{} : cache.values(source);
                const Tensor key_pos     = cache.positions(source);
                const std::int32_t ring  = cache.ring_tokens();
                // A quantized ring is decoded here too: the drafter reads the same rows the target
                // wrote, in the same basis.
                Tensor ring_keys   = keys;
                Tensor ring_values = values;
                if (quantized) {
                    ring_keys   = arena.alloc(DType::BF16, {d, hkv, ring});
                    ring_values = arena.alloc(DType::BF16, {d, hkv, ring});
                    ops::dense_kv_ring_decode(cache.key_codes(source), cache.key_scales(source),
                                              cache.value_codes(source), cache.value_scales(source),
                                              cache.storage(), ring_keys, ring_values, stream);
                }
                const Tensor k_ring(ring_keys.data, DType::BF16, {d, hkv, ring, 1});
                const Tensor v_ring(ring_values.data, DType::BF16, {d, hkv, ring, 1});
                const Tensor p_ring(key_pos.data, DType::I32, {ring, 1});
                const Tensor none;
                const std::size_t split =
                    ops::sliding_causal_attention_workspace_bytes(heads, 1, ring, 1);
                Tensor workspace;
                if (split > 0)
                    workspace = arena.alloc(DType::U8, {static_cast<std::int32_t>(split)});
                ops::sliding_causal_attention(q_lane, k_ring, v_ring, visible_position(column),
                                              none, p_ring, none, none, none, heads,
                                              target.sliding_window, target.attention_scale, out,
                                              workspace, execution);
            } else {
                const std::size_t source    = draft.target_global_layer;
                const std::int32_t width    = d + 2 * pairs;
                const std::int32_t position = lanes[column].position;
                const bool coded           = cache.coded_global(source);
                const Tensor row_pos        = cache.positions(source);
                const Tensor p_rows(row_pos.data, DType::I32, {position, 1});
                const std::size_t split =
                    ops::causal_compact_attention_workspace_bytes(heads, d, 1, position, 1);
                Tensor workspace;
                if (split > 0)
                    workspace = arena.alloc(DType::U8, {static_cast<std::int32_t>(split)});
                if (coded) {
                    const Tensor codes  = cache.key_codes(source);
                    const Tensor scales = cache.key_scales(source);
                    const Tensor code_batch(codes.data, DType::FP8_E4M3FN,
                                            {width, hkv, position, 1});
                    const Tensor scale_batch(scales.data, DType::FP16, {hkv, position, 1, 1});
                    ops::causal_compact_attention_codes(
                        q_lane, code_batch, scale_batch, visible_position(column), p_rows, heads, d,
                        pairs, target.attention_scale, true, out, workspace, execution);
                } else {
                    const Tensor rows = cache.keys(source);
                    const Tensor kv(rows.data, DType::BF16, {width, hkv, position, 1});
                ops::causal_compact_attention(q_lane, kv, visible_position(column), p_rows, heads,
                                              d, pairs, target.attention_scale, true, out,
                                              workspace, execution);
                }
            }
        }
        ops::linear(attended, to_weight(layer.output), projected, stream);
        ops::rmsnorm(projected, model.tensor(layer.post_attention_norm), eps, false, branch,
                     stream);
        ops::residual_add(branch, residual, stream);

        ops::rmsnorm(residual, model.tensor(layer.pre_feedforward_norm), eps, false, normed,
                     stream);
        ops::linear(normed, to_weight(layer.mlp.gate), gate, stream);
        ops::linear(normed, to_weight(layer.mlp.up), up, stream);
        ops::gelu_mul(gate, up, activated, stream);
        ops::linear(activated, to_weight(layer.mlp.down), down, stream);
        ops::rmsnorm(down, model.tensor(layer.post_feedforward_norm), eps, false, branch, stream);
        ops::residual_add(branch, residual, stream);
        ops::mul_scalar(residual, model.layer_scalar(layer.layer_scalar), stream);
    }

    ops::rmsnorm(residual, model.tensor(weights.final_norm), eps, false, normed, stream);
    ops::linear(normed, to_weight(weights.output_head), logits, stream);
    ops::linear(normed, to_weight(weights.post_projection), hidden_out, stream);
}

void forward_layer(const Model& model, std::size_t layer, const Tensor& hidden_in,
                   std::int32_t first_position, std::int32_t tokens, const PassIndices& indices,
                   KvCache& cache, DeviceArena& arena, Tensor& hidden_out,
                   DeviceExecutionView execution, const Tensor& position_high) {
    const PassSegment segment{&cache, first_position, 0, tokens};
    forward_layer(model, layer, hidden_in, std::span(&segment, 1), indices, arena, hidden_out,
                  execution, position_high);
}

void forward_layer(const Model& model, std::size_t layer, const Tensor& hidden_in,
                   std::span<const PassSegment> segments, const PassIndices& indices,
                   DeviceArena& arena, Tensor& hidden_out, DeviceExecutionView execution,
                   const Tensor& position_high) {
    const std::int32_t tokens = static_cast<std::int32_t>(indices.positions.ne[0]);
    check_segments(segments, tokens, "forward_layer");
    if (indices.positions.dtype != DType::I32 || indices.ring_slots.dtype != DType::I32 ||
        indices.ring_slots.ne[0] != tokens) {
        throw std::invalid_argument("forward_layer: indices must be I32 [tokens]");
    }
    if (position_high.data != nullptr &&
        (segments.size() != 1 || position_high.dtype != DType::I32 ||
         position_high.ne[0] != tokens || position_high.ne[1] != 1)) {
        throw std::invalid_argument(
            "forward_layer: position_high must be I32 [tokens,1] over a single segment");
    }
    if (model.config().sliding_attention(layer)) {
        forward_sliding_layer(model, layer, hidden_in, segments, tokens, indices, arena, hidden_out,
                              execution, position_high);
    } else {
        forward_global_layer(model, layer, hidden_in, segments, tokens, indices, arena, hidden_out,
                             execution);
    }
}

} // namespace ninfer::models::gemma4
