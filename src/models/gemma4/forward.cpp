#include "models/gemma4/forward.h"

#include "core/device.h"
#include "ninfer/ops/causal_compact_attention.h"
#include "ninfer/ops/compact_kv_rows.h"
#include "ninfer/ops/gelu_mul.h"
#include "ninfer/ops/gemma_moe_experts.h"
#include "ninfer/ops/gemma_moe_router.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/mul_scalar.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/scale_columns.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/soft_cap.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/sliding_causal_attention.h"
#include "ninfer/ops/weight_input.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::models::gemma4 {
namespace {

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
    Tensor activated; // [I,T]
    Tensor down;      // [H,T]
    // The MoE block's router output, allocated only when the model enables the block.
    Tensor moe_ids;     // I32 [K,T]  one chosen expert per row
    Tensor moe_weights; // FP32 [K,T] the router's weights, renormalized and expert-scaled
};

// I32 [n] as BF16 [2,n], so `scatter`'s exact column copy moves whole 32-bit positions.
Tensor positions_as_pairs(const Tensor& positions) {
    return Tensor(positions.data, DType::BF16, {2, static_cast<std::int32_t>(positions.ne[0])});
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
    const Tensor index        = slots.slice(0, skip, kept);
    Tensor destination        = cache_rows;
    ops::scatter(rows.slice(1, skip, kept), index, destination, stream);
    Tensor position_destination = positions_as_pairs(cache_positions);
    ops::scatter(positions_as_pairs(positions).slice(1, skip, kept), index, position_destination,
                 stream);
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

// The attention input projections are declared as one group by the converter
// (tools/convert/gemma4.py `group(query, key, value)`) and bound as row slices of a single packed
// object. When those slices are contiguous from the object's first row, one Linear over the union
// runs exactly the rows the three separate launches run, through the same route: a Q6 GEMV splits a
// row's K range from K and the split index alone, so which block computes a row does not change its
// reduction order. The three results are bit-identical -- the generated token ids of a greedy
// decode match the unfused build exactly -- so this trades 150 launches a token for 50 without
// touching arithmetic.
//
// The check below is by plane offset and covers the row-split Q6 layout only; any other format,
// a differing quantisation, or a slice that is not exactly where it should be keeps the three
// separate launches. `prepare_linear_weight` returns a copy of the stored weight, so the addresses
// compared here are the artifact's own slice addresses.
bool qkv_slices_are_contiguous(const Weight& query, const Weight& key, const Weight& value) {
    if (query.qtype != QType::Q6_G64_FP16 || query.layout != QuantLayout::RowSplit ||
        query.group != 64) {
        return false;
    }
    const Weight* members[] = {&query, &key, &value};
    for (const Weight* member : members) {
        if (member->qtype != query.qtype || member->layout != query.layout ||
            member->group != query.group || member->scale_dtype != query.scale_dtype ||
            member->k != query.k || member->padded_shape[1] != query.padded_shape[1] ||
            member->weight_scale_divisor != query.weight_scale_divisor ||
            member->input_scale_divisor != query.input_scale_divisor) {
            return false;
        }
        if (member->qdata == nullptr || member->qhigh == nullptr || member->scales == nullptr) {
            return false;
        }
    }
    // Per-group plane widths of the row-split Q6 layout, from q6_a16_simt.cuh: a row's group
    // index advances by padded_k/64 in a 32-byte code plane, a 16-byte high plane and a 2-byte
    // scale plane.
    const std::uint64_t groups = static_cast<std::uint64_t>(query.padded_shape[1]) / 64;
    const std::uint64_t rows_q = static_cast<std::uint64_t>(query.n);
    const std::uint64_t rows_k = static_cast<std::uint64_t>(key.n);
    const auto is_at = [](const void* base, std::uint64_t groups_ahead, std::uint64_t per_group,
                          const void* expected) {
        return static_cast<const void*>(static_cast<const std::uint8_t*>(base) +
                                        groups_ahead * per_group) == expected;
    };
    return is_at(query.qdata, rows_q * groups, 32, key.qdata) &&
           is_at(query.qhigh, rows_q * groups, 16, key.qhigh) &&
           is_at(query.scales, rows_q * groups, 2, key.scales) &&
           is_at(key.qdata, rows_k * groups, 32, value.qdata) &&
           is_at(key.qhigh, rows_k * groups, 16, value.qhigh) &&
           is_at(key.scales, rows_k * groups, 2, value.scales);
}

// The same weight with the union row count. `query`'s planes begin at the object's first row once
// `qkv_slices_are_contiguous` holds, so only `n` changes.
Weight fused_qkv_weight(const Weight& query, std::int32_t rows) {
    Weight fused = query;
    fused.n      = rows;
    return fused;
}

} // namespace

std::size_t layer_workspace_bytes(const TextConfig& config, std::int32_t tokens) {
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
    // The MoE block's router output -- K expert ids and K FP32 weights per token -- plus the
    // alignment of its two slices.
    const std::size_t route = config.enable_moe_block
                                  ? 2 * static_cast<std::size_t>(config.top_k_experts) *
                                            static_cast<std::size_t>(t) * sizeof(float) +
                                        2 * 256
                                  : 0;
    return static_cast<std::size_t>(std::max(sliding, global) * t * 2 + 18 * 256 + t * 8) + split +
           route;
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
};

ops::GemmaMoeRouterWeights router_weights(const Model& model, const MoeWeights& moe) {
    return {ops::prepare_linear_weight(model.input(moe.router)).weight,
            model.tensor(moe.router_scale), model.tensor(moe.per_expert_scale)};
}

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
// The MoE branch reuses `normed`, `projected` and `down` in place rather than three more [H,T]
// buffers: the dense branch has already consumed `normed` (gate/up) and `down` (its own post-norm),
// and `projected` held the attention output, which was normed into `branch`. Only the router's two
// outputs are new storage, and the caller allocates them when the block is enabled.
void feed_forward_sublayer(const Model& model, const LayerWeights& weights, FfnScratch s,
                           Tensor& moe_ids, Tensor& moe_weights, Tensor& residual,
                           cudaStream_t stream) {
    const TextConfig& config = model.config();
    const float eps          = config.rms_norm_eps;
    auto to_weight           = [&](WeightId id) {
        return ops::prepare_linear_weight(model.input(id)).weight;
    };

    ops::rmsnorm(residual, model.tensor(weights.pre_feedforward_norm), eps, false, s.normed,
                 stream);
    ops::linear(s.normed, to_weight(weights.mlp.gate), s.gate, stream);
    ops::linear(s.normed, to_weight(weights.mlp.up), s.up, stream);
    ops::gelu_mul(s.gate, s.up, s.activated, stream);
    ops::linear(s.activated, to_weight(weights.mlp.down), s.down, stream);

    if (!weights.moe) {
        ops::rmsnorm(s.down, model.tensor(weights.post_feedforward_norm), eps, false, s.branch,
                     stream);
        ops::residual_add(s.branch, residual, stream);
        ops::mul_scalar(residual, model.layer_scalar(weights.layer_scalar), stream);
        return;
    }

    const MoeWeights& moe = *weights.moe;
    // h1: the dense branch through its own post-norm. The MoE branch reads the pre-FFN residual
    // directly -- the router weightlessly, the experts through their own second pre-norm.
    ops::rmsnorm(s.down, model.tensor(moe.post_feedforward_norm_1), eps, false, s.branch, stream);
    ops::gemma_moe_router(residual, router_weights(model, moe), router_geometry(config), moe_ids,
                          moe_weights, stream);
    ops::rmsnorm(residual, model.tensor(moe.pre_feedforward_norm_2), eps, false, s.normed, stream);
    ops::gemma_moe_experts(s.normed, moe_ids, moe_weights, expert_weights(model, moe),
                           expert_geometry(config), s.projected, stream);
    // h2, then h = h1 + h2.
    ops::rmsnorm(s.projected, model.tensor(moe.post_feedforward_norm_2), eps, false, s.down,
                 stream);
    ops::residual_add(s.down, s.branch, stream);
    // The shared post-norm, the residual add and the scalar, exactly as the dense path ends.
    ops::rmsnorm(s.branch, model.tensor(weights.post_feedforward_norm), eps, false, s.normed,
                 stream);
    ops::residual_add(s.normed, residual, stream);
    ops::mul_scalar(residual, model.layer_scalar(weights.layer_scalar), stream);
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
    // The union is one frame the three can be viewed from only at T=1, which is why decode fuses
    // and every wider pass keeps the three launches: tensors are column-major with dim 0 fastest,
    // so a
    // row slice of a [Q+2KV,T] frame carries that frame's token stride and is not contiguous for
    // T>1 -- which rmsnorm requires. At T=1 the frame's row 0 is byte 0, so a [rows,1] slice is
    // exactly a standalone [rows,1] tensor. See qkv_slices_are_contiguous for the weight side.
    const Weight query_weight =
        ops::prepare_linear_weight(model.input(weights.attention.query)).weight;
    const Weight key_weight = ops::prepare_linear_weight(model.input(weights.attention.key)).weight;
    const Weight value_weight =
        ops::prepare_linear_weight(model.input(weights.attention.value)).weight;
    const bool fuse_qkv =
        tokens == 1 && qkv_slices_are_contiguous(query_weight, key_weight, value_weight);
    scratch.normed = arena.alloc(DType::BF16, {h, tokens});
    scratch.branch = arena.alloc(DType::BF16, {h, tokens});
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
    scratch.gate      = arena.alloc(DType::BF16, {i, tokens});
    scratch.up        = arena.alloc(DType::BF16, {i, tokens});
    scratch.activated = arena.alloc(DType::BF16, {i, tokens});
    scratch.down      = arena.alloc(DType::BF16, {h, tokens});
    if (weights.moe) {
        const std::int32_t k = static_cast<std::int32_t>(config.top_k_experts);
        scratch.moe_ids      = arena.alloc(DType::I32, {k, tokens});
        scratch.moe_weights  = arena.alloc(DType::FP32, {k, tokens});
    }

    Tensor residual = hidden_in;
    auto to_weight  = [&](WeightId id) {
        return ops::prepare_linear_weight(model.input(id)).weight;
    };

    // Attention sublayer. The key and value sets are the token itself, so the same position vector
    // serves both sides of the causal window.
    ops::rmsnorm(residual, model.tensor(weights.input_norm), eps, false, scratch.normed, stream);
    if (fuse_qkv) {
        ops::linear(scratch.normed, fused_qkv_weight(query_weight, q_width + 2 * kv_width),
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
        Tensor cache_keys                  = cache.keys(layer);
        Tensor cache_values                = cache.values(layer);
        Tensor cache_pos                   = cache.positions(layer);
        const std::int32_t ring            = cache.ring_tokens();
        const bool write_first             = n <= cache.ring_slack() + 1;
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
            const Tensor key_rows(cache_keys.data, DType::BF16, {row, ring});
            const Tensor value_rows(cache_values.data, DType::BF16, {row, ring});
            write_cache_rows(k_rows, positions, slots, kept, key_rows, cache_pos, stream);
            Tensor value_destination = value_rows;
            ops::scatter(v_rows.slice(1, n - kept, kept), slots.slice(0, n - kept, kept),
                         value_destination, stream);
        };
        if (write_first) write_kept();

        const ops::AttentionHeadGeometry geometry{d, hq, hkv};
        const std::size_t q_column = static_cast<std::size_t>(q_width) * sizeof(std::uint16_t);
        Tensor q_batch(column_data(scratch.q_normed, q_column, column), DType::BF16, {d, hq, n, 1});
        // A narrow pass attends to the whole ring, whose unwritten slots are invisible, so its
        // launches are the same at every position. A wide pass runs eagerly and passes only the
        // leading slots the ring has filled before it (one unwritten slot when it is empty).
        const std::int32_t filled =
            write_first ? ring : std::clamp(segment.first_position, 1, ring);
        Tensor k_ring(static_cast<std::uint8_t*>(cache_keys.data), DType::BF16,
                      {d, hkv, filled, 1});
        Tensor v_ring(static_cast<std::uint8_t*>(cache_values.data), DType::BF16,
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
    ops::rmsnorm(scratch.projected, model.tensor(weights.post_attention_norm), eps, false,
                 scratch.branch, stream);
    ops::residual_add(scratch.branch, residual, stream);

    // MLP sublayer: the dense branch, and the MoE block beside it when the model has one.
    feed_forward_sublayer(model, weights,
                          {scratch.normed, scratch.branch, scratch.projected, scratch.down,
                           scratch.gate, scratch.up, scratch.activated},
                          scratch.moe_ids, scratch.moe_weights, residual, stream);

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
    Tensor gate      = arena.alloc(DType::BF16, {i, tokens});
    Tensor up        = arena.alloc(DType::BF16, {i, tokens});
    Tensor activated = arena.alloc(DType::BF16, {i, tokens});
    Tensor moe_ids;     // I32 [K,T]  the router's selection, when the model enables the block
    Tensor moe_weights; // FP32 [K,T] its weights
    if (weights.moe) {
        const std::int32_t k = static_cast<std::int32_t>(config.top_k_experts);
        moe_ids              = arena.alloc(DType::I32, {k, tokens});
        moe_weights          = arena.alloc(DType::FP32, {k, tokens});
    }
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
        Tensor cache_rows                  = cache.keys(layer);
        Tensor cache_pos                   = cache.positions(layer);
        const std::int32_t capacity        = cache.capacity();
        const Tensor segment_positions     = indices.positions.slice(0, column, n);
        write_cache_rows(
            Tensor(column_data(compact, compact_column, column), DType::BF16, {width * hkv, n}),
            segment_positions, segment_positions, n,
            Tensor(cache_rows.data, DType::BF16, {width * hkv, capacity}), cache_pos, stream);

        const ops::AttentionHeadGeometry heads{d, hq, hkv};
        Tensor q_batch(column_data(q_normed, q_column, column), DType::BF16, {d, hq, n, 1});
        // The Op is handed the whole capacity: with `indexed_keys` it stops at the latest query's
        // position on the device, so the launches do not depend on how much of the cache is filled.
        Tensor kv_batch(static_cast<std::uint8_t*>(cache_rows.data), DType::BF16,
                        {width, hkv, capacity, 1});
        Tensor attended_batch(column_data(attended, q_column, column), DType::BF16, {d, hq, n, 1});
        Tensor position_q = segment_positions.view({n, 1});
        Tensor position_k(static_cast<std::uint8_t*>(cache_pos.data), DType::I32, {capacity, 1});
        const std::size_t split =
            ops::causal_compact_attention_workspace_bytes(heads, d, n, capacity, 1);
        Tensor workspace;
        if (split > 0) workspace = arena.alloc(DType::U8, {static_cast<std::int32_t>(split)});
        // Rows sit at their own positions, which is what `indexed_keys` states.
        ops::causal_compact_attention(q_batch, kv_batch, position_q, position_k, heads, d, pairs,
                                      config.attention_scale, true, attended_batch, workspace,
                                      execution);
    }
    ops::linear(attended, to_weight(weights.attention.output), projected, stream);
    ops::rmsnorm(projected, model.tensor(weights.post_attention_norm), eps, false, branch, stream);
    ops::residual_add(branch, residual, stream);

    // MLP sublayer: the dense branch, and the MoE block beside it when the model has one.
    feed_forward_sublayer(model, weights, {normed, branch, projected, down, gate, up, activated},
                          moe_ids, moe_weights, residual, stream);

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
                const Tensor keys        = cache.keys(source);
                const Tensor values      = cache.values(source);
                const Tensor key_pos     = cache.positions(source);
                const std::int32_t ring  = cache.ring_tokens();
                const Tensor k_ring(keys.data, DType::BF16, {d, hkv, ring, 1});
                const Tensor v_ring(values.data, DType::BF16, {d, hkv, ring, 1});
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
                const Tensor rows           = cache.keys(source);
                const Tensor row_pos        = cache.positions(source);
                const Tensor kv(rows.data, DType::BF16, {width, hkv, position, 1});
                const Tensor p_rows(row_pos.data, DType::I32, {position, 1});
                const std::size_t split =
                    ops::causal_compact_attention_workspace_bytes(heads, d, 1, position, 1);
                Tensor workspace;
                if (split > 0)
                    workspace = arena.alloc(DType::U8, {static_cast<std::int32_t>(split)});
                ops::causal_compact_attention(q_lane, kv, visible_position(column), p_rows, heads,
                                              d, pairs, target.attention_scale, true, out,
                                              workspace, execution);
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
