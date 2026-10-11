#include "models/gemma4/forward.h"

#include "core/device.h"
#include "ninfer/ops/causal_compact_attention.h"
#include "ninfer/ops/compact_kv_rows.h"
#include "ninfer/ops/gelu_mul.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/mul_scalar.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/scale_columns.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/soft_cap.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/sliding_causal_attention.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace ninfer::models::gemma4 {
namespace {

struct Scratch {
    Tensor normed;      // [H,T]     the normed layer input, reused by both sublayers
    Tensor branch;      // [H,T]     a sublayer's normalized output before the residual add
    Tensor q;           // [Hq*D,T]  raw query projection
    Tensor k;           // [Hkv*D,T]
    Tensor v;           // [Hkv*D,T]
    Tensor qkv;         // the one frame q, k and v are row views of
    Tensor q_heads;     // [D,Hq,T]  the same storage, as heads for the norms and RoPE
    Tensor k_heads;     // [D,Hkv,T]
    Tensor v_heads;     // [D,Hkv,T]
    Tensor q_normed;    // [D,Hq,T]
    Tensor k_normed;    // [D,Hkv,T]
    Tensor v_normed;    // [D,Hkv,T]
    Tensor attended;    // [Hq*D,T]
    Tensor projected;   // [H,T]     the attention output projection
    Tensor gate;        // [I,T]     the MLP's gate projection
    Tensor up;          // [I,T]     the MLP's up projection
    Tensor activated;   // [I,T]
    Tensor down;        // [H,T]
};

// I32 [n] as BF16 [2,n], so `scatter`'s exact column copy moves whole 32-bit positions.
Tensor positions_as_pairs(const Tensor& positions) {
    return Tensor(positions.data, DType::BF16, {2, static_cast<std::int32_t>(positions.ne[0])});
}

// Writes the `kept` trailing tokens' rows of `rows` (BF16 [width,tokens]) and their positions into a
// cache layer whose slot for each token `slots` names: rows into `cache_rows` (BF16 [width,slots]),
// positions into `cache_positions` (I32 [slots]). Every index is on the device.
void write_cache_rows(const Tensor& rows, const PassIndices& indices, const Tensor& slots,
                      std::int32_t kept, const Tensor& cache_rows, const Tensor& cache_positions,
                      cudaStream_t stream) {
    const std::int32_t tokens = static_cast<std::int32_t>(rows.ne[1]);
    const std::int32_t skip   = tokens - kept;
    const Tensor index        = slots.slice(0, skip, kept);
    Tensor destination        = cache_rows;
    ops::scatter(rows.slice(1, skip, kept), index, destination, stream);
    Tensor position_destination = positions_as_pairs(cache_positions);
    ops::scatter(positions_as_pairs(indices.positions).slice(1, skip, kept), index,
                 position_destination, stream);
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
    const std::uint64_t slide_q  = static_cast<std::uint64_t>(config.num_attention_heads) *
                                   config.sliding.head_dim;
    const std::uint64_t slide_kv = static_cast<std::uint64_t>(config.sliding.num_key_value_heads) *
                                   config.sliding.head_dim;
    const std::uint64_t global_q  = static_cast<std::uint64_t>(config.num_attention_heads) *
                                    config.global.shared.head_dim;
    const std::uint64_t global_kv =
        static_cast<std::uint64_t>(config.global.shared.num_key_value_heads) *
        config.global.shared.head_dim;
    const std::uint64_t compact = (config.global.shared.head_dim + 2 * config.global.rope_angles) *
                                  config.global.shared.num_key_value_heads;

    // Every buffer either kind allocates, counted exactly so the bound cannot drift from the scratch:
    //   hidden-sized:        normed, branch, projected, down
    //   intermediate-sized:  gate, up, activated
    //   query-sized:         q, q_normed, attended
    //   key/value-sized:     k, value, k_normed
    //   compact (global):    the R + 2P by Hkv compact row
    // plus the pass's two index vectors. Every buffer scales with the batch. Sliding layers add a v
    // projection on top of the key/value buffers.
    // The arena aligns every allocation to 256 bytes, so the bound carries that per slice, and an
    // undercount makes the arena refuse an allocation rather than overrun.
    const std::uint64_t sliding = 4 * h + 3 * slide_q + 4 * slide_kv + 3 * i;
    const std::uint64_t global  = 4 * h + 3 * global_q + 3 * global_kv + compact + 3 * i;
    if (tokens < 1) throw std::invalid_argument("layer_workspace_bytes: tokens must be positive");
    const std::uint64_t t = static_cast<std::uint64_t>(tokens);
    // The attention's key split needs its own workspace when a pass has few rows. Its size depends on
    // the pass width, not the key count once the keys are many, so the largest over every width up
    // to `tokens` against an unbounded cache bounds it.
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
        split = std::max(split, ops::sliding_causal_attention_workspace_bytes(sliding_heads, width,
                                                                              1 << 24, 1));
        split = std::max(split, ops::causal_compact_attention_workspace_bytes(
                                    global_heads, global_heads.head_dim, width, 1 << 24, 1));
    }
    return static_cast<std::size_t>(std::max(sliding, global) * t * 2 + 18 * 256 + t * 8) + split;
}

PassIndices write_pass_indices(const KvCache& cache, std::int32_t first_position,
                               std::int32_t tokens, DeviceArena& arena, cudaStream_t stream) {
    if (tokens < 1) throw std::invalid_argument("write_pass_indices: tokens must be positive");
    if (first_position < 0) {
        throw std::invalid_argument("write_pass_indices: positions must not be negative");
    }
    PassIndices indices;
    indices.positions  = arena.alloc(DType::I32, {tokens});
    indices.ring_slots = arena.alloc(DType::I32, {tokens});
    const std::int32_t ring = cache.ring_tokens();
    std::vector<std::int32_t> host(2 * static_cast<std::size_t>(tokens));
    for (std::int32_t token = 0; token < tokens; ++token) {
        host[static_cast<std::size_t>(token)] = first_position + token;
        host[static_cast<std::size_t>(tokens + token)] = (first_position + token) % ring;
    }
    const std::size_t bytes = static_cast<std::size_t>(tokens) * sizeof(std::int32_t);
    CUDA_CHECK(cudaMemcpyAsync(indices.positions.data, host.data(), bytes, cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(indices.ring_slots.data, host.data() + tokens, bytes,
                               cudaMemcpyHostToDevice, stream));
    return indices;
}

void forward_sliding_layer(const Model& model, std::size_t layer, const Tensor& hidden_in,
                           std::int32_t first_position, std::int32_t tokens,
                           const PassIndices& indices, KvCache& cache, DeviceArena& arena,
                           Tensor& hidden_out, DeviceExecutionView execution,
                           const Tensor& position_high) {
    const cudaStream_t stream = execution.stream;
    const TextConfig& config  = model.config();
    if (layer >= model.weights().text.layers.size()) {
        throw std::invalid_argument("forward_sliding_layer: layer index out of range");
    }
    const LayerWeights& weights = model.weights().text.layers[layer];
    if (weights.mixer != MixerKind::SlidingAttention) {
        throw std::invalid_argument("forward_sliding_layer: layer is not a sliding layer");
    }
    if (tokens < 1) { throw std::invalid_argument("forward_sliding_layer: tokens must be positive"); }

    // The scratch is a per-call frame: take a scope so a caller can hand the same arena to every
    // layer of a pass, and so `sliding_layer_workspace_bytes` bounds one call rather than all of them.
    DeviceArena::Scope scope = arena.scope();

    const std::int32_t h   = static_cast<std::int32_t>(config.hidden_size);
    const std::int32_t i   = static_cast<std::int32_t>(config.intermediate_size);
    const std::int32_t d   = static_cast<std::int32_t>(config.sliding.head_dim);
    const std::int32_t hq  = static_cast<std::int32_t>(config.num_attention_heads);
    const std::int32_t hkv = static_cast<std::int32_t>(config.sliding.num_key_value_heads);
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
    scratch.normed    = arena.alloc(DType::BF16, {h, tokens});
    scratch.branch    = arena.alloc(DType::BF16, {h, tokens});
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

    Tensor residual = hidden_in;
    auto to_weight  = [&](WeightId id) { return ops::prepare_linear_weight(model.input(id)).weight; };

    // Attention sublayer. The key and value sets are the token itself, so the same position vector
    // serves both sides of the causal window.
    ops::rmsnorm(residual, model.tensor(weights.input_norm), eps, false, scratch.normed, stream);
    if (fuse_qkv) {
        ops::linear(scratch.normed,
                    fused_qkv_weight(query_weight, q_width + 2 * kv_width), scratch.qkv, stream);
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

    // The ring holds the last window plus slack of tokens at position modulo the ring; a slot no token
    // has written carries an invisible position. A pass's keys enter the ring as its last `kept`
    // tokens. A pass of at most slack + 1 tokens writes first and attends to the ring alone: the rows
    // it replaces are a window older than its first query, and the key set stays one. A wider pass
    // attends to the ring as it was plus its own keys as the Op's second key set, and writes
    // afterwards, so no write replaces a key one of its own queries still sees; the stale rows a
    // rolled-back speculation left at its positions are forgotten first.
    Tensor cache_keys   = cache.keys(layer);
    Tensor cache_values = cache.values(layer);
    Tensor cache_pos    = cache.positions(layer);
    const std::int32_t ring = cache.ring_tokens();
    const bool write_first  = tokens <= cache.ring_slack() + 1;
    if (!write_first) cache.forget_stale(layer, first_position, stream);

    // The kept rows go to the slots `indices` names, so the writes carry no host offset; the last
    // ring of a pass has distinct slots.
    const std::int32_t kept = std::min(tokens, ring);
    const std::int32_t row  = d * hkv;
    const auto write_kept   = [&] {
        const Tensor key_rows(cache_keys.data, DType::BF16, {row, ring});
        const Tensor value_rows(cache_values.data, DType::BF16, {row, ring});
        write_cache_rows(scratch.k_normed.view({row, tokens}), indices, indices.ring_slots, kept,
                         key_rows, cache_pos, stream);
        Tensor value_destination = value_rows;
        ops::scatter(scratch.v_normed.view({row, tokens}).slice(1, tokens - kept, kept),
                     indices.ring_slots.slice(0, tokens - kept, kept), value_destination, stream);
    };
    if (write_first) write_kept();

    {
        const ops::AttentionHeadGeometry geometry{d, hq, hkv};
        Tensor q_batch = scratch.q_normed.view({d, hq, tokens, 1});
        // A narrow pass attends to the whole ring, whose unwritten slots are invisible, so its
        // launches are the same at every position. A wide pass runs eagerly and passes only the
        // leading slots the ring has filled before it (one unwritten slot when it is empty).
        const std::int32_t filled = write_first ? ring : std::clamp(first_position, 1, ring);
        Tensor k_ring(static_cast<std::uint8_t*>(cache_keys.data), DType::BF16, {d, hkv, filled, 1});
        Tensor v_ring(static_cast<std::uint8_t*>(cache_values.data), DType::BF16, {d, hkv, filled, 1});
        Tensor position_ring(static_cast<std::uint8_t*>(cache_pos.data), DType::I32, {filled, 1});
        Tensor position_pass  = indices.positions.view({tokens, 1});
        Tensor attended_batch = scratch.attended.view({d, hq, tokens, 1});
        const Tensor none;
        const Tensor k_pass = write_first ? none : scratch.k_normed.view({d, hkv, tokens, 1});
        const Tensor v_pass = write_first ? none : scratch.v_normed.view({d, hkv, tokens, 1});
        const Tensor position_extra = write_first ? none : position_pass;
        const std::size_t split = ops::sliding_causal_attention_workspace_bytes(
            geometry, tokens, filled + (write_first ? 0 : tokens), 1);
        Tensor workspace;
        if (split > 0) workspace = arena.alloc(DType::U8, {static_cast<std::int32_t>(split)});
        // An image's later keys are part of this pass, in the ring (written first) or in the pass's
        // own key set, so its tokens' upper bound always finds them.
        ops::sliding_causal_attention(q_batch, k_ring, v_ring, position_pass, position_high,
                                      position_ring, k_pass, v_pass, position_extra, geometry,
                                      config.sliding_window, config.attention_scale, attended_batch,
                                      workspace, execution);
    }
    if (!write_first) write_kept();
    ops::linear(scratch.attended, to_weight(weights.attention.output), scratch.projected, stream);
    ops::rmsnorm(scratch.projected, model.tensor(weights.post_attention_norm), eps, false,
                 scratch.branch, stream);
    ops::residual_add(scratch.branch, residual, stream);

    // MLP sublayer: the fused gate/up parent, GeGLU, down, then the sandwich and the layer scalar.
    ops::rmsnorm(residual, model.tensor(weights.pre_feedforward_norm), eps, false, scratch.normed,
                 stream);
    ops::linear(scratch.normed, to_weight(weights.mlp.gate), scratch.gate, stream);
    ops::linear(scratch.normed, to_weight(weights.mlp.up), scratch.up, stream);
    ops::gelu_mul(scratch.gate, scratch.up, scratch.activated, stream);
    ops::linear(scratch.activated, to_weight(weights.mlp.down), scratch.down, stream);
    ops::rmsnorm(scratch.down, model.tensor(weights.post_feedforward_norm), eps, false,
                 scratch.branch, stream);
    ops::residual_add(scratch.branch, residual, stream);
    ops::mul_scalar(residual, model.layer_scalar(weights.layer_scalar), stream);

    // The layer's output is the residual stream itself; the next layer applies its own input norm.
    CUDA_CHECK(cudaMemcpyAsync(hidden_out.data, residual.data,
                               static_cast<std::size_t>(h) * static_cast<std::size_t>(tokens) *
                                   sizeof(std::uint16_t),
                               cudaMemcpyDeviceToDevice, stream));
}

void forward_global_layer(const Model& model, std::size_t layer, const Tensor& hidden_in,
                          std::int32_t first_position, std::int32_t tokens,
                          const PassIndices& indices, KvCache& cache, DeviceArena& arena,
                          Tensor& hidden_out, DeviceExecutionView execution) {
    const cudaStream_t stream = execution.stream;
    const TextConfig& config  = model.config();
    if (layer >= model.weights().text.layers.size()) {
        throw std::invalid_argument("forward_global_layer: layer index out of range");
    }
    const LayerWeights& weights = model.weights().text.layers[layer];
    if (weights.mixer != MixerKind::FullAttention) {
        throw std::invalid_argument("forward_global_layer: layer is not a global layer");
    }
    if (tokens < 1) { throw std::invalid_argument("forward_global_layer: tokens must be positive"); }
    // These layers keep one row per token at the token's own position, so the batch must fit.
    if (first_position + tokens > cache.capacity()) {
        throw std::invalid_argument("forward_global_layer: the batch exceeds the cache capacity");
    }

    // The scratch is a per-call frame, exactly as in the sliding path.
    DeviceArena::Scope scope = arena.scope();

    const GlobalAttentionGeometry& geometry = config.global;
    const std::int32_t h     = static_cast<std::int32_t>(config.hidden_size);
    const std::int32_t i     = static_cast<std::int32_t>(config.intermediate_size);
    const std::int32_t d     = static_cast<std::int32_t>(geometry.shared.head_dim);
    const std::int32_t hq    = static_cast<std::int32_t>(config.num_attention_heads);
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
    const Tensor positions = indices.positions.view({tokens, 1, 1});

    Tensor residual = hidden_in;
    auto to_weight  = [&](WeightId id) { return ops::prepare_linear_weight(model.input(id)).weight; };

    ops::rmsnorm(residual, model.tensor(weights.input_norm), eps, false, normed, stream);
    ops::linear(normed, to_weight(weights.attention.query), q, stream);
    ops::linear(normed, to_weight(weights.attention.key), k, stream);

    // K is V on these layers, so the value side is the key normalized without a weight, read from the
    // raw projection; the weighted key norm is a second, independent normalization of the same input.
    const Tensor k_heads = k.view({d, hkv, tokens});
    ops::rmsnorm(k_heads, eps, value, stream);
    ops::rmsnorm(q.view({d, hq, tokens}), model.tensor(weights.attention.query_norm), eps, false,
                 q_normed, stream);
    ops::rmsnorm(k_heads, model.tensor(weights.attention.key_norm), eps, false, k_normed, stream);

    // Proportional RoPE: `rope_angles` pairs taken against the full head width, so the pair partner of
    // dim i is i + d/2 and the dims outside those two runs carry no rotation.
    ops::rope(positions, d, pairs, geometry.shared.rope_theta, q_normed, execution);
    ops::rope(positions, d, pairs, geometry.shared.rope_theta, k_normed, execution);

    // On the non-rotated dims the key the score needs is the value times the key norm's weight, so the
    // query carries that weight instead: q[d] * w_kn[d] against the stored value.
    const Tensor& w_kn = model.tensor(weights.attention.key_norm);
    ops::scale_columns(w_kn, q_normed, pairs, d / 2, stream);
    ops::scale_columns(w_kn, q_normed, d / 2 + pairs, d, stream);

    ops::compact_kv_rows(value, k_normed, compact, d, pairs, stream);

    // The compact row is written at the token's own position, the slot the positions themselves
    // name, and read back causally, so a slot no token has written carries an invisible position and
    // the whole cache can be handed to the Op.
    Tensor cache_rows = cache.keys(layer);
    Tensor cache_pos  = cache.positions(layer);
    const std::int32_t capacity = cache.capacity();
    write_cache_rows(compact.view({width * hkv, tokens}), indices, indices.positions, tokens,
                     Tensor(cache_rows.data, DType::BF16, {width * hkv, capacity}), cache_pos,
                     stream);

    {
        const ops::AttentionHeadGeometry heads{d, hq, hkv};
        Tensor q_batch        = q_normed.view({d, hq, tokens, 1});
        // The Op is handed the whole capacity: with `indexed_keys` it stops at the latest query's
        // position on the device, so the launches do not depend on how much of the cache is filled.
        Tensor kv_batch(static_cast<std::uint8_t*>(cache_rows.data), DType::BF16,
                        {width, hkv, capacity, 1});
        Tensor attended_batch = attended.view({d, hq, tokens, 1});
        Tensor position_q     = positions.view({tokens, 1});
        Tensor position_k(static_cast<std::uint8_t*>(cache_pos.data), DType::I32, {capacity, 1});
        const std::size_t split =
            ops::causal_compact_attention_workspace_bytes(heads, d, tokens, capacity, 1);
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

    ops::rmsnorm(residual, model.tensor(weights.pre_feedforward_norm), eps, false, normed, stream);
    ops::linear(normed, to_weight(weights.mlp.gate), gate, stream);
    ops::linear(normed, to_weight(weights.mlp.up), up, stream);
    ops::gelu_mul(gate, up, activated, stream);
    ops::linear(activated, to_weight(weights.mlp.down), down, stream);
    ops::rmsnorm(down, model.tensor(weights.post_feedforward_norm), eps, false, branch, stream);
    ops::residual_add(branch, residual, stream);
    ops::mul_scalar(residual, model.layer_scalar(weights.layer_scalar), stream);

    CUDA_CHECK(cudaMemcpyAsync(hidden_out.data, residual.data,
                               static_cast<std::size_t>(h) * static_cast<std::size_t>(tokens) *
                                   sizeof(std::uint16_t),
                               cudaMemcpyDeviceToDevice, stream));
}

void forward_head(const Model& model, const Tensor& hidden_in, std::int32_t tokens, DeviceArena& arena,
                  Tensor& logits, DeviceExecutionView execution, Tensor* normed_out) {
    const cudaStream_t stream = execution.stream;
    const TextConfig& config  = model.config();

    DeviceArena::Scope scope = arena.scope();
    const std::int32_t h = static_cast<std::int32_t>(config.hidden_size);

    // The caller's buffer may be wider than this pass: its leading columns are contiguous.
    Tensor normed = normed_out != nullptr ? Tensor(normed_out->data, DType::BF16, {h, tokens})
                                          : arena.alloc(DType::BF16, {h, tokens});
    ops::rmsnorm(hidden_in, model.tensor(model.weights().text.final_norm), config.rms_norm_eps, false,
                 normed, stream);
    auto weight = ops::prepare_linear_weight(model.input(model.weights().text.output_head)).weight;
    ops::linear(normed, weight, logits, stream);
    ops::soft_cap(logits, config.final_logit_softcapping, stream);
}

void forward_draft(const Model& model, const Tensor& input, std::int32_t position,
                   const KvCache& cache, DeviceArena& arena, Tensor& logits, Tensor& hidden_out,
                   DeviceExecutionView execution) {
    const cudaStream_t stream = execution.stream;
    const TextConfig& target  = model.config();
    if (!model.draft_config() || !model.weights().draft) {
        throw std::logic_error("forward_draft: the drafter is not loaded");
    }
    const DraftConfig& draft    = *model.draft_config();
    const DraftWeights& weights = *model.weights().draft;
    // The drafter reads keys strictly before its token, so a token at position 0 has none.
    if (position < 1 || position > cache.capacity()) {
        throw std::invalid_argument("forward_draft: position must be in [1, capacity]");
    }

    DeviceArena::Scope scope = arena.scope();
    const std::int32_t h  = static_cast<std::int32_t>(draft.hidden_size);
    const std::int32_t i  = static_cast<std::int32_t>(draft.intermediate_size);
    const std::int32_t hq = static_cast<std::int32_t>(target.num_attention_heads);
    const float eps       = draft.rms_norm_eps;
    auto to_weight = [&](WeightId id) { return ops::prepare_linear_weight(model.input(id)).weight; };

    // Two position vectors: RoPE rotates at the token's own position, while visibility is that of the
    // previous position, which admits exactly the keys before the token. A row the target wrote at or
    // past the token (a rejected draft's, or none at all) stays invisible either way.
    Tensor positions = arena.alloc(DType::I32, {2, 1, 1});
    const std::int32_t host_positions[2] = {position, position - 1};
    CUDA_CHECK(cudaMemcpyAsync(positions.data, host_positions, sizeof(host_positions),
                               cudaMemcpyHostToDevice, stream));
    // The pageable source is staged before the call returns, so the stack array may go.
    const Tensor rope_position(positions.data, DType::I32, {1, 1, 1});
    const Tensor visible_position(static_cast<std::uint8_t*>(positions.data) + sizeof(std::int32_t),
                                  DType::I32, {1, 1});

    Tensor residual  = arena.alloc(DType::BF16, {h, 1});
    Tensor normed    = arena.alloc(DType::BF16, {h, 1});
    Tensor branch    = arena.alloc(DType::BF16, {h, 1});
    Tensor projected = arena.alloc(DType::BF16, {h, 1});
    Tensor gate      = arena.alloc(DType::BF16, {i, 1});
    Tensor up        = arena.alloc(DType::BF16, {i, 1});
    Tensor activated = arena.alloc(DType::BF16, {i, 1});
    Tensor down      = arena.alloc(DType::BF16, {h, 1});
    const std::int32_t widest_q = hq * static_cast<std::int32_t>(
        std::max(target.sliding.head_dim, target.global.shared.head_dim));
    Tensor q_storage        = arena.alloc(DType::BF16, {widest_q, 1});
    Tensor q_normed_storage = arena.alloc(DType::BF16, {widest_q, 1});
    Tensor attended_storage = arena.alloc(DType::BF16, {widest_q, 1});

    ops::linear(input, to_weight(weights.pre_projection), residual, stream);

    for (const DraftLayerWeights& layer : weights.layers) {
        const bool sliding = layer.mixer == MixerKind::SlidingAttention;
        const auto& geometry = sliding ? target.sliding : target.global.shared;
        const std::int32_t d   = static_cast<std::int32_t>(geometry.head_dim);
        const std::int32_t hkv = static_cast<std::int32_t>(geometry.num_key_value_heads);
        // The buffers are sized for the wider global head; a sliding layer uses their prefix.
        Tensor q(q_storage.data, DType::BF16, {hq * d, 1});
        Tensor q_normed(q_normed_storage.data, DType::BF16, {d, hq, 1});
        Tensor attended(attended_storage.data, DType::BF16, {hq * d, 1});

        ops::rmsnorm(residual, model.tensor(layer.input_norm), eps, false, normed, stream);
        ops::linear(normed, to_weight(layer.query), q, stream);
        ops::rmsnorm(q.view({d, hq, 1}), model.tensor(layer.query_norm), eps, false, q_normed,
                     stream);
        const ops::AttentionHeadGeometry heads{d, hq, hkv};
        if (sliding) {
            const std::size_t source = draft.target_sliding_layer;
            ops::rope(rope_position, d, d / 2, geometry.rope_theta, q_normed, execution);
            const Tensor keys      = cache.keys(source);
            const Tensor values    = cache.values(source);
            const Tensor key_pos   = cache.positions(source);
            const std::int32_t ring = cache.ring_tokens();
            const Tensor k_ring(keys.data, DType::BF16, {d, hkv, ring, 1});
            const Tensor v_ring(values.data, DType::BF16, {d, hkv, ring, 1});
            const Tensor p_ring(key_pos.data, DType::I32, {ring, 1});
            const Tensor none;
            Tensor out = attended.view({d, hq, 1, 1});
            const std::size_t split =
                ops::sliding_causal_attention_workspace_bytes(heads, 1, ring, 1);
            Tensor workspace;
            if (split > 0) workspace = arena.alloc(DType::U8, {static_cast<std::int32_t>(split)});
            ops::sliding_causal_attention(q_normed.view({d, hq, 1, 1}), k_ring, v_ring,
                                          visible_position, none, p_ring, none, none, none, heads,
                                          target.sliding_window, target.attention_scale, out,
                                          workspace, execution);
        } else {
            const std::size_t source = draft.target_global_layer;
            const std::int32_t pairs = static_cast<std::int32_t>(target.global.rope_angles);
            const std::int32_t width = d + 2 * pairs;
            ops::rope(rope_position, d, pairs, geometry.rope_theta, q_normed, execution);
            // The compact rows hold the weightless value and the rotated key; on the dims that do not
            // rotate, the key the target scores against is the value times the source layer's key-norm
            // weight, which the query carries instead, exactly as in the target's own global layer.
            const Tensor& w_kn = model.tensor(target_layer_key_norm(model, source));
            ops::scale_columns(w_kn, q_normed, pairs, d / 2, stream);
            ops::scale_columns(w_kn, q_normed, d / 2 + pairs, d, stream);
            const Tensor rows    = cache.keys(source);
            const Tensor row_pos = cache.positions(source);
            const Tensor kv(rows.data, DType::BF16, {width, hkv, position, 1});
            const Tensor p_rows(row_pos.data, DType::I32, {position, 1});
            Tensor out = attended.view({d, hq, 1, 1});
            const std::size_t split =
                ops::causal_compact_attention_workspace_bytes(heads, d, 1, position, 1);
            Tensor workspace;
            if (split > 0) workspace = arena.alloc(DType::U8, {static_cast<std::int32_t>(split)});
            ops::causal_compact_attention(q_normed.view({d, hq, 1, 1}), kv, visible_position, p_rows,
                                          heads, d, pairs, target.attention_scale, true, out,
                                          workspace, execution);
        }
        ops::linear(attended, to_weight(layer.output), projected, stream);
        ops::rmsnorm(projected, model.tensor(layer.post_attention_norm), eps, false, branch, stream);
        ops::residual_add(branch, residual, stream);

        ops::rmsnorm(residual, model.tensor(layer.pre_feedforward_norm), eps, false, normed, stream);
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
    if (indices.positions.dtype != DType::I32 || indices.positions.ne[0] != tokens ||
        indices.ring_slots.dtype != DType::I32 || indices.ring_slots.ne[0] != tokens) {
        throw std::invalid_argument("forward_layer: indices must be I32 [tokens]");
    }
    if (position_high.data != nullptr &&
        (position_high.dtype != DType::I32 || position_high.ne[0] != tokens ||
         position_high.ne[1] != 1)) {
        throw std::invalid_argument("forward_layer: position_high must be I32 [tokens,1]");
    }
    if (model.config().sliding_attention(layer)) {
        forward_sliding_layer(model, layer, hidden_in, first_position, tokens, indices, cache,
                              arena, hidden_out, execution, position_high);
    } else {
        forward_global_layer(model, layer, hidden_in, first_position, tokens, indices, cache,
                             arena, hidden_out, execution);
    }
}

} // namespace ninfer::models::gemma4
