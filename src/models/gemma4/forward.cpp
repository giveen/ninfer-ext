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
    Tensor positions;   // I32 [T]
};

// The key-norm weight of a target global layer, which a query attending its compact rows carries.
WeightId target_layer_key_norm(const Model& model, std::size_t layer) {
    return model.weights().text.layers.at(layer).attention.key_norm;
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
    // plus one position vector per token. Every buffer scales with the batch. Sliding layers add a v
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
    return static_cast<std::size_t>(std::max(sliding, global) * t * 2 + 17 * 256 + t * 4) + split;
}

void forward_sliding_layer(const Model& model, std::size_t layer, const Tensor& hidden_in,
                           std::int32_t first_position, std::int32_t tokens, KvCache& cache,
                           DeviceArena& arena, Tensor& hidden_out, DeviceExecutionView execution,
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
    scratch.normed    = arena.alloc(DType::BF16, {h, tokens});
    scratch.branch    = arena.alloc(DType::BF16, {h, tokens});
    scratch.q         = arena.alloc(DType::BF16, {q_width, tokens});
    scratch.k         = arena.alloc(DType::BF16, {kv_width, tokens});
    scratch.v         = arena.alloc(DType::BF16, {kv_width, tokens});
    scratch.q_normed  = arena.alloc(DType::BF16, {d, hq, tokens});
    scratch.k_normed  = arena.alloc(DType::BF16, {d, hkv, tokens});
    scratch.v_normed  = arena.alloc(DType::BF16, {d, hkv, tokens});
    scratch.attended  = arena.alloc(DType::BF16, {q_width, tokens});
    scratch.projected = arena.alloc(DType::BF16, {h, tokens});
    scratch.gate      = arena.alloc(DType::BF16, {i, tokens});
    scratch.up        = arena.alloc(DType::BF16, {i, tokens});
    scratch.activated = arena.alloc(DType::BF16, {i, tokens});
    scratch.down      = arena.alloc(DType::BF16, {h, tokens});
    scratch.positions = arena.alloc(DType::I32, {tokens, 1, 1});

    std::vector<std::int32_t> host_positions(static_cast<std::size_t>(tokens));
    for (std::int32_t token = 0; token < tokens; ++token) {
        host_positions[static_cast<std::size_t>(token)] = first_position + token;
    }
    CUDA_CHECK(cudaMemcpyAsync(scratch.positions.data, host_positions.data(),
                               static_cast<std::size_t>(tokens) * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream));

    Tensor residual = hidden_in;
    auto to_weight  = [&](WeightId id) { return ops::prepare_linear_weight(model.input(id)).weight; };

    // Attention sublayer. The key and value sets are the token itself, so the same position vector
    // serves both sides of the causal window.
    ops::rmsnorm(residual, model.tensor(weights.input_norm), eps, false, scratch.normed, stream);
    ops::linear(scratch.normed, to_weight(weights.attention.query), scratch.q, stream);
    ops::linear(scratch.normed, to_weight(weights.attention.key), scratch.k, stream);
    ops::linear(scratch.normed, to_weight(weights.attention.value), scratch.v, stream);

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
    ops::rope(scratch.positions, d, d / 2, config.sliding.rope_theta, scratch.q_normed,
              scratch.k_normed, execution);

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

    const std::int32_t kept  = std::min(tokens, ring);
    const std::int32_t first = first_position + tokens - kept;
    const std::size_t row_bytes =
        static_cast<std::size_t>(d) * static_cast<std::size_t>(hkv) * sizeof(std::uint16_t);
    // The kept rows are adjacent in the ring unless the run wraps, which splits it into two copies.
    const auto write_kept = [&] {
        const std::int32_t first_slot = cache.slot(layer, first);
        const std::int32_t head_run   = std::min(kept, ring - first_slot);
        const auto copy_run = [&](std::int32_t slot, std::int32_t from, std::int32_t count) {
            const std::size_t source = static_cast<std::size_t>(tokens - kept + from) * row_bytes;
            CUDA_CHECK(cudaMemcpyAsync(static_cast<std::uint8_t*>(cache_keys.data) +
                                           static_cast<std::size_t>(slot) * row_bytes,
                                       static_cast<std::uint8_t*>(scratch.k_normed.data) + source,
                                       static_cast<std::size_t>(count) * row_bytes,
                                       cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(static_cast<std::uint8_t*>(cache_values.data) +
                                           static_cast<std::size_t>(slot) * row_bytes,
                                       static_cast<std::uint8_t*>(scratch.v_normed.data) + source,
                                       static_cast<std::size_t>(count) * row_bytes,
                                       cudaMemcpyDeviceToDevice, stream));
        };
        copy_run(first_slot, 0, head_run);
        if (head_run < kept) copy_run(0, head_run, kept - head_run);
        cache.mark_range(layer, first, kept, stream);
    };
    if (write_first) write_kept();

    {
        const ops::AttentionHeadGeometry geometry{d, hq, hkv};
        Tensor q_batch = scratch.q_normed.view({d, hq, tokens, 1});
        // Until the ring first wraps, only its leading slots hold tokens, and the Op scans every key
        // it is given. An empty ring is passed as one unwritten slot.
        const std::int32_t filled =
            std::clamp(first_position + (write_first ? tokens : 0), 1, ring);
        Tensor k_ring(static_cast<std::uint8_t*>(cache_keys.data), DType::BF16, {d, hkv, filled, 1});
        Tensor v_ring(static_cast<std::uint8_t*>(cache_values.data), DType::BF16, {d, hkv, filled, 1});
        Tensor position_ring(static_cast<std::uint8_t*>(cache_pos.data), DType::I32, {filled, 1});
        Tensor position_pass  = scratch.positions.view({tokens, 1});
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
                                      workspace, stream);
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
                          std::int32_t first_position, std::int32_t tokens, KvCache& cache,
                          DeviceArena& arena, Tensor& hidden_out, DeviceExecutionView execution) {
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
    Tensor positions = arena.alloc(DType::I32, {tokens, 1, 1});

    std::vector<std::int32_t> host_positions(static_cast<std::size_t>(tokens));
    for (std::int32_t token = 0; token < tokens; ++token) {
        host_positions[static_cast<std::size_t>(token)] = first_position + token;
    }
    CUDA_CHECK(cudaMemcpyAsync(positions.data, host_positions.data(),
                               static_cast<std::size_t>(tokens) * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream));

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

    // The compact row is written at the token's own position and read back causally, so a slot no
    // token has written carries an invisible position and the whole cache can be handed to the Op.
    Tensor cache_rows = cache.keys(layer);
    Tensor cache_pos  = cache.positions(layer);
    const std::int32_t capacity  = cache.capacity();
    const std::int32_t row_index = cache.slot(layer, first_position);
    const std::size_t row_bytes =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(hkv) * sizeof(std::uint16_t);
    // A batch occupies consecutive positions, so its rows are consecutive too.
    CUDA_CHECK(cudaMemcpyAsync(static_cast<std::uint8_t*>(cache_rows.data) +
                                   static_cast<std::size_t>(row_index) * row_bytes,
                               compact.data, static_cast<std::size_t>(tokens) * row_bytes,
                               cudaMemcpyDeviceToDevice, stream));
    cache.mark_range(layer, first_position, tokens, stream);

    {
        const ops::AttentionHeadGeometry heads{d, hq, hkv};
        Tensor q_batch        = q_normed.view({d, hq, tokens, 1});
        // Rows sit at their own positions, so the written ones are exactly the leading `filled`, and
        // the Op scans every key it is given.
        const std::int32_t filled = first_position + tokens;
        Tensor kv_batch(static_cast<std::uint8_t*>(cache_rows.data), DType::BF16,
                        {width, hkv, filled, 1});
        Tensor attended_batch = attended.view({d, hq, tokens, 1});
        Tensor position_q     = positions.view({tokens, 1});
        Tensor position_k(static_cast<std::uint8_t*>(cache_pos.data), DType::I32, {filled, 1});
        const std::size_t split =
            ops::causal_compact_attention_workspace_bytes(heads, d, tokens, filled, 1);
        Tensor workspace;
        if (split > 0) workspace = arena.alloc(DType::U8, {static_cast<std::int32_t>(split)});
        // Rows sit at their own positions, which is what `indexed_keys` states.
        ops::causal_compact_attention(q_batch, kv_batch, position_q, position_k, heads, d, pairs,
                                      config.attention_scale, true, attended_batch, workspace,
                                      stream);
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
                                          workspace, stream);
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
                                          workspace, stream);
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
                   std::int32_t first_position, std::int32_t tokens, KvCache& cache,
                   DeviceArena& arena, Tensor& hidden_out, DeviceExecutionView execution,
                   const Tensor& position_high) {
    if (position_high.data != nullptr &&
        (position_high.dtype != DType::I32 || position_high.ne[0] != tokens ||
         position_high.ne[1] != 1)) {
        throw std::invalid_argument("forward_layer: position_high must be I32 [tokens,1]");
    }
    if (model.config().sliding_attention(layer)) {
        forward_sliding_layer(model, layer, hidden_in, first_position, tokens, cache, arena,
                              hidden_out, execution, position_high);
    } else {
        forward_global_layer(model, layer, hidden_in, first_position, tokens, cache, arena,
                             hidden_out, execution);
    }
}

} // namespace ninfer::models::gemma4
