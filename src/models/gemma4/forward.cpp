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
#include "ninfer/ops/rope.h"
#include "ninfer/ops/sliding_causal_attention.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace ninfer::models::gemma4 {
namespace {

struct Scratch {
    Tensor normed;      // [H,1]     the normed layer input, reused by both sublayers
    Tensor branch;      // [H,1]     a sublayer's normalized output before the residual add
    Tensor q;           // [Hq*D,1]  raw query projection
    Tensor k;           // [Hkv*D,1]
    Tensor v;           // [Hkv*D,1]
    Tensor q_heads;     // [D,Hq,1]  the same storage, as heads for the norms and RoPE
    Tensor k_heads;     // [D,Hkv,1]
    Tensor v_heads;     // [D,Hkv,1]
    Tensor q_normed;    // [D,Hq,1]
    Tensor k_normed;    // [D,Hkv,1]
    Tensor v_normed;    // [D,Hkv,1]
    Tensor attended;    // [Hq*D,1]
    Tensor projected;   // [H,1]     the attention output projection
    Tensor gate;        // [I,1]     the MLP's gate projection
    Tensor up;          // [I,1]     the MLP's up projection
    Tensor activated;   // [I,1]
    Tensor down;        // [H,1]
    Tensor positions;   // I32 [1]
};

} // namespace

std::size_t layer_workspace_bytes(const TextConfig& config) {
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
    // plus one position vector. Sliding layers add a v projection on top of the key/value buffers.
    // The arena aligns every allocation to 256 bytes, so the bound carries that per slice, and an
    // undercount makes the arena refuse an allocation rather than overrun.
    const std::uint64_t sliding = 4 * h + 3 * slide_q + 4 * slide_kv + 3 * i;
    const std::uint64_t global  = 4 * h + 3 * global_q + 3 * global_kv + compact + 3 * i;
    return static_cast<std::size_t>(std::max(sliding, global) * 2 + 16 * 256 + 64);
}

void forward_sliding_layer(const Model& model, std::size_t layer, const Tensor& hidden_in,
                           std::int32_t position, KvCache& cache, DeviceArena& arena,
                           Tensor& hidden_out, DeviceExecutionView execution) {
    const cudaStream_t stream = execution.stream;
    const TextConfig& config  = model.config();
    if (layer >= model.weights().text.layers.size()) {
        throw std::invalid_argument("forward_sliding_layer: layer index out of range");
    }
    const LayerWeights& weights = model.weights().text.layers[layer];
    if (weights.mixer != MixerKind::SlidingAttention) {
        throw std::invalid_argument("forward_sliding_layer: layer is not a sliding layer");
    }

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
    scratch.normed    = arena.alloc(DType::BF16, {h, 1, 1});
    scratch.branch    = arena.alloc(DType::BF16, {h, 1, 1});
    scratch.q         = arena.alloc(DType::BF16, {q_width, 1, 1});
    scratch.k         = arena.alloc(DType::BF16, {kv_width, 1, 1});
    scratch.v         = arena.alloc(DType::BF16, {kv_width, 1, 1});
    scratch.q_normed  = arena.alloc(DType::BF16, {d, hq, 1});
    scratch.k_normed  = arena.alloc(DType::BF16, {d, hkv, 1});
    scratch.v_normed  = arena.alloc(DType::BF16, {d, hkv, 1});
    scratch.attended  = arena.alloc(DType::BF16, {q_width, 1, 1});
    scratch.projected = arena.alloc(DType::BF16, {h, 1, 1});
    scratch.gate      = arena.alloc(DType::BF16, {i, 1, 1});
    scratch.up        = arena.alloc(DType::BF16, {i, 1, 1});
    scratch.activated = arena.alloc(DType::BF16, {i, 1, 1});
    scratch.down      = arena.alloc(DType::BF16, {h, 1, 1});
    scratch.positions = arena.alloc(DType::I32, {1, 1, 1});

    const std::vector<std::int32_t> host_position{position};
    CUDA_CHECK(cudaMemcpyAsync(scratch.positions.data, host_position.data(), sizeof(std::int32_t),
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
    scratch.q_heads = scratch.q.view({d, hq, 1});
    scratch.k_heads = scratch.k.view({d, hkv, 1});
    scratch.v_heads = scratch.v.view({d, hkv, 1});
    ops::rmsnorm(scratch.q_heads, model.tensor(weights.attention.query_norm), eps, false,
                 scratch.q_normed, stream);
    ops::rmsnorm(scratch.k_heads, model.tensor(weights.attention.key_norm), eps, false,
                 scratch.k_normed, stream);
    ops::rmsnorm(scratch.v_heads, eps, scratch.v_normed, stream);

    // RoPE rotates in place over the head dimension; the sliding layers rotate the whole head, so
    // the pair count is half of it.
    ops::rope(scratch.positions, d, d / 2, config.sliding.rope_theta, scratch.q_normed,
              scratch.k_normed, execution);

    // The ring holds the window's worth of rows, so the slot is the position modulo the window and the
    // query sees exactly the tokens still inside it. A slot no token has written carries an invisible
    // position, so the whole cache can be handed to the Op without tracking how much is filled.
    Tensor cache_keys   = cache.keys(layer);
    Tensor cache_values = cache.values(layer);
    Tensor cache_pos    = cache.positions(layer);
    const std::int32_t window_index = cache.slot(layer, position);
    const std::int32_t window       = static_cast<std::int32_t>(config.sliding_window);
    const std::size_t row_bytes =
        static_cast<std::size_t>(d) * static_cast<std::size_t>(hkv) * sizeof(std::uint16_t);
    CUDA_CHECK(cudaMemcpyAsync(static_cast<std::uint8_t*>(cache_keys.data) +
                                   static_cast<std::size_t>(window_index) * row_bytes,
                               scratch.k_normed.data, row_bytes, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(static_cast<std::uint8_t*>(cache_values.data) +
                                   static_cast<std::size_t>(window_index) * row_bytes,
                               scratch.v_normed.data, row_bytes, cudaMemcpyDeviceToDevice, stream));
    cache.mark(layer, position, stream);

    {
        const ops::AttentionHeadGeometry geometry{d, hq, hkv};
        Tensor q_batch        = scratch.q_normed.view({d, hq, 1, 1});
        Tensor k_batch        = cache_keys.view({d, hkv, window, 1});
        Tensor v_batch        = cache_values.view({d, hkv, window, 1});
        Tensor attended_batch = scratch.attended.view({d, hq, 1, 1});
        Tensor position_q     = scratch.positions.view({1, 1});
        Tensor position_k     = cache_pos.view({window, 1});
        ops::sliding_causal_attention(q_batch, k_batch, v_batch, position_q, position_k, geometry,
                                      config.sliding_window, config.attention_scale, attended_batch,
                                      stream);
    }
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
                               static_cast<std::size_t>(h) * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToDevice, stream));
}

void forward_global_layer(const Model& model, std::size_t layer, const Tensor& hidden_in,
                          std::int32_t position, KvCache& cache, DeviceArena& arena,
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

    Tensor normed    = arena.alloc(DType::BF16, {h, 1, 1});
    Tensor branch    = arena.alloc(DType::BF16, {h, 1, 1});
    Tensor projected = arena.alloc(DType::BF16, {h, 1, 1});
    Tensor down      = arena.alloc(DType::BF16, {h, 1, 1});
    Tensor q         = arena.alloc(DType::BF16, {hq * d, 1, 1});
    Tensor q_normed  = arena.alloc(DType::BF16, {d, hq, 1});
    Tensor attended  = arena.alloc(DType::BF16, {hq * d, 1, 1});
    Tensor k         = arena.alloc(DType::BF16, {hkv * d, 1, 1});
    Tensor value     = arena.alloc(DType::BF16, {d, hkv, 1});
    Tensor k_normed  = arena.alloc(DType::BF16, {d, hkv, 1});
    Tensor compact   = arena.alloc(DType::BF16, {width, hkv, 1});
    Tensor gate      = arena.alloc(DType::BF16, {i, 1, 1});
    Tensor up        = arena.alloc(DType::BF16, {i, 1, 1});
    Tensor activated = arena.alloc(DType::BF16, {i, 1, 1});
    Tensor positions = arena.alloc(DType::I32, {1, 1, 1});

    const std::vector<std::int32_t> host_position{position};
    CUDA_CHECK(cudaMemcpyAsync(positions.data, host_position.data(), sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream));

    Tensor residual = hidden_in;
    auto to_weight  = [&](WeightId id) { return ops::prepare_linear_weight(model.input(id)).weight; };

    ops::rmsnorm(residual, model.tensor(weights.input_norm), eps, false, normed, stream);
    ops::linear(normed, to_weight(weights.attention.query), q, stream);
    ops::linear(normed, to_weight(weights.attention.key), k, stream);

    // K is V on these layers, so the value side is the key normalized without a weight, read from the
    // raw projection; the weighted key norm is a second, independent normalization of the same input.
    const Tensor k_heads = k.view({d, hkv, 1});
    ops::rmsnorm(k_heads, eps, value, stream);
    ops::rmsnorm(q.view({d, hq, 1}), model.tensor(weights.attention.query_norm), eps, false,
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
    const std::int32_t row_index = cache.slot(layer, position);
    const std::size_t row_bytes =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(hkv) * sizeof(std::uint16_t);
    CUDA_CHECK(cudaMemcpyAsync(static_cast<std::uint8_t*>(cache_rows.data) +
                                   static_cast<std::size_t>(row_index) * row_bytes,
                               compact.data, row_bytes, cudaMemcpyDeviceToDevice, stream));
    cache.mark(layer, position, stream);

    {
        const ops::AttentionHeadGeometry heads{d, hq, hkv};
        Tensor q_batch        = q_normed.view({d, hq, 1, 1});
        Tensor kv_batch       = cache_rows.view({width, hkv, capacity, 1});
        Tensor attended_batch = attended.view({d, hq, 1, 1});
        Tensor position_q     = positions.view({1, 1});
        Tensor position_k     = cache_pos.view({capacity, 1});
        ops::causal_compact_attention(q_batch, kv_batch, position_q, position_k, heads, d, pairs,
                                      config.attention_scale, attended_batch, stream);
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
                               static_cast<std::size_t>(h) * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToDevice, stream));
}

void forward_layer(const Model& model, std::size_t layer, const Tensor& hidden_in,
                   std::int32_t position, KvCache& cache, DeviceArena& arena, Tensor& hidden_out,
                   DeviceExecutionView execution) {
    if (model.config().sliding_attention(layer)) {
        forward_sliding_layer(model, layer, hidden_in, position, cache, arena, hidden_out, execution);
    } else {
        forward_global_layer(model, layer, hidden_in, position, cache, arena, hidden_out, execution);
    }
}

} // namespace ninfer::models::gemma4
