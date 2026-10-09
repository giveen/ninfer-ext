#include "models/gemma4/forward.h"

#include "core/device.h"
#include "ninfer/ops/gelu_mul.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/mul_scalar.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/sliding_causal_attention.h"

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

std::size_t sliding_layer_workspace_bytes(const TextConfig& config) {
    const std::uint64_t h  = config.hidden_size;
    const std::uint64_t i  = config.intermediate_size;
    const std::uint64_t q  = static_cast<std::uint64_t>(config.num_attention_heads) *
                            config.sliding.head_dim;
    const std::uint64_t kv = static_cast<std::uint64_t>(config.sliding.num_key_value_heads) *
                             config.sliding.head_dim;
    // Every buffer the scratch below allocates, counted exactly so the bound cannot drift from it:
    //   hidden-sized:        normed, branch, projected, down
    //   query-sized:         q, q_normed (D by Hq), attended
    //   key/value-sized:     k, v, k_normed, v_normed
    //   intermediate-sized:  gate, up, activated
    // plus one position vector. The arena aligns every allocation to 256 bytes, so the bound carries
    // that per slice, and an undercount makes the arena refuse an allocation rather than overrun.
    const std::uint64_t elements = 4 * h + 3 * q + 4 * kv + 3 * i;
    return static_cast<std::size_t>(elements * 2 + 16 * 256 + 64);
}

void forward_sliding_layer(const Model& model, std::size_t layer, const Tensor& hidden_in,
                           std::int32_t position, DeviceArena& arena, Tensor& hidden_out,
                           DeviceExecutionView execution) {
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

    {
        const ops::AttentionHeadGeometry geometry{d, hq, hkv};
        Tensor q_batch        = scratch.q_normed.view({d, hq, 1, 1});
        Tensor k_batch        = scratch.k_normed.view({d, hkv, 1, 1});
        Tensor v_batch        = scratch.v_normed.view({d, hkv, 1, 1});
        Tensor attended_batch = scratch.attended.view({d, hq, 1, 1});
        ops::sliding_causal_attention(q_batch, k_batch, v_batch, scratch.positions,
                                      scratch.positions, geometry, config.sliding_window,
                                      config.attention_scale, attended_batch, stream);
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

} // namespace ninfer::models::gemma4
