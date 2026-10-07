#include "models/qwen3_5/execution/text.h"

#include "models/qwen3_5/execution/ffn.h"

#include "ninfer/ops/embedding.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/mtp_pack.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/softmax_attention.h"

namespace ninfer::models::qwen3_5::execution {
namespace {

void project(const Tensor& x, const LinearParameters& parameters, Tensor& out,
             WorkspaceArena& workspace, cudaStream_t stream) {
    ops::linear(x, parameters.weight, out, parameters.policy, workspace, stream);
}

} // namespace

// EAGLE3 encoder: fuse the three concatenated target hidden states into the draft-width feature g.
// With `norm_before_fc` the fusion would be preceded by an RMSNorm, but the SpecForge head carries
// no such weight, so the projection is the whole encoder.
void TextContext::eagle3_encode(const Tensor& features, Tensor& g) {
    const auto& draft = *parameters_.draft;
    project(features, draft.feature_projection, g, work_, ctx_.stream);
}

// EAGLE3 decoder step. At draft position P the pair is (t_{P+1}, g_P): the token embedding and the
// fused target feature, each normalized, concatenated into the 2h-wide attention input. The layer
// has no attention gate and no fused input projection, unlike the MTP layer it is hosted beside.
void TextContext::eagle3_forward_ar_step(const Tensor& token, const Tensor& g,
                                         const Tensor& position,
                                         ops::CausalAttentionExecutionEnvelope envelope,
                                         Tensor& hidden, Tensor& logits) {
    if (batch_eagle3_kv_ == nullptr) { throw std::runtime_error("EAGLE3 forward is not enabled"); }
    const auto& config = parameters_.model.config();
    const auto& draft  = *parameters_.draft;
    const auto& eagle  = *config.draft->eagle3;
    const auto& layer  = draft.layers.at(0);
    const auto& attn   = config.draft->attention;
    const std::int32_t h        = dimension(config_.hidden_size);
    const std::int32_t head_dim = dimension(attn.head_dim);
    const std::int32_t q_heads  = dimension(attn.num_attention_heads);
    const std::int32_t kv_heads = dimension(attn.num_key_value_heads);
    const std::int32_t T        = 1;
    cudaStream_t s              = ctx_.stream;
    // The SpecForge draft is a plain Llama: full rotary over the head dimension.
    const int rotary = head_dim;

    Tensor emb = work_.alloc(DType::BF16, {h, T});
    ops::embedding(token, *embed_, emb, s);
    Tensor e = work_.alloc(DType::BF16, {h, T});
    Tensor u = work_.alloc(DType::BF16, {h, T});
    ops::rmsnorm(emb, layer.input_norm, config_.rms_norm_eps, true, e, s);
    ops::rmsnorm(g, layer.hidden_norm, config_.rms_norm_eps, true, u, s);

    Tensor x = work_.alloc(DType::BF16, {2 * h, T});
    ops::mtp_pack_fc_input(e, u, x, s);

    Tensor q = work_.alloc(DType::BF16, {head_dim, q_heads, T});
    Tensor k = work_.alloc(DType::BF16, {head_dim, kv_heads, T});
    Tensor v = work_.alloc(DType::BF16, {head_dim, kv_heads, T});
    Tensor q_flat = q.view({q_heads * head_dim, T});
    Tensor k_flat = k.view({kv_heads * head_dim, T});
    Tensor v_flat = v.view({kv_heads * head_dim, T});
    project(x, layer.query, q_flat, work_, s);
    project(x, layer.key, k_flat, work_, s);
    project(x, layer.value, v_flat, work_, s);

    ops::rope(position, rotary, config.draft->rope_theta, q, k, ctx_.execution_view());

    Tensor a = work_.alloc(DType::BF16, {head_dim, q_heads, T});
    ops::causal_softmax_attention(
        q, k, v, position, Tensor{}, io_.backend_kv_table_row, {head_dim, q_heads, kv_heads},
        static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim))),
        batch_eagle3_kv_->batch_layer_view(0), envelope, work_, a, ctx_.execution_view());

    Tensor y = work_.alloc(DType::BF16, {h, T});
    Tensor a_flat = a.view({q_heads * head_dim, T});
    project(a_flat, layer.output, y, work_, s);
    ops::residual_add(eagle.norm_before_residual ? u : g, y, s);

    Tensor m = work_.alloc(DType::BF16, {h, T});
    ops::rmsnorm(y, layer.post_attention_norm, config_.rms_norm_eps, true, m, s);
    ffn(m, layer.mlp, y, {}, work_, ctx_.execution_view(), true);

    // The pre-norm hidden becomes the next draft step's g.
    CUDA_CHECK(cudaMemcpyAsync(hidden.data, y.data, y.bytes(), cudaMemcpyDeviceToDevice, s));
    Tensor normed = work_.alloc(DType::BF16, {h, T});
    ops::rmsnorm(y, draft.final_norm, config_.rms_norm_eps, true, normed, s);
    project(normed, draft.output_head, logits, work_, s);
}

} // namespace ninfer::models::qwen3_5::execution