#include "models/qwen3_5/execution/text.h"

#include "models/qwen3_5/execution/ffn.h"

#include "ninfer/ops/argmax.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/mtp_pack.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/softmax_attention.h"
#include "ninfer/ops/speculative_round.h"

namespace ninfer::models::qwen3_5::execution {
namespace {

void project(const Tensor& x, const LinearParameters& parameters, Tensor& out,
             WorkspaceArena& workspace, cudaStream_t stream) {
    ops::linear(x, parameters.weight, out, parameters.policy, workspace, stream);
}

} // namespace

// EAGLE3 encoder: fuse the three concatenated target hidden states into the draft-width feature g.
// With `norm_before_fc` the fusion would be preceded by an RMSNorm, but the SpecForge head carries
// no such weight, so the projection is the whole encoder. `features` is [3h, width, batch]; `g` is
// [h, width, batch].
void TextContext::eagle3_encode_batch(const Tensor& features, Tensor& g) {
    const auto& draft       = *parameters_.draft;
    const std::int32_t h    = dimension(parameters_.model.config().text.hidden_size);
    const std::int32_t fused = features.ne[0];
    const std::int32_t width = features.ne[1];
    const std::int32_t batch = features.ne[2];
    if (features.dtype != DType::BF16 || width <= 0 || batch <= 0 ||
        fused != static_cast<std::int32_t>(draft.feature_projection.weight.k) ||
        g.dtype != DType::BF16 || g.ne[0] != h || g.ne[1] != width || g.ne[2] != batch) {
        throw std::invalid_argument("EAGLE3 encoder shapes are invalid");
    }
    const std::int32_t columns = width * batch;
    Tensor g_flat = g.view({h, columns});
    project(features.view({fused, columns}), draft.feature_projection, g_flat, work_, ctx_.stream);
}

// EAGLE3 decoder step. At draft position P the pair is (t_{P+1}, g_P): the token embedding and the
// fused target feature, each normalized, concatenated into the 2h-wide attention input. The layer
// has no attention gate and no fused input projection, unlike the MTP layer it is hosted beside.
// `ids` is [width, batch]; `g`/`hidden` are [h, width, batch]; `cache_positions`/`rope_positions`
// are [width, batch]. The draft head is applied separately, only at the selected column.
void TextContext::eagle3_forward_decode_batch(const Tensor& ids, const Tensor& g,
                                              const Tensor& cache_positions,
                                              const Tensor& rope_positions,
                                              const Tensor& valid_columns,
                                              const Tensor& kv_table_rows,
                                              ops::CausalAttentionExecutionEnvelope envelope,
                                              Tensor& hidden) {
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
    const std::int32_t width    = ids.ne[0];
    const std::int32_t batch    = ids.ne[1];
    if (ids.dtype != DType::I32 || rope_positions.dtype != DType::I32 ||
        cache_positions.dtype != DType::I32 || valid_columns.dtype != DType::I32 ||
        kv_table_rows.dtype != DType::I32 || width <= 0 || batch <= 0 ||
        cache_positions.ne[0] != width || cache_positions.ne[1] != batch ||
        rope_positions.ne[0] != width || rope_positions.ne[1] != batch ||
        valid_columns.ne[0] != batch || kv_table_rows.ne[0] != batch ||
        g.dtype != DType::BF16 || g.ne[0] != h || g.ne[1] != width || g.ne[2] != batch ||
        hidden.dtype != DType::BF16 || hidden.ne[0] != h || hidden.ne[1] != width ||
        hidden.ne[2] != batch) {
        throw std::invalid_argument("EAGLE3 decode batch shapes are invalid");
    }
    const std::int32_t columns = width * batch;
    cudaStream_t s             = ctx_.stream;
    auto scratch               = work_.scope();
    // The SpecForge draft is a plain Llama: full rotary over the head dimension.
    const int rotary = head_dim;

    Tensor g_flat = g.view({h, columns});
    Tensor emb    = work_.alloc(DType::BF16, {h, columns});
    ops::embedding(ids.view({columns}), *embed_, emb, s);
    Tensor e = work_.alloc(DType::BF16, {h, columns});
    Tensor u = work_.alloc(DType::BF16, {h, columns});
    ops::rmsnorm(emb, layer.input_norm, config_.rms_norm_eps, false, e, s);
    ops::rmsnorm(g_flat, layer.hidden_norm, config_.rms_norm_eps, false, u, s);

    Tensor x = work_.alloc(DType::BF16, {2 * h, columns});
    ops::mtp_pack_fc_input(e, u, x, s);

    Tensor q = work_.alloc(DType::BF16, {head_dim, q_heads, columns});
    Tensor k = work_.alloc(DType::BF16, {head_dim, kv_heads, columns});
    Tensor v = work_.alloc(DType::BF16, {head_dim, kv_heads, columns});
    Tensor q_flat = q.view({q_heads * head_dim, columns});
    Tensor k_flat = k.view({kv_heads * head_dim, columns});
    Tensor v_flat = v.view({kv_heads * head_dim, columns});
    project(x, layer.query, q_flat, work_, s);
    project(x, layer.key, k_flat, work_, s);
    project(x, layer.value, v_flat, work_, s);

    ops::rope(rope_positions.view({columns}), rotary, config.draft->rope_theta, q, k,
              ctx_.execution_view());

    Tensor a = work_.alloc(DType::BF16, {head_dim, q_heads, width, batch});
    ops::causal_softmax_attention(
        q.view({head_dim, q_heads, width, batch}), k.view({head_dim, kv_heads, width, batch}),
        v.view({head_dim, kv_heads, width, batch}), cache_positions.view({width, batch}),
        valid_columns, kv_table_rows, {head_dim, q_heads, kv_heads},
        static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim))),
        batch_eagle3_kv_->batch_layer_view(0), envelope, work_, a, ctx_.execution_view());

    Tensor y = work_.alloc(DType::BF16, {h, columns});
    project(a.view({q_heads * head_dim, columns}), layer.output, y, work_, s);
    ops::residual_add(eagle.norm_before_residual ? u : g_flat, y, s);

    Tensor m = work_.alloc(DType::BF16, {h, columns});
    ops::rmsnorm(y, layer.post_attention_norm, config_.rms_norm_eps, false, m, s);
    ffn(m, layer.mlp, y, {}, work_, ctx_.execution_view(), true);

    // The pre-norm hidden becomes the next draft step's g.
    CUDA_CHECK(cudaMemcpyAsync(hidden.data, y.data, y.bytes(), cudaMemcpyDeviceToDevice, s));
}

// EAGLE3 draft proposal: the draft-vocabulary head over the pre-norm hidden, then the stored
// draft-to-target map lifts the argmax index to a real token id. `hidden` is [h, batch].
void TextContext::eagle3_propose_batch(const Tensor& hidden, Tensor& logits, Tensor& draft_tokens) {
    if (eagle3_head_ == nullptr || eagle3_d2t_ == nullptr) {
        throw std::runtime_error("EAGLE3 proposal head is not set");
    }
    const std::int32_t batch = hidden.ne[1];
    const std::int32_t rows  = eagle3_head_->weight.n; // stored rows; the converter pads the head
    if (hidden.dtype != DType::BF16 || batch <= 0 || logits.dtype != DType::BF16 ||
        logits.ne[0] < rows || logits.ne[1] != batch || draft_tokens.dtype != DType::I32 ||
        draft_tokens.ne[0] != batch) {
        throw std::invalid_argument("EAGLE3 proposal batch shapes are invalid");
    }
    const auto& draft = *parameters_.draft;
    Tensor lg         = logits.slice(0, 0, rows);
    Tensor normed     = work_.alloc(DType::BF16, {hidden.ne[0], batch});
    ops::rmsnorm(hidden, draft.final_norm, config_.rms_norm_eps, false, normed, ctx_.stream);
    project(normed, *eagle3_head_, lg, work_, ctx_.stream);
    Tensor real = lg.slice(0, 0, eagle3_draft_vocab_);
    ops::argmax(real, draft_tokens, eagle3_draft_vocab_, ctx_.stream);
    ops::proposal_remap_token_ids(draft_tokens, eagle3_d2t_, eagle3_draft_vocab_, ctx_.stream);
}

} // namespace ninfer::models::qwen3_5::execution
