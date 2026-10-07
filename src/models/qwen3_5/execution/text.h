#pragma once
#include "models/qwen3_5/program/internal.h"


#include "core/arena.h"
#include "core/device.h"
#include "core/gdn_replay_records.h"
#include "core/linear_attention_state.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/offload_moe.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/softmax_attention.h"
#include "ninfer/ops/sparse_moe.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/program/round_buffers.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

using Phase = qwen3_5::TextPhase;

enum class GdnStateAction : std::uint8_t {
    UpdateInPlace,
    RecordForReplay,
};

struct NullTap {
    static constexpr bool enabled = false;
};

struct PrefillChunkResult {
    std::uint32_t processed_tokens = 0;
    bool finalized                 = false;
    runtime::ExecutionTiming timing;
};

struct DFlashFeatureSink {
    static constexpr bool enabled = true;
    using PrefillConsumer         = std::function<void(const Tensor&, const Tensor&, bool)>;

    Tensor* features                  = nullptr;
    Tensor* positions                 = nullptr;
    Tensor* batch_features            = nullptr;
    const Tensor* batch_lanes         = nullptr;
    const Tensor* batch_valid_columns = nullptr;
    std::int32_t batch_width          = 0;
    std::int32_t batch_size           = 0;
    std::span<const std::uint32_t> layers;
    PrefillConsumer consume_prefill;
    std::uint32_t captured_mask = 0;
    std::int32_t active_tokens  = 0;

    void begin(const Tensor& value);
    void capture_layer(int layer, const Tensor& value, cudaStream_t stream);
    void capture_positions(const Tensor& source, cudaStream_t stream);
    void consume_prefill_chunk(std::int32_t tokens, bool rewrite_checkpoint);
};

class VisionPrefillSession;
class PleGather;

class Qwen4ExpertPager;

// Program-owned mutable resources of the Qwen4Exp route. The expert pager and PLE state are
// performance or recurrent side state; the Program allocates them once and binds them here.
struct Qwen4Runtime {
    Qwen4ExpertPager* experts = nullptr;
    Tensor ple_states; // BF16 [history * residual, StateImage slots]
    Tensor ple_record; // BF16 [residual, record_width, rows] speculative conv inputs, or empty
    Tensor ple_input;  // BF16 [ple_width, columns] decode-round embedding written before launch
    PleGather* ple_gather       = nullptr;
    std::uint16_t* ple_host     = nullptr; // pinned BF16 [ple_width, prefill_chunk]
};

class TextContext {
public:
    TextContext(DeviceContext& ctx, const execution::Parameters& weights, WorkspaceArena& work,
                qwen3_5::PagedKVCacheView kv, LinearAttentionStatePool& state,
                qwen3_5::RoundState& io, Tensor& prefill_hidden, std::uint32_t prefill_chunk,
                std::uint32_t text_kv_base,
                qwen3_5::PagedKVCacheView mtp_kv           = qwen3_5::PagedKVCacheView(),
                const qwen3_5::PagedKVCache* batch_text_kv = nullptr,
                const qwen3_5::PagedKVCache* batch_mtp_kv  = nullptr,
                const qwen3_5::PagedKVCache* batch_eagle3_kv = nullptr);
    ~TextContext();

    TextContext(const TextContext&)            = delete;
    TextContext& operator=(const TextContext&) = delete;

    void set_proposal_head(const LinearParameters* weight, const std::int32_t* ids,
                           int count) noexcept {
        proposal_head_     = weight;
        proposal_head_ids_ = ids;
        proposal_head_n_   = count;
    }

    void set_sampling(const ops::SamplingConfig* config) noexcept { sampling_config_ = config; }

    void set_prefill_split_frontier(std::int64_t position) noexcept {
        prefill_split_frontier_ = position;
    }

    void set_rewrite_checkpoint_hidden_output(Tensor* output) noexcept {
        rewrite_checkpoint_hidden_output_ = output;
    }

    // Streamed prefill: full-attention layers read the chunk's Host pages from this staging.
    void set_text_kv_staging(qwen3_5::KVHostStaging* staging) noexcept {
        text_kv_staging_ = staging;
    }

    // Streamed Qwen4Exp prefill: the MTP layer reads the chunk's Host pages from this staging.
    void set_mtp_kv_staging(qwen3_5::KVHostStaging* staging) noexcept { mtp_kv_staging_ = staging; }

    // Streamed decode: QSA layers score Host pages' index records from these mirrors.
    void set_qsa_index_mirrors(qwen3_5::QsaIndexMirrors mirrors) noexcept {
        qsa_index_mirrors_ = mirrors;
    }

    void set_mtp_proposal_extent(std::uint32_t extent) noexcept { mtp_proposal_extent_ = extent; }

    void set_qwen4_runtime(const Qwen4Runtime* runtime) noexcept { qwen4_runtime_ = runtime; }

    // Width of the continuation hidden this context reads and writes (the wide HC residual for
    // Qwen4Exp, the normalized hidden otherwise).
    [[nodiscard]] std::int32_t continuation_width() const noexcept { return continuation_width_; }

    void set_linear_state_slots(std::int32_t source_slot, std::int32_t destination_slot);
    void set_gdn_state_action(GdnStateAction action, const GdnReplayRecords* replay_records);

    [[nodiscard]] const LinearParameters* proposal_head() const noexcept { return proposal_head_; }

    [[nodiscard]] const std::int32_t* proposal_head_ids() const noexcept {
        return proposal_head_ids_;
    }

    [[nodiscard]] int proposal_head_n() const noexcept { return proposal_head_n_; }

    [[nodiscard]] PrefillChunkResult prefill_chunk(std::span<const int> full_ids,
                                                   std::uint32_t begin,
                                                   std::uint32_t nominal_length,
                                                   bool finalize_at_end,
                                                   std::int32_t rope_delta);
    [[nodiscard]] PrefillChunkResult prefill_chunk(std::span<const int> full_ids,
                                                   std::uint32_t begin,
                                                   std::uint32_t nominal_length,
                                                   bool finalize_at_end, std::int32_t rope_delta,
                                                   DFlashFeatureSink& sink);
    [[nodiscard]] PrefillChunkResult
    prefill_chunk(const qwen3_5::PreparedPromptData& input, std::uint32_t begin,
                  std::uint32_t nominal_length, VisionPrefillSession& vision, bool finalize_at_end);
    [[nodiscard]] PrefillChunkResult prefill_chunk(const qwen3_5::PreparedPromptData& input,
                                                   std::uint32_t begin,
                                                   std::uint32_t nominal_length,
                                                   VisionPrefillSession& vision,
                                                   bool finalize_at_end, DFlashFeatureSink& sink);
    void ordinary_decode_batch(const Tensor& ids, const Tensor& cache_positions,
                               const Tensor& rope_positions, const Tensor& kv_table_rows,
                               const Tensor& linear_state_source_slots,
                               const Tensor& linear_state_destination_slots,
                               ops::CausalAttentionExecutionEnvelope envelope, Tensor& hidden,
                               Tensor& logits);
    void target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                             const Tensor& rope_positions, const Tensor& valid_columns,
                             const Tensor& kv_table_rows, const Tensor& linear_state_source_slots,
                             ops::CausalAttentionExecutionEnvelope envelope, Tensor& hidden,
                             Tensor& logits, Tensor& target_tokens);
    void target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                             const Tensor& rope_positions, const Tensor& valid_columns,
                             const Tensor& kv_table_rows, const Tensor& linear_state_source_slots,
                             ops::CausalAttentionExecutionEnvelope envelope, Tensor& hidden,
                             Tensor& logits, Tensor& target_tokens, DFlashFeatureSink& sink);
    void mtp_forward_decode_batch(const Tensor& ids, const Tensor& hidden,
                                  const Tensor& cache_positions, const Tensor& rope_positions,
                                  const Tensor& valid_columns, const Tensor& kv_table_rows,
                                  ops::CausalAttentionExecutionEnvelope envelope,
                                  Tensor& mtp_hidden);
    // Qwen4Exp: write the MTP layer's KV for `ids [W,B]` over target `hidden [D,W,B]` without the
    // rest of the layer. A round that verifies no drafts keeps the MTP cache current this way.
    void mtp_append_decode_batch(const Tensor& ids, const Tensor& hidden,
                                 const Tensor& cache_positions, const Tensor& rope_positions,
                                 const Tensor& kv_table_rows,
                                 ops::CausalAttentionExecutionEnvelope envelope);
    void mtp_propose_batch(const Tensor& hidden, Tensor& logits, Tensor& draft_tokens);
    void mtp_forward_batch(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                           ops::CausalAttentionExecutionEnvelope envelope, Tensor& mtp_hidden,
                           int logits_column, Tensor* logits, Tensor* draft_token,
                           const Tensor* explicit_rope_positions = nullptr,
                           const Tensor* input_embeddings        = nullptr);
    void mtp_forward_ar_step(const Tensor& token, const Tensor& previous_hidden,
                             const Tensor& position, ops::CausalAttentionExecutionEnvelope envelope,
                             Tensor& mtp_hidden, Tensor& logits, Tensor& draft_token);

    // EAGLE3 draft: fuse the three captured target layers into the draft-width feature g.
    void eagle3_encode_batch(const Tensor& features, Tensor& g);
    // EAGLE3 decoder step: the pair (token embedding, g) through the one decoder layer produces the
    // draft pre-norm hidden (the next step's g) and the draft-vocabulary logits.
    void eagle3_forward_decode_batch(const Tensor& ids, const Tensor& g,
                                     const Tensor& cache_positions, const Tensor& rope_positions,
                                     const Tensor& valid_columns, const Tensor& kv_table_rows,
                                     ops::CausalAttentionExecutionEnvelope envelope, Tensor& hidden);
    // EAGLE3 draft proposal: the draft-vocabulary head, then the draft-to-target token-id map.
    void eagle3_propose_batch(const Tensor& hidden, Tensor& logits, Tensor& draft_tokens);
    void set_eagle3_head(const LinearParameters* head, const std::int32_t* d2t,
                         int draft_vocab) noexcept {
        eagle3_head_        = head;
        eagle3_d2t_         = d2t;
        eagle3_draft_vocab_ = draft_vocab;
    }
private:
    [[nodiscard]] bool mtp_enabled() const noexcept {
        return mtp_kv_.valid() || batch_mtp_kv_ != nullptr;
    }

    void attn_mix(const BlockParameters& weights, Tensor& x, int index, int layer, Phase phase);
    void gdn_mix(const BlockParameters& weights, Tensor& x, int index, int layer, Phase phase);
    void mlp_tail(const BlockParameters& weights, Tensor& x, int layer, Phase phase,
                  const ops::SparseMoeHints& hints);
    [[nodiscard]] ops::SparseMoeHints next_projection_hints(int layer) const;
    void run_layers(Tensor& x, Phase phase);
    template <class Tap>
    void run_layers(Tensor& x, Phase phase, Tap& tap);
    template <class Tap>
    void target_verify_batch_impl(const Tensor& ids, const Tensor& cache_positions,
                                  const Tensor& rope_positions, const Tensor& valid_columns,
                                  const Tensor& kv_table_rows,
                                  const Tensor& linear_state_source_slots,
                                  ops::CausalAttentionExecutionEnvelope envelope, Tensor& hidden,
                                  Tensor& logits, Tensor& target_tokens, Tap& tap);

    void mtp_forward_stem(const Tensor& ids, const Tensor& hidden, const Tensor* input_embeddings,
                          Tensor& x, Tensor& ah);
    void mtp_forward_tail(Tensor& x, const Tensor& ah, const Tensor& positions,
                          const Tensor& rope_positions,
                          ops::CausalAttentionExecutionEnvelope envelope, Tensor& mtp_hidden);
    void mtp_forward_core(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                          const Tensor& rope_positions,
                          ops::CausalAttentionExecutionEnvelope envelope, Tensor& mtp_hidden,
                          const Tensor* input_embeddings);
    void mtp_prefill_chunk(const Tensor& ids, const Tensor& hidden, const Tensor* input_embeddings,
                           const Tensor& positions, const Tensor& rope_positions,
                           ops::CausalAttentionExecutionEnvelope envelope, bool final_chunk,
                           Tensor* final_hidden, Tensor* logits, Tensor* draft_token);
    void proposal_argmax(const Tensor& hidden, Tensor& logits, Tensor& proposal_tokens);

    struct MultimodalPrefill {
        std::span<const int> token_ids;
        std::span<const std::int32_t> positions;
        VisionPrefillSession* vision = nullptr;
        std::uint32_t begin          = 0;
        std::int32_t rope_delta      = 0;
    };

    struct TextPrefill {
        std::span<const int> token_ids;
        std::uint32_t begin     = 0;
        std::int32_t rope_delta = 0;
    };

    template <class Tap>
    [[nodiscard]] PrefillChunkResult
    prefill_impl(std::span<const int> ids, const TextPrefill* text_prefill,
                 const MultimodalPrefill* multimodal, Tap& tap, bool finalize_at_end);

    // Qwen4Exp route (qwen4_text.cpp).
    struct Qwen4Sequence {
        const Tensor* cache_positions = nullptr; // I32 [W*B]
        const Tensor* rope_positions  = nullptr; // I32 [W*B] or [W*B,3]
        const Tensor* valid_columns   = nullptr; // I32 [B] or null
        const Tensor* table_rows      = nullptr; // I32 [B]
        std::int32_t width            = 0;
        std::int32_t batch            = 0;
        std::uint32_t max_visible     = 0;
    };

    [[nodiscard]] const Qwen4Runtime& qwen4_runtime() const;
    [[nodiscard]] Qwen4Sequence qwen4_text_sequence(std::int32_t columns) const;
    void qwen4_embed(const Tensor& ids, const Tensor* scatter_indices,
                     const Tensor* scatter_embeddings, const Tensor* input_embeddings,
                     Tensor& wide);
    void qwen4_hc_mix(const HyperConnectionParameters& p, const Tensor& wide, Tensor& x,
                      Tensor* inject);
    // A null `y` only appends this layer's KV and index keys.
    void qwen4_qsa(const QsaParameters& p, const Tensor& x, const Qwen4Sequence& sequence,
                   const PagedKVCache& cache, std::uint32_t layer, Tensor* y);
    void qwen4_gdn(const Qwen4GdnParameters& p, const Tensor& x, int gdn_index, Phase phase,
                   Tensor& y);
    void qwen4_ple(const PleParameters& p, const Tensor& embedding, Phase phase, Tensor& wide);
    void qwen4_moe(const OffloadMoeParameters& p, const Tensor& x, std::int32_t cache_layer,
                   Tensor& y);
    // Offline calibration: route every token through every expert on scratch outputs so each
    // expert's down projection observes the whole calibration set.
    void qwen4_capture_all_experts(const Tensor& x, const ops::ExpertWeights& staged, int layer);
    void qwen4_block(const Qwen4BlockParameters& p, Tensor& wide, Phase phase,
                     const Qwen4Sequence& sequence, const PagedKVCache& cache,
                     std::uint32_t kv_layer, int gdn_index, std::int32_t cache_layer,
                     const Tensor* ple_embedding);
    void qwen4_layers(Tensor& wide, Phase phase, const Qwen4Sequence& sequence,
                      const Tensor* ple_embedding);
    void qwen4_logits(const HyperConnectionParameters& head, const Tensor& wide, Tensor& logits);
    void qwen4_mtp_stem(const Tensor& ids, const Tensor& hidden, const Tensor* input_embeddings,
                        Tensor& wide);
    void qwen4_mtp_core(const Tensor& ids, const Tensor& hidden, const Tensor* input_embeddings,
                        const Qwen4Sequence& sequence, Tensor& mtp_hidden);
    void qwen4_mtp_append(const Tensor& ids, const Tensor& hidden, const Qwen4Sequence& sequence);
    void qwen4_proposal(const Tensor& hidden, Tensor& logits, Tensor& tokens);
    [[nodiscard]] PrefillChunkResult qwen4_prefill(std::span<const int> ids,
                                                   const TextPrefill* text_prefill,
                                                   const MultimodalPrefill* multimodal,
                                                   bool finalize_at_end);
    DeviceContext& ctx_;
    const Parameters& parameters_;
    const TextConfig& config_;
    WorkspaceArena& work_;
    qwen3_5::PagedKVCacheView kv_;
    qwen3_5::PagedKVCacheView mtp_kv_;
    qwen3_5::PagedKVCacheView eagle3_kv_;
    const qwen3_5::PagedKVCache* batch_text_kv_ = nullptr;
    qwen3_5::KVHostStaging* text_kv_staging_ = nullptr;
    qwen3_5::KVHostStaging* mtp_kv_staging_  = nullptr;
    qwen3_5::QsaIndexMirrors qsa_index_mirrors_;
    const qwen3_5::PagedKVCache* batch_mtp_kv_  = nullptr;
    const qwen3_5::PagedKVCache* batch_eagle3_kv_ = nullptr;
    const LinearParameters* eagle3_head_    = nullptr;
    const std::int32_t* eagle3_d2t_         = nullptr;
    int eagle3_draft_vocab_                 = 0;
    LinearAttentionStatePool& state_;
    qwen3_5::RoundState& io_;
    Tensor& prefill_hidden_;
    std::uint32_t prefill_chunk_;
    std::uint32_t text_kv_base_;
    const Tensor* active_cache_positions_                                          = nullptr;
    const Tensor* active_rope_positions_                                           = nullptr;
    const Tensor* active_kv_table_rows_                                            = nullptr;
    const Tensor* active_linear_state_source_slots_                                = nullptr;
    const Tensor* active_linear_state_destination_slots_                           = nullptr;
    const Tensor* active_valid_columns_                                            = nullptr;
    const Tensor* active_backend_kv_table_rows_                                    = nullptr;
    const ops::CausalAttentionExecutionEnvelope* active_causal_attention_envelope_ = nullptr;
    std::int32_t active_sequence_batch_                                            = 0;
    std::int32_t active_sequence_width_                                            = 0;
    std::int32_t rope_delta_                                                       = 0;
    std::int32_t linear_state_source_slot_                                         = 0;
    std::int32_t linear_state_destination_slot_                                    = 0;
    GdnStateAction gdn_state_action_          = GdnStateAction::UpdateInPlace;
    const GdnReplayRecords* replay_records_   = nullptr;
    std::int64_t prefill_split_frontier_      = -1;
    Tensor* rewrite_checkpoint_hidden_output_ = nullptr;
    std::uint32_t mtp_proposal_extent_        = 0;

    const Weight* embed_                        = nullptr;
    const Tensor* final_norm_                   = nullptr;
    const LinearParameters* lm_head_            = nullptr;
    const LinearParameters* proposal_head_      = nullptr;
    const std::int32_t* proposal_head_ids_      = nullptr;
    int proposal_head_n_                        = 0;
    const ops::SamplingConfig* sampling_config_ = nullptr;
    const MtpParameters* mtp_                   = nullptr;
    const Qwen4Parameters* qwen4_               = nullptr;
    const Qwen4MtpParameters* qwen4_mtp_        = nullptr;
    const Qwen4Runtime* qwen4_runtime_          = nullptr;
    std::int32_t continuation_width_            = 0;
};

} // namespace ninfer::models::qwen3_5::execution
