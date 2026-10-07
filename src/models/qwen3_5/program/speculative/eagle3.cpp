#include "models/qwen3_5/program/context.h"
#include "models/qwen3_5/program/graph_execution.h"
#include "models/qwen3_5/program/internal.h"
#include "core/nvtx.h"
#include "ninfer/ops/mtp_round.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/speculative_round.h"
#include <cuda_runtime.h>
#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {
namespace {

DFlashFeatureSink eagle3_feature_sink(Eagle3BatchContext& state, const Tensor& lanes,
                                     const Tensor& valid_columns, std::int32_t width,
                                     std::int32_t batch_size) {
    const auto& config = *state.execution.parameters.model.config().draft;
    return DFlashFeatureSink{
        .batch_features      = &state.frame.features,
        .batch_lanes         = &lanes,
        .batch_valid_columns = &valid_columns,
        .batch_width         = width,
        .batch_size          = batch_size,
        .layers              = std::span<const std::uint32_t>(config.target_layer_ids),
    };
}

} // namespace

// One exact-B EAGLE3 transaction: the target verifies the anchor chain while capturing the three
// draft layers, the encoder fuses them into g, and the one-layer draft decoder autoregressively
// proposes the next round's tokens from the accepted prefix. The round reuses the MTP host bridge,
// frame and envelopes; only the draft cache and the draft forward differ.
void eagle3_decode_batch(Eagle3BatchContext& state, std::int32_t batch_size, std::uint32_t k,
                         MtpCausalAttentionEnvelopes envelopes, DecodeGraphExecutable* executable) {
    auto body = [&state, batch_size, k, envelopes] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumConcurrency) ||
            k == 0 || k > kLookupDecodeMaximumDrafts) {
            throw std::logic_error("EAGLE3 decode batch state is incomplete");
        }

        qwen3_5::MtpDecodeState& frame = state.frame;
        const std::int32_t width       = static_cast<std::int32_t>(k) + 1;
        CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, &state.host_ingress,
                                   sizeof(qwen3_5::MtpDecodeIngress), cudaMemcpyHostToDevice,
                                   state.execution.device.stream));

        const auto& config = state.execution.parameters.model.config();
        const auto& draft  = *state.execution.parameters.draft;
        const auto& eagle  = *config.draft->eagle3;

        TextContext card(state.execution.device, state.execution.parameters, state.execution.work,
                         {}, state.execution.linear_attention, state.execution.io,
                         state.execution.prefill_hidden, state.execution.prefill_chunk, 0, {},
                         &state.text_cache, nullptr, &state.eagle3_cache);
        card.set_qwen4_runtime(state.execution.qwen4);
        card.set_text_kv_staging(state.text_kv_staging);
        card.set_qsa_index_mirrors(state.qsa_index_mirrors);
        card.set_eagle3_head(&draft.output_head,
                             static_cast<const std::int32_t*>(draft.d2t.data),
                             static_cast<int>(eagle.draft_vocab_size));

        upload_qwen4_round_input(state.execution, static_cast<std::int32_t>(k + 1U) * batch_size);
        Tensor anchors            = frame.anchors.slice(0, 0, batch_size);
        Tensor frontiers          = frame.base_frontiers.slice(0, 0, batch_size);
        Tensor budgets            = frame.remaining_budgets.slice(0, 0, batch_size);
        Tensor current_extents    = frame.current_extents.slice(0, 0, batch_size);
        Tensor target_valid       = frame.target_valid_columns.slice(0, 0, batch_size);
        Tensor current_drafts     = frame.current_drafts.slice(1, 0, batch_size);
        Tensor target_rope        = frame.target_rope_positions.slice(1, 0, batch_size);
        Tensor text_rows          = frame.text_kv_table_rows.slice(0, 0, batch_size);
        Tensor eagle3_rows        = frame.mtp_kv_table_rows.slice(0, 0, batch_size);
        Tensor state_sources      = frame.state_source_slots.slice(0, 0, batch_size);
        Tensor state_destinations = frame.state_destination_slots.slice(0, 0, batch_size);
        Tensor rope_deltas        = frame.rope_deltas.slice(0, 0, batch_size);
        Tensor verify_ids         = frame.verify_ids.slice(1, 0, batch_size);
        Tensor target_positions   = frame.target_positions.slice(1, 0, batch_size);
        Tensor target_tokens      = frame.target_argmax.slice(1, 0, batch_size);
        Tensor target_logits      = frame.target_logits.slice(2, 0, batch_size);
        Tensor features           = frame.features.slice(2, 0, batch_size);
        Tensor g                  = frame.target_hidden.slice(2, 0, batch_size);
        Tensor selected_hidden    = frame.target_continuation_hidden.slice(1, 0, batch_size);
        Tensor licensed_tokens    = frame.licensed_tokens.slice(1, 0, batch_size);
        Tensor licensed_counts    = frame.licensed_counts.slice(0, 0, batch_size);
        Tensor accepted           = frame.accepted_drafts.slice(0, 0, batch_size);
        Tensor next_extents       = frame.next_extents.slice(0, 0, batch_size);
        Tensor alignment_ids      = frame.alignment_ids.slice(1, 0, batch_size);
        Tensor alignment_hidden   = frame.alignment_hidden.slice(2, 0, batch_size);
        Tensor ar_hidden          = frame.ar_hidden.slice(1, 0, batch_size);
        Tensor next_hidden        = frame.next_hidden.slice(1, 0, batch_size);
        Tensor ar_positions       = frame.ar_positions.slice(0, 0, batch_size);
        Tensor ar_rope_positions  = frame.ar_rope_positions.slice(0, 0, batch_size);
        Tensor ar_valid_columns   = frame.ar_valid_columns.slice(0, 0, batch_size);
        Tensor next_drafts        = frame.next_drafts.slice(0, 0, batch_size);
        Tensor feature_lanes      = frame.feature_lanes.slice(0, 0, batch_size);

        ops::speculative_prepare_verify_inputs(anchors, current_drafts, frontiers, current_extents,
                                               verify_ids, target_positions,
                                               state.execution.device.stream);
        {
            nvtx::ScopedRange target_range(nvtx::Name::DecodeEagle3Target, nvtx::Category::Eagle3,
                                           static_cast<std::uint64_t>(width) * batch_size);
            DFlashFeatureSink sink =
                eagle3_feature_sink(state, feature_lanes, target_valid, width, batch_size);
            target_verify_accept(
                state.execution, state.continuation_hidden_store, card,
                TargetVerifyFrameView{
                    .ids                     = verify_ids,
                    .cache_positions         = target_positions,
                    .rope_positions          = target_rope,
                    .valid_columns           = target_valid,
                    .kv_table_rows           = text_rows,
                    .state_source_slots      = state_sources,
                    .state_destination_slots = state_destinations,
                    .target_hidden           = g,
                    .target_logits           = target_logits,
                    .target_tokens           = target_tokens,
                    .drafts                  = current_drafts,
                    .current_extents         = current_extents,
                    .frontiers               = frontiers,
                    .anchors                 = anchors,
                    .licensed_tokens         = licensed_tokens,
                    .licensed_counts         = licensed_counts,
                    .accepted_drafts         = accepted,
                    .selected_hidden         = selected_hidden,
                    .replay_records          = state.execution.replay_records,
                    .sampling                = frame.sampling,
                    .feature_sink            = &sink,
                },
                envelopes.target_verify);
        }

        nvtx::ScopedRange draft_range(nvtx::Name::DecodeEagle3Draft, nvtx::Category::Eagle3,
                                      static_cast<std::uint64_t>(k) * batch_size);
        ops::mtp_prepare_next_round(verify_ids, anchors, accepted, frontiers, budgets,
                                    licensed_counts, rope_deltas, alignment_ids, next_extents,
                                    ar_positions, ar_rope_positions, ar_valid_columns,
                                    static_cast<std::int32_t>(state.text_cache.max_context()),
                                    state.execution.device.stream);
        // The verify wrote the three target layers into `features`; the encoder fuses them into g,
        // which the decoder overwrites as it consumes the accepted prefix.
        card.eagle3_encode_batch(features, g);
        card.eagle3_forward_decode_batch(alignment_ids, g, target_positions, target_rope,
                                         licensed_counts, eagle3_rows, envelopes.batch,
                                         alignment_hidden);
        ops::speculative_select_accepted_hidden(alignment_hidden, accepted, ar_hidden,
                                                state.execution.device.stream);

        const std::int32_t h   = dimension(config.text.hidden_size);
        Tensor proposal_logits = frame.proposal_logits.slice(1, 0, batch_size);
        Tensor draft0          = next_drafts.slice(1, 0, 1).view({batch_size});
        card.eagle3_propose_batch(ar_hidden, proposal_logits, draft0);
        for (std::uint32_t step = 0; step + 1 < k; ++step) {
            Tensor previous = next_drafts.slice(1, static_cast<std::int32_t>(step), 1).view(
                {batch_size});
            Tensor next = next_drafts.slice(1, static_cast<std::int32_t>(step + 1), 1).view(
                {batch_size});
            Tensor position = ar_positions.slice(1, static_cast<std::int32_t>(step), 1).view(
                {1, batch_size});
            Tensor rope = ar_rope_positions.slice(1, static_cast<std::int32_t>(step), 1).view(
                {1, batch_size});
            Tensor valid = ar_valid_columns.slice(1, static_cast<std::int32_t>(step), 1).view(
                {batch_size});
            Tensor previous_batch    = previous.view({1, batch_size});
            Tensor hidden_batch      = ar_hidden.view({h, 1, batch_size});
            Tensor next_hidden_batch = next_hidden.view({h, 1, batch_size});
            card.eagle3_forward_decode_batch(previous_batch, hidden_batch, position, rope, valid,
                                             eagle3_rows, envelopes.ar[step], next_hidden_batch);
            card.eagle3_propose_batch(next_hidden, proposal_logits, next);
            CUDA_CHECK(cudaMemcpyAsync(ar_hidden.data, next_hidden.data, ar_hidden.bytes(),
                                       cudaMemcpyDeviceToDevice, state.execution.device.stream));
        }

        CUDA_CHECK(cudaMemcpyAsync(&state.host_egress, frame.egress.data,
                                   sizeof(qwen3_5::MtpDecodeEgress), cudaMemcpyDeviceToHost,
                                   state.execution.device.stream));
    };
    run_prepared(state, executable, body);
}

} // namespace ninfer::models::qwen3_5::execution
