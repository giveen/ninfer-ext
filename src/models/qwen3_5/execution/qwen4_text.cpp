// Qwen4Exp Text execution: hyper-connection residual, QSA and GDN mixers, PLE and the offloaded
// routed-expert MoE. The mathematics is defined in docs/maintainer/qwen4-exp-model.md.
#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/execution/text.h"
#include "models/qwen3_5/execution/attention.h"
#include "models/qwen3_5/execution/linear.h"
#include "models/qwen3_5/execution/qwen4_expert_pager.h"
#include "models/qwen3_5/execution/ple_gather.h"
#include "models/qwen3_5/execution/qwen4_workspace.h"
#include "models/qwen3_5/execution/workspace.h"
#include "models/qwen3_5/execution/vision.h"
#include "models/qwen3_5/execution/visual_scatter.h"
#include "models/qwen3_5/program/vision_control.h"

#include "core/nvtx.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/gdn_gating.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/hyper_connection.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/offload_moe.h"
#include "ninfer/ops/ple.h"
#include "ninfer/ops/position.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ninfer/ops/silu_mul.h"
#include "ninfer/ops/sparse_attention.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

namespace {

// Rows [row0, row0 + rows) of a contiguous [R, T] matrix as a contiguous tensor shaped like
// `destination` ([rows, T] elements). One column is already contiguous, so the result is a view
// into the source and nothing is copied; wider sources are copied into `destination`. Callers
// only read the result.
Tensor split_rows(const Tensor& source, std::int32_t row0, std::int32_t rows,
                  const Tensor& destination, cudaStream_t stream) {
    const std::size_t element  = dtype_size(source.dtype);
    const std::int64_t columns = source.numel() / source.ne[0];
    if (destination.dtype != source.dtype || destination.ne[0] != rows ||
        destination.numel() != static_cast<std::int64_t>(rows) * columns ||
        row0 + rows > source.ne[0] || !source.is_contiguous() || !destination.is_contiguous()) {
        throw std::invalid_argument("Qwen4Exp row split does not match its source");
    }
    if (columns == 1) {
        Tensor view = destination;
        view.data   = static_cast<std::byte*>(source.data) + static_cast<std::size_t>(row0) * element;
        return view;
    }
    CUDA_CHECK(cudaMemcpy2DAsync(
        destination.data, static_cast<std::size_t>(rows) * element,
        static_cast<const std::byte*>(source.data) + static_cast<std::size_t>(row0) * element,
        static_cast<std::size_t>(source.ne[0]) * element, static_cast<std::size_t>(rows) * element,
        static_cast<std::size_t>(columns), cudaMemcpyDeviceToDevice, stream));
    return destination;
}

void copy_i32(const std::int32_t* source, Tensor& destination, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(destination.data, source, destination.bytes(),
                               cudaMemcpyHostToDevice, stream));
}

template <class T>
class ScopedBinding {
public:
    ScopedBinding(T& slot, T value) : slot_(slot), previous_(slot) { slot_ = value; }

    ScopedBinding(const ScopedBinding&)            = delete;
    ScopedBinding& operator=(const ScopedBinding&) = delete;

    ~ScopedBinding() { slot_ = previous_; }

private:
    T& slot_;
    T previous_;
};

void hc_mix(const TextConfig& config, const HyperConnectionParameters& p, const Tensor& wide,
            Tensor& x, Tensor* inject, WorkspaceArena& work, cudaStream_t s) {
    const int T           = wide.ne[1];
    const std::int32_t hc = dimension(config.hyper_connection->hc_count);
    auto scope            = work.scope();
    auto roots            = workspace::qwen4_hc(work, config, T, p.inject_rows);
    ops::grouped_offset_rmsnorm(wide, p.norm, hc, config.rms_norm_eps, roots.normalized, s);
    project(roots.normalized, p.down, roots.projection, work, s);
    ops::hyper_connection_gates(roots.projection, hc, roots.lowrank,
                                p.inject_rows != 0 ? inject : nullptr, s);
    project(roots.lowrank, p.up, roots.up, work, s);
    ops::hyper_connection_collapse(roots.up, roots.normalized, hc, x, s);
}

} // namespace

void output_logits(const Parameters& parameters, const Tensor& hidden, Tensor& logits,
                   WorkspaceArena& work, cudaStream_t stream) {
    auto scope = work.scope();
    if (!parameters.qwen4) {
        ops::linear(hidden, parameters.text.output_head.weight, logits,
                    parameters.text.output_head.policy, work, stream);
        return;
    }
    const auto& config = parameters.model.config().text;
    Tensor x           = work.alloc(DType::BF16, {dimension(config.hidden_size), hidden.ne[1]});
    hc_mix(config, parameters.qwen4->head, hidden, x, nullptr, work, stream);
    ops::linear(x, parameters.text.output_head.weight, logits, parameters.text.output_head.policy,
                work, stream);
}

const Qwen4Runtime& TextContext::qwen4_runtime() const {
    if (qwen4_runtime_ == nullptr) {
        throw std::logic_error("Qwen4Exp execution has no bound Program runtime");
    }
    return *qwen4_runtime_;
}

TextContext::Qwen4Sequence TextContext::qwen4_text_sequence(std::int32_t columns) const {
    Qwen4Sequence out;
    out.cache_positions = active_cache_positions_ != nullptr ? active_cache_positions_ : &io_.pos;
    out.rope_positions = active_rope_positions_ != nullptr ? active_rope_positions_ : &io_.rope_pos;
    out.valid_columns  = active_valid_columns_;
    out.table_rows =
        active_kv_table_rows_ != nullptr ? active_kv_table_rows_ : &io_.text_kv_table_row;
    out.width = active_sequence_batch_ != 0 ? active_sequence_width_ : columns;
    out.batch = active_sequence_batch_ != 0 ? active_sequence_batch_ : 1;
    if (active_causal_attention_envelope_ == nullptr) {
        throw std::logic_error("Qwen4Exp attention execution envelope is not set");
    }
    out.max_visible = active_causal_attention_envelope_->max_visible_keys;
    if (out.width * out.batch != columns) {
        throw std::logic_error("Qwen4Exp sequence binding does not match aggregate columns");
    }
    return out;
}

void TextContext::qwen4_embed(const Tensor& ids, const Tensor* scatter_indices,
                              const Tensor* scatter_embeddings, const Tensor* input_embeddings,
                              Tensor& wide) {
    cudaStream_t s = ctx_.stream;
    const int T    = wide.ne[1];
    auto scope     = work_.scope();
    Tensor e;
    if (input_embeddings != nullptr) {
        e = input_embeddings->view({dimension(config_.hidden_size), T});
    } else {
        e = work_.alloc(DType::BF16, {dimension(config_.hidden_size), T});
        ops::embedding(ids.view({T}), *embed_, e, s);
        if (scatter_indices != nullptr) {
            ops::scatter(*scatter_embeddings, *scatter_indices, e, s);
        }
    }
    ops::hyper_connection_expand(e, dimension(config_.hyper_connection->hc_count), wide, s);
}

void TextContext::qwen4_hc_mix(const HyperConnectionParameters& p, const Tensor& wide, Tensor& x,
                               Tensor* inject) {
    hc_mix(config_, p, wide, x, inject, work_, ctx_.stream);
}

void TextContext::qwen4_qsa(const QsaParameters& p, const Tensor& x, const Qwen4Sequence& sequence,
                            const PagedKVCache& cache, std::uint32_t layer, Tensor* y) {
    cudaStream_t s                  = ctx_.stream;
    const int T                     = x.ne[1];
    const auto& attention           = *config_.attention;
    const auto head_dim             = dimension(attention.head_dim);
    const auto heads                = dimension(attention.num_attention_heads);
    const auto kv_heads             = dimension(attention.num_key_value_heads);
    const auto qw                   = dimension(attention.query_width());
    const auto kw                   = dimension(attention.key_width());
    const ops::QsaGeometry geometry = workspace::qwen4_qsa_geometry(config_);
    const std::int32_t W = sequence.width, B = sequence.batch;

    auto scope = work_.scope();
    auto roots = workspace::qwen4_qsa(work_, config_, T);
    project(x, p.projection, roots.projection, work_, s);
    const Tensor q    = split_rows(roots.projection, 0, qw, roots.query, s);
    const Tensor k    = split_rows(roots.projection, qw, kw, roots.key, s);
    const Tensor gate = split_rows(roots.projection, qw + kw, qw, roots.gate, s);
    const Tensor v    = split_rows(roots.projection, 2 * qw + kw, kw, roots.value, s);

    Tensor q3   = q.view({head_dim, heads, T});
    Tensor k3   = k.view({head_dim, kv_heads, T});
    Tensor qn   = roots.normalized_query.view({head_dim, heads, T});
    Tensor kn   = roots.normalized_key.view({head_dim, kv_heads, T});
    // Batched sequences bind [W,B] positions; only unbatched prefill carries [T,3] MRoPE rows.
    const bool mrope = active_sequence_batch_ == 0 && sequence.rope_positions->ne[1] == 3;
    Tensor rope      = mrope ? *sequence.rope_positions : sequence.rope_positions->view({T});
    text_qk_norm_rope(rope, *config_.rope_parameters, attention, config_.rms_norm_eps, p.query_norm,
                      p.key_norm, q3, k3, qn, kn, s);

    project(x, p.indexer, roots.indexer, work_, s);
    const std::int32_t index_query_rows = geometry.index_heads * geometry.index_dim;
    const Tensor index_query =
        split_rows(roots.indexer, 0, index_query_rows, roots.index_query, s);
    const Tensor index_key =
        split_rows(roots.indexer, index_query_rows, geometry.index_dim, roots.index_key, s);

    const Tensor positions         = sequence.cache_positions->view({W, B});
    const Tensor rope_rows =
        mrope ? *sequence.rope_positions : sequence.rope_positions->view({T, 1});
    const PagedKVBatchLayerView kv = cache.batch_layer_view(layer);
    const ops::QsaIndexPlane index = cache.index_plane(layer);
    // Appends write the step's own Device tail through the published tables. A staged step reads
    // its Host pages, K/V and index records, from the Device staging buffer instead of in place.
    ops::qsa_append(kn.view({head_dim, kv_heads, W, B}), v.view({head_dim, kv_heads, W, B}),
                    index_key.view({geometry.index_dim, W, B}), rope_rows, positions,
                    sequence.valid_columns, *sequence.table_rows, kv, index, s);
    if (y == nullptr) { return; }

    const bool text                         = &cache == batch_text_kv_;
    qwen3_5::KVHostStaging* staging         = text ? text_kv_staging_
                                              : &cache == batch_mtp_kv_ ? mtp_kv_staging_
                                                                        : nullptr;
    const qwen3_5::QsaIndexMirror* mirror   = text ? qsa_index_mirrors_.text
                                              : &cache == batch_mtp_kv_ ? qsa_index_mirrors_.mtp
                                                                        : nullptr;
    const PagedKVBatchLayerView read_kv     = staging ? staging->stage(kv, layer) : kv;
    const ops::QsaIndexPlane read_index     = staging  ? staging->stage_index(index, layer)
                                              : mirror ? mirror->plane(index, layer)
                                                       : index;
    Tensor selected = roots.selected.view({geometry.max_selected(), W, B});
    Tensor counts   = roots.counts.view({W, B});
    ops::qsa_select(index_query.view({index_query_rows, W, B}), rope_rows, positions,
                    sequence.valid_columns, *sequence.table_rows, p.indexer_query_norm,
                    p.indexer_key_norm, read_index, geometry, sequence.max_visible, work_, selected,
                    counts, s);

    Tensor a = roots.attention.view({head_dim, heads, W, B});
    ops::qsa_attention(qn.view({head_dim, heads, W, B}), selected, counts, *sequence.table_rows,
                       read_kv,
                       static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim))), work_, a,
                       s);
    Tensor a_flat = a.view({qw, T});
    ops::sigmoid_mul(gate, a_flat, s);
    project(a_flat, p.output, *y, work_, s);
}

void TextContext::qwen4_gdn(const Qwen4GdnParameters& p, const Tensor& x, int gidx, Phase ph,
                            Tensor& y) {
    cudaStream_t s    = ctx_.stream;
    const int T       = x.ne[1];
    const auto& gdn   = *config_.gdn;
    const auto heads  = dimension(gdn.linear_num_value_heads);
    const auto kwidth = dimension(gdn.key_width());
    const auto vwidth = dimension(gdn.value_width());
    const auto conv   = dimension(gdn.conv_channels());

    auto scope = work_.scope();
    auto roots = workspace::qwen4_gdn(work_, config_, T);
    project(x, p.projection, roots.projection, work_, s);
    project(x, p.control, roots.control, work_, s);
    const Tensor a = split_rows(roots.control, 0, heads, roots.a, s);
    const Tensor b = split_rows(roots.control, heads, heads, roots.b, s);
    Tensor g = roots.g, beta = roots.beta;
    ops::gdn_gating(a, b, p.a_log, p.dt_bias, g, beta, s);

    const Tensor qkv = split_rows(roots.projection, 0, conv, roots.qkv, s);
    const Tensor z   = split_rows(roots.projection, conv, vwidth, roots.z, s);
    Tensor qc = roots.query, kc = roots.key, vc = roots.value;

    if (ph == Phase::Verify) {
        if (active_sequence_batch_ == 0 || active_linear_state_source_slots_ == nullptr) {
            throw std::logic_error(
                "Verify GDN requires an explicit sequence batch and state slots");
        }
        const std::int32_t W = active_sequence_width_, B = active_sequence_batch_;
        if (gdn_state_action_ == GdnStateAction::UpdateInPlace && W != 1) {
            throw std::logic_error("In-place batched GDN update requires width one");
        }
        Tensor conv_states = state_.layer_view(static_cast<std::uint32_t>(gidx)).conv;
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        Tensor q3          = qc.view({kwidth, W, B});
        Tensor k3          = kc.view({kwidth, W, B});
        Tensor v3          = vc.view({vwidth, W, B});
        if (gdn_state_action_ == GdnStateAction::RecordForReplay) {
            GdnReplayRecordLayer records = replay_records_->layer(gidx, B);
            CUDA_CHECK(cudaMemcpyAsync(records.conv.data, qkv.data, qkv.bytes(),
                                       cudaMemcpyDeviceToDevice, s));
            ops::gdn_projected_conv_record(records.conv, p.convolution, conv_states, valid,
                                           *active_linear_state_source_slots_, q3, k3, v3, s);
        } else {
            ops::gdn_projected_conv_snapshot(qkv.view({conv, W, B}), p.convolution, conv_states,
                                             valid, *active_linear_state_source_slots_,
                                             *active_linear_state_destination_slots_, q3, k3, v3,
                                             s);
        }
    } else {
        Tensor state_in =
            state_.conv_slot(static_cast<std::uint32_t>(gidx), linear_state_source_slot_);
        Tensor state_out =
            state_.conv_slot(static_cast<std::uint32_t>(gidx), linear_state_destination_slot_);
        ops::causal_conv1d_silu_split(qkv, p.convolution, state_in, state_out, qc, kc, vc, s);
    }

    const auto dk  = dimension(gdn.linear_key_head_dim);
    const auto dv  = dimension(gdn.linear_value_head_dim);
    const auto hk  = dimension(gdn.linear_num_key_heads);
    Tensor q_r     = qc.view({dk, hk, T});
    Tensor k_r     = kc.view({dk, hk, T});
    Tensor v_r     = vc.view({dv, heads, T});
    Tensor o       = roots.output.view({dv, heads, T});
    const float sc = static_cast<float>(1.0 / std::sqrt(static_cast<double>(dk)));
    if (ph == Phase::Verify) {
        const std::int32_t W = active_sequence_width_, B = active_sequence_batch_;
        Tensor recurrent   = state_.layer_view(static_cast<std::uint32_t>(gidx)).recurrent;
        Tensor q4          = q_r.view({dk, hk, W, B});
        Tensor k4          = k_r.view({dk, hk, W, B});
        Tensor v4          = v_r.view({dv, heads, W, B});
        Tensor g3          = g.view({heads, W, B});
        Tensor beta3       = beta.view({heads, W, B});
        Tensor o4          = o.view({dv, heads, W, B});
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        if (gdn_state_action_ == GdnStateAction::RecordForReplay) {
            GdnReplayRecordLayer records = replay_records_->layer(gidx, B);
            ops::gated_delta_net_replay_record(q4, k4, v4, g3, beta3, sc, recurrent, valid,
                                               *active_linear_state_source_slots_, records.key,
                                               records.value, records.gate, o4, s);
        } else {
            ops::gated_delta_net_batch_update(q4, k4, v4, g3, beta3, sc, /*normalize_qk=*/true,
                                              recurrent, *active_linear_state_source_slots_,
                                              *active_linear_state_destination_slots_, o4, s);
        }
    } else {
        Tensor in =
            state_.recurrent_slot(static_cast<std::uint32_t>(gidx), linear_state_source_slot_);
        Tensor out =
            state_.recurrent_slot(static_cast<std::uint32_t>(gidx), linear_state_destination_slot_);
        ops::gated_delta_net(q_r, k_r, v_r, g, beta, sc, /*normalize_qk=*/true, work_, in, out, o,
                             s);
    }

    Tensor on = roots.normalized_output.view({dv, heads, T});
    ops::gated_rmsnorm_sigmoid(o, p.norm, z.view({dv, heads, T}), config_.rms_norm_eps, on, s);
    project(on.view({vwidth, T}), p.output, y, work_, s);
}

void TextContext::qwen4_ple(const PleParameters& p, const Tensor& embedding, Phase ph,
                            Tensor& wide) {
    cudaStream_t s               = ctx_.stream;
    const int T                  = wide.ne[1];
    const std::int32_t hc        = dimension(config_.hyper_connection->hc_count);
    const std::int32_t H         = dimension(config_.hidden_size);
    const std::int32_t wide_rows = hc * H;
    const auto& runtime          = qwen4_runtime();
    const auto& ple              = *config_.ple;

    auto scope = work_.scope();
    auto roots = workspace::qwen4_ple(work_, config_, T);
    project(embedding, p.key_value, roots.key_value, work_, s);
    const Tensor key   = split_rows(roots.key_value, 0, wide_rows, roots.key, s);
    const Tensor value = split_rows(roots.key_value, wide_rows, H, roots.value, s);
    Tensor key_n   = roots.normalized_key;
    Tensor query_n = roots.normalized_query;
    Tensor gated   = roots.gated;
    Tensor normed  = roots.normalized_gated;
    ops::grouped_offset_rmsnorm(key, p.key_norm, hc, config_.rms_norm_eps, key_n, s);
    ops::grouped_offset_rmsnorm(wide, p.query_norm, hc, config_.rms_norm_eps, query_n, s);
    ops::ple_gate(key_n, query_n, value, hc, gated, s);
    ops::grouped_offset_rmsnorm(gated, p.conv_norm, hc, config_.rms_norm_eps, normed, s);

    const auto dilation = static_cast<std::int32_t>(ple.conv_dilation());
    const auto history  = static_cast<std::int32_t>(ple.conv_history());
    Tensor states       = runtime.ple_states;
    std::int32_t W = T, B = 1;
    Tensor source_slots, destination_slots;
    if (ph == Phase::Verify) {
        W            = active_sequence_width_;
        B            = active_sequence_batch_;
        source_slots = *active_linear_state_source_slots_;
        if (active_linear_state_destination_slots_ != nullptr) {
            destination_slots = *active_linear_state_destination_slots_;
        }
    } else {
        source_slots      = roots.slots.slice(0, 0, 1);
        destination_slots = roots.slots.slice(0, 1, 1);
        ops::set_i32_scalar(source_slots, linear_state_source_slot_, s);
        ops::set_i32_scalar(destination_slots, linear_state_destination_slot_, s);
    }
    Tensor normed3 = normed.view({wide_rows, W, B});
    Tensor gated3  = gated.view({wide_rows, W, B});
    Tensor wide3   = wide.view({wide_rows, W, B});
    ops::ple_dilated_conv(normed3, gated3, p.convolution, dilation, states, source_slots, wide3, s);
    if (ph == Phase::Verify && gdn_state_action_ == GdnStateAction::RecordForReplay) {
        // The commit advances the history by the accepted prefix of these inputs.
        const Tensor& record = runtime.ple_record;
        if (record.data == nullptr || record.ne[0] != wide_rows || record.ne[1] < W ||
            record.ne[2] < B) {
            throw std::logic_error("Qwen4Exp PLE speculative record is unavailable");
        }
        const std::size_t column_bytes = static_cast<std::size_t>(wide_rows) * 2U;
        CUDA_CHECK(cudaMemcpy2DAsync(record.data, column_bytes * record.ne[1], normed3.data,
                                     column_bytes * W, column_bytes * W,
                                     static_cast<std::size_t>(B), cudaMemcpyDeviceToDevice, s));
    } else {
        const Tensor* valid = ph == Phase::Verify ? active_valid_columns_ : nullptr;
        ops::ple_conv_advance(normed3, valid, history, states, source_slots, destination_slots, s);
    }
}

void TextContext::qwen4_moe(const OffloadMoeParameters& p, const Tensor& x,
                            std::int32_t cache_layer, Tensor& y) {
    cudaStream_t s            = ctx_.stream;
    const int T               = x.ne[1];
    Qwen4ExpertPager& experts = *qwen4_runtime().experts;

    auto scope = work_.scope();
    auto roots = workspace::qwen4_moe(work_, config_, T, experts.cache().slots);
    ops::moe_route(x, p.router, work_, roots.ids, roots.weights, roots.shared_gate, s);
    project(x, p.shared_gate_up, roots.shared_gate_up, work_, s);
    const std::int32_t width = roots.shared_gate_up.ne[0] / 2;
    Tensor shared_act        = roots.shared_act;
    ops::silu_mul(roots.shared_gate_up.slice(0, 0, width),
                  roots.shared_gate_up.slice(0, width, width), shared_act, s);
    project(shared_act, p.shared_down, roots.shared, work_, s);

    if (experts.stages(T)) {
        // Stream whole layers: every expert sits at its own id in a staged bank. The next main
        // layer's fill starts now and overlaps this layer's compute.
        const auto next = static_cast<std::size_t>(cache_layer) + 1;
        const bool has_next = next < qwen4_->layers.size();
        // Layer 0 starts a main forward; the MTP block (cache layer past the main layers) runs as
        // its own forward. Either may follow cache-route calls that used the bank slots.
        const bool forward_start =
            cache_layer == 0 || static_cast<std::size_t>(cache_layer) == qwen4_->layers.size();
        const ops::ExpertWeights staged = experts.acquire_staged(
            cache_layer, p.bank, static_cast<std::int32_t>(next),
            has_next ? &qwen4_->layers[next].moe.bank : nullptr, forward_start, s);
        if (p.bank.gate_up_input_divisor > 0.0F) {
            ops::moe_experts_a4(x, roots.ids, roots.weights, roots.shared_gate, roots.shared,
                                staged, work_, y, s);
        } else {
            ops::moe_experts(x, roots.ids, roots.ids, roots.weights, roots.shared_gate,
                             roots.shared, staged, ops::kOffloadMoeExperts, work_, y, s);
        }
        experts.release_staged(cache_layer, s);
        return;
    }

    // A speculative round pads short drafts to the round width; those columns' outputs are never
    // consumed, so they reuse their lane's experts instead of fetching their own.
    if (active_valid_columns_ != nullptr && active_sequence_batch_ != 0 &&
        active_sequence_width_ > 1) {
        ops::moe_route_share_padding(roots.ids, *active_valid_columns_, active_sequence_width_, s);
    }

    // Cache route in column groups the slot pool can hold at once.
    const std::int32_t group        = std::min(T, experts.max_resolve_columns());
    const ops::ExpertWeights slots  = experts.slot_weights(p.bank);
    const std::int32_t slot_count   = experts.cache().slots;
    for (std::int32_t begin = 0; begin < T; begin += group) {
        const std::int32_t count = std::min(group, T - begin);
        Tensor ids               = roots.ids.slice(1, begin, count);
        Tensor slot_ids          = roots.slot_ids.slice(1, begin, count);
        Tensor misses            = roots.misses.slice(0, 0, 2 * count * ops::kOffloadMoeTopK + 1);
        experts.resolve(ids, cache_layer, p.bank, slot_ids, misses, s);
        Tensor y_slice = y.slice(1, begin, count);
        ops::moe_experts(
            x.slice(1, begin, count), ids, slot_ids, roots.weights.slice(1, begin, count),
            roots.shared_gate.slice(0, begin, count), roots.shared.slice(1, begin, count), slots,
            slot_count, work_, y_slice, s);
    }
}

void TextContext::qwen4_block(const Qwen4BlockParameters& p, Tensor& wide, Phase ph,
                              const Qwen4Sequence& sequence, const PagedKVCache& cache,
                              std::uint32_t kv_layer, int gdn_index, std::int32_t cache_layer,
                              const Tensor* ple_embedding) {
    cudaStream_t s = ctx_.stream;
    const int T    = wide.ne[1];
    if (p.ple) {
        if (ple_embedding == nullptr) {
            throw std::logic_error("Qwen4Exp PLE block has no n-gram embedding input");
        }
        qwen4_ple(*p.ple, *ple_embedding, ph, wide);
    }
    auto scope = work_.scope();
    auto roots = workspace::qwen4_block(work_, config_, T);
    Tensor x = roots.mixed, inject = roots.inject, y = roots.output;

    qwen4_hc_mix(p.attention_hc, wide, x, &inject);
    if (const auto* qsa = std::get_if<QsaParameters>(&p.mixer)) {
        qwen4_qsa(*qsa, x, sequence, cache, kv_layer, &y);
    } else {
        qwen4_gdn(std::get<Qwen4GdnParameters>(p.mixer), x, gdn_index, ph, y);
    }
    ops::hyper_connection_combine(y, inject, wide, s);

    qwen4_hc_mix(p.ffn_hc, wide, x, &inject);
    qwen4_moe(p.moe, x, cache_layer, y);
    ops::hyper_connection_combine(y, inject, wide, s);
}

void TextContext::qwen4_layers(Tensor& wide, Phase ph, const Qwen4Sequence& sequence,
                               const Tensor* ple_embedding) {
    const bool prefill = ph == Phase::Prefill;
    for (std::size_t layer = 0; layer < qwen4_->layers.size(); ++layer) {
        const bool full    = config_.layer_types[layer] == MixerKind::FullAttention;
        const auto compact = dimension(config_.compact_layer_indices[layer]);
        nvtx::ScopedRange layer_range(
            full ? (prefill ? nvtx::Name::PrefillLayerFull : nvtx::Name::VerifyLayerFull)
                 : (prefill ? nvtx::Name::PrefillLayerGdn : nvtx::Name::VerifyLayerGdn),
            full ? nvtx::Category::Attention : nvtx::Category::Gdn, layer);
        try {
            qwen4_block(qwen4_->layers[layer], wide, ph, sequence, *batch_text_kv_,
                        static_cast<std::uint32_t>(compact), compact,
                        static_cast<std::int32_t>(layer), ple_embedding);
        } catch (const std::exception& error) {
            throw std::runtime_error(
                "text/layers/" + std::to_string(layer) + (prefill ? " prefill" : " verify") +
                " columns=" + std::to_string(wide.ne[1]) + ": " + error.what());
        }
    }
}

void TextContext::qwen4_logits(const HyperConnectionParameters& head, const Tensor& wide,
                               Tensor& logits) {
    auto scope = work_.scope();
    Tensor x   = work_.alloc(DType::BF16, {dimension(config_.hidden_size), wide.ne[1]});
    qwen4_hc_mix(head, wide, x, nullptr);
    project(x, *lm_head_, logits, work_, ctx_.stream);
}

void TextContext::qwen4_proposal(const Tensor& hidden, Tensor& logits, Tensor& tokens) {
    const int T   = hidden.ne[1];
    Tensor window = logits.slice(1, 0, T);
    qwen4_logits(qwen4_mtp_->head, hidden, window);
    ops::argmax(window, tokens, dimension(parameters_.model.resources().public_token_count),
                ctx_.stream);
}

void TextContext::qwen4_mtp_stem(const Tensor& ids, const Tensor& hidden,
                                 const Tensor* input_embeddings, Tensor& wide) {
    cudaStream_t s        = ctx_.stream;
    const int T           = static_cast<int>(ids.numel());
    const std::int32_t H  = dimension(config_.hidden_size);
    const std::int32_t hc = dimension(config_.hyper_connection->hc_count);
    Tensor flat_input     = hidden.view({hc * H, T});
    {
        auto stem        = work_.scope();
        auto roots       = workspace::qwen4_mtp_stem(work_, config_, T);
        Tensor embedding = roots.embedding;
        if (input_embeddings != nullptr) {
            embedding = input_embeddings->view({H, T});
        } else {
            ops::embedding(ids.view({T}), *embed_, embedding, s);
        }
        Tensor normalized = roots.normalized_embedding;
        ops::rmsnorm(embedding, qwen4_mtp_->embedding_norm, config_.rms_norm_eps, true, normalized,
                     s);
        Tensor projected = roots.projected_embedding;
        project(normalized, qwen4_mtp_->embedding_projection, projected, work_, s);
        Tensor normalized_hidden = roots.normalized_hidden;
        ops::rmsnorm(flat_input, qwen4_mtp_->hidden_norm, config_.rms_norm_eps, true,
                     normalized_hidden, s);
        Tensor streams = roots.projected_hidden.view({H, hc * T});
        project(normalized_hidden.view({H, hc * T}), qwen4_mtp_->hidden_projection, streams, work_,
                s);
        ops::hyper_connection_expand(projected, hc, wide, s);
        ops::residual_add(roots.projected_hidden, wide, s);
    }
}

void TextContext::qwen4_mtp_core(const Tensor& ids, const Tensor& hidden,
                                 const Tensor* input_embeddings, const Qwen4Sequence& sequence,
                                 Tensor& mtp_hidden) {
    if (batch_mtp_kv_ == nullptr || qwen4_mtp_ == nullptr) {
        throw std::runtime_error("Qwen4Exp MTP forward is not enabled");
    }
    const int T           = static_cast<int>(ids.numel());
    const std::int32_t H  = dimension(config_.hidden_size);
    const std::int32_t hc = dimension(config_.hyper_connection->hc_count);
    nvtx::ScopedRange range(nvtx::Name::MtpForward, nvtx::Category::Mtp,
                            static_cast<std::uint64_t>(T));
    auto scope  = work_.scope();
    Tensor wide = mtp_hidden.view({hc * H, T});
    qwen4_mtp_stem(ids, hidden, input_embeddings, wide);
    qwen4_block(qwen4_mtp_->layer, wide, Phase::Verify, sequence, *batch_mtp_kv_, 0, -1,
                dimension(config_.num_hidden_layers), nullptr);
}

void TextContext::qwen4_mtp_append(const Tensor& ids, const Tensor& hidden,
                                   const Qwen4Sequence& sequence) {
    if (batch_mtp_kv_ == nullptr || qwen4_mtp_ == nullptr) {
        throw std::runtime_error("Qwen4Exp MTP forward is not enabled");
    }
    const auto* qsa = std::get_if<QsaParameters>(&qwen4_mtp_->layer.mixer);
    if (qsa == nullptr) { throw std::logic_error("Qwen4Exp MTP layer has no QSA mixer"); }
    const int T           = static_cast<int>(ids.numel());
    const std::int32_t H  = dimension(config_.hidden_size);
    const std::int32_t hc = dimension(config_.hyper_connection->hc_count);
    nvtx::ScopedRange range(nvtx::Name::MtpForward, nvtx::Category::Mtp,
                            static_cast<std::uint64_t>(T));
    auto scope  = work_.scope();
    Tensor wide = work_.alloc(DType::BF16, {hc * H, T});
    qwen4_mtp_stem(ids, hidden, nullptr, wide);
    auto roots = workspace::qwen4_block(work_, config_, T);
    qwen4_hc_mix(qwen4_mtp_->layer.attention_hc, wide, roots.mixed, &roots.inject);
    qwen4_qsa(*qsa, roots.mixed, sequence, *batch_mtp_kv_, 0, nullptr);
}

PrefillChunkResult TextContext::qwen4_prefill(std::span<const int> ids,
                                              const TextPrefill* text_prefill,
                                              const MultimodalPrefill* multimodal,
                                              bool finalize_at_end) {
    runtime::ExecutionTimingRecorder timing;
    if (ids.empty()) { throw std::invalid_argument("TextContext::prefill requires tokens"); }
    const auto& runtime          = qwen4_runtime();
    cudaStream_t s               = ctx_.stream;
    const int T                  = static_cast<int>(ids.size());
    const int chunk              = static_cast<int>(prefill_chunk_);
    const std::uint32_t base     = text_kv_base_;
    const std::int32_t H         = dimension(config_.hidden_size);
    const std::int32_t wide_rows = dimension(config_.residual_width());

    std::span<const int> history = ids;
    if (text_prefill != nullptr) {
        if (multimodal != nullptr || base != text_prefill->begin ||
            text_prefill->token_ids.size() < static_cast<std::size_t>(base) + ids.size()) {
            throw std::invalid_argument("text prefill chunk does not match its full prompt");
        }
        history = text_prefill->token_ids;
    }
    if (multimodal != nullptr) {
        if (base != multimodal->begin ||
            multimodal->token_ids.size() < static_cast<std::size_t>(base) + ids.size() ||
            multimodal->positions.size() != 3 * multimodal->token_ids.size() ||
            multimodal->vision == nullptr) {
            throw std::invalid_argument("multimodal prefill does not match its cache base");
        }
        rope_delta_ = multimodal->rope_delta;
        history     = multimodal->token_ids;
    } else if (text_kv_base_ == 0) {
        rope_delta_ = 0;
    }
    if (text_prefill == nullptr && multimodal == nullptr && base != 0) {
        throw std::invalid_argument("Qwen4Exp continued prefill requires its full prompt");
    }
    ops::set_i32_scalar(io_.rope_delta, rope_delta_, s);
    if (static_cast<std::uint64_t>(base) + static_cast<std::uint64_t>(T) >
        static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("TextContext::prefill absolute position exceeds int32");
    }
    const int base_i              = static_cast<int>(base);
    const std::int64_t split_abs  = prefill_split_frontier_;
    const bool has_split          = split_abs > base_i && split_abs <= base_i + T;
    const int split_rel           = has_split ? static_cast<int>(split_abs - base_i) : -1;
    const bool prepare_mtp_prompt = mtp_enabled() && io_.mtp.has_value();
    if (prepare_mtp_prompt &&
        mtp_proposal_extent_ > static_cast<std::uint32_t>(io_.mtp->draft_tokens.ne[0])) {
        throw std::logic_error("MTP proposal extent exceeds the configured draft window");
    }

    int len = std::min(chunk, T);
    if (split_rel > 0 && len > split_rel) { len = split_rel; }
    work_.reset();
    VisionChunk vision_chunk;
    const std::uint32_t prompt_t0 = base;
    if (multimodal != nullptr) {
        vision_chunk =
            multimodal->vision->prepare_chunk(prompt_t0, static_cast<std::uint32_t>(len));
        len = vision_chunk.length;
    }
    const bool is_last = finalize_at_end && len == T;
    nvtx::ScopedRange chunk_range(nvtx::Name::PrefillChunk, nvtx::Category::Prefill,
                                  static_cast<std::uint64_t>(len));
    {
        std::vector<std::int32_t> local_scatter;
        std::int32_t visual_begin = 0;
        if (!vision_chunk.scatter.empty()) {
            const auto scatter =
                vision_chunk.scatter;
            const auto begin = std::ranges::lower_bound(scatter, prompt_t0);
            const auto end   = std::lower_bound(begin, scatter.end(), prompt_t0 + len);
            visual_begin     = static_cast<std::int32_t>(begin - scatter.begin());
            for (auto it = begin; it != end; ++it) {
                local_scatter.push_back(*it - static_cast<std::int32_t>(prompt_t0));
            }
        }
        const std::int32_t rope_axes = multimodal != nullptr ? 3 : (rope_delta_ != 0 ? 1 : 0);
        auto roots                   = workspace::qwen4_prefill_roots(
            work_, config_, len, rope_axes, static_cast<std::int32_t>(local_scatter.size()));
        Tensor ids_device = roots.ids;
        copy_i32(ids.data(), ids_device, s);
        Tensor positions = roots.positions;
        ops::fill_i32_positions(positions, base_i, s);
        Tensor rope_positions = positions;
        std::vector<std::int32_t> rope_host;
        if (multimodal != nullptr) {
            rope_positions = roots.rope_positions;
            rope_host.resize(static_cast<std::size_t>(3) * len);
            const std::size_t prompt_tokens = multimodal->token_ids.size();
            for (int axis = 0; axis < 3; ++axis) {
                std::copy_n(multimodal->positions.data() +
                                static_cast<std::size_t>(axis) * prompt_tokens + prompt_t0,
                            len, rope_host.data() + static_cast<std::size_t>(axis) * len);
            }
            copy_i32(rope_host.data(), rope_positions, s);
        } else if (rope_delta_ != 0) {
            rope_positions = roots.rope_positions;
            ops::offset_i32_positions(positions, io_.rope_delta, rope_positions, s);
        }
        const auto visible = static_cast<std::uint32_t>(base_i + len);
        const ops::CausalAttentionExecutionEnvelope envelope{visible, visible};
        const Qwen4Sequence sequence{.cache_positions = &positions,
                                     .rope_positions  = &rope_positions,
                                     .valid_columns   = nullptr,
                                     .table_rows      = &io_.text_kv_table_row,
                                     .width           = len,
                                     .batch           = 1,
                                     .max_visible     = envelope.max_visible_keys};

        Tensor scatter_indices, scatter_embeddings;
        if (!local_scatter.empty()) {
            scatter_indices = roots.scatter_indices;
            copy_i32(local_scatter.data(), scatter_indices, s);
            scatter_embeddings = vision_chunk.embeddings.slice(
                1, visual_begin, static_cast<std::int32_t>(local_scatter.size()));
        }
        Tensor wide = roots.residual;
        qwen4_embed(ids_device, local_scatter.empty() ? nullptr : &scatter_indices,
                    &scatter_embeddings, nullptr, wide);

        Tensor ple_embedding;
        if (config_.ple) {
            if (runtime.ple_gather == nullptr || runtime.ple_host == nullptr) {
                throw std::logic_error("Qwen4Exp prefill has no PLE gather staging");
            }
            const std::uint32_t width = runtime.ple_gather->width();
            std::vector<std::int32_t> tokens(history.begin(), history.end());
            runtime.ple_gather->gather(
                tokens, text_prefill != nullptr || multimodal != nullptr ? prompt_t0 : 0,
                static_cast<std::size_t>(len),
                std::span<std::uint16_t>(runtime.ple_host, static_cast<std::size_t>(width) * len));
            ple_embedding = roots.ple;
            CUDA_CHECK(cudaMemcpyAsync(ple_embedding.data, runtime.ple_host, ple_embedding.bytes(),
                                       cudaMemcpyHostToDevice, s));
        }
        {
            ScopedBinding<const Tensor*> cache_binding(active_cache_positions_, &positions);
            ScopedBinding<const Tensor*> rope_binding(active_rope_positions_, &rope_positions);
            ScopedBinding<const ops::CausalAttentionExecutionEnvelope*> envelope_binding(
                active_causal_attention_envelope_, &envelope);
            qwen4_layers(wide, Phase::Prefill, sequence, config_.ple ? &ple_embedding : nullptr);
        }

        Tensor xf = prefill_hidden_.data != nullptr ? prefill_hidden_.slice(1, 0, len)
                                                    : work_.alloc(DType::BF16, {wide_rows, len});
        CUDA_CHECK(cudaMemcpyAsync(xf.data, wide.data, wide.bytes(), cudaMemcpyDeviceToDevice, s));

        if (is_last) {
            Tensor logits = io_.logits.slice(1, 0, 1);
            qwen4_logits(qwen4_->head, xf.slice(1, len - 1, 1), logits);
            ops::set_i32_scalar(io_.pos, base_i + T, s);
            ops::set_i32_scalar(io_.rope_pos, base_i + T + rope_delta_, s);
            if (sampling_config_ != nullptr) {
                ops::sample(logits, io_.token,
                            dimension(parameters_.model.resources().public_token_count),
                            sampling_config_, io_.pos, ops::kSamplePurposePrefill, work_, s);
            } else {
                ops::argmax(logits, io_.token,
                            dimension(parameters_.model.resources().public_token_count), s);
            }
        }

        if (prepare_mtp_prompt) {
            const std::uint32_t alignment_tokens     = static_cast<std::uint32_t>(history.size());
            const qwen3_5::MtpAlignmentWindow window = qwen3_5::plan_mtp_alignment_window(
                alignment_tokens, prompt_t0, static_cast<std::uint32_t>(len));
            const int prompt_columns =
                len - static_cast<int>(window.final_column_uses_generated_token);
            auto mtp_scope = work_.scope();
            Tensor mtp_ids = work_.alloc(DType::I32, {len});
            if (prompt_columns != 0) {
                Tensor prompt_ids = mtp_ids.slice(0, 0, prompt_columns);
                copy_i32(history.data() + window.shifted_embedding_begin, prompt_ids, s);
            }
            if (window.final_column_uses_generated_token) {
                CUDA_CHECK(cudaMemcpyAsync(mtp_ids.slice(0, len - 1, 1).data, io_.token.data,
                                           sizeof(std::int32_t), cudaMemcpyDeviceToDevice, s));
            }
            Tensor mtp_embeddings;
            const Tensor* mtp_embeddings_ptr = nullptr;
            if (multimodal != nullptr) {
                mtp_embeddings = work_.alloc(DType::BF16, {H, len});
                ops::embedding(mtp_ids, *embed_, mtp_embeddings, s);
                if (!vision_chunk.scatter.empty()) {
                    const qwen3_5::MtpVisualOverlap overlap = qwen3_5::shifted_visual_overlap(
                        vision_chunk.scatter, alignment_tokens, window);
                    if (!overlap.empty()) {
                        Tensor shifted = workspace::visual_scatter_indices(
                            work_, static_cast<std::int32_t>(overlap.size()));
                        qwen3_5::detail::scatter_shifted_visual_embeddings(
                            mtp_embeddings, vision_chunk.embeddings, overlap, shifted, s);
                    }
                }
                mtp_embeddings_ptr = &mtp_embeddings;
            }
            const Qwen4Sequence mtp_sequence{.cache_positions = &positions,
                                             .rope_positions  = &rope_positions,
                                             .valid_columns   = nullptr,
                                             .table_rows      = &io_.backend_kv_table_row,
                                             .width           = len,
                                             .batch           = 1,
                                             .max_visible     = envelope.max_visible_keys};
            Tensor mtp_hidden = work_.alloc(DType::BF16, {wide_rows, len});
            qwen4_mtp_core(mtp_ids, xf, mtp_embeddings_ptr, mtp_sequence, mtp_hidden);
            if (is_last && mtp_proposal_extent_ != 0) {
                Tensor logits     = io_.logits.slice(1, 0, 1);
                Tensor draft0     = io_.mtp->draft_tokens.slice(0, 0, 1);
                Tensor last_draft = mtp_hidden.slice(1, len - 1, 1);
                CUDA_CHECK(cudaMemcpyAsync(io_.mtp->ar_hidden.data, last_draft.data,
                                           last_draft.bytes(), cudaMemcpyDeviceToDevice, s));
                qwen4_proposal(io_.mtp->ar_hidden, logits, draft0);
                Tensor ar_position = io_.mtp->position.slice(0, 0, 1);
                ops::set_i32_scalar(ar_position, base_i + T, s);
                for (int i = 1; i < static_cast<int>(mtp_proposal_extent_); ++i) {
                    Tensor previous       = io_.mtp->draft_tokens.slice(0, i - 1, 1);
                    Tensor next           = io_.mtp->draft_tokens.slice(0, i, 1);
                    Tensor next_hidden    = work_.alloc(DType::BF16, {wide_rows, 1});
                    const auto ar_visible = static_cast<std::uint32_t>(base_i + T + i);
                    mtp_forward_ar_step(previous, io_.mtp->ar_hidden, ar_position,
                                        {ar_visible, ar_visible}, next_hidden, logits, next);
                    CUDA_CHECK(cudaMemcpyAsync(io_.mtp->ar_hidden.data, next_hidden.data,
                                               io_.mtp->ar_hidden.bytes(), cudaMemcpyDeviceToDevice,
                                               s));
                    ops::increment_i32_scalar(ar_position, s);
                }
            }
        }

        if (split_rel > 0 && len == split_rel && rewrite_checkpoint_hidden_output_ != nullptr) {
            const Tensor checkpoint = xf.slice(1, len - 1, 1);
            CUDA_CHECK(cudaMemcpyAsync(rewrite_checkpoint_hidden_output_->data, checkpoint.data,
                                       checkpoint.bytes(), cudaMemcpyDeviceToDevice, s));
        }
    }

    prefill_split_frontier_ = -1;
    timing.begin_wait();
    ctx_.synchronize();
    timing.end_wait();
    work_.reset();
    return PrefillChunkResult{.processed_tokens = static_cast<std::uint32_t>(len),
                              .finalized        = finalize_at_end && len == T,
                              .timing           = timing.finish()};
}

} // namespace ninfer::models::qwen3_5::execution
