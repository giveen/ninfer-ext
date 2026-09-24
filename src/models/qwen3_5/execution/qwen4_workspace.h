#pragma once

// Qwen4Exp phase allocations shared by execution (qwen4_text.cpp) and startup sizing.

#include "models/qwen3_5/execution/parameters.h"
#include "ninfer/ops/offload_moe.h"
#include "ninfer/ops/sparse_attention.h"

#include <algorithm>
#include <stdexcept>
#include <cstdint>

namespace ninfer::models::qwen3_5::execution::workspace {

[[nodiscard]] inline std::int32_t qwen4_wide(const TextConfig& config) {
    return dimension(config.residual_width());
}

[[nodiscard]] inline std::int32_t qwen4_streams(const TextConfig& config) {
    return dimension(config.hyper_connection->hc_count);
}

[[nodiscard]] inline ops::QsaGeometry qwen4_qsa_geometry(const TextConfig& config) {
    const auto& attention = *config.attention;
    const auto& qsa       = *config.sparse_attention;
    const auto& rope      = *config.rope_parameters;
    ops::QsaGeometry g;
    g.query_heads  = dimension(attention.num_attention_heads);
    g.kv_heads     = dimension(attention.num_key_value_heads);
    g.head_dim     = dimension(attention.head_dim);
    g.index_heads  = dimension(qsa.indexer_heads);
    g.index_dim    = dimension(qsa.indexer_head_dim);
    g.budget       = dimension(qsa.budget);
    g.ratio        = dimension(qsa.compress_ratio);
    g.rotary_dim   = dimension(rope.rotary_dim);
    g.rope_theta   = rope.rope_theta;
    g.rms_norm_eps = config.rms_norm_eps;
    if (rope.pair_axes.size() > g.pair_axes.size()) {
        throw std::invalid_argument("QSA rotary pairs exceed the indexer geometry");
    }
    std::copy(rope.pair_axes.begin(), rope.pair_axes.end(), g.pair_axes.begin());
    return g;
}

struct Qwen4HcRoots {
    Tensor normalized, projection, lowrank, up;
};

template <class Allocator>
Qwen4HcRoots qwen4_hc(Allocator& a, const TextConfig& config, std::int32_t tokens,
                      std::int32_t inject_rows) {
    const auto rank = dimension(config.hyper_connection->hc_lowrank);
    return {a.alloc(DType::BF16, {qwen4_wide(config), tokens}),
            a.alloc(DType::BF16, {rank + inject_rows, tokens}),
            a.alloc(DType::BF16, {rank, tokens}),
            a.alloc(DType::BF16, {qwen4_wide(config), tokens})};
}

struct Qwen4QsaRoots {
    Tensor projection, query, key, gate, value, normalized_query, normalized_key;
    Tensor indexer, index_query, index_key, selected, counts, attention;
};

template <class Allocator>
Qwen4QsaRoots qwen4_qsa(Allocator& a, const TextConfig& config, std::int32_t tokens) {
    const auto qw    = dimension(config.attention->query_width());
    const auto kw    = dimension(config.attention->key_width());
    const auto& qsa  = *config.sparse_attention;
    const auto iq    = dimension(std::uint64_t(qsa.indexer_heads) * qsa.indexer_head_dim);
    const auto id    = dimension(qsa.indexer_head_dim);
    const auto width = dimension(qsa.max_selected_tokens());
    return {a.alloc(DType::BF16, {2 * qw + 2 * kw, tokens}),
            a.alloc(DType::BF16, {qw, tokens}),
            a.alloc(DType::BF16, {kw, tokens}),
            a.alloc(DType::BF16, {qw, tokens}),
            a.alloc(DType::BF16, {kw, tokens}),
            a.alloc(DType::BF16, {qw, tokens}),
            a.alloc(DType::BF16, {kw, tokens}),
            a.alloc(DType::BF16, {iq + id, tokens}),
            a.alloc(DType::BF16, {iq, tokens}),
            a.alloc(DType::BF16, {id, tokens}),
            a.alloc(DType::I32, {width, tokens}),
            a.alloc(DType::I32, {tokens}),
            a.alloc(DType::BF16, {qw, tokens})};
}

struct Qwen4GdnRoots {
    Tensor projection, control, a, b, g, beta, qkv, z, query, key, value, output;
    Tensor normalized_output;
};

template <class Allocator>
Qwen4GdnRoots qwen4_gdn(Allocator& a, const TextConfig& config, std::int32_t tokens) {
    const auto& gdn  = *config.gdn;
    const auto heads = dimension(gdn.linear_num_value_heads);
    const auto kw    = dimension(gdn.key_width());
    const auto vw    = dimension(gdn.value_width());
    const auto conv  = dimension(gdn.conv_channels());
    return {a.alloc(DType::BF16, {conv + vw, tokens}), a.alloc(DType::BF16, {2 * heads, tokens}),
            a.alloc(DType::BF16, {heads, tokens}),     a.alloc(DType::BF16, {heads, tokens}),
            a.alloc(DType::FP32, {heads, tokens}),     a.alloc(DType::FP32, {heads, tokens}),
            a.alloc(DType::BF16, {conv, tokens}),      a.alloc(DType::BF16, {vw, tokens}),
            a.alloc(DType::BF16, {kw, tokens}),        a.alloc(DType::BF16, {kw, tokens}),
            a.alloc(DType::BF16, {vw, tokens}),        a.alloc(DType::BF16, {vw, tokens}),
            a.alloc(DType::BF16, {vw, tokens})};
}

struct Qwen4PleRoots {
    Tensor key_value, key, value, normalized_key, normalized_query, gated, normalized_gated;
    Tensor slots;
};

template <class Allocator>
Qwen4PleRoots qwen4_ple(Allocator& a, const TextConfig& config, std::int32_t tokens) {
    const auto wide = qwen4_wide(config);
    const auto h    = dimension(config.hidden_size);
    return {a.alloc(DType::BF16, {wide + h, tokens}), a.alloc(DType::BF16, {wide, tokens}),
            a.alloc(DType::BF16, {h, tokens}),        a.alloc(DType::BF16, {wide, tokens}),
            a.alloc(DType::BF16, {wide, tokens}),     a.alloc(DType::BF16, {wide, tokens}),
            a.alloc(DType::BF16, {wide, tokens}),     a.alloc(DType::I32, {2})};
}

// Columns one cache resolution covers: the slot pool must hold every assignment of the group.
[[nodiscard]] inline std::int32_t qwen4_moe_group(std::int32_t tokens, std::int32_t slots) {
    return std::max(1, std::min(tokens, slots / ops::kOffloadMoeTopK));
}

struct Qwen4MoeRoots {
    Tensor ids, weights, shared_gate, shared_gate_up, shared_act, shared, slot_ids, misses;
};

template <class Allocator>
Qwen4MoeRoots qwen4_moe(Allocator& a, const TextConfig& config, std::int32_t tokens,
                        std::int32_t slots) {
    const auto& moe  = std::get<MoeConfig>(config.ffn);
    const auto width = dimension(moe.shared_expert_intermediate_size);
    const auto group = qwen4_moe_group(tokens, slots);
    return {a.alloc(DType::I32, {ops::kOffloadMoeTopK, tokens}),
            a.alloc(DType::FP32, {ops::kOffloadMoeTopK, tokens}),
            a.alloc(DType::FP32, {tokens}),
            a.alloc(DType::BF16, {2 * width, tokens}),
            a.alloc(DType::BF16, {width, tokens}),
            a.alloc(DType::BF16, {dimension(config.hidden_size), tokens}),
            a.alloc(DType::I32, {ops::kOffloadMoeTopK, tokens}),
            a.alloc(DType::I32, {2 * ops::kOffloadMoeTopK * group + 1})};
}

struct Qwen4BlockRoots {
    Tensor mixed, inject, output;
};

template <class Allocator>
Qwen4BlockRoots qwen4_block(Allocator& a, const TextConfig& config, std::int32_t tokens) {
    const auto h = dimension(config.hidden_size);
    return {a.alloc(DType::BF16, {h, tokens}),
            a.alloc(DType::FP32, {qwen4_streams(config), tokens}),
            a.alloc(DType::BF16, {h, tokens})};
}

struct Qwen4MtpStemRoots {
    Tensor embedding, normalized_embedding, projected_embedding, normalized_hidden,
        projected_hidden;
};

template <class Allocator>
Qwen4MtpStemRoots qwen4_mtp_stem(Allocator& a, const TextConfig& config, std::int32_t tokens) {
    const auto h = dimension(config.hidden_size);
    return {a.alloc(DType::BF16, {h, tokens}), a.alloc(DType::BF16, {h, tokens}),
            a.alloc(DType::BF16, {h, tokens}), a.alloc(DType::BF16, {qwen4_wide(config), tokens}),
            a.alloc(DType::BF16, {qwen4_wide(config), tokens})};
}

struct Qwen4PrefillRoots {
    Tensor ids, positions, rope_positions, residual, scatter_indices, ple;
};

template <class Allocator>
Qwen4PrefillRoots qwen4_prefill_roots(Allocator& a, const TextConfig& config, std::int32_t tokens,
                                      std::int32_t rope_axes, std::int32_t scatter_tokens) {
    Qwen4PrefillRoots out;
    out.ids       = a.alloc(DType::I32, {tokens});
    out.positions = a.alloc(DType::I32, {tokens});
    if (rope_axes != 0) { out.rope_positions = a.alloc(DType::I32, {tokens, rope_axes}); }
    out.residual = a.alloc(DType::BF16, {qwen4_wide(config), tokens});
    if (scatter_tokens != 0) { out.scatter_indices = a.alloc(DType::I32, {scatter_tokens}); }
    if (config.ple) { out.ple = a.alloc(DType::BF16, {dimension(config.ple->embed_dim), tokens}); }
    return out;
}

} // namespace ninfer::models::qwen3_5::execution::workspace
