#pragma once

#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/** Indexer key record per cached token: 128 raw BF16 key words, then three I32 rope positions. */
inline constexpr std::int32_t kQsaIndexKeyDim      = 128;
inline constexpr std::int32_t kQsaIndexRecordWords = 136;

/**
 * Qwen Sparse Attention geometry. Query heads read KV head floor(h / (query_heads/kv_heads)).
 * The indexer has `index_heads` query heads of `index_dim` and one key head. The first
 * `rotary_dim` features of both indexer heads rotate with interleaved multimodal RoPE: frequency
 * pair i uses `theta^(-2i/rotary_dim)` and axis `pair_axes[i]`, rotate-half convention.
 */
struct QsaGeometry {
    std::int32_t query_heads = 0;
    std::int32_t kv_heads    = 0;
    std::int32_t head_dim    = 0;
    std::int32_t index_heads = 0;
    std::int32_t index_dim   = 0;
    std::int32_t budget      = 0; // tokens; budget / ratio whole blocks are kept
    std::int32_t ratio       = 0;
    std::int32_t rotary_dim  = 0;
    float rope_theta         = 0.0F;
    float rms_norm_eps       = 0.0F;
    std::array<std::uint8_t, 64> pair_axes{};

    [[nodiscard]] constexpr std::int32_t block_topk() const noexcept { return budget / ratio; }

    [[nodiscard]] constexpr std::int32_t max_selected() const noexcept {
        return budget + ratio - 1;
    }
};

/** Paged indexer-key plane sharing the Text KV block tables (BF16 records, one per token). */
struct QsaIndexPlane {
    Tensor pages;        // BF16 [kQsaIndexRecordWords, page_tokens, 1, pages]
    Tensor block_tables; // I32 [table_stride, rows]
};

/**
 * Append attention K/V and indexer records for `W` columns of `B` lanes.
 *
 * `k`/`v` are contiguous BF16 `[256, kv_heads, W, B]` after the attention key norm and RoPE;
 * `index_keys` is contiguous BF16 `[128, W, B]` raw indexer keys (no norm, no RoPE);
 * `rope_positions` is I32 `[W*B, 1|3]` axis-major (one column means all three axes are equal);
 * `cache_positions` is I32 `[W, B]`; `table_rows` I32 `[B]` selects each lane's block-table row.
 * `valid_columns` (optional, I32 `[B]`) limits each lane to its first columns. KV storage follows
 * the cache profile (BFloat16 or Fp8E4M3Row256); the index record is stored exactly.
 */
void qsa_append(const Tensor& k, const Tensor& v, const Tensor& index_keys,
                const Tensor& rope_positions, const Tensor& cache_positions,
                const Tensor* valid_columns, const Tensor& table_rows,
                const PagedKVBatchLayerView& kv, const QsaIndexPlane& index, cudaStream_t stream);

[[nodiscard]] std::size_t qsa_select_workspace_bytes(const QsaGeometry& geometry,
                                                     std::uint32_t max_visible_keys,
                                                     std::int32_t columns);

/**
 * Select the visible cache indices of every query column.
 *
 * `index_query` is contiguous BF16 `[index_heads*index_dim, W, B]` raw indexer query rows;
 * `query_norm`/`key_norm` are BF16 `[index_dim]` zero-centred weights. Column (w,b) sees cache
 * indices `[0, cache_positions[w,b] + 1)`. For n visible tokens with m = floor(n/ratio) complete
 * blocks, block keys are `RoPE(offset_rmsnorm(mean(raw keys of the block)), position of its first
 * token)`; the score is `sum_heads relu(q_h . k_b) / sqrt(index_dim)` with q_h normalized and
 * rotated at the query position. The `min(block_topk, m)` highest-scoring blocks (lower block
 * index wins an exact tie) and the incomplete tail are written in ascending order to `selected`
 * (I32 `[max_selected, W, B]`) and their count to `counts` (I32 `[W, B]`). Columns beyond
 * `valid_columns` produce count zero.
 */
void qsa_select(const Tensor& index_query, const Tensor& query_rope_positions,
                const Tensor& cache_positions, const Tensor* valid_columns,
                const Tensor& table_rows, const Tensor& query_norm, const Tensor& key_norm,
                const QsaIndexPlane& index, const QsaGeometry& geometry,
                std::uint32_t max_visible_keys, WorkspaceArena& workspace, Tensor& selected,
                Tensor& counts, cudaStream_t stream);

/**
 * Softmax attention restricted to selected cache indices.
 *
 * `q` is contiguous BF16 `[head_dim, query_heads, W, B]` after norm and RoPE; `selected`/`counts`
 * come from qsa_select. For each column and query head the oracle is the scaled dot-product
 * softmax over exactly the selected keys, reading K/V at their storage boundary (FP16 V for the
 * BFloat16 profile; represented FP8 codes times scales, K in the stored Hadamard domain, for the
 * FP8 profile). `out` has the shape of `q`; a column with count zero writes zeros.
 */
void qsa_attention(const Tensor& q, const Tensor& selected, const Tensor& counts,
                   const Tensor& table_rows, const PagedKVBatchLayerView& kv, float scale,
                   Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
