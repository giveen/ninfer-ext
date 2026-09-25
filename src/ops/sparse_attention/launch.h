#pragma once

// ninfer::ops::detail - private launch prototypes for Qwen Sparse Attention.

#include "ninfer/ops/sparse_attention.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void qsa_append_launch(const Tensor& k, const Tensor& v, const Tensor& index_keys,
                       const Tensor& rope_positions, const Tensor& cache_positions,
                       const Tensor* valid_columns, const Tensor& table_rows,
                       const PagedKVBatchLayerView& kv, const QsaIndexPlane& index,
                       cudaStream_t stream);
[[nodiscard]] std::int32_t qsa_select_pass_columns(std::int32_t columns);
void qsa_select_launch(const Tensor& index_query, const Tensor& query_rope_positions,
                       const Tensor& cache_positions, const Tensor* valid_columns,
                       const Tensor& table_rows, const Tensor& query_norm, const Tensor& key_norm,
                       const QsaIndexPlane& index, const QsaGeometry& geometry,
                       std::int32_t max_blocks, float* scores, Tensor& selected, Tensor& counts,
                       cudaStream_t stream);

struct QsaAttentionSplit {
    std::int32_t splits;     // CTAs per (column, KV head)
    std::int32_t split_keys; // selected keys per split
};

[[nodiscard]] QsaAttentionSplit qsa_attention_split(std::int32_t columns, std::int32_t kv_heads,
                                                    std::int32_t max_selected);
[[nodiscard]] std::size_t qsa_attention_partial_bytes(std::int32_t columns,
                                                      std::int32_t query_heads,
                                                      std::int32_t kv_heads,
                                                      std::int32_t max_selected);
// `partial` holds qsa_attention_partial_bytes; it is unused when the call does not split.
void qsa_attention_launch(const Tensor& q, const Tensor& selected, const Tensor& counts,
                          const Tensor& table_rows, const PagedKVBatchLayerView& kv, float scale,
                          float* partial, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
