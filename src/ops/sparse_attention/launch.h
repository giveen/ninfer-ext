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
void qsa_attention_launch(const Tensor& q, const Tensor& selected, const Tensor& counts,
                          const Tensor& table_rows, const PagedKVBatchLayerView& kv, float scale,
                          Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
