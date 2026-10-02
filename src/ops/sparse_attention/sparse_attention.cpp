// ninfer::ops - Qwen Sparse Attention wrappers: validate the public contract, then launch.
#include "ninfer/ops/sparse_attention.h"

#include "ops/sparse_attention/launch.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require(bool condition, const char* op, const std::string& message) {
    if (!condition) { throw std::invalid_argument(std::string(op) + ": " + message); }
}

void require_dense(const Tensor& t, DType dtype, const char* op, const char* label) {
    require(t.dtype == dtype && t.data != nullptr && t.is_contiguous() && t.numel() > 0, op,
            std::string(label) + " must be contiguous and non-empty with the expected dtype");
}

void require_storage(const PagedKVBatchLayerView& kv, const char* op) {
    require(kv.head_dim == 256 && kv.num_kv_heads > 0 && kv.num_kv_heads <= 8, op,
            "KV geometry must be head_dim 256 with at most 8 KV heads");
    const PagedKVStorageLayout layout = paged_kv_storage_layout(kv.storage, kv.head_dim);
    const auto plane                  = [&](const Tensor& data, const Tensor& scales,
                           const PagedKVVectorLayout& vector, const char* label) {
        require(data.data != nullptr && data.dtype == vector.data_dtype &&
                    data.ne[0] == vector.data_leading_extent,
                op, std::string(label) + " pages do not match the KV-cache profile");
        require(!vector.has_scale() ||
                    (scales.data != nullptr && scales.dtype == vector.scale_dtype),
                op, std::string(label) + " scales do not match the KV-cache profile");
    };
    plane(kv.k_pages, kv.k_scale_pages, layout.key, "K");
    plane(kv.v_pages, kv.v_scale_pages, layout.value, "V");
    require(kv.block_tables.dtype == DType::I32 && kv.block_tables.data != nullptr, op,
            "block tables must be I32");
}

void require_positions(const Tensor& rope_positions, std::int64_t tokens, const char* op) {
    require(rope_positions.dtype == DType::I32 && rope_positions.data != nullptr &&
                rope_positions.is_contiguous() && rope_positions.ne[0] == tokens &&
                (rope_positions.ne[1] == 1 || rope_positions.ne[1] == 3),
            op, "rope positions must be I32 [W*B, 1|3]");
}

} // namespace

void qsa_append(const Tensor& k, const Tensor& v, const Tensor& index_keys,
                const Tensor& rope_positions, const Tensor& cache_positions,
                const Tensor* valid_columns, const Tensor& table_rows,
                const PagedKVBatchLayerView& kv, const QsaIndexPlane& index, cudaStream_t stream) {
    constexpr const char* op = "qsa_append";
    require_storage(kv, op);
    require_dense(cache_positions, DType::I32, op, "cache positions");
    const std::int64_t tokens = cache_positions.numel();
    require_dense(k, DType::BF16, op, "k");
    require_dense(v, DType::BF16, op, "v");
    require(k.numel() == tokens * kv.num_kv_heads * 256 && v.numel() == k.numel(), op,
            "k/v must be [256,kv_heads,W,B]");
    require_dense(index_keys, DType::BF16, op, "index keys");
    require(index_keys.numel() == tokens * kQsaIndexKeyDim, op, "index keys must be [128,W,B]");
    require_positions(rope_positions, tokens, op);
    require_dense(table_rows, DType::I32, op, "table rows");
    require(table_rows.numel() == cache_positions.ne[1], op, "one table row per lane");
    if (valid_columns != nullptr) {
        require_dense(*valid_columns, DType::I32, op, "valid columns");
        require(valid_columns->numel() == cache_positions.ne[1], op, "one valid count per lane");
    }
    require(index.pages.dtype == DType::BF16 && index.pages.data != nullptr &&
                index.pages.ne[0] == kQsaIndexRecordWords,
            op, "index plane must hold BF16 records of 136 words");
    detail::qsa_append_launch(k, v, index_keys, rope_positions, cache_positions, valid_columns,
                              table_rows, kv, index, stream);
}

std::size_t qsa_select_workspace_bytes(const QsaGeometry& geometry, std::uint32_t max_visible_keys,
                                       std::int32_t columns, std::int32_t lanes) {
    if (geometry.ratio <= 0 || columns <= 0 || lanes <= 0) {
        throw std::invalid_argument("qsa_select_workspace_bytes: invalid geometry");
    }
    const std::size_t blocks = max_visible_keys / static_cast<std::uint32_t>(geometry.ratio) + 1;
    std::size_t bytes        = blocks * sizeof(float) *
                                   static_cast<std::size_t>(detail::qsa_select_pass_columns(columns)) +
                               256;
    if (lanes == 1) {
        // Pooled block keys [1, blocks, index_dim], computed once per call.
        bytes += blocks * sizeof(float) * static_cast<std::size_t>(geometry.index_dim);
    }
    return bytes;
}

void qsa_select(const Tensor& index_query, const Tensor& query_rope_positions,
                const Tensor& cache_positions, const Tensor* valid_columns,
                const Tensor& table_rows, const Tensor& query_norm, const Tensor& key_norm,
                const QsaIndexPlane& index, const QsaGeometry& geometry,
                std::uint32_t max_visible_keys, WorkspaceArena& workspace, Tensor& selected,
                Tensor& counts, cudaStream_t stream) {
    constexpr const char* op = "qsa_select";
    require(geometry.index_dim == kQsaIndexKeyDim && geometry.index_heads > 0 &&
                geometry.index_heads <= 8 && geometry.ratio > 0 && geometry.budget > 0 &&
                geometry.budget % geometry.ratio == 0 && geometry.rotary_dim > 0 &&
                geometry.rotary_dim <= geometry.index_dim && geometry.rotary_dim % 8 == 0,
            op, "unsupported indexer geometry");
    require_dense(cache_positions, DType::I32, op, "cache positions");
    const std::int64_t columns = cache_positions.numel();
    require_dense(index_query, DType::BF16, op, "index query");
    require(index_query.numel() == columns * geometry.index_heads * geometry.index_dim, op,
            "index query must be [heads*dim,W,B]");
    require_positions(query_rope_positions, columns, op);
    require_dense(table_rows, DType::I32, op, "table rows");
    require_dense(query_norm, DType::BF16, op, "query norm");
    require_dense(key_norm, DType::BF16, op, "key norm");
    require_dense(selected, DType::I32, op, "selected");
    require(selected.ne[0] == geometry.max_selected() &&
                selected.numel() == columns * selected.ne[0],
            op, "selected must be [max_selected,W,B]");
    require_dense(counts, DType::I32, op, "counts");
    require(counts.numel() == columns, op, "counts must be [W,B]");
    if (valid_columns != nullptr) {
        require_dense(*valid_columns, DType::I32, op, "valid columns");
    }
    const std::int32_t max_blocks = static_cast<std::int32_t>(
        max_visible_keys / static_cast<std::uint32_t>(geometry.ratio) + 1);
    const std::int32_t lanes = static_cast<std::int32_t>(cache_positions.ne[1]);
    auto scope   = workspace.scope();
    auto scratch = workspace.alloc_bytes(qsa_select_workspace_bytes(
        geometry, max_visible_keys, static_cast<std::int32_t>(columns), lanes));
    float* scores = static_cast<float*>(scratch.data);
    float* pooled = nullptr;
    if (lanes == 1) {
        // The pooled block keys follow the per-column score rows in the same allocation.
        const std::size_t scores_bytes =
            static_cast<std::size_t>(max_blocks) *
            static_cast<std::size_t>(
                detail::qsa_select_pass_columns(static_cast<std::int32_t>(columns))) *
            sizeof(float);
        pooled = reinterpret_cast<float*>(static_cast<std::byte*>(scratch.data) + scores_bytes);
    }
    detail::qsa_select_launch(index_query, query_rope_positions, cache_positions, valid_columns,
                              table_rows, query_norm, key_norm, index, geometry, max_blocks,
                              scores, pooled, selected, counts, stream);
}

std::size_t qsa_attention_workspace_bytes(const QsaGeometry& geometry, std::int32_t columns) {
    if (geometry.kv_heads <= 0 || geometry.query_heads % geometry.kv_heads != 0 || columns <= 0) {
        throw std::invalid_argument("qsa_attention_workspace_bytes: invalid geometry");
    }
    // Narrower calls split further, so size for every width up to `columns`.
    std::size_t bytes = 0;
    for (std::int32_t c = 1; c <= columns; ++c) {
        const std::size_t call = detail::qsa_attention_partial_bytes(
            c, geometry.query_heads, geometry.kv_heads, geometry.max_selected());
        if (call == 0) { break; } // wider calls fill the GPU without splitting
        bytes = std::max(bytes, call);
    }
    return bytes + 256;
}

void qsa_attention(const Tensor& q, const Tensor& selected, const Tensor& counts,
                   const Tensor& table_rows, const PagedKVBatchLayerView& kv, float scale,
                   WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    constexpr const char* op = "qsa_attention";
    require_storage(kv, op);
    require_dense(q, DType::BF16, op, "q");
    require(q.ne[0] == 256 && q.ne[1] % kv.num_kv_heads == 0 && q.ne[1] / kv.num_kv_heads <= 16, op,
            "q must be [256,heads,W,B] with at most 16 heads per KV head");
    require_dense(out, DType::BF16, op, "out");
    require(out.numel() == q.numel() && out.data != q.data, op, "out must match q and not alias");
    require_dense(selected, DType::I32, op, "selected");
    require_dense(counts, DType::I32, op, "counts");
    require(counts.numel() == static_cast<std::int64_t>(q.ne[2]) * q.ne[3], op,
            "counts must be [W,B]");
    require_dense(table_rows, DType::I32, op, "table rows");
    require(table_rows.numel() == q.ne[3], op, "one table row per lane");
    const std::int32_t columns = q.ne[2] * q.ne[3];
    const std::size_t bytes    = detail::qsa_attention_partial_bytes(
        columns, q.ne[1], kv.num_kv_heads, static_cast<std::int32_t>(selected.ne[0]));
    auto scope     = workspace.scope();
    float* partial = bytes == 0 ? nullptr : static_cast<float*>(workspace.alloc_bytes(bytes).data);
    detail::qsa_attention_launch(q, selected, counts, table_rows, kv, scale, partial, out, stream);
}

} // namespace ninfer::ops
