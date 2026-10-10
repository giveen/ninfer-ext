#pragma once

// Host-side entry to the tensor-core attention kernel family both Gemma attention Ops use.

#include "ops/kernel/gemma_flash_attention.cuh"

#include "core/device.h" // CUDA_CHECK

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {

inline std::int64_t flash_row_tiles(std::int32_t query_tokens, std::int32_t query_heads,
                                    std::int32_t kv_heads) {
    const std::int64_t rows = static_cast<std::int64_t>(query_tokens) * (query_heads / kv_heads);
    return (rows + kFlashRows - 1) / kFlashRows;
}

// Key splits per row tile: when the rows alone give fewer blocks than the device has SMs, the keys
// are divided until there are about two blocks per SM, keeping at least two key tiles per split.
inline std::int32_t flash_splits(std::int64_t row_tiles, std::int32_t kv_heads, std::int32_t batch,
                                 std::int32_t key_tokens) {
    const std::int64_t blocks   = row_tiles * kv_heads * batch;
    const std::int32_t key_tiles = (key_tokens + kFlashKeys - 1) / kFlashKeys;
    if (blocks >= 128 || key_tiles < 4) return 1;
    const std::int64_t wanted = std::min<std::int64_t>(64, (256 + blocks - 1) / blocks);
    return static_cast<std::int32_t>(std::max<std::int64_t>(1, std::min<std::int64_t>(wanted, key_tiles / 2)));
}

inline std::size_t flash_workspace_bytes(std::int32_t value_width, std::int32_t query_tokens,
                                         std::int32_t query_heads, std::int32_t kv_heads,
                                         std::int32_t key_tokens, std::int32_t batch) {
    if (query_tokens <= 0 || key_tokens <= 0 || batch <= 0 || kv_heads <= 0) return 0;
    const std::int64_t row_tiles = flash_row_tiles(query_tokens, query_heads, kv_heads);
    const std::int32_t splits    = flash_splits(row_tiles, kv_heads, batch, key_tokens);
    if (splits == 1) return 0;
    const std::int64_t rows = flash_partial_rows(row_tiles, kv_heads, batch);
    return static_cast<std::size_t>(splits) * static_cast<std::size_t>(rows) *
           (static_cast<std::size_t>(value_width) + 2) * sizeof(float);
}

// `workspace` may be empty; the keys are split only when it holds what the split needs, so a caller
// without one gets the one-pass route.
template <int QK, int V, bool kCompact>
void launch_gemma_flash_attention(FlashAttentionParams params, std::int32_t batch,
                                  const Tensor& workspace, cudaStream_t stream) {
    using Layout      = FlashLayout<QK, V, kCompact>;
    const auto kernel = gemma_flash_attention_kernel<QK, V, kCompact>;
    // Every route needs more than the default 48 KiB of dynamic shared memory at its widest.
    static const bool configured = [&] {
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        Layout::kBytes));
        return true;
    }();
    (void)configured;
    const std::int64_t row_tiles =
        flash_row_tiles(params.query_tokens, params.query_heads, params.kv_heads);
    const std::int32_t all_keys = params.key_tokens + params.extra_tokens;
    const std::size_t needed    = flash_workspace_bytes(V, params.query_tokens, params.query_heads,
                                                        params.kv_heads, all_keys, batch);
    params.splits = 1;
    if (needed > 0 && workspace.data != nullptr && workspace.bytes() >= needed) {
        params.splits = flash_splits(row_tiles, params.kv_heads, batch, all_keys);
        const std::int64_t rows = flash_partial_rows(row_tiles, params.kv_heads, batch);
        params.partial_out   = static_cast<float*>(workspace.data);
        params.partial_stats = params.partial_out + static_cast<std::int64_t>(params.splits) * rows * V;
    }
    const dim3 grid(static_cast<unsigned>(row_tiles * params.splits),
                    static_cast<unsigned>(params.kv_heads), static_cast<unsigned>(batch));
    kernel<<<grid, kFlashThreads, Layout::kBytes, stream>>>(params);
    CUDA_CHECK(cudaGetLastError());
    if (params.splits > 1) {
        const std::int64_t rows = flash_partial_rows(row_tiles, params.kv_heads, batch);
        gemma_flash_attention_combine_kernel<V>
            <<<static_cast<unsigned>(rows), 128, 0, stream>>>(params, row_tiles);
        CUDA_CHECK(cudaGetLastError());
    }
}

// Rows are copied in 16-byte units, so every row must start 16-byte aligned.
inline void require_flash_alignment(const void* data, std::int32_t row_elements, const char* op) {
    if (reinterpret_cast<std::uintptr_t>(data) % 16 != 0 || row_elements % 8 != 0) {
        throw std::invalid_argument(std::string(op) +
                                    ": rows must be 16-byte aligned for the tensor-core route");
    }
}

} // namespace ninfer::ops::detail
