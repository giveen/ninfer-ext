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

// The most resident blocks a split decision assumes, which bounds the workspace on any device; a
// device with more is treated as having this many.
inline constexpr std::int64_t kFlashMaxResident = 512;

// Key splits per row tile. `resident` is how many blocks of the kernel the device runs at once.
// When the rows alone fill less than half of that, the keys are divided so the grid is one full
// wave: a second, partial wave would leave most SMs idle while it runs. A split keeps at least two
// key tiles. The count depends only on the shapes and the device, never on how many keys are filled,
// so a captured decode step keeps its grid; each block finds its key range from the filled length.
inline std::int32_t flash_splits(std::int64_t row_tiles, std::int32_t kv_heads, std::int32_t batch,
                                 std::int32_t key_tokens, std::int64_t resident) {
    resident                     = std::min(resident, kFlashMaxResident);
    const std::int64_t blocks    = row_tiles * kv_heads * batch;
    if (blocks <= 0) return 1;
    const std::int32_t key_tiles = (key_tokens + kFlashKeys - 1) / kFlashKeys;
    if (2 * blocks > resident || key_tiles < 4) return 1;
    const std::int64_t wanted = std::min<std::int64_t>(kFlashMaxSplits, resident / blocks);
    return static_cast<std::int32_t>(std::max<std::int64_t>(1, std::min<std::int64_t>(wanted, key_tiles / 2)));
}

// Bytes the split route may need on any device: the splits at kFlashMaxResident.
inline std::size_t flash_workspace_bytes(std::int32_t value_width, std::int32_t query_tokens,
                                         std::int32_t query_heads, std::int32_t kv_heads,
                                         std::int32_t key_tokens, std::int32_t batch) {
    if (query_tokens <= 0 || key_tokens <= 0 || batch <= 0 || kv_heads <= 0) return 0;
    const std::int64_t row_tiles = flash_row_tiles(query_tokens, query_heads, kv_heads);
    const std::int32_t splits =
        flash_splits(row_tiles, kv_heads, batch, key_tokens, kFlashMaxResident);
    if (splits == 1) return 0;
    const std::int64_t rows = flash_partial_rows(row_tiles, kv_heads, batch);
    return static_cast<std::size_t>(splits) * static_cast<std::size_t>(rows) *
           (static_cast<std::size_t>(value_width) + 2) * sizeof(float);
}

// Row tiles per KV head from which a pass takes the tiled kernel. Below it the tiled grid is too
// small to fill the GPU, and the decode kernel's 16-row blocks and key split do better.
inline constexpr std::int64_t kFlashTiledMinTiles = 8;

template <int QK, int V, bool kCompact>
void launch_gemma_flash_attention_tiled(const FlashAttentionParams& params, std::int32_t batch,
                                        cudaStream_t stream) {
    using Config      = FlashTiledConfig<QK, V, kCompact>;
    const auto kernel = gemma_flash_attention_tiled_kernel<QK, V, kCompact>;
    static const bool configured = [&] {
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        Config::kBytes));
        return true;
    }();
    (void)configured;
    const std::int64_t rows =
        static_cast<std::int64_t>(params.query_tokens) * (params.query_heads / params.kv_heads);
    const dim3 grid(static_cast<unsigned>((rows + Config::kRows - 1) / Config::kRows),
                    static_cast<unsigned>(params.kv_heads), static_cast<unsigned>(batch));
    kernel<<<grid, Config::kThreads, Config::kBytes, stream>>>(params);
    CUDA_CHECK(cudaGetLastError());
}

// `workspace` may be empty; the keys are split only when it holds what the split needs, so a caller
// without one gets the one-pass route. Passes with many query rows take the tiled kernel, which
// never splits.
template <int QK, int V, bool kCompact>
void launch_gemma_flash_attention(FlashAttentionParams params, std::int32_t batch,
                                  const Tensor& workspace, DeviceExecutionView execution) {
    const cudaStream_t stream = execution.stream;
    if (static_cast<std::int64_t>(params.query_tokens) * (params.query_heads / params.kv_heads) >=
        kFlashTiledMinTiles * FlashTiledConfig<QK, V, kCompact>::kRows) {
        launch_gemma_flash_attention_tiled<QK, V, kCompact>(params, batch, stream);
        return;
    }
    using Layout      = FlashLayout<QK, V, kCompact>;
    const auto kernel = gemma_flash_attention_kernel<QK, V, kCompact>;
    // Every route needs more than the default 48 KiB of dynamic shared memory at its widest. The
    // blocks one SM holds at once fix how many a full wave is.
    static const int per_sm = [&] {
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        Layout::kBytes));
        int blocks = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, kFlashThreads,
                                                                 Layout::kBytes));
        return std::max(blocks, 1);
    }();
    if (execution.multiprocessor_count <= 0) {
        throw std::invalid_argument("gemma flash attention: the execution view has no SM count");
    }
    const std::int64_t resident =
        static_cast<std::int64_t>(per_sm) * execution.multiprocessor_count;
    const std::int64_t row_tiles =
        flash_row_tiles(params.query_tokens, params.query_heads, params.kv_heads);
    const std::int32_t all_keys = params.key_tokens + params.extra_tokens;
    const std::size_t needed    = flash_workspace_bytes(V, params.query_tokens, params.query_heads,
                                                        params.kv_heads, all_keys, batch);
    params.splits = 1;
    if (needed > 0 && workspace.data != nullptr && workspace.bytes() >= needed) {
        params.splits = flash_splits(row_tiles, params.kv_heads, batch, all_keys, resident);
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
            <<<static_cast<unsigned>(rows), kFlashCombineThreads, 0, stream>>>(params, row_tiles);
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
