#pragma once

// Implements: include/ninfer/ops/compact_kv_rows.h
// Match: one block dimension covers a compact row and one row is visited per grid step. Every
// entry is a bit-exact copy, so this is a gather with a computed source index and needs no vector
// route to be correct; the prefix that copies the value vector contiguously is the part a pack
// route would cover if this Op ever needs one.

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

__launch_bounds__(256) __global__ void compact_kv_rows_kernel(
    const __nv_bfloat16* v, const __nv_bfloat16* k, __nv_bfloat16* out, std::int32_t rotary_dim,
    std::int32_t rotary_pairs, std::int32_t compact_width, std::int64_t rows) {
    const std::int32_t entry = static_cast<std::int32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (entry >= compact_width) { return; }

    // The source dimension of this output entry: the value vector, the low rotated run, or the
    // high rotated run.
    const bool from_value = entry < rotary_dim;
    std::int32_t source   = entry;
    if (!from_value) {
        const std::int32_t i = entry - rotary_dim;
        source = i < rotary_pairs ? i : (rotary_dim / 2 + i - rotary_pairs);
    }

    for (std::int64_t row = blockIdx.y; row < rows; row += gridDim.y) {
        const std::int64_t source_offset = row * static_cast<std::int64_t>(rotary_dim) + source;
        out[row * static_cast<std::int64_t>(compact_width) + entry] =
            from_value ? v[source_offset] : k[source_offset];
    }
}

} // namespace ninfer::ops
