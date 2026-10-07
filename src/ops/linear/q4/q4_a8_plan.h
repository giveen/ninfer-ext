#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/layout.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {

// Materialized per-token int8 activation for the Q4 W4A8 route: one signed code per input row
// plus a per-token absmax scale.  The weights stay Q4 group-64; the codes are sign-extended to
// int8 in the MMA kernel's shared stage.
struct Q4A8Workspace {
    std::int8_t* codes = nullptr;
    float* scales      = nullptr;
};

inline std::size_t q4_a8_checked_bytes(std::int32_t tokens, std::size_t bytes_per_token) {
    if (tokens <= 0) { throw std::invalid_argument("q4 A8 workspace: T must be positive"); }
    const auto count = static_cast<std::size_t>(tokens);
    if (count > std::numeric_limits<std::size_t>::max() / bytes_per_token) {
        throw std::overflow_error("q4 A8 workspace size overflow");
    }
    return count * bytes_per_token;
}

template <class Arena>
Q4A8Workspace allocate_q4_a8_workspace(Arena& arena, std::int32_t tokens, std::int32_t input_rows) {
    if (input_rows <= 0 || (input_rows % 64) != 0) {
        throw std::invalid_argument("q4 A8 workspace: invalid K");
    }
    const DeviceSpan codes =
        arena.alloc_bytes(q4_a8_checked_bytes(tokens, static_cast<std::size_t>(input_rows)), 256);
    const DeviceSpan scales = arena.alloc_bytes(q4_a8_checked_bytes(tokens, sizeof(float)), 256);
    return {static_cast<std::int8_t*>(codes.data), static_cast<float*>(scales.data)};
}

inline std::size_t q4_a8_workspace_capacity_bytes(std::int32_t tokens, std::int32_t input_rows) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_q4_a8_workspace(layout, tokens, input_rows);
    return layout.peak_bytes(1);
}

void launch_q4_a8_quantize(const Tensor& x, Q4A8Workspace workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
