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

// `codes`/`scales` hold the first (high) E4M3 term. The two-term activation route adds the residual
// E4M3 term in its own plane pair; both are null when the selected route uses a single term.
struct Fp8A8Workspace {
    std::uint8_t* codes    = nullptr;
    float* scales          = nullptr;
    std::uint8_t* codes_lo = nullptr;
    float* scales_lo       = nullptr;
    float* partials        = nullptr;
};

inline std::size_t fp8_a8_checked_bytes(std::int32_t tokens, std::size_t bytes_per_token) {
    if (tokens <= 0) { throw std::invalid_argument("fp8 A8 workspace: T must be positive"); }
    const auto count = static_cast<std::size_t>(tokens);
    if (count > std::numeric_limits<std::size_t>::max() / bytes_per_token) {
        throw std::overflow_error("fp8 A8 workspace size overflow");
    }
    return count * bytes_per_token;
}

template <class Arena>
Fp8A8Workspace allocate_fp8_a8_workspace(Arena& arena, std::int32_t tokens, std::int32_t input_rows,
                                         std::size_t partial_bytes = 0, int terms = 1) {
    if (input_rows <= 0 || (input_rows % 32) != 0) {
        throw std::invalid_argument("fp8 A8 workspace: invalid K");
    }
    if (terms != 1 && terms != 2) {
        throw std::invalid_argument("fp8 A8 workspace: terms must be one or two");
    }
    const DeviceSpan codes =
        arena.alloc_bytes(fp8_a8_checked_bytes(tokens, static_cast<std::size_t>(input_rows)), 256);
    const DeviceSpan scales = arena.alloc_bytes(fp8_a8_checked_bytes(tokens, sizeof(float)), 256);
    DeviceSpan codes_lo{};
    DeviceSpan scales_lo{};
    if (terms == 2) {
        codes_lo = arena.alloc_bytes(
            fp8_a8_checked_bytes(tokens, static_cast<std::size_t>(input_rows)), 256);
        scales_lo = arena.alloc_bytes(fp8_a8_checked_bytes(tokens, sizeof(float)), 256);
    }
    float* partials =
        partial_bytes ? static_cast<float*>(arena.alloc_bytes(partial_bytes, 256).data) : nullptr;
    return {static_cast<std::uint8_t*>(codes.data), static_cast<float*>(scales.data),
            static_cast<std::uint8_t*>(codes_lo.data), static_cast<float*>(scales_lo.data),
            partials};
}

inline std::size_t fp8_a8_workspace_capacity_bytes(std::int32_t tokens, std::int32_t input_rows,
                                                   std::size_t partial_bytes = 0, int terms = 1) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_fp8_a8_workspace(layout, tokens, input_rows, partial_bytes, terms);
    return layout.peak_bytes(1);
}

// Quantizes the first term, and the residual term as well when the workspace holds its planes.
void launch_fp8_a8_quantize(const Tensor& x, const Weight& weight, Fp8A8Workspace workspace,
                            cudaStream_t stream);


} // namespace ninfer::ops::detail
