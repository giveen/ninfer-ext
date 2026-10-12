#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/layout.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {

// Two-term E4M3 A8 pair route. Both terms' MMAs run against one weight load on this 128x128 tile,
// which keeps two accumulator sets inside today's register profile, so the split-K plan holds one
// partial plane per term. The wave count is the tuned resident grid of one CTA per SM.
inline constexpr int kFp8A8PairTerms = 2;
// Below this token count the pair tile's waves do not pay for themselves; short sequences keep the
// sequential two-pass route on their tuned small-T schedules.
inline constexpr int kFp8A8PairMinTokens   = 256;
inline constexpr int kFp8A8PairWaveCtas    = 170;
inline constexpr int kFp8A8PairBlockTokens = 128;
inline constexpr int kFp8A8PairBlockRows   = 128;
// Both terms' split-K partial planes together; the caller owns this much of the workspace.
inline constexpr std::size_t kFp8A8PairPartialBytes = std::size_t(kFp8A8PairTerms) *
                                                      kFp8A8PairWaveCtas * kFp8A8PairBlockTokens *
                                                      kFp8A8PairBlockRows * sizeof(float);
// One token chunk's pair activation planes must stay resident in L2 between row tiles, otherwise
// every row tile re-reads both planes from DRAM; the budget leaves room for the weight stream and
// the output on the tuned device.
inline constexpr std::int64_t kFp8A8PairResidentBytes = 32 << 20;

// Token width of one pair-route chunk: whole tiles that fit the resident budget.
inline std::int32_t fp8_a8_pair_token_slice(std::int32_t input_rows) {
    const std::int64_t per_tile = 2 * std::int64_t(kFp8A8PairBlockTokens) * input_rows;
    const auto tiles =
        static_cast<std::int32_t>(std::max<std::int64_t>(1, kFp8A8PairResidentBytes / per_tile));
    return tiles * kFp8A8PairBlockTokens;
}

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
