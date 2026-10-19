#pragma once

// Host arithmetic for Qwen4Exp NVFP4 routed experts, read in place from a stored Host bank.
//
//   gate/up: h[t][r] = silu(inv_gate * <W_g[r], x_t>) * (inv_up * <W_u[r], x_t>),  r < 640
//   down:    y[t][r] = inv_down * <W_d[r], h_t>,                                     r < 2560
//
// Activations enter as Q8_0: per 32 consecutive values one int8 code row and one FP32 scale
// d = amax / 127 (q = round-to-nearest(x / d), zero when the block is all zero). Each 16-value
// NVFP4 group is an exact int32 sum of e2m1 x int8 products; the group's E4M3 scale and the
// activation block's scale multiply it in FP32, and the row reduces those products in FP32. The
// results therefore depend on the Q8_0 rounding of x and h, not on how rows are split across
// threads: a row is always reduced by one thread in one order.
//
// Weights are the bank's own planes for one expert: gate rows then up rows, row-major codes
// (K/2 bytes per row, the even element in the low nibble) and the K16M128x4 scale plane (per row,
// four consecutive groups share one 32-bit word inside the row's 512-byte tile).

#include <cstdint>

namespace ninfer::ops::cpu {

inline constexpr int kNvfp4Hidden       = 2560;
inline constexpr int kNvfp4Intermediate = 640;

/** One expert of an NVFP4 bank: its four planes and its three weight divisors as reciprocals. */
struct Nvfp4Expert {
    const std::uint8_t* gate_up_codes  = nullptr; // [1280, 2560/2]
    const std::uint8_t* gate_up_scales = nullptr; // K16M128x4, 1280 rows x 160 groups
    const std::uint8_t* down_codes     = nullptr; // [2560, 640/2]
    const std::uint8_t* down_scales    = nullptr; // K16M128x4, 2560 rows x 40 groups
    float inv_gate                     = 0.0F;
    float inv_up                       = 0.0F;
    float inv_down                     = 0.0F;
};

/** A Q8_0 activation row of `n` values: `n` int8 codes and `n / 32` FP32 scales. */
struct Q8Row {
    const std::int8_t* codes = nullptr;
    const float* scales      = nullptr;
};

/** Quantize `n` FP32 values (a multiple of 32) to Q8_0. */
void quantize_q8_0(const float* x, int n, std::int8_t* codes, float* scales);

/** Rows [r0, r1) of the expert's SwiGLU output for `tokens` (1..8) activation rows of 2560. */
void nvfp4_gate_up_rows(const Nvfp4Expert& expert, int r0, int r1, const Q8Row* x, int tokens,
                        float* const* h);

/** Rows [r0, r1) of the expert's down output for `tokens` (1..8) Q8_0 rows of 640. */
void nvfp4_down_rows(const Nvfp4Expert& expert, int r0, int r1, const Q8Row* h, int tokens,
                     float* const* out);

/** True when this CPU runs the kernels (AVX2, FMA, F16C). */
[[nodiscard]] bool nvfp4_host_supported() noexcept;

/** Which instruction set the kernels dispatch to: "avx-vnni" or "avx2". */
[[nodiscard]] const char* nvfp4_host_isa() noexcept;

} // namespace ninfer::ops::cpu
