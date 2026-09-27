#pragma once

// LDLQ for EXL3: quantizes W one 16-row strip at a time, from the last strip to the first, feeding
// each strip the Hessian-weighted error of the strips already quantized:
//     target[b] = W[b] + L[later, b]ᵀ (W - Wq)[later]
// where L is the unit block-lower factor of H = L D Lᵀ (block_ldl.h). Each strip's 16x16 tiles are
// encoded with the tail-biting trellis encoder (trellis_encoder.h).

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::quantize::exl3 {

// Tile element order (trellis_t16_v1): state t of a 16x16 tile covers
//     k = 2 q + (r & 1) + 8 (r >> 1),  n = g + 8 h,
// with t = 8 lane + 4 h + r and lane = 4 g + q (the mma.m16n8k16 B-operand fragment order).
__host__ __device__ constexpr int tile_k(int t) {
    return 2 * ((t >> 3) & 3) + (t & 1) + 8 * ((t >> 1) & 1);
}

__host__ __device__ constexpr int tile_n(int t) { return (t >> 5) + 8 * ((t >> 2) & 1); }

[[nodiscard]] std::size_t ldlq_scratch_bytes(std::int64_t k, std::int64_t n, int bitrate_half_bits);

// w: [k][n] FP32 (rows are input channels), l: [k][k] unit block-lower. Writes states in
// trellis_t16_v1 tile order ([n/16][k/16][256], output-major) and the decoded matrix wq [k][n].
// k % 16 == 0 and n % 128 == 0.
void ldlq_quantize(const float* w, const float* l, std::int64_t k, std::int64_t n,
                   int bitrate_half_bits, float scale, std::uint16_t* states, float* wq,
                   void* scratch, std::size_t scratch_bytes, cudaStream_t stream);

} // namespace ninfer::quantize::exl3
