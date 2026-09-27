#pragma once

// GPU tail-biting Viterbi encoder for EXL3 mul1 trellis tiles (offline quantizer, M2).
//
// Each 256-weight tile is encoded as one ring of 16-bit states with the trellis_t16_v1 convention
// (artifact/exl3_trellis.h): s_t = (s_{t-1} >> w_t) | (b_t << (16 - w_t)); the value of state s is
// mul1(s) * scale. The encoder minimizes the tile's squared error with FP32 path costs, using the
// two-pass tail-biting method: an unconstrained pass over the ring rotated by 128 fixes the overlap
// entering weight 0, and a pass constrained to that overlap closes the ring.
//
// Oracle: tests/quantize/exl3_viterbi_reference.h (FP64, same method).

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::quantize::exl3 {

inline constexpr int kTrellisTileWeights = 256;

// Device scratch for `blocks` tiles encoded concurrently at this bitrate.
[[nodiscard]] std::size_t trellis_encoder_scratch_bytes(int blocks, int bitrate_half_bits);

// tiles: [count][256] FP32 targets in trellis order. Writes the ring's states and the decoded
// values (mul1(state) * scale). The scratch must hold trellis_encoder_scratch_bytes(blocks, rate);
// blocks tiles are encoded concurrently and the launch strides over count.
void encode_trellis_tiles(const float* tiles, std::int64_t count, int bitrate_half_bits,
                          float scale, std::uint16_t* states, float* decoded, void* scratch,
                          std::size_t scratch_bytes, int blocks, cudaStream_t stream);

} // namespace ninfer::quantize::exl3
