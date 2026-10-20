#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstdint>

namespace ninfer::ops {

/**
 * Encodes Gemma 4's compact global rows into row-scaled FP8 E4M3 planes.
 *
 * `rows` is contiguous BF16 [640, Hkv, T] viewed as [640*Hkv, T]: one 640-value compact row per
 * (KV head, token), the value vector followed by the rotated key dims exactly as `compact_kv_rows`
 * writes them. `positions` is contiguous I32 [T] and names the cache slot of each token, in
 * [0, capacity). `codes` is contiguous FP8_E4M3FN [640, Hkv, capacity] and `scales` contiguous FP16
 * [Hkv, capacity], one scale per stored row.
 *
 * For each written row, let a = max |value| over its 640 entries. The stored scale is
 *
 *   s = round_fp16(clamp(a / 448, 2^-24, 65504))
 *
 * and every code is the E4M3 encoding of `value / s` saturated to the format's finite range. An
 * all-zero row stores s = 0 and zero codes. The destination slot is the row's token position, so a
 * cache hands the Op every token of a pass and reads the rows back through
 * `causal_compact_attention_codes`. The width is fixed at 640, the compact row of an R = 512 head
 * with 64 rotated pairs; any other width is a contract error.
 *
 * Every destination entry of every written row is written; slots no position names are untouched.
 * The Op uses no workspace or persistent state and does not synchronize.
 */
void compact_kv_rows_quantize(const Tensor& rows, const Tensor& positions, Tensor& codes,
                              Tensor& scales, cudaStream_t stream);

} // namespace ninfer::ops
