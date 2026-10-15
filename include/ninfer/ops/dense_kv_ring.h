#pragma once

#include "core/tensor.h"
#include "ninfer/types.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * Encode BF16 key and value rows into a slot-indexed dense D256 KV ring's quantized planes.
 *
 * Logical shapes, all contiguous, with Hkv the head count, T the rows written and S the ring's
 * slot count:
 *
 *   keys, values   BF16 [256, Hkv, T]      one 256-wide row per (head, token)
 *   slots          I32  [T]                the destination slot of each token's row, in [0,S)
 *   key_codes      the storage's code dtype  [code_extent,  Hkv, S]
 *   key_scales     the storage's scale dtype [scale_extent, Hkv, S]
 *   value_codes, value_scales              the same for the value side
 *
 * The plane schema is exactly `paged_kv_storage_layout(storage, 256)`: one code element per
 * dimension (INT8, FP8, or two E2M1 codes a byte for NVFP4) and the format's scale group
 * (INT8-G64 4 x FP16, FP8-row 1 x FP16, NVFP4-G16 16 x UE4M3). `BFloat16` has no quantized planes
 * and is not a legal storage here.
 *
 * Numeric contract: each row is encoded exactly as `kv_cache_append`'s same-format row is, from the
 * same represented BF16 input values -- the format's row preparation (the fixed normalized D256
 * Hadamard, applied to K for INT8/FP8 and to both K and V for NVFP4/K8V4), then the format's own
 * represented FP16 or UE4M3 scale and its saturation arithmetic. The stored bytes are therefore
 * the paged append's bytes for the same input row, and a row's codes and scale are overwritten
 * wherever they are addressed.
 *
 * Effects: for every token index i < T and every head h, the code and scale entries of
 * `keys[:, h, i]` and `values[:, h, i]` are written at slot `slots[i]`; no other slot's entry is
 * read or written, and the inputs are unchanged. The caller guarantees every `slots[i]` is in
 * [0,S), that the four planes never alias the inputs or each other, and that a row's plane
 * addresses are 2-byte aligned (the FP16 scale planes), and that the `slots[i]` of one call are
 * distinct: two tokens addressing one slot would have two warps write the same row in an
 * indeterminate order. The Op owns no workspace and no persistent state.
 */
void dense_kv_ring_encode(const Tensor& keys, const Tensor& values, const Tensor& slots,
                          KvCacheStorage storage, Tensor& key_codes, Tensor& key_scales,
                          Tensor& value_codes, Tensor& value_scales, cudaStream_t stream);

/**
 * Decode a dense D256 KV ring's quantized planes back to BF16 key and value rows.
 *
 * Logical shapes, all contiguous, with Hkv the head count and S the ring's slot count:
 *
 *   key_codes      the storage's code dtype  [code_extent,  Hkv, S]
 *   key_scales     the storage's scale dtype [scale_extent, Hkv, S]
 *   value_codes, value_scales              the same for the value side
 *   keys, values   BF16 [256, Hkv, S]      one decoded 256-wide row per (head, slot)
 *
 * Numeric contract: every destination entry is `BF16_RNE` of the represented value the storage's
 * codec defines for its code and scale -- `FP32(E4M3(code)) * FP32(FP16(scale))` for FP8,
 * `FP32(I8(code)) * FP32(FP16(scale))` per 64-dimension group for INT8, and
 * `FP32(E2M1(code)) * FP32(UE4M3(scale))` per 16-dimension group for NVFP4 -- with the format's
 * row rotation undone first, because the rotation is an involution: a decoded row is expressed in
 * the basis the encoder read, so a reader pairs it with an unrotated query and no query-side or
 * output-side transform. The rotation is applied in FP32 and its own rounding is far below one
 * BF16 ulp of the represented value.
 *
 * Effects: writes every element of `keys` and `values`; the planes are read-only and no input is
 * modified. The caller guarantees the planes never alias the destinations. The Op owns no
 * workspace and no persistent state.
 */
void dense_kv_ring_decode(const Tensor& key_codes, const Tensor& key_scales,
                          const Tensor& value_codes, const Tensor& value_scales,
                          KvCacheStorage storage, Tensor& keys, Tensor& values,
                          cudaStream_t stream);

} // namespace ninfer::ops
