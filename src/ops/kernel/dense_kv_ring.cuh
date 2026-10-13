#pragma once

// Implements: include/ninfer/ops/dense_kv_ring.h
//
// One warp owns one (head, row) pair: lane l holds dimensions l + 32 r of the 256-wide row, the
// distribution every D256 KV codec is written for. Encoding rotates the row with the codec's own
// fixed normalized Hadamard (for the side the format rotates), forms the format's represented scale
// and codes with the codec's own arithmetic, and writes them into the ring's planes at the row's
// slot. Decoding forms the represented value of each code and applies the same rotation again --
// the normalized Hadamard is an involution, so what lands in the destination is the value in the
// basis the encoder read, and the reader of a decoded row needs no rotated query. NVFP4's codes
// ride two to a byte and its groups are sixteen consecutive dimensions, so the two NVFP4 cases
// stage one row through the owning warp's shared scratch, exactly as the paged append does.

#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/kv_cache/fp8_e4m3_row_codec.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"
#include "ops/kv_cache/nvfp4_group16_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr int kDenseKvHeadDim = 256;
inline constexpr int kDenseKvWarps   = 8;
inline constexpr int kDenseKvThreads = kDenseKvWarps * 32;

// One ring side's codec. `kCodeBytes` is the plane's leading extent (one byte per code element, two
// NVFP4 codes a byte) and `kScaleBytes` the bytes one group's scale occupies.
enum class DenseKvFormat : int {
    Fp8Row       = 0,
    Int8Group64  = 1,
    Nvfp4Group16 = 2,
};

template <DenseKvFormat Format>
struct DenseKvFormatTraits;

template <>
struct DenseKvFormatTraits<DenseKvFormat::Fp8Row> {
    static constexpr int kCodeBytes  = kKVCacheFp8HeadDim;
    static constexpr int kScaleBytes = kKVCacheFp8Groups * static_cast<int>(sizeof(__half));
    static constexpr bool kFp16Scale = true;
};

template <>
struct DenseKvFormatTraits<DenseKvFormat::Int8Group64> {
    static constexpr int kCodeBytes  = kKVCacheInt8HeadDim;
    static constexpr int kScaleBytes = kKVCacheInt8Groups * static_cast<int>(sizeof(__half));
    static constexpr bool kFp16Scale = true;
};

template <>
struct DenseKvFormatTraits<DenseKvFormat::Nvfp4Group16> {
    static constexpr int kCodeBytes  = kKVCacheNvfp4CodeBytes;
    static constexpr int kScaleBytes = kKVCacheNvfp4Groups;
    static constexpr bool kFp16Scale = false;
};

__device__ __forceinline__ float dense_kv_fp8_to_float(std::uint8_t code) {
    __nv_fp8_e4m3 value;
    value.__x = code;
    return __half2float(static_cast<__half>(value));
}

// Encodes one row held as values[r] = dimension lane + 32 r into the plane row `codes`/`scales`,
// which are the byte addresses of the row's first code and first scale.
template <DenseKvFormat Format, bool Rotate>
__device__ __forceinline__ void
dense_kv_encode_row(const float (&source)[8], int lane, std::uint8_t* __restrict__ codes,
                    std::uint8_t* __restrict__ scales, float* scratch) {
    float values[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) values[r] = source[r];
    if (Rotate) normalized_hadamard_d256_inplace(values, lane);

    if constexpr (Format == DenseKvFormat::Fp8Row) {
        float local_absmax = 0.0F;
#pragma unroll
        for (float value : values) local_absmax = fmaxf(local_absmax, fabsf(value));
        const auto quant = kv_cache_fp8_quant_params(warp_max(local_absmax, kFullWarpMask));
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            codes[lane + 32 * r] = kv_cache_fp8_quant_code(values[r], quant.inverse_scale);
        }
        if (lane == 0) *reinterpret_cast<__half*>(scales) = quant.scale;
    } else if constexpr (Format == DenseKvFormat::Int8Group64) {
#pragma unroll
        for (int group = 0; group < kKVCacheInt8Groups; ++group) {
            const float low    = values[2 * group];
            const float high   = values[2 * group + 1];
            const float absmax = warp_max(fmaxf(fabsf(low), fabsf(high)), kFullWarpMask);
            const auto quant   = kv_cache_int8_quant_params(absmax);
            codes[group * kKVCacheInt8Group + lane] =
                static_cast<std::uint8_t>(kv_cache_int8_quant_code(low, quant.inverse_scale));
            codes[group * kKVCacheInt8Group + lane + 32] =
                static_cast<std::uint8_t>(kv_cache_int8_quant_code(high, quant.inverse_scale));
            if (lane == 0) reinterpret_cast<__half*>(scales)[group] = quant.scale;
        }
    } else {
        static_assert(Format == DenseKvFormat::Nvfp4Group16, "unknown dense KV format");
#pragma unroll
        for (int r = 0; r < 8; ++r) scratch[lane + 32 * r] = values[r];
        __syncwarp(kFullWarpMask);
        if (lane < kKVCacheNvfp4Groups) {
            const auto quantized =
                kv_cache_nvfp4_quantize_group16(scratch + lane * kKVCacheNvfp4Group);
            store_vec(codes + lane * (kKVCacheNvfp4Group / 2),
                      make_uint2(quantized.codes_lo, quantized.codes_hi));
            scales[lane] = quantized.scale;
        }
        __syncwarp(kFullWarpMask);
    }
}

// Decodes the plane row at `codes`/`scales` into values[r] = dimension lane + 32 r.
template <DenseKvFormat Format, bool Rotate>
__device__ __forceinline__ void dense_kv_decode_row(const std::uint8_t* __restrict__ codes,
                                                    const std::uint8_t* __restrict__ scales,
                                                    int lane, float* scratch, float (&values)[8]) {
    if constexpr (Format == DenseKvFormat::Fp8Row) {
        const float scale = __half2float(*reinterpret_cast<const __half*>(scales));
#pragma unroll
        for (int r = 0; r < 8; ++r) values[r] = dense_kv_fp8_to_float(codes[lane + 32 * r]) * scale;
    } else if constexpr (Format == DenseKvFormat::Int8Group64) {
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int group   = r / 2;
            const float scale = __half2float(reinterpret_cast<const __half*>(scales)[group]);
            values[r] = static_cast<float>(static_cast<std::int8_t>(codes[lane + 32 * r])) * scale;
        }
    } else {
        static_assert(Format == DenseKvFormat::Nvfp4Group16, "unknown dense KV format");
        if (lane < kKVCacheNvfp4Groups) {
            const auto decoded = kv_cache_nvfp4_dequant_f16x16(
                codes + lane * (kKVCacheNvfp4Group / 2), scales[lane]);
            const __half* halves = reinterpret_cast<const __half*>(&decoded);
#pragma unroll
            for (int i = 0; i < kKVCacheNvfp4Group; ++i) {
                scratch[lane * kKVCacheNvfp4Group + i] = __half2float(halves[i]);
            }
        }
        __syncwarp(kFullWarpMask);
#pragma unroll
        for (int r = 0; r < 8; ++r) values[r] = scratch[lane + 32 * r];
        __syncwarp(kFullWarpMask);
    }
    if (Rotate) normalized_hadamard_d256_inplace(values, lane);
}

// One warp per (head, token) row: the token's slot comes from `slots`, and the row's code and scale
// entries are written there.
template <DenseKvFormat KeyFormat, bool KeyRotate, DenseKvFormat ValueFormat, bool ValueRotate>
__launch_bounds__(kDenseKvThreads) __global__
    void dense_kv_ring_encode_kernel(const __nv_bfloat16* __restrict__ keys,
                                     const __nv_bfloat16* __restrict__ values,
                                     const std::int32_t* __restrict__ slots, std::int32_t kv_heads,
                                     std::int32_t units, std::uint8_t* __restrict__ key_codes,
                                     std::uint8_t* __restrict__ key_scales,
                                     std::uint8_t* __restrict__ value_codes,
                                     std::uint8_t* __restrict__ value_scales) {
    constexpr int kKeyCodeBytes    = DenseKvFormatTraits<KeyFormat>::kCodeBytes;
    constexpr int kKeyScaleBytes   = DenseKvFormatTraits<KeyFormat>::kScaleBytes;
    constexpr int kValueCodeBytes  = DenseKvFormatTraits<ValueFormat>::kCodeBytes;
    constexpr int kValueScaleBytes = DenseKvFormatTraits<ValueFormat>::kScaleBytes;
    __shared__ float scratch[kDenseKvWarps][kDenseKvHeadDim];

    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int unit = static_cast<int>(blockIdx.x) * kDenseKvWarps + warp;
    if (unit >= units) return;

    const int head  = unit % kv_heads;
    const int token = unit / kv_heads;
    const int slot  = slots[token];
    const std::int64_t source =
        static_cast<std::int64_t>(head + kv_heads * token) * kDenseKvHeadDim;
    const std::int64_t destination = static_cast<std::int64_t>(head + kv_heads * slot);

    float row[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) { row[r] = __bfloat162float(keys[source + lane + 32 * r]); }
    dense_kv_encode_row<KeyFormat, KeyRotate>(row, lane, key_codes + destination * kKeyCodeBytes,
                                              key_scales + destination * kKeyScaleBytes,
                                              scratch[warp]);
#pragma unroll
    for (int r = 0; r < 8; ++r) { row[r] = __bfloat162float(values[source + lane + 32 * r]); }
    dense_kv_encode_row<ValueFormat, ValueRotate>(
        row, lane, value_codes + destination * kValueCodeBytes,
        value_scales + destination * kValueScaleBytes, scratch[warp]);
}

// One warp per (head, slot): every destination entry is the represented value of its code and
// scale.
template <DenseKvFormat KeyFormat, bool KeyRotate, DenseKvFormat ValueFormat, bool ValueRotate>
__launch_bounds__(kDenseKvThreads) __global__
    void dense_kv_ring_decode_kernel(const std::uint8_t* __restrict__ key_codes,
                                     const std::uint8_t* __restrict__ key_scales,
                                     const std::uint8_t* __restrict__ value_codes,
                                     const std::uint8_t* __restrict__ value_scales,
                                     std::int32_t kv_heads, std::int32_t units,
                                     __nv_bfloat16* __restrict__ keys,
                                     __nv_bfloat16* __restrict__ values) {
    constexpr int kKeyCodeBytes    = DenseKvFormatTraits<KeyFormat>::kCodeBytes;
    constexpr int kKeyScaleBytes   = DenseKvFormatTraits<KeyFormat>::kScaleBytes;
    constexpr int kValueCodeBytes  = DenseKvFormatTraits<ValueFormat>::kCodeBytes;
    constexpr int kValueScaleBytes = DenseKvFormatTraits<ValueFormat>::kScaleBytes;
    __shared__ float scratch[kDenseKvWarps][kDenseKvHeadDim];

    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int unit = static_cast<int>(blockIdx.x) * kDenseKvWarps + warp;
    if (unit >= units) return;

    const int head                 = unit % kv_heads;
    const int slot                 = unit / kv_heads;
    const std::int64_t source      = static_cast<std::int64_t>(head + kv_heads * slot);
    const std::int64_t destination = source * kDenseKvHeadDim;

    float row[8];
    dense_kv_decode_row<KeyFormat, KeyRotate>(key_codes + source * kKeyCodeBytes,
                                              key_scales + source * kKeyScaleBytes, lane,
                                              scratch[warp], row);
#pragma unroll
    for (int r = 0; r < 8; ++r) { keys[destination + lane + 32 * r] = __float2bfloat16_rn(row[r]); }
    dense_kv_decode_row<ValueFormat, ValueRotate>(value_codes + source * kValueCodeBytes,
                                                  value_scales + source * kValueScaleBytes, lane,
                                                  scratch[warp], row);
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        values[destination + lane + 32 * r] = __float2bfloat16_rn(row[r]);
    }
}

} // namespace ninfer::ops::detail
