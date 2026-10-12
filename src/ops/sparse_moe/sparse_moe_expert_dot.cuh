#pragma once

// The stored-plane codecs and the warp dot products the sparse-MoE decode routes read their expert
// banks with. They belong to no one model: any route that streams expert rows of a registered
// codec at a compile-time column count can take them.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"
#include "ops/linear/q5/q5_rowsplit_storage.cuh"
#include "ops/linear/q6/q6_rowsplit_storage.cuh"
#include "ops/linear/q8/q8_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// The stored planes of one matrix. They travel together because a dot product takes one matrix at
// a time and each codec reads a different subset of them: a call site handing them over one by one
// had to know that subset before it could name the codec.
struct SparseMoePlanes {
    const std::uint8_t* __restrict__ codes  = nullptr;
    const std::uint8_t* __restrict__ high   = nullptr;
    const std::uint8_t* __restrict__ scales = nullptr;
    // One divisor per source matrix the plane was assembled from, each covering `divisor_rows`
    // consecutive rows. Only NVFP4 has them.
    const float* __restrict__ divisors = nullptr;
    int divisor_rows                   = 1;
};

// Codecs come in two lane ownerships. Under `kPackedWord8` a lane owns eight consecutive K values
// and a single decode feeds eight FP32 FMAs, so four adjacent groups of eight lanes each form one
// 128-byte warp transaction. The other ownership is scalar - one value per lane in D3, a pair in
// D4 - which is what a 32-wide group leaves room for. Either way the codec takes the matrix's
// column count as a template argument, because it walks its own row and only it knows how its
// planes are laid out.

struct Q4Codec {
    static constexpr int kGroupK              = 64;
    static constexpr bool kPackedWord8        = true;
    static constexpr bool kSingleValuePerLane = false;

    __device__ static __forceinline__ float row_coefficient(const SparseMoePlanes&, int) {
        return 1.0F;
    }

    // The raw words of one eight-value fetch, kept apart from their decode so a caller can issue
    // several fetches before decoding any of them.
    struct Raw {
        std::uint32_t packed;
        std::uint16_t scale_bits;
    };

    template <int K>
    __device__ static __forceinline__ Raw fetch(const SparseMoePlanes& planes, int row, int group,
                                                int lane_in_group) {
        const std::int64_t index = static_cast<std::int64_t>(row) * (K / kGroupK) + group;
        return {
            *reinterpret_cast<const std::uint32_t*>(
                planes.codes + index * Q4RowSplitStorage::kCodeBytesPerGroup + lane_in_group * 4),
            *reinterpret_cast<const std::uint16_t*>(
                planes.scales + index * Q4RowSplitStorage::kScaleBytesPerGroup)};
    }

    __device__ static __forceinline__ void decode(Raw raw, float (&weights)[8]) {
        Q4SimtDecodeAtom::decode_eight(raw.packed, raw.scale_bits, weights);
    }

    template <int K>
    __device__ static __forceinline__ void load_eight(const SparseMoePlanes& planes, int row,
                                                      int group, int lane_in_group, float,
                                                      float (&weights)[8]) {
        decode(fetch<K>(planes, row, group, lane_in_group), weights);
    }
};

struct Q5Codec {
    static constexpr int kGroupK              = 64;
    static constexpr bool kPackedWord8        = true;
    static constexpr bool kSingleValuePerLane = false;

    __device__ static __forceinline__ float row_coefficient(const SparseMoePlanes&, int) {
        return 1.0F;
    }

    template <int K>
    __device__ static __forceinline__ void load_eight(const SparseMoePlanes& planes, int row,
                                                      int group, int lane_in_group, float,
                                                      float (&weights)[8]) {
        const std::int64_t index   = static_cast<std::int64_t>(row) * (K / kGroupK) + group;
        const std::uint32_t packed = *reinterpret_cast<const std::uint32_t*>(
            planes.codes + index * Q5RowSplitStorage::kCodeBytesPerGroup + lane_in_group * 4);
        const std::uint8_t high_bits =
            planes.high[index * Q5RowSplitStorage::kHighBytesPerGroup + lane_in_group];
        const auto scale_bits = *reinterpret_cast<const std::uint16_t*>(
            planes.scales + index * Q5RowSplitStorage::kScaleBytesPerGroup);
        Q5SimtDecodeAtom::decode_eight(packed, high_bits, scale_bits, weights);
    }
};

struct Q6Codec {
    static constexpr int kGroupK              = 64;
    static constexpr bool kPackedWord8        = true;
    static constexpr bool kSingleValuePerLane = false;

    __device__ static __forceinline__ float row_coefficient(const SparseMoePlanes&, int) {
        return 1.0F;
    }

    template <int K>
    __device__ static __forceinline__ void load_eight(const SparseMoePlanes& planes, int row,
                                                      int group, int lane_in_group, float,
                                                      float (&weights)[8]) {
        const std::int64_t index   = static_cast<std::int64_t>(row) * (K / kGroupK) + group;
        const std::uint32_t packed = *reinterpret_cast<const std::uint32_t*>(
            planes.codes + index * Q6RowSplitStorage::kCodeBytesPerGroup + lane_in_group * 4);
        const std::uint16_t high_bits = *reinterpret_cast<const std::uint16_t*>(
            planes.high + index * Q6RowSplitStorage::kHighBytesPerGroup + lane_in_group * 2);
        const auto scale_bits = *reinterpret_cast<const std::uint16_t*>(
            planes.scales + index * Q6RowSplitStorage::kScaleBytesPerGroup);
        Q6SimtDecodeAtom::decode_eight(packed, high_bits, scale_bits, weights);
    }
};

struct Q8Codec {
    static constexpr int kGroupK              = 32;
    static constexpr bool kPackedWord8        = false;
    static constexpr bool kSingleValuePerLane = true;

    __device__ static __forceinline__ float row_coefficient(const SparseMoePlanes&, int) {
        return 1.0F;
    }

    template <int K>
    __device__ static __forceinline__ float load_one(const SparseMoePlanes& planes, int row,
                                                     int group, int lane) {
        const std::int64_t index = static_cast<std::int64_t>(row) * (K / kGroupK) + group;
        const float scale = __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(
            planes.scales + index * Q8RowSplitStorage::kScaleBytesPerGroup)));
        return static_cast<float>(static_cast<std::int8_t>(
                   planes.codes[index * Q8RowSplitStorage::kCodeBytesPerGroup + lane])) *
               scale;
    }

    template <int K>
    __device__ static __forceinline__ void load_pair(const SparseMoePlanes& planes, int row,
                                                     int group, int lane, float& w0, float& w1) {
        Q8ScalarDecodeAtom::load_pair(planes.codes, planes.high, planes.scales,
                                      static_cast<std::int64_t>(row) * (K / kGroupK) + group, lane,
                                      w0, w1);
    }
};

// q4_g32_fp16_rows in row_group_v1: a row is K/2 code bytes, element 2i in the low nibble of byte
// i, each nibble the signed code plus 8; the scale plane holds one binary16 word per 32 elements
// of a row. The walk takes 64 columns a group like the other codecs, so a lane's eight values lie
// inside one 32-wide scale group, (2 * group + lane_in_group / 4). Forming 1024 + n in binary16
// and subtracting 1032 lands exactly on the code n - 8.
struct Q4RowsCodec {
    static constexpr int kGroupK              = 64;
    static constexpr bool kPackedWord8        = true;
    static constexpr bool kSingleValuePerLane = false;

    __device__ static __forceinline__ float row_coefficient(const SparseMoePlanes&, int) {
        return 1.0F;
    }

    template <int K>
    __device__ static __forceinline__ void load_eight(const SparseMoePlanes& planes, int row,
                                                      int group, int lane_in_group, float,
                                                      float (&weights)[8]) {
        static_assert(K % 64 == 0);
        const std::uint32_t packed = *reinterpret_cast<const std::uint32_t*>(
            planes.codes + static_cast<std::int64_t>(row) * (K / 2) + group * 32 +
            lane_in_group * 4);
        const std::uint16_t scale_bits = *reinterpret_cast<const std::uint16_t*>(
            planes.scales +
            (static_cast<std::int64_t>(row) * (K / 32) + group * 2 + (lane_in_group >> 2)) * 2);
        const float scale  = __half2float(__ushort_as_half(scale_bits));
        const __half2 bias = __half2half2(__ushort_as_half(0x6408)); // 1032.0
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const std::uint32_t bits = ((packed >> (4 * pair)) & 0x000f000fu) | 0x64006400u;
            const __half2 decoded    = __hsub2(*reinterpret_cast<const __half2*>(&bits), bias);
            const float2 values      = __half22float2(decoded);
            weights[pair]            = values.x * scale;
            weights[pair + 4]        = values.y * scale;
        }
    }
};

// NVFP4 packs two e2m1 codes per byte and one e4m3 scale per sixteen values, so a lane's eight
// values are four code bytes and exactly one scale: whichever half of a block the lane owns, its
// eight values lie inside that one block. Only the scale plane is swizzled; the codes are plain
// row-major. The per-tensor divisor is folded into the decoded block scale, once per eight values
// rather than once per value.
struct Nvfp4Codec {
    static constexpr int kGroupK              = 64;
    static constexpr bool kPackedWord8        = true;
    static constexpr bool kSingleValuePerLane = false;

    // Which stored divisor this row was quantised against. Taken once per row, not per group.
    __device__ static __forceinline__ float row_coefficient(const SparseMoePlanes& planes,
                                                            int row) {
        return __frcp_rn(planes.divisors[row / planes.divisor_rows]);
    }

    template <int K>
    __device__ static __forceinline__ void load_eight(const SparseMoePlanes& planes, int row,
                                                      int group, int lane_in_group, float row_scale,
                                                      float (&weights)[8]) {
        const std::uint32_t packed = *reinterpret_cast<const std::uint32_t*>(
            planes.codes + static_cast<std::int64_t>(row) * (K / 2) + group * 32 +
            lane_in_group * 4);
        const std::uint8_t scale =
            planes.scales[nvfp4_scale_byte<K / 64>(row, group * 4 + (lane_in_group >> 1))];
        const float coefficient = decode_nvfp4_e4m3(scale) * row_scale;
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const float2 code =
                decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(packed >> (8 * pair)));
            weights[pair * 2]     = code.x * coefficient;
            weights[pair * 2 + 1] = code.y * coefficient;
        }
    }
};

// A codec declares exactly one lane ownership. The trap below catches a codec that declares
// neither, which would otherwise compile and reduce a zero accumulator.
template <class>
inline constexpr bool kNoLaneOwnership = false;

// A codec that can separate its raw fetch from its decode: `Raw`, `fetch` and `decode`.
template <class Codec>
inline constexpr bool kSplitFetch = requires { typename Codec::Raw; };

// One lane's eight weights of each row against eight consecutive activations.
__device__ __forceinline__ void accumulate_eight(const __nv_bfloat16* x_lane,
                                                 const float (&weights0)[8],
                                                 const float (&weights1)[8], float& acc0,
                                                 float& acc1) {
    const uint4 input     = load_vec<uint4>(x_lane);
    const float2 x0       = bf16x2_bits_to_float2(input.x);
    const float2 x1       = bf16x2_bits_to_float2(input.y);
    const float2 x2       = bf16x2_bits_to_float2(input.z);
    const float2 x3       = bf16x2_bits_to_float2(input.w);
    const float values[8] = {x0.x, x0.y, x1.x, x1.y, x2.x, x2.y, x3.x, x3.y};
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        acc0 = fmaf(weights0[item], values[item], acc0);
        acc1 = fmaf(weights1[item], values[item], acc1);
    }
}

// QuadsInFlight is the number of four-group quads whose raw words are fetched before any is
// decoded, for codecs that can split fetch from decode. One quad in flight leaves the load unit
// idle for the whole decode; two overlap one quad's decode and FMA chain with the next quad's
// memory latency, which pays where the path is latency-bound rather than bandwidth-bound. Groups
// are still visited in ascending order at any depth, so the accumulation order, and therefore the
// output, is bit-identical. Codecs without a split fetch ignore it.
template <class Codec, int K, int QuadsInFlight = 2>
__device__ __forceinline__ void dot_two_rows(const SparseMoePlanes& planes, int row0, int row1,
                                             const __nv_bfloat16* x, int k_begin, int k_end,
                                             float& result0, float& result1) {
    const int lane        = static_cast<int>(threadIdx.x) & 31;
    float acc0            = 0.0f;
    float acc1            = 0.0f;
    const int first_group = k_begin / Codec::kGroupK;
    const int last_group  = k_end / Codec::kGroupK;
    if constexpr (Codec::kPackedWord8) {
        const int lane_group    = lane >> 3;
        const int lane_in_group = lane & 7;
        const float scale0      = Codec::row_coefficient(planes, row0);
        const float scale1      = Codec::row_coefficient(planes, row1);
        if constexpr (kSplitFetch<Codec> && QuadsInFlight > 1) {
            // The pairing needs the walked span, (k_end - k_begin) / kGroupK, to be a multiple of
            // 4 * QuadsInFlight. That is a property of the arguments; every caller passes
            // k_begin = 0 and k_end = K, which reduces it to this compile-time condition.
            static_assert((K / Codec::kGroupK) % (4 * QuadsInFlight) == 0);
            for (int group_base = first_group; group_base < last_group;
                 group_base += 4 * QuadsInFlight) {
                typename Codec::Raw raw0[QuadsInFlight];
                typename Codec::Raw raw1[QuadsInFlight];
#pragma unroll
                for (int quad = 0; quad < QuadsInFlight; ++quad) {
                    const int group = group_base + quad * 4 + lane_group;
                    raw0[quad]      = Codec::template fetch<K>(planes, row0, group, lane_in_group);
                    raw1[quad]      = Codec::template fetch<K>(planes, row1, group, lane_in_group);
                }
#pragma unroll
                for (int quad = 0; quad < QuadsInFlight; ++quad) {
                    const int group = group_base + quad * 4 + lane_group;
                    float weights0[8];
                    float weights1[8];
                    Codec::decode(raw0[quad], weights0);
                    Codec::decode(raw1[quad], weights1);
                    accumulate_eight(x + group * Codec::kGroupK + lane_in_group * 8, weights0,
                                     weights1, acc0, acc1);
                }
            }
        } else {
            for (int group_base = first_group; group_base < last_group; group_base += 4) {
                const int group = group_base + lane_group;
                float weights0[8];
                float weights1[8];
                Codec::template load_eight<K>(planes, row0, group, lane_in_group, scale0, weights0);
                Codec::template load_eight<K>(planes, row1, group, lane_in_group, scale1, weights1);
                accumulate_eight(x + group * Codec::kGroupK + lane_in_group * 8, weights0, weights1,
                                 acc0, acc1);
            }
        }
    } else if constexpr (Codec::kSingleValuePerLane) {
        for (int group = first_group; group < last_group; ++group) {
            const float w0 = Codec::template load_one<K>(planes, row0, group, lane);
            const float w1 = Codec::template load_one<K>(planes, row1, group, lane);
            const float xv = __bfloat162float(x[group * Codec::kGroupK + lane]);
            acc0           = fmaf(w0, xv, acc0);
            acc1           = fmaf(w1, xv, acc1);
        }
    } else {
        static_assert(kNoLaneOwnership<Codec>, "dot_two_rows: codec declares no lane ownership");
    }
    result0 = warp_reduce_sum(acc0);
    result1 = warp_reduce_sum(acc1);
}

template <class Codec, int K, int Rows>
__device__ __forceinline__ void dot_fp32_rows(const SparseMoePlanes& planes, int row_base,
                                              const float* x, int first_group, int last_group,
                                              float (&result)[Rows]) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    float acc[Rows];
#pragma unroll
    for (int row = 0; row < Rows; ++row) { acc[row] = 0.0f; }
    if constexpr (Codec::kPackedWord8) {
        // The same eight-value lane ownership as D3, over the FP32 SwiGLU result rather than the
        // BF16 hidden state. Each row decodes from its own stored planes before the FP32
        // accumulation.
        const int lane_group    = lane >> 3;
        const int lane_in_group = lane & 7;
        float row_scale[Rows];
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
            row_scale[row] = Codec::row_coefficient(planes, row_base + row);
        }
        for (int group_base = first_group; group_base < last_group; group_base += 4) {
            const int group = group_base + lane_group;
            const float4 x0 = load_vec<float4>(x + group * Codec::kGroupK + lane_in_group * 8);
            const float4 x1 = load_vec<float4>(x + group * Codec::kGroupK + lane_in_group * 8 + 4);
            const float values[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
            for (int row = 0; row < Rows; ++row) {
                float weights[8];
                Codec::template load_eight<K>(planes, row_base + row, group, lane_in_group,
                                              row_scale[row], weights);
#pragma unroll
                for (int item = 0; item < 8; ++item) {
                    acc[row] = fmaf(weights[item], values[item], acc[row]);
                }
            }
        }
    } else if constexpr (Codec::kSingleValuePerLane) {
        // Two values a lane here rather than one: the FP32 input is half the width of the BF16
        // hidden state, so the same group needs half the lanes.
        if (lane < Codec::kGroupK / 2) {
            for (int group = first_group; group < last_group; ++group) {
                const int k     = group * Codec::kGroupK + lane * 2;
                const float2 xv = load_vec<float2>(x + k);
#pragma unroll
                for (int row = 0; row < Rows; ++row) {
                    float w0, w1;
                    Codec::template load_pair<K>(planes, row_base + row, group, lane, w0, w1);
                    acc[row] = fmaf(w0, xv.x, acc[row]);
                    acc[row] = fmaf(w1, xv.y, acc[row]);
                }
            }
        }
    } else {
        static_assert(kNoLaneOwnership<Codec>, "dot_fp32_rows: codec declares no lane ownership");
    }
#pragma unroll
    for (int row = 0; row < Rows; ++row) { result[row] = warp_reduce_sum(acc[row]); }
}

} // namespace ninfer::ops::detail
