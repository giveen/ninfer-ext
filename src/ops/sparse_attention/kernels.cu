// Implements: include/ninfer/ops/sparse_attention.h
// Qwen Sparse Attention: paged append, indexer block selection and selected-key attention.
// One CTA per query column (selection) or per (column, KV head, key split) (attention), with FP32
// accumulation and K/V read through the same codecs the dense cache uses. Attention runs each
// query group on Tensor Cores over either cache profile. Narrow calls split each selection across
// CTAs and merge the partial softmax states.
#include "ops/sparse_attention/launch.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/fp8_e4m3_row_codec.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"
#include "ops/kv_cache/nvfp4_group16_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <math_constants.h>

#include <algorithm>
#include <stdexcept>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kHeadDim      = 256;
constexpr int kIndexDim     = kQsaIndexKeyDim;
constexpr int kRecord       = kQsaIndexRecordWords;
constexpr int kMaxIndexHead = 8;

__device__ __forceinline__ std::int64_t record_offset(std::int32_t physical_page,
                                                      std::int32_t page_offset) {
    return (static_cast<std::int64_t>(physical_page) * kPagedKVPageSize + page_offset) * kRecord;
}

__device__ __forceinline__ std::int32_t lane_valid(const std::int32_t* valid, std::int32_t lane,
                                                   std::int32_t width) {
    if (valid == nullptr) { return width; }
    return max(0, min(width, valid[lane]));
}

// ---------------------------------------------------------------------------------------------
// Cache codecs

// Codec of one quantized K or V plane, as the dense cache stores it. Every quantized profile
// stores K Hadamard-rotated; V is rotated where its codec is NVFP4.
enum class Codec { Fp8Row, Int8G64, Nvfp4G16 };

struct ProfileCodecs {
    Codec k;
    Codec v;
    bool v_rotated;
};

__host__ __device__ constexpr ProfileCodecs profile_codecs(KvCacheStorage storage) {
    switch (storage) {
    case KvCacheStorage::Int8Group64: return {Codec::Int8G64, Codec::Int8G64, false};
    case KvCacheStorage::Nvfp4Group16: return {Codec::Nvfp4G16, Codec::Nvfp4G16, true};
    case KvCacheStorage::Fp8KeyNvfp4Value: return {Codec::Fp8Row, Codec::Nvfp4G16, true};
    default: return {Codec::Fp8Row, Codec::Fp8Row, false};
    }
}

// Code bytes of one 256-element row.
__host__ __device__ constexpr int code_bytes(Codec codec) {
    return codec == Codec::Nvfp4G16 ? 128 : 256;
}

// Encodes one row held by a warp as values[r] = x[lane + 32 r] into cache row `row`. NVFP4 groups
// gather through the warp's 256-float `scratch`.
template <Codec C>
__device__ __forceinline__ void encode_row(const float (&values)[8], std::int64_t row, void* codes,
                                           void* scales, float* scratch, int lane) {
    constexpr unsigned kMask = 0xffffffffu;
    if constexpr (C == Codec::Fp8Row) {
        float absmax = 0.0F;
#pragma unroll
        for (float x : values) { absmax = fmaxf(absmax, fabsf(x)); }
        const auto q = kv_cache_fp8_quant_params(warp_max(absmax, kMask));
        auto* c      = static_cast<std::uint8_t*>(codes) + row * kHeadDim;
#pragma unroll
        for (int r = 0; r < 8; ++r) { c[lane + 32 * r] = kv_cache_fp8_quant_code(values[r], q.inverse_scale); }
        if (lane == 0) { static_cast<__half*>(scales)[row] = q.scale; }
    } else if constexpr (C == Codec::Int8G64) {
        // Group g holds values[2g] (feature 64g + lane) and values[2g + 1] (feature 64g + 32 + lane).
        auto* c = static_cast<std::int8_t*>(codes) + row * kHeadDim;
#pragma unroll
        for (int group = 0; group < kKVCacheInt8Groups; ++group) {
            const float x0 = values[2 * group], x1 = values[2 * group + 1];
            const auto q   = kv_cache_int8_quant_params(warp_max(fmaxf(fabsf(x0), fabsf(x1)), kMask));
            c[group * kKVCacheInt8Group + lane]      = kv_cache_int8_quant_code(x0, q.inverse_scale);
            c[group * kKVCacheInt8Group + lane + 32] = kv_cache_int8_quant_code(x1, q.inverse_scale);
            if (lane == 0) { static_cast<__half*>(scales)[row * kKVCacheInt8Groups + group] = q.scale; }
        }
    } else {
#pragma unroll
        for (int r = 0; r < 8; ++r) { scratch[lane + 32 * r] = values[r]; }
        __syncwarp();
        if (lane < kKVCacheNvfp4Groups) {
            const auto q = kv_cache_nvfp4_quantize_group16(scratch + lane * kKVCacheNvfp4Group);
            store_vec(static_cast<std::uint8_t*>(codes) + row * kKVCacheNvfp4CodeBytes + lane * 8,
                      make_uint2(q.codes_lo, q.codes_hi));
            static_cast<std::uint8_t*>(scales)[row * kKVCacheNvfp4Groups + lane] = q.scale;
        }
        __syncwarp();
    }
}

// ---------------------------------------------------------------------------------------------
// Append

constexpr int kAppendThreads = 256;

template <KvCacheStorage Storage>
__global__ void __launch_bounds__(kAppendThreads) qsa_append_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ index_keys, const std::int32_t* __restrict__ rope_positions,
    std::int32_t rope_axes, const std::int32_t* __restrict__ cache,
    const std::int32_t* __restrict__ valid, const std::int32_t* __restrict__ table_rows,
    const std::int32_t* __restrict__ tables, std::int32_t table_stride, void* k_pages,
    void* v_pages, void* k_scales, void* v_scales, __nv_bfloat16* __restrict__ records,
    std::int32_t kv_heads, std::int32_t width, std::int32_t lanes) {
    constexpr bool kNvfp4 = Storage == KvCacheStorage::Nvfp4Group16 ||
                            Storage == KvCacheStorage::Fp8KeyNvfp4Value;
    __shared__ float scratch[kNvfp4 ? kAppendThreads / 32 : 1][kNvfp4 ? kHeadDim : 1];
    // One warp per (column, lane, kv head) plus one warp per (column, lane) for the record.
    const int warp_global = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane_id     = static_cast<int>(threadIdx.x) & 31;
    const int units       = width * lanes * (kv_heads + 1);
    if (warp_global >= units) { return; }
    const int head   = warp_global % (kv_heads + 1);
    const int column = (warp_global / (kv_heads + 1)) % width;
    const int lane   = warp_global / ((kv_heads + 1) * width);
    if (column >= lane_valid(valid, lane, width)) { return; }
    const std::int64_t token    = static_cast<std::int64_t>(lane) * width + column;
    const std::int32_t position = cache[token];
    if (position < 0) { return; }
    const std::int32_t* table = tables + static_cast<std::int64_t>(table_rows[lane]) * table_stride;
    const std::int32_t page   = paged_kv_physical_page(table, position);
    const std::int32_t offset = position & kPagedKVPageMask;

    if (head == kv_heads) {
        const std::int64_t base = record_offset(page, offset);
        for (int d = lane_id; d < kIndexDim; d += 32) {
            records[base + d] = index_keys[token * kIndexDim + d];
        }
        if (lane_id < 3) {
            const std::int64_t tokens = static_cast<std::int64_t>(width) * lanes;
            const std::int32_t p = rope_positions[(rope_axes == 3 ? lane_id : 0) * tokens + token];
            auto* words          = reinterpret_cast<std::uint16_t*>(records + base + kIndexDim);
            words[2 * lane_id]   = static_cast<std::uint16_t>(static_cast<std::uint32_t>(p));
            words[2 * lane_id + 1] =
                static_cast<std::uint16_t>(static_cast<std::uint32_t>(p) >> 16);
        }
        return;
    }

    const std::int64_t source = (token * kv_heads + head) * kHeadDim;
    const std::int64_t row =
        (static_cast<std::int64_t>(page) * kv_heads + head) * kPagedKVPageSize + offset;
    if constexpr (Storage == KvCacheStorage::BFloat16) {
        auto* kb = static_cast<__nv_bfloat16*>(k_pages);
        auto* vh = static_cast<__half*>(v_pages);
        for (int d = lane_id; d < kHeadDim; d += 32) {
            kb[row * kHeadDim + d] = k[source + d];
            vh[row * kHeadDim + d] = __float2half_rn(__bfloat162float(v[source + d]));
        }
    } else {
        constexpr ProfileCodecs codecs = profile_codecs(Storage);
        float* warp_scratch            = scratch[kNvfp4 ? static_cast<int>(threadIdx.x) >> 5 : 0];
        float values[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) { values[r] = __bfloat162float(k[source + lane_id + 32 * r]); }
        normalized_hadamard_d256_inplace(values, lane_id);
        encode_row<codecs.k>(values, row, k_pages, k_scales, warp_scratch, lane_id);
#pragma unroll
        for (int r = 0; r < 8; ++r) { values[r] = __bfloat162float(v[source + lane_id + 32 * r]); }
        if constexpr (codecs.v_rotated) { normalized_hadamard_d256_inplace(values, lane_id); }
        encode_row<codecs.v>(values, row, v_pages, v_scales, warp_scratch, lane_id);
    }
}

// ---------------------------------------------------------------------------------------------
// Selection

// Kernel-side geometry (std::array is host-only).
struct KernelGeometry {
    std::int32_t index_heads, index_dim, budget, ratio, rotary_dim;
    float rope_theta, rms_norm_eps;
    std::uint8_t pair_axes[64];

    __device__ std::int32_t block_topk() const { return budget / ratio; }
};

KernelGeometry kernel_geometry(const QsaGeometry& g) {
    KernelGeometry out{g.index_heads, g.index_dim,  g.budget,       g.ratio,
                       g.rotary_dim,  g.rope_theta, g.rms_norm_eps, {}};
    for (int i = 0; i < 64; ++i) { out.pair_axes[i] = g.pair_axes[i]; }
    return out;
}

struct SelectArgs {
    const __nv_bfloat16* index_query;
    const std::int32_t* query_positions;
    std::int32_t query_axes;
    const std::int32_t* cache;
    const std::int32_t* valid;
    const std::int32_t* table_rows;
    const std::int32_t* tables;
    std::int32_t table_stride;
    const __nv_bfloat16* records;
    const __nv_bfloat16* query_norm;
    const __nv_bfloat16* key_norm;
    float* scores; // [max_blocks] per CTA
    std::int32_t max_blocks;
    std::int32_t* selected;
    std::int32_t* counts;
    std::int32_t width;
    std::int32_t lanes;
    std::int32_t column_begin; // first global column of this pass
    std::int32_t max_selected;
    KernelGeometry geometry;
};

__device__ __forceinline__ float rope_angle(const KernelGeometry& g, int pair,
                                            const std::int32_t* p) {
    const float exponent = -2.0F * static_cast<float>(pair) / static_cast<float>(g.rotary_dim);
    return static_cast<float>(p[g.pair_axes[pair]]) * powf(g.rope_theta, exponent);
}

__device__ __forceinline__ std::uint32_t order_key(float x) {
    const std::uint32_t bits = __float_as_uint(x);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ std::int32_t read_position(const __nv_bfloat16* record, int axis) {
    const auto* words = reinterpret_cast<const std::uint16_t*>(record + kIndexDim);
    return static_cast<std::int32_t>(static_cast<std::uint32_t>(words[2 * axis]) |
                                     (static_cast<std::uint32_t>(words[2 * axis + 1]) << 16));
}

// Scoring split policy: narrow passes spread each column's blocks over about kScoreTargetCtas
// CTAs, at least kMinScoreBlocks blocks (four per warp) apiece.
constexpr std::int32_t kScoreTargetCtas = 256;
constexpr std::int32_t kMinScoreBlocks  = 32;

// Visible tokens of this CTA's query column (zero for an invalid column) and its blocks.
struct SelectColumn {
    int column;
    int lane;
    std::int32_t visible;
    std::int32_t blocks;
};

__device__ __forceinline__ SelectColumn select_column(const SelectArgs& a) {
    SelectColumn c{};
    c.column    = a.column_begin + static_cast<int>(blockIdx.x);
    c.lane      = c.column / a.width;
    const int w = c.column % a.width;
    c.visible   = w < lane_valid(a.valid, c.lane, a.width) ? a.cache[c.column] + 1 : 0;
    c.blocks    = c.visible / a.geometry.ratio;
    return c;
}

// Block scores of the columns that must choose among their blocks. CTA (x, y) scores blocks
// [y * split_blocks, (y + 1) * split_blocks) of pass column x into its scores row.
__global__ void __launch_bounds__(256) qsa_score_kernel(SelectArgs a, std::int32_t split_blocks) {
    __shared__ float query[kMaxIndexHead * kIndexDim];

    const KernelGeometry& g  = a.geometry;
    const int tid            = static_cast<int>(threadIdx.x);
    const int warp           = tid >> 5;
    const int lane_id        = tid & 31;
    const SelectColumn col   = select_column(a);
    const int column         = col.column;
    const int lane           = col.lane;
    const std::int32_t first = static_cast<int>(blockIdx.y) * split_blocks;
    const std::int32_t last  = min(col.blocks, first + split_blocks);
    if (col.blocks <= g.block_topk() || first >= last) { return; }
    const std::int32_t ratio = g.ratio;

    // Query heads: offset RMSNorm then RoPE at the query position.
    const std::int64_t tokens = static_cast<std::int64_t>(a.width) * a.lanes;
    std::int32_t qpos[3];
    for (int axis = 0; axis < 3; ++axis) {
        qpos[axis] = a.query_positions[(a.query_axes == 3 ? axis : 0) * tokens + column];
    }
    const int di = g.index_dim;
    for (int h = warp; h < g.index_heads; h += blockDim.x / 32) {
        const __nv_bfloat16* raw =
            a.index_query + (static_cast<std::int64_t>(column) * g.index_heads + h) * di;
        float sum = 0.0F;
        for (int d = lane_id; d < di; d += 32) {
            const float x = __bfloat162float(raw[d]);
            sum += x * x;
        }
        const float inv = rsqrtf(warp_sum(sum) / static_cast<float>(di) + g.rms_norm_eps);
        for (int d = lane_id; d < di; d += 32) {
            query[h * di + d] =
                __bfloat162float(raw[d]) * inv * (1.0F + __bfloat162float(a.query_norm[d]));
        }
        __syncwarp();
        if (lane_id < g.rotary_dim / 2) {
            const int i   = lane_id;
            const int j   = i + g.rotary_dim / 2;
            const float t = rope_angle(g, i, qpos);
            const float c = cosf(t), s = sinf(t);
            const float x0 = query[h * di + i], x1 = query[h * di + j];
            query[h * di + i] = x0 * c - x1 * s;
            query[h * di + j] = x1 * c + x0 * s;
        }
    }
    __syncthreads();

    // Block scores, one warp per block; each lane owns di/32 features.
    const std::int32_t* table =
        a.tables + static_cast<std::int64_t>(a.table_rows[lane]) * a.table_stride;
    float* scores     = a.scores + static_cast<std::int64_t>(blockIdx.x) * a.max_blocks;
    const float scale = rsqrtf(static_cast<float>(di));
    const int per     = di / 32;
    for (int b = first + warp; b < last; b += blockDim.x / 32) {
        float key[8];
        for (int r = 0; r < per; ++r) { key[r] = 0.0F; }
        std::int32_t kpos[3] = {0, 0, 0};
        for (int t = 0; t < ratio; ++t) {
            const std::int32_t token = b * ratio + t;
            const __nv_bfloat16* record =
                a.records +
                record_offset(paged_kv_physical_page(table, token), token & kPagedKVPageMask);
            for (int r = 0; r < per; ++r) { key[r] += __bfloat162float(record[lane_id * per + r]); }
            if (t == 0) {
                for (int axis = 0; axis < 3; ++axis) { kpos[axis] = read_position(record, axis); }
            }
        }
        float sum = 0.0F;
        for (int r = 0; r < per; ++r) {
            // Upstream rounds the pooled mean to the activation dtype before normalizing.
            key[r] = __bfloat162float(__float2bfloat16_rn(key[r] / static_cast<float>(ratio)));
            sum += key[r] * key[r];
        }
        const float inv = rsqrtf(warp_sum(sum) / static_cast<float>(di) + g.rms_norm_eps);
        for (int r = 0; r < per; ++r) {
            key[r] *= inv * (1.0F + __bfloat162float(a.key_norm[lane_id * per + r]));
        }
        // RoPE pairs (i, i + rotary/2) inside the first rotary features.
        const int half  = g.rotary_dim / 2;
        const int shift = half / per; // lanes between pair members
        for (int r = 0; r < per; ++r) {
            const int d       = lane_id * per + r;
            const float other = __shfl_xor_sync(0xffffffffu, key[r], shift);
            if (d < g.rotary_dim) {
                const int pair = d < half ? d : d - half;
                const float t  = rope_angle(g, pair, kpos);
                const float c = cosf(t), s = sinf(t);
                key[r] = d < half ? key[r] * c - other * s : key[r] * c + other * s;
            }
        }
        float score = 0.0F;
        for (int h = 0; h < g.index_heads; ++h) {
            float dot = 0.0F;
            for (int r = 0; r < per; ++r) { dot += query[h * di + lane_id * per + r] * key[r]; }
            score += fmaxf(warp_sum(dot), 0.0F);
        }
        if (lane_id == 0) { scores[b] = score * scale; }
    }
}

// One CTA (256 threads) per query column: the kept blocks from qsa_score_kernel's scores.
__global__ void __launch_bounds__(256) qsa_select_kernel(SelectArgs a) {
    __shared__ std::uint32_t histogram[256];
    __shared__ std::int32_t scan[256];
    __shared__ std::uint32_t prefix_s;
    __shared__ std::int32_t remaining_s;
    __shared__ std::int32_t emitted_s;
    __shared__ std::int32_t ties_s;

    const KernelGeometry& g    = a.geometry;
    const int tid              = static_cast<int>(threadIdx.x);
    const SelectColumn col     = select_column(a);
    const int column           = col.column;
    const std::int32_t visible = col.visible;
    const std::int32_t ratio   = g.ratio;
    const std::int32_t blocks  = col.blocks;
    const std::int32_t keep    = min(g.block_topk(), blocks);
    const std::int32_t tail    = visible - blocks * ratio;
    std::int32_t* selected     = a.selected + static_cast<std::int64_t>(column) * a.max_selected;

    if (blocks <= g.block_topk()) {
        for (int i = tid; i < visible; i += blockDim.x) { selected[i] = i; }
        if (tid == 0) { a.counts[column] = visible; }
        return;
    }
    const float* scores = a.scores + static_cast<std::int64_t>(blockIdx.x) * a.max_blocks;

    // Radix select the keep-th largest key.
    if (tid == 0) {
        prefix_s    = 0;
        remaining_s = keep;
    }
    __syncthreads();
    std::uint32_t mask = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        histogram[tid] = 0;
        __syncthreads();
        for (int b = tid; b < blocks; b += blockDim.x) {
            const std::uint32_t key = order_key(scores[b]);
            if ((key & mask) == prefix_s) { atomicAdd(&histogram[(key >> shift) & 0xFFu], 1u); }
        }
        __syncthreads();
        if (tid == 0) {
            std::int32_t need = remaining_s;
            for (int bin = 255; bin >= 0; --bin) {
                const std::int32_t count = static_cast<std::int32_t>(histogram[bin]);
                if (count >= need) {
                    prefix_s |= static_cast<std::uint32_t>(bin) << shift;
                    remaining_s = need;
                    break;
                }
                need -= count;
            }
        }
        mask |= 0xFFu << shift;
        __syncthreads();
    }
    const std::uint32_t threshold = prefix_s;
    // remaining_s ties at the threshold are admitted in ascending block order.
    if (tid == 0) {
        emitted_s = 0;
        ties_s    = remaining_s;
    }
    __syncthreads();
    for (int base = 0; base < blocks; base += blockDim.x) {
        const int b             = base + tid;
        const std::uint32_t key = b < blocks ? order_key(scores[b]) : 0u;
        const int greater       = b < blocks && key > threshold ? 1 : 0;
        const int tie           = b < blocks && key == threshold ? 1 : 0;
        // Inclusive scans of greater and tie flags.
        scan[tid] = greater | (tie << 16);
        __syncthreads();
        for (int offset = 1; offset < static_cast<int>(blockDim.x); offset <<= 1) {
            const std::int32_t add = tid >= offset ? scan[tid - offset] : 0;
            __syncthreads();
            scan[tid] += add;
            __syncthreads();
        }
        const std::int32_t inclusive = scan[tid];
        const std::int32_t total     = scan[blockDim.x - 1];
        const std::int32_t tie_rank  = (inclusive >> 16) - tie; // ties before this one
        const std::int32_t ties_left = ties_s;
        const bool take_tie          = tie && tie_rank < ties_left;
        __syncthreads(); // every thread has read the chunk totals before scan is reused
        // Emit order among taken blocks in this chunk.
        scan[tid] = (greater || take_tie) ? 1 : 0;
        __syncthreads();
        for (int offset = 1; offset < static_cast<int>(blockDim.x); offset <<= 1) {
            const std::int32_t add = tid >= offset ? scan[tid - offset] : 0;
            __syncthreads();
            scan[tid] += add;
            __syncthreads();
        }
        if (greater || take_tie) {
            const std::int32_t slot = emitted_s + scan[tid] - 1;
            for (int t = 0; t < ratio; ++t) { selected[slot * ratio + t] = b * ratio + t; }
        }
        __syncthreads();
        if (tid == 0) {
            emitted_s += scan[blockDim.x - 1];
            ties_s = max(0, ties_left - (total >> 16));
        }
        __syncthreads();
    }
    for (int t = tid; t < tail; t += blockDim.x) {
        selected[keep * ratio + t] = blocks * ratio + t;
    }
    if (tid == 0) { a.counts[column] = keep * ratio + tail; }
}

// ---------------------------------------------------------------------------------------------
// Attention

constexpr int kKeyTile = 16;

// Split policy: a call whose (column, KV head) CTAs cannot fill the GPU splits each selection
// into contiguous ranges of at least kMinSplitKeys keys until about kSplitTargetCtas CTAs run.
constexpr std::int32_t kSplitTargetCtas = 256;
constexpr std::int32_t kMinSplitKeys    = 2 * kKeyTile;

// Partial state of one (column, KV head, split, head): the unnormalized FP32 value sum, then
// the running max and the softmax denominator.
constexpr std::int32_t kPartialWords = kHeadDim + 2;

// Tensor Core attention. One CTA of four warps per (column, KV head, split) runs the query group
// as a 16-row MMA tile (rows past the group are zero) against gathered 32-key tiles: S = Q K^T, an
// FP32 online softmax, then O += P V with FP16 MMA where P is rounded to FP16.
//   BF16 cache (BF16 K, FP16 V): S uses BF16 MMA on the stored rows; K/V tiles are double-buffered
//   through cp.async.
//   Quantized caches (K always Hadamard-rotated): the codes are double-buffered through cp.async
//   and widened to FP16 tiles; Q is rotated like K and rounded to FP16 and S uses FP16 MMA. A
//   row-scaled E4M3 plane widens exactly and applies its per-key scale afterwards (K to S, V folded
//   into P before rounding); an int8 G64 plane widens with its group scales (one FP16 rounding);
//   an NVFP4 G16 plane widens exactly with its group scales. A rotated V leaves O in the rotated
//   domain, so the epilogue rotates each head's row back (the rotation is its own inverse).
// With `partial` null the CTA owns the whole selection and writes normalized output; otherwise it
// covers keys [split * split_keys, +split_keys) and writes its partial state.
namespace mma_attention {

constexpr int kRows      = 16;
constexpr int kKeys      = 32;
constexpr int kThreads   = 128;
constexpr int kChunks    = kHeadDim / 8; // 16-byte chunks per 256-element row
constexpr int kPStride   = kKeys + 8;    // FP16 P row stride, off the 128-byte bank period
constexpr int kRowBytes  = kHeadDim * 2;
constexpr int kTileBytes = kKeys * kRowBytes;

// 16-bit K/V tiles: two BF16/FP16 buffers, or one FP16 buffer the FP8 codes widen into.
template <bool Fp8>
struct Shared {
    static constexpr int kTiles = Fp8 ? 1 : 2;
    alignas(16) std::uint8_t q[kRows * kRowBytes];
    alignas(16) std::uint8_t k[kTiles][kTileBytes];
    alignas(16) std::uint8_t v[kTiles][kTileBytes];
    alignas(16) std::uint8_t codes[Fp8 ? 2 : 1][2][Fp8 ? kKeys * kHeadDim : 16]; // [buf][K|V]
    std::int64_t rows[Fp8 ? 2 : 1][kKeys];                                     // -1 when absent
    alignas(16) __half p[kRows * kPStride];
    float s[kRows * kKeys];
    float m[kRows], l[kRows], alpha[kRows];
    float k_scale[kKeys], v_scale[kKeys];
};

// Byte offset of 16-byte chunk `chunk` of row `row` in a swizzled 512-byte-row tile.
__device__ __forceinline__ int swizzle(int row, int chunk) {
    return row * kRowBytes + ((chunk ^ (row & 7)) << 4);
}

__device__ __forceinline__ unsigned pack_half2(float a, float b) {
    const __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const unsigned*>(&h);
}

// Widens the staged codes of one plane (`codes` [kKeys][256 bytes], row j at j * 256) into the
// swizzled FP16 tile `target`. `rows` are the keys' cache rows (-1 when absent); group scales are
// read from `scales`, and a row-scaled plane widens unscaled.
template <Codec C>
__device__ __forceinline__ void widen_tile(const std::uint8_t* codes, const std::int64_t* rows,
                                           const void* scales, std::uint8_t* target, int tid) {
    if constexpr (C == Codec::Nvfp4G16) {
        // 16 bytes hold 32 features: groups 2c and 2c + 1 of the row.
        const auto* group_scales = static_cast<const std::uint8_t*>(scales);
        for (int item = tid; item < kKeys * 8; item += kThreads) {
            const int j            = item / 8;
            const int chunk        = item % 8;
            const std::int64_t row = rows[j];
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int group = 2 * chunk + half;
                const std::uint8_t s =
                    row >= 0 ? group_scales[row * kKVCacheNvfp4Groups + group] : std::uint8_t{0};
                const auto wide =
                    kv_cache_nvfp4_dequant_f16x16(&codes[j * kHeadDim + chunk * 16 + half * 8], s);
                *reinterpret_cast<int4*>(&target[swizzle(j, 2 * group)])     = wide.lo;
                *reinterpret_cast<int4*>(&target[swizzle(j, 2 * group + 1)]) = wide.hi;
            }
        }
    } else {
        for (int item = tid; item < kKeys * 16; item += kThreads) {
            const int j       = item / 16;
            const int chunk   = item % 16;
            const uint4 bytes = *reinterpret_cast<const uint4*>(&codes[j * kHeadDim + chunk * 16]);
            uint4 wide[2];
            auto* halves = reinterpret_cast<__half2*>(wide);
            if constexpr (C == Codec::Fp8Row) {
                const auto* pairs = reinterpret_cast<const std::uint16_t*>(&bytes);
#pragma unroll
                for (int i = 0; i < 8; ++i) { halves[i] = kv_cache_fp8_code2_to_half2(pairs[i]); }
            } else {
                const std::int64_t row = rows[j];
                const __half s =
                    row >= 0 ? static_cast<const __half*>(
                                   scales)[row * kKVCacheInt8Groups + chunk / 4]
                             : __float2half_rn(0.0F);
                const __half2 s2  = __halves2half2(s, s);
                const auto* ints  = reinterpret_cast<const std::int8_t*>(&bytes);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const __half2 x = __halves2half2(__short2half_rn(ints[2 * i]),
                                                     __short2half_rn(ints[2 * i + 1]));
                    halves[i]       = __hmul2(x, s2);
                }
            }
            *reinterpret_cast<uint4*>(&target[swizzle(j, 2 * chunk)])     = wide[0];
            *reinterpret_cast<uint4*>(&target[swizzle(j, 2 * chunk + 1)]) = wide[1];
        }
    }
}

} // namespace mma_attention

template <KvCacheStorage Storage>
__global__ void __launch_bounds__(mma_attention::kThreads)
    qsa_attention_mma_kernel(const __nv_bfloat16* __restrict__ q,
                             const std::int32_t* __restrict__ selected,
                             const std::int32_t* __restrict__ counts, std::int32_t max_selected,
                             const std::int32_t* __restrict__ table_rows,
                             const std::int32_t* __restrict__ tables, std::int32_t table_stride,
                             const void* __restrict__ k_pages, const void* __restrict__ v_pages,
                             const void* __restrict__ k_scales,
                             const void* __restrict__ v_scales, std::int32_t query_heads,
                             std::int32_t kv_heads, std::int32_t width, float scale,
                             std::int32_t split_keys, float* __restrict__ partial,
                             __nv_bfloat16* __restrict__ out) {
    using namespace mma_attention;
    constexpr bool Fp8                = Storage != KvCacheStorage::BFloat16; // quantized cache
    constexpr ProfileCodecs codecs    = profile_codecs(Storage);
    constexpr bool kKeyRowScale       = Fp8 && codecs.k == Codec::Fp8Row;
    constexpr bool kValueRowScale     = Fp8 && codecs.v == Codec::Fp8Row;
    constexpr bool kRotatedValue      = Fp8 && codecs.v_rotated;
    extern __shared__ __align__(16) std::uint8_t raw_shared[];
    Shared<Fp8>& sh = *reinterpret_cast<Shared<Fp8>*>(raw_shared);

    const int tid            = static_cast<int>(threadIdx.x);
    const int warp           = tid >> 5;
    const int lane_id        = tid & 31;
    const int gid            = lane_id >> 2;
    const int lid            = lane_id & 3;
    const int kv_head        = static_cast<int>(blockIdx.x) % kv_heads;
    const int column         = static_cast<int>(blockIdx.x) / kv_heads;
    const int group          = query_heads / kv_heads;
    const std::int32_t* list = selected + static_cast<std::int64_t>(column) * max_selected;
    const std::int32_t first = partial == nullptr ? 0 : static_cast<int>(blockIdx.y) * split_keys;
    const std::int32_t count =
        partial == nullptr ? counts[column] : min(counts[column], first + split_keys);
    const std::int32_t* table =
        tables + static_cast<std::int64_t>(table_rows[column / width]) * table_stride;

    const auto key_row = [&](int base, int j) -> std::int64_t {
        if (base + j >= count) { return -1; }
        const std::int32_t token = list[base + j];
        return (static_cast<std::int64_t>(paged_kv_physical_page(table, token)) * kv_heads +
                kv_head) *
                   kPagedKVPageSize +
               (token & kPagedKVPageMask);
    };
    // Gather keys [base, base + kKeys) of the selection into buffer `buf`; missing keys are zero.
    auto stage = [&](int buf, int base) {
        if constexpr (!Fp8) {
            const auto* kb = static_cast<const __nv_bfloat16*>(k_pages);
            const auto* vh = static_cast<const __half*>(v_pages);
#pragma unroll 4
            for (int item = tid; item < kKeys * kChunks; item += kThreads) {
                const int j                = item / kChunks;
                const int chunk            = item - j * kChunks;
                const std::int64_t row     = key_row(base, j);
                const std::int64_t element = max(row, std::int64_t{0}) * kHeadDim + chunk * 8;
                const int offset           = swizzle(j, chunk);
                const int bytes            = row >= 0 ? 16 : 0;
                cp_async_zfill<16, Cache::cg>(&sh.k[buf][offset], kb + element, bytes);
                cp_async_zfill<16, Cache::cg>(&sh.v[buf][offset], vh + element, bytes);
            }
        } else {
            // Each row's codes in 16-byte chunks, unswizzled; the widening pass swizzles.
            if (tid < kKeys) { sh.rows[buf][tid] = key_row(base, tid); }
            const auto plane = [&](std::uint8_t* dst, const void* pages, int row_bytes) {
                const auto* codes = static_cast<const std::uint8_t*>(pages);
                const int chunks  = row_bytes / 16;
#pragma unroll 4
                for (int item = tid; item < kKeys * chunks; item += kThreads) {
                    const int j                = item / chunks;
                    const int chunk            = item - j * chunks;
                    const std::int64_t row     = key_row(base, j);
                    const std::int64_t element = max(row, std::int64_t{0}) * row_bytes + chunk * 16;
                    cp_async_zfill<16, Cache::cg>(&dst[j * kHeadDim + chunk * 16], codes + element,
                                                  row >= 0 ? 16 : 0);
                }
            };
            plane(sh.codes[buf][0], k_pages, code_bytes(codecs.k));
            plane(sh.codes[buf][1], v_pages, code_bytes(codecs.v));
        }
    };

    if (first < count) { stage(0, first); }
    cp_commit();

    // The query group, rows past the group zero.
    if constexpr (!Fp8) {
        for (int item = tid; item < kRows * kChunks; item += kThreads) {
            const int h     = item / kChunks;
            const int chunk = item - h * kChunks;
            uint4 value     = make_uint4(0, 0, 0, 0);
            if (h < group) {
                value = *reinterpret_cast<const uint4*>(
                    q + (static_cast<std::int64_t>(column) * query_heads + kv_head * group + h) *
                            kHeadDim +
                    chunk * 8);
            }
            *reinterpret_cast<uint4*>(&sh.q[swizzle(h, chunk)]) = value;
        }
    } else {
        // One warp per head: lane l rotates dimensions l + 32 r, then stores them as FP16.
        for (int h = warp; h < kRows; h += kThreads / 32) {
            float values[8] = {};
            if (h < group) {
                const std::int64_t base =
                    (static_cast<std::int64_t>(column) * query_heads + kv_head * group + h) *
                    kHeadDim;
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    values[r] = __bfloat162float(q[base + lane_id + 32 * r]);
                }
                normalized_hadamard_d256_inplace(values, lane_id);
            }
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int d = lane_id + 32 * r;
                *reinterpret_cast<__half*>(&sh.q[swizzle(h, d / 8) + (d % 8) * 2]) =
                    __float2half_rn(values[r]);
            }
        }
    }
    if (tid < kRows) {
        sh.m[tid] = -CUDART_INF_F;
        sh.l[tid] = 0.0F;
    }
    __syncthreads();

    // Q fragments for all sixteen K-steps stay in registers.
    unsigned qf[kHeadDim / 16][4];
#pragma unroll
    for (int step = 0; step < kHeadDim / 16; ++step) {
        ldmatrix_x4(qf[step][0], qf[step][1], qf[step][2], qf[step][3],
                    smem_addr(&sh.q[swizzle(lane_id & 15, 2 * step + (lane_id >> 4))]));
    }

    // This warp's output features [64 * warp, +64): eight n8 tiles.
    float acc[8][4];
#pragma unroll
    for (int t = 0; t < 8; ++t) { acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.0F; }

    int buf = 0;
    for (int base = first; base < count; base += kKeys, buf ^= 1) {
        if (base + kKeys < count) {
            stage(buf ^ 1, base + kKeys);
            cp_commit();
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();
        const int tile = Fp8 ? 0 : buf;

        if constexpr (Fp8) {
            // Widen this tile's codes to FP16 (swizzled) and read the keys' row scales.
            widen_tile<codecs.k>(sh.codes[buf][0], sh.rows[buf], k_scales, sh.k[0], tid);
            widen_tile<codecs.v>(sh.codes[buf][1], sh.rows[buf], v_scales, sh.v[0], tid);
            if (tid < kKeys) {
                const std::int64_t row = sh.rows[buf][tid];
                if constexpr (kKeyRowScale) {
                    sh.k_scale[tid] =
                        row >= 0 ? __half2float(static_cast<const __half*>(k_scales)[row]) : 0.0F;
                }
                if constexpr (kValueRowScale) {
                    sh.v_scale[tid] =
                        row >= 0 ? __half2float(static_cast<const __half*>(v_scales)[row]) : 0.0F;
                }
            }
            __syncthreads();
        }

        // S for keys [8 * warp, +8).
        {
            float c[4] = {0.0F, 0.0F, 0.0F, 0.0F};
            const int key = warp * 8 + (lane_id & 7);
#pragma unroll
            for (int pair = 0; pair < kHeadDim / 32; ++pair) {
                unsigned b[4];
                ldmatrix_x4(b[0], b[1], b[2], b[3],
                            smem_addr(&sh.k[tile][swizzle(key, 4 * pair + (lane_id >> 3))]));
                if constexpr (Fp8) {
                    mma_f16(c[0], c[1], c[2], c[3], qf[2 * pair][0], qf[2 * pair][1],
                            qf[2 * pair][2], qf[2 * pair][3], b[0], b[1]);
                    mma_f16(c[0], c[1], c[2], c[3], qf[2 * pair + 1][0], qf[2 * pair + 1][1],
                            qf[2 * pair + 1][2], qf[2 * pair + 1][3], b[2], b[3]);
                } else {
                    mma_bf16(c[0], c[1], c[2], c[3], qf[2 * pair][0], qf[2 * pair][1],
                             qf[2 * pair][2], qf[2 * pair][3], b[0], b[1]);
                    mma_bf16(c[0], c[1], c[2], c[3], qf[2 * pair + 1][0], qf[2 * pair + 1][1],
                             qf[2 * pair + 1][2], qf[2 * pair + 1][3], b[2], b[3]);
                }
            }
            const int col  = warp * 8 + 2 * lid;
            const float s0 = kKeyRowScale ? scale * sh.k_scale[col] : scale;
            const float s1 = kKeyRowScale ? scale * sh.k_scale[col + 1] : scale;
            sh.s[gid * kKeys + col]           = c[0] * s0;
            sh.s[gid * kKeys + col + 1]       = c[1] * s1;
            sh.s[(gid + 8) * kKeys + col]     = c[2] * s0;
            sh.s[(gid + 8) * kKeys + col + 1] = c[3] * s1;
        }
        __syncthreads();

        // Online softmax: eight threads per row, four keys apiece.
        {
            const int row  = tid >> 3;
            const int col0 = (tid & 7) * 4;
            float s[4];
            float tile_max = -CUDART_INF_F;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                s[i]     = base + col0 + i < count ? sh.s[row * kKeys + col0 + i] : -CUDART_INF_F;
                tile_max = fmaxf(tile_max, s[i]);
            }
#pragma unroll
            for (int offset = 1; offset < 8; offset <<= 1) {
                tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, offset));
            }
            const float m_old = sh.m[row];
            const float m_new = fmaxf(m_old, tile_max);
            float sum         = 0.0F;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                s[i] = base + col0 + i < count ? __expf(s[i] - m_new) : 0.0F;
                sum += s[i];
            }
#pragma unroll
            for (int offset = 1; offset < 8; offset <<= 1) {
                sum += __shfl_xor_sync(0xffffffffu, sum, offset);
            }
            float pv[4] = {s[0], s[1], s[2], s[3]};
            if constexpr (kValueRowScale) {
#pragma unroll
                for (int i = 0; i < 4; ++i) { pv[i] *= sh.v_scale[col0 + i]; }
            }
            *reinterpret_cast<uint2*>(&sh.p[row * kPStride + col0]) =
                make_uint2(pack_half2(pv[0], pv[1]), pack_half2(pv[2], pv[3]));
            __syncwarp();
            if ((tid & 7) == 0) {
                const float alpha = __expf(m_old - m_new);
                sh.alpha[row]     = alpha;
                sh.l[row]         = sh.l[row] * alpha + sum;
                sh.m[row]         = m_new;
            }
        }
        __syncthreads();

        // O = alpha O + P V over this warp's 64 features.
        {
            const float a_lo = sh.alpha[gid];
            const float a_hi = sh.alpha[gid + 8];
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                acc[t][0] *= a_lo;
                acc[t][1] *= a_lo;
                acc[t][2] *= a_hi;
                acc[t][3] *= a_hi;
            }
#pragma unroll
            for (int step = 0; step < kKeys / 16; ++step) {
                unsigned a[4];
                ldmatrix_x4(a[0], a[1], a[2], a[3],
                            smem_addr(&sh.p[(lane_id & 15) * kPStride + step * 16 +
                                            (lane_id >> 4) * 8]));
                const int key = step * 16 + (lane_id & 7) + ((lane_id >> 3) & 1) * 8;
#pragma unroll
                for (int pair = 0; pair < 4; ++pair) {
                    const int chunk = warp * 8 + pair * 2 + (lane_id >> 4);
                    unsigned b[4];
                    ldmatrix_x4_t(b[0], b[1], b[2], b[3],
                                  smem_addr(&sh.v[tile][swizzle(key, chunk)]));
                    mma_f16(acc[2 * pair][0], acc[2 * pair][1], acc[2 * pair][2],
                            acc[2 * pair][3], a[0], a[1], a[2], a[3], b[0], b[1]);
                    mma_f16(acc[2 * pair + 1][0], acc[2 * pair + 1][1], acc[2 * pair + 1][2],
                            acc[2 * pair + 1][3], a[0], a[1], a[2], a[3], b[2], b[3]);
                }
            }
        }
        __syncthreads(); // the next prefetch reuses this buffer
    }
    cp_wait<0>();

    if constexpr (kRotatedValue) {
        // Gather O's unnormalized rows through the free K tile, then one warp per head rotates its
        // row back before normalizing or writing the partial state.
        auto* rows = reinterpret_cast<float*>(sh.k[0]);
        static_assert(sizeof(sh.k[0]) >= kRows * kHeadDim * sizeof(float));
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const int d = warp * 64 + t * 8 + 2 * lid;
            rows[gid * kHeadDim + d]           = acc[t][0];
            rows[gid * kHeadDim + d + 1]       = acc[t][1];
            rows[(gid + 8) * kHeadDim + d]     = acc[t][2];
            rows[(gid + 8) * kHeadDim + d + 1] = acc[t][3];
        }
        __syncthreads();
        for (int h = warp; h < group; h += kThreads / 32) {
            float values[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) { values[r] = rows[h * kHeadDim + lane_id + 32 * r]; }
            normalized_hadamard_d256_inplace(values, lane_id);
            if (partial != nullptr) {
                const std::int64_t state =
                    (static_cast<std::int64_t>(blockIdx.x) * gridDim.y + blockIdx.y) * group + h;
                float* p = partial + state * kPartialWords;
#pragma unroll
                for (int r = 0; r < 8; ++r) { p[lane_id + 32 * r] = values[r]; }
                if (lane_id == 0) {
                    p[kHeadDim]     = sh.m[h];
                    p[kHeadDim + 1] = sh.l[h];
                }
            } else {
                const float l   = sh.l[h];
                const float inv = count > 0 && l > 0.0F ? 1.0F / l : 0.0F;
                __nv_bfloat16* o =
                    out + (static_cast<std::int64_t>(column) * query_heads + kv_head * group + h) *
                              kHeadDim;
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    o[lane_id + 32 * r] = __float2bfloat16_rn(values[r] * inv);
                }
            }
        }
        return;
    }

    const float m_lo = sh.m[gid], m_hi = sh.m[gid + 8];
    const float l_lo = sh.l[gid], l_hi = sh.l[gid + 8];
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        const int d = warp * 64 + t * 8 + 2 * lid;
#pragma unroll
        for (int upper = 0; upper < 2; ++upper) {
            const int h = gid + upper * 8;
            if (h >= group) { continue; }
            const float x0 = acc[t][2 * upper], x1 = acc[t][2 * upper + 1];
            if (partial != nullptr) {
                const std::int64_t state =
                    (static_cast<std::int64_t>(blockIdx.x) * gridDim.y + blockIdx.y) * group + h;
                float* p = partial + state * kPartialWords;
                p[d]     = x0;
                p[d + 1] = x1;
                if (t == 0 && lid == 0) {
                    p[kHeadDim]     = upper == 0 ? m_lo : m_hi;
                    p[kHeadDim + 1] = upper == 0 ? l_lo : l_hi;
                }
            } else {
                const float l     = upper == 0 ? l_lo : l_hi;
                const float inv   = count > 0 && l > 0.0F ? 1.0F / l : 0.0F;
                const std::int64_t index =
                    (static_cast<std::int64_t>(column) * query_heads + kv_head * group + h) *
                        kHeadDim +
                    d;
                *reinterpret_cast<__nv_bfloat162*>(&out[index]) =
                    __floats2bfloat162_rn(x0 * inv, x1 * inv);
            }
        }
    }
}

// One CTA per (column, KV head) and head of its group; thread d merges feature d over the splits.
__global__ void __launch_bounds__(256)
    qsa_attention_combine_kernel(const float* __restrict__ partial, std::int32_t splits,
                                 std::int32_t query_heads, std::int32_t kv_heads,
                                 __nv_bfloat16* __restrict__ out) {
    const int d       = static_cast<int>(threadIdx.x);
    const int kv_head = static_cast<int>(blockIdx.x) % kv_heads;
    const int column  = static_cast<int>(blockIdx.x) / kv_heads;
    const int group   = query_heads / kv_heads;
    const int h       = static_cast<int>(blockIdx.y);
    const float* p =
        partial + (static_cast<std::int64_t>(blockIdx.x) * splits * group + h) * kPartialWords;
    const std::int64_t stride = static_cast<std::int64_t>(group) * kPartialWords;
    float m                   = -CUDART_INF_F;
    for (int s = 0; s < splits; ++s) { m = fmaxf(m, p[s * stride + kHeadDim]); }
    float numerator = 0.0F, denominator = 0.0F;
    if (m > -CUDART_INF_F) {
        for (int s = 0; s < splits; ++s) {
            const float* state = p + s * stride;
            const float l      = state[kHeadDim + 1];
            if (l > 0.0F) {
                const float w = __expf(state[kHeadDim] - m);
                numerator += w * state[d];
                denominator += w * l;
            }
        }
    }
    const std::int64_t index =
        (static_cast<std::int64_t>(column) * query_heads + kv_head * group + h) * kHeadDim + d;
    out[index] = __float2bfloat16_rn(denominator > 0.0F ? numerator / denominator : 0.0F);
}

// Calls fn.template operator()<Storage>() for the cache profile.
template <typename Fn>
void dispatch_storage(KvCacheStorage storage, Fn&& fn) {
    switch (storage) {
    case KvCacheStorage::BFloat16: fn.template operator()<KvCacheStorage::BFloat16>(); return;
    case KvCacheStorage::Int8Group64: fn.template operator()<KvCacheStorage::Int8Group64>(); return;
    case KvCacheStorage::Fp8E4M3Row256:
        fn.template operator()<KvCacheStorage::Fp8E4M3Row256>();
        return;
    case KvCacheStorage::Nvfp4Group16: fn.template operator()<KvCacheStorage::Nvfp4Group16>(); return;
    case KvCacheStorage::Fp8KeyNvfp4Value:
        fn.template operator()<KvCacheStorage::Fp8KeyNvfp4Value>();
        return;
    }
    throw std::invalid_argument("Qwen Sparse Attention: unknown KV-cache profile");
}

} // namespace

void qsa_append_launch(const Tensor& k, const Tensor& v, const Tensor& index_keys,
                       const Tensor& rope_positions, const Tensor& cache_positions,
                       const Tensor* valid_columns, const Tensor& table_rows,
                       const PagedKVBatchLayerView& kv, const QsaIndexPlane& index,
                       cudaStream_t stream) {
    const std::int32_t width    = cache_positions.ne[0];
    const std::int32_t lanes    = cache_positions.ne[1];
    const std::int32_t kv_heads = kv.num_kv_heads;
    const std::int64_t warps    = static_cast<std::int64_t>(width) * lanes * (kv_heads + 1);
    const int blocks            = static_cast<int>((warps * 32 + kAppendThreads - 1) / kAppendThreads);
    const auto* valid =
        valid_columns ? static_cast<const std::int32_t*>(valid_columns->data) : nullptr;
    dispatch_storage(kv.storage, [&]<KvCacheStorage Storage>() {
        qsa_append_kernel<Storage><<<blocks, kAppendThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
            static_cast<const __nv_bfloat16*>(index_keys.data),
            static_cast<const std::int32_t*>(rope_positions.data), rope_positions.ne[1],
            static_cast<const std::int32_t*>(cache_positions.data), valid,
            static_cast<const std::int32_t*>(table_rows.data),
            static_cast<const std::int32_t*>(kv.block_tables.data), kv.block_tables.ne[0],
            kv.k_pages.data, kv.v_pages.data, kv.k_scale_pages.data, kv.v_scale_pages.data,
            static_cast<__nv_bfloat16*>(index.pages.data), kv_heads, width, lanes);
    });
    CUDA_CHECK(cudaGetLastError());
}

std::int32_t qsa_select_pass_columns(std::int32_t columns) { return std::min(columns, 256); }

void qsa_select_launch(const Tensor& index_query, const Tensor& query_rope_positions,
                       const Tensor& cache_positions, const Tensor* valid_columns,
                       const Tensor& table_rows, const Tensor& query_norm, const Tensor& key_norm,
                       const QsaIndexPlane& index, const QsaGeometry& geometry,
                       std::int32_t max_blocks, float* scores, Tensor& selected, Tensor& counts,
                       cudaStream_t stream) {
    const std::int32_t width   = cache_positions.ne[0];
    const std::int32_t lanes   = cache_positions.ne[1];
    const std::int32_t columns = width * lanes;
    SelectArgs args{};
    args.index_query     = static_cast<const __nv_bfloat16*>(index_query.data);
    args.query_positions = static_cast<const std::int32_t*>(query_rope_positions.data);
    args.query_axes      = query_rope_positions.ne[1];
    args.cache           = static_cast<const std::int32_t*>(cache_positions.data);
    args.valid = valid_columns ? static_cast<const std::int32_t*>(valid_columns->data) : nullptr;
    args.table_rows         = static_cast<const std::int32_t*>(table_rows.data);
    args.tables             = static_cast<const std::int32_t*>(index.block_tables.data);
    args.table_stride       = index.block_tables.ne[0];
    args.records            = static_cast<const __nv_bfloat16*>(index.pages.data);
    args.query_norm         = static_cast<const __nv_bfloat16*>(query_norm.data);
    args.key_norm           = static_cast<const __nv_bfloat16*>(key_norm.data);
    args.scores             = scores;
    args.max_blocks         = max_blocks;
    args.selected           = static_cast<std::int32_t*>(selected.data);
    args.counts             = static_cast<std::int32_t*>(counts.data);
    args.width              = width;
    args.lanes              = lanes;
    args.max_selected       = selected.ne[0];
    args.geometry           = kernel_geometry(geometry);
    const std::int32_t pass = qsa_select_pass_columns(columns);
    for (std::int32_t begin = 0; begin < columns; begin += pass) {
        args.column_begin       = begin;
        const std::int32_t cols = std::min(pass, columns - begin);
        // Narrow passes split each column's blocks across CTAs, whole warps' worth apiece.
        const std::int32_t by_blocks = (max_blocks + kMinScoreBlocks - 1) / kMinScoreBlocks;
        const std::int32_t splits =
            std::max(1, std::min((kScoreTargetCtas + cols - 1) / cols, by_blocks));
        const std::int32_t split_blocks = (max_blocks + splits - 1) / splits;
        qsa_score_kernel<<<dim3(cols, splits), 256, 0, stream>>>(args, split_blocks);
        CUDA_CHECK(cudaGetLastError());
        qsa_select_kernel<<<cols, 256, 0, stream>>>(args);
        CUDA_CHECK(cudaGetLastError());
    }
}

QsaAttentionSplit qsa_attention_split(std::int32_t columns, std::int32_t kv_heads,
                                      std::int32_t max_selected) {
    const std::int64_t ctas    = static_cast<std::int64_t>(columns) * kv_heads;
    const std::int32_t by_keys = (max_selected + kMinSplitKeys - 1) / kMinSplitKeys;
    const auto wanted          = static_cast<std::int32_t>(
        std::min<std::int64_t>((kSplitTargetCtas + ctas - 1) / ctas, by_keys));
    if (wanted <= 1) { return {.splits = 1, .split_keys = max_selected}; }
    // Whole key tiles per split; the last split may be short or empty.
    const std::int32_t tiles = (max_selected + kKeyTile - 1) / kKeyTile;
    const std::int32_t keys  = (tiles + wanted - 1) / wanted * kKeyTile;
    return {.splits = (max_selected + keys - 1) / keys, .split_keys = keys};
}

std::size_t qsa_attention_partial_bytes(std::int32_t columns, std::int32_t query_heads,
                                        std::int32_t kv_heads, std::int32_t max_selected) {
    const QsaAttentionSplit split = qsa_attention_split(columns, kv_heads, max_selected);
    if (split.splits == 1) { return 0; }
    return static_cast<std::size_t>(columns) * kv_heads * split.splits * (query_heads / kv_heads) *
           kPartialWords * sizeof(float);
}

void qsa_attention_launch(const Tensor& q, const Tensor& selected, const Tensor& counts,
                          const Tensor& table_rows, const PagedKVBatchLayerView& kv, float scale,
                          float* partial, Tensor& out, cudaStream_t stream) {
    const std::int32_t query_heads = q.ne[1];
    const std::int32_t width       = q.ne[2];
    const std::int32_t lanes       = q.ne[3];
    const std::int32_t kv_heads    = kv.num_kv_heads;
    const int blocks               = width * lanes * kv_heads;
    const QsaAttentionSplit split  = qsa_attention_split(width * lanes, kv_heads, selected.ne[0]);
    float* states                  = split.splits == 1 ? nullptr : partial;
    if (query_heads % kv_heads != 0 || query_heads / kv_heads > mma_attention::kRows) {
        throw std::invalid_argument("qsa_attention: a KV head serves at most 16 query heads");
    }
    dispatch_storage(kv.storage, [&]<KvCacheStorage Storage>() {
        constexpr int kShared =
            sizeof(mma_attention::Shared<Storage != KvCacheStorage::BFloat16>);
        static const bool configured = [] {
            CUDA_CHECK(cudaFuncSetAttribute(qsa_attention_mma_kernel<Storage>,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize, kShared));
            return true;
        }();
        (void)configured;
        qsa_attention_mma_kernel<Storage>
            <<<dim3(blocks, split.splits), mma_attention::kThreads, kShared, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data),
                static_cast<const std::int32_t*>(selected.data),
                static_cast<const std::int32_t*>(counts.data), selected.ne[0],
                static_cast<const std::int32_t*>(table_rows.data),
                static_cast<const std::int32_t*>(kv.block_tables.data), kv.block_tables.ne[0],
                kv.k_pages.data, kv.v_pages.data, kv.k_scale_pages.data, kv.v_scale_pages.data,
                query_heads, kv_heads, width, scale, split.split_keys, states,
                static_cast<__nv_bfloat16*>(out.data));
    });
    CUDA_CHECK(cudaGetLastError());
    if (states != nullptr) {
        qsa_attention_combine_kernel<<<dim3(blocks, query_heads / kv_heads), 256, 0, stream>>>(
            states, split.splits, query_heads, kv_heads, static_cast<__nv_bfloat16*>(out.data));
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace ninfer::ops::detail
