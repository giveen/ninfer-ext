// Implements: include/ninfer/ops/sparse_attention.h
// Qwen Sparse Attention: paged append, indexer block selection and selected-key attention.
// One CTA per query column (selection) or per (column, KV head, key split) (attention), with FP32
// accumulation and K/V read through the same codecs the dense cache uses. BF16 K/V attention runs
// the query group on Tensor Cores; FP8 K/V attention is scalar. Narrow calls split each selection
// across CTAs and merge the partial softmax states.
#include "ops/sparse_attention/launch.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/fp8_e4m3_row_codec.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <math_constants.h>

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kHeadDim      = 256;
constexpr int kMaxKvHeads   = 8;
constexpr int kMaxGroup     = 16;
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
// Append

template <bool Fp8>
__global__ void qsa_append_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ index_keys, const std::int32_t* __restrict__ rope_positions,
    std::int32_t rope_axes, const std::int32_t* __restrict__ cache,
    const std::int32_t* __restrict__ valid, const std::int32_t* __restrict__ table_rows,
    const std::int32_t* __restrict__ tables, std::int32_t table_stride, void* k_pages,
    void* v_pages, __half* k_scales, __half* v_scales, __nv_bfloat16* __restrict__ records,
    std::int32_t kv_heads, std::int32_t width, std::int32_t lanes) {
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
    const std::int64_t target =
        ((static_cast<std::int64_t>(page) * kv_heads + head) * kPagedKVPageSize + offset) *
        kHeadDim;
    if constexpr (!Fp8) {
        auto* kb = static_cast<__nv_bfloat16*>(k_pages);
        auto* vh = static_cast<__half*>(v_pages);
        for (int d = lane_id; d < kHeadDim; d += 32) {
            kb[target + d] = k[source + d];
            vh[target + d] = __float2half_rn(__bfloat162float(v[source + d]));
        }
    } else {
        constexpr unsigned kMask = 0xffffffffu;
        const std::int64_t scale_index =
            (static_cast<std::int64_t>(page) * kv_heads + head) * kPagedKVPageSize + offset;
        auto* kc = static_cast<std::uint8_t*>(k_pages);
        auto* vc = static_cast<std::uint8_t*>(v_pages);
        float values[8];
        float absmax = 0.0F;
#pragma unroll
        for (int r = 0; r < 8; ++r) { values[r] = __bfloat162float(k[source + lane_id + 32 * r]); }
        normalized_hadamard_d256_inplace(values, lane_id);
#pragma unroll
        for (float x : values) { absmax = fmaxf(absmax, fabsf(x)); }
        const auto kq = kv_cache_fp8_quant_params(warp_max(absmax, kMask));
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            kc[target + lane_id + 32 * r] = kv_cache_fp8_quant_code(values[r], kq.inverse_scale);
        }
        if (lane_id == 0) { k_scales[scale_index] = kq.scale; }
        absmax = 0.0F;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            values[r] = __bfloat162float(v[source + lane_id + 32 * r]);
            absmax    = fmaxf(absmax, fabsf(values[r]));
        }
        const auto vq = kv_cache_fp8_quant_params(warp_max(absmax, kMask));
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            vc[target + lane_id + 32 * r] = kv_cache_fp8_quant_code(values[r], vq.inverse_scale);
        }
        if (lane_id == 0) { v_scales[scale_index] = vq.scale; }
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

// With `partial` null the CTA owns the whole selection and writes normalized output; otherwise
// it covers keys [split * split_keys, +split_keys) and writes its partial state.
// FP8 E4M3 row-scaled K/V: the query group is Hadamard-rotated to match the stored keys.
__global__ void __launch_bounds__(256)
    qsa_attention_fp8_kernel(const __nv_bfloat16* __restrict__ q,
                         const std::int32_t* __restrict__ selected,
                         const std::int32_t* __restrict__ counts, std::int32_t max_selected,
                         const std::int32_t* __restrict__ table_rows,
                         const std::int32_t* __restrict__ tables, std::int32_t table_stride,
                         const void* k_pages, const void* v_pages, const __half* k_scales,
                         const __half* v_scales, std::int32_t query_heads, std::int32_t kv_heads,
                         std::int32_t width, float scale, std::int32_t split_keys,
                         float* __restrict__ partial, __nv_bfloat16* __restrict__ out) {
    __shared__ float qs[kMaxGroup * kHeadDim];
    __shared__ float ks[kKeyTile * kHeadDim];
    __shared__ float vs[kKeyTile * kHeadDim];
    __shared__ float ps[kMaxGroup * kKeyTile];
    __shared__ float m_s[kMaxGroup], l_s[kMaxGroup], alpha_s[kMaxGroup];

    const int tid            = static_cast<int>(threadIdx.x);
    const int warp           = tid >> 5;
    const int lane_id        = tid & 31;
    const int kv_head        = static_cast<int>(blockIdx.x) % kv_heads;
    const int column         = static_cast<int>(blockIdx.x) / kv_heads;
    const int lane           = column / width;
    const int group          = query_heads / kv_heads;
    const std::int32_t* list = selected + static_cast<std::int64_t>(column) * max_selected;
    const std::int32_t first = partial == nullptr ? 0 : static_cast<int>(blockIdx.y) * split_keys;
    const std::int32_t count =
        partial == nullptr ? counts[column] : min(counts[column], first + split_keys);

    // Load (and for FP8, Hadamard-rotate) this KV head's query group.
    for (int h = warp; h < group; h += 8) {
        const std::int64_t base =
            (static_cast<std::int64_t>(column) * query_heads + kv_head * group + h) * kHeadDim;
        float values[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) { values[r] = __bfloat162float(q[base + lane_id + 32 * r]); }
        normalized_hadamard_d256_inplace(values, lane_id);
#pragma unroll
        for (int r = 0; r < 8; ++r) { qs[h * kHeadDim + lane_id + 32 * r] = values[r]; }
    }
    if (tid < group) {
        m_s[tid] = -CUDART_INF_F;
        l_s[tid] = 0.0F;
    }
    float acc[kMaxGroup];
#pragma unroll
    for (int h = 0; h < kMaxGroup; ++h) { acc[h] = 0.0F; }
    __syncthreads();

    const std::int32_t* table = tables + static_cast<std::int64_t>(table_rows[lane]) * table_stride;
    for (int base = first; base < count; base += kKeyTile) {
        const int tile = min(kKeyTile, count - base);
        // Load K/V rows of the tile, one warp per key.
        for (int j = warp; j < tile; j += 8) {
            const std::int32_t token  = list[base + j];
            const std::int32_t page   = paged_kv_physical_page(table, token);
            const std::int32_t offset = token & kPagedKVPageMask;
            const std::int64_t row =
                ((static_cast<std::int64_t>(page) * kv_heads + kv_head) * kPagedKVPageSize +
                 offset);
            const auto* kc  = static_cast<const std::uint8_t*>(k_pages);
            const auto* vc  = static_cast<const std::uint8_t*>(v_pages);
            const float ksc = __half2float(k_scales[row]);
            const float vsc = __half2float(v_scales[row]);
            for (int d = lane_id; d < kHeadDim; d += 32) {
                __nv_fp8_e4m3 kcode, vcode;
                kcode.__x            = kc[row * kHeadDim + d];
                vcode.__x            = vc[row * kHeadDim + d];
                ks[j * kHeadDim + d] = static_cast<float>(kcode) * ksc;
                vs[j * kHeadDim + d] = static_cast<float>(vcode) * vsc;
            }
        }
        __syncthreads();
        // Scores: one thread per (head, key).
        for (int pair = tid; pair < group * kKeyTile; pair += blockDim.x) {
            const int h = pair / kKeyTile;
            const int j = pair - h * kKeyTile;
            float s     = -CUDART_INF_F;
            if (j < tile) {
                float dot = 0.0F;
                for (int d = 0; d < kHeadDim; ++d) {
                    dot += qs[h * kHeadDim + d] * ks[j * kHeadDim + d];
                }
                s = dot * scale;
            }
            ps[h * kKeyTile + j] = s;
        }
        __syncthreads();
        if (tid < group) {
            const int h    = tid;
            float tile_max = -CUDART_INF_F;
            for (int j = 0; j < tile; ++j) { tile_max = fmaxf(tile_max, ps[h * kKeyTile + j]); }
            const float m_new = fmaxf(m_s[h], tile_max);
            const float alpha = __expf(m_s[h] - m_new);
            float l           = l_s[h] * alpha;
            for (int j = 0; j < kKeyTile; ++j) {
                const float p        = j < tile ? __expf(ps[h * kKeyTile + j] - m_new) : 0.0F;
                ps[h * kKeyTile + j] = p;
                l += p;
            }
            m_s[h]     = m_new;
            l_s[h]     = l;
            alpha_s[h] = alpha;
        }
        __syncthreads();
        // Value accumulation: thread d owns feature d for every head of the group.
        const int d = tid;
        for (int h = 0; h < group; ++h) {
            float sum = 0.0F;
            for (int j = 0; j < tile; ++j) { sum += ps[h * kKeyTile + j] * vs[j * kHeadDim + d]; }
            acc[h] = acc[h] * alpha_s[h] + sum;
        }
        __syncthreads();
    }
    const int d = tid;
    if (partial != nullptr) {
        const std::int64_t state =
            (static_cast<std::int64_t>(blockIdx.x) * gridDim.y + blockIdx.y) * group;
        for (int h = 0; h < group; ++h) {
            float* p = partial + (state + h) * kPartialWords;
            p[d]     = acc[h];
            if (d == 0) {
                p[kHeadDim]     = m_s[h];
                p[kHeadDim + 1] = l_s[h];
            }
        }
        return;
    }
    for (int h = 0; h < group; ++h) {
        const std::int64_t index =
            (static_cast<std::int64_t>(column) * query_heads + kv_head * group + h) * kHeadDim + d;
        const float value = count > 0 && l_s[h] > 0.0F ? acc[h] / l_s[h] : 0.0F;
        out[index]        = __float2bfloat16_rn(value);
    }
}

// Tensor Core attention over a BF16 K / FP16 V cache. One CTA of four warps per (column, KV head,
// split) runs the query group as a 16-row MMA tile (rows past the group are zero) against
// gathered 32-key tiles: S = Q K^T with BF16 MMA, an FP32 online softmax, then O += P V with FP16
// MMA where P is rounded to FP16. K/V tiles are double-buffered through cp.async. Output and
// partial states match qsa_attention_fp8_kernel.
namespace mma_attention {

constexpr int kRows      = 16;
constexpr int kKeys      = 32;
constexpr int kThreads   = 128;
constexpr int kChunks    = kHeadDim / 8; // 16-byte chunks per 256-element row
constexpr int kPStride   = kKeys + 8;    // FP16 P row stride, off the 128-byte bank period
constexpr int kRowBytes  = kHeadDim * 2;
constexpr int kTileBytes = kKeys * kRowBytes;

struct Shared {
    alignas(16) std::uint8_t q[kRows * kRowBytes];
    alignas(16) std::uint8_t k[2][kTileBytes];
    alignas(16) std::uint8_t v[2][kTileBytes];
    alignas(16) __half p[kRows * kPStride];
    float s[kRows * kKeys];
    float m[kRows], l[kRows], alpha[kRows];
};

// Byte offset of 16-byte chunk `chunk` of row `row` in a swizzled 512-byte-row tile.
__device__ __forceinline__ int swizzle(int row, int chunk) {
    return row * kRowBytes + ((chunk ^ (row & 7)) << 4);
}

__device__ __forceinline__ unsigned pack_half2(float a, float b) {
    const __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const unsigned*>(&h);
}

} // namespace mma_attention

__global__ void __launch_bounds__(mma_attention::kThreads)
    qsa_attention_mma_kernel(const __nv_bfloat16* __restrict__ q,
                             const std::int32_t* __restrict__ selected,
                             const std::int32_t* __restrict__ counts, std::int32_t max_selected,
                             const std::int32_t* __restrict__ table_rows,
                             const std::int32_t* __restrict__ tables, std::int32_t table_stride,
                             const __nv_bfloat16* __restrict__ k_pages,
                             const __half* __restrict__ v_pages, std::int32_t query_heads,
                             std::int32_t kv_heads, std::int32_t width, float scale,
                             std::int32_t split_keys, float* __restrict__ partial,
                             __nv_bfloat16* __restrict__ out) {
    using namespace mma_attention;
    extern __shared__ __align__(16) std::uint8_t raw_shared[];
    Shared& sh = *reinterpret_cast<Shared*>(raw_shared);

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

    // Gather keys [base, base + kKeys) of the selection into buffer `buf`; missing keys are zero.
    auto stage = [&](int buf, int base) {
#pragma unroll 4
        for (int item = tid; item < kKeys * kChunks; item += kThreads) {
            const int j     = item / kChunks;
            const int chunk = item - j * kChunks;
            const bool live = base + j < count;
            std::int64_t row = 0;
            if (live) {
                const std::int32_t token = list[base + j];
                row = (static_cast<std::int64_t>(paged_kv_physical_page(table, token)) * kv_heads +
                       kv_head) *
                          kPagedKVPageSize +
                      (token & kPagedKVPageMask);
            }
            const std::int64_t element = row * kHeadDim + chunk * 8;
            const int offset           = swizzle(j, chunk);
            cp_async_zfill<16, Cache::cg>(&sh.k[buf][offset], k_pages + element, live ? 16 : 0);
            cp_async_zfill<16, Cache::cg>(&sh.v[buf][offset], v_pages + element, live ? 16 : 0);
        }
    };

    if (first < count) { stage(0, first); }
    cp_commit();

    // The query group, rows past the group zero.
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

        // S for keys [8 * warp, +8).
        {
            float c[4] = {0.0F, 0.0F, 0.0F, 0.0F};
            const int key = warp * 8 + (lane_id & 7);
#pragma unroll
            for (int pair = 0; pair < kHeadDim / 32; ++pair) {
                unsigned b[4];
                ldmatrix_x4(b[0], b[1], b[2], b[3],
                            smem_addr(&sh.k[buf][swizzle(key, 4 * pair + (lane_id >> 3))]));
                mma_bf16(c[0], c[1], c[2], c[3], qf[2 * pair][0], qf[2 * pair][1],
                         qf[2 * pair][2], qf[2 * pair][3], b[0], b[1]);
                mma_bf16(c[0], c[1], c[2], c[3], qf[2 * pair + 1][0], qf[2 * pair + 1][1],
                         qf[2 * pair + 1][2], qf[2 * pair + 1][3], b[2], b[3]);
            }
            const int col = warp * 8 + 2 * lid;
            sh.s[gid * kKeys + col]           = c[0] * scale;
            sh.s[gid * kKeys + col + 1]       = c[1] * scale;
            sh.s[(gid + 8) * kKeys + col]     = c[2] * scale;
            sh.s[(gid + 8) * kKeys + col + 1] = c[3] * scale;
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
            *reinterpret_cast<uint2*>(&sh.p[row * kPStride + col0]) =
                make_uint2(pack_half2(s[0], s[1]), pack_half2(s[2], s[3]));
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
                                  smem_addr(&sh.v[buf][swizzle(key, chunk)]));
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
    const int blocks            = static_cast<int>((warps * 32 + 255) / 256);
    const auto* valid =
        valid_columns ? static_cast<const std::int32_t*>(valid_columns->data) : nullptr;
    const bool fp8 = kv.storage == KvCacheStorage::Fp8E4M3Row256;
    auto launch    = [&](auto kernel) {
        kernel<<<blocks, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
            static_cast<const __nv_bfloat16*>(index_keys.data),
            static_cast<const std::int32_t*>(rope_positions.data), rope_positions.ne[1],
            static_cast<const std::int32_t*>(cache_positions.data), valid,
            static_cast<const std::int32_t*>(table_rows.data),
            static_cast<const std::int32_t*>(kv.block_tables.data), kv.block_tables.ne[0],
            kv.k_pages.data, kv.v_pages.data,
            fp8 ? static_cast<__half*>(kv.k_scale_pages.data) : nullptr,
            fp8 ? static_cast<__half*>(kv.v_scale_pages.data) : nullptr,
            static_cast<__nv_bfloat16*>(index.pages.data), kv_heads, width, lanes);
    };
    if (fp8) {
        launch(qsa_append_kernel<true>);
    } else {
        launch(qsa_append_kernel<false>);
    }
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
    if (kv.storage == KvCacheStorage::Fp8E4M3Row256) {
        qsa_attention_fp8_kernel<<<dim3(blocks, split.splits), 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const std::int32_t*>(selected.data),
            static_cast<const std::int32_t*>(counts.data), selected.ne[0],
            static_cast<const std::int32_t*>(table_rows.data),
            static_cast<const std::int32_t*>(kv.block_tables.data), kv.block_tables.ne[0],
            kv.k_pages.data, kv.v_pages.data, static_cast<const __half*>(kv.k_scale_pages.data),
            static_cast<const __half*>(kv.v_scale_pages.data), query_heads, kv_heads, width,
            scale, split.split_keys, states, static_cast<__nv_bfloat16*>(out.data));
    } else {
        constexpr int kShared = sizeof(mma_attention::Shared);
        static const bool configured = [] {
            CUDA_CHECK(cudaFuncSetAttribute(qsa_attention_mma_kernel,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize, kShared));
            return true;
        }();
        (void)configured;
        qsa_attention_mma_kernel<<<dim3(blocks, split.splits), mma_attention::kThreads, kShared,
                                   stream>>>(
            static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const std::int32_t*>(selected.data),
            static_cast<const std::int32_t*>(counts.data), selected.ne[0],
            static_cast<const std::int32_t*>(table_rows.data),
            static_cast<const std::int32_t*>(kv.block_tables.data), kv.block_tables.ne[0],
            static_cast<const __nv_bfloat16*>(kv.k_pages.data),
            static_cast<const __half*>(kv.v_pages.data), query_heads, kv_heads, width, scale,
            split.split_keys, states, static_cast<__nv_bfloat16*>(out.data));
    }
    CUDA_CHECK(cudaGetLastError());
    if (states != nullptr) {
        qsa_attention_combine_kernel<<<dim3(blocks, query_heads / kv_heads), 256, 0, stream>>>(
            states, split.splits, query_heads, kv_heads, static_cast<__nv_bfloat16*>(out.data));
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace ninfer::ops::detail
