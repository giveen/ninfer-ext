#pragma once

// Implements: include/ninfer/ops/sliding_causal_attention.h,
// include/ninfer/ops/causal_compact_attention.h
//
// Tensor-core attention for Gemma 4's two attention Ops. Passes with many query rows take the tiled
// kernel at the end of this file; the kernel described here serves the rest. A block owns 16 query
// rows of one KV head: the rows are the (token, query head) pairs that read that KV head, so a key
// tile loaded once serves all of them. Keys stream in tiles of 32 with an online Softmax:
//
//   S = scale * Q K^T      four warps, eight keys each, BF16 MMA with FP32 accumulation
//   m, l, P                one warp per four rows, FP32
//   O = alpha O + P V      four warps, a quarter of the value columns each
//
// P is split into a BF16 head and a BF16 remainder and both are multiplied, so the weights carry
// about 16 significant bits rather than BF16's 8: the Op criterion is a little over one BF16 output
// ulp, which a single BF16 rounding of P can exceed on its own.
//
// Visibility is evaluated per element from the stored positions, so the sliding ring's order does
// not matter; a tile no row can see is skipped before its keys are loaded. When the caller states
// that keys are indexed by position (key k holds position k, as the global cache stores them), the
// scan also stops after the last key the block's latest query can see, so a cache handed over at
// its full capacity costs only what is filled.
//
// Few query rows (decode) give few blocks, so the keys may be split across `splits` blocks per row
// tile: each writes its unnormalized partial output with its running max and sum, and
// gemma_flash_attention_combine_kernel merges them.
//
// The compact global representation is handled by building a widened query: a stored row is
// [value (R) | rotated key dims (2A)], and with q' = [q with the rotated dims zeroed | q's rotated
// dims in stored order], q' . row is exactly the Op's score, while the value is the row's first R
// entries, so one tile serves QK^T and PV.

#include "ops/kv_cache/fp8_e4m3_row_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <climits>
#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr int kFlashRows    = 16;
inline constexpr int kFlashKeys    = 32;
inline constexpr int kFlashWarps   = 4;
inline constexpr int kFlashThreads = kFlashWarps * 32;
inline constexpr int kFlashPad     = 8; // BF16 elements of row padding against bank conflicts

struct FlashAttentionParams {
    const __nv_bfloat16* q; // [q_width, Hq, T, B]
    const __nv_bfloat16* k; // key row (kvh, key, b) at k + (kvh + Hkv * (key + S * b)) * k_stride
    const __nv_bfloat16* v; // value row likewise with v_stride; the compact rows serve as both
    // Row-scaled FP8 codes of the main compact key set, null when `k` holds BF16 rows. A staged
    // value is its E4M3 code times the row's FP16 scale; `k_stride` is then the code row's byte
    // stride and `k_scale` is indexed by the same (kvh, key, b) row.
    const std::uint8_t* k_code;
    const __half* k_scale;
    const std::int32_t* position_q; // [T, B]
    // The latest key position each query may see, [T, B]; null means its own position (causal).
    const std::int32_t* position_q_high;
    const std::int32_t* position_k; // [S, B]
    __nv_bfloat16* out;             // [V, Hq, T, B]
    std::int32_t query_heads;
    std::int32_t kv_heads;
    std::int32_t query_tokens;
    std::int32_t key_tokens;
    std::int32_t window; // 0: causal only
    std::int32_t q_width;
    std::int32_t rope_angles; // compact only
    std::int32_t k_stride;
    std::int32_t v_stride;
    float scale;
    bool indexed_keys;    // key k holds position k; keys past the latest query are invisible
    std::int32_t splits;  // key splits per row tile; 1 writes `out` directly
    float* partial_out;   // [splits, rows, V] unnormalized, when splits > 1
    float* partial_stats; // [splits, rows, 2] running max and sum
    // A second key set, scanned after the first with the same strides and visibility: a pass's own
    // keys beside a cache that does not hold them yet. extra_tokens == 0 means none.
    const __nv_bfloat16* extra_k;
    const __nv_bfloat16* extra_v;
    const std::int32_t* extra_position; // [extra_tokens, B]
    std::int32_t extra_tokens;
    // One extra exponent term per query head with no value, added unscaled after the score's scale.
    // Null means none.
    const __nv_bfloat16* sink; // [Hq]
};

// Partial rows are numbered (batch, kv head, row tile, row) so the combine needs no geometry beyond
// the row count.
__host__ __device__ inline std::int64_t
flash_partial_rows(std::int64_t row_tiles, std::int32_t kv_heads, std::int32_t batch) {
    return row_tiles * kFlashRows * kv_heads * batch;
}

template <int QK, int V, bool kCompact>
struct FlashLayout {
    static constexpr int kQStride    = QK + kFlashPad;
    static constexpr int kVStride    = V + kFlashPad;
    static constexpr int kPStride    = kFlashKeys + kFlashPad;
    static constexpr int kQBytes     = kFlashRows * kQStride * 2;
    static constexpr int kKBytes     = kFlashKeys * kQStride * 2;
    static constexpr int kVBytes     = kCompact ? 0 : kFlashKeys * kVStride * 2;
    static constexpr int kPBytes     = 2 * kFlashRows * kPStride * 2;
    static constexpr int kSBytes     = kFlashRows * (kFlashKeys + 1) * 4;
    static constexpr int kStatsBytes = 3 * kFlashRows * 4 + (2 * kFlashRows + kFlashKeys) * 4;
    static constexpr int kBytes = kQBytes + kKBytes + kVBytes + kPBytes + kSBytes + kStatsBytes;
};

__device__ __forceinline__ void flash_mma(float (&c)[4], const std::uint32_t (&a)[4],
                                          std::uint32_t b0, std::uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, "
                 "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ std::uint32_t flash_pair(const __nv_bfloat16* p) {
    return *reinterpret_cast<const std::uint32_t*>(p);
}

__device__ __forceinline__ std::uint32_t flash_pack(__nv_bfloat16 lo, __nv_bfloat16 hi) {
    return static_cast<std::uint32_t>(__bfloat16_as_ushort(lo)) |
           (static_cast<std::uint32_t>(__bfloat16_as_ushort(hi)) << 16);
}

__device__ __forceinline__ float flash_warp_max(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
    }
    return value;
}

__device__ __forceinline__ float flash_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

// A key is visible when it is no later than the query's upper bound and, with a window, less than a
// window behind the query itself.
__device__ __forceinline__ bool flash_visible(std::int32_t query, std::int32_t high,
                                              std::int32_t key, std::int32_t window) {
    return key <= high && (window == 0 || static_cast<std::int64_t>(query) - key < window);
}

// Copies `count` BF16 values (a multiple of 8) from global to shared in 16-byte units.
__device__ __forceinline__ void flash_copy_row(__nv_bfloat16* dst, const __nv_bfloat16* src,
                                               int count, int lane, int lanes) {
    const uint4* from = reinterpret_cast<const uint4*>(src);
    uint4* to         = reinterpret_cast<uint4*>(dst);
    for (int i = lane; i < count / 8; i += lanes) { to[i] = from[i]; }
}

// One 16-byte asynchronous copy from global to shared, or 16 zero bytes when `copy` is false (`src`
// is then not read).
__device__ __forceinline__ void flash_copy16(__nv_bfloat16* dst, const __nv_bfloat16* src,
                                             bool copy) {
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :
                 : "r"(address), "l"(src), "r"(copy ? 16 : 0));
}

// One 8-byte asynchronous copy from global to shared, or 8 zero bytes when `copy` is false. The
// coded tile stages its codes with this and converts them when the tile is consumed, so the global
// load keeps overlapping the previous tile's compute.
__device__ __forceinline__ void flash_copy8(std::uint8_t* dst, const std::uint8_t* src, bool copy) {
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n"
                 :
                 : "r"(address), "l"(src), "r"(copy ? 8 : 0));
}

// Eight E4M3 codes of one row as BF16, each code times the row scale rounded once: the coded
// counterpart of flash_copy16, which stores the same eight values from BF16 bytes.
__device__ __forceinline__ void flash_decode8(const std::uint8_t* src, __nv_bfloat16* dst,
                                              float scale) {
    const uint2 raw = *reinterpret_cast<const uint2*>(src);
    const auto pair = [&](std::uint32_t word, int half) {
        const auto bits    = static_cast<std::uint16_t>(word >> (16 * half));
        const float2 value = __half22float2(kv_cache_fp8_code2_to_half2(bits));
        return flash_pack(__float2bfloat16_rn(value.x * scale),
                          __float2bfloat16_rn(value.y * scale));
    };
    *reinterpret_cast<uint4*>(dst) =
        make_uint4(pair(raw.x, 0), pair(raw.x, 1), pair(raw.y, 0), pair(raw.y, 1));
}

template <int QK, int V, bool kCompact>
__launch_bounds__(kFlashThreads) __global__
    void gemma_flash_attention_kernel(FlashAttentionParams p) {
    using Layout = FlashLayout<QK, V, kCompact>;
    static_assert(QK % 16 == 0 && V % (kFlashWarps * 8) == 0, "MMA tiling");
    extern __shared__ __align__(16) unsigned char shared[];
    auto* qs = reinterpret_cast<__nv_bfloat16*>(shared);
    auto* ks = reinterpret_cast<__nv_bfloat16*>(shared + Layout::kQBytes);
    auto* vs = kCompact
                   ? ks
                   : reinterpret_cast<__nv_bfloat16*>(shared + Layout::kQBytes + Layout::kKBytes);
    auto* ph = reinterpret_cast<__nv_bfloat16*>(shared + Layout::kQBytes + Layout::kKBytes +
                                                Layout::kVBytes);
    auto* pl = ph + kFlashRows * Layout::kPStride;
    auto* ss = reinterpret_cast<float*>(shared + Layout::kQBytes + Layout::kKBytes +
                                        Layout::kVBytes + Layout::kPBytes);
    float* row_max      = ss + kFlashRows * (kFlashKeys + 1);
    float* row_sum      = row_max + kFlashRows;
    float* row_alpha    = row_sum + kFlashRows;
    auto* qpos          = reinterpret_cast<std::int32_t*>(row_alpha + kFlashRows);
    std::int32_t* qhigh = qpos + kFlashRows;
    std::int32_t* kpos  = qhigh + kFlashRows;
    const int vstride   = kCompact ? Layout::kQStride : Layout::kVStride;

    const int thread = static_cast<int>(threadIdx.x);
    const int warp   = thread >> 5;
    const int lane   = thread & 31;
    const int g      = lane >> 2;
    const int t      = lane & 3;

    const std::int32_t group = p.query_heads / p.kv_heads;
    const std::int32_t kvh   = static_cast<std::int32_t>(blockIdx.y);
    const std::int32_t b     = static_cast<std::int32_t>(blockIdx.z);
    const std::int32_t split = static_cast<std::int32_t>(blockIdx.x) % p.splits;
    const std::int64_t tile  = static_cast<std::int64_t>(blockIdx.x) / p.splits;
    const std::int64_t first = tile * kFlashRows;
    const std::int64_t rows  = static_cast<std::int64_t>(p.query_tokens) * group;

    // Queries: row r is the flat (token, head-in-group) pair first + r.
    for (int r = warp; r < kFlashRows; r += kFlashWarps) {
        const std::int64_t flat = first + r;
        __nv_bfloat16* dst      = qs + r * Layout::kQStride;
        if (flat >= rows) {
            for (int j = lane; j < QK; j += 32) dst[j] = __float2bfloat16_rn(0.0F);
            if (lane == 0) qpos[r] = qhigh[r] = INT_MIN;
            continue;
        }
        const std::int32_t token = static_cast<std::int32_t>(flat / group);
        const std::int32_t head  = kvh * group + static_cast<std::int32_t>(flat % group);
        const __nv_bfloat16* src =
            p.q + (static_cast<std::int64_t>(head) +
                   static_cast<std::int64_t>(p.query_heads) *
                       (token + static_cast<std::int64_t>(p.query_tokens) * b)) *
                      p.q_width;
        if constexpr (kCompact) {
            const int half = p.q_width / 2;
            const int a    = p.rope_angles;
            for (int j = lane; j < QK; j += 32) {
                __nv_bfloat16 value = __float2bfloat16_rn(0.0F);
                if (j < p.q_width) {
                    const bool rotated = j < a || (j >= half && j < half + a);
                    if (!rotated) value = src[j];
                } else {
                    const int index = j - p.q_width;
                    value           = src[index < a ? index : half + index - a];
                }
                dst[j] = value;
            }
        } else {
            flash_copy_row(dst, src, QK, lane, 32);
        }
        if (lane == 0) {
            const std::int64_t at = token + static_cast<std::int64_t>(p.query_tokens) * b;
            qpos[r]               = p.position_q[at];
            qhigh[r]              = p.position_q_high ? p.position_q_high[at] : qpos[r];
        }
    }
    if (thread < kFlashRows) {
        // A sink is one exponent term with no value: seeding the running max and sum with it is the
        // reference's `den = sum exp(w - m) + exp(sink - m)` exactly, and it costs one add. Row r
        // is the flat (token, head-in-group) pair, so its sink is that head's.
        const std::int64_t flat = first + thread;
        float sink              = -CUDART_INF_F;
        // Seeded only when this kernel is the whole reduction. With splits the partials are summed
        // by the merge, which folds the sink once; seeding here too would count it per split.
        if (p.sink != nullptr && p.splits == 1 && flat < rows) {
            sink = __bfloat162float(p.sink[kvh * group + static_cast<std::int32_t>(flat % group)]);
        }
        row_max[thread] = sink;
        row_sum[thread] = sink == -CUDART_INF_F ? 0.0F : 1.0F;
    }
    __syncthreads();
    std::int32_t query_low = INT_MAX, query_high = INT_MIN;
    for (int r = 0; r < kFlashRows; ++r) {
        if (qpos[r] == INT_MIN) continue;
        query_low  = min(query_low, qpos[r]);
        query_high = max(query_high, qhigh[r]);
    }

    constexpr int kColumns = V / kFlashWarps;
    constexpr int kTiles   = kColumns / 8;
    float o[kTiles][4];
#pragma unroll
    for (int n = 0; n < kTiles; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0F;

    std::int32_t key_end = p.key_tokens;
    if (p.indexed_keys) { key_end = query_high == INT_MIN ? 0 : min(p.key_tokens, query_high + 1); }
    const std::int32_t main_tiles = (key_end + kFlashKeys - 1) / kFlashKeys;
    const std::int32_t key_tiles  = main_tiles + (p.extra_tokens + kFlashKeys - 1) / kFlashKeys;
    const std::int32_t per_split  = (key_tiles + p.splits - 1) / p.splits;
    const std::int32_t tile_begin = split * per_split;
    const std::int32_t tile_end   = min(key_tiles, tile_begin + per_split);
    for (std::int32_t tile_index = tile_begin; tile_index < tile_end; ++tile_index) {
        // The tile's key set: the main keys, then the extra ones.
        const bool extra              = tile_index >= main_tiles;
        const std::int32_t base       = (extra ? tile_index - main_tiles : tile_index) * kFlashKeys;
        const std::int32_t keys       = extra ? p.extra_tokens : p.key_tokens;
        const __nv_bfloat16* key_rows = extra ? p.extra_k : p.k;
        const __nv_bfloat16* value_rows  = extra ? p.extra_v : p.v;
        const std::int32_t* key_position = extra ? p.extra_position : p.position_k;
        // Positions first: a tile no row can see is skipped before its keys move.
        bool seen = false;
        if (thread < kFlashKeys) {
            const std::int32_t key = base + thread;
            std::int32_t position  = INT_MAX;
            if (key < keys) position = key_position[key + static_cast<std::int64_t>(keys) * b];
            kpos[thread] = position;
            seen = key < keys && query_high != INT_MIN && position <= query_high &&
                   (p.window == 0 || static_cast<std::int64_t>(query_low) - position < p.window);
        }
        if (!__syncthreads_or(seen)) continue;

        // The whole tile is issued at once as 16-byte asynchronous copies, so its loads overlap
        // instead of costing a memory latency per row. A key no row can see is zero-filled rather
        // than loaded: an unwritten cache slot may hold any bits, and a masked NaN value would
        // still poison the P*V product.
        const auto copied = [&](int r) {
            const std::int32_t position = kpos[r];
            return base + r < keys && position <= query_high &&
                   (p.window == 0 || static_cast<std::int64_t>(query_low) - position < p.window);
        };
        const auto key_row = [&](int r) {
            return static_cast<std::int64_t>(kvh) +
                   static_cast<std::int64_t>(p.kv_heads) *
                       (base + r + static_cast<std::int64_t>(keys) * b);
        };
        constexpr int kKChunks = QK / 8;
        const bool coded       = p.k_code != nullptr && !extra;
        if (coded) {
            for (int at = thread; at < kFlashKeys * kKChunks; at += kFlashThreads) {
                const int r        = at / kKChunks;
                const int c        = at % kKChunks;
                __nv_bfloat16* dst = ks + r * Layout::kQStride + c * 8;
                if (copied(r)) {
                    const std::int64_t row = key_row(r);
                    flash_decode8(p.k_code + row * p.k_stride + c * 8, dst,
                                  __half2float(p.k_scale[row]));
                } else {
                    *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
                }
            }
        } else {
        for (int at = thread; at < kFlashKeys * kKChunks; at += kFlashThreads) {
            const int r     = at / kKChunks;
            const int c     = at % kKChunks;
            const bool copy = copied(r);
            flash_copy16(ks + r * Layout::kQStride + c * 8,
                         copy ? key_rows + key_row(r) * p.k_stride + c * 8 : key_rows, copy);
            }
        }
        if constexpr (!kCompact) {
            constexpr int kVChunks = V / 8;
            for (int at = thread; at < kFlashKeys * kVChunks; at += kFlashThreads) {
                const int r     = at / kVChunks;
                const int c     = at % kVChunks;
                const bool copy = copied(r);
                flash_copy16(vs + r * vstride + c * 8,
                             copy ? value_rows + key_row(r) * p.v_stride + c * 8 : value_rows,
                             copy);
            }
        }
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();

        // S: warp w scores keys [8w, 8w + 8) for all sixteen rows.
        {
            float acc[4]              = {0.0F, 0.0F, 0.0F, 0.0F};
            const __nv_bfloat16* krow = ks + (warp * 8 + g) * Layout::kQStride;
#pragma unroll 4
            for (int kk = 0; kk < QK; kk += 16) {
                const std::uint32_t a[4] = {
                    flash_pair(qs + g * Layout::kQStride + kk + 2 * t),
                    flash_pair(qs + (g + 8) * Layout::kQStride + kk + 2 * t),
                    flash_pair(qs + g * Layout::kQStride + kk + 2 * t + 8),
                    flash_pair(qs + (g + 8) * Layout::kQStride + kk + 2 * t + 8)};
                flash_mma(acc, a, flash_pair(krow + kk + 2 * t), flash_pair(krow + kk + 2 * t + 8));
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int r   = g + (i >= 2 ? 8 : 0);
                const int col = warp * 8 + 2 * t + (i & 1);
                const bool ok = qpos[r] != INT_MIN && base + col < keys &&
                                flash_visible(qpos[r], qhigh[r], kpos[col], p.window);
                ss[r * (kFlashKeys + 1) + col] = ok ? acc[i] * p.scale : -CUDART_INF_F;
            }
        }
        __syncthreads();

        // Online Softmax: warp w owns rows [4w, 4w + 4), a lane per key.
        for (int r = warp * 4; r < warp * 4 + 4; ++r) {
            const float s      = ss[r * (kFlashKeys + 1) + lane];
            const float old    = row_max[r];
            const float high   = fmaxf(old, flash_warp_max(s));
            const bool empty   = high == -CUDART_INF_F;
            const float weight = empty || s == -CUDART_INF_F ? 0.0F : __expf(s - high);
            const float alpha =
                empty || old == -CUDART_INF_F ? (empty ? 1.0F : 0.0F) : __expf(old - high);
            const float total               = flash_warp_sum(weight);
            const __nv_bfloat16 head        = __float2bfloat16_rn(weight);
            ph[r * Layout::kPStride + lane] = head;
            pl[r * Layout::kPStride + lane] = __float2bfloat16_rn(weight - __bfloat162float(head));
            __syncwarp();
            if (lane == 0) {
                row_sum[r]   = row_sum[r] * alpha + total;
                row_max[r]   = high;
                row_alpha[r] = alpha;
            }
        }
        __syncthreads();

        // O: warp w accumulates value columns [w * V/4, (w + 1) * V/4).
        {
            const float alpha_low  = row_alpha[g];
            const float alpha_high = row_alpha[g + 8];
#pragma unroll
            for (int n = 0; n < kTiles; ++n) {
                o[n][0] *= alpha_low;
                o[n][1] *= alpha_low;
                o[n][2] *= alpha_high;
                o[n][3] *= alpha_high;
            }
#pragma unroll
            for (int kk = 0; kk < kFlashKeys; kk += 16) {
                const std::uint32_t ah[4] = {
                    flash_pair(ph + g * Layout::kPStride + kk + 2 * t),
                    flash_pair(ph + (g + 8) * Layout::kPStride + kk + 2 * t),
                    flash_pair(ph + g * Layout::kPStride + kk + 2 * t + 8),
                    flash_pair(ph + (g + 8) * Layout::kPStride + kk + 2 * t + 8)};
                const std::uint32_t al[4] = {
                    flash_pair(pl + g * Layout::kPStride + kk + 2 * t),
                    flash_pair(pl + (g + 8) * Layout::kPStride + kk + 2 * t),
                    flash_pair(pl + g * Layout::kPStride + kk + 2 * t + 8),
                    flash_pair(pl + (g + 8) * Layout::kPStride + kk + 2 * t + 8)};
                const __nv_bfloat16* v0 = vs + (kk + 2 * t) * vstride;
#pragma unroll
                for (int n = 0; n < kTiles; ++n) {
                    const int col          = warp * kColumns + n * 8 + g;
                    const std::uint32_t b0 = flash_pack(v0[col], v0[vstride + col]);
                    const std::uint32_t b1 =
                        flash_pack(v0[8 * vstride + col], v0[9 * vstride + col]);
                    flash_mma(o[n], ah, b0, b1);
                    flash_mma(o[n], al, b0, b1);
                }
            }
        }
        __syncthreads();
    }

    if (p.splits > 1) {
        // Partial: the unnormalized output and this split's running max and sum, per row.
        const std::int64_t row_tiles = (rows + kFlashRows - 1) / kFlashRows;
        const std::int64_t total =
            flash_partial_rows(row_tiles, p.kv_heads, static_cast<std::int32_t>(gridDim.z));
        const std::int64_t base_row =
            static_cast<std::int64_t>(split) * total +
            ((static_cast<std::int64_t>(b) * p.kv_heads + kvh) * row_tiles + tile) * kFlashRows;
        // Padding rows are never combined, so they are not written.
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int r = g + 8 * half;
            if (first + r >= rows) continue;
            float* dst = p.partial_out + (base_row + r) * V;
#pragma unroll
            for (int n = 0; n < kTiles; ++n) {
                const int col = warp * kColumns + n * 8 + 2 * t;
                dst[col]      = o[n][2 * half];
                dst[col + 1]  = o[n][2 * half + 1];
            }
        }
        if (thread < kFlashRows && first + thread < rows) {
            p.partial_stats[(base_row + thread) * 2]     = row_max[thread];
            p.partial_stats[(base_row + thread) * 2 + 1] = row_sum[thread];
        }
        return;
    }

    // Normalize and store. A row that saw no key writes exact zero.
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int r             = g + 8 * half;
        const std::int64_t flat = first + r;
        if (flat >= rows) continue;
        const std::int32_t token = static_cast<std::int32_t>(flat / group);
        const std::int32_t head  = kvh * group + static_cast<std::int32_t>(flat % group);
        const float total        = row_sum[r];
        const float inverse      = total > 0.0F ? 1.0F / total : 0.0F;
        __nv_bfloat16* dst = p.out + (static_cast<std::int64_t>(head) +
                                      static_cast<std::int64_t>(p.query_heads) *
                                          (token + static_cast<std::int64_t>(p.query_tokens) * b)) *
                                         V;
#pragma unroll
        for (int n = 0; n < kTiles; ++n) {
            const int col = warp * kColumns + n * 8 + 2 * t;
            *reinterpret_cast<__nv_bfloat162*>(dst + col) =
                __floats2bfloat162_rn(o[n][2 * half] * inverse, o[n][2 * half + 1] * inverse);
        }
    }
}

// The most key splits a row tile takes; the combine's weights live in one warp's two lanes each.
inline constexpr int kFlashMaxSplits      = 64;
inline constexpr int kFlashCombineThreads = 256;

// Merges the splits of every partial row: out = sum_s O_s e^(m_s - M) / sum_s l_s e^(m_s - M).
// One block per partial row (every batch). One warp derives the splits' weights e^(m_s - M) once;
// then a thread owns four value columns of one group of splits, every group's partial sums meet in
// shared memory, and the first group writes the row. The splits are summed in groups rather than in
// order, which is a reassociation of FP32 additions.
template <int V>
__launch_bounds__(kFlashCombineThreads) __global__
    void gemma_flash_attention_combine_kernel(FlashAttentionParams p, std::int64_t row_tiles) {
    static_assert(V % 4 == 0 && V / 4 <= kFlashCombineThreads, "four columns a thread");
    constexpr int kColumnThreads = V / 4;
    constexpr int kGroups        = kFlashCombineThreads / kColumnThreads;
    __shared__ float weight[kFlashMaxSplits];
    __shared__ float inverse_sum;
    __shared__ float4 partial[kGroups > 1 ? kGroups - 1 : 1][kColumnThreads];

    const std::int64_t row   = blockIdx.x;
    const std::int64_t total = gridDim.x;
    const std::int32_t group = p.query_heads / p.kv_heads;
    const std::int64_t r     = row % kFlashRows;
    const std::int64_t tile  = (row / kFlashRows) % row_tiles;
    const std::int32_t kvh = static_cast<std::int32_t>((row / kFlashRows / row_tiles) % p.kv_heads);
    const std::int32_t b   = static_cast<std::int32_t>(row / kFlashRows / row_tiles / p.kv_heads);
    const std::int64_t flat = tile * kFlashRows + r;
    if (flat >= static_cast<std::int64_t>(p.query_tokens) * group) return;
    const int thread = static_cast<int>(threadIdx.x);

    if (thread < 32) {
        float m[2], l[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int s = thread + 32 * i;
            m[i]        = -CUDART_INF_F;
            l[i]        = 0.0F;
            if (s < p.splits) {
                m[i] = p.partial_stats[(s * total + row) * 2];
                l[i] = p.partial_stats[(s * total + row) * 2 + 1];
            }
        }
        // The sink is one term belonging to the row rather than to any split, so it raises the
        // merge's maximum before the split weights are formed and enters the denominator once.
        const float sink_value =
            p.sink != nullptr
                ? __bfloat162float(p.sink[kvh * group + static_cast<std::int32_t>(flat % group)])
                : -CUDART_INF_F;
        const float high = flash_warp_max(fmaxf(fmaxf(m[0], m[1]), sink_value));
        float sum        = 0.0F;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const float w = m[i] == -CUDART_INF_F ? 0.0F : __expf(m[i] - high);
            if (thread + 32 * i < p.splits) weight[thread + 32 * i] = w;
            sum += l[i] * w;
        }
        if (thread == 0 && sink_value != -CUDART_INF_F) { sum += __expf(sink_value - high); }
        sum = flash_warp_sum(sum);
        if (thread == 0) inverse_sum = sum > 0.0F ? 1.0F / sum : 0.0F;
    }
    __syncthreads();

    const int column = thread % kColumnThreads;
    const int split0 = thread / kColumnThreads;
    float4 value     = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
#pragma unroll 4
    for (int s = split0; s < p.splits; s += kGroups) {
        const float w = weight[s];
        if (w == 0.0F) continue;
        const float4 o =
            reinterpret_cast<const float4*>(p.partial_out + (s * total + row) * V)[column];
        value.x += o.x * w;
        value.y += o.y * w;
        value.z += o.z * w;
        value.w += o.w * w;
    }
    if constexpr (kGroups > 1) {
        if (split0 > 0) partial[split0 - 1][column] = value;
        __syncthreads();
        if (split0 > 0) return;
#pragma unroll
        for (int g = 0; g < kGroups - 1; ++g) {
            const float4 o = partial[g][column];
            value.x += o.x;
            value.y += o.y;
            value.z += o.z;
            value.w += o.w;
        }
    }
    const std::int32_t token = static_cast<std::int32_t>(flat / group);
    const std::int32_t head  = kvh * group + static_cast<std::int32_t>(flat % group);
    __nv_bfloat16* dst = p.out + (static_cast<std::int64_t>(head) +
                                  static_cast<std::int64_t>(p.query_heads) *
                                      (token + static_cast<std::int64_t>(p.query_tokens) * b)) *
                                     V;
    const float scale  = inverse_sum;
    *reinterpret_cast<__nv_bfloat162*>(dst + 4 * column) =
        __floats2bfloat162_rn(value.x * scale, value.y * scale);
    *reinterpret_cast<__nv_bfloat162*>(dst + 4 * column + 2) =
        __floats2bfloat162_rn(value.z * scale, value.w * scale);
}

// ---------------------------------------------------------------------------------------------
// Tiled route for passes with many query rows (prefill and scoring). The arithmetic is the one
// above - BF16 Q K^T with FP32 accumulation, an online Softmax in FP32, and P V with P split into a
// BF16 head and remainder - but organized for reuse and overlap instead of for few rows:
//
//   * a warp owns 16 query rows for the whole scan, so S, the running max/sum and P stay in its
//     registers and a key tile costs two block barriers rather than four;
//   * a block holds 64 rows (32 when the value is 512 wide), so a key tile loaded once serves four
//     times the rows of the decode kernel;
//   * key tiles stream through two cp.async stages, the next tile's copy overlapping this tile's
//     math, into XOR-swizzled rows read with ldmatrix.
//
// With a 512-wide value one warp cannot hold a 16 x 512 FP32 accumulator, so a row group is two
// warps: each scores the 16 rows over half of the QK dims, the halves are exchanged through shared
// memory and summed (a + b and b + a are the same float, so both warps hold the same S), and each
// warp then accumulates half of the value columns from that common P.
//
// Visibility is still evaluated per element from the stored positions. A key tile no row of the
// block can see is skipped, and a key the block cannot see is zero-filled instead of copied, for
// the same reason as above: an unwritten cache slot may hold NaN.

template <int Cols>
struct FlashTile {
    static constexpr int kChunks   = Cols / 8;
    static constexpr bool kSwizzle = kChunks % 8 == 0;
    static constexpr int kStride   = kSwizzle ? Cols : Cols + kFlashPad;

    // Element offset of the 16-byte chunk holding column `col` (a multiple of 8) of row `row`.
    __device__ static __forceinline__ int offset(int row, int col) {
        const int chunk = col >> 3;
        return row * kStride + ((kSwizzle ? (chunk ^ (row & 7)) : chunk) << 3);
    }
};

template <int QK, int V, bool kCompact>
struct FlashTiledConfig {
    static constexpr int kColWarps   = V > 256 ? 2 : 1;
    static constexpr int kRowWarps   = kColWarps == 2 ? 2 : 4;
    static constexpr int kWarps      = kRowWarps * kColWarps;
    static constexpr int kThreads    = kWarps * 32;
    static constexpr int kRows       = 16 * kRowWarps;
    static constexpr int kKeys       = QK >= 512 ? 16 : 32;
    static constexpr int kQKPerWarp  = QK / kColWarps;
    static constexpr int kVPerWarp   = V / kColWarps;
    static constexpr int kKeyTiles   = kKeys / 8;
    static constexpr int kValueTiles = kVPerWarp / 8;
    static_assert(kQKPerWarp % 16 == 0 && kVPerWarp % 16 == 0, "MMA tiling");
    static_assert(!kCompact || V <= QK, "a compact row carries its value first");
    using QTile                      = FlashTile<QK>;
    using KTile                      = FlashTile<QK>;
    using VTile                      = FlashTile<kCompact ? QK : V>;
    static constexpr int kQBytes     = kRows * QTile::kStride * 2;
    static constexpr int kKBytes     = kKeys * KTile::kStride * 2;
    static constexpr int kVBytes     = kCompact ? 0 : kKeys * VTile::kStride * 2;
    static constexpr int kStageBytes = kKBytes + kVBytes;
    static constexpr int kExchangeBytes =
        kColWarps == 2 ? kRowWarps * kColWarps * kKeyTiles * 4 * 32 * 4 : 0;
    static constexpr int kKposBytes = 3 * kKeys * 4;
    static constexpr int kQposBytes = 2 * kRows * 4;
    static constexpr int kBytes =
        kQBytes + 2 * kStageBytes + kExchangeBytes + kKposBytes + kQposBytes;
};

__device__ __forceinline__ void flash_ldsm_x4(std::uint32_t (&r)[4], const __nv_bfloat16* p) {
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(address));
}

__device__ __forceinline__ void flash_ldsm_x4_trans(std::uint32_t (&r)[4], const __nv_bfloat16* p) {
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(address));
}

__device__ __forceinline__ std::uint32_t flash_pack_float(float lo, float hi) {
    return flash_pack(__float2bfloat16_rn(lo), __float2bfloat16_rn(hi));
}

template <int QK, int V, bool kCompact>
__launch_bounds__(FlashTiledConfig<QK, V, kCompact>::kThreads, 1) __global__
    void gemma_flash_attention_tiled_kernel(FlashAttentionParams p) {
    using C     = FlashTiledConfig<QK, V, kCompact>;
    using QTile = typename C::QTile;
    using KTile = typename C::KTile;
    using VTile = typename C::VTile;
    // A coded tile stages its raw E4M3 rows in two alternated buffers and converts the consumed
    // stage into one BF16 tile. The three buffers fit inside the two BF16 stages the layout
    // reserves.
    constexpr int kCodeStage = kCompact ? C::kKeys * QK : 0;
    static_assert(!kCompact || 2 * kCodeStage + C::kKBytes <= 2 * C::kStageBytes,
                  "coded stages must fit the tiled kernel's stage region");
    extern __shared__ __align__(16) unsigned char shared[];
    auto* qs       = reinterpret_cast<__nv_bfloat16*>(shared);
    auto* stages   = shared + C::kQBytes;
    auto* exchange = reinterpret_cast<float*>(stages + 2 * C::kStageBytes);
    auto* kpos = reinterpret_cast<std::int32_t*>(stages + 2 * C::kStageBytes + C::kExchangeBytes);
    std::int32_t* qpos  = kpos + 3 * C::kKeys;
    std::int32_t* qhigh = qpos + C::kRows;

    const int thread = static_cast<int>(threadIdx.x);
    const int warp   = thread >> 5;
    const int lane   = thread & 31;
    const int g      = lane >> 2;
    const int t      = lane & 3;
    const int rg     = warp / C::kColWarps;
    const int cw     = warp % C::kColWarps;

    const std::int32_t group = p.query_heads / p.kv_heads;
    const std::int32_t kvh   = static_cast<std::int32_t>(blockIdx.y);
    const std::int32_t b     = static_cast<std::int32_t>(blockIdx.z);
    // Later row tiles see more keys; they are scheduled first.
    const std::int64_t tile  = static_cast<std::int64_t>(gridDim.x) - 1 - blockIdx.x;
    const std::int64_t first = tile * C::kRows;
    const std::int64_t rows  = static_cast<std::int64_t>(p.query_tokens) * group;

    // Queries: row r is the flat (token, head-in-group) pair first + r, built a 16-byte chunk at a
    // time. A compact query is widened as in the decode kernel.
    constexpr int kQChunks = QK / 8;
    for (int index = thread; index < C::kRows * kQChunks; index += C::kThreads) {
        const int r             = index / kQChunks;
        const int c             = index % kQChunks;
        const std::int64_t flat = first + r;
        __nv_bfloat16* dst      = qs + QTile::offset(r, c * 8);
        if (flat >= rows) {
            *reinterpret_cast<uint4*>(dst) = uint4{0u, 0u, 0u, 0u};
            continue;
        }
        const std::int32_t token = static_cast<std::int32_t>(flat / group);
        const std::int32_t head  = kvh * group + static_cast<std::int32_t>(flat % group);
        const __nv_bfloat16* src =
            p.q + (static_cast<std::int64_t>(head) +
                   static_cast<std::int64_t>(p.query_heads) *
                       (token + static_cast<std::int64_t>(p.query_tokens) * b)) *
                      p.q_width;
        if constexpr (kCompact) {
            const int half = p.q_width / 2;
            const int a    = p.rope_angles;
            __align__(16) __nv_bfloat16 values[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int j         = c * 8 + e;
                __nv_bfloat16 value = __float2bfloat16_rn(0.0F);
                if (j < p.q_width) {
                    const bool rotated = j < a || (j >= half && j < half + a);
                    if (!rotated) value = src[j];
                } else {
                    const int at = j - p.q_width;
                    value        = src[at < a ? at : half + at - a];
                }
                values[e] = value;
            }
            *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(values);
        } else {
            *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src + c * 8);
        }
    }
    for (int r = thread; r < C::kRows; r += C::kThreads) {
        const std::int64_t flat = first + r;
        if (flat >= rows) {
            qpos[r] = qhigh[r] = INT_MIN;
            continue;
        }
        const std::int64_t at = flat / group + static_cast<std::int64_t>(p.query_tokens) * b;
        qpos[r]               = p.position_q[at];
        qhigh[r]              = p.position_q_high ? p.position_q_high[at] : qpos[r];
    }
    __syncthreads();
    std::int32_t query_low = INT_MAX, query_high = INT_MIN;
    for (int r = 0; r < C::kRows; ++r) {
        if (qpos[r] == INT_MIN) continue;
        query_low  = min(query_low, qpos[r]);
        query_high = max(query_high, qhigh[r]);
    }
    const int row0         = rg * 16 + g;
    const std::int32_t qp0 = qpos[row0];
    const std::int32_t qh0 = qhigh[row0];
    const std::int32_t qp1 = qpos[row0 + 8];
    const std::int32_t qh1 = qhigh[row0 + 8];
    // A key the block cannot see is neither scored by any row nor copied.
    const auto block_sees = [&](std::int32_t position) {
        return query_high != INT_MIN && position <= query_high &&
               (p.window == 0 || static_cast<std::int64_t>(query_low) - position < p.window);
    };

    std::int32_t key_end = p.key_tokens;
    if (p.indexed_keys) key_end = query_high == INT_MIN ? 0 : min(p.key_tokens, query_high + 1);
    const std::int32_t main_tiles = (key_end + C::kKeys - 1) / C::kKeys;
    const std::int32_t tiles      = main_tiles + (p.extra_tokens + C::kKeys - 1) / C::kKeys;

    struct KeyTile {
        const __nv_bfloat16* k;
        const __nv_bfloat16* v;
        const std::int32_t* position;
        std::int32_t keys;
        std::int32_t base;
    };

    const auto key_tile = [&](std::int32_t index) {
        const bool extra = index >= main_tiles;
        return KeyTile{extra ? p.extra_k : p.k, extra ? p.extra_v : p.v,
                       extra ? p.extra_position : p.position_k,
                       extra ? p.extra_tokens : p.key_tokens,
                       (extra ? index - main_tiles : index) * C::kKeys};
    };
    // The first tile at or after `index` the block can see, its key positions left in `slot`.
    const auto next_visible = [&](std::int32_t index, int slot) {
        for (; index < tiles; ++index) {
            const KeyTile kt = key_tile(index);
            bool seen        = false;
            if (thread < C::kKeys) {
                const std::int32_t key = kt.base + thread;
                std::int32_t position  = INT_MAX;
                if (key < kt.keys)
                    position = kt.position[key + static_cast<std::int64_t>(kt.keys) * b];
                kpos[slot * C::kKeys + thread] = position;
                seen                           = key < kt.keys && block_sees(position);
            }
            if (__syncthreads_or(seen)) return index;
        }
        return tiles;
    };
    const auto issue = [&](std::int32_t index, int stage, int slot) {
        const KeyTile kt       = key_tile(index);
        auto* ks               = reinterpret_cast<__nv_bfloat16*>(stages + stage * C::kStageBytes);
        constexpr int kKChunks = QK / 8;
        const bool coded       = kCompact && p.k_code != nullptr && index < main_tiles;
        auto* codes_dst        = reinterpret_cast<std::uint8_t*>(stages + stage * kCodeStage);
        for (int at = thread; at < C::kKeys * kKChunks; at += C::kThreads) {
            const int r            = at / kKChunks;
            const int c            = at % kKChunks;
            const std::int32_t key = kt.base + r;
            const bool copy        = key < kt.keys && block_sees(kpos[slot * C::kKeys + r]);
            const std::int64_t row =
                static_cast<std::int64_t>(kvh) + static_cast<std::int64_t>(p.kv_heads) *
                                                     (key + static_cast<std::int64_t>(kt.keys) * b);
            if (coded) {
                flash_copy8(codes_dst + r * QK + c * 8,
                            copy ? p.k_code + row * p.k_stride + c * 8 : p.k_code, copy);
            } else {
                flash_copy16(ks + KTile::offset(r, c * 8),
                             copy ? kt.k + row * p.k_stride + c * 8 : kt.k, copy);
            }
        }
        if constexpr (!kCompact) {
            auto* vs =
                reinterpret_cast<__nv_bfloat16*>(stages + stage * C::kStageBytes + C::kKBytes);
            constexpr int kVChunks = V / 8;
            for (int at = thread; at < C::kKeys * kVChunks; at += C::kThreads) {
                const int r            = at / kVChunks;
                const int c            = at % kVChunks;
                const std::int32_t key = kt.base + r;
                const bool copy        = key < kt.keys && block_sees(kpos[slot * C::kKeys + r]);
                const std::int64_t row = static_cast<std::int64_t>(kvh) +
                                         static_cast<std::int64_t>(p.kv_heads) *
                                             (key + static_cast<std::int64_t>(kt.keys) * b);
                flash_copy16(vs + VTile::offset(r, c * 8),
                             copy ? kt.v + row * p.v_stride + c * 8 : kt.v, copy);
            }
        }
    };

    float o[C::kValueTiles][4];
#pragma unroll
    for (int n = 0; n < C::kValueTiles; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0F;
    float max0 = -CUDART_INF_F, max1 = -CUDART_INF_F;
    float sum0 = 0.0F, sum1 = 0.0F; // this thread's columns only, reduced over the quad at the end

    int processed    = 0;
    std::int32_t cur = next_visible(0, 0);
    if (cur < tiles) issue(cur, 0, 0);
    asm volatile("cp.async.commit_group;\n");
    while (cur < tiles) {
        const int next_slot   = (processed + 1) % 3;
        const std::int32_t nx = next_visible(cur + 1, next_slot);
        if (nx < tiles) issue(nx, (processed + 1) & 1, next_slot);
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_group 1;\n");
        __syncthreads();

        const int stage = processed & 1;
        const int slot  = processed % 3;
        const __nv_bfloat16* ks =
            reinterpret_cast<const __nv_bfloat16*>(stages + stage * C::kStageBytes);
        if constexpr (kCompact) {
            if (p.k_code != nullptr) {
                // Convert the consumed stage's codes into the one BF16 tile. The next stage's
                // codes are already in flight, so the global load keeps overlapping this tile's
                // compute; only the conversion is new work.
                const KeyTile kt = key_tile(cur);
                auto* bf         = reinterpret_cast<__nv_bfloat16*>(stages + 2 * kCodeStage);
                const auto* cs = reinterpret_cast<const std::uint8_t*>(stages + stage * kCodeStage);
                constexpr int kKChunks = QK / 8;
                for (int at = thread; at < C::kKeys * kKChunks; at += C::kThreads) {
                    const int r            = at / kKChunks;
                    const int c            = at % kKChunks;
                    const std::int32_t key = kt.base + r;
                    const std::int64_t row = static_cast<std::int64_t>(kvh) +
                                             static_cast<std::int64_t>(p.kv_heads) *
                                                 (key + static_cast<std::int64_t>(kt.keys) * b);
                    const float scale      = key < kt.keys ? __half2float(p.k_scale[row]) : 0.0F;
                    flash_decode8(cs + r * QK + c * 8, bf + KTile::offset(r, c * 8), scale);
                }
                __syncthreads();
                ks = bf;
            }
        }
        const auto* vs  = kCompact ? ks
                                   : reinterpret_cast<const __nv_bfloat16*>(
                                         stages + stage * C::kStageBytes + C::kKBytes);
        const std::int32_t* tile_pos = kpos + slot * C::kKeys;

        // S over this warp's share of the QK dims.
        float s[C::kKeyTiles][4];
#pragma unroll
        for (int n = 0; n < C::kKeyTiles; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0F;
#pragma unroll 4
        for (int kk = cw * C::kQKPerWarp; kk < (cw + 1) * C::kQKPerWarp; kk += 16) {
            std::uint32_t a[4];
            flash_ldsm_x4(a, qs + QTile::offset(rg * 16 + (lane & 7) + ((lane >> 3) & 1) * 8,
                                                kk + (lane >> 4) * 8));
#pragma unroll
            for (int np = 0; np < C::kKeyTiles / 2; ++np) {
                std::uint32_t bk[4];
                flash_ldsm_x4(bk, ks + KTile::offset(np * 16 + (lane & 7) + (lane >> 4) * 8,
                                                     kk + ((lane >> 3) & 1) * 8));
                flash_mma(s[2 * np], a, bk[0], bk[1]);
                flash_mma(s[2 * np + 1], a, bk[2], bk[3]);
            }
        }
        if constexpr (C::kColWarps == 2) {
            float* mine    = exchange + (rg * 2 + cw) * C::kKeyTiles * 4 * 32;
            float* partner = exchange + (rg * 2 + (cw ^ 1)) * C::kKeyTiles * 4 * 32;
#pragma unroll
            for (int n = 0; n < C::kKeyTiles; ++n)
#pragma unroll
                for (int e = 0; e < 4; ++e) mine[(n * 4 + e) * 32 + lane] = s[n][e];
            asm volatile("bar.sync %0, 64;\n" : : "r"(1 + rg) : "memory");
#pragma unroll
            for (int n = 0; n < C::kKeyTiles; ++n)
#pragma unroll
                for (int e = 0; e < 4; ++e) s[n][e] += partner[(n * 4 + e) * 32 + lane];
        }

        // Scale and mask, then the online Softmax over the rows g and g + 8.
        float tile_max0 = -CUDART_INF_F, tile_max1 = -CUDART_INF_F;
#pragma unroll
        for (int n = 0; n < C::kKeyTiles; ++n) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const std::int32_t key_position = tile_pos[n * 8 + 2 * t + (e & 1)];
                const bool upper                = e >= 2;
                const std::int32_t qp           = upper ? qp1 : qp0;
                const bool ok =
                    qp != INT_MIN && flash_visible(qp, upper ? qh1 : qh0, key_position, p.window);
                s[n][e] = ok ? s[n][e] * p.scale : -CUDART_INF_F;
            }
            tile_max0 = fmaxf(tile_max0, fmaxf(s[n][0], s[n][1]));
            tile_max1 = fmaxf(tile_max1, fmaxf(s[n][2], s[n][3]));
        }
#pragma unroll
        for (int offset = 1; offset < 4; offset <<= 1) {
            tile_max0 = fmaxf(tile_max0, __shfl_xor_sync(0xffffffffu, tile_max0, offset));
            tile_max1 = fmaxf(tile_max1, __shfl_xor_sync(0xffffffffu, tile_max1, offset));
        }
        const float high0  = fmaxf(max0, tile_max0);
        const float high1  = fmaxf(max1, tile_max1);
        const float alpha0 = high0 == -CUDART_INF_F  ? 1.0F
                             : max0 == -CUDART_INF_F ? 0.0F
                                                     : __expf(max0 - high0);
        const float alpha1 = high1 == -CUDART_INF_F  ? 1.0F
                             : max1 == -CUDART_INF_F ? 0.0F
                                                     : __expf(max1 - high1);
        max0               = high0;
        max1               = high1;
        float part0 = 0.0F, part1 = 0.0F;
#pragma unroll
        for (int n = 0; n < C::kKeyTiles; ++n) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float high = e >= 2 ? high1 : high0;
                s[n][e]          = s[n][e] == -CUDART_INF_F ? 0.0F : __expf(s[n][e] - high);
            }
            part0 += s[n][0] + s[n][1];
            part1 += s[n][2] + s[n][3];
        }
        sum0 = sum0 * alpha0 + part0;
        sum1 = sum1 * alpha1 + part1;
#pragma unroll
        for (int n = 0; n < C::kValueTiles; ++n) {
            o[n][0] *= alpha0;
            o[n][1] *= alpha0;
            o[n][2] *= alpha1;
            o[n][3] *= alpha1;
        }

        // O += P V over this warp's value columns, P as a BF16 head plus remainder.
#pragma unroll
        for (int ks16 = 0; ks16 < C::kKeys / 16; ++ks16) {
            // Keys [16 ks16, 16 ks16 + 16) as an A fragment: the first key tile's accumulator
            // gives registers 0 and 1, the second's 2 and 3.
            std::uint32_t ah[4], al[4];
            const float w[8] = {s[2 * ks16][0],     s[2 * ks16][1],     s[2 * ks16][2],
                                s[2 * ks16][3],     s[2 * ks16 + 1][0], s[2 * ks16 + 1][1],
                                s[2 * ks16 + 1][2], s[2 * ks16 + 1][3]};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const __nv_bfloat16 h0 = __float2bfloat16_rn(w[2 * i]);
                const __nv_bfloat16 h1 = __float2bfloat16_rn(w[2 * i + 1]);
                ah[i]                  = flash_pack(h0, h1);
                al[i]                  = flash_pack_float(w[2 * i] - __bfloat162float(h0),
                                                          w[2 * i + 1] - __bfloat162float(h1));
            }
#pragma unroll
            for (int vp = 0; vp < C::kValueTiles / 2; ++vp) {
                std::uint32_t bv[4];
                flash_ldsm_x4_trans(
                    bv, vs + VTile::offset(ks16 * 16 + (lane & 7) + ((lane >> 3) & 1) * 8,
                                           cw * C::kVPerWarp + vp * 16 + (lane >> 4) * 8));
                flash_mma(o[2 * vp], ah, bv[0], bv[1]);
                flash_mma(o[2 * vp], al, bv[0], bv[1]);
                flash_mma(o[2 * vp + 1], ah, bv[2], bv[3]);
                flash_mma(o[2 * vp + 1], al, bv[2], bv[3]);
            }
        }
        cur = nx;
        ++processed;
    }
    asm volatile("cp.async.wait_group 0;\n");

    // Normalize and store. A row that saw no key writes exact zero.
#pragma unroll
    for (int offset = 1; offset < 4; offset <<= 1) {
        sum0 += __shfl_xor_sync(0xffffffffu, sum0, offset);
        sum1 += __shfl_xor_sync(0xffffffffu, sum1, offset);
    }
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const std::int64_t flat = first + row0 + 8 * half;
        if (flat >= rows) continue;
        const std::int32_t token = static_cast<std::int32_t>(flat / group);
        const std::int32_t head  = kvh * group + static_cast<std::int32_t>(flat % group);
        const float total        = half == 0 ? sum0 : sum1;
        const float inverse      = total > 0.0F ? 1.0F / total : 0.0F;
        __nv_bfloat16* dst = p.out + (static_cast<std::int64_t>(head) +
                                      static_cast<std::int64_t>(p.query_heads) *
                                          (token + static_cast<std::int64_t>(p.query_tokens) * b)) *
                                         V;
#pragma unroll
        for (int n = 0; n < C::kValueTiles; ++n) {
            const int col = cw * C::kVPerWarp + n * 8 + 2 * t;
            *reinterpret_cast<__nv_bfloat162*>(dst + col) =
                __floats2bfloat162_rn(o[n][2 * half] * inverse, o[n][2 * half + 1] * inverse);
        }
    }
}

} // namespace ninfer::ops::detail
