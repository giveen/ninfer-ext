#pragma once

// Implements: include/ninfer/ops/sliding_causal_attention.h, include/ninfer/ops/causal_compact_attention.h
//
// Tensor-core attention for Gemma 4's two attention Ops. A block owns 16 query rows of one KV head:
// the rows are the (token, query head) pairs that read that KV head, so a key tile loaded once serves
// all of them. Keys stream in tiles of 32 with an online Softmax:
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

#include <cuda_bf16.h>
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
    const __nv_bfloat16* q;   // [q_width, Hq, T, B]
    const __nv_bfloat16* k;   // key row (kvh, key, b) at k + (kvh + Hkv * (key + S * b)) * k_stride
    const __nv_bfloat16* v;   // value row likewise with v_stride; the compact rows serve as both
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
    bool indexed_keys;      // key k holds position k; keys past the latest query are invisible
    std::int32_t splits;    // key splits per row tile; 1 writes `out` directly
    float* partial_out;     // [splits, rows, V] unnormalized, when splits > 1
    float* partial_stats;   // [splits, rows, 2] running max and sum
    // A second key set, scanned after the first with the same strides and visibility: a pass's own
    // keys beside a cache that does not hold them yet. extra_tokens == 0 means none.
    const __nv_bfloat16* extra_k;
    const __nv_bfloat16* extra_v;
    const std::int32_t* extra_position; // [extra_tokens, B]
    std::int32_t extra_tokens;
};

// Partial rows are numbered (batch, kv head, row tile, row) so the combine needs no geometry beyond
// the row count.
__host__ __device__ inline std::int64_t flash_partial_rows(std::int64_t row_tiles, std::int32_t kv_heads,
                                                           std::int32_t batch) {
    return row_tiles * kFlashRows * kv_heads * batch;
}

template <int QK, int V, bool kCompact> struct FlashLayout {
    static constexpr int kQStride = QK + kFlashPad;
    static constexpr int kVStride = V + kFlashPad;
    static constexpr int kPStride = kFlashKeys + kFlashPad;
    static constexpr int kQBytes  = kFlashRows * kQStride * 2;
    static constexpr int kKBytes  = kFlashKeys * kQStride * 2;
    static constexpr int kVBytes  = kCompact ? 0 : kFlashKeys * kVStride * 2;
    static constexpr int kPBytes  = 2 * kFlashRows * kPStride * 2;
    static constexpr int kSBytes  = kFlashRows * (kFlashKeys + 1) * 4;
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
__device__ __forceinline__ bool flash_visible(std::int32_t query, std::int32_t high, std::int32_t key,
                                              std::int32_t window) {
    return key <= high && (window == 0 || static_cast<std::int64_t>(query) - key < window);
}

// Copies `count` BF16 values (a multiple of 8) from global to shared in 16-byte units.
__device__ __forceinline__ void flash_copy_row(__nv_bfloat16* dst, const __nv_bfloat16* src,
                                               int count, int lane, int lanes) {
    const uint4* from = reinterpret_cast<const uint4*>(src);
    uint4* to         = reinterpret_cast<uint4*>(dst);
    for (int i = lane; i < count / 8; i += lanes) { to[i] = from[i]; }
}

template <int QK, int V, bool kCompact>
__launch_bounds__(kFlashThreads) __global__ void gemma_flash_attention_kernel(FlashAttentionParams p) {
    using Layout = FlashLayout<QK, V, kCompact>;
    static_assert(QK % 16 == 0 && V % (kFlashWarps * 8) == 0, "MMA tiling");
    extern __shared__ __align__(16) unsigned char shared[];
    auto* qs = reinterpret_cast<__nv_bfloat16*>(shared);
    auto* ks = reinterpret_cast<__nv_bfloat16*>(shared + Layout::kQBytes);
    auto* vs = kCompact ? ks : reinterpret_cast<__nv_bfloat16*>(shared + Layout::kQBytes + Layout::kKBytes);
    auto* ph = reinterpret_cast<__nv_bfloat16*>(shared + Layout::kQBytes + Layout::kKBytes + Layout::kVBytes);
    auto* pl = ph + kFlashRows * Layout::kPStride;
    auto* ss = reinterpret_cast<float*>(shared + Layout::kQBytes + Layout::kKBytes + Layout::kVBytes +
                                        Layout::kPBytes);
    float* row_max   = ss + kFlashRows * (kFlashKeys + 1);
    float* row_sum   = row_max + kFlashRows;
    float* row_alpha = row_sum + kFlashRows;
    auto* qpos       = reinterpret_cast<std::int32_t*>(row_alpha + kFlashRows);
    std::int32_t* qhigh = qpos + kFlashRows;
    std::int32_t* kpos  = qhigh + kFlashRows;
    const int vstride  = kCompact ? Layout::kQStride : Layout::kVStride;

    const int thread = static_cast<int>(threadIdx.x);
    const int warp   = thread >> 5;
    const int lane   = thread & 31;
    const int g      = lane >> 2;
    const int t      = lane & 3;

    const std::int32_t group  = p.query_heads / p.kv_heads;
    const std::int32_t kvh    = static_cast<std::int32_t>(blockIdx.y);
    const std::int32_t b      = static_cast<std::int32_t>(blockIdx.z);
    const std::int32_t split  = static_cast<std::int32_t>(blockIdx.x) % p.splits;
    const std::int64_t tile   = static_cast<std::int64_t>(blockIdx.x) / p.splits;
    const std::int64_t first  = tile * kFlashRows;
    const std::int64_t rows   = static_cast<std::int64_t>(p.query_tokens) * group;

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
        row_max[thread] = -CUDART_INF_F;
        row_sum[thread] = 0.0F;
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
    if (p.indexed_keys) {
        key_end = query_high == INT_MIN ? 0 : min(p.key_tokens, query_high + 1);
    }
    const std::int32_t main_tiles = (key_end + kFlashKeys - 1) / kFlashKeys;
    const std::int32_t key_tiles  = main_tiles + (p.extra_tokens + kFlashKeys - 1) / kFlashKeys;
    const std::int32_t per_split  = (key_tiles + p.splits - 1) / p.splits;
    const std::int32_t tile_begin = split * per_split;
    const std::int32_t tile_end   = min(key_tiles, tile_begin + per_split);
    for (std::int32_t tile_index = tile_begin; tile_index < tile_end; ++tile_index) {
        // The tile's key set: the main keys, then the extra ones.
        const bool extra                 = tile_index >= main_tiles;
        const std::int32_t base          = (extra ? tile_index - main_tiles : tile_index) * kFlashKeys;
        const std::int32_t keys          = extra ? p.extra_tokens : p.key_tokens;
        const __nv_bfloat16* key_rows    = extra ? p.extra_k : p.k;
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
                   (p.window == 0 ||
                    static_cast<std::int64_t>(query_low) - position < p.window);
        }
        if (!__syncthreads_or(seen)) continue;

        for (int r = warp; r < kFlashKeys; r += kFlashWarps) {
            const std::int32_t key = base + r;
            __nv_bfloat16* kdst    = ks + r * Layout::kQStride;
            __nv_bfloat16* vdst    = vs + r * vstride;
            // A key no row can see is not loaded: an unwritten cache slot may hold any bits, and a
            // masked NaN value would still poison the P*V product.
            const std::int32_t position = kpos[r];
            const bool unseen           = position > query_high ||
                                (p.window != 0 &&
                                 static_cast<std::int64_t>(query_low) - position >= p.window);
            if (key >= keys || unseen) {
                for (int j = lane; j < QK; j += 32) kdst[j] = __float2bfloat16_rn(0.0F);
                if constexpr (!kCompact) {
                    for (int j = lane; j < V; j += 32) vdst[j] = __float2bfloat16_rn(0.0F);
                }
                continue;
            }
            const std::int64_t row = static_cast<std::int64_t>(kvh) +
                                     static_cast<std::int64_t>(p.kv_heads) *
                                         (key + static_cast<std::int64_t>(keys) * b);
            flash_copy_row(kdst, key_rows + row * p.k_stride, QK, lane, 32);
            if constexpr (!kCompact) flash_copy_row(vdst, value_rows + row * p.v_stride, V, lane, 32);
        }
        __syncthreads();

        // S: warp w scores keys [8w, 8w + 8) for all sixteen rows.
        {
            float acc[4] = {0.0F, 0.0F, 0.0F, 0.0F};
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
            const float alpha  = empty || old == -CUDART_INF_F ? (empty ? 1.0F : 0.0F)
                                                              : __expf(old - high);
            const float total  = flash_warp_sum(weight);
            const __nv_bfloat16 head = __float2bfloat16_rn(weight);
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
                const std::uint32_t ah[4] = {flash_pair(ph + g * Layout::kPStride + kk + 2 * t),
                                             flash_pair(ph + (g + 8) * Layout::kPStride + kk + 2 * t),
                                             flash_pair(ph + g * Layout::kPStride + kk + 2 * t + 8),
                                             flash_pair(ph + (g + 8) * Layout::kPStride + kk + 2 * t + 8)};
                const std::uint32_t al[4] = {flash_pair(pl + g * Layout::kPStride + kk + 2 * t),
                                             flash_pair(pl + (g + 8) * Layout::kPStride + kk + 2 * t),
                                             flash_pair(pl + g * Layout::kPStride + kk + 2 * t + 8),
                                             flash_pair(pl + (g + 8) * Layout::kPStride + kk + 2 * t + 8)};
                const __nv_bfloat16* v0 = vs + (kk + 2 * t) * vstride;
#pragma unroll
                for (int n = 0; n < kTiles; ++n) {
                    const int col         = warp * kColumns + n * 8 + g;
                    const std::uint32_t b0 = flash_pack(v0[col], v0[vstride + col]);
                    const std::uint32_t b1 = flash_pack(v0[8 * vstride + col], v0[9 * vstride + col]);
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
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int r = g + 8 * half;
            float* dst  = p.partial_out + (base_row + r) * V;
#pragma unroll
            for (int n = 0; n < kTiles; ++n) {
                const int col = warp * kColumns + n * 8 + 2 * t;
                dst[col]      = o[n][2 * half];
                dst[col + 1]  = o[n][2 * half + 1];
            }
        }
        if (thread < kFlashRows) {
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
        __nv_bfloat16* dst =
            p.out + (static_cast<std::int64_t>(head) +
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

// Merges the splits of every partial row: out = sum_s O_s e^(m_s - M) / sum_s l_s e^(m_s - M).
// One block per partial row (every batch), a thread per value column.
template <int V>
__global__ void gemma_flash_attention_combine_kernel(FlashAttentionParams p, std::int64_t row_tiles) {
    const std::int64_t row   = blockIdx.x;
    const std::int64_t total = gridDim.x;
    const std::int32_t group = p.query_heads / p.kv_heads;
    const std::int64_t r     = row % kFlashRows;
    const std::int64_t tile  = (row / kFlashRows) % row_tiles;
    const std::int32_t kvh   = static_cast<std::int32_t>((row / kFlashRows / row_tiles) % p.kv_heads);
    const std::int32_t b     = static_cast<std::int32_t>(row / kFlashRows / row_tiles / p.kv_heads);
    const std::int64_t flat  = tile * kFlashRows + r;
    if (flat >= static_cast<std::int64_t>(p.query_tokens) * group) return;
    float high = -CUDART_INF_F;
    for (int s = 0; s < p.splits; ++s) high = fmaxf(high, p.partial_stats[(s * total + row) * 2]);
    float sum = 0.0F;
    for (int s = 0; s < p.splits; ++s) {
        const float m = p.partial_stats[(s * total + row) * 2];
        if (m != -CUDART_INF_F) sum += p.partial_stats[(s * total + row) * 2 + 1] * __expf(m - high);
    }
    const std::int32_t token = static_cast<std::int32_t>(flat / group);
    const std::int32_t head  = kvh * group + static_cast<std::int32_t>(flat % group);
    __nv_bfloat16* dst =
        p.out + (static_cast<std::int64_t>(head) +
                 static_cast<std::int64_t>(p.query_heads) *
                     (token + static_cast<std::int64_t>(p.query_tokens) * b)) *
                    V;
    for (int col = static_cast<int>(threadIdx.x); col < V; col += static_cast<int>(blockDim.x)) {
        float value = 0.0F;
        if (sum > 0.0F) {
            for (int s = 0; s < p.splits; ++s) {
                const float m = p.partial_stats[(s * total + row) * 2];
                if (m != -CUDART_INF_F) value += p.partial_out[(s * total + row) * V + col] * __expf(m - high);
            }
            value /= sum;
        }
        dst[col] = __float2bfloat16_rn(value);
    }
}

} // namespace ninfer::ops::detail
