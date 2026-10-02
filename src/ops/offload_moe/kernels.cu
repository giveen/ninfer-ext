// Implements: include/ninfer/ops/offload_moe.h
// Routed experts for the offloaded Qwen4Exp MoE. Weights stay NVFP4 in their slot or staging
// buffer and are decoded in registers; activations stay BF16 (A16). Every assignment (k,t) owns
// its own intermediate and partial output, and the merge sums k in order, so cache placement and
// job order never change a result.
#include "ops/offload_moe/launch.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math_constants.h>

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int H                         = kOffloadMoeHidden;
constexpr int E                         = kOffloadMoeExperts;
constexpr int K                         = kOffloadMoeTopK;
constexpr int I                         = kOffloadMoeIntermediate;
constexpr int kJobTokens                = 64;

__device__ __constant__ float kE2m1[16] = {0.0F,  0.5F,  1.0F,  1.5F,  2.0F,  3.0F,  4.0F,  6.0F,
                                           -0.0F, -0.5F, -1.0F, -1.5F, -2.0F, -3.0F, -4.0F, -6.0F};

// Swizzled block-scale offset of (row, group) in a matrix with `cols` logical columns.
__device__ __forceinline__ std::int64_t nvfp4_scale_offset(std::int32_t row, std::int32_t group,
                                                           std::int32_t cols) {
    const std::int32_t k_tiles   = cols / 64;
    const std::int32_t row_tile  = row >> 7;
    const std::int32_t row_inner = row & 127;
    return (static_cast<std::int64_t>(row_tile) * k_tiles + (group >> 2)) * 512 +
           (row_inner & 31) * 16 + (row_inner >> 5) * 4 + (group & 3);
}

__device__ __forceinline__ float nvfp4_value(const std::uint8_t* codes, const std::uint8_t* scales,
                                             std::int32_t row, std::int32_t col, std::int32_t cols,
                                             float inverse_divisor) {
    const std::uint8_t byte = codes[static_cast<std::int64_t>(row) * (cols / 2) + (col >> 1)];
    const std::uint8_t code = (col & 1) ? (byte >> 4) : (byte & 15);
    __nv_fp8_e4m3 scale;
    scale.__x = scales[nvfp4_scale_offset(row, col >> 4, cols)];
    return kE2m1[code] * static_cast<float>(scale) * inverse_divisor;
}

// ---------------------------------------------------------------------------------------------
// Routing. Logits: one warp per router row keeps its row in registers and sweeps a column tile, so
// the 2.6 MB router streams through many SMs even for one token. Top-k: one warp per column.

constexpr int kRouteWarps   = 4;
constexpr int kRouteColumns = 64;
constexpr int kRouteVectors = H / 8 / 32; // uint4 (8 x BF16) per lane

__device__ __forceinline__ float dot_bf16x8(uint4 a, uint4 b) {
    const auto* x = reinterpret_cast<const __nv_bfloat162*>(&a);
    const auto* y = reinterpret_cast<const __nv_bfloat162*>(&b);
    float sum     = 0.0F;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 u = __bfloat1622float2(x[i]);
        const float2 v = __bfloat1622float2(y[i]);
        sum            = fmaf(u.x, v.x, fmaf(u.y, v.y, sum));
    }
    return sum;
}

__global__ void __launch_bounds__(kRouteWarps * 32)
    moe_route_logits_kernel(const __nv_bfloat16* __restrict__ x,
                            const __nv_bfloat16* __restrict__ router, float* __restrict__ logits,
                            std::int32_t columns) {
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int row  = static_cast<int>(blockIdx.x) * kRouteWarps + warp;
    if (row > E) { return; }
    const auto* w4 = reinterpret_cast<const uint4*>(router + static_cast<std::int64_t>(row) * H);
    uint4 w[kRouteVectors];
#pragma unroll
    for (int i = 0; i < kRouteVectors; ++i) { w[i] = __ldg(w4 + i * 32 + lane); }
    const int begin = static_cast<int>(blockIdx.y) * kRouteColumns;
    const int end   = min(columns, begin + kRouteColumns);
    for (int col = begin; col < end; ++col) {
        const auto* x4 = reinterpret_cast<const uint4*>(x + static_cast<std::int64_t>(col) * H);
        float sum      = 0.0F;
#pragma unroll
        for (int i = 0; i < kRouteVectors; ++i) { sum += dot_bf16x8(w[i], x4[i * 32 + lane]); }
        sum = warp_sum(sum);
        if (lane == 0) { logits[static_cast<std::int64_t>(col) * (E + 1) + row] = sum; }
    }
}

__global__ void __launch_bounds__(256)
    moe_route_topk_kernel(const float* __restrict__ logits, std::int32_t columns,
                          std::int32_t* __restrict__ ids, float* __restrict__ weights,
                          float* __restrict__ shared_gate) {
    constexpr int kPerLane = E / 32;
    const int lane         = static_cast<int>(threadIdx.x) & 31;
    const std::int64_t col = static_cast<std::int64_t>(blockIdx.x) * 8 + (threadIdx.x >> 5);
    if (col >= columns) { return; }
    const float* l = logits + col * (E + 1);
    float v[kPerLane];
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) { v[i] = l[i * 32 + lane]; }
    unsigned taken = 0;
    float top[K];
    int top_id[K];
    for (int k = 0; k < K; ++k) {
        float best  = -CUDART_INF_F;
        int best_id = E;
#pragma unroll
        for (int i = 0; i < kPerLane; ++i) {
            const int e = i * 32 + lane;
            if (!((taken >> i) & 1U) && (v[i] > best || (v[i] == best && e < best_id))) {
                best    = v[i];
                best_id = e;
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other_v = __shfl_xor_sync(0xffffffffu, best, offset);
            const int other_id  = __shfl_xor_sync(0xffffffffu, best_id, offset);
            if (other_v > best || (other_v == best && other_id < best_id)) {
                best    = other_v;
                best_id = other_id;
            }
        }
        top[k]    = best;
        top_id[k] = best_id;
        if ((best_id & 31) == lane) { taken |= 1U << (best_id >> 5); }
    }
    if (lane == 0) {
        float total = 0.0F;
        for (int k = 0; k < K; ++k) { total += __expf(top[k] - top[0]); }
        for (int k = 0; k < K; ++k) {
            ids[col * K + k]     = top_id[k];
            weights[col * K + k] = __expf(top[k] - top[0]) / total;
        }
        shared_gate[col] = sigmoid(l[E]);
    }
}

// ---------------------------------------------------------------------------------------------
// Cache resolution: one CTA of 1024 threads per call.

// CTA b copies lane b's last valid routing over its padding columns. The source column is valid,
// so it is never itself overwritten.
constexpr int kShareThreads = 128;

__global__ void __launch_bounds__(kShareThreads)
    moe_route_share_padding_kernel(std::int32_t* __restrict__ ids,
                                   const std::int32_t* __restrict__ valid_columns,
                                   std::int32_t width) {
    const int lane  = static_cast<int>(blockIdx.x);
    const int valid = valid_columns[lane];
    if (valid < 1 || valid >= width) { return; }
    std::int32_t* row        = ids + static_cast<std::int64_t>(lane) * width * K;
    const std::int32_t* from = row + (valid - 1) * K;
    for (int i = static_cast<int>(threadIdx.x); i < (width - valid) * K; i += blockDim.x) {
        row[valid * K + i] = from[i % K];
    }
}

constexpr int kResolveThreads = 1024;

// Exclusive scan over the CTA (blockDim.x a multiple of 32, at most 1024): warp shuffles, then
// one warp scans the per-warp totals. `scratch` holds at least 32 ints.
__device__ __forceinline__ int block_exclusive_scan(int value, int* scratch, int& total) {
    const int tid  = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    int inclusive  = value;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int up = __shfl_up_sync(0xffffffffu, inclusive, offset);
        if (lane >= offset) { inclusive += up; }
    }
    if (lane == 31) { scratch[warp] = inclusive; }
    __syncthreads();
    if (warp == 0) {
        const int warps = static_cast<int>(blockDim.x) >> 5;
        int sum         = lane < warps ? scratch[lane] : 0;
#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            const int up = __shfl_up_sync(0xffffffffu, sum, offset);
            if (lane >= offset) { sum += up; }
        }
        scratch[lane] = sum; // inclusive over warps
    }
    __syncthreads();
    total            = scratch[31];
    const int result = (warp > 0 ? scratch[warp - 1] : 0) + inclusive - value;
    __syncthreads();
    return result;
}

// Eviction key of a slot: smaller is older. Stamps at or after `now` (touched by this call) are
// protected; older stamps order by age, saturated to 32 bits.
__device__ __forceinline__ std::uint32_t eviction_key(unsigned long long stamp,
                                                      unsigned long long now) {
    if (stamp >= now) { return 0xFFFFFFFFu; }
    const unsigned long long age = now - stamp;
    return 0xFFFFFFFFu - static_cast<std::uint32_t>(min(age, 0xFFFFFFFFULL));
}

__global__ void __launch_bounds__(kResolveThreads)
    expert_cache_resolve_kernel(const std::int32_t* __restrict__ ids, std::int32_t assignments,
                                std::int32_t layer, ExpertCacheState cache,
                                std::int32_t* __restrict__ slot_ids,
                                std::int32_t* __restrict__ misses) {
    __shared__ int used[E];
    __shared__ int slot_for[E];
    __shared__ int miss_expert[E];
    __shared__ int scratch[32];
    __shared__ unsigned long long now_s;
    __shared__ std::uint32_t prefix_s;
    __shared__ int remaining_s;
    __shared__ unsigned histogram[256];
    __shared__ int miss_count_s;
    __shared__ int emitted_s;
    __shared__ int ties_s;
    const int tid = static_cast<int>(threadIdx.x);
    if (tid == 0) { now_s = atomicAdd(cache.clock, 1ULL) + 1ULL; }
    for (int e = tid; e < E; e += blockDim.x) { used[e] = 0; }
    __syncthreads();
    for (int i = tid; i < assignments; i += blockDim.x) { used[ids[i]] = 1; }
    __syncthreads();
    const unsigned long long now = now_s;
    // Hits refresh their stamp; misses are listed in ascending expert order.
    int miss = 0;
    if (tid < E && used[tid]) {
        const int slot = cache.slot_of[layer * E + tid];
        if (slot >= 0) {
            cache.stamp[slot] = now;
            slot_for[tid]     = slot;
        } else {
            miss = 1;
        }
    }
    int miss_total = 0;
    const int rank = block_exclusive_scan(miss, scratch, miss_total);
    if (miss) { miss_expert[rank] = tid; }
    if (tid == 0) {
        miss_count_s = miss_total;
        atomicAdd(&cache.statistics[1], static_cast<unsigned long long>(miss_total));
    }
    int hit  = tid < E && used[tid] && !miss ? 1 : 0;
    int hits = 0;
    (void)block_exclusive_scan(hit, scratch, hits);
    if (tid == 0) { atomicAdd(&cache.statistics[0], static_cast<unsigned long long>(hits)); }
    __syncthreads();
    const int need_total = miss_count_s;

    if (need_total > 0) {
        // Radix-select the need_total-th smallest eviction key.
        if (tid == 0) {
            prefix_s    = 0U;
            remaining_s = need_total;
        }
        __syncthreads();
        std::uint32_t mask = 0U;
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (int b = tid; b < 256; b += blockDim.x) { histogram[b] = 0U; }
            __syncthreads();
            for (int s = tid; s < cache.slots; s += blockDim.x) {
                const std::uint32_t key = eviction_key(cache.stamp[s], now);
                if ((key & mask) == prefix_s) { atomicAdd(&histogram[(key >> shift) & 0xFFU], 1U); }
            }
            __syncthreads();
            if (tid < 32) {
                // Lane l owns bins [8l, 8l + 8); the first lane whose running count reaches the
                // remaining need holds the bin.
                int counts[8];
                int own = 0;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    counts[i] = static_cast<int>(histogram[tid * 8 + i]);
                    own += counts[i];
                }
                int inclusive = own;
#pragma unroll
                for (int offset = 1; offset < 32; offset <<= 1) {
                    const int up = __shfl_up_sync(0xffffffffu, inclusive, offset);
                    if (tid >= offset) { inclusive += up; }
                }
                const int need   = remaining_s;
                const unsigned hit = __ballot_sync(0xffffffffu, inclusive >= need);
                if (hit != 0U && tid == __ffs(static_cast<int>(hit)) - 1) {
                    int left = need - (inclusive - own);
                    for (int i = 0; i < 8; ++i) {
                        if (counts[i] >= left) {
                            prefix_s |= static_cast<std::uint32_t>(tid * 8 + i) << shift;
                            remaining_s = left;
                            break;
                        }
                        left -= counts[i];
                    }
                }
            }
            mask |= 0xFFU << shift;
            __syncthreads();
        }
        const std::uint32_t threshold = prefix_s;
        if (tid == 0) {
            emitted_s = 0;
            ties_s    = remaining_s;
        }
        __syncthreads();
        for (int base = 0; base < cache.slots; base += blockDim.x) {
            const int s             = base + tid;
            const std::uint32_t key = s < cache.slots ? eviction_key(cache.stamp[s], now)
                                                      : 0xFFFFFFFFu;
            const int less = s < cache.slots && key < threshold ? 1 : 0;
            const int tie  = s < cache.slots && key == threshold && key != 0xFFFFFFFFu ? 1 : 0;
            int tie_total      = 0;
            const int tie_rank = block_exclusive_scan(tie, scratch, tie_total);
            const int take     = less || (tie && tie_rank < ties_s);
            int taken_total    = 0;
            const int order    = block_exclusive_scan(take, scratch, taken_total);
            if (take) {
                const int j = emitted_s + order;
                if (j < need_total) {
                    const int expert = miss_expert[j];
                    const int old    = cache.owner[s];
                    if (old >= 0) { cache.slot_of[old] = -1; }
                    cache.owner[s]                    = layer * E + expert;
                    cache.slot_of[layer * E + expert] = s;
                    cache.stamp[s]                    = now;
                    slot_for[expert]                  = s;
                    misses[1 + 2 * j]                 = s;
                    misses[2 + 2 * j]                 = expert;
                }
            }
            __syncthreads();
            if (tid == 0) {
                emitted_s += taken_total;
                ties_s = max(0, ties_s - tie_total);
            }
            __syncthreads();
        }
    }
    __syncthreads();
    if (tid == 0) { misses[0] = min(need_total, emitted_s); }
    for (int i = tid; i < assignments; i += blockDim.x) { slot_ids[i] = slot_for[ids[i]]; }
}

// ---------------------------------------------------------------------------------------------
// Fetch: a fixed grid grid-strides over the flattened (miss, 16-byte vector) work. SM reads of
// pinned Host memory lose PCIe throughput as more CTAs contend, so the CTA count stays constant
// instead of growing with the miss count (32 CTAs per miss fell to ~20 GB/s at 16 misses; a fixed
// 64 holds ~36 GB/s on RTX 5090).

constexpr int kFetchChunks = 32; // staging: CTAs per expert
constexpr int kFetchCtas   = 64;

__global__ void __launch_bounds__(256)
    expert_cache_fetch_kernel(ExpertWeights bank, const std::int32_t* __restrict__ misses,
                              std::int32_t max_misses, std::byte* pool) {
    constexpr std::int64_t kVectors[4] = {kExpertGateUpCodeBytes / 16, kExpertGateUpScaleBytes / 16,
                                          kExpertDownCodeBytes / 16, kExpertDownScaleBytes / 16};
    constexpr std::int64_t kPerExpert = kExpertSlotBytes / 16;
    const std::int64_t total  = static_cast<std::int64_t>(min(misses[0], max_misses)) * kPerExpert;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t w = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         w < total; w += stride) {
        const std::int64_t j = w / kPerExpert;
        std::int64_t v       = w - j * kPerExpert;
        std::int64_t offset  = 0;
        int p                = 0;
        while (v >= kVectors[p]) {
            v -= kVectors[p];
            offset += kVectors[p];
            ++p;
        }
        const std::int64_t slot   = misses[1 + 2 * j];
        const std::int64_t expert = misses[2 + 2 * j];
        const auto* src = reinterpret_cast<const uint4*>(bank.base[p] + expert * bank.stride[p]);
        auto* dst       = reinterpret_cast<uint4*>(pool + slot * kExpertSlotBytes) + offset;
        dst[v]          = src[v];
    }
}

__global__ void __launch_bounds__(256)
    expert_cache_reclaim_kernel(ExpertCacheState cache, std::int32_t first_slot) {
    for (int s = first_slot + static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
         s < cache.slots; s += static_cast<int>(gridDim.x * blockDim.x)) {
        const int owner = cache.owner[s];
        if (owner >= 0) { cache.slot_of[owner] = -1; }
        cache.owner[s] = -1;
        cache.stamp[s] = 0ULL;
    }
}

// CTA (e, chunk) copies chunk `chunk` of each plane of expert e into the staged bank when the
// snapshot marked e resident: from its slot if the device still holds it, else from the bank.
__global__ void __launch_bounds__(256)
    expert_cache_stage_kernel(ExpertCacheState cache, std::int32_t layer,
                              const std::int32_t* __restrict__ resident, ExpertWeights bank,
                              ExpertWeights staged) {
    const int expert = static_cast<int>(blockIdx.x);
    const std::int64_t entry = static_cast<std::int64_t>(layer) * E + expert;
    if (resident[entry] < 0) { return; }
    const std::int32_t slot     = cache.slot_of[entry];
    const std::int64_t sizes[4] = {kExpertGateUpCodeBytes, kExpertGateUpScaleBytes,
                                   kExpertDownCodeBytes, kExpertDownScaleBytes};
    std::int64_t plane_offset   = 0;
    for (int p = 0; p < 4; ++p) {
        const std::byte* from = slot >= 0 ? cache.pool + slot * kExpertSlotBytes + plane_offset
                                          : bank.base[p] + expert * bank.stride[p];
        const auto* src          = reinterpret_cast<const uint4*>(from);
        auto* dst                = reinterpret_cast<uint4*>(
            const_cast<std::byte*>(staged.base[p]) + expert * staged.stride[p]);
        const std::int64_t vectors = sizes[p] / 16;
        const std::int64_t per     = (vectors + kFetchChunks - 1) / kFetchChunks;
        const std::int64_t begin   = static_cast<std::int64_t>(blockIdx.y) * per;
        const std::int64_t end     = min(vectors, begin + per);
        for (std::int64_t v = begin + threadIdx.x; v < end; v += blockDim.x) { dst[v] = src[v]; }
        plane_offset += sizes[p];
    }
}

// ---------------------------------------------------------------------------------------------
// Job building: sort assignments by slot, then cut each slot's run into jobs of <= 64 tokens.

constexpr int kJobThreads = 1024;

__global__ void __launch_bounds__(kJobThreads)
    moe_jobs_kernel(const std::int32_t* __restrict__ slot_ids, std::int32_t assignments,
                    std::int32_t slots, std::int32_t* __restrict__ counts,
                    std::int32_t* __restrict__ sorted_assign,
                    std::int32_t* __restrict__ sorted_slot, std::int32_t* __restrict__ jobs,
                    std::int32_t* __restrict__ job_count) {
    __shared__ unsigned long long keys[kJobThreads];
    __shared__ int scratch[kJobThreads];
    __shared__ int base_s;
    const int tid = static_cast<int>(threadIdx.x);
    if (assignments <= kJobThreads) {
        // Bitonic sort of (slot, assignment) keys in shared memory.
        keys[tid] = tid < assignments ? (static_cast<unsigned long long>(slot_ids[tid]) << 32) |
                                            static_cast<unsigned>(tid)
                                      : ~0ULL;
        __syncthreads();
        for (int size = 2; size <= kJobThreads; size <<= 1) {
            for (int stride = size >> 1; stride > 0; stride >>= 1) {
                const int partner = tid ^ stride;
                if (partner > tid) {
                    const bool ascending       = (tid & size) == 0;
                    const unsigned long long a = keys[tid];
                    const unsigned long long b = keys[partner];
                    if ((a > b) == ascending) {
                        keys[tid]     = b;
                        keys[partner] = a;
                    }
                }
                __syncthreads();
            }
        }
        if (tid < assignments) {
            sorted_assign[tid] = static_cast<std::int32_t>(keys[tid] & 0xFFFFFFFFULL);
            sorted_slot[tid]   = static_cast<std::int32_t>(keys[tid] >> 32);
        }
        __syncthreads();
    } else {
        // Counting sort over a small slot domain (staged banks index by expert id).
        for (int s = tid; s < slots; s += blockDim.x) { counts[s] = 0; }
        __syncthreads();
        for (int i = tid; i < assignments; i += blockDim.x) { atomicAdd(&counts[slot_ids[i]], 1); }
        __syncthreads();
        if (tid == 0) { base_s = 0; }
        __syncthreads();
        for (int chunk = 0; chunk < slots; chunk += blockDim.x) {
            const int s     = chunk + tid;
            const int value = s < slots ? counts[s] : 0;
            int total       = 0;
            const int start = block_exclusive_scan(value, scratch, total);
            if (s < slots) { counts[s] = base_s + start; }
            __syncthreads();
            if (tid == 0) { base_s += total; }
            __syncthreads();
        }
        // Scatter; order inside a slot run is irrelevant because every assignment is computed
        // independently.
        for (int i = tid; i < assignments; i += blockDim.x) {
            const int s             = slot_ids[i];
            const int position      = atomicAdd(&counts[s], 1);
            sorted_assign[position] = i;
            sorted_slot[position]   = s;
        }
        __syncthreads();
    }
    // Job starts: first entry of a slot run and every 64th entry within it.
    if (tid == 0) { base_s = 0; }
    __syncthreads();
    for (int chunk = 0; chunk < assignments; chunk += blockDim.x) {
        const int i   = chunk + tid;
        int start_job = 0;
        if (i < assignments) {
            // i starts a job when its offset inside its slot run is a multiple of 64.
            int offset = 0;
            for (int j = i; j > 0 && sorted_slot[j - 1] == sorted_slot[i]; --j) { ++offset; }
            start_job = offset % kJobTokens == 0 ? 1 : 0;
        }
        int total       = 0;
        const int index = block_exclusive_scan(start_job, scratch, total);
        if (start_job) { jobs[base_s + index] = i; }
        __syncthreads();
        if (tid == 0) { base_s += total; }
        __syncthreads();
    }
    if (tid == 0) { *job_count = base_s; }
}

// ---------------------------------------------------------------------------------------------
// Expert projections. One CTA per (job, row tile); 256 threads; 64x64 FP32 staging tiles.

constexpr int kTile = 64;

struct JobView {
    int begin;
    int count;
    int slot;
    int expert;
};

// Selects which jobs a projection pass runs: all of them, or only the slots that are (not) listed
// in a resolve miss list.
struct JobFilter {
    const std::int32_t* misses; // [count, (slot, expert)...]; null runs every job
    bool missed;                // run the listed slots (true) or the others (false)

    __device__ __forceinline__ bool admits(int slot) const {
        if (misses == nullptr) { return true; }
        const int count = misses[0];
        bool listed     = false;
        for (int j = 0; j < count && !listed; ++j) { listed = misses[1 + 2 * j] == slot; }
        return listed == missed;
    }
};

// First assignment and slot of this CTA's job; count 0 when the CTA has no job or `filter` skips
// it.
__device__ __forceinline__ JobView job_head(const std::int32_t* jobs, const std::int32_t* job_count,
                                            const std::int32_t* sorted_slot, JobFilter filter) {
    JobView view{-1, 0, -1, -1};
    const int job = static_cast<int>(blockIdx.x);
    if (job >= *job_count) { return view; }
    view.begin = jobs[job];
    view.slot  = sorted_slot[view.begin];
    if (!filter.admits(view.slot)) { return view; }
    view.count = 1;
    return view;
}

// Token count and expert of a job whose head is known.
__device__ __forceinline__ void job_tail(JobView& view, const std::int32_t* sorted_slot,
                                         const std::int32_t* sorted_assign,
                                         const std::int32_t* expert_ids, int assignments) {
    int count = 0;
    while (count < kJobTokens && view.begin + count < assignments &&
           sorted_slot[view.begin + count] == view.slot) {
        ++count;
    }
    view.count  = count;
    view.expert = expert_ids[sorted_assign[view.begin]];
}

__device__ __forceinline__ JobView load_job(const std::int32_t* jobs, const std::int32_t* job_count,
                                            const std::int32_t* sorted_slot,
                                            const std::int32_t* sorted_assign,
                                            const std::int32_t* expert_ids, int assignments,
                                            JobFilter filter) {
    JobView view = job_head(jobs, job_count, sorted_slot, filter);
    if (view.count == 0) { return view; }
    job_tail(view, sorted_slot, sorted_assign, expert_ids, assignments);
    return view;
}

__global__ void __launch_bounds__(256) moe_gate_up_kernel(
    const __nv_bfloat16* __restrict__ x, const std::int32_t* __restrict__ expert_ids,
    const std::int32_t* __restrict__ sorted_assign, const std::int32_t* __restrict__ sorted_slot,
    const std::int32_t* __restrict__ jobs, const std::int32_t* __restrict__ job_count,
    std::int32_t assignments, JobFilter filter, ExpertWeights weights,
    __nv_bfloat16* __restrict__ act) {
    __shared__ float xs[kJobTokens][kTile + 1];
    __shared__ float ws[kTile][kTile + 1];
    const JobView job =
        load_job(jobs, job_count, sorted_slot, sorted_assign, expert_ids, assignments, filter);
    if (job.count == 0) { return; }
    const int tid  = static_cast<int>(threadIdx.x);
    const int ty   = tid / 16;                          // rows ty*4 .. ty*4+3 of the 64-row tile
    const int tx   = tid % 16;                          // tokens tx*4 .. tx*4+3
    const int row0 = static_cast<int>(blockIdx.y) * 32; // gate rows row0..row0+31, up +I
    const auto* codes =
        reinterpret_cast<const std::uint8_t*>(weights.base[0] + job.slot * weights.stride[0]);
    const auto* scale =
        reinterpret_cast<const std::uint8_t*>(weights.base[1] + job.slot * weights.stride[1]);
    const std::int64_t bank_row = static_cast<std::int64_t>(job.expert) * 2 * I;
    const float inv_gate = 1.0F / weights.gate_up_divisors[bank_row / weights.gate_up_divisor_rows];
    const float inv_up =
        1.0F / weights.gate_up_divisors[(bank_row + I) / weights.gate_up_divisor_rows];
    float acc[4][4] = {};
    for (int k0 = 0; k0 < H; k0 += kTile) {
        for (int i = tid; i < kJobTokens * kTile; i += blockDim.x) {
            const int token = i / kTile, kk = i % kTile;
            float value = 0.0F;
            if (token < job.count) {
                const std::int64_t column = sorted_assign[job.begin + token] / K;
                value                     = __bfloat162float(x[column * H + k0 + kk]);
            }
            xs[token][kk] = value;
        }
        for (int i = tid; i < kTile * kTile; i += blockDim.x) {
            const int local = i / kTile, kk = i % kTile;
            const int row = local < 32 ? row0 + local : I + row0 + (local - 32);
            ws[local][kk] =
                nvfp4_value(codes, scale, row, k0 + kk, H, local < 32 ? inv_gate : inv_up);
        }
        __syncthreads();
        for (int kk = 0; kk < kTile; ++kk) {
            float wr[4], xr[4];
            for (int r = 0; r < 4; ++r) { wr[r] = ws[ty * 4 + r][kk]; }
            for (int c = 0; c < 4; ++c) { xr[c] = xs[tx * 4 + c][kk]; }
            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c) { acc[r][c] += wr[r] * xr[c]; }
            }
        }
        __syncthreads();
    }
    // Exchange through shared memory so gate row r meets up row r.
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) { ws[ty * 4 + r][tx * 4 + c] = acc[r][c]; }
    }
    __syncthreads();
    for (int i = tid; i < 32 * job.count; i += blockDim.x) {
        const int r = i % 32, token = i / 32;
        const float g         = ws[r][token];
        const float u         = ws[32 + r][token];
        const std::int64_t a  = sorted_assign[job.begin + token];
        act[a * I + row0 + r] = __float2bfloat16_rn(silu(g) * u);
    }
}

__global__ void __launch_bounds__(256) moe_down_kernel(
    const __nv_bfloat16* __restrict__ act, const std::int32_t* __restrict__ expert_ids,
    const std::int32_t* __restrict__ sorted_assign, const std::int32_t* __restrict__ sorted_slot,
    const std::int32_t* __restrict__ jobs, const std::int32_t* __restrict__ job_count,
    std::int32_t assignments, JobFilter filter, ExpertWeights weights,
    float* __restrict__ partial) {
    __shared__ float xs[kJobTokens][kTile + 1];
    __shared__ float ws[kTile][kTile + 1];
    const JobView job =
        load_job(jobs, job_count, sorted_slot, sorted_assign, expert_ids, assignments, filter);
    if (job.count == 0) { return; }
    const int tid  = static_cast<int>(threadIdx.x);
    const int ty   = tid / 16;
    const int tx   = tid % 16;
    const int row0 = static_cast<int>(blockIdx.y) * kTile;
    const auto* codes =
        reinterpret_cast<const std::uint8_t*>(weights.base[2] + job.slot * weights.stride[2]);
    const auto* scale =
        reinterpret_cast<const std::uint8_t*>(weights.base[3] + job.slot * weights.stride[3]);
    const float inv = 1.0F / weights.down_divisors[static_cast<std::int64_t>(job.expert) * H /
                                                   weights.down_divisor_rows];
    float acc[4][4] = {};
    for (int k0 = 0; k0 < I; k0 += kTile) {
        for (int i = tid; i < kJobTokens * kTile; i += blockDim.x) {
            const int token = i / kTile, kk = i % kTile;
            float value = 0.0F;
            if (token < job.count) {
                const std::int64_t a = sorted_assign[job.begin + token];
                value                = __bfloat162float(act[a * I + k0 + kk]);
            }
            xs[token][kk] = value;
        }
        for (int i = tid; i < kTile * kTile; i += blockDim.x) {
            const int local = i / kTile, kk = i % kTile;
            ws[local][kk] = nvfp4_value(codes, scale, row0 + local, k0 + kk, I, inv);
        }
        __syncthreads();
        for (int kk = 0; kk < kTile; ++kk) {
            float wr[4], xr[4];
            for (int r = 0; r < 4; ++r) { wr[r] = ws[ty * 4 + r][kk]; }
            for (int c = 0; c < 4; ++c) { xr[c] = xs[tx * 4 + c][kk]; }
            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c) { acc[r][c] += wr[r] * xr[c]; }
            }
        }
        __syncthreads();
    }
    for (int c = 0; c < 4; ++c) {
        const int token = tx * 4 + c;
        if (token >= job.count) { continue; }
        const std::int64_t a = sorted_assign[job.begin + token];
        for (int r = 0; r < 4; ++r) { partial[a * H + row0 + ty * 4 + r] = acc[r][c]; }
    }
}

// ---------------------------------------------------------------------------------------------
// Decode projections: at most 64 token columns, so a job holds only a few tokens, staged in shared
// memory 8 at a time. Gate/up: one warp owns two gate/up pairs and sweeps each 2560-column row
// with 8-byte code loads. Down: four lanes own one 640-column row; right after the job's slot is
// known each lane issues all of its 16-byte code loads and block-scale pairs, so the weight stream
// does not wait on the rest of the job lookup, and a row finishes with a 4-lane reduction. Rows
// reduce only the job's live tokens. Each token's sum depends only on that token and the weights,
// never on its neighbours in the job. (Giving gate/up the down layout measured slower in the
// engine: its larger unrolled body cost more than the extra loads in flight gained.)

constexpr int kGemvTokens         = 8;
constexpr int kGemvWarps          = 8;
constexpr int kGemvThreads        = kGemvWarps * 32;
constexpr int kGemvAssignments    = 640; // 64 columns
constexpr int kGateUpPairsPerWarp = 2;
constexpr int kDownLanes          = 4; // lanes per down row
constexpr int kDownRowsPerCta     = kGemvThreads / kDownLanes;

// One lane's share of a weight row of `Cols` columns: 16-byte code vectors v = sub + Lanes*i, each
// covering scale groups 2v and 2v+1, whose two E4M3 scales are adjacent bytes of the swizzle.
template <int Cols, int Lanes>
struct RowShare {
    static constexpr int kVectors = Cols / 32 / Lanes;
    static_assert(Cols % (32 * Lanes) == 0, "a row must split evenly into 16-byte lane shares");
    uint4 codes[kVectors];
    std::uint16_t scales[kVectors];

    __device__ __forceinline__ void load(const std::uint8_t* __restrict__ plane_codes,
                                         const std::uint8_t* __restrict__ plane_scales, int row,
                                         int sub) {
        const auto* row_codes = reinterpret_cast<const uint4*>(
            plane_codes + static_cast<std::int64_t>(row) * (Cols / 2));
#pragma unroll
        for (int i = 0; i < kVectors; ++i) {
            const int v = sub + Lanes * i;
            codes[i]    = row_codes[v];
            scales[i]   = *reinterpret_cast<const std::uint16_t*>(
                plane_scales + nvfp4_scale_byte<Cols / 64>(row, 2 * v));
        }
    }
};

// Accumulate one 16-weight scale group against `tokens` activation rows of stride `cols`.
template <int Cols>
__device__ __forceinline__ void gemv_group(uint2 codes, float scale,
                                           const __nv_bfloat16* __restrict__ xs, int k, int tokens,
                                           float (&acc)[kGemvTokens]) {
    float w[16];
#pragma unroll
    for (int b = 0; b < 8; ++b) {
        const unsigned word =
            b < 4 ? static_cast<unsigned>(codes.x) : static_cast<unsigned>(codes.y);
        const float2 pair = decode_nvfp4_e2m1x2(
            static_cast<std::uint8_t>(byte_of_word(word, static_cast<unsigned>(b) & 3u)));
        w[2 * b]     = pair.x * scale;
        w[2 * b + 1] = pair.y * scale;
    }
#pragma unroll
    for (int j = 0; j < kGemvTokens; ++j) {
        if (j >= tokens) { break; }
        const auto* x4 = reinterpret_cast<const uint4*>(xs + j * Cols + k);
        const uint4 lo = x4[0], hi = x4[1];
        const auto* a = reinterpret_cast<const __nv_bfloat162*>(&lo);
        const auto* c = reinterpret_cast<const __nv_bfloat162*>(&hi);
        float sum     = 0.0F;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 u = __bfloat1622float2(a[i]);
            const float2 v = __bfloat1622float2(c[i]);
            sum            = fmaf(w[2 * i], u.x, fmaf(w[2 * i + 1], u.y, sum));
            sum            = fmaf(w[8 + 2 * i], v.x, fmaf(w[9 + 2 * i], v.y, sum));
        }
        acc[j] += sum;
    }
}

// Sum of one lane's row share against `tokens` activation rows.
template <int Cols, int Lanes>
__device__ __forceinline__ void row_share_dot(const RowShare<Cols, Lanes>& share,
                                              const __nv_bfloat16* __restrict__ xs, int sub,
                                              int tokens, float (&acc)[kGemvTokens]) {
#pragma unroll
    for (int i = 0; i < RowShare<Cols, Lanes>::kVectors; ++i) {
        const int k   = (sub + Lanes * i) * 32;
        const uint4 c = share.codes[i];
        gemv_group<Cols>(make_uint2(c.x, c.y), decode_nvfp4_e4m3(share.scales[i] & 0xFF), xs, k,
                         tokens, acc);
        gemv_group<Cols>(make_uint2(c.z, c.w), decode_nvfp4_e4m3(share.scales[i] >> 8), xs, k + 16,
                         tokens, acc);
    }
}

// Copy the activation rows of tokens [t0, t0+tokens) of a job into shared memory; `row_of(j)` is
// the source row of the job's j-th token.
template <int Cols, class RowOf>
__device__ __forceinline__ void stage_rows(__nv_bfloat16* __restrict__ xs,
                                           const __nv_bfloat16* __restrict__ source, int tokens,
                                           RowOf row_of) {
    for (int i = static_cast<int>(threadIdx.x); i < tokens * (Cols / 8); i += blockDim.x) {
        const int token                 = i / (Cols / 8);
        const int v                     = i - token * (Cols / 8);
        const std::int64_t row          = row_of(token);
        reinterpret_cast<uint4*>(xs)[i] = reinterpret_cast<const uint4*>(source + row * Cols)[v];
    }
}

// One warp owns two gate/up pairs and sweeps their rows with 32 lanes.
__global__ void __launch_bounds__(kGemvThreads) moe_gate_up_gemv_kernel(
    const __nv_bfloat16* __restrict__ x, const std::int32_t* __restrict__ expert_ids,
    const std::int32_t* __restrict__ sorted_assign, const std::int32_t* __restrict__ sorted_slot,
    const std::int32_t* __restrict__ jobs, const std::int32_t* __restrict__ job_count,
    std::int32_t assignments, JobFilter filter, ExpertWeights weights,
    __nv_bfloat16* __restrict__ act) {
    __shared__ __align__(16) __nv_bfloat16 xs[kGemvTokens * H];
    const JobView job =
        load_job(jobs, job_count, sorted_slot, sorted_assign, expert_ids, assignments, filter);
    if (job.count == 0) { return; }
    const int tid   = static_cast<int>(threadIdx.x);
    const int warp  = tid >> 5;
    const int lane  = tid & 31;
    const int pair0 = (static_cast<int>(blockIdx.y) * kGemvWarps + warp) * kGateUpPairsPerWarp;
    const auto* codes =
        reinterpret_cast<const std::uint8_t*>(weights.base[0] + job.slot * weights.stride[0]);
    const auto* scale =
        reinterpret_cast<const std::uint8_t*>(weights.base[1] + job.slot * weights.stride[1]);
    const std::int64_t bank_row = static_cast<std::int64_t>(job.expert) * 2 * I;
    const float inv_gate = 1.0F / weights.gate_up_divisors[bank_row / weights.gate_up_divisor_rows];
    const float inv_up =
        1.0F / weights.gate_up_divisors[(bank_row + I) / weights.gate_up_divisor_rows];
    for (int t0 = 0; t0 < job.count; t0 += kGemvTokens) {
        const int tokens = min(kGemvTokens, job.count - t0);
        __syncthreads();
        for (int i = tid; i < tokens * (H / 8); i += blockDim.x) {
            const int token                 = i / (H / 8);
            const int v                     = i - token * (H / 8);
            const std::int64_t column       = sorted_assign[job.begin + t0 + token] / K;
            reinterpret_cast<uint4*>(xs)[i] = reinterpret_cast<const uint4*>(x + column * H)[v];
        }
        __syncthreads();
        float acc[kGateUpPairsPerWarp][2][kGemvTokens] = {};
        for (int q = lane; q < H / 16; q += 32) {
#pragma unroll
            for (int p = 0; p < kGateUpPairsPerWarp; ++p) {
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const int row = half * I + pair0 + p;
                    const uint2 c = *reinterpret_cast<const uint2*>(
                        codes + static_cast<std::int64_t>(row) * (H / 2) + q * 8);
                    const float sc = decode_nvfp4_e4m3(scale[nvfp4_scale_byte<H / 64>(row, q)]);
                    gemv_group<H>(c, sc, xs, q * 16, tokens, acc[p][half]);
                }
            }
        }
#pragma unroll
        for (int p = 0; p < kGateUpPairsPerWarp; ++p) {
#pragma unroll
            for (int j = 0; j < kGemvTokens; ++j) {
                if (j >= tokens) { break; }
                const float g = warp_sum(acc[p][0][j]) * inv_gate;
                const float u = warp_sum(acc[p][1][j]) * inv_up;
                if (lane == 0) {
                    const std::int64_t a   = sorted_assign[job.begin + t0 + j];
                    act[a * I + pair0 + p] = __float2bfloat16_rn(silu(g) * u);
                }
            }
        }
    }
}

// Lane group g of 4 lanes owns down row g of this CTA's rows.
__global__ void __launch_bounds__(kGemvThreads) moe_down_gemv_kernel(
    const __nv_bfloat16* __restrict__ act, const std::int32_t* __restrict__ expert_ids,
    const std::int32_t* __restrict__ sorted_assign, const std::int32_t* __restrict__ sorted_slot,
    const std::int32_t* __restrict__ jobs, const std::int32_t* __restrict__ job_count,
    std::int32_t assignments, JobFilter filter, ExpertWeights weights,
    float* __restrict__ partial) {
    __shared__ __align__(16) __nv_bfloat16 xs[kGemvTokens * I];
    JobView job = job_head(jobs, job_count, sorted_slot, filter);
    if (job.count == 0) { return; }
    const int tid = static_cast<int>(threadIdx.x);
    const int sub = tid % kDownLanes;
    const int row = static_cast<int>(blockIdx.y) * kDownRowsPerCta + tid / kDownLanes;
    RowShare<I, kDownLanes> share;
    share.load(
        reinterpret_cast<const std::uint8_t*>(weights.base[2] + job.slot * weights.stride[2]),
        reinterpret_cast<const std::uint8_t*>(weights.base[3] + job.slot * weights.stride[3]), row,
        sub);
    job_tail(job, sorted_slot, sorted_assign, expert_ids, assignments);
    const float inv = 1.0F / weights.down_divisors[static_cast<std::int64_t>(job.expert) * H /
                                                   weights.down_divisor_rows];
    for (int t0 = 0; t0 < job.count; t0 += kGemvTokens) {
        const int tokens = min(kGemvTokens, job.count - t0);
        __syncthreads();
        stage_rows<I>(xs, act, tokens, [&](int j) { return sorted_assign[job.begin + t0 + j]; });
        __syncthreads();
        float acc[kGemvTokens] = {};
        row_share_dot(share, xs, sub, tokens, acc);
#pragma unroll
        for (int j = 0; j < kGemvTokens; ++j) {
            if (j >= tokens) { break; }
            const float sum = warp_sum<kDownLanes>(acc[j]) * inv;
            if (sub == 0) {
                const std::int64_t a = sorted_assign[job.begin + t0 + j];
                partial[a * H + row] = sum;
            }
        }
    }
}

__global__ void moe_combine_kernel(const float* __restrict__ partial,
                                   const float* __restrict__ weights,
                                   const float* __restrict__ shared_gate,
                                   const __nv_bfloat16* __restrict__ shared,
                                   __nv_bfloat16* __restrict__ y, std::int64_t columns) {
    const std::int64_t total = columns * H;
    for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
         i < total; i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t t = i / H;
        const int h          = static_cast<int>(i % H);
        float sum            = 0.0F;
        for (int k = 0; k < K; ++k) { sum += weights[t * K + k] * partial[(t * K + k) * H + h]; }
        sum += shared_gate[t] * __bfloat162float(shared[i]);
        y[i] = __float2bfloat16_rn(sum);
    }
}

} // namespace

void moe_route_launch(const Tensor& x, const Tensor& router, float* logits, Tensor& ids,
                      Tensor& weights, Tensor& shared_gate, cudaStream_t stream) {
    const int columns = x.ne[1];
    const dim3 grid((E + 1 + kRouteWarps - 1) / kRouteWarps,
                    (columns + kRouteColumns - 1) / kRouteColumns);
    moe_route_logits_kernel<<<grid, kRouteWarps * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(router.data),
        logits, columns);
    CUDA_CHECK(cudaGetLastError());
    moe_route_topk_kernel<<<(columns + 7) / 8, 256, 0, stream>>>(
        logits, columns, static_cast<std::int32_t*>(ids.data), static_cast<float*>(weights.data),
        static_cast<float*>(shared_gate.data));
    CUDA_CHECK(cudaGetLastError());
}

void moe_route_share_padding_launch(Tensor& ids, const Tensor& valid_columns, std::int32_t width,
                                    cudaStream_t stream) {
    const auto lanes = static_cast<std::int32_t>(valid_columns.numel());
    moe_route_share_padding_kernel<<<lanes, kShareThreads, 0, stream>>>(
        static_cast<std::int32_t*>(ids.data), static_cast<const std::int32_t*>(valid_columns.data),
        width);
    CUDA_CHECK(cudaGetLastError());
}

void expert_cache_resolve_launch(const Tensor& ids, std::int32_t layer,
                                 const ExpertCacheState& cache, Tensor& slot_ids, Tensor& misses,
                                 cudaStream_t stream) {
    expert_cache_resolve_kernel<<<1, kResolveThreads, 0, stream>>>(
        static_cast<const std::int32_t*>(ids.data), static_cast<std::int32_t>(ids.numel()), layer,
        cache, static_cast<std::int32_t*>(slot_ids.data), static_cast<std::int32_t*>(misses.data));
    CUDA_CHECK(cudaGetLastError());
}

void expert_cache_stage_launch(const ExpertCacheState& cache, std::int32_t layer,
                               const std::int32_t* resident, const ExpertWeights& bank,
                               const ExpertWeights& staged, cudaStream_t stream) {
    expert_cache_stage_kernel<<<dim3(E, kFetchChunks), 256, 0, stream>>>(cache, layer, resident,
                                                                          bank, staged);
    CUDA_CHECK(cudaGetLastError());
}

void expert_cache_reclaim_launch(const ExpertCacheState& cache, std::int32_t first_slot,
                                 cudaStream_t stream) {
    const int count = cache.slots - first_slot;
    expert_cache_reclaim_kernel<<<(count + 255) / 256, 256, 0, stream>>>(cache, first_slot);
    CUDA_CHECK(cudaGetLastError());
}

void expert_cache_fetch_launch(const ExpertWeights& bank, const Tensor& misses,
                               std::int32_t max_misses, const ExpertCacheState& cache,
                               cudaStream_t stream) {
    expert_cache_fetch_kernel<<<kFetchCtas, 256, 0, stream>>>(
        bank, static_cast<const std::int32_t*>(misses.data), max_misses, cache.pool);
    CUDA_CHECK(cudaGetLastError());
}

namespace {

// Gate/up then down for the jobs `filter` admits.
void moe_project(const MoeChunk& chunk, JobFilter filter, cudaStream_t stream) {
    const int assignments = chunk.columns * K;
    if (assignments <= kGemvAssignments) {
        moe_gate_up_gemv_kernel<<<dim3(assignments, I / (kGemvWarps * kGateUpPairsPerWarp)),
                                  kGemvThreads, 0, stream>>>(
            chunk.x, chunk.expert_ids, chunk.sorted_assign, chunk.sorted_slot, chunk.jobs,
            chunk.job_count, assignments, filter, chunk.source, chunk.act);
        CUDA_CHECK(cudaGetLastError());
        moe_down_gemv_kernel<<<dim3(assignments, H / kDownRowsPerCta), kGemvThreads, 0, stream>>>(
            chunk.act, chunk.expert_ids, chunk.sorted_assign, chunk.sorted_slot, chunk.jobs,
            chunk.job_count, assignments, filter, chunk.source, chunk.partial);
        CUDA_CHECK(cudaGetLastError());
    } else {
        moe_gate_up_kernel<<<dim3(assignments, I / 32), 256, 0, stream>>>(
            chunk.x, chunk.expert_ids, chunk.sorted_assign, chunk.sorted_slot, chunk.jobs,
            chunk.job_count, assignments, filter, chunk.source, chunk.act);
        CUDA_CHECK(cudaGetLastError());
        moe_down_kernel<<<dim3(assignments, H / kTile), 256, 0, stream>>>(
            chunk.act, chunk.expert_ids, chunk.sorted_assign, chunk.sorted_slot, chunk.jobs,
            chunk.job_count, assignments, filter, chunk.source, chunk.partial);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace

void moe_experts_chunk_launch(const MoeChunk& chunk, cudaStream_t stream) {
    const int assignments = chunk.columns * K;
    moe_jobs_kernel<<<1, kJobThreads, 0, stream>>>(chunk.slot_ids, assignments, chunk.slots,
                                                   chunk.counts, chunk.sorted_assign,
                                                   chunk.sorted_slot, chunk.jobs, chunk.job_count);
    CUDA_CHECK(cudaGetLastError());
    if (chunk.misses == nullptr) {
        moe_project(chunk, JobFilter{nullptr, false}, stream);
    } else {
        // Cache-resident experts run while the fetch fills the missed slots; each assignment
        // writes its own partial row, so the split leaves every result unchanged.
        moe_project(chunk, JobFilter{chunk.misses, false}, stream);
        CUDA_CHECK(cudaStreamWaitEvent(stream, chunk.fetched));
        moe_project(chunk, JobFilter{chunk.misses, true}, stream);
    }
    const std::int64_t total = static_cast<std::int64_t>(chunk.columns) * H;
    const int grid = static_cast<int>(std::min<std::int64_t>((total + 255) / 256, 1 << 16));
    moe_combine_kernel<<<grid, 256, 0, stream>>>(chunk.partial, chunk.weights, chunk.shared_gate,
                                                 chunk.shared, chunk.y, chunk.columns);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
