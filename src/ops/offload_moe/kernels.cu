// Implements: include/ninfer/ops/offload_moe.h
// Routed experts for the offloaded Qwen4Exp MoE. Weights stay NVFP4 in their slot or staging
// buffer and are decoded in registers; activations stay BF16 (A16). Every assignment (k,t) owns
// its own intermediate and partial output, and the merge sums k in order, so cache placement and
// job order never change a result.
#include "ops/offload_moe/launch.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"

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
constexpr unsigned long long kProtected = ~0ULL;

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
// Routing: one CTA per column; one warp per router row.

__global__ void __launch_bounds__(256)
    moe_route_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ router,
                     std::int32_t* __restrict__ ids, float* __restrict__ weights,
                     float* __restrict__ shared_gate) {
    __shared__ float xs[H];
    __shared__ float logits[E + 1];
    __shared__ int taken[E];
    const int tid          = static_cast<int>(threadIdx.x);
    const int warp         = tid >> 5;
    const int lane         = tid & 31;
    const std::int64_t col = blockIdx.x;
    for (int i = tid; i < H; i += blockDim.x) { xs[i] = __bfloat162float(x[col * H + i]); }
    for (int e = tid; e < E; e += blockDim.x) { taken[e] = 0; }
    __syncthreads();
    for (int row = warp; row <= E; row += blockDim.x / 32) {
        const __nv_bfloat16* w = router + static_cast<std::int64_t>(row) * H;
        float sum              = 0.0F;
        for (int i = lane; i < H; i += 32) { sum += __bfloat162float(w[i]) * xs[i]; }
        sum = warp_sum(sum);
        if (lane == 0) { logits[row] = sum; }
    }
    __syncthreads();
    if (warp != 0) { return; }
    float top[K];
    int top_id[K];
    for (int k = 0; k < K; ++k) {
        float best  = -CUDART_INF_F;
        int best_id = E;
        for (int e = lane; e < E; e += 32) {
            const float v = logits[e];
            if (!taken[e] && (v > best || (v == best && e < best_id))) {
                best    = v;
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
        if (lane == 0) { taken[best_id] = 1; }
        __syncwarp();
    }
    if (lane == 0) {
        float total = 0.0F;
        for (int k = 0; k < K; ++k) { total += __expf(top[k] - top[0]); }
        for (int k = 0; k < K; ++k) {
            ids[col * K + k]     = top_id[k];
            weights[col * K + k] = __expf(top[k] - top[0]) / total;
        }
        shared_gate[col] = sigmoid(logits[E]);
    }
}

// ---------------------------------------------------------------------------------------------
// Cache resolution: one CTA of 1024 threads per call.

constexpr int kResolveThreads = 1024;

__device__ __forceinline__ int block_exclusive_scan(int value, int* scratch, int& total) {
    const int tid = static_cast<int>(threadIdx.x);
    scratch[tid]  = value;
    __syncthreads();
    for (int offset = 1; offset < static_cast<int>(blockDim.x); offset <<= 1) {
        const int add = tid >= offset ? scratch[tid - offset] : 0;
        __syncthreads();
        scratch[tid] += add;
        __syncthreads();
    }
    total            = scratch[blockDim.x - 1];
    const int result = scratch[tid] - value;
    __syncthreads();
    return result;
}

__global__ void __launch_bounds__(kResolveThreads)
    expert_cache_resolve_kernel(const std::int32_t* __restrict__ ids, std::int32_t assignments,
                                std::int32_t layer, ExpertCacheState cache,
                                std::int32_t* __restrict__ slot_ids,
                                std::int32_t* __restrict__ misses) {
    __shared__ int used[E];
    __shared__ int slot_for[E];
    __shared__ int miss_expert[E];
    __shared__ int scratch[kResolveThreads];
    __shared__ unsigned long long now_s;
    __shared__ unsigned long long prefix_s;
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
        // Radix-select the need_total-th smallest stamp among unprotected slots.
        if (tid == 0) {
            prefix_s    = 0ULL;
            remaining_s = need_total;
        }
        __syncthreads();
        unsigned long long mask = 0ULL;
        for (int shift = 56; shift >= 0; shift -= 8) {
            for (int b = tid; b < 256; b += blockDim.x) { histogram[b] = 0U; }
            __syncthreads();
            for (int s = tid; s < cache.slots; s += blockDim.x) {
                const unsigned long long stamp = cache.stamp[s];
                const unsigned long long key   = stamp >= now ? kProtected : stamp;
                if ((key & mask) == prefix_s) {
                    atomicAdd(&histogram[(key >> shift) & 0xFFULL], 1U);
                }
            }
            __syncthreads();
            if (tid == 0) {
                int need = remaining_s;
                for (int bin = 0; bin < 256; ++bin) {
                    const int count = static_cast<int>(histogram[bin]);
                    if (count >= need) {
                        prefix_s |= static_cast<unsigned long long>(bin) << shift;
                        remaining_s = need;
                        break;
                    }
                    need -= count;
                }
            }
            mask |= 0xFFULL << shift;
            __syncthreads();
        }
        const unsigned long long threshold = prefix_s;
        if (tid == 0) {
            emitted_s = 0;
            ties_s    = remaining_s;
        }
        __syncthreads();
        for (int base = 0; base < cache.slots; base += blockDim.x) {
            const int s            = base + tid;
            unsigned long long key = kProtected;
            if (s < cache.slots) {
                const unsigned long long stamp = cache.stamp[s];
                key                            = stamp >= now ? kProtected : stamp;
            }
            const int less     = s < cache.slots && key < threshold ? 1 : 0;
            const int tie      = s < cache.slots && key == threshold && key != kProtected ? 1 : 0;
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
// Fetch: grid (max_misses, chunks); every CTA copies one chunk of one missing expert.

constexpr int kFetchChunks = 32;

__global__ void __launch_bounds__(256)
    expert_cache_fetch_kernel(ExpertWeights bank, const std::int32_t* __restrict__ misses,
                              std::byte* pool) {
    const int j = static_cast<int>(blockIdx.x);
    if (j >= misses[0]) { return; }
    const std::int64_t slot     = misses[1 + 2 * j];
    const std::int64_t expert   = misses[2 + 2 * j];
    const std::int64_t sizes[4] = {kExpertGateUpCodeBytes, kExpertGateUpScaleBytes,
                                   kExpertDownCodeBytes, kExpertDownScaleBytes};
    std::int64_t plane_offset   = 0;
    auto* target                = pool + slot * kExpertSlotBytes;
    for (int p = 0; p < 4; ++p) {
        const std::int64_t vectors = sizes[p] / 16;
        const auto* src = reinterpret_cast<const uint4*>(bank.base[p] + expert * bank.stride[p]);
        auto* dst       = reinterpret_cast<uint4*>(target + plane_offset);
        const std::int64_t per   = (vectors + kFetchChunks - 1) / kFetchChunks;
        const std::int64_t begin = static_cast<std::int64_t>(blockIdx.y) * per;
        const std::int64_t end   = min(vectors, begin + per);
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

__device__ __forceinline__ JobView load_job(const std::int32_t* jobs, const std::int32_t* job_count,
                                            const std::int32_t* sorted_slot,
                                            const std::int32_t* sorted_assign,
                                            const std::int32_t* expert_ids, int assignments) {
    JobView view{-1, 0, -1, -1};
    const int job = static_cast<int>(blockIdx.x);
    if (job >= *job_count) { return view; }
    view.begin = jobs[job];
    view.slot  = sorted_slot[view.begin];
    int count  = 0;
    while (count < kJobTokens && view.begin + count < assignments &&
           sorted_slot[view.begin + count] == view.slot) {
        ++count;
    }
    view.count  = count;
    view.expert = expert_ids[sorted_assign[view.begin]];
    return view;
}

__global__ void __launch_bounds__(256) moe_gate_up_kernel(
    const __nv_bfloat16* __restrict__ x, const std::int32_t* __restrict__ expert_ids,
    const std::int32_t* __restrict__ sorted_assign, const std::int32_t* __restrict__ sorted_slot,
    const std::int32_t* __restrict__ jobs, const std::int32_t* __restrict__ job_count,
    std::int32_t assignments, ExpertWeights weights, __nv_bfloat16* __restrict__ act) {
    __shared__ float xs[kJobTokens][kTile + 1];
    __shared__ float ws[kTile][kTile + 1];
    const JobView job =
        load_job(jobs, job_count, sorted_slot, sorted_assign, expert_ids, assignments);
    if (job.count == 0) { return; }
    const int tid  = static_cast<int>(threadIdx.x);
    const int ty   = tid / 16;                          // rows ty*4 .. ty*4+3 of the 64-row tile
    const int tx   = tid % 16;                          // tokens tx*4 .. tx*4+3
    const int row0 = static_cast<int>(blockIdx.y) * 32; // gate rows row0..row0+31, up +I
    const auto* codes =
        reinterpret_cast<const std::uint8_t*>(weights.base[0] + job.slot * weights.stride[0]);
    const auto* scale =
        reinterpret_cast<const std::uint8_t*>(weights.base[1] + job.slot * weights.stride[1]);
    const float inv_gate = 1.0F / weights.divisors[job.expert * 3 + 0];
    const float inv_up   = 1.0F / weights.divisors[job.expert * 3 + 1];
    float acc[4][4]      = {};
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
    std::int32_t assignments, ExpertWeights weights, float* __restrict__ partial) {
    __shared__ float xs[kJobTokens][kTile + 1];
    __shared__ float ws[kTile][kTile + 1];
    const JobView job =
        load_job(jobs, job_count, sorted_slot, sorted_assign, expert_ids, assignments);
    if (job.count == 0) { return; }
    const int tid  = static_cast<int>(threadIdx.x);
    const int ty   = tid / 16;
    const int tx   = tid % 16;
    const int row0 = static_cast<int>(blockIdx.y) * kTile;
    const auto* codes =
        reinterpret_cast<const std::uint8_t*>(weights.base[2] + job.slot * weights.stride[2]);
    const auto* scale =
        reinterpret_cast<const std::uint8_t*>(weights.base[3] + job.slot * weights.stride[3]);
    const float inv = 1.0F / weights.divisors[job.expert * 3 + 2];
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

void moe_route_launch(const Tensor& x, const Tensor& router, Tensor& ids, Tensor& weights,
                      Tensor& shared_gate, cudaStream_t stream) {
    const int columns = x.ne[1];
    moe_route_kernel<<<columns, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(router.data),
        static_cast<std::int32_t*>(ids.data), static_cast<float*>(weights.data),
        static_cast<float*>(shared_gate.data));
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

void expert_cache_fetch_launch(const ExpertWeights& bank, const Tensor& misses,
                               std::int32_t max_misses, const ExpertCacheState& cache,
                               cudaStream_t stream) {
    expert_cache_fetch_kernel<<<dim3(max_misses, kFetchChunks), 256, 0, stream>>>(
        bank, static_cast<const std::int32_t*>(misses.data), cache.pool);
    CUDA_CHECK(cudaGetLastError());
}

void moe_experts_chunk_launch(const MoeChunk& chunk, cudaStream_t stream) {
    const int assignments = chunk.columns * K;
    moe_jobs_kernel<<<1, kJobThreads, 0, stream>>>(chunk.slot_ids, assignments, chunk.slots,
                                                   chunk.counts, chunk.sorted_assign,
                                                   chunk.sorted_slot, chunk.jobs, chunk.job_count);
    CUDA_CHECK(cudaGetLastError());
    moe_gate_up_kernel<<<dim3(assignments, I / 32), 256, 0, stream>>>(
        chunk.x, chunk.expert_ids, chunk.sorted_assign, chunk.sorted_slot, chunk.jobs,
        chunk.job_count, assignments, chunk.source, chunk.act);
    CUDA_CHECK(cudaGetLastError());
    moe_down_kernel<<<dim3(assignments, H / kTile), 256, 0, stream>>>(
        chunk.act, chunk.expert_ids, chunk.sorted_assign, chunk.sorted_slot, chunk.jobs,
        chunk.job_count, assignments, chunk.source, chunk.partial);
    CUDA_CHECK(cudaGetLastError());
    const std::int64_t total = static_cast<std::int64_t>(chunk.columns) * H;
    const int grid = static_cast<int>(std::min<std::int64_t>((total + 255) / 256, 1 << 16));
    moe_combine_kernel<<<grid, 256, 0, stream>>>(chunk.partial, chunk.weights, chunk.shared_gate,
                                                 chunk.shared, chunk.y, chunk.columns);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
