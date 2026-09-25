// Implements: include/ninfer/ops/offload_moe.h (moe_experts_a4)
// The A4 prefill route of the offloaded Qwen4Exp MoE. A staged layer bank is one contiguous NVFP4
// plane per projection, so both projections run on nvfp4_w4a4_mma_kernel over a work list of
// (expert, 64-column tile) jobs:
//
//   * the chunk's activations are quantised once with the layer's gate/up input divisor;
//   * assignments are grouped by expert, so a job reads one expert's rows and only its columns;
//   * the gate/up epilogue applies SwiGLU and writes the intermediate as NVFP4 with the down input
//     divisor, so down needs no quantiser of its own;
//   * down writes one BF16 row per assignment, and the merge sums the ten routes in k order and
//     adds the gated shared expert, so the grouping order never changes a result.
#include "ops/offload_moe/launch.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include "ops/linear/nvfp4/nvfp4_geometry.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int H = kOffloadMoeHidden;
constexpr int E = kOffloadMoeExperts;
constexpr int K = kOffloadMoeTopK;
constexpr int I = kOffloadMoeIntermediate;

using GateUpGeometry = Nvfp4Geometry<E * 2 * I, H>;
using DownGeometry   = Nvfp4Geometry<E * H, I>;
using InputGeometry  = Nvfp4ActivationGeometry<H>;

// The routed schedule the 35B-A3B NVFP4 route settled on: a 128-deep tile keeps staging small
// enough for three CTAs an SM, and a job is one 64-column tile of one expert.
using Schedule = Nvfp4W4a4MmaSchedule<64, 128, 128, 2, 4, 2, 3>;
static_assert(Schedule::kBlockM == kA4JobColumns);
constexpr int kPairRows = Schedule::kBlockN / 2;
static_assert((kPairRows & (kPairRows - 1)) == 0 && I % kPairRows == 0);
static_assert(H % Schedule::kBlockN == 0);

// The grid is sized from a host bound on the work list; the real count lives on the device, so
// tiles past its end leave before they touch a weight plane.
struct Jobs {
    const std::int32_t* __restrict__ experts;
    const std::int32_t* __restrict__ columns;
    const std::int32_t* __restrict__ count;

    __device__ __forceinline__ int expert() const { return experts[blockIdx.y]; }
    __device__ __forceinline__ int column_base() const { return columns[blockIdx.y]; }
    __device__ __forceinline__ bool live() const { return static_cast<int>(blockIdx.y) < *count; }
};

struct GateUpRaster {
    Jobs jobs;

    __device__ __forceinline__ void blocks(int& block_row, int& block_token) const {
        block_row   = static_cast<int>(blockIdx.x);
        block_token = jobs.column_base() / Schedule::kBlockM;
    }
    __device__ __forceinline__ bool live() const { return jobs.live(); }
};

// Down rows are contiguous inside an expert, so folding the expert's row base into the block row
// lets the identity row policy and the contiguous scale staging address the plane.
struct DownRaster {
    Jobs jobs;

    __device__ __forceinline__ void blocks(int& block_row, int& block_token) const {
        block_row   = jobs.expert() * (H / Schedule::kBlockN) + static_cast<int>(blockIdx.x);
        block_token = jobs.column_base() / Schedule::kBlockM;
    }
    __device__ __forceinline__ bool live() const { return jobs.live(); }
};

// A pair tile stages kPairRows gate rows and the matching up rows of the job's expert; every
// expert's gate rows precede its up rows.
struct GateUpRows {
    static constexpr bool kContiguous = false;

    Jobs jobs;

    __device__ __forceinline__ int weight_row(int row_begin, int local_row) const {
        const int within =
            row_begin + (local_row & (kPairRows - 1)) + (local_row >= kPairRows ? I : 0);
        return jobs.expert() * (2 * I) + within;
    }
};

// gate/up reads the chunk's quantised activations through the grouping.
struct GatherTokens {
    const std::int32_t* __restrict__ packed_token;
    const std::int32_t* __restrict__ offsets;
    Jobs jobs;

    __device__ __forceinline__ int source_token(int column) const {
        return packed_token[offsets[jobs.expert()] + column];
    }
    __device__ __forceinline__ int active_tokens(int) const {
        const int expert = jobs.expert();
        return offsets[expert + 1] - offsets[expert];
    }
};

// down reads what gate/up wrote, already in grouped order.
struct PackedTokens {
    const std::int32_t* __restrict__ offsets;
    Jobs jobs;

    __device__ __forceinline__ int source_token(int column) const {
        return offsets[jobs.expert()] + column;
    }
    __device__ __forceinline__ int active_tokens(int) const {
        const int expert = jobs.expert();
        return offsets[expert + 1] - offsets[expert];
    }
};

// Each stacked source matrix carries its own weight divisor: gate and up of an expert (I rows
// each) and down of an expert (H rows).
struct DivisorEpilogue {
    const float* __restrict__ divisors;
    int divisor_rows;

    __device__ __forceinline__ float apply(std::int32_t row, std::int32_t, float value) const {
        return value * __frcp_rn(divisors[row / divisor_rows]);
    }
};

union Bf16Pair {
    unsigned bits;
    __nv_bfloat162 values;
};

__device__ __forceinline__ unsigned swiglu_pair(unsigned gate_bits, unsigned up_bits) {
    const Bf16Pair gate{gate_bits};
    const Bf16Pair up{up_bits};
    const float2 g = __bfloat1622float2(gate.values);
    const float2 u = __bfloat1622float2(up.values);
    Bf16Pair result;
    result.values = __floats2bfloat162_rn(silu(g.x) * u.x, silu(g.y) * u.y);
    return result.bits;
}

// One store is half an NVFP4 group of the intermediate; the group maximum is completed across
// the neighbouring lane, which always holds the same token.
struct GateUpOutput {
    std::uint8_t* codes;
    std::uint8_t* scales;
    const std::int32_t* __restrict__ offsets;
    Jobs jobs;
    float divisor;

    __device__ __forceinline__ void store_pair_vector(std::int32_t row, std::int32_t column,
                                                      uint4 gate, uint4 up) const {
        const std::int64_t packed   = offsets[jobs.expert()] + column;
        const std::uint32_t bits[4] = {swiglu_pair(gate.x, up.x), swiglu_pair(gate.y, up.y),
                                       swiglu_pair(gate.z, up.z), swiglu_pair(gate.w, up.w)};
        float2 values[4];
        float max_abs = 0.0F;
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            values[pair] = bf16x2_bits_to_float2(bits[pair]);
            max_abs      = fmaxf(max_abs, fmaxf(fabsf(values[pair].x), fabsf(values[pair].y)));
        }
        const unsigned pair_mask = 3U << (threadIdx.x & 31U & ~1U);
        max_abs                  = fmaxf(max_abs, __shfl_xor_sync(pair_mask, max_abs, 1));

        const float scale_unencoded = __fdiv_rn(divisor * max_abs, 6.0F);
        const std::uint8_t scale =
            __nv_cvt_float_to_fp8(scale_unencoded, __NV_SATFINITE, __NV_E4M3);
        std::uint32_t word = 0;
        if (scale != 0) {
            const float reciprocal = __frcp_rn(decode_nvfp4_e4m3(scale));
#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                values[pair].x = values[pair].x * divisor * reciprocal;
                values[pair].y = values[pair].y * divisor * reciprocal;
            }
            word = pack_nvfp4_e2m1x8(values);
        }
        const int half      = (row >> 3) & 1;
        const int group_row = row & ~15;
        store_vec(codes + packed * (I / 2) + (group_row >> 1) + half * 4, word);
        if (half == 0) { scales[packed * (I / 16) + (group_row >> 4)] = scale; }
    }
};

struct DownOutput {
    __nv_bfloat16* data;
    const std::int32_t* __restrict__ offsets;
    Jobs jobs;

    __device__ __forceinline__ void store_vector(std::int32_t row, std::int32_t column,
                                                 uint4 values) const {
        const int expert          = jobs.expert();
        const std::int64_t packed = offsets[expert] + column;
        store_vec(data + packed * H + (row - expert * H), values);
    }
};

// rank[a] orders assignment a among those of its expert.
__global__ void a4_count_kernel(const std::int32_t* __restrict__ ids, std::int32_t assignments,
                                std::int32_t* __restrict__ counts,
                                std::int32_t* __restrict__ rank) {
    const int a = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (a < assignments) { rank[a] = atomicAdd(&counts[ids[a]], 1); }
}

// One CTA of E threads: expert offsets and the (expert, column tile) work list.
__global__ void __launch_bounds__(E)
    a4_scan_kernel(const std::int32_t* __restrict__ counts, std::int32_t* __restrict__ offsets,
                   std::int32_t* __restrict__ job_experts, std::int32_t* __restrict__ job_columns,
                   std::int32_t* __restrict__ job_count) {
    __shared__ std::int32_t scan[E];
    const int expert = static_cast<int>(threadIdx.x);
    const int count  = counts[expert];
    const int jobs   = (count + kA4JobColumns - 1) / kA4JobColumns;
    for (int pass = 0; pass < 2; ++pass) {
        const int own = pass == 0 ? count : jobs;
        scan[expert]  = own;
        __syncthreads();
        for (int offset = 1; offset < E; offset <<= 1) {
            const int add = expert >= offset ? scan[expert - offset] : 0;
            __syncthreads();
            scan[expert] += add;
            __syncthreads();
        }
        const int exclusive = scan[expert] - own;
        if (pass == 0) {
            offsets[expert] = exclusive;
            if (expert == E - 1) { offsets[E] = scan[expert]; }
        } else {
            for (int job = 0; job < jobs; ++job) {
                job_experts[exclusive + job] = expert;
                job_columns[exclusive + job] = job * kA4JobColumns;
            }
            if (expert == E - 1) { *job_count = scan[expert]; }
        }
        __syncthreads();
    }
}

__global__ void a4_place_kernel(const std::int32_t* __restrict__ ids,
                                const std::int32_t* __restrict__ rank, std::int32_t assignments,
                                const std::int32_t* __restrict__ offsets,
                                std::int32_t* __restrict__ packed_token,
                                std::int32_t* __restrict__ packed_index) {
    const int a = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (a >= assignments) { return; }
    const int position     = offsets[ids[a]] + rank[a];
    packed_token[position] = a / K;
    packed_index[a]        = position;
}

// One CTA per column; thread i owns features [8i, 8i+8).
__global__ void __launch_bounds__(H / 8)
    a4_merge_kernel(const __nv_bfloat16* __restrict__ grouped,
                    const std::int32_t* __restrict__ packed_index,
                    const float* __restrict__ weights, const float* __restrict__ shared_gate,
                    const __nv_bfloat16* __restrict__ shared, __nv_bfloat16* __restrict__ y) {
    const int column  = static_cast<int>(blockIdx.x);
    const int feature = static_cast<int>(threadIdx.x) * 8;
    float sum[8]      = {};
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int row      = packed_index[column * K + k];
        const float weight = weights[column * K + k];
        const uint4 raw = load_vec<uint4>(grouped + static_cast<std::int64_t>(row) * H + feature);
        const std::uint32_t words[4]{raw.x, raw.y, raw.z, raw.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 v = bf16x2_bits_to_float2(words[i]);
            sum[2 * i]     = fmaf(weight, v.x, sum[2 * i]);
            sum[2 * i + 1] = fmaf(weight, v.y, sum[2 * i + 1]);
        }
    }
    const float gate        = shared_gate[column];
    const std::int64_t base = static_cast<std::int64_t>(column) * H + feature;
    const uint4 shared_raw  = load_vec<uint4>(shared + base);
    const std::uint32_t words[4]{shared_raw.x, shared_raw.y, shared_raw.z, shared_raw.w};
    std::uint32_t out[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 s = bf16x2_bits_to_float2(words[i]);
        Bf16Pair pair;
        pair.values =
            __floats2bfloat162_rn(fmaf(gate, s.x, sum[2 * i]), fmaf(gate, s.y, sum[2 * i + 1]));
        out[i] = pair.bits;
    }
    store_vec(y + base, make_uint4(out[0], out[1], out[2], out[3]));
}

} // namespace

void moe_experts_a4_launch(const MoeA4Chunk& c, cudaStream_t stream) {
    const int assignments = c.columns * K;
    const int blocks      = (assignments + 255) / 256;
    CUDA_CHECK(cudaMemsetAsync(c.counts, 0, sizeof(std::int32_t) * E, stream));
    a4_count_kernel<<<blocks, 256, 0, stream>>>(c.expert_ids, assignments, c.counts, c.rank);
    CUDA_CHECK(cudaGetLastError());
    a4_scan_kernel<<<1, E, 0, stream>>>(c.counts, c.offsets, c.job_experts, c.job_columns,
                                        c.job_count);
    CUDA_CHECK(cudaGetLastError());
    a4_place_kernel<<<blocks, 256, 0, stream>>>(c.expert_ids, c.rank, assignments, c.offsets,
                                                c.packed_token, c.packed_index);
    CUDA_CHECK(cudaGetLastError());

    constexpr int kQuantizeThreads = 256;
    const int groups               = c.columns * InputGeometry::kGroupsPerRow;
    nvfp4_w4a4_quantize_kernel<InputGeometry, kQuantizeThreads, Nvfp4ScaleLayout::RowMajor>
        <<<(groups + kQuantizeThreads - 1) / kQuantizeThreads, kQuantizeThreads, 0, stream>>>(
            c.x, c.input_codes, c.input_scales, c.columns, c.columns,
            c.source.gate_up_input_divisor);
    CUDA_CHECK(cudaGetLastError());

    const Jobs jobs{c.job_experts, c.job_columns, c.job_count};
    const int max_jobs = a4_max_jobs(c.columns);
    {
        const Nvfp4W4a4MaterializedActivation input{c.input_codes, c.input_scales};
        const GateUpOutput output{c.middle_codes, c.middle_scales, c.offsets, jobs,
                                  c.source.down_input_divisor};
        const DivisorEpilogue epilogue{c.source.gate_up_divisors, c.source.gate_up_divisor_rows};
        const dim3 grid(I / kPairRows, max_jobs);
        nvfp4_w4a4_mma_kernel<GateUpGeometry, Schedule, DivisorEpilogue, GateUpOutput, GateUpRows,
                              true, GatherTokens, GateUpRaster>
            <<<grid, Schedule::kThreads, 0, stream>>>(
                input, reinterpret_cast<const std::uint8_t*>(c.source.base[0]),
                reinterpret_cast<const std::uint8_t*>(c.source.base[1]), assignments,
                1.0F / c.source.gate_up_input_divisor, epilogue, output, GateUpRows{jobs},
                GatherTokens{c.packed_token, c.offsets, jobs}, GateUpRaster{jobs});
        CUDA_CHECK(cudaGetLastError());
    }
    {
        const Nvfp4W4a4MaterializedActivation middle{c.middle_codes, c.middle_scales};
        const DownOutput output{c.grouped, c.offsets, jobs};
        const DivisorEpilogue epilogue{c.source.down_divisors, c.source.down_divisor_rows};
        const dim3 grid(H / Schedule::kBlockN, max_jobs);
        nvfp4_w4a4_mma_kernel<DownGeometry, Schedule, DivisorEpilogue, DownOutput,
                              Nvfp4W4a4IdentityRows, false, PackedTokens, DownRaster>
            <<<grid, Schedule::kThreads, 0, stream>>>(
                middle, reinterpret_cast<const std::uint8_t*>(c.source.base[2]),
                reinterpret_cast<const std::uint8_t*>(c.source.base[3]), assignments,
                1.0F / c.source.down_input_divisor, epilogue, output, Nvfp4W4a4IdentityRows{},
                PackedTokens{c.offsets, jobs}, DownRaster{jobs});
        CUDA_CHECK(cudaGetLastError());
    }
    a4_merge_kernel<<<c.columns, H / 8, 0, stream>>>(c.grouped, c.packed_index, c.weights,
                                                     c.shared_gate, c.shared, c.y);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
