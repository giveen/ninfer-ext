#pragma once

// Small-T NVFP4 weight x BF16 activation Tensor Core kernel.
//
//   out[row, t] = inverse_weight_divisor * sum_k (e2m1[row, k] * e4m3[row, k / 16]) * x[k, t]
//
// Every code times its block scale is exact in BF16 (at most six significant bits, inside BF16
// range), so the weight operand is widened exactly and activations stay BF16. Tensor Cores
// accumulate in FP32 and the tensor divisor is applied once before the tile's epilogue.
//
// The kernel keeps both operands in registers. Inside each 128-wide K chunk a lane owns 32
// contiguous K values: one 16-byte code load per weight row and four 16-byte activation loads per
// token. The m16n8k16 K slots of that lane are mapped onto its own contiguous values, the same
// permutation for A and B, so the reduction is unchanged while no operand goes through shared
// memory. K chunks are split across the warps of a CTA and reduced through shared memory at the
// end; the cost per weight stays at one widening regardless of T, which is what keeps T = 2..8 at
// the single-token weight-streaming rate.
//
// A tile policy names the physical weight row behind each MMA row and consumes the reduced
// accumulators. Rows are chosen so that the rows of one CTA share whole 16-byte lines of the
// K16M128x4 scale plane.

#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

template <int RowTiles, int TokenTiles, int KWarps>
struct Nvfp4A16MmaSchedule {
    static constexpr int kRowTiles   = RowTiles;
    static constexpr int kTokenTiles = TokenTiles;
    static constexpr int kKWarps     = KWarps;
    static constexpr int kThreads    = KWarps * 32;
    static constexpr int kTokens     = TokenTiles * 8;
    static_assert(RowTiles > 0 && TokenTiles > 0 && KWarps > 0 && kThreads <= 1024);
};

// The rows of `kBlockRows` consecutive logical rows inside one 128-row scale tile, laid out as
// kBlockRows / 4 row residues times the four 32-row quartiles that share a scale line.
template <int BlockRows>
__device__ __forceinline__ int nvfp4_a16_mma_scale_line_row(int block, int index) {
    static_assert(BlockRows >= 4 && (BlockRows % 4) == 0 && (128 % BlockRows) == 0);
    constexpr int kResidues = BlockRows / 4;
    const int m_tile        = block / (128 / BlockRows);
    const int residue_base  = (block % (128 / BlockRows)) * kResidues;
    return m_tile * 128 + residue_base + (index % kResidues) + 32 * (index / kResidues);
}

__device__ __forceinline__ unsigned nvfp4_a16_widen_pair(std::uint32_t word, int shift,
                                                         float scale) {
    const float2 value = decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(word >> shift));
    const __nv_bfloat162 widened = __floats2bfloat162_rn(value.x * scale, value.y * scale);
    return *reinterpret_cast<const unsigned*>(&widened);
}

// Tile contract:
//   static constexpr int kBlocks;                          grid size
//   __device__ int row(int block, int mma_row) const;      mma_row = row_tile * 16 + 0..15
//   __device__ void finish(const int (&rows)[RowTiles][2],
//                          const float (&values)[RowTiles][TokenTiles][4],
//                          int lane_token, int tokens) const;
// `values` follow the m16n8 accumulator layout: [0], [1] belong to rows[..][0] and tokens
// lane_token, lane_token + 1 of each 8-token tile; [2], [3] to rows[..][1].
template <class Geometry, class Schedule, class Tile>
__global__ __launch_bounds__(Schedule::kThreads) void nvfp4_a16_mma_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, float inverse_weight_divisor, Tile tile,
    std::int32_t tokens) {
    constexpr int K       = Geometry::kInputRows;
    constexpr int kChunks = K / 128;
    constexpr int MT      = Schedule::kRowTiles;
    constexpr int NT      = Schedule::kTokenTiles;
    constexpr int WK      = Schedule::kKWarps;
    static_assert((K % 128) == 0);

    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int gid   = lane >> 2;
    const int lid   = lane & 3;
    const int block = static_cast<int>(blockIdx.x);

    int rows[MT][2];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        rows[mt][0] = tile.row(block, mt * 16 + gid);
        rows[mt][1] = tile.row(block, mt * 16 + gid + 8);
    }

    float accumulators[MT][NT][4] = {};

#pragma unroll 1
    for (int chunk = warp; chunk < kChunks; chunk += WK) {
        const int k_begin = chunk * 128 + lid * 32;
        uint4 row_codes[MT][2];
        std::uint16_t row_scales[MT][2];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                row_codes[mt][half] =
                    load_vec<uint4>(codes +
                                    static_cast<std::int64_t>(rows[mt][half]) *
                                        Geometry::kCodeBytesPerRow +
                                    k_begin / 2);
                // Groups k_begin / 16 and the next one are adjacent bytes of one scale word.
                row_scales[mt][half] = *reinterpret_cast<const std::uint16_t*>(
                    scales +
                    nvfp4_scale_byte<Geometry::kScaleTilesPerRow>(rows[mt][half], k_begin / 16));
            }
        }
        uint4 activation[NT][4];
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int token = nt * 8 + gid;
#pragma unroll
            for (int part = 0; part < 4; ++part) {
                activation[nt][part] =
                    token < tokens
                        ? load_vec<uint4>(x + static_cast<std::int64_t>(token) * K + k_begin +
                                          part * 8)
                        : make_uint4(0U, 0U, 0U, 0U);
            }
        }

#pragma unroll
        for (int group = 0; group < 2; ++group) {
            float scale[MT][2];
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    scale[mt][half] = decode_nvfp4_e4m3(
                        static_cast<std::uint8_t>(row_scales[mt][half] >> (8 * group)));
                }
            }
#pragma unroll
            for (int group_step = 0; group_step < 4; ++group_step) {
                // Step s covers this lane's values 4s..4s+3: code bytes 2s and 2s + 1.
                const int step  = group * 4 + group_step;
                const int word  = step / 2;
                const int shift = (step & 1) * 16;
                unsigned a[MT][4];
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    const auto* first  = reinterpret_cast<const std::uint32_t*>(&row_codes[mt][0]);
                    const auto* second = reinterpret_cast<const std::uint32_t*>(&row_codes[mt][1]);
                    a[mt][0] = nvfp4_a16_widen_pair(first[word], shift, scale[mt][0]);
                    a[mt][1] = nvfp4_a16_widen_pair(second[word], shift, scale[mt][1]);
                    a[mt][2] = nvfp4_a16_widen_pair(first[word], shift + 8, scale[mt][0]);
                    a[mt][3] = nvfp4_a16_widen_pair(second[word], shift + 8, scale[mt][1]);
                }
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    const auto* pairs = reinterpret_cast<const unsigned*>(activation[nt]);
                    const unsigned b0 = pairs[2 * step];
                    const unsigned b1 = pairs[2 * step + 1];
#pragma unroll
                    for (int mt = 0; mt < MT; ++mt) {
                        mma_bf16(accumulators[mt][nt][0], accumulators[mt][nt][1],
                                 accumulators[mt][nt][2], accumulators[mt][nt][3], a[mt][0],
                                 a[mt][1], a[mt][2], a[mt][3], b0, b1);
                    }
                }
            }
        }
    }

    if constexpr (WK > 1) {
        __shared__ float partials[WK][MT][NT][4][32];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
                for (int e = 0; e < 4; ++e) { partials[warp][mt][nt][e][lane] = accumulators[mt][nt][e]; }
            }
        }
        __syncthreads();
        if (warp != 0) { return; }
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    float total = 0.0F;
#pragma unroll
                    for (int source = 0; source < WK; ++source) {
                        total += partials[source][mt][nt][e][lane];
                    }
                    accumulators[mt][nt][e] = total;
                }
            }
        }
    }

#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) { accumulators[mt][nt][e] *= inverse_weight_divisor; }
        }
    }
    tile.finish(rows, accumulators, 2 * lid, tokens);
}

// Plain rows: the block's 16 * RowTiles rows share scale lines; each output goes through the
// caller's epilogue and output policy.
template <class Geometry, class Schedule, class Epilogue, class Output>
struct Nvfp4A16MmaRowTile {
    static constexpr int kBlockRows = Schedule::kRowTiles * 16;
    static constexpr int kBlocks    = Geometry::kOutputRows / kBlockRows;
    static_assert((Geometry::kOutputRows % 128) == 0);

    Epilogue epilogue;
    Output output;

    __device__ __forceinline__ int row(int block, int mma_row) const {
        return nvfp4_a16_mma_scale_line_row<kBlockRows>(block, mma_row);
    }

    __device__ __forceinline__ void
    finish(const int (&rows)[Schedule::kRowTiles][2],
           const float (&values)[Schedule::kRowTiles][Schedule::kTokenTiles][4], int lane_token,
           int tokens) const {
#pragma unroll
        for (int mt = 0; mt < Schedule::kRowTiles; ++mt) {
#pragma unroll
            for (int nt = 0; nt < Schedule::kTokenTiles; ++nt) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int row   = rows[mt][e >> 1];
                    const int token = nt * 8 + lane_token + (e & 1);
                    if (token < tokens) {
                        output.store(row, token, epilogue.apply(row, token, values[mt][nt][e]));
                    }
                }
            }
        }
    }
};

} // namespace ninfer::ops::detail
