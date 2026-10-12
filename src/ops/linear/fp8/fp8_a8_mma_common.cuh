#pragma once

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/linear/common/vector_output.cuh"
#include "ops/linear/fp8/fp8_schedule.cuh"

namespace ninfer::ops::detail {

// Epilogues may consume adjacent aligned rows together; the scalar contract remains valid.
template <class Epilogue>
__device__ __forceinline__ float2 fp8_apply_row_pair(Epilogue epilogue, int row, int next_row,
                                                     int token, float2 value) {
    if constexpr (requires { epilogue.apply_row_pair(row, token, value); }) {
        if (next_row == row + 1) return epilogue.apply_row_pair(row, token, value);
    }
    return make_float2(epilogue.apply(row, token, value.x),
                       epilogue.apply(next_row, token, value.y));
}

template <class Schedule>
__device__ __forceinline__ int fp8_mma_shared_byte(int row, int logical_byte) {
    if constexpr (Schedule::kTmaSwizzle) {
        // TMA swizzles bits from the byte address, including the row's K stride.
        return logical_byte ^
               ((((row * Schedule::kBlockK + logical_byte) >> 7) & (Schedule::kSegmentsPerRow - 1))
                << 4);
    }
    const int logical_segment  = logical_byte >> 4;
    const int byte_in_segment  = logical_byte & 15;
    const int physical_segment = logical_segment ^ (row & (Schedule::kSegmentsPerRow - 1));
    return physical_segment * 16 + byte_in_segment;
}

template <class Schedule>
__device__ __forceinline__ void
fp8_mma_tile_coordinates(std::int32_t linear, std::int32_t row_tiles, std::int32_t token_tiles,
                         std::int32_t& row_tile, std::int32_t& token_tile) {
    if constexpr (Schedule::kRaster == Fp8MmaRaster::RowFast) {
        token_tile = linear / row_tiles;
        row_tile   = linear - token_tile * row_tiles;
    } else if constexpr (Schedule::kRaster == Fp8MmaRaster::TokenFast) {
        row_tile   = linear / token_tiles;
        token_tile = linear - row_tile * token_tiles;
    } else {
        constexpr int group_rows       = Schedule::kRasterGroupRows;
        const std::int32_t group_span  = group_rows * token_tiles;
        const std::int32_t group       = linear / group_span;
        const std::int32_t first_row   = group * group_rows;
        const std::int32_t active_rows = min(group_rows, row_tiles - first_row);
        const std::int32_t within      = linear - group * group_span;
        row_tile                       = first_row + within % active_rows;
        token_tile                     = within / active_rows;
    }
}

// `Terms` E4M3 activation planes share one weight load: each term's A fragments feed their own
// accumulator set, so a two-term route issues twice the MMAs per staged weight tile. Accumulators
// are indexed `[term * kMmaTokens + mma_token]`, keeping one flat set per term.
template <class Schedule, int Terms = 1>
__device__ __forceinline__ void
fp8_mma_compute_stage(const std::uint8_t* activation_stage, const std::uint8_t* activation_lo_stage,
                      const std::uint8_t* weight_stage,
                      float (&accumulators)[Terms * Schedule::kMmaTokens][Schedule::kMmaRows][4],
                      int warp, int lane) {
    static_assert(Terms == 1 || Terms == 2);
    constexpr int BK        = Schedule::kBlockK;
    constexpr int Tokens    = Schedule::kMmaTokens;
    constexpr int Rows      = Schedule::kMmaRows;
    const int warp_token    = warp / Schedule::kWarpsRows;
    const int warp_row      = warp % Schedule::kWarpsRows;
    const int a_matrix      = lane >> 3;
    const int a_row_offset  = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_column_byte = (a_matrix >> 1) * 16;
    const int b_row_offset  = lane & 7;
    const int b_column_byte = ((lane >> 3) & 1) * 16;
    auto activation_plane   = [&](int term) {
        if constexpr (Terms == 1) {
            (void)term;
            return activation_stage;
        } else {
            return term == 0 ? activation_stage : activation_lo_stage;
        }
    };
    auto load_fragments = [&](int k_step, unsigned (&a_fragments)[Terms][Tokens][4],
                              unsigned (&b_fragments)[Rows][2]) {
#pragma unroll
        for (int term = 0; term < Terms; ++term) {
            const std::uint8_t* plane = activation_plane(term);
#pragma unroll
            for (int mma_token = 0; mma_token < Tokens; ++mma_token) {
                const int row = warp_token * Schedule::kWarpTokens + mma_token * 16 + a_row_offset;
                const int logical_byte  = k_step * 32 + a_column_byte;
                const int physical_byte = fp8_mma_shared_byte<Schedule>(row, logical_byte);
                ldmatrix_x4(a_fragments[term][mma_token][0], a_fragments[term][mma_token][1],
                            a_fragments[term][mma_token][2], a_fragments[term][mma_token][3],
                            smem_addr(plane + row * BK + physical_byte));
            }
        }
#pragma unroll
        for (int mma_row = 0; mma_row < Rows; ++mma_row) {
            const int row           = warp_row * Schedule::kWarpRows + mma_row * 8 + b_row_offset;
            const int logical_byte  = k_step * 32 + b_column_byte;
            const int physical_byte = fp8_mma_shared_byte<Schedule>(row, logical_byte);
            ldmatrix_x2(b_fragments[mma_row][0], b_fragments[mma_row][1],
                        smem_addr(weight_stage + row * BK + physical_byte));
        }
    };

    if constexpr (Schedule::kFragmentPipeline == Fp8MmaFragmentPipeline::PingPong) {
        unsigned a_fragments[2][Terms][Tokens][4];
        unsigned b_fragments[2][Rows][2];
        load_fragments(0, a_fragments[0], b_fragments[0]);
#pragma unroll
        for (int k_step = 0; k_step < Schedule::kMmaK; ++k_step) {
            const int slot = k_step & 1;
            if (k_step + 1 < Schedule::kMmaK) {
                load_fragments(k_step + 1, a_fragments[slot ^ 1], b_fragments[slot ^ 1]);
            }
#pragma unroll
            for (int term = 0; term < Terms; ++term) {
#pragma unroll
                for (int mma_token = 0; mma_token < Tokens; ++mma_token) {
#pragma unroll
                    for (int mma_row = 0; mma_row < Rows; ++mma_row) {
                        const int accumulators_row = term * Tokens + mma_token;
                        mma_fp8_e4m3(accumulators[accumulators_row][mma_row][0],
                                     accumulators[accumulators_row][mma_row][1],
                                     accumulators[accumulators_row][mma_row][2],
                                     accumulators[accumulators_row][mma_row][3],
                                     a_fragments[slot][term][mma_token][0],
                                     a_fragments[slot][term][mma_token][1],
                                     a_fragments[slot][term][mma_token][2],
                                     a_fragments[slot][term][mma_token][3],
                                     b_fragments[slot][mma_row][0], b_fragments[slot][mma_row][1]);
                    }
                }
            }
        }
    } else {
        unsigned a_fragments[Terms][Tokens][4];
        unsigned b_fragments[Rows][2];
#pragma unroll
        for (int k_step = 0; k_step < Schedule::kMmaK; ++k_step) {
            load_fragments(k_step, a_fragments, b_fragments);
#pragma unroll
            for (int term = 0; term < Terms; ++term) {
#pragma unroll
                for (int mma_token = 0; mma_token < Tokens; ++mma_token) {
#pragma unroll
                    for (int mma_row = 0; mma_row < Rows; ++mma_row) {
                        const int accumulators_row = term * Tokens + mma_token;
                        mma_fp8_e4m3(
                            accumulators[accumulators_row][mma_row][0],
                            accumulators[accumulators_row][mma_row][1],
                            accumulators[accumulators_row][mma_row][2],
                            accumulators[accumulators_row][mma_row][3],
                            a_fragments[term][mma_token][0], a_fragments[term][mma_token][1],
                            a_fragments[term][mma_token][2], a_fragments[term][mma_token][3],
                            b_fragments[mma_row][0], b_fragments[mma_row][1]);
                    }
                }
            }
        }
    }
}

template <class Schedule, bool FullTokens, class Epilogue, class Output, class RowPolicy,
          int Terms = 1>
__device__ __forceinline__ void
fp8_finish_mma_tile(Output output, Epilogue epilogue, RowPolicy row_policy,
                    unsigned char* shared_raw,
                    float (&accumulators)[Terms * Schedule::kMmaTokens][Schedule::kMmaRows][4],
                    const float* activation_scales, const __nv_bfloat16* weight_scales,
                    int row_begin, int token_begin, int rows, int tokens, int warp, int lane,
                    const float* activation_scales_lo = nullptr) {
    constexpr bool PairRows = RowPolicy::kPaired;
    constexpr int BM = Schedule::kBlockTokens, BN = Schedule::kBlockRows;
    constexpr int producers = [] {
        if constexpr (requires { Schedule::kProducerThreads; })
            return Schedule::kProducerThreads;
        else
            return 0;
    }();
    constexpr int THREADS     = Schedule::kThreads - producers;
    const int tid             = warp * 32 + lane;
    const int warp_token      = warp / Schedule::kWarpsRows;
    const int warp_row        = warp % Schedule::kWarpsRows;
    constexpr bool collective = requires {
        epilogue.template finish_tile<Schedule, FullTokens>(output, shared_raw, accumulators,
                                                            row_begin, token_begin, rows, tokens);
    };
    static_assert(!PairRows || collective, "paired rows require a collective epilogue");
    static_assert(!collective || Terms == 1,
                  "the two-term route combines its terms before a collective epilogue");
    const int accumulator_token = lane >> 2;
    const int accumulator_row   = 2 * (lane & 3);
    constexpr int output_stride = BN + 8;
    auto* shared_output         = reinterpret_cast<__nv_bfloat16*>(shared_raw);
#pragma unroll
    for (int mma_token = 0; mma_token < Schedule::kMmaTokens; ++mma_token) {
        const int token0 =
            token_begin + warp_token * Schedule::kWarpTokens + mma_token * 16 + accumulator_token;
        const int token1 = token0 + 8;
        const float activation_scale0 =
            (FullTokens || token0 < tokens) ? __ldg(activation_scales + token0) : 0.0F;
        const float activation_scale1 =
            (FullTokens || token1 < tokens) ? __ldg(activation_scales + token1) : 0.0F;
        float activation_scale0_lo = 0.0F, activation_scale1_lo = 0.0F;
        if constexpr (Terms > 1) {
            activation_scale0_lo =
                (FullTokens || token0 < tokens) ? __ldg(activation_scales_lo + token0) : 0.0F;
            activation_scale1_lo =
                (FullTokens || token1 < tokens) ? __ldg(activation_scales_lo + token1) : 0.0F;
        }
#pragma unroll
        for (int mma_row = 0; mma_row < Schedule::kMmaRows; ++mma_row) {
            const int local_row0  = warp_row * Schedule::kWarpRows + mma_row * 8 + accumulator_row;
            const int parent_row0 = row_policy.weight_row(row_begin, local_row0, rows);
            const int parent_row1 = row_policy.weight_row(row_begin, local_row0 + 1, rows);
            const float2 weight_scale = [&] {
                if constexpr (requires { RowPolicy::kContiguousPairs; }) {
                    if constexpr (RowPolicy::kContiguousPairs)
                        return bf16x2_bits_to_float2(
                            load_ldg<std::uint32_t>(weight_scales + parent_row0));
                }
                return make_float2(__bfloat162float(weight_scales[parent_row0]),
                                   __bfloat162float(weight_scales[parent_row1]));
            }();
            // Each E4M3 term keeps its own per-token activation scale until both are summed in
            // FP32.
            float value00 = accumulators[mma_token][mma_row][0] * activation_scale0;
            float value01 = accumulators[mma_token][mma_row][1] * activation_scale0;
            float value10 = accumulators[mma_token][mma_row][2] * activation_scale1;
            float value11 = accumulators[mma_token][mma_row][3] * activation_scale1;
            if constexpr (Terms > 1) {
                value00 += accumulators[Schedule::kMmaTokens + mma_token][mma_row][0] *
                           activation_scale0_lo;
                value01 += accumulators[Schedule::kMmaTokens + mma_token][mma_row][1] *
                           activation_scale0_lo;
                value10 += accumulators[Schedule::kMmaTokens + mma_token][mma_row][2] *
                           activation_scale1_lo;
                value11 += accumulators[Schedule::kMmaTokens + mma_token][mma_row][3] *
                           activation_scale1_lo;
            }
            value00 *= weight_scale.x;
            value01 *= weight_scale.y;
            value10 *= weight_scale.x;
            value11 *= weight_scale.y;
            if constexpr (collective) {
                accumulators[mma_token][mma_row][0] = value00;
                accumulators[mma_token][mma_row][1] = value01;
                accumulators[mma_token][mma_row][2] = value10;
                accumulators[mma_token][mma_row][3] = value11;
            } else {
                if (FullTokens || token0 < tokens) {
                    const float2 value = fp8_apply_row_pair(epilogue, parent_row0, parent_row1,
                                                            token0, make_float2(value00, value01));
                    value00            = value.x;
                    value01            = value.y;
                }
                if (FullTokens || token1 < tokens) {
                    const float2 value = fp8_apply_row_pair(epilogue, parent_row0, parent_row1,
                                                            token1, make_float2(value10, value11));
                    value10            = value.x;
                    value11            = value.y;
                }
                auto* destination0 = reinterpret_cast<__nv_bfloat162*>(
                    shared_output + (token0 - token_begin) * output_stride + local_row0);
                auto* destination1 = reinterpret_cast<__nv_bfloat162*>(
                    shared_output + (token1 - token_begin) * output_stride + local_row0);
                *destination0 = __floats2bfloat162_rn(value00, value01);
                *destination1 = __floats2bfloat162_rn(value10, value11);
            }
        }
    }
    __syncthreads();

    if constexpr (collective) {
        epilogue.template finish_tile<Schedule, FullTokens>(output, shared_raw, accumulators,
                                                            row_begin, token_begin, rows, tokens);
    } else {
        constexpr int stored_rows       = PairRows ? BN / 2 : BN;
        constexpr int vectors_per_token = stored_rows / 8;
        constexpr int output_vectors    = BM * vectors_per_token;
        for (int task = tid; task < output_vectors; task += THREADS) {
            const int token_local = task / vectors_per_token;
            const int row_vector  = task - token_local * vectors_per_token;
            const int token       = token_begin + token_local;
            if constexpr (FullTokens) {
                const uint4 values =
                    load_vec<uint4>(shared_output + token_local * output_stride + row_vector * 8);
                linear_store_bf16_vector(output, row_begin + row_vector * 8, token, values);
            } else if (token < tokens) {
                const uint4 values =
                    load_vec<uint4>(shared_output + token_local * output_stride + row_vector * 8);
                linear_store_bf16_vector(output, row_begin + row_vector * 8, token, values);
            }
        }
    }
}

} // namespace ninfer::ops::detail
