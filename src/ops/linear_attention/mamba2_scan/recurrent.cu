#include "ops/linear_attention/mamba2_scan/launch.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"

namespace ninfer::ops::detail::mamba2_scan {
namespace {

// The recurrence is sequential in the token, so its parallelism is the 128 heads times the value
// tiles, and the state of one tile (16 values x 128 keys) stays in registers across the whole
// sequence: the operands stream through once per token and the state is read and written once for
// the call. Per token, per warp, the work is one decay-multiply, one rank-1 update and one readout
// contraction per state element, plus the kValuesPerWarp warp reductions of the readout.
__global__ __launch_bounds__(kWarpSize* kWarps, 4) void recurrent_kernel(Arguments args) {
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;

    const int head       = static_cast<int>(blockIdx.x);
    const int value_base = static_cast<int>(blockIdx.z) * kBlockValues + warp * kValuesPerWarp;
    const int key_base   = lane * kKeysPerLane;
    // B and C are per group, shared by the heads/groups heads of that group. The reference expands
    // the groups with repeat_interleave over the head axis, which is this same head/h denominator.
    const int group            = head / (args.heads / args.groups);
    const auto state_head_base = static_cast<std::int64_t>(head) * args.value_dim * kStateSize;

    const float bias = args.dt_bias[head];
    const float skip = args.d_skip[head];
    const float a    = -expf(args.a_log[head]);

    // The state's key extent is its fastest axis and its value extent is the row, so a row is
    // kStateSize floats and the lane's four keys are one aligned float4.
    float state[kValuesPerWarp][kKeysPerLane];
#pragma unroll
    for (int row = 0; row < kValuesPerWarp; ++row) {
        const int value = value_base + row;
        store_vec(state[row],
                  value < args.value_dim
                      ? load_vec<float4>(args.state_in + state_head_base +
                                         static_cast<std::int64_t>(value) * kStateSize + key_base)
                      : make_float4(0.0F, 0.0F, 0.0F, 0.0F));
    }

    // Measured: hand-rolling a two-slot prefetch of the next token's operands (gates included)
    // makes this loop 2.1x slower rather than faster, so the operands are loaded where they are
    // used and the loop stays this simple.
    for (int token = 0; token < args.tokens; ++token) {
        const auto group_offset =
            (static_cast<std::int64_t>(token) * args.groups + group) * kStateSize + key_base;
        float key[kKeysPerLane];
        float query[kKeysPerLane];
        load_bf16x4(key, args.b + group_offset);
        load_bf16x4(query, args.c + group_offset);

        const float raw_dt =
            __bfloat162float(args.dt[static_cast<std::int64_t>(token) * args.heads + head]);
        // Only the minimum clamps: the reference's maximum clamp is commented out. The softplus
        // plateau at 20 is the reference's own, and it keeps the gate away from exp overflow.
        const float step  = fmaxf(softplus(raw_dt + bias), args.time_step_min);
        const float decay = expf(step * a);

        const auto value_offset =
            (static_cast<std::int64_t>(token) * args.heads + head) * args.value_dim;
        float dx[kValuesPerWarp];
        float raw[kValuesPerWarp];
#pragma unroll
        for (int row = 0; row < kValuesPerWarp; ++row) {
            const int value = value_base + row;
            raw[row] =
                value < args.value_dim ? __bfloat162float(args.x[value_offset + value]) : 0.0F;
            dx[row] = step * raw[row];
        }

        float total[kValuesPerWarp];
#pragma unroll
        for (int row = 0; row < kValuesPerWarp; ++row) {
            float partial = 0.0F;
#pragma unroll
            for (int k = 0; k < kKeysPerLane; ++k) {
                state[row][k] = fmaf(dx[row], key[k], decay * state[row][k]);
                partial       = fmaf(state[row][k], query[k], partial);
            }
            // The xor reduction hands every lane the complete contraction over all 128 keys.
            total[row] = warp_sum(partial);
        }

        // Lane r publishes value row r. The D skip multiplies the raw (pre-dt) x and is added
        // before the gated grouped norm that follows this Op, which cannot absorb it.
        float result = 0.0F;
#pragma unroll
        for (int row = 0; row < kValuesPerWarp; ++row) {
            if (lane == row) { result = total[row] + skip * raw[row]; }
        }
        if (lane < kValuesPerWarp && value_base + lane < args.value_dim) {
            args.out[value_offset + value_base + lane] = __float2bfloat16_rn(result);
        }
    }

#pragma unroll
    for (int row = 0; row < kValuesPerWarp; ++row) {
        const int value = value_base + row;
        if (value < args.value_dim) {
            store_vec(args.state_out + state_head_base +
                          static_cast<std::int64_t>(value) * kStateSize + key_base,
                      load_vec<float4>(state[row]));
        }
    }
}

} // namespace

void launch_recurrent(const Arguments& args, cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(args.heads), 1,
                    static_cast<unsigned>(div_up(args.value_dim, kBlockValues)));
    const dim3 block(kWarpSize, kWarps);
    recurrent_kernel<<<grid, block, 0, stream>>>(args);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail::mamba2_scan
