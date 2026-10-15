#pragma once

#include "core/device.h"
#include "ops/common/math.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail::mamba2_scan {

// The state (key) extent, which is `ssm_state_size`. It is fixed here for the same reason the
// gated-linear-attention kernels fix it: the lane that owns a 4-key vector needs the extent to be
// a multiple of 32, and one warp's reduction must cover the whole readout contraction.
inline constexpr int kStateSize = 128;

// One CTA covers a 16-value tile of a head: kWarps warps of kValuesPerWarp value rows, each row
// spread over the 32 lanes as kKeysPerLane consecutive keys. A value extent that is not a multiple
// of the tile leaves the last tile's tail rows predicated out.
inline constexpr int kWarps         = 4;
inline constexpr int kValuesPerWarp = 4;
inline constexpr int kKeysPerLane   = kStateSize / 32;
inline constexpr int kBlockValues   = kWarps * kValuesPerWarp;

static_assert(kStateSize % 32 == 0);
static_assert(kKeysPerLane == 4);

struct Arguments {
    const __nv_bfloat16 *x, *b, *c, *dt;
    const float *dt_bias, *a_log, *d_skip, *state_in;
    float* state_out;
    __nv_bfloat16* out;
    int value_dim, heads, groups, tokens;
    float time_step_min;
};

// One CTA per (head, 16-value tile). The state lives in registers for the whole sequence, so the
// recurrence streams the operands once each and touches the state exactly twice (initial load,
// final store).
//
// Measured at the checkpoint's geometry (128 heads x 64 values, 128-wide state, RTX 5090): 0.41 us
// per token at T=128 rising to 0.54 us at T=4096, i.e. 2.2 ms for one layer's 4096-token prefill
// (about 11% of the card's FP32 rate, and roughly half the instruction-issue bound of this
// decomposition at 12 resident warps per SM). A chunked or tensor-core form is the way to go
// further; the sequential form was chosen because the state's arithmetic is then identical to the
// oracle's and the Op needs no workspace at all.
void launch_recurrent(const Arguments& args, cudaStream_t stream);

} // namespace ninfer::ops::detail::mamba2_scan
