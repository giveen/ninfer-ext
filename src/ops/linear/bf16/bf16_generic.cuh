#pragma once

// The dense BF16 fallback the Linear dispatch consults when no registered shape matches.
//
// Every format in this tree is a finite registry of hand-written, separately qualified kernels
// (bf16_dispatch.cpp, and the q4..q8, fp8, nvfp4 and exl3 dispatches beside it) and none of them
// has a generic fallback: a problem no model has registered has no route at all. That is the right
// default for a measured route, but it means a family whose own widths are not in the registry
// cannot run at all -- Nemotron-H is the first such family (every one of its projections carries a
// K of 4096, 1024 or 18560 rows, and no registered pair holds any of them).
//
// So this route exists for correctness, not for speed: it accepts any positive (N,K) and any
// positive token count and is *only* reached when the registry has already declined the problem,
// so no existing model's route changes. The shapes a family actually runs are still expected to
// get hand-written kernels; this is what lets the mathematics be checked before those land.
// Nemotron-H is the family that first needed it: no pair the registry carried matched a width of
// it, and the two latent projections among them now have their own route in
// shapes/nemotron_h_latent.cu while the rest still arrive here.
//
// Arithmetic: FP32 accumulation in ascending K order with one BF16 rounding at the output, which
// is the Linear Op's numeric contract for its dense routes.

#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <cmath>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// One thread owns this many consecutive tokens of one output row, so a weight element is loaded
// once and reused across them. The fallback's whole job is to be correct for shapes nobody wrote a
// kernel for, and this keeps its weight traffic to a single pass over [N,K].
inline constexpr int kBf16GenericTokensPerThread = 4;

// Threads are indexed row-fastest, which makes consecutive threads read consecutive weight rows and
// keeps a warp's weight loads coalesced. `x` is token major with K fastest and remains untouched.
static __global__ void bf16_generic_kernel(const __nv_bfloat16* __restrict__ x,
                                           const __nv_bfloat16* __restrict__ weight,
                                           __nv_bfloat16* __restrict__ out, const std::int32_t n,
                                           const std::int32_t k, const std::int32_t tokens) {
    const std::int64_t owner  = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int32_t row    = static_cast<std::int32_t>(owner % n);
    const std::int32_t token0 = static_cast<std::int32_t>(owner / n) * kBf16GenericTokensPerThread;
    if (token0 >= tokens) { return; }

    const __nv_bfloat16* weight_row                = weight + static_cast<std::int64_t>(row) * k;
    float accumulator[kBf16GenericTokensPerThread] = {0.0F, 0.0F, 0.0F, 0.0F};
    for (std::int32_t column = 0; column < k; ++column) {
        const float w = __bfloat162float(weight_row[column]);
#pragma unroll
        for (int slot = 0; slot < kBf16GenericTokensPerThread; ++slot) {
            const std::int32_t token = token0 + slot;
            if (token < tokens) {
                accumulator[slot] =
                    fmaf(w, __bfloat162float(x[static_cast<std::int64_t>(token) * k + column]),
                         accumulator[slot]);
            }
        }
    }
#pragma unroll
    for (int slot = 0; slot < kBf16GenericTokensPerThread; ++slot) {
        const std::int32_t token = token0 + slot;
        if (token >= tokens) { break; }
        out[static_cast<std::int64_t>(token) * n + row] = __float2bfloat16_rn(accumulator[slot]);
    }
}

// The fallback's launch. Defined in shapes/generic.cu, which is the only place the kernel above is
// instantiated: a definition in the header would be dropped from every translation unit that does
// not call it, which is all of them.
void bf16_generic_launch(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
