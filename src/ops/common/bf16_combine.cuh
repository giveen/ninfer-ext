#pragma once

// The two elementwise BF16 pair expressions that `residual_add` and `mul_scalar` store at, in one
// place so an Op that folds them around another computation is built from the same definitions
// rather than from a copy that could drift from them.

#include <cuda_bf16.h>

namespace ninfer::ops {

// `x += y`, one BF16 store: `x` is the accumulator and `y` the addend.
__device__ __forceinline__ __nv_bfloat162 residual_add_pair(__nv_bfloat162 y, __nv_bfloat162 x) {
    const float r0 = __low2float(x) + __low2float(y);
    const float r1 = __high2float(x) + __high2float(y);
    return __floats2bfloat162_rn(r0, r1);
}

__device__ __forceinline__ float mul_scalar_one(float value, float factor) {
    return value * factor;
}

__device__ __forceinline__ __nv_bfloat162 mul_scalar_pair(__nv_bfloat162 input, float factor) {
    return __floats2bfloat162_rn(mul_scalar_one(__low2float(input), factor),
                                 mul_scalar_one(__high2float(input), factor));
}

} // namespace ninfer::ops
