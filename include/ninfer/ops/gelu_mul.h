#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Elementwise GeGLU activation, the tanh-GELU sibling of `silu_mul`:
 *
 *   ideal[i] = gelu_tanh(gate[i]) * up[i]
 *            = 0.5*gate[i]*(1 + tanh(sqrt(2/pi)*(gate[i] + 0.044715*gate[i]^3))) * up[i].
 *
 * The formula is the one `ops::gelu`'s Tanh mode uses, so the two Ops agree bit for bit on the gate
 * half, and it is the activation Gemma 4's MLP names `gelu_pytorch_tanh`.
 *
 * `gate`, `up`, and `out` are same-shaped BF16 tensors. out is contiguous; gate and up may use
 * arbitrary valid Tensor strides, which is what lets a caller pass the two halves of a fused
 * gate/up projection without copying them. out must not overlap either input (the two read-only
 * inputs may overlap one another). The oracle evaluates `ideal` in FP64 from the represented
 * inputs; the BF16 output is promoted and compared directly with that result, and output storage
 * rounding belongs to the Op's numerical criterion, not the oracle. Private kernel arithmetic is
 * implementation-defined. The Op writes all of out and uses no workspace or persistent state.
 */
void gelu_mul(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
