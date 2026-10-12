#pragma once

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * Op: gemma_moe_experts
 *
 * Math / indexing:
 *   For each token t in [0,T), over contiguous BF16 x [H,T] and the router's selection:
 *
 *     out[:,t] = sum over the K selected ranks j of
 *                  w[j,t] * down_e · ( gelu_tanh(gate_e · x[:,t]) ⊙ (up_e · x[:,t]) )
 *
 *     with e = expert_ids[j,t] and w = expert_weights[j,t]. `gelu_tanh` is the tanh approximation
 *     of the GELU that the checkpoint names as its activation.
 *
 *   The experts are stored expert-major in two banks and are read in place, with no gather and no
 *   repack: expert e's gate rows are [e*2*I, e*2*I + I) of `gate_up` and its up rows are
 *   [e*2*I + I, e*2*I + 2*I); `down` holds expert e in [e*H, (e+1)*H) with I columns. I is the
 *   *stored* intermediate width, which the artifact pads to the block-scaled layout's row multiple;
 *   its gate and up padding rows and its down padding columns are zero, so they contribute nothing
 *   and the padded width is arithmetically invisible.
 *
 * Logical shapes:
 *   x               contiguous BF16 [H,T].
 *   expert_ids      contiguous I32 [K,T], rank fastest axis.
 *   expert_weights  contiguous FP32 [K,T], rank fastest axis.
 *   gate_up         Weight [E*2*I, H].
 *   down            Weight [E*H, I].
 *   out             contiguous BF16 [H,T], completely overwritten.
 *
 * Supported domain:
 *   T >= 1, H >= 1, E >= 1, 1 <= K <= E, I >= 1. Both banks are BF16, or both are NVFP4 in the
 *   registered K16M128x4 block-scaled layout (N a multiple of 128 and K of 64, which the stored
 *   padded widths satisfy), and the two banks share one format. An NVFP4 bank is a stack of
 *   separately quantised source matrices, so it carries a weight divisor per source and a stored
 *   value is `e2m1(codes) * e4m3(block scale) / divisor(row)` -- the same reading the offload MoE
 *   route applies to its own stacked planes.
 *
 * Numeric:
 *   The activations are the represented BF16 values and the weights the exact decode of their
 * stored form. The oracle evaluates the formula above naively in FP64 from the same represented
 * inputs; the reduction association, any private accumulator and the placement of the activation
 * are execution choices rather than semantic boundaries. The checkpoint's own reference adds each
 * selected expert's contribution into an accumulator held in the output dtype, so a test compares
 * the returned BF16 with the FP64 formula under a criterion rather than exactly. Every expert id in
 * [0,E) is assumed, which the producing router guarantees.
 *
 * Effects:
 *   `out` is completely overwritten. Every input and both banks are read only. No state is kept
 *   between calls.
 *
 * Workspace:
 *   None. The Op allocates nothing and needs no caller scratch.
 *
 * Execution:
 *   The caller's stream; graph-capturable (no host synchronization, no allocation).
 */
struct GemmaMoeExpertWeights {
    Weight gate_up; // [num_experts * 2 * intermediate, hidden]
    Weight down;    // [num_experts * hidden, intermediate]
};

struct GemmaMoeExpertGeometry {
    std::int32_t hidden_size       = 0; // H
    std::int32_t num_experts       = 0; // E
    std::int32_t top_k             = 0; // K
    std::int32_t intermediate_size = 0; // I, the stored width
};

void gemma_moe_experts(const Tensor& x, const Tensor& expert_ids, const Tensor& expert_weights,
                       const GemmaMoeExpertWeights& weights, const GemmaMoeExpertGeometry& geometry,
                       Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
