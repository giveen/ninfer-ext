#pragma once

#include "core/weight.h"
#include "ninfer/ops/linear.h"

#include <cstdint>

namespace ninfer::ops::detail {

// Executes an EXL3 mul1 projection: out[N,T] = x[K,T] · W[N,K]ᵀ with
// W = diag(su) · H128 · Z · H128 · diag(sv), the stored trellis_t16_v1 representation.
void exl3_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                   WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
