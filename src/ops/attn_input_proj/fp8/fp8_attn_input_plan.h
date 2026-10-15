#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/fp8/fp8_a8_plan.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t fp8_attn_input_partial_capacity_bytes(std::int32_t max_tokens);

// Inclusive upper bound of the token range handled by fp8_attn_input_a16_small_mma_launch; must
// stay in sync with the schedule coverage in fp8_attn_input_a16_small_t.cu.

inline constexpr int kFp8AttnInputLastSmallMmaT = 32;

// Split-K partial plane bytes required when the A8 route runs at `max_tokens` (0 when the route
// needs no partials). Must be passed to fp8_attn_input_workspace_capacity_bytes for the same token
// range so the caller's arena is neither under- nor over-sized.

[[nodiscard]] std::size_t fp8_attn_input_workspace_capacity_bytes(LinearPolicy policy,
                                                                  std::int32_t min_tokens,
                                                                  std::int32_t max_tokens);

// Route contract: small_mma covers T in [2,kFp8AttnInputLastSmallMmaT], decode is T == 1, and gemm
// covers T > kFp8AttnInputLastSmallMmaT. dispatch picks between them (and the A8 route) itself and
// requires a non-null `workspace` only when it selects A8.

void fp8_attn_input_a16_small_mma_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                         Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream);
void fp8_attn_input_a16_gemm_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                    Tensor& k, Tensor& v, cudaStream_t stream);

void fp8_attn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                  Tensor& k, Tensor& v, cudaStream_t stream);

void fp8_attn_input_a8_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                              Tensor& k, Tensor& v, Fp8A8Workspace workspace, cudaStream_t stream);

void fp8_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                             Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                             cudaStream_t stream);

} // namespace ninfer::ops::detail
