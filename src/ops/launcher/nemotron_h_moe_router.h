// Preconditions, all established by the wrapper: x is contiguous, 16-byte aligned BF16 [H,T] with
// T>=1 and H a multiple of 8; the router payload is a 16-byte aligned BF16 [E,H] matrix and
// correction_bias is a contiguous FP32 [E] vector; logits is FP32 [E,T] of caller workspace,
// expert_ids is I32 [K,T] and expert_weights FP32 [K,T] with 1<=K<=16, K<=E<=512, 1<=G<=32 with E a
// multiple of G, 1<=U<=G and K<=U*(E/G).
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void nemotron_h_moe_router_launch(const void* x, const void* router, const void* correction_bias,
                                  std::int32_t hidden, std::int32_t experts, std::int32_t top_k,
                                  std::int32_t groups, std::int32_t groups_used, bool normalize,
                                  float scaling_factor, std::int32_t tokens, float* logits,
                                  void* expert_ids, float* expert_weights, cudaStream_t stream);

} // namespace ninfer::ops::detail
