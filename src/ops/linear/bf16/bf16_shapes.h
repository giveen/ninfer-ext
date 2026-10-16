#pragma once

#include "ops/linear/bf16/bf16_launch.h"

#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] Bf16Launch select_bf16_n14336_k5120(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n5120_k6144(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n256_k5120(std::int32_t tokens);
// The Gemma 4 output head: the tied embedding matrix, projected once per token.
[[nodiscard]] Bf16Launch select_bf16_n262144_k5376(std::int32_t tokens);
// The Gemma 4 assistant drafter: input and output projections, attention, MLP and head.
[[nodiscard]] Bf16Launch select_bf16_n1024_k10752(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n8192_k1024(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n16384_k1024(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n1024_k8192(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n1024_k16384(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n262144_k1024(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n5376_k1024(std::int32_t tokens);
// The Gemma 4 vision tower: patch embedding and the soft-token projection into the text width.
[[nodiscard]] Bf16Launch select_bf16_n1152_k768(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n5376_k1152(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n2816_k1152(std::int32_t tokens);
// The Nemotron-H latent round trip: the two projections that bracket a MoE bank's routed experts,
// at the hidden width and the latent width. Both are GEMV-shaped -- 1024 and 4096 rows against K of
// 4096 and 1024 -- and both keep the dense fallback's accumulation order rather than a warp-reduced
// one (see the shape file), so they are bit-identical to the route they displace.
[[nodiscard]] Bf16Launch select_bf16_n1024_k4096(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n4096_k1024(std::int32_t tokens);

// The dense fallback for a problem no registered shape matches. Two of Nemotron-H's widths, the
// latent projections, left it for their own routes above; every other BF16 problem in the family
// still lands here, as does anything else nobody has written a kernel for (see bf16_generic.cuh).
// N, K and T are read from the call, so one route serves every shape.
void bf16_generic_launch(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
