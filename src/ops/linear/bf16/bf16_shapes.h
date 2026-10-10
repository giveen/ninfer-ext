#pragma once

#include "ops/linear/bf16/bf16_launch.h"

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

} // namespace ninfer::ops::detail
