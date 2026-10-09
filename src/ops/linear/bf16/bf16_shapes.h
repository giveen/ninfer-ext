#pragma once

#include "ops/linear/bf16/bf16_launch.h"

namespace ninfer::ops::detail {

[[nodiscard]] Bf16Launch select_bf16_n14336_k5120(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n5120_k6144(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n256_k5120(std::int32_t tokens);
// The Gemma 4 output head: the tied embedding matrix, projected once per token.
[[nodiscard]] Bf16Launch select_bf16_n262144_k5376(std::int32_t tokens);

} // namespace ninfer::ops::detail
