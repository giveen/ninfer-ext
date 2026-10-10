#pragma once

#include "ops/linear/q6/q6_launch.h"

namespace ninfer::ops::detail {

[[nodiscard]] Q6Launch select_q6_n248320_k5120(std::int32_t tokens);
[[nodiscard]] Q6Launch select_q6_n34816_k5120(std::int32_t tokens);
[[nodiscard]] Q6Launch select_q6_n248320_k2048(std::int32_t tokens);
[[nodiscard]] Q6Launch select_q6_n1152_k1536(std::int32_t tokens);
[[nodiscard]] Q6Launch select_q6_n248320_k2560(std::int32_t tokens);
[[nodiscard]] Q6Launch select_q6_qwen4_exp(std::int32_t tokens);
[[nodiscard]] Q6Launch select_q6_gemma4_n4096_k5376(std::int32_t tokens);
[[nodiscard]] Q6Launch select_q6_gemma4_n8192_k5376(std::int32_t tokens);
[[nodiscard]] Q6Launch select_q6_gemma4_n16384_k5376(std::int32_t tokens);
[[nodiscard]] Q6Launch select_q6_gemma4_n2048_k5376(std::int32_t tokens);
[[nodiscard]] Q6Launch select_q6_gemma4_n21504_k5376(std::int32_t tokens);
[[nodiscard]] Q6Launch select_q6_gemma4_n262144_k5376(std::int32_t tokens);
[[nodiscard]] Q6Launch select_q6_gemma4_k8192(std::int32_t tokens);
[[nodiscard]] Q6Launch select_q6_gemma4_k16384(std::int32_t tokens);
[[nodiscard]] Q6Launch select_q6_gemma4_k21504(std::int32_t tokens);
[[nodiscard]] Q6Launch select_q6_gemma4_drafter_wide_k(std::int32_t tokens);
[[nodiscard]] Q6Launch select_q6_gemma4_drafter_n16384_k1024(std::int32_t tokens);
[[nodiscard]] Q6Launch select_q6_gemma4_drafter_n262144_k1024(std::int32_t tokens);
[[nodiscard]] Q6Launch select_q6_gemma4_drafter_small(std::int32_t tokens);
[[nodiscard]] Q6Launch select_q6_gemma4_vision_n1152_k1152(std::int32_t tokens);
[[nodiscard]] Q6Launch select_q6_gemma4_vision_n1152_k4352(std::int32_t tokens);
[[nodiscard]] Q6Launch select_q6_gemma4_vision_n4352_k1152(std::int32_t tokens);
[[nodiscard]] Q6Launch select_q6_gemma4(std::int32_t tokens);
[[nodiscard]] Q6Launch select_q6_qwen4_exp_wide_k(std::int32_t tokens);
[[nodiscard]] Q6Launch select_q6_qwen4_exp_small(std::int32_t tokens);

} // namespace ninfer::ops::detail
