#include "ops/linear/q6/q6_shapes.h"

namespace ninfer::ops::detail {

// Qwen4Exp (hidden 2560) dense projections reuse the schedule tuned for the N=248320, K=2560 head;
// the rest of this family has no tuned instances of its own.
Q6Launch select_q6_qwen4_exp(std::int32_t tokens) {
    return select_q6_n248320_k2560(tokens);
}

// The hyper-connection projections onto the low rank ([320|324, 10240]) have too few rows for a
// warp per row to fill the GPU at decode widths; several warps share each row there.
Q6Launch select_q6_qwen4_exp_wide_k(std::int32_t tokens) {
    if (tokens <= 1) return launch_q6_a16_gemv_r4_w2_g16;
    if (tokens <= 8) return launch_q6_a16_sliced_r16_t8_w4_s2;
    return select_q6_qwen4_exp(tokens);
}

} // namespace ninfer::ops::detail
