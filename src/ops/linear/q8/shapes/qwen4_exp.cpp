#include "ops/linear/q8/q8_shapes.h"

namespace ninfer::ops::detail {

// Qwen4Exp (hidden 2560) projections use the generic routes; T selection follows the untuned
// default of the smaller registered problems.
Q8Launch select_q8_generic(std::int32_t tokens) {
    if (tokens <= 4) return launch_q8_a16_simt_r8_t4;
    if (tokens <= 16) return launch_q8_a16_simt_r8_t8;
    if (tokens <= 128) return launch_q8_a16_mma_r32_t128;
    return launch_q8_a16_mma_r64_t128;
}

Q8Launch select_q8_qwen4_exp(std::int32_t tokens) { return select_q8_generic(tokens); }

// Shapes with few output rows at decode widths: two warps share each row, which measured 10.24 ->
// 8.19 us on the 1280-row MoE shared gate. Eight warps per row is right only where k is wide (the
// 10240-wide hyper-connection pair, see select_q8_qwen4_exp_wide_k); on a narrow k it leaves each
// warp too little of the reduction and regresses badly (10240x320 measured 5.2x worse).
Q8Launch select_q8_qwen4_exp_few_rows(std::int32_t tokens) {
    if (tokens <= 4) return launch_q8_a16_simt_r4_t4_w2_g16_s2;
    return select_q8_qwen4_exp(tokens);
}

// The hyper-connection projections onto the low rank ([320|324, 10240]) have too few rows for a
// warp per row to fill the GPU at decode widths; several warps share each row there.
Q8Launch select_q8_qwen4_exp_wide_k(std::int32_t tokens) {
    if (tokens <= 4) return launch_q8_a16_simt_r2_t4_w8_g16_s2;
    if (tokens <= 16) return launch_q8_a16_sliced_r16_t16_w8_s2;
    return select_q8_qwen4_exp(tokens);
}

} // namespace ninfer::ops::detail
