#include "ops/linear/q8/q8_shapes.h"

namespace ninfer::ops::detail {

// Qwen4Exp (hidden 2560) projections use the generic row-split routes; T selection follows the
// untuned default of the smaller registered problems.
Q8Launch select_q8_qwen4_exp(std::int32_t tokens) {
    if (tokens <= 4) return launch_q8_simt_r8_c4;
    if (tokens <= 16) return launch_q8_simt_r8_c8;
    if (tokens <= 128) return launch_q8_mma_r32_c128;
    return launch_q8_mma_r64_c128;
}

// The hyper-connection projections onto the low rank ([320|324, 10240]) have too few rows for a
// warp per row to fill the GPU at decode widths; several warps share each row there.
Q8Launch select_q8_qwen4_exp_wide_k(std::int32_t tokens) {
    if (tokens <= 4) return launch_q8_simt_ksplit_c4;
    if (tokens <= 8) return launch_q8_simt_ksplit_c8;
    return select_q8_qwen4_exp(tokens);
}

} // namespace ninfer::ops::detail
