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

} // namespace ninfer::ops::detail
