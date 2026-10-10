#include "ops/linear/q5/q5_shapes.h"

namespace ninfer::ops::detail {

// Gemma 4 31B MLP (gate/up [21504,5376], down [5376,21504]). The direct single-token routes are
// specialized per K, and none exists for K 5376 or 21504, so decode uses the two-row direct route;
// wider passes follow the down-projection list of N=5120, K=17408. Untuned for these shapes.
Q5Launch select_q5_gemma4(std::int32_t tokens) {
    if (tokens <= 8) return launch_q5_a16_direct_r2_t4_w2_g8_b4;
    if (tokens <= 16) return launch_q5_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 32) return launch_q5_a16_sliced_r32_t32_w4_s2;
    if (tokens <= 128) return launch_q5_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 256) return launch_q5_a16_mma_r32_t128;
    return launch_q5_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
