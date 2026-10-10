#include "ops/linear/q5/q5_shapes.h"

namespace ninfer::ops::detail {
namespace {
// Gemma 4 31B MLP at hidden 5376. The inherited list followed the down projection of N=5120,
// K=17408, whose rows are far narrower than 21504: with that list a 48-to-256 token pass cost 1.1x
// to 1.6x the time a route sweep measures as achievable at this shape, because the sliced-K routes
// it selects leave the wide row extent to too few blocks. Widths below 48 and above 256 were
// already the best registered route and keep their choice.
Q5Launch select_wide(std::int32_t tokens) {
    if (tokens <= 4) return launch_q5_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 8) return launch_q5_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 16) return launch_q5_a16_sliced_r32_t24_w4_s2_pairwise;
    if (tokens <= 32) return launch_q5_a16_sliced_r32_t32_w4_s2;
    if (tokens <= 64) return launch_q5_a16_sliced_r32_t64_w2_s1;
    if (tokens <= 96) return launch_q5_a16_mma_r64_t96_k128_s1_a1;
    if (tokens <= 128) return launch_q5_a16_mma_r64_t128;
    if (tokens <= 192) return launch_q5_a16_mma_r64_t96_k128_s1_a1;
    return launch_q5_a16_mma_r64_t128;
}
} // namespace

Q5Launch select_q5_gemma4_k5376(std::int32_t tokens) {
    if (tokens == 1) return launch_q5_a16_direct_r1_t1_w4_k5376;
    return select_wide(tokens);
}

Q5Launch select_q5_gemma4_k21504(std::int32_t tokens) {
    // The down projection re-reads its weights past four tokens with the two-row direct route; the
    // four-row-block route stays ahead through there and nothing else in the list beats the
    // inherited choice, so only that interval changes.
    if (tokens == 1) return launch_q5_a16_direct_r1_t1_w4_k21504;
    if (tokens <= 4) return launch_q5_a16_direct_r2_t4_w2_g8_b4;
    if (tokens <= 8) return launch_q5_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 16) return launch_q5_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 32) return launch_q5_a16_sliced_r32_t32_w4_s2;
    if (tokens <= 128) return launch_q5_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 256) return launch_q5_a16_mma_r32_t128;
    return launch_q5_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
