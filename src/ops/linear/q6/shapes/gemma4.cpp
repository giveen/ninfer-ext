#include "ops/linear/q6/q6_shapes.h"

namespace ninfer::ops::detail {
// Gemma 4 31B text projections, hidden 5376. Each selector below overrides only the token extents
// where a measured sweep beat the route list inherited from N=34816, K=5120; every other extent
// keeps that list, which the sweep confirmed is already the best registered route there. The
// numbers are from bench/ops sweeps at this exact (N, K): the widest gains are 1.9x at T=64 on the
// K-heavy shapes, because the inherited list leaves the 32..256 interval to sliced-K routes tuned
// for a much wider N. Every N here is a multiple of 64 and every K of 128, which the MMA routes need.

// Sliding key and value: the most numerous text projection, 100 of the 240 MLP/attention objects.
Q6Launch select_q6_gemma4_n4096_k5376(std::int32_t tokens) {
    if (tokens == 1) return launch_q6_a16_gemv_r4_w2_g16;
    if (tokens <= 8) return launch_q6_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 16) return launch_q6_a16_sliced_r32_t16_w4_s2;
    if (tokens <= 32) return launch_q6_a16_sliced_r32_t32_w4_s2;
    if (tokens <= 96) return launch_q6_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 128) return launch_q6_a16_sliced_r32_t64_w2_s1;
    if (tokens <= 192) return launch_q6_a16_mma_r64_t40_k128;
    if (tokens <= 256) return launch_q6_a16_sliced_r32_t64_w2_s1;
    return select_q6_n34816_k5120(tokens);
}

// Sliding query.
Q6Launch select_q6_gemma4_n8192_k5376(std::int32_t tokens) {
    if (tokens == 1) return launch_q6_a16_gemv_r4_w2_g16;
    if (tokens <= 8) return launch_q6_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 16) return launch_q6_a16_sliced_r32_t16_w4_s2;
    if (tokens <= 64) return select_q6_n34816_k5120(tokens);
    if (tokens <= 96) return launch_q6_a16_mma_r64_t48_k128;
    if (tokens <= 128) return launch_q6_a16_sliced_r32_t64_w2_s1;
    return select_q6_n34816_k5120(tokens);
}

// Global query.
Q6Launch select_q6_gemma4_n16384_k5376(std::int32_t tokens) {
    if (tokens == 1) return launch_q6_a16_gemv_r4_w2_g16;
    if (tokens <= 8) return launch_q6_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 16) return launch_q6_a16_sliced_r32_t16_w4_s2;
    if (tokens <= 128) return select_q6_n34816_k5120(tokens);
    if (tokens <= 192) return launch_q6_a16_mma_r64_t96;
    return select_q6_n34816_k5120(tokens);
}

// Global key, which is also its value.
Q6Launch select_q6_gemma4_n2048_k5376(std::int32_t tokens) {
    if (tokens == 1) return launch_q6_a16_gemv_r4_w2_g16;
    if (tokens <= 8) return launch_q6_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 16) return launch_q6_a16_sliced_r16_t24_w4_s2;
    if (tokens <= 32) return launch_q6_a16_sliced_r16_t32_w4_s2;
    if (tokens <= 64) return launch_q6_a16_sliced_r32_t32_w4_s2;
    if (tokens <= 512) return launch_q6_a16_sliced_r32_t32_w4_s1;
    return select_q6_n34816_k5120(tokens);
}

// MLP gate and up, the Q6 half of the projection.
Q6Launch select_q6_gemma4_n21504_k5376(std::int32_t tokens) {
    if (tokens == 1) return launch_q6_a16_gemv_r4_w2_g16;
    if (tokens <= 8) return launch_q6_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 128) return select_q6_n34816_k5120(tokens);
    if (tokens <= 192) return launch_q6_a16_mma_r64_t96;
    return select_q6_n34816_k5120(tokens);
}

// The vocabulary head keeps the inherited list: the sweep found nothing above 9% there.
Q6Launch select_q6_gemma4_n262144_k5376(std::int32_t tokens) { return select_q6_gemma4(tokens); }

// Sliding output projection.
Q6Launch select_q6_gemma4_k8192(std::int32_t tokens) {
    if (tokens <= 8) return launch_q6_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 16) return launch_q6_a16_sliced_r32_t16_w4_s2;
    if (tokens <= 32) return launch_q6_a16_sliced_r32_t32_w4_s2;
    if (tokens <= 96) return launch_q6_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 128) return launch_q6_a16_sliced_r32_t64_w2_s1;
    if (tokens <= 192) return launch_q6_a16_mma_r64_t48_k128;
    if (tokens <= 256) return launch_q6_a16_sliced_r32_t64_w2_s1;
    return select_q6_n34816_k5120(tokens);
}

// Global output projection.
Q6Launch select_q6_gemma4_k16384(std::int32_t tokens) {
    if (tokens <= 8) return launch_q6_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 16) return launch_q6_a16_sliced_r32_t16_w4_s2;
    if (tokens <= 32) return launch_q6_a16_sliced_r32_t32_w4_s2;
    if (tokens <= 96) return launch_q6_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 128) return launch_q6_a16_sliced_r32_t64_w2_s1;
    if (tokens <= 192) return launch_q6_a16_mma_r64_t48_k128;
    if (tokens <= 256) return launch_q6_a16_sliced_r32_t64_w2_s1;
    return select_q6_n34816_k5120(tokens);
}

// MLP down, the Q6 half.
Q6Launch select_q6_gemma4_k21504(std::int32_t tokens) {
    if (tokens == 1) return launch_q6_a16_gemv_r4_w2_g16;
    if (tokens <= 8) return launch_q6_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 16) return launch_q6_a16_sliced_r32_t16_w4_s2;
    if (tokens <= 32) return launch_q6_a16_sliced_r32_t32_w4_s2;
    if (tokens <= 96) return launch_q6_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 128) return launch_q6_a16_sliced_r32_t64_w2_s1;
    if (tokens <= 192) return launch_q6_a16_mma_r64_t48_k128;
    if (tokens <= 256) return launch_q6_a16_mma_r64_t64_k128;
    return select_q6_n34816_k5120(tokens);
}

// The assistant drafter, hidden 1024. A draft step consumes one token, so the single-token pass is
// what production runs; these selectors fix that pass and the narrow verify widths.
Q6Launch select_q6_gemma4_drafter_wide_k(std::int32_t tokens) {
    if (tokens == 1) return launch_q6_a16_gemv_r4_w2_g16;
    if (tokens <= 4) return launch_q6_a16_simt_r4_t4_w2_g16;
    if (tokens <= 16) return launch_q6_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 384) return launch_q6_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 768) return launch_q6_a16_mma_r64_t40_k128;
    return select_q6_gemma4(tokens);
}

Q6Launch select_q6_gemma4_drafter_n16384_k1024(std::int32_t tokens) {
    if (tokens <= 16) return launch_q6_a16_sliced_r16_t8_w4_s2;
    return select_q6_gemma4(tokens);
}

Q6Launch select_q6_gemma4_drafter_n262144_k1024(std::int32_t tokens) {
    if (tokens == 1) return launch_q6_a16_sliced_r16_t8_w4_s2;
    if (tokens == 2) return launch_q6_a16_sliced_r32_t16_w4_s2;
    return select_q6_gemma4(tokens);
}

Q6Launch select_q6_gemma4_drafter_small(std::int32_t tokens) {
    if (tokens == 2) return launch_q6_a16_simt_r4_t4_w2_g16;
    if (tokens <= 8) return launch_q6_a16_sliced_r16_t8_w4_s2;
    if (tokens == 16) return launch_q6_a16_sliced_r16_t32_w4_s2;
    return select_q6_gemma4(tokens);
}

// The image tower: hidden 1152 and the MLP width stored as 4304 padded to 4352. An encoder call uses
// every patch of one image as its token extent, so the extents are 9..2520 rather than a request
// length, and the inherited list was the worst possible fit: from 32 to 390 patches it left two to
// four times the achievable time on the table, mostly because N is far too small for its row tiles.
Q6Launch select_q6_gemma4_vision_n1152_k1152(std::int32_t tokens) {
    if (tokens <= 72) return launch_q6_a16_sliced_r16_t32_w4_s2;
    if (tokens <= 324) return launch_q6_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 648) return launch_q6_a16_mma_r64_t40_k128;
    if (tokens <= 1296) return launch_q6_a16_mma_r64_t72_k128;
    if (tokens <= 2304) return launch_q6_a16_mma_r64_t128;
    // Past 2304 a 128-token tile usually ends in a masked tail of up to 127 columns, which costs
    // more than the whole tile is worth; a 96-token tile keeps the tail under a quarter tile.
    return tokens % 128 == 0 ? launch_q6_a16_mma_r64_t128 : launch_q6_a16_mma_r64_t96;
}

Q6Launch select_q6_gemma4_vision_n1152_k4352(std::int32_t tokens) {
    if (tokens <= 8) return launch_q6_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 162) return launch_q6_a16_sliced_r16_t24_w4_s2;
    if (tokens <= 384) return launch_q6_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 648) return launch_q6_a16_mma_r64_t40_k128;
    if (tokens <= 1296) return launch_q6_a16_mma_r64_t72_k128;
    if (tokens <= 2304) return launch_q6_a16_mma_r64_t128;
    return tokens % 128 == 0 ? launch_q6_a16_mma_r64_t128 : launch_q6_a16_mma_r64_t96;
}

Q6Launch select_q6_gemma4_vision_n4352_k1152(std::int32_t tokens) {
    if (tokens <= 18) return launch_q6_a16_sliced_r16_t24_w4_s2;
    if (tokens <= 72) return launch_q6_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 162) return launch_q6_a16_mma_r64_t40_k128;
    if (tokens <= 768) return launch_q6_a16_mma_r64_t80;
    if (tokens <= 1536) return launch_q6_a16_mma_r64_t112;
    return launch_q6_a16_mma_r64_t128;
}

// Gemma 4 text (hidden 5376) that the sweep found no better route for: the head, and any geometry
// reached through it. Attention q/k/v and output, the MLP halves and down have their own selectors.
Q6Launch select_q6_gemma4(std::int32_t tokens) {
    if (tokens == 1) return launch_q6_a16_gemv_r4_w2_g16;
    return select_q6_n34816_k5120(tokens);
}

} // namespace ninfer::ops::detail
