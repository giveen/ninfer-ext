#include "ops/linear/q6/q6_shapes.h"

namespace ninfer::ops::detail {

// Gemma 4 31B (hidden 5376): attention q/k/v and output, the MLP halves and down, and the head. They
// reuse the route list tuned for N=34816, K=5120, the nearest registered problem; every N here is a
// multiple of 64 and every K of 128, which the MMA routes need. Untuned for these shapes.
Q6Launch select_q6_gemma4(std::int32_t tokens) {
    return select_q6_n34816_k5120(tokens);
}

} // namespace ninfer::ops::detail
