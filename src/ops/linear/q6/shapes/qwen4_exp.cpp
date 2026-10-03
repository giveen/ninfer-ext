#include "ops/linear/q6/q6_shapes.h"

namespace ninfer::ops::detail {

// Qwen4Exp (hidden 2560) dense projections reuse the schedule tuned for the N=248320, K=2560 head;
// the rest of this family has no tuned instances of its own.
Q6Launch select_q6_qwen4_exp(std::int32_t tokens) {
    return select_q6_n248320_k2560(tokens);
}

} // namespace ninfer::ops::detail
