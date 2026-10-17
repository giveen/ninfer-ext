#pragma once

#include "core/device.h"
#include "core/tensor.h"

#include <cstdint>

namespace ninfer::ops {

/**
 * The grouped tap application of a DFlash-style draft's finish convolution.
 *
 * Math / indexing:
 *   H = x.ne[0], W = x.ne[1], B = x.ne[2], G = H / group_width, g(h) = h / group_width, and the
 *   tap count is `base.ne[1] / H`. `side` selects the prepare (0) or finish (1) half of the
 *   checkpoint's two, and each is applied separately -- the prepare half to the normed block input,
 *   the finish half to that branch's projected output -- so this Op computes one side per call:
 *
 *     y[h,t,b] = sum_{k < taps, k <= t mod block_size}
 *                    ( base[side*taps*H + k*H + h] + delta[(side*taps + k)*G + g(h), t, b] )
 *                  * x[h, t-k, b]
 *
 *   The bound `k <= t mod block_size` is what makes the convolution block-relative: the first token
 *   of a block sees only tap zero, so a block's tokens never read the previous block's hidden state
 *   through this path. That is the reference's own rule, not an optimization.
 *
 * Logical shapes / supported domain:
 *   `x` and `out` are contiguous BF16 [H, W, B] in the engine's dim0-fastest order (element
 *   `h + H*(t + W*b)`, so a tap is exactly `k*H` earlier in the buffer); `base` is contiguous BF16
 *   [2, taps*H] carrying the artifact's payload order, which is side-major taps
 *   (`side * taps * H + tap * H + channel`) rather than dim0-fastest, because it is a bound weight
 *   rather than an activation; `delta` is contiguous BF16 [2*taps*G, W, B] with its row fastest, in
 *   the projection's own row order (`(side * taps + tap) * G + group`), so nothing is shuffled
 *   between the projection and this Op.
 *   `group_width` must divide H, `block_size` must be positive, and every tensor must be distinct
 *   storage of the declared dtype.
 *
 * Numeric:
 *   The oracle evaluates the formula naively in FP64 from the represented BF16 inputs and compares
 *   the promoted BF16 output directly. `x`, `base` and `delta` are all cast boundaries; the sum
 *   order and any private precision are implementation choices.
 *
 * Effects:
 *   Writes every element of `out` and preserves `x`, `base` and `delta`. A shape, dtype, alignment
 *   or overlap violation throws std::invalid_argument.
 */
void grouped_tap_apply(const Tensor& x, const Tensor& base, const Tensor& delta, Tensor& out,
                       std::int32_t side, std::int32_t group_width, std::int32_t block_size,
                       DeviceExecutionView execution);

} // namespace ninfer::ops
