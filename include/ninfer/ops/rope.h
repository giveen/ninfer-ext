#pragma once

#include "core/device.h"
#include "core/tensor.h"

namespace ninfer::ops {

/**
 * Applies split-half NeoX RoPE in place. For pair i in [0,rotary_pairs), angle phi(i,t), and
 * each head:
 *
 *   ideal[i]              = x[i] * cos(phi) - x[i+R/2] * sin(phi)
 *   ideal[i+rotary_dim/2] = x[i+R/2] * cos(phi) + x[i] * sin(phi).
 *
 * `rotary_pairs` is the number of rotated pairs, in [1,rotary_dim/2] and at most 128. The
 * frequency denominator remains rotary_dim, so a partial profile rotates its leading pairs at the
 * full rotary span's frequencies rather than at its own span's. Pairs in
 * [rotary_pairs,rotary_dim/2) and their partners in [rotary_dim/2+rotary_pairs,rotary_dim) are
 * bit-exact unchanged, and so are dimensions [rotary_dim,head_dim). Supported modes are:
 *
 * - Text 1-D: positions I32 [T], head_dim=256 with even 0<rotary_dim<=256, or the DFlash
 *   full-head domain head_dim=rotary_dim=128, or the proportional domain
 *   head_dim=rotary_dim=512; phi=positions[t]*theta^(-2*i/rotary_dim).
 * - Text MRoPE: positions I32 [T,3], head_dim=256, rotary_dim=64 and rotary_pairs=32; pair i
 *   uses axis i%3 with the same frequency as Text 1-D.
 * - Vision 2-D: positions I32 [T,2], head_dim=rotary_dim=72 and rotary_pairs=36; pairs 0..17 use
 *   axis 0 and pairs 18..35 use axis 1, each with local frequency theta^(-2*(i%18)/36).
 *
 * positions is contiguous and theta is positive and finite. Q/K tensors are BF16
 * [head_dim,heads,T] with positive head counts, contiguous head features and heads, and an optional
 * padded token stride. The registered optimized domains are D256/R64 Text Q/K head geometries
 * 24/4 and 16/2, D128/R128 1-D Text geometry 32/8, plus Vision geometry 16/16; every other
 * geometry, including the full-span and proportional profiles, takes the generic kernel. q and k
 * must not
 * overlap one another or positions. The Op mutates only dimensions [0,rotary_dim) of the supplied
 * Q/K tensor storage. The oracle evaluates the rotated dimensions naively in FP64 from the
 * represented inputs. The updated BF16 values are promoted and compared directly with that result;
 * output storage rounding belongs to the Op's numerical criterion, not the oracle. Unrotated
 * dimensions remain bit-exact. Private kernel arithmetic is implementation-defined. The Op uses no
 * workspace or persistent state. `execution` supplies the stream and the selected device's
 * positive physical SM count; the count selects launch geometry, not the transformation.
 */
void rope(const Tensor& positions, int rotary_dim, int rotary_pairs, float theta, Tensor& q,
          Tensor& k, DeviceExecutionView execution);

// Single-tensor form with the same formula and storage contract. The head count comes directly
// from x; Q versus K role does not change the transformation.
void rope(const Tensor& positions, int rotary_dim, int rotary_pairs, float theta, Tensor& x,
          DeviceExecutionView execution);

} // namespace ninfer::ops
