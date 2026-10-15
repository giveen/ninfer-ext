#pragma once

#include "runtime/contract/resources.h"

#include <cstddef>

namespace ninfer::runtime {

// Resolves the Main KV capacity a target may reserve out of `available_runtime_bytes`, which is the
// device memory left after the weights, under `policy` and the target's `curve`.
//   Explicit: `policy.explicit_tokens` fixes the page count; `policy.automatic_headroom_bytes`
//             must be zero.
//   Automatic: `automatic_headroom_bytes` is withheld from the budget first; the curve's largest
//              page count that still fits is then chosen.
// Throws std::invalid_argument for an invalid curve, an unknown policy mode, a zero explicit
// capacity, an explicit policy that carries automatic headroom, or a budget smaller than the
// required reservation; std::overflow_error when the resolved page count or token capacity does not
// fit its type. The returned resolution is always within the curve.

[[nodiscard]] KvCapacityResolution resolve_kv_capacity(const KvCapacityPolicy& policy,
                                                       const SequenceCapacityCurve& curve,
                                                       std::size_t available_runtime_bytes);

} // namespace ninfer::runtime
