#pragma once

#include "ninfer/types.h"

namespace ninfer::runtime {

// Resolves one request at the Engine boundary. The registered preset supplies every omitted
// model-owned field; an omitted seed remains deterministic for direct Engine callers. Throws
// std::invalid_argument when a resolved value is not finite, temperature is outside [0,2], top_k
// is outside [1,20] (an override of zero resolves to the target cap of 20), top_p or min_p is
// outside [0,1], or either penalty is outside [-2,2].
[[nodiscard]] ResolvedSamplingParameters resolve_sampling(const ModelSamplingDefaults& defaults,
                                                          SamplingMode mode,
                                                          const SamplingOverrides& overrides);

} // namespace ninfer::runtime
