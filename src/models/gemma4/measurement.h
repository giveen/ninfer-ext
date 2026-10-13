#pragma once

#include "models/gemma4/model.h"

#include <string>

namespace ninfer::models::gemma4 {

// Measurement applicability, independent of checkpoint names, object IDs, payload values and
// device addresses. This key never selects or admits an execution implementation.
[[nodiscard]] std::string prefill_signature(const Model& model);

} // namespace ninfer::models::gemma4
