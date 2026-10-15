#pragma once

#include <string>
#include <string_view>
#include "ninfer/types.h"

namespace xgrammar {
class Grammar;
}

namespace ninfer::text {

// Validate the supported JSON Schema dialect, retaining property order and literal data.
// Returns normalized schema text for the vendor converter; errors carry a JSON Pointer.
// Throws RequestError with InvalidJsonSchema, UnsupportedJsonSchema or UnsatisfiableJsonSchema;
// the pointer() identifies the offending schema location.
std::string prepare_json_schema(std::string_view source);

// Build a JSON grammar from a JsonObject or JsonSchema constraint.
// Throws RequestError for any other constraint kind, for literal alternatives, or when the source
// schema fails validation or compilation.
xgrammar::Grammar build_json_grammar(const OutputConstraint& constraint);

} // namespace ninfer::text
