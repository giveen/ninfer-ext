#pragma once

#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::text {
struct ParsedJsonNumbers {
    nlohmann::ordered_json value;
    std::vector<std::string> inexact_numbers;
};

// Build the ordinary ordered DOM in one pass, retaining locations whose numeric spelling
// would change value. Consumers apply the precision contract only to active schemas.
// Malformed input throws nlohmann::json::parse_error.
[[nodiscard]]
ParsedJsonNumbers parse_json_numbers(std::string_view source);

// The JSON-pointer location, absolute in the document, of the first number whose spelling the DOM
// did not preserve. An empty `schema_pointer` selects the root document; a non-empty one must be a
// valid JSON pointer (it is parsed, so a malformed pointer throws nlohmann::json::parse_error).
// Returns nothing when the document does not contain the pointer or holds no such number.
[[nodiscard]]
std::optional<std::string> inexact_schema_number(const ParsedJsonNumbers& parsed,
                                                 const std::string& schema_pointer = {});
} // namespace ninfer::text
