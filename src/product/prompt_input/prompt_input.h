#pragma once

#include "ninfer/types.h"

#include <filesystem>
#include <string>

namespace ninfer::product {

// Throws std::invalid_argument when `text` is empty.

[[nodiscard]] PromptInput prompt_from_text(std::string text, std::optional<bool> enable_thinking);
// Throws std::runtime_error if the file cannot be opened, and std::invalid_argument on malformed or
// empty JSON.
[[nodiscard]] PromptInput prompt_from_messages(const std::filesystem::path& path,
                                               std::optional<bool> enable_thinking,
                                               bool vision_enabled);

} // namespace ninfer::product
