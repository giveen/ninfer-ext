#pragma once

// Test-only access to the tokenizer's per-family word boundaries. The tokenizer reaches them
// through encode; a test compares them piece by piece against the reference pipeline's own
// pre-tokenization, which is the only way to see a split apart from the merge table it feeds.

#include "models/qwen3_5/frontend/tokenizer.h"

#include <cstddef>
#include <string_view>

namespace ninfer::models::qwen3_5::frontend {

// The end offset of the word beginning at `begin`, exactly as `family`'s declared pre-tokenizer
// splits the text there.
[[nodiscard]] std::size_t tokenizer_word_end(std::string_view text, std::size_t begin,
                                             TokenizerFamily family);

} // namespace ninfer::models::qwen3_5::frontend
