#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace ninfer::perplexity {

struct CorpusStream {
    std::string id;
    std::string domain;
    std::filesystem::path path;
    std::string text;
};

struct CorpusSelection {
    std::string corpus_id;
    std::string mode;
    std::filesystem::path source;
    std::vector<CorpusStream> streams;
};

// Both loaders signal every failure by throwing std::runtime_error: an unreadable or malformed
// manifest (invalid JSON, a missing `streams`/`modes` block, an empty/duplicate/unknown stream id,
// or a stream path that is absolute or escapes the manifest), a text file that cannot be read, is
// empty, or holds a NUL byte or an invalid UTF-8 sequence. A returned selection always holds at
// least one stream.

[[nodiscard]] CorpusSelection load_corpus(const std::filesystem::path& manifest, bool quick);
[[nodiscard]] CorpusSelection load_custom_text(const std::filesystem::path& path);

} // namespace ninfer::perplexity
