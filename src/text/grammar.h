#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::text {

// One immutable vocabulary/compiler per model; one transactional matcher per request.
class GrammarSession {
public:
    ~GrammarSession();
    GrammarSession(GrammarSession&&) noexcept;
    GrammarSession& operator=(GrammarSession&&) noexcept;
    // Mask for the accepted state and one per draft token, laid out consecutively. Tokens that
    // cannot continue the language are cleared; a position with no legal token at all sets bit
    // `position` in the result and leaves a placeholder mask for the caller to discard.
    [[nodiscard]] std::uint32_t masks(std::span<const std::int32_t> drafts,
                                      std::span<std::uint32_t> words);
    // Tentatively accept a sampled token; `confirm` keeps it and `discard` rolls it back.
    void accept(std::int32_t token);
    void confirm() noexcept;
    void discard();
    [[nodiscard]] std::size_t mask_words() const noexcept;

private:
    class Impl;
    explicit GrammarSession(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
    friend class GrammarCompiler;
};

class GrammarCompiler {
public:
    // Empty entries explicitly identify forbidden/special IDs. EOS is supplied separately.
    GrammarCompiler(std::vector<std::string> vocabulary, std::vector<std::int32_t> eos,
                    std::size_t cache_bytes);
    ~GrammarCompiler();
    // `reasoning_close` frames the constrained text after the model's thinking marker and
    // `continuation` is assistant text already generated; both may be empty.
    [[nodiscard]] std::unique_ptr<GrammarSession> compile(const std::string& source,
                                                          std::string_view reasoning_close,
                                                          std::string_view continuation);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ninfer::text
