#pragma once

#include "ninfer/types.h"
#include "models/qwen3_5/frontend/output_session.h"
#include "models/registry.h"
#include "runtime/contract/request.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::models::qwen3_5 {

[[nodiscard]] ModelSamplingDefaults default_sampling(Architecture architecture);

struct FrontendOptions {
    std::filesystem::path chat_template_path;
    Architecture architecture              = Architecture::Qwen3_5;
    bool vision_enabled                    = true;
    std::uint32_t max_context              = 2'048;
    std::size_t media_cache_bytes          = kDefaultMediaCacheBytes;
    std::size_t media_live_bytes           = kDefaultMediaLiveBytes;
    std::uint32_t media_preprocess_threads = 0;
    // Per-continuation long-anchor capacity L. When nonzero, preparation synthesizes
    // engine-automatic PrivateLongAnchor opportunities at up to L message boundaries, walking back
    // from the prompt end on a geometrically widening grid (`long_anchor_min_spacing_tokens`), so a
    // later history rewrite diverging there resumes from the retained anchor instead of root.
    // The Engine may raise it after startup through Frontend::publish_long_anchor_limit.
    std::uint32_t max_long_anchors_per_continuation = 0;
    // Minimum token gap between consecutive engine-automatic anchors (and between the deepest
    // prompt-end grid point and the first anchor), doubling per anchor; 0 disables spacing.
    std::uint32_t long_anchor_min_spacing_tokens = 0;
};

struct FrontendResources;
struct PreparedPromptData;
class Frontend;
class FrontendTestAccess;
class PreparedPromptAccess;

class PreparedPrompt {
public:
    PreparedPrompt() noexcept;
    ~PreparedPrompt();
    PreparedPrompt(PreparedPrompt&&) noexcept;
    PreparedPrompt& operator=(PreparedPrompt&&) noexcept;

    PreparedPrompt(const PreparedPrompt&)            = delete;
    PreparedPrompt& operator=(const PreparedPrompt&) = delete;

    [[nodiscard]] PromptSummary summary() const;
    [[nodiscard]] PromptPreparationStats preparation_stats() const noexcept;
    [[nodiscard]] explicit operator bool() const noexcept;

private:
    explicit PreparedPrompt(std::unique_ptr<PreparedPromptData> data) noexcept;
    std::unique_ptr<PreparedPromptData> data_;

    friend class Frontend;
    friend class FrontendTestAccess;
    friend class PreparedPromptAccess;
};

class Frontend {
public:
    Frontend(const Frontend&);
    Frontend& operator=(const Frontend&);
    Frontend(Frontend&&) noexcept;
    Frontend& operator=(Frontend&&) noexcept;
    ~Frontend();

    [[nodiscard]] PreparedPrompt prepare(PromptInput input,
                                         const PreparationControl& control = {}) const;
    [[nodiscard]] std::uint32_t count_tokens(PromptInput input,
                                             const PreparationControl& control = {}) const;
    [[nodiscard]] PreparedPrompt prepare_tokens(std::vector<TokenId> token_ids,
                                                bool allow_prefix_identity = true) const;
    [[nodiscard]] std::vector<TokenId> tokenize_text(std::string_view text) const;
    [[nodiscard]] MediaCacheSummary media_cache_summary() const;
    [[nodiscard]] OutputSession
    make_output_session(const PreparedPrompt& prompt, const StopPolicy& caller_stop,
                        const OutputOptions& output            = {},
                        const ThinkingControlOptions& thinking = {}) const;
    [[nodiscard]] const StopPolicy& default_stop_policy() const noexcept;
    [[nodiscard]] const ModelSamplingDefaults& sampling_defaults() const noexcept;
    // Publishes the resolved long-anchor count. Startup builds the frontend before the sequence
    // plan exists, so the Engine hands the host-cache-resolved value to the grid the capture path
    // will create checkpoints for, before any request is prepared.
    void publish_long_anchor_limit(std::uint32_t anchors) noexcept;

private:
    class Impl;
    explicit Frontend(std::shared_ptr<const Impl> impl) noexcept;
    std::shared_ptr<const Impl> impl_;

    friend class FrontendTestAccess;
    friend Frontend make_frontend(const FrontendResources& resources, FrontendOptions options);
};

[[nodiscard]] Frontend make_frontend(const FrontendResources& resources, FrontendOptions options);

} // namespace ninfer::models::qwen3_5
