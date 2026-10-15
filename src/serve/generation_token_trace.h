#pragma once

#include "ninfer/types.h"

#include <atomic>

#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace spdlog {
class logger;
}

namespace ninfer::serve {

inline constexpr std::string_view kGenerationTokenTraceSchema     = "ninfer_generated_token_trace";
inline constexpr std::uint32_t kGenerationTokenTraceSchemaVersion = 2;

// `prompt_token_ids` carry the exact Engine prompt (chat template included) so a ledger can be
// replayed offline; they are recorded only while this trace is enabled. The record is therefore
// sensitive: it holds raw user prompt tokens in plaintext, and the file is opened for append with
// the process umask and is never rotated or truncated, so an operator who enables it must choose an
// access-controlled path and expect the file to grow.
[[nodiscard]] std::string
format_generation_token_trace_json(std::uint64_t request_id, std::uint32_t prompt_tokens,
                                   const std::vector<TokenId>& prompt_token_ids,
                                   const std::vector<TokenId>& generated_token_ids);

// Opt-in local trace output, separate from the stable request-measurement JSONL schema. It stores
// only Engine token IDs and counts; protocol payloads and ordinary request logs are unchanged.
class GenerationTokenTraceJsonl {
public:
    explicit GenerationTokenTraceJsonl(const std::string& path,
                                       std::shared_ptr<spdlog::logger> logger = {});

    GenerationTokenTraceJsonl(const GenerationTokenTraceJsonl&)            = delete;
    GenerationTokenTraceJsonl& operator=(const GenerationTokenTraceJsonl&) = delete;

    // Also false once a write has failed, so a caller stops collecting prompt tokens for a trace
    // that can no longer record them. Read before the writer takes `mutex_`, so it stays lock-free.

    [[nodiscard]] bool enabled() const noexcept { return output_.is_open() && !failed_.load(std::memory_order_relaxed); }

    [[nodiscard]] bool write(std::uint64_t request_id, std::uint32_t prompt_tokens,
                             const std::vector<TokenId>& prompt_token_ids,
                             const std::vector<TokenId>& generated_token_ids);

private:
    std::ofstream output_;
    std::mutex mutex_;
    std::shared_ptr<spdlog::logger> logger_;
    // Written by the writer thread alone and read by enabled() on any thread.
    std::atomic<
    bool> failed_{ false};
};

} // namespace ninfer::serve
