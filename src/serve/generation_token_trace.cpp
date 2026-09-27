#include "serve/generation_token_trace.h"

#include <nlohmann/json.hpp>

#include <spdlog/logger.h>

#include <stdexcept>
#include <utility>

namespace ninfer::serve {

std::string format_generation_token_trace_json(std::uint64_t request_id,
                                               std::uint32_t prompt_tokens,
                                               const std::vector<TokenId>& generated_token_ids) {
    // An empty completion is a valid record: skipping it would misalign request order.
    const nlohmann::json record = {
        {"schema", std::string(kGenerationTokenTraceSchema)},
        {"schema_version", kGenerationTokenTraceSchemaVersion},
        {"request_id", request_id},
        {"prompt_tokens", prompt_tokens},
        {"completion_tokens", generated_token_ids.size()},
        {"generated_token_ids", generated_token_ids},
    };
    return record.dump();
}

GenerationTokenTraceJsonl::GenerationTokenTraceJsonl(const std::string& path,
                                                     std::shared_ptr<spdlog::logger> logger)
    : logger_(std::move(logger)) {
    if (path.empty()) { return; }
    output_.open(path, std::ios::out | std::ios::app);
    if (!output_) { throw std::runtime_error("cannot open generation token trace: " + path); }
}

bool GenerationTokenTraceJsonl::write(std::uint64_t request_id, std::uint32_t prompt_tokens,
                                      const std::vector<TokenId>& generated_token_ids) {
    if (!enabled()) { return true; }
    std::lock_guard lock(mutex_);
    if (failed_) { return false; }
    try {
        output_ << format_generation_token_trace_json(request_id, prompt_tokens,
                                                      generated_token_ids)
                << '\n';
        output_.flush();
    } catch (const std::exception& error) {
        if (logger_) { logger_->error("generation token trace write failed: {}", error.what()); }
        failed_ = true;
        return false;
    }
    if (output_) { return true; }
    if (logger_) {
        logger_->error("generation token trace write failed: output stream is not writable");
    }
    failed_ = true;
    return false;
}

} // namespace ninfer::serve
