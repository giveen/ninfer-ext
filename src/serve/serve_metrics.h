#pragma once

#include "ninfer/types.h"
#include "serve/generation_service.h"
#include "serve/request_events.h"

#include <array>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

// Prometheus metrics for GET /metrics. Names follow vLLM's (`vllm:` -> `ninfer:`) and, where vLLM
// has no counterpart, llama.cpp's. Per-request observations (histograms, finish reasons,
// speculative decoding) are recorded once when the request reaches its terminal record; token
// counters and gauges are read from the Engine's published RuntimeStats at scrape time, so they
// advance while a long request is still running. docs/serving.md lists every metric.
class ServeMetrics {
public:
    explicit ServeMetrics(std::uint32_t max_context);

    // Called once the Engine is ready. `memory` supplies the fixed KV pool size; it is not read
    // per scrape because Engine::memory_summary() waits for the execution lock. Engine counters
    // are reported relative to `baseline`, which excludes startup warmup.
    void attach(std::string model_name, const MemorySummary& memory, const RuntimeStats& baseline);

    void observe_done(const GenerationOutcome& outcome);
    void observe_failure(RequestFailureClass classification);
    // A request refused before it reached the Engine, classified from its HTTP status.
    void observe_rejection(int http_status);

    // Text exposition format 0.0.4; `runtime` is the Engine's current published RuntimeStats.
    [[nodiscard]] std::string render(const RuntimeStats& runtime) const;

    static constexpr std::string_view kContentType = "text/plain; version=0.0.4; charset=utf-8";

private:
    struct Histogram {
        std::vector<double> bounds;
        std::vector<std::uint64_t> counts; // per bucket, not cumulative; last is +Inf
        double sum          = 0.0;
        std::uint64_t count = 0;

        explicit Histogram(std::vector<double> upper_bounds);
        void observe(double value);
    };

    std::string model_name_;
    std::uint32_t kv_capacity_tokens_ = 0;
    std::uint32_t kv_capacity_pages_  = 0;
    RuntimeStats baseline_;

    mutable std::mutex mutex_;
    Histogram time_to_first_token_;
    Histogram e2e_latency_;
    Histogram queue_time_;
    Histogram prefill_time_;
    Histogram decode_time_;
    Histogram time_per_output_token_;
    Histogram prompt_tokens_;
    Histogram generation_tokens_;
    std::map<std::string, std::uint64_t> finished_;
    std::map<std::string, std::uint64_t> failed_;
    std::uint64_t first_tokens_          = 0;
    std::uint64_t spec_drafts_           = 0;
    std::uint64_t spec_draft_tokens_     = 0;
    std::uint64_t spec_accepted_tokens_  = 0;
    std::vector<std::uint64_t> spec_accepted_per_position_;
    // Settled constrained requests: completion outcome and compile-cache access, plus the work the
    // constraint itself reported as subintervals of the request's existing timings.
    std::array<std::uint64_t, 3> constraint_outcomes_{};
    std::array<std::uint64_t, 3> constraint_cache_{};
    double constraint_prepare_seconds_     = 0;
    double constraint_mask_seconds_        = 0;
    double constraint_matcher_seconds_     = 0;
    std::uint64_t constraint_positions_    = 0;
    std::uint64_t constraint_upload_bytes_ = 0;
};

// vLLM `finished_reason` label of a completed request: stop, length or abort.
[[nodiscard]] const char* finished_reason_label(FinishReason reason, bool tool_calls) noexcept;
// `reason` label of a failed request.
[[nodiscard]] const char* failure_reason_label(RequestFailureClass classification) noexcept;
// Prometheus label-value escaping: backslash, double quote and newline.
[[nodiscard]] std::string escape_label_value(std::string_view value);

} // namespace ninfer::serve
