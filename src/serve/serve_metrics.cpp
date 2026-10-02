#include "serve/serve_metrics.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <sstream>
#include <utility>

namespace ninfer::serve {
namespace {

// vLLM's default histogram buckets (vllm/v1/metrics/buckets.py).
std::vector<double> time_to_first_token_buckets() {
    return {0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1,  0.25,  0.5,   0.75,
            1.0,   2.5,   5.0,  7.5,  10.0, 20.0, 40.0, 80.0, 160.0, 640.0, 2560.0};
}

std::vector<double> inter_token_latency_buckets() {
    return {0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2,  0.3,  0.4, 0.5,
            0.75, 1.0,   2.5,  5.0,   7.5, 10.0, 20.0, 40.0, 80.0};
}

std::vector<double> request_latency_buckets() {
    return {0.3,  0.5,  0.8,  1.0,  1.5,  2.0,   2.5,   5.0,   10.0,   15.0,  20.0,
            30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0};
}

// vLLM's 1-2-5 series capped at the maximum context.
std::vector<double> request_token_buckets(std::uint32_t max_context) {
    std::vector<double> bounds;
    for (double scale = 1.0;; scale *= 10.0) {
        for (const double mantissa : {1.0, 2.0, 5.0}) {
            const double value = mantissa * scale;
            if (value > static_cast<double>(max_context)) { return bounds; }
            bounds.push_back(value);
        }
    }
}

std::string format_number(double value) {
    if (std::isinf(value)) { return value > 0 ? "+Inf" : "-Inf"; }
    if (std::isnan(value)) { return "NaN"; }
    char buffer[64];
    const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value);
    return std::string(buffer, result.ptr);
}

class Writer {
public:
    explicit Writer(std::string_view model_name)
        : model_label_("model_name=\"" + escape_label_value(model_name) + "\"") {}

    void family(std::string_view name, std::string_view type, std::string_view help) {
        out_ << "# HELP " << name << ' ' << help << "\n# TYPE " << name << ' ' << type << '\n';
    }

    // One sample; `labels` is extra `key="value"` pairs, already escaped, comma-separated.
    void sample(std::string_view name, double value, std::string_view labels = {}) {
        out_ << name << '{' << model_label_;
        if (!labels.empty()) { out_ << ',' << labels; }
        out_ << "} " << format_number(value) << '\n';
    }

    void scalar(std::string_view name, std::string_view type, std::string_view help,
                double value) {
        family(name, type, help);
        sample(name, value);
    }

    template <class Histogram>
    void histogram(std::string_view name, std::string_view help, const Histogram& h) {
        family(name, "histogram", help);
        const std::string bucket = std::string(name) + "_bucket";
        std::uint64_t cumulative = 0;
        for (std::size_t index = 0; index < h.bounds.size(); ++index) {
            cumulative += h.counts[index];
            sample(bucket, static_cast<double>(cumulative),
                   "le=\"" + format_number(h.bounds[index]) + "\"");
        }
        cumulative += h.counts.back();
        sample(bucket, static_cast<double>(cumulative), "le=\"+Inf\"");
        sample(std::string(name) + "_sum", h.sum);
        sample(std::string(name) + "_count", static_cast<double>(h.count));
    }

    [[nodiscard]] std::string str() const { return out_.str(); }

private:
    std::string model_label_;
    std::ostringstream out_;
};

} // namespace

const char* finished_reason_label(FinishReason reason, bool tool_calls) noexcept {
    if (tool_calls) { return "stop"; }
    switch (reason) {
    case FinishReason::StopToken:
    case FinishReason::StopString:
        return "stop";
    case FinishReason::OutputLimit:
    case FinishReason::ContextCapacity:
        return "length";
    case FinishReason::Cancelled:
        return "abort";
    case FinishReason::None:
        break;
    }
    return "stop";
}

const char* failure_reason_label(RequestFailureClass classification) noexcept {
    switch (classification) {
    case RequestFailureClass::ClientInput:
        return "client_input";
    case RequestFailureClass::ClientDisconnected:
        return "client_disconnected";
    case RequestFailureClass::Overload:
        return "overload";
    case RequestFailureClass::Timeout:
        return "timeout";
    case RequestFailureClass::Unavailable:
        return "unavailable";
    case RequestFailureClass::Upstream:
        return "upstream";
    case RequestFailureClass::Internal:
        return "internal";
    }
    return "internal";
}

std::string escape_label_value(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (const char c : value) {
        switch (c) {
        case '\\':
            out += "\\\\";
            break;
        case '"':
            out += "\\\"";
            break;
        case '\n':
            out += "\\n";
            break;
        default:
            out += c;
        }
    }
    return out;
}

ServeMetrics::Histogram::Histogram(std::vector<double> upper_bounds)
    : bounds(std::move(upper_bounds)), counts(bounds.size() + 1, 0) {}

void ServeMetrics::Histogram::observe(double value) {
    const auto bucket = std::lower_bound(bounds.begin(), bounds.end(), value) - bounds.begin();
    ++counts[static_cast<std::size_t>(bucket)];
    sum += value;
    ++count;
}

ServeMetrics::ServeMetrics(std::uint32_t max_context)
    : time_to_first_token_(time_to_first_token_buckets()),
      e2e_latency_(request_latency_buckets()), queue_time_(request_latency_buckets()),
      prefill_time_(request_latency_buckets()), decode_time_(request_latency_buckets()),
      time_per_output_token_(inter_token_latency_buckets()),
      prompt_tokens_(request_token_buckets(max_context)),
      generation_tokens_(request_token_buckets(max_context)) {
    for (const char* reason : {"stop", "length", "abort"}) { finished_[reason] = 0; }
}

void ServeMetrics::observe_done(const GenerationOutcome& outcome) {
    const GenerationMetrics& metrics = outcome.metrics;
    const int completion             = std::max(outcome.completion_tokens, 0);
    std::lock_guard lock(mutex_);
    ++finished_[finished_reason_label(outcome.finish_reason, !outcome.tool_calls.empty())];
    e2e_latency_.observe(metrics.total_seconds);
    queue_time_.observe(metrics.engine_timing.queue_wait_seconds);
    prompt_tokens_.observe(static_cast<double>(std::max(outcome.prompt_tokens, 0)));
    generation_tokens_.observe(static_cast<double>(completion));
    if (completion > 0) {
        ++first_tokens_;
        time_to_first_token_.observe(metrics.ttft_seconds);
        prefill_time_.observe(metrics.prompt_wall_seconds);
        decode_time_.observe(metrics.generation_wall_seconds);
    }
    if (completion > 1) {
        time_per_output_token_.observe(metrics.generation_wall_seconds /
                                       static_cast<double>(completion - 1));
    }
    spec_drafts_ += metrics.speculative_rounds + metrics.speculative_lookup_rounds;
    spec_draft_tokens_ +=
        metrics.speculative_draft_tokens + metrics.speculative_lookup_draft_tokens;
    spec_accepted_tokens_ +=
        metrics.speculative_accepted_tokens + metrics.speculative_lookup_accepted_tokens;
    const auto& per_position = metrics.speculative_accepted_per_position;
    if (spec_accepted_per_position_.size() < per_position.size()) {
        spec_accepted_per_position_.resize(per_position.size(), 0);
    }
    for (std::size_t position = 0; position < per_position.size(); ++position) {
        spec_accepted_per_position_[position] += per_position[position];
    }
}

void ServeMetrics::observe_failure(RequestFailureClass classification) {
    std::lock_guard lock(mutex_);
    ++failed_[failure_reason_label(classification)];
}

void ServeMetrics::observe_rejection(int http_status) {
    RequestFailureClass classification = RequestFailureClass::Internal;
    if (http_status == 429) {
        classification = RequestFailureClass::Overload;
    } else if (http_status == 503) {
        classification = RequestFailureClass::Unavailable;
    } else if (http_status == 504 || http_status == 408) {
        classification = RequestFailureClass::Timeout;
    } else if (http_status >= 400 && http_status < 500) {
        classification = RequestFailureClass::ClientInput;
    }
    observe_failure(classification);
}

void ServeMetrics::attach(std::string model_name, const MemorySummary& memory,
                          const RuntimeStats& baseline) {
    model_name_         = std::move(model_name);
    kv_capacity_tokens_ = memory.kv_capacity;
    kv_capacity_pages_  = memory.kv_capacity_page_groups;
    baseline_           = baseline;
}

std::string ServeMetrics::render(const RuntimeStats& runtime) const {
    Writer w(model_name_);
    // Engine counters are monotonic from Engine construction; report serving traffic only.
    const auto since = [](std::uint64_t current, std::uint64_t base) {
        return static_cast<double>(current >= base ? current - base : 0);
    };

    w.scalar("ninfer:num_requests_running", "gauge",
             "Number of requests occupying an Engine lane (prefill or decode).",
             runtime.running_requests);
    w.scalar("ninfer:num_requests_waiting", "gauge",
             "Number of requests submitted to the Engine and waiting for a lane.",
             runtime.waiting_requests);
    const double kv_pages = static_cast<double>(kv_capacity_pages_);
    w.scalar("ninfer:kv_cache_usage_perc", "gauge",
             "Device Main KV pages occupied (written, reserved for growth, or retained by the "
             "context cache, which admission pressure can evict) over the KV pool. 1 means 100 "
             "percent usage.",
             kv_pages > 0.0 ? static_cast<double>(runtime.device_main_kv_occupied_pages) / kv_pages
                            : 0.0);
    w.scalar("ninfer:kv_cache_capacity_tokens", "gauge",
             "Device Main KV pool capacity in tokens.", kv_capacity_tokens_);

    const double cached   = since(runtime.reused_prompt_tokens, baseline_.reused_prompt_tokens);
    const double computed =
        since(runtime.computed_prefill_tokens, baseline_.computed_prefill_tokens);
    w.scalar("ninfer:prompt_tokens_total", "counter",
             "Number of prefill tokens processed: computed plus reused from the prefix cache.",
             computed + cached);
    w.scalar("ninfer:prompt_tokens_cached_total", "counter",
             "Number of prompt tokens reused from the prefix cache.", cached);
    w.scalar("ninfer:prefix_cache_queries_total", "counter",
             "Prefix cache queries, in terms of number of queried tokens.", computed + cached);
    w.scalar("ninfer:prefix_cache_hits_total", "counter",
             "Prefix cache hits, in terms of number of cached tokens.", cached);

    std::lock_guard lock(mutex_);
    w.scalar("ninfer:generation_tokens_total", "counter",
             "Number of generation tokens produced: tokens committed by decode rounds, plus each "
             "finished request's first token.",
             since(runtime.committed_decode_tokens, baseline_.committed_decode_tokens) +
                 static_cast<double>(first_tokens_));
    const double rounds = since(runtime.decode_rounds, baseline_.decode_rounds);
    w.scalar("ninfer:n_decode_total", "counter", "Number of decode batch executions.", rounds);
    w.scalar("ninfer:n_busy_slots_per_decode", "gauge",
             "Average number of requests per decode batch execution since startup.",
             rounds > 0.0
                 ? since(runtime.decode_row_rounds, baseline_.decode_row_rounds) / rounds
                 : 0.0);

    w.family("ninfer:request_success_total", "counter",
             "Count of successfully processed requests, by finish reason.");
    for (const auto& [reason, count] : finished_) {
        w.sample("ninfer:request_success_total", static_cast<double>(count),
                 "finished_reason=\"" + reason + "\"");
    }
    w.family("ninfer:request_failure_total", "counter",
             "Count of requests that failed or were rejected, by reason.");
    for (const auto& [reason, count] : failed_) {
        w.sample("ninfer:request_failure_total", static_cast<double>(count),
                 "reason=\"" + reason + "\"");
    }

    w.histogram("ninfer:time_to_first_token_seconds",
                "Histogram of time to first token in seconds.", time_to_first_token_);
    w.histogram("ninfer:e2e_request_latency_seconds",
                "Histogram of e2e request latency in seconds.", e2e_latency_);
    w.histogram("ninfer:request_queue_time_seconds",
                "Histogram of time spent waiting for an Engine lane.", queue_time_);
    w.histogram("ninfer:request_prefill_time_seconds",
                "Histogram of time from admission to the first output token.", prefill_time_);
    w.histogram("ninfer:request_decode_time_seconds",
                "Histogram of time from the first to the last output token.", decode_time_);
    w.histogram("ninfer:request_time_per_output_token_seconds",
                "Histogram of decode time per output token after the first, per request.",
                time_per_output_token_);
    w.histogram("ninfer:request_prompt_tokens", "Number of prompt tokens per request.",
                prompt_tokens_);
    w.histogram("ninfer:request_generation_tokens", "Number of generation tokens per request.",
                generation_tokens_);

    w.scalar("ninfer:spec_decode_num_drafts_total", "counter",
             "Number of speculative verification rounds (MTP, DFlash and prompt lookup).",
             static_cast<double>(spec_drafts_));
    w.scalar("ninfer:spec_decode_num_draft_tokens_total", "counter",
             "Number of draft tokens proposed.", static_cast<double>(spec_draft_tokens_));
    w.scalar("ninfer:spec_decode_num_accepted_tokens_total", "counter",
             "Number of draft tokens accepted.", static_cast<double>(spec_accepted_tokens_));
    w.family("ninfer:spec_decode_num_accepted_tokens_per_pos_total", "counter",
             "Accepted draft tokens by draft position (model drafts only).");
    for (std::size_t position = 0; position < spec_accepted_per_position_.size(); ++position) {
        w.sample("ninfer:spec_decode_num_accepted_tokens_per_pos_total",
                 static_cast<double>(spec_accepted_per_position_[position]),
                 "position=\"" + std::to_string(position) + "\"");
    }
    return w.str();
}

} // namespace ninfer::serve
