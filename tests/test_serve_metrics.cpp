#include "serve/serve_metrics.h"

#include <cstdint>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace ninfer::serve;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

struct Sample {
    std::string name;
    std::string labels;
    double value = 0.0;
};

struct Exposition {
    std::map<std::string, std::string> types; // family -> type
    std::set<std::string> helped;
    std::vector<Sample> samples;
    std::vector<std::string> malformed;
};

Exposition parse(const std::string& text) {
    Exposition out;
    std::istringstream lines(text);
    std::string line;
    while (std::getline(lines, line)) {
        if (line.starts_with("# HELP ")) {
            out.helped.insert(line.substr(7, line.find(' ', 7) - 7));
        } else if (line.starts_with("# TYPE ")) {
            const std::size_t name_end = line.find(' ', 7);
            out.types[line.substr(7, name_end - 7)] = line.substr(name_end + 1);
        } else {
            const std::size_t open  = line.find('{');
            const std::size_t close = line.rfind("} ");
            if (open == std::string::npos || close == std::string::npos || close < open) {
                out.malformed.push_back(line);
                continue;
            }
            out.samples.push_back({line.substr(0, open), line.substr(open + 1, close - open - 1),
                                   std::stod(line.substr(close + 2))});
        }
    }
    return out;
}

std::string family_of(const std::string& sample, const std::map<std::string, std::string>& types) {
    if (types.contains(sample)) { return sample; }
    for (const char* suffix : {"_bucket", "_sum", "_count"}) {
        const std::string s(suffix);
        if (sample.ends_with(s)) {
            const std::string base = sample.substr(0, sample.size() - s.size());
            if (types.contains(base) && types.at(base) == "histogram") { return base; }
        }
    }
    return {};
}

double value_of(const Exposition& e, const std::string& name, const std::string& label = {}) {
    for (const Sample& s : e.samples) {
        if (s.name == name && (label.empty() || s.labels.find(label) != std::string::npos)) {
            return s.value;
        }
    }
    return -1.0;
}

GenerationOutcome outcome(ninfer::FinishReason reason, int prompt, int completion, double total) {
    GenerationOutcome o;
    o.finish_reason                   = reason;
    o.prompt_tokens                   = prompt;
    o.completion_tokens               = completion;
    o.metrics.total_seconds           = total;
    o.metrics.ttft_seconds            = 0.2;
    o.metrics.prompt_wall_seconds     = 0.15;
    o.metrics.generation_wall_seconds = total - 0.2;
    return o;
}

} // namespace

int main() {
    int failures = 0;

    failures += check(escape_label_value("a\\b\"c\nd") == "a\\\\b\\\"c\\nd",
                      "label escaping must cover backslash, quote and newline");
    using ninfer::FinishReason;
    failures += check(std::string(finished_reason_label(FinishReason::StopToken, false)) == "stop" &&
                          std::string(finished_reason_label(FinishReason::StopString, false)) ==
                              "stop" &&
                          std::string(finished_reason_label(FinishReason::OutputLimit, false)) ==
                              "length" &&
                          std::string(finished_reason_label(FinishReason::ContextCapacity,
                                                            false)) == "length" &&
                          std::string(finished_reason_label(FinishReason::Cancelled, false)) ==
                              "abort" &&
                          std::string(finished_reason_label(FinishReason::OutputLimit, true)) ==
                              "stop",
                      "finish reasons must map to vLLM's stop/length/abort");

    ninfer::MemorySummary memory;
    memory.kv_capacity             = 262144;
    memory.kv_capacity_page_groups = 4096;
    ninfer::RuntimeStats warmup; // Engine startup traffic, excluded from counters.
    warmup.computed_prefill_tokens = 11;
    warmup.committed_decode_tokens = 2;
    warmup.decode_rounds           = 2;
    warmup.decode_row_rounds       = 2;

    ServeMetrics metrics(262144);
    metrics.attach("qwen\"3.8", memory, warmup);
    metrics.observe_done(outcome(FinishReason::StopToken, 24, 456, 5.5));
    metrics.observe_done(outcome(FinishReason::OutputLimit, 2048, 512, 6.0));
    metrics.observe_done(outcome(FinishReason::Cancelled, 100, 0, 0.05));
    metrics.observe_failure(RequestFailureClass::ClientDisconnected);
    metrics.observe_rejection(429);

    ninfer::RuntimeStats runtime;
    runtime.running_requests              = 2;
    runtime.waiting_requests              = 3;
    runtime.computed_prefill_tokens       = 2011;
    runtime.reused_prompt_tokens          = 172;
    runtime.committed_decode_tokens       = 968;
    runtime.decode_rounds                 = 502;
    runtime.decode_row_rounds             = 968;
    runtime.device_main_kv_occupied_pages = 1024;

    const std::string text = metrics.render(runtime);
    const Exposition e     = parse(text);

    for (const std::string& line : e.malformed) {
        failures += check(false, "malformed sample line: " + line);
    }
    for (const Sample& s : e.samples) {
        const std::string family = family_of(s.name, e.types);
        failures += check(!family.empty(), "sample without a # TYPE family: " + s.name);
        failures += check(e.helped.contains(family), "family without # HELP: " + family);
        failures += check(s.name.starts_with("ninfer:"), "metric outside ninfer: " + s.name);
        failures += check(s.labels.starts_with("model_name=\"qwen\\\"3.8\""),
                          "model_name label missing or unescaped on " + s.name);
        if (e.types.contains(family) && e.types.at(family) == "counter") {
            failures += check(s.name.ends_with("_total"), "counter without _total: " + s.name);
        }
    }

    // Every histogram: cumulative non-decreasing buckets, +Inf equals _count.
    for (const auto& [family, type] : e.types) {
        if (type != "histogram") { continue; }
        double previous = 0.0;
        double inf      = -1.0;
        for (const Sample& s : e.samples) {
            if (s.name != family + "_bucket") { continue; }
            failures += check(s.value >= previous, family + " buckets are not cumulative");
            previous = s.value;
            if (s.labels.find("le=\"+Inf\"") != std::string::npos) { inf = s.value; }
        }
        failures += check(inf >= 0.0 && inf == value_of(e, family + "_count"),
                          family + " +Inf bucket differs from _count");
    }

    failures += check(value_of(e, "ninfer:e2e_request_latency_seconds_count") == 3,
                      "e2e latency must count every completed request");
    failures += check(value_of(e, "ninfer:time_to_first_token_seconds_count") == 2,
                      "TTFT must skip requests without an output token");
    failures += check(value_of(e, "ninfer:request_time_per_output_token_seconds_count") == 2,
                      "time per output token needs at least two tokens");
    failures += check(value_of(e, "ninfer:request_success_total", "finished_reason=\"stop\"") == 1 &&
                          value_of(e, "ninfer:request_success_total",
                                   "finished_reason=\"length\"") == 1 &&
                          value_of(e, "ninfer:request_success_total",
                                   "finished_reason=\"abort\"") == 1,
                      "request_success_total by finished_reason");
    failures += check(value_of(e, "ninfer:request_failure_total",
                               "reason=\"client_disconnected\"") == 1 &&
                          value_of(e, "ninfer:request_failure_total", "reason=\"overload\"") == 1,
                      "request_failure_total by reason");
    failures += check(value_of(e, "ninfer:prompt_tokens_total") == 2172 &&
                          value_of(e, "ninfer:prompt_tokens_cached_total") == 172,
                      "prompt token counters come from RuntimeStats minus warmup");
    failures += check(value_of(e, "ninfer:n_decode_total") == 500 &&
                          value_of(e, "ninfer:n_busy_slots_per_decode") == 966.0 / 500.0,
                      "decode counters exclude warmup");
    failures += check(value_of(e, "ninfer:generation_tokens_total") == 968,
                      "generation tokens add each finished request's first token");
    failures += check(value_of(e, "ninfer:kv_cache_usage_perc") == 0.25,
                      "KV usage is occupied pages over pool pages");
    failures += check(value_of(e, "ninfer:num_requests_running") == 2 &&
                          value_of(e, "ninfer:num_requests_waiting") == 3,
                      "request gauges come from RuntimeStats");

    if (failures != 0) {
        std::cerr << text;
        return 1;
    }
    std::cout << "serve metrics OK\n";
    return 0;
}
