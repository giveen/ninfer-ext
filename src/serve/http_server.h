#pragma once

#include "serve/generation_service.h"
#include "serve/generation_token_trace.h"
#include "serve/http_transport.h"
#include "serve/operational_log.h"
#include "serve/openai_responses_store.h"
#include "serve/request_log.h"
#include "serve/serve_metrics.h"
#include "serve/serve_options.h"

#include <httplib.h>

#include <atomic>
#include <condition_variable>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <memory>
#include <string>
#include <string_view>
#include <thread>

namespace ninfer::serve {

void write_openai_error(httplib::Response& response, const ApiError& error);
void write_anthropic_error(httplib::Response& response, const ApiError& error,
                           const std::string& request_id);

// cpp-httplib invokes the error handler for every application response with status >= 400. Only
// an empty 413 is its own pre-routing payload-limit rejection; application-authored errors must be
// left untouched.
httplib::Server::HandlerResponse handle_unrendered_http_error(const ServeOptions& options,
                                                              const httplib::Request& request,
                                                              httplib::Response& response);

[[nodiscard]] bool matches_bearer_credential(std::string_view authorization,
                                             std::string_view api_key) noexcept;

class HttpServer {
public:
    HttpServer(ServeOptions options, std::shared_ptr<spdlog::logger> logger);

    // Reserves the configured address before model loading. The service is attached only after its
    // Engine is ready, then listen() enters the blocking accept loop on the already-bound socket.
    bool bind();
    void attach(GenerationService& service);
    bool listen();
    void stop();

    [[nodiscard]] const std::string& public_model_id() const noexcept { return public_model_id_; }

private:
    class RequestLifecycle {
    public:
        RequestLifecycle(HttpServer& owner, RequestLogContext context);

        void done(const GenerationOutcome& outcome);
        void failure(const RequestFailure& failure);
        void response_failure(const RequestFailure& failure);

        [[nodiscard]] std::uint64_t request_id() const noexcept { return context_.id; }

    private:
        enum class State : std::uint8_t {
            Pending,
            Done,
            Error,
        };

        [[nodiscard]] bool claim(State terminal) noexcept;

        HttpServer* owner_ = nullptr;
        RequestLogContext context_;
        std::atomic<State> state_{State::Pending};
    };

    [[nodiscard]] std::shared_ptr<RequestLifecycle> begin_request(RequestLogContext context);

    // Drives one streamed response: SSE transport, cancellation latch, engine run, and request
    // lifecycle. The encoder owns the wire shape; write_error renders a protocol JSON error when
    // the stream cannot be set up.
    void stream_generation(httplib::Response& response, std::shared_ptr<HttpGenerationStream> stream,
                           std::shared_ptr<RequestLifecycle> lifecycle,
                           std::shared_ptr<SseStreamEncoder> encoder,
                           std::function<void(const ApiError&)> write_error);

    void register_routes();
    void handle_chat_completions(const httplib::Request& req, httplib::Response& res);
    void handle_messages(const httplib::Request& req, httplib::Response& res);
    void handle_count_tokens(const httplib::Request& req, httplib::Response& res);
    void handle_responses(const httplib::Request& req, httplib::Response& res);
    void handle_response_input_tokens(const httplib::Request& req, httplib::Response& res);
    void handle_response_get(const httplib::Request& req, httplib::Response& res);
    void handle_response_delete(const httplib::Request& req, httplib::Response& res);
    void handle_response_input_items(const httplib::Request& req, httplib::Response& res);
    void handle_response_cancel(const httplib::Request& req, httplib::Response& res);
    void handle_response_compact(const httplib::Request& req, httplib::Response& res);
    void handle_models(const httplib::Request& req, httplib::Response& res) const;
    void handle_model(const httplib::Request& req, httplib::Response& res) const;
    void handle_metrics(const httplib::Request& req, httplib::Response& res) const;

    void record_request_start(const RequestLogContext& context);
    void record_request_rejected(const RequestRejectionLogContext& context);
    void record_request_done(const RequestLogContext& context, const GenerationOutcome& outcome);
    void record_generation_token_trace(const RequestLogContext& context,
                                       const GenerationOutcome& outcome);
    void record_request_failure(const RequestLogContext& context, const RequestFailure& failure);
    void record_response_failure(std::uint64_t request_id, const RequestFailure& failure);
    void record_throughput(const ThroughputReport& report);
    void run_stats_reporter();
    void stop_stats_reporter();

    GenerationService* service_ = nullptr;
    ServeOptions options_;
    std::string public_model_id_;
    ninfer::ModelMetadata model_metadata_;
    OpenAIResponsesStore openai_responses_store_;
    OperationalLog operational_log_;
    JsonlRequestLog request_jsonl_;
    GenerationTokenTraceJsonl generation_token_trace_jsonl_;
    ServeMetrics metrics_;
    httplib::Server server_;
    std::atomic<std::uint64_t> request_seq_{0};
    std::mutex stats_mutex_;
    std::condition_variable stats_cv_;
    std::thread stats_thread_;
    bool stats_stopping_ = false;
};

} // namespace ninfer::serve
