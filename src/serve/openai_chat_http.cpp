#include "serve/http_server.h"

#include "serve/http_transport.h"
#include "serve/openai_chat.h"
#include "serve/openai_common.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::serve {
namespace {

std::string sse_error_event(const ApiError& error) {
    return "data: " + make_error_body(error) + "\n\n";
}

class ChatSseEncoder final : public SseStreamEncoder {
public:
    ChatSseEncoder(OpenAIChatResponseIdentity identity, bool include_usage, bool timings_per_token,
                   bool return_progress)
        : stream_(std::move(identity), include_usage, timings_per_token, return_progress),
          timings_per_token_(timings_per_token), return_progress_(return_progress) {}

    std::vector<std::string> open() override { return {stream_.start()}; }

    std::vector<std::string> on_start(const ninfer::GenerationStart& start) override {
        stream_.note_start(start);
        if (return_progress_) { return {stream_.initial_prompt_progress()}; }
        return {};
    }

    std::vector<std::string> on_progress(const ninfer::PromptProgress& progress) override {
        if (return_progress_) { return {stream_.prompt_progress(progress)}; }
        return {};
    }

    void on_timing(const ninfer::GenerationTimingObservation& timing) override {
        if (timings_per_token_) { stream_.note_timing(timing); }
    }

    std::vector<std::string> reasoning_delta(const std::string& text) override {
        return {stream_.reasoning_delta(text)};
    }

    std::vector<std::string> content_delta(const std::string& text) override {
        return {stream_.content_delta(text)};
    }

    std::vector<std::string> error_event(const ApiError& error) override {
        return {sse_error_event(error)};
    }

    SseStreamClose close(const GenerationOutcome& outcome) override {
        SseStreamClose result;
        try {
            result.events = stream_.finish(outcome);
        } catch (const std::exception& exception) {
            result.failure = make_internal_request_failure(RequestFailurePhase::ResponseRender,
                                                           exception.what());
            result.error   = internal_error(exception.what());
        }
        return result;
    }

private:
    OpenAIChatStream stream_;
    bool timings_per_token_;
    bool return_progress_;
};

} // namespace

void HttpServer::handle_chat_completions(const httplib::Request& req, httplib::Response& res) {
    OpenAIChatRequest request;
    try {
        RequestLimits limits;
        limits.default_max_tokens = options_.default_max_tokens;
        request = parse_chat_completion_request(parse_json_body(req), limits,
                                                options_.auto_system_shared_prefix);
        validate_openai_model(request.model, public_model_id_);
    } catch (const ApiException& exception) {
        write_openai_error(res, exception.error());
        return;
    }

    const std::uint64_t req_id = ++request_seq_;
    const RequestLogMetadata metadata{.model                  = request.model,
                                      .stream                 = request.stream,
                                      .output_tokens_explicit = request.output_tokens_explicit};
    PreparedRequest prepared;
    try {
        const ninfer::GenerationObservationOptions observation{
            .phase_timings   = true,
            .live_timings    = request.stream && request.timings_per_token,
            .prompt_progress = request.stream && request.return_progress,
        };
        prepared = service_->prepare(request.generation,
                                     request.stream ? GenerationConsumerMode::Streaming
                                                    : GenerationConsumerMode::Aggregate,
                                     observation, [&req] { return client_disconnected(req); });
    } catch (const ApiException& exception) {
        record_request_rejected(make_request_rejection_log_context(
            req_id, "openai_chat_completions", request.generation, metadata, exception.error()));
        write_openai_error(res, exception.error());
        return;
    } catch (const std::exception& exception) {
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = exception.what();
        record_request_rejected(make_request_rejection_log_context(
            req_id, "openai_chat_completions", request.generation, metadata, error));
        write_openai_error(res, error);
        return;
    }

    const OpenAIChatResponseIdentity identity = make_openai_chat_response_identity(request.model);
    auto lifecycle                            = begin_request(make_request_log_context(
        req_id, "openai_chat_completions", request.generation, metadata, prepared));

    if (!request.stream) {
        GenerationOutcome outcome;
        try {
            outcome = service_->run(prepared, nullptr, [&req] { return client_disconnected(req); });
        } catch (const ApiException& exception) {
            lifecycle->failure(make_generation_request_failure(exception.error()));
            write_openai_error(res, exception.error());
            return;
        } catch (const std::exception& exception) {
            const RequestFailure failure =
                make_internal_request_failure(RequestFailurePhase::Generation, exception.what());
            lifecycle->failure(failure);
            ApiError error;
            error.status  = 500;
            error.type    = "internal_error";
            error.message = exception.what();
            write_openai_error(res, error);
            return;
        }
        lifecycle->done(outcome);
        try {
            set_owned_json_content(res, make_chat_completion_response(identity, outcome),
                                   prepared.lifetime);
        } catch (const std::exception& exception) {
            lifecycle->response_failure(make_internal_request_failure(
                RequestFailurePhase::ResponseRender, exception.what()));
            ApiError error;
            error.status  = 500;
            error.type    = "internal_error";
            error.message = exception.what();
            write_openai_error(res, error);
        }
        return;
    }

    auto stream = std::make_shared<HttpGenerationStream>(std::move(prepared));
    auto encoder =
        std::make_shared<ChatSseEncoder>(identity, request.include_usage, request.timings_per_token,
                                         request.return_progress);
    stream_generation(res, std::move(stream), std::move(lifecycle), std::move(encoder),
                      [&res](const ApiError& error) { write_openai_error(res, error); });
}

} // namespace ninfer::serve
