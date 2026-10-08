#include "serve/http_server.h"

#include "serve/anthropic_messages.h"
#include "serve/http_transport.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::serve {

namespace {

class MessagesSseEncoder final : public SseStreamEncoder {
public:
    MessagesSseEncoder(AnthropicResponseIdentity identity, int input_tokens, bool hide_thinking)
        : stream_(std::move(identity), input_tokens, hide_thinking) {}

    std::vector<std::string> open() override { return {}; }

    std::vector<std::string> on_start(const ninfer::GenerationStart& start) override {
        return {stream_.start(start)};
    }

    std::vector<std::string> reasoning_delta(const std::string& text) override {
        return stream_.reasoning_delta(text);
    }

    std::vector<std::string> content_delta(const std::string& text) override {
        return stream_.content_delta(text);
    }

    std::vector<std::string> error_event(const ApiError& error) override {
        std::vector<std::string> events;
        if (!stream_.started()) { events.push_back(stream_.start()); }
        events.push_back(stream_.error(error));
        return events;
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

    ApiError normalize(ApiError error) override {
        return normalize_anthropic_error(std::move(error));
    }

    ApiError internal_error(const std::string& message) const override {
        return ApiError{500, "invalid_request_error", message};
    }

private:
    AnthropicMessagesStream stream_;
};

} // namespace

void HttpServer::handle_count_tokens(const httplib::Request& req, httplib::Response& res) {
    const std::string request_id = new_anthropic_request_id();
    res.set_header("request-id", request_id);
    try {
        const AnthropicCountTokensRequest request =
            parse_anthropic_count_tokens_request(parse_json_body(req));
        const int input_tokens = service_->count_prompt_tokens(
            request.generation, [&req] { return client_disconnected(req); });
        res.set_content(make_anthropic_count_tokens_response(input_tokens), "application/json");
    } catch (const ApiException& exception) {
        write_anthropic_error(res, exception.error(), request_id);
    } catch (const std::exception& exception) {
        operational_log_.http_failure(
            "anthropic_count_tokens",
            make_internal_request_failure(RequestFailurePhase::Http, exception.what()), request_id);
        ApiError error;
        error.status  = 500;
        error.message = exception.what();
        write_anthropic_error(res, error, request_id);
    }
}

void HttpServer::handle_messages(const httplib::Request& req, httplib::Response& res) {
    const std::string request_id = new_anthropic_request_id();
    res.set_header("request-id", request_id);

    AnthropicMessagesRequest request;
    try {
        RequestLimits limits;
        limits.default_max_tokens = options_.default_max_tokens;
        request                   = parse_anthropic_messages_request(parse_json_body(req), limits);
    } catch (const ApiException& exception) {
        write_anthropic_error(res, exception.error(), request_id);
        return;
    } catch (const std::exception& exception) {
        operational_log_.http_failure(
            "anthropic_messages",
            make_internal_request_failure(RequestFailurePhase::Http, exception.what()), request_id);
        ApiError error;
        error.status  = 500;
        error.message = exception.what();
        write_anthropic_error(res, error, request_id);
        return;
    }

    const std::uint64_t req_id = ++request_seq_;
    const RequestLogMetadata metadata{.model                  = request.model,
                                      .stream                 = request.stream,
                                      .output_tokens_explicit = request.output_tokens_explicit};
    PreparedRequest prepared;
    try {
        prepared = service_->prepare(request.generation,
                                     request.stream ? GenerationConsumerMode::Streaming
                                                    : GenerationConsumerMode::Aggregate,
                                     ninfer::GenerationObservationOptions{.phase_timings = true},
                                     [&req] { return client_disconnected(req); });
    } catch (const ApiException& exception) {
        const ApiError error = normalize_anthropic_error(exception.error());
        record_request_rejected(make_request_rejection_log_context(
            req_id, "anthropic_messages", request.generation, metadata, error));
        write_anthropic_error(res, error, request_id);
        return;
    } catch (const std::exception& exception) {
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = exception.what();
        record_request_rejected(make_request_rejection_log_context(
            req_id, "anthropic_messages", request.generation, metadata, error));
        write_anthropic_error(res, error, request_id);
        return;
    }

    const AnthropicResponseIdentity identity =
        make_anthropic_response_identity(request_id, request.model);
    const int input_tokens = prepared.prompt_tokens;

    auto lifecycle = begin_request(make_request_log_context(
        req_id, "anthropic_messages", request.generation, metadata, prepared));

    if (!request.stream) {
        GenerationOutcome outcome;
        try {
            outcome = service_->run(prepared, nullptr, [&req] { return client_disconnected(req); });
        } catch (const ApiException& exception) {
            const ApiError error = normalize_anthropic_error(exception.error());
            lifecycle->failure(make_generation_request_failure(error));
            write_anthropic_error(res, error, request_id);
            return;
        } catch (const std::exception& exception) {
            lifecycle->failure(
                make_internal_request_failure(RequestFailurePhase::Generation, exception.what()));
            ApiError error;
            error.status  = 500;
            error.message = exception.what();
            write_anthropic_error(res, error, request_id);
            return;
        }
        lifecycle->done(outcome);
        try {
            set_owned_json_content(
                res, make_anthropic_messages_response(identity, outcome, request.hide_thinking),
                prepared.lifetime);
        } catch (const ApiException& exception) {
            const ApiError error = normalize_anthropic_error(exception.error());
            lifecycle->response_failure(
                make_request_failure(RequestFailurePhase::ResponseRender, error));
            write_anthropic_error(res, error, request_id);
        } catch (const std::exception& exception) {
            lifecycle->response_failure(make_internal_request_failure(
                RequestFailurePhase::ResponseRender, exception.what()));
            ApiError error;
            error.status  = 500;
            error.message = exception.what();
            write_anthropic_error(res, error, request_id);
        }
        return;
    }

    auto stream = std::make_shared<HttpGenerationStream>(std::move(prepared));
    auto encoder =
        std::make_shared<MessagesSseEncoder>(identity, input_tokens, request.hide_thinking);
    stream_generation(
        res, std::move(stream), std::move(lifecycle), std::move(encoder),
        [&res, &request_id](const ApiError& error) {
            write_anthropic_error(res, error, request_id);
        });
}

} // namespace ninfer::serve
