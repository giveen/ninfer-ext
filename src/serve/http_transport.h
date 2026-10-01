#pragma once

#include "serve/generation_service.h"
#include "serve/request_events.h"
#include "serve/request_json.h"

#include <httplib.h>

#include <atomic>
#include <chrono>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::serve {

class ClientDisconnected final : public std::exception {
public:
    [[nodiscard]] const char* what() const noexcept override { return "client disconnected"; }
};

class ResponseRenderFailure final : public std::runtime_error {
public:
    explicit ResponseRenderFailure(const std::string& message) : std::runtime_error(message) {}
};

struct HttpGenerationStream {
    explicit HttpGenerationStream(PreparedRequest request) : prepared(std::move(request)) {}
    HttpGenerationStream() = default;

    PreparedRequest prepared;
    std::atomic<bool> cancelled{false};
    std::atomic<bool> started{false};
};

class SseTransport final {
public:
    using Clock = std::chrono::steady_clock;

    static constexpr std::chrono::seconds kHeartbeatInterval{5};
    static constexpr std::string_view kHeartbeatComment = ": keep-alive\n\n";

    explicit SseTransport(httplib::DataSink& sink, std::atomic<bool>& cancelled,
                          Clock::duration heartbeat_interval = kHeartbeatInterval,
                          Clock::time_point now              = Clock::now());

    void write(std::string_view item, Clock::time_point now = Clock::now());
    void write(const std::vector<std::string>& items, Clock::time_point now = Clock::now());

    // Called by the Engine wait loop. Besides observing an already-closed socket, a quiet stream
    // periodically writes an SSE comment so TCP_USER_TIMEOUT has traffic with which to detect an
    // unacknowledged peer. A failed probe marks the request for cancellation.
    [[nodiscard]] bool poll(Clock::time_point now = Clock::now());

private:
    bool mark_cancelled() noexcept;

    httplib::DataSink& sink_;
    std::atomic<bool>& cancelled_;
    Clock::duration heartbeat_interval_;
    Clock::time_point last_write_;
};

// Result of a protocol stream's final step. When `failure` is set the driver reports it and sends
// `error` instead of `events`, so a closing step keeps ownership of its failure phase.
struct SseStreamClose {
    std::vector<std::string> events;
    std::optional<RequestFailure> failure;
    ApiError error;
};

// The wire half of one streamed response. HttpServer::stream_generation owns the transport, the
// cancellation latch, the engine run, and the request lifecycle; an encoder owns only the shape of
// the events. Any method that produces events may throw ResponseRenderFailure (a render error, which
// the driver reports against ResponseRender) or ClientDisconnected (a transport error).
class SseStreamEncoder {
public:
    virtual ~SseStreamEncoder() = default;

    // Events written before the engine runs.
    virtual std::vector<std::string> open() = 0;
    // The engine's generation start; may emit the stream's opening events.
    virtual std::vector<std::string> on_start(const ninfer::GenerationStart&) { return {}; }
    virtual std::vector<std::string> on_progress(const ninfer::PromptProgress&) { return {}; }
    virtual void on_timing(const ninfer::GenerationTimingObservation&) {}
    virtual std::vector<std::string> reasoning_delta(const std::string& text) = 0;
    virtual std::vector<std::string> content_delta(const std::string& text) = 0;
    // The protocol's error event(s) for an already-normalized error.
    virtual std::vector<std::string> error_event(const ApiError& error) = 0;
    // Everything still to write after the engine run, including any store commit.
    virtual SseStreamClose close(const GenerationOutcome& outcome) = 0;
    // The protocol's error normalization for an Engine exception (identity by default).
    virtual ApiError normalize(ApiError error) { return error; }
    // The protocol's internal (500) error shape.
    virtual ApiError internal_error(const std::string& message) const {
        return ApiError{500, "internal_error", message};
    }
};

RequestJson parse_json_body(const httplib::Request& request);
[[nodiscard]] bool client_disconnected(const httplib::Request& request);

void prepare_sse_response(httplib::Response& response);
void configure_http_server_socket(socket_t socket) noexcept;
void set_owned_json_content(httplib::Response& response, std::string body,
                            std::shared_ptr<RequestLifetime> lifetime);

} // namespace ninfer::serve
