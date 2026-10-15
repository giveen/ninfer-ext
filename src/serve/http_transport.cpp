#include "serve/http_transport.h"

#include "serve/http_server.h"
#include "serve/request_validation.h"
#include "text/json_input.h"

#if defined(__linux__)
#    include <netinet/tcp.h>
#    include <sys/socket.h>
#elif defined(_WIN32)
// httplib has already included <winsock2.h>/<ws2tcpip.h> for the socket_t it hands this file; the
// TCP-level option names below come from there, so the includes are repeated to keep the block
// self-contained.
#    include <winsock2.h>
#    include <ws2tcpip.h>
#endif

#include <stdexcept>
#include <utility>

namespace ninfer::serve {
namespace {

#if defined(__linux__)
constexpr int kKeepAliveIdleSeconds                = 10;
constexpr int kKeepAliveIntervalSeconds            = 3;
constexpr int kKeepAliveProbeCount                 = 3;
constexpr unsigned int kTcpUserTimeoutMilliseconds = 15000;

template <class T>
void set_socket_option(socket_t socket, int level, int option, const T& value) noexcept {
    (void)::setsockopt(socket, level, option, &value, sizeof(value));
}
#elif defined(_WIN32)
// The same keep-alive timers as the Linux block, with the meanings Windows gives them: idle
// seconds before the first probe, seconds between probes, and failed probes before the connection
// is dropped.
constexpr int kKeepAliveIdleSeconds     = 10;
constexpr int kKeepAliveIntervalSeconds = 3;
constexpr int kKeepAliveProbeCount      = 3;

template <class T>
void set_socket_option(socket_t socket, int level, int option, const T& value) noexcept {
    // Winsock spells the option value as a char pointer where POSIX spells it as a void pointer.
    (void)::setsockopt(socket, level, option, reinterpret_cast<const char*>(&value),
                       static_cast<int>(sizeof(value)));
}
#endif

} // namespace

RequestJson parse_json_body(const httplib::Request& request) {
    // Parse once, retaining schema locations whose numeric spelling would change value: a schema
    // that cannot round-trip through the JSON number representation is refused here, before it can
    // describe a language the engine could not have generated.
    text::ParsedJsonNumbers parsed;
    try {
        parsed = text::parse_json_numbers(request.body);
    } catch (const std::exception&) { bad_request("request body is not valid JSON"); }
    validate_schema_number_input(parsed);
    return std::move(parsed.value);
}

bool client_disconnected(const httplib::Request& request) { return request.is_connection_closed(); }

void prepare_sse_response(httplib::Response& response) {
    response.set_header("Cache-Control", "no-cache");
    response.set_header("X-Accel-Buffering", "no");
}

SseTransport::SseTransport(httplib::DataSink& sink, std::atomic<bool>& cancelled,
                           Clock::duration heartbeat_interval, Clock::time_point now)
    : sink_(sink), cancelled_(cancelled), heartbeat_interval_(heartbeat_interval),
      last_write_(now) {
    if (heartbeat_interval_ <= Clock::duration::zero()) {
        throw std::invalid_argument("SSE heartbeat interval must be positive");
    }
}

bool SseTransport::mark_cancelled() noexcept {
    cancelled_.store(true, std::memory_order_release);
    return true;
}

void SseTransport::write(std::string_view item, Clock::time_point now) {
    if (cancelled_.load(std::memory_order_acquire) || !sink_.write(item.data(), item.size())) {
        mark_cancelled();
        throw ClientDisconnected();
    }
    last_write_ = now;
}

void SseTransport::write(const std::vector<std::string>& items, Clock::time_point now) {
    for (const std::string& item : items) { write(item, now); }
}

bool SseTransport::poll(Clock::time_point now) {
    if (cancelled_.load(std::memory_order_acquire)) { return true; }
    if (sink_.is_writable && !sink_.is_writable()) { return mark_cancelled(); }
    if (now - last_write_ < heartbeat_interval_) { return false; }
    if (!sink_.write(kHeartbeatComment.data(), kHeartbeatComment.size())) {
        return mark_cancelled();
    }
    last_write_ = now;
    return false;
}

void configure_http_server_socket(socket_t socket) noexcept {
    httplib::default_socket_options(socket);
#if defined(__linux__)
    const int enabled = 1;
    // httplib sets only SO_REUSEPORT. Linux allows binding over a port's
    // TIME_WAIT connections only when the old and new sockets both set
    // SO_REUSEADDR, or both set SO_REUSEPORT under one uid. Without this, a
    // server using SO_REUSEADDR cannot take the port for about a minute after
    // NInfer stops, and NInfer cannot bind for as long after such a server stops.
    set_socket_option(socket, SOL_SOCKET, SO_REUSEADDR, enabled);
    set_socket_option(socket, SOL_SOCKET, SO_KEEPALIVE, enabled);
    set_socket_option(socket, IPPROTO_TCP, TCP_KEEPIDLE, kKeepAliveIdleSeconds);
    set_socket_option(socket, IPPROTO_TCP, TCP_KEEPINTVL, kKeepAliveIntervalSeconds);
    set_socket_option(socket, IPPROTO_TCP, TCP_KEEPCNT, kKeepAliveProbeCount);
    set_socket_option(socket, IPPROTO_TCP, TCP_USER_TIMEOUT, kTcpUserTimeoutMilliseconds);
#elif defined(_WIN32)
    const int enabled = 1;
    // Windows 10 1709 and later implement the three Linux keep-alive timers under the same names,
    // so the settings below are the Linux ones. TCP_USER_TIMEOUT has no Windows equivalent and is
    // left out: the 5 s send timeout httplib sets on every accepted socket bounds an
    // unacknowledged write, and the 5 s SSE heartbeat keeps traffic flowing, so a dead SSE peer
    // is still noticed, just on the next write rather than by a timer firing.
    set_socket_option(socket, SOL_SOCKET, SO_KEEPALIVE, enabled);
    set_socket_option(socket, IPPROTO_TCP, TCP_KEEPIDLE, kKeepAliveIdleSeconds);
    set_socket_option(socket, IPPROTO_TCP, TCP_KEEPINTVL, kKeepAliveIntervalSeconds);
    set_socket_option(socket, IPPROTO_TCP, TCP_KEEPCNT, kKeepAliveProbeCount);
#endif
}

void set_owned_json_content(httplib::Response& response, std::string body,
                            std::shared_ptr<RequestLifetime> lifetime) {
    response.set_content(std::move(body), "application/json");
    response.user_data.set("ninfer.request_lifetime", std::move(lifetime));
}

void HttpServer::stream_generation(httplib::Response& res,
                                   std::shared_ptr<HttpGenerationStream> stream,
                                   std::shared_ptr<RequestLifecycle> lifecycle,
                                   std::shared_ptr<SseStreamEncoder> encoder,
                                   std::function<void(const ApiError&)> write_error) {
    // Wrap a protocol encoder call so a render failure stays distinguishable from a transport one.
    const auto render = [](auto&& produce) -> std::vector<std::string> {
        try {
            return produce();
        } catch (const ClientDisconnected&) {
            throw;
        } catch (const ResponseRenderFailure&) {
            throw;
        } catch (const std::exception& exception) {
            throw ResponseRenderFailure(exception.what());
        }
    };

    try {
        prepare_sse_response(res);
        res.set_chunked_content_provider(
            "text/event-stream",
            [this, stream, lifecycle, encoder,
             render](std::size_t, httplib::DataSink& sink) -> bool {
                if (stream->started.exchange(true, std::memory_order_acq_rel)) {
                    sink.done();
                    return true;
                }
                SseTransport transport(sink, stream->cancelled);
                const auto send_error = [&](const ApiError& error) -> bool {
                    try {
                        transport.write(render([&] { return encoder->error_event(error); }));
                        sink.done();
                        return true;
                    } catch (const ClientDisconnected&) {
                        lifecycle->response_failure(
                            make_client_disconnected_failure(RequestFailurePhase::Transport));
                        return false;
                    } catch (const ResponseRenderFailure& exception) {
                        lifecycle->response_failure(make_internal_request_failure(
                            RequestFailurePhase::ResponseRender, exception.what()));
                        return false;
                    }
                };

                try {
                    transport.write(render([&] { return encoder->open(); }));
                } catch (const ClientDisconnected&) {
                    lifecycle->failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                    return false;
                } catch (const ResponseRenderFailure& exception) {
                    lifecycle->failure(make_internal_request_failure(
                        RequestFailurePhase::ResponseRender, exception.what()));
                    return send_error(encoder->internal_error(exception.what()));
                }

                GenerationOutcome outcome;
                try {
                    StreamSink output;
                    output.on_start = [&](const ninfer::GenerationStart& start) {
                        transport.write(render([&] { return encoder->on_start(start); }));
                    };
                    output.on_progress = [&](const ninfer::PromptProgress& progress) {
                        transport.write(render([&] { return encoder->on_progress(progress); }));
                    };
                    output.on_timing = [&](const ninfer::GenerationTimingObservation& timing) {
                        // on_timing emits no event of its own, but it still goes through `render`
                        // so a throw is classified as a render failure like every other callback.
                        (void)render([&] {
                        encoder->on_timing(timing);
                            return std::vector<std::string>{};
                        });
                    };
                    output.on_reasoning = [&](const std::string& text) {
                        transport.write(render([&] { return encoder->reasoning_delta(text); }));
                    };
                    output.on_content = [&](const std::string& text) {
                        transport.write(render([&] { return encoder->content_delta(text); }));
                    };
                    output.is_cancelled = [&] { return transport.poll(); };

                    outcome = service_->run(stream->prepared, &output);
                } catch (const ClientDisconnected&) {
                    lifecycle->failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                    return false;
                } catch (const ResponseRenderFailure& exception) {
                    lifecycle->failure(make_internal_request_failure(
                        RequestFailurePhase::ResponseRender, exception.what()));
                    return send_error(encoder->internal_error(exception.what()));
                } catch (const ApiException& exception) {
                    const ApiError error = encoder->normalize(exception.error());
                    lifecycle->failure(make_generation_request_failure(error));
                    return send_error(error);
                } catch (const std::exception& exception) {
                    lifecycle->failure(make_internal_request_failure(
                        RequestFailurePhase::Generation, exception.what()));
                    return send_error(encoder->internal_error(exception.what()));
                }

                lifecycle->done(outcome);
                SseStreamClose closing;
                try {
                    closing = encoder->close(outcome);
                } catch (const ClientDisconnected&) {
                    // The socket is already gone: report the disconnect rather than a render 500.
                    lifecycle->response_failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                    return false;
                } catch (const std::exception& exception) {
                    lifecycle->response_failure(make_internal_request_failure(
                        RequestFailurePhase::ResponseRender, exception.what()));
                    return send_error(encoder->internal_error(exception.what()));
                }
                if (closing.failure) {
                    lifecycle->response_failure(*closing.failure);
                    return send_error(closing.error);
                }
                try {
                    transport.write(closing.events);
                    sink.done();
                    return true;
                } catch (const ClientDisconnected&) {
                    lifecycle->response_failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                    return false;
                }
            },
            [stream, lifecycle](bool successful) {
                stream->cancelled.store(true, std::memory_order_release);
                if (!successful || !stream->started.load(std::memory_order_acquire)) {
                    lifecycle->failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                }
            });
    } catch (const std::exception& exception) {
        lifecycle->failure(
            make_internal_request_failure(RequestFailurePhase::ResponseRender, exception.what()));
        write_error(encoder->internal_error(exception.what()));
    }
}

} // namespace ninfer::serve
