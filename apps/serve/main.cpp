#include "product/logging/logging.h"
#include "product/logging/startup_log.h"
#include "serve/generation_service.h"
#include "serve/http_server.h"
#include "serve/serve_options.h"

#include <spdlog/logger.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <exception>
#include <print>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>

namespace {

std::atomic<ninfer::serve::HttpServer*> g_server{nullptr};
std::atomic<ninfer::product::LoggingRuntime*> g_logging{nullptr};

// An exception that escapes a request boundary ends the process through
// std::terminate, and the default handler's message is the only record of which
// exception it was. That message is worth writing through the server's own log:
// under a container this process is pid 1, the kernel discards the SIGABRT that
// abort() raises against itself, glibc falls through to its abort instruction,
// and all the kernel reports is a bare protection fault inside libc.
[[noreturn]] void log_terminate() {
    std::string detail = "terminate called with no active exception";
    if (std::current_exception() != nullptr) {
        try {
            std::rethrow_exception(std::current_exception());
        } catch (const std::exception& error) {
            detail = std::string("terminate called after throwing ") + typeid(error).name() + ": " +
                     error.what();
        } catch (...) { detail = "terminate called after throwing a non-std exception"; }
    }
    if (ninfer::product::LoggingRuntime* logging = g_logging.load(); logging != nullptr) {
        logging->logger()->critical("{}", detail);
        logging->flush();
    } else {
        std::println(stderr, "ninfer-serve: {}", detail);
    }
    std::abort();
}

void handle_signal(int) {
    ninfer::serve::HttpServer* server = g_server.load();
    if (server != nullptr) { server->stop(); }
}

} // namespace

int main(int argc, char** argv) {
    std::set_terminate(log_terminate);
    ninfer::serve::ServeOptions options;
    try {
        options = ninfer::serve::parse_serve_options(argc, argv);
    } catch (const std::invalid_argument& exception) {
        std::println(stderr, "ninfer-serve: {}", exception.what());
        std::print(stderr, "{}", ninfer::serve::serve_usage_text(argv[0]));
        return 1;
    } catch (const std::exception& exception) {
        std::println(stderr, "ninfer-serve: {}", exception.what());
        return 1;
    }
    if (options.help_requested) {
        std::print("{}", ninfer::serve::serve_usage_text(argv[0]));
        return 0;
    }

    ninfer::product::LoggingRuntime logging(
        {.logger_name  = "ninfer-serve",
         .level        = options.log_level,
         .presentation = ninfer::product::LogPresentation::Service});
    const std::shared_ptr<spdlog::logger> logger = logging.logger();
    g_logging.store(&logging);
    ninfer::product::StartupLogRenderer startup_log(logging);
    ninfer::serve::OperationalLog operational_log(logger);
    bool serving = false;

    try {
        ninfer::serve::HttpServer server(options, logger);
        if (!server.bind()) {
            operational_log.bind_failure(options.host, options.port);
            return 1;
        }

        ninfer::serve::GenerationService service(options, startup_log.observer());
        startup_log.engine_ready(service.load_summary());
        operational_log.engine_capacity(service);

        using Clock                            = std::chrono::steady_clock;
        const Clock::time_point warmup_started = Clock::now();
        operational_log.warmup_started();
        try {
            service.warmup();
        } catch (const std::exception& exception) {
            const double seconds =
                std::chrono::duration<double>(Clock::now() - warmup_started).count();
            operational_log.warmup_failure(seconds, exception.what());
            return 1;
        }
        operational_log.warmup_complete(
            std::chrono::duration<double>(Clock::now() - warmup_started).count());
        server.attach(service);

        g_server.store(&server);
        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);

        serving = true;
        operational_log.server_ready(options.host, options.port, server.public_model_id(),
                                     !options.api_key.empty());

        const bool ok = server.listen();
        g_server.store(nullptr);
        if (!ok) {
            operational_log.listen_failure(options.host, options.port);
            return 1;
        }
        operational_log.server_stopped();
        return 0;
    } catch (const std::exception& exception) {
        g_server.store(nullptr);
        operational_log.server_failure(serving, exception.what());
        return 1;
    }
}
