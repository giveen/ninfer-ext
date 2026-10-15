#include "product/logging/logging.h"
#include "product/logging/pretty_format.h"
#include "product/logging/startup_log.h"

#include "test_platform.h"

#include <spdlog/logger.h>

#if defined(_WIN32)
#    include <fcntl.h>
#    include <io.h>
#else
#    include <unistd.h>
#endif

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>

namespace {

class StderrCapture {
public:
    StderrCapture() {
#if defined(_WIN32)
        // The CRT's descriptor calls carry the underscore prefix, and _pipe takes a buffer size and
        // a text mode: binary mode keeps the captured bytes unmangled. STDERR_FILENO is not
        // portable, so the stream's own descriptor is asked for instead.
        if (::_pipe(pipe_, 4096, _O_BINARY) != 0) {
            throw std::runtime_error(ninfer::test::errno_message(errno));
        }
        saved_ = ::_dup(::_fileno(stderr));
        if (saved_ < 0 || ::_dup2(pipe_[1], ::_fileno(stderr)) < 0) {
            throw std::runtime_error(ninfer::test::errno_message(errno));
        }
        ::_close(pipe_[1]);
#else
        if (::pipe(pipe_) != 0) { throw std::runtime_error(ninfer::test::errno_message(errno)); }
        saved_ = ::dup(STDERR_FILENO);
        if (saved_ < 0 || ::dup2(pipe_[1], STDERR_FILENO) < 0) {
            throw std::runtime_error(ninfer::test::errno_message(errno));
        }
        ::close(pipe_[1]);
#endif
        pipe_[1] = -1;
    }

    ~StderrCapture() {
        if (saved_ >= 0) {
#if defined(_WIN32)
            (void)::_dup2(saved_, ::_fileno(stderr));
            ::_close(saved_);
#else
            (void)::dup2(saved_, STDERR_FILENO);
            ::close(saved_);
#endif
        }
#if defined(_WIN32)
        if (pipe_[0] >= 0) { ::_close(pipe_[0]); }
#else
        if (pipe_[0] >= 0) { ::close(pipe_[0]); }
#endif
    }

    std::string finish() {
        std::fflush(stderr);
#if defined(_WIN32)
        if (::_dup2(saved_, ::_fileno(stderr)) < 0) {
            throw std::runtime_error(ninfer::test::errno_message(errno));
        }
        ::_close(saved_);
#else
        if (::dup2(saved_, STDERR_FILENO) < 0) {
            throw std::runtime_error(ninfer::test::errno_message(errno));
        }
        ::close(saved_);
#endif
        saved_ = -1;

        std::string output;
        std::array<char, 4096> buffer{};
        for (;;) {
#if defined(_WIN32)
            // _read returns an int and has no ssize_t; nothing else about the loop differs.
            const int count = ::_read(pipe_[0], buffer.data(), static_cast<unsigned>(buffer.size()));
#else
            const ssize_t count = ::read(pipe_[0], buffer.data(), buffer.size());
#endif
            if (count == 0) { break; }
            if (count < 0) {
                if (errno == EINTR) { continue; }
                throw std::runtime_error(ninfer::test::errno_message(errno));
            }
            output.append(buffer.data(), static_cast<std::size_t>(count));
        }
#if defined(_WIN32)
        ::_close(pipe_[0]);
#else
        ::close(pipe_[0]);
#endif
        pipe_[0] = -1;
        return output;
    }

private:
    int pipe_[2]{-1, -1};
    int saved_ = -1;
};

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

std::size_t line_count(std::string_view value) {
    return static_cast<std::size_t>(std::ranges::count(value, '\n'));
}

int check_rejects_log_level(std::string_view spelling) {
    try {
        (void)ninfer::product::parse_log_level(spelling);
    } catch (const std::invalid_argument&) { return 0; }
    std::cerr << "parse_log_level accepted an invalid spelling\n";
    return 1;
}

} // namespace

int main() {
    int failures = 0;
    using ninfer::product::LogLevel;
    using ninfer::product::parse_log_level;
    failures += check(parse_log_level("trace") == LogLevel::Trace, "trace mismatch");
    failures += check(parse_log_level("debug") == LogLevel::Debug, "debug mismatch");
    failures += check(parse_log_level("info") == LogLevel::Info, "info mismatch");
    failures += check(parse_log_level("warning") == LogLevel::Warning, "warning mismatch");
    failures += check(parse_log_level("warn") == LogLevel::Warning, "warn alias mismatch");
    failures += check(parse_log_level("error") == LogLevel::Error, "error mismatch");
    failures += check(parse_log_level("critical") == LogLevel::Critical, "critical mismatch");
    failures += check(parse_log_level("fatal") == LogLevel::Critical, "fatal alias mismatch");
    failures += check(parse_log_level("off") == LogLevel::Off, "off mismatch");
    for (const std::string_view spelling : {"", "verbose", "INFO", "Warning", "off "}) {
        failures += check_rejects_log_level(spelling);
    }
    std::string service_output;
    {
        StderrCapture capture;
        {
            ninfer::product::LoggingRuntime logging(
                {.logger_name  = "ninfer-serve",
                 .color        = ninfer::product::LogColorMode::Auto,
                 .presentation = ninfer::product::LogPresentation::Service});
            logging.logger()->info("throughput | sample");
            logging.flush();
        }
        service_output = capture.finish();
    }
    failures += check(
        std::regex_match(
            service_output,
            std::regex(
                R"(^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}  INFO  throughput \| sample\n$)")),
        "service pretty prefix mismatch");
    failures += check(!service_output.contains("ninfer-serve"),
                      "service pretty output repeated the executable name");
    failures += check(!service_output.contains("\x1b["),
                      "redirected service output contains ANSI escapes");

    std::string startup_output;
    {
        StderrCapture capture;
        {
            ninfer::product::LoggingRuntime logging(
                {.logger_name  = "ninfer-serve",
                 .color        = ninfer::product::LogColorMode::Never,
                 .presentation = ninfer::product::LogPresentation::Service});
            ninfer::product::StartupLogRenderer startup(logging);
            ninfer::StartupObserver observer = startup.observer();
            observer.callback({.phase  = ninfer::StartupPhase::EngineStartup,
                               .status = ninfer::StartupStatus::Begin});
            observer.callback({.phase  = ninfer::StartupPhase::CudaInitialize,
                               .status = ninfer::StartupStatus::Begin});
            observer.callback({.phase      = ninfer::StartupPhase::CudaInitialize,
                               .status     = ninfer::StartupStatus::Complete,
                               .elapsed_ns = 1'000'000'000});
            observer.callback({.phase         = ninfer::StartupPhase::WeightsMaterialize,
                               .status        = ninfer::StartupStatus::Begin,
                               .progress_unit = ninfer::StartupProgressUnit::Bytes,
                               .total         = 16ULL << 30});
            observer.callback({.phase         = ninfer::StartupPhase::WeightsMaterialize,
                               .status        = ninfer::StartupStatus::Complete,
                               .progress_unit = ninfer::StartupProgressUnit::Bytes,
                               .current       = 16ULL << 30,
                               .total         = 16ULL << 30,
                               .elapsed_ns    = 2'000'000'000});
            observer.callback({.phase      = ninfer::StartupPhase::EngineStartup,
                               .status     = ninfer::StartupStatus::Complete,
                               .elapsed_ns = 3'000'000'000});
            startup.engine_ready({.model_name           = "qwen3.6-27b",
                                  .cuda_sync_mode       = "blocking",
                                  .weight_formats       = {"q4_g64_fp16", "q8_g32_fp16"},
                                  .host_to_device_bytes = 16ULL << 30});
            logging.flush();
        }
        startup_output = capture.finish();
    }
    failures += check(
        line_count(startup_output) == 4 &&
            startup_output.contains("starting engine") &&
            startup_output.contains("loading weights | 16.0 GiB") &&
            startup_output.find("weights ready | 16.0 GiB | 2.0s | 8.00 GiB/s") !=
                std::string::npos &&
            startup_output.find("engine ready | qwen3.6-27b | total 3.0s | weights 16.0 GiB") !=
                std::string::npos &&
            startup_output.contains("CUDA sync blocking") &&
            !startup_output.contains("CUDA initialized"),
        "normal startup pretty output is noisy or incomplete");

    std::string tool_output;
    {
        StderrCapture capture;
        {
            ninfer::product::LoggingRuntime logging(
                {.logger_name  = "ninfer",
                 .color        = ninfer::product::LogColorMode::Never,
                 .presentation = ninfer::product::LogPresentation::Tool});
            logging.logger()->info("engine ready");
            logging.logger()->error("failed");
            logging.flush();
        }
        tool_output = capture.finish();
    }
    failures +=
        check(tool_output == "engine ready\nerror: failed\n", "tool pretty prefix mismatch");

    using ninfer::product::format_pretty_duration;
    // A sub-unit component that rounds up to its own unit's threshold must carry into that unit.
    failures += check(format_pretty_duration(119.99) == "2m 0.0s" &&
                          format_pretty_duration(59.98) == "1m 0.0s" &&
                          format_pretty_duration(3599.99) == "1h 0m" &&
                          format_pretty_duration(0.9999) == "1.0s" &&
                          format_pretty_duration(0.0009999) == "1.00 ms",
                      "a rounded duration component did not carry into the next unit");
    // A finite but astronomically large duration must not narrow out of range.
    failures += check(format_pretty_duration(1.0e30) == "1000000000000000h 0m",
                      "a huge duration did not clamp before the integer conversion");
    failures += check(
        format_pretty_duration(0.0) == "0 us" && format_pretty_duration(59.94) == "59.9s" &&
            format_pretty_duration(3600.0) == "1h 0m" && format_pretty_duration(-1.0) == "n/a",
        "ordinary duration formatting changed");
    return failures == 0 ? 0 : 1;
}
