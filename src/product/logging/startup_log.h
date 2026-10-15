#pragma once

#include "ninfer/types.h"

#include <memory>

namespace ninfer::product {

class LoggingRuntime;

// Product-side renderer for the neutral Engine startup observer. Returned observers retain the
// rendering state, so the callback remains self-contained if copied with Engine options. That
// state is not synchronized: one renderer renders one engine startup at a time, and a renderer
// shared by engines that start concurrently must be serialized by the caller.
class StartupLogRenderer {
public:
    explicit StartupLogRenderer(LoggingRuntime& logging);
    ~StartupLogRenderer();

    StartupLogRenderer(const StartupLogRenderer&)            = delete;
    StartupLogRenderer& operator=(const StartupLogRenderer&) = delete;
    StartupLogRenderer(StartupLogRenderer&&)                 = delete;
    StartupLogRenderer& operator=(StartupLogRenderer&&)      = delete;

    [[nodiscard]] StartupObserver observer();
    void engine_ready(const LoadSummary& load);

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace ninfer::product
