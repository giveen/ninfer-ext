#pragma once

#include <cuda_runtime.h>

#include <functional>

namespace ninfer {

// CUDA decode graphs. Neither wrapper is thread-safe: both hold mutable CUDA handles and their
// mutators capture, instantiate, or launch on the calling thread's current device and stream, so
// one instance belongs to one thread at a time. capture() additionally relies on a thread-local
// CUDA capture mode.

class DecodeGraphDefinition {
public:
    DecodeGraphDefinition() = default;
    ~DecodeGraphDefinition();

    DecodeGraphDefinition(const DecodeGraphDefinition&)            = delete;
    DecodeGraphDefinition& operator=(const DecodeGraphDefinition&) = delete;
    DecodeGraphDefinition(DecodeGraphDefinition&& other) noexcept;
    DecodeGraphDefinition& operator=(DecodeGraphDefinition&& other) noexcept;

    void capture(cudaStream_t stream, const std::function<void()>& body);
    [[nodiscard]] bool ready() const noexcept;
    void reset() noexcept;

private:
    friend class DecodeGraphExecutable;
    cudaGraph_t graph_ = nullptr;
};

class DecodeGraphExecutable {
public:
    DecodeGraphExecutable() = default;
    ~DecodeGraphExecutable();

    DecodeGraphExecutable(const DecodeGraphExecutable&)            = delete;
    DecodeGraphExecutable& operator=(const DecodeGraphExecutable&) = delete;
    DecodeGraphExecutable(DecodeGraphExecutable&& other) noexcept;
    DecodeGraphExecutable& operator=(DecodeGraphExecutable&& other) noexcept;

    void instantiate(const DecodeGraphDefinition& definition);
    // Updates the executable in place. Both the executable and the definition must be ready
    // (std::logic_error otherwise), and a rejected update — a topology change or any other
    // constraint of an in-place update — throws std::runtime_error. Callers that expect a topology
    // change should prefer try_update() and fall back to instantiate().
    void update(const DecodeGraphDefinition& definition);
    // Attempts the update; returns false, leaving the executable unchanged, when the definition
    // has a different topology or otherwise violates the constraints of an in-place update.
    [[nodiscard]] bool try_update(const DecodeGraphDefinition& definition);
    void upload(cudaStream_t stream);
    void launch(cudaStream_t stream);
    [[nodiscard]] bool ready() const noexcept;
    void reset() noexcept;

private:
    cudaGraphExec_t exec_ = nullptr;
};

} // namespace ninfer
