// GPU <-> host handshake cost inside a captured CUDA Graph, for the decode miss split.
//
// One graph holds `layers` steps. In each step a one-thread kernel stores the step number to a
// mapped Host word (the "doorbell"), a host thread spinning on that word answers by storing the
// same number to a second mapped word (the "flag"), and a one-block kernel spins on the flag before
// the step ends; a small busy kernel stands in for the step's own GPU work. The same graph without
// the doorbell/flag pair is the baseline, so the difference per step is the handshake's critical-
// path cost. `--work-us` sets the stand-in work, and `--host-work-us` how long the host takes to
// answer (the CPU share's compute), so overlap is visible.
//
//   ninfer_host_handshake_bench [--layers 48] [--work-us 20] [--host-work-us 0] [--iters 200]
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#define CHECK(x)                                                                                  \
    do {                                                                                          \
        const cudaError_t e_ = (x);                                                               \
        if (e_ != cudaSuccess) {                                                                  \
            std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_));                         \
            std::exit(1);                                                                         \
        }                                                                                         \
    } while (0)

namespace {

__global__ void ring(volatile std::uint32_t* doorbell, std::uint32_t value) {
    __threadfence_system();
    *doorbell = value;
}

__global__ void wait_flag(const volatile std::uint32_t* flag, std::uint32_t value) {
    if (threadIdx.x == 0) {
        while (*flag < value) { __nanosleep(32); }
    }
    __syncthreads();
}

__global__ void busy(long long cycles, int* sink) {
    const long long t0 = clock64();
    while (clock64() - t0 < cycles) {}
    if (threadIdx.x == 0 && blockIdx.x == 0) { atomicAdd(sink, 1); }
}

struct Args {
    int layers = 48;
    double work_us = 20.0;
    double host_work_us = 0.0;
    int iters = 200;
};

double run(const Args& args, bool handshake) {
    cudaStream_t stream;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::uint32_t *h_bell = nullptr, *h_flag = nullptr, *d_bell = nullptr, *d_flag = nullptr;
    CHECK(cudaHostAlloc(&h_bell, 64, cudaHostAllocMapped));
    CHECK(cudaHostAlloc(&h_flag, 64, cudaHostAllocMapped));
    CHECK(cudaHostGetDevicePointer(&d_bell, h_bell, 0));
    CHECK(cudaHostGetDevicePointer(&d_flag, h_flag, 0));
    int* sink = nullptr;
    CHECK(cudaMalloc(&sink, sizeof(int)));
    int clock_khz = 0;
    CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
    const long long cycles = static_cast<long long>(args.work_us * clock_khz / 1000.0);

    // The step numbers grow across replays (`base` + step), so a replay never sees a stale answer.
    // A graph's kernel arguments are fixed, so each replay uses its own graph slot: two graphs
    // alternate, each built for its own range of values and rebuilt once that range is used up.
    auto build = [&](std::uint32_t base) {
        cudaGraph_t graph;
        CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for (int l = 0; l < args.layers; ++l) {
            const std::uint32_t value = base + static_cast<std::uint32_t>(l) + 1;
            if (handshake) { ring<<<1, 1, 0, stream>>>(d_bell, value); }
            busy<<<1, 32, 0, stream>>>(cycles, sink);
            if (handshake) { wait_flag<<<1, 32, 0, stream>>>(d_flag, value); }
        }
        CHECK(cudaStreamEndCapture(stream, &graph));
        cudaGraphExec_t exec;
        CHECK(cudaGraphInstantiate(&exec, graph, 0));
        CHECK(cudaGraphDestroy(graph));
        return exec;
    };

    std::atomic<bool> stop{false};
    std::thread host;
    if (handshake) {
        host = std::thread([&] {
            std::uint32_t seen = 0;
            while (!stop.load(std::memory_order_relaxed)) {
                const std::uint32_t v = *reinterpret_cast<volatile std::uint32_t*>(h_bell);
                if (v != seen) {
                    seen = v;
                    if (args.host_work_us > 0) {
                        const auto t0 = std::chrono::steady_clock::now();
                        while (std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() <
                               args.host_work_us) {}
                    }
                    std::atomic_thread_fence(std::memory_order_release);
                    *reinterpret_cast<volatile std::uint32_t*>(h_flag) = v;
                } else {
                    __builtin_ia32_pause();
                }
            }
        });
    }

    std::vector<double> us;
    std::uint32_t base = 0;
    for (int it = 0; it < args.iters + 10; ++it) {
        cudaGraphExec_t exec = build(base);
        const auto t0 = std::chrono::steady_clock::now();
        CHECK(cudaGraphLaunch(exec, stream));
        CHECK(cudaStreamSynchronize(stream));
        const auto t1 = std::chrono::steady_clock::now();
        CHECK(cudaGraphExecDestroy(exec));
        base += static_cast<std::uint32_t>(args.layers);
        if (it >= 10) { us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count()); }
    }
    stop.store(true);
    if (host.joinable()) { host.join(); }
    CHECK(cudaFreeHost(h_bell));
    CHECK(cudaFreeHost(h_flag));
    CHECK(cudaFree(sink));
    CHECK(cudaStreamDestroy(stream));
    std::sort(us.begin(), us.end());
    return us[us.size() / 2];
}

} // namespace

int main(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--layers" && i + 1 < argc) { args.layers = std::atoi(argv[++i]); }
        else if (a == "--work-us" && i + 1 < argc) { args.work_us = std::atof(argv[++i]); }
        else if (a == "--host-work-us" && i + 1 < argc) { args.host_work_us = std::atof(argv[++i]); }
        else if (a == "--iters" && i + 1 < argc) { args.iters = std::atoi(argv[++i]); }
        else {
            std::fprintf(stderr, "usage: %s [--layers N] [--work-us F] [--host-work-us F] [--iters N]\n", argv[0]);
            return 2;
        }
    }
    const double base = run(args, false), shake = run(args, true);
    std::printf("layers %d, work %.1f us, host work %.1f us: graph %.1f us plain, %.1f us with handshake; "
                "%.2f us per step\n",
                args.layers, args.work_us, args.host_work_us, base, shake, (shake - base) / args.layers);
    return 0;
}
