// Host-pointer-for-registered-memory diagnostic for the Windows (WDDM) port.
//
// The one assumption the port cannot settle on Linux is whether a kernel may dereference a pinned
// host buffer at the address the CPU holds for it. `cudaDevAttrCanUseHostPointerForRegisteredMem`,
// the attribute src/core/arena.cu uses to decide whether PinnedHostPages::Huge can hand out a
// pointer that device code reads directly, is advisory: it reports what the driver can do in
// general, not what this machine's driver actually does. Two shipped features depend on the real
// behavior rather than on the attribute. The Gemma 4 token-embedding gather reads a pinned host
// table from a kernel through UVA, and the Qwen4 expert pager copies with direction-inferred
// cudaMemcpyDefault. So this probe does not trust the attribute: it registers genuine 2 MiB-aligned
// anonymous memory and then reads it from a kernel and through a direction-inferred copy.
//
// The output is one fact per line, with no colour and no timing, so a GitHub issue can paste it
// whole and two machines can be diffed line by line. Exit code 0 means the registered-memory path
// worked end to end, 1 means it did not, 2 means the probe itself failed.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <new>
#include <string>

namespace ninfer {
namespace {

// The huge-page pinned path in src/core/arena.cu reserves whole 2 MiB pages; the probe registers a
// buffer with the same shape, so a pass here means that path is usable.
constexpr std::size_t kHostRegisterBytes = std::size_t{4} << 20;
constexpr std::size_t kHostRegisterAlign = std::size_t{2} << 20;

// A small grid is enough for a 1M-element sum and keeps the reduction cheap.
constexpr int kBlocks          = 8;
constexpr int kThreadsPerBlock = 256;

// The engine is compiled only for sm_120a. The runtime exposes no way to ask for the arch-specific
// ("a") feature set, so the closest observable check is the numeric compute capability of the
// device the driver reports.
constexpr int kRequiredMajor = 12;
constexpr int kRequiredMinor = 0;
constexpr int kDevice        = 0;

std::string describe_cuda(const char* operation, cudaError_t err) {
    return std::string(operation) + " failed: " + cudaGetErrorName(err) + ": " +
           cudaGetErrorString(err);
}

// A cheap integer mix, so a read that lands on a stale, shifted or truncated page changes the
// checksum instead of cancelling itself out.
std::uint32_t pattern_value(std::size_t index) {
    std::uint32_t value = static_cast<std::uint32_t>(index) * 2654435761U + 1013904223U;
    value ^= value >> 15;
    value *= 2246822519U;
    value ^= value >> 13;
    return value;
}

void fill_pattern(void* buffer, std::size_t bytes) {
    auto* words             = static_cast<std::uint32_t*>(buffer);
    const std::size_t count = bytes / sizeof(std::uint32_t);
    for (std::size_t index = 0; index < count; ++index) { words[index] = pattern_value(index); }
}

std::uint64_t pattern_sum(std::size_t bytes) {
    const std::size_t count = bytes / sizeof(std::uint32_t);
    std::uint64_t sum       = 0;
    for (std::size_t index = 0; index < count; ++index) { sum += pattern_value(index); }
    return sum;
}

// Reads the buffer at exactly the pointer the host holds. If the driver maps the registration at a
// different device address, or does not expose it to device code at all, this read either faults or
// returns different bytes, and the checksum disagrees.
__global__ void read_through_host_pointer(const std::uint32_t* values, std::size_t count,
                                          unsigned long long* total) {
    unsigned long long partial = 0;
    const std::size_t stride   = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += stride) {
        partial += values[index];
    }
    // Integer addition is associative and exact, so the atomic reduction is order-independent and
    // the summary is reproducible.
    atomicAdd(total, partial);
}

// Anonymous 2 MiB-aligned memory, as the arena's huge-page path allocates it. The block is aligned
// by hand instead of with an aligned-allocation entry point, because MSVC's spelling differs from
// glibc's and this file has to build under both.
struct AlignedHostAllocation {
    void* reservation = nullptr;
    void* data        = nullptr;

    AlignedHostAllocation(std::size_t bytes, std::size_t align) {
        reservation = std::malloc(bytes + align);
        if (reservation == nullptr) { throw std::bad_alloc(); }
        const auto address = reinterpret_cast<std::uintptr_t>(reservation);
        const std::uintptr_t aligned =
            (address + align - 1) & ~static_cast<std::uintptr_t>(align - 1);
        data = reinterpret_cast<void*>(aligned);
    }

    ~AlignedHostAllocation() { std::free(reservation); }

    AlignedHostAllocation(const AlignedHostAllocation&)            = delete;
    AlignedHostAllocation& operator=(const AlignedHostAllocation&) = delete;
};

// One kernel read of a host buffer: whether it ran at all, and the checksum it produced.
struct ReadTest {
    bool ran               = false;
    bool passed            = false;
    std::uint64_t observed = 0;
    std::uint64_t expected = 0;
    std::string detail;
};

// Direction-inferred copy of a registered host buffer into device memory and back. The copy is the
// operation the Qwen4 expert pager relies on, so a direction the driver infers wrongly shows up as
// mismatching bytes rather than as an error.
struct CopyTest {
    bool ran                  = false;
    bool passed               = false;
    std::size_t compared      = 0;
    std::size_t mismatches    = 0;
    std::size_t first_differs = 0;
    std::string detail;
};

// `device_total` is an 8-byte device slot the kernel atomically accumulates into, `host_total` its
// host mirror. Shared by the registered buffer and the cudaMallocHost fallback.
ReadTest read_through_host_buffer(void* host_buffer, void* device_total, void* host_total,
                                  cudaStream_t stream) {
    ReadTest result;
    result.expected = pattern_sum(kHostRegisterBytes);

    cudaError_t err = cudaMemsetAsync(device_total, 0, sizeof(unsigned long long), stream);
    if (err == cudaSuccess) {
        read_through_host_pointer<<<kBlocks, kThreadsPerBlock, 0, stream>>>(
            static_cast<const std::uint32_t*>(host_buffer),
            kHostRegisterBytes / sizeof(std::uint32_t),
            static_cast<unsigned long long*>(device_total));
        err = cudaGetLastError();
    }
    if (err == cudaSuccess) {
        // A device-side fault is reported by the synchronization, not by the launch.
        err = cudaStreamSynchronize(stream);
    }
    if (err != cudaSuccess) {
        result.detail = describe_cuda("checksum kernel", err);
        return result;
    }

    err = cudaMemcpyAsync(host_total, device_total, sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost, stream);
    if (err == cudaSuccess) { err = cudaStreamSynchronize(stream); }
    if (err != cudaSuccess) {
        result.detail = describe_cuda("cudaMemcpyAsync of the checksum", err);
        return result;
    }

    result.ran      = true;
    result.observed = *static_cast<unsigned long long*>(host_total);
    result.passed   = result.observed == result.expected;
    return result;
}

void print_read_test(const char* label, const ReadTest& result) {
    if (!result.ran) {
        std::printf("%s: FAIL (%s)\n", label, result.detail.c_str());
        return;
    }
    std::printf("%s: %s (sum 0x%016llx, expected 0x%016llx)\n", label,
                result.passed ? "PASS" : "FAIL", static_cast<unsigned long long>(result.observed),
                static_cast<unsigned long long>(result.expected));
}

void print_build_facts() {
    // The compiler name matters here because the Windows toolchain is MSVC, whose standard library
    // and floating-point behavior differ from the GCC the tree is developed against.
#if defined(_MSC_VER)
    std::printf("compiler: MSVC _MSC_VER=%d\n", _MSC_VER);
#elif defined(__clang__)
    std::printf("compiler: clang %d.%d.%d\n", __clang_major__, __clang_minor__,
                __clang_patchlevel__);
#elif defined(__GNUC__)
    std::printf("compiler: GCC %d.%d.%d\n", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#else
    std::printf("compiler: unknown\n");
#endif
    // The reference oracle's precision changes with this width, so it belongs in the baseline.
    std::printf("long double: %zu bytes\n", sizeof(long double));
    std::printf("pointer size: %zu bytes\n", sizeof(void*));

    int runtime_version             = 0;
    const cudaError_t runtime_error = cudaRuntimeGetVersion(&runtime_version);
    if (runtime_error == cudaSuccess) {
        std::printf("cuda runtime: %d (%d.%d)\n", runtime_version, runtime_version / 1000,
                    (runtime_version % 1000) / 10);
    } else {
        std::printf("cuda runtime: unknown (%s)\n",
                    describe_cuda("cudaRuntimeGetVersion", runtime_error).c_str());
    }

    int driver_version             = 0;
    const cudaError_t driver_error = cudaDriverGetVersion(&driver_version);
    if (driver_error == cudaSuccess) {
        std::printf("cuda driver: %d (%d.%d)\n", driver_version, driver_version / 1000,
                    (driver_version % 1000) / 10);
    } else {
        std::printf("cuda driver: unknown (%s)\n",
                    describe_cuda("cudaDriverGetVersion", driver_error).c_str());
    }
}

void print_device_facts() {
    cudaDeviceProp prop{};
    const cudaError_t prop_error = cudaGetDeviceProperties(&prop, kDevice);
    if (prop_error != cudaSuccess) {
        std::printf("device: unavailable (%s)\n",
                    describe_cuda("cudaGetDeviceProperties", prop_error).c_str());
    } else {
        std::printf("device: %s\n", prop.name);
        std::printf("compute capability: %d.%d\n", prop.major, prop.minor);
        if (prop.major != kRequiredMajor || prop.minor != kRequiredMinor) {
            // Such a device cannot run this engine at all, so the results below are informational.
            std::printf(
                "note: this engine supports only sm_120a (compute capability %d.%d); the device is "
                "not that target\n",
                kRequiredMajor, kRequiredMinor);
        }
        std::printf("uuid: %02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x\n",
                    static_cast<unsigned char>(prop.uuid.bytes[0]),
                    static_cast<unsigned char>(prop.uuid.bytes[1]),
                    static_cast<unsigned char>(prop.uuid.bytes[2]),
                    static_cast<unsigned char>(prop.uuid.bytes[3]),
                    static_cast<unsigned char>(prop.uuid.bytes[4]),
                    static_cast<unsigned char>(prop.uuid.bytes[5]),
                    static_cast<unsigned char>(prop.uuid.bytes[6]),
                    static_cast<unsigned char>(prop.uuid.bytes[7]),
                    static_cast<unsigned char>(prop.uuid.bytes[8]),
                    static_cast<unsigned char>(prop.uuid.bytes[9]),
                    static_cast<unsigned char>(prop.uuid.bytes[10]),
                    static_cast<unsigned char>(prop.uuid.bytes[11]),
                    static_cast<unsigned char>(prop.uuid.bytes[12]),
                    static_cast<unsigned char>(prop.uuid.bytes[13]),
                    static_cast<unsigned char>(prop.uuid.bytes[14]),
                    static_cast<unsigned char>(prop.uuid.bytes[15]));
        // The LUID is how Windows identifies the display adapter, which is what a WDDM report needs
        // to line this device up with the machine's GPU; the driver leaves it undefined off
        // Windows.
        std::printf(
            "luid: %02x%02x%02x%02x%02x%02x%02x%02x (device node mask %u)\n",
            static_cast<unsigned char>(prop.luid[0]), static_cast<unsigned char>(prop.luid[1]),
            static_cast<unsigned char>(prop.luid[2]), static_cast<unsigned char>(prop.luid[3]),
            static_cast<unsigned char>(prop.luid[4]), static_cast<unsigned char>(prop.luid[5]),
            static_cast<unsigned char>(prop.luid[6]), static_cast<unsigned char>(prop.luid[7]),
            static_cast<unsigned int>(prop.luidDeviceNodeMask));
    }

    std::size_t free_bytes      = 0;
    std::size_t total_bytes     = 0;
    const cudaError_t mem_error = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (mem_error != cudaSuccess) {
        std::printf("vram: unavailable (%s)\n", describe_cuda("cudaMemGetInfo", mem_error).c_str());
        return;
    }
    std::printf("vram total: %llu bytes (%llu MiB)\n", static_cast<unsigned long long>(total_bytes),
                static_cast<unsigned long long>(total_bytes >> 20));
    std::printf("vram free: %llu bytes (%llu MiB)\n", static_cast<unsigned long long>(free_bytes),
                static_cast<unsigned long long>(free_bytes >> 20));
}

int run_probe() {
    std::printf("ninfer-probe: host-pointer-for-registered-memory diagnostic\n");
    print_build_facts();

    int device_count              = 0;
    const cudaError_t count_error = cudaGetDeviceCount(&device_count);
    if (count_error != cudaSuccess) {
        std::printf("device: none (%s)\n",
                    describe_cuda("cudaGetDeviceCount", count_error).c_str());
        std::printf("host-pointer-for-registered-memory: UNSUPPORTED\n");
        return 1;
    }
    if (device_count == 0) {
        std::printf("device: none (cudaGetDeviceCount reported 0 devices)\n");
        std::printf("host-pointer-for-registered-memory: UNSUPPORTED\n");
        return 1;
    }

    print_device_facts();

    // -1 keeps "the query failed" distinct from "the driver reported 0", because the notes below
    // read very differently in those two cases.
    int host_pointer_usable           = -1;
    const cudaError_t attribute_error = cudaDeviceGetAttribute(
        &host_pointer_usable, cudaDevAttrCanUseHostPointerForRegisteredMem, kDevice);
    if (attribute_error != cudaSuccess) {
        host_pointer_usable = -1;
        std::printf("cudaDevAttrCanUseHostPointerForRegisteredMem: query failed (%s)\n",
                    describe_cuda("cudaDeviceGetAttribute", attribute_error).c_str());
    } else {
        std::printf("cudaDevAttrCanUseHostPointerForRegisteredMem: %d\n", host_pointer_usable);
    }

    // The functional test needs a working context; without one there is nothing left to measure.
    const cudaError_t set_error = cudaSetDevice(kDevice);
    if (set_error != cudaSuccess) {
        std::printf("cudaSetDevice: FAIL (%s)\n",
                    describe_cuda("cudaSetDevice", set_error).c_str());
        std::printf("host-pointer-for-registered-memory: UNSUPPORTED\n");
        return 1;
    }

    cudaStream_t stream = nullptr;
    cudaError_t err     = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        std::printf("cudaStreamCreateWithFlags: FAIL (%s)\n",
                    describe_cuda("cudaStreamCreateWithFlags", err).c_str());
        std::printf("host-pointer-for-registered-memory: UNSUPPORTED\n");
        return 1;
    }

    void* device_copy  = nullptr;
    void* device_total = nullptr;
    err                = cudaMalloc(&device_copy, kHostRegisterBytes);
    if (err == cudaSuccess) { err = cudaMalloc(&device_total, sizeof(unsigned long long)); }
    void* host_total = std::malloc(sizeof(unsigned long long));
    void* host_stage = std::malloc(kHostRegisterBytes);
    if (err != cudaSuccess || host_total == nullptr || host_stage == nullptr) {
        std::printf("test buffers: FAIL (%s)\n", err != cudaSuccess
                                                     ? describe_cuda("cudaMalloc", err).c_str()
                                                     : "host staging allocation failed");
        if (device_copy != nullptr) { cudaFree(device_copy); }
        if (device_total != nullptr) { cudaFree(device_total); }
        std::free(host_total);
        std::free(host_stage);
        cudaStreamDestroy(stream);
        std::printf("host-pointer-for-registered-memory: UNSUPPORTED\n");
        return 1;
    }

    AlignedHostAllocation host(kHostRegisterBytes, kHostRegisterAlign);
    fill_pattern(host.data, kHostRegisterBytes);

    ReadTest registered_read;
    CopyTest registered_copy;
    bool registered = false;
    err             = cudaHostRegister(host.data, kHostRegisterBytes, cudaHostRegisterDefault);
    if (err != cudaSuccess) {
        std::printf("cudaHostRegister: FAIL (%s)\n",
                    describe_cuda("cudaHostRegister", err).c_str());
    } else {
        registered = true;
        std::printf(
            "registered host buffer: %llu bytes at 2 MiB alignment, cudaHostRegisterDefault\n",
            static_cast<unsigned long long>(kHostRegisterBytes));

        // The copy runs first even though it is reported second: a kernel that faults on the host
        // pointer leaves the context unusable, and a copy measured after that would report the
        // kernel's error instead of its own result.
        err =
            cudaMemcpyAsync(device_copy, host.data, kHostRegisterBytes, cudaMemcpyDefault, stream);
        if (err == cudaSuccess) { err = cudaStreamSynchronize(stream); }
        if (err == cudaSuccess) {
            // The way back is not in question, so it names its direction explicitly.
            err = cudaMemcpyAsync(host_stage, device_copy, kHostRegisterBytes,
                                  cudaMemcpyDeviceToHost, stream);
        }
        if (err == cudaSuccess) { err = cudaStreamSynchronize(stream); }
        if (err != cudaSuccess) {
            registered_copy.detail = describe_cuda("cudaMemcpyDefault copy", err);
        } else {
            const auto* expected          = static_cast<const std::uint8_t*>(host.data);
            const auto* actual            = static_cast<const std::uint8_t*>(host_stage);
            registered_copy.ran           = true;
            registered_copy.first_differs = kHostRegisterBytes;
            for (std::size_t index = 0; index < kHostRegisterBytes; ++index) {
                if (expected[index] == actual[index]) { continue; }
                if (registered_copy.first_differs == kHostRegisterBytes) {
                    registered_copy.first_differs = index;
                }
                ++registered_copy.mismatches;
            }
            registered_copy.compared = kHostRegisterBytes;
            registered_copy.passed   = registered_copy.mismatches == 0;
        }

        registered_read = read_through_host_buffer(host.data, device_total, host_total, stream);
    }

    print_read_test("registered kernel read", registered_read);
    if (registered_copy.ran) {
        std::printf("registered cudaMemcpyDefault copy: %s (%llu bytes compared, %llu mismatching, "
                    "first difference at %llu)\n",
                    registered_copy.passed ? "PASS" : "FAIL",
                    static_cast<unsigned long long>(registered_copy.compared),
                    static_cast<unsigned long long>(registered_copy.mismatches),
                    static_cast<unsigned long long>(registered_copy.first_differs));
        if (!registered_copy.passed) {
            // The bytes themselves, because a direction inferred wrongly usually shows up as a
            // shifted region and the offset alone does not say which side the shift came from.
            const auto* expected = static_cast<const std::uint8_t*>(host.data);
            const auto* actual   = static_cast<const std::uint8_t*>(host_stage);
            const std::size_t at = registered_copy.first_differs;
            std::printf(
                "first difference: byte %llu expected 0x%02x, observed 0x%02x\n",
                static_cast<unsigned long long>(at), static_cast<unsigned int>(expected[at]),
                static_cast<unsigned int>(actual[at]));
        }
    } else {
        std::printf("registered cudaMemcpyDefault copy: FAIL (%s)\n",
                    registered_copy.detail.c_str());
    }

    // A device-side fault leaves the context unusable, and a later call only repeats its error.
    // Probing the context keeps the fallback's line meaningful: it reports the fallback itself
    // rather than the registered test's failure.
    std::size_t probe_free    = 0;
    std::size_t probe_total   = 0;
    const bool context_usable = cudaMemGetInfo(&probe_free, &probe_total) == cudaSuccess;

    // The fallback the arena uses when the attribute is unavailable: cudaMallocHost memory, which
    // the driver itself pins. A kernel that cannot read this either points at the driver rather
    // than at the registration.
    void* pinned = nullptr;
    if (!context_usable) {
        std::printf("cudaMallocHost kernel read: not measured (the device context is invalid after "
                    "the registered test)\n");
    } else {
        err = cudaMallocHost(&pinned, kHostRegisterBytes);
        if (err != cudaSuccess) {
            std::printf("cudaMallocHost: FAIL (%s)\n", describe_cuda("cudaMallocHost", err).c_str());
        } else {
            std::printf("cudaMallocHost buffer: %llu bytes\n",
                        static_cast<unsigned long long>(kHostRegisterBytes));
            fill_pattern(pinned, kHostRegisterBytes);
            print_read_test("cudaMallocHost kernel read",
                            read_through_host_buffer(pinned, device_total, host_total, stream));
        }
    }

    // Registration is released explicitly so its outcome is reported separately from a successful
    // unregistration that the process teardown would otherwise hide.
    if (registered) {
        const cudaError_t unregister_error = cudaHostUnregister(host.data);
        if (unregister_error == cudaSuccess) {
            std::printf("cudaHostUnregister: PASS\n");
        } else {
            std::printf("cudaHostUnregister: FAIL (%s)\n",
                        describe_cuda("cudaHostUnregister", unregister_error).c_str());
        }
    } else {
        std::printf("cudaHostUnregister: not attempted (registration failed)\n");
    }

    // Cleanup failures are reported only when they happen, so a passing run stays deterministic.
    const cudaError_t free_copy_error =
        device_copy != nullptr ? cudaFree(device_copy) : cudaSuccess;
    if (free_copy_error != cudaSuccess) {
        std::printf("cudaFree(device copy): FAIL (%s)\n",
                    describe_cuda("cudaFree", free_copy_error).c_str());
    }
    const cudaError_t free_total_error =
        device_total != nullptr ? cudaFree(device_total) : cudaSuccess;
    if (free_total_error != cudaSuccess) {
        std::printf("cudaFree(device checksum slot): FAIL (%s)\n",
                    describe_cuda("cudaFree", free_total_error).c_str());
    }
    if (pinned != nullptr) {
        const cudaError_t free_pinned_error = cudaFreeHost(pinned);
        if (free_pinned_error != cudaSuccess) {
            std::printf("cudaFreeHost: FAIL (%s)\n",
                        describe_cuda("cudaFreeHost", free_pinned_error).c_str());
        }
    }
    const cudaError_t destroy_error = cudaStreamDestroy(stream);
    if (destroy_error != cudaSuccess) {
        std::printf("cudaStreamDestroy: FAIL (%s)\n",
                    describe_cuda("cudaStreamDestroy", destroy_error).c_str());
    }
    std::free(host_stage);
    std::free(host_total);

    // The verdict is the functional result, because that is what the two shipped features do. A
    // disagreement with the attribute is worth a line: the engine's gate reads the attribute, so a
    // mismatch means the gate is either too strict or too trusting on this machine.
    const bool supported = registered_read.passed && registered_copy.passed;
    if (host_pointer_usable > 0 && !supported) {
        std::printf("note: the attribute reports 1 but the functional test failed, so device-side "
                    "use of host pointers cannot be trusted on this machine\n");
    } else if (host_pointer_usable == 0 && supported) {
        std::printf("note: the attribute reports 0 but the functional test passed, so the gate in "
                    "src/core/arena.cu disables the huge-page pinned path unnecessarily\n");
    }
    std::printf("host-pointer-for-registered-memory: %s\n",
                supported ? "SUPPORTED" : "UNSUPPORTED");
    return supported ? 0 : 1;
}

} // namespace
} // namespace ninfer

int main() {
    // Nothing may escape: a Windows tester pastes this output into an issue, and an uncaught
    // exception would replace the verdict with a bare abort.
    try {
        return ninfer::run_probe();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "fatal: %s\n", error.what());
        return 2;
    } catch (...) {
        std::fprintf(stderr, "fatal: unknown exception\n");
        return 2;
    }
}
