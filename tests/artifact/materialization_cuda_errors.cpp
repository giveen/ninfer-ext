#include "artifact/binder.h"
#include "artifact/fixture.h"

#include <cuda_runtime.h>

#if defined(_WIN32)
// windows.h defines min and max as macros; the headers above are already parsed, and this keeps a
// later include or call site from silently losing std::min/std::max.
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace {

enum class Failure { None, EventCreation, EventRecord };

struct Trace {
    Failure failure                = Failure::None;
    bool injected                  = false;
    bool upload_pending            = false;
    bool released_during_upload    = false;
    std::size_t uploads            = 0;
    std::size_t host_allocations   = 0;
    std::size_t host_releases      = 0;
    std::size_t device_allocations = 0;
    std::size_t device_releases    = 0;
};

Trace trace;

bool active() { return trace.failure != Failure::None; }

} // namespace

// The wrappers affect only this test executable. The production path has no fault hooks.
//
// POSIX redirects the runtime calls at link time: --wrap makes the linker bind every reference to
// cudaMalloc to __wrap_cudaMalloc and define __real_cudaMalloc for the original. MSVC has no
// equivalent, and /alternatename cannot stand in: it maps only a symbol that nothing defines, while
// the runtime arrives through an import library that does define these. Windows therefore
// redirects them at run time instead, through the executable's import address table, which every
// call to the runtime reaches -- whether the compiler emitted a direct call to the import thunk or
// an indirect call through the import slot -- so replacing the slot's value catches all of them and
// the value it held is the function the wrapper calls through.
extern "C" {
#if defined(_WIN32)
cudaError_t(CUDARTAPI* __real_cudaMalloc)(void**, std::size_t)                      = nullptr;
cudaError_t(CUDARTAPI* __real_cudaMallocHost)(void**, std::size_t)                  = nullptr;
cudaError_t(CUDARTAPI* __real_cudaFree)(void*)                                      = nullptr;
cudaError_t(CUDARTAPI* __real_cudaFreeHost)(void*)                                  = nullptr;
cudaError_t(CUDARTAPI* __real_cudaEventCreateWithFlags)(cudaEvent_t*, unsigned int) = nullptr;
cudaError_t(CUDARTAPI* __real_cudaEventRecord)(cudaEvent_t, cudaStream_t)           = nullptr;
cudaError_t(CUDARTAPI* __real_cudaMemcpyAsync)(void*, const void*, std::size_t, cudaMemcpyKind,
                                               cudaStream_t)                        = nullptr;
cudaError_t(CUDARTAPI* __real_cudaStreamSynchronize)(cudaStream_t)                  = nullptr;
#else
cudaError_t CUDARTAPI __real_cudaMalloc(void**, std::size_t);
cudaError_t CUDARTAPI __real_cudaMallocHost(void**, std::size_t);
cudaError_t CUDARTAPI __real_cudaFree(void*);
cudaError_t CUDARTAPI __real_cudaFreeHost(void*);
cudaError_t CUDARTAPI __real_cudaEventCreateWithFlags(cudaEvent_t*, unsigned int);
cudaError_t CUDARTAPI __real_cudaEventRecord(cudaEvent_t, cudaStream_t);
cudaError_t CUDARTAPI __real_cudaMemcpyAsync(void*, const void*, std::size_t, cudaMemcpyKind,
                                             cudaStream_t);
cudaError_t CUDARTAPI __real_cudaStreamSynchronize(cudaStream_t);
#endif

cudaError_t CUDARTAPI __wrap_cudaMalloc(void** pointer, std::size_t bytes) {
    const auto status = __real_cudaMalloc(pointer, bytes);
    if (active() && status == cudaSuccess) { ++trace.device_allocations; }
    return status;
}

cudaError_t CUDARTAPI __wrap_cudaMallocHost(void** pointer, std::size_t bytes) {
    const auto status = __real_cudaMallocHost(pointer, bytes);
    if (active() && status == cudaSuccess) { ++trace.host_allocations; }
    return status;
}

cudaError_t CUDARTAPI __wrap_cudaFree(void* pointer) {
    if (active()) { trace.released_during_upload |= trace.upload_pending; }
    const auto status = __real_cudaFree(pointer);
    if (active() && status == cudaSuccess) { ++trace.device_releases; }
    return status;
}

cudaError_t CUDARTAPI __wrap_cudaFreeHost(void* pointer) {
    if (active()) { trace.released_during_upload |= trace.upload_pending; }
    const auto status = __real_cudaFreeHost(pointer);
    if (active() && status == cudaSuccess) { ++trace.host_releases; }
    return status;
}

cudaError_t CUDARTAPI __wrap_cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) {
    if (trace.failure == Failure::EventCreation && !trace.injected) {
        trace.injected = true;
        *event         = nullptr;
        return cudaErrorMemoryAllocation;
    }
    return __real_cudaEventCreateWithFlags(event, flags);
}

cudaError_t CUDARTAPI __wrap_cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    if (trace.failure == Failure::EventRecord && !trace.injected) {
        trace.injected = true;
        return cudaErrorInvalidResourceHandle;
    }
    return __real_cudaEventRecord(event, stream);
}

cudaError_t CUDARTAPI __wrap_cudaMemcpyAsync(void* destination, const void* source,
                                             std::size_t bytes, cudaMemcpyKind kind,
                                             cudaStream_t stream) {
    const auto status = __real_cudaMemcpyAsync(destination, source, bytes, kind, stream);
    if (active() && status == cudaSuccess) {
        trace.upload_pending = true;
        ++trace.uploads;
    }
    return status;
}

cudaError_t CUDARTAPI __wrap_cudaStreamSynchronize(cudaStream_t stream) {
    const auto status = __real_cudaStreamSynchronize(stream);
    if (active() && status == cudaSuccess) { trace.upload_pending = false; }
    return status;
}
}

#if defined(_WIN32)
namespace {

// Replace one CUDA runtime function this executable imports, returning the address the slot held.
// The entry is matched by function name rather than by DLL name because the runtime library carries
// its toolkit version in that name.
template <class Function>
Function interpose_import(const char* name, Function replacement) {
    auto* base = reinterpret_cast<std::byte*>(::GetModuleHandleW(nullptr));
    if (base == nullptr) { return nullptr; }
    const auto* dos     = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
    const auto* nt      = reinterpret_cast<const IMAGE_NT_HEADERS*>(base + dos->e_lfanew);
    const auto& imports = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    const auto* descriptor =
        reinterpret_cast<const IMAGE_IMPORT_DESCRIPTOR*>(base + imports.VirtualAddress);
    for (; descriptor->Name != 0; ++descriptor) {
        auto* slot        = reinterpret_cast<Function*>(base + descriptor->FirstThunk);
        const auto* thunk = reinterpret_cast<const IMAGE_THUNK_DATA*>(
            base + (descriptor->OriginalFirstThunk != 0 ? descriptor->OriginalFirstThunk
                                                        : descriptor->FirstThunk));
        for (; thunk->u1.Function != 0; ++thunk, ++slot) {
            if (IMAGE_SNAP_BY_ORDINAL(thunk->u1.Ordinal)) { continue; }
            const auto* entry =
                reinterpret_cast<const IMAGE_IMPORT_BY_NAME*>(base + thunk->u1.AddressOfData);
            if (std::strcmp(reinterpret_cast<const char*>(entry->Name), name) != 0) { continue; }
            DWORD previous = 0;
            if (::VirtualProtect(slot, sizeof(void*), PAGE_READWRITE, &previous) == 0) {
                return nullptr;
            }
            const Function real = *slot;
            *slot               = replacement;
            DWORD restored      = 0;
            (void)::VirtualProtect(slot, sizeof(void*), previous, &restored);
            return real;
        }
    }
    return nullptr;
}

// Interpose once, before the first allocation. A second attempt would find the wrapper in the slot
// and hand it back as the function to call through, which is why the result is cached.
void interpose_cuda_runtime() {
    static const bool installed = [] {
        const auto install = [](const char* name, auto wrapper, auto& real) {
            real = interpose_import(name, wrapper);
            if (real == nullptr) {
                // Either this executable imports nothing under that name -- a statically linked
                // runtime has no import table entry to redirect -- or the slot is not writable.
                // Both leave the fault injection inoperative, so the test must not continue as if
                // the wrapper were in place.
                throw std::runtime_error(std::string("cannot interpose the imported ") + name);
            }
        };
        install("cudaMalloc", __wrap_cudaMalloc, __real_cudaMalloc);
        install("cudaMallocHost", __wrap_cudaMallocHost, __real_cudaMallocHost);
        install("cudaFree", __wrap_cudaFree, __real_cudaFree);
        install("cudaFreeHost", __wrap_cudaFreeHost, __real_cudaFreeHost);
        install("cudaEventCreateWithFlags", __wrap_cudaEventCreateWithFlags,
                __real_cudaEventCreateWithFlags);
        install("cudaEventRecord", __wrap_cudaEventRecord, __real_cudaEventRecord);
        install("cudaMemcpyAsync", __wrap_cudaMemcpyAsync, __real_cudaMemcpyAsync);
        install("cudaStreamSynchronize", __wrap_cudaStreamSynchronize,
                __real_cudaStreamSynchronize);
        return true;
    }();
    (void)installed;
}

} // namespace
#endif

namespace ninfer::test {

void materialization_cuda_errors(DeviceContext& device) {
    using namespace artifact;
    using namespace artifact_fixture;
#if defined(_WIN32)
    interpose_cuda_runtime();
#endif
    Fixture fixture;
    fixture.write();
    for (const auto failure : {Failure::EventCreation, Failure::EventRecord}) {
        Reader reader(fixture.entry);
        Binder binder(reader);
        (void)binder.parameter("matrix", {2, 130});
        trace       = {.failure = failure};
        bool caught = false;
        try {
            (void)materialize(reader, std::move(binder).finish(), device);
        } catch (const ArtifactError&) { caught = true; }
        const auto result = std::exchange(trace, {});
        require(result.injected && caught, "CUDA loading error did not propagate as an exception");
        require(result.host_allocations > 0 && result.device_allocations > 0 &&
                    result.host_allocations == result.host_releases &&
                    result.device_allocations == result.device_releases,
                "failed loading leaked pinned or device allocations");
        require(!result.upload_pending && !result.released_during_upload,
                "failed loading released storage before completing submitted transfers");
        if (failure == Failure::EventRecord) {
            require(result.uploads > 0, "event failure did not exercise an in-flight upload");
        }
    }
    Reader reader(fixture.entry);
    Binder binder(reader);
    (void)binder.parameter("matrix", {2, 130});
    auto recovered = materialize(reader, std::move(binder).finish(), device);
    std::array<std::byte, 528> received{};
    CUDA_CHECK(cudaMemcpy(received.data(), recovered.device_parent(reader.find("q5")).data,
                          received.size(), cudaMemcpyDeviceToHost));
    require(std::equal(received.begin(), received.end(), fixture.payload.begin() + 256),
            "successful loading after CUDA failure changed weight bytes");
}

} // namespace ninfer::test
