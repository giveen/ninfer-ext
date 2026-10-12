// Tensor-core instruction ceiling probe: the source of the kMma* constants in
// ninfer_bench_common.h.
//
// Every other Op bench divides a measured rate by one of those constants, so those numbers need a
// derivation rather than a memory of an old measurement. This bench supplies it: a register-only
// MMA loop with no memory traffic and no epilogue, so what it measures is the tensor pipe alone.
//
// It also documents a trap on this part: the plain FP8 form pays a half-rate FP32-accumulate
// penalty that the block-scaled kind::mxf8f6f4 form does not, so an A8 Op must not be ranked
// against the plain form's number.

#include "ninfer_bench_common.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

using namespace ninfer;

namespace {

struct Options {
    int iterations = 4096;
    int reps       = 10;
    std::string csv_out;
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto next = bench::required_argument(index, argc, argv);
        if (argument == "--iterations") {
            options.iterations = std::stoi(std::string(next("--iterations")));
        } else if (argument == "--reps") {
            options.reps = std::stoi(std::string(next("--reps")));
        } else if (argument == "--csv-out") {
            options.csv_out = next("--csv-out");
        } else if (argument == "--help" || argument == "-h") {
            std::printf("usage: %s [--iterations N] [--reps N] [--csv-out PATH]\n", argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argument));
        }
    }
    return options;
}

// --- the four instructions the engine's routes issue (mirrors src/ops/common/mma.cuh) ---

__device__ __forceinline__ void mma_bf16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                         unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_fp8_plain(float& c0, float& c1, float& c2, float& c3,
                                              unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                              unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_fp8_block(float& c0, float& c1, float& c2, float& c3,
                                              unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                              unsigned b0, unsigned b1) {
    constexpr unsigned kUnitScale8          = 0x7Fu; // UE8M0 encoding of 1.0
    constexpr unsigned short kScaleBlockId  = 0;
    constexpr unsigned short kScaleThreadId = 0;
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X."
                 "m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
                 "{%10}, {%11,%12}, {%13}, {%14,%15};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(kUnitScale8),
                   "h"(kScaleBlockId), "h"(kScaleThreadId), "r"(kUnitScale8), "h"(kScaleBlockId),
                   "h"(kScaleThreadId));
}

__device__ __forceinline__ void mma_nvfp4_block(float& c0, float& c1, float& c2, float& c3,
                                                unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                                unsigned b0, unsigned b1, unsigned sfa,
                                                unsigned sfb) {
    constexpr unsigned short kScaleBlockId  = 0;
    constexpr unsigned short kScaleThreadId = 0;
    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
                 "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, "
                 "{%13}, {%14,%15};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa),
                   "h"(kScaleBlockId), "h"(kScaleThreadId), "r"(sfb), "h"(kScaleBlockId),
                   "h"(kScaleThreadId));
}

template <int Form>
__device__ __forceinline__ void issue(unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                      unsigned b0, unsigned b1, float* c) {
    constexpr unsigned kNvfp4UnitScale = 0x38383838u; // four E4M3 1.0 bytes
    if constexpr (Form == 0) {
        mma_bf16(c[0], c[1], c[2], c[3], a0, a1, a2, a3, b0, b1);
    } else if constexpr (Form == 1) {
        mma_fp8_plain(c[0], c[1], c[2], c[3], a0, a1, a2, a3, b0, b1);
    } else if constexpr (Form == 2) {
        mma_fp8_block(c[0], c[1], c[2], c[3], a0, a1, a2, a3, b0, b1);
    } else {
        mma_nvfp4_block(c[0], c[1], c[2], c[3], a0, a1, a2, a3, b0, b1, kNvfp4UnitScale,
                        kNvfp4UnitScale);
    }
}

// Four independent accumulator quads: back-to-back MMAs instead of a latency-bound chain.
template <int Form, int Iterations>
__global__ void peak_kernel(float* sink) {
    float acc[16]     = {0.0F};
    const unsigned a0 = threadIdx.x * 3U + 1U;
    const unsigned a1 = a0 + 7U;
    const unsigned a2 = a0 + 13U;
    const unsigned a3 = a0 + 29U;
    const unsigned b0 = a0 ^ 0x5A5A5A5AU;
    const unsigned b1 = b0 + 11U;
    for (int iteration = 0; iteration < Iterations; ++iteration) {
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            issue<Form>(a0, a1, a2, a3, b0, b1, acc + slot * 4);
        }
    }
    float total = 0.0F;
    for (int index = 0; index < 16; ++index) { total += acc[index]; }
    if (total == 12345.678F) { *sink = total; } // never taken: keeps the loop alive
}

struct Measurement {
    const char* name;
    const char* instruction;
    double declared;
};

constexpr Measurement kForms[] = {
    {"bf16", "mma.sync.aligned.m16n8k16...bf16.bf16.f32", bench::kMmaBf16F32AccumTflops},
    {"fp8-plain", "mma.sync.aligned.m16n8k32...e4m3.e4m3.f32", bench::kMmaFp8PlainF32AccumTflops},
    {"fp8-blockscale", "kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32",
     bench::kMmaFp8BlockScaleTflops},
    {"nvfp4-blockscale", "kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64",
     bench::kMmaNvfp4BlockScaleTflops},
};

template <int Form, int Iterations>
void measure(int form_index, int blocks, int threads, int reps, double* out) {
    float* sink = nullptr;
    CUDA_CHECK(cudaMalloc(&sink, sizeof(float)));
    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    peak_kernel<Form, 64><<<blocks, threads>>>(sink); // warm
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int rep = 0; rep < reps; ++rep) {
        peak_kernel<Form, Iterations><<<blocks, threads>>>(sink);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    const double warps = static_cast<double>(blocks) * threads / 32.0;
    const double mmas  = warps * 4.0 * Iterations * reps;
    const double flops_per_mma =
        Form == 0 || Form == 1 || Form == 2 ? (Form == 0 ? 4096.0 : 8192.0) : 16384.0;
    out[form_index] = mmas * flops_per_mma / (milliseconds * 1.0e-3) / 1.0e12;
    CUDA_CHECK(cudaFree(sink));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
        const int threads  = 128;
        const int blocks   = properties.multiProcessorCount * 8;
        double measured[4] = {0.0, 0.0, 0.0, 0.0};
        measure<0, 4096>(0, blocks, threads, options.reps, measured);
        measure<1, 4096>(1, blocks, threads, options.reps, measured);
        measure<2, 4096>(2, blocks, threads, options.reps, measured);
        measure<3, 4096>(3, blocks, threads, options.reps, measured);

        std::printf(
            "device %s sm_%d%d, %d SMs, grid %d blocks x %d threads, %d iterations x %d reps\n",
            properties.name, properties.major, properties.minor, properties.multiProcessorCount,
            blocks, threads, options.iterations, options.reps);
        std::printf("%-17s %10s %10s %9s  %s\n", "form", "measured", "declared", "ratio",
                    "instruction");
        for (int form = 0; form < 4; ++form) {
            std::printf("%-17s %10.1f %10.1f %8.2fx  %s\n", kForms[form].name, measured[form],
                        kForms[form].declared, measured[form] / kForms[form].declared,
                        kForms[form].instruction);
        }
        std::printf("\nTFLOP/s, register-only loop: these bound the tensor pipe, not an Op's\n"
                    "achieved rate (an Op also pays memory, tiling and the epilogue).\n");

        if (!options.csv_out.empty()) {
            std::ofstream out = bench::open_csv(options.csv_out);
            out << "form,instruction,measured_tflops,declared_tflops,ratio\n";
            for (int form = 0; form < 4; ++form) {
                out << kForms[form].name << ',' << kForms[form].instruction << ',' << measured[form]
                    << ',' << kForms[form].declared << ','
                    << (measured[form] / kForms[form].declared) << '\n';
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_mma_peak_bench: %s\n", error.what());
        return 1;
    }
}
