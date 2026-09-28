#pragma once

#include "core/device.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"

#include <cstdint>

namespace ninfer::ops::detail {

// Executes an EXL3 mul1 projection: out[N,T] = x[K,T] · W[N,K]ᵀ with
// W = diag(su) · H128 · Z · H128 · diag(sv), the stored trellis_t16_v1 representation.
void exl3_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                   WorkspaceArena& workspace, cudaStream_t stream);

// Measurement hook: one EXL3 weight can be perturbed by seeded Gaussian noise while a sensitivity
// run is in flight, without touching the artifact. `target` is that weight's trellis plane; `rfn`
// is the relative Frobenius norm of the added noise and `rms` the weight's per-element rms (the
// noise anchor), both zero when disabled. The dispatch copies the active probe into its kernels as
// scalars, so the host registry never races a launch that is already queued.
struct Exl3WeightProbe {
    const void* target = nullptr;
    float rfn          = 0.0F;
    float rms          = 0.0F;
    std::uint32_t seed = 0;
};

[[nodiscard]] Exl3WeightProbe& exl3_weight_probe();

// Copy `rows` complete rows of a column-major BF16 [*, columns] tensor to another from `source_row`.
// Rows of a column-major tensor are strided, so this is a pitched 2D copy. The fused consumers split
// one decoded parent into its member outputs this way.
inline void exl3_copy_rows(const Tensor& source, std::int32_t source_row, Tensor& destination,
                           std::int32_t rows, std::int32_t columns, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpy2DAsync(
        destination.data, static_cast<std::size_t>(destination.ne[0]) * sizeof(std::uint16_t),
        static_cast<const std::uint8_t*>(source.data) +
            static_cast<std::size_t>(source_row) * sizeof(std::uint16_t),
        static_cast<std::size_t>(source.ne[0]) * sizeof(std::uint16_t),
        static_cast<std::size_t>(rows) * sizeof(std::uint16_t),
        static_cast<std::size_t>(columns), cudaMemcpyDeviceToDevice, stream));
}

} // namespace ninfer::ops::detail
