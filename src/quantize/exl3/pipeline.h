#pragma once

// Single-tensor EXL3 quantization: damping, random signs and Hadamard rotation of H, block LDL,
// weight regularization (channel scales, Hadamard rotations, global-scale search), LDLQ with the
// trellis encoder, and Hessian-metric scale refit.
//
// The stored representation is W ≈ diag(su) · H128_k · Z · H128_n · diag(sv), where Z holds the
// exact mul1 integers of the trellis states (the codebook scale is folded into sv).

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::quantize::exl3 {

struct TensorOptions {
    int bitrate_half_bits = 8;
    std::uint64_t seed    = 0;
    float damping         = 0.025f; // fraction of the mean Hessian diagonal
    // Output-channel scales are applied when the input is regular: the top 2% of sqrt(diag(H))
    // carry less than this fraction of its sum (after exllamav3's measured cutoff).
    float out_scale_skew_limit = 0.15f;
    int refit_rounds           = 2;
};

struct TensorReport {
    double proxy_error_rotated = 0.0; // tr(Eᵀ H E) / tr(Wᵀ H W) in the rotated domain, after LDLQ
    double proxy_error_before_refit = 0.0; // same in the original domain
    double proxy_error              = 0.0; // after scale refit
    float global_scale              = 1.0f;
    bool out_scales                 = false;
    int damping_retries             = 0;
};

// w: [k][n] FP32 device (rows are input channels), h: [k][k] FP32 device mean XᵀX (undamped; left
// unchanged). k % 128 == 0, n % 128 == 0. Outputs (device): states [n/16][k/16][256] in
// trellis_t16_v1 order, su [k], sv [n], and the reconstruction wq [k][n].
TensorReport quantize_tensor(const float* w, const float* h, std::int64_t k, std::int64_t n,
                             const TensorOptions& options, std::uint16_t* states, float* su,
                             float* sv, float* wq, cudaStream_t stream);

} // namespace ninfer::quantize::exl3
