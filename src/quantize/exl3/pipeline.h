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

enum class OutScales {
    // Applied when the input is regular: the top 2% of sqrt(diag(H)) carry less than
    // out_scale_skew_limit of its sum (exllamav3's measured cutoff).
    Auto,
    Always,
    Never,
};

struct TensorOptions {
    int bitrate_half_bits      = 8;
    std::uint64_t seed         = 0;
    float damping              = 0.025f; // fraction of the mean Hessian diagonal
    OutScales out_scales       = OutScales::Auto;
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
    int refit_input_skipped         = 0; // refit rounds whose input-scale solve did not converge
};

// w: [k][n] FP32 device (rows are input channels), h: [k][k] FP32 device mean XᵀX (undamped; left
// unchanged). k % 128 == 0, n % 128 == 0. Outputs: states_host receives [n/16][k/16][256] trellis
// states in trellis_t16_v1 order (written by the call), su [k] and sv [n] are device vectors, and
// wq [k][n] is the device reconstruction. The pipeline releases each device transient before the
// phase that no longer needs it, so a vocabulary-sized [248320, 5120] tensor fits on one 32 GiB GPU.
TensorReport quantize_tensor(const float* w, const float* h, std::int64_t k, std::int64_t n,
                             const TensorOptions& options, std::uint16_t* states_host, float* su,
                             float* sv, float* wq, cudaStream_t stream);

} // namespace ninfer::quantize::exl3
