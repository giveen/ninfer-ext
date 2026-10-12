#include "ops/linear/q4rows/q4rows_dispatch.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

void q4rows_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy,
                     cudaStream_t stream) {
    // Every policy computes in the represented activations: this format has no activation
    // quantizer, so a permission to quantize them grants nothing.
    if (w.layout != QuantLayout::RowGroup || w.qdata == nullptr || w.scales == nullptr) {
        throw std::invalid_argument("linear: q4_g32_fp16_rows must be a bound row_group_v1 weight");
    }
    if (w.n % 64 != 0 || w.k % 64 != 0) {
        throw std::invalid_argument("linear: q4_g32_fp16_rows needs N and K multiples of 64");
    }
    if (reinterpret_cast<std::uintptr_t>(w.qdata) % 16 != 0 ||
        reinterpret_cast<std::uintptr_t>(w.scales) % 2 != 0) {
        throw std::invalid_argument("linear: q4_g32_fp16_rows planes are misaligned");
    }
    q4rows_launch(static_cast<const __nv_bfloat16*>(x.data),
                  static_cast<const std::uint8_t*>(w.qdata),
                  static_cast<const std::uint8_t*>(w.scales),
                  static_cast<__nv_bfloat16*>(out.data), w.n, w.k, x.ne[1], stream);
}

} // namespace ninfer::ops::detail
