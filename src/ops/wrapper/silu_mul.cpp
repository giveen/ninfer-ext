// ninfer::ops — silu_mul wrapper: implements the public api, validates parameters, and
// dispatches to the launcher. Host-compiled; never includes the kernel header.
// See docs/op-development.md §2.
#include "ninfer/ops/silu_mul.h"

#include "ops/launcher/silu_and_mul.h" // detail::silu_and_mul_launch

#include <cstdint>

#include <stdexcept>

namespace ninfer::ops {
namespace {

// The contract excludes shared storage: the kernel reads gate/up while writing out.
bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    if (lhs_begin <= rhs_begin) { return rhs_begin - lhs_begin < lhs.bytes(); }
    return lhs_begin - rhs_begin < rhs.bytes();
}

} // namespace

void silu_mul(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream) {
    const bool fp32 =
        gate.dtype == DType::FP32 && up.dtype == DType::FP32 && out.dtype == DType::FP32;
    if (!fp32 &&
        (gate.dtype != DType::BF16 || up.dtype != DType::BF16 || out.dtype != DType::BF16)) {
        throw std::invalid_argument("silu_mul: gate/up/out must all be BF16 or all be F32");
    }
    for (int d = 0; d < 4; ++d) {
        if (gate.ne[d] != up.ne[d] || gate.ne[d] != out.ne[d]) {
            throw std::invalid_argument("silu_mul: gate/up/out shapes must match");
        }
    }
    if (!out.is_contiguous()) { throw std::invalid_argument("silu_mul: out must be contiguous"); }
    if (gate.data == nullptr || up.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("silu_mul: gate/up/out data must be non-null");
    }
    if (overlaps(out, gate) || overlaps(out, up)) {
        throw std::invalid_argument("silu_mul: out must not overlap gate or up");
    }

    if (fp32) {
        detail::silu_and_mul_fp32_launch(gate, up, out, stream);
        return;
    }
    detail::silu_and_mul_launch(gate, up, out, stream); // single variant -> direct dispatch
}

} // namespace ninfer::ops
