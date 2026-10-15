#include "ninfer/ops/mamba2_scan.h"

#include "ops/linear_attention/mamba2_scan/launch.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

using detail::mamba2_scan::kStateSize;

struct Geometry {
    std::int32_t value_dim;
    std::int32_t heads;
    std::int32_t groups;
    std::int32_t tokens;
};

void require_dtype(const Tensor& tensor, DType dtype, const char* name) {
    if (tensor.dtype != dtype) { throw std::invalid_argument(std::string("mamba2_scan: ") + name); }
}

void require_shape(const Tensor& tensor, std::int32_t n0, std::int32_t n1, std::int32_t n2,
                   std::int32_t n3, const char* name) {
    if (tensor.ne[0] != n0 || tensor.ne[1] != n1 || tensor.ne[2] != n2 || tensor.ne[3] != n3) {
        throw std::invalid_argument(std::string("mamba2_scan: invalid shape for ") + name);
    }
}

void require_contiguous_nonnull(const Tensor& tensor, const char* name) {
    if (!tensor.is_contiguous()) {
        throw std::invalid_argument(std::string("mamba2_scan: ") + name + " must be contiguous");
    }
    if (tensor.data == nullptr) {
        throw std::invalid_argument(std::string("mamba2_scan: ") + name + " data must be non-null");
    }
}

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    return lhs_begin < rhs_begin + rhs.bytes() && rhs_begin < lhs_begin + lhs.bytes();
}

// The kernel declares every represented operand, the parameters and the output __restrict__: they
// must be pairwise disjoint. The state pair has its own rule at the call site.
void require_operand_nonoverlap(const Tensor& x, const Tensor& b, const Tensor& c, const Tensor& dt,
                                const Tensor& dt_bias, const Tensor& a_log, const Tensor& d_skip,
                                const Tensor& out, const Tensor& state_in,
                                const Tensor& state_out) {
    const std::array<const Tensor*, 8> operands{&x, &b, &c, &dt, &dt_bias, &a_log, &d_skip, &out};
    for (std::size_t lhs = 0; lhs < operands.size(); ++lhs) {
        for (std::size_t rhs = lhs + 1; rhs < operands.size(); ++rhs) {
            if (overlaps(*operands[lhs], *operands[rhs])) {
                throw std::invalid_argument("mamba2_scan: tensor operands must not overlap");
            }
        }
        if (overlaps(*operands[lhs], state_in) || overlaps(*operands[lhs], state_out)) {
            throw std::invalid_argument("mamba2_scan: tensor operands must not overlap the state");
        }
    }
}

Geometry validate(const Tensor& x, const Tensor& b, const Tensor& c, const Tensor& dt,
                  const Tensor& dt_bias, const Tensor& a_log, const Tensor& d_skip,
                  float time_step_min, const Tensor& state_in, const Tensor& state_out,
                  const Tensor& out) {
    require_dtype(x, DType::BF16, "x must be BF16");
    require_dtype(b, DType::BF16, "B must be BF16");
    require_dtype(c, DType::BF16, "C must be BF16");
    require_dtype(dt, DType::BF16, "dt must be BF16");
    require_dtype(out, DType::BF16, "out must be BF16");
    require_dtype(dt_bias, DType::FP32, "dt_bias must be FP32");
    require_dtype(a_log, DType::FP32, "A_log must be FP32");
    require_dtype(d_skip, DType::FP32, "D must be FP32");
    require_dtype(state_in, DType::FP32, "state_in must be FP32");
    require_dtype(state_out, DType::FP32, "state_out must be FP32");

    const Geometry geometry{x.ne[0], x.ne[1], b.ne[1], x.ne[2]};
    if (b.ne[0] != kStateSize) {
        throw std::invalid_argument("mamba2_scan: the state size must be 128");
    }
    if (geometry.value_dim <= 0 || geometry.heads <= 0 || geometry.groups <= 0 ||
        geometry.tokens <= 0 || geometry.heads % geometry.groups != 0) {
        throw std::invalid_argument(
            "mamba2_scan: require value_dim>=1, heads>=1, groups>=1, heads%groups=0, and T>=1");
    }

    require_shape(x, geometry.value_dim, geometry.heads, geometry.tokens, 1, "x");
    require_shape(b, kStateSize, geometry.groups, geometry.tokens, 1, "B");
    require_shape(c, kStateSize, geometry.groups, geometry.tokens, 1, "C");
    require_shape(dt, geometry.heads, geometry.tokens, 1, 1, "dt");
    require_shape(dt_bias, geometry.heads, 1, 1, 1, "dt_bias");
    require_shape(a_log, geometry.heads, 1, 1, 1, "A_log");
    require_shape(d_skip, geometry.heads, 1, 1, 1, "D");
    require_shape(out, geometry.value_dim, geometry.heads, geometry.tokens, 1, "out");
    require_shape(state_in, kStateSize, geometry.value_dim, geometry.heads, 1, "state_in");
    require_shape(state_out, kStateSize, geometry.value_dim, geometry.heads, 1, "state_out");

    require_contiguous_nonnull(x, "x");
    require_contiguous_nonnull(b, "B");
    require_contiguous_nonnull(c, "C");
    require_contiguous_nonnull(dt, "dt");
    require_contiguous_nonnull(dt_bias, "dt_bias");
    require_contiguous_nonnull(a_log, "A_log");
    require_contiguous_nonnull(d_skip, "D");
    require_contiguous_nonnull(state_in, "state_in");
    require_contiguous_nonnull(state_out, "state_out");
    require_contiguous_nonnull(out, "out");

    // The recurrence's only clamp is the floor. A NaN would silently poison every following step
    // and a negative floor is a mis-set config rather than an admission.
    if (!std::isfinite(time_step_min) || time_step_min < 0.0F) {
        throw std::invalid_argument("mamba2_scan: time_step_min must be finite and non-negative");
    }
    if (state_in.data != state_out.data && overlaps(state_in, state_out)) {
        throw std::invalid_argument(
            "mamba2_scan: state input/output may only be disjoint or exactly alias");
    }
    require_operand_nonoverlap(x, b, c, dt, dt_bias, a_log, d_skip, out, state_in, state_out);
    return geometry;
}

detail::mamba2_scan::Arguments arguments(const Tensor& x, const Tensor& b, const Tensor& c,
                                         const Tensor& dt, const Tensor& dt_bias,
                                         const Tensor& a_log, const Tensor& d_skip,
                                         float time_step_min, const Tensor& state_in,
                                         Tensor& state_out, Tensor& out, const Geometry& geometry) {
    return {static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(b.data),
            static_cast<const __nv_bfloat16*>(c.data),
            static_cast<const __nv_bfloat16*>(dt.data),
            static_cast<const float*>(dt_bias.data),
            static_cast<const float*>(a_log.data),
            static_cast<const float*>(d_skip.data),
            static_cast<const float*>(state_in.data),
            static_cast<float*>(state_out.data),
            static_cast<__nv_bfloat16*>(out.data),
            geometry.value_dim,
            geometry.heads,
            geometry.groups,
            geometry.tokens,
            time_step_min};
}

} // namespace

void mamba2_scan(const Tensor& x, const Tensor& b, const Tensor& c, const Tensor& dt,
                 const Tensor& dt_bias, const Tensor& a_log, const Tensor& d_skip,
                 float time_step_min, Tensor& state, Tensor& out, cudaStream_t stream) {
    mamba2_scan(x, b, c, dt, dt_bias, a_log, d_skip, time_step_min, state, state, out, stream);
}

void mamba2_scan(const Tensor& x, const Tensor& b, const Tensor& c, const Tensor& dt,
                 const Tensor& dt_bias, const Tensor& a_log, const Tensor& d_skip,
                 float time_step_min, const Tensor& state_in, Tensor& state_out, Tensor& out,
                 cudaStream_t stream) {
    const Geometry geometry =
        validate(x, b, c, dt, dt_bias, a_log, d_skip, time_step_min, state_in, state_out, out);
    detail::mamba2_scan::launch_recurrent(arguments(x, b, c, dt, dt_bias, a_log, d_skip,
                                                    time_step_min, state_in, state_out, out,
                                                    geometry),
                                          stream);
}

} // namespace ninfer::ops
