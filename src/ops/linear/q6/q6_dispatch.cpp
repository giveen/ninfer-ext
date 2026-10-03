#include "ops/linear/q6/q6_dispatch.h"
#include "ops/linear/q6/q6_shapes.h"
#include <array>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
struct ShapeEntry {
    std::int32_t n, k;
    Q6Launch (*select)(std::int32_t);
};

constexpr std::array kShapes{
    ShapeEntry{248320, 5120, select_q6_n248320_k5120},
    ShapeEntry{34816, 5120, select_q6_n34816_k5120},
    ShapeEntry{248320, 2048, select_q6_n248320_k2048},
    ShapeEntry{1152, 1536, select_q6_n1152_k1536},
    ShapeEntry{248320, 2560, select_q6_n248320_k2560},
    // Qwen4Exp dense projections (same list as the Q8 table).
    ShapeEntry{13312, 2560, select_q6_qwen4_exp},
    ShapeEntry{640, 2560, select_q6_qwen4_exp},
    ShapeEntry{2560, 6144, select_q6_qwen4_exp},
    ShapeEntry{16384, 2560, select_q6_qwen4_exp},
    ShapeEntry{1280, 2560, select_q6_qwen4_exp},
    ShapeEntry{2560, 640, select_q6_qwen4_exp},
    ShapeEntry{324, 10240, select_q6_qwen4_exp_wide_k},
    ShapeEntry{320, 10240, select_q6_qwen4_exp_wide_k},
    ShapeEntry{10240, 320, select_q6_qwen4_exp},
    ShapeEntry{12800, 2560, select_q6_qwen4_exp},
    ShapeEntry{2560, 2560, select_q6_qwen4_exp},
    ShapeEntry{2560, 4608, select_q6_qwen4_exp},
    ShapeEntry{96, 2560, select_q6_qwen4_exp},
};
} // namespace

Q6Launch select_q6_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    if (t <= 0) throw std::invalid_argument("q6 linear: T must be positive");
    for (const auto& entry : kShapes) {
        if (entry.n == n && entry.k == k) return entry.select(t);
    }
    throw std::invalid_argument("q6 linear: unsupported shape");
}

Q6Launch select_q6_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    if (!valid_linear_policy(policy)) throw std::invalid_argument("q6 linear: unsupported policy");
    return select_q6_a16_launch(n, k, t);
}

void q6_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream) {
    select_q6_launch(weight.n, weight.k, x.ne[1], policy)(x, weight, out, stream);
}
} // namespace ninfer::ops::detail
