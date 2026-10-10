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
    // Gemma 4 31B: attention, MLP and head.
    ShapeEntry{8192, 5376, select_q6_gemma4_n8192_k5376},
    ShapeEntry{4096, 5376, select_q6_gemma4_n4096_k5376},
    ShapeEntry{16384, 5376, select_q6_gemma4_n16384_k5376},
    ShapeEntry{2048, 5376, select_q6_gemma4_n2048_k5376},
    ShapeEntry{5376, 8192, select_q6_gemma4_k8192},
    ShapeEntry{5376, 16384, select_q6_gemma4_k16384},
    ShapeEntry{21504, 5376, select_q6_gemma4_n21504_k5376},
    ShapeEntry{5376, 21504, select_q6_gemma4_k21504},
    ShapeEntry{262144, 5376, select_q6_gemma4_n262144_k5376},
    // Qwen4Exp dense projections (same list as the Q8 table).
    ShapeEntry{13312, 2560, select_q6_qwen4_exp},
    ShapeEntry{640, 2560, select_q6_qwen4_exp_small},
    ShapeEntry{2560, 6144, select_q6_qwen4_exp},
    ShapeEntry{16384, 2560, select_q6_qwen4_exp},
    ShapeEntry{1280, 2560, select_q6_qwen4_exp_small},
    ShapeEntry{2560, 640, select_q6_qwen4_exp_small},
    ShapeEntry{324, 10240, select_q6_qwen4_exp_wide_k},
    ShapeEntry{320, 10240, select_q6_qwen4_exp_wide_k},
    ShapeEntry{10240, 320, select_q6_qwen4_exp},
    ShapeEntry{12800, 2560, select_q6_qwen4_exp},
    ShapeEntry{2560, 2560, select_q6_qwen4_exp},
    ShapeEntry{2560, 4608, select_q6_qwen4_exp},
    ShapeEntry{96, 2560, select_q6_qwen4_exp_small},
    // Gemma 4 vision: q/k/v and output, and the MLP at its stored width (4304 padded to 4352).
    ShapeEntry{1152, 1152, select_q6_gemma4_vision_n1152_k1152},
    ShapeEntry{4352, 1152, select_q6_gemma4_vision_n4352_k1152},
    ShapeEntry{1152, 4352, select_q6_gemma4_vision_n1152_k4352},
    // The Gemma 4 assistant drafter: input and output projections, attention, MLP and head.
    ShapeEntry{1024, 10752, select_q6_gemma4_drafter_wide_k},
    ShapeEntry{8192, 1024, select_q6_gemma4_drafter_small},
    ShapeEntry{16384, 1024, select_q6_gemma4_drafter_n16384_k1024},
    ShapeEntry{1024, 8192, select_q6_gemma4_drafter_wide_k},
    ShapeEntry{1024, 16384, select_q6_gemma4_drafter_wide_k},
    ShapeEntry{262144, 1024, select_q6_gemma4_drafter_n262144_k1024},
    ShapeEntry{5376, 1024, select_q6_gemma4_drafter_small},
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
