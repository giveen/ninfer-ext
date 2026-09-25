#include "ops/linear/q8/q8_dispatch.h"
#include "ops/linear/q8/q8_shapes.h"

#include <array>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
using ShapeSelector = Q8Launch (*)(std::int32_t);

struct ShapeEntry {
    std::int32_t n;
    std::int32_t k;
    ShapeSelector select;
};

template <class Geometry>
constexpr ShapeEntry shape(ShapeSelector select) {
    return {Geometry::kOutputRows, Geometry::kInputRows, select};
}

// A shape whose K is padded (padded_columns != k) is not served correctly by the tiled MMA
// route's scale-cache schedule. Route it through the SIMT gemm, which the shape's conformance
// already covers, instead of the K-padded MMA path.
Q8Launch select_q8_qwen4_exp_padded(std::int32_t) { return launch_q8_simt_r8_c8; }

constexpr std::array kShapes{
    shape<Q8N1024K2048>(select_q8_n1024_k2048),    shape<Q8N1024K5120>(select_q8_n1024_k5120),
    shape<Q8N2048K4096>(select_q8_n2048_k4096),
    shape<Q8N2048K4608>(select_q8_n2048_k4608),
    shape<Q8N2048K16384>(select_q8_n2048_k16384),
    shape<Q8N4608K4608>(select_q8_n4608_k4608),
    shape<Q8N5120K4608>(select_q8_n5120_k4608),
    shape<Q8N5120K6144>(select_q8_n5120_k6144),
    shape<Q8N5120K10240>(select_q8_n5120_k10240),
    shape<Q8N5120K17408>(select_q8_n5120_k17408),
    shape<Q8N5120K25600>(select_q8_n5120_k25600),
    shape<Q8N6144K5120>(select_q8_n6144_k5120),
    shape<Q8N9216K2048>(select_q8_n9216_k2048),
    shape<Q8N12288K2048>(select_q8_n12288_k2048),
    shape<Q8N14336K5120>(select_q8_n14336_k5120),
    shape<Q8N34816K5120>(select_q8_n34816_k5120),
    shape<Q8N248320K5120>(select_q8_n248320_k5120),
    // Qwen4Exp: attention q/k/gate/v, indexer q/k, mixer outputs, GDN q/k/v/z, shared expert,
    // hyper-connection mixers, PLE key/value, MTP stems and the Vision merger output.
    ShapeEntry{13312, 2560, select_q8_qwen4_exp},
    ShapeEntry{640, 2560, select_q8_qwen4_exp},
    ShapeEntry{2560, 6144, select_q8_qwen4_exp},
    ShapeEntry{16384, 2560, select_q8_qwen4_exp},
    ShapeEntry{1280, 2560, select_q8_qwen4_exp},
    ShapeEntry{2560, 640, select_q8_qwen4_exp_padded},
    ShapeEntry{324, 10240, select_q8_qwen4_exp},
    ShapeEntry{320, 10240, select_q8_qwen4_exp},
    ShapeEntry{10240, 320, select_q8_qwen4_exp_padded},
    ShapeEntry{12800, 2560, select_q8_qwen4_exp},
    ShapeEntry{2560, 2560, select_q8_qwen4_exp},
    ShapeEntry{2560, 4608, select_q8_qwen4_exp},
    ShapeEntry{96, 2560, select_q8_qwen4_exp},
};
} // namespace

Q8Launch select_q8_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    if (t <= 0) throw std::invalid_argument("q8 linear: T must be positive");
    for (const auto& entry : kShapes) {
        if (entry.n == n && entry.k == k) return entry.select(t);
    }
    throw std::invalid_argument("q8 linear: unsupported shape");
}

Q8Launch select_q8_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    if (!valid_linear_policy(policy)) throw std::invalid_argument("q8 linear: unsupported policy");
    return select_q8_a16_launch(n, k, t);
}

void q8_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream) {
    select_q8_launch(w.n, w.k, x.ne[1], policy)(x, w, out, stream);
}

} // namespace ninfer::ops::detail
