// ninfer::ops - hyper-connection wrappers: validate the public contract, then launch.
#include "ninfer/ops/hyper_connection.h"

#include "ops/hyper_connection/launch.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require(bool condition, const char* op, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string(op) + ": " + message); }
}

void require_matrix(const Tensor& t, DType dtype, const char* op, const char* label) {
    require(t.dtype == dtype, op, (std::string(label) + " dtype mismatch").c_str());
    require(t.data != nullptr && t.is_contiguous(), op,
            (std::string(label) + " must be contiguous and non-null").c_str());
    require(t.ne[0] > 0 && t.ne[1] > 0 && t.ne[2] == 1 && t.ne[3] == 1, op,
            (std::string(label) + " must be a positive matrix").c_str());
}

} // namespace

void grouped_offset_rmsnorm(const Tensor& x, const Tensor& weight, std::int32_t groups, float eps,
                            Tensor& out, cudaStream_t stream) {
    constexpr const char* op = "grouped_offset_rmsnorm";
    require_matrix(x, DType::BF16, op, "x");
    require_matrix(out, DType::BF16, op, "out");
    require(weight.dtype == DType::BF16 && weight.is_contiguous() && weight.data != nullptr &&
                weight.numel() == x.ne[0],
            op, "weight must be contiguous BF16 [G*H]");
    require(groups > 0 && x.ne[0] % groups == 0, op, "groups must divide the row width");
    require(out.ne[0] == x.ne[0] && out.ne[1] == x.ne[1], op, "x/out shapes differ");
    require(eps > 0 && std::isfinite(eps), op, "eps must be positive and finite");
    require(x.data != out.data, op, "x and out must not overlap");
    detail::grouped_offset_rmsnorm_launch(x, weight, groups, eps, out, stream);
}

void hyper_connection_gates(const Tensor& projection, std::int32_t streams, Tensor& lowrank,
                            Tensor* inject, cudaStream_t stream) {
    constexpr const char* op = "hyper_connection_gates";
    require_matrix(projection, DType::BF16, op, "projection");
    require_matrix(lowrank, DType::BF16, op, "lowrank");
    require(streams > 0, op, "streams must be positive");
    const std::int32_t inject_rows = inject != nullptr ? streams : 0;
    require(projection.ne[0] == lowrank.ne[0] + inject_rows && projection.ne[1] == lowrank.ne[1],
            op, "projection rows must be lowrank rows plus inject rows");
    if (inject != nullptr) {
        require_matrix(*inject, DType::FP32, op, "inject");
        require(inject->ne[0] == streams && inject->ne[1] == projection.ne[1], op,
                "inject must be FP32 [streams,T]");
    }
    detail::hyper_connection_gates_launch(projection, streams, lowrank, inject, stream);
}

void hyper_connection_collapse(const Tensor& up, const Tensor& normalized, std::int32_t streams,
                               Tensor& mixed, cudaStream_t stream) {
    constexpr const char* op = "hyper_connection_collapse";
    require_matrix(up, DType::BF16, op, "up");
    require_matrix(normalized, DType::BF16, op, "normalized");
    require_matrix(mixed, DType::BF16, op, "mixed");
    require(streams > 0 && up.ne[0] == mixed.ne[0] * streams && normalized.ne[0] == up.ne[0] &&
                up.ne[1] == mixed.ne[1] && normalized.ne[1] == mixed.ne[1],
            op, "up/normalized must be [S*H,T] for mixed [H,T]");
    detail::hyper_connection_collapse_launch(up, normalized, streams, mixed, stream);
}

void hyper_connection_combine(const Tensor& y, const Tensor& inject, Tensor& residual,
                              cudaStream_t stream) {
    constexpr const char* op = "hyper_connection_combine";
    require_matrix(y, DType::BF16, op, "y");
    require_matrix(inject, DType::FP32, op, "inject");
    require_matrix(residual, DType::BF16, op, "residual");
    require(residual.ne[0] == y.ne[0] * inject.ne[0] && residual.ne[1] == y.ne[1] &&
                inject.ne[1] == y.ne[1],
            op, "residual must be [S*H,T] for y [H,T] and inject [S,T]");
    detail::hyper_connection_combine_launch(y, inject, residual, stream);
}

void hyper_connection_expand(const Tensor& x, std::int32_t streams, Tensor& residual,
                             cudaStream_t stream) {
    constexpr const char* op = "hyper_connection_expand";
    require_matrix(x, DType::BF16, op, "x");
    require_matrix(residual, DType::BF16, op, "residual");
    require(streams > 0 && residual.ne[0] == x.ne[0] * streams && residual.ne[1] == x.ne[1], op,
            "residual must be [S*H,T]");
    detail::hyper_connection_expand_launch(x, streams, residual, stream);
}

} // namespace ninfer::ops
