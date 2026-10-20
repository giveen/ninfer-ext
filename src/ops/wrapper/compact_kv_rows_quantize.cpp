// ninfer::ops - compact_kv_rows_quantize wrapper: implements the public api, validates parameters,
// and dispatches to the launcher. Host-compiled; never includes the kernel header.
// See docs/op-development.md §2.
#include "ninfer/ops/compact_kv_rows_quantize.h"

#include "ops/launcher/compact_kv_rows_quantize.h" // detail::compact_kv_rows_quantize_launch

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kWidth = 640;

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    if (lhs_begin <= rhs_begin) { return rhs_begin - lhs_begin < lhs.bytes(); }
    return lhs_begin - rhs_begin < rhs.bytes();
}

void require(const Tensor& tensor, DType dtype, const char* role) {
    if (tensor.dtype != dtype) {
        throw std::invalid_argument(std::string("compact_kv_rows_quantize: ") + role +
                                    " has the wrong dtype");
    }
    if (tensor.data == nullptr) {
        throw std::invalid_argument(std::string("compact_kv_rows_quantize: ") + role +
                                    " data must be non-null");
    }
    if (!tensor.is_contiguous()) {
        throw std::invalid_argument(std::string("compact_kv_rows_quantize: ") + role +
                                    " must be contiguous");
    }
}

} // namespace

void compact_kv_rows_quantize(const Tensor& rows, const Tensor& positions, Tensor& codes,
                              Tensor& scales, cudaStream_t stream) {
    require(rows, DType::BF16, "rows");
    require(positions, DType::I32, "positions");
    require(codes, DType::FP8_E4M3FN, "codes");
    require(scales, DType::FP16, "scales");

    const std::int32_t heads    = codes.ne[1];
    const std::int32_t capacity = codes.ne[2];
    const std::int32_t tokens   = positions.ne[0];
    if (codes.ne[0] != kWidth || codes.ne[3] != 1) {
        throw std::invalid_argument(
            "compact_kv_rows_quantize: codes must be FP8_E4M3FN [640,Hkv,capacity]");
    }
    if (scales.ne[0] != heads || scales.ne[1] != capacity || scales.ne[2] != 1 ||
        scales.ne[3] != 1) {
        throw std::invalid_argument("compact_kv_rows_quantize: scales must be FP16 [Hkv,capacity]");
    }
    if (heads <= 0 || capacity <= 0 || tokens < 0) {
        throw std::invalid_argument(
            "compact_kv_rows_quantize: Hkv and capacity must be positive and tokens nonnegative");
    }
    if (positions.ne[1] != 1 || positions.ne[2] != 1 || positions.ne[3] != 1) {
        throw std::invalid_argument("compact_kv_rows_quantize: positions must be I32 [tokens]");
    }
    if (rows.ne[0] != kWidth * heads || rows.ne[1] != tokens || rows.ne[2] != 1 ||
        rows.ne[3] != 1) {
        throw std::invalid_argument("compact_kv_rows_quantize: rows must be BF16 [640*Hkv,tokens]");
    }
    if (overlaps(codes, rows) || overlaps(codes, positions) || overlaps(scales, rows) ||
        overlaps(scales, positions) || overlaps(codes, scales)) {
        throw std::invalid_argument(
            "compact_kv_rows_quantize: the destination planes must not overlap the inputs");
    }

    detail::compact_kv_rows_quantize_launch(rows, positions, codes, scales, heads, capacity,
                                            stream);
}

} // namespace ninfer::ops
