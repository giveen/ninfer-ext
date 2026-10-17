#include "ninfer/ops/cast.h"

#include "ops/launcher/cast.h"

#include <cstdint>

#include <stdexcept>

namespace ninfer::ops {
namespace {

// The contract excludes any shared storage, not just identical base pointers.
bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    if (lhs_begin <= rhs_begin) { return rhs_begin - lhs_begin < lhs.bytes(); }
    return lhs_begin - rhs_begin < rhs.bytes();
}

} // namespace

void cast_fp32_to_bf16(const Tensor& source, Tensor& destination, cudaStream_t stream) {
    if (source.dtype != DType::FP32 || destination.dtype != DType::BF16) {
        throw std::invalid_argument("cast_fp32_to_bf16: source must be FP32 and destination BF16");
    }
    for (int dim = 0; dim < 4; ++dim) {
        if (source.ne[dim] != destination.ne[dim]) {
            throw std::invalid_argument("cast_fp32_to_bf16: shapes must match");
        }
    }
    if (!source.is_contiguous() || !destination.is_contiguous() || source.data == nullptr ||
        destination.data == nullptr) {
        throw std::invalid_argument("cast_fp32_to_bf16: tensors must be contiguous and non-null");
    }
    if (overlaps(source, destination)) {
        throw std::invalid_argument("cast_fp32_to_bf16: source and destination must not alias");
    }
    detail::cast_fp32_to_bf16_launch(source, destination, stream);
}

void cast_bf16_to_fp32(const Tensor& source, Tensor& destination, cudaStream_t stream) {
    if (source.dtype != DType::BF16 || destination.dtype != DType::FP32) {
        throw std::invalid_argument("cast_bf16_to_fp32: source must be BF16 and destination FP32");
    }
    for (int dim = 0; dim < 4; ++dim) {
        if (source.ne[dim] != destination.ne[dim]) {
            throw std::invalid_argument("cast_bf16_to_fp32: shapes must match");
        }
    }
    if (!source.is_contiguous() || !destination.is_contiguous() || source.data == nullptr ||
        destination.data == nullptr) {
        throw std::invalid_argument("cast_bf16_to_fp32: tensors must be contiguous and non-null");
    }
    if (overlaps(source, destination)) {
        throw std::invalid_argument("cast_bf16_to_fp32: source and destination must not alias");
    }
    detail::cast_bf16_to_fp32_launch(source, destination, stream);
}

} // namespace ninfer::ops
