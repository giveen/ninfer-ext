#include "ninfer/ops/prepare_ragged_prefix.h"

#include "ops/launcher/prepare_ragged_prefix.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {

void prepare_ragged_prefix(const Tensor& source, const Tensor& lanes, const Tensor& starts,
                           const Tensor& ends, Tensor& destination, Tensor& positions,
                           Tensor& counts, cudaStream_t stream) {
    const std::int32_t width = source.ne[1];
    const std::int32_t batch = destination.ne[2];
    const auto require_i32_vector  = [batch](const Tensor& tensor, const char* name) {
        if ( tensor.dtype != DType::I32 || tensor.ne[0] != batch || tensor.ne[1] != 1 ||
               tensor.ne[2] != 1 || tensor.ne[3] != 1 || ! tensor.is_contiguous() ||
               tensor.data == nullptr) {
            throw std::invalid_argument(std::string("prepare_ragged_prefix: ") + name +
                                        " must be a contiguous I32 [batch] vector");
        }
    };
    if (source.dtype != DType::BF16 || destination.dtype != DType::BF16 || source.ne[0] <= 0 ||
        (source.ne[0] % 8) != 0 || width <= 0 || source.ne[2] <= 0 || source.ne[3] != 1 ||
        destination.ne[0] != source.ne[0] || destination.ne[1] != width || batch <= 0 ||
        destination.ne[3] != 1) {
        throw std::invalid_argument("prepare_ragged_prefix: invalid source/destination geometry");
    }
    if ( positions.dtype != DType::I32 || positions.ne[0] != width ||
        positions.ne[1] != batch || positions.ne[2] != 1 || positions.ne[3] != 1) {
        throw std::invalid_argument("prepare_ragged_prefix: positions must be I32 [width,batch]");
    }
    require_i32_vector(lanes, "lanes");
    require_i32_vector(starts, "starts");
    require_i32_vector(ends, "ends");
    require_i32_vector(counts, "counts");
    if (source.data == nullptr || destination.data == nullptr || !destination.is_contiguous() ||
        !positions.is_contiguous() || positions.data == nullptr || source.nb[0] != 2 ||
        (source.nb[1] % static_cast<std::int64_t>(sizeof(uint4))) != 0 ||
        (source.nb[2] % static_cast<std::int64_t>(sizeof(uint4))) != 0 ||
        ((reinterpret_cast<std::uintptr_t>(source.data) |
          reinterpret_cast<std::uintptr_t>(destination.data)) &
         15U) != 0U) {
        throw std::invalid_argument(
            "prepare_ragged_prefix: tensors must have aligned dense-column storage");
    }
    detail::prepare_ragged_prefix_launch(source, lanes, starts, ends, destination, positions,
                                         counts, stream);
}

} // namespace ninfer::ops
