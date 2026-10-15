#include "ninfer/ops/gelu.h"

#include "ops/launcher/gelu.h"
#include <stdexcept>

namespace ninfer::ops {

void gelu(Tensor& x, GeluMode mode, cudaStream_t stream) {
    if (x.dtype != DType::BF16) { throw std::invalid_argument("gelu: x must be BF16"); }
    for (int i = 0; i < 4; ++i) {
        if (x.ne[i] < 0) { throw std::invalid_argument("gelu: negative dimension"); }
        if (x.ne[i] == 0) { return; }
    }
    if (!x.is_contiguous()) { throw std::invalid_argument("gelu: x must be contiguous"); }
    if (x.data == nullptr) { throw std::invalid_argument("gelu: x data must be non-null"); }
    detail::gelu_launch(x, mode, stream);
}

} // namespace ninfer::ops
