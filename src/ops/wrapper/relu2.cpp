// ninfer::ops - relu2 wrapper: public api validation and launcher dispatch.
#include "ninfer/ops/relu2.h"

#include "ops/launcher/relu2.h" // detail::relu2_launch

#include <stdexcept>

namespace ninfer::ops {

void relu2(Tensor& x, cudaStream_t stream) {
    if (x.dtype != DType::BF16) { throw std::invalid_argument("relu2: x must be BF16"); }
    for (int i = 0; i < 4; ++i) {
        if (x.ne[i] < 0) { throw std::invalid_argument("relu2: negative dimension"); }
        if (x.ne[i] == 0) { return; }
    }
    if (!x.is_contiguous()) { throw std::invalid_argument("relu2: x must be contiguous"); }
    if (x.data == nullptr) { throw std::invalid_argument("relu2: x data must be non-null"); }
    detail::relu2_launch(x, stream);
}

} // namespace ninfer::ops
