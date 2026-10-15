#pragma once

#include "core/dtype.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>

namespace ninfer {

struct Tensor {
    void* data         = nullptr;
    DType dtype        = DType::BF16;
    std::int32_t ne[4] = {1, 1, 1, 1};
    std::int64_t nb[4] = {0, 0, 0, 0};

    // A default-constructed Tensor is unset: `data` is null and the extents are the placeholder
    // 1x1x1x1 shape, so numel()/bytes() describe that shape rather than any storage. Test for a
    // bound tensor with `data != nullptr`.
    //
    // `data` is a non-owning view: a Tensor never allocates or frees its storage, and
    // view()/reshape()/slice()/permute() alias the same buffer, so the caller's storage must
    // outlive every Tensor derived from it.

    Tensor() noexcept = default;
    Tensor(void* data, DType dtype, std::initializer_list<std::int32_t> shape);

    std::int64_t numel() const;
    std::size_t bytes() const;
    bool is_contiguous() const;

    Tensor view(std::initializer_list<std::int32_t> shape) const;
    Tensor reshape(std::initializer_list<std::int32_t> shape) const;
    Tensor slice(int dim, std::int32_t start, std::int32_t len) const;
    Tensor permute(std::initializer_list<int> order) const;
};

} // namespace ninfer
