// ninfer::ops - dense_kv_ring wrapper: implements the public api, validates parameters, and
// dispatches to the launcher. Host-compiled; never includes the kernel header.
#include "ninfer/ops/dense_kv_ring.h"

#include "core/paged_kv_storage.h"
#include "ops/launcher/dense_kv_ring.h" // detail::dense_kv_ring_{encode,decode}_launch

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHeadDim = 256;

PagedKVStorageLayout ring_layout(KvCacheStorage storage, const char* op) {
    if (storage == KvCacheStorage::BFloat16) {
        throw std::invalid_argument(std::string(op) + ": BFloat16 has no quantized planes");
    }
    try {
        return paged_kv_storage_layout(storage, kHeadDim);
    } catch (const std::invalid_argument&) {
        throw std::invalid_argument(std::string(op) + ": unsupported KV storage");
    }
}

void require_contiguous(const Tensor& tensor, const char* op, const char* name) {
    if (tensor.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": " + name + " data must be non-null");
    }
    if (!tensor.is_contiguous()) {
        throw std::invalid_argument(std::string(op) + ": " + name + " must be contiguous");
    }
}

// One plane of one side: the storage's code or scale element type over [extent, hkv, slots].
// NVFP4's code and scale elements are both U8, so the caller names the plane.
void require_plane(const Tensor& tensor, const PagedKVVectorLayout& layout, bool codes,
                   std::int32_t heads, const char* op, const char* name) {
    require_contiguous(tensor, op, name);
    const DType dtype         = codes ? layout.data_dtype : layout.scale_dtype;
    const std::int32_t extent = codes ? layout.data_leading_extent : layout.scale_leading_extent;
    if (extent <= 0 || tensor.dtype != dtype || tensor.ne[0] != extent || tensor.ne[1] != heads ||
        tensor.ne[2] <= 0 || tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": " + name + " has the wrong shape");
    }
}

// A ring and its four planes must agree on the head count and the slot count.
std::int32_t require_planes(const PagedKVStorageLayout& layout, const Tensor& key_codes,
                            const Tensor& key_scales, const Tensor& value_codes,
                            const Tensor& value_scales, std::int32_t heads, const char* op) {
    require_plane(key_codes, layout.key, true, heads, op, "key codes");
    require_plane(key_scales, layout.key, false, heads, op, "key scales");
    require_plane(value_codes, layout.value, true, heads, op, "value codes");
    require_plane(value_scales, layout.value, false, heads, op, "value scales");
    const std::int32_t slots = key_codes.ne[2];
    if (key_scales.ne[2] != slots || value_codes.ne[2] != slots || value_scales.ne[2] != slots) {
        throw std::invalid_argument(std::string(op) + ": the four planes must share a slot count");
    }
    return slots;
}

void require_rows(const Tensor& keys, const Tensor& values, std::int32_t rows, std::int32_t heads,
                  const char* op) {
    if (keys.dtype != DType::BF16 || values.dtype != DType::BF16) {
        throw std::invalid_argument(std::string(op) + ": keys/values must be BF16");
    }
    for (int d = 0; d < 4; ++d) {
        if (keys.ne[d] != values.ne[d]) {
            throw std::invalid_argument(std::string(op) + ": keys/values shapes must match");
        }
    }
    if (heads <= 0 || rows <= 0 || keys.ne[0] != kHeadDim || keys.ne[1] != heads ||
        keys.ne[2] != rows || keys.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": keys/values must be [256,hkv,rows]");
    }
    require_contiguous(keys, op, "keys");
    require_contiguous(values, op, "values");
}

} // namespace

void dense_kv_ring_encode(const Tensor& keys, const Tensor& values, const Tensor& slots,
                          KvCacheStorage storage, Tensor& key_codes, Tensor& key_scales,
                          Tensor& value_codes, Tensor& value_scales, cudaStream_t stream) {
    constexpr const char* op          = "dense_kv_ring_encode";
    const PagedKVStorageLayout layout = ring_layout(storage, op);
    if (slots.dtype != DType::I32 || slots.ne[0] <= 0 || slots.ne[1] != 1 || slots.ne[2] != 1 ||
        slots.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": slots must be I32 [rows]");
    }
    require_contiguous(slots, op, "slots");
    require_rows(keys, values, slots.ne[0], keys.ne[1], op);
    require_planes(layout, key_codes, key_scales, value_codes, value_scales, keys.ne[1], op);
    detail::dense_kv_ring_encode_launch(keys, values, slots, storage, key_codes, key_scales,
                                        value_codes, value_scales, stream);
}

void dense_kv_ring_decode(const Tensor& key_codes, const Tensor& key_scales,
                          const Tensor& value_codes, const Tensor& value_scales,
                          KvCacheStorage storage, Tensor& keys, Tensor& values,
                          cudaStream_t stream) {
    constexpr const char* op          = "dense_kv_ring_decode";
    const PagedKVStorageLayout layout = ring_layout(storage, op);
    const std::int32_t slots          = require_planes(layout, key_codes, key_scales, value_codes,
                                                       value_scales, key_codes.ne[1], op);
    require_rows(keys, values, slots, key_codes.ne[1], op);
    detail::dense_kv_ring_decode_launch(key_codes, key_scales, value_codes, value_scales, storage,
                                        keys, values, stream);
}

} // namespace ninfer::ops
