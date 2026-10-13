#pragma once

#include "core/tensor.h"
#include "ninfer/types.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void dense_kv_ring_encode_launch(const Tensor& keys, const Tensor& values, const Tensor& slots,
                                 KvCacheStorage storage, const Tensor& key_codes,
                                 const Tensor& key_scales, const Tensor& value_codes,
                                 const Tensor& value_scales, cudaStream_t stream);

void dense_kv_ring_decode_launch(const Tensor& key_codes, const Tensor& key_scales,
                                 const Tensor& value_codes, const Tensor& value_scales,
                                 KvCacheStorage storage, Tensor& keys, Tensor& values,
                                 cudaStream_t stream);

} // namespace ninfer::ops::detail
