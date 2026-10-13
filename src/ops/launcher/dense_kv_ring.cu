// Implements: include/ninfer/ops/dense_kv_ring.h
// One route: a warp per row for every storage, so the format only decides the plane arithmetic.
#include "ops/launcher/dense_kv_ring.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/kernel/dense_kv_ring.cuh"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int64_t kRows = kDenseKvWarps;

std::int64_t grid_for(std::int64_t units) { return (units + kRows - 1) / kRows; }

template <DenseKvFormat KeyFormat, bool KeyRotate, DenseKvFormat ValueFormat, bool ValueRotate>
void encode(const Tensor& keys, const Tensor& values, const Tensor& slots, const Tensor& key_codes,
            const Tensor& key_scales, const Tensor& value_codes, const Tensor& value_scales,
            cudaStream_t stream) {
    const std::int64_t units = static_cast<std::int64_t>(keys.ne[1]) * keys.ne[2];
    if (units <= 0) { return; }
    dense_kv_ring_encode_kernel<KeyFormat, KeyRotate, ValueFormat, ValueRotate>
        <<<static_cast<unsigned>(grid_for(units)), kDenseKvThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(keys.data),
            static_cast<const __nv_bfloat16*>(values.data),
            static_cast<const std::int32_t*>(slots.data), keys.ne[1],
            static_cast<std::int32_t>(units), static_cast<std::uint8_t*>(key_codes.data),
            static_cast<std::uint8_t*>(key_scales.data),
            static_cast<std::uint8_t*>(value_codes.data),
            static_cast<std::uint8_t*>(value_scales.data));
    CUDA_CHECK(cudaGetLastError());
}

template <DenseKvFormat KeyFormat, bool KeyRotate, DenseKvFormat ValueFormat, bool ValueRotate>
void decode(const Tensor& key_codes, const Tensor& key_scales, const Tensor& value_codes,
            const Tensor& value_scales, Tensor& keys, Tensor& values, cudaStream_t stream) {
    const std::int64_t units = static_cast<std::int64_t>(keys.ne[1]) * keys.ne[2];
    if (units <= 0) { return; }
    dense_kv_ring_decode_kernel<KeyFormat, KeyRotate, ValueFormat, ValueRotate>
        <<<static_cast<unsigned>(grid_for(units)), kDenseKvThreads, 0, stream>>>(
            static_cast<const std::uint8_t*>(key_codes.data),
            static_cast<const std::uint8_t*>(key_scales.data),
            static_cast<const std::uint8_t*>(value_codes.data),
            static_cast<const std::uint8_t*>(value_scales.data), keys.ne[1],
            static_cast<std::int32_t>(units), static_cast<__nv_bfloat16*>(keys.data),
            static_cast<__nv_bfloat16*>(values.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void dense_kv_ring_encode_launch(const Tensor& keys, const Tensor& values, const Tensor& slots,
                                 KvCacheStorage storage, const Tensor& key_codes,
                                 const Tensor& key_scales, const Tensor& value_codes,
                                 const Tensor& value_scales, cudaStream_t stream) {
    switch (storage) {
    case KvCacheStorage::Fp8E4M3Row256:
        encode<DenseKvFormat::Fp8Row, true, DenseKvFormat::Fp8Row, false>(
            keys, values, slots, key_codes, key_scales, value_codes, value_scales, stream);
        return;
    case KvCacheStorage::Int8Group64:
        encode<DenseKvFormat::Int8Group64, true, DenseKvFormat::Int8Group64, false>(
            keys, values, slots, key_codes, key_scales, value_codes, value_scales, stream);
        return;
    case KvCacheStorage::Nvfp4Group16:
        encode<DenseKvFormat::Nvfp4Group16, true, DenseKvFormat::Nvfp4Group16, true>(
            keys, values, slots, key_codes, key_scales, value_codes, value_scales, stream);
        return;
    case KvCacheStorage::Fp8KeyNvfp4Value:
        encode<DenseKvFormat::Fp8Row, true, DenseKvFormat::Nvfp4Group16, true>(
            keys, values, slots, key_codes, key_scales, value_codes, value_scales, stream);
        return;
    case KvCacheStorage::BFloat16:
        break;
    }
    throw std::invalid_argument("dense_kv_ring_encode: BFloat16 has no quantized planes");
}

void dense_kv_ring_decode_launch(const Tensor& key_codes, const Tensor& key_scales,
                                 const Tensor& value_codes, const Tensor& value_scales,
                                 KvCacheStorage storage, Tensor& keys, Tensor& values,
                                 cudaStream_t stream) {
    switch (storage) {
    case KvCacheStorage::Fp8E4M3Row256:
        decode<DenseKvFormat::Fp8Row, true, DenseKvFormat::Fp8Row, false>(
            key_codes, key_scales, value_codes, value_scales, keys, values, stream);
        return;
    case KvCacheStorage::Int8Group64:
        decode<DenseKvFormat::Int8Group64, true, DenseKvFormat::Int8Group64, false>(
            key_codes, key_scales, value_codes, value_scales, keys, values, stream);
        return;
    case KvCacheStorage::Nvfp4Group16:
        decode<DenseKvFormat::Nvfp4Group16, true, DenseKvFormat::Nvfp4Group16, true>(
            key_codes, key_scales, value_codes, value_scales, keys, values, stream);
        return;
    case KvCacheStorage::Fp8KeyNvfp4Value:
        decode<DenseKvFormat::Fp8Row, true, DenseKvFormat::Nvfp4Group16, true>(
            key_codes, key_scales, value_codes, value_scales, keys, values, stream);
        return;
    case KvCacheStorage::BFloat16:
        break;
    }
    throw std::invalid_argument("dense_kv_ring_decode: BFloat16 has no quantized planes");
}

} // namespace ninfer::ops::detail
