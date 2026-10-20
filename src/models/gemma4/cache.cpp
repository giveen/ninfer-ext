#include "models/gemma4/cache.h"

#include "core/device.h"
#include "core/paged_kv_storage.h"

#include <stdexcept>
#include <vector>

namespace ninfer::models::gemma4 {
namespace {

Tensor device_tensor(void* pointer, DType dtype, std::initializer_list<std::int32_t> shape) {
    return Tensor(static_cast<std::uint8_t*>(pointer), dtype, shape);
}

std::int32_t element_count(std::initializer_list<std::int32_t> shape) {
    std::int32_t count = 1;
    for (const std::int32_t extent : shape) count *= extent;
    return count;
}

} // namespace

void KvCache::release_owned() noexcept {
    for (void* pointer : owned_) {
        if (pointer != nullptr) (void)cudaFree(pointer);
    }
    owned_.clear();
}

KvCache::~KvCache() { release_owned(); }

KvCache::KvCache(KvCache&& other) noexcept
    : config_(other.config_), layers_(std::move(other.layers_)), owned_(std::move(other.owned_)),
      capacity_(other.capacity_), ring_slack_(other.ring_slack_), stale_end_(other.stale_end_),
      bytes_(other.bytes_), storage_(other.storage_), global_fp8_(other.global_fp8_) {
    other.layers_.clear();
    other.owned_.clear();
    other.bytes_ = 0;
}

KvCache& KvCache::operator=(KvCache&& other) noexcept {
    if (this != &other) {
        release_owned();
        config_     = other.config_;
        layers_     = std::move(other.layers_);
        owned_      = std::move(other.owned_);
        capacity_   = other.capacity_;
        ring_slack_ = other.ring_slack_;
        stale_end_  = other.stale_end_;
        bytes_      = other.bytes_;
        storage_    = other.storage_;
        global_fp8_ = other.global_fp8_;
        other.layers_.clear();
        other.owned_.clear();
        other.bytes_ = 0;
    }
    return *this;
}

void KvCache::configure(const TextConfig& config, std::int32_t capacity, std::int32_t ring_slack,
                        KvCacheStorage storage) {
    if (capacity < 1) throw std::invalid_argument("KvCache: capacity must be positive");
    if (ring_slack < 0) throw std::invalid_argument("KvCache: ring slack must not be negative");
    if (storage != KvCacheStorage::BFloat16) {
        // The quantized storages are the D256 codecs, and they cover the sliding rings.
        try {
            (void)paged_kv_storage_layout(storage,
                                          static_cast<std::int32_t>(config.sliding.head_dim));
        } catch (const std::invalid_argument&) {
            throw std::invalid_argument("KvCache: unsupported sliding KV-cache storage");
        }
    }
    // FP8 is the one storage with a codec for the global layers' compact rows; it is registered for
    // the 640-value row of an R = 512 head with 64 rotated pairs.
    const bool global_fp8 = storage == KvCacheStorage::Fp8E4M3Row256;
    const std::int32_t global_width =
        static_cast<std::int32_t>(config.global.shared.head_dim) + 2 * config.global.rope_angles;
    if (global_fp8 && global_width != 640) {
        throw std::invalid_argument(
            "KvCache: the FP8 global codec is registered for 640-wide compact rows only");
    }
    config_     = config;
    capacity_   = capacity;
    ring_slack_ = ring_slack;
    storage_    = storage;
    global_fp8_ = global_fp8;
    stale_end_  = 0;
    layers_.assign(config.num_hidden_layers, Layer{});
    release_owned();
    bytes_ = 0;
}

void KvCache::ensure(std::size_t layer) {
    if (layer >= layers_.size()) {
        throw std::invalid_argument("KvCache: layer index out of range");
    }
    Layer& entry = layers_[layer];
    if (entry.allocated) return;

    const bool global           = !config_.sliding_attention(layer);
    const std::int32_t head_dim = static_cast<std::int32_t>(global ? config_.global.shared.head_dim
                                                                   : config_.sliding.head_dim);
    const std::int32_t heads    = static_cast<std::int32_t>(
        global ? config_.global.shared.num_key_value_heads : config_.sliding.num_key_value_heads);
    const std::int32_t tokens = global ? capacity_ : ring_tokens();
    const std::int32_t width =
        global ? head_dim + 2 * static_cast<std::int32_t>(config_.global.rope_angles) : head_dim;

    const auto allocate = [&](DType dtype, std::initializer_list<std::int32_t> shape, bool zero) {
        const std::size_t bytes =
            static_cast<std::size_t>(element_count(shape)) * dtype_size(dtype);
        void* pointer = nullptr;
        if (cudaMalloc(&pointer, bytes) != cudaSuccess) {
            throw std::runtime_error("KvCache: could not allocate device storage");
        }
        if (zero && cudaMemset(pointer, 0, bytes) != cudaSuccess) {
            (void)cudaFree(pointer);
            throw std::runtime_error("KvCache: could not clear device storage");
        }
        owned_.push_back(pointer);
        bytes_ += bytes;
        return device_tensor(pointer, dtype, shape);
    };

    const bool quantized = !global && storage_ != KvCacheStorage::BFloat16;
    if (global && global_fp8_) {
        // One E4M3 code per compact-row value and one FP16 scale per (position, KV head), cleared
        // so that an unwritten slot decodes to zero rather than to whatever the allocation held;
        // the position plane makes such a slot invisible before any row of it is read.
        entry.key_codes  = allocate(DType::FP8_E4M3FN, {width, heads, tokens}, true);
        entry.key_scales = allocate(DType::FP16, {heads, tokens}, true);
    } else
    if (quantized) {
        // The quantized ring's planes, cleared so that a slot no token has written decodes to zero
        // rather than to whatever the allocation held. The planes' schema is the storage's own
        // (paged_kv_storage_layout), so the model and the codec Ops cannot disagree about it.
        const PagedKVStorageLayout layout =
            paged_kv_storage_layout(storage_, static_cast<std::int32_t>(head_dim));
        entry.key_codes =
            allocate(layout.key.data_dtype, {layout.key.data_leading_extent, heads, tokens}, true);
        entry.key_scales   = allocate(layout.key.scale_dtype,
                                      {layout.key.scale_leading_extent, heads, tokens}, true);
        entry.value_codes  = allocate(layout.value.data_dtype,
                                      {layout.value.data_leading_extent, heads, tokens}, true);
        entry.value_scales = allocate(layout.value.scale_dtype,
                                      {layout.value.scale_leading_extent, heads, tokens}, true);
    } else {
        entry.keys = allocate(DType::BF16, {width, heads, tokens}, false);
        if (!global) entry.values = allocate(DType::BF16, {head_dim, heads, tokens}, false);
    }
    entry.positions = allocate(DType::I32, {tokens}, false);
    entry.tokens    = tokens;
    entry.allocated = true;
    reset_layer(layer);
}

void KvCache::reset_layer(std::size_t layer) {
    Layer& entry = layers_[layer];
    std::vector<std::int32_t> unwritten(static_cast<std::size_t>(entry.tokens), kUnwrittenPosition);
    if (cudaMemcpy(entry.positions.data, unwritten.data(),
                   static_cast<std::size_t>(entry.tokens) * sizeof(std::int32_t),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        throw std::runtime_error("KvCache: could not reset the positions");
    }
}

void KvCache::allocate() {
    for (std::size_t layer = 0; layer < layers_.size(); ++layer) ensure(layer);
}

std::size_t KvCache::device_bytes(const TextConfig& config, std::int32_t capacity,
                                  std::int32_t ring_slack, KvCacheStorage storage) {
    std::size_t total = 0;
    for (std::size_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        const bool global = !config.sliding_attention(layer);
        const std::size_t head_dim =
            global ? config.global.shared.head_dim : config.sliding.head_dim;
        const std::size_t heads =
            global ? config.global.shared.num_key_value_heads : config.sliding.num_key_value_heads;
        const std::size_t tokens =
            global ? static_cast<std::size_t>(capacity)
                   : config.sliding_window + static_cast<std::size_t>(ring_slack);
        const std::size_t width = global ? head_dim + 2 * config.global.rope_angles : head_dim;
        if (!global && storage != KvCacheStorage::BFloat16) {
            const PagedKVStorageLayout layout =
                paged_kv_storage_layout(storage, static_cast<std::int32_t>(head_dim));
            total += (layout.key.physical_bytes() + layout.value.physical_bytes()) * heads * tokens;
        } else if (global && storage == KvCacheStorage::Fp8E4M3Row256) {
            total += width * heads * tokens;                 // E4M3 codes
            total += heads * tokens * sizeof(std::uint16_t); // one scale per row
        } else {
            total += width * heads * tokens * sizeof(std::uint16_t);
            if (!global) total += head_dim * heads * tokens * sizeof(std::uint16_t);
        }
        total += tokens * sizeof(std::int32_t);
    }
    return total;
}

void KvCache::reset() {
    for (std::size_t layer = 0; layer < layers_.size(); ++layer) {
        if (layers_[layer].allocated) reset_layer(layer);
    }
    stale_end_ = 0;
}

namespace {
const Tensor& existing(const Tensor& tensor, bool allocated) {
    if (!allocated) throw std::logic_error("KvCache: the layer has no storage yet");
    return tensor;
}

// A plane of a quantized ring: absent for a BF16 ring, which stores rows instead.
const Tensor& existing_plane(const Tensor& tensor, bool quantized) {
    if (!quantized) throw std::logic_error("KvCache: the layer's ring is not quantized");
    if (tensor.data == nullptr) throw std::logic_error("KvCache: the layer has no storage yet");
    return tensor;
}
} // namespace

Tensor KvCache::key_codes(std::size_t layer) const {
    const bool coded = quantized(layer) || coded_global(layer);
    return existing_plane(layers_.at(layer).key_codes, coded);
}

Tensor KvCache::key_scales(std::size_t layer) const {
    const bool coded = quantized(layer) || coded_global(layer);
    return existing_plane(layers_.at(layer).key_scales, coded);
}

Tensor KvCache::value_codes(std::size_t layer) const {
    return existing_plane(layers_.at(layer).value_codes, quantized(layer));
}

Tensor KvCache::value_scales(std::size_t layer) const {
    return existing_plane(layers_.at(layer).value_scales, quantized(layer));
}

Tensor KvCache::keys(std::size_t layer) const {
    if (quantized(layer) || coded_global(layer)) { throw std::logic_error("KvCache: the layer's rows are quantized");
    }
    return existing(layers_.at(layer).keys, layers_.at(layer).allocated);
}

Tensor KvCache::values(std::size_t layer) const {
    if (quantized(layer) || coded_global(layer)) { throw std::logic_error("KvCache: the layer's rows are quantized");
    }
    return existing(layers_.at(layer).values, layers_.at(layer).allocated);
}

Tensor KvCache::positions(std::size_t layer) const {
    return existing(layers_.at(layer).positions, layers_.at(layer).allocated);
}

void KvCache::forget_stale(std::size_t layer, std::int32_t from, cudaStream_t stream) {
    if (!config_.sliding_attention(layer) || stale_end_ <= from) return;
    ensure(layer);
    const std::int32_t count = std::min(stale_end_ - from, ring_tokens());
    // The marker vector must outlive the copies, which a pageable source guarantees by staging.
    const std::vector<std::int32_t> unwritten(static_cast<std::size_t>(count), kUnwrittenPosition);
    auto* base               = static_cast<std::uint8_t*>(layers_[layer].positions.data);
    const std::int32_t first = slot(layer, from);
    const std::int32_t head  = std::min(count, ring_tokens() - first);
    const auto write         = [&](std::int32_t at, std::int32_t n) {
        if (n > 0 &&
            cudaMemcpyAsync(base + static_cast<std::size_t>(at) * sizeof(std::int32_t),
                            unwritten.data(), static_cast<std::size_t>(n) * sizeof(std::int32_t),
                            cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            throw std::runtime_error("KvCache: could not forget stale rows");
        }
    };
    write(first, head);
    write(0, count - head);
}

Tensor KvCache::keys(std::size_t layer) {
    ensure(layer);
    if (quantized(layer) || coded_global(layer)) { throw std::logic_error("KvCache: the layer's rows are quantized");
    }
    return layers_[layer].keys;
}

Tensor KvCache::key_codes(std::size_t layer) {
    ensure(layer);
    if (!quantized(layer) && !coded_global(layer)) { throw std::logic_error("KvCache: the layer's rows are not quantized");
    }
    return layers_[layer].key_codes;
}

Tensor KvCache::key_scales(std::size_t layer) {
    ensure(layer);
    if (!quantized(layer) && !coded_global(layer)) { throw std::logic_error("KvCache: the layer's rows are not quantized");
    }
    return layers_[layer].key_scales;
}

Tensor KvCache::value_codes(std::size_t layer) {
    ensure(layer);
    if (!quantized(layer)) throw std::logic_error("KvCache: the layer's ring is not quantized");
    return layers_[layer].value_codes;
}

Tensor KvCache::value_scales(std::size_t layer) {
    ensure(layer);
    if (!quantized(layer)) throw std::logic_error("KvCache: the layer's ring is not quantized");
    return layers_[layer].value_scales;
}

Tensor KvCache::values(std::size_t layer) {
    ensure(layer);
    if (quantized(layer)) throw std::logic_error("KvCache: the layer's ring is quantized");
    return layers_[layer].values;
}

Tensor KvCache::positions(std::size_t layer) {
    ensure(layer);
    return layers_[layer].positions;
}

std::int32_t KvCache::slot(std::size_t layer, std::int32_t position) const {
    if (layer >= layers_.size()) throw std::invalid_argument("KvCache: layer index out of range");
    if (!config_.sliding_attention(layer)) {
        if (position >= capacity_) {
            throw std::invalid_argument("KvCache: position is beyond the configured capacity");
        }
        return position;
    }
    return position % ring_tokens();
}

} // namespace ninfer::models::gemma4
