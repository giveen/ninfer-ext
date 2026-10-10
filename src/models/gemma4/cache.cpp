#include "models/gemma4/cache.h"

#include "core/device.h"

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

KvCache::~KvCache() {
    for (void* pointer : owned_) {
        if (pointer != nullptr) (void)cudaFree(pointer);
    }
}

KvCache::KvCache(KvCache&& other) noexcept
    : config_(other.config_), layers_(std::move(other.layers_)), owned_(std::move(other.owned_)),
      capacity_(other.capacity_), ring_slack_(other.ring_slack_), stale_end_(other.stale_end_),
      bytes_(other.bytes_) {
    other.layers_.clear();
    other.owned_.clear();
    other.bytes_ = 0;
}

KvCache& KvCache::operator=(KvCache&& other) noexcept {
    if (this != &other) {
        for (void* pointer : owned_) {
            if (pointer != nullptr) (void)cudaFree(pointer);
        }
        config_   = other.config_;
        layers_   = std::move(other.layers_);
        owned_    = std::move(other.owned_);
        capacity_   = other.capacity_;
        ring_slack_ = other.ring_slack_;
        stale_end_  = other.stale_end_;
        bytes_      = other.bytes_;
        other.layers_.clear();
        other.owned_.clear();
        other.bytes_ = 0;
    }
    return *this;
}

void KvCache::configure(const TextConfig& config, std::int32_t capacity, std::int32_t ring_slack) {
    if (capacity < 1) throw std::invalid_argument("KvCache: capacity must be positive");
    if (ring_slack < 0) throw std::invalid_argument("KvCache: ring slack must not be negative");
    config_     = config;
    capacity_   = capacity;
    ring_slack_ = ring_slack;
    stale_end_  = 0;
    layers_.assign(config.num_hidden_layers, Layer{});
    std::vector<void*>().swap(owned_);
    bytes_ = 0;
}

void KvCache::ensure(std::size_t layer) {
    if (layer >= layers_.size()) {
        throw std::invalid_argument("KvCache: layer index out of range");
    }
    Layer& entry = layers_[layer];
    if (entry.allocated) return;

    const bool global = !config_.sliding_attention(layer);
    const std::int32_t head_dim =
        static_cast<std::int32_t>(global ? config_.global.shared.head_dim : config_.sliding.head_dim);
    const std::int32_t heads = static_cast<std::int32_t>(
        global ? config_.global.shared.num_key_value_heads : config_.sliding.num_key_value_heads);
    const std::int32_t tokens =
        global ? capacity_ : ring_tokens();
    const std::int32_t width =
        global ? head_dim + 2 * static_cast<std::int32_t>(config_.global.rope_angles) : head_dim;

    const auto allocate = [&](DType dtype, std::initializer_list<std::int32_t> shape) {
        const std::size_t bytes =
            static_cast<std::size_t>(element_count(shape)) * (dtype == DType::BF16 ? 2u : 4u);
        void* pointer = nullptr;
        if (cudaMalloc(&pointer, bytes) != cudaSuccess) {
            throw std::runtime_error("KvCache: could not allocate device storage");
        }
        owned_.push_back(pointer);
        bytes_ += bytes;
        return device_tensor(pointer, dtype, shape);
    };

    entry.keys = allocate(DType::BF16, {width, heads, tokens});
    if (!global) entry.values = allocate(DType::BF16, {head_dim, heads, tokens});
    entry.positions = allocate(DType::I32, {tokens});
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
                                  std::int32_t ring_slack) {
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
        total += width * heads * tokens * sizeof(std::uint16_t);
        if (!global) total += head_dim * heads * tokens * sizeof(std::uint16_t);
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
} // namespace

Tensor KvCache::keys(std::size_t layer) const {
    return existing(layers_.at(layer).keys, layers_.at(layer).allocated);
}

Tensor KvCache::values(std::size_t layer) const {
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
    auto* base = static_cast<std::uint8_t*>(layers_[layer].positions.data);
    const std::int32_t first = slot(layer, from);
    const std::int32_t head  = std::min(count, ring_tokens() - first);
    const auto write = [&](std::int32_t at, std::int32_t n) {
        if (n > 0 && cudaMemcpyAsync(base + static_cast<std::size_t>(at) * sizeof(std::int32_t),
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
    return layers_[layer].keys;
}

Tensor KvCache::values(std::size_t layer) {
    ensure(layer);
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

void KvCache::mark_range(std::size_t layer, std::int32_t first_position, std::int32_t tokens,
                        cudaStream_t stream) {
    if (tokens < 1) { throw std::invalid_argument("KvCache: tokens must be positive"); }
    std::vector<std::int32_t> positions(static_cast<std::size_t>(tokens));
    for (std::int32_t token = 0; token < tokens; ++token) {
        positions[static_cast<std::size_t>(token)] = first_position + token;
    }
    const std::int32_t first = slot(layer, first_position);
    const std::int32_t last  = slot(layer, first_position + tokens - 1);
    auto* base               = static_cast<std::uint8_t*>(layers_[layer].positions.data);
    const std::size_t bytes  = static_cast<std::size_t>(tokens) * sizeof(std::int32_t);
    if (first + tokens - 1 <= last) {
        // The run does not wrap: one write covers it.
        if (cudaMemcpyAsync(base + static_cast<std::size_t>(first) * sizeof(std::int32_t),
                            positions.data(), bytes, cudaMemcpyHostToDevice, stream) !=
            cudaSuccess) {
            throw std::runtime_error("KvCache: could not record the positions");
        }
        return;
    }
    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::int32_t index = slot(layer, first_position + token);
        if (cudaMemcpyAsync(base + static_cast<std::size_t>(index) * sizeof(std::int32_t),
                            positions.data() + token, sizeof(std::int32_t), cudaMemcpyHostToDevice,
                            stream) != cudaSuccess) {
            throw std::runtime_error("KvCache: could not record the positions");
        }
    }
}

} // namespace ninfer::models::gemma4
