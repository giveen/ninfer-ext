#pragma once

#include "core/host_kv_arena.h"
#include "core/layout.h"
#include "core/paged_kv_cache.h"
#include "ninfer/ops/sparse_attention.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace ninfer::models::qwen3_5 {

inline constexpr std::int32_t kKvInt8QuantGroup = 64;
inline constexpr std::int32_t kKvFp8QuantGroup  = 256;

struct DecoderStateSpec {
    std::uint32_t full_attention_layers     = 0;
    std::uint32_t mtp_layers                = 0;
    std::uint32_t capacity                  = 0;
    std::int32_t kv_heads                   = 0;
    std::int32_t attention_head_dim         = 0;
    KvCacheStorage kv_storage               = KvCacheStorage::BFloat16;
    bool enable_mtp                         = false;
    std::int32_t kv_table_rows              = 1;
    std::uint32_t text_physical_page_groups = 0;
    std::uint32_t mtp_physical_page_groups  = 0;
    // Qwen Sparse Attention: every attention layer also pages one indexer-key record per token.
    bool qsa_index = false;
    // KV streaming: the physical pools may be smaller than one full-capacity sequence, since
    // older pages are served from Host records.
    bool kv_stream = false;
    // KV streaming: Host pages one text prefill chunk can stage on Device (KVHostStaging).
    std::uint32_t kv_staging_pages = 0;
};

struct PagedKVCacheLayout {
    DeviceKVPagePoolLayout pages;
    KVExecutionTableLayout execution_tables;
    std::uint32_t layers      = 0;
    std::uint32_t max_context = 0;
    std::int32_t kv_heads     = 0;
    PagedKVStorageLayout layer_storage;
    bool qsa_index = false;

    [[nodiscard]] std::size_t payload_bytes() const noexcept { return pages.payload_bytes(); }
};

class PagedKVCache;

class PagedKVCacheView {
public:
    PagedKVCacheView() noexcept = default;

    [[nodiscard]] bool valid() const noexcept { return cache_ != nullptr; }

    [[nodiscard]] std::uint32_t max_context() const noexcept;
    [[nodiscard]] PagedKVLayerView layer_view(std::uint32_t layer) const;

private:
    friend class PagedKVCache;
    PagedKVCacheView(const PagedKVCache& cache, Tensor block_table) noexcept;

    const PagedKVCache* cache_ = nullptr;
    Tensor block_table_;
};

class PagedKVCache {
public:
    PagedKVCache(DeviceSpan backing, const PagedKVCacheLayout& layout);

    PagedKVCache(const PagedKVCache&)            = delete;
    PagedKVCache& operator=(const PagedKVCache&) = delete;
    PagedKVCache(PagedKVCache&&)                 = delete;
    PagedKVCache& operator=(PagedKVCache&&)      = delete;

    [[nodiscard]] std::uint32_t max_context() const noexcept { return max_context_; }

    [[nodiscard]] std::uint32_t layers() const noexcept { return layers_; }

    [[nodiscard]] auto& page_pool(this auto& self) noexcept { return self.pages_; }

    [[nodiscard]] auto& execution_tables(this auto& self) noexcept {
        return self.execution_tables_;
    }

    [[nodiscard]] PagedKVCacheView execution_view(const KVExecutionRowLease& row) const;

    [[nodiscard]] PagedKVBatchLayerView batch_layer_view(std::uint32_t layer) const;

    [[nodiscard]] bool has_qsa_index() const noexcept { return qsa_index_; }

    // Indexer-key plane of one layer over all execution table rows; requires has_qsa_index().
    [[nodiscard]] ops::QsaIndexPlane index_plane(std::uint32_t layer) const;

    // KV streaming: layer views carry `arena`'s Host plane bases so execution rows may hold Host
    // words (KVPageRef). Bind before any view is captured into a CUDA Graph; the arena outlives
    // the cache and its base is stable for the Engine lifetime.
    void bind_host_records(const HostKVArena& arena);

    [[nodiscard]] const std::byte* host_base() const noexcept { return host_base_; }
    [[nodiscard]] const HostKVPageLayout& host_layout() const noexcept { return host_layout_; }
    [[nodiscard]] const PagedKVStorageLayout& layer_storage() const noexcept {
        return layer_storage_;
    }
    [[nodiscard]] std::size_t planes_per_layer() const noexcept { return layer_plane_stride(); }

private:
    [[nodiscard]] std::size_t layer_plane_stride() const noexcept {
        return layer_storage_.planes_per_layer() + (qsa_index_ ? 1U : 0U);
    }

    friend class PagedKVCacheView;
    [[nodiscard]] PagedKVLayerView layer_view(std::uint32_t layer, Tensor block_table) const;

    DeviceKVPagePool pages_;
    KVExecutionTablePool execution_tables_;
    std::uint32_t layers_      = 0;
    std::uint32_t max_context_ = 0;
    std::int32_t kv_heads_     = 0;
    PagedKVStorageLayout layer_storage_;
    bool qsa_index_ = false;
    const std::byte* host_base_ = nullptr;
    HostKVPageLayout host_layout_;
};

// Device staging for a streamed text prefill chunk (paged-kv §6.5). A prompt kernel reads every
// visible page once per query tile, so reading Host pages in place multiplies PCIe traffic by the
// tile count. Instead each layer's slice of the chunk's Host pages is copied once into `buffer`,
// laid out as compact one-layer records, and a staged copy of the execution tables points the
// Host words at those records; paged Ops then resolve them through the Host arm from Device memory.
struct KVHostStagingLayout {
    TensorRegion buffer; // U8 [record_stride, capacity_pages]
    TensorRegion tables; // I32 [logical pages, table rows], shaped like the execution tables
    std::size_t record_stride    = 0;
    std::uint32_t capacity_pages = 0;
};

class KVHostStaging {
public:
    KVHostStaging(DeviceSpan backing, const KVHostStagingLayout& layout,
                     const PagedKVCache& cache);

    KVHostStaging(const KVHostStaging&)            = delete;
    KVHostStaging& operator=(const KVHostStaging&) = delete;

    // Stages execution row `row` for a chunk that reads logical pages [0, pages). Returns false,
    // staging nothing, when those pages hold no Host word or more than the buffer fits; the chunk
    // then reads its Host pages in place. The row's words must not change until the chunk ends.
    [[nodiscard]] bool prepare(std::int32_t row, std::uint32_t pages, cudaStream_t stream);

    // Enqueues the copy of layer `layer`'s slices of the prepared Host pages and returns `view`
    // reading them from the staging buffer. Stream-ordered: the next layer's copy overwrites the
    // buffer after this layer's readers.
    [[nodiscard]] PagedKVBatchLayerView stage(const PagedKVBatchLayerView& view,
                                              std::uint32_t layer, cudaStream_t stream) const;

private:
    struct Run {
        std::uint32_t unit  = 0; // Host unit of the first page
        std::uint32_t index = 0; // staging record of the first page
        std::uint32_t count = 0;
    };

    const PagedKVCache* cache_ = nullptr;
    Tensor buffer_;
    Tensor tables_;
    std::size_t record_stride_    = 0;
    std::uint32_t capacity_pages_ = 0;
    std::vector<std::int32_t> words_;
    std::vector<Run> runs_;
};

struct DecoderStateLayout {
    PagedKVCacheLayout text_kv;
    std::optional<PagedKVCacheLayout> mtp_kv;
    std::optional<KVHostStagingLayout> text_kv_staging;

    [[nodiscard]] std::size_t kv_payload_bytes() const noexcept;
};

[[nodiscard]] DecoderStateLayout plan_decoder_state(LayoutBuilder& builder,
                                                    const DecoderStateSpec& spec);

struct DecoderState {
    PagedKVCache text_kv;
    std::optional<PagedKVCache> mtp_kv;
    std::optional<KVHostStaging> text_kv_staging;

    DecoderState(DeviceSpan backing, const DecoderStateLayout& layout);

    [[nodiscard]] PagedKVCache* mtp_cache() noexcept;
    [[nodiscard]] const PagedKVCache* mtp_cache() const noexcept;
};

} // namespace ninfer::models::qwen3_5
