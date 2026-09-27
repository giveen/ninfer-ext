#pragma once

#include "core/host_kv_arena.h"
#include "core/layout.h"
#include "core/paged_kv_cache.h"
#include "ninfer/ops/sparse_attention.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
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
    // KV streaming: Host pages one text step can stage on Device (KVHostStaging).
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

// Device staging of streamed text KV for one execution step (paged-kv §6.5). Each layer's slice of
// the step's Host pages is copied once, through the copy engine, into `buffer` as compact one-layer
// records, and a staged copy of the execution tables points the Host words at those records; paged
// Ops then resolve them through the Host arm from Device memory. A prompt chunk would otherwise read
// every Host page over PCIe once per query tile; a decode round reads each once, but SM zero-copy
// over the all-layer Host records reaches well under the copy engine's rate.
struct KVHostStagingLayout {
    TensorRegion buffer; // U8 [record_stride, capacity_pages, 2]: two layer buffers
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

    struct RowPages {
        std::int32_t row    = 0; // execution-table row
        std::uint32_t pages = 0; // logical pages [0, pages) the step reads
    };

    // Stages the rows of one step executing on `stream`; layer copies run on `transfer`. Returns
    // false, staging nothing, when those pages hold no Host word or more than a buffer fits; the
    // step then reads its Host pages in place. The rows' words must not change until the step
    // ends.
    [[nodiscard]] bool prepare(std::span<const RowPages> rows, cudaStream_t stream,
                               cudaStream_t transfer);
    [[nodiscard]] bool prepare(std::int32_t row, std::uint32_t pages, cudaStream_t stream,
                               cudaStream_t transfer) {
        const RowPages single{.row = row, .pages = pages};
        return prepare(std::span<const RowPages>(&single, 1), stream, transfer);
    }

    // Returns `view` reading layer `layer`'s slices of the prepared Host pages from Device, and
    // starts copying the next layer's into the other buffer so it overlaps this layer's compute.
    // Layers are staged in increasing order within a step; the step's stream waits for each
    // copy, and a buffer is refilled only after the readers enqueued before the refill.
    [[nodiscard]] PagedKVBatchLayerView stage(const PagedKVBatchLayerView& view,
                                              std::uint32_t layer);
    // The staged QSA index plane of layer `layer`, read like stage(layer)'s view and only after it.
    [[nodiscard]] ops::QsaIndexPlane stage_index(const ops::QsaIndexPlane& plane,
                                                 std::uint32_t layer) const;

private:
    struct EventDeleter {
        void operator()(cudaEvent_t event) const noexcept { (void)cudaEventDestroy(event); }
    };
    using Event = std::unique_ptr<CUevent_st, EventDeleter>;

    void copy_layer(std::uint32_t layer, cudaStream_t stream) const;
    [[nodiscard]] std::byte* buffer(std::uint32_t layer) const;

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
    cudaStream_t stream_   = nullptr;
    cudaStream_t transfer_ = nullptr;
    std::array<Event, 2> copied_;
    Event consumed_;
    std::int64_t prefetched_layer_ = -1;
};

// Device mirror of the QSA index records of streamed pages (paged-kv §6.5). qsa_select scores
// every visible token's index record each step, so reading Host pages' records in place over PCIe
// dominates a streamed decode step; K/V is read only for the selected tokens and stays in place.
// Each (table row, logical page) owns one fixed mirror record per layer, filled from the Host
// record the first time a step reads that page from Host, and a mirrored copy of the execution
// tables points the Host words at those records.
struct QsaIndexMirrorLayout {
    TensorRegion records; // U8 [page record bytes, table rows * logical pages, layers]
    TensorRegion tables;  // I32 [logical pages, table rows], shaped like the execution tables
};

class QsaIndexMirror {
public:
    QsaIndexMirror(DeviceSpan backing, const QsaIndexMirrorLayout& layout,
                   const PagedKVCache& cache);

    QsaIndexMirror(const QsaIndexMirror&)            = delete;
    QsaIndexMirror& operator=(const QsaIndexMirror&) = delete;

    struct RowPages {
        std::int32_t row    = 0; // execution-table row
        std::uint32_t pages = 0; // logical pages [0, pages) the step reads
    };

    // Forgets `row`'s mirrored records; called when the row is bound to a sequence, since a Host
    // record may be reused by another sequence's page with the same word.
    void reset(std::int32_t row) noexcept;

    // Mirrors the rows' Host index records not yet mirrored and publishes the mirrored tables on
    // `stream`. Returns false, changing nothing, when those pages hold no Host word.
    [[nodiscard]] bool prepare(std::span<const RowPages> rows, cudaStream_t stream);

    // `plane` reading the prepared rows' Host pages from the mirror.
    [[nodiscard]] ops::QsaIndexPlane plane(const ops::QsaIndexPlane& plane,
                                           std::uint32_t layer) const;

private:
    struct Run {
        std::uint32_t unit   = 0; // Host unit of the first page
        std::uint32_t record = 0; // mirror record of the first page
        std::uint32_t count  = 0;
    };

    const PagedKVCache* cache_ = nullptr;
    Tensor records_;
    Tensor tables_;
    std::size_t record_bytes_  = 0;
    std::uint32_t row_pages_   = 0;
    std::vector<std::int32_t> mirrored_; // Host word each record holds, 0 when none
    std::vector<std::int32_t> words_;
    std::vector<Run> runs_;
};

// The prepared index mirrors one streamed round reads, or null where its rows hold no Host page.
struct QsaIndexMirrors {
    const QsaIndexMirror* text = nullptr;
    const QsaIndexMirror* mtp  = nullptr;

    [[nodiscard]] bool active() const noexcept { return text != nullptr || mtp != nullptr; }
};

struct DecoderStateLayout {
    PagedKVCacheLayout text_kv;
    std::optional<PagedKVCacheLayout> mtp_kv;
    std::optional<KVHostStagingLayout> text_kv_staging;
    // Prefill staging of the MTP layer's own cache (Qwen4Exp runs that layer over every chunk). It
    // shares the text staging buffer: the MTP layer runs after the last text layer on one stream.
    std::optional<KVHostStagingLayout> mtp_kv_staging;
    std::optional<QsaIndexMirrorLayout> text_qsa_mirror;
    std::optional<QsaIndexMirrorLayout> mtp_qsa_mirror;

    [[nodiscard]] std::size_t kv_payload_bytes() const noexcept;
};

[[nodiscard]] DecoderStateLayout plan_decoder_state(LayoutBuilder& builder,
                                                    const DecoderStateSpec& spec);

struct DecoderState {
    PagedKVCache text_kv;
    std::optional<PagedKVCache> mtp_kv;
    std::optional<KVHostStaging> text_kv_staging;
    std::optional<KVHostStaging> mtp_kv_staging;
    std::optional<QsaIndexMirror> text_qsa_mirror;
    std::optional<QsaIndexMirror> mtp_qsa_mirror;

    DecoderState(DeviceSpan backing, const DecoderStateLayout& layout);

    [[nodiscard]] PagedKVCache* mtp_cache() noexcept;
    [[nodiscard]] const PagedKVCache* mtp_cache() const noexcept;
};

} // namespace ninfer::models::qwen3_5
