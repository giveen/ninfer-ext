#include "models/qwen3_5/state/decoder_state.h"

#include "core/device.h"
#include "core/kv_page_ref.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen3_5 {
namespace {

std::uint32_t page_count(std::uint32_t capacity) {
    if (capacity == 0) { throw std::invalid_argument("Paged KV capacity must be positive"); }
    return 1U + (capacity - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

PagedKVCacheLayout plan_cache(LayoutBuilder& builder, std::uint32_t layers, std::uint32_t capacity,
                              std::int32_t kv_heads, std::int32_t head_dim, KvCacheStorage storage,
                              std::int32_t table_rows, std::uint32_t physical_page_groups,
                              bool qsa_index, bool kv_stream) {
    if (layers == 0 ||
        layers > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        kv_heads <= 0 || head_dim <= 0 || table_rows <= 0) {
        throw std::invalid_argument("Paged KV cache geometry is invalid");
    }
    const PagedKVStorageLayout layer_storage = paged_kv_storage_layout(storage, head_dim);

    const std::uint32_t logical_pages = page_count(capacity);
    if (!kv_stream && physical_page_groups < logical_pages) {
        throw std::invalid_argument("Paged KV physical pages are below logical capacity");
    }

    KVPageGeometry geometry;
    geometry.planes.reserve(static_cast<std::size_t>(layers) *
                            (layer_storage.planes_per_layer() + (qsa_index ? 1U : 0U)));
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        geometry.planes.push_back(
            {layer_storage.key.data_dtype, layer_storage.key.data_leading_extent, kv_heads, 256});
        geometry.planes.push_back({layer_storage.value.data_dtype,
                                   layer_storage.value.data_leading_extent, kv_heads, 256});
        if (layer_storage.key.has_scale()) {
            geometry.planes.push_back({layer_storage.key.scale_dtype,
                                       layer_storage.key.scale_leading_extent, kv_heads, 256});
        }
        if (layer_storage.value.has_scale()) {
            geometry.planes.push_back({layer_storage.value.scale_dtype,
                                       layer_storage.value.scale_leading_extent, kv_heads, 256});
        }
        if (qsa_index) {
            geometry.planes.push_back({DType::BF16, ops::kQsaIndexRecordWords, 1, 256});
        }
    }
    geometry.layer_planes = layer_storage.planes_per_layer() + (qsa_index ? 1U : 0U);
    return PagedKVCacheLayout{
        .pages = plan_device_kv_page_pool(
            builder, DeviceKVPagePoolSpec{.page_group_count = physical_page_groups,
                                          .geometry         = std::move(geometry)}),
        .execution_tables = plan_kv_execution_tables(
            builder,
            KVExecutionTableSpec{.logical_page_capacity = logical_pages, .table_rows = table_rows}),
        .layers        = layers,
        .max_context   = capacity,
        .kv_heads      = kv_heads,
        .layer_storage = layer_storage,
        .qsa_index     = qsa_index,
    };
}

// Bytes of one page's index record of one layer: the last plane of each layer.
std::size_t qsa_index_record_bytes(const PagedKVCacheLayout& cache) {
    const HostKVPageLayout host = plan_host_kv_page_layout(cache.pages.spec.geometry);
    return host.planes[host.planes.size() / cache.layers - 1U].page_payload_bytes;
}

QsaIndexMirrorLayout plan_qsa_mirror(LayoutBuilder& builder, const PagedKVCacheLayout& cache,
                                     const char* name) {
    const std::size_t bytes = qsa_index_record_bytes(cache);
    const auto pages        = cache.execution_tables.spec.logical_page_capacity;
    const auto rows         = cache.execution_tables.spec.table_rows;
    return QsaIndexMirrorLayout{
        .records = builder.add_tensor(DType::U8,
                                      {static_cast<std::int32_t>(bytes),
                                       static_cast<std::int32_t>(pages) * rows,
                                       static_cast<std::int32_t>(cache.layers)},
                                      256, name),
        .tables  = builder.add_tensor(DType::I32, {static_cast<std::int32_t>(pages), rows}, 256,
                                      name),
    };
}

std::size_t staging_record_stride(const PagedKVCacheLayout& cache) {
    return plan_host_kv_page_layout(cache.pages.spec.geometry).layer_span;
}

} // namespace

DecoderStateLayout plan_decoder_state(LayoutBuilder& builder, const DecoderStateSpec& spec) {
    DecoderStateLayout layout;
    layout.text_kv = plan_cache(builder, spec.full_attention_layers, spec.capacity, spec.kv_heads,
                                spec.attention_head_dim, spec.kv_storage, spec.kv_table_rows,
                                spec.text_physical_page_groups, spec.qsa_index, spec.kv_stream);
    if (spec.enable_mtp) {
        layout.mtp_kv = plan_cache(builder, spec.mtp_layers, spec.capacity, spec.kv_heads,
                                   spec.attention_head_dim, spec.kv_storage, spec.kv_table_rows,
                                   spec.mtp_physical_page_groups, spec.qsa_index, spec.kv_stream);
    }
    if (spec.enable_eagle3) {
        layout.eagle3_kv = plan_cache(builder, 1, spec.capacity, spec.eagle3_kv_heads,
                                      spec.attention_head_dim, spec.kv_storage, spec.kv_table_rows,
                                      spec.eagle3_physical_page_groups, false, spec.kv_stream);
    }
    if (spec.kv_staging_pages != 0) {
        if (!spec.kv_stream) {
            throw std::invalid_argument("prefill KV staging requires KV streaming");
        }
        const std::size_t stride = staging_record_stride(layout.text_kv);
        layout.text_kv_staging = KVHostStagingLayout{
            .buffer = builder.add_tensor(
                DType::U8,
                {static_cast<std::int32_t>(stride),
                 static_cast<std::int32_t>(spec.kv_staging_pages), 2},
                256, "Paged KV prefill staging"),
            .tables = builder.add_tensor(
                DType::I32,
                {static_cast<std::int32_t>(
                     layout.text_kv.execution_tables.spec.logical_page_capacity),
                 layout.text_kv.execution_tables.spec.table_rows},
                256, "Paged KV prefill staging tables"),
            .record_stride  = stride,
            .capacity_pages = spec.kv_staging_pages,
        };
        if (layout.mtp_kv && spec.qsa_index) {
            if (staging_record_stride(*layout.mtp_kv) > stride) {
                throw std::logic_error("MTP KV staging records exceed the text staging stride");
            }
            layout.mtp_kv_staging        = *layout.text_kv_staging;
            layout.mtp_kv_staging->tables = builder.add_tensor(
                DType::I32,
                {static_cast<std::int32_t>(layout.mtp_kv->execution_tables.spec.logical_page_capacity),
                 layout.mtp_kv->execution_tables.spec.table_rows},
                256, "MTP Paged KV prefill staging tables");
        }
    }
    if (spec.kv_stream && spec.qsa_index) {
        layout.text_qsa_mirror = plan_qsa_mirror(builder, layout.text_kv, "QSA index mirror");
        if (layout.mtp_kv) {
            layout.mtp_qsa_mirror =
                plan_qsa_mirror(builder, *layout.mtp_kv, "MTP QSA index mirror");
        }
    }
    return layout;
}

PagedKVCache::PagedKVCache(DeviceSpan backing, const PagedKVCacheLayout& layout)
    : pages_(backing, layout.pages), execution_tables_(backing, layout.execution_tables, pages_),
      layers_(layout.layers), max_context_(layout.max_context), kv_heads_(layout.kv_heads),
      layer_storage_(layout.layer_storage), qsa_index_(layout.qsa_index) {
    if (pages_.plane_count() != static_cast<std::size_t>(layers_) * layer_plane_stride()) {
        throw std::invalid_argument("Paged KV layer plane inventory is inconsistent");
    }
}

PagedKVCacheView::PagedKVCacheView(const PagedKVCache& cache, Tensor block_table) noexcept
    : cache_(&cache), block_table_(block_table) {}

std::uint32_t PagedKVCacheView::max_context() const noexcept {
    return cache_ == nullptr ? 0 : cache_->max_context();
}

PagedKVLayerView PagedKVCacheView::layer_view(std::uint32_t layer) const {
    if (cache_ == nullptr) { throw std::logic_error("Paged KV execution view is empty"); }
    return cache_->layer_view(layer, block_table_);
}

PagedKVCacheView PagedKVCache::execution_view(const KVExecutionRowLease& row) const {
    if (!row.belongs_to(execution_tables_)) {
        throw std::invalid_argument("Paged KV execution row belongs to another cache");
    }
    return PagedKVCacheView(*this, execution_tables_.row(row.handle()));
}

PagedKVLayerView PagedKVCache::layer_view(std::uint32_t layer, Tensor block_table) const {
    if (layer >= layers_) { throw std::out_of_range("Paged KV layer is out of range"); }
    const std::size_t stride        = layer_plane_stride();
    const std::size_t base          = static_cast<std::size_t>(layer) * stride;
    const std::size_t k_scale_index = base + 2;
    const std::size_t v_scale_index =
        k_scale_index + static_cast<std::size_t>(layer_storage_.key.has_scale());
    return PagedKVLayerView{
        .k_pages       = pages_.plane(base),
        .v_pages       = pages_.plane(base + 1),
        .k_scale_pages = layer_storage_.key.has_scale() ? pages_.plane(k_scale_index) : Tensor(),
        .v_scale_pages = layer_storage_.value.has_scale() ? pages_.plane(v_scale_index) : Tensor(),
        .block_table   = block_table,
        .head_dim      = layer_storage_.head_dim,
        .num_kv_heads  = kv_heads_,
        .storage       = layer_storage_.storage,
        .host =
            PagedKVHostPlanes{
                .k       = host_plane(base),
                .v       = host_plane(base + 1),
                .k_scale = layer_storage_.key.has_scale() ? host_plane(k_scale_index) : nullptr,
                .v_scale = layer_storage_.value.has_scale() ? host_plane(v_scale_index) : nullptr,
            },
    };
}

void PagedKVCache::bind_host_records(const HostKVArena& arena) {
    const HostKVPageLayout* layout = arena.layout_for(pages_.geometry());
    if (layout == nullptr || arena.plane_base(*layout, 0) == nullptr) {
        throw std::invalid_argument("Paged KV streaming requires a Host arena for this pool");
    }
    if (pages_.geometry().device_plane_order != PagedKVPlaneOrder::PageMajor) {
        throw std::invalid_argument("Paged KV streaming requires page-major Device planes");
    }
    host_layout_ = *layout;
    host_planes_.clear();
    for (std::size_t plane = 0; plane < layout->planes.size(); ++plane) {
        host_planes_.push_back(arena.plane_base(*layout, plane));
    }
}

PagedKVBatchLayerView PagedKVCache::batch_layer_view(std::uint32_t layer) const {
    const PagedKVLayerView direct = layer_view(layer, Tensor());
    return PagedKVBatchLayerView{
        .k_pages       = direct.k_pages,
        .v_pages       = direct.v_pages,
        .k_scale_pages = direct.k_scale_pages,
        .v_scale_pages = direct.v_scale_pages,
        .block_tables  = execution_tables_.matrix(),
        .head_dim      = direct.head_dim,
        .num_kv_heads  = direct.num_kv_heads,
        .storage       = direct.storage,
        .host          = direct.host,
    };
}

ops::QsaIndexPlane PagedKVCache::index_plane(std::uint32_t layer) const {
    if (!qsa_index_) { throw std::logic_error("Paged KV cache has no QSA index planes"); }
    if (layer >= layers_) { throw std::out_of_range("Paged KV layer is out of range"); }
    const std::size_t index =
        static_cast<std::size_t>(layer) * layer_plane_stride() + layer_storage_.planes_per_layer();
    return ops::QsaIndexPlane{
        .pages        = pages_.plane(index),
        .block_tables = execution_tables_.matrix(),
        .host         = host_plane(index),
    };
}

std::size_t DecoderStateLayout::kv_payload_bytes() const noexcept {
    return text_kv.payload_bytes() + (mtp_kv ? mtp_kv->payload_bytes() : 0) +
           (eagle3_kv ? eagle3_kv->payload_bytes() : 0);
}

DecoderState::DecoderState(DeviceSpan backing, const DecoderStateLayout& layout)
    : text_kv(backing, layout.text_kv) {
    if (layout.mtp_kv) { mtp_kv.emplace(backing, *layout.mtp_kv); }
    if (layout.eagle3_kv) { eagle3_kv.emplace(backing, *layout.eagle3_kv); }
    if (layout.text_kv_staging) {
        text_kv_staging.emplace(backing, *layout.text_kv_staging, text_kv);
    }
    if (layout.mtp_kv_staging) {
        if (!mtp_kv) { throw std::logic_error("MTP KV staging has no MTP KV cache"); }
        mtp_kv_staging.emplace(backing, *layout.mtp_kv_staging, *mtp_kv);
    }
    if (layout.text_qsa_mirror) { text_qsa_mirror.emplace(backing, *layout.text_qsa_mirror, text_kv); }
    if (layout.mtp_qsa_mirror) {
        if (!mtp_kv) { throw std::logic_error("MTP QSA index mirror has no MTP KV cache"); }
        mtp_qsa_mirror.emplace(backing, *layout.mtp_qsa_mirror, *mtp_kv);
    }
}

KVHostStaging::KVHostStaging(DeviceSpan backing, const KVHostStagingLayout& layout,
                                   const PagedKVCache& cache)
    : cache_(&cache), buffer_(layout.buffer.bind(backing)), tables_(layout.tables.bind(backing)),
      record_stride_(layout.record_stride), capacity_pages_(layout.capacity_pages) {
    if (record_stride_ == 0 || record_stride_ % kHostKVPageUnitBytes != 0 ||
        tables_.ne[0] != cache.execution_tables().matrix().ne[0] ||
        tables_.ne[1] != cache.execution_tables().matrix().ne[1]) {
        throw std::logic_error("Paged KV prefill staging layout is inconsistent");
    }
    words_.reserve(static_cast<std::size_t>(tables_.ne[0]) *
                   static_cast<std::size_t>(tables_.ne[1]));
    runs_.reserve(capacity_pages_);
    const auto make_event = [] {
        cudaEvent_t event = nullptr;
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        return Event(event);
    };
    copied_   = {make_event(), make_event()};
    consumed_ = make_event();
}

bool KVHostStaging::prepare(std::span<const RowPages> rows, cudaStream_t stream,
                            cudaStream_t transfer) {
    // A step that ended before its last layer may leave a prefetch in flight into a buffer the
    // next step refills on its own stream.
    if (prefetched_layer_ >= 0) {
        CUDA_CHECK(cudaStreamWaitEvent(stream, copied_[prefetched_layer_ & 1].get()));
    }
    runs_.clear();
    prefetched_layer_ = -1;
    stream_           = stream;
    transfer_         = transfer;
    if (!cache_->host_bound()) { return false; }
    if (cache_->host_layout().layer_span > record_stride_) {
        throw std::logic_error("Host KV layer span exceeds the staging record");
    }
    const auto page_units =
        static_cast<std::uint32_t>(cache_->host_layout().layer_span / kHostKVPageUnitBytes);
    const auto record_units = static_cast<std::uint32_t>(record_stride_ / kHostKVPageUnitBytes);
    const auto row_words    = static_cast<std::size_t>(tables_.ne[0]);
    // Every row's staged words, then one upload each; records are shared across the rows.
    words_.clear();
    std::uint32_t staged = 0;
    for (const RowPages& request : rows) {
        const std::span<const std::int32_t> published =
            cache_->execution_tables().published_words(request.row);
        if (request.pages > published.size()) {
            throw std::out_of_range("KV staging reads past the execution row");
        }
        const std::size_t first = words_.size();
        words_.insert(words_.end(), published.begin(), published.begin() + request.pages);
        for (std::size_t index = first; index < words_.size(); ++index) {
            const KVPageRef ref = KVPageRef::from_word(words_[index]);
            if (!ref.host()) { continue; }
            if (staged == capacity_pages_) {
                runs_.clear();
                return false;
            }
            const auto unit = static_cast<std::uint32_t>(ref.host_unit());
            if (!runs_.empty() && runs_.back().index + runs_.back().count == staged &&
                runs_.back().unit + runs_.back().count * page_units == unit) {
                ++runs_.back().count;
            } else {
                runs_.push_back(Run{.unit = unit, .index = staged, .count = 1});
            }
            words_[index] = KVPageRef(static_cast<HostPageUnit>(staged * record_units)).word();
            ++staged;
        }
    }
    if (runs_.empty()) { return false; }
    // Pageable source: each call returns once the words are captured, so words_ may be reused.
    std::size_t offset = 0;
    for (const RowPages& request : rows) {
        auto* destination = static_cast<std::int32_t*>(tables_.data) +
                            static_cast<std::size_t>(request.row) * row_words;
        CUDA_CHECK(cudaMemcpyAsync(destination, words_.data() + offset,
                                   request.pages * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                   stream));
        offset += request.pages;
    }
    return true;
}

std::byte* KVHostStaging::buffer(std::uint32_t layer) const {
    return static_cast<std::byte*>(buffer_.data) +
           static_cast<std::size_t>(layer & 1U) * record_stride_ * capacity_pages_;
}

void KVHostStaging::copy_layer(std::uint32_t layer, cudaStream_t stream) const {
    // A layer's first plane starts its span, so a run of adjacent Host pages is one copy.
    const std::size_t span = cache_->host_layout().layer_span;
    const std::byte* source =
        cache_->host_plane(static_cast<std::size_t>(layer) * cache_->planes_per_layer());
    std::byte* destination = buffer(layer);
    for (const Run& run : runs_) {
        std::byte* target      = destination + static_cast<std::size_t>(run.index) * record_stride_;
        const std::byte* first = source + static_cast<std::size_t>(run.unit) * kHostKVPageUnitBytes;
        if (span == record_stride_) {
            CUDA_CHECK(
                cudaMemcpyAsync(target, first, span * run.count, cudaMemcpyHostToDevice, stream));
        } else {
            CUDA_CHECK(cudaMemcpy2DAsync(target, record_stride_, first, span, span, run.count,
                                         cudaMemcpyHostToDevice, stream));
        }
    }
}

PagedKVBatchLayerView KVHostStaging::stage(const PagedKVBatchLayerView& view,
                                           std::uint32_t layer) {
    if (runs_.empty()) { throw std::logic_error("KV staging is not prepared"); }
    if (prefetched_layer_ == static_cast<std::int64_t>(layer)) {
        CUDA_CHECK(cudaStreamWaitEvent(stream_, copied_[layer & 1U].get()));
    } else {
        copy_layer(layer, stream_);
    }
    // The other buffer's last readers (layer - 1) are enqueued on the step stream by now.
    if (layer + 1U < cache_->layers()) {
        CUDA_CHECK(cudaEventRecord(consumed_.get(), stream_));
        CUDA_CHECK(cudaStreamWaitEvent(transfer_, consumed_.get()));
        copy_layer(layer + 1U, transfer_);
        CUDA_CHECK(cudaEventRecord(copied_[(layer + 1U) & 1U].get(), transfer_));
        prefetched_layer_ = static_cast<std::int64_t>(layer) + 1;
    }
    const HostKVPageLayout& host = cache_->host_layout();
    const std::size_t first      = static_cast<std::size_t>(layer) * cache_->planes_per_layer();
    std::byte* base              = buffer(layer);
    const auto plane             = [&](std::size_t index) -> const std::byte* {
        return base + host.planes[first + index].offset;
    };
    PagedKVBatchLayerView staged = view;
    staged.block_tables          = tables_;
    staged.host                  = PagedKVHostPlanes{
                         .k       = plane(0),
                         .v       = plane(1),
                         .k_scale = cache_->layer_storage().key.has_scale() ? plane(2) : nullptr,
                         .v_scale = cache_->layer_storage().value.has_scale()
                                        ? plane(cache_->layer_storage().key.has_scale() ? 3 : 2)
                                        : nullptr,
    };
    return staged;
}

ops::QsaIndexPlane KVHostStaging::stage_index(const ops::QsaIndexPlane& plane,
                                              std::uint32_t layer) const {
    if (runs_.empty()) { throw std::logic_error("KV staging is not prepared"); }
    // The index plane is the last plane of the layer span stage(layer) made ready.
    const std::size_t planes  = cache_->planes_per_layer();
    const std::size_t index   = static_cast<std::size_t>(layer) * planes + planes - 1U;
    ops::QsaIndexPlane staged = plane;
    staged.block_tables       = tables_;
    staged.host               = buffer(layer) + cache_->host_layout().planes[index].offset;
    return staged;
}

QsaIndexMirror::QsaIndexMirror(DeviceSpan backing, const QsaIndexMirrorLayout& layout,
                               const PagedKVCache& cache)
    : cache_(&cache), records_(layout.records.bind(backing)), tables_(layout.tables.bind(backing)) {
    const Tensor& published = cache.execution_tables().matrix();
    record_bytes_           = static_cast<std::size_t>(records_.ne[0]);
    row_pages_              = static_cast<std::uint32_t>(tables_.ne[0]);
    if (!cache.has_qsa_index() || record_bytes_ % kHostKVPageUnitBytes != 0 ||
        tables_.ne[0] != published.ne[0] || tables_.ne[1] != published.ne[1] ||
        records_.ne[1] != tables_.ne[0] * tables_.ne[1] ||
        records_.ne[2] != static_cast<std::int32_t>(cache.layers())) {
        throw std::logic_error("QSA index mirror layout is inconsistent");
    }
    mirrored_.assign(static_cast<std::size_t>(records_.ne[1]), 0);
    words_.reserve(static_cast<std::size_t>(records_.ne[1]));
    runs_.reserve(row_pages_);
}

void QsaIndexMirror::reset(std::int32_t row) noexcept {
    if (row < 0 || row >= tables_.ne[1]) { return; }
    const auto first = mirrored_.begin() + static_cast<std::ptrdiff_t>(row) * row_pages_;
    std::fill(first, first + row_pages_, 0);
}

bool QsaIndexMirror::prepare(std::span<const RowPages> rows, cudaStream_t stream) {
    if (!cache_->host_bound()) { return false; }
    const HostKVPageLayout& host = cache_->host_layout();
    const auto page_units   = static_cast<std::uint32_t>(host.layer_span / kHostKVPageUnitBytes);
    const auto record_units = static_cast<std::uint32_t>(record_bytes_ / kHostKVPageUnitBytes);
    words_.clear();
    runs_.clear();
    bool host_words = false;
    for (const RowPages& request : rows) {
        if (request.row < 0 || request.row >= tables_.ne[1]) {
            throw std::out_of_range("QSA index mirror row is out of range");
        }
        const std::span<const std::int32_t> published =
            cache_->execution_tables().published_words(request.row);
        if (request.pages > published.size()) {
            throw std::out_of_range("QSA index mirror reads past the execution row");
        }
        const std::size_t first = words_.size();
        words_.insert(words_.end(), published.begin(), published.begin() + request.pages);
        for (std::uint32_t page = 0; page < request.pages; ++page) {
            std::int32_t& word = words_[first + page];
            const KVPageRef ref = KVPageRef::from_word(word);
            if (!ref.host()) { continue; }
            host_words = true;
            const auto record = static_cast<std::uint32_t>(request.row) * row_pages_ + page;
            if (mirrored_[record] != word) {
                const auto unit = static_cast<std::uint32_t>(ref.host_unit());
                if (!runs_.empty() && runs_.back().record + runs_.back().count == record &&
                    runs_.back().unit + runs_.back().count * page_units == unit) {
                    ++runs_.back().count;
                } else {
                    runs_.push_back(Run{.unit = unit, .record = record, .count = 1});
                }
                mirrored_[record] = word;
            }
            word = KVPageRef(static_cast<HostPageUnit>(record * record_units)).word();
        }
    }
    if (!host_words) { return false; }
    // Host records are published after their copies complete (paged-kv §5.4) and stay referenced
    // by these rows, so the pinned source is stable while the copies run.
    const std::size_t planes      = cache_->planes_per_layer();
    const std::size_t layer_bytes = record_bytes_ * static_cast<std::size_t>(records_.ne[1]);
    for (std::uint32_t layer = 0; layer < cache_->layers(); ++layer) {
        const std::byte* source =
            cache_->host_plane(static_cast<std::size_t>(layer) * planes + planes - 1U);
        std::byte* destination = static_cast<std::byte*>(records_.data) + layer * layer_bytes;
        for (const Run& run : runs_) {
            CUDA_CHECK(cudaMemcpy2DAsync(
                destination + static_cast<std::size_t>(run.record) * record_bytes_, record_bytes_,
                source + static_cast<std::size_t>(run.unit) * kHostKVPageUnitBytes, host.layer_span,
                record_bytes_, run.count, cudaMemcpyHostToDevice, stream));
        }
    }
    // Pageable source: each call returns once the words are captured, so words_ may be reused.
    const auto row_words = static_cast<std::size_t>(tables_.ne[0]);
    std::size_t offset   = 0;
    for (const RowPages& request : rows) {
        auto* destination = static_cast<std::int32_t*>(tables_.data) +
                            static_cast<std::size_t>(request.row) * row_words;
        CUDA_CHECK(cudaMemcpyAsync(destination, words_.data() + offset,
                                   request.pages * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                   stream));
        offset += request.pages;
    }
    return true;
}

ops::QsaIndexPlane QsaIndexMirror::plane(const ops::QsaIndexPlane& plane,
                                         std::uint32_t layer) const {
    if (layer >= cache_->layers()) { throw std::out_of_range("QSA index mirror layer is out of range"); }
    ops::QsaIndexPlane mirrored = plane;
    mirrored.block_tables       = tables_;
    mirrored.host               = static_cast<const std::byte*>(records_.data) +
                    static_cast<std::size_t>(layer) * record_bytes_ *
                        static_cast<std::size_t>(records_.ne[1]);
    return mirrored;
}

PagedKVCache* DecoderState::mtp_cache() noexcept { return mtp_kv ? &*mtp_kv : nullptr; }

const PagedKVCache* DecoderState::mtp_cache() const noexcept { return mtp_kv ? &*mtp_kv : nullptr; }

PagedKVCache* DecoderState::eagle3_cache() noexcept { return eagle3_kv ? &*eagle3_kv : nullptr; }

const PagedKVCache* DecoderState::eagle3_cache() const noexcept {
    return eagle3_kv ? &*eagle3_kv : nullptr;
}

} // namespace ninfer::models::qwen3_5
