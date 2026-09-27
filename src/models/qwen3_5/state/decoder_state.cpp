#include "models/qwen3_5/state/decoder_state.h"

#include "core/device.h"
#include "core/kv_page_ref.h"

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

// One layer's planes are adjacent in a Host record: [first plane offset, last plane end).
struct HostLayerSpan {
    std::size_t begin = 0;
    std::size_t bytes = 0;
};

HostLayerSpan host_layer_span(const HostKVPageLayout& layout, std::size_t planes_per_layer,
                              std::uint32_t layer) {
    const std::size_t first = static_cast<std::size_t>(layer) * planes_per_layer;
    if (planes_per_layer == 0 || first + planes_per_layer > layout.planes.size()) {
        throw std::out_of_range("Host KV layer span is outside the record");
    }
    const HostKVPlaneLayout& last = layout.planes[first + planes_per_layer - 1];
    return {.begin = layout.planes[first].offset,
            .bytes = last.offset + last.page_payload_bytes - layout.planes[first].offset};
}

std::size_t staging_record_stride(const PagedKVCacheLayout& cache) {
    const HostKVPageLayout host = plan_host_kv_page_layout(cache.pages.spec.geometry);
    const std::size_t planes    = host.planes.size() / cache.layers;
    std::size_t stride          = 0;
    for (std::uint32_t layer = 0; layer < cache.layers; ++layer) {
        stride = std::max(stride, host_layer_span(host, planes, layer).bytes);
    }
    return (stride + kHostKVPageUnitBytes - 1) / kHostKVPageUnitBytes * kHostKVPageUnitBytes;
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
    if (spec.kv_staging_pages != 0) {
        if (!spec.kv_stream || spec.qsa_index) {
            throw std::invalid_argument("prefill KV staging requires dense KV streaming");
        }
        const std::size_t stride = staging_record_stride(layout.text_kv);
        layout.text_kv_staging = KVHostStagingLayout{
            .buffer = builder.add_tensor(
                DType::U8,
                {static_cast<std::int32_t>(stride),
                 static_cast<std::int32_t>(spec.kv_staging_pages)},
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
        .host          = host_base_ == nullptr
                             ? PagedKVHostPlanes{}
                             : PagedKVHostPlanes{
                                   .k       = host_base_ + host_layout_.planes[base].offset,
                                   .v       = host_base_ + host_layout_.planes[base + 1].offset,
                                   .k_scale = layer_storage_.key.has_scale()
                                                  ? host_base_ +
                                                        host_layout_.planes[k_scale_index].offset
                                                  : nullptr,
                                   .v_scale = layer_storage_.value.has_scale()
                                                  ? host_base_ +
                                                        host_layout_.planes[v_scale_index].offset
                                                  : nullptr,
                               },
    };
}

void PagedKVCache::bind_host_records(const HostKVArena& arena) {
    const HostKVPageLayout* layout = arena.layout_for(pages_.geometry());
    if (layout == nullptr || arena.base() == nullptr) {
        throw std::invalid_argument("Paged KV streaming requires a Host arena for this pool");
    }
    if (pages_.geometry().device_plane_order != PagedKVPlaneOrder::PageMajor) {
        throw std::invalid_argument("Paged KV streaming requires page-major Device planes");
    }
    host_layout_ = *layout;
    host_base_   = arena.base();
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
    return ops::QsaIndexPlane{.pages        = pages_.plane(index),
                              .block_tables = execution_tables_.matrix()};
}

std::size_t DecoderStateLayout::kv_payload_bytes() const noexcept {
    return text_kv.payload_bytes() + (mtp_kv ? mtp_kv->payload_bytes() : 0);
}

DecoderState::DecoderState(DeviceSpan backing, const DecoderStateLayout& layout)
    : text_kv(backing, layout.text_kv) {
    if (layout.mtp_kv) { mtp_kv.emplace(backing, *layout.mtp_kv); }
    if (layout.text_kv_staging) {
        text_kv_staging.emplace(backing, *layout.text_kv_staging, text_kv);
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
    words_.reserve(static_cast<std::size_t>(tables_.ne[0]));
    runs_.reserve(capacity_pages_);
}

bool KVHostStaging::prepare(std::int32_t row, std::uint32_t pages, cudaStream_t stream) {
    runs_.clear();
    const std::span<const std::int32_t> published = cache_->execution_tables().published_words(row);
    if (pages > published.size()) {
        throw std::out_of_range("prefill KV staging reads past the execution row");
    }
    if (cache_->host_base() == nullptr) { return false; }
    const auto page_units = static_cast<std::uint32_t>(cache_->host_layout().page_stride /
                                                       kHostKVPageUnitBytes);
    const auto record_units = static_cast<std::uint32_t>(record_stride_ / kHostKVPageUnitBytes);
    words_.assign(published.begin(), published.begin() + pages);
    std::uint32_t staged = 0;
    for (std::int32_t& word : words_) {
        const KVPageRef ref = KVPageRef::from_word(word);
        if (!ref.host()) { continue; }
        if (staged == capacity_pages_) {
            runs_.clear();
            return false;
        }
        const auto unit = static_cast<std::uint32_t>(ref.host_unit());
        if (!runs_.empty() && runs_.back().unit + runs_.back().count * page_units == unit) {
            ++runs_.back().count;
        } else {
            runs_.push_back(Run{.unit = unit, .index = staged, .count = 1});
        }
        word = KVPageRef(static_cast<HostPageUnit>(staged * record_units)).word();
        ++staged;
    }
    if (runs_.empty()) { return false; }
    // Pageable source: the call returns once the words are captured, so words_ may be reused.
    auto* destination = static_cast<std::int32_t*>(tables_.data) +
                        static_cast<std::size_t>(row) * static_cast<std::size_t>(tables_.ne[0]);
    CUDA_CHECK(cudaMemcpyAsync(destination, words_.data(), words_.size() * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream));
    return true;
}

PagedKVBatchLayerView KVHostStaging::stage(const PagedKVBatchLayerView& view,
                                              std::uint32_t layer, cudaStream_t stream) const {
    if (runs_.empty()) { throw std::logic_error("prefill KV staging is not prepared"); }
    const HostKVPageLayout& host = cache_->host_layout();
    const std::size_t planes     = cache_->planes_per_layer();
    const HostLayerSpan span     = host_layer_span(host, planes, layer);
    auto* buffer                 = static_cast<std::byte*>(buffer_.data);
    for (const Run& run : runs_) {
        CUDA_CHECK(cudaMemcpy2DAsync(
            buffer + static_cast<std::size_t>(run.index) * record_stride_, record_stride_,
            cache_->host_base() + static_cast<std::size_t>(run.unit) * kHostKVPageUnitBytes +
                span.begin,
            host.page_stride, span.bytes, run.count, cudaMemcpyHostToDevice, stream));
    }
    const std::size_t first = static_cast<std::size_t>(layer) * planes;
    const auto plane        = [&](std::size_t index) -> const std::byte* {
        return buffer + (host.planes[first + index].offset - span.begin);
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

PagedKVCache* DecoderState::mtp_cache() noexcept { return mtp_kv ? &*mtp_kv : nullptr; }

const PagedKVCache* DecoderState::mtp_cache() const noexcept { return mtp_kv ? &*mtp_kv : nullptr; }

} // namespace ninfer::models::qwen3_5
