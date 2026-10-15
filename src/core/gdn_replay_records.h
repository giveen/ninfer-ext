#pragma once

#include "core/layout.h"
#include "core/tensor.h"

#include <cstddef>
#include <cstdint>

namespace ninfer {

// Every field must be positive, record_capacity is capped at eight rows, value_heads must be
// divisible by qk_heads, and layers * record_capacity must fit in int32_t. A spec that violates
// one of these fails in plan_gdn_replay_records with std::invalid_argument or
// std::overflow_error.

struct GdnReplayRecordSpec {
    std::int32_t layers          = 0;
    std::int32_t record_capacity = 0;
    std::int32_t width           = 0;
    std::int32_t conv_channels   = 0;
    std::int32_t qk_heads        = 0;
    std::int32_t value_heads     = 0;
    std::int32_t key_dim         = 0;
    std::int32_t value_dim       = 0;
};

struct GdnReplayRecordLayout {
    GdnReplayRecordSpec spec;
    TensorRegion conv;
    TensorRegion key;
    TensorRegion value;
    TensorRegion gate;

    [[nodiscard]] std::size_t payload_bytes() const noexcept;
};

[[nodiscard]] GdnReplayRecordLayout plan_gdn_replay_records(LayoutBuilder& builder,
                                                            const GdnReplayRecordSpec& spec);

struct GdnReplayRecordLayer {
    Tensor conv;  // BF16 [conv_channels, width, rows]
    Tensor key;   // BF16 [key_dim, qk_heads, width, rows]
    Tensor value; // BF16 [value_dim, value_heads, width, rows]
    Tensor gate;  // FP32 [2, value_heads, width, rows], ordered {g, beta}
};

/**
 * Bound non-owning all-layer ReplaySSM transition records.
 *
 * Layer and physical record row share the outer index `layer * record_capacity + row` in every
 * plane. The object owns no allocation and stores no active-row or valid-prefix metadata.
 */
struct GdnReplayRecords {
    Tensor conv;
    Tensor key;
    Tensor value;
    Tensor gate;
    GdnReplayRecordSpec spec;

    GdnReplayRecords() = default;
    GdnReplayRecords(DeviceSpan backing, const GdnReplayRecordLayout& layout);

    // Preconditions (validated with the spec first, then the indices): 0 <= layer_index <
    // spec.layers and 0 < rows <= spec.record_capacity. Throws std::invalid_argument for an
    // invalid spec and std::out_of_range for either index outside its range.

    [[nodiscard]] GdnReplayRecordLayer layer(std::int32_t layer_index, std::int32_t rows) const;

    /**
     * The same planes over the same memory at a narrower width. Each plane is reinterpreted as
     * its contiguous prefix with the width extent replaced, so a round that verifies fewer columns
     * than the layout was planned for records and folds a tighter layout. Writers and the fold
     * must use the same view for a round.
     *
     * Requires 2 <= width <= spec.width; throws std::out_of_range otherwise.
     */
    [[nodiscard]] GdnReplayRecords with_width(std::int32_t width) const;
};

} // namespace ninfer
