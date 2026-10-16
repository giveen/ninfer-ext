#pragma once

// The attention K/V state of up to `lanes` Nemotron-H sequences, and the attention route over it.
//
// The family's attention is plain causal grouped-query attention with no window, no rotation, no
// sinks and no bias (forward.h): positions are absolute from the sequence's own length, the scale
// is `1/sqrt(head_dim)` passed explicitly rather than derived, and a key is visible exactly when it
// is at or before the query. That is what decides this state's shape:
//
//  * One buffer per lane per attention layer, dense and unbounded: keys and values are contiguous
//    BF16 `[head_dim, kv_heads, capacity, 1]`, exactly the operand `ops::sliding_causal_attention`
//    declares, so the route hands the state to the Op without a copy, a transpose or a repacking.
//    Nothing aliases between lanes, and a lane's slots are its own.
//  * Causality is by position, not by a window: the route passes `window == 0`, the Op's unbounded
//    causal route, because a finite window is not a substitute for this family -- its context is
//    262144 and its mask is causal -- and the Op's finite span is 4096.
//  * A slot no token has written holds `kUnwrittenPosition`, a position far past any real one, so
//    the Op's `position_k <= high` test excludes it. That is what lets the route hand the Op the
//    lane's whole capacity and attend over it without knowing how much of it is filled (the Op
//    skips a key tile no query row can see), which is also why the state needs no page table and no
//    block table: the positions are the only index there is.
//  * The step's own K/V is appended *before* its query runs, so a token's own key and value
//    participate in its own output -- the reference's order. `attend` refuses a layer whose append
//    has not happened, so the order is enforced rather than documented.
//
// The layout arithmetic, for `A` attention layers, `L` lanes and `C` tokens of capacity, with
// `D = head_dim` and `Hkv = num_key_value_heads`:
//
//   keys      = A * L * C * Hkv * D * 2 bytes   (BF16)
//   values    = A * L * C * Hkv * D * 2 bytes   (BF16)
//   positions = L * C * 4 bytes                 (I32, one row per lane: the sequence's
//                                                positions are the same for every layer)
//
// so one lane of the 74B checkpoint's geometry (A = 8, D = 128, Hkv = 2) holds 2 * 8 * C * 2 * 128
// * 2 = 8192 * C bytes: 8 KB per token, 2 GiB at the family's 262144-token context.
//
// What this state does not hold, deliberately: a quantized K/V (the Op's operands are BF16, so a
// codec would have to be decoded into BF16 scratch per call, which is a different route and is not
// wired) and a carried head width (a config whose `head_dim` is not one of the Op's 32, 64, 128 or
// 256 needs the zero-carry the composition implements for its passes, `attend` refuses that width
// by name rather than padding the state). Neither is this family's geometry: its head is 128.

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "models/nemotron_h/config.h"
#include "ninfer/ops/attention_geometry.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime_api.h>

namespace ninfer::models::nemotron_h {

// A slot no token has written. Far past any real position, so `position_k <= high` -- the Op's own
// causal bound -- excludes it, and its bytes are never read by a query.
inline constexpr std::int32_t kUnwrittenPosition = 1 << 30;

// The `window` that means unbounded causal to `ops::sliding_causal_attention`: this model has no
// window of its own, so every key at or before the query is visible.
inline constexpr std::uint32_t kUnboundedWindow = 0;

// One lane's K/V state for one attention layer, as the Op takes it. `filled` is the number of
// tokens written (slots `[0, filled)` hold their own position); the rest hold `kUnwrittenPosition`
// and are invisible.
struct AttentionStateView {
    Tensor keys;      // contiguous BF16 [head_dim, kv_heads, capacity, 1]
    Tensor values;    // contiguous BF16 [head_dim, kv_heads, capacity, 1]
    Tensor positions; // contiguous I32 [capacity, 1]
    std::int32_t filled = 0;
};

class AttentionState {
public:
    // `capacity` is the tokens one lane may hold, `lanes` the sequences that may hold state at
    // once. Every allocation is made here: a lane never discovers half way through a request that
    // its state does not fit. The positions are not initialized until `reset_all`, which the owner
    // calls once with the stream it will use.
    AttentionState(const AttentionConfig& attention, std::uint32_t layers, std::int32_t capacity,
                   std::int32_t lanes);
    ~AttentionState();
    AttentionState(const AttentionState&)            = delete;
    AttentionState& operator=(const AttentionState&) = delete;
    AttentionState(AttentionState&&)                 = delete;
    AttentionState& operator=(AttentionState&&)      = delete;

    // The three classes above for these extents, so a caller can size a lane before constructing
    // one. Pure: it computes the same bytes the constructor allocates.
    [[nodiscard]] static std::size_t device_bytes(const AttentionConfig& attention,
                                                  std::uint32_t layers, std::int32_t capacity,
                                                  std::int32_t lanes);

    [[nodiscard]] std::uint32_t layers() const noexcept { return layers_; }

    [[nodiscard]] std::int32_t lanes() const noexcept {
        return static_cast<std::int32_t>(lanes_state_.size());
    }

    [[nodiscard]] std::int32_t capacity() const noexcept { return capacity_; }

    [[nodiscard]] std::size_t allocated_bytes() const noexcept { return state_.bytes; }

    // The state storage of the three classes, as allocated (their sum plus alignment padding).
    [[nodiscard]] std::size_t keys_bytes() const noexcept { return keys_.bytes(); }

    [[nodiscard]] std::size_t values_bytes() const noexcept { return values_.bytes(); }

    [[nodiscard]] std::size_t positions_bytes() const noexcept { return positions_.bytes(); }

    // Device bytes one attention call needs for `tokens` query tokens over a lane's whole state:
    // the Op's own key-split workspace. A caller that passes less gets the Op's one-pass route,
    // which computes the same result.
    [[nodiscard]] std::size_t workspace_bytes(std::int32_t tokens) const;

    // Every lane's positions become unwritten and every cursor returns to zero, keeping the
    // allocation. Every lane is reset before its first request: an unwritten state is invisible,
    // but an *initialized* one is what makes it so.
    void reset_all(cudaStream_t stream);

    // One lane's state: every position unwritten, its cursor at zero, no pending step. Refuses a
    // lane whose step has not been advanced.
    void reset(std::int32_t lane, cudaStream_t stream);

    // Tokens the lane has appended and advanced past: its sequence length.
    [[nodiscard]] std::int32_t filled(std::int32_t lane) const;

    // Tokens appended in the step in progress, which is the query count its attention runs with.
    [[nodiscard]] std::int32_t pending(std::int32_t lane) const;

    // The lane's state for one attention layer.
    [[nodiscard]] AttentionStateView layer(std::int32_t lane, std::uint32_t layer) const;

    /**
     * Appends `tokens` tokens of one layer's K and V at the lane's cursor, and writes their
     * positions. `keys` and `values` are contiguous BF16 `[head_dim, kv_heads, tokens, 1]` -- the
     * projections' own output for this layer, in the Op's own element order, so the append is one
     * contiguous copy. Every attention layer of a step must append the same token count before the
     * step is advanced, and a layer may append once per step.
     *
     * The caller orders `stream` against prior state accesses; the state enqueues nothing else.
     */
    void append(std::int32_t lane, std::uint32_t layer, const Tensor& keys, const Tensor& values,
                std::int32_t tokens, cudaStream_t stream);

    // Ends the step: the cursor moves past every layer's appended tokens. Refuses unless every
    // attention layer appended the step's count, so a step cannot half-advance.
    void advance(std::int32_t lane);

    /**
     * One layer's attention over the lane's whole state, for the step's `pending` tokens: exactly
     * the causal attention this model has, with the scale it has and no window.
     *
     * `query` and `out` are contiguous BF16 `[head_dim, query_heads, pending, 1]`. The layer's K/V
     * must have been appended first (`append`), so the step's own key and value are in its own
     * output; the positions are the lane's own, absolute from the sequence's length.
     *
     * `workspace` may be empty, or `workspace_bytes(pending)` bytes for the Op's key-split route.
     */
    void attend(std::int32_t lane, std::uint32_t layer, const Tensor& query, Tensor& out,
                const Tensor& workspace, DeviceExecutionView execution) const;

private:
    struct Lane {
        std::int32_t filled  = 0;
        std::int32_t pending = 0;
        // One flag per attention layer: whether this step's append has happened. `attend` reads it,
        // which is what makes append-before-query an enforced order rather than a convention.
        std::vector<bool> appended;
    };

    [[nodiscard]] Lane& lane_at(std::int32_t lane);
    [[nodiscard]] const Lane& lane_at(std::int32_t lane) const;
    [[nodiscard]] std::size_t layer_index(std::int32_t lane, std::uint32_t layer) const;
    // Elements of one (layer, lane) K or V plane: head_dim * kv_heads * capacity.
    [[nodiscard]] std::size_t plane_elements() const;
    // The Op's geometry for this config, validated against the widths its launcher admits.
    [[nodiscard]] ops::AttentionHeadGeometry route_geometry() const;

    DeviceBuffer state_;
    Tensor keys_;      // BF16 [head_dim, kv_heads, capacity, layers * lanes]
    Tensor values_;    // BF16 [head_dim, kv_heads, capacity, layers * lanes]
    Tensor positions_; // I32 [capacity, lanes]
    std::vector<Lane> lanes_state_;
    // Host staging for one append's positions, reused so an append allocates nothing.
    std::vector<std::int32_t> positions_host_;
    std::int32_t head_dim_    = 0;
    std::int32_t kv_heads_    = 0;
    std::int32_t query_heads_ = 0;
    std::int32_t capacity_    = 0;
    std::uint32_t layers_     = 0;
};

} // namespace ninfer::models::nemotron_h
