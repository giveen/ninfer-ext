#include "models/qwen3_5/program/speculative/suffix_drafter.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace ninfer::models::qwen3_5 {
namespace {

// Murmur-style finalizer (Strata's `mix`): spreads the trigram over the whole key so the low bits
// used for the table index are independent of token-id structure.
std::uint64_t mix(std::uint64_t value) {
    value ^= value >> 33;
    value *= 0xff51afd7ed558ccdULL;
    value ^= value >> 33;
    value *= 0xc4ceb9fe1a85ec53ULL;
    value ^= value >> 33;
    return value;
}

} // namespace

SuffixDrafter::SuffixDrafter(std::uint32_t capacity_tokens) {
    if (capacity_tokens == 0) { throw std::invalid_argument("suffix drafter needs a nonzero capacity"); }
    const std::uint32_t indexed = std::min(capacity_tokens, kMaxIndexedTokens);
    std::size_t entries = 1;
    // Load factor at or below 0.5 at the nominal capacity: linear probing stays short and a full
    // table is detected before every slot is claimed.
    while (entries < static_cast<std::size_t>(indexed) * 2U) { entries <<= 1U; }
    table_.assign(entries, Slot{});
    mask_ = static_cast<std::uint32_t>(entries - 1U);
    history_.reserve(indexed);
}

void SuffixDrafter::reset() {
    history_.clear();
    std::fill(table_.begin(), table_.end(), Slot{});
    indexed_ = 0;
    match_   = 0;
    full_    = false;
}

std::uint64_t SuffixDrafter::key_at(std::uint32_t end) const {
    const auto a = static_cast<std::uint32_t>(history_[end - 2U]);
    const auto b = static_cast<std::uint32_t>(history_[end - 1U]);
    const auto c = static_cast<std::uint32_t>(history_[end]);
    const std::uint64_t key =
        mix(static_cast<std::uint64_t>(a) * 0x9E3779B97F4A7C15ULL ^
            mix(static_cast<std::uint64_t>(b) + 0x632BE59BD9B4E019ULL) ^
            (static_cast<std::uint64_t>(c) << 1U));
    // Bit 0 of a slot key marks an empty slot, so a key is never zero.
    return key | 1ULL;
}

SuffixDrafter::Slot* SuffixDrafter::find_slot(std::uint64_t key, bool insert) {
    std::uint32_t index    = static_cast<std::uint32_t>(key) & mask_;
    const std::uint32_t probes_max = mask_ + 1U;
    for (std::uint32_t probes = 0; probes < probes_max; ++probes) {
        Slot& slot = table_[index];
        if (slot.key == key) { return &slot; }
        if (slot.key == 0) {
            if (!insert) { return nullptr; }
            slot.key = key;
            return &slot;
        }
        index = (index + 1U) & mask_;
    }
    return nullptr; // table full: the history outgrew its capacity
}

void SuffixDrafter::sync(std::span<const TokenId> tokens) {
    if (full_) { return; }
    // `tokens` is the whole sequence, so a span shorter than the indexed prefix means the caller
    // changed the sequence underneath us; restart rather than silently index the wrong tokens.
    if (tokens.size() < indexed_) { reset(); }
    for (std::size_t i = indexed_; i < tokens.size(); ++i) {
        if (indexed_ >= kMaxIndexedTokens) {
            full_ = true;
            return;
        }
        history_.push_back(tokens[i]);
        const std::uint32_t end = indexed_;
        ++indexed_;
        if (end < 2U) { continue; }
        Slot* slot = find_slot(key_at(end), true);
        if (slot == nullptr) {
            // The table is full. Keep the history (it is still the comparison source) but stop
            // inserting: an uninserted trigram simply proposes nothing.
            full_ = true;
            return;
        }
        for (std::uint32_t way = kWays - 1U; way > 0U; --way) {
            slot->position[way] = slot->position[way - 1U];
        }
        slot->position[0] = end;
        if (slot->count < kWays) { ++slot->count; }
    }
}

std::uint32_t SuffixDrafter::propose(std::uint32_t max_drafts, TokenId* out) {
    match_ = 0;
    const std::size_t size = history_.size();
    if (size < 4U || max_drafts == 0U) { return 0; }
    const std::uint32_t current = static_cast<std::uint32_t>(size - 1U);
    const Slot* slot            = find_slot(key_at(current), false);
    if (slot == nullptr) { return 0; }

    std::uint32_t best_end = 0;
    std::uint32_t best_len = 0;
    for (std::uint32_t way = 0; way < slot->count; ++way) {
        const std::uint32_t candidate = slot->position[way];
        if (candidate >= current) { continue; } // the current suffix itself
        std::uint32_t length = 0;
        while (length < kMaxMatch && length <= candidate &&
               history_[candidate - length] == history_[current - length]) {
            ++length;
        }
        // Positions are newest-first, so a strict comparison keeps the most recent winner.
        if (length > best_len) {
            best_len = length;
            best_end = candidate;
        }
    }
    if (best_len < kMinMatch) { return 0; }
    match_ = best_len;

    std::uint32_t count = 0;
    // The continuation may run into the current suffix (periodic text); reading history up to
    // `current` is valid and is exactly what a repeat implies.
    for (std::uint32_t position = best_end + 1U; position <= current && count < max_drafts;
         ++position) {
        out[count++] = history_[position];
    }
    return count;
}

} // namespace ninfer::models::qwen3_5
