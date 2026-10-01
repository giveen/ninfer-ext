#pragma once

// Host prompt-lookup (suffix) drafter for speculative decoding.
//
// It proposes the tokens that followed the longest earlier occurrence of the sequence's current
// suffix. On text that repeats itself - an edit returning the file it was given, a refactor
// quoting its input, a tool result echoed into the next call - the continuation is often exact, so
// one verify pass accepts several tokens. It needs no weights, no draft KV and no GPU work: a hash
// probe and a bounded backward comparison on the Host.
//
// The proposal is a set of candidate tokens whose draft distribution is one-hot at each proposed
// token. The target's rejection sampling therefore accepts them without changing what the model
// commits, exactly as it does for MTP drafts.
//
// This is a faithful port of Strata's `src/spec/suffix_drafter.cpp`: a trigram key -> up to `kWays`
// recent end positions in an open-addressed table, a backward extension to `max_match`, and the
// continuation of the winner, bounded by the current end of the history. The index is bounded: once
// the table is full, new keys are dropped rather than evicted, so a very long sequence loses some
// recall but never corrupts a proposal.

#include "ninfer/types.h"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_5 {

class SuffixDrafter {
public:
    // Recent end positions remembered per trigram. Four is Strata's measured value: more costs
    // memory and comparison work for candidates that almost never win.
    static constexpr std::uint32_t kWays = 4;
    // A shorter match is not worth a proposal: its continuation is noise.
    static constexpr std::uint32_t kMinMatch = 3;
    // Backward extension limit. Beyond this the match is already long enough that the win comes
    // from the match bucket, not from a longer comparison.
    static constexpr std::uint32_t kMaxMatch = 32;
    // Upper bound on indexed history, so the table (and its zeroed allocation) stays bounded.
    static constexpr std::uint32_t kMaxIndexedTokens = 1U << 20;

    // `capacity_tokens`: the sequence length the index is sized for. The trigram table is the next
    // power of two covering twice that value, capped at twice `kMaxIndexedTokens`.
    explicit SuffixDrafter(std::uint32_t capacity_tokens);

    SuffixDrafter(const SuffixDrafter&) = delete;
    SuffixDrafter& operator=(const SuffixDrafter&) = delete;

    // Drop every indexed token. Called when a sequence is (re)used by a new request, so a reused
    // continuation can never draft from another request's tokens.
    void reset();

    // Append every token of `tokens` after the already-indexed prefix. Idempotent for a prefix the
    // caller has already synced, which lets the round loop sync the whole ledger each round.
    void sync(std::span<const TokenId> tokens);

    // Write up to `max_drafts` proposed tokens to `out`; return the count (0 = no usable match).
    std::uint32_t propose(std::uint32_t max_drafts, TokenId* out);

    // Match length behind the last proposal (0 when none).
    [[nodiscard]] std::uint32_t match_length() const noexcept { return match_; }
    // Tokens indexed so far.
    [[nodiscard]] std::uint32_t indexed() const noexcept { return indexed_; }

private:
    struct Slot {
        std::uint64_t key = 0;               // trigram hash + 1 (0 marks an empty slot)
        std::array<std::uint32_t, kWays> position{};
        std::uint8_t count = 0;
    };

    [[nodiscard]] std::uint64_t key_at(std::uint32_t end) const;
    [[nodiscard]] Slot* find_slot(std::uint64_t key, bool insert);

    std::vector<TokenId> history_;
    std::vector<Slot> table_;
    std::uint32_t mask_    = 0;            // table_.size() - 1
    std::uint32_t indexed_ = 0;            // tokens already inserted into `table_`
    std::uint32_t match_   = 0;
    bool full_             = false;        // the table stopped accepting new keys
};

} // namespace ninfer::models::qwen3_5
