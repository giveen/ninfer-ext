#include "models/qwen3_5/program/planning/startup.h"
#include "runtime/engine/model_instance.h"

#include <iostream>
#include <optional>
#include <stdexcept>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

// Oracle for the Host RAM budget split. Every expectation below is computed by hand from the
// resolver's contract: one StateImage per retained position (2 + A per private owner plus one per
// shared entry), a state footprint capped at half the budget, extra anchors bought with the
// headroom under that cap at one image per private owner each, and a usefulness ceiling of
// pages(capacity) / pages(buyback tokens) - 2 where one image buys back
// state_image_bytes / host_kv_group_bytes page groups of re-prefill.
//
// The unit costs are round numbers on purpose: one image is 1 MiB and one Main page group is
// 256 KiB, so the image is exactly four page groups, buyback tokens are 4 * 64 = 256, and one page
// group is 4 logical pages. Nothing here mirrors the implementation's expression; the values are
// the arithmetic a reader can repeat.
constexpr std::uint64_t kImageBytes = 1ULL << 20; // 1 MiB per Host StateImage.
constexpr std::uint64_t kGroupBytes = 1ULL << 18; // 256 KiB per Main Host KV page group.
constexpr std::uint64_t kMiB        = 1ULL << 20;

ninfer::ContextCacheOptions budgeted(std::size_t budget_mib, std::uint32_t private_capacity,
                                     std::uint32_t shared_capacity, std::uint32_t configured) {
    ninfer::ContextCacheOptions cache;
    cache.host_cache_budget_bytes           = static_cast<std::size_t>(budget_mib * kMiB);
    cache.max_private_continuations         = private_capacity;
    cache.max_shared_prefixes               = shared_capacity;
    cache.max_long_anchors_per_continuation = configured;
    return cache;
}

void resolve(ninfer::ContextCacheOptions& cache, std::uint32_t capacity) {
    ninfer::models::qwen3_5::detail::resolve_host_cache_budget(
        cache, *cache.max_private_continuations, *cache.max_shared_prefixes, capacity, kImageBytes,
        kGroupBytes);
}

int check_split(const ninfer::ContextCacheOptions& cache, std::uint32_t anchors,
                std::uint32_t state_slots, std::uint64_t host_kv_mib, const char* message) {
    const std::uint64_t host_kv = cache.host_kv_capacity_bytes;
    if (*cache.max_long_anchors_per_continuation == anchors &&
        cache.host_state_slots == state_slots && host_kv == host_kv_mib * kMiB) {
        return 0;
    }
    std::cerr << message << ": anchors " << *cache.max_long_anchors_per_continuation << " (want "
              << anchors << "), state slots " << cache.host_state_slots << " (want " << state_slots
              << "), host KV " << host_kv << " B (want " << host_kv_mib << " MiB)\n";
    return 1;
}

} // namespace

int main() {
    using ninfer::EngineOptions;
    using ninfer::kMaximumPreparedPromptCacheCandidatesPerRequest;
    using ninfer::runtime::normalize_engine_options;

    int failures = 0;

    // A single request can produce up to kMaximumPreparedPromptCacheCandidatesPerRequest distinct
    // shared-prefix candidates (frontend.cpp's opportunities.reserve(7U): four explicit markers
    // plus the engine's tool/leading-instruction/full-prompt automatic candidates). The default
    // Engine-wide shared catalog must be able to hold at least one request's own candidates even
    // at the smallest concurrency, or ordinary DefaultAutomatic-evidence traffic starts losing
    // cache hits to its own prior turns as soon as the catalog fills.
    for (const std::uint32_t concurrency : {1U, 2U, 8U}) {
        EngineOptions options;
        options.max_concurrency = concurrency;
        const EngineOptions normalized = normalize_engine_options(options);
        const std::uint32_t default_shared_prefixes = *normalized.context_cache.max_shared_prefixes;
        failures += check(
            default_shared_prefixes >=
                static_cast<std::uint32_t>(kMaximumPreparedPromptCacheCandidatesPerRequest),
            "default shared-prefix catalog capacity is smaller than one request's own candidate ceiling");
        failures += check(default_shared_prefixes >= concurrency,
                          "default shared-prefix catalog capacity did not cover active concurrency");
    }

    // An explicit override is still respected verbatim, including a deliberately small value.
    {
        EngineOptions options;
        options.max_concurrency               = 1;
        options.context_cache.max_shared_prefixes = 1;
        const EngineOptions normalized = normalize_engine_options(options);
        failures += check(*normalized.context_cache.max_shared_prefixes == 1,
                          "explicit max_shared_prefixes override was not preserved");
    }

    // A disabled context cache still normalizes to a root-only zero capacity.
    {
        EngineOptions options;
        options.max_concurrency        = 1;
        options.context_cache.enabled  = false;
        const EngineOptions normalized = normalize_engine_options(options);
        failures += check(*normalized.context_cache.max_shared_prefixes == 0,
                          "disabled context cache did not normalize shared-prefix capacity to zero");
    }

    // A comfortable budget buys anchors. 64 MiB halves to a 32 MiB state cap; the mandatory
    // 2 + 4 per owner plus 7 shared is 19 images = 19 MiB, leaving 13 images of headroom = 6 extra
    // anchors per owner, so A = 4 + 6 = 10 and the pool holds (2 + 10) * 2 + 7 = 31 images. The
    // usefulness ceiling at 32768 tokens is 512 pages / 4 pages per 256 tokens - 2 = 126, far
    // above the budget's 10, so the budget binds.
    {
        ninfer::ContextCacheOptions cache = budgeted(64, 2, 7, 4);
        resolve(cache, 32768);
        failures += check_split(cache, 10, 31, 33, "comfortable budget did not buy anchors");
    }

    // The half-budget cap is respected exactly when no shared entry competes with the per-owner
    // images: 64 MiB / 2 is exactly (2 + 14) * 2 images of 1 MiB.
    {
        ninfer::ContextCacheOptions cache = budgeted(64, 2, 0, 4);
        resolve(cache, 32768);
        failures += check_split(cache, 14, 32, 32,
                                "budget-funded anchors did not fill the state cap exactly");
    }

    // A budget that exactly covers the mandatory inventory buys nothing extra and still starts.
    {
        ninfer::ContextCacheOptions cache = budgeted(38, 2, 7, 4);
        resolve(cache, 32768);
        failures += check_split(cache, 4, 19, 19,
                                "boundary budget did not keep the configured anchor count");
    }

    // A budget below the mandatory inventory falls back to the configured count and rejects, so
    // the caller reports the real shortfall instead of an inventory the budget never funded.
    {
        bool rejected                     = false;
        ninfer::ContextCacheOptions cache = budgeted(37, 2, 7, 4);
        try {
            resolve(cache, 32768);
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "an undersized budget did not reject the checkpoint inventory");
    }

    // An explicitly disabled anchor count stays disabled rather than being grown into a budget it
    // cannot afford: the default 4 needs 19 MiB of the 12 MiB half-budget, while the 11 images of
    // endpoint, rewrite and shared inventory fit it.
    {
        ninfer::ContextCacheOptions cache = budgeted(24, 2, 7, 0);
        resolve(cache, 32768);
        failures += check_split(cache, 0, 11, 13,
                                "an explicitly disabled anchor count was grown past the budget");
    }

    // The payoff ceiling binds before the budget: at 4096 tokens the logical space holds
    // 64 / 4 - 2 = 14 useful anchors, so a 256 MiB budget stops there and the rest stays Host KV.
    {
        ninfer::ContextCacheOptions cache = budgeted(256, 2, 7, 4);
        resolve(cache, 4096);
        failures += check_split(cache, 14, 39, 217,
                                "the anchor payoff ceiling did not bound the budget");
    }

    // The ceiling never lowers a count the caller configured, even one above it.
    {
        ninfer::ContextCacheOptions cache = budgeted(256, 2, 7, 20);
        resolve(cache, 4096);
        failures += check_split(cache, 20, 51, 205,
                                "the anchor payoff ceiling lowered a configured count");
    }

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
