// The 26B's KV footprint per `--kv-dtype`, checked where the sizes are decided rather than where
// they are spent. A quantized sliding ring costs a code plane plus a scale plane against two bytes
// an element, so it must be strictly cheaper than BF16 at the same capacity, and the sliding part
// must not scale with the context capacity at all -- the ring is the window. Host-only, so it runs
// beside a busy device and catches a sizing regression before any allocation happens.
#include "models/gemma4/cache.h"
#include "models/gemma4/config.h"
#include "models/gemma4/forward.h"
#include "models/gemma4/program.h"

#include "artifact/schema.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::KvCacheStorage;
using ninfer::models::gemma4::KvCache;
using ninfer::models::gemma4::Program;
using ninfer::models::gemma4::TextConfig;

std::string read_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) { throw std::runtime_error("cannot read " + path); }
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

struct Storage {
    KvCacheStorage storage;
    const char* name;
};

constexpr Storage kStorages[] = {
    {KvCacheStorage::BFloat16, "bf16"},         {KvCacheStorage::Int8Group64, "int8"},
    {KvCacheStorage::Fp8E4M3Row256, "fp8"},     {KvCacheStorage::Nvfp4Group16, "nvfp4"},
    {KvCacheStorage::Fp8KeyNvfp4Value, "k8v4"},
};

// The program's pass width, so the scratch bound is the one the engine reserves.
constexpr std::int32_t kPass = Program::kPass;

} // namespace

int main() {
    int failures = 0;
    const std::string fixture =
        std::string(NINFER_SOURCE_DIR) + "/tests/fixtures/gemma4/text_config_26b.json";
    const auto json         = ninfer::artifact::Json::parse(read_file(fixture)).at("config");
    const TextConfig config = ninfer::models::gemma4::parse_text_config(json);

    const std::vector<std::int32_t> capacities{8192, 17408, 32768};
    for (const std::int32_t capacity : capacities) {
        std::cout << "capacity " << capacity << "\n";
        std::size_t bf16_cache = 0, bf16_total = 0, bf16_arena = 0;
        for (const Storage& entry : kStorages) {
            const std::size_t cache = KvCache::device_bytes(config, capacity, 0, entry.storage);
            const std::size_t arena =
                ninfer::models::gemma4::layer_workspace_bytes(config, kPass, entry.storage, 0);
            const std::size_t total =
                Program::device_bytes(config, capacity, 1, 0, nullptr, entry.storage);
            if (entry.storage == KvCacheStorage::BFloat16) {
                bf16_cache = cache;
                bf16_arena = arena;
                bf16_total = total;
            }
            std::printf("  %-5s cache %10zu B  pass scratch %10zu B  program %12zu B\n", entry.name,
                        cache, arena, total);
            if (entry.storage != KvCacheStorage::BFloat16 &&
                (cache >= bf16_cache || total >= bf16_total)) {
                std::cerr << "  " << entry.name
                          << " must cost less than bf16 at the same capacity\n";
                ++failures;
            }
        }

        // The sliding ring is the window plus the slack, so a wider capacity may only add the five
        // global layers' rows. Every storage must add the same amount: a quantized storage that
        // scaled with capacity would add more.
        const std::int32_t wider = 32768;
        const std::size_t global_rows =
            static_cast<std::size_t>(wider - capacity) * config.global_layers *
                (config.global.shared.head_dim + 2 * config.global.rope_angles) *
                config.global.shared.num_key_value_heads * sizeof(std::uint16_t) +
            static_cast<std::size_t>(wider - capacity) * config.global_layers *
                sizeof(std::int32_t);
        for (const Storage& entry : kStorages) {
            const std::size_t grown = KvCache::device_bytes(config, wider, 0, entry.storage) -
                                      KvCache::device_bytes(config, capacity, 0, entry.storage);
            if (grown != global_rows) {
                std::cerr << "  " << entry.name << " grew by " << grown << " B over a capacity of "
                          << capacity << " where only the global layers' " << global_rows
                          << " B may grow\n";
                ++failures;
            }
        }
        // The pass scratch is a per-pass bound: a quantized storage adds the BF16 ring each layer
        // decodes, and nothing about either scales with the capacity.
        (void)bf16_arena;
    }

    // The scratch a quantized storage adds is the two decoded rings, no more.
    const std::size_t bf16_scratch =
        ninfer::models::gemma4::layer_workspace_bytes(config, kPass, KvCacheStorage::BFloat16, 0);
    const std::size_t rings =
        2 * static_cast<std::size_t>(config.sliding.head_dim) * config.sliding.num_key_value_heads *
        static_cast<std::size_t>(config.sliding_window) * sizeof(std::uint16_t);
    for (const Storage& entry : kStorages) {
        const std::size_t scratch =
            ninfer::models::gemma4::layer_workspace_bytes(config, kPass, entry.storage, 0);
        const std::size_t added    = scratch - bf16_scratch;
        const std::size_t expected = entry.storage == KvCacheStorage::BFloat16 ? 0 : rings;
        if (added < expected || added > expected + 512) {
            std::cerr << entry.name << " adds " << added << " B of pass scratch where the decoded "
                      << "rings need " << expected << " B\n";
            ++failures;
        }
    }

    std::cout << (failures ? "FAIL" : "OK") << " gemma4_kv_quant_footprint\n";
    return failures ? 1 : 0;
}
