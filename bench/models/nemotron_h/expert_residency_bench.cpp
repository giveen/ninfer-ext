// Nemotron-H routed-expert residency: the same prefill pass over the same real artifact, with the
// expert banks read in place over UVA and with the cache-resident ones uploaded and the rest staged
// into device buffers.
//
// This is the measurement the split has to justify itself with, so it loads the artifact exactly as
// the Engine's instance does -- same residency request, same `plan_load`, same `materialize_model`,
// same `forward_model` -- and reports what the two configurations actually cost:
//
//   * the split itself: resident experts and bytes, the pinned host remainder, and the pool's own
//     device bytes;
//   * per-pass wall time at a few widths, `--tokens` wide included, with a warm-up pass excluded;
//   * the host-link traffic: the bytes a staged pass copies (from the pool's own accounting, not
//     from the plan) and the bytes the in-place route streams through its kernel loads.
//
// The in-place figure is an *estimate of the same shape* from the same run: the device-side bytes
// the in-place pass reads are `top_k * slot_bytes` per layer per token, which no counter observes
// because those reads are kernel loads rather than copies. It is reported as an estimate and named
// as one.
//
// The prefill width is the engine's own chunk, and the widths are the ones a request actually
// arrives in. Nothing here is a tok/s claim: generation does not exist for this family yet.
#include "core/arena.h"
#include "core/device.h"
#include "models/nemotron_h/experts.h"
#include "models/nemotron_h/forward.h"
#include "models/nemotron_h/load.h"
#include "models/nemotron_h/program.h"
#include "ninfer/types.h"
#include "product/offload_options.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace nemotron = ninfer::models::nemotron_h;

void usage(const char* argv0) {
    std::cerr << "usage: " << argv0 << " <model.ninfer> [--tokens N,N,...] [--cache auto|0|MiB]\n"
              << "       [--lanes N] [--context N] [--chunk N] [--repeats N]\n";
}

std::vector<std::int32_t> parse_tokens(std::string_view text) {
    std::vector<std::int32_t> widths;
    std::size_t begin = 0;
    while (begin <= text.size()) {
        const std::size_t comma = text.find(',', begin);
        const std::size_t end   = comma == std::string_view::npos ? text.size() : comma;
        if (end > begin) {
            widths.push_back(std::stoi(std::string(text.substr(begin, end - begin))));
        }
        if (comma == std::string_view::npos) { break; }
        begin = comma + 1;
    }
    if (widths.empty()) { throw std::runtime_error("no token counts"); }
    return widths;
}

double mib(std::size_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0); }

double gib(std::size_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0); }

} // namespace

using namespace ninfer;

int main(int argc, char** argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 2;
    }
    const std::string artifact = argv[1];
    std::vector<std::int32_t> widths{512, 1024, 4096};
    std::string cache     = "auto";
    std::uint32_t lanes   = 1;
    std::uint32_t context = 2048;
    std::uint32_t chunk   = 4096;
    std::int32_t repeats  = 2;
    for (int index = 2; index < argc; ++index) {
        const std::string_view arg = argv[index];
        const auto next            = [&](const char* what) -> std::string {
            if (index + 1 >= argc) { throw std::runtime_error(std::string("missing ") + what); }
            return argv[++index];
        };
        if (arg == "--tokens") {
            widths = parse_tokens(next("--tokens"));
        } else if (arg == "--cache") {
            cache = next("--cache");
        } else if (arg == "--lanes") {
            lanes = static_cast<std::uint32_t>(std::stoul(next("--lanes")));
        } else if (arg == "--context") {
            context = static_cast<std::uint32_t>(std::stoul(next("--context")));
        } else if (arg == "--chunk") {
            chunk = static_cast<std::uint32_t>(std::stoul(next("--chunk")));
        } else if (arg == "--repeats") {
            repeats = static_cast<std::int32_t>(std::stoi(next("--repeats")));
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    DeviceContext device(0);
    DeviceExecutionView execution = device.execution_view();
    artifact::Reader reader(artifact);
    const models::nemotron_h::Config artifact_config = nemotron::parse_config(reader.directory());
    const nemotron::TextConfig& config               = artifact_config.text;

    EngineOptions options;
    options.max_concurrency = lanes;
    options.max_context     = context;
    options.prefill_chunk   = chunk;
    options.kv_cache        = KvCacheStorage::BFloat16;
    options.kv_capacity     = KvCapacityPolicy::explicit_capacity(context);
    options.expert_cache    = product::parse_expert_cache(cache);

    // The same reservation the Engine's instance resolves, so the two configurations differ only in
    // the split: the family's state and one pass's workspace, the pass's own output buffer, the
    // runtime's reservation and its headroom all stay out of the cache's way.
    nemotron::ExpertResidencyRequest request;
    request.automatic      = options.expert_cache.mode == ExpertCacheMode::Automatic;
    request.explicit_bytes = options.expert_cache.explicit_bytes;
    request.reserve_bytes  = nemotron::Program::device_bytes(config, options) +
                             nemotron::pass_output_bytes(config, static_cast<std::int32_t>(chunk)) +
                             nemotron::kRuntimeReservationBytes;
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));

    nemotron::LoadPlan plan = nemotron::plan_load(reader, models::load_options(options), request);
    // A *copy*: the residency is the plan's own and `materialize_model` consumes the plan, so a
    // reference here would dangle into the freed Impl for every read below -- which is exactly how
    // a split's own numbers come out as garbage.
    const nemotron::ExpertResidencyPlan residency = plan.expert_residency();
    std::unique_ptr<nemotron::Model> model = nemotron::materialize_model(std::move(plan), device);
    device.synchronize();

    std::size_t after_words = 0;
    CUDA_CHECK(cudaMemGetInfo(&after_words, &total_bytes));

    std::optional<nemotron::RoutedExpertPool> pool;
    if (residency.staging_bytes != 0) { pool.emplace(*model, nemotron::expert_top_k(config)); }
    std::size_t after_pool = 0;
    CUDA_CHECK(cudaMemGetInfo(&after_pool, &total_bytes));

    const std::vector<ops::ExpertBankShape> banks = nemotron::expert_bank_geometry(config);
    const std::size_t device_bank_bytes           = static_cast<std::size_t>(
        std::accumulate(banks.begin(), banks.end(), std::int64_t{0},
                        [](std::int64_t sum, const ops::ExpertBankShape& bank) {
                            return sum + nemotron::expert_bank_bytes(bank);
                        }));

    std::cout << "artifact            " << artifact << "\n"
              << "lanes/context/chunk " << lanes << "/" << context << "/" << chunk << "\n"
              << "expert cache         " << cache << "\n"
              << "card free before     " << mib(free_bytes) << " MiB\n"
              << "reservation          " << mib(request.reserve_bytes) << " MiB\n"
              << "expert budget        " << mib(residency.budget_bytes) << " MiB"
              << (residency.clamped
                      ? "  (requested " + std::to_string(mib(residency.requested_bytes)) +
                            " MiB, clamped to the card)"
                      : "")
              << "\n"
              << "staging              " << mib(residency.staging_bytes) << " MiB\n"
              << "pool device bytes    " << mib(pool ? pool->allocated_bytes() : 0) << " MiB\n"
              << "resident banks       "
              << std::count(residency.resident.begin(), residency.resident.end(), true) << " of "
              << residency.resident.size() << "\n"
              << "resident slots       " << residency.resident_experts << " of "
              << residency.pool_experts << "  (" << (residency.resident_fraction() * 100.0)
              << "%)\n"
              << "resident bytes       " << gib(residency.resident_bytes) << " GiB\n"
              << "pinned host banks    " << gib(residency.pinned_bytes(device_bank_bytes))
              << " GiB\n"
              << "storage stats: h2d " << gib(model->storage_stats().h2d_bytes) << " GiB, pinned "
              << gib(model->storage_stats().pinned_host_bytes) << " GiB\n"
              << "free after model     " << mib(after_words) << " MiB, after pool "
              << mib(after_pool) << " MiB\n"
              << std::flush;

    // The split's two descriptions of the same bytes have to agree, and they count exactly one
    // thing differently: this materializer counts whole stored objects, and each of a bank's stored
    // objects carries its own weight divisor word *inside* its bytes, which the bank's plane size
    // -- and so the residency's pinned remainder -- does not. An NVFP4 bank stores one object per
    // pair of planes (its codes and its scales), so what is left is one word per non-resident
    // object.
    const std::size_t pinned_banks = residency.pinned_bytes(device_bank_bytes);
    std::size_t pinned_objects     = 0;
    for (std::size_t layer = 0; layer < banks.size(); ++layer) {
        if (!residency.resident[layer]) { pinned_objects += banks[layer].layout.planes / 2; }
    }
    const std::size_t pinned_stats = model->storage_stats().pinned_host_bytes;
    if (pinned_stats != pinned_banks + sizeof(float) * pinned_objects) {
        std::cerr << "residency implies " << pinned_banks << " pinned host bytes plus "
                  << pinned_objects << " divisor words, where the storage stats report "
                  << pinned_stats << "\n";
        return 1;
    }

    const std::size_t arena_bytes = nemotron::model_workspace_bytes(config, chunk);
    DeviceArena arena(arena_bytes);
    std::vector<std::int32_t> ids(
        static_cast<std::size_t>(*std::max_element(widths.begin(), widths.end())), 0);
    for (std::size_t index = 0; index < ids.size(); ++index) {
        ids[index] = static_cast<std::int32_t>(index % config.vocab_size);
    }

    std::cout << "\n  tokens   pass_ms   staged_MiB  staged_GB|s  inplace_GB_est\n";
    for (const std::int32_t tokens : widths) {
        Tensor token_ids(ids.data(), DType::I32, {tokens});
        DeviceBuffer logits(nemotron::pass_output_bytes(config, tokens));
        Tensor logits_view(logits.p, DType::BF16,
                           {static_cast<std::int32_t>(config.vocab_size), tokens});
        // Warm-up: the first pass pays the in-place reads' page faults and the staged route's first
        // reclaim, neither of which is a steady-state cost.
        nemotron::forward_model(*model, token_ids, arena, logits_view, execution, nullptr,
                                nemotron::HeadColumns::Every, nullptr, pool ? &*pool : nullptr);
        device.synchronize();
        // Taken *after* the warm-up: the pool's count is cumulative, so a snapshot before it would
        // charge the warm-up's traffic to the report. What follows is the timed passes' own bytes.
        const std::size_t before        = pool ? pool->staged_bytes() : 0;
        const std::size_t before_layers = pool ? pool->staged_layers() : 0;
        const std::int32_t passes       = std::max(repeats, 1);
        double best                     = 0.0;
        for (std::int32_t run = 0; run < passes; ++run) {
            const auto start = std::chrono::steady_clock::now();
            nemotron::forward_model(*model, token_ids, arena, logits_view, execution, nullptr,
                                    nemotron::HeadColumns::Every, nullptr, pool ? &*pool : nullptr);
            device.synchronize();
            const double seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            best = best == 0.0 ? seconds : std::min(best, seconds);
        }
        // One pass's own traffic: the column is quoted against `best`, which is one pass.
        const std::size_t staged =
            pool ? (pool->staged_bytes() - before) / static_cast<std::size_t>(passes) : 0;
        // The column only means anything if the pool really copied the non-resident banks: a pass
        // stages every one of them whole, exactly once, so one pass's traffic is at least their
        // planes and at most those plus the divisor tails the layout reserves for them. A counter
        // nothing writes reads zero; one that counts a prefetch and its own layer separately reads
        // twice the planes; both fail here.
        if (pool) {
            const std::size_t planes = device_bank_bytes - residency.resident_bytes;
            std::size_t tails = 0, non_resident = 0;
            for (std::size_t layer = 0; layer < banks.size(); ++layer) {
                if (!residency.resident[layer]) {
                    tails += static_cast<std::size_t>(banks[layer].layout.tail_bytes);
                    ++non_resident;
                }
            }
            const std::size_t layers_staged =
                (pool->staged_layers() - before_layers) / static_cast<std::size_t>(passes);
            if (staged < planes || staged > planes + tails || layers_staged != non_resident) {
                std::cerr << tokens << " tokens: the pool reports " << staged
                          << " staged bytes and " << layers_staged << " layers in a pass where the "
                          << non_resident << " non-resident banks are " << planes
                          << " bytes of planes (at most " << planes + tails
                          << " with their divisor tails)\n";
                return 1;
            }
        }
        // The in-place route's device-side traffic: every selected expert's matrix pair, per layer,
        // per token, read by the kernel's own loads.
        // The in-place route's traffic: `top_k` selected experts of each layer, each of them
        // `1152 * intermediate` bytes of stored NVFP4, re-read for every token of the pass.
        std::uint64_t inplace = 0;
        for (std::uint32_t block = 0; block < config.num_hidden_layers(); ++block) {
            if (config.block_kind(block) != nemotron::BlockKind::Moe) { continue; }
            const nemotron::MoeBankConfig& bank = config.moe_bank(block);
            inplace += static_cast<std::uint64_t>(bank.experts_per_token) * 1152ULL *
                       bank.intermediate_size * static_cast<std::uint64_t>(tokens);
        }
        std::printf("  %6d  %8.1f  %11.1f  %11.2f  %14.2f\n", tokens, best * 1000.0, mib(staged),
                    staged / 1e9 / best, static_cast<double>(inplace) / 1e9 / best);
    }
    return 0;
}
