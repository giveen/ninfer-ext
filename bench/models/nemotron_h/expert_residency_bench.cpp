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
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace nemotron = ninfer::models::nemotron_h;

void usage(const char* argv0) {
    std::cerr << "usage: " << argv0 << " <model.ninfer> [--tokens N,N,...] [--cache auto|0|MiB]\n"
              << "       [--slots N] [--lanes N] [--context N] [--chunk N] [--repeats N]\n";
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

// A BF16 word as the float it represents: the format is the leading half of an FP32, which is what
// makes the two routes' outputs comparable word for word rather than by a tolerance.
float bf16_to_float(std::uint16_t word) {
    const std::uint32_t bits = static_cast<std::uint32_t>(word) << 16;
    float value              = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

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
    std::int32_t slots    = 0;
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
        } else if (arg == "--slots") {
            slots = std::stoi(next("--slots"));
            if (slots < 0) { throw std::runtime_error("--slots must not be negative"); }
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
    // runtime's reservation and its headroom all stay out of the cache's way. The pool is *inside*
    // the budget the residency plan spends, so it comes out of the reservation here rather than
    // being counted in both places.
    nemotron::ExpertResidencyRequest request;
    request.automatic      = options.expert_cache.mode == ExpertCacheMode::Automatic;
    request.explicit_bytes = options.expert_cache.explicit_bytes;
    request.cache_slots    = slots;
    request.reserve_bytes  = nemotron::Program::device_bytes(config, options) -
                             nemotron::Program::expert_pool_device_bytes(config, options) +
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
              << "slot cache           " << slots << " slots requested, " << residency.cache_slots
              << " granted"
              << (residency.slots_clamped ? "  (clamped to what the card affords)" : "") << "\n"
              << "pool slots           "
              << (pool ? static_cast<std::size_t>(pool->cache().slots) : 0) << " = two staged "
              << (pool ? static_cast<std::size_t>(2) * pool->geometry().max_experts() : 0)
              << "-expert banks + " << residency.cache_slots << " cached\n"
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
        // twice the planes; both fail here. The accounting is the *staged* route's, so a width
        // below the pool's own staged width -- which now gathers instead -- is reported by the
        // one-column section below rather than checked against a staged count it will not have.
        if (pool && pool->stages(tokens)) {
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

    // --- The narrow route against the in-place one, at one column --------------------------------
    //
    // A one-column call routes 9 experts out of a bank of 240 to 496 of them, so staging that bank
    // whole would copy about 27 times the bytes the call reads. The gather route copies only the
    // experts the call routed to, and this is the section that proves it: one column over the same
    // artifact with the banks split and the routed experts gathered into the pool's slots, against
    // one column with every bank pinned and read over UVA where it lies (`--expert-cache 0`). Both
    // read the same stored bytes -- W4A16 with no activation quantization on either route, one slot
    // per routed expert -- so the two passes must agree *word for word*. That is a stricter claim
    // than the in-place column above, which estimates that route's traffic: this compares the
    // outputs themselves.
    //
    // The two configurations are materialized one at a time: the split exists because 37.21 GB of
    // banks do not fit on the card at once, and holding both would need 54.65 GB of pinned host
    // memory resident simultaneously.
    //
    // What this is *not*: a claim that a decode step gets 35 times faster. At one column the pass
    // is ~95% one kernel -- the experts' own arithmetic, `nemotron_h_moe_experts_kernel`, one block
    // per token with the nine ranks serialized behind it -- so the staged route's 22 GB of copies
    // were already hidden behind that kernel and the gather's saving arrives as host-link traffic
    // rather than as time. The three times below are the measurement of exactly that, and they are
    // reported together precisely so the traffic claim cannot be read as a speedup.
    const std::int32_t vocabulary = static_cast<std::int32_t>(config.vocab_size);
    // One pass over a materialized configuration: its wall time, and -- when the caller asked for
    // them -- the logits it wrote, so two configurations can be compared on one input at the same
    // column count. `columns` defaults to one, which is what most of this section measures.
    const auto column_pass = [&](nemotron::Model& subject, nemotron::RoutedExpertPool* subject_pool,
                                 const std::int32_t* token, double& ms,
                                 std::vector<std::uint16_t>* logits_out,
                                 std::int32_t columns = 1) {
        // The ids are uploaded per pass, so a pointer into the host vector is all the pass needs.
        const Tensor token_ids(const_cast<std::int32_t*>(token), DType::I32, {columns});
        DeviceBuffer logits(nemotron::pass_output_bytes(config, columns));
        Tensor logits_view(logits.p, DType::BF16, {vocabulary, columns});
        const auto start = std::chrono::steady_clock::now();
        nemotron::forward_model(subject, token_ids, arena, logits_view, execution, nullptr,
                                nemotron::HeadColumns::Every, nullptr, subject_pool);
        device.synchronize();
        ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                 .count();
        if (logits_out != nullptr) {
            logits_out->resize(static_cast<std::size_t>(vocabulary) *
                               static_cast<std::size_t>(columns));
            CUDA_CHECK(cudaMemcpy(logits_out->data(), logits.p,
                                  logits_out->size() * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost));
        }
    };

    int failures = 0;
    if (pool) {
        const std::int32_t top_k = nemotron::expert_top_k(config);
        // The bytes one column's gather asks for: `top_k` experts of every non-resident bank at
        // that layer's own expert size. The comparison pass below runs on a cache the staged passes
        // of the table above left empty, so the cache's own miss counter is what confirms every one
        // of those assignments was in fact fetched rather than counted.
        std::uint64_t routed_bytes = 0;
        std::int32_t assignments   = 0;
        for (std::size_t layer = 0; layer < banks.size(); ++layer) {
            if (residency.resident[layer]) { continue; }
            routed_bytes += static_cast<std::uint64_t>(top_k) *
                            static_cast<std::uint64_t>(banks[layer].layout.slot_bytes);
            assignments += top_k;
        }
        const ops::ExpertCacheState& cache_state = pool->cache();
        unsigned long long counters[3]           = {};
        const auto read_counters                 = [&]() {
            CUDA_CHECK(cudaMemcpy(counters, cache_state.statistics, sizeof(counters),
                                  cudaMemcpyDeviceToHost));
        };
        read_counters();
        const unsigned long long misses_before = counters[1];
        // The comparison pass, which is also this route's warm-up: the cache holds nothing, so
        // every routed expert is gathered -- a decode step's own case, and the one the routed
        // figure states.
        std::vector<std::uint16_t> gathered_logits, pinned_logits;
        double cold_ms = 0.0;
        column_pass(*model, &*pool, ids.data(), cold_ms, &gathered_logits);
        read_counters();
        const unsigned long long cold_misses = counters[1] - misses_before;
        // The timed passes: a decode step asks about a token the cache has not seen, so each one
        // draws another of the vocabulary's own ids rather than repeating the last.
        const std::int32_t runs = std::max(repeats, 1);
        double narrow_ms        = 0.0;
        for (std::int32_t run = 0; run < runs; ++run) {
            const std::size_t index = (static_cast<std::size_t>(run) * 7 + 1) % ids.size();
            double ms               = 0.0;
            column_pass(*model, &*pool, ids.data() + index, ms, nullptr);
            narrow_ms = narrow_ms == 0.0 ? ms : std::min(narrow_ms, ms);
        }
        read_counters();
        const unsigned long long timed_misses = counters[1] - misses_before - cold_misses;

        // The widths this route serves in production are one to eight columns -- the engine's lanes
        // -- and the resolve contract is per call: its slot protection covers one call, and the
        // misses it lists may not be read again before the next resolve. Neither can be seen at one
        // column, so identity is checked at every width a lane set produces, cold each time, with
        // only the pass's output compared.
        const std::int32_t widths[3]      = {2, 4, 8};
        std::vector<std::uint16_t> gathered_wide[3];
        for (int index = 0; index < 3; ++index) {
            double wide_ms = 0.0;
            column_pass(*model, &*pool, ids.data(), wide_ms, &gathered_wide[index], widths[index]);
            (void)wide_ms;
        }

        // The control the change has to be read beside: the *same* one-column pass over the staged
        // route, which is what every decode step did before the width rule existed. A second pool
        // over the same model with its staged width forced to one column -- the constructor takes
        // that width, so a harness can ask for the old behaviour -- and the pool is rebuilt rather
        // than kept beside the first because two of them hold 5.6 GB of Device memory between them.
        double staged_column_ms         = 0.0;
        std::size_t staged_column_bytes = 0;
        std::vector<std::uint16_t> staged_logits;
        pool.reset();
        pool.emplace(*model, top_k, 1);
        {
            // The first pass reclaims the bank slots and pays the first fill of every layer; the
            // second is the steady-state one a decode round costs.
            double warm = 0.0;
            column_pass(*model, &*pool, ids.data(), warm, nullptr);
            const std::size_t before_bytes = pool->staged_bytes();
            column_pass(*model, &*pool, ids.data(), staged_column_ms, &staged_logits);
            staged_column_bytes = pool->staged_bytes() - before_bytes;
        }
        pool.reset();

        // The in-place configuration of the same artifact: every bank pinned, no pool at all, so
        // the pass reads each routed expert out of the host mapping it lies in. Released first,
        // because the second materialization needs their Device memory.
        pool.reset();
        model.reset();
        device.synchronize();
        EngineOptions pinned_options = options;
        pinned_options.expert_cache  = product::parse_expert_cache("0");
        nemotron::ExpertResidencyRequest pinned_request;
        pinned_request.automatic      = false;
        pinned_request.explicit_bytes = 0;
        pinned_request.reserve_bytes  = request.reserve_bytes;
        nemotron::LoadPlan pinned_plan =
            nemotron::plan_load(reader, models::load_options(pinned_options), pinned_request);
        if (pinned_plan.expert_residency().staging_bytes != 0) {
            std::cerr << "\nthe in-place configuration staged "
                      << mib(pinned_plan.expert_residency().staging_bytes)
                      << " MiB where --expert-cache 0 asks for none\n";
            return 1;
        }
        std::unique_ptr<nemotron::Model> pinned_model =
            nemotron::materialize_model(std::move(pinned_plan), device);
        device.synchronize();
        double inplace_ms = 0.0;
        column_pass(*pinned_model, nullptr, ids.data(), inplace_ms, &pinned_logits);
        for (std::int32_t run = 0; run < runs; ++run) {
            double ms = 0.0;
            column_pass(*pinned_model, nullptr, ids.data(), ms, nullptr);
            inplace_ms = std::min(inplace_ms, ms);
        }
        std::vector<std::uint16_t> pinned_wide[3];
        for (int index = 0; index < 3; ++index) {
            double wide_ms = 0.0;
            column_pass(*pinned_model, nullptr, ids.data(), wide_ms, &pinned_wide[index],
                        widths[index]);
            (void)wide_ms;
        }
        pinned_model.reset();

        // The two routes read the same stored bytes, so the claim is identity: the same words in
        // the same order, one BF16 word per vocabulary entry.
        std::size_t differing    = 0;
        double max_abs           = 0.0;
        double error_squared     = 0.0;
        double reference_squared = 0.0;
        for (std::size_t index = 0; index < pinned_logits.size(); ++index) {
            if (gathered_logits[index] != pinned_logits[index]) { ++differing; }
            const double left  = bf16_to_float(gathered_logits[index]);
            const double right = bf16_to_float(pinned_logits[index]);
            max_abs            = std::max(max_abs, std::abs(left - right));
            error_squared += (left - right) * (left - right);
            reference_squared += right * right;
        }
        const double relative_l2 =
            reference_squared > 0.0 ? std::sqrt(error_squared / reference_squared) : 0.0;

        // Every miss copies one expert of its own layer, and a layer's expert is its own width, so
        // the cache route's traffic is the miss count times the mean non-resident expert. The cold
        // pass's figure is exact -- all `assignments` missed -- and the warm one is that mean
        // applied to the misses counted over the timed passes, which is what makes the two routes'
        // host-link traffic comparable without a profile.
        const double mean_expert_mib =
            assignments != 0 ? mib(routed_bytes) / static_cast<double>(assignments) : 0.0;
        const double warm_misses = static_cast<double>(timed_misses) / static_cast<double>(runs);

        std::printf("\n  narrow   1 column  %9.1f ms/pass   (best of %d)\n"
                    "  staged   1 column  %9.1f ms/pass   (%.1f MiB staged: the route decode used "
                    "before)\n"
                    "  in-place 1 column  %9.1f ms/pass   (best of %d; every bank pinned)\n"
                    "  narrow vs in-place: %zu logits, %zu differ, max_abs %.6g, rel_l2 %.6g\n"
                    "  one column routes %d assignments: %.1f MiB gathered on the cold pass\n"
                    "  (%llu of %d missed), %.0f missed per warm column, %.1f MiB gathered\n",
                    narrow_ms, runs, staged_column_ms, mib(staged_column_bytes), inplace_ms,
                    runs + 1, pinned_logits.size(), differing, max_abs, relative_l2, assignments,
                    mib(routed_bytes), cold_misses, assignments, warm_misses,
                    warm_misses * mean_expert_mib);
        std::cout << std::flush;

        // The evidence the gather happened at all: a route that silently gathered nothing would
        // report a counter at zero and the same timings as an idle pool.
        if (cold_misses == 0 || cold_misses > static_cast<unsigned long long>(assignments)) {
            std::cerr << "the cold pass gathered " << cold_misses
                      << " assignments where one column routes " << assignments << "\n";
            ++failures;
        }
        // And the claim of the section: the two routes' outputs are the same words.
        if (differing != 0) {
            std::cerr << "the narrow route and the in-place route disagree on " << differing
                      << " of " << pinned_logits.size() << " logits\n";
            ++failures;
        }
        // The staged route reads the same stored bytes through the same decoder, so it agrees too:
        // that is what the gather was allowed to replace, and a difference here would mean the
        // width rule changed a result rather than only where the bytes came from.
        if (staged_logits != gathered_logits) {
            std::cerr << "the staged route and the gathered one disagree on one column's logits\n";
            ++failures;
        }
        // And the same comparison at the widths a lane set actually produces, where the pager's
        // per-call slot protection and miss list are what a mistake would show up in.
        std::printf("\n  columns     words  differ    max_abs    rel_l2\n");
        for (int index = 0; index < 3; ++index) {
            std::size_t wide_differing = 0;
            double wide_max_abs        = 0.0;
            double wide_error          = 0.0;
            double wide_reference      = 0.0;
            for (std::size_t at = 0; at < pinned_wide[index].size(); ++at) {
                if (gathered_wide[index][at] != pinned_wide[index][at]) { ++wide_differing; }
                const double left  = bf16_to_float(gathered_wide[index][at]);
                const double right = bf16_to_float(pinned_wide[index][at]);
                wide_max_abs       = std::max(wide_max_abs, std::abs(left - right));
                wide_error += (left - right) * (left - right);
                wide_reference += right * right;
            }
            const double wide_relative_l2 =
                wide_reference > 0.0 ? std::sqrt(wide_error / wide_reference) : 0.0;
            std::printf("  %7d  %8zu  %6zu  %9.6g  %8.6g\n", widths[index],
                        pinned_wide[index].size(), wide_differing, wide_max_abs, wide_relative_l2);
            if (wide_differing != 0) {
                std::cerr << "the narrow route and the in-place route disagree at " << widths[index]
                          << " columns\n";
                ++failures;
            }
        }
    }
    return failures == 0 ? 0 : 1;
}
