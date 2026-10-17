// The LiLiCorr round, end to end on the real artifact: propose, verify, publish.
//
// A round is the unit speculative decoding runs in, and everything below exists to measure one
// thing: how many of the drafter's seven proposals this engine's target actually accepts, per
// round, with the round's own cost beside it. The reference implementation of the same draft
// accepts **2.122** drafts per round on this engine's taps (slot 0: 70.7%), measured by
// `tools/verify/lilicorr_acceptance.py` against the target's recorded greedy stream. This test
// measures the same quantity through the C++ round -- the draft's forward, the target's 8-row
// verify pass, the accepted prefix and the published bonus token -- so a number far below the
// reference points at the round rather than at the drafter's arithmetic, which the stage diff
// already covers.
//
// The accounting is the reference's, verbatim: the round's anchor is the token already drawn at the
// frontier position `F`, the draft's mask rows sit at `F+1 .. F+7` and each predicts the token at
// its own position, and the target's logits at row `r` verify draft slot `r`. Accepting `n` drafts
// commits `n + 1` tokens (the anchor and the accepted drafts), which is what moves the cursor.
//
// Every round also reports its split: the drafter's own forward against the target's block, the
// state rollback and the committed pass. The card allows the head ~3% of a round, so a draft that
// dominates the number is a finding rather than a detail -- the head's index arithmetic is still
// host-side.
//
//   NINFER_NEMOTRON_ARTIFACT   the artifact carrying the draft; unset skips (77)
//   NINFER_DRAFT_ROUNDS        rounds to run (default 32)
//   NINFER_DRAFT_TEXT          a UTF-8 file to use as the prompt; unset uses the built-in prose
//   NINFER_DRAFT_PROMPT_TOKENS prompt length in tokens (default 512)
//   NINFER_DRAFT_MIN_ACCEPT    the acceptance bar, drafts per round (default 1.5)

#include "artifact/reader.h"
#include "core/device.h"
#include "core/tensor.h"
#include "models/nemotron_h/frontend.h"
#include "models/nemotron_h/load.h"
#include "models/nemotron_h/program.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace nemotron = ninfer::models::nemotron_h;

int failures = 0;

void expect(bool condition, const std::string& message) {
    if (condition) { return; }
    ++failures;
    std::cout << "FAIL: " << message << '\n';
}

// The same paragraph the tap dump was taken on, and therefore the same text the reference
// implementation's 2.122 drafts per round was measured on: acceptance is a property of the text
// as much as of the pair, so a different prompt would not be comparable to it.
constexpr const char* kDefaultText =
    "The Nemotron hybrid stack that this engine serves is a mixture of three mixers rather than "
    "one "
    "attention: most of its blocks are Mamba-2 selective scans, most of the rest are latent "
    "mixture "
    "of experts layers whose routed experts live in host memory, and only eight of its eighty "
    "eight "
    "blocks attend over the sequence at all. That composition is what a speculative drafter has to "
    "condition on, and it is why the drafter does not read the target's attention cache but the "
    "residual stream itself, at six points along the way, projected into its own width. A drafter "
    "that reads the right records and still accepts almost nothing is not a drafter with a bug in "
    "its arithmetic; it is usually a drafter trained against a different post-trained checkpoint, "
    "or against a different quantization of the same one, and the only way to tell the difference "
    "is to measure the accepted length on the real pair rather than to trust the mechanism.";

std::string read_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) { throw std::runtime_error("cannot read " + path); }
    std::ostringstream text;
    text << input.rdbuf();
    return text.str();
}

void write_file(const std::filesystem::path& path, const void* bytes, std::size_t size) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) { throw std::runtime_error("cannot write " + path.string()); }
    output.write(static_cast<const char*>(bytes), static_cast<std::streamsize>(size));
    if (!output) { throw std::runtime_error("cannot write " + path.string()); }
}

std::int32_t env_int(const char* name, std::int32_t fallback) {
    const char* text = std::getenv(name);
    if (text == nullptr || *text == '\0') { return fallback; }
    const int value = std::stoi(text);
    if (value <= 0) { throw std::runtime_error(std::string(name) + " must be positive"); }
    return value;
}

double env_double(const char* name, double fallback) {
    const char* text = std::getenv(name);
    if (text == nullptr || *text == '\0') { return fallback; }
    return std::stod(text);
}

double millis_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
        .count();
}

} // namespace

int main() {
    const char* artifact_path = std::getenv("NINFER_NEMOTRON_ARTIFACT");
    if (artifact_path == nullptr || *artifact_path == '\0') {
        std::cout << "SKIP: NINFER_NEMOTRON_ARTIFACT is not set\n";
        return 77;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        const char* draft_text           = std::getenv("NINFER_DRAFT_TEXT");
        const std::string text           = draft_text != nullptr && *draft_text != '\0'
                                               ? read_file(draft_text)
                                               : std::string(kDefaultText);
        const std::int32_t prompt_tokens = env_int("NINFER_DRAFT_PROMPT_TOKENS", 512);
        const std::int32_t rounds        = env_int("NINFER_DRAFT_ROUNDS", 32);
        const double bar                 = env_double("NINFER_DRAFT_MIN_ACCEPT", 1.5);
        // A plain DFlash stores no block length: the round's width is the request's. This is the
        // same number the Engine's `--draft-tokens` carries, and the Program plans the ring, the
        // proposal logits and the verify block from it.
        const std::int32_t draft_tokens = env_int("NINFER_DRAFT_TOKENS", 7);

        // Three tokens a round is the reference's own committed rate; the capacity is generous so a
        // round that accepts everything is not the one that runs out of room.
        const std::int32_t capacity = prompt_tokens + 3 * rounds + 64;

        ninfer::EngineOptions options;
        options.device          = 0;
        options.max_context     = static_cast<std::uint32_t>(capacity);
        options.max_concurrency = 1;
        options.prefill_chunk   = static_cast<std::uint32_t>(capacity);
        options.kv_cache        = ninfer::KvCacheStorage::BFloat16;
        options.kv_capacity =
            ninfer::KvCapacityPolicy::explicit_capacity(static_cast<std::uint32_t>(capacity));
        options.speculative.backend      = ninfer::SpeculativeBackend::DFlash;
        options.speculative.draft_tokens = static_cast<std::uint32_t>(draft_tokens);
        // Measurement hook: the proposal block routes to experts the prompt never touched, so the
        // verify's first-touch staging may be a pool-size question. Off by default.
        if (const char* slots = std::getenv("NINFER_DRAFT_CACHE_SLOTS");
            slots != nullptr && *slots != '\0') {
            options.expert_cache.cache_slots = std::atoi(slots);
        }

        // The resident banks' budget is everything the card has after the state, one pass's
        // storage, the draft's own state, the expert pool and the runtime's reservation -- the same
        // arithmetic the Engine does. The draft's bytes are part of it because this Program will
        // engage the draft; leaving them out would let the loader spend them on resident banks and
        // the round would fail to allocate its ring.
        ninfer::artifact::Reader reader(artifact_path);
        const nemotron::Config artifact_config = nemotron::parse_config(reader.directory());
        if (!artifact_config.draft) {
            std::cout << "SKIP: " << artifact_path
                      << " carries no draft; this test measures the round's acceptance\n";
            return 77;
        }
        nemotron::ExpertResidencyRequest residency;
        residency.automatic      = options.expert_cache.mode == ninfer::ExpertCacheMode::Automatic;
        residency.explicit_bytes = options.expert_cache.explicit_bytes;
        residency.cache_slots    = options.expert_cache.cache_slots;
        residency.reserve_bytes =
            nemotron::Program::device_bytes(artifact_config, options) -
            nemotron::Program::expert_pool_device_bytes(artifact_config.text, options) +
            nemotron::kRuntimeReservationBytes +
            (options.kv_capacity.mode == ninfer::KvCapacityMode::Automatic
                 ? options.kv_capacity.automatic_headroom_bytes
                 : 0);
        auto plan = nemotron::plan_load(reader, ninfer::models::load_options(options), residency);
        const nemotron::TextResources& resources = plan.resources();
        const nemotron::FrontendResources frontend_resources{
            .tokenizer_json         = std::string(resources.tokenizer_json),
            .tokenizer_config_json  = std::string(resources.tokenizer_config_json),
            .generation_config_json = std::string(resources.generation_config_json),
            .chat_template_jinja    = std::string(resources.chat_template_jinja),
        };

        ninfer::DeviceContext device;
        auto model                         = nemotron::materialize_model(std::move(plan), device);
        const nemotron::TextConfig& config = model->config();
        (void)config;

        nemotron::Frontend frontend(frontend_resources, static_cast<std::uint32_t>(capacity));
        const std::vector<ninfer::TokenId> ids = frontend.tokenize_text(text);
        if (static_cast<std::int32_t>(ids.size()) < 64) {
            throw std::runtime_error("the prompt text tokenized to fewer than 64 tokens, which is "
                                     "no context for a drafter to condition on");
        }
        const std::int32_t actual_prompt =
            std::min(prompt_tokens, static_cast<std::int32_t>(ids.size()));
        const std::vector<std::int32_t> prompt(ids.begin(), ids.begin() + actual_prompt);

        nemotron::Program program(*model, options, device);
        const ninfer::DeviceExecutionView execution = device.execution_view();

        // The draft is engaged before the request's first pass, so the prefill itself conditions
        // the draft's ring: the round then reads a context assembled by the passes that committed
        // those positions, which is what a request longer than one pass needs.
        const auto setup_started = std::chrono::steady_clock::now();
        program.engage_draft(execution);
        program.create(0, execution);
        program.prefill(0, prompt, execution);
        const double setup_ms = millis_since(setup_started);

        const bool plain_dflash =
            artifact_config.draft->flavor == ninfer::models::draft::Spec::Flavor::Dflash;
        const std::int32_t slots =
            plain_dflash ? draft_tokens
                         : static_cast<std::int32_t>(artifact_config.draft->block_size) - 1;
        std::vector<std::int32_t> accepted_per_slot(static_cast<std::size_t>(slots), 0);
        std::int32_t accepted_total      = 0;
        std::int32_t bonus_disagreements = 0;
        // The drafter's own top-1 chain: at every slot, the candidate the target's head ranked
        // first. The walk cannot propose it -- it commits to one predecessor's row -- but as a
        // measurement it separates the two ways a round loses: a candidate set that does not hold
        // the answer, and a lattice that out-votes it. A walk far below its own top-1 chain is a
        // broken score path rather than a weak drafter.
        std::int32_t top1_total = 0;
        double draft_ms = 0.0, verify_ms = 0.0, accept_ms = 0.0, restore_ms = 0.0, commit_ms = 0.0;
        double round_wall_ms = 0.0;
        std::vector<std::int32_t> stream(prompt);

        // The conditioning experiment's arm: advance the target one row at a time, so the draft
        // reads the taps a plain decode produces instead of the ones a 2-4 row commit pass
        // produces.
        const bool single_row_commit = std::getenv("NINFER_DRAFT_W1_COMMIT") != nullptr;
        // The width arm: score `NINFER_DRAFT_SLOTS` mask rows this round (1..7) with the artifact's
        // own tables, so a shorter block needs no regeneration.
        std::int32_t round_slots = slots;
        // The width actually run; the artifact's own tables are indexed 0..width-1, so a narrower
        // block must bound every per-slot loop and every `verified`/`proposed` index.
        std::int32_t width_slots = 0;
        if (const char* slot_text = std::getenv("NINFER_DRAFT_SLOTS");
            slot_text != nullptr && *slot_text != '\0') {
            round_slots = std::stoi(slot_text);
        }
        width_slots         = round_slots > 0 ? round_slots : 0;
        std::int32_t anchor = program.sample(0, execution);

        // An optional diagnostic, off by default: when a round accepts far less than it should, the
        // first question is whether the target's verify block reproduces the target's own decode.
        // This arm draws that stream one token per pass over the positions the first round checks,
        // then runs one round from the same prefill state and prints proposal, verify block and
        // decode side by side. Two prefills and ten decodes is not a cost every run should pay.
        if (std::getenv("NINFER_DRAFT_VERIFY_CHECK") != nullptr) {
            std::vector<std::int32_t> decoded;
            for (std::int32_t step = 0; step < slots + 3; ++step) {
                const std::int32_t drawn = program.sample(0, execution);
                decoded.push_back(drawn);
                program.decode(0, drawn, execution);
            }
            program.finish(0);
            program.create(0, execution);
            program.prefill(0, prompt, execution);
            anchor = program.sample(0, execution);
            const nemotron::DraftRoundReport probe =
                program.draft_round(0, anchor, execution, nullptr, false, round_slots);
            std::cout << "diagnose: anchor " << anchor << " | decode draws its successor "
                      << decoded[0] << '\n';
            std::cout << "  slot | proposed | verified |         decoded row+1 | p~v v~d p~d\n";
            for (std::int32_t index = 0; index < slots; ++index) {
                const std::int32_t proposed = probe.proposed[static_cast<std::size_t>(index)];
                const std::int32_t verified = probe.verified[static_cast<std::size_t>(index)];
                const std::int32_t actual   = decoded[static_cast<std::size_t>(index) + 1];
                // The draft's candidate list at this slot, and where the target's own token sits
                // in it. A slot whose answer is not in the list cannot be accepted whatever the
                // walk does, which is why the ceiling and the walk are reported apart.
                const std::vector<std::int32_t>& candidates = program.draft()->candidates(0);
                const std::size_t base                      = static_cast<std::size_t>(index) * 8U;
                std::string listed                          = "  cand[";
                std::int32_t rank                           = -1;
                for (std::size_t k = 0; k < 8U && base + k < candidates.size(); ++k) {
                    listed += (k == 0 ? "" : " ");
                    listed += std::to_string(candidates[base + k]);
                    if (candidates[base + k] == verified) { rank = static_cast<std::int32_t>(k); }
                }
                listed += "] rank=" + (rank < 0 ? std::string("-") : std::to_string(rank));
                std::cout << "  " << index << " | " << proposed << " | " << verified << " | "
                          << actual << " | " << (proposed == verified ? "y" : "n") << " "
                          << (verified == actual ? "y" : "n") << " "
                          << (proposed == actual ? "y" : "n") << listed << '\n';
            }
            // The diagnostic consumed a round; put the lane back where the loop expects it.
            program.finish(0);
            program.create(0, execution);
            program.prefill(0, prompt, execution);
            anchor = program.sample(0, execution);
        }

        // ---- Measurement arms, off by default. Neither changes what the round computes: one dumps
        // the engine's own head inputs so the oracle's head arithmetic can be run on them, the
        // other times the pieces of a round. ----
        const char* dump_dir   = std::getenv("NINFER_DRAFT_STAGE_DUMP");
        const char* cost_probe = std::getenv("NINFER_DRAFT_COST_PROBE");
        if (dump_dir != nullptr && *dump_dir != '\0') {
            const std::filesystem::path directory(dump_dir);
            std::filesystem::create_directories(directory);

            struct DumpObserver final : nemotron::DraftObserver {
                std::filesystem::path directory;
                std::int32_t round          = 0;
                mutable std::int32_t stages = 0;
                mutable double host_ms      = 0.0;
                mutable double worst_ms     = 0.0;
                mutable double second_ms    = 0.0;
                mutable std::string worst_stage;
                mutable std::string second_stage;
                mutable std::chrono::steady_clock::time_point last =
                    std::chrono::steady_clock::now();

                void on_stage(std::string_view stage, const ninfer::Tensor& tensor) const override {
                    const auto now = std::chrono::steady_clock::now();
                    const double gap =
                        std::chrono::duration<double, std::milli>(now - last).count();
                    last = now;
                    ++stages;
                    host_ms += gap;
                    if (gap > worst_ms) {
                        second_ms    = worst_ms;
                        second_stage = worst_stage;
                        worst_ms     = gap;
                        worst_stage  = std::string(stage);
                    } else if (gap > second_ms) {
                        second_ms    = gap;
                        second_stage = std::string(stage);
                    }
                    // The stages a reference forward is rebuilt from: the tapped operand, the fused
                    // feature, the normed context, one layer's q/k/v post-RoPE, its attention and
                    // MLP outputs, the final normed hidden, and the tower head's logits. The rest
                    // (the head's candidate machinery) is LiLiCorr's and has no plain counterpart.
                    static constexpr const char* kDumped[] = {
                        "body.tap_operand", "body.fc_out", "body.hidden_norm_out",
                        "body.embeddings", "body.layer0.norm_in", "body.layer0.q_pre_norm",
                        "body.layer0.q_post_norm", "body.layer0.k_post_norm",
                        "body.layer0.k_ctx_post_rope", "body.layer0.v_ctx",
                        "body.layer0.q_post_rope", "body.layer0.k_post_rope",
                        "body.layer0.v_pre_norm", "body.layer0.attn_out",
                        "body.layer0.mlp_act", "body.layer0.out", "body.norm_out",
                        "body.anchor_hidden", "head.logits", "head.pass_hidden",
                        "ring.layer0.k", "ring.layer0.v", "ring.positions"};
                    bool wanted = false;
                    for (const char* name : kDumped) { wanted = wanted || stage == name; }
                    if (!wanted) { return; }
                    std::vector<std::byte> bytes(static_cast<std::size_t>(tensor.bytes()));
                    // The stages are written on the Program's execution stream; this copy is on the
                    // default one, so the stream has to be joined before it is read.
                    cudaDeviceSynchronize();
                    const cudaError_t status =
                        cudaMemcpy(bytes.data(), tensor.data, bytes.size(), cudaMemcpyDeviceToHost);
                    if (status != cudaSuccess) {
                        throw std::runtime_error("stage dump copy failed");
                    }
                    std::ostringstream name;
                    name << "r" << round << "." << stage << ".bin";
                    write_file(directory / name.str(), bytes.data(), bytes.size());
                }
            } observer;

            observer.directory = directory;
            std::ostringstream rounds_text;
            for (std::int32_t i = 0; i < rounds; ++i) {
                observer.round                          = i;
                observer.stages                         = 0;
                observer.host_ms                        = 0.0;
                observer.worst_ms                       = 0.0;
                observer.second_ms                      = 0.0;
                const auto started                      = std::chrono::steady_clock::now();
                observer.last                           = started;
                const std::int32_t here                 = program.position(0);
                const nemotron::DraftRoundReport report = program.draft_round(
                    0, anchor, execution, &observer, single_row_commit, round_slots);
                const double wall = millis_since(started);
                std::cout << "  dump round " << (i + 1) << ": wall " << wall << " ms | "
                          << observer.stages << " stages | host gaps " << observer.host_ms
                          << " ms | worst gap " << observer.worst_ms << " ms @ "
                          << observer.worst_stage << " | next " << observer.second_ms << " ms @ "
                          << observer.second_stage << '\n';
                rounds_text << i << ' ' << anchor << ' ' << here << ' ' << report.accepted;
                for (const std::int32_t id : report.proposed) { rounds_text << ' ' << id; }
                for (const std::int32_t id : report.verified) { rounds_text << ' ' << id; }
                rounds_text << '\n';
                anchor = program.sample(0, execution);
            }
            const std::string text = rounds_text.str();
            write_file(directory / "rounds.txt", text.data(), text.size());
            program.finish(0);
            program.create(0, execution);
            program.prefill(0, prompt, execution);
            anchor = program.sample(0, execution);
        }
        if (cost_probe != nullptr && *cost_probe != '\0') {
            // The pieces of a round, each measured on a fresh state: the round's own verify pass (8
            // columns of *proposed* tokens), the same 8 ids as an ordinary committed pass (the bare
            // pp8 the doc quotes), the same again (does a repeat reuse the staged experts?), and
            // one row.
            const auto reset = [&]() {
                program.finish(0);
                program.create(0, execution);
                program.prefill(0, prompt, execution);
            };
            reset();
            const std::int32_t probe_anchor = program.sample(0, execution);
            const nemotron::DraftRoundReport probe =
                program.draft_round(0, probe_anchor, execution);
            std::vector<std::int32_t> eight{probe_anchor};
            eight.insert(eight.end(), probe.proposed.begin(), probe.proposed.end());
            reset();
            auto mark = std::chrono::steady_clock::now();
            program.prefill(0, eight, execution);
            const double bare8 = millis_since(mark);
            mark               = std::chrono::steady_clock::now();
            program.prefill(0, eight, execution);
            const double bare8_repeat = millis_since(mark);
            mark                      = std::chrono::steady_clock::now();
            program.prefill(0, std::span(&eight.front(), 1), execution);
            const double bare1                             = millis_since(mark);
            const nemotron::ExpertResidencyPlan& residency = program.expert_residency();
            std::cout << "cost: verify(8 proposed) " << probe.verify_ms << " ms | committed(8) "
                      << bare8 << " ms | committed(8) again " << bare8_repeat
                      << " ms | committed(1) " << bare1 << " ms | draft " << probe.draft_ms
                      << " ms | commit pass " << probe.commit_ms << " ms\n"
                      << "cost: expert staged bytes " << residency.staging_bytes
                      << " | resident bytes " << residency.resident_bytes << " | cache slots "
                      << residency.cache_slots << "/" << residency.requested_slots << " (clamped "
                      << residency.clamped << ")\n";
            reset();
            anchor = program.sample(0, execution);
        }

        for (std::int32_t round = 0; round < rounds; ++round) {
            const std::int32_t before = program.position(0);
            const auto started        = std::chrono::steady_clock::now();
            const nemotron::DraftRoundReport report =
                program.draft_round(0, anchor, execution, nullptr, single_row_commit, round_slots);
            const double wall        = millis_since(started);
            const std::int32_t bonus = program.sample(0, execution);

            expect(report.anchor == anchor,
                   "round " + std::to_string(round) + " reports the anchor it was handed");
            expect(report.anchor_position == before,
                   "round " + std::to_string(round) + " starts at the lane's frontier");
            expect(static_cast<std::int32_t>(report.proposed.size()) == width_slots,
                   "round " + std::to_string(round) + " proposes one token per mask slot");
            expect(report.accepted >= 0 && report.accepted <= width_slots,
                   "round " + std::to_string(round) + " accepts within its block");
            expect(static_cast<std::int32_t>(report.committed.size()) == report.accepted + 1,
                   "round " + std::to_string(round) +
                       " commits the anchor and the accepted drafts");
            expect(program.position(0) == before + report.accepted + 1,
                   "round " + std::to_string(round) + " moves the cursor by accepted + 1");
            expect(report.committed[0] == anchor,
                   "round " + std::to_string(round) + " consumes the anchor it was handed");
            for (std::int32_t index = 0; index < report.accepted; ++index) {
                expect(report.committed[static_cast<std::size_t>(index) + 1] ==
                           report.proposed[static_cast<std::size_t>(index)],
                       "round " + std::to_string(round) + " commits its accepted proposals");
            }
            if (report.bonus != bonus) { ++bonus_disagreements; }

            // The round commits `accepted + 1` positions: the anchor and the accepted drafts. The
            // bonus is the *next* round's anchor, so it is committed by that round, not this one.
            for (const std::int32_t id : report.committed) { stream.push_back(id); }

            for (std::int32_t index = 0; index < report.accepted; ++index) {
                accepted_per_slot[static_cast<std::size_t>(index)] += 1;
            }
            accepted_total += report.accepted;
            // The LiLiCorr head's candidate ceiling. A plain DFlash has no candidate set -- its
            // proposal is the tower's own argmax -- so the ceiling and the walk are the same thing
            // there and the measurement is skipped rather than invented.
            {
                const std::vector<std::int32_t>& candidates = program.draft()->candidates(0);
                if (!plain_dflash && candidates.size() >= static_cast<std::size_t>(width_slots) * 8U) {
                    std::int32_t top1 = 0;
                    while (top1 < width_slots &&
                           candidates[static_cast<std::size_t>(top1) * 8U] ==
                               report.verified[static_cast<std::size_t>(top1)]) {
                        ++top1;
                    }
                    top1_total += top1;
                }
            }
            draft_ms += report.draft_ms;
            verify_ms += report.verify_ms;
            accept_ms += report.accept_ms;
            restore_ms += report.restore_ms;
            commit_ms += report.commit_ms;
            round_wall_ms += wall;

            std::cout << "  round " << (round + 1) << ": frontier " << before << " | accepted "
                      << report.accepted << " of " << slots << " | proposed [";
            for (std::size_t index = 0; index < report.proposed.size(); ++index) {
                std::cout << (index == 0 ? "" : " ") << report.proposed[index];
            }
            std::cout << "] | wall " << wall << " ms [draft " << report.draft_ms << ", verify "
                      << report.verify_ms << ", accept " << report.accept_ms << ", rollback "
                      << report.restore_ms << ", commit " << report.commit_ms << "]\n";
            anchor = bonus;
        }

        const double rounds_d     = static_cast<double>(rounds);
        const double mean         = static_cast<double>(accepted_total) / rounds_d;
        const double slot0        = static_cast<double>(accepted_per_slot.front()) / rounds_d;
        const double mean_draft   = draft_ms / rounds_d;
        const double mean_verify  = verify_ms / rounds_d;
        const double mean_accept  = accept_ms / rounds_d;
        const double mean_restore = restore_ms / rounds_d;
        const double mean_commit  = commit_ms / rounds_d;
        const double mean_wall    = round_wall_ms / rounds_d;

        // The last round's bonus is drawn but not consumed, so it is not one of the committed
        // positions.
        stream.push_back(anchor);
        std::cout << "\nrounds " << rounds << " | commit width "
                  << (single_row_commit ? "1 (experiment arm)" : "accepted + 1") << " | prompt "
                  << actual_prompt << " tokens | committed " << (stream.size() - prompt.size() - 1)
                  << " tokens in " << rounds << " rounds\n";
        std::cout << "accepted drafts per round: " << mean << " (reference 2.122, bar " << bar
                  << ")\n";
        std::cout << "committed tokens per round: " << (mean + 1.0) << '\n';
        std::cout << "the same rounds' top-1 chain: " << static_cast<double>(top1_total) / rounds_d
                  << " accepted (the drafter's own rank-0 candidate at every slot)\n";
        std::cout << "slot 0 rate: " << slot0 << " (reference 0.707)\n";
        for (std::int32_t index = 0; index < width_slots; ++index) {
            std::cout << "  slot " << index << ": "
                      << static_cast<double>(accepted_per_slot[static_cast<std::size_t>(index)]) /
                             rounds_d
                      << '\n';
        }
        std::cout << "per-round wall time: " << mean_wall << " ms (setup " << setup_ms
                  << " ms excluded)\n"
                  << "  draft forward   " << mean_draft << " ms ("
                  << (mean_wall > 0.0 ? 100.0 * mean_draft / mean_wall : 0.0) << "%)\n"
                  << "  target verify   " << mean_verify << " ms ("
                  << (mean_wall > 0.0 ? 100.0 * mean_verify / mean_wall : 0.0) << "%)\n"
                  << "  block argmax    " << mean_accept << " ms\n"
                  << "  state rollback  " << mean_restore << " ms\n"
                  << "  committed pass  " << mean_commit << " ms\n";
        std::cout << "rounds whose verify-pass bonus differed from the committed pass's draw: "
                  << bonus_disagreements << " of " << rounds
                  << " (equal widths are not the same kernel routes; a near tie can flip one)\n";

        expect(mean >= bar, "accepted drafts per round is " + std::to_string(mean) +
                                ", below the " + std::to_string(bar) + " bar");
        if (failures != 0) {
            std::cout << "nemotron_h draft round: " << failures << " check(s) failed\n";
            return 1;
        }
        std::cout << "nemotron_h draft round: ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cout << "nemotron_h draft round: " << error.what() << '\n';
        return 1;
    }
}
