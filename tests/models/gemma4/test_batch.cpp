// Decodes several lanes in one pass through the model's own Program and checks that the lanes stay
// independent: the pass shares the weight reads across one column per lane, while each lane attends
// to its own cache.
//
//   NINFER_GEMMA_ARTIFACT  the artifact
//   NINFER_GEMMA_PROMPT    a file of int32 little-endian token ids, at least 1,100 of them
//
// Three prompts are cut from the file: A is its first 1,100 tokens, longer than the sliding ring, and
// B and C are shorter slices from inside it. Generation is greedy throughout.
//
//   * Partner swap: lane 0 decodes A beside B, then beside C (named in the other order). The
//     columns are independent and the attention per lane, so lane 0's tokens must be identical.
//   * Idle lane: a lane prefilled with C sits out the batched rounds and then decodes alone; its
//     tokens must equal C decoded alone from the start, so a batched round touches no other cache.
//   * Graph and eager: three lanes decoded with and without CUDA Graphs must be identical.
//   * Solo: a lane decoded alone takes the single-token Linear routes and a batch the multi-token
//     ones, whose accumulation order differs, so lane 0's batched tokens need not equal its solo ones.
//     Where they first differ, the solo token must be within a BF16 rounding of the batch's best
//     logit there.

#include "core/device.h"
#include "models/gemma4/load.h"
#include "models/gemma4/program.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <span>
#include <vector>

namespace {

using namespace ninfer;
namespace gemma = ninfer::models::gemma4;

constexpr std::int32_t kSteps = 32;

std::vector<std::int32_t> read_ids(const char* path) {
    std::ifstream source(path, std::ios::binary | std::ios::ate);
    const std::streamsize bytes = source.tellg();
    source.seekg(0);
    std::vector<std::int32_t> ids(static_cast<std::size_t>(bytes) / sizeof(std::int32_t));
    source.read(reinterpret_cast<char*>(ids.data()), bytes);
    return ids;
}

float decode_bf16(std::uint16_t value) {
    std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
    float decoded;
    std::memcpy(&decoded, &bits, sizeof(decoded));
    return decoded;
}

using Prompt = std::span<const std::int32_t>;

struct Assignment {
    std::int32_t lane;
    Prompt prompt;
};

// Prefills every assigned lane, draws its first token alone, then decodes all of them together for
// kSteps rounds with `order` naming the lanes. Returns each assignment's tokens. `observe(step)` runs
// before each round's draw.
template <class Observe>
std::vector<std::vector<std::int32_t>> generate(gemma::Program& program,
                                                std::span<const Assignment> lanes,
                                                std::span<const std::int32_t> order,
                                                DeviceExecutionView execution, Observe observe) {
    std::vector<std::vector<std::int32_t>> tokens(lanes.size());
    for (std::size_t i = 0; i < lanes.size(); ++i) {
        program.reset(lanes[i].lane);
        program.prefill(lanes[i].lane, lanes[i].prompt, execution);
        tokens[i].push_back(program.sample(lanes[i].lane, execution));
    }
    for (std::int32_t step = 0; step < kSteps; ++step) {
        std::vector<std::int32_t> ids;
        for (const std::int32_t lane : order) {
            for (std::size_t i = 0; i < lanes.size(); ++i) {
                if (lanes[i].lane == lane) ids.push_back(tokens[i].back());
            }
        }
        program.decode(order, ids, execution);
        observe(step);
        const std::vector<std::int32_t> drawn = program.sample(order, execution);
        for (std::size_t k = 0; k < order.size(); ++k) {
            for (std::size_t i = 0; i < lanes.size(); ++i) {
                if (lanes[i].lane == order[k]) tokens[i].push_back(drawn[k]);
            }
        }
    }
    return tokens;
}

std::vector<std::vector<std::int32_t>> generate(gemma::Program& program,
                                                std::span<const Assignment> lanes,
                                                std::span<const std::int32_t> order,
                                                DeviceExecutionView execution) {
    return generate(program, lanes, order, execution, [](std::int32_t) {});
}

// Prefills every assigned lane and draws its first token, then runs batched MTP rounds, each lane
// drafting its own count and keeping its whole licensed run, until every lane has kSteps + 1 tokens.
std::vector<std::vector<std::int32_t>> speculate(gemma::Program& program,
                                                 std::span<const Assignment> lanes,
                                                 std::span<const std::int32_t> drafts,
                                                 DeviceExecutionView execution) {
    std::vector<std::vector<std::int32_t>> tokens(lanes.size());
    for (std::size_t i = 0; i < lanes.size(); ++i) {
        program.reset(lanes[i].lane);
        program.prefill(lanes[i].lane, lanes[i].prompt, execution);
        tokens[i].push_back(program.sample(lanes[i].lane, execution));
    }
    for (;;) {
        std::vector<std::int32_t> active, anchors, counts;
        std::vector<std::size_t> index;
        for (std::size_t i = 0; i < lanes.size(); ++i) {
            if (tokens[i].size() > static_cast<std::size_t>(kSteps)) continue;
            active.push_back(lanes[i].lane);
            anchors.push_back(tokens[i].back());
            counts.push_back(drafts[i]);
            index.push_back(i);
        }
        if (active.empty()) break;
        const auto rounds = program.speculate(active, anchors, counts, execution);
        for (std::size_t k = 0; k < active.size(); ++k) {
            program.commit_round(active[k], static_cast<std::int32_t>(rounds[k].tokens.size()),
                                 execution);
            auto& out = tokens[index[k]];
            out.insert(out.end(), rounds[k].tokens.begin(), rounds[k].tokens.end());
        }
    }
    for (auto& out : tokens) out.resize(static_cast<std::size_t>(kSteps) + 1);
    return tokens;
}

bool equal(const std::vector<std::int32_t>& a, const std::vector<std::int32_t>& b, const char* what) {
    const auto mismatch = std::mismatch(a.begin(), a.end(), b.begin(), b.end());
    if (mismatch.first == a.end() && mismatch.second == b.end()) {
        std::cout << "  " << what << ": " << a.size() << " tokens identical\n";
        return true;
    }
    std::cerr << "gemma4 batch: " << what << " differs at token " << (mismatch.first - a.begin())
              << '\n';
    return false;
}

} // namespace

int main() {
    const char* artifact    = std::getenv("NINFER_GEMMA_ARTIFACT");
    const char* prompt_path = std::getenv("NINFER_GEMMA_PROMPT");
    if (artifact == nullptr || *artifact == '\0' || prompt_path == nullptr) {
        std::cout << "SKIP: NINFER_GEMMA_ARTIFACT and NINFER_GEMMA_PROMPT must be set\n";
        return 77;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        DeviceContext device;
        const DeviceExecutionView execution = device.execution_view();
        // With the drafter when the artifact carries one, for the MTP rounds.
        models::LoadOptions options;
        options.speculative = SpeculativeBackend::Mtp;
        auto model = [&] {
            try {
                return gemma::load_model(artifact, options, device);
            } catch (const std::exception&) {
                return gemma::load_model(artifact, models::LoadOptions{}, device);
            }
        }();
        const std::vector<std::int32_t> ids = read_ids(prompt_path);
        if (ids.size() < 1100) {
            std::cerr << "gemma4 batch: the prompt needs at least 1,100 tokens\n";
            return 1;
        }
        const Prompt a(ids.data(), 1100);
        const Prompt b(ids.data() + 200, 300);
        const Prompt c(ids.data() + 600, 300);
        const std::int32_t capacity = 1100 + kSteps + 8;
        const std::int32_t vocabulary = static_cast<std::int32_t>(model->config().vocab_size);
        int failures = 0;

        gemma::Program graphs(*model, capacity, 3, device, 0, true);

        // Solo: lane 0 decodes A alone, then C alone.
        const Assignment solo_a[] = {{0, a}};
        const std::int32_t lane0[] = {0};
        const auto alone_a = generate(graphs, solo_a, lane0, execution)[0];
        const Assignment solo_c[] = {{0, c}};
        const auto alone_c = generate(graphs, solo_c, lane0, execution)[0];

        // Partner swap, with the idle lane: C waits in lane 2 through the first run.
        const Assignment with_b[] = {{0, a}, {1, b}};
        const std::int32_t lanes01[] = {0, 1};
        const std::int32_t lane0b[]  = {1};
        graphs.reset(2);
        graphs.prefill(2, c, execution);
        const std::int32_t idle_first = graphs.sample(2, execution);
        // Lane 0's batched logits at each round, against the solo token there.
        std::int32_t first_difference = -1;
        float shortfall               = 0.0F;
        std::vector<std::uint16_t> host(static_cast<std::size_t>(vocabulary));
        const auto beside_b = generate(graphs, with_b, lanes01, execution, [&](std::int32_t step) {
            // The pass runs on the device's own stream, which a legacy-stream copy does not wait for.
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(host.data(), graphs.logits(0).data,
                                  host.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
            const auto best = std::max_element(host.begin(), host.end(), [](auto x, auto y) {
                return decode_bf16(x) < decode_bf16(y);
            });
            const std::int32_t chosen = static_cast<std::int32_t>(best - host.begin());
            const std::int32_t wanted = alone_a[static_cast<std::size_t>(step) + 1];
            if (first_difference < 0 && chosen != wanted) {
                first_difference = step + 1;
                shortfall = decode_bf16(*best) - decode_bf16(host[static_cast<std::size_t>(wanted)]);
            }
        });
        // The idle lane resumes alone: its first token, then kSteps rounds.
        std::vector<std::int32_t> idle{idle_first};
        for (std::int32_t step = 0; step < kSteps; ++step) {
            graphs.decode(2, idle.back(), execution);
            idle.push_back(graphs.sample(2, execution));
        }
        failures += !equal(idle, alone_c, "idle lane against C alone");

        const Assignment with_c[] = {{0, a}, {1, c}};
        const std::int32_t lanes10[] = {1, 0};
        const auto beside_c = generate(graphs, with_c, lanes10, execution);
        failures += !equal(beside_b[0], beside_c[0], "lane 0 beside B against beside C");

        // Solo against batched: the first difference must be a near tie.
        if (first_difference < 0) {
            failures += !equal(beside_b[0], alone_a, "lane 0 batched against alone");
        } else {
            std::cout << "  lane 0 batched against alone: first difference at token "
                      << first_difference << ", the solo token " << shortfall
                      << " below the batch's best\n";
            if (shortfall > 0.25F) {
                std::cerr << "gemma4 batch: the solo token is not a near tie in the batch\n";
                ++failures;
            }
        }

        // Graph and eager, three lanes.
        const Assignment three[] = {{0, a}, {1, b}, {2, c}};
        const std::int32_t lanes012[] = {0, 1, 2};
        const auto replayed = generate(graphs, three, lanes012, execution);
        {
            gemma::Program eager(*model, capacity, 3, device, 0, false);
            const auto direct = generate(eager, three, lanes012, execution);
            for (std::size_t i = 0; i < 3; ++i) {
                failures += !equal(replayed[i], direct[i], "three lanes, graph against eager");
            }
        }

        // MTP rounds batched across lanes at different draft counts, against each lane's own rounds
        // and against plain greedy decode.
        if (model->draft_config()) {
            gemma::Program mtp(*model, capacity, 3, device, gemma::Program::kMaximumDraftTokens, true);
            const Assignment pair[]       = {{0, a}, {2, b}};
            const std::int32_t counts[]   = {7, 3};
            const auto together           = speculate(mtp, pair, counts, execution);
            const auto solo_mtp_a         = speculate(mtp, std::span(pair, 1), std::span(counts, 1), execution);
            const auto solo_mtp_b         = speculate(mtp, std::span(pair + 1, 1), std::span(counts + 1, 1), execution);
            failures += !equal(together[0], solo_mtp_a[0], "MTP lane 0 batched against its own rounds");
            failures += !equal(together[1], solo_mtp_b[0], "MTP lane 2 batched against its own rounds");
            const auto plain_b = generate(graphs, std::span(with_b + 1, 1), lane0b, execution)[0];
            failures += !equal(together[0], alone_a, "MTP lane 0 batched against plain greedy");
            failures += !equal(together[1], plain_b, "MTP lane 2 batched against plain greedy");
        } else {
            std::cout << "  no drafter in the artifact: MTP batching not checked\n";
        }

        std::cout << (failures == 0 ? "OK" : "FAIL") << " gemma4 batch\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "gemma4 batch: " << error.what() << '\n';
        return 1;
    }
}
