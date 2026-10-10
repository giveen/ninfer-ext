// MTP rounds against the target alone.
//
// A greedy lane generates with the widest draft window, so most rounds reject drafts and roll back.
// Two checks follow:
//  - exact: every sliding ring holds exactly the committed positions of the last window, and every
//    global row its own position, so no rejected draft evicted a key a later query needs and none
//    survives where a query could see it;
//  - behavioural: a fresh Program scores prompt + output in one go, and each generated token is
//    compared with that pass's argmax. Batched and token-at-a-time passes round differently, so the
//    same comparison is made for a plain greedy decode of the same length, and the MTP run must not
//    disagree more often than it does.
//
// NINFER_GEMMA_ARTIFACT an artifact converted with --components text,mtp
// NINFER_GEMMA_PROMPT   I32 token ids (the first 1100 are used, so the rings wrap)

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
#include <string>
#include <vector>

namespace {

using namespace ninfer;
namespace gemma = ninfer::models::gemma4;

constexpr std::size_t kPrompt  = 1100;
constexpr std::int32_t kOutput = 384;

float decode_bf16(std::uint16_t value) {
    const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
    float decoded;
    std::memcpy(&decoded, &bits, sizeof(decoded));
    return decoded;
}

struct Agreement {
    std::int32_t mismatches = 0;
    float worst_margin      = 0.0F; // the largest logit gap by which a generated token lost
};

// Scores prompt + output in one pass per batch and compares each generated token with the argmax.
Agreement agreement(gemma::Program& program, const std::vector<std::int32_t>& prompt,
                    const std::vector<std::int32_t>& output, DeviceExecutionView execution) {
    std::vector<std::int32_t> sequence = prompt;
    sequence.insert(sequence.end(), output.begin(), output.end());
    Agreement out;
    const auto sink = [&](const ScoredLogits& view) {
        for (std::uint32_t column = 0; column < view.columns; ++column) {
            const auto target = view.first_target + column;
            const std::uint16_t* row = view.values.data() + static_cast<std::size_t>(column) *
                                                                view.vocab_size;
            float best = -1.0e30F;
            for (std::uint32_t id = 0; id < view.vocab_size; ++id) {
                best = std::max(best, decode_bf16(row[id]));
            }
            const float chosen = decode_bf16(row[sequence[target]]);
            if (chosen < best) {
                ++out.mismatches;
                out.worst_margin = std::max(out.worst_margin, best - chosen);
            }
        }
    };
    (void)program.causal_score(sequence, static_cast<std::int32_t>(prompt.size()), sink, execution);
    return out;
}

} // namespace

int main() {
    const char* path        = std::getenv("NINFER_GEMMA_ARTIFACT");
    const char* prompt_path = std::getenv("NINFER_GEMMA_PROMPT");
    if (path == nullptr || prompt_path == nullptr) {
        std::cout << "SKIP: NINFER_GEMMA_ARTIFACT and NINFER_GEMMA_PROMPT must be set\n";
        return 77;
    }
    try {
        DeviceContext device;
        const DeviceExecutionView execution = device.execution_view();
        models::LoadOptions options;
        options.speculative = SpeculativeBackend::Mtp;
        auto model         = gemma::load_model(path, options, device);
        const auto& config = model->config();

        std::vector<std::int32_t> prompt;
        {
            std::ifstream in(prompt_path, std::ios::binary);
            std::int32_t id = 0;
            while (prompt.size() < kPrompt && in.read(reinterpret_cast<char*>(&id), sizeof(id))) {
                prompt.push_back(id);
            }
        }
        // Past the window, so the rings wrap during generation.
        const std::int32_t capacity = static_cast<std::int32_t>(prompt.size()) + kOutput + 16;
        constexpr std::int32_t kDrafts = gemma::Program::kMaximumDraftTokens;
        int failures = 0;

        // The MTP run: every licensed token is kept, so each round commits its whole prefix.
        gemma::Program speculative(*model, capacity, 1, device, kDrafts);
        speculative.reset(0);
        speculative.prefill(0, prompt, execution);
        std::vector<std::int32_t> mtp{speculative.sample(0, execution)};
        std::int64_t drafted = 0, accepted = 0, rounds = 0;
        while (static_cast<std::int32_t>(mtp.size()) < kOutput) {
            const std::int32_t room = kOutput - static_cast<std::int32_t>(mtp.size());
            const auto round = speculative.speculate(0, mtp.back(), std::min(kDrafts, room), execution);
            const auto kept  = static_cast<std::int32_t>(
                std::min<std::size_t>(round.tokens.size(), static_cast<std::size_t>(room)));
            speculative.commit_round(0, kept, execution);
            mtp.insert(mtp.end(), round.tokens.begin(), round.tokens.begin() + kept);
            drafted += std::min(kDrafts, room);
            accepted += std::min(round.accepted_drafts, kept);
            ++rounds;
        }

        // Exact: the ring and the rows hold the committed positions.
        CUDA_CHECK(cudaDeviceSynchronize());
        const gemma::KvCache& cache = speculative.cache(0);
        const std::int32_t end      = speculative.position(0);
        const std::int32_t ring     = cache.ring_tokens();
        std::int32_t wrong_slots    = 0;
        for (std::size_t layer = 0; layer < config.num_hidden_layers; ++layer) {
            const bool sliding         = config.sliding_attention(layer);
            const std::int32_t entries = sliding ? ring : end;
            std::vector<std::int32_t> held(static_cast<std::size_t>(entries));
            CUDA_CHECK(cudaMemcpy(held.data(), cache.positions(layer).data,
                                  held.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost));
            const std::int32_t first =
                sliding ? std::max(0, end - static_cast<std::int32_t>(config.sliding_window)) : 0;
            for (std::int32_t position = first; position < end; ++position) {
                if (held[static_cast<std::size_t>(cache.slot(layer, position))] != position) {
                    ++wrong_slots;
                }
            }
        }
        if (wrong_slots != 0) {
            std::cerr << "gemma4 mtp: " << wrong_slots
                      << " cache slots do not hold their committed position\n";
            ++failures;
        }

        // The baseline: plain greedy decode on a Program without drafts.
        gemma::Program plain(*model, capacity, 1, device);
        plain.reset(0);
        plain.prefill(0, prompt, execution);
        std::vector<std::int32_t> baseline{plain.sample(0, execution)};
        while (static_cast<std::int32_t>(baseline.size()) < kOutput) {
            plain.decode(0, baseline.back(), execution);
            baseline.push_back(plain.sample(0, execution));
        }

        const Agreement mtp_agreement      = agreement(plain, prompt, mtp, execution);
        const Agreement baseline_agreement = agreement(plain, prompt, baseline, execution);
        std::int32_t same = 0;
        while (same < kOutput && mtp[same] == baseline[same]) ++same;
        std::cout << "gemma4 mtp: " << rounds << " rounds, " << accepted << " of " << drafted
                  << " drafts accepted; " << same << " leading tokens equal the plain decode's\n"
                  << "  disagreement with one fresh pass: MTP " << mtp_agreement.mismatches << " of "
                  << kOutput << " (worst margin " << mtp_agreement.worst_margin << "), plain "
                  << baseline_agreement.mismatches << " (worst margin "
                  << baseline_agreement.worst_margin << ")\n";
        // The plain decode is the noise floor of route rounding; MTP may meet it but not exceed it by
        // more than a handful of near-ties.
        if (mtp_agreement.mismatches > baseline_agreement.mismatches + 3) {
            std::cerr << "gemma4 mtp: MTP disagrees with the target more than plain decode does\n";
            ++failures;
        }
        if (drafted == 0 || accepted == drafted) {
            std::cerr << "gemma4 mtp: the run never rolled back, so it tested nothing\n";
            ++failures;
        }
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "gemma4 mtp: " << error.what() << '\n';
        return 1;
    }
}
