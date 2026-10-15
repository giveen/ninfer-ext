#pragma once
#include "core/tensor.h"
#include "core/arena.h"
#include "ninfer/ops/sampling.h"
#include <cuda_runtime.h>

namespace ninfer::ops::detail {
// Candidate width of the lattice kernel, which indexes edges[(column * kSelectorCandidates + p) *
// kSelectorCandidates + lane]; the edge buffer below is sized and laid out by it.
inline constexpr int kSelectorCandidates = 16;
enum class SelectorRoute { Direct, Lattice };

struct SelectorWorkspace {
    Tensor edges;
};

SelectorRoute candidate_selector_path_route(int steps, int batch);
const char* selector_route_name(SelectorRoute route);

template <class Allocator>
SelectorWorkspace allocate_selector_workspace(Allocator& allocator, SelectorRoute route, int steps,
                                              int batch) {
    SelectorWorkspace out;
    if (route == SelectorRoute::Lattice)
        out.edges = allocator.alloc(DType::FP32, {kSelectorCandidates, kSelectorCandidates, steps, batch});
    return out;
}

void candidate_selector_path_dispatch(const Tensor& candidate_ids, const Tensor& unary_scores,
                                      const Tensor& projected_hidden, const Tensor& anchors,
                                      const Tensor& predecessor_codebook,
                                      const Tensor& successor_codebook,
                                      const Tensor& base_positions, const SamplingConfig* configs,
                                      Tensor& drafts, Tensor& proposal_q, WorkspaceArena& workspace,
                                      cudaStream_t stream);
} // namespace ninfer::ops::detail
