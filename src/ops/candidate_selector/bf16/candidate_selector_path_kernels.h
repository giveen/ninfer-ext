#pragma once
#include "ops/candidate_selector/bf16/candidate_selector_path_plan.h"

namespace ninfer::ops::detail {
// Requires: `configs` holds candidate_ids.ne[2] entries and is not null, steps and batch are
// candidate_ids.ne[1] and ne[2], workspace.edges is read only for SelectorRoute::Lattice, and no
// operand or workspace range aliases another. The Op wrapper enforces all of these; the launcher
// checks nothing itself.
void candidate_selector_path_launch(SelectorRoute route, const Tensor& candidate_ids,
                                    const Tensor& unary_scores, const Tensor& projected_hidden,
                                    const Tensor& anchors, const Tensor& predecessor_codebook,
                                    const Tensor& successor_codebook, const Tensor& base_positions,
                                    const SamplingConfig* configs, Tensor& drafts,
                                    Tensor& proposal_q, const SelectorWorkspace& workspace,
                                    cudaStream_t stream);
} // namespace ninfer::ops::detail
