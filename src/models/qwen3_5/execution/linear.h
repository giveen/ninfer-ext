#pragma once

#include "models/qwen3_5/execution/parameters.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"

namespace ninfer::models::qwen3_5::execution {

inline void project(const Tensor& input, const LinearParameters& p, Tensor& output,
                    WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope = workspace.scope();
    ops::linear(input, p.weight, output, p.policy, workspace, stream);
}

inline void project_add(const Tensor& input, const LinearParameters& p, Tensor& residual,
                        WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope = workspace.scope();
    ops::linear_add(input, p.weight, residual, p.policy, workspace, stream);
}

inline void project_swiglu(const Tensor& input, const LinearParameters& p, Tensor& output,
                           WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope = workspace.scope();
    ops::linear_swiglu(input, p.weight, output, p.policy, workspace, stream);
}

// Output-head logits of continuation hidden columns: the normalized hidden for Qwen3.5, the wide
// hyper-connection residual (through the output mixer) for Qwen4Exp.
void output_logits(const Parameters& parameters, const Tensor& hidden, Tensor& logits,
                   WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::models::qwen3_5::execution
