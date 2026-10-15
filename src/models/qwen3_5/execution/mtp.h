#pragma once

#include "models/qwen3_5/execution/parameters.h"

namespace ninfer::models::qwen3_5::execution {

// Workspace sizing for the projection entry points below. `first`/`last` are the token-column
// bounds (minimum and maximum token count) of the corresponding call and must match it: the
// projection takes its actual extent from `hidden.ne[1]`, so planning a smaller range
// under-allocates the arena.

[[nodiscard]] std::size_t mtp_projection_workspace_bytes(const MtpProjectionParameters& parameters,
                                                         std::int32_t first, std::int32_t last);
[[nodiscard]] std::size_t mtp_kv_workspace_bytes(const MtpProjectionParameters& parameters,
                                                 const AttentionConfig& config, std::int32_t first,
                                                 std::int32_t last);
[[nodiscard]] std::size_t mtp_query_gate_workspace_bytes(const MtpProjectionParameters& parameters,
                                                         const AttentionConfig& config,
                                                         std::int32_t first, std::int32_t last);
void mtp_projection(const Tensor& hidden, const MtpProjectionParameters& parameters,
                    const AttentionConfig& config, Tensor& query, Tensor& gate, Tensor& key,
                    Tensor& value, WorkspaceArena& workspace, cudaStream_t stream);
void mtp_kv_projection(const Tensor& hidden, const MtpProjectionParameters& parameters,
                       const AttentionConfig& config, Tensor& key, Tensor& value,
                       WorkspaceArena& workspace, cudaStream_t stream);
void mtp_query_gate_projection(const Tensor& hidden, const MtpProjectionParameters& parameters,
                               const AttentionConfig& config, Tensor& query, Tensor& gate,
                               WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::models::qwen3_5::execution
