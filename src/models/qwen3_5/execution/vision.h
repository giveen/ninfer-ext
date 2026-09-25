#pragma once

#include "models/qwen3_5/program/internal.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "models/qwen3_5/program/vision_control.h"
#include "models/qwen3_5/program/planning/startup.h"
#include "models/qwen3_5/program/vision_prefill.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

using detail::VisionWorkspacePlan;
using detail::VisionPrefillPlan;
using detail::VisionUseSpan;

struct VisionItemView {
    std::span<const std::uint16_t> patches;
    const qwen3_5::VisionItemControl* control = nullptr;
};

class VisionContext {
public:
    VisionContext(DeviceContext& device, const execution::Parameters& parameters);

    [[nodiscard]] static std::size_t workspace_bytes(const VisionConfig& config,
                                                     const VisionParameters& parameters,
                                                     std::size_t patches, std::size_t merged_tokens,
                                                     const VisionWorkspacePlan& plan);
    [[nodiscard]] static VisionWorkspacePlan plan_workspace(const VisionConfig& config,
                                                            const VisionParameters& parameters,
                                                            std::uint32_t max_merged_tokens,
                                                            std::size_t general_capacity_bytes);

    [[nodiscard]] const VisionConfig& config() const noexcept { return config_; }
    [[nodiscard]] const VisionParameters& parameters() const noexcept { return parameters_; }

    // Output of `merged_tokens` tokens at `offset_bytes` into the handoff region.
    [[nodiscard]] static Tensor bind_output(DeviceSpan backing, const VisionWorkspacePlan& plan,
                                            std::size_t merged_tokens,
                                            std::size_t offset_bytes = 0);
    // Whether `control` can be encoded with its output at `offset_bytes`: the output and the final
    // merger scratch fit the workspace, and past offset 0 the encoder's scratch stays below the
    // handoff region so outputs already there survive.
    [[nodiscard]] static bool fits_output(const VisionConfig& config,
                                          const VisionParameters& parameters,
                                          const VisionWorkspacePlan& plan,
                                          const qwen3_5::VisionItemControl& control,
                                          std::size_t offset_bytes);
    // Bytes one packed output of `merged_tokens` occupies when outputs can sit back to back as
    // one [H, tokens] tensor (the column size keeps the workspace alignment); 0 otherwise.
    [[nodiscard]] static std::size_t packed_output_bytes(const VisionWorkspacePlan& plan,
                                                         std::size_t merged_tokens);
    void encode(const VisionItemView& item, Tensor& output, DeviceSpan backing,
                const VisionWorkspacePlan& plan, std::size_t offset_bytes = 0) const;

private:
    DeviceContext& ctx_;
    const VisionConfig& config_;
    const VisionParameters& parameters_;
};

// A prefill chunk and the encoded Vision items it may scatter: `scatter` lists the prompt
// positions of `embeddings`' columns, ascending, over every live item (empty for a text-only
// chunk). Items are those whose outputs are live, in prompt order. The spans stay valid until the
// next prepare_chunk.
struct VisionChunk {
    std::int32_t length = 0;
    std::span<const std::int32_t> scatter;
    std::span<const qwen3_5::VisionItemControl* const> items;
    Tensor embeddings;
};

class VisionPrefillSession {
public:
    VisionPrefillSession(DeviceContext& device, const execution::Parameters& parameters,
                         DeviceSpan workspace, const VisionWorkspacePlan& workspace_plan,
                         qwen3_5::PreparedPromptData& prompt, const VisionPrefillPlan& plan,
                         std::size_t& handoff_peak_bytes);

    [[nodiscard]] VisionChunk prepare_chunk(std::uint32_t begin, std::uint32_t nominal_length);
    void release_encoded_media_payloads() noexcept;
    void retire_handoff() noexcept;
    [[nodiscard]] double elapsed_seconds() const;

    [[nodiscard]] std::size_t active_handoff_bytes() const noexcept {
        return active_handoff_bytes_;
    }

private:
    DeviceContext& device_;
    DeviceSpan workspace_;
    const VisionWorkspacePlan& workspace_plan_;
    qwen3_5::PreparedPromptData& prompt_;
    const VisionPrefillPlan& plan_;
    std::size_t& handoff_peak_bytes_;
    VisionContext context_;
    // An encoded item whose output is live in the handoff region.
    struct LiveItem {
        std::uint32_t prepared_item                = 0;
        const qwen3_5::VisionItemControl* control  = nullptr;
        std::uint32_t end                          = 0; // prompt end of its span
        std::size_t offset                         = 0; // bytes into the handoff region
        std::size_t bytes                          = 0;
    };

    std::size_t next_use_ = 0;
    std::vector<LiveItem> live_;
    std::vector<std::int32_t> chunk_scatter_;
    std::vector<const qwen3_5::VisionItemControl*> chunk_items_;
    std::size_t active_handoff_bytes_ = 0;
    std::vector<std::uint32_t> encoded_payloads_pending_release_;
    std::vector<CudaEventTimer> timers_;
};

} // namespace ninfer::models::qwen3_5::execution
