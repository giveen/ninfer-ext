#pragma once

// Offline quantization hook. The native EXL3 quantizer needs E[XᵀX] for every projection's input,
// which only the model's own execution produces. The model calls observe_projection() with the
// exact activation that feeds a projection, before that projection runs.
//
// This is not a serving facility: serving never installs an observer, so the empty check is the
// only cost on the normal path. The site and layer identify which parameters share the activation.

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <functional>

namespace ninfer::models::qwen3_5::execution {

enum class CalibrationSite : int {
    AttentionInput = 0, // input to attention q/k/gate/v
    AttentionOutput,    // input to the attention output projection
    GdnInput,           // input to GDN q/k/v/z
    GdnResidual,        // input to the GDN a/b control projections
    GdnOutput,          // input to the GDN output projection
    MlpInput,           // input to the MLP gate/up pair
    MlpActivation,      // input to the MLP down projection
    OutputHead,         // input to the vocabulary output head
    // The MTP layer consumes the trunk's final hidden state and the next-token embedding, so its
    // activations are not any Text layer's and need their own sites. Its MLP reuses MlpInput /
    // MlpActivation with the MTP layer sentinel (-1), which no Text layer uses.
    MtpStem,            // input to mtp/input_projection (the packed hidden + embedding)
    MtpAttentionInput,  // input to the MTP attention q/k/gate/v parent
    MtpAttentionOutput, // input to the MTP attention output projection
    Count,
};

// layer is the Text layer index, or a negative sentinel for calls outside a Text layer.
using CalibrationObserver =
    std::function<void(CalibrationSite, int layer, const Tensor& input, cudaStream_t stream)>;

inline CalibrationObserver& calibration_observer() {
    static CalibrationObserver observer;
    return observer;
}

inline void observe_projection(CalibrationSite site, int layer, const Tensor& input,
                               cudaStream_t stream) {
    if (calibration_observer()) { calibration_observer()(site, layer, input, stream); }
}

} // namespace ninfer::models::qwen3_5::execution
