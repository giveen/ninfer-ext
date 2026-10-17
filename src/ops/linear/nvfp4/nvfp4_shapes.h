#pragma once
#include "core/tensor.h"
#include "core/weight.h"
#include "ops/linear/nvfp4/nvfp4_launch.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"

#include <cstdint>

namespace ninfer::ops::detail {
struct Nvfp4LinearShape {
    std::int32_t n, k;
    Nvfp4Launch a16;
    void (*a4)(const Tensor&, const Weight&, Tensor&, Nvfp4W4a4Workspace, cudaStream_t);
    bool (*uses_a4)(std::int32_t min_tokens, std::int32_t max_tokens);
};

extern const Nvfp4LinearShape kNvfp4N14336K5120;
extern const Nvfp4LinearShape kNvfp4N16384K5120;
extern const Nvfp4LinearShape kNvfp4N34816K5120;
extern const Nvfp4LinearShape kNvfp4N5120K6144;
extern const Nvfp4LinearShape kNvfp4N5120K17408;
extern const Nvfp4LinearShape kNvfp4N43008K5376;
extern const Nvfp4LinearShape kNvfp4N5376K21504;
extern const Nvfp4LinearShape kNvfp4N21504K5376;
// Nemotron-H's shared expert, bound as an unfused up matrix and down matrix.
extern const Nvfp4LinearShape kNvfp4N5376K4096;
extern const Nvfp4LinearShape kNvfp4N4096K5376;
// The Lightning 30B DFlash companion's tap fusion and its MLP down projection, preserved as the
// ModelOpt NVFP4 the checkpoint stores.
extern const Nvfp4LinearShape kNvfp4N2688K16128;
extern const Nvfp4LinearShape kNvfp4N2688K6144;
extern const Nvfp4LinearShape kNvfp4N6144K2688;
} // namespace ninfer::ops::detail
