#pragma once
#include "ops/linear/fp8/fp8_launch.h"
#include "ops/linear/fp8/fp8_a8_plan.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {
struct Fp8LinearShape {
    std::int32_t n, k;
    Fp8Launch a16;
    void (*a8)(const Tensor&, const Weight&, Tensor&, Fp8A8Workspace, cudaStream_t);
    bool (*uses_a8)(std::int32_t min_tokens, std::int32_t max_tokens);
    std::size_t (*partial_capacity_bytes)(std::int32_t max_tokens) = nullptr;
};

extern const Fp8LinearShape kFp8N14336K5120;
extern const Fp8LinearShape kFp8N16384K5120;
extern const Fp8LinearShape kFp8N8192K5376;
// The G0 layout keeps the MLP at FP8 rather than NVFP4 in the sensitive layers.
extern const Fp8LinearShape kFp8N21504K5376;
extern const Fp8LinearShape kFp8N5376K21504;
extern const Fp8LinearShape kFp8N4096K5376;
extern const Fp8LinearShape kFp8N16384K5376;
extern const Fp8LinearShape kFp8N2048K5376;
extern const Fp8LinearShape kFp8N5376K8192;
extern const Fp8LinearShape kFp8N5376K16384;
extern const Fp8LinearShape kFp8N262144K5376;
extern const Fp8LinearShape kFp8N34816K5120;
extern const Fp8LinearShape kFp8N5120K6144;
extern const Fp8LinearShape kFp8N5120K17408;
extern const Fp8LinearShape kFp8N248320K5120;
// Nemotron-H's dense FP8 projections: the mamba mixer's two, the attention mixer's four, and the
// head. n = 256 for the key/value pair, 4096 for query/output and the mamba output, 18560 for the
// mamba input, and the vocabulary's 131072 for the head; k = 4096 everywhere but the mamba output's
// 8192.
extern const Fp8LinearShape kFp8N256K4096;
extern const Fp8LinearShape kFp8N4096K4096;
extern const Fp8LinearShape kFp8N4096K8192;
extern const Fp8LinearShape kFp8N18560K4096;
extern const Fp8LinearShape kFp8N131072K4096;
} // namespace ninfer::ops::detail
