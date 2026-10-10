#include "ops/linear/bf16/bf16_launch.cuh"
#include "ops/linear/bf16/bf16_shapes.h"

#include <stdexcept>

// The Gemma 4 assistant drafter (hidden 1024), which runs one token per step: single-token GEMV only.
// Wide outputs over K=1024 take eight rows per warp; the 1024-row outputs over long K split each row
// across warps so the grid still covers the device. Untuned beyond that.

namespace ninfer::ops::detail {
namespace {
using Wide   = Bf16GemvSchedule<4, 1, 8, 8, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                                Bf16PhaseOrder::RowSwizzled, 1, 1, 1, 2>;
using Narrow = Bf16GemvSchedule<8, 8, 1, 8, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                                Bf16PhaseOrder::RowSwizzled, 1, 2, 1, 1>;
using Input  = Bf16GemvSchedule<4, 2, 1, 8, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                                Bf16PhaseOrder::RowSwizzled, 1, 2, 1, 1>;

template <class Geometry, class Schedule>
Bf16Launch single_token(std::int32_t tokens) {
    if (tokens != 1) {
        throw std::invalid_argument("bf16 linear: the Gemma drafter shapes run one token at a time");
    }
    return launch_bf16_gemv<Geometry, Schedule>;
}
} // namespace

Bf16Launch select_bf16_n1024_k10752(std::int32_t t) {
    return single_token<Bf16Geometry<1024, 10752>, Input>(t);
}
Bf16Launch select_bf16_n8192_k1024(std::int32_t t) {
    return single_token<Bf16Geometry<8192, 1024>, Wide>(t);
}
Bf16Launch select_bf16_n16384_k1024(std::int32_t t) {
    return single_token<Bf16Geometry<16384, 1024>, Wide>(t);
}
Bf16Launch select_bf16_n1024_k8192(std::int32_t t) {
    return single_token<Bf16Geometry<1024, 8192>, Narrow>(t);
}
Bf16Launch select_bf16_n1024_k16384(std::int32_t t) {
    return single_token<Bf16Geometry<1024, 16384>, Narrow>(t);
}
Bf16Launch select_bf16_n262144_k1024(std::int32_t t) {
    return single_token<Bf16Geometry<262144, 1024>, Wide>(t);
}
Bf16Launch select_bf16_n5376_k1024(std::int32_t t) {
    return single_token<Bf16Geometry<5376, 1024>, Wide>(t);
}

} // namespace ninfer::ops::detail
