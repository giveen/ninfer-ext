#include "ops/linear/nvfp4/nvfp4_group_gemv.cuh"
#include "ops/linear/nvfp4/nvfp4_shapes.h"

namespace ninfer::ops::detail {
namespace {
// The Lightning 30B DFlash companion's MLP gate and up projections, bound unfused:
// n = 6144 = the companion's MLP width, k = 2688 = the tower's width. K is 168 groups of 16, which
// the chunked SIMT route's 256-wide phase cannot tile, so this shape takes the dedicated group
// route (nvfp4_group_gemv.cuh) at every token count. The A4 routes are absent: the draft takes the
// A16 route.
using Geometry = Nvfp4Geometry<6144, 2688>;
void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_nvfp4_group_gemv<Geometry>(x, weight, out, stream);
}
bool uses_a4(std::int32_t, std::int32_t) { return false; }
} // namespace

const Nvfp4LinearShape kNvfp4N6144K2688{6144, 2688, launch_a16, nullptr, uses_a4};
} // namespace ninfer::ops::detail
