#include "ops/linear/q6/q6_launch.h"
#include "ops/linear/q6/q6_simt_launch.cuh"
#include "ops/linear/q6/q6_instances.cuh"

namespace ninfer::ops::detail {

void launch_q6_a16_simt_r8_t4(const Tensor& x, const Weight& weight, Tensor& out,
                              cudaStream_t stream) {
    launch_q6_a16_simt<q6_instances::SimtR8T4>(
        q6_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n}, LinearIdentityEpilogue{},
        stream);
}

void launch_q6_a16_simt_r4_t4_w2_g16(const Tensor& x, const Weight& weight, Tensor& out,
                                     cudaStream_t stream) {
    launch_q6_a16_simt<q6_instances::SimtR4T4W2G16>(
        q6_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n}, LinearIdentityEpilogue{},
        stream);
}

void launch_q6_a16_simt_r4_t8_w2_g16(const Tensor& x, const Weight& weight, Tensor& out,
                                     cudaStream_t stream) {
    launch_q6_a16_simt<q6_instances::SimtR4T8W2G16>(
        q6_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n}, LinearIdentityEpilogue{},
        stream);
}

void launch_q6_a16_gemv_r4_w2_g16(const Tensor& x, const Weight& weight, Tensor& out,
                                  cudaStream_t stream) {
    launch_q6_a16_gemv<q6_instances::GemvR4W2G16>(
        q6_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n}, LinearIdentityEpilogue{},
        stream);
}

namespace {
template <class Schedule>
void launch_q6_a16_direct_instance(const Tensor& x, const Weight& weight, Tensor& out,
                                   cudaStream_t stream) {
    launch_q6_a16_direct_simt<Schedule>(
        q6_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n}, LinearIdentityEpilogue{},
        stream);
}
} // namespace

void launch_q6_a16_direct_r1_t1_w4_k5376(const Tensor& x, const Weight& weight, Tensor& out,
                                         cudaStream_t stream) {
    launch_q6_a16_direct_instance<q6_instances::DirectR1T1W4K5376>(x, weight, out, stream);
}

void launch_q6_a16_direct_r1_t1_w4_k8192(const Tensor& x, const Weight& weight, Tensor& out,
                                         cudaStream_t stream) {
    launch_q6_a16_direct_instance<q6_instances::DirectR1T1W4K8192>(x, weight, out, stream);
}

void launch_q6_a16_direct_r1_t1_w4_k16384(const Tensor& x, const Weight& weight, Tensor& out,
                                          cudaStream_t stream) {
    launch_q6_a16_direct_instance<q6_instances::DirectR1T1W4K16384>(x, weight, out, stream);
}

void launch_q6_a16_direct_r1_t1_w4_k21504(const Tensor& x, const Weight& weight, Tensor& out,
                                          cudaStream_t stream) {
    launch_q6_a16_direct_instance<q6_instances::DirectR1T1W4K21504>(x, weight, out, stream);
}

} // namespace ninfer::ops::detail
