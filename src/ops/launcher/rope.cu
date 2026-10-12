// ninfer::ops - rope launcher: private token-count tuning and generic fallback.
#include "ops/launcher/rope.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/kernel/rope.cuh"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kLargeBlock               = 256;
constexpr int kFullChunkBlock           = 192;
constexpr int kSmallBlock               = 128;
constexpr int kDefaultChunkTargetTokens = 1024;
// Qualified residency of this 256-thread instance on the sm_120a target.
constexpr int kLargeBlockResidentCtasPerSm = 6;

template <RopeKernelMode Mode>
inline constexpr bool kTextMode =
    Mode == RopeKernelMode::Text1D || Mode == RopeKernelMode::TextMrope ||
    Mode == RopeKernelMode::DflashText1D;

std::int64_t token_stride(const Tensor* tensor) {
    return tensor == nullptr ? 0 : tensor->nb[2] / static_cast<std::int64_t>(sizeof(__nv_bfloat16));
}

bool bf16x2_aligned(const Tensor& tensor) {
    return (reinterpret_cast<std::uintptr_t>(tensor.data) & (alignof(__nv_bfloat162) - 1)) == 0 &&
           tensor.nb[2] % static_cast<std::int64_t>(alignof(__nv_bfloat162)) == 0;
}

template <RopeKernelMode Mode, int QHeads, int KHeads>
void launch_fixed_block(const Tensor& positions, Tensor* q, Tensor* k, int block,
                        cudaStream_t stream) {
    const int tokens = positions.ne[0];
    rope_fixed_kernel<Mode, QHeads, KHeads><<<tokens, block, 0, stream>>>(
        static_cast<const std::int32_t*>(positions.data),
        q == nullptr ? nullptr : static_cast<__nv_bfloat16*>(q->data),
        k == nullptr ? nullptr : static_cast<__nv_bfloat16*>(k->data), tokens, token_stride(q),
        token_stride(k));
}

template <RopeKernelMode Mode, int QHeads, int KHeads>
void launch_fixed(const Tensor& positions, Tensor* q, Tensor* k, DeviceExecutionView execution) {
    const int tokens = positions.ne[0];
    int block        = kSmallBlock;
    if constexpr (kTextMode<Mode>) {
        if (tokens <= 6) {
            block = (QHeads + KHeads) * 32;
        } else if (tokens <= static_cast<std::int64_t>(execution.multiprocessor_count) *
                                 kLargeBlockResidentCtasPerSm) {
            block = kLargeBlock;
        } else if (tokens <= kDefaultChunkTargetTokens) {
            block = kFullChunkBlock;
        }
        const int head_warps = (QHeads + KHeads) * 32;
        if (block > head_warps) { block = head_warps; }
        if (block > 1024) { block = 1024; }
    }
    launch_fixed_block<Mode, QHeads, KHeads>(positions, q, k, block, execution.stream);
}

template <int HeadsPerBlock, int QHeads, int KHeads>
void launch_dflash_split(const Tensor& positions, Tensor* q, Tensor* k, cudaStream_t stream) {
    constexpr int kGroups = (QHeads + KHeads + HeadsPerBlock - 1) / HeadsPerBlock;
    constexpr int kBlock  = HeadsPerBlock <= 2 ? 64 : HeadsPerBlock * 32;
    const int tokens      = positions.ne[0];
    rope_fixed_split_kernel<RopeKernelMode::DflashText1D, QHeads, KHeads, HeadsPerBlock>
        <<<tokens * kGroups, kBlock, 0, stream>>>(
            static_cast<const std::int32_t*>(positions.data),
            q == nullptr ? nullptr : static_cast<__nv_bfloat16*>(q->data),
            k == nullptr ? nullptr : static_cast<__nv_bfloat16*>(k->data), tokens, token_stride(q),
            token_stride(k));
}

bool launch_fixed_pair(const Tensor& positions, int rotary_dim, int rotary_pairs, float theta,
                       Tensor& q, Tensor& k, DeviceExecutionView execution) {
    const auto stream = execution.stream;
    // The specialized kernels span the whole rotary dimension; a partial profile takes the generic
    // kernel, which is the only one that bounds its pair loop and its coefficient cache.
    if (rotary_pairs != rotary_dim / 2) { return false; }
    if (!bf16x2_aligned(q) || !bf16x2_aligned(k)) { return false; }
    const int axes = positions.ne[1];
    if (axes == 1 && q.ne[0] == 128 && rotary_dim == 128 && theta == 1.0e7F && q.ne[1] == 32 &&
        k.ne[1] == 8) {
        const int tokens = positions.ne[0];
        if (tokens <= 16) {
            launch_dflash_split<5, 32, 8>(positions, &q, &k, stream);
        } else if (tokens <= 400) {
            launch_dflash_split<8, 32, 8>(positions, &q, &k, stream);
        } else {
            launch_fixed_block<RopeKernelMode::DflashText1D, 32, 8>(positions, &q, &k, 160, stream);
        }
        return true;
    }
    if (rotary_dim == 64 && theta == 1.0e7F) {
        if (q.ne[1] == 24 && k.ne[1] == 4) {
            if (axes == 1) {
                launch_fixed<RopeKernelMode::Text1D, 24, 4>(positions, &q, &k, execution);
                return true;
            }
            if (axes == 3) {
                launch_fixed<RopeKernelMode::TextMrope, 24, 4>(positions, &q, &k, execution);
                return true;
            }
        }
        if (q.ne[1] == 16 && k.ne[1] == 2) {
            if (axes == 1) {
                launch_fixed<RopeKernelMode::Text1D, 16, 2>(positions, &q, &k, execution);
                return true;
            }
            if (axes == 3) {
                launch_fixed<RopeKernelMode::TextMrope, 16, 2>(positions, &q, &k, execution);
                return true;
            }
        }
    }
    if (axes == 2 && rotary_dim == 72 && theta == 10'000.0F && q.ne[1] == 16 && k.ne[1] == 16) {
        launch_fixed<RopeKernelMode::Vision2D, 16, 16>(positions, &q, &k, execution);
        return true;
    }
    return false;
}

template <RopeKernelMode Mode, int Heads>
void launch_fixed_single(const Tensor& positions, Tensor& x, DeviceExecutionView execution) {
    launch_fixed<Mode, Heads, 0>(positions, &x, nullptr, execution);
}

template <int Heads>
bool launch_text_single(const Tensor& positions, int axes, Tensor& x,
                        DeviceExecutionView execution) {
    if (x.ne[1] != Heads) { return false; }
    if (axes == 1) {
        launch_fixed_single<RopeKernelMode::Text1D, Heads>(positions, x, execution);
        return true;
    }
    if (axes == 3) {
        launch_fixed_single<RopeKernelMode::TextMrope, Heads>(positions, x, execution);
        return true;
    }
    return false;
}

bool launch_fixed_single_dispatch(const Tensor& positions, int rotary_dim, int rotary_pairs,
                                  float theta, Tensor& x, DeviceExecutionView execution) {
    if (rotary_pairs != rotary_dim / 2) { return false; }
    if (!bf16x2_aligned(x)) { return false; }
    const int axes = positions.ne[1];
    if (axes == 1 && x.ne[0] == 128 && rotary_dim == 128 && theta == 1.0e7F) {
        if (x.ne[1] == 32) {
            launch_fixed_single<RopeKernelMode::DflashText1D, 32>(positions, x, execution);
            return true;
        }
        if (x.ne[1] == 8) {
            launch_fixed_single<RopeKernelMode::DflashText1D, 8>(positions, x, execution);
            return true;
        }
    }
    if (rotary_dim == 64 && theta == 1.0e7F) {
        if (launch_text_single<24>(positions, axes, x, execution) ||
            launch_text_single<4>(positions, axes, x, execution) ||
            launch_text_single<16>(positions, axes, x, execution) ||
            launch_text_single<2>(positions, axes, x, execution)) {
            return true;
        }
    }
    if (axes == 2 && rotary_dim == 72 && theta == 10'000.0F && x.ne[1] == 16) {
        launch_fixed_single<RopeKernelMode::Vision2D, 16>(positions, x, execution);
        return true;
    }
    return false;
}

// The per-pair frequencies of one generic geometry. A Program issues the same few geometries on
// every step, so each launching thread keeps the last tables it evaluated.
const RopeGenericFrequencies& generic_frequencies(int axes, int head_dim, int rotary_dim,
                                                  int rotary_pairs, float theta) {
    struct Entry {
        int axes;
        int head_dim;
        int rotary_dim;
        int rotary_pairs;
        float theta;
        RopeGenericFrequencies table;
    };
    constexpr std::size_t kEntries = 4;
    thread_local std::array<Entry, kEntries> entries{};
    thread_local std::size_t filled = 0;
    thread_local std::size_t next   = 0;
    for (std::size_t i = 0; i < filled; ++i) {
        const Entry& e = entries[i];
        if (e.axes == axes && e.head_dim == head_dim && e.rotary_dim == rotary_dim &&
            e.rotary_pairs == rotary_pairs && e.theta == theta) {
            return e.table;
        }
    }
    Entry& e       = entries[next];
    next           = (next + 1) % kEntries;
    filled         = filled < kEntries ? filled + 1 : filled;
    e.axes         = axes;
    e.head_dim     = head_dim;
    e.rotary_dim   = rotary_dim;
    e.rotary_pairs = rotary_pairs;
    e.theta        = theta;
    e.table        = {};
    for (int pair = 0; pair < rotary_pairs; ++pair) {
        int axis        = 0;
        double exponent = 0.0;
        generic_axis_exponent(axes, head_dim, rotary_dim, pair, &axis, &exponent);
        e.table.inv_frequency[pair] = std::pow(static_cast<double>(theta), exponent);
    }
    return e.table;
}

void launch_generic(const Tensor& positions, int rotary_dim, int rotary_pairs, float theta,
                    Tensor* q, Tensor* k, cudaStream_t stream) {
    constexpr int block = kRopeGenericHeadsPerBlock * 32;
    Tensor& sample      = q != nullptr ? *q : *k;
    const int tokens    = sample.ne[2];
    const int heads     = (q == nullptr ? 0 : q->ne[1]) + (k == nullptr ? 0 : k->ne[1]);
    const int axes      = positions.ne[1];
    const dim3 grid(static_cast<unsigned>(tokens),
                    static_cast<unsigned>((heads + kRopeGenericHeadsPerBlock - 1) /
                                          kRopeGenericHeadsPerBlock));
    rope_generic_kernel<<<grid, block, 0, stream>>>(
        static_cast<const std::int32_t*>(positions.data), axes,
        q == nullptr ? nullptr : static_cast<__nv_bfloat16*>(q->data),
        k == nullptr ? nullptr : static_cast<__nv_bfloat16*>(k->data), sample.ne[0], rotary_dim,
        rotary_pairs, theta, q == nullptr ? 0 : q->ne[1], k == nullptr ? 0 : k->ne[1], tokens,
        token_stride(q), token_stride(k),
        generic_frequencies(axes, sample.ne[0], rotary_dim, rotary_pairs, theta));
}

} // namespace

void rope_launch(const Tensor& positions, int rotary_dim, int rotary_pairs, float theta,
                 Tensor& q, Tensor& k, DeviceExecutionView execution) {
    if (!launch_fixed_pair(positions, rotary_dim, rotary_pairs, theta, q, k, execution)) {
        launch_generic(positions, rotary_dim, rotary_pairs, theta, &q, &k, execution.stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

void rope_single_launch(const Tensor& positions, int rotary_dim, int rotary_pairs, float theta,
                        Tensor& x, DeviceExecutionView execution) {
    if (!launch_fixed_single_dispatch(positions, rotary_dim, rotary_pairs, theta, x, execution)) {
        launch_generic(positions, rotary_dim, rotary_pairs, theta, &x, nullptr, execution.stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
