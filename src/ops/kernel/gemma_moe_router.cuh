// Implements: include/ninfer/ops/gemma_moe_router.h
//
// One CTA per token. The scoring, the selection and the weights are one pass because the CTA holds
// the whole hidden row and the whole expert logit vector in shared memory: the admitted geometry is
// H=2816 and E=128, so the row is 11 KB and the logits are 512 B.
//
// The selection is a K-pass argmax over the expert probabilities with the already-selected experts
// masked out. Ties compare equal floats exactly and resolve to the lower expert id, which is what
// scanning every pass in ascending id order with a strict `>` gives.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops::detail {

// The (value, id) pair a pass reduces to. The comparison is the whole tie-break contract.
struct RouterBest {
    float value;
    int id;
};

__device__ __forceinline__ RouterBest router_better(RouterBest a, RouterBest b) {
    if (b.value > a.value) { return b; }
    if (b.value < a.value) { return a; }
    return b.id < a.id ? b : a;
}

// Reduces one (value, id) pair per thread into every thread's copy of the block's best.
__device__ __forceinline__ RouterBest router_block_argmax(RouterBest mine, RouterBest* scratch) {
    scratch[threadIdx.x] = mine;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (static_cast<int>(threadIdx.x) < stride) {
            scratch[threadIdx.x] =
                router_better(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    return scratch[0];
}

__device__ __forceinline__ float router_block_sum(float mine, float* scratch) {
    scratch[threadIdx.x] = mine;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (static_cast<int>(threadIdx.x) < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    return scratch[0];
}

/**
 * `shared` holds, in order: [hidden] scaled activations, [experts] probabilities, [experts]
 * selection flags, one RouterBest per thread, then one float per thread of reduction scratch.
 *
 * blockDim.x must be a power of two, at least `top_k`, and at most the launch's thread count.
 */
__global__ void gemma_moe_router_kernel(const __nv_bfloat16* __restrict__ x,
                                        const __nv_bfloat16* __restrict__ router,
                                        const __nv_bfloat16* __restrict__ router_scale,
                                        const __nv_bfloat16* __restrict__ per_expert_scale,
                                        float eps, int hidden, int experts, int top_k, int tokens,
                                        std::int32_t* __restrict__ ids,
                                        float* __restrict__ weights) {
    extern __shared__ float shared[];
    float* scaled          = shared;
    float* probabilities   = scaled + hidden;
    std::uint8_t* selected = reinterpret_cast<std::uint8_t*>(probabilities + experts);
    RouterBest* ranks      = reinterpret_cast<RouterBest*>(selected + experts);
    float* scratch         = reinterpret_cast<float*>(ranks + blockDim.x);

    const int token = blockIdx.x;
    if (token >= tokens) { return; }
    const __nv_bfloat16* row = x + static_cast<std::size_t>(token) * hidden;

    // 1. The weightless RMS: the unit-RMS row, then router_scale and H^-1/2. The norm carries no
    //    weight and no gain of its own, which is why it is not the learned-weight rmsnorm form.
    float squares = 0.0F;
    for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
        const float value = __bfloat162float(row[h]);
        squares += value * value;
    }
    const float mean    = router_block_sum(squares, scratch) / static_cast<float>(hidden);
    const float inverse = rsqrtf(mean + eps);
    const float root    = rsqrtf(static_cast<float>(hidden));
    for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
        scaled[h] = __bfloat162float(row[h]) * inverse * __bfloat162float(router_scale[h]) * root;
    }
    __syncthreads();

    // 2. The router projection: one expert per thread, accumulated in FP32 over the whole row.
    for (int e = threadIdx.x; e < experts; e += blockDim.x) {
        const __nv_bfloat16* weights_row = router + static_cast<std::size_t>(e) * hidden;
        float accumulator                = 0.0F;
        for (int h = 0; h < hidden; ++h) {
            accumulator += __bfloat162float(weights_row[h]) * scaled[h];
        }
        probabilities[e] = accumulator;
    }
    __syncthreads();

    // 3. Softmax over the expert axis.
    float maximum = -CUDART_INF_F;
    for (int e = threadIdx.x; e < experts; e += blockDim.x) {
        maximum = fmaxf(maximum, probabilities[e]);
    }
    maximum     = router_block_argmax(RouterBest{maximum, 0}, ranks).value;
    float total = 0.0F;
    for (int e = threadIdx.x; e < experts; e += blockDim.x) {
        const float value = __expf(probabilities[e] - maximum);
        probabilities[e]  = value;
        total += value;
    }
    total = router_block_sum(total, scratch);
    for (int e = threadIdx.x; e < experts; e += blockDim.x) { probabilities[e] /= total; }
    for (int e = threadIdx.x; e < experts; e += blockDim.x) { selected[e] = 0; }
    __syncthreads();

    // 4. The K ranks, each a masked argmax over the probabilities. `ranks` is the reduction's own
    //    scratch and the next pass overwrites it, so the chosen ranks are kept out of it: only
    //    thread 0 needs them and it holds them locally. top_k <= 16 is a launch precondition.
    RouterBest chosen_ranks[16];
    for (int rank = 0; rank < top_k; ++rank) {
        RouterBest mine{-CUDART_INF_F, experts};
        for (int e = threadIdx.x; e < experts; e += blockDim.x) {
            if (selected[e] == 0) { mine = router_better(mine, RouterBest{probabilities[e], e}); }
        }
        const RouterBest chosen = router_block_argmax(mine, ranks);
        if (threadIdx.x == 0) {
            selected[chosen.id] = 1;
            chosen_ranks[rank]  = chosen;
        }
        __syncthreads();
    }

    // 5. Renormalize the selected probabilities to sum to one, then apply the per-expert scale. The
    //    order is the contract: the scale multiplies an already-normalized weight, so the applied
    //    weights do not sum to one when the scales differ.
    if (threadIdx.x == 0) {
        float sum = 0.0F;
        for (int rank = 0; rank < top_k; ++rank) { sum += chosen_ranks[rank].value; }
        const float inverse_sum = sum > 0.0F ? 1.0F / sum : 0.0F;
        for (int rank = 0; rank < top_k; ++rank) {
            const int expert                                     = chosen_ranks[rank].id;
            ids[static_cast<std::size_t>(rank) * tokens + token] = expert;
            weights[static_cast<std::size_t>(rank) * tokens + token] =
                chosen_ranks[rank].value * inverse_sum * __bfloat162float(per_expert_scale[expert]);
        }
    }
}

} // namespace ninfer::ops::detail
