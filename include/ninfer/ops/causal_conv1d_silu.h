#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Depthwise causal width-4 convolution followed by SiLU. Let u[c,-3..-1] be the three values in
 * the input state and u[c,t]=x[c,t] for t>=0. Then
 *
 *   ideal[c,t] = SiLU(sum_{j=0..3} weight[c,j] * u[c,t-3+j]).
 *
 * `x` and `out` are contiguous BF16 [C,T] and a state is contiguous BF16 [C,3] ordered oldest to
 * newest, in this tree's own axis order, which puts the innermost axis first: element (c,t) of x is
 * at the flat index `t*C + c`. `weight` is contiguous BF16 [C,4] in that same order, so it is
 * *tap-major*: the tap j of channel c is at the flat index `j*C + c` -- the payload is four runs of
 * C channels, one run per tap -- and `weight[c,j]` in the formula above denotes that element. The
 * kernels read it that way. That is *not* the checkpoint's order: its `mixer.conv1d.weight` is
 * `[conv_channels,1,conv_kernel]`, whose own flattening is `c*4 + j`, so binding the stored payload
 * as it is stored yields a shape-valid [C,4] tensor in which every tap of every channel belongs to
 * another channel -- a silently wrong model rather than a refusal. The transpose belongs to whoever
 * binds the parameter.
 *
 * The wrapper requires the declared shape to be exactly [C,4] (ne[0] = C), which refuses the
 * checkpoint's own declaration ([4,1,C], or a flattened [4,C]) whenever C != 4. Two bindings cannot
 * be refused and are stated here instead: at C == 4 the two declarations are the same tensor and
 * differ only in the payload, and a checkpoint-order payload inside a tensor that already declares
 * [C,4] carries no shape evidence at all. The flat index above is therefore the whole contract for
 * the order, and a caller that reshapes a stored `[C,1,4]` parameter has to transpose it.
 *
 * The oracle evaluates `ideal` naively in FP64 from the represented inputs. The BF16 output is
 * promoted and compared directly with that result;
 * output storage rounding belongs to the Op's numerical criterion, not the oracle. Kernel
 * accumulator and staging precision are implementation choices. Input, weight, output, and state
 * storage do not overlap except for the explicitly allowed exact alias between state input and
 * state output. No caller workspace is used. T may be any positive value.
 *
 * Every entry also has a biased overload that takes a `const Tensor* bias`, a contiguous BF16 [C]
 * row added to the convolution sum before the fused activation:
 *
 *   ideal[c,t] = SiLU(bias[c] + sum_{j=0..3} weight[c,j] * u[c,t-3+j]).
 *
 * The bias has to be inside the Op because the activation is fused: SiLU(conv + bias) is not
 * SiLU(conv) + bias, so a call site cannot fold it in afterwards. This is the depthwise
 * convolution of a mixer that carries a convolution bias while its projections do not. A null
 * pointer is exactly the bias-free entry above, and a set bias is read-only, so it must not share
 * storage with any operand the call reads or writes. Everything else - the state rules, the
 * destinations, and the numerical criterion - is shared with the bias-free entry.
 */

// Biased forms of the two state forms. `bias` may be null, which is exactly the bias-free call.
void causal_conv1d_silu(const Tensor& x, const Tensor& weight, const Tensor* bias,
                        Tensor& conv_state, Tensor& out, cudaStream_t stream);
void causal_conv1d_silu(const Tensor& x, const Tensor& weight, const Tensor* bias,
                        const Tensor& conv_state_in, Tensor& conv_state_out, Tensor& out,
                        cudaStream_t stream);

// Reads conv_state as the initial window and replaces it with the final three values after x.
void causal_conv1d_silu(const Tensor& x, const Tensor& weight, Tensor& conv_state, Tensor& out,
                        cudaStream_t stream);

// Distinct-state form. conv_state_in and conv_state_out may be disjoint or exactly the same
// storage; conv_state_out receives the trailing width-3 window of concat(conv_state_in,x).
void causal_conv1d_silu(const Tensor& x, const Tensor& weight, const Tensor& conv_state_in,
                        Tensor& conv_state_out, Tensor& out, cudaStream_t stream);

// Split-output form. `ideal` is unchanged; its rows are written to three destinations instead of
// one, partitioned in row order as out0, out1, out2. Every destination is contiguous BF16 with the
// column count of x, and every operand, a set bias included, is four-byte aligned. The supported
// row profiles are (2048, 2048, 4096) over C = 8192 and (2048, 2048, 6144) over C = 10240; any
// other profile is rejected. conv_state_in and conv_state_out follow the family rule above:
// disjoint or exactly the same storage. No destination may overlap another, x, weight, bias, or
// either state.
// Biased form of the split entry; a null bias is the bias-free call.
void causal_conv1d_silu_split(const Tensor& x, const Tensor& weight, const Tensor* bias,
                              const Tensor& conv_state_in, Tensor& conv_state_out, Tensor& out0,
                              Tensor& out1, Tensor& out2, cudaStream_t stream);
void causal_conv1d_silu_split(const Tensor& x, const Tensor& weight, const Tensor& conv_state_in,
                              Tensor& conv_state_out, Tensor& out0, Tensor& out1, Tensor& out2,
                              cudaStream_t stream);

/**
 * Snapshot form for B independent sequences. `x` and `out` are contiguous BF16 [C,W,B],
 * `conv_states` is contiguous BF16 [C,3,Slots], and `initial_state_slots` and
 * `snapshot_base_slots` are contiguous I32 [B]. `valid_columns` is either contiguous I32 [B],
 * with every value in [1,W], or an empty Tensor meaning every row has W valid columns. B=1
 * accepts every positive W; B=2..8 accepts W=1..16.
 *
 * Row b starts from the window selected by initial_state_slots[b]. After valid column j, the new
 * window is written to snapshot_base_slots[b]+j. Invalid-tail output columns are exact BF16 zero
 * and do not mutate state. The caller reserves the complete [base,base+W) interval for every row;
 * all row reservations are disjoint and no row overwrites another row's initial slot. A row's own
 * initial slot may lie in its destination reservation. `conv_states` is the only persistent state
 * mutated and must not overlap x, weight, bias, metadata, or out.
 */
void causal_conv1d_silu_snapshot(const Tensor& x, const Tensor& weight, const Tensor* bias,
                                 Tensor& conv_states, const Tensor& valid_columns,
                                 const Tensor& initial_state_slots,
                                 const Tensor& snapshot_base_slots, Tensor& out,
                                 cudaStream_t stream);
void causal_conv1d_silu_snapshot(const Tensor& x, const Tensor& weight, Tensor& conv_states,
                                 const Tensor& valid_columns, const Tensor& initial_state_slots,
                                 const Tensor& snapshot_base_slots, Tensor& out,
                                 cudaStream_t stream);

} // namespace ninfer::ops
