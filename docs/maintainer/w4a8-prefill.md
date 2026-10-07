# W4A8 prefill for groupwise-int (active work)

Status: **active**. M1 (feasibility) and M2 (the production Q4 W4A8 `linear_swiglu` route) are
implemented and oracle-tested. In the engine the route is a modest win over A16 (1.03-1.09x); a
prototype of the same kernel reaches ~1.2x, so a gap remains. This records the measured evidence,
the design, and the remaining gap so the numbers are not re-derived. Remove or fold into
[op development](op-development.md) / [engine architecture](engine-architecture.md) when the route
is fully characterized.

## In-engine result (M2, `ninfer_q4_linear_swiglu_bench --policy a8`)

| `[34816,5120] x [5120,T]` | T=2048 | T=4096 | T=8192 |
|---|---:|---:|---:|
| A16 (`--policy a16`) | 202.5 | 176.1 | 174.4 TFLOP/s |
| **W4A8 (`--policy a8`)** | **208.7** | **192.4** | **187.1** |
| ratio | 1.03x | 1.09x | 1.07x |

The standalone prototype of the same kernel reaches ~242 TFLOP/s, so ~15-25% is lost inside the
engine. Profiling shows the in-engine kernel is L2-throughput bound (~93%) at a lower SM clock
(2.41 vs 2.67 GHz) and ~25% more elapsed cycles; the cause is not yet isolated. This is the open
work before the route is worth enabling by default.

## Enabling it (no reconversion)

The W4A8 route consumes the **same** stored Q4 codes and fp16 group scales as A16, so the artifact
payload is unchanged. Only the per-use `activation_policy` metadata differs. The groupwise artifact
currently declares `A16Only` for every use (the `recipe.py` default, which `_dense_groupwise` never
overrode), so the route is unreachable until the artifact declares `AllowA8` for the gate/up
projections. `_dense_groupwise` now sets `activation_policy="AllowA8"` for the Q4 gate/up; because
`"A16Only"` and `"AllowA8"` are both 7 bytes, an existing artifact can be enabled by rewriting that
schema field in place - no re-quantization from tensors is required.

## Why

Qwen3.8-27B `groupwise-int` prefill is **bf16-Tensor-pipe bound at the roofline**, so there is no
scheduling headroom left in the Q4/Q5 A16 kernels. int8 is ~3.5-4x bf16 in hardware:

| kernel on `[34816,5120] x [5120,T]` | T=1024 | T=4096 |
|---|---:|---:|
| ninfer Q4 A16 SwiGLU (`ninfer_q4_linear_swiglu_bench`, this machine) | 200.8 | 177.3 TFLOP/s |
| cuBLAS bf16 (same shape) | 237.7 | 256.5 |
| **cuBLAS int8 (same shape)** | **830.3** | **909.0** |
| hand-written wmma int8 prototype | 217.5 | 241.9 |
| hand-written raw `mma.s8` GEMM prototype (correct) | 397.5 | 420.9 |
| **groupwise Q4-A8 SwiGLU prototype (correct, realistic data + L2 flush)** | **246.6** | **247.3** |

`wmma::mma_sync` (m16n16k16) cannot reach the int8 rate; a raw
`mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` kernel with `cp.async` staging reaches ~2x bf16
(~0.5x cuBLAS). The fused **groupwise** SwiGLU reaches ~1.2-1.4x bf16: the groupwise structure and
the SwiGLU's dual gate/up stream keep a ~1.6x gap against the plain int8 GEMM.

### Measurement caveat (important)

Uniform (all-ones) inputs **inflate** this kernel ~18% via clock/power boost: all-ones reported 288
TFLOP/s standalone, but 279 under ncu at 2.7 GHz and only ~247 with random inputs. All numbers above
use random inputs and a 256 MB L2 flush per iteration, matching
`ninfer_q4_linear_swiglu_bench` (`flush_bytes=268435456`). Do not benchmark this route on uniform
data.

## What actually moved the prototype (145 -> 247)

1. **Weight residency is the first-order effect.** Pre-expanding Q4 to int8 doubles the weight to
   178 MB, which does not fit the ~96 MB L2, so the weight streams from DRAM (DRAM 58%). Keeping the
   weights **int4-packed (89 MB)** and sign-extending to int8 in the shared stage dropped DRAM to
   ~5% and lifted the prototype from 145 to 242 TFLOP/s.
2. **Expand with packed stores.** Byte-at-a-time int4->int8 stores made LSU the top pipe (47%); two
   16-byte `uint4` stores per stream (the swizzle keeps 16 logical bytes contiguous) removed it.
3. **Occupancy.** The kernel is register-bound; `-maxrregcount` (80 at 64x64x256, 64 at
   128x64x512) admits 3/2 blocks per SM and lifted occupancy from 33% to 50%.
4. **Grid order.** Making the token block the fast grid axis keeps the active weight slice
   L2-resident across token blocks (DRAM 58% -> 5%).
5. Larger K tiles (STAGES=3), more warps per CTA, and a smaller row tile (BN=32) all *hurt*
   (redundant A `ldmatrix`, less reuse).

The remaining limiter is instruction issue (tensor pipe ~29%): the per-group scale FMA, the
`ldmatrix` address math, and the int4 expansion together are ~4x the MMA instruction count. Closing
that is the M2 kernel work.

## Design

Reuse the FP8 A8 route as the template: the `fp8_mma_compute_stage` ldmatrix fragment pattern for
k32 8-bit operands applies unchanged to int8.

- **Weights**: keep Q4 group-64 packed (int4, 89 MB); sign-extend the 4-bit codes to int8 `[-8,7]`
  in the shared stage (packed `uint4` stores). MMA = `mma_s8` (`src/ops/common/mma.cuh`).
- **Activations**: per-token absmax int8 quantization with a per-token scale.
- **Groupwise scale**: Q4 scales vary per K-group-64, so accumulate a group into an int fragment and
  `total += (w_group_scale x a_token_scale) * group_acc`. Folding after each k32 is valid by
  linearity and shortens the accumulator live range.
- **Tile**: 128 tokens x 64 gate rows (+64 up rows), 512 threads, `maxrregcount` ~64; token block as
  the fast grid axis.
- **Epilogue**: `silu(gate) * up`.

## Milestones

- **M2**: production raw-`mma_s8` Q4 route for `linear_swiglu [34816,5120]` over ninfer's rowsplit Q4
  layout; register as an A8 plan under `LinearPolicy::AllowA8`; wire the activation quant and
  workspace; benchmark against the bf16 route on realistic data with the L2 flush.
- **M3**: extend to Q5/Q6 and `linear`/`linear_add`.
- **M4**: FP64-oracle A8 qualification (per [op-development.md](op-development.md)) and end-to-end
  prefill measurement.

## Alternative

The `nvfp4` artifact already runs an A4 prefill at ~12.8k tok/s (~4x groupwise-int) and needs no
work; W4A8 exists to keep groupwise-int's smaller footprint (16.96 GiB vs 22.09 GiB) with faster
prefill.
