# W4A8 prefill for groupwise-int (active work)

Status: **active**. M1 (feasibility) complete. A groupwise Q4-A8 SwiGLU prototype is correct but
does **not yet beat** the bf16 route (see below); M2 (production int8 route) is not started. This
records the measured evidence and the plan so the numbers are not re-derived. Remove or fold into
[op development](op-development.md) / [engine architecture](engine-architecture.md) when the route
lands.

## Why

Qwen3.8-27B `groupwise-int` prefill is **bf16-Tensor-pipe bound at the roofline**, so there is no
scheduling headroom left in the Q4/Q5 A16 kernels:

| kernel on `[34816,5120] x [5120,T]` | T=1024 | T=4096 |
|---|---:|---:|
| ninfer Q4 A16 SwiGLU (`ninfer_q4_linear_swiglu_bench`) | 202.6 | 205.1 TFLOP/s |
| cuBLAS bf16 (same shape) | 237.7 | 256.5 |
| **int8 (cuBLAS, same shape)** | **830.3** | **909.0** |
| hand-written wmma int8 prototype | 217.5 | 241.9 |
| hand-written raw `mma.s8` prototype (correct) | **397.5** | **420.9** |
| groupwise Q4-A8 SwiGLU prototype (correct) | 145.7 | 142.2 |

int8 is ~3.5-4× bf16 in hardware, but `wmma::mma_sync` (m16n16k16) does not reach it. A raw
`mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` kernel with `cp.async` staging reaches ~2× the
bf16 route (~0.5x cuBLAS), verified against a CPU oracle.

However the **groupwise Q4-A8 SwiGLU prototype lags the bf16 route** (146 vs 205): the 64-K group
tile, three staging streams (activation + gate + up), dual accumulators, and per-group scaling cost
more than the int8 MMA saves. The plain int8 GEMM reaching 397 shows the ~2× is achievable in
principle, but the groupwise SwiGLU needs the production structure (interleaved gate/up rows in one
accumulator, larger K tiles with an inner per-group scale) plus a deeper pipeline. That is the M2
work, and it is not guaranteed.

## Design

Reuse the existing FP8 A8 route as the template: the `fp8_mma_compute_stage` ldmatrix fragment
pattern for k32 8-bit operands applies unchanged to int8 (same 8-bit x k32 fragment layout). The
route adds:

- **Weights**: keep Q4 group-64 packed; sign-extend the 4-bit codes to int8 `[-8,7]` in the shared
  stage. MMA = `mma_s8` (in `src/ops/common/mma.cuh`).
- **Activations**: per-token absmax int8 quantization in-kernel with a per-token scale.
- **Groupwise scale (the crux)**: unlike FP8's per-row weight scale, Q4 scales vary per K-group-64.
  Accumulate a group into a temp int fragment, then `total += (w_group_scale x a_token_scale) *
  group_acc`. The prototype shows the naive form is expensive (it forces a 64-K tile and dual
  accumulators), so the production kernel must amortize it over a larger K tile.
- **Epilogue**: existing SwiGLU epilogue after dequant.

## Milestones

- **M2**: production raw-`mma_s8` Q4 route for `linear_swiglu [34816,5120]`; register as an A8 plan
  under `LinearPolicy::AllowA8`; wire workspace; benchmark against 205 TFLOP/s.
- **M3**: extend to Q5/Q6 and `linear`/`linear_add`.
- **M4**: FP64-oracle A8 qualification (per [op-development.md](op-development.md)) and end-to-end
  prefill measurement.

## Alternative

The `nvfp4` artifact already runs an A4 prefill at ~12.8k tok/s (~4x groupwise-int) and needs no
work; W4A8 exists to keep groupwise-int's smaller footprint (16.96 GiB vs 22.09 GiB) with faster
prefill.
