# W4A8 prefill for groupwise-int (active work)

Status: **active**. M1 (feasibility) and M2 (the production Q4 W4A8 `linear_swiglu` route) are
complete: the route is implemented, FP64-oracle-tested, enabled on the `qwen3_8_27b` artifact, and
measured end-to-end. It runs ~1.6-1.7x the A16 route at the op and ~1.16x on a 1410-token prefill.
M3 (Q5/Q6 and `linear`/`linear_add`) and the remaining M4 work are open. This records the measured
evidence, the design, and the open work. Remove or fold into [op development](op-development.md) /
[engine architecture](engine-architecture.md) when the route is fully characterized.

## Result

### Op (`ninfer_q4_linear_swiglu_bench`), `[34816,5120] x [5120,T]`

Realistic data, 256 MB L2 flush (`flush_bytes=268435456`), three interleaved A16/A8 passes (median):

| policy | T=2048 | T=4096 | T=8192 |
|---|---:|---:|---:|
| A16 (`--policy a16`) | 203.1 | 179.9 | 175.2 TFLOP/s |
| **W4A8 (`--policy a8`)** | **324.7** | **310.7** | **288.6** |
| ratio | 1.60x | 1.73x | 1.65x |

### End-to-end prefill

1410-token prompt, `--max-new 1 --greedy`; two interleaved passes each:

| artifact policy | prefill | speed |
|---|---:|---:|
| A16 (785 uses `A16Only`) | 457 ms | 3.09k tok/s |
| **A8 (130 gate/up uses `AllowA8`)** | **392 ms** | **3.60k tok/s** |

End-to-end is 1.16x, not the op's 1.6-1.7x: prefill also runs attention and the other projections,
which stay A16.

## Energy per output was the remaining lever

The route's first in-engine measurement was only 1.03-1.09x A16. Investigation showed the kernel was
not tensor-throughput limited but **instruction-issue limited**: the per-group scale FMA, the
`ldmatrix` address math, and above all the int4->int8 expansion were ~4x the MMA instruction count.
The expansion decoded each 32-bit weight word with four scalar sign-extends; replacing it with two
`__byte_perm` gathers plus a SWAR sign-extend (`q4a8_sext4`/`q4a8_expand4`, forcing the high bit of
every byte before the subtract so no borrow crosses a byte) roughly halved the non-MMA ALU work and
lifted the op from ~1.05x to the 1.6-1.7x above.

The earlier note that W4A8 is "power-limited at 475 W" was incomplete: the scalar expansion was
also drawing ALU power and blocking issue, so removing it improved both the clock and the issue
rate. The int8 tensor advantage is now mostly realized, not spent on the clock penalty.

## Enabling it (no reconversion)

The W4A8 route consumes the **same** stored Q4 codes and fp16 group scales as A16, so the artifact
payload is unchanged. Only the per-use `activation_policy` metadata differs. The groupwise artifact
originally declared `A16Only` for every use (the `recipe.py` default, which `_dense_groupwise` never
overrode), so the route was unreachable. `_dense_groupwise` now sets `activation_policy="AllowA8"`
for the Q4 gate/up; because `"A16Only"` and `"AllowA8"` are both 7 bytes, the existing artifact was
enabled by rewriting that schema field in place (130 gate/up uses to `AllowA8`, the other 655 uses
left `A16Only`) - no re-quantization from tensors was required. For an already-downloaded artifact,
[`set_activation_policy`](../../tools/artifact/set_activation_policy.py) performs that rewrite and
writes a new file with a new `artifact_id`.

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
(~0.5x cuBLAS). The fused **groupwise** SwiGLU prototype reached ~1.2-1.4x bf16. The production
route now exceeds that prototype (324.7/310.7 at T=2048/4096), so the prototype-vs-engine gap is
closed.

### Measurement caveat (important)

Uniform (all-ones) inputs **inflate** this kernel ~18% via clock/power boost: all-ones reported 288
TFLOP/s standalone, but 279 under ncu at 2.7 GHz and only ~247 with random inputs. All numbers above
use random inputs and a 256 MB L2 flush per iteration, matching
`ninfer_q4_linear_swiglu_bench` (`flush_bytes=268435456`). Do not benchmark this route on uniform
data.

## What actually moved it

Prototype, 145 -> 247 TFLOP/s:

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

Engine route, 208 -> 325 TFLOP/s (T=2048):

6. **Cheap int4 decode.** Two `__byte_perm` gathers plus a SWAR sign-extend replace the four scalar
   sign-extends per word (see "Energy per output" above).
7. **Packed expansion stores.** The engine stage writes two `uint4` per stream, matching the
   prototype's LSU fix; a byte-store version left LSU at 47%.
8. **Register cap for occupancy.** `__launch_bounds__(512, 2)` admits 2 blocks/SM.
9. **Token-major grid.** `grid.x` is the token block, `grid.y` the row block, so the active weight
   slice stays L2-resident.

## Design

Reuse the FP8 A8 route as the template: the `fp8_mma_compute_stage` ldmatrix fragment pattern for
k32 8-bit operands applies unchanged to int8.

- **Weights**: keep Q4 group-64 packed (int4, 89 MB); sign-extend the 4-bit codes to int8 `[-8,7]`
  in the shared stage with `__byte_perm` + SWAR sign-extend and packed `uint4` stores. MMA =
  `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`.
- **Activations**: per-token absmax int8 quantization with a per-token scale.
- **Groupwise scale**: Q4 scales vary per K-group-64, so accumulate a group into an int fragment and
  `total += (w_group_scale x a_token_scale) * group_acc`. Folding after each k32 is valid by
  linearity and shortens the accumulator live range.
- **Tile**: 128 tokens x 64 gate rows (+64 up rows), 512 threads, `__launch_bounds__(512, 2)`; token
  block as the fast grid axis.
- **Epilogue**: `silu(gate) * up`.

## Milestones

- **M2** (done): production raw-`mma.s8` Q4 route for `linear_swiglu [34816,5120]` over ninfer's
  rowsplit Q4 layout; registered as an A8 plan under `LinearPolicy::AllowA8`; activation quant and
  workspace wired; FP64-oracle qualified; artifact enabled; benchmarked against A16 on realistic
  data with the L2 flush and end-to-end.
- **M3**: extend to Q5/Q6 and `linear`/`linear_add`.
- **M4**: complete the remaining end-to-end characterization (longer prompts, other prefill chunks)
  under [op-development.md](op-development.md).

## Alternative

The `nvfp4` artifact already runs an A4 prefill at ~12.8k tok/s (~4x groupwise-int) and needs no
work; W4A8 exists to keep groupwise-int's smaller footprint (16.96 GiB vs 22.09 GiB) with faster
prefill.
