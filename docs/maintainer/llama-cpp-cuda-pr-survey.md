# llama.cpp CUDA PR survey — what is worth acting on for NInfer

Snapshot 2026-10-14. Scope: the 137 open PRs carrying `label:CUDA` on `ggml-org/llama.cpp`
(<https://github.com/ggml-org/llama.cpp/pulls?q=is%3Apr+state%3Aopen+label%3ACUDA>), triaged by title,
then Tier A evaluated in depth against this repository.

Method and limits:

- Criterion: could the PR change a decision **here** — sm_120a on one RTX 5090, the Qwen3.5/3.6/3.8
  hybrid (GDN + attention) family, Qwen4Exp host-resident experts, Gemma 4, nvfp4 / fp8 / int8 /
  group-64, paged KV + host KV tier, MTP + ReplaySSM, CUDA Graphs + PDL?
- Triage was title-only; the label overlaps other labels, so a title can mislead in both directions.
- Each Tier A entry below was evaluated by reading the PR body/diff/comments and the corresponding
  NInfer code. Load-bearing citations were then re-checked against the tree; two path citations in the
  evaluations were wrong and are corrected here (`sparse_moe` kernels live in
  `src/ops/sparse_moe/prefill/`, the KV codecs' head-dim pin is in `src/core/paged_kv_storage.h`).
- No GPU measurement was taken for this survey; where a number would decide a question, the exact
  check is named instead.

## Outcome

Tier A: **six evaluated, one worth acting on** (A3, and only after an offline measurement), five
closed with in-repo evidence that the capability already exists. The survey's by-product is four
NInfer-internal leads, listed at the end.

| # | PR | Title | Verdict | Bound |
|---|---|---|---|---|
| A1 | [#26159](https://github.com/ggml-org/llama.cpp/pull/26159) | compact Blackwell NVFP4 MoE work scheduling | not transferable (both halves present) | 27B dense 0%; 35B-A3B ≤66% nearest measurement; Flash-Next ≤7.8% |
| A2 | [#26704](https://github.com/ggml-org/llama.cpp/pull/26704) | SM120 CUTLASS MoE prefill for MXFP4/NVFP4 | not transferable | ≤3.3% end-to-end by FLOP share |
| A3 | [#26914](https://github.com/ggml-org/llama.cpp/pull/26914) + [#28481](https://github.com/ggml-org/llama.cpp/pull/28481) | native INT8 convrot (rotation) | **plausible, magnitude unverified — measure first** | must cut A8 KLD 0.362 → ≲0.05–0.09 (≥4–7×) |
| A4 | [#29051](https://github.com/ggml-org/llama.cpp/pull/29051) | Kronecker FWHT | not transferable | <0.05% of a decode step; capability only |
| A5 | [#22671](https://github.com/ggml-org/llama.cpp/pull/22671) | MXFP6 (CPU) | KV: not transferable; weights: needs measurement | equal-bit gate is `q6_g64_fp16` (6.25 b/w) |
| A6 | [#21897](https://github.com/ggml-org/llama.cpp/pull/21897) | concurrent streams for linear attention | not transferable as a gain | prefill ≈0; decode ≤~5% best case |

---

## A1 — compact Blackwell NVFP4 MoE work scheduling (#26159)

**Mechanism.** Two separable changes. (A) `build_moe_job_map<J>`: one block, `threadIdx.x == 0`, walks
experts in id order and cuts each expert's already-compacted assignment range into tiles of `J`,
writing `(job_experts, job_tiles)`; the grid becomes flat `max_jobs * nty` with
`max_jobs = min(ncols_dst, ceil(ncols_dst/J) + E - 1)`, i.e. full tiles plus one partial per expert,
and padding exits on `job_experts[job] < 0`. Deterministic: fixed job order, one scalar store per
output element, no atomics. (B) the per-expert NVFP4 output scale is folded into the writeback
(`output_scale[zt]`, `ggml_cuda_mmq_write_back_mma_scaled`), deleting a GET_ROWS + MUL pass; the
author states most of the measured benefit is here. Measured on GB10, `-p 8192 -ub 2048`: +15.5% /
+10.0% on two NVFP4 MoE models, noise elsewhere.

**NInfer counterpart.** Mechanism A exists three times, with the same grid formula:
`src/ops/launcher/gemma_moe_experts.cu:125` computes
`carve.max_jobs = assignments / kGroupedBN + std::min(experts, assignments)`;
`src/ops/kernel/gemma_moe_experts_grouped.cuh:36,90,170` walks those jobs and clamps the column range
to the expert's offset range; `src/ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu:248-305`
builds `expert_offsets` / `route_job_experts` / `route_job_columns` / `route_job_count`, and the expert
GEMMs then walk **exactly the useful items** with a device-side count (`:531`, `:958`:
`total_work = *route_job_count * row_blocks; for (work = blockIdx.x; work < total_work; work += gridDim.x)`),
which is strictly stronger than the PR's padded host grid; `src/ops/offload_moe/a4_kernels.cu:322-343`
uses `a4_max_jobs(columns)` on the same shape. Mechanism B is already the epilogue:
`DivisorEpilogue` (`a4_kernels.cu:125`, applied at `:327` and `:341`).

**Why the determinism question is unchanged.** The recorded rejection of NInfer's own gather inversion
(`docs/maintainer/gemma4-26b-plan.md:452-462`) was *not* about compaction order — that was already
fixed — but about an `atomicAdd`-filled FP32 accumulator (two identical runs gave mean NLL 5.464852
and 5.520692). This PR adds no cross-block reduction and no atomic, so it neither fixes nor worsens
that question. NInfer's remaining cross-expert sum is rank-ordered (`:1654`).

**Shares.** 27B dense: 0% (no MoE op). 35B-A3B: ~38% of arithmetic, ≤66% by the nearest measurement
(the Gemma 26B grouped route's profile, `docs/maintainer/gemma4-26b-plan.md:716-724`: expert gate/up
30.7% + down 24.3% MMAs + router 11.3%). Flash-Next: ≤7.8% (0.172 s device wait of 2.221 s for a
7,006-token prefill, `profiles/bench/flash-next-host-bound-2026-10-09/sweep_p8k.jsonl`).

**Decisive check if the area is ever touched.** The one real gap named in-repo is tile width against
per-expert count ("~half of every MMA tile is padding", `gemma4-26b-plan.md:720`), and it comes from
master's AMD-only `J` heuristic, not from this PR. Order: `ninfer_sparse_moe_bench --codec q4-q5
--tokens 1024 --execution graph --cache cold` x 40 layers / prefill wall time; then
`ninfer_bench --profile-measured` for launched-block and tail waste (flip if > ~1% of prefill); then a
`kGroupedBN` 64 -> 32 A/B (flip if the 55% MMA share drops >= 3 points with KLD unmoved).

## A2 — SM120 CUTLASS MoE prefill for MXFP4 and NVFP4 (#26704)

**What is reviewable.** The diff is **dense-only** (weight path gated on `src0->ne[2] != 1 -> false`,
no `MUL_MAT_ID`; commit `37fb133f` "limit cutlass repack path to dense models"). The MoE revision was
force-pushed out on 2026-08-12, so only its benchmark rows survive and there is no MoE scheduling code
to read. Reported wins: MXFP4 dense +8.5–13.3%, NVFP4 dense +11.0–12.8% (RTX PRO 6000), a third-party
5090 NVFP4 dense +30–43%; all measured against llama.cpp's own dequant -> q8_1 -> DP4A MMQ path, and
the PR's own table shows that baseline 3.14x behind vLLM. Scheduling actually present: CUTLASS
`GemmUniversal` with `Sm120` + `OpClassBlockScaledTensorOp`, tile 128x128x128, **cluster 1x1x1**, auto
stage count, no stream-K, no persistent CTAs, and no expert-dimension mapping.

**NInfer counterpart.** Dense NVFP4 prefill is a hand-written TMA warp-specialized block-scaled FP4
GEMM (`src/ops/linear/nvfp4/nvfp4_w4a4_tma.cuh:55-123`; BlockN 128, block-scale tiles, descriptor
pinned to `kNvfp4TmaBlockM`), measured at `docs/maintainer/linear-benchmark.md:511-517`:
`[34816,5120]` T=1024 390.112 us = 935.81 TFLOP/s = **55.84% of the 1676 TFLOP/s dense FP4 peak**
(58.79% best case at `[14336,5120]`). MoE prefill is grouped/dynamic and reuses the same linear kernel
with routed row policies (`row_policy.weight_row(row_begin, row)`), the routed job list coming from the
scan in A1. MXFP4 has zero hits in `src/`, `include/`, `docs/` — that half is inert here.

**Verdict.** Not transferable: llama.cpp adopted CUTLASS for a format NInfer does not store, against a
baseline NInfer does not have. Bound: expert GEMMs are ~23.6% of 35B-A3B active FLOPs, so the op-level
ceiling (55.84% -> 65% of peak) is <= 3.3% end-to-end.

**Internal lead.** The routed MoE expert path uses cp.async staging — `sparse_moe_prefill_kernels.cu`
has **0** TMA occurrences — while the dense NVFP4 path is TMA warp-specialized. Check:
`./build/bench/ninfer_sparse_moe_bench --execution graph --cache cold --repeat 20` was run, one session:

| codec | T=1024 | T=8192 |
|---|---|---|
| nvfp4 | 368.6 µs, 160.2 TFLOP/s, **69.0%** of DRAM peak | 1400.8 µs, 337.3 TFLOP/s, **18.2%** |
| q4-q5 | 647.2 µs, 91.3 TFLOP/s, 40.2% | 3560.0 µs, 132.7 TFLOP/s, 7.3% |
| q4-q6 | — | 3625.9 µs, 130.3 TFLOP/s, 7.7% |

Read: at the production prefill chunk (1024) the nvfp4 routed body is near bandwidth-bound, leaving at
most ~30% on that op; at wide extents it is compute/occupancy-bound at 18% of DRAM peak. Both sit far
below the dense nvfp4 GEMM's 55.84% of FP4 peak, so there is a real op-level gap — but it is
shape-confounded (small N per expert, gather overhead), not a TMA-or-not question, and q4-q5 at T=1024
(40.2% of peak) is the point with visible headroom. This is an NInfer-internal investigation, not a
port of anything upstream.

## A3 — INT8 ConvRot (#26914 + #28481)

**Mechanism.** A block-diagonal rotation by a **regular** Hadamard (the Kronecker power of a 4x4 H4
with no all-ones column), groups of 256 along K. Weights are rotated offline at conversion
(`W_rot = grouped(W) @ H.T`, then per-output-row absmax int8, one scale per output row); activations
are rotated online by a dedicated quantize op that emits rotated int8 codes plus one F32 row scale, so
the packed activation can be cached and shared by consumers. Correct because `R` is symmetric and
orthogonal, `X_rot W_rot^T = X W^T`. Only G = 256 is supported on the dedicated paths
(`GGML_ASSERT(convrot_group_size == 0 || convrot_group_size == 256)`), and the CUDA side is a **cuBLAS
INT8** call, not a hand-written MMA.

**Evidence quality.** Both PRs carry **no accuracy data of any kind** — only op correctness at 1e-5
against a quantized reference and synthetic speed (73.5 TOPS on Vulkan; Krea2 9.47 -> 1.72 s/step).
`#28481`'s body is empty. The intended consumer is stable-diffusion.cpp, "no llama.cpp model path uses
these operations yet". The published evidence is QuaRot (arXiv 2404.00456v2), whose rotated W8A8KV8 RTN
is lossless on Llama-2 (7B 5.47 -> 5.50) and whose grouped ablation puts group-256 at 5.98 against the
global rotation's 6.10 — but on a *global* rotation, measured as WikiText PPL, not KLD against an A16
reference. **No published or in-PR measurement of grouped-256 rotated INT8 KLD exists.**

**NInfer counterpart.** Every part is present, in a different place: a normalized Hadamard in the KV
path (`src/ops/kv_cache/hadamard_d256.cuh:46`, warp in-register, applied at `src/ops/kv_cache/append/kernel.cuh:34,239,310`),
KV codecs that rotate rows with the paired reader rotating queries
(`src/ops/kernel/dense_kv_ring.cuh:75-82,124-155`; "the Hadamard question is settled: the readers rotate
the queries", `docs/maintainer/gemma4-26b-plan.md:789-795`), a linear-activation rotation already shipped
for EXL3 (`src/ops/linear/exl3/exl3_dispatch.cu:51-59`, H128 over 128-blocks of K with a row scale into
BF16 scratch), and the quantizer — per-token absmax int8, **unrotated** —
at `src/ops/linear/q4/q4_a8.cu:18-62`, which is exactly the insertion point.

**Arithmetic.** The recorded Gemma 4 31B experiment: INT8 activations on the Q6/Q5 row-split prefill
linears bought **+47% pp8192 for 0.362 wikitext KLD against an A16 reference, inside a 0.419 drift
budget**, and was dropped; the error was spread evenly with no KLD floor, single-layer A8 cost only
0.004–0.011, and group 64 -> 32 bought 13% accuracy for 15% prefill. To be affordable beside the weight
formats, the A8-attributable KLD must fall to **≲0.05–0.09, i.e. a >=4–7x reduction**. Cost side:
QuaRot's online Hadamard is +10.5% on a 4096x4096 batch-1 linear, which would take the route from
~1.65x A16 to ~1.5x A16 at the op [INFERENCE]. If it lands it **descopes the two-term FP8 plan** to the
fp8-weight artifact, because one `mma.s8` (368.6 us at `[14336,5120]`) beats A8Pair's two terms
(721.9 us) — and rotation is not expected to help E4M3 as much, since NInfer already measured that
per-32 MX scaling of E4M3 activations buys nothing over per-token (0.0372 vs 0.0375).

**Nearest in-engine support.** Rotated int8 G64 at 8.25 bits has 0.0059 attention-score error against
rotated fp8 E4M3 row-256 at 8.06 bits with 0.0260 — NInfer's own rotated *integer* codec is ~4.4x more
accurate than its *float* one at equal bits. [INFERENCE] The bet is that the same holds for gate/up
activation rows.

**Required work if it transfers.** (1) Converter: rotate FP32 weights along K in groups of 256 *before*
the groupwise quantizer wherever `K % 256 == 0` — the load-bearing change, since rotating after
quantization adds a second lossy step, so the stored codes change and today's "no reconversion"
enablement of W4A8 (`docs/maintainer/w4a8-prefill.md:50-58`) is lost. (2) A per-tensor/per-use
`rotation` marker in the artifact schema, because *every* consumer of a rotated weight must rotate its
activation, not only the A8 one. (3) Runtime: the existing Sylvester `hadamard_d256` butterfly, which
the measurement above shows is equivalent to ConvRot's regular H256, called at the head of
`launch_q4_a8_quantize` (`q4_a8.cu:19`) and at the head of any A16 kernel that reads a rotated weight. (4) Scales unchanged — per-token activation scale and per-row/group weight
scales are exactly ConvRot's contract.

**Result (measured 2026-10-14, branch `review/llama-cuda-tier-a`) — the rotation works, and by more
than the gate asked for.**

Tools: `tools/verify/rotation.py` (both Hadamard families, block rotation along K, the per-token INT8
quantizer), `tools/verify/gemma4_rotation_probe.py` (row-level Stage 0, reusing this harness's
checkpoint, forward and converter encoders), and three candidates in `tools/verify/gemma4_sensitivity.py`
(`q6_i8`, `q6_i8r256`, `q6_r256`) added through `parse_candidate`/`candidate_weight`. The probe asserts
that the unquantized rotated route reproduces the FP32 oracle to <1e-6 at every layer, so a broken
rotation cannot produce a flattering number.

Stage 0 — all 60 Gemma 4 31B layers, gate and down inputs, 256 rows each:

| | unrotated | rotated |
|---|---|---|
| `absmax/RMS` per row (median) | 5.9–68 | 3.8–6.1 |
| excess kurtosis | 1.4–1415 | -0.0–7.0 |
| per-row INT8 round-trip rel. L2 | 1.3e-2 – 1.2e-1 | 8.6e-3 – 1.9e-2 |
| A8-attributable excess vs the FP32 oracle | 1.00x | **gate 5.78x median, 44/60 layers >= 4x, worst 1.77x; down 7.05x median, worst 2.26x** |

`regular256` and Sylvester-256 agree within noise on all 60 layers of both projections (5.78 / 5.79x on
gate), so **the in-register Sylvester `hadamard_d256` already in the tree is sufficient** — no new
butterfly and no new block size. Correction: that run also carried a `regular128` column, which was a
silent duplicate of `regular256`; the probe's own transform builder overshot any non-power-of-four size
instead of validating it, and 128 is not a Kronecker power of H4. The shared module added since
(`tools/artifact/rotation.py`) raises on such a size and the name is gone, so the registered set is
`sylvester128/256` and `regular64/256`. The one genuinely narrower block that was measured, `regular64`,
is consistently worse (4.5x against 6.0–7.8x on gate).

Stage 1 — KLD through this harness's layout mode, A8 on all 60 layers' `mlp_gu`, 3 x 4096 tokens:

| layout | KLD vs the BF16 reference |
|---|---:|
| Q6 weights, unrotated | 0.20252 |
| Q6 weights, rotated (`q6_r256`) | 0.18491 |
| Q6 + INT8 activations | 0.37637 |
| Q6 + INT8 activations, rotated | 0.19637 |

Decomposition: the A8-attributable drift is **+0.1739 unrotated and +0.0115 rotated, a 15.1x
reduction**, and that is the aggregate over all 60 layers with the weak layers included, not a best
case. Rotating the weights alone is worth another 0.0176 (8.7%) of the weight-format drift, so the
rotated A8 route lands *below* the unrotated weight-only baseline.

Protocol note: the recorded layout KLDs in `gemma4-model.md` were taken with **FP32 activations** (BF16
activations put a ~0.15 KLD floor under every candidate), on whole layouts. The figures below are
`mlp_gu`-only and were taken twice — once under the FP32-activation protocol, once with the default
BF16 activations. Their absolute values are **not** comparable to the recorded 0.362, which covers the
full A8 set on M1's weights; what is comparable is the delta measured inside one run.

| A8-attributable drift (KLD, mlp_gu, all 60 layers) | unrotated | rotated | reduction |
|---|---:|---:|---:|
| FP32 activations, 4 x 2048 tokens, 4 domains | +0.2762 | +0.0104 | **26.6x** |
| BF16 activations, 3 x 4096 tokens, 3 domains | +0.1739 | +0.0115 | 15.1x |

Weight-side effect of rotating before the groupwise encoder, same runs: 0.22502 -> 0.18702 (−17%) and
0.20252 -> 0.18491 (−9%) — **KLD only**: the engine result further down shows the same change costs
perplexity, so read those two figures as agreement with the reference distribution, not as "better". In both, the rotated A8 total lands at or below the unrotated weight-only
baseline, i.e. A8 stops being the limiting error once the operands are rotated. The BF16 run's smaller
ratio is the expected masking: int8 and BF16 residual rounding are both activation-domain errors.

Cost of the rotation itself, now an Op in the tree (`include/ninfer/ops/block_hadamard.h`,
`src/ops/{kernel,launcher,wrapper}/block_hadamard.*`, FP64-matrix oracle test
`tests/ops/test_block_hadamard.cpp`, bench `bench/ops/block_hadamard_bench.cu`), measured against a
plain BF16 copy of the same bytes:

| point | copy | rotate |
|---|---:|---:|
| K=5376, T=1024 | 6.26 us | 6.34 us |
| K=5120, T=1024 | 6.01 us | 6.32 us |
| K=5376, T=1 | 1.39 us | 1.41 us |
| K=21504, T=1024 | 25.85 us | 29.66 us |

The butterfly is entirely hidden behind the memory traffic, and the traffic is L2-resident in
production (the activation was just produced). Per layer that is ~6 us at the prefill chunk and ~1.4 us
at T=1: ~0.06% of a 1024-token prefill chunk and ~0.7% of a decode step, which is negligible beside
the +47% pp8192 the A8 route already measured.

Against the recorded arithmetic: the A8 term that consumed 0.362 of Gemma 4 31B's 0.419 budget becomes
~0.02–0.03, so A8 becomes affordable at the +47% pp8192 the unrotated route already measured. That
projection scales the rotated delta measured here (mlp_gu only, +0.0115) by the ratio between the
recorded A8 set's drift and this proxy's mlp_gu drift, 0.362 / 0.174 = 2.1 [INFERENCE]; the engine run
below replaces it with a number. Keep the
weight-side gain separate: it is an accuracy improvement at equal bits, but it is not free — a rotated
weight needs its activation rotated too, so it carries the same rotation pass. Since the weight-side
gain is negative for the layout (−0.0176 here) and the A8 term collapses, the combined Gemma 4 31B
layout should land at or below today's 0.419 while gaining the A8 prefill route.

**Engine result (measured 2026-10-14) — the weight-side rotation is a regression, not the gain the proxy
predicted.**

Three artifacts from the same tree, same recipe otherwise: `out/m1_plain.ninfer`,
`out/m1_rot.ninfer` (rotation `sylvester256` on all 60 layers' `mlp/gate|up`), and the shipped
`models/gemma4_31b_it_m1.ninfer`.

| check | result |
|---|---|
| all 1077 objects hashed, both artifacts | exactly **120 differ, all gate/up**; every other object byte-identical |
| converter rotation against the source checkpoint | `stored == encode(rotate_k(W, "sylvester256"))` exactly, plain `== encode(W)`; the Python transform equals the popcount matrix the CUDA Op's oracle pins (identity permutation) |
| fresh plain conversion vs the shipped artifact | mean NLL **identical to 15 digits** (7.577467419430873): no converter drift, unrotated path untouched |
| `gemma4_load`, `gemma4_model`, `gemma4_layer` (repo's own tests) | pass on **both** artifacts; the rotated one is honoured at load (exactly 120 claimed Uses) |
| engine mean NLL (wikitext/00, context 4096, stride 2048, bf16 KV) | plain 7.5775 → rotated **7.6633 (+0.0859)**; PPL 1953.7 → 2128.9 (**+9.0%**) |
| per-window deltas, 31 windows | rotation *better* early (−0.17 … −0.04) and *worse* late (+0.25 … +0.30): trajectory divergence, not a localized error |

The harness — the tool this repo chooses layouts with — was then run on the **exact** M1 `mlp_gu` mix
(36 q5 + 24 q6 layers), not the q6-everywhere set used earlier:

| harness run | KLD | delta-NLL |
|---|---|---|
| FP32 activations, M1 mix | 0.27132 → 0.23037 (**−15%**) | −0.06849 → −0.06809 (**+0.0004**) |
| BF16 activations (the engine's rounding) | 0.29596 → 0.28477 | −0.01473 → −0.01188 (**+0.0029**) |

So the harness now agrees with the engine's *sign* — NLL is not improved by rotation — while its
magnitude is ~30x smaller (+0.003 against +0.086). That gap is **unresolved**, and it is not a pairing
bug: the four checks above establish that the stored codes, the transform, and every consumer agree,
and the repo's own layer/model tests pass with rotation on. It does mean the harness under-predicts the
cost of this change, and that any future rotation claim needs an engine run, not a layout score.

**Verdict: not worth shipping for the weight-side purpose.** (The Op's own files —
`include/ninfer/ops/block_hadamard.h`, `src/ops/{kernel,launcher,wrapper}/block_hadamard.*`,
`bench/ops/block_hadamard_bench.cu`, `tests/ops/test_block_hadamard.cpp` — were removed with the
reverted feature; they are named here only because this section records what was built and measured.) At equal bits the rotation buys KLD (the
budget currency, −15%) and top-1 agreement (+1.8 points) but costs ~9% perplexity — the number
`docs/perplexity.md` publishes — for a new artifact and an extra activation pass. The A8 motivation is
unaffected in principle (an int8 activation error is ~10x the rotation's own rounding), but no Gemma 4
A8 route exists to carry it (`w4a8-prefill.md` M3 is open), so the plumbing in this branch currently
has no beneficiary and should be reverted or held until that route lands.

## A4 — Kronecker FWHT (#29051)

**What it is.** Not a new op: it extends the `GGML_HINT_SRC0_IS_HADAMARD` path (an ordinary `MUL_MAT`
by a dense Hadamard) with a Kronecker-structured transform `R = S_{N/M} ⊗ H_M^T`, `M ∈ {12,20}` — a
dense `H_M^T` inside contiguous M-blocks plus `log2(N/M)` Sylvester stages, at `M + log2(N/M)` ops per
element (384 -> 17, 1280 -> 26, against 384–1280 for the dense form). Consumers are the quantized KV
rotation and folded weights.

**Two facts decide it.**

1. `H12`/`H20` are orthogonal but **non-symmetric**, so `R` is not an involution — unlike Sylvester and
   unlike ConvRot's symmetric `H4^{⊗n}`. NInfer's KV codecs rely on running the *same* transform on the
   encoder and on the reader's query, which a non-symmetric `R` breaks.
2. It is unreachable upstream by construction: `n_rot_k` is derived by doubling from 64 and the
   generator asserts power-of-two, and the kernel itself asserts `N/M >= 32`, which excludes 192/320 —
   the odd widths real MLA models have. No model becomes rotatable through it.

**NInfer counterpart.** `normalized_hadamard_d256_inplace` is the D256 counterpart at identical FLOP
count, used on the KV append and on the query side of every paged decoder, with an H128 FWHT for EXL3
(`src/quantize/exl3/hadamard.cu`). Every quantized KV plane pins `head_dim == 256`, and 640 = 2^7 x 5
admits no Sylvester Hadamard at all, so the 640-wide global row was already resolved by per-slice
rotation (`docs/maintainer/gemma4-26b-plan.md:804-806`).

**Magnitude.** ~0.5–0.9 MFLOP/token for the rotation, against an O(10 ms) decode step: **<0.05%**.
This is a capability question, never a time question. If it is ever revisited, the cheap check is the
existing FP64 KV round-trip oracle: does one whole-640 `H32 ⊗ H20` beat two Sylvester slices on score
error?

## A5 — MXFP6 (#22671)

**Mechanism.** `MXFP6_E2M3`: a 6-bit sign-magnitude code indexing an E2M3 magnitude table, with one
UE8M0 byte per K32 sub-block (`QK_MXFP6 = 64`), stored as an 832-byte tile (16 rows x 64 K) in the
GGUF — 6.5 bits/element, of which ~0.25 is scale-array padding — driven through ggml's dormant
`blck_size_interleave = 16` plus a 32-byte tensor header. The CPU `vec_dot` compares against Q8_0 with
integer table-lookup MACs and no float dequant in the hot path. The PR is 16 files, all CPU/gguf-py;
master has no MXFP6 MMQ/MXVariant, though its merged MXFP4 already runs the hardware block-scale MMA.

**NInfer counterpart.** `src/ops/common/mma.cuh:64-78` already issues
`mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0`
with UE8M0 `0x7f` as the unit scale (the comment records why: on this GPU the block-scaled form avoids
the plain form's half-rate FP32 accumulation). `block_scale_k16_m128x4_v1` is a shipped tensor layout
(`docs/maintainer/storage-layouts.md:16,253`), fp32 divisors are already in-plane, and CUDA 13.3
provides the conversions (`cvt.rn.satfinite.e2m3x2.f32`, `/usr/local/cuda/include/cuda_fp6.hpp:250`).
NInfer has no 6-bit minifloat weight format; `q6_g64_fp16` is the same 6.25 bits/weight but integer and
not an MMA operand type.

**KV: not transferable.** An e2m3/ue8m0-32 D256 row is 400 B/token/head, the same point K8V4 already
occupies, and the recorded codec accuracy table (`docs/maintainer/paged-kv-cache.md:289-300`, 261,167
tokens) shows nvfp4 at 4.50 bits is both smaller and better at 32k than k8v4, with e2m3's K-side error
worse than the fp8 K that K8V4 uses. There is no accuracy basis for a sixth KV profile here. If a
capacity need ever appears, the cheap variant is keeping the FP16 row scale and swapping only the code
plane (388 B, same score error, same decode instruction count).

**Weights: needs measurement, and this is the half the PR informs.** A W6A8 route would ride the
shipped 1X ue8m0 MMA at a bit budget identical to `q6_g64_fp16`, so the entire question is accuracy
plus operand-ness; [INFERENCE] from the measured GEMV bandwidth it would decode at ~1.15x q6 and 0.77x
nvfp4. Check: extend the existing per-group format-substitution harness
(`tools/verify/gemma4_sensitivity.py`) with `mxfp6_e2m3`/`e3m2` in both scale pairings (ue8m0-32,
fp16-row-256) against real checkpoint weights, gating on `q6_g64_fp16` at equal bits and targeting
displacement of `fp8_e4m3fn_row`.

## A6 — concurrent streams for linear attention (#21897)

**Mechanism.** Not op-specific: `ggml_backend_cuda_graph_optimize` auto-detects fork/join regions
(fan-out 3–4, `ggml_nrows <= 8`) and puts every in-region node on a branch stream, with explicit CUDA
events for fork and join; it survives graph capture and is gated behind `GGML_CUDA_GRAPH_OPT=1`. For
Qwen3.5's GDN block the fork is the normed input with four consumers (wqkv, wqkv_gate, ssm_beta,
ssm_alpha) joined at the gated-delta-net op. Measured gain: 1.04–1.07 at pp512 on a 5090, with the
author noting the remaining effect is TG-only and suspecting the `graph_optimize` cost itself.

**NInfer counterpart.** NInfer already overlaps the same work by **PDL** instead of streams:
`src/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_conv_snapshot.cu:122-159` is templated on
`<TriggerPdl, JoinPdl, Dependent>` and launches the Q5 value/z GEMV (`TriggerPdl`) followed by the Q4
q/k GEMV (`Dependent` + `JoinPdl`, joining at exit) — live for the dense 27B, since the recipe assigns
Q4 to attention/GDN query and key and Q5 to everything else
(`tools/convert/official_recipes.py:100-110`). There is one compute stream and one transfer-only
stream (`src/core/device.cu:90-107`), the decode/verify step is captured whole
(`src/core/decode_graph.cpp`), and PDL entry-trigger/exit-join is already used across the GEMV family.
Other formats need no overlap: nvfp4 decode is a single GEMV over q+k+v+z, nvfp4 prefill is one grouped
MMA, MoE shared and routed experts share kernels. The allocator hazard the PR had to fix (keep-alive
nodes) has no analogue here — the arenas are four distinct buffers.

**Verdict and bound.** Not transferable as a gain; the pair already exists. Prefill: ≈ 0, since the
GDN 6.4% is strictly dependent. Decode: <= ~5% best case (the Q4 half is 4096 of 16384 rows in the GDN
input projections) if the pair were serialized. The one open question is already recorded in-repo —
whether PDL pays at all under a replayed graph (`docs/exl3-quant-plan.md:891-893`). Check: `nsys` a
single decode step and look at whether the q5/q4 GDN GEMV intervals overlap; if they are strictly
back-to-back while the recurrent kernel leaves SMs idle, the PR's event design is the fallback
(magnitude ~25% of GDN projection time).

---

## NInfer-internal leads found by this survey

Not ports; each is a question this repository can answer with its own benches.

1. **A3 was the one Tier A item worth pursuing, and it is now measured: pursue it.** Rotation collapses
   the A8-attributable drift on Gemma 4 31B from +0.174 to +0.0115 (15.1x, aggregate over all 60
   layers), so the W8A8 route becomes affordable at the +47% pp8192 it already measured, and the
   required butterfly is the one already in the tree (Sylvester H256 measured equivalent to ConvRot's
   regular H256). Detail and remaining checks in the A3 section above.
2. **Routed MoE has no TMA** while dense NVFP4 prefill does (`sparse_moe_prefill_kernels.cu`: 0 TMA
   occurrences). Now measured: nvfp4 sits at 69.0% of DRAM peak at T=1024 (368.6 µs, 160.2 TFLOP/s) and
   337.3 TFLOP/s at T=8192 (18.2%), q4-q5 at 40.2% of peak at T=1024 (647.2 µs) — a shape-confounded gap
   against the dense GEMM's 55.84% of FP4 peak, worth an occupancy investigation but not a port.
3. **PR #26704's carried-over number** (llama.cpp 3.14x behind vLLM on the same GEMM) is a reminder
   that NInfer's own dense NVFP4 sits at 55.84% of peak; the five-shape table in
   `docs/maintainer/linear-benchmark.md:511-517` is the baseline to beat, not llama.cpp.
4. **MoE tile width vs per-expert count** — "~half of every MMA tile is padding"
   (`docs/maintainer/gemma4-26b-plan.md:720`): a `kGroupedBN` 64 -> 32 A/B on the gemma grouped route,
   checking KLD at the same time.
5. Carried forward from earlier in this session, not part of the 137: the GDN chunked kernel's
   `value_tile` picks `DV=64` at 48 heads where a forced `DV=16` measured ~10% faster on the op
   (96.3 us vs 106.5 us at T=1024; tile 8 worse at 124.9), worth ~0.6% end-to-end and only if that
   heuristic is touched anyway.

---

## Tier B results

Four clusters were evaluated first, chosen for "could this change a decision here". Three close as
*already present*; one that names a genuinely measurable internal item is B1.

### B1 — KV dequant on the fly (#30202 + #27140)

**Not transferable, and confirmatory.** #30202 deletes the F16 KV materialization for `q4_0`/`q8_0`
only (D=128/256/512, 1-8 queries) by dequantizing rows into the same smem tile inside the MMA kernel
(`need_f16_KV = type_KV == F16`, `flash_attn_ext_f16_load_tile_q`); the win is traffic, not launch
(~1.0x at 8k depth, 1.03-1.12x at 32k, **1.35-1.48x at 128k**, because the deleted pass is
O(depth)). #27140 does not delete that pass — it makes the same copy 16x faster for
`q4_0/q4_1/q5_0/q5_1` (bit-identical output) to work around a llama.cpp config fallback, so it is inert
here.

NInfer's paged decoders already do the stronger form: all four quantized profiles read the code/scale
planes and dequantize into a per-key-tile FP16 stage inside attention
(`src/ops/softmax_attention/dense/causal_cache/int8/grouped_mma.cuh:68-72` and the query rotation at `:224`,
`src/ops/softmax_attention/dense/causal_cache/nvfp4/grouped_mma.cuh:290-294`, `src/ops/softmax_attention/dense/causal_cache/fp8/tile_io.cuh:19`, `src/ops/softmax_attention/dense/causal_cache/k8v4/grouped_mma.cuh:60-61`), and
`int8_g64_codec.cuh:3-5` states no transcode kernel exists. The one exception is the Gemma 4 sliding
ring, which decodes into BF16 scratch (`src/ops/kernel/dense_kv_ring.cuh:204`, called at
`src/models/gemma4/forward.cpp:615-619`). That exception is already recorded, with its size, in
`gemma4-26b-plan.md:890-891` (decode 350.80 tok/s bf16 against 334.99 int8 / 320.21 nvfp4, i.e.
**4.5-8.7% of decode**) and `:818-829` ("Step 2 must not inherit Step 1's decode into BF16 scratch …
Step 2's decode therefore belongs in the attention's tile loads"), including the argument that on the
global cache a scratch decode would be a net loss at 262144 tokens. So the PR is third-party
confirmation of a planned internal step, not a new lever.

Correction to this survey's own suggested instrument: `ninfer_context_kv_materialize_bench` is the
DFlash2 five-layer context-state benchmark, not a quantized-KV-read bench. The right check is a
`--kv-dtype` sweep plus one `nsys` capture, flipping if `dense_kv_ring_decode_kernel` is >= 2% of a
decode token.

### B2 — pipeline parallelism with MoE experts in host RAM (#29963)

**Not transferable.** The PR adds `ggml_backend_tensor_set_async` on a split's own stream (one copy per
device, whole `ffn_*_exps` tensor, dependency by stream order plus the split's events), measured on 3x
RTX PRO 4000 PCIe5 P2P: Flash-Next Q8_0 PP 1009 -> 1808 tok/s. It is cross-device overlap on a
multi-GPU split — and its own table has generation regressing in one configuration (42.0 -> 29.2
tok/s), with no gain in mmap load mode.

NInfer is single-GPU by product definition and already pipelines one layer deep, more strongly:
`qwen4_expert_pager.h:21-23` documents two whole-layer banks filled on the pager's own stream with the
next layer's fill overlapping this layer's compute, `fill()` fences on the previous release with
`cudaStreamWaitEvent(stream_, released_[b])` (`.cpp:96-176`), `qwen4_text.cpp:446-447` starts the next
layer's fill before this layer computes, and the narrow route splits fetch from compute in two passes
(`src/ops/offload_moe/kernels.cu:976-985`, "cache-resident experts run while the fetch fills the missed
slots"). What is not overlapped: `snapshot_residency`'s two syncs plus a D2H once per forward, the drain
at each chunk boundary, and the fact that fill launches are host-driven.

Bound: the recorded Flash-Next prefill has **0.172 s of device-blocked wait in 2.221 s (7.8%)**
(`profiles/bench/flash-next-host-bound-2026-10-09/sweep_p8k.jsonl`), which caps any transfer/overlap
idea here; the host path is the lever, and the landed version of that is chunk widening
(`startup.h:30-40`, `kQwen4PrefillChunk = 4096`, "19-21% on the MoE over 1024").

### B3 — sync, wait and allocation discipline (#26487, #29796, #29768, #28044)

**All four already answered; no port.**

- #26487 (blocking sync instead of CPU spin) — NInfer has the knob already, as a superset:
  `NINFER_CUDA_SYNC = spin|blocking|yield|auto` (`src/core/device.cu:24-30`, default
  `cudaDeviceScheduleSpin`, applied via `cudaSetDeviceFlags` at `:79-83`), reported as
  `load.cuda_sync_mode` (`src/runtime/engine/engine.cpp:215`), documented (`docs/cli.md:285-293`), and
  covered in CI for all four modes plus invalid/empty (`tests/cmake/CoreTests.cmake:11-25`).
- #29796 (skip redundant sync) — NInfer syncs **once per decode round for a real dependency**
  (`src/models/qwen3_5/program/decode.cpp:418-424`, bracketed by `timing.begin_wait/end_wait`), immediately before the host
  reads the egress struct the graph wrote D2H (`:72-75`, `:430-435`). There is no second backend-level
  sync per step to elide.
- #29768 (avoid repeated warmup) — warmup is startup-only and explicit, every (batch x frontier)
  profile is captured once at load, and runtime is `cudaGraphExecUpdate` + `cudaGraphUpload` + launch
  (`graphs.cpp:83-105`, `decode.cpp:133-142`): no runtime recapture, so the warmup the PR removes is
  never paid.
- #28044 (graceful pool-allocation failure) — no allocation happens inside graph replay at all (scratch
  is the pre-sized arena; overflow is `std::bad_alloc` at `arena.cu:234`), a refused reservation is an
  optional, and the worker catches it per request, returns **429**, and keeps serving
  (`engine_core.h:2280,2304`, `kOomMaxRecoveries = 8` at `:2369`), with the in-flight-capture unwind
  already present (`decode_graph.cpp:59-78`).

The only genuine deltas are policy and coverage, not code: after 8 consecutive OOM recoveries NInfer
stops the worker (later requests 503) where llama.cpp's post-PR shape is 500-and-always-serving, and
**nothing drives `EngineCore` through an OOM recovery** — `tests/test_engine_oom_recovery.cpp` does not
exist. That is a test-coverage gap worth closing on its own merits (tiny KV capacity, two requests, one
429 and one decodes), independent of any PR.

### B4 — MoE route bookkeeping (#26631, #26294, #30477)

**Not transferable.** `-1` in #26631 is a per-slot skip a *graph* emits, and no merged llama.cpp graph
emits one today (its tests inject it); the model that needs it carries a TODO and clamps zero-expert
routes onto a dummy expert. NInfer has no `-1` anywhere and handles an empty route by *absence*: the
scan emits a job only for a nonempty tile (`src/ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu:296-305`)
and consumers exit through `count <= 0` (`:740`, `:1131`) or a device-counted `live()`
(`:1415`, `src/ops/offload_moe/a4_kernels.cu:55-77`).

The duplicate-id fixes (#26294, #30477) defend against an id repeated within one token, which NInfer's
selectors cannot produce: the Gemma router drops a chosen slot to `-inf` so it cannot win again
(`src/ops/kernel/gemma_moe_router.cuh:255-269`), the offload router carries a `taken` bitmask
(`src/ops/offload_moe/kernels.cu:117-136`), and `sparse_moe_route.cuh` orders totally. The suite asserts it
(`tests/ops/test_offload_moe.cpp:135-137`). Even under a hypothetical repeat, its bookkeeping ranks
*assignments* rather than tokens — `local_rank[assignment] = atomicAdd(&counts[ids[assignment]], 1)`
(`sparse_moe_prefill_kernels.cu:236-240`), `rank[a]` (`a4_kernels.cu:211`) — so an id would inflate a
count, not collide in rank space; llama.cpp's defect was token-keyed. Neither PR bears on the recorded
determinism constraint (`gemma4-26b-plan.md:452-461`): that rejection was an `atomicAdd`-filled
accumulator, while these are dropped *index* entries leading to uninitialized ids and illegal access.

One robustness note, not a bug: the **expert-id domain is unvalidated where ids enter** —
`counts[ids[assignment]]` indexes a `__shared__ int counts[kExperts]` (`sparse_moe_prefill_kernels.cu:236-240`)
and `used[ids[i]]` likewise (`src/ops/offload_moe/kernels.cu:236-238`), while the host-side checks cover dtype
and shape only (`src/ops/offload_moe/offload_moe.cpp:247`, `:335`). Ranges hold today because only in-tree
routers produce ids and they are provably within `[0, E)`, so this is hardening; the cheap proof is to
inject a repeated id and an all-one-expert route into `tests/ops/test_offload_moe.cpp:575-596` against
the existing oracle, which is expected to pass.

## Qwen4 / Qwen3.8-Flash-Next sweep (llama.cpp, 16 PRs)

Query `repo:ggml-org/llama.cpp qwen4 type:pr` → 16 PRs. Five clusters evaluated; **none is a port, and
three merged ones are behind or level with NInfer**:

| cluster | PRs | verdict |
|---|---|---|
| masks | #29824, #30480 (both merged) | **nothing to port.** They optimize a materialized mask (`[1, n_kv+n_sel, n_tokens]`); NInfer has no mask at all — selection is an int32 index list built on device (`src/ops/sparse_attention/kernels.cu:424-531`) and consumed by index (`:675-708`). Reducible amount: exactly zero. #29824 also fixes a Metal-only `repeat` slow path and its author reports it neutral on CUDA. |
| sparse FA | #27970, #28770 (both merged) | **not transferable, NInfer stronger.** Same selection rule and block granularity (`include/ninfer/ops/sparse_attention.h:25-43`, `qwen4-exp-model.md:39`: indexer 4×128, budget 2048 / block 4 → 512 blocks + tail = 2051), but llama.cpp materializes an F16 mask and compacts it to indices before the MMA, while NInfer gathers only the selected keys through the paged block table and dequantizes all five KV profiles. Only idea worth borrowing: the model-agnostic `n_kv_max` hint. |
| expert cache | #29887 (merged, `d6cf9acb`) | **not transferable, NInfer level or ahead.** The PR walks an LRU list on the host, copies ids D2H and `ggml_backend_synchronize`s per layer per call, and is prefill-inert by construction (batches > 32 tokens skip the cache). NInfer resolves misses in a device kernel (`src/ops/offload_moe/kernels.cu:216`), splits resident/missed work in two passes (`:976-985`), auto-sizes capacity, and already records its eviction result ("recency beats LFRU by 3.9-4.0% more misses at 8,237 slots; do not re-add frequency protection without a new measurement", `:200-215`). |
| format / multimodal MTP | #30316 (open) | **nothing needed.** NInfer's format excludes padded expert banks and rejects them loudly, requires `mrope_section {11,11,10}` (`src/models/qwen3_5/config.cpp:82-105`), and multimodal MTP already exists (`qwen4_mtp_stem`). |
| graph splits | #27880 (merged, `6fe7498`) | **not transferable, and the premise is dead.** It is a graph-construction *ordering* change with upstream-measured parity (665.65 -> 665.27 tok/s); NInfer has no backend boundaries to split. |

### Measured Qwen4Exp prefill composition (in-tree nsys exports — mind the dates)

**Every Qwen4Exp export in the tree is from 2026-09-24/25.** `profiles/nsys/flash_pp3072_c4096.sqlite`
(2026-09-24, chunk 4096, staged expert route; 5,370 ms run, device union-occupied **99.9%**, kernels
3,927 ms + copies 1,438 ms) predates that day's own QSA rewrite, so its attention row is history:

| kernel | share of occupied |
|---|---:|
| `moe_gate_up_kernel` + `moe_down_kernel` (staged expert GEMMs) | **47.6%** |
| `q8_rowsplit_gemm_simt_kernel` | 10.8% |
| `qsa_attention_kernel` (**the pre-rewrite kernel**, 46.4 ms per call, 12 calls) | 10.4% |
| host→device expert streaming (copies) | ~27% of the run |

That 46 ms/call figure is what prompted a look at the QSA kernel, and the look answers itself:
`48f2bdb2 perf(sparse_attention): run BF16-KV QSA attention on Tensor Cores` is dated **2026-09-24**,
and `qsa_attention_kernel` no longer exists in `src/` at all — the launcher
(`src/ops/sparse_attention/kernels.cu:1215-1248`) only ever instantiates
`qsa_attention_mma_kernel<Storage>`. The post-rewrite export (`profiles/nsys/flash_pg2048.sqlite`,
2026-09-25) confirms it: **0.90 ms per call** (60 calls, 53.9 ms of 940.5 ms of kernel time), with
`qsa_score_kernel` at 0.00 ms per call — against 0.11 ms per call in the pre-rewrite build. A third
tuning landed later: `08e27d21 perf(sparse-attention): target 8192 CTAs for the QSA score split`
(2026-10-09).

So the QSA attention lever was pulled before this survey started, and the score/selection lever with
it. What the Sep 24 export still describes correctly is the **MoE side** — ~74% of the run in staged
expert GEMMs plus the streaming into them — and that path has not been rewritten since. There is no
post-Oct-9 Qwen4Exp profile to check it against: the newest attempt (`profiles/nsys/c1/fn_plain2`,
2026-10-13) left a 0-byte `.sqlite` and its `.nsys-rep` is missing (the capture failed).

Two claims from the cluster evaluations do **not** survive any of this: the sparse-FA evaluation's
"attention MMA 3.4-4.4% / QSA 13-16.5% of pp8192 GPU time" (no such export exists in-tree) and its
"bound <= 4-7% of prefill". The launch census from the graph-splits evaluation *is* confirmed to the
digit (3,927 ms kernels, 1,438 ms copies, 99.9% of 5,370 ms).

### Final composition after the four prefill changes (2026-10-21, RTX 5090, `nsys`)

Same prompt and flags as the table below, after the pooled-key grouping, the n-gram read-ahead, the
attention widening and the MTP bank's route: wall **10.36 s**, kernels **10.30 s** (47,447
launches), host->device 216 GB in 4.35 s (hidden), against 12.98 s / 12.87 s at the start.

| kernel group | before | after |
|---|---:|---:|
| dense projections (`q8_a16_mma`) | 3.01 s | 3.08 s |
| routed experts, `nvfp4_w4a4_mma` | 2.32 s | 2.49 s |
| QSA select/score | 2.68 s | **1.13 s** |
| QSA attention | 1.33 s | **1.01 s** |
| hyper-connection | 0.72 s | 0.77 s |
| router + A4 route plan | 0.63 s | 0.66 s |
| expert cache/fetch | 0.42 s | 0.46 s |
| GDN + conv | 0.34 s | 0.34 s |
| norms/rope/elementwise | 0.27 s | 0.30 s |
| routed experts, SIMT A16 (the MTP bank) | 1.10 s | **0.03 s** |

Wall-clock, two requests per process (the second one warm), same settings: **12.76 -> 9.89 s**
(5,057 -> 6,525 tok/s) warm, 16.68 -> 11.55 s with a cold page cache, and `--ngram-residency
stream` 18.89 -> 11.43 s.

### Where the remaining 10.3 s of device time sits, and why `program_submit` is not a host lever

NVTX phase ranges over the same capture: `prefill.layer.gdn` **6.19 s** (360 instances),
`prefill.layer.full` **3.77 s** (120) - the QSA layers - and `mtp.forward` 0.42 s. Per instance that is
17.2 ms for a GDN layer and 31.4 ms for a QSA one at 16,384 tokens; the QSA layer adds 9.4 ms of
selection and 8.4 ms of attention to the same MoE and dense work.

`program_submit` reading 9.7 s is an accounting artifact of being *device*-bound, not a host cost: the
kernels are 10.30 s of a 10.36 s wall, and running the same prompt with `NINFER_CUDA_SYNC=blocking`
leaves `program_submit` at 9.75 s while `device_wait` falls from 1.35 s to 0.43 s. The Program's
submit phase spans the device's execution because each chunk's submission is interleaved with it, so
what is left of the host is ~0.25 s. Earlier notes in this section that named the host as the lever
were wrong; the n-gram gather was a real cost, but only off the page cache.

Two levers were priced and both need a numerics decision rather than more tuning:

- **Dense projections (3.08 s, 30%).** The Q8-A16 kernel is at 80% of the bf16 peak where it can be
  measured; the way down is an int8-activation Q8 route, which changes prefill activation precision
  on every dense projection (`w4a8-prefill.md` measures that step at 1.6-1.7x on the Q4 route).
- **QSA score (1.13 s, 11%)** is shuffle-bound, not traffic-bound: four query columns per CTA
  (dividing the pooled-key traffic by four) measured *slower* (38.2 ms against 33.5 ms for a
  16,384-column chunk at 65,536 tokens), and removing the 32-lane butterfly by ablation gives
  34.1 -> 18.7 ms. So the 20 shuffles of four bit-identical butterflies are ~45% of the step, and an
  eight-lane-per-head reduction (five shuffles) would buy ~25-30% while changing the summation order,
  i.e. which blocks win a near tie.

Still open besides those: the BF16 KV attention profile (77,792 B of shared, one CTA per SM), and the
routed experts at 122 TFLOP/s (2.49 s) against the dense nvfp4 route's 935.

### Post-Oct-9 composition (2026-10-21, RTX 5090, `nsys`)

A 64,512-token prompt (NIAH fixture), one request, `--max-context 65536 --kv-capacity auto --kv-dtype
fp8 --expert-cache auto --ngram-residency mapped --spec mtp`, idle prefill chunk 16,384, expert cache
8,093 slots. Wall **12.98 s**, kernels **12.87 s** (47,258 launches), host->device **215 GB in 74,150
copies = 4.09 s**, which the kernels hide. The machine had other users, so read the ratios, not the
absolute rate; the A/B below is what attributes the change.

| kernel group | time | share |
|---|---:|---:|
| dense projections (`q8_a16_mma`) | 3.01 s | 23.4% |
| QSA select/score | 2.68 s | 20.8% |
| routed experts, `nvfp4_w4a4_mma` (48 text layers) | 2.32 s | 18.0% |
| QSA attention | 1.33 s | 10.3% |
| routed experts, SIMT A16 (`mtp/layers/0` only) | 1.10 s | 8.6% |
| hyper-connection | 0.72 s | 5.6% |
| router + A4 route plan | 0.63 s | 4.9% |
| expert cache/fetch | 0.42 s | 3.2% |
| GDN + conv | 0.34 s | 2.6% |
| norms/rope/elementwise | 0.27 s | 2.1% |

The score step was then given four blocks per warp iteration (`perf(sparse-attention): keep four
pooled-key loads in flight`): back-to-back captures of the same prompt give kernels **13.30 s ->
11.48 s** and the score step **2.52 s -> 0.86 s**, selection bit-identical; nothing else in the
table moved by more than run noise.

That pair also shows the shape of the remaining problem. The wall followed only 0.45 s of the 1.82 s
the GPU gave back (15.51 s -> 15.06 s, `device_wait` 1.48 s of it, `program_submit` 14.21 s ->
13.77 s): on that machine the host was at ~98% duty and the device had 11.48 s of work for a 15 s
prompt, so prefill is `max(host submission, device)`.

The host's largest single cost was then measured rather than inferred: `perf` samples one prefill and
puts 92% of the prompt's CPU time under `PleGather::decode_row` (self time in the scalar E4M3->BF16
loop plus page-fault frames). Note what that number is and is not: it is *threaded* CPU time - the
gather fans out over up to 16 workers - so its wall share is small when the table is in the page
cache (~9 ms per 16,384-token chunk, 0.3% of the prompt) and large when it is not. Gathering the next
chunk a chunk ahead (`PleReadAhead`, see the model reference) fixed exactly that: 12.51 -> 10.86 s
warm, 16.63 -> 12.53 s with the page cache evicted, 18.89 -> 12.41 s under `--ngram-residency
stream`. What remains of `program_submit` (~9.7 s against a 9.7-11.5 s device) is not this gather and
has not been attributed yet.

Two of those were then taken: the QSA attention kernel staged every quantized tile's codes in
shared memory (32,768 B of a 78,016 B block, on a part with 100 KB per SM = one CTA per SM), and
widening straight from the paged codes leaves 45,248 B and two CTAs: attention 1.445 -> 1.027 s and
kernels 12.215 -> 11.725 s on the same prompt. The Q8-A16 dense route turned out to be at 80% of the
bf16 peak already (`ninfer_linear_bench`: 209 TFLOP/s at [2560,6144]x16384, 199 at [1280,2560], 154
at [10240,320] against a 262.7 TFLOP/s peak), so the 3.25 s dense slice is the shape mix of the
fused `gdn_input_proj`/`attn_input_proj`/`hyper_connection` ops, and beating it needs an
int8-activation Q8 route rather than a retile - a numerics decision (`w4a8-prefill.md` measures that
step at 1.6-1.7x on the Q4 route).

What is left, in order: the host path, the dense projections' activation precision, and the MTP
bank's missing A4 divisor.

### The staged route's SIMT branch runs at ~5.4 TFLOP/s (measured)

The Sep 24 export's 47.6% is the *staged* expert route (`moe_gate_up_kernel` + `moe_down_kernel`), and
the in-tree bench can measure that route directly — `ninfer_offload_moe_bench` builds a 512-expert
nvfp4 bank and takes arbitrary token counts, and `T >= kQwen4StagedColumns` (103) selects the staged
route, so a call at T=1024 or 3072 is the production prefill call:

| T | median per layer-call | distinct experts | implied compute rate |
|---:|---:|---:|---:|
| 1 | 52.6 us | 10 | (decode; 525 GB/s of distinct expert bytes) |
| 64 | 923 us | 371.5 | 6.8 TFLOP/s |
| 1024 | 18.0 ms | 512 | **5.6 TFLOP/s** |
| 3072 | 55.3 ms | 512 | **5.4 TFLOP/s** |

At T=3072 that is 55.3 ms x 48 layers = 2.65 s, matching the export's staged GEMMs (1,833 + 718 =
2,551 ms), so the bench reproduces the production cost. For the *same* nvfp4 codec, the repository's
*sparse*-MoE route measures **160 TFLOP/s at T=1024** and 337 at T=8192 (`ninfer_sparse_moe_bench`,
which runs its own 35B-A3B geometry - H=2048/E=256 against Qwen4Exp's H=2560/E=512; a 28x gap is far
too large to be geometry), and the dense nvfp4 route 935.8 TFLOP/s
(`docs/maintainer/linear-benchmark.md:511-517`) - this is what showed the staged route was leaving the
tree's own capability unused, but the actionable bound is the next section's, which measures the
staged route's *other* branch rather than another route's rate. The staged kernel is SIMT FP32: it dequantizes every
weight element into a shared `float` tile once per job (`nvfp4_value`, `src/ops/offload_moe/kernels.cu:530-595`;
a job is ≤ `kJobTokens` = 64 assignments sharing a slot, `jobs.cuh:12`), and gets four FMAs per staged
weight, so the dequant/staging work rivals the FMA work [INFERENCE from the kernel shape plus the
5.4%-of-FP32-peak measurement]. The bench output is *not* evidence of a traffic problem: 1.4 GB of
distinct expert bytes per layer-call is read, and the `GB/s (touched)` column is just that divided by
the call time.

### The same route's tensor-core branch is implemented and dormant

5.4 TFLOP/s is only the branch this artifact selects. One stored scalar away sits a second
implementation of the identical route (`src/models/qwen3_5/execution/qwen4_text.cpp:457-464`):

```cpp
if (p.bank.gate_up_input_divisor > 0.0F) ops::moe_experts_a4(...);  // W4A4, mxf4nvf4 tensor cores
else                                     ops::moe_experts(...);     // SIMT A16, measured above
```

`moe_experts_a4` quantizes the activation to NVFP4 with the bank's per-layer input divisors and runs
gate/up and down on `mxf4nvf4` (`include/ninfer/ops/offload_moe.h:271-276`; landed as `ff6c6213` and
`a960ee36`, both 2026-09-24). `OffloadMoeParameters` fills the divisor only when *every* expert of
the layer is `AllowA4` with an identical divisor per projection
(`src/models/qwen3_5/execution/parameters.cpp:440-471`). The recipe grants `AllowA4`
(`tools/convert/official_recipes.py:428-435`). The divisor is metadata, never weight data - A16 does
not read it (`src/ops/weight_input.cpp:92-93`).

**Correction (2026-10-21): this section described the 2026-09-24 artifact, and the shipped
`qwen3_8_flash_next_nvfp4.ninfer` is not it.** That file's conversion report names
`/mnt/storage/models/qwen3.8/flash/ninfer-flash` as its source, and every one of its 75,264 text
expert uses carries `AllowA4` plus an `activation_input_divisor` auxiliary whose value is common to
the layer and projection (verified by reading the artifact: 48 layers x 3 projections, one value
each) - so all 48 text layers run the `mxf4nvf4` route, and an in-engine profile of a 64,512-token
prefill shows the `nvfp4_w4a4_mma_kernel` launches, not the SIMT pair. What the artifact does *not*
carry is a divisor for `mtp/layers/0` (1,536 uses, `AllowA4` and no auxiliary), so that one bank
falls to the SIMT A16 route: 1.10 s of the 12.87 s of prefill kernel time measured on 2026-10-21,
against 2.32 s for the other 48 layers together. The rest of this section (the SIMT branch's rate,
the cuBLAS reference, the W4A4 measurement) still stands as the qualification of both routes.

**Solved for the MTP bank without new metadata (2026-10-21).** What that block was missing turned
out to be two things, and neither needed a calibrated value. It carries no *activation* divisor, and
the divisor cancels exactly - the quantiser multiplies by it and the MMA epilogue by its reciprocal -
so the model now sets 1.0 for a bank that permits A4 and stores none: it only chooses which E4M3 bin
the per-16 group scale rounds into, and 1.0 puts a group whose maximum is O(1) at full mantissa
precision. The block's *weight* divisors were also one value for a whole stacked matrix
(`divisor_rows` 655,360) rather than one per expert, and `moe_experts_a4` rejected anything that was
not 640/2560 rows - although its epilogue addresses the table by output row and `row / 655360` is
simply 0. Relaxing that check to `divisor_rows > 0` moved the bank onto `mxf4nvf4`: the SIMT pair
(0.806 + 0.316 s) disappears from a 64,512-token prefill and kernels fall 11.725 -> 10.287 s, with
the greedy completion token-identical and draft acceptance unchanged (353/474).

What would still be left for a *calibrated* placement is the older, optional route:
- convert from `nvidia/Qwen3.8-Flash-Next-NVFP4` (ModelOpt 0.46, 119.6 GB, tags `ModelOpt`/`FP4`),
  whose 73,728 expert `input_scale` tensors are the calibrated divisors, reduced to one value per
  layer and projection - the A4 route quantizes the activation once for the whole layer, which is why
  `parameters.cpp` demands a common value - through the converter's `auxiliary_overrides`
  (`tools/convert/methods.py:405-410`); or
- calibrate the layer-level activation scales on the current artifact and write them in.

Precedent: the 35B-A3B artifact ships 30,720 expert uses with `AllowA4` *and* divisors, so W4A4
experts are accepted practice in this tree; what Qwen4Exp lacks is the stored value.

Measured with that option (`--format nvfp4 --tokens 64,1024,3072,8192 --iters 50`, two interleaved
A16/A4 rounds in one session; the rounds agreed to within 0.6% on A16 and 0.2% on A4):

| T | SIMT A16 | W4A4 tensor-core | ratio | A4 implied rate |
|---:|---:|---:|---:|---:|
| 64 | 916 us | 721 us | 1.27x | 8.7 TFLOP/s (DRAM-bound: 1.4 TB/s of expert bytes) |
| 1024 | 18.33 ms | 1.009 ms | 18.2x | 99.8 TFLOP/s |
| 3072 | 55.06 ms | 2.979 ms | 18.5x | 101.4 TFLOP/s |
| 8192 | 148.02 ms | 7.905 ms | 18.7x | 101.9 TFLOP/s |

The A16 column reproduces the export's staged cost to 0.6% (55.06 ms x 48 = 2.64 s against
2,551 ms), so the bench is measuring the production call; the A4 branch is ~100 TFLOP/s against 5.5,
flat across prefill widths, and falls to DRAM-bound only below the staged threshold (T=64 spends its
time streaming 368 distinct experts).

Bound: the two staged kernels are 47.6% of the Sep 24 chunk-4096 prefill. A 4096-token chunk routes
40,960 rows, i.e. 19.3 TFLOP of expert GEMMs (three K x N = 2560 x 640 matrices), which the A4 branch
would run in ~190 ms instead of 2,551 ms - **~44% of that prefill**, the same ceiling the earlier
extrapolation reached, now anchored on a measurement rather than the sparse route's rate. The
remaining unknown is accuracy, not speed.

### What is actually left (all NInfer-internal, none from llama.cpp)

1. **Expert streaming per chunk** — the recorded lever, already exploited: chunk widening takes the
   same 30,627-token prompt from 2,095 to 3,146 tok/s (`src/models/qwen3_5/program/planning/startup.h:29-45`).
2. **The staged experts' A4 divisor** — the measured 5.4 TFLOP/s is the SIMT branch that this
   artifact's missing divisor selects; the same route's W4A4 tensor-core branch is implemented and
   dormant (`qwen4_text.cpp:457-464`). One layer-common input divisor per projection flips it (see
   above), and 47.6% of the prefill sits on the slow branch.
3. **Decode expert-miss traffic** — ~147 MB per token at the recorded 88.9% hit rate with 8,059
   resident slots (`profiles/bench/flash-next-host-bound-2026-10-09/sweep_p8k.jsonl`); capacity is the
   obvious but already-spent lever.
4. **#30380's trimmed draft vocabulary** — the one concrete idea in the sweep that NInfer has not
   taken: its Qwen4Exp MTP head is full-vocab and shares the target's head object
   (`src/models/qwen3_5/load/text.cpp:191-196`, projected in `src/models/qwen3_5/execution/draft.cpp:531-537`), and
   `src/models/qwen3_5/program/planning/startup.cpp:1382-1384` pins `ProposalHead::Full` for Qwen4Exp while other
   drafters already use the static shortlist (`tools/convert/proposal.py`). Bound: the head is re-read
   per draft column (~113 GB/s at the recorded C=1 MTP3 rate, ~6% of HBM), and the host expert fetch
   dominates that path, so <= ~5% end-to-end.

## Tier B — cheap to check, plausible

| PR | Title | Angle |
|---|---|---|
| [#30202](https://github.com/ggml-org/llama.cpp/pull/30202) | dequant quantized kv on-the-fly for mma kernel | KV materialization is a named stage here; on-the-fly dequant deletes a pass |
| [#27140](https://github.com/ggml-org/llama.cpp/pull/27140) | slow prefill on small KV quants fixed | same territory |
| [#29963](https://github.com/ggml-org/llama.cpp/pull/29963) | pipeline parallelism with MoE experts in host RAM | same problem as the Flash-Next expert cache |
| [#29948](https://github.com/ggml-org/llama.cpp/pull/29948) / [#28702](https://github.com/ggml-org/llama.cpp/pull/28702) / [#30489](https://github.com/ggml-org/llama.cpp/pull/30489) | fuse FFN gate/up + GLU in MMQ, incl. into the Q8_1 activation quant | interleaved gate/up rows; adjacent to the recorded two-term E4M3 A8 plan |
| [#26311](https://github.com/ggml-org/llama.cpp/pull/26311) | linear weight repack + MMVQ for NVFP4 | layout only; decode GEMV is already ~89% of DRAM at T=1 |
| [#26048](https://github.com/ggml-org/llama.cpp/pull/26048) | fuse w_s scale in epilogue MMQ for nvfp4 prefill | weight-side scale in the epilogue |
| [#28898](https://github.com/ggml-org/llama.cpp/pull/28898) / [#28413](https://github.com/ggml-org/llama.cpp/pull/28413) | quant scales for fp8/nvfp4; OCP FP8 | confirm whether the latter is a block-scaled variant or naming |
| [#29187](https://github.com/ggml-org/llama.cpp/pull/29187) / [#28447](https://github.com/ggml-org/llama.cpp/pull/28447) | fuse GDN alpha/beta projections; RDNA4 GDN | GDN gating fusion; already present here |
| [#26631](https://github.com/ggml-org/llama.cpp/pull/26631) / [#26294](https://github.com/ggml-org/llama.cpp/pull/26294) / [#30477](https://github.com/ggml-org/llama.cpp/pull/30477) | skip MUL_MAT_ID -1; duplicate expert-ID compaction | route compaction and dedup |
| [#29768](https://github.com/ggml-org/llama.cpp/pull/29768) / [#26771](https://github.com/ggml-org/llama.cpp/pull/26771) / [#29796](https://github.com/ggml-org/llama.cpp/pull/29796) / [#28044](https://github.com/ggml-org/llama.cpp/pull/28044) / [#26487](https://github.com/ggml-org/llama.cpp/pull/26487) | graph warmup / recapture after pool flush / skip sync / graceful pool failure / blocking sync | graph and stream lifecycle; CPU spin cost in serving |
| [#24565](https://github.com/ggml-org/llama.cpp/pull/24565) / [#30285](https://github.com/ggml-org/llama.cpp/pull/30285) | Blackwell FA MMA config; head sizes 40/72 | tile-config method; our head dims are 128/256 |
| [#29910](https://github.com/ggml-org/llama.cpp/pull/29910) / [#30077](https://github.com/ggml-org/llama.cpp/pull/30077) / [#26621](https://github.com/ggml-org/llama.cpp/pull/26621) | VGPR spills on Q2_K; slow Blackwell dequant under CUDA 12.8; row padding vs cache-set aliasing | failure classes worth recognizing in our own kernels |
| [#22692](https://github.com/ggml-org/llama.cpp/pull/22692) + [#22691](https://github.com/ggml-org/llama.cpp/pull/22691) / [#29247](https://github.com/ggml-org/llama.cpp/pull/29247) | streamed weights + placement plan; RFC memory-eliding fusions | adjacent to the host tier; the RFC is a design read |
| [#26301](https://github.com/ggml-org/llama.cpp/pull/26301) | dequant-float matvec (mmvdq) for Q4_K/Q5_K/Q6_K | our row-split GEMV already fuses decode |

## Tier C — not relevant here

- Wrong arch or multi-GPU: all HIP/RDNA/gfx tuning, Pascal, Kepler, sm70, NCCL and multi-node
  (#25479, #27558, #28613, #28709, #28714, #29534, #29535, #29536, #29927, #28398, #28616, #28303,
  #25940, #26419, #29050, #29126, #27721, #30338, #28967, #27328, #30344, #30021, #30022, #30370,
  #30099, #29025, #29427, #27379, #26932, #28178, #25384, #28313, #26592, #27523, #27997).
- New llama.cpp model architectures: #25731, #24423/#24427, #26298, #29012, #30306, #25206/#27523/#27267.
- New llama.cpp quant types: #27322, #27324, #27325, #27983, #29676, #30278, #28468, #19941, #29896.
- Generic ggml API and op coverage / housekeeping / other backends: the remainder of the 137, grouped
  in the desktop triage list.
