# EXL3 trellis weights in NInfer — plan

- Repo: `/mnt/storage/Projects/ninfer-ext`. Milestone status is tracked in Section 7 and the progress log.
- First model: **Qwen3.8-27B** from full-precision tensors at `/mnt/storage/models/qwen3.8/full`: `Qwen3_5ForConditionalGeneration`, 52 GB, with MTP and Vision.
- Sources (2026-09-27):
  - exllamav3 `master` `fa714d0`. `dev` (`c5a4b43`) only adds a merge and has no quantizer or kernel changes, so `fa714d0` is the latest EXL3.
  - QTIP (Tseng et al., arXiv 2406.11235), QuIP# (arXiv 2402.04396), and YAQA (the two-sided Kronecker Hessian that `ldlq_2hess` implements).
- License: exllamav3 is MIT, NInfer is Apache-2.0. We hand-write our own implementation; where a constant or algorithmic detail follows exllamav3, a comment cites it.

## Decisions (answered 2026-09-27)

| Question | Decision |
|---|---|
| What "native" means | Same as `q4_g64_fp16` and `nvfp4`: a registered NInfer numeric format and layout, produced **by our converter from full-precision source tensors**. No import of exllamav3 checkpoints. Quantization maths, calibration, codec, loader and kernels are C++23/CUDA. |
| Which EXL3 | Latest only: the **mul1** codebook, integer and half-integer bitrates, scale refit, drift-compensated global-scale search, skew-gated output scales, `-hq` promotion, and trace-based calibration with sensitivity-optimized recipes. No `mcg`/`3inst`. |
| Implementation policy | Best method, hand-written where that is better than porting. No cuSOLVER/cuBLAS/torch in the maths: our own FWHT, `XᵀX`, blocked Cholesky/LDL, LDLQ, Viterbi encoder, packer, refit solver and inference kernels. This matches the repo, which links no CUDA math libraries today. |
| First target | Qwen3.8-27B dense. Flash-Next/MoE comes later. |
| int8-activation GEMV | Deferred. Only ever behind a recipe activation permission. |

---

## 1. EXL3, precisely

Each linear `W` (shape `k × n`, used as `y = x·W`) is represented as:
- `trellis`: one tail-biting bitstream per 16×16 tile, K bits per weight, `16·K` uint16 words per tile;
- `suh` (`[k]`) and `svh` (`[n]`): per-channel sign × scale;
- `W = diag(suh) · H_k · Q · H_n · diag(svh)`, where `H_k` and `H_n` are block-diagonal normalized 128-point Hadamards and `Q` is the decoded trellis.

**Bitshift trellis (L = 16):**
- The tile's 256 weights are in `mma.m16n8k16` fragment order.
- The state of weight `t` is the 16-bit window ending at stream bit `(t+1)·K`, read circularly around the tile.
- Half-integer K alternates n- and (n+1)-bit steps (mask `0xAAAA`), so every tile stays a whole number of 32-bit words.

**mul1 codebook:** `z(s) = bytesum((s·0x83DCD12D) mod 2³²) − 510 ∈ [−510, 510]`, a near-Gaussian integer. The value is `z·κ`, with κ ≈ 1/147.7.

**Quantizer (exllamav3 `quantize_exl3`):**
1. `H = E[xᵀx]` from calibration. Damp the diagonal by `0.025·mean(diag)`.
2. Random input signs, blockwise-Hadamard rotation of `H`, and a 16-block LDL.
3. Regularize: output-channel RMS scales when the input skew is below 0.15 (fraction of RMS on the top 2% of channels); output Hadamard; input-channel RMS scales ÷ 1.2437; input Hadamard; global-scale search on sampled tiles, corrected for LDLQ drift.
4. LDLQ from the last 16-row block to the first. Each 16×16 tile is encoded by Viterbi over 2¹⁶ states, with two-pass approximate tail-biting (rotate by L/2).
5. Refit `suh` and `svh` in the Hessian metric with the trellis fixed (closed form per output column; a k×k solve per input row; 2 alternating rounds).
6. Layers are quantized in order, and calibration runs forward through already-quantized layers.
7. Bits come from a global budget, `-hq`, or a recipe from `sc_measure` + `sc_optimize`.

## 2. Qwen3.8-27B shapes

hidden 5120, intermediate 17408, 64 layers (48 GDN + 16 full attention, `full_attention_interval` 4), vocab 248,320, 1 MTP layer. EXL3 needs both dimensions divisible by 128, for the Hadamard blocks.

| Projection | k → n | Count | EXL3 |
|---|---|---:|---|
| attention q + gate | 5120 → 12288 | 16 | yes |
| attention k, v | 5120 → 1024 | 16 each | yes |
| attention o | 6144 → 5120 | 16 | yes |
| GDN qkv | 5120 → 10240 | 48 | yes |
| GDN z | 5120 → 6144 | 48 | yes |
| GDN a, b | 5120 → 48 | 48 each | **no**: stays BF16 (0.5 MB total) |
| GDN out | 6144 → 5120 | 48 | yes |
| MLP gate, up | 5120 → 17408 | 64 each | yes |
| MLP down | 17408 → 5120 | 64 | yes |
| LM head | 5120 → 248320 | 1 | yes; exllamav3 defaults it to 6 bpw |
| MTP layer (attention + MLP + fc 10240 → 5120) | — | 1 | yes; separate bits (exllamav3 default 4, uncalibrated) |
| Embedding, norms, conv, A_log, dt | — | — | unchanged |
| Vision tower | — | — | unchanged at first |

## 3. Numeric format and layout

### 3.1 Format `exl3_mul1` (bitrate as a format parameter in half-bits: 2…16, i.e. K = 1…8 in steps of 0.5)

- Registered in `tools/artifact/formats.py` and `src/artifact/formats.cpp`, and specified in `docs/maintainer/tensor-formats.md`.
- **Represented value:** `Q = Z·κ` with exact integers `Z`. `W = diag(suh) · H_k · Q · H_n · diag(svh)`, where `H` is the Sylvester Hadamard scaled by `1/√128`.
- `κ` is folded into `svh` at quantization, so the stored reconstruction is `W = diag(suh) · H_k · Z · H_n · diag(svh')`. Kernels therefore only ever see exact integers `|Z| ≤ 510`, which are exact in FP16.
  - exllamav3 instead rounds each codebook value once in fp16. Since we produce our own quants, our definition is the ground truth, the oracle is exact, and there is no rounding to reproduce.
- **Scales.** `suh` and `svh'` are stored as FP32 unless qualification shows fp16 is lossless for the refit values. They are a few KB per tensor, so fidelity wins over bytes.

### 3.2 Layout `trellis_t16_v1`

- **Planes:** trellis words, `suh`, `svh'`; object metadata stores `bitrate_half_bits` (2..16).
  Encoded size and plane offsets are documented in `docs/maintainer/storage-layouts.md`.
- **Tile order:** output-major `[n/16][k/16][words]`; each output tile's K stream is contiguous.
- **Bit order:** little-endian within bytes, aligned 32-bit windows, low state bit is oldest; half-rate
  codes use `floor(K)` steps on even states and `ceil(K)` on odd states. This supports two-word
  extraction with one funnel shift and no byte-order permutation.
- **M1 layout microbenchmark:** `bench/exl3_layout_bench.cu`, built and run with
  `nvcc -O3 -std=c++23 -arch=sm_120a bench/exl3_layout_bench.cu -o build/exl3_layout_bench && build/exl3_layout_bench`.
  On the RTX 5090/CUDA 13.3, output-major and K-major tile scans differed by at most about 1.3% over
  Qwen3.8 MLP `[17408,5120]` (3.5/4.0 bpw), attention Q+gate `[12288,5120]`, and attention output
  `[5120,6144]`; no order won consistently. LSB-first and MSB-first window decode differed by at most
  about 1.2% on half/integer-rate tile samples, also without a stable winner. The selected output-major
  and LSB-first order is the tie-breaker matching output-tile GEMV traversal and native little-endian
  words, not a claimed end-to-end speedup. Revisit only if production-kernel profiling shows a real
  difference.
- Slices and row views are tile-aligned (16) and Hadamard-aligned (128).
- **Fused parents** (q+gate+k+v, gate+up) follow the existing grouping rules.
  - Our quantizer gives all members of a shared-input group **one shared `suh`**. Members share the Hessian and the random signs; the input-channel scales are computed jointly over the group's concatenated columns.
  - Then one input transform serves the whole parent, which exllamav3 cannot do because it scales members separately. Quality impact is measured; per-member `suh` is the fallback.

## 4. Native quantizer

### 4.1 Pipeline and ownership

```
1. BF16 artifact     `tools/convert` builds a BF16 `.ninfer` from the source checkpoint. It is the
                      quantizer's single model input; M0 answered QBench's BF16 reference from the
                      source directly, but M3 needs the artifact for config, bindings and weights.
2. Calibration trace  `tools/exl3/sample_traces.py` starts `ninfer-serve` on the existing Q4 artifact,
                      samples chat continuations from disjoint corpus shards, and writes qbench JSON,
                      packed token rows, and a text stream for evaluation.
3. Quantize           `ninfer-quantize` (new offline C++23/CUDA app) maps the BF16 artifact's host
                      objects, runs calibration layer by layer, quantizes every eligible linear, and
                      writes each tensor's `.trellis/.su/.sv` source
4. Final artifact     tools/convert recipe `qwen3_8_27b_exl3` assigns format exl3_mul1 via
                      import_encoded from that source; everything else as in the BF16/NVFP4 recipes
```

- Artifact writing stays in one place (the converter). The maths lives in one native tool.
- An official recipe wraps steps 3–4, so the user-facing flow matches today's: one command per artifact after the trace exists.

### 4.2 Calibration Program (new Engine capability)

- A dedicated offline **calibration Program** owns:
  - a hidden-state buffer for N calibration rows × T tokens (exllamav3's default is 250 × 2048);
  - per-group `H` accumulators;
  - the layer-by-layer schedule.
- **It reuses the model's own layer execution** (the `qwen3_5` text layer code: GDN, gated attention, MLP), so calibration activations are exactly what inference computes.
  - Linear call sites expose an **observer** that accumulates `XᵀX` for the group's input, in FP32 with our own GEMM (FP32 accumulation, or 3×TF32 where qualified).
- **Sequential:** run layer ℓ on the BF16 weights while observing → quantize ℓ's linears → re-run ℓ
  with the FP32 reconstruction `wq` (M3 decision 3) to produce layer ℓ+1's input.
  - The re-run uses the existing BF16 linears, so the first artifact does not wait for the M4 EXL3
    kernels; M4 replaces it and re-quantizes.
- **MTP layer:** calibrated from the final hidden states plus next-token embeddings. This is an improvement on exllamav3, which quantizes MTP uncalibrated.
- `engine-architecture.md` gets a section on this. It is an offline Program kind and never part of serving.

### 4.3 Per-tensor GPU maths (all hand-written, each with its own oracle)

| Stage | Implementation | Oracle |
|---|---|---|
| 128-point FWHT (weights, `H`, activations) | in-register/shared butterfly, FP32 | explicit Hadamard matmul, FP64 |
| `XᵀX` accumulation | tiled FP32 GEMM | FP64 naive |
| Damping, signs, `H` rotation | elementwise + FWHT on both sides | FP64 |
| Block LDL (16) via Cholesky | right-looking blocked Cholesky, FP32 panels with FP64 diagonal blocks; failure → progressive damping as exllamav3 does | FP64 LAPACK-style reference on host (our own code) |
| Regularization + global-scale search | coarse grid on a tile subsample → fine grid → parabolic refine, drift factors per K | host reference on the same sample |
| Viterbi tile encoder | one CTA per tile ring; 2¹⁶ state costs in shared memory (FP16/FP32 by K), one backpointer byte per state and step; two-pass tail-biting; integer K and half-integer patterns | exhaustive/brute-force search on small synthetic tries + host FP64 Viterbi on real tiles (bit-exact path and cost) |
| LDLQ | block loop from the end, error feedback `rows += Lᵀ·err` via our GEMM, a 128-row buffer as exllamav3 uses | host FP64 LDLQ on small matrices |
| Pack | bitstream writer in our chosen layout | exact unpack round-trip |
| Scale refit | closed-form output scales; input scales via a k×k solve (Cholesky, reusing our kernel); 2 rounds | FP64 host |

- **Two-sided YAQA LDLQ** (`H_out`) is a later option.
  - exllamav3 has it only as an unexposed library hook.
  - It needs an output-side Hessian (a gradient or Fisher estimate), so it waits until the basic pipeline is measured.
- **Throughput.** The Viterbi encoder dominates. The target is exllamav3's single-GPU conversion time for 27B, with the encoder tuned for sm_120 (exllamav3 notes that its sm_120 ptxas unrolling needs care).

### 4.4 Bit allocation

1. **Global budget** (`--bits`, integer or half-integer) plus **`-hq`** promotion (attention and small tensors +1). This is the first artifact.
2. **Recipe optimization**, the latest EXL3 method:
   - per-tensor sensitivity `S_t` from LDLQ-shaped noise injected into the unquantized model;
   - per-tensor error amplitude `rfn_t(K)`;
   - an exact greedy allocation minimizing `Σ S_t · rfn_t(K)^α` under the bit budget.
   - All of this is run through NInfer's own forward and KLD measurement (`ninfer-perplexity`), with the self-sampled trace. The result is a recipe the converter reads.

## 5. Inference kernels (sm_120a)

Every route computes `y = svh' ∘ H_n( H_k(x ∘ suh) · Z )`, plus the consumer epilogue.

1. **Input transform.** 128-point FWHT per row with `suh` into an Op workspace (declared through the resource query; no hidden allocation).
   - FP16 A operand with an exact power-of-two row prescale, folded back in the epilogue, so outliers cannot overflow.
   - Where the producer is an RMSNorm with a single consumer group, fuse the transform into its epilogue later.
2. **Decode, T ≤ 8** (bandwidth bound):
   - funnel-shift window extraction;
   - `s·0x83DCD12D`;
   - `dp4a(·, 0x01010101)` → `Z`;
   - FP32 FMA against FP32 `xh`.
   - One tile's stream is read with 16-byte `cp.async`/`ld.global.nc`. Target: at or above q4 decode throughput per stored byte.
3. **T = 4–64** (MTP verification, small prefill): sliced-K MMA with the decode written straight into B fragments, since the tile element order is the fragment order.
4. **Prefill:** FP16 `mma.m16n8k16`, a multi-stage `cp.async` pipeline, decode fused into the B loads, and a CTA N tile that is a multiple of 128 so the **output Hadamard runs in the epilogue**. Then × `svh'` and the consumer epilogue.
5. **Consumers for 27B, in order:** `linear`, `linear_swiglu` (gate+up parent), `linear_add` (o, down, GDN out), `attn_input_proj` (q+gate+k+v parent), `gdn_input_proj` (qkv+z with conv epilogue), the LM head (its own dispatch, very wide N), and MTP.
6. Each route is tuned per shape with the `linear-tuning.md` method, and every route is qualified against the FP64 oracle at the real 27B shapes.

## 6. Qualification and acceptance

- **Codec:** exact round-trip tests (all K, half-integer patterns, tail wrap, tile order, parent slices).
- **Ops:** naive FP64 oracle from stored planes, per `op-development.md`, for every route and epilogue.
- **Quantizer stages:** the oracles in 4.3. An end-to-end single-tensor test compares the proxy error `tr(EHEᵀ)/tr(WHWᵀ)` with an FP64 host pipeline on a real 27B tensor.
- **Model:**
  - KLD and PPL (`ninfer-perplexity`) against the BF16 artifact, on our corpus and on the disjoint self-sampled eval trace.
  - Compared with our `q4_g64_fp16` and `nvfp4` 27B artifacts at their bpw.
  - Quality reference: exllamav3's own quant of the same source at the same bpw, run once in a throwaway tooling venv and not a product dependency. **Target: our KLD ≤ exllamav3's at equal bpw.**
- **Speed:** end-to-end decode and prefill tok/s (plain and MTP K=3/5) against the q4, NVFP4 and FP8 artifacts, same hardware and flags.

## 7. Milestones

| # | Status | Deliverable | Done when |
|---|---|---|---|
| M0 | Complete (calibration trace sampling) | Baselines | Existing Q4/NVFP4 NInfer PPL baselines recorded; BF16 source and exllamav3 3.0/4.0-bpw references scored on the self-sampled qbench trace (isolated tooling venv, streaming HF reference); self-sampled calibration and eval traces generated with `ninfer-serve`. No separate BF16 `.ninfer` artifact is required. |
| M1 | Complete | Format, layout, codec | `exl3_mul1` + `trellis_t16_v1` registered (Python + C++), docs written, tile and bit order chosen by microbenchmark, exact codec tests pass |
| M2 | Complete | Quantizer maths | FWHT, `XᵀX`, blocked Cholesky/LDL, Viterbi (integer + half-integer K), LDLQ, pack, refit, all passing their oracles; single-tensor proxy error matches the FP64 host pipeline |
| M3 | Complete | Calibration Program + first artifact | `ninfer-quantize` produces a 4.0 bpw `-hq` EXL3 artifact (shared-input parents, one shared input-scale vector) that loads and serves; full-corpus PPL 4.2939 beats Q4 4.3439 and NVFP4 4.3149 over 261,167 tokens; exllamav3 KLD comparison is M5 |
| M4 | Complete | Fast inference kernels | Tensor-core `m16n8k16` contraction for prefill (per-warp trellis→B-fragment decode, double-buffered `cp.async` BF16 A stage, sub-tiled epilogue) and the decode GEMV (split-K FFMA at T = 1, small-m tensor-core verify at T = 2..8), all consumers, MTP (quantized, calibrated) and Vision are in. On one RTX 5090 at a 2,805-token prompt: prefill **2.29-2.47k tok/s** (TTFT 1.1-1.2 s) and decode **75.6-77.7 tok/s** at 4.0 bpw, against Q4's 2.86k and 78.4 on the same prompt -- 1.17x and 1.03x -- on an artifact 1.6 GiB smaller; the 3.5 bpw artifact prefills at 2.11k and decodes at 64.7, its heavier odd-rate decode being the price of the smaller file. The contraction issues exactly the required MMA count (89,128,960 at the MLP-up shape, identical to Q4's); what remains above Q4 is the funnel, bit-field extracts and `IMAD`/`DP4A` per decoded window that a 4-bit trellis costs and a nibble plane does not. The consumers oracle takes T = 32 and T = 96 so `exl3_mma` is exercised through the consumer row mapping at both tile sizes, and its tolerance is expressed in BF16 ULP of the largest magnitude: the split-K GEMV reassociates its atomic K reduction per launch, so two runs of one parent differ by a few ULP and a 1e-3 relative gate sat below one ULP of a near-max element. |
| M5 | Complete | Recipe optimization | Uniform `-hq` artifacts at 3.5 and 4.0 bpw -- 13.91 / 15.35 GiB, PPL 4.31009 / 4.29390. On the plan's KLD-vs-BF16 metric (2,940 positions at context 4096 / stride 2048, ordering reproduced in both halves of the reference) the artifacts rank 4.0 bpw 0.0332 < Q4 0.0429 < NVFP4 0.0510 < 3.5 bpw 0.0624: **4.0 bpw beats Q4 and NVFP4 on both metrics and is smaller than both**, while 3.5 bpw's PPL advantage over them does not survive the divergence. The 3.0 bpw tier was built, measured worst on both metrics, and removed. The sensitivity-based layer allocation was measured and rejected; the odd half-rate decode is fast. |
| M6 | Pending | Later | Two-sided YAQA LDLQ; Flash-Next (GDN + MoE experts + expert pager); int8-activation route behind a permission. Vision is in the recipe but stays groupwise: its MLP intermediate is not 128-aligned. |
| M7 | Pending | Reference KLD | **M7a**: a logits-export route plus `ninfer-perplexity --reference FILE` and a KL accumulator, with the reference produced externally -- this lands M5's KLD evidence. **M7b**: `ninfer-reference`, an offline app that streams the BF16 `.ninfer` and writes the exact per-position distribution file, removing the external dependency. Done when the KLD agrees with an FP64 oracle and every M5 point carries a KLD. Design and gaps in section 9. |

**Order:** M0 baselines should precede M2/M3 quality work. M1 is independent of the reference
quantization and self-trace work and can complete while M0 remains open. M0 → M1 → M2 → M3 gives the
first real 27B EXL3 artifact with measured quality. M4 makes it fast; M5 makes it best-in-class per
bit. M7 supplies M5's KLD evidence and is independent of M4.

## 8. Risks

- **Decode ALU budget at T = 1.** About 6 integer ops per weight; 27B at 4 bpw is about 13.5 GB, a ~7.5 ms/token bandwidth floor. M4 must show the decode hidden under memory time. If it isn't, the fallbacks are:
  - more weights per decoded 32-bit window (aligned 2/4-bit fast paths as exllamav3 has);
  - the int8-activation route behind a permission.
- **Shared-`suh` fused parents** could cost quality. Measured in M3; per-member `suh` is the fallback.
- **Calibration Program scope.** It reuses layer execution but adds an observer and a layer-by-layer schedule. It must not leak into serving Programs.
- **Host-side FP64 oracles** are slow on 17408² Hessians, so they run on small and synthetic cases plus a few sampled real tensors.
- **FP16 activation range.** Handled by the exact power-of-two prescale; verified on real activations in M3.
- **27B has one MTP layer:** its quality at its own bitrate affects MTP acceptance, so acceptance rate is reported alongside KLD.

## 9. Reference KLD through a streamed BF16 forward (sketch, 2026-09-28)

M5's named metric is KLD against BF16, and PPL is a poor substitute: on wikitext-2 the exllamav3 3.0 and 4.0
bpw references span only 0.018 PPL, and a cross-engine PPL needs a matching window protocol that NInfer
forbids (`stride < context`) while exllamav3 fixes its stride at 2048. One shared BF16 reference removes both
problems: every artifact is compared to the same distribution at the same scored positions, in either engine.

### 9.1 NInfer can produce that reference itself

The BF16 weights already exist as a streamable `.ninfer`: `models/qwen3.8/bf16.ninfer` (29.8 GiB) plus
`bf16.ninfer.part-0001` (22.0 GiB), about 52 GiB, against 247 GiB of host RAM and a 30 GiB device. The
precedent is in-tree: `ninfer-quantize` is a standalone offline app that walks that artifact one layer at a
time and owns its device allocations. A reference forward is the same kind of tool, so the capability needs no
other engine and does not touch the serving stance.

### 9.2 Producer and consumer

1. `ninfer-reference` (new offline app): stream the BF16 artifact, score a text under the window protocol
   `ninfer-perplexity` uses, and write a compact reference file.
2. `ninfer-perplexity --reference FILE`: score the artifact at the same positions and report KLD alongside the
   existing per-domain PPL.

Separating them means one reference is computed once and reused by every artifact and every bitrate.

### 9.3 What the reference file holds

The BF16 next-token distribution at each scored position. Full logits are 248,320 x N x 2 B, so a
2,048-position reference is about 1 GiB in BF16 -- small enough to keep exact rather than approximate. Start
exact (BF16 logits `[positions, vocab]`, the scored token ids, and the text/protocol tags so the consumer can
check it scores the same thing). A top-k + logsumexp summary is a size option for later, not a correctness one.

### 9.4 Execution: a dedicated offline executor for the producer

The open question is settled against reuse. The composed Program cannot run this: `construct_model` marks every
weight `Residency::Device` and materializes one device arena of the whole payload (about 52 GiB on a 30 GiB
device), planning already throws for the missing BF16 geometries, and `ProgramImpl` holds an immutable
`const Parameters&` of baked device pointers with no weight-provider indirection. A resident-set contract would
first have to be threaded through `TextContext`/`run_layers` and every Op -- a serving hot-path and ABI change
far larger than an offline tool.

Reuse happens one level lower, where the composition is already weight-agnostic:
`execution::prefill_text_chunk` and `TextContext::run_layers` take weights, activations and state, so the
reference app supplies one layer's weights from a double-buffered staging arena and owns its own KV, GDN state
and workspace. `execution::Qwen4ExpertPager` is the in-tree precedent for on-demand device staging, though it
is MoE-specific and pinned-host based. Weights stay memory-mapped or pread from the artifact in host RAM; the
small pieces (embedding, final norm, head, state and KV) stay resident, and the 64 blocks stream with a
prefetch: stage `i+1` while computing `i`, release `i`. A full 52 GiB pass is a couple of seconds of PCIe
traffic, so a few hundred windows is minutes.

The consumer does reuse the composed Program: the CausalScoring route already computes full vocab logits per
position and fits the quantized artifact, so only a logits-export route is added.

### 9.5 Verification

- **The streamed logits are the claim.** Check one window's BF16 logits against a trusted BF16 run on the same
  text (exllamav3 `-hf`, or a Transformers backend) within BF16 tolerance. One-time evidence, not a runtime
  dependency.
- **The KLD agrees with a reference implementation**: compare against exllamav3's
  `util/measures.py::compute_kl_div`, or an FP64 oracle, on the same two distributions.
- **The scored positions match** `ninfer-perplexity`'s window protocol, on a short text whose windows can be
  enumerated by hand.

### 9.6 Scope and non-goals

Offline only. The serving stance is unchanged -- one resident model, one to eight requests, no offload; this is
not a serving capability and adds none. It is the KLD producer M5 asks for and, once built, a standing quality
tool; the capability description moves to a maintainer doc when it lands.

### 9.7 Gaps, and why the work is staged

Reading the tree corrected two premises of the first sketch. There is **no BF16 forward anywhere**: the
quantizer's calibration pass runs a servable *quantized* activation model, and the BF16 `linear` registry
covers only `[14336,5120]`, `[5120,6144]` and `[256,5120]`, so MLP gate/up `[34816,5120]`, MLP down
`[5120,17408]`, GDN input `[16384,5120]` and the head `[248320,5120]` have no BF16 route. And there is no
public logits API: `Engine::score_tokens` returns target logprobs only, while the full logits exist transiently
inside `ProgramImpl::causal_score`.

The gaps are (1) a logits-export route through `Program::causal_score` -> `CausalScoreCore` -> `Engine`; (2)
BF16 coverage for the missing Text geometries, or a reference-only generic BF16 GEMV/GEMM that leaves the
serving registries untouched; (3) the streaming schedule plus the app's own state and workspace; (4) the
reference-file format and the KL accumulator.

Staged so M5's evidence lands first:

- **M7a -- consumer.** The logits route, `ninfer-perplexity --reference FILE`, and the KL accumulator and
  report fields, with the reference produced externally for now. Small, and it delivers the KLD comparison M5
  needs.
- **M7b -- producer.** `ninfer-reference`, the offline streaming BF16 executor above, which replaces the
  external dependency. This is the large piece, and it is the milestone's own goal rather than a prerequisite
  for the evidence.

## Progress log

- 2026-09-27: research done; plan revised with the decisions above.
- 2026-09-27: M0 is in progress. On RTX 5090 (32,607 MiB), CUDA 13.3, the existing Q4
  (`models/qwen3_8_27b.ninfer`) and NVFP4 (`models/qwen3_8_27b_nvfp4.ninfer`) artifacts were
  scored with `build/apps/ninfer-perplexity --corpus eval/corpora/perplexity-1m/manifest.json
  --quick --context 4096 --stride 2048 --kv-dtype fp8`. Both runs scored the same 261,167 tokens
  over four streams: Q4 NLL/PPL 1.4687714 / 4.3438949; NVFP4 1.4620800 / 4.3149254. Reports are
  in ignored local `profiles/perplexity/exl3_m0_{q4,nvfp4}/report.json`.
  The user waived a separate BF16 `.ninfer` artifact. The 52 GB source cannot be resident in NInfer's
  32 GB no-offload Engine, so the BF16 reference will be attempted directly with QBench's streaming
  Transformers backend instead. Do not treat the Q4/NVFP4 results as substitutes for EXL3 reference
  KLD or self-sampled traces.
- 2026-09-27: M1 complete. Registered `exl3_mul1`, required `bitrate_half_bits` object metadata, and
  `trellis_t16_v1` geometry in Python/C++; added the exact C++ tile pack/unpack and mul1 codebook;
  documented numeric and storage contracts. `ninfer_exl3_trellis_test` passes for all 15 rates,
  circular states, all 65,536 codebook inputs, directory metadata, and Qwen3.8 MLP geometry;
  Python artifact tests pass. The layout microbenchmark found no stable winner, so the documented
  tie-breaks select output-major and LSB-first. Native model execution remains unimplemented.
- 2026-09-27: Per user direction, a BF16 baseline `.ninfer` artifact is not an M0 gate. Installed CPU PyTorch
  2.14.0, `safetensors` and `tokenizers` in the maintained Python 3.11 environment; created an isolated Python 3.11
  EXL3 reference environment at `/tmp/opencode/exllamav3-reference/.venv` with CUDA PyTorch 2.14.0
  (`cu132`). The pinned exllamav3 `fa714d0` source extension built successfully on first CLI import.
- 2026-09-27: Added `tools/exl3/sample_traces.py` to run the public `ninfer-serve` route and write
  qbench JSON, packed Safetensors rows, and text streams. It assigns fixed-corpus shards `00/01` to
  calibration and `02/03` to evaluation, using the selected artifact's own tokenizer and chat
  template, with a small tool-call slice backed by deterministic local tool fixtures. Prompt token
  counts are checked against server usage; returned assistant text is re-tokenized and token-count
  mismatches are recorded. Its split,
  packing, token-prefix, tool-fixture and artifact-resource tests pass; actual sampling awaits the
  active GPU reference quantization. Installed `transformers`, `datasets`, and QBench plotting
  dependencies in the isolated EXL3 environment. A 3.0-bpw reference conversion is running from the
  local checkpoint at `/mnt/storage/models/qwen3.8/full`.
- 2026-09-27: The exllamav3 3.0 bpw reference finished at 09:58, about 45 minutes including one resume:
  `/mnt/storage/models/qwen3.8/exl3-reference/3bpw`, 13.5 GB, v1.5.2, mul1, `head_bits` 6, `mtp_bits` 4,
  `out_scales` always, calibrated on exllamav3's built-in data (250 x 2048).
- 2026-09-27: The sampler's prompt-token cross-check failed on tool rows (1857 vs 1881 tokens). NInfer's template
  `tojson` uses Python's default `", "`/`": "` separators, and the server renders tools in its canonical form
  (`name`, `parameters`, `strict: false`, `description`); the sampler now matches both (`7bfe1012`). The 100-row
  evaluation trace (shards 02/03, `--max-new-tokens 256`, Q4 artifact) is in ignored
  `profiles/exl3/traces/evaluation.*`: about 1,473 prompt + 261 sampled tokens per row.
- 2026-09-27: M0 reference scoring with QBench (`profiles/exl3/qwen3_8_27b_reference.yaml`, results and plots
  beside it) on the evaluation trace, against the layer-streamed HF BF16 source:

  | Model | PPL | Mean KLD | Median KLD | p90 KLD |
  |---|---:|---:|---:|---:|
  | BF16 reference | 1.7406 | — | — | — |
  | BF16 noise floor | 1.7409 | 0.000833 | 0.000136 | 0.00215 |
  | exllamav3 3.0 bpw | 1.7832 | 0.034747 | 0.010227 | 0.094392 |

  PPL is low because the trace is the model's own sampled continuations. The 4.0 bpw reference conversion is
  running (`exl3-reference/4bpw`, `-hb 6`); add it to the QBench project when it finishes. The calibration trace
  (250 rows, shards 00/01) remains to be sampled.
- 2026-09-27: M2 started, in `ninfer_quantize` (`src/quantize/`, offline only, not linked into the Engine):
  - `tests/quantize/exl3_viterbi_reference`: FP64 host oracle with exact tail-biting search (one constrained
    Viterbi per ring overlap) and the two-pass method. The exact search matches brute-force enumeration of every
    ring bitstream on small trellises. Unit-Gaussian MSE is 0.067 / 0.035 / 0.016 at K = 2 / 2.5 / 3, 5-11% above
    the rate-distortion bound.
  - `src/quantize/exl3/trellis_encoder`: GPU two-pass encoder with FP32 path costs (exllamav3 uses FP16) indexed by
    carry node. Its rings are bit-identical to the FP64 reference at K = 1, 1.5, 2, 2.5, 3 and 4 (24/24 tiles each)
    and pack through `trellis_t16_v1`. Cost buffers and back-pointers live in global scratch; shared-memory costs for
    K >= 3 and packed back-pointers are pending throughput work.
  - `src/quantize/exl3/hadamard`: in-place 128-point FWHT along rows and columns, checked against the explicit FP64
    Sylvester product.
  - Next: `XᵀX` accumulation, blocked LDL, LDLQ, scale refit, and the single-tensor FP64 pipeline comparison.
- 2026-09-27: M2 complete. `ninfer_quantize` now holds every stage, each checked against its own oracle:
  - `hessian`: `XᵀX` from BF16 activations in FP32, exactly symmetric; FP64 Gram oracle.
  - `block_ldl`: blocked Cholesky (64-wide FP32 panels, FP64 diagonal blocks) plus 16x16 block normalization.
    Matches FP64 to 2.6e-6, reports non-positive pivots for progressive damping.
  - `ldlq`: strip-by-strip LDLQ with the GPU encoder; states identical to an FP64 host LDLQ built on the reference
    encoder. Error feedback lowers the proxy error from 0.067 to 0.038 at K = 2 on a correlated Hessian.
  - Tile element order: the `mma.m16n8k16` B-fragment order, now written out as a formula in
    `storage-layouts.md` 9.2. It matches exllamav3's `tensor_core_perm`.
  - `pipeline` (`quantize_tensor`): damping, random signs, rotated `H` and block LDL with retries; skew-gated
    output scales; Hadamard rotations; input scales; global-scale search (coarse grid on a third of a
    wrapped-diagonal plus extreme-RMS tile sample, fine grid, parabolic refine, LDLQ drift); LDLQ; refit of
    su/sv (closed-form output scales, CG input scales, 2 rounds). The codebook scale is folded into sv, so the
    stored form is exactly W = diag(su) H128 Z H128 diag(sv) with integer Z; the test decodes it independently
    in FP64.
  - Single-tensor parity with exllamav3 1.5.2 (`tools/exl3/compare_tensor.py`, real layer-0 Hessian from the
    trace, same W and H). The deviation from the plan: the stage oracles plus this parity replace one end-to-end
    FP64 host pipeline.

    | Tensor | K | NInfer proxy | exllamav3 proxy | Ratio |
    |---|---:|---:|---:|---:|
    | in_proj_z 5120→6144 | 2 / 2.5 / 3 / 3.5 / 4 | 0.004801 / 0.002350 / 0.001165 / 0.000585 / 0.000295 | 0.004802 / 0.002351 / 0.001167 / 0.000586 / 0.000295 | 0.9986–0.9997 |
    | gate_proj 5120→17408 | 3 | 0.004843 | 0.004845 | 0.9996 |

  - Encoder throughput (bit-identical to the FP64 reference after each change): per-rate compile-time
    specialization, shared-memory FP32 costs for steps of at least 3 bits, conflict-free work mapping, coalesced
    back pointers, one 1024-thread block per SM. gate_proj single-tensor time is 7.1 s at K = 3 (exllamav3
    6.1 s), 8.0 s at K = 4, and 14.7 s at K = 2.
  - Open performance items, none blocking M3:
    - K <= 2.5 (smallest step 2 bits) keeps its costs in global scratch.
    - Back pointers use a byte per node (could pack).
    - The LDLQ compensation and refit GEMMs are SIMT FP32.
    - The encoder is about 93% of single-tensor time, so M3's full-model quantization time follows it.
- 2026-09-27: M0 reference scoring finished. The exllamav3 4.0 bpw reference (`exl3-reference/4bpw`, `-hb 6`,
  otherwise as 3.0) was added to the QBench project; the cached BF16 logits were reused.

  | Model | PPL | Mean KLD | Median KLD | p90 KLD |
  |---|---:|---:|---:|---:|
  | BF16 reference | 1.7406 | — | — | — |
  | BF16 noise floor | 1.7409 | 0.000833 | 0.000136 | 0.00215 |
  | exllamav3 3.0 bpw | 1.7832 | 0.034747 | 0.010227 | 0.094392 |
  | exllamav3 4.0 bpw | 1.7546 | 0.009969 | 0.002794 | 0.026393 |

  - Output-scale mode: the references used always-on output scales. `TensorOptions::out_scales` now has
    `Always`, and single-tensor parity holds in that mode too (in_proj_z K = 3/4 0.9989/0.9993, in_proj_qkv K = 3
    0.9970).
  - Largest shape: a k = 17408 down_proj probe (synthetic Hessian, K = 4) takes 9.9 s and about 13 GB of device
    memory, with the CG input refit converged. `TensorReport::refit_input_skipped` now counts non-converged
    solves.
  - Sampler fixes (`5c3c58e9`, review item 11):
    - Tool rows sample with thinking off by default, so they reach real tool calls and follow-ups.
    - Follow-ups render without tool definitions when `tool_choice` is `"none"`, and with tool-call arguments as
      objects, as ninfer-serve does.
  - Calibration trace: 250 rows, shards 00/01, `--max-new-tokens 600`, sampling into ignored
    `profiles/exl3/traces/calibration.*`. Check its log for completion.
- 2026-09-27: The converter reads the native quantizer's output (`f116ed0e`).
  `tools/convert/sources/exl3.py` resolves an `exl3_mul1` matrix from `prefix.trellis` U8 tiles plus
  `prefix.su` F32[K] and `prefix.sv` F32[N]; `import_encoded` copies tiles and scales unchanged, and a shared
  parent's sources must agree on the bitrate. This repaired a missing source import that had left six convert
  tests failing, and `tests/convert/test_sources.py` now decodes a written source independently in FP64.
- 2026-09-27: M3 decisions recorded (`ae56f284`) and the BF16 input artifact built (`b8c442ff`). Added the
  `qwen3_8_27b_bf16` official recipe (every weight at BF16, documented in the conversion guide). Converted
  `/mnt/storage/models/qwen3.8/full` with `--recipe qwen3_8_27b_bf16 --components text,vision,mtp --device cpu`
  in 43.7 s to `/mnt/storage/models/qwen3.8/bf16.ninfer` (30 GB + 22 GB parts; 1,052 tensors — 956 bf16, 96 fp32,
  all `contiguous_le_v1`; 1,420 bindings). This is the quantizer's input artifact, not a scored baseline.
- 2026-09-27: Next producer step, not started: `ninfer-quantize` (offline app). It reads the BF16 artifact,
  computes each eligible linear's Hessian from the calibration trace, calls `quantize_tensor`, and writes a
  source safetensors keyed by **logical parameter name** (`text/layers/<l>/mlp/gate.trellis/.su/.sv`). The EXL3
  recipe then reads that store, so the app never needs the checkpoint's tensor names. Calibration re-runs each
  layer with the BF16 reconstruction `wq`.

- 2026-09-27: `ninfer-quantize` app v1 (host surface + quantization driver). It reads the BF16 artifact,
  enumerates the 522 eligible Text/MTP BF16 projections (`--list`, no device), reads each as FP32, calls
  `quantize_tensor` with a per-parameter Hessian from `--hessians`, packs the returned states through
  `trellis_t16_v1`, and writes one `exl3.safetensors` (`<name>.trellis/.su/.sv`) plus `report.json`.
  `ninfer_exl3_quantize_source_test` (options, Hessian IO, safetensors writer, tile packing, transpose) and
  `ninfer_exl3_quantize_interop_test` (Python writer -> C++ enumeration) pass CPU-only. The calibration stage
  that produces the Hessians is next; device runs await a free GPU.

- 2026-09-27: Added the `qwen3_8_27b_exl3` official recipe. It assigns `exl3_mul1` (`import_encoded`) to
  every Text/MTP projection present in the quantizer's `exl3.safetensors`, and leaves the embedding, GDN
  a/b, norms and Vision at BF16; EXL3 parents stay separate until shared-`suh` fusion. Documented the
  format and recipe in the conversion guide.

- 2026-09-27: End-to-end CPU test of the producer format path (`tests/convert/test_exl3_pipeline.py`):
  a `ninfer-quantize` source store flows through `qwen3_8_27b_exl3` and `import_encoded` into an
  `exl3_mul1` artifact, and the artifact's own planes decode to the source in FP64. Fixed
  `import_encoded`'s format probe to read a whole 16-row tile for EXL3.

- 2026-09-27: Added the C++ packed-trace reader (`quantize/trace_reader.cpp`) that the calibration
  stage will consume: `input_ids` I64 [rows, tokens] and `lengths` I64 [rows], validated and converted
  to I32. `ninfer_exl3_trace_reader_test` covers the round trip and its rejections, CPU-only.

- 2026-09-27: Verified the quantizer on the GPU. `ninfer-quantize bf16.ninfer --hessians <identity H> --limit 1`
  quantized `mtp/input_projection` [5120,10240] in 2.7 s and wrote a 25 MiB `exl3.safetensors`. The Python
  converter's `exl3_matrix_source` reads that store back with the expected plane shapes (`codes [1,640,128]` U8,
  `su [10240]`, `sv [16]`) and rate 8. All 11 `ctest -R exl3` tests pass, including the M2 GPU kernels.

- 2026-09-27: M3 quantizer calibration and native decode are in (`ee6eb28c`).
  - Calibration: the model exposes a projection observer (`execution/calibration.h`), so
    `ninfer-quantize --trace --activation-model` scores the packed trace through a servable
    activation artifact, accumulates each projection's `E[XᵀX]` (transposed Hessian kernel for the
    model's `[K,T]` activation) in device-budgeted layer groups, and writes per-parameter
    `.h.f32`. Layer groups are separate passes over unchanged weights, so the Hessians are identical
    to one pass while peak device memory stays bounded.
  - Decode: `native_weight` bridges EXL3 and `src/ops/linear/exl3` decodes `trellis_t16_v1` with the
    two 128-point Hadamards. `ninfer_linear_exl3_a16_test` passes the FP64 plane oracle at rates
    3/4/8 and shapes up to `[384,256] x T=64`.
  - First 4.0 bpw Text artifact produced by `qwen3_8_27b_exl3` from the quantizer store.
- 2026-09-27: Remaining for full serving (the plan's M4 consumer set): `linear_add`,
  `linear_swiglu`, `attn_input_proj` and `gdn_input_proj` need EXL3 paths; the attention/GDN fusion
  gates in `ops/weight_input.cpp` must admit EXL3; and each shared-input group (attention
  q/gate/k/v, GDN q/k/v/z, MLP gate+up) must be quantized as one parent with a shared `suh`, which
  the app does not yet do.

- 2026-09-27: First native EXL3 artifact produced. `ninfer-quantize` calibrated from 80 trace rows and
  quantized all 512 Text projections (~50 min): proxies around 0.003–0.004 at 4.0 bpw, attention and the
  small GDN controls promoted to 5.0 bpw by `--hq`, the output head at 6.0 bpw. `qwen3_8_27b_exl3` turned the
  12.4 GB store into `/mnt/storage/models/qwen3.8/exl3-4bpw.ninfer` (18 GB, 1,266 objects: 512 `exl3_mul1`,
  652 `bf16` Vision/MTP/embedding, 96 `fp32`). The loader reads and validates every EXL3 object
  (`loading weights | 16.3 GiB | 2.7s`). All 12 `ctest -R exl3` tests pass.
- 2026-09-27: Blocking finding for "serves". The Engine fails binding this artifact with
  `text/layers/0/mlp/gate: native input requires one contiguous parent region`. The model composes each
  shared-input group as one fused parent (`ops::prepare_linear_swiglu_weight`, `prepare_attn_input_proj_weights`,
  `prepare_gdn_input_proj_weights`), while the app quantized each projection separately. Making the artifact
  servable requires, in order:
  1. quantize each shared-input parent (attention q/gate/k/v, GDN q/k/v/z, MLP gate+up) as one EXL3 matrix
     with one shared `suh`, and have `qwen3_8_27b_exl3` assign the group through one parent source;
  2. admit EXL3 in the fusion gates in `src/ops/weight_input.cpp`;
  3. add EXL3 paths to `linear_add`, `linear_swiglu`, `attn_input_proj` and `gdn_input_proj` (the plan's M4
     consumer set), each composing the existing decode with its epilogue.
- 2026-09-27: Resolved all three steps. `ninfer-quantize` now enumerates parent objects and quantizes each as one
  matrix with one shared input-scale vector; `qwen3_8_27b_exl3` binds members to row ranges; the fusion gates and
  the four consumers (plus the GDN verify snapshot/record) gained EXL3 paths; the vocabulary head is assigned too.
  The first attempt produced a PPL of 1.8e6 because the decode wrote and read the parent **row-major** while NInfer
  tensors are column-major: every projection was transposed. The decode now indexes column-major and the consumers
  split a parent with a pitched 2D copy. A sampled FP64 oracle over the model's real parent shapes (including the
  [248320, 5120] head) and a new fused-consumer test caught and now guard the layout.
- 2026-09-27: M3 artifact and evidence. `ninfer-quantize` (80 calibration rows, `--bits 8 --head-bits 12 --hq`)
  wrote 257 EXL3 tensors (12.4 GB); `qwen3_8_27b_exl3` produced `exl3-text.ninfer` (14.9 GiB). The head needed the
  quantizer's device memory bound (`fix(exl3): bound quantizer device memory...`) to fit a 32 GiB GPU; it quantizes
  in 51 s. Evidence on `perplexity-1m/00.txt` at context/stride 512/256: Q4 PPL 2.1358, EXL3 first-window PPL
  2.3056; `ninfer-serve` returns coherent generation. Decode is ~1.7 tok/s (SIMT), which is what M4's kernels fix.
  The full-corpus KLD and MTP quantization remain.
- 2026-09-27: M4 first step. The SIMT contraction re-decoded the trellis per `(n,t)` pair and extracted each
  16-bit window bit by bit. It now reads the window as two 32-bit words and a funnel shift (even rates), and one
  block covers 128 output rows and up to 16 columns so each window is decoded once and reused across them. For
  `T = 1` the contraction splits K across eight 128-thread slice groups (with the 128-point output Hadamard run
  per slice) to lift occupancy. Measured on `perplexity-1m/00.txt` prefix (12,126 tokens, context/stride 512/256):
  scoring 5.4 -> 60 tok/s, and EXL3 PPL 2.0307 against Q4's 2.0399 at the same prefix; `ninfer-serve` decode
  1.75 -> 13.2 tok/s with coherent output. Tensor-core MMA and the full-corpus KLD remain.
- 2026-09-27: M4 tensor cores. The stored trellis tile is already the `mma.m16n8k16` B-fragment order, so a
  warp lane decodes its eight windows straight into two B fragments with no shuffle; a block covers one 128-row
  Hadamard block and 64 tokens, warp `w` owning n = 16w..16w+15, and the epilogue applies the 128-point output
  Hadamard (two columns at a time across 256 threads) and `sv`. The FP64 oracle now exercises the MMA path at
  `[14336, 5120] x T=32` and `[256, 128] x T=64`. Measured on the same 12,126-token prefix: scoring
  60 -> 318 tok/s and PPL 2.0291 (Q4 2.0399). Decode (`T=1`, verify) still uses the SIMT decode path; a
  dedicated GEMV is the next step.
- 2026-09-27: M4 window plan. The inner loop still did two integer modulos per window (state -> word and
  shift). The plan is now computed once per thread and packed into 16 bits, so only two loads and a funnel
  shift remain, and the decode kernel uses four accumulators to break the FMA chain. The decode kernel keeps
  four k-slices (eight exceeded the 1024-thread register budget). Measured on the same prefix: scoring
  318 -> 406 tok/s, decode 13.2 -> 15.5 tok/s, PPL unchanged at 2.0291. A dedicated decode GEMV remains.
- 2026-09-27: Full-corpus quality. `ninfer-perplexity --corpus perplexity-1m/manifest.json --quick
  --context 4096 --stride 2048 --kv-dtype fp8` over the same 261,167 tokens as the M0 baselines gives EXL3
  4.0 bpw overall PPL **4.2939** (chinese 4.974, english_long_form 6.846, english_reference 6.007, code 1.649)
  against Q4 4.3439 and NVFP4 4.3149, at 380.7 tok/s. The 4.0 bpw artifact is therefore the best PPL of the
  three native routes on this corpus, and the run qualifies the tensor-core path at `T = 4096`.
- 2026-09-27: M4 decode GEMV. Graphsignal (`--cuda-graph-trace node`) showed the decode contraction at 77% of a
  decode step. The GEMV now splits K across the grid into a small FP32 partial buffer (removing the block
  register ceiling that capped it at four k-slices), and each lane decodes the eight m16n8k16 B-fragment
  windows `t = 8L..8L+7`, which at 4 bits is one coalesced 32-bit word per lane. A dispatch trace showed the
  serving decode is a `T = 2/3` verify pass plus a `T = 1` head, so those verify columns now use the MMA and
  the T-tiled SIMT kernel is gone. Decode 15.2 -> 41.0 tok/s, prefill 61 -> 71 tok/s; the EXL3 workspace
  capacities include the GEMV partial at `T = 1`. The remaining gap to the ~120 tok/s bandwidth floor is the
  head GEMV (n = 248320) and per-round overhead.
- 2026-09-27: M4 GEMV split tuning. Graphsignal telemetry (`ninfer_decode_device_wait_microseconds_per_round`
  = 25.2 ms, `ninfer_decode_batch_average_size` = 1) plus a per-call sample count (252 GEMV calls per token,
  87 us each) showed the head is only about 3% of the decode; the 251 layer projections are the cost. The
  k-split is now adaptive, `min(32, max(1, 8192 / (N/128)))`, and each of the eight windows accumulates into
  its own register. Decode 38.2 -> 41.2 tok/s on a 162-token paragraph; a smaller split target (25.9) and a
  larger one (39.4) are both worse. The whole-model GEMV is still about 2.5x the DRAM floor.
- 2026-09-27: M4 4-bit GEMV extraction. exllamav3's GEMV (`dq8_regs_4bits`) resolves a tile's two words
  in-warp and takes the eight windows with one funnel plus five BFE16. The 4-bit rates are about 90% of the
  weights (MLP and GDN at 8 half bits), so the decode GEMV now does the same instead of unpacking a plan and
  funnelling per window. Decode 41.2 -> 62.1 tok/s on a 162-token paragraph. The 5-bit (attention) and 6-bit
  (head) rates still use the generic path.
- 2026-09-27: M4 5/6-bit extraction and the GEMV floor. The attention projections (5 bits after `-hq`) and
  the head (6 bits) now resolve each four-window group from a 64-bit window over the lane's word and its
  successor (exllamav3's `dq4`, widened because a 5- or 6-bit group can exceed 32 bits), with lane-constant
  word indices and shifts. Decode 62.1 -> 68.6 tok/s; graphsignal then reads the GEMV at **48.4 us/call**
  against a 49 us DRAM floor for the largest layer projection, so the contraction is now bandwidth-bound and
  the remaining decode time is the MMA verify/prefill pass (391 ms) and the small ops. The Linear
  qualification test covers rates 10 and 12 at T = 1.
- 2026-09-27: M4 small-m routing. graphsignal showed the tiled MMA at short T (a short prompt's prefill and
  the verify pass) at 763 us/call, grid.y = 1 having too few blocks to fill the GPU; exllamav3 routes m = 1..8
  to its GEMV and only m >= 16 to the tiled MMA. The dispatch now runs the GEMV once per column for T <= 8.
  Short-prompt TTFT 985 -> 344 ms, long-prompt prefill 634 -> 733 tok/s, decode unchanged at 68.2 tok/s. The
  workspace capacities include the GEMV partial for T <= 8 and the Linear test covers T = 4.
- 2026-09-27: M4 small-T MMA tile. The tiled MMA covered 64 columns per block, so any T below 64 left
  grid.y = 1 and occupancy-starved, a typical short prompt (T = 59) included. The kernel is now templated on
  its tile and T <= 64 takes a 16-column tile. T = 59 TTFT 344 -> 274 ms; long prefill and decode unchanged
  (726 and 68.0 tok/s). The Linear test covers T = 64 and 128.
- 2026-09-27: M4 MTP. The layer was never quantized: the quantizer skips a projection with no calibration
  Hessian and MTP had no site, so it stayed BF16 and BF16 has no MTP geometry ([5120,10240] stem, [34816,5120]
  MLP); the served artifact was Text-only. MTP now has observation sites (the stem, its attention, and the
  MLP through ffn's existing sites at the -1 sentinel), `site_for` maps the `mtp/` names, and an unobserved MTP
  projection falls back to the identity Hessian instead of being skipped (the scoring Program forbids a
  speculative backend, so the calibration cannot enable MTP itself). `Prepare::mtp` builds its dense `rows`
  split only for Q8, since `ops::linear_pair` implements Q8 alone and a row slice of an EXL3 parent is not a
  valid native Weight. The verify pass carries the drafted tokens in the m16n8k16 A fragment, as the
  reference's `exl3_gemv_kernel` does for 2 <= m <= 8; the FFMA multi-column alternative needed 128 registers
  at four columns and lost 2.4x. Measured: `exl3-mtp.ninfer` `--spec mtp --draft-tokens 3 --fixed-draft` decodes
  130 tok/s against 71 plain (1.8x, 53% acceptance), verify 53.6 us/call, plain decode unchanged.
- 2026-09-28: M4 MTP calibration and Vision. The MTP sites were observed but never fired: the scoring
  Program rejects a speculative backend by design. `calibrate()` now runs a second pass with a generation
  engine (CUDA graphs off, since the observer allocates and copies; the scoring engine is released first
  through an optional because both engines are ~16 GiB). The stem proxy error fell 0.0458 -> 0.00656 (7x)
  and the MTP attention input 0.00144 -> 0.00035 (4x); on predictable completions the acceptance rose
  accordingly (counting 93.0% -> 96.7%, 160 -> 186 tok/s; code 71.9% -> 79.3%, 135 -> 166 tok/s). The
  attention output observer now passes the projection's 2-D view (the head-major attention tensor failed
  the observer's rank check). Vision is added to the recipe: it cannot be EXL3 (its MLP intermediate 4304
  is not a multiple of the 128-point Hadamard block) and cannot stay BF16 (the BF16 registry carries only
  the Text/MTP geometries), so it takes the groupwise formats every other official recipe uses (Q6 patch,
  Q8 merger, Q4 qkv and fc1, Q5 else). `--components text,mtp,vision` builds a 15.35 GiB artifact that
  loads, answers an image prompt, and decodes 140 tok/s with MTP3 (vision) and 191 tok/s (text).
- 2026-09-28: Canonical artifact name and the fully calibrated MTP. The artifact is
  `models/qwen3_8_27b_exl3_4bpw.ninfer` (15.35 GiB), named for its body rate the way the other artifacts are
  (`qwen3_8_27b`, `qwen3_8_27b_nvfp4`); the superseded EXL3 artifacts are removed. With the complete MTP
  Hessian set (the attention output observer now passes the projection's 2-D view, and the generation pass runs
  four output tokens so `mtp_forward_tail` fires) every MTP projection is calibrated: proxy errors 0.0066
  (stem), 0.00035 (attention qkv), 0.000079 (attention output) and 0.00014-0.00016 (MLP), against
  0.0012-0.0045 uncalibrated.
- 2026-09-28: M5 first point, uniform 3.0 bpw. `--bits 6 --hq` over the same combined Hessians gives
  `models/qwen3_8_27b_exl3_3bpw.ninfer`, 12.47 GiB against 15.35 at 4.0 bpw, and full-corpus PPL **4.3840**
  against 4.2939 (chinese 5.129/4.974, english_long_form 7.018/6.846, english_reference 6.062/6.007, code
  1.680/1.649). So 19% smaller costs ~2% PPL, and uniform 3.0 bpw lands behind Q4 (4.3439) and NVFP4 (4.3149):
  the allocation is what M5 has to fix, because the per-tensor sensitivities are far from equal. The
  sensitivity measurement is being built -- `Exl3WeightProbe` injects seeded Gaussian noise into one EXL3
  weight's decoded values (committed `b4c3cbc4`), and preparation registers each projection as a probe target
  (committed `6c4becd1`).
- 2026-09-28: M5 sensitivity result and the layer allocation. The 16-row `ninfer-sensitivity` sweep (256
  targets, seeded Gaussian `rfn` 0.145 on the 4.0 bpw weights) measures each projection's end-to-end ΔNLL. The
  signal sits on the **layer** axis: the per-layer mean |ΔNLL| spans 84x, 9.3 (L63) to 784.2 (L8), and the 17
  quietest layers (45-51, 54-63) are all below 100. The projection **type** axis is nearly flat, 1.28x
  (gdn/query 230.6 to gdn/output 296.2), and the per-tensor signs flip between adjacent projections of one
  layer -- so per-tensor allocation would fit mostly noise. `tools/exl3/allocate.py` therefore pools each Text
  layer as one unit (`--group-by-layer`) and gives its projections a single rate, summing each member's KL at
  the common rate through that member's **own** quantizer anchor,
  `kld(k) = sum_m S_m (rfn0_m * 0.5 ** (k - k0_m)) ** 2` (`--hq` anchored attention and GDN a bit above the
  MLP, so one anchor per layer misprices the MLP). Two bugs surfaced and are fixed: the error curve exponent
  was inverted (`0.5 ** (k0 - k)` makes a lower rate *more* accurate) and the signed ΔNLL must enter as a
  magnitude.
  At the uniform 3.0 bpw body rate (6.1379 half bits, measured from the `--bits 6 --hq` report) the allocation
  is a 4-7 half bit schedule (mean 6.126, 3.06 bpw): layers 0-44 keep 3.0-3.5 bpw and the quiet tail 45-63
  drops to 2.0-2.5 (L56 2.0, L63 2.0); the head is unchanged at 12 half bits. `ninfer-quantize --rates` is
  re-running over the same `hess-all`; the conversion, artifact replacement and full-corpus PPL follow.
- 2026-09-28: M5 allocation trial -- negative, and an odd-rate slow path. The 3.06 bpw layer allocation
  converted to 12.44 GiB against the uniform 12.47, and scored **4.595078** on the same corpus -- worse than
  the uniform 3.0 bpw 4.383991 by 0.21 -- while scoring at 163 tok/s against 452. Two causes are visible.
  (1) The contraction resolves a window directly for **even** half bits and falls back to the bit-by-bit
  `tile_state` reference decode for odd ones. The allocation put 40 of 64 layers on 5 or 7 half bits; the
  fallback is about 4x slower, which is the whole 2.7x. (2) The demotion is real weight error -- the tail's
  quantizer proxy error rose ~1.99x per half bit below the anchor (L56 0.0136 -> 0.0539, L63 0.0079 -> 0.0504)
  while promotion only halved it, so `kld ~ rfn^2` over-credited trading the quiet tail for the loud middle.
  The scheme the allocation displaced also keeps a type prior the pooled measurement does not see: `--hq`
  promotes only `attention/*` (6.9% of the body numel) by two half bits, and per-layer pooling flattened it.
  `allocate.py` gained `--rate-step` (even-only rates stay on the fast path) and `--offset SUBSTRING:DELTA`
  (the type prior, inside the budget); the corrected 3.0 bpw point is re-running.
- 2026-09-28: M5 allocation concluded negative. The corrected variant -- even rates plus the restored
  attention offset, a pure layer redistribution at mean 6.110 half bits (12.42 GiB) -- scored **4.609115** on
  the same corpus: no better than the first attempt's 4.595078 and 0.225 worse than the uniform 3.0 bpw
  4.383991. Both redistributions lose by about the same amount, so pooled layer sensitivity does not identify
  a profitable trade at a fixed budget. The per-layer |ΔNLL| spread is real, but the model built on it orders
  the candidates **backwards** -- predicted KL 520.7 uniform / 356.5 odd / 461.5 even against measured PPL
  4.384 / 4.595 / 4.609. The likely cause is the injected regime: the sweep used `rfn` 0.145 (a 14.5% relative
  weight error) while the quantized tensors sit 1-5% from their anchors, so the tail's insensitivity at 14.5%
  does not transfer to the demoted rates. The even schedule did fix the throughput (401 tok/s against the
  odd run's 163), confirming the odd-half-bit fallback. M5 therefore keeps the uniform `-hq` recipe for its
  points and reports the PPL-vs-size frontier; the layer allocation is dropped, and a sweep re-measured at
  the operating error is the prerequisite for reviving it.
- 2026-09-28: M5 consults exllamav3, and ports its half-rate decode. Reading the reference settled three
  things. (1) EXL3 `K` is an integer 1..8 or a half-integer **1.5 / 2.5 / 3.5** (mul1 only;
  `quant/bits_k.cuh`), and `k2_from_K = 2*bits + half` is exactly NInfer's `bitrate_half_bits` -- so 3.5 bpw
  is a first-class rate and the M5 point is uniform 3.5, not a 6/8 mix. NInfer's 2..16 is a superset: it also
  admits 4.5/5.5/6.5/7.5 bpw, which `bits_from_K` rejects. (2) exllamav3's
  `conversion/allocation.py::create_q_strategy` floors every tensor at `floor(bpw)`, then spends the
  remainder promoting groups one step at a time ordered by group priority and then by distance to the nearer
  end of the forward pass -- "End layers contribute disproportionately to end-to-end error" -- with the
  sensitivity recipe only an optional path (`create_q_strategy_from_recipe`). Its prior is the opposite of
  our measured layer signal and agrees with what the corpus showed when the tail was demoted. Also
  `-hq` is mostly an MoE knob: in `architecture/qwen3_5.py`, `select_hq_bits = 2 if use_moe else 0`, so on
  this dense model exllamav3 promotes nothing while NInfer's `-hq` raises `attention/*` by a bit. (3)
  exllamav3 decodes the half rates fast (`exl3_dq.cuh::dq8_half`, instances `comp_units/exl3_comp_unit_h{1,2,3}`):
  consecutive windows alternate KA and KA+1 bits, so four share an `18 + 3*KA` bit field taken with one funnel
  shift. NInfer's odd fallback was the 16-iteration `tile_state`; `exl3_windows_half` ports the funnel form
  for `18 + 3*KA <= 32` (up to 4.5 bpw, where exllamav3 stops) in the GEMV, small-m GEMV and tiled MMA,
  leaving wider half rates on the bitwise decode. Adding rates 5, 7 and 9 to the A16 oracle also exposed a
  latent bug: both GEMVs guarded their 4-bit specialisation on `bits == 4`, and `bits = half_bits >> 1` is 4
  for **both** 8 and 9, so 4.5 bpw was decoded with the 4-bit kernel (which also wraps in a 32-word tile that
  a 36-word rate does not have). The guard is now `half_bits == 8`; the scoring path was unaffected because
  the tiled MMA branches on `half_bits & 1`, so the 3.5 bpw point's PPL was and is valid while its decode was
  not.
- 2026-09-28: M5 curve, three points. Uniform `-hq` at 3.5 bpw (`--bits 7`, attention raised to 4.5 by the
  promotion) gives `models/qwen3_8_27b_exl3_3p5bpw.ninfer`, 13.91 GiB, full-corpus PPL **4.310089** (chinese
  5.009, english_long_form 6.914, english_reference 5.968, code 1.657) at 440 tok/s once the half-rate decode
  landed. The three EXL3 points on the same 261,167-token corpus are 3.0 bpw 12.47 GiB / 4.38399, 3.5 bpw
  13.91 GiB / 4.31009 and 4.0 bpw 15.35 GiB / 4.29390, against Q4 4.34389 at 16.96 GiB and NVFP4 4.31493 at
  22.09 GiB. So **3.5 bpw beats both at 82% and 63% of their size**, 4.0 bpw beats them at 91% and 70%, and
  3.0 bpw is 0.9% behind Q4's PPL at 74% of its size. The curve is monotone in every domain, and the two
  redistributed 3.0 bpw artifacts (`_alloc` 4.595, `_even` 4.609) stay out of it.
- 2026-09-28: M5 external reference, first pass (cross-engine PPL). Both engines scored the identical
  `wiki.test.raw` (sha256 `173c87a5...`) at 2048 context. NInfer (`--context 2048 --stride 2047`; 297,192
  scored tokens) gives 3.0 bpw 6.977040, 3.5 bpw 6.895462, 4.0 bpw 6.868214. exllamav3's `eval/ppl.py` on its
  `exl3-reference` 3bpw and 4bpw (non-overlapping 2048 rows, 145 rows) gives 7.056015 and 7.037993 -- its own
  3.0 to 4.0 spacing is only 0.018, so the corpus is forgiving, yet NInfer measures 0.08 and 0.17 lower. Two
  caveats keep this suggestive rather than a claim: NInfer cannot take `stride == context`, so its windows
  overlap by one token and its scored-token count differs (297,192 against about 296,960); and NInfer's `-hq`
  raises `attention/*` by a bit while exllamav3's `select_hq_bits` is 0 on this dense model, so the two
  recipes are not bit-for-bit comparable at equal body bpw. That is the argument for the shared-reference KLD
  of section 9; this PPL pass stands as the cheap external check.
- 2026-09-28: M7 scoped and staged. Mapping the tree corrected two premises of the section 9 sketch. There is
  **no** BF16 forward anywhere -- the quantizer's calibration runs a servable *quantized* activation model, and
  the BF16 `linear` registry has only `[14336,5120]`, `[5120,6144]` and `[256,5120]`, so the MLP, GDN input and
  head geometries are missing -- and there is no public logits API (`score_tokens` returns target logprobs
  only, while full logits live transiently in `ProgramImpl::causal_score`). The composed Program also cannot
  run the reference: it materializes one ~52 GiB device arena and holds immutable baked pointers with no
  weight-provider indirection. So the producer becomes a dedicated offline executor reusing
  `prefill_text_chunk`/`run_layers` with a double-buffered per-layer staging arena, while the consumer reuses
  the CausalScoring Program and needs only a logits-export route. To land M5's evidence first, M7 splits into
  M7a (consumer: logits route, `--reference`, KL accumulator, reference produced externally) and M7b (the
  streamed BF16 producer).
- 2026-09-28: M7a lands (the producer run is the remaining piece). `score_tokens` gained a defaulted
  `LogitsSink`; the flush that already materializes the BF16 vocab logits copies them to a pinned buffer
  (allocated on first use only) and hands back a `ScoredLogits` view. The real-artifact scoring test exports
  them and re-derives the log probability from the column, agreeing to **1.8e-06**. `ninfer-perplexity
  --reference FILE` reads a self-describing KL reference (magic, version, vocab, rows, context, stride, text
  sha256, scored target indices, BF16 rows) and reports `KL(P_ref || P_model)` per stream and domain beside the
  PPL, refusing a protocol mismatch and erroring when no position matches; `tools/perplexity/kl_reference.py`
  writes that format from a BF16 Hugging Face run that mirrors `plan_windows`. Verified end to end with a
  synthetic 300-row reference: 283/283 scored positions matched and a finite KL. Still owed: the interim
  producer run over the real BF16 model, which needs the streaming M7b executor or an offloading HF pass.
- 2026-09-28: M5's KLD against BF16, on the plan's own metric. A BF16 reference (HF producer over a
  23.5k-token wikitext slice, text digest verified, every 8th scored position, 2,940 rows at context 4096 /
  stride 2048) scored the same positions for all five artifacts:

  4.0 bpw **0.033235** (15.35 GiB, PPL 4.29390), Q4 0.042903 (16.96 GiB, 4.34389), NVFP4 0.050978
  (22.09 GiB, 4.31493), 3.5 bpw 0.062384 (13.91 GiB, 4.31009), 3.0 bpw 0.101758 (12.47 GiB, 4.38399).

  EXL3 4.0 bpw wins on both metrics. **The metrics rank the rest differently**: 3.5 bpw has a better PPL than
  Q4 and NVFP4 but a *worse* KL divergence, and Q4 and NVFP4 swap between the metrics. Interleaving the
  reference into halves reproduces the exact ordering in both (0.0233 / 0.0353 / 0.0418 / 0.0482 / 0.0774 and
  0.0432 / 0.0505 / 0.0602 / 0.0765 / 0.1261), so the ranking is robust while the absolute values are
  text-dependent (about 2x between halves) and only the ordering should be quoted. M5's "3.5 bpw beats
  Q4/NVFP4" was a PPL-only claim and does not survive the plan's metric; 4.0 bpw's win does.
- 2026-09-28: The KLD pipeline caught two of its own bugs, both silent. The writer padded the 32-byte field
  block to 96 before the digest, so the reader took digest bytes as scored positions and misaligned every
  logit row (2,924 of 2,940 rows "matched"); and the producer used the input position predicting `t+1`
  instead of `t`. Both still produced a finite KL of about 18.8 -- only the ordering contradicting the PPL
  and the absurd magnitude gave them away. Guarded now by a Python test on the writer's byte layout, the C++
  reader/KL oracle test, and `tools/perplexity/check_reference.py`, which asserts a reference predicts its
  own text (argmax == next token, 60.8% healthy against 1.8% broken).
- 2026-09-28: The 3.0 bpw tier is removed. It measured worst of the five on KL divergence (0.1018 against
  4.0 bpw's 0.0332) and behind Q4 on PPL, so its only argument was being the smallest artifact, and the user
  judged that not worth shipping. `models/qwen3_8_27b_exl3_3bpw.ninfer` and the two redistributed allocation
  variants are deleted, leaving 3.5 and 4.0 bpw; the progress log keeps their measurements.
- 2026-09-28: A stale-build trap, and the corrected M4 numbers. `build/apps/ninfer` and `build/apps/ninfer-serve`
  had not been relinked since the M4 kernel rewrite, so the day's speed work was measuring the *pre-M4 SIMT*
  path: a 1.7 tok/s decode (the M3 log's own SIMT number), a 0.61 s stream sync per token, and a 3.5 bpw
  artifact that stopped after one token and repeated. `ninfer-perplexity` and the test binaries were fresh, so
  the same artifacts scored and passed correctly while the apps did not -- which is exactly what made it
  convincing. After `cmake --build build -j` (and deleting the unused `build-cxx23/` and `build-port/` trees),
  the same sweep reads: 4.0 bpw 38.5 decode / 315.2 prefill, 97.5 decode with MTP3 at 51.3% acceptance; 3.5 bpw
  36.7 / 147.6, 100.6 with MTP3 at 52.5%. So the 3.5 artifact is sound and the half-rate decode carries **no
  decode penalty**, as the oracle tests said. Two items survive the correction: the 3.5 prefill is 2.1x slower
  than the 4.0's, and the same message renders to 26 tokens on 3.5 against 68 on 4.0. The rule that prevents a
  repeat: build the app you are about to measure, not just its neighbours. The M4 log's 71 tok/s decode is now
  38.5, so a real decode regression remains to be explained.
- 2026-09-28: The decode regression is explained, and fixed: contraction occupancy. On the identical
  request: Q4 78.5 decode / 425.3 prefill, NVFP4 72.6 / 1.21k, EXL3 4.0 38.1 / 314.6 -- EXL3 slowest while
  being the *smallest* artifact (15.35 GiB against 16.96 and 22.09), i.e. about 0.59 TB/s against NVFP4's
  1.59. NCU on the three contraction kernels showed none of them near the DRAM floor: `Block Limit Registers
  = 2` (2 blocks/SM, ~16% occupancy) with `SM Active Cycles` 743 of `Elapsed Cycles` 9,406, so they were
  latency-bound -- waiting, not streaming. No kernel carried `__launch_bounds__`, so nvcc spent registers
  until occupancy collapsed. Sweeping the target: 3 blocks 49.5 / 326.6, 4 -> 57.3 / 342.0, 5 -> 62.4 / 251.5,
  6 -> 41.4 / 196.3, 8 -> 22.1 / 204.1 (decode / prefill). The families peak differently -- the GEMV at 5, the
  MMA at 4 before its accumulators spill -- so they carry separate targets now: **decode 38.1 -> 62.5 tok/s
  (+64%), prefill 314.6 -> 345.2, TTFT 224 -> 204 ms**, oracles unchanged. The earlier M4 claim that the GEMV
  sat at the DRAM floor came from a per-call figure on the largest projection; at model scale it was
  occupancy-bound. EXL3 still trails Q4 and NVFP4 on both, so the remaining gap is kernel work, not the format.
- 2026-09-28: M4 prefill investigation -- located, not yet fixed. On an equal request EXL3's prefill is several
  times behind the engine's other formats: at 0.5k / 7.8k prompt tokens, EXL3 4.0 495 / 579 tok/s, Q4 2,340 /
  2,880, NVFP4 5,230 / 8,870. Decode is much closer (62 / 59 against 77 / 74 and 72 / 69), and the depth curve
  disposed of the first hypothesis -- decode falls only about 5% from 0.5k to 7.8k for all three, so there is no
  structural depth problem (the "144 ms per step" read was graphsignal's mean over a window mixing prefill and
  decode). The prefill gap is not kernel time either: a trace of a 2,805-token prefill shows the EXL3 kernels at
  0.62 s of a 5.1 s wall, with **4.03 s of blocking stream sync over 34 syncs (118 ms each)** while
  `gpu_utilization_percent` reads 5% and process CPU 0.04%. Q4 and NVFP4 do not stall, so the stall is specific
  to the EXL3 prefill path rather than engine scheduling. Two cheap levers are ruled out: chunk size barely moves
  it (1024 / 2048 / 4096 -> 571 / 579 / 585 tok/s, and a single 4096 chunk still stalls), and this box's memory
  clock is already 14,551 of 14,751 MHz, so the +4500 offset a comparable 2x5090 ExLlamaV3 setup relies on is not
  available. Bounding the two 128-thread kernels (`exl3_input_transform`, `exl3_gemv_finish`) was tried and
  reverted: no prefill change, and it pushed the fused-consumer oracle to 2.4e-3 on `gdn_input_proj` past its
  1e-3 gate. Next: identify the sync site in the EXL3 prefill path.
- 2026-09-28: M4 prefill -- "blocking stream sync" red herring resolved; `exl3_mma` rewritten. An nsys trace of the
  same 2,805-token prefill showed the 4.03 s of blocking sync is just the host waiting at the chunk boundary: the
  real cost was the prefill contraction kernel itself, **4.82 s of a 5.06 s wall over 256 `exl3_mma<64>` calls
  (avg 18.8 ms)**. The old block-wide form staged one decoded B per 128-column block behind a `__syncthreads()`
  per k-tile and re-read a shared A tile for every fragment; NCU had it register-limited at 2 blocks/SM. The
  rewrite decodes per warp (each warp's 16 output rows decode their own trellis tile once per k-tile),
  double-buffers the A stage through `cp.async` (16 KiB smem: FP32 A stages plus a `[16][128]` epilogue buffer
  reused across sub-tiles), and publishes each 16-row sub-tile's Hadamard and store before the next. Three
  correctness traps on the way, all caught by the A16 oracle: the first pipeline draft waited on the wrong group
  and read uninitialized shared memory (nondeterministic failures across rates); staging BF16 directly under a
  raw-byte `cp.async` copy sliced FP32 bit patterns into BF16 halves (NaN); and the m16n8k16 A fragment loaded
  `col+8` where row+8 belongs. Final `exl3_mma<64>`: 80 registers, zero spills, 16 KiB smem, 3 blocks/SM
  (4-block target fits in 64 registers only by spilling and measured no better). **Prefill 555 -> ~830 tok/s
  (1.5x), decode unchanged (~60)**; per-call kernel average 18.8 -> 13.5 ms; both oracles and the fused consumers
  pass. EXL3 still trails Q4 (~2.9k) and NVFP4 (~9.5k) by 3.5x/11x, so the remaining gap is memory- and
  decode-bound kernel structure, not scheduling; the next lever would be wider N per block to amortize the
  per-block decode across more output, or an FP8/BF16 decode-to-smem B stage feeding `wgmma`-shaped tiles.
- 2026-09-28: The fused-consumer oracle was flaky, not broken: its gate sat below one BF16 ULP. The rewrite above
  claimed "both oracles and the fused consumers pass", but `ninfer_linear_exl3_consumers_test` failed roughly 40%
  of runs on `gdn_input_proj` at T = 2. Rebuilding the pre-rewrite kernel (`git checkout 708d4437 -- exl3_dispatch.cu`)
  and counting 20 runs gave 9/20 failures against 8/20 for the rewritten kernel, so the flake predates the rewrite --
  which anyway touches only `exl3_mma`, while the failing case is the T = 2 GEMV. Counted elementwise, every failure
  is 0-7 elements off by 1-5 BF16 ULP, and forcing the GEMV to `splits = 1` (one ordered K pass) makes the reference
  and the consumer bit-identical on every run. The cause is the split-K GEMV's `atomicAdd` K reduction, whose order
  varies per launch, so two evaluations of one parent differ by a few ULP while the old 1e-3 relative-to-max gate
  sat below one ULP of a near-max element. The attn and gdn gates now bound the worst element by four ULP of the
  largest magnitude -- a row-mapping defect displaces whole rows by O(128) ULP -- and two T-above-the-cutoff cases
  (T = 32 and T = 96) take the consumer path through `exl3_mma` at both tile sizes. 0 failures in 20 runs; the A16
  oracle still passes.
- 2026-09-28: M4 prefill, the A-staging bank conflict -- prefill 806 -> 1.41k tok/s. NCU on the real prefill
  (`--prefill-chunk 1024`, 2,805-token prompt) put the contraction at 95.7% of GPU time: 768 `exl3_mma<64>` calls over
  3.41 s (91.6%, 4.44 ms average), 256 `exl3_mma<16>` at 4.1%, `exl3_input_transform` at 0.9%; DRAM was 5% of peak, so
  nothing was memory-bound. The limiter was the A operand: it was staged FP32 in a 16-wide shared row and every warp
  re-read the whole `[TILE_T][16]` tile with eight strided loads per fragment per k-tile, which NCU measured as a
  3.9-way bank conflict -- "64,167,936 excessive wavefronts, 74% of 87,197,184", estimated local speedup 51.85%. A is
  now BF16 end to end (`exl3_input_transform` rounds once on store) and staged in two 8-wide k-planes: a plane row is
  four shared banks, so the eight rows one fragment spans land exactly once on the 32 banks and each of the four A
  registers becomes a single conflict-free 32-bit load -- no convert, no permute, half the scratch and half the
  cp.async bytes. `exl3_mma<64>` fell 4.44 -> 2.46 ms per call (1.80x), `<16>` 0.59 -> 0.46 ms (1.30x), total GPU
  3.73 -> 2.16 s, and the served 2,805-token prefill 806 -> 1.41k tok/s with TTFT 3.6 -> 2.0 s; decode is unchanged
  (60-62 tok/s). Prefill arithmetic is bit-identical (the MMA always consumed BF16 A, so the rounding only moved from
  the fragment pack into the transform); the T = 1 FFMA path used to keep A in FP32 and now shares that rounding,
  which is the operand precision the T = 2..8 GEMV already had. Both EXL3 oracles pass.
- 2026-09-28: M4 prefill, where the remaining time goes. After the A-staging fix the contraction measures
  74.4% compute and 74.4% memory on the real `<64>` launch (grid 112x16, T = 1024, 1.90 ms): NCU calls it
  balanced, the bank-conflict rules are gone, DRAM is 2.4%, L2 hit 97.8%, and IPC rose 1.82 -> 2.16. The
  tensor pipe is only 43.9% of cycles while the ALU and LSU pipes sit near 74%, so the tensor cores are
  starved by the work around them -- and that work is already at its floor: SASS shows one `IDP.4A` per
  `mul1_value`, one `F2FP.BF16.F32.PACK_AB` per fragment register, `LDGSTS` for the cp.async stage, and the
  A fragment is one 32-bit load per value, each used by exactly two mma (h = 0, 1). Amortising further means
  more mma per decoded weight, which costs the other pipe: 2 n16 rows per warp halves the A loads but doubles
  the accumulator. The one NCU still offers -- ~25% from occupancy -- is closed: 3 blocks/SM is already the
  register limit (the `[2][MT][4]` FP32 accumulator), and raising the target to 4 spills and drops prefill
  from 1,420 to 832 tok/s (measured, then reverted). So what is left is the intrinsic price of decoding the
  weights at run time: Q4 dequantises for free and runs at ~74% of the BF16 peak, EXL3 lands at ~38% with the
  tensor pipe at 44% -- the ~2x that separates them (1.43k tok/s against Q4's 2.86k on the same 2,805-token
  prompt, TTFT 2.0 s against 0.98 s). The 3.5 bpw artifact prefills at 2.03k tok/s -- within 1.4x of Q4 -- with the
  same A-staging fix, even though its odd-rate trellis decode is the heavier one on the decode side. Closing the 4.0
  bpw gap needs the accumulator out of registers (a streamed sub-tile epilogue) so one decoded weight can serve more
  tokens, not another launch-bounds or layout tweak.
- 2026-09-29: M4 decode -- measured, and the obvious levers closed. Graphsignal with `--cuda-graph-trace node` (the
  decode replays CUDA graphs, which is why an nsys kernel sum saw only 309 ms of a 3.4 s request) puts
  `exl3_gemv_split<1>` at 2.71 s of 3.3 s of kernel time over 200 tokens: 51,961 calls, 52 us each, 260 calls/token,
  with the profiler itself costing 4% (57.7 against 60 tok/s). `ninfer_decode_device_wait_microseconds_per_round` is
  15.8 ms against 1.6 ms of host and a batch of 1.0, so the GPU is busy and there is no batching slack. NCU on the
  real shapes: the head (n = 248320, 6 bits) reaches 73.9% of DRAM, but the byte-heavy MLP projections sit at 44-60%
  DRAM with the ALU pipe at 58% and an ALU-heavy instruction mix -- NCU's own "below 60% ... typically indicate
  latency issues" -- at 5 blocks/SM and 78% achieved occupancy. That is 13.5 ms/token of GEMV against a ~8-9.5 ms
  weight-read floor, so ~1.3x is reachable and Q4's 78.6 tok/s on the same prompt is the target.
  `upstream/dev` (Neroued/ninfer, tip 2026-09-29) has no EXL3 but the same techniques: split-K with explicit FP32
  partials and a merge kernel, caller-owned partial storage (21.25 MiB), native pair conversions in place of an
  FP32/FP16 bridge, and a wider-vector T1 GEMV. Our own tree agrees that EXL3 is the outlier -- `grep atomicAdd
  src/ops/linear/` hits only `exl3_dispatch.cu`, while q4/q5 use `ksplit_mma`, which "reduces FP32 partials in shared
  memory". Two attempts at that lever lost and were reverted: an in-CTA row split on the `q5_ksplit_mma` shape (16 rows
  per CTA, eight K warps, shared-memory reduce, no memset, no atomic) gave 52.2 tok/s against 61.5 because it trades
  grid parallelism for locality -- the whole reason the grid-level split exists at T = 1 -- and a 6-block register
  target spilled to 41.3 tok/s. So the atomic grid-level split-K is the right shape for T = 1, the load pipes are not
  the limiter (`Mem Pipes Busy` 35%, `L1/TEX` 37.7%, so widening the operand loads cannot pay), and the one unspent
  lever is software-pipelining the K loop to prefetch the next trellis word -- raising memory-level parallelism
  without more warps, inside the 51-register budget. Decode holds at 61.9-62.6 tok/s and prefill at 1.44-1.45k, both
  oracles pass, and the experiments left no trace in the tree.
- 2026-09-29: M4 decode -- the probe instantiation, six blocks per SM, and paired operand loads. Decode
  61.9-62.6 -> 71.0-71.2 tok/s at 4.0 bpw (+15%) and 57.6 -> 64.5 at 3.5 bpw, prefill unchanged at 1.45-1.46k. Three
  changes. First, the served T = 1
  GEMV carried the sensitivity probe's per-window Gaussian (`if (probe_sigma > 0.0F)`, never taken in service) in its
  own instantiation; a `kProbe` template parameter compiles it out for service while `ninfer-sensitivity` keeps it.
  That block was costing far more than the branch it guarded -- its reconvergence barriers sat in the hot K loop --
  and it is what freed the register budget. Second, with the probe gone the kernel needs 40 registers and no stack
  frame where the shared instantiation needed 48, so the launch bound moved from five blocks per SM to six (seven
  makes ptxas abandon the allocation, REG:255, and decode collapses to 23.3 tok/s). Third, the lane's four k
  positions are the adjacent pairs 2c, 2c+1 and 2c+8, 2c+9, so one 32-bit BF16 pair load now serves each half of the
  fragment instead of two 16-bit halves -- two latency-bearing loads per column instead of four, with the register
  count unchanged at 40.
  Reading q4/nvfp4 for the same idea turned up three house patterns we are not using. Q4's T = 1 GEMV
  (`q4_instances.cuh`) runs **tiny row tiles** -- `GemvR1W8K5120` is one row with eight K-split warps, `GemvR4W1` four
  rows -- with the K reduction inside the CTA and no global partial at all, which is only possible because its group
  granularity is 64 k positions, not 16 rows; EXL3's decode yields a whole 16-row trellis tile per warp, so its row
  tile cannot shrink below 16 and the grid-level K split is what supplies the CTAs (which is exactly why the in-CTA
  row split lost). It stages codes with `CodeTransfer::AsyncVector16` -- a cp.async 16-byte ring -- where this GEMV
  reads the trellis straight from global and consumes it in the same iteration, leaving the DRAM latency exposed; that
  is the transferable idea, and the one unspent lever named above. And it launches through `pdl::launch_dependent`
  (programmatic dependent launch) for the transform/gemv/finish chain, which may or may not pay under a replayed
  graph. Its `Cache` policy is only `{ca, cg}` and Q4 uses `ca`, the same L1-cached path our trellis loads already
  take, so the "streaming weights" hint is not the difference.
- 2026-09-29: M4 decode -- the 4-bit rate pinned, and the cp.async staging is a negative. Decode 71.0-71.2 -> 75.4-75.9
  tok/s at 4.0 bpw (+21% over the 61.9-62.6 this work started from), prefill unchanged at 1.42-1.46k, both oracles
  pass. The kernel evaluated `half_bits == 8 / fast / exl3_half_fast` once per k-tile; the 4-bit rate carries most of
  the model's projections by call count, so pinning it as a template parameter removes that chain -- and the wide
  rates' six window registers -- from the instantiation the bulk of the decode runs. It is the same lesson as the
  probe block: a cold branch's reconvergence in the hot loop costs more than the work it guards, while the register
  count did not move (40 either way). A seven-block target still overruns (REG:45) and is slower. Separately, the
  pattern q4 (`AsyncVector16`) and nvfp4 (`Nvfp4GemvSharedStorage`, `cp_async_zfill<16, Cache::cg>`) both use --
  stage the weight stream into shared memory with 16-byte L2-only copies so the DRAM fetch overlaps the decode -- was
  built for this GEMV as a per-warp eight-tile ring with one committed group per tile and a warp-level wait. It is
  correct (both oracles pass with it) but 9% slower, 71.0-71.2 -> 64.1-64.4 tok/s, so it is reverted: after the probe and
  occupancy changes there are 48 warps per SM with two loads in flight each, the direct read already hides the DRAM
  latency, and the staging only added a shared round trip, a per-tile warp wait, and `cg` on a stream whose neighbour
  word used to be an L1 hit. Explicit staging pays when there are not enough warps to hide the latency; this kernel
  no longer is that case.
- 2026-09-29: M4 prefill -- the two cold branches were the gap. Prefill 1.42-1.46k -> 2.29-2.47k tok/s on the
  2,805-token prompt (+68%), TTFT 2.0 -> 1.2 s, by applying to `exl3_mma` what the decode GEMV had just taught: the
  hot K loop carried the sensitivity probe's inline per-window Gaussian behind `if (probe_sigma > 0.0F)`, never taken
  in service, and the 4-bit rate -- most of the model's projections by call count -- went through the wide-even-rate
  `plan`/`exl3_window_value` path, eight funnel shifts per tile plus the eight registers holding the window plan. A
  `kProbe` parameter compiles the first out of the served instantiation; a `kRate4Bit` parameter routes the second to
  `exl3_windows_4bit`, one funnel and five bit-field extracts, the extraction the GEMV and the A16 oracle already
  use. The register count falls 80 -> 77. The 3.5 bpw tier's rates are odd, so it gains only the probe removal
  (2.04 -> 2.11k), which is what the split predicts.
  That also closes the head-to-head from the entry above. At one identical shape Q4 issued 11.1 instructions per MMA
  against EXL3's 42.2, both issuing exactly the required 89,128,960 MMAs -- Q4 tensor-bound at 94.6% of compute,
  EXL3 issue-bound at 78.8% with its tensor pipe at 43.9%. The 31 extra instructions per MMA were not the decode but
  the control around it, and removing two cold branches recovered most of them: served prefill is now 2.44k against
  Q4's 2.86k (**1.17x**, from 3.5x) and decode 76-78 against 78.4 (**1.03x**), while the EXL3 artifact is 1.6 GiB
  smaller (15.35 against 16.96 GiB).
  The re-profile confirms it at that shape: `exl3_mma<64>` 4.25 -> 2.41 ms and its instruction count 3.759e9 -> 1.346e9,
  i.e. **42.2 -> 15.1 instructions per MMA** against Q4's 11.1, with the tensor instruction count identical at
  89,128,960 in both before and after. So the 31 "extra" instructions per MMA really were control, not decode, and
  what remains above Q4 is roughly four instructions per MMA per tile -- the one funnel, five bit-field extracts and
  `IMAD`/`DP4A` per window that the 4-bit trellis costs and a nibble plane does not.
- 2026-09-29: M4 release verification -- quality, MTP, Vision; the route is published. The full-corpus perplexity was
  re-run on the current kernels for both tiers and is **bit-identical** to the published figures: 4.0 bpw overall
  **4.293931** (chinese 4.974198, english_long_form 6.845779, english_reference 6.007342, ninfer_code 1.649449) and
  3.5 bpw overall **4.310089** (5.008571, 6.914397, 5.967606, 1.657355). Both come from the large-T MMA path, where
  the runtime work stayed bit-identical -- the BF16 A stage moved the rounding earlier rather than changing it -- and
  KL divergence is computed from the same logits, so it carries over with them. The scoring pass itself now runs at
  679.6 tok/s (4.0 bpw) and 993.6 (3.5 bpw) against the 380.7 and 440 recorded when those numbers were taken.
  MTP measured for all four artifacts at a fixed three-token draft: EXL3 4.0 bpw 124, 3.5 bpw 120, Q4 134, NVFP4 142.
  Q4 and NVFP4 reach 144 and 166 at K=5 with `--lm-head-draft`; **the EXL3 artifacts cannot**, because the conversion
  does not emit the separate draft-head projection the official artifacts carry -- the server refuses at startup with
  "selected proposal head is absent from artifact" and K=5 does not help (124.6). Emitting that projection is the
  follow-up that would close the gap.
  Vision was smoke-tested on the 4.0 bpw artifact with `--vision` against `examples/cli/media/visual_chart.png`, whose
  expected content is documented in `examples/cli/README.md`. It returned the title `NIFER VISION 731`, the three red
  circles and the blue square on the left, plus the `COUNT`/`POSITION` labels and a green triangle: HTTP 200 in 2.9 s,
  media reported as 1 image prepared in 7.66 ms.
  Documentation now publishes the route: the README gained a fourth "what the fork adds" bullet, artifact rows with
  the download line, a recommended-settings row and an EXL3 quantization section; `docs/performance.md` gained
  coverage rows (Not published for the methodology tables) with a pointer to the card; `docs/README.md` links the
  Hugging Face repo and a versioned card; and `model-cards/ninfer-ext-models/README.md` is now the card's source in
  the repository, rewritten where it was stale (it had claimed prefill was "several times slower" and "under active
  work" at 495/579 tok/s). M4 is Complete.

## M3 status and decisions

M3 is complete functionally: `ninfer-quantize` quantizes every eligible parent from the BF16 artifact,
the `qwen3_8_27b_exl3` recipe turns the store into a `.ninfer`, and the engine loads and serves it.
The full-corpus KLD against BF16 and exllamav3, and the fast decode that makes it practical, are M4.

Three decisions the earlier handoff reserved were answered on 2026-09-27:

1. **How the quantizer loads the model — BF16 `.ninfer`, streamed layer by layer.** A BF16 `.ninfer` artifact
   (produced by `tools/convert`) is the single model input; `ninfer-quantize` maps its host objects and reads one
   layer at a time. This keeps `.ninfer` the only model artifact and avoids a second safetensors reader in C++.
   AGENTS.md and artifact materialization both support it.
2. **Device allocations — the offline app owns them.** `ninfer-quantize` is a standalone offline app, outside any
   Engine Program, so `quantize_tensor`'s temporary device buffers (about 13 GB peak at k = 17408) are the app's
   own. It reports peak usage; it is not a server capability and does not allocate on behalf of a Program.
3. **Sequential re-run — BF16 reconstruction.** After quantizing a layer, the calibration pass re-runs that layer
   with the FP32 reconstruction `wq` (already produced by `quantize_tensor`) through the existing BF16 linears, so
   calibration and the M3 artifact do not depend on the M4 EXL3 kernels. This deviates from the original
   "re-run uses the EXL3 inference kernels" wording; it is accepted so the first artifact can be measured before
   M4. M4 later replaces the re-run with the real kernels and re-quantizes.

Deliverable order inside M3:
- `ninfer-quantize` app: BF16 `.ninfer` loading, parameter enumeration, BF16→FP32 weights, calibration Hessians,
  `quantize_tensor`, and a `.trellis/.su/.sv` source writer.
- The official `qwen3_8_27b_exl3` converter recipe.
- The EXL3 linear Op for "loads and serves" (a simple decode kernel qualified against an FP64 oracle decoded from
  the stored planes).

## Review notes (2026-09-27, external read-only review of `385bb9af` and the uncommitted sampler)

Verified correct: the mul1 codebook (checked exhaustively), tile sizing for every rate including half-bit rates,
circular 16-bit state windows with tail-biting, the packer's path check, plane offsets and alignment, the overflow check,
and the guards that keep EXL3 tensors out of `native_weight` and `weight_row_planes`.
`ninfer_exl3_trellis_test` and the 32 Python EXL3 tests pass on a rerun. Open items, in priority order:

1. **Resolve before M3: the plan contradicts itself on model loading.** Section 4.1 step 3 still says
   `ninfer-quantize` "loads the BF16 artifact through the normal loader". The BF16 artifact was waived, and the
   52 GB source cannot be resident in 32 GB. The calibration Program has to stream one layer's source tensors at a
   time from `/mnt/storage/models/qwen3.8/full`. Section 4.1/4.2 should state that, along with who owns reading the
   source safetensors, since the loader today only reads `.ninfer`.
2. **Codec test independence.**
   - `reference_state` in `test_exl3_trellis.cpp` repeats the implementation's formula, so it is not an independent
     oracle.
   - Random tiles always decode to a valid path, so the packer's "not a circular trellis path" branch is never
     exercised.
   - Add a hand-worked tile with literal expected states (one integer and one half-bit rate), plus a negative case
     with a broken state sequence.
3. **Geometry field aliasing.** The `TrellisT16` branch of `weight_geometry` also writes `high_offset`/`high_bytes`
   (input scales), `scale_offset`/`scale_bytes` (output scales) and `code_bytes`. Any future generic reader of the Q5
   high plane or the scale plane would silently misread EXL3. Prefer leaving those fields zero and using only the
   `input_scale_*`/`output_scale_*`/`trellis_*` fields.
4. **Calibration row padding (M3).**
   - `pack_calibration_rows` zero-pads to `--row-tokens`, and token id 0 is a real vocabulary token. The `XᵀX`
     observer must mask by the stored `lengths`, or padding rows distort `H`.
   - Tool follow-up rows repeat the previous turn's prompt tokens, so that shared prefix is counted twice in `H`.
     Either accept and document this, or count only the new suffix of a follow-up.
5. **M2 oracle discipline.** Viterbi and LDLQ are where silent quality bugs hide: a slightly wrong path cost or
   error-feedback sign still produces a working but worse quant. Keep the section 4.3 oracles mandatory:
   - brute-force minimum-cost search on small synthetic trellises;
   - bit-exact path and cost against the host FP64 Viterbi on real tiles;
   - FP64 LDLQ on small matrices.
   Also compare the single-tensor proxy error with exllamav3 on the same tensor and Hessian before running the
   full model.
6. **Environment durability.** The exllamav3 reference venv lives in `/tmp/opencode/...` and will not survive a
   reboot. Its work directory on `/mnt/storage` does. Record how to recreate the venv, or move it off `/tmp`, before
   M3 needs the reference again.
7. **Style.** The new enumerators in `src/core/weight.h` and the new fields in `src/core/weight_view.h` are not
   aligned with their neighbours. Run clang-format on the touched files.
8. **Resolved: the BF16 reference loads through QBench's layer-major streaming.** This item originally said a
   whole-model load could not fit. QBench's `streaming: true` backend is exactly the recommended design: it builds a
   meta skeleton, materializes one module at a time from the shards, pushes every row through it, then frees it, so
   peak VRAM is one layer plus about 2 GB of activations. The 100-row evaluation pass plus its noise-floor pass took
   minutes, even with Transformers' pure-PyTorch GDN fallbacks (no `fla`/`causal_conv1d` in the tooling venv).
   Note that QBench infers row length from the last nonzero token, so a real trailing token id 0 would be dropped
   (irrelevant for chat traces, which end in template tokens). The M3 calibration Program still needs its own native
   layer streaming (item 1).
9. **Resolved.** The 3.0 bpw reference is recorded in the progress log and scored below. Its calibration is still
   exllamav3's built-in data, not our trace, so that confound remains when our quant is compared against it.
10. **Resolved.** The serve option was reviewed and committed (`e587576b`). The review found that an empty completion
    made the formatter throw, which permanently disabled the trace; empty completions are now valid records. The
    indentation was restored.
11. **Sampler: every evaluation response hit the 256-token cap mid-thinking.** All 100 rows report 261 sampled tokens
    (256 plus template closers), so the 8% tool rows never reached an actual tool call and no follow-up turns were
    produced. For the calibration trace, either raise `--max-new-tokens`, or sample tool rows with thinking disabled,
    so the calibration activations include real tool-call syntax.
