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
| M3 | In progress | Calibration Program + first artifact | `ninfer-quantize` produces a full 27B EXL3 artifact at 4.0 bpw with `-hq`, using a simple correct decode kernel for the sequential re-runs; artifact loads and serves; KLD recorded vs BF16 and vs exllamav3 |
| M4 | Pending | Fast inference kernels | Decode GEMV, sliced-K MMA, prefill MMA with epilogue Hadamard; consumers `linear`, `swiglu`, `add`, `attn_input`, `gdn_input`, LM head, MTP; end-to-end speed report |
| M5 | Pending | Recipe optimization | Sensitivity measurement + greedy allocation through NInfer; recipe artifacts at 3.0 / 3.5 / 4.0 bpw with a KLD-vs-size curve against q4/NVFP4/exllamav3 |
| M6 | Pending | Later | Two-sided YAQA LDLQ; Vision tower; Flash-Next (GDN + MoE experts + expert pager); int8-activation route behind a permission |

**Order:** M0 baselines should precede M2/M3 quality work. M1 is independent of the reference
quantization and self-trace work and can complete while M0 remains open. M0 → M1 → M2 → M3 gives the
first real 27B EXL3 artifact with measured quality. M4 makes it fast; M5 makes it best-in-class per
bit.

## 8. Risks

- **Decode ALU budget at T = 1.** About 6 integer ops per weight; 27B at 4 bpw is about 13.5 GB, a ~7.5 ms/token bandwidth floor. M4 must show the decode hidden under memory time. If it isn't, the fallbacks are:
  - more weights per decoded 32-bit window (aligned 2/4-bit fast paths as exllamav3 has);
  - the int8-activation route behind a permission.
- **Shared-`suh` fused parents** could cost quality. Measured in M3; per-member `suh` is the fallback.
- **Calibration Program scope.** It reuses layer execution but adds an observer and a layer-by-layer schedule. It must not leak into serving Programs.
- **Host-side FP64 oracles** are slow on 17408² Hessians, so they run on small and synthetic cases plus a few sampled real tensors.
- **FP16 activation range.** Handled by the exact power-of-two prescale; verified on real activations in M3.
- **27B has one MTP layer:** its quality at its own bitrate affects MTP acceptance, so acceptance rate is reported alongside KLD.

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

## M3 status and decisions

M3 is in progress. Its first half is landed: the converter reads the native quantizer's output format
(`feat(exl3): read native quantizer sources in the converter`). The producer (`ninfer-quantize`) and the EXL3
linear Op are not started.

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
