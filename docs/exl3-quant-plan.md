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
1. Source checkpoint `/mnt/storage/models/qwen3.8/full` is the full-precision authority. M0 uses it
                      directly as QBench's streaming HF reference; a separate BF16 `.ninfer`
                      baseline artifact is not required. Produce one later only if the selected M3
                      loader path requires it.
2. Calibration trace  `tools/exl3/sample_traces.py` starts `ninfer-serve` on the existing Q4 artifact,
                      samples chat continuations from disjoint corpus shards, and writes qbench JSON,
                      packed token rows, and a text stream for evaluation.
3. Quantize           ninfer-quantize (new C++23/CUDA app) loads the BF16 artifact through the normal
                      loader, runs calibration layer by layer, quantizes every eligible linear, and
                      writes encoded EXL3 rows as a converter source
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
- **Sequential:** run layer ℓ on the BF16 weights while observing → quantize ℓ's linears → re-run ℓ with the quantized weights to produce layer ℓ+1's input.
  - The re-run uses the EXL3 inference kernels, which also exercises them end to end.
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
| M0 | In progress | Baselines | Existing Q4/NVFP4 NInfer PPL baselines recorded; BF16 source and exllamav3 3.0/4.0-bpw references scored on the self-sampled qbench trace (isolated tooling venv, streaming HF reference); self-sampled calibration and eval traces generated with `ninfer-serve`. No separate BF16 `.ninfer` artifact is required. |
| M1 | Complete | Format, layout, codec | `exl3_mul1` + `trellis_t16_v1` registered (Python + C++), docs written, tile and bit order chosen by microbenchmark, exact codec tests pass |
| M2 | Pending | Quantizer maths | FWHT, `XᵀX`, blocked Cholesky/LDL, Viterbi (integer + half-integer K), LDLQ, pack, refit, all passing their oracles; single-tensor proxy error matches the FP64 host pipeline |
| M3 | Pending | Calibration Program + first artifact | `ninfer-quantize` produces a full 27B EXL3 artifact at 4.0 bpw with `-hq`, using a simple correct decode kernel for the sequential re-runs; artifact loads and serves; KLD recorded vs BF16 and vs exllamav3 |
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
8. **Blocks M0: the BF16 reference cannot be loaded whole.** The 52 GB source does not fit on the 32 GB GPU,
   so a plain Transformers or QBench load will run out of memory. Options, in order of preference:
   1. **Layer-streamed reference forward (recommended).**
      - For each layer ℓ, load only that layer's source tensors to the GPU, run every evaluation row through it,
        keep the hidden states, and free the layer.
      - Hidden states for 100 rows × 2048 tokens × 5120 in BF16 are about 2 GB. The source is read once, so a full
        pass costs roughly the 52 GB disk read.
      - At the end, apply the final norm and LM head, and store per-token top-k log-probs. Use k of about 64–128
        plus the log of the tail mass, since full 248,320-way logits are too large to keep. KLD for any candidate
        (EXL3 reference, our EXL3, Q4, NVFP4) is then computed against these stored log-probs.
      - This is the same layer-streaming schedule the M3 calibration Program needs (item 1). Build it once and
        reuse it: first as the BF16 reference scorer, then with the `XᵀX` observer and quantize-then-rerun steps
        for calibration.
      - A quick first version can run in the tooling venv with PyTorch and the HF layer modules loaded one at a time.
      - The native C++23 version belongs to M3.
   2. **Transformers with CPU offload** (`device_map="auto"`, about 28 GiB GPU budget, the rest in the 247 GB of
      host RAM). No new code, but every forward streams about 25 GB of weights over PCIe. Acceptable for about 100
      evaluation rows if batched, too slow for anything larger. Verify QBench's "streaming" backend really offloads
      before relying on it.
   3. **Near-lossless proxy.** An exllamav3 6.0 bpw quant (about 21 GB) fits in VRAM, and its KLD against BF16 is
      far below the 3–4 bpw differences being measured. Acceptable for ranking quants, but label it clearly in M0
      as a proxy, not BF16.
   4. **CPU-only BF16.** It fits in RAM but takes hours for the evaluation trace. Not recommended.

   Do not substitute the Q4/NVFP4 artifacts for the reference, as the progress log already notes.
9. **The reference quant finished; record it.** The exllamav3 3.0 bpw conversion completed at 09:58, about 45 minutes
   including one resume.
   - Output: `/mnt/storage/models/qwen3.8/exl3-reference/3bpw`, 13.5 GB.
   - Settings: v1.5.2, mul1, `head_bits` 6, `mtp_bits` 4, `out_scales` always.
   - Calibration: exllamav3's built-in data (250 × 2048), not our self-sampled trace. Keep that in mind when
     comparing KLD with our quant: calibration data is a confound unless both use the same trace, or unless it is
     reported explicitly.
   - The progress log still says "running".
10. **Scope of the uncommitted `--generation-token-trace-jsonl` serve option.**
    - Capturing exact generated token IDs is better than re-tokenizing text. But it changes the product server
      (`http_server`, `generation_service`, `serve_options`, `docs/serving.md`), so it needs the normal serving
      review and tests, not just EXL3 tooling review.
    - The usage-text lines it touched in `serve_options.cpp` gained an extra space of indentation (`request-log`
      and the new line). Restore the original alignment.
