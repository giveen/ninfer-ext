# Gemma 4 26B-A4B (MoE) plan

Temporary plan for branch `gemma4-26b`. Its outputs move into `docs/maintainer/gemma4-model.md`
(the permanent authority) as each phase lands; delete this file when P5 is done, per AGENTS.md's
one-current-authority rule. Acknowledged gaps that need a product decision are marked **DECIDE**.

## 0. Source and target

- **Source**: `/mnt/storage/models/gemma/26b-full` — QAT, BF16 safetensors, 2 shards, 51.6 GB,
  26,544,131,376 parameters, 1,013 tensors. The assistant is the sibling directory
  `26b-full/assist` (`Gemma4AssistantForCausalLM`, 839 MB). The instruction-tuned checkpoint is the
  right choice for this engine (D2 of the earlier plan): NVIDIA's NVFP4 and the assistant drafter
  exist only for `-it`.
- **Shard state**: both shards are complete (46.5 GiB + 1.6 GiB; 955 + 58 tensors). Every shape below
  is read from the real safetensors headers plus `config.json` and `model.safetensors.index.json`, so
  nothing here is guessed.
- **Target**: `models/gemma4_26b_a4b.ninfer` plus a drafter artifact, served by the same Engine and
  the same public route as the 31B. Text only: the vision tower (355 tensors, 27 layers) and the
  audio config are out of scope exactly as they are for the 31B, and the artifact build must skip
  them the way the 31B recipe does (only `text/` roles are bound).

## 1. Shape facts (config + measured headers)

| field | 31B (shipped) | 26B-A4B | note |
|---|---|---|---|
| layers | 60 (50 sliding / 10 global) | **30 (25 sliding / 5 global)** | global at 5, 11, 17, 23, 29 |
| hidden | 5376 | **2816** | |
| dense FFN | 21504 | **2112** | kept, in parallel with MoE |
| MoE | — | **128 experts, top-8, I_moe 704** | `enable_moe_block: true` |
| attention heads | 32 / 16 KV sliding, 4 global KV | **16 / 8 KV sliding, 2 global KV** | |
| head_dim / global_head_dim | 256 / 512 | 256 / 512 | unchanged |
| sliding window | 1024 | 1024 | |
| vocab / softcap | 262144 tied / 30 | same | |
| RoPE | sliding default θ=1e4, global proportional 0.25 θ=1e6 | same | |
| `v_proj` | sliding layers only | **sliding layers only** (global is K=V) | confirmed: absent on layer 5, present on layer 0 |
| params | ~31B dense | 26.5B total, **≈3.6B active/token** | experts 22.8B = 86% |
| KV per token | — | 5 global layers keep compact rows; 25 sliding layers are windowed at 1024 | long context stays cheap |

Measured tensor shapes (text tower, `model.language_model.`):

| role | shape | dtype |
|---|---|---|
| `layers.N.experts.gate_up_proj` | `[128, 1408, 2816]` (from the reference; shard 1) | BF16 |
| `layers.N.experts.down_proj` | `[128, 2816, 704]` | BF16 |
| `layers.N.router.proj.weight` | `[128, 2816]` | BF16 |
| `layers.N.router.scale` | `[2816]` | BF16 |
| `layers.N.router.per_expert_scale` | `[128]` | BF16 |
| `layers.N.mlp.{gate,up}_proj` | `[2112, 2816]` | BF16 |
| `layers.N.mlp.down_proj` | `[2816, 2112]` | BF16 |
| 6 × `*layernorm.weight` + `layer_scalar` | `[2816]` / `[1]` | BF16 |
| `embed_tokens.weight` (tied head) | `[262144, 2816]` | BF16 |
| `norm.weight` (final) | `[2816]` | BF16 |

Six norms per layer now, not four: `input`, `post_attention`, `pre_feedforward`, `post_feedforward`,
plus `pre_feedforward_layernorm_2`, `post_feedforward_layernorm_1`, `post_feedforward_layernorm_2`.
That is the visible signature of the parallel MoE block.

## 2. The MoE math, pinned from the reference

Reference: `transformers` **v5.6.2** `src/transformers/models/gemma4/modeling_gemma4.py` — the version
the checkpoint's own `config.json` declares. `Gemma4TextRouter` / `Gemma4TextExperts` /
`Gemma4TextDecoderLayer` (lines ~1230–1400 of that file). Everything below is verbatim that code, not
inference.

**Router** (`router.proj.weight`, `router.scale`, `router.per_expert_scale`):

```text
h  = rmsnorm_weightless(x)          # x is the pre-FFN residual, flattened to [T, H]
h  = h * router.scale * H^-0.5      # H^-0.5 is a constant, not a learned value
s  = router.proj(h)                 # [T, 128], no bias
p  = softmax(s, dim=-1)
(w, idx) = topk(p, k=8, dim=-1)
w  = w / w.sum(-1, keepdim=True)    # renormalize BEFORE the per-expert scale
w  = w * router.per_expert_scale[idx]
```

**Experts** (fused tensors, per-expert gather):

```text
for each expert e hit by any token:
    gate, up = linear(x, gate_up_proj[e]).chunk(2, dim=-1)   # [0] = gate, [1] = up
    y = gelu_tanh(gate) * up
    y = linear(y, down_proj[e])
    out[idx_of(tokens on e)] += y * w[token, slot]
```

**Decoder layer** (MoE branch in bold, showing exactly where the three FFN norms land):

```text
residual = h
h = input_layernorm(h)
h,_ = self_attn(h, ...)
h = post_attention_layernorm(h)
h = residual + h

residual = h
h1 = mlp(pre_feedforward_layernorm(h))                       # dense MLP, I = 2112
if enable_moe_block:
    h1 = post_feedforward_layernorm_1(h1)                    # norm on the DENSE output
    _, w, idx = router(residual)                             # router reads the PRE-FFN residual
    h2 = pre_feedforward_layernorm_2(residual)               # experts read the PRE-FFN residual
    h2 = experts(h2, idx, w)
    h2 = post_feedforward_layernorm_2(h2)
    h = h1 + h2                                              # parallel combine, plain sum
h = post_feedforward_layernorm(h)                            # shared norm on the SUM
h = residual + h
h *= layer_scalar
```

Two details that are easy to get wrong and are worth a test each:

1. **The router and the experts read the pre-FFN residual**, not the normed MLP input. The dense MLP
   input (`pre_feedforward_layernorm`) is a different tensor from the experts' input
   (`pre_feedforward_layernorm_2`).
2. **Top-k weights are renormalized to sum to 1 before `per_expert_scale` is applied**, so the applied
   weights do not sum to 1 in general.

Everything else in the layer (embedding `√H` scale, sandwich norms, `layer_scalar`, K=V global
attention with compact KV, proportional RoPE, softcap 30) is the 31B math already implemented in
`docs/maintainer/gemma4-model.md` §§2–5.

## 2b. Sibling reference: gewell's 26B A4B

`github.com/leDissolution/gewell` (Apache-2.0, HEAD `0ee5bc3` "26A4 performance improvements") ships a
complete 26B A4B subsystem for sm120 — the same card class — so its decisions are a free cross-check
and, for speed, the thing to beat. `docs/maintainer/gemma4-model.md` §14 already cites it for the 31B.

| area | gewell | consequence for us |
|---|---|---|
| routed experts | CUTLASS `GroupProblemShape` block-scaled GEMM, `Sm120` + `OpClassBlockScaledTensorOp` + `KernelPtrArrayTmaWarpSpecializedCooperative`, tile **128×128×128**, cluster 1×1 (`moe_nvfp4.cuh`) | their expert GEMM is a vendor template. Our hand-written NVFP4 A4 route already measures **1246 TFLOP/s against 1092 for that lineage** on this card (cuBLAS 1116), so a faster grouped kernel is credible — but our *grouped* shape is unproven and must be measured, not assumed |
| thin decode | a separate hand-written tiled path at **16×64 tiles**, dequantizing NVFP4 weights to half in-kernel for W4A16 | matches TileGym's autotuner finding (thin-M wants BLOCK_M 16 at occupancy 1; clusters were 1.76× slower there). Its BF16 path is `GemmGrouped` fed by a device-side `prepare_grouped` kernel that fills 256 per-group pointer sets |
| activation policy | `--nvfp4-activation-policy prefill` → **W4A16 decode, W4A4 prefill**; `always` → W4A4; 26B only | the same split we already have (A16 below 8 tokens, A4 above). For experts the decode shape is M≈1 per expert, so an A16 streaming route is only worth anything if the expert layout is expert-major |
| formats | per-projection mask (`bf16` / `fp8_w8a8` / `nvfp4_w4a4`) over attention, shared MLP and routed MLP, with **router, norms, embeddings and the tied head left BF16** | agrees with §4's precision plan; keeping the router and norms BF16 is a sound default to adopt **and test**, since our shipped 31B head is FP8-row |
| routing | device-side only — `dispatch()` never downloads routing data — deterministic top-k with tie-break on the **lower expert index**, renormalize then `per_expert_scale` | match both: device-side routing, and the tie-break. A different tie-break changes greedy output on adversarial rows and fails an oracle |
| combine | the 8 expert outputs are summed **in BF16** (`Reduction`), matching the reference's `index_add_` in the output dtype | the reference accumulates in BF16. FP32 accumulation is a real quality improvement we can offer, but it *deviates* from the reference and must be qualified by KLD, not asserted |
| workspace | one workspace shared by all layers, sized for `8·rows` assignments; no per-round allocation or host round-trip | matches our Program-owned workspace doctrine; adopt "no allocation and no host round-trip inside a round" as a design constraint for the MoE route |
| conversion | `tools/convert.py --architecture gemma4_26b_a4b` accepts Google's stacked BF16 experts **and NVIDIA's packed NVFP4** 26B checkpoint; native NVFP4 pads physical output rows while preserving logical width | two things to copy: accept a packed-NVFP4 source, and pad-but-preserve on the 704-wide expert MLP (704 = 11·64, not a power of two) |
| MTP | `mtp_target.cu`, `mtp_staging.cu`, `mtp_verifier` test for the 26B | the drafter is real work for the 26B; our adaptive {2,3,4,7} ladder is already built and measured for the 31B |

**It publishes no 26B numbers.** Its README and `docs/benchmark.md` contain only the benchmarking tool,
not results. So "faster" cannot be argued from prose — build gewell, run it on this card, and compare
measured saturated decodes. That is P0's last bullet.

## 3. What transfers, what is new

| area | disposition |
|---|---|
| attention Ops (sliding causal + compact global), RoPE, sandwich norms, `layer_scalar`, softcap, embedding scale | **transfers**, geometry from config |
| `src/models/gemma4/` module (config, load, weights, forward, program, cache) | transfers, extended |
| frontend, tokenizer, chat template, protocol, sampler, graphs, context cache | transfers; verify the tokenizer is byte-identical to the 31B's |
| MoE block (router + 128 experts + parallel combine + 3 FFN norms) | **new math** (§2) |
| fused 3-D expert tensors in conversion and in the artifact | **new roles** |
| decoder/prefill MoE execution (gather/scatter, top-8 streaming) | **new op** |
| assistant drafter (`assist/`, hidden 1024, 4 KV-shared layers, backbone 2816) | **verify** against the shipped 31B assistant path |

## 4. Phases

### P0 — Ground truth before any code
- Cross-check §2 against llama.cpp `src/models/gemma4.cpp` and gewell's 26B path (the §14 method),
  as was done for the 31B. Record the formulas in `gemma4-model.md` as the 26B section.
- Settled from shard 1's header: `experts.gate_up_proj` is `[128, 1408, 2816]` — `[E, 2·I_moe, H]` with
  gate first, exactly as the reference predicted. Layer 0 also confirms `q_proj [4096, 2816]`, `k_proj`
  and `v_proj` `[2048, 2816]` (the sliding geometry, so `v_proj` is present off the global layers as in
  §1), `router.scale [2816]`, `router.per_expert_scale [128]`, `layer_scalar [1]`, and the tied
  `embed_tokens [262144, 2816]`.
- **DECIDE / check**: the 26B assistant declares `transformers_version: 5.10.0.dev0` and carries
  `num_centroids: 2048`, `centroid_intermediate_top_k: 32`, `use_ordered_embeddings: false`,
  `backbone_hidden_size: 2816`. `v5.6.2`'s `modeling_gemma4.py` has no assistant class at all, so the
  drafter's reference is a different source. Diff the shipped 31B assistant's `config.json` against
  this one: if the 31B also carries the centroid fields, our existing path already ignores them
  correctly and the 26B needs only the retargeted `backbone_hidden_size`; if not, the centroid path is
  unimplemented and is its own slice.
- Evidence: the reference source recorded in `gemma4-model.md`, plus the two config diffs.
- **Head-to-head baseline**: build gewell at `0ee5bc3` and run its 26B on this card. Record single-lane
  greedy tok/s on **saturated** completions, prefill tok/s at 8K and 32K, per-round accounting
  (tokens/round and ms/round, which separates acceptance from round cost), and its MTP acceptance if it
  runs — with the thinking mode stated. Comparing headline tok/s across engines without those controls
  is dominated by completion length and reasoning mode, not by the engine. This run defines the target
  the rest of the plan has to beat.

### P1 — Config, roles, conversion
- `src/models/gemma4/config.{h,cpp}`: parse `enable_moe_block`, `num_experts`, `top_k_experts`,
  `moe_intermediate_size`; keep the dense path bit-identical when the flag is off (the 31B must not
  move). Tests for both configurations from the two real config files.
- `weights.h` / `load.cpp`: roles for `router.proj`, `router.scale`, `router.per_expert_scale`,
  `experts.gate_up`, `experts.down`, and the three extra norms.
- Converter: name vocabulary for the MoE roles and a `gemma4_26b_a4b_base` recipe modelled on
  `gemma4_31b_base` (`tools/convert/official_recipes.py:513`). Precision plan: experts + dense MLP
  NVFP4, attention + embedding FP8 rows (`AllowA8`), router/norms/scales BF16/FP32. Where the 31B
  recipe already skips `/moe/router` and `/moe/shared_score`, extend rather than duplicate.
- Deliverable: an artifact that loads and reports its layout, with a conversion report.
- Evidence: conversion report + load smoke test; `tests/convert/` extended.

### P2 — The MoE route
- **Landed**: the router and its selection are `ops::gemma_moe_router` — contract in
  `include/ninfer/ops/gemma_moe_router.h`, kernel/launcher/wrapper in `src/ops/{kernel,launcher,wrapper}/`,
  FP64 oracle in `tests/ops/gemma_moe_router_ref.h` and three cases in
  `tests/ops/test_gemma_moe_router.cpp` (the stored geometry, an exact tie at the boundary, and the
  renormalize-then-scale order). What remains here is the expert apply and the parallel combine.
- **Landed**: the expert apply is `ops::gemma_moe_experts` over the two expert-major banks, with
  `tests/ops/gemma_moe_experts_ref.h` and a case that runs it at the *stored* width over a bank whose
  padding rows and columns are zero and requires the oracle's logical-width result — which is what
  makes the padding arithmetically invisible rather than merely believed to be. Both routes are
  BF16-bank routes: the artifact's banks are NVFP4, so a block-scaled bank profile is the next route
  here (a BF16-expert recipe is what exercises this pair end to end today), and the layer's combine
  with the dense branch is the remaining piece of P2.
- **Landed: the block-scaled (NVFP4) bank profile.** The kernel takes a bank *reader* policy — a BF16
  bank hands back its stored element, an NVFP4 bank decodes `e2m1(codes) · e4m3(16-element block
  scale) / divisor(row)` — so one kernel body covers both formats and the launcher instantiates the
  pair. The divisor is the per-source one the stacked layout carries, read exactly as the offload MoE
  route reads its own stacked planes (`value * __frcp_rn(divisors[row / divisor_rows])`). The test
  reads the oracle through `tests/ops/quantized_weight.h`'s own FP64 decode rather than the Op's
  decoder, and the fixture now writes *distinct* divisors per source, so a reader that ignored the
  array cannot pass. `nvfp4_scale_byte` gained a runtime-tile-count form (`nvfp4_scale_byte_runtime`)
  that the template now delegates to, because a route over several geometries cannot use a
  compile-time tile count; the six NVFP4 suites pass unchanged after it.
- **The block-scaled bank profile is on the critical path, not a performance nicety.** A BF16-expert
  variant — how the 35B's own `qwen3_8_flash_next_bf16` recipe exercises its pair — is not available
  to this model: 24.9 B expert parameters are 50 GB at two bytes against a 32 GB card, and even
  8-bit experts would leave no room for KV. So the artifact's NVFP4 banks are what the Op must read.
- Its enablers, located rather than searched for: the decode is `nvfp4_codec.cuh`
  (`decode_nvfp4_e2m1x2`, `decode_nvfp4_e4m3`, `nvfp4_scale_byte<ScaleTilesPerRow>`) with the per-row
  access in `nvfp4_gemv.cuh`; a bank is a **stack of per-expert planes** (`require_nvfp4`: "every
  expert matrix was quantised on its own, so each of these planes is a stack"); the expert shapes are
  now registered in `nvfp4_geometry.h` (`Nvfp4N768K2816`, `Nvfp4N2816K768`, which the converted
  artifact already contains); and `tests/ops/quantized_weight.h` builds NVFP4 fixtures at the same
  N/128 and K/64 constraints.
- **DECIDE**: a Gemma-specific MoE op, or generalizing `ops/wrapper/sparse_moe.cpp`. That op is
  specialized to Qwen4Exp's geometry by `constexpr` (`kHidden = 2048`, `kExperts = 256`,
  `kExpertRows = 1024`, `kIntermediate = 512`, a shared-expert bias row), and Gemma's layer is a
  different shape *and* a different computation (parallel dense MLP, `per_expert_scale`, renormalized
  weights). Recommendation: a Gemma-specific op, following §5.1's precedent that Gemma keeps its own
  attention Ops rather than bending a shared route.
- **Reference: the in-project Qwen3.6-35B-A3B MoE** (`ops::sparse_moe`, 256 experts, top-8). Its shape
  decides the rest of P2 rather than leaving it to taste:
  - **One `Weight` per bank, not one per expert.** `SparseMoeWeights` is `routed_gate_up` +
    `routed_down`, and the wrapper validates them as one matrix each
    (`kRoutedGateRows = kExperts * kExpertRows`, `kRoutedDownRows = kExperts * kHidden`). The artifact
    side agrees: the binding documentation says the experts "keep the per-layer gate/up and down
    parents, so an expert is a row range of its bank". The Gemma banks are therefore
    `gate_up [128 * 1536, 2816]` (per expert: 768 gate rows then 768 up rows, which is the source's
    own `[E, 2I, H]` order preserved) and `down [128 * 2816, 768]`.
  - **`kExpertRows = 2 * kIntermediate`**: gate and up are interleaved per expert inside one bank.
    Gemma's checkpoint already stores them that way, so the grouping needs no reordering.
  - **The padded width is transparent.** The op works over the stored 768 and the zero rows and
    columns are inert, so nothing downstream needs to know the checkpoint's 704.
  - **Routes per regime**, each with its own plan: `decode/`, `prefill/`, `small_t/`. The Gemma route
    starts with one correct route and splits when the performance phase asks for it.
  - **Routing belongs to the Op's workspace**, not to the caller: `sparse_moe` keeps ids and alpha in
    its own workspace tensors. `gemma_moe_router` currently exposes them as outputs, which is what the
    qualification test wants; the fused layer Op may hold them internally instead.
  - **The tie-break matches**: `sparse_moe_ranked_better` is
    `a.value > b.value || (a.value == b.value && a.id < b.id)`, the same total order
    `gemma_moe_router` implements and tests.
- Shape it around the payoff: per decode token, stream **only the top-8 experts** (8 × 5.95 M ≈ 47.6 M
  params/layer), which is the whole reason this model exists. The expert weight layout must therefore
  be expert-major so a round reads 8 contiguous expert slabs.
- Both routes need real shapes: prefill (large T, every expert hit) and decode (T = 1..8).
- Evidence: FP64 oracle per op per `docs/maintainer/op-development.md` — routing (selected indices and
  weights exactly match the reference), expert gate/up/down math, the combine, and the two tricky
  points in §2 each covered by a case.

### P3 — Execution and end-to-end correctness
- **Landed**: `feed_forward_sublayer` in `src/models/gemma4/forward.cpp` now carries the layer's
  feed-forward sublayer for both layer kinds. A dense model takes the branch it always took; a model
  whose config enables the MoE block runs the router and the experts beside it, sums the two branches
  and applies the shared post-norm. `prepare_gemma_moe_expert_weights` unites each bank into one
  contiguous `Weight` through the same `single(...)` the 35B's `prepare_sparse_moe_weights` uses, so
  an expert is a row range of its bank and nothing is unpacked per expert.
- **Landed**: the MoE branch adds no large scratch. It reuses the dense branch's `[H,T]` buffers in
  place — `normed`, `down` and the attention's `projected`, each after its last reader — so the only
  new storage is the router's `K`-per-token output, which `layer_workspace_bytes` counts.
- **Correction the bridge forced**: the 26B recipe had granted the routed experts `AllowA4` with no
  activation divisor — a permission no route honours, since the expert apply decodes the stored codes
  itself and computes in the represented activations. They are `A16Only` now, the NVFP4 permission
  guard skips routed-expert banks for the same reason, and the artifact predating the change was
  rebuilt.
- **Finding the first run produced**: every quantized linear route resolves its geometry from a fixed
  registry of hand-tuned shape files, and the FP8, NVFP4, Q4–Q6 and BF16 registries hold only the
  Qwen and 31B geometries, so hidden 2816 failed at the first attention projection. The row-split int8
  registry is the only one with a generic selector, so the 26B's ten `linear` shapes are registered
  there and the recipe gives the attention, the dense MLP, the token table and the head Q8. The routed
  experts keep NVFP4 because no linear route sees them. Registering the two dense-MLP shapes for NVFP4
  is what would restore one layout across both branches, and that is P5's precision comparison rather
  than a bring-up step.
- **Evidence so far**: P2's op oracles; the 31B's `ninfer_gemma4_forward_test` and
  `ninfer_gemma4_layer_test` passing against the refactored sublayer, so the dense execution path is
  unchanged; the ten new Q8 geometries passing `ninfer_linear_q8_a16_test`'s T sweep against the FP64
  oracle; and the 26B itself loading (12118 parameters over 30 layers, 5 global) and generating the
  correct greedy continuation.
- **Note on the 31B-shaped engine tests**: `ninfer_gemma4_model_test` and `ninfer_gemma4_forward_test`
  assert 31B facts (60 layers, 833 bound weights, hidden 5376, layer 59) and fail when pointed at the
  26B artifact. They are artifact-gated and nothing registers the 26B with them, so CI is unaffected;
  giving the 26B its own artifact-derived versions is a follow-up, the same way `test_load.cpp` was
  made to derive from the artifact's config in P1.
- **Remaining**: logits agreement against the HF reference at the §11 protocol, then greedy output
  equality on the standard prompt set, then concurrency 1..8 and the KV/state checks the 31B already
  has.

### P4 — Speculation
- Support `assist/` after P0's config diff; reuse the shipped 31B assistant machinery (target-cache
  binding: sliding ring for the last sliding layer, compact global rows for the last global layer, the
  shared-vs-context accept policy).
- Evidence: acceptance rate and tok/s with and without the drafter, in the shape of §12a's table.

### P5 — Precision and performance qualification
- KLD against the BF16 reference on the standard corpus at two contexts, against the accepted drift
  budget already recorded for Gemma (0.419). A/B the expert encoding between `nvfp4_mse` and
  `nvfp4_local_hessian`: the earlier ModelOpt work measured local-Hessian as 3–6% better on A4-route
  output error at conversion time only, and this is the first from-BF16 NVFP4 conversion since.
- Measure decode (expect roughly 2 ms/token against the 31B's ~19 ms: 3.6 B active parameters is about
  1.9 GB streamed per token at 4.5 bits) and prefill, then write both into the 26B section of
  `gemma4-model.md` and, if the product scope allows, `docs/performance.md`.
- **QAT caveat**: the source is QAT-trained but we do not know which 4-bit grid it was trained against.
  If our G16/E4M3 NVFP4 grid harvests nothing measurable, the fallback is FP8 for the experts. Measure
  before claiming the QAT benefit.
- **First measurement (this pass)**: `ninfer-perplexity --text <40 KB> --context 1024 --stride 512
  --kv-dtype bf16` scores the 26B at 75.4 tok/s against the 31B's 473.9 over the same 9,104 scored
  tokens, text and protocol; both are correctness-first numbers, not tuned ones. The expert apply is
  one CTA per token that reads each of its top-8 experts' three banks with scalar loads, so a
  1024-token chunk moves roughly 590 GB of expert weights in 13.6 s — 43 GB/s, about 2.4% of this
  card's bandwidth. The lever list below begins with the expert-major layout; the measurement says the
  gather comes first, because a chunk's 8 × 1024 slots address only 128 matrices, so the same bytes
  are re-read tens of times before any layout question is even reachable.
- **Absolute perplexity is not a defect signal on `profiles/kld/quick-concat.txt`**: that corpus was
  chosen for quantization sensitivity, not for a low number. On the same 40 KB and protocol the 26B
  scores 233 against the 31B's 1414, and llama.cpp's own 26B (Q5_K_S, its own chunking) scores 27,641
  on it. The 26B is therefore in the right range by the coarsest available comparison; the HF logits
  agreement of P3's remaining list is still what establishes correctness.
- **A negative result worth keeping**: rewriting the expert apply's inner loops so that a warp walks one
  output row together — 32 lanes over consecutive columns, one warp-shuffle reduction per row — made the
  same corpus *slower*: 41.7 tok/s against 75.4, with the perplexity unmoved (229.6 against 233.1, which
  is the reduction-order difference alone, so the rewrite was numerically clean). The 43 GB/s figure is
  real, but attributing it to uncoalesced sectors was wrong: every row a warp issues is consumed by the
  same pass, so no sector is wasted end to end, and the pattern costs load/store-unit throughput rather
  than DRAM traffic. The kernel is decode-bound instead: each NVFP4 element pays a byte load, a nibble
  select, a 4-bit conversion, a block-scale byte, an E4M3 conversion and a reciprocal multiply. The
  levers that follow are (1) the gather inversion, which cuts the element count by the average expert hit
  count, since one matrix is currently read once per token that selects it, and (2) a vectorized decode —
  `uint4` code loads, a hoisted per-row divisor, a register LUT for E2M1. Reverted; the tree is back at
  the measured 75.4 tok/s, and no performance claim is made until a decode measurement exists.
- **Claim discipline for "faster"**: only against the P0 baseline, on this card, the same checkpoint,
  greedy, thinking mode stated, saturated completions, at C=1..8, with per-round accounting. A dense
  NVFP4 advantage (1246 against 1092 TFLOP/s) does not transfer to the grouped thin-M shape by itself;
  the grouped route has to be measured. The kernel levers to try, in order: an expert-major weight
  layout so a decode round streams only the top-8 experts' bytes; thin-M tiles rather than cooperative
  128×128 (gewell's own thin path is 16×64, and TileGym's tuner independently chose BLOCK_M 16 at
  occupancy 1); FP32 accumulation of the 8-way expert sum where the reference and gewell use BF16, kept
  only if KLD and the oracle both accept it as a deviation.

## 5. Risks

| risk | why it matters | mitigation |
|---|---|---|
| MoE math misread | a wrong combine or norm order is invisible in spot checks | §2 is taken verbatim from the declared reference and cross-checked in P0 before code |
| Assistant reference missing locally | the drafter's centroid/ordered-embedding path may be unimplemented | P0 config diff decides the size of the slice |
| Experts do not fit with long context across 8 lanes | 22.8 B params at 4.5 bits is ≈12.1 GB of experts; the artifact is ≈14 GB total | initially device-resident (no offload MoE); the existing host-KV tier is the fallback, and its cost at spill is now measured |
| Everything streams per token in prefill | the expert set is 12 GB, so a large batch is what makes prefill efficient | prefill amortizes across tokens; measure, and if a short prefill is slow the fix is the standard one (chunking) |
| The grouped/thin-M expert shape has no baseline for us | our 1246-against-1092 TFLOP/s result is on *dense* shapes; the grouped path could go the other way, and gewell's CUTLASS grouped route is the thing to beat | P2's oracle establishes correctness, P5 measures speed against the P0 baseline before any claim |
| Source download incomplete | no value-level work is possible | index and headers were enough to plan; P1 starts on config only, which needs no weights |

## 6. First commits

1. `docs(gemma4)`: the 26B section in `gemma4-model.md` (§1 shapes, §2 math) plus this plan.
2. `feat(gemma4)`: MoE config fields, parser, and tests from both real config files.
3. `feat(gemma4)`: weight roles, loader bindings, converter naming, and the recipe skeleton.
4. `test(ops)`: the MoE route's FP64 oracle cases.
