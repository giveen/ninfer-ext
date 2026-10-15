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
- **Resolved (P4)**: the two assistant configs are field-for-field the same outside the geometry a
  drafter inherits from its target, and both carry all three centroid fields with
  `use_ordered_embeddings: false`. Our path consumes none of them, so nothing was missing and no
  centroid slice is needed. Diff recorded in `gemma4-model.md` §16.5.
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
- **Note on the 31B-shaped engine tests** (resolved): `ninfer_gemma4_model_test` and `ninfer_gemma4_forward_test` used to
  assert 31B facts (60 layers, 833 bound weights, hidden 5376, layer 59) and fail when pointed at the
  26B artifact. Every expectation now comes from the artifact's parsed config — the layer count, hidden size, the per-layer attention geometry that
  gives the query/key/value slice shapes, the sliding/global split, and the last global layer index —
  and the bound-weight count from the structure in `tests/models/gemma4/expected_weights.h`, which `test_load.cpp` now counts with too ( the shared oracle that P1 gave the load test). Both files also
  indexed `weights.text.layers[59]` unchecked, which on the 26B read past the end of a 30-entry layer
  vector and turned garbage bytes into a `WeightId`; that read is now the last global layer. Both
  artifacts pass (`ctest -R 'gemma4_(model|forward)_test'`: 30 layers / 12118 parameters for the 26B
  QAT artifact, 60 / 833 for the 31B).
- **Remaining**: logits agreement against the HF reference at the §11 protocol, then greedy output
  equality on the standard prompt set, then concurrency 1..8 and the KV/state checks the 31B already
  has.

### P4 — Speculation
- **Landed.** `out/gemma4_26b_a4b_qat.ninfer` is built with `--components text,mtp --source
  mtp=/mnt/storage/models/gemma/26b-full/assist`. The shipped 31B assistant machinery carried it with
  no runtime change: the operator's config is the 31B's with the target-inherited geometry retargeted
  (backbone 2816, 16 heads, 8 KV, 2 global KV), and `draft_config` already reads each of those from
  the target. The only code the drafter needed is four Q6 registry entries, because its attention
  widths are the target's head count and so nothing the 31B's drafter registered fits. P0's centroid
  check is resolved: both assistant configs carry `num_centroids: 2048`,
  `centroid_intermediate_top_k: 32` and `use_ordered_embeddings: false`, so the ordered head is off in
  both and the centroid path stays unimplemented.
- **Evidence** (greedy, 384 output tokens, one request, adaptive `--draft-tokens 7`, `ninfer` CLI
  decode speed; full table in `gemma4-model.md` §16.5): 341.2 to 489.0 tok/s on the mean of five
  prompts, +13.6% (story) to +69.6% (math), acceptance 53.5-88.9%. A round costs 2.9 ms (the target's
  flat verify pass) plus 1.2 ms a draft step, from fixed-draft runs at K = 1, 3, 7; the drafter's
  0.4 GB a step imply 0.22 ms, so its step is 5x off the transfer bound and is the next lever.
  `ninfer_gemma4_mtp_test` and `ninfer_gemma4_draft_test` pass against this artifact, and the added
  component leaves the target bit-identical (mean NLL `5.398730771493524`).
- **Not done, and it needs a decision**: the drafter's step cost (its selectors are noted untuned at
  hidden 1024), and the serving-side check (`--spec mtp` through `apps/serve`, plus the concurrency
  1..8 rounds the 31B has).

### P4b — DFlash (tried and reverted)

Kept as a record only: the DFlash component is not in the tree. The work tried the official
block-diffusion drafter `z-lab/gemma-4-26B-A4B-it-DFlash` beside the MTP assistant, because it
denoises a whole 16-token block in one pass, so its draft cost barely grows with block length, where
the MTP drafter is sequential (a round costs 2.9 ms of verify plus 1.2 ms a draft step, P4 above).
The port was written in full and its mechanisms verified, then reverted: the drafts it produces
against this artifact's target are near-worthless, and the finding rather than the code is what is
kept.

**The port was complete and mechanically verified.** A converter component and its config parse --
the drafter's own config is a Qwen3 one, five layers at the target's width, `head_dim` 128 with 32
query and 8 key/value heads and a 2048 sliding window, built by `--components text,mtp,dflash` with
`--source dflash=<the drafter>` -- and weight binding of **68 logical parameters over 43 objects**.
The **six taps** the drafter reads from the target's residual stream (layers `[1, 6, 11, 17, 22,
27]` through a `[2816, 16896]` feature projection) are exact: **8,650,752 elements**, six taps of
512 columns, equal the same layers run one at a time. The context projection agrees with an **FP64
oracle that decodes the artifact's own stored weight** from its row-split planes: **relative L2
0.0033**, which is bf16 rounding. The cyclic context cache and its slot arithmetic were verified,
and the proposal ran end to end -- a prefill's taps, the append and a 16-column block all produced
well-formed tokens.

**The block does read its context.** At a fixed prefix, 511 context rows carrying data give draft
49321, while the same prefix with only its last 32 rows carrying data (the other 479 left zero, the
planes being zeroed when they are allocated) gives 4077. The drafts differ, so the context reaches
the block.

**The failure is quality, not mechanism.** The first draft matched the token the target itself names
at only **1 of 5** prefix lengths. Ranking that draft inside the target's own next-token
distribution puts it at **1, 19, 49, 143 and 292 out of 262,144** -- correct inputs, a slightly
shifted distribution, which is what a mismatched conditioning target looks like; a genuinely wrong
port scatters the rank instead.

**The cause is the target pairing.** This artifact's target is the **QAT** checkpoint
(`/mnt/storage/models/gemma/26b-full`, Google's `gemma-4-qat-q4-0` collection), and the drafter was
trained against `google/gemma-4-26B-A4B-it`. Its whole context is that target's residual stream
through the `fc` projector, so a target it was not trained against puts that context outside the
projector's distribution and degrades the drafts without breaking any mechanism -- exactly what is
measured. The drafter's own model page reports the same symptom, unanswered, with no QAT DFlash
released: `z-lab/gemma-4-26B-A4B-it-DFlash` discussion 8, "If i use QAT model and existing DFlash,
i got 0-1 accepted tokens".

**Decision: dropped and reverted.** The MTP assistant remains the shipping drafter at **+43.3%**
(341.2 to 489.0 tok/s; P4 above and `gemma4-model.md` §16.5). Confirming the pairing would need the
same draft path against the non-QAT `google/gemma-4-26B-A4B-it`, which is not on this box.

**The method correction is worth keeping.** An earlier conclusion of this work -- that "the block
reads no context" -- was wrong twice over. The comparison it rested on also moved the anchor token
(it compared a 64-token prefix against a 512-token one), and its two "envelopes" were never an A/B,
because `resolve_plan` walks `min(envelope.max_context, window - 1)` rows, so `{0, 2048}` and
`{511, 511}` both walk the same 511. A control must vary only the variable under test.

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
  `uint4` code loads, a hoisted per-row divisor, a register LUT for E2M1. Reverted at the time, and the
  divisor hoist was later measured to buy nothing at all (see the three-way table below); the speed came
  from independent loads instead, which is the next bullet.
- **The accumulation order is worth 0.04 nats; the unroll is worth 50%.** Three formulations of the same
  expert apply, each measured twice, interleaved in one session on the same 40 KB corpus:

  | expert apply | mean NLL | PPL | tok/s |
  |---|---|---|---|
  | serial, divisor per element (baseline) | 5.451330 | 233.068 | 75.8 / 76.7 |
  | four partial sums, divisor per element (kept) | 5.492602 | 242.888 | 114.1 / 115.3 |
  | four partial sums, divisor hoisted per row | 5.503557 | 245.564 | 115.5 / 115.4 |

  Every row is bit-identical across its two runs, so those differences are the code and not run noise.
  The unroll carries the whole 50% and the hoist carries none of it, so the hoist was dropped: the kept
  variant is the middle row, which is also the one whose reader contract is unchanged. Its cost is
  +0.041 nats, from reordered FP32 accumulation alone — the same mathematics, a different summation
  tree. A fourth formulation moved the same metric the other way (5.436548 for the reverted
  warp-cooperative rewrite), so this is a noise floor of roughly 0.04 nats rather than a bias, and it is
  10% of the 0.419 drift budget Gemma is held to. Consequence: a KLD or perplexity comparison must fix
  the accumulation order across the variants it compares, and the P2 decision that accumulation is a
  private execution choice now has a number attached.
- **The gather inversion's design (built, measured, rejected — see below)**: the element count is the
  target, not the access pattern. Per
  layer and 1024-token chunk the apply reads 8,192 slot-hits × 6.49 M elements = 53.2 G elements, while
  those slots address only 128 distinct expert matrices — 0.83 G elements, a 64× reduction. Inverting the
  loop also makes the reuse *cacheable*: one expert's three matrices are about 3.6 MB in NVFP4, so an
  expert that a block works through stays resident in L2 (which is far larger than that on this card)
  while its hits are served, which is where the multiple comes from rather than from DRAM traffic.
  Shape: (1) a compaction pass writing, per expert, its list of (token, weight) slots — deliberately a
  fixed per-expert region scanned in a fixed (rank, token) order rather than a global atomic cursor,
  because a nondeterministic order would poison exactly the comparisons P5 depends on; (2) a per-expert
  block computing gate/up, GeGLU and down over its hits and accumulating into an FP32 `[H,T]` buffer,
  which the op's contract must now declare as workspace along with a workspace-size function; (3) a
  convert pass to the BF16 output, with the accumulator cleared per call. The wrapper and the model's
  scratch follow (`layer_workspace_bytes`), as do the oracle at the model's shapes and an interleaved
  A/B against the kept variant's 114.1 tok/s and 5.492602 mean NLL.
- **The gather inversion, implemented and rejected**: the full route was built — a fixed slot region per
  expert written by one thread per expert in a fixed (rank, token) order, one block per (expert, hit
  tile) computing gate/up, GeGLU and down over the hits it owns, an FP32 [H,T] accumulator, a narrowing
  pass, and the workspace declared in the Op's contract and counted in the layer bound. It is correct:
  the FP64 oracles pass every case. It is worse on both axes.
  - **Nondeterministic.** Two runs of the same command gave mean NLL 5.464852 and 5.520692, where every
    earlier formulation was bit-identical run to run. The compaction order is fixed, but the accumulator
    is filled by `atomicAdd` from blocks whose relative order the scheduler picks, so the sum's
    association varies — the one property P5's comparisons cannot tolerate, and the reason the design
    note above calls for a fixed order.
  - **Slower.** 51.9 and 52.2 tok/s against the kept variant's 114.1 and 115.3, although it reads each
    expert's matrices about once per pass instead of once per hit. The element-count argument was
    necessary but not sufficient: with eight blocks per expert, the blocks resident on one SM are
    working on different experts, so the reuse the inversion exists to create is spread across the
    machine instead of captured per SM. Keeping an expert on an SM and giving it enough warps to hide
    the bank latency is the schedule, and eight tiles per expert does the opposite.
  Reverted. A future attempt should start from the schedule and from a deterministic reduction
  (per-expert partials summed in a fixed order, no atomics), not from the element count.
- **The streamed expert route, the router rewrite and the 2816 norm route (landed).** `ninfer_bench`
  on this card measured decode at **2.05 tok/s** before this pass, not the perplexity-derived "4×
  behind": the per-token expert kernel runs a decode token on one SM. Three changes, each from a
  measurement:
  - the NVFP4 expert pair at the 26B's geometry takes the 35B decode's two-stage split (a CTA per
    intermediate row and token with a warp per rank for gate/up + GeGLU into an FP32 workspace,
    then a CTA per pair of hidden rows summing the ranks in ascending order). The codec and warp
    dot products moved from the 35B's decode file into `sparse_moe_expert_dot.cuh` unchanged;
  - the router projection read its 720 KB weight one uncoalesced thread per expert in one CTA per
    token, 149 µs a layer and 48% of decode time; it is now a warp per expert over (4 experts × 4
    tokens) CTAs into an FP32 [E,T] workspace, then a warp per token for the selection;
  - RMSNorm at 2816 fell to the generic kernel (5.6 µs a row, seven a layer) and now has the
    fixed-width route the 31B's 5376 has.

  | `ninfer_bench`, bf16 KV | before | after |
  |---|---|---|
  | tg128 (tok/s) | 2.05 | 215.9 |
  | pp512 (tok/s) | 189 | 2,040 |
  | perplexity score rate, 40 KB, ctx 1024/512 (tok/s) | 111.3 | 506 |

  Both MoE Ops gained a workspace in their contracts, shared by the layer (router logits, then
  expert products). Mean NLL on that 40 KB slice (9,319 tokens): 5.500364 before, 5.471673 after,
  bit-identical across two runs; the 0.029 difference is inside the 0.04 accumulation-order floor
  above. The 35B is unmoved (tg128 404.6 vs 405.4; pp512 22,111 ± 421 against 22,238 ± 602 with
  the change stashed). Prefill still runs the streamed decode route per token; the grouped route
  is next.
- **The QAT grid is int4 group-32, not NVFP4.** The checkpoint's README names its QAT target as
  Q4_0, and the weights agree: in every sampled tensor (experts, dense MLP, attention, embedding)
  every group of 32 holds at most 16 distinct values, and 94–99.5% of groups are exactly `d·k`
  with `k` in [-8,7]. The scale is learned, not Q4_0's amax/8 rule: expert and MLP groups reach
  their amax at 5–7 steps. NVFP4's E2M1 grid has no ±5 or ±7, so it re-quantizes these weights:
  a best-case per-block-MSE NVFP4 encoding has 7.4–7.7% relative RMS weight error on every tensor,
  against 0.19% (the BF16 rounding of the checkpoint itself) for int4 group-32 with the recovered
  scale stored as FP16, at the same 4.5 bits a weight. No recovered scale is near FP16's subnormal
  range (smallest 8.9e-4). **Decided**: the fast routes are built for an exact int4 group-32 codec,
  experts first (streamed decode, then a grouped W4A16 prefill that dequantizes to the
  checkpoint's own BF16 values), then the dense linears and the head, which at Q8 are about 2.4 GB
  of the ~3.3 GB a decode token reads.
- **The QAT experts and the grouped prefill route (landed).** Recipe `gemma4_26b_a4b_qat` stores the
  routed experts as `q4_g32_fp16_rows` (row_group_v1) through the new converter method
  `q4_rows_qat`, which recovers each group's learned step (the coarsest `max|x| / q` that puts every
  value on an integer multiple, falling back to the least-squares step for a group on no grid)
  instead of `q4_rows_maxabs`'s `max|x| / 7`; a converted value is the checkpoint's BF16 word to
  within one ulp. The conversion of `out/gemma4_26b_a4b_qat.ninfer` takes 4 minutes. The expert Op
  reads the format on all three routes: the per-token reader, the streamed route (a `Q4RowsCodec`
  beside the NVFP4 one, below 16 tokens) and a new grouped route from 16 tokens up: a
  deterministic per-expert placement (count, plan, place, no atomics), a BF16 m16n8k16 gate/up
  GEMM over each expert's gathered tokens with the GeGLU in its epilogue, the same GEMM for down,
  and a rank-ordered FP32 combine, in slices of at most 1024 tokens. That route decodes weights to
  BF16 and rounds the GeGLU product and each expert's output to BF16, where the checkpoint's own
  reference rounds; the oracle models those three boundaries, and the grouped cases (T = 16, 37,
  300, 1030 at the real H/I/K) bound the error by `1e-3 + 1e-2·|ref| + 2^-7·Σ|w·y|`, the last term
  being a one-ulp flip at the expert-output boundary where outputs cancel. Worst case 0.44 of that
  bound.

  | `ninfer_bench`, bf16 KV | NVFP4 experts | QAT int4 g32 experts |
  |---|---|---|
  | pp512 (tok/s) | 2,040 | 8,246 |
  | pp2048 (tok/s) | — | 8,030 |
  | tg128 (tok/s) | 215.9 | 219.4 |

  Mean NLL on the 40 KB slice: 5.425723 (QAT) against 5.471673 (NVFP4), each bit-identical across
  two runs. The 0.046 difference is just outside the 0.04 accumulation-order floor, so it points the
  expected way but is not a quality measurement; KLD against the BF16 reference is. Greedy output on
  a short prompt is coherent for both. The NVFP4 bring-up recipe stays until that comparison.
- **Dense projections and head on the QAT grid (landed).** `gemma4_26b_a4b_qat` now stores attention,
  the dense MLP and the head as `q4_g32_fp16_rows` too (the token table stays Q8: the embedding
  gather reads row-split int8); the artifact is 16.2 GB against 17.4. The linear Op reads the format
  with no shape registry: up to 8 tokens a warp per four rows streams 16-byte groups, above that the
  64×64 tile the grouped expert route uses (now shared in `ops/common/q4_rows_mma.cuh`). Up to 8
  tokens the router splits each expert's row over one CTA (13.9 → 2.1 µs a layer at decode).

  | QAT artifact, `ninfer_bench` | experts only | + projections and head | + GEMV and router |
  |---|---|---|---|
  | tg128 (tok/s) | 219.4 | 264.5 | 309.7 |
  | pp512 (tok/s) | 8,246 | 8,277 | 8,246 |

  Mean NLL on the 40 KB slice: 5.398731, bit-identical twice (experts only 5.425723, NVFP4 experts
  5.471673). A rejected GEMV variant (eight rows a CTA, items spread over threads, shared-memory
  reduction) was slower on every shape (tg128 241) and is gone. What decode time remains, per token
  from an nsys trace (`--trace=cuda --sample=none`, which avoids the hang the default options hit
  on this box): GEMVs 1.24 ms (the head at 1.70 TB/s, the per-layer ones at 0.8–1.2 TB/s, which is
  about 2 µs of fixed cost each over ~200 launches), expert stream 0.63 ms, rmsnorm 0.2 ms (211
  launches), router selection 0.17 ms, attention ~0.2 ms. Fewer, fused launches are the next lever;
  prefill (8.2k against the 35B's 22k) has not been attributed yet.
- **The decode launch map (measured), and what it produced (landed).** One decode token
  captured with `nsys profile --trace=cuda --sample=none --capture-range=cudaProfilerApi
  --capture-range-end=stop` around `ninfer_bench --profile-measured` (which brackets exactly one
  measured repetition; the profiling API is what makes the capture cheap, the un-bracketed form
  never returned). The capture covers exactly one token -- the 2816-wide rmsnorm route's 211
  launches are the 210 layer norms plus the final one -- so these are per-token figures at bf16 KV,
  T=1:

  | kernel | instances | total | share |
  |---|---|---|---|
  | `q4rows_simt_kernel<1>` | 206 | 1.222 ms | 42.6% |
  | `gemma_moe_stream_gate_up_kernel` | 30 | 0.413 ms | 14.4% |
  | `gemma_moe_stream_down_kernel` | 30 | 0.214 ms | 7.5% |
  | rmsnorm, 2816 fixed width | 211 | 0.198 ms | 6.9% |
  | rmsnorm, 512 head width (q/k/v norms, both epilogues) | 90 | 0.082 ms | 2.9% |
  | `gemma_moe_router_select_kernel` | 30 | 0.163 ms | 5.7% |
  | sliding flash attention + its combine | 50 | 0.155 ms | 5.4% |
  | global flash attention + its combine | 10 | 0.062 ms | 2.2% |
  | `residual_add` | 90 | 0.072 ms | 2.5% |
  | `scatter` (both forms) | 85 | 0.068 ms | 2.4% |
  | `gemma_moe_router_logits_split_kernel` | 30 | 0.062 ms | 2.1% |
  | `rope_generic` | 35 | 0.056 ms | 2.0% |
  | sampling | 18 | 0.042 ms | 1.5% |
  | `gelu_mul` + `mul_scalar` | 60 | 0.046 ms | 1.6% |

  Kernel time sums to 2.871 ms of a 3.27 ms token and the median gap between graph nodes is 256 ns,
  so the token is kernel-execution-bound, not gap-bound: fusion pays through the exposed latency of
  each short launch, never through graph scheduling. The rmsnorm family is 301 launches and 0.280 ms
  (9.7%) -- pure launch cost for a 2816-element row, ~0.9 us each -- and the earlier note above
  counted only the 211 of the fixed-width route.
- **What a small GEMV costs, and why the projections are short of parallelism.** The head's own
  launch serves 415 MB at 1.70 TB/s; the other 205 launches average 956 GB/s. Across the uniform
  grid sizes the cost is affine in the launch's bytes: about 2 us of exposed latency plus 0.63 us/MB
  (1024 rows 3.56 us, 2048 3.95, 2176 4.03, 4096 5.54, 8192 10.75, 262144 244.4 us). The
  expert-stream kernels reach 1.45 TB/s because they spread over 768-1408 CTAs; the dense GEMVs have
  64-512, which is one CTA on an SM or less. It is parallelism, not DRAM traffic, that the small
  projections lack.
- **The attention inputs were three objects in the QAT artifact, and that is a converter bug
  (fixed).** The automatic packing path only groups a method listed in `tools/convert/recipe.py`'s
  `standard` tuple, and `q4_rows_qat` was not in it, so the mapping's `group(query, key, value)` was
  silently dropped for this recipe while the expert banks -- which the recipe groups explicitly --
  still became parents. The 31B's `grouped_absmax` had always been in the tuple, which is why only
  the 26B lost the fusion. The artifact went from 662 objects to 607 (25 sliding qkv unions and 5
  global q+k unions), and `qkv_slices_are_contiguous` now accepts the row-grouped int4 layout by
  plane offset, so a decode runs one launch over the 8192-row union instead of three (and one over
  the global layers' 9216-row union instead of two). The new assertion in
  `test_gemma4_26b_qat_recipe_puts_every_projection_on_its_trained_grid` is what would have caught
  it.
- **The gate/up pair rides the same mechanism (landed).** A layer's gate and up are now one parent
  too -- another 30 objects merged, 607 to 577 -- and decode runs them as one launch, which took its
  126 GEMV launches from 156 and is worth +1.6% on its own (341.0 to 346.5 tok/s). The group is
  declared in the *recipe* rather than in the mapping's `block`, and deliberately so:
  `gemma4_31b_base` puts NVFP4 on everything under `/mlp/`, a BlockScale parent has a plane a row
  slice cannot address, and `native_weight` refuses such a slice -- so a group declared in the block
  would produce a 31B artifact the loader will not accept. `row_slices_are_contiguous` accordingly
  admits only the row-split Q6 and the row-grouped int4 layouts, and a parent whose width it cannot
  derive simply declines to fuse, which is why the 26B's bring-up recipe (Q8 projections) keeps its
  two launches. Its arithmetic is the attention-input case's: the union's rows are computed by the
  same route with the same per-row reduction order, so `pp512` and both models' perplexities are
  unchanged to the last bit.
- **The row-grouped GEMV takes one output row a warp, not four.** Four rows a warp is what leaves
  the projections at 64-512 CTAs; one row multiplies the grid by four, and a warp's 32 lanes then
  cover 32 of the row's groups with the lane's strided range keeping the row's own loads in flight.
  Warps per CTA made no difference (2 warps 334.11, 4 warps 334.02, 8 warps 330.67 tok/s), so the CTA
  stays at four. A row's arithmetic does not depend on the block that covers it, so the change is
  exact.

  | QAT artifact, `ninfer_bench`, bf16 KV | tg128 (tok/s) | pp512 (tok/s) | mean NLL, 40 KB |
  |---|---|---|---|
  | as this pass began: three qkv objects, four rows a warp | 309.60 | 8,327 | 5.398731 |
  | + the attention inputs grouped into one parent | 318.62 | 8,388 | 5.398731 |
  | + one output row a warp in the row-grouped GEMV | 334.02 | 8,415 | 5.398731 |
  | + `ops::residual_add_norm` replacing the post-norm, add and scale | 338.30 | 8,374 | 5.398731 |
  | + the layer's gate and up grouped into one parent | **346.5** | 8,284 | 5.398731 |

  The last row is decode on its own (`-n 128 -r 5 --warmup 2`, three runs: 347.21, 346.69, 345.73).
  The rows above ran `pp512` and then `tg128` in one process, where decode alone reads 340.51, 340.98
  and 341.20 -- so the pass is +11.9% on decode by the decode-only protocol (309.60 to 346.5) and
  +9.3% by the mixed one, depending on which protocol is held fixed.

  The first three were measured with slightly different repetition counts, so the steps are also
  separated by an interleaved control in one session at `-r 5 --warmup 2`: the ungrouped artifact
  with the one-row-a-warp kernel is 322.41 +- 1.48 and the grouped one 332.24 +- 0.63, so the
  grouping alone is +3.0% and the kernel step +4.1-4.8% depending on the control; the fusion's three
  repeats then read 338.81, 338.30 and 336.21 against that 332.24. A row's arithmetic does not depend
  on the block that covers it, the two q4rows readers decode identically over a row, and the fused
  accumulate reproduces both of the unfused stores, so every step is bit-identical on the scoring
  corpus -- 5.398730771493524 exactly, before and after -- and leaves pp512 where it was, because the
  MMA route above eight tokens is untouched. That is +9% to +10% over the pass's starting decode,
  against the 35B's ~405; the q4rows conformance test (FP64 oracle, T = 1, 5, 8, 9, 100) passes on
  both routes, as do the 26 linear suites and the new Op's own oracle and unfused-sequence cases.
- **The three post-norm accumulations are one Op (`ops::residual_add_norm`, landed).** Every
  post-norm whose only reader is the residual accumulate -- the shared one, and the two the decoder
  layer's own sublayers add -- now folds the norm, the accumulate and (on the shared one) the layer
  scalar into a single launch: seven launches a layer become three, and the rmsnorm family drops from
  301 launches to 181. The Op is documented in `include/ninfer/ops/residual_add_norm.h`, has an FP64
  oracle test, and is bit-identical to the sequence it replaces: it calls `ops::rmsnorm`'s cell
  helper, its `block_reduce_sum`, and the very `residual_add_pair` / `mul_scalar_pair` expressions,
  which moved to `src/ops/common/bf16_combine.cuh` so the fused body cannot drift from them.
  - **Its first version was a 8% regression, and the profile said why.** At 4.66 us a call the fused
    kernel cost more than the 1.73-2.51 us of the two or three kernels it replaced: it issued the
    weight and accumulator loads *after* the block reduction, so one row of 2816 exposed two memory
    round trips where `ops::rmsnorm`'s route exposes one -- the existing kernel prefetches its weight
    in the sum pass for exactly that reason. Staging all three operands in the sum pass takes the
    call to 1.44 us and the fusion to +1.7% (`ninfer_perplexity`'s score rate rose 642.5 to 650.7
    tok/s over the same corpus). The width's unrolled pair count is instantiated per row width as
    well: covering 2816 with 21 pairs predicated off cost 2.7% of prefill.
  - The dense (31B) path takes the same fusion at hidden 5376, which the Op's 21-pair instantiation
    covers. Its perplexity on the recorded `ppl40k.txt` protocol is unchanged bit for bit, and
    `ninfer_gemma4_model_test` / `ninfer_gemma4_forward_test` pass against the 31B artifact.
  - `residual_add` and `mul_scalar` remain Ops in their own right -- the drafter and other models use
    them -- but the 26B's decoder tail no longer issues them.
- **The decode launch map after all three changes.** The same capture re-run on the changed artifact: the
  GEMV family is 156 launches and 1.040 ms (from 206 and 1.222), of which the head is 257 us, so the
  155 small launches are 783 us for the same 933 MB -- 1.19 TB/s against 0.96 before, with the cost
  curve's intercept down from ~2.5 us to ~1.76 us a launch and its slope at ~0.57 us/MB (1.75 TB/s):

  | rows per launch | bytes | avg | rate |
  |---|---|---|---|
  | 1024 | 1.62 MB | 2.68 us | 605 GB/s |
  | 2176 | 3.45 MB | 3.49 us | 989 GB/s |
  | 8192 (a fused qkv union) | 12.98 MB | 9.16 us | 1417 GB/s |
  | 262144 (the head) | 415.3 MB | 257.2 us | 1615 GB/s |

  The 8192-row union at 9.16 us replaces three launches that cost 13.45 us, so the grouping alone is
  worth 107 us of the measured gain. With every change the token's kernels number 792 at 2.585 ms,
  from 992 at 2.871 ms:

  | family | launches | time | share |
  |---|---|---|---|
  | row-grouped GEMVs (the head included) | 126 | 0.984 ms | 38.0% |
  | expert stream (gate/up + down) | 60 | 0.627 ms | 24.3% |
  | router (projection + selection) | 60 | 0.226 ms | 8.8% |
  | attention (+ combines, both kinds) | 60 | 0.216 ms | 8.4% |
  | rmsnorm (was 301 launches / 0.280 ms) | 211 | 0.196 ms | 7.6% |
  | `residual_add_norm` | 90 | 0.129 ms | 5.0% |
  | scatter | 85 | 0.070 ms | 2.7% |
  | rope | 35 | 0.056 ms | 2.2% |
  | sampling | 18 | 0.042 ms | 1.6% |
  | `gelu_mul` | 30 | 0.023 ms | 0.9% |

  `residual_add` and `mul_scalar` no longer appear in the decoder at all. Roughly 0.4 ms is still node
  scheduling over those 822 nodes: ~0.27 ms of sub-2 us gaps plus a few multi-us stalls. A
  gate_up-shaped dependency stall appears in every capture (0.08 ms of 2-1000 us gaps at the start of
  the pass, 0.24 ms after) and one 153 us gap sat between a layer's gate and up GEMVs, but nothing
  about it repeated across layers, so it reads as a node-scheduling outlier rather than a dependency
  defect.
- **What is left, ranked by measured cost.** The expert stream is the largest single item at 0.626 ms
  a token (24%) and already runs at 1.45 TB/s, so it has ~0.15 ms of headroom at most. The GEMV family
  keeps about 0.22 ms of pure exposed latency in its 125 non-head launches: at 1.76 us a launch
  apiece, that is all that is left of it now that the attention inputs and the gate/up pair are one
  launch each. Folding a pre-norm into a GEMV prologue costs a redundant row reduction in every CTA
  that consumes it, so that trade needs measuring rather than assuming. The
  router is 0.227 ms (8.7%) -- its projection reads 128 rows of 2816 BF16 over 128 CTAs at 351 GB/s,
  and its selection kernel is 5.5 us for a single warp on one SM, which is latency and not work. The
  rmsnorm family that remains is 211 launches and 0.197 ms, of which the three per-head norms a layer
  could be one launch (~56 us). The attention's 0.215 ms is the 1024-token sliding window's KV read
  (8.4 MB a layer) rather than a kernel defect. Between those and the ~0.4 ms of node overhead lies
  the 0.7 ms needed to reach the 35B's ~405.
- **Prefill, attributed (P5).** A warm single `pp512` repetition (`--profile-measured`, `-p 512 -r 1
  --warmup 2`; one cold repetition measures 302 tok/s and is worthless, so the warmup is not
  optional here) runs at 7,491 tok/s and decomposes into 946 kernels over 67.95 ms, 97% of it in
  seven families:

  | family | time | share | note |
  |---|---|---|---|
  | expert grouped MMA, gate/up (`grid=(24, jobs)`, 128 threads) | 20.84 ms | 30.7% | 51 TFLOP/s |
  | dense row-grouped MMA tile (205 launches, `grid=128`) | 16.61 ms | 24.4% | ~102 TFLOP/s |
  | expert grouped MMA, down (`grid=(44, jobs)`) | 16.52 ms | 24.3% | 32 TFLOP/s |
  | router wide-token projection (`grid=(32, 128)`, 45 KiB shared a CTA) | 7.68 ms | 11.3% | ~1.4 TFLOP/s |
  | sliding tiled flash attention | 1.97 ms | 2.9% | |
  | the head's 262144 rows on the simt route, one launch | 1.05 ms | 1.5% | <= 8 tokens reached it |
  | `residual_add_norm` (90 launches) + global attention | 1.07 ms | 1.6% | |
  | remainder (combines, plan, scatter, norms) | 2.21 ms | 3.3% | |

  So prefill is compute-bound, not bandwidth-bound, and the 26B reaches ~54 TFLOP/s where the 35B's
  ~22,000 tok/s implies ~130. The two identified causes are both structural, not bandwidth:
  - **The expert grouped tile is 64 columns wide and the average expert gets ~36 of them.** 4096
    assignments over 128 experts means most jobs are half empty, so roughly half of every MMA tile
    is padding -- and the same `kGroupedBN = kQ4RowsTileN = 64` is shared with the dense route,
    where the column count is not a lottery.
  - **The router's wide-token projection re-stages its activations per expert group.** Its grid is
    (experts/4, tokens/4) and each CTA rebuilds `4 x hidden` scaled rows in 45 KiB of shared memory,
    so a 512 x 2816 activation (2.9 MB) is read 32 times -- 92 MB a layer for 0.72 MB of weights,
    which is where its 360 GB/s and 1.4 TFLOP/s come from.
  The dense route at ~102 TFLOP/s and the attention at 4% are not the problem; the 0.6 ms a layer the
  experts and the router cost are.
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

## 7. Quantized KV for this model (active)

The engine exposes five KV storages (`--kv-dtype bf16|int8|fp8|nvfp4|k8v4`) and the 26B uses one.
Every other choice is refused at `src/runtime/engine/gemma_instance.cpp:767` with "a KV cache format
other than BF16". The refusal is honest but wider than the code: the four quantized formats exist
only in the *paged* KV subsystem on the Qwen full-attention side, and this model's KV is a separate
dense subsystem, so there is no wiring path -- not even for the sliding layers, whose geometry the
existing codecs already match.

| | Qwen (paged) | Gemma 4 (this model) |
|---|---|---|
| Layout | `paged_kv_storage.h:58-88`: one plane schema per (storage, head_dim), every non-BF16 branch guarded by `head_dim == 256` | `cache.cpp:69-103`: sliding `keys`/`values` BF16 `[d, hkv, ring]`; global `keys` BF16 `[d + 2*rope_angles, hkv, capacity]` and **no values** (`attention_k_eq_v`); no scale plane |
| Write | `kv_cache_append.cpp`: bf16/nvfp4/k8v4 launchers, the prefix (cyclic) variant BF16 only | `ops::scatter` of BF16 rows and `write_cache_rows` (`forward.cpp:105-116`) |
| Read | `softmax_attention/dense/causal_cache/{bf16,int8,nvfp4,fp8,k8v4}/`, geometries (24,4)/(16,2)/(16,16), `scale == 0.0625` enforced | one BF16 flash kernel (`gemma_flash_attention.cuh:47-56`) behind `ops::sliding_causal_attention` (D256/Hq16/Hkv8) and `ops::causal_compact_attention` |
| Codecs | `int8_g64_codec.cuh`, `fp8_e4m3_row_codec.cuh`, `nvfp4_group16_codec.cuh`, `hadamard_d256.cuh` | none |

**Step 1, the 25 sliding layers (D256, 8 KV heads).** Their geometry already matches the codecs
(`kKVCacheInt8HeadDim` / `Fp8` / `Nvfp4` = 256). Three pieces: the ring's storage gains the format's
planes beside the BF16 ones and `device_bytes` stops assuming two bytes an element; an append that
quantizes rows on write, reusing the codecs' own encoders rather than re-deriving them; and a
dequantizing sliding attention, either by reusing the per-format decode math in
`dense/causal_cache/{int8,fp8,nvfp4,k8v4}/` at this model's geometry (Hq16/Hkv8, `attention_scale
1.0`, where the paged ones register (24,4)/(16,2)/(16,16) with scale 0.0625) or by adding the formats
to `gemma_flash_attention.cuh`'s loads.

**The Hadamard question is settled: the readers rotate the queries.** The codecs rotate rows with
`hadamard_d256.cuh` before quantizing, which preserves a dot product only if the paired operand is
rotated too, and every paged decoder does exactly that at read time -- `int8/tiled_mma.cuh:83` and
`int8/grouped_mma.cuh:224` apply `normalized_hadamard_d256_inplace` to the query fragment, as do the
fp8, nvfp4 and k8v4 decoders. Porting a format to the sliding attention therefore means writing the
stored rows through the same codec and rotating the queries the same way, rather than re-deriving the
mathematics; what differs is this model's geometry (Hq16/Hkv8 at `attention_scale 1.0`) and the ring
addressing.

**Step 2, the 5 global layers.** The harder half: D512 rather than 256, two KV heads (eight queries
each), K and V the same tensor, and a stored row of 640 (512 plus the two RoPE angle planes). No
existing codec covers any of that, so this is new codecs rather than a re-plumb. Until it lands, a
quantized choice leaves the global layers BF16 -- a hybrid that still moves most of the KV, since the
sliding rings are 25 of the 30 layers.

**What Step 1's work gives Step 2, and the one thing it must not.** The codecs themselves do not
transfer: all four pin `head_dim == 256` (`kKVCache{Fp8,Int8,Nvfp4}HeadDim`, `hadamard_d256.cuh`, and
every `paged_kv_storage_layout` branch), and 640 = 2^7 x 5 admits no Sylvester Hadamard at all, so
the rotation has to be per power-of-two slice. What does transfer is the shape of the solution and
everything around it: the `KvCacheStorage` plumbing through `Program`/`KvCache` with a per-layer
`quantized()` and plane accessors, `device_bytes` and the pass-scratch bound, the wrapper/launcher/
kernel split of `dense_kv_ring`, the round-trip + FP64-oracle + host-only footprint test triple, and
the rotation-at-decode trick (undo the encoder's rotation in the decode so no query or output path
changes). Two group schemes tile the 640 row exactly -- INT8-G64 in ten groups and NVFP4-G16 in
forty -- while FP8-row-256 does not (2.5 groups), so a first global format should be one of those
two. K == V needs no extra structure: one decoded 640-wide row already serves both products in
`gemma_flash_attention.cuh`'s compact path (`<640, 512, true>`), which reads the row for QK^T and its
first 512 entries for P V.

The thing Step 2 must not inherit is Step 1's **decode into BF16 scratch**, and the numbers say so
plainly. Materializing the decoded row costs, per layer per step: read the codes, write a BF16 copy,
then read that copy back, against BF16's single read. On the *ring* that amplification is 4.5-8.7%
of decode time (measured above), because the ring is a fixed 8.4 MB whose working set stays in
cache. On the *global* cache the same amplification lands on the one tensor that grows: at 262,144
tokens one layer's cache is 640 x 2 x 2 B = 671 MB, all of it streamed per token in BF16 today (3.36
GB a step over five layers, ~1.9 ms at 1.8 TB/s). INT8-G64 would cut that read to ~346 MB a layer,
but a scratch decode would add ~671 MB of writes and ~671 MB of reads a layer on top, turning a
1.9 ms saving into roughly a 1 ms *loss*. Step 2's decode therefore belongs in the attention's tile
loads (a 640/512-tile dequant), where the codec work above is what makes it cheap to write; the
plumbing, the tests and the guard stay as they are.

**Verification.** Each format needs the Op's own oracle: an FP64 reference that dequantizes the
stored rows and runs the attention at the real geometry (hkv 8, D256, ring 1024), the way the linear
routes are qualified. Above that, the model-level check is the documented scoring protocol
(`ninfer-perplexity --context 1024 --stride 512 --kv-dtype <fmt>`) against the BF16 number
`5.398731`; the earlier codec survey measured int8 row-64's attention-score error at 0.0059, so a
small shift is expected and a large one is a defect. The guard moves per format, only as each passes.

### 7.1 Step 1 landed (2026-10-12)

All four formats are wired for the sliding rings; the global layers stay BF16, so Step 2 is still
open. What the code does, and how it differs from the sketch above:

- **Two Ops, not a ported decoder.** `ops/dense_kv_ring.h` owns the ring codec: `dense_kv_ring_encode`
  quantizes BF16 `[256,Hkv,T]` rows into a slot-indexed ring's planes (reusing the codec headers'
  own `kv_cache_fp8_quant_params` / `int8_quant_params` / `nvfp4_quantize_group16` and their Hadamard,
  so the stored bytes are the paged append's for the same row), and `dense_kv_ring_decode` forms the
  represented value of every code and scale into BF16 `[256,Hkv,S]` rows. The plane schema is the
  storage's own `paged_kv_storage_layout(storage, 256)`.
- **The rotation moved to the decode, not the query.** The codecs rotate the row before quantizing;
  the normalized D256 Hadamard is an involution (`T² = I`), so the decode applies it again and hands
  the attention rows in the basis the encoder read. That is the same mathematics as rotating the
  query -- and rotating the value for the formats whose V is rotated -- but it leaves
  `gemma_flash_attention.cuh` and every query path bit-identical, which the BF16 default's exact
  reproduction depends on. Reusing the paged decoders at this geometry was rejected: they fix
  (24,4)/(16,2)/(16,16) and `scale == 0.0625`, read a block table, and scan the whole logical prefix
  rather than a ring.
- **Storage.** A quantized sliding layer allocates only the format's planes (no BF16 rows), cleared
  at allocation so an unwritten slot decodes to zero; `device_bytes` reports the planes' bytes; the
  storage threads from `EngineOptions::kv_cache` through `GemmaProgram` -> `Program` -> `KvCache`.
- **Read.** `forward_sliding_layer` and `forward_draft` decode a segment's ring into BF16 scratch
  (two rings, in the pass arena, whose bound `layer_workspace_bytes` now includes) and call the same
  sliding Op as the BF16 path. A wide pass still hands its own keys as the Op's second key set, so
  its rows are full precision until they are written.

Measured (40 KB slice, context 1024, stride 512, 9,319 scored tokens), against BF16 `5.398731`:

| `--kv-dtype` | mean NLL | PPL | oracle (attention, rel L2) | round trip (rel L2) |
|---|---|---|---|---|
| bf16 | 5.398731 | 221.126 | -- | -- |
| int8 | 5.419401 | 225.744 | 0.00309 / 0.00304 | 0.00614 / 0.00437 |
| fp8 | 5.430185 | 228.191 | 0.00300 / 0.00316 | 0.02610 / 0.02542 |
| nvfp4 | 5.450106 | 232.783 | 0.00316 / 0.00304 | 0.09529 / 0.09518 |
| k8v4 | 5.457335 | 234.472 | 0.00304 / 0.00313 | 0.02610 / 0.09518 |

The shifts (+0.021 to +0.059 nats, against an artifact budget of 0.419) order exactly by precision,
and each format's round-trip error reproduces an independent FP64 emulation of the same codec to
three digits. `tests/models/gemma4/test_kv_quant_test.cpp` holds both checks at this geometry.

**Footprint.** A quantized ring is a code plane plus a scale plane against BF16's two bytes an
element, so a quantized run must reserve less than a BF16 one and must not scale with the context
capacity at all -- the ring is the window. Engine-reported sequence bytes at `--max-ctx` 17408 /
32768 (bf16 601,844,800 / 798,760,000): int8 508,653,120 / 705,568,320, fp8 506,195,520 /
703,110,720, nvfp4 459,501,120 / 656,416,320, k8v4 482,848,320 / 679,763,520. The reservation
(`Program::device_bytes`) matches what the Program allocates to 64 bytes.
`ninfer_gemma4_kv_quant_footprint_test` pins all of it on the host: quantized below BF16 at 8192,
17408 and 32768, a capacity growth that adds exactly the five global layers' rows for every storage,
and a pass scratch that grows by exactly the two decoded rings.

**Where the payoff is, and it is not in this step yet.** At a short context the quantized formats cost
decode time rather than saving it, because there is nothing to read back: `ninfer_bench --weights
<26B> --max-ctx 2048 -n 128 -r 3 --warmup 1` reads bf16 350.80 tok/s, fp8 336.57, int8 334.99, k8v4
326.09 and nvfp4 320.21, so encoding and dequantizing a nearly empty ring is a 4.5-8.7% loss. That is
the expected shape and this step cannot change it: the sliding ring is capped at 1024 rows, so its
read is 25 layers x 8 heads x 256 dims x 2 tensors x 2 bytes = 205 KB a token however long the
context is. The layers whose KV grows with the context are the global ones, which are not a ring at
all: 5 layers x 2 heads x 512 dims x 2 bytes = 20 KB a token per stored row of context (10 KB if
`attention_k_eq_v` means one row holds both, which does not change the comparison), i.e. ~0.34 GB a
token over 16K. Those two differ by three orders of magnitude, so whatever a KV byte costs, Step 2 is
where it is spent -- and Step 2 is also where no codec exists yet (D512, two KV heads, K == V, a
640-wide stored row). Step 1's worth is the mechanism, the four codecs proven at this model's
geometry, and a measured quality cost of 0.021-0.059 nats; the performance case belongs to Step 2.

**A retraction, with the numbers that replace it.** This work was reported as over-allocating, because
`ninfer_bench --kv-dtype nvfp4|int8|fp8 --max-ctx 17408` died with `cudaMalloc failed:
cudaErrorMemoryAllocation` while the same command with bf16 ran. That was contention, not size. On a
free device all five formats run that exact command and at a larger capacity, and the peak device bytes
sit *below* bf16 at every capacity -- at 32,768 tokens: bf16 798,760,000 against int8 705,568,320, fp8
703,110,720, k8v4 679,763,520 and nvfp4 656,416,320, i.e. 93-142 MB less, with the global (non-ring)
layers byte-identical to BF16. The same failure reproduces with *bf16* while one other 26B process is in
flight: a 32 GB box fits exactly two of them (about 15.4 GB of weights each), and the third fails
whichever storage it asked for -- and bf16, needing the most, fails first. `device_bytes` equals what
the Program actually allocates (to 64 B); a quantized ring is sized from `ring_tokens()` and never from
`capacity_`; and the dense path carries no page/pad term. All three are now pinned by
`tests/models/gemma4/test_kv_quant_footprint.cpp`, a host-only test that runs beside a busy device --
which is the part worth keeping, since it cannot be fooled the way a one-off measurement was.

## 8. The all-models serve benchmark (active)

`tools/bench/run_bench_sweep.py` drives `run_serve_concurrency.py` over every local artifact and
`plot_concurrency.py` draws one figure per model: prefill and decode against context, a line per
concurrency. Run it with no arguments for the default suite; figures land in
`profiles/bench/all_models/figures/`. Twenty-four campaigns take about 58 minutes.

**Two harness defects had to be fixed first, and both returned wrong numbers rather than errors.**

*The KV window is reserved per lane, not shared.* `gemma_engine_options` forces
`kv_capacity = max_context` and `GemmaInstance` sizes the cache through
`Program::device_bytes(config, capacity, lanes, ...)`, refusing with "Gemma needs N MiB for L lane(s)
of C tokens ... lower --max-ctx or --max-concurrency". Every lane therefore reserves a *full* window,
so a 32 GB card holds about 262,144 tokens across all lanes -- not per lane. The driver originally ran
all four concurrencies under one window, so C=2 and C=4 oversubscribed and C=8 could not start at all
(its campaign exited 1 with no summary). It now sets `window = lane_tokens / lanes` (c1 262144, c2
131072, c4 65536, c8 32768) and walks a ladder downward when a startup refuses, costing one weight
load (~3 s) per rung. Measured: qwen3.8-27b refuses 262,144 (needs 17,480,752,480 B against
14,043,054,080 free) and serves at 131,072 with an 8.28 GiB runtime reservation.

*`decode_tokens_per_second` over a makespan is not a decode rate.* For `corpus-makespan` the runner
computes it as total decode tokens over the whole makespan, and these NIAH requests answer in about 17
tokens: 240 decode tokens over 220 s reads 1.09 tok/s against a true per-request 54.4. The plotter also
hardcoded `concurrency: 1` for every per-request record, folding all four campaigns' records into the
c1 line while c2/c4/c8 kept that aggregate. Records now carry their point's own concurrency, the decode
panel excludes `corpus-makespan` aggregates, and the prefill panel keeps them (total prefill tokens
over the makespan is a real throughput). A per-request decode rate exists at C=1 only, by design: those
records are written only for `corpus-makespan` at one lane, because the collector pairs each client
result with the server's `request_done` event by sequence, which is ambiguous with several in flight.

**Measured, C=1, per request.** Prefill and decode tokens/s at the smallest and largest context each
model reached (five seeds each, spread within 1-2%):

| model | ~8k context | largest reachable context |
|---|---|---|
| gemma4_26b_a4b_qat | 6,840 / 317 | 132,694 -> 2,801 / 210 |
| qwen3_6_35b_a3b_nvfp4 | 23,730 / 381 | 64,512 -> 14,346 / 299 |
| qwen3_6_35b_a3b | 16,887 / 379 | 260,096 reached |
| qwen3_8_27b_nvfp4 | 10,079 / 74 | 130,048 -> 4,034 / 54 |
| gemma4_31b_it_m1 | 2,030 / 52 | 65,837 -> 1,159 / 48 |
| qwen3_8_27b | 3,333 / 80 | 64,512 -> 2,769 / 68 |

The context each concurrency reaches follows its window: the 26B covers 7846/65837/132694 at c1, 65837
at c2, and 7846 at c4 and c8. `qwen3_6_27b` has no artifact (`models/qwen3_6_27b.ninfer` is absent), so
the sweep records it as a skip. The 26B's 256k NIAH fixture is unreachable at any window: its prepared
prompt is 265,411 tokens against the model's own 262,144-token limit.

**Steady decode at every concurrency.** `--suite decode-saturation --decode-tokens 1024` measures
`metrics.steady.decode_tokens_per_second` over complete intervals whose decode batch equals the
concurrency, which is valid at every concurrency, unlike the makespan aggregate. Measured over a fixed
~300-token context (aggregate system throughput, so c8 counts all eight lanes):

| model | c1 | c2 | c4 | c8 |
|---|---|---|---|---|
| gemma4_26b_a4b_qat | 330.5 | 474.0 | 586.0 | 674.9 |
| qwen3_6_35b_a3b | 389.0 | 631.0 | 1008.0 | 1376.0 |
| qwen3_6_35b_a3b_nvfp4 | 397.0 | 640.0 | 985.3 | 1337.6 |
| qwen3_8_27b | 81.0 | 144.3 | 255.7 | 422.2 |
| qwen3_8_27b_nvfp4 | 76.3 | 149.0 | 282.2 | 544.6 |
| gemma4_31b_it_m1 | 54.0 | 86.9 | 154.6 | did not fit |

Decode throughput scales 2.0x (26B) to 5.2x (qwen3.8-27b) from one lane to eight. The 31B's c8 does not
fit the card at any window down to 8192, so it is a recorded skip rather than a failure. The c1 column
cross-validates the makespan pass: 330.5 steady against 317-326 per request on the 26B.

**tg against context, at every concurrency (closed).** `--suite decode-saturation --pad-tokens N`
prefixes the saturation prompt with in-distribution prose from `make_bench_corpus.PARAGRAPHS`, which
puts steady decode at a chosen context while keeping the reasoning question last, so the output still
runs to its limit (verified: a 123,342-token prompt generated all 1,024 requested tokens rather than
answering early). Measured on the 26B, aggregate over intervals whose decode batch equals the
concurrency:

| c | ~300 tokens | ~8k | ~31k | ~123k |
|---|---|---|---|---|
| 1 | 330.5 | 320.0 | 291.0 | **212.5** |
| 2 | 474.0 | 445.3 | 374.8 | -- |
| 4 | 586.0 | 550.9 | 461.1 | -- |
| 8 | 674.9 | 620.7 | -- | -- |

qwen3_6_35b_a3b follows the same shape deeper in: c1 389.0 / 378.5 / 342.0 / 256.7 at 300 / 7.3k /
28.4k / 113.0k tokens, c2 631 / 592 / 519, c4 1008 / 914, c8 1376. Every figure now carries prefill
for c1/c2/c4 (c8 where the card fitted) *and* tg against context for the same concurrencies.
`gemma4_31b_it_m1` and `qwen3_8_27b` have only two c1 points (300 and 8k); their deeper padded runs
exist on disk but produced no point, which is unexplained and should not be read as the model's reach.

**A third round of harness defects, all fixed.** (1) The plotter dropped *every* aggregate for a
concurrency that had per-request records -- including the decode-saturation point -- so c1 lost its
entire tg curve; only a `corpus-makespan` aggregate is replaced by records now. (2) The driver's window
ladder read a non-zero exit as "does not fit", but a run whose requests were partly refused exits
non-zero *after* writing its points, so it re-ran whole campaigns one rung down: six times per Gemma
model, which is where most of a 2.8 h MTP pass went. A run that wrote `points/` is now complete.
(3) `block_fixture_names()` selects the prompt set from the speculative backend, so a plain-vs-MTP A/B
ran disjoint requests (4 NIAH fixtures against 3 long-decode + 12 scenario prompts, zero shared pairs)
and its ratios measured the workload change rather than MTP; `--block-fixtures {auto,niah,decode}`
overrides it, verified to give 15 shared prompts on both sides.

**Quality, repo protocol.** `ninfer-perplexity --kv-dtype bf16` over `ninfer-ppl-1m-v1` full --
1,092,299 tokens, 526 windows, context/stride 4096/2048 -- reads overall mean NLL **5.039971**
(PPL **154.466**) at 714.3 tok/s. Per domain: chinese_reference 5.184 (PPL 178.4), english_long_form
5.948 (383.1), english_reference 6.642 (766.4), ninfer_code 2.701 (14.9).

**MTP on the serve route, and what it is worth.** The Gemma 4 engine supports MTP with the *full*
proposal head only (1-7 draft tokens) and refuses the optimized one, so every Gemma MTP point died with
"Gemma 4 does not support an optimized proposal head yet" while the benchmark runner asked for
`--lm-head-draft` unconditionally and its config validation expected `proposal_head: optimized`. Both
are conditional now (`mtp3_full` / `mtp5_full`, plus `--block-fixtures long` for a 15-request matched
A/B rather than 75). Same-session plain against MTP at a ~300-token context, both modes in one
invocation, gain at two / four / eight lanes:

| model | c2 | c4 | c8 | acceptance |
|---|---|---|---|---|
| gemma4_26b_a4b_qat | 1.39x | **0.67x** | 1.03x | 0.82-0.85 |
| qwen3_6_35b_a3b | 1.46x | 1.12x | 0.98x | 0.72-0.75 |
| qwen3_6_35b_a3b_nvfp4 | 1.43x | 1.42x | 1.57x | 0.71-0.74 |
| qwen3_8_27b | 2.21x | 2.03x | 1.78x | 0.81-0.82 |
| qwen3_8_27b_nvfp4 | 2.10x | 1.98x | 1.87x | 0.59-0.63 |

Speculation pays where decode is weight-bound -- the dense 27B models gain about 2x at every
concurrency -- and stops paying once the engine already has a full batch: on the 26B at four lanes it is
*slower* than plain decode, because the draft passes cost more than the tokens they commit. On long
single-request generations (32k-token outputs) the same drafter reads 1.75x (288 -> 500 tok/s) at
0.87-0.95 acceptance, measured across two sessions rather than in one. The one-lane MTP point did not
run for any model and is unexplained.

**Master charts.** `tools/bench/plot_master.py` draws one figure per model with four panels -- prefill
and decode against context (a line per concurrency) and the same two against concurrency (a line per
context) -- into `profiles/bench/all_models/figures/{model}_master.{png,svg}`, and the sweep driver now
emits it next to the concurrency chart. Panels are sparse by design (a concurrency can only reach the
contexts whose windows fit the card), and the figure states its two sources: prefill at c1 from the
per-request records and at c2-c8 the point aggregate (prompt tokens over the whole makespan, which
includes each request's decode), decode from decode-saturation steady intervals whose batch equals the
concurrency. Cells per model: 26B 6 prefill / 12 decode, the two 35B-A3B builds 7/10, the two
Qwen3.8-27B builds 6/8, the 31B 4/6. MTP is deliberately not on the master -- it has its own figure in
`profiles/bench/mtp_decode/figures/`.