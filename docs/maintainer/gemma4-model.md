# Gemma 4 model reference

Gemma 4 31B (`Gemma4ForCausalLM`, text `gemma4_text`) as NInfer executes it. This is the
mathematical authority for the implementation and the oracle; it is written from the checkpoint and
the plan that preceded this work. Vision (P5) and the assistant drafter (P6) are not described here
yet.

**Status.** Text tower in progress. The converter, the L1 artifact and the tokenizer are done; the
Program and its Ops are not. Open points are listed at the end with what would close them.

## 1. Configuration (the base checkpoint)

Verified against `/mnt/storage/models/gemma/full-31b/config.json` and the artifact's own config.

| Quantity | Value |
|---|---:|
| Hidden H / MLP intermediate I | 5376 / 21504 |
| Layers | 60: 50 sliding + 10 global, global at indices 5, 11, …, 59 (every sixth, last included) |
| Query heads | 32 in every layer |
| Sliding: KV heads / head dim / window | 16 / 256 / 1024 |
| Global: KV heads / head dim | 4 / 512, `attention_k_eq_v` — there is no `v_proj` |
| RoPE, sliding | `default`, θ = 10,000, all 256 dims |
| RoPE, global | `proportional`, θ = 1,000,000, `partial_rotary_factor` 0.25 |
| Activation | `gelu_pytorch_tanh` (GeGLU MLP) |
| RMSNorm ε | 1e-6 |
| Vocabulary | 262,144, embeddings tied to the LM head (the checkpoint has no `lm_head`) |
| Final logit soft-cap | 30.0 |
| Context | `max_position_embeddings` 262,144 |

Not used by this checkpoint: MoE block, per-layer inputs, KV-shared layers, double-wide MLP, audio.

## 2. Norms

`rmsnorm(x, w) = x · (mean(x²) + ε)^(-1/2) · w`, computed in FP32, with the weight used **as is**.
This is not the Gemma 2/3 or Qwen `(1 + w)` offset — ModelOpt's Gemma 4 spec omits `+1` as well — so
`ninfer::ops::rmsnorm` is used with its unit offset off. `v_norm`, the global layers' shared norm
and the vision projector's pre-norm are the same formula without a weight.

## 3. Decoder layer

```text
h = embed(token) · √H
```
The embedding scale is a rounding boundary, not a constant to fold away: Transformers gathers the
row (already rounded to BF16 by the table's dequantization) and multiplies it by `√H` cast to the
weight dtype, so the product is rounded a second time. Folding `√H` into the stored row scale would
multiply first and round once, which is a different BF16 value, so the scale has to be applied to
the gathered row.

```text
for layer ℓ:
    a = attn_ℓ(rmsnorm(h, w_in))
    h = h + rmsnorm(a, w_post_attn)          # sandwich norm on the branch output
    x = rmsnorm(h, w_pre_ff)
    m = W_down · (gelu_tanh(W_gate · x) ⊙ (W_up · x))
    h = h + rmsnorm(m, w_post_ff)
    h = h · s_ℓ                              # layer_scalar
logits = 30 · tanh((W_embᵀ · rmsnorm(h, w_final)) / 30)
```

The checkpoint carries four norms per layer (`input`, `post_attention`, `pre_feedforward`,
`post_feedforward`), one `layer_scalar` per layer as BF16, and both post-norms reduce over the whole
output row, so neither can live in a column-tiled GEMM epilogue. `layer_scalar` scales the entire
residual stream (layer 0 = 0.0894, layer 1 = 0.0654, layer 59 = 0.0364, the rest 0.44–0.99), so it
cannot be folded into a projection weight; it belongs in the residual update.

## 4. Sliding-window attention (50 layers)

```text
q = rope₁₀ₖ(rmsnorm(W_q x, w_qn))     # [32 heads, 256]
k = rope₁₀ₖ(rmsnorm(W_k x, w_kn))     # [16 heads, 256]
v = rmsnorm(W_v x)                    # [16 heads, 256], no weight
o = W_o · softmax(q·kᵀ · 1.0 + mask) · v
```

- RoPE is `rotate_half` over all 256 dims: pairs (j, j+128), `inv_freq_j = 10000^(-2j/256)`.
- The norms without a weight (`v` here and the shared normalization below) use the `rmsnorm`
  weightless form, whose gain is exactly one: no weight tensor is read at all, and the
  normalized row is materialized in BF16, which is the value the cache stores.
- Query head h reads KV head ⌊h/2⌋.
- **Scale is exactly 1.0**, not `1/√D`.
- **Mask:** key position `p_k` is visible to query `p_q` when `0 ≤ p_q − p_k < 1024`. The bound is
  one-sided, so the window is a causal band of 1024 with the query included.

## 5. Global attention (10 layers): K = V and proportional RoPE

```text
r = W_k x                             # [4 heads, 512]; there is no W_v
n = r · (mean(r²) + ε)^(-1/2)         # weightless shared normalization
v = n

k = rope_prop(n ⊙ w_kn)
q = rope_prop(rmsnorm(W_q x, w_qn))   # [32 heads, 512]
o = W_o · softmax(q·kᵀ · 1.0 + causal) · v
```

- Query head h reads KV head ⌊h/8⌋; always causal, with no image overlay.
- The checkpoint's `k_norm.weight` is 512 wide on these layers (256 on the sliding ones) and scales
  the normalized vector.
- **Proportional RoPE:** with D = 512 and factor 0.25, `rope_angles = 64`;
  `inv_freq_j = 10⁶^(−2j/512)` for j < 64 — the denominator is the **full** head dim — and 0 for
  j = 64..255, and `rotate_half` pairs (j, j+256). Only dims {0..63} ∪ {256..319} rotate. This is
  **not** Qwen's partial RoPE (a contiguous leading block with frequencies normalized to the rotary
  dim): the rotated set is the leading pairs, and the frequency denominator stays the full head
  dimension. The `rope` Op expresses both kinds through `rotary_pairs`, the count of rotated pairs,
  taking the denominator from `rotary_dim`: global layers pass `rotary_dim = 512` with
  `rotary_pairs = 64`, sliding layers `rotary_dim = 256` with `rotary_pairs = 128`. Pairs outside
  the count and their partners are bit-exact unchanged, which reproduces the checkpoint's zeroed
  frequencies instead of approximating them with a near-zero rotation. Long-context phases make the
  generic kernel evaluate its angle in double: at position 262144 the float spacing alone is about
  0.016 rad, wider than the deviation the rotation is held to.
- **Compact global KV (private representation, exact reformulation).** On the 384 non-rotated dims
  `k[d] = v[d]·w_kn[d]`, so
  `q·k = Σ_{d∈rot} q[d]·k[d] + Σ_{d∉rot} (q[d]·w_kn[d])·v[d]`.
  The cache may therefore hold, per token and KV head, only `v` (512 values) and the 128 rotated `k`
  dims — 640 instead of 1024, 37.5% less global KV — with the query prescaled by `w_kn` on the
  non-rotated dims during Q preparation, so one load of `v` serves both QKᵀ and PV. The formula above
  remains the oracle; the reformulation changes rounding, which the Op criterion must cover.

  Both multiplications are the `scale_columns` Op: the whole 512-wide head for `k = n ⊙ w_kn`, and
  the two non-rotated runs `[64,256)` and `[320,512)` for the query prescale, whose dimensions
  outside the range stay bit-exact. The compact row itself is assembled by `compact_kv_rows`, which
  puts the value vector first and then the rotated key dimensions in low-then-high order, `[0,64)`
  followed by `[256,320)`; the checkpoint defines no order, so this is the representation's own
  contract and the global attention reads it in that order.

### 5.1 What the attention Ops cost us (read before planning rows 4 and 5)

- **The sliding Op's geometry is compile-time and shared.** `kContextQueryHeadDim`,
  `kContextQueryQHeads` and `kContextQueryKVHeads` live in
  `ops/softmax_attention/common/context_query.cuh` and are referenced 43 times across five files,
  including the dense causal route that Qwen uses. A Gemma profile (D 256, 32 query heads, 16 KV
  heads) is therefore a refactor of shared attention code rather than a configuration, and it
  rebuilds the dense route too. The window and scale *are* runtime values, but the wrapper admits
  only windows 2048/4096 and scale 1/sqrt(128), so the profile's window 1024 and scale 1.0 are
  refused before any kernel is chosen.
- **The sliding Op is symmetric on the current chunk.** Every live query sees every live temporary
  row, and context keys within the window; that is the image-block rule. Gemma's text layers need
  the causal rule (`0 <= p_q - p_k < 1024`), which is one predicate in principle. An attempt that
  routed the four mask sites through a policy hook was reverted: the existing bidirectional cases
  stayed green, but causal cases disagreed with the causal oracle *even where the two oracles
  coincide* — a case with at most one live temporary row — so the kernel's causal path diverges
  somewhere the mask must be invariant. The cause is unresolved. A next attempt should start from a
  minimal reproducer (one batch, one live row, an empty committed cache) instead of the full matrix.

## 6. Weight inventory

From `model.safetensors.index.json` (1,188 tensors). Text lives under `model.language_model.`:

| Role | Tensors |
|---|---|
| Embedding (tied to the head) | `model.language_model.embed_tokens.weight` |
| Per layer | `input_layernorm`, `post_attention_layernorm`, `pre_feedforward_layernorm`, `post_feedforward_layernorm`, `layer_scalar` |
| Attention | `self_attn.{q,k,o}_proj`, `self_attn.{q,k}_norm`, and `self_attn.v_proj` **only on the 50 sliding layers** |
| MLP | `mlp.{gate,up,down}_proj` |
| Final | `model.language_model.norm.weight` |

Vision is `model.vision_tower.*` (27 layers, `patch_embedder`, `std_bias`/`std_scale`) plus
`model.embed_vision.embedding_projection`; the plan puts it in P5.

## 7. Tokenizer

Gemma 4's `tokenizer.json` is a **BPE whose symbols are raw UTF-8 characters**, not GPT-2 bytes:

- normalizer: `Replace` — a space becomes U+2581 (`▁`). There is no NFC step, and one must not be
  added: a decomposed `e` + U+0301 stays two tokens (`[236744, 238288]`), which NFC would compose.
- pre-tokenizer: one `Split` on `" "` with `MergedWithPrevious`, so BPE merges run over the whole
  text with no word-level pre-splitting (`f(x)` is `'('`, `'x'`, `')'`, `':'` grouped by the merges).
  Words are therefore the runs of newlines and of everything else, because a merge may not cross a
  newline.
- model: `byte_fallback: true`, 262,144 vocabulary entries, 514,906 merges, and 256 `<0xXX>` tokens
  for the fallback. A character the vocabulary lacks is written as its bytes' fallback tokens
  (`U+323B0` becomes four tokens).
- decoder: `Sequence[Replace(▁ → " "), ByteFallback, Fuse]`.
- `tokenizer_config.json` names `pad_token` `<pad>` (there is no `<|endoftext|>`), omits
  `added_tokens_decoder`, and lists its 24 specials — including `<turn|>` and `<|tool_response>` for
  the serving protocol — in `added_tokens` with the flags off.

Evidence: `tests/models/qwen3_5/test_gemma4_tokenizer.cpp` compares exact token ids against the
`tokenizers` reference for 24 cases recorded in `tests/fixtures/gemma4/tokenizer_ids.json`, which
also records the source digest.

## 8. Artifact

`out/gemma4_31b_base_nvfp4.ninfer`, produced by the `gemma4_31b_base` recipe from the **base** BF16
checkpoint: 667 objects, 23 GB, 673 s. Layout per the plan's D3/L1: MLP NVFP4 encoded by the
converter itself (NVIDIA's checkpoint is instruction-tuned, so its NVFP4 cannot be imported on this
path), attention and embedding FP8 `e4m3fn_row_bf16`, norms BF16, `layer_scalar` FP32. Counts
confirm the mapping: 120 NVFP4, 121 FP8, 362 BF16, 60 FP32.

Chat, tools and thinking need an instruction-tuned artifact later; the same recipe produces it from
an `-it` checkpoint.

## 9. What the sibling engines settle

Checked against llama.cpp (MIT) `src/models/gemma4.cpp` and gewell (Apache-2.0)
`include/gewell/models/gemma4/31b/`, which implement this model:

- **Serving scale.** llama.cpp's loader carries `f_attention_scale = 1.0f` with the comment that Gemma 4
  uses `self.scaling = 1.0` with no pre-attention scaling — the same invariant this reference states.
- **Per-layer kind.** It also keeps a separate SWA rope base (`rope_freq_base_train_swa`, 10k) from the
  global one (1e6), and separate SWA head widths (`n_embd_head_k_swa` / `n_embd_head_v_swa`), so the
  two geometries this reference describes are the two it branches on.
- **Sliding layers need no new attention mathematics**: llama.cpp treats them as a standard sliding
  window (`LLAMA_SWA_TYPE_STANDARD`) over a windowed cache with a mask for the prompt, not as a
  different attention. The window is a property of the cache and the mask.
- **Proportional RoPE is a parameterization, not a kernel.** llama.cpp supplies explicit per-pair
  frequency factors for the global layers. Our `rope` Op cannot express it as it stands: its
  `phi = pos·θ^(-2i/rotary_dim)` puts the frequency denominator at the rotary width, and it pairs
  `(i, i + rotary_dim/2)`, where the global layers need the denominator over the *full* head and the
  pairs `(j, j + head_dim/2)`.
- **The global K = V compression is already implemented** in gewell's
  `include/gewell/compact_global_cache.h`, alongside `kv_format.h` and `fp8_attention.h`, for the same
  card and the same quantization layout this artifact uses.

## 10. Open points

| Question | How it closes |
|---|---|
| Sliding mask with images. The checkpoint sets `text.use_bidirectional_attention: "vision"`, and llama.cpp reads that flag as *bidirectional attention on the SWA layers only*, dense layers staying causal (`LLAMA_NON_CAUSAL_TYPE_SWA_ONLY`), which agrees with the plan's note that global layers have no overlay — but the plan words it more narrowly, as an image block seeing itself | Transformers `create_masks_for_vision_model` / `sliding_window_overlay` at the pinned commit, then an oracle case — P5 |
| The assistant drafter's hidden state (before or after the final norm) | The drafter's `modeling_gemma4_assistant.py` — P6 |
| Whether the compact global KV keeps the oracle's rounding at 128K/256K | The Op criterion at those lengths; fall back to separate K/V rows (+60% global KV) if it does not |
| `layer_scalar` at layers 0, 1 and 59 (0.036–0.089) in BF16 residual | Oracle error at those layers |

## 11. Unlanded attempts so far

Recorded so the next attempt starts where this one stopped, not from scratch.

- **Fused sublayer boundary (plan row 6), specified and attempted, not landed.** The Op was written
  as `h' = (h + rmsnorm(y, w_post)) · s` then `x = rmsnorm(h', w_next)`, with `h'` treated as an
  observable BF16 boundary — the second normalization consumes the stored value, and the test's
  oracle for `x` therefore starts from the *rounded* scaled residual, which widens its criterion to
  two roundings. With the layer scalar equal to one, every case passed. With a scalar other than
  one, elements at small *cancelling* values deviated by up to 0.7% relative — about 3.5 BF16
  half-ulps, which a single output rounding cannot produce — and the deviation was identical
  (7.8e-3, one half-step at magnitude 2) across unrelated rows, so it is not a tie case. The Op was
  reverted rather than shipped. The next attempt should compare `inv` and the residual sum directly
  between the kernel and an FP64 recomputation, since the terms are fp32-reduced while the ideal is
  fp64 and the observed deviation is *larger* than that difference explains.
- **A real bug found while attempting it.** A kernel that reduces a row twice through one shared
  buffer needs a barrier before the *second* reduction writes its slots, or a thread overwrites the
  previous total before a slow thread has read it. The first version of the Op raced and produced
  wrong `inv` values in some rows; the fix is a `__syncthreads()` at the top of the reused reduction.
- **Causal temporary visibility for sliding attention (plan row 4), attempted, reverted.** See
  section 5.1 for the evidence and the recommendation.

A related finding, not an attempt: **the tanh-GELU primitive already exists.** `ops::gelu` carries a
`GeluMode::Tanh` whose formula is exactly PyTorch's `gelu_pytorch_tanh`
(`0.5*z*(1 + tanh(sqrt(2/pi)*(z + 0.044715*z^3)))`), qualified by its own FP64 oracle in the suite,
and the device helper is reusable. Row 3 therefore needs no new activation mathematics — only the
fused epilogue and the plumbing that selects it, since the `linear_swiglu` routes hard-code `silu`
at 37 call sites across about ten tuned files.

- **Fused GeGLU for the NVFP4 `linear_swiglu` parent: scoped, not started.** The activation's home is
  each route's epilogue, and the four NVFP4 route files carry eight `silu` sites between them:
  `nvfp4_linear_swiglu_decode.cu` (the kernel body and its launcher), `nvfp4_linear_swiglu_small_t.cu`
  (the `SwiGluTile` member and its launcher), `nvfp4_linear_swiglu_w4a4.cu` (`Nvfp4SwiGluOutput::combine`
  and its launcher) and `nvfp4_linear_swiglu_w4a4_tma.cuh` (four sites in the TMA epilogue, launched
  through `nvfp4_linear_swiglu_w4a4_tma_launch.h`). The activation has to reach them as a kernel
  argument: `nvfp4_linear_swiglu_plan.h`'s launch declarations, `nvfp4_linear_swiglu_dispatch`'s five
  cases, and `linear_swiglu`'s wrapper in `src/ops/wrapper/linear_swiglu.cpp`. No new formula is
  needed — `gelu_one<true>` is it — and support must be all-or-nothing per parent, because an
  activation that depends on the resolved token range would work at one batch size and throw at
  another, which is worse than not shipping it. Until then, the unfused path exists:
  `gelu_mul` on the two halves of a projection output.

### 8.1 The stored bindings (read from the artifact, not from the recipe)

Read with `python -m tools.artifact.inspect out/gemma4_31b_base_nvfp4.ninfer --objects --bindings`.
833 bindings: 50 sliding layers x 14 + 10 global layers x 13 + 3 text-level, which matches the
structure below exactly.

| Logical name | Shape | Stored as |
|---|---|---|
| `text/token_embedding` | 262144 x 5376 | `fp8_e4m3fn_row_bf16` |
| `text/output_head` | 262144 x 5376 | **BF16**, not the FP8 the plan's row 1 lists |
| `text/final_norm` | 5376 | BF16 |
| `text/layers/{l}/input_norm`, `post_attention_norm`, `pre_feedforward_norm`, `post_feedforward_norm` | 5376 | BF16 |
| `text/layers/{l}/layer_scalar` | 1 | FP32 |
| sliding `attention/query`, `key`, `value` | 8192, 4096, 4096 x 5376 | one FP8 object of 16384 rows, sliced at rows 0 / 8192 / 12288 |
| global `attention/query`, `key` | 16384, 2048 x 5376 | one FP8 object of 18432 rows, sliced at rows 0 / 16384; **no `value` binding** |
| `attention/query_norm`, `key_norm` | 256 sliding, 512 global | BF16 |
| sliding `attention/output`, global `attention/output` | 5376 x 8192 / 5376 x 16384 | FP8 |
| `mlp/gate`, `mlp/up` | 21504 x 5376 each | NVFP4, two halves of one 43008-row object |
| `mlp/down` | 5376 x 21504 | NVFP4 |

Three consequences:

- **The query, key and value parameters are slices of a fused parent**, not separate objects, so a
  binding must request the *slice* shape (8192 x 5376 and so on) and let the artifact layer resolve
  the row offset and the scale plane. The absence of a `value` binding on global layers is the
  `attention_k_eq_v` structure showing up in the artifact, and it is what a loader must branch on.
- **The MLP is bound as two halves** (21504 each), not as the single 43008-row parent the plan's
  row 1 lists. The natural route is therefore the unfused one — a linear per half and `gelu_mul`,
  both of which exist — and `n21504/k5376` is the shape that is actually needed. The fused
  `n43008/k5376` stays registered for a recipe that fuses the halves.
- **The head is BF16**, so the FP8 head shape registered for it is not what this artifact uses.

**Owner decision on the attention geometry (see 5.1).** Gemma 4 gets its own attention Op in new
files rather than a refactor of the shared body: the sliding profile first (D 256, 32 query heads,
16 KV heads, window 1024, causal, scale exactly 1.0), then the global one (D 512, 4 KV heads, the
compact row). The shared body and Qwen's dense route stay untouched. The cost is real and accepted:
the softmax core is implemented twice rather than parameterized once.

**Blocking finding on the attention input projections (round 18).** The artifact binds
`attention/query`, `attention/key` and `attention/value` as *slices* of one fused FP8 object, which is
what the converter's grouping produces and what llama.cpp matches by reading a single `wqkv` tensor in
one matmul. But nothing in this runtime can turn those slices back into a parent weight:

- `ops::prepare_linear_weight` refuses a slice with "this native Weight input requires a complete
  FP8/NVFP4 parent".
- `ops::prepare_attn_input_proj_weights`, which exists to combine exactly these three inputs, refuses
  the combination with "QKV input projection: unsupported logical geometry" because its registered
  geometries are Qwen's.

So a Gemma layer cannot consume the artifact's per-projection bindings at all. The fix belongs in the
converter: bind one parameter for the fused parent (sliding 16384 rows = 8192 + 4096 + 4096, global
18432 = 16384 + 2048) instead of three slices, which is also the shape `n16384_k5376` and
`n18432_k5376` that the linear registry expects. The recipe's `group(selectors, shape=...)` looks like
the mechanism, since grouping without a shape is what produced the physical parent with separate
logical bindings; that needs verifying by re-converting and re-inspecting.

Two corrections to the layer composition came out of reading llama.cpp at the same time: the layer's
output is the residual stream itself (the *next* layer applies its own `attn_norm`, so a layer must
not norm its own output), and the sliding layers norm V weightlessly (`ggml_rms_norm(Vcur, eps)`),
which the first forward attempt omitted.

**The MLP binding is two slices, and that is the last blocker on a token.** Re-inspecting the
re-converted artifact: `text/layers/N/mlp/gate` covers bytes [0, 115605504) and `mlp/up` covers
[115605504, 231211008) of *one* object, exactly as the attention's q/k/v cover three ranges of one
object. The recipe's `group(selectors, shape=...)` does not merge logical bindings, only the physical
packing, so it changed nothing: a layer's MLP is still two parameters.

That is fine for the attention, whose parent is FP8 RowScale and whose slices the runtime now accepts,
and fatal for the MLP, whose parent is BlockScaleK16M128x4 with a swizzled scale plane. Two ways out,
in order of alignment with the plan:

- Teach the runtime to slice a *128-row-aligned* NVFP4 parent as well, which should be expressible
  because the slice boundaries land on the swizzle's own block grid (both halves here are 21504 = 168
  times 128 rows). This keeps the plan's NVFP4 MLP layout and is the smaller change.
- Or emit the MLP in FP8 like the attention, which makes the slices work through the mechanism that
  already exists at the cost of the plan's L1 layout for the MLP.

The forward is written, compiles, and fails exactly here: `text/layers/0/mlp/gate: logical shape
differs from Binding coverage`. Everything before it — the attention path, the norms, the RoPE, the
sandwich and the fused projections — runs.

## 12. A sliding layer runs end to end

`forward_sliding_layer` now executes one sliding decoder layer over the produced artifact. The
verification is structural rather than numerical: 50 sliding layers run in sequence, every hidden
state is finite, the largest magnitude is 6.59 from an input whose scale is the embedding's own
(hidden times the square root of the hidden size), a global layer is refused by the sliding path, and
the fast suite is green at 166 tests. The magnitudes are *not* checked against a reference; the
perplexity route is the gate for that, and nothing here substitutes for it.

Three things about the artifact and the runtime came out of this, and two of them cost earlier rounds:

- **The MLP's gate and up projections are separate objects** (727 objects for 833 bindings, up 60 from
  the packed version). Packing them bound each half as a row slice of a `BlockScaleK16M128x4` parent,
  and such a slice cannot re-derive the swizzled scale plane it shares with its sibling, so no
  consumer could execute it. `recipe.group(selectors, shape=...)` does not merge logical bindings,
  only physical packing, so the fix had to come from the builder, which no longer groups them. Each
  half now carries its own per-tensor scale as a side effect, and separate objects are what the
  reference engine's converter produces.
- **The FP8 launcher demanded a parent's absolute plane layout.** It required `qdata` to be the
  payload base and `scales` to sit at `align_up(n*k, 256)` from it, which no row slice can satisfy —
  so every sliced FP8 projection was rejected at launch. The check is now relative: the planes must be
  ordered and inside the payload, which is what the kernel actually indexes. The four FP8 linear
  tests still pass, so complete weights take the same qualified route.
- **`DeviceArena` accumulates, so a layer must scope its own scratch.** The `std::bad_alloc` that cost
  two earlier rounds was this and not the op sequence: the first diagnosis blamed a workspace that was
  undercounted by one hidden-sized buffer, and correcting the count changed nothing because the real
  fault was reuse across layers. The layer now opens a `Scope`, which also makes
  `sliding_layer_workspace_bytes` bound one call rather than a whole pass.

Still open: the global (full-attention) layers, which need the compact K/V path, the KV caches, the
Program with its state, workspaces and CUDA graphs, the engine's load path and architecture dispatch,
and the output head.

## 13. The whole decoder stack runs

`forward_layer` dispatches on each layer's attention kind, and a pass over the artifact now runs all
60 layers in order — 50 sliding and 10 global — sharing one arena, every hidden state finite, the
largest magnitude 5.97, and a layer index outside the stack refused. The fast suite is green at 166
tests.

**The global path is the compact one.** Its layers store no value projection because `attention_k_eq_v`
holds: the value side is the key normalized *without* a weight, read from the same raw projection, and
a second, weighted `key_norm` pass produces the key. Proportional RoPE takes `rope_angles` pairs
against the full 512-wide head, so the pair partner of dim i is i + 256 and the dims outside those two
runs carry no rotation — which is exactly the pair structure `compact_kv_rows` stores, giving the
640-value rows the plan describes. On the non-rotated dims the key the score needs is the value times
the key norm's weight, so the query carries that weight instead (`w_kn`, two `scale_columns` calls over
[64,256) and [320,512)).

**The layer scalars verify the load path.** The plan's analysis of this checkpoint records layer 59 at
0.0364 and the others in 0.44–0.99, and those values are what scale the whole residual stream once per
layer — which is also why the stream shrinks from the embedding's scale to a few units. The model reads
back **0.036377 for layer 59** and a band of **[0.036377, 0.992188]**, matching the independent
prediction. So the magnitudes that looked suspicious earlier are expected behavior, and the load path
at the end of a 60-layer stack is doing what it should.

**A note on shape order, because it cost a false alarm.** `Tensor` shapes here are column-major: the
first axis is contiguous. So the attention ops' `[D, Hq, T, B]` places a head's `D` values in one
contiguous run, which is the *same* layout as the checkpoint's row-major `(head, dim)`. Reasoning from
the plan's `[16 heads, 256]` notation alone suggested a transpose was needed between the projections
and the attention; the kernel's own addressing comment (`d + D * (head + Hq * column)`) shows there is
not, and `rope` and `scale_columns` agree. Read a shape against its convention before concluding a
transpose is missing.

Still open, in the order they gate a running model: the KV caches (the sliding ring and the global
pages) and the windowed/compact attention over more than one token, the Program with its state,
workspaces and CUDA graphs, the engine's load path and architecture dispatch, the output head with its
soft cap, and the numerical gate — the perplexity route against a reference — which nothing here
substitutes for.

## 14. Cache-backed decoding

`KvCache` gives a sequence its attention state, layer by layer, and `forward_layer` now writes the
token's rows into it and attends over everything it makes visible. Sliding layers keep a ring of
`sliding_window` key and value rows, so a token's slot is `position % window` and the query sees
exactly the tokens still inside the window. Global layers keep one compact row per token at the
token's own position, up to the cache's capacity.

**Unwritten slots need no bookkeeping.** A slot no token has written carries `kUnwrittenPosition`
(2^30), and each Op decides visibility by comparing positions: the sliding Op needs
`0 <= pq - pk < window`, which a far-future key fails, and the compact Op needs `0 <= pq - pk`, which
it also fails. So a caller can hand the *whole* cache to either Op without tracking how much is filled
— no fill counter, no per-step position rewrite.

**What the test verifies.** Eight tokens through all 60 layers, then three invariants that together pin
the ring, the masking, and the cache's involvement: replaying the last token over the same rows and
input reproduces its hidden state bit for bit; filling *every* unwritten slot with a large pattern
leaves that hidden state unchanged; and the control, corrupting slot 3 — a token the window still sees
— *does* change it. The control matters: without it the second check would also pass if the cache were
ignored altogether.

**Limits, stated plainly.** Capacity bounds the global layers' tokens (64 here), which the plan's
paging is what lifts; sliding layers are unbounded because the ring is exactly the window. Tokens are
processed one at a time — correct, and equivalent to a batched prefill because each Op already takes
the whole cache and its own position array, but not yet measured as a prefill. And the numbers are
still only structurally checked: finiteness, bounds, and these cache invariants, not a reference.

## 15. The layer forward against an independent reference

`tools/verify/gemma4_layer_reference.py` recomputes one decoder layer from the BF16 checkpoint in
FP32 — independently of the engine and of the Ops — and compares it with the engine's own output on
exactly the same input, which `ninfer_gemma4_layer_test` dumps.

**Result: structural agreement on both layer kinds, over four cached tokens.**

| Layer | Kind | worst cosine | worst relative L2 |
|---|---|---|---|
| 0 | sliding | 0.999996 | 0.0030 |
| 5 | global | 0.999999 | 0.0019 |

The engine's weights are FP8 row-scale and NVFP4, so a fraction of a percent is the quantization floor,
not slack: the check's job is to rule out a *structural* error — a transposed layout, a missing
rotation, a mis-grouped head, a score computed without the key norm's weight — each of which moves the
cosine far below 1.

What the four tokens cover: the input norm, both projections, the per-head q and k norms and the
weightless value norm, both RoPE kinds, the head grouping, the causal attention scores and softmax, the
output projection, both sandwich norms, GeGLU, the MLP, and the layer scalar. The multi-token case is
what makes the attention real — with a single key every score would be 1 and neither the sliding dot
product nor the global compact score with its `w_kn` prescale would be exercised at all.

What it does not cover: the sliding window's boundary (needs more than 1024 tokens), the cross-layer
composition (errors would compound), the final norm and the head, and the quantized routes' *quality*
as opposed to their structure. Those are the perplexity route's business.

### 15.1 The output head, and a realistic input

`forward_head` is the final norm, the projection through the tied embedding matrix, and the logit soft
cap. Its projection is BF16 `[262144, 5376]`, so the BF16 linear needed the shape registered
(`shapes/n262144_k5376.cu`); the schedule set is copied from an existing shape and is untuned, which
follows the plan's ordering of correctness before speed.

The layer stack's real input is the embedding of a token scaled by the checkpoint's embedding scale,
which is the square root of the hidden size (73.321211 for 5376) — the text config does not carry that
value, the converter derives it. Feeding that instead of white noise puts the run at the magnitudes
service sees, and the head can then be compared on the engine's own layer output, so the head's error
is isolated from the layers'.

| Check | Cosine | Relative L2 | Worst max abs diff |
|---|---|---|---|
| Layer 0, sliding, 4 tokens | 0.999857 | 0.0170 | 0.129 |
| Layer 5, global, 4 tokens | 0.999857 | 0.0169 | 0.329 |
| Head (either layer's output) | 0.999989 | 0.0024 | — |

The relative L2 is higher here than the 0.3% the white-noise runs showed, and the reason is worth
stating rather than glossing: the *absolute* worst error is smaller (0.129 against 0.224), but a real
activation's layer output has a smaller norm, so the same absolute error is a larger fraction of it.

**The head check is weaker than the layer check, on purpose stated plainly.** One layer of 60 does not
produce a final hidden state, so the logits run past the soft cap and `|logits|max` sits exactly at
30.000. What that verifies is the sign structure — 262144 values agreeing — not the magnitude
distribution, and the top-1 token disagrees between the engine and the reference because every
saturated value is ±30 and a 0.2% difference decides which one caps first. A deeper stack would make
that comparison meaningful.

**A comparison must refuse stale inputs.** An early run of this check compared the reference against a
dump a *failed* engine run had left behind, and reported a relative L2 of 0.898 for a layer that
actually agrees to 0.017. The reference now refuses to run unless both dumps exist.

## 16. The model generates, and the reference picks the same token

`ninfer_gemma4_generate_test` runs the whole pipeline end to end — the embedding, sixty cached decoder
layers, the head and a greedy argmax — for a prompt given as token ids, and tokenizing and detokenizing
happen outside it. The prompt "The capital of France is" (five tokens, `818 5279 529 7001 563`) produces

    The capital of France is France is France is France is France is

which is real English in the right semantic domain, so the pipeline works — and it says *France* where
*Paris* is expected, and then loops. That is exactly the kind of result that needs an outside opinion
rather than a guess, so `tools/verify/gemma4_prompt_reference.py` runs the same prompt through all
sixty layers with the checkpoint's BF16 weights and reports what it would pick:

| | Next token | Its logit |
|---|---|---|
| Engine (FP8 attention, NVFP4 MLP) | 7001 | 17.63 |
| Reference (BF16, FP32 math) | 7001 | 19.19 |

**They agree on the token.** The base checkpoint — this artifact is converted from the base model, not
an instruction-tuned one — greedily continues "…is France is France…", and the engine reproduces that
choice. So the loop is the checkpoint's behavior at this prompt, not an engine defect, and the 8%
logit gap is the quantization's.

The reference also shows why the one-layer head check saturated while this one does not: a full stack
ends at `|h|max 7.8` with `|logits|max 21.05`, inside the 30.0 cap, whereas a single layer's output is
not a final hidden state at all.

**What this does and does not establish.** It is symbol-level agreement between the engine and an
independent implementation of the whole model, which no single-layer check can give. It is still one
prompt, the reference is still my own reading of the architecture rather than HuggingFace's code, and
the logit gap is consistent with the quantization without being attributed to FP8 versus NVFP4 versus
rounding. A raw prompt is also the right input here: the checkpoint's chat template belongs to the
instruction-tuned model, which this artifact is not.

## 17. The Program owns the sequence

`gemma4::Program` is the model's own execution object: it holds the KV cache, the per-layer scratch
arena, the token and hidden buffers and the logits buffer, and exposes `reset`, `position`, `prefill`,
`decode` and `logits`. It shares no device allocation with any other Program, and it takes the model
and the execution view per call, so it is bound to streaming but not to a stream. Sampling stays out of
it on purpose — the Program produces soft-capped logits and the caller decides what to do with them.

The generation logic used to live in the test that drove it; it now lives here, which is where the
architecture document puts mutable state and execution composition. The verification is that it
reproduces the previous behavior exactly: the same prompt yields the same tokens *and the same logits*
— 7001 at 17.625, then 563 at 20.875 — through the Program.

**The engine still cannot load this model, and the reason is architectural rather than small.**
`runtime::ModelInstance` declares `ModelContract = models::qwen3_5::RuntimeTypes` and holds a Qwen
`Model`, `Parameters`, `Frontend` and `Program` by name, so `EngineCore<Instance>` is only instantiated
for Qwen. Reaching `ninfer` and `ninfer-perplexity` needs the plan's P2 boundary extraction: a Program
contract that both models satisfy, extracted one boundary at a time with the real-model Qwen tests as
the guard — which the plan's own risk table names as the thing that can destabilize the shipped models.
That is the next step, and it is why the Program was worth building first: it is the piece the engine
will call, and it is verifiable on its own today.

## 18. Causal scoring, and the missing BOS that made everything look broken

`Program::causal_score` returns the natural log-probability of every target token from `first_target`
on, which is what `ninfer-perplexity` consumes: `ScoreAggregate::add` accumulates
`total_nll -= logprob` and `ppl()` is `exp(mean_nll)`. The probability is taken from the soft-capped
logits of the preceding position — the distribution the model would sample from — and normalized over
the whole vocabulary in double precision.

**The first measurement said the model was broken, and it was the harness.** Scoring a paragraph of
plain English gave mean NLL 9.96, perplexity 21092. The independent reference, which shares no code,
agreed it was broken — 11.21 and 74217. Two independent implementations agreeing on an absurd number
is a signal about the *input*, not the math: the corpus was tokenized without a leading `<bos>`, which
Gemma requires, and which `tokenizers` does not add from `tokenizer.json` alone.

With the BOS the same corpus scores **mean NLL 2.676, perplexity 14.53**. The per-token
log-probabilities show the model doing exactly what it should: −0.0097 for "France" after
"capital of", and −0.0049 for "Paris" after "The capital of France is". Generation agrees — the same
prompt now produces

    The capital of France is Paris.

**This corrects section 16.** The looping "France is France is" was not the base checkpoint's
character, as that section concluded; it was the un-BOS-ed prompt, and the reference agreed with the
engine there because it was fed the same ids.

**Engine against reference, position by position** on 70 targets of the same corpus:

| | Mean NLL | Perplexity |
|---|---|---|
| Engine (FP8 attention, NVFP4 MLP) | 2.676 | 14.53 |
| Reference (BF16 weights, FP32 math) | 2.878 | 17.77 |

Correlation 0.9824, mean difference 0.20 nats, worst position 3.70 nats, and the first two positions
agree to 0.02. The gap is the size the single-layer checks predicted would accumulate over sixty
layers, and it is *not* attributed: quantization, the engine's BF16 activation storage at every layer
boundary, and the reference's own FP32 assumptions are all candidates. Note the direction — the
quantized engine is the *more* confident of the two, which is not what quantization alone would do, so
some of the difference is probably the reference.

What remains before this is P3: scoring runs through a test binary, not the public Engine, so
`ninfer-perplexity` cannot drive it yet. `runtime::ModelInstance` is still typed to Qwen, and the
Engine's scoring core reaches a Program through `ModelContract::PreparedPrompt` plus `causal_score`,
`memory_summary` and `reset_memory_peaks` — a small enough surface that the P2 boundary is now a
bounded piece of work rather than an open-ended one.

## 19. The Engine's instance boundary, second implementation

The Engine's cores are templated on an instance: `EngineCore<Instance>` for generation and
`CausalScoreCore<Instance>` for scoring, and `runtime::ModelInstance` is the only thing they have ever
been instantiated with. `runtime::GemmaInstance`, `runtime::GemmaEngineProgram` and
`runtime::GemmaPreparedPrompt` are now a second implementation of the same boundary, in the runtime
layer beside the Qwen one rather than in the model, so `models/gemma4` stays free of engine types.

Sizing it first paid off: the Engine reaches through `active` in only ten places (six `frontend`, four
`capacity`), and the scoring core asks an instance for exactly three things — `causal_score`,
`memory_summary` and `reset_memory_peaks` — plus a `PreparedPrompt` and a
`KvCapacityResolution`. That is small enough that the boundary can be widened one piece at a time.

**Verified by driving the Engine's own core.** `ninfer_gemma4_score_test` with
`NINFER_GEMMA_ENGINE_CORE=1` instantiates `runtime::CausalScoreCore<runtime::GemmaInstance>` and runs
the same corpus through it, then compares entry by entry against the direct `Program::causal_score`
path: **70 of 70 positions identical**, and the same mean NLL 2.676 and perplexity 14.53. So the
boundary is real and behaves, without the Engine itself having changed.

Two things are deliberately absent and say so. Per-position logits are not produced, and the adapter
*throws* if a `LogitsSink` is passed rather than scoring without it — those rows are what the KLD route
reads, and silence there would look like a working KLD path. And `memory_summary` reports an empty
summary: the paged-KV accounting the Qwen program fills does not exist here, so zeros are honest where
invented numbers would not be.

The test also had to be restructured for a physical reason worth recording: two 24 GB models do not fit
beside each other on a 32 GB device, so the direct path runs in its own scope and releases its model
before the instance loads one.

What remains for P3: `Engine::Impl` holds `active` and `core` as concrete Qwen types, so the dispatch
is the next step — an alternative in each variant, a helper for the ten `active` sites, a branch in
`construct_model` on the artifact's architecture, and a frontend for this model. The artifact already
carries its tokenizer resources, which is the part that would otherwise need a re-conversion.

## 20. The Engine drives this model

`Engine::load` now probes the artifact's declared architecture and takes a Gemma branch: it loads the
model through `gemma4::load_model`, wraps it as `runtime::GemmaInstance`, and constructs
`runtime::CausalScoreCore<GemmaInstance>` instead of the Qwen cores. `Engine::Impl::active` and
`Impl::Core` are variants over the two model kinds, the ten places the Engine reached through `active`
became visits, and `PreparedPrompt::Impl` holds whichever model's prepared prompt it was given, since
only that model's core can consume it.

**Verified through the public API**, not through a copy of the path:

    the Engine, through its public API, matches the Program on 70 positions

with the same mean NLL 2.676 and perplexity 14.53 the direct route produces. So an artifact with
`architectures: ["Gemma4ForCausalLM"]` loads, is recognized, and is scored by the Engine's own variant
dispatch and scoring core.

Four things are deliberately narrow, and each says so rather than pretending:

- **Generation is refused, loudly.** A Gemma artifact with `EnginePurpose::Generation` throws "this
  artifact is a Gemma model, which the Engine can score but not yet generate". The generation core's
  Programme surface is fourteen methods of leases, capture, output sessions and context transactions,
  and none of that exists here yet.
- **A prompt cannot be built from text or media.** `GemmaFrontend::prepare` and `tokenize_text` throw;
  only `prepare_tokens` works, which is what scoring needs because the caller already has ids. The
  artifact carries its tokenizer resources, so the data is there when a tokenizer frontend is written.
- **The logits sink is still refused**, so the KLD route cannot run for this model.
- **`LoadSummary` and `ModelMetadata` are mostly defaulted** for this branch — only the architecture
  string is set. They are Qwen-shaped, and filling them with invented numbers would be worse than
  leaving them empty.

Also removed: `gemma4::plan_load` had rejected every purpose except `Generation`, a placeholder from
when only the generation route existed. Scoring binds the same parameters — the head it reads is the
head generation samples from — so the guard was wrong, and the Engine's scoring load hit it
immediately.

## 21. The engine against HuggingFace's own implementation

Sections 15 through 20 compared this model against a reference I wrote. That is weaker evidence than it
looks: a misreading shared by the engine and my reference would pass, and I flagged that every round.
The alternative turned out to be on the machine all along. `transformers` 5.x ships
`transformers.models.gemma4` — the maintainer environment's 4.57.6 does not, which is why I had
concluded none existed — and `tools/verify/gemma4_hf_reference.py` runs HF's own model in BF16 on the
CPU, on ids the engine is then fed unchanged.

| Text | HF, BF16, CPU | Engine, FP8 attention and NVFP4 MLPs |
|---|---|---|
| France sentence | mean NLL **2.837** | mean NLL **2.509** |
| wikitext slice | mean NLL **8.215** | mean NLL **9.070** |

**Two conclusions, and the second one is the important one.**

First, the engine is faithful to the official implementation on both an easy and a hard text: the gaps
are 0.33 and 0.86 nats, in the direction and size quantization produces, and they are the only
quantitative end-to-end agreement this model has.

Second, **the perplexity that looked absurd is the model's own.** HuggingFace scores real prose at
NLL 8.2 as well — PPL 3,700 — so this checkpoint is simply weak on prose, and no amount of engine work
will change that. Every earlier symptom now fits: the "France is France is" loop at NLL 9.96, the
corpus scores near 9, and the checkpoint's own `.eval_results` card pointing at
`google/gemma-4-31B-**it**` rather than at a base model. What is *not* explained is why these base
weights behave this way — that is a question about the checkpoint, not about this engine.

One more finding worth recording: HF's tokenizer does not add the beginning-of-sequence token by
itself, and without it HF's own reference scores the France sentence at NLL 11.99. An implementation
that adds it is right here, and the engine does.

## 22. The open findings, resolved

### The long-text comparison is measured, not inferred

The HuggingFace oracle was extended past the 43-token runs that made the agreement look provisional, on
the exact texts the engine had already scored:

| Text | Tokens | HF, official, BF16 CPU | Engine, FP8 and NVFP4 |
|---|---|---|---|
| France sentence | 53 | mean NLL **2.6746** | 2.509 |
| wikitext slice | 335 | mean NLL **9.3332** | 9.0697 |
| wikitext slice | 1,514 | mean NLL **9.7178** | 9.6205 |

At 335 tokens the gap is 0.26 nats and at 1,514 it is 0.10, so the engine is faithful to the official
implementation at the lengths the perplexity tool actually runs. The 1,514-token case also matters for
another reason: it crosses the 1,024-token sliding window, so the ring, the wrap and the windowed
visibility are now numerically validated against the reference rather than only structurally.

### The reference I wrote is superseded where it mattered

Every end-to-end number in sections 15 to 20 came from a reference of my own, which cannot catch a
misreading I share with it. Those claims are now covered by HuggingFace's implementation instead, at
three lengths and on both an easy and a hard text. The hand-written reference still earns its place for
per-layer diagnosis, where HF's model would have to be instrumented, but it is no longer the evidence
for anything end to end.

### The real-model guard: 11 of 14

The ctest run reported 7 passed, 1 failed and 6 skipped. Four of those six run once their inputs are
supplied, and the way to supply them is worth writing down:

| Test | What it needs |
|---|---|
| `loading_real` | an explicit `--artifact <path>`, not the environment variable |
| `prefix_real` | `NINFER_PREFIX_REAL_SCENARIO`; the default `all` is invalid for this artifact and raises "constraints require default EOS, text output, no custom stops, and one output language" — `pressure-resume` works |
| `tools_real` | `NINFER_TEST_TOOL_REPORT=<path>` |
| `grammar_real` | `NINFER_TEST_SCHEMA_REPORT=<path>` |
| `dflash_real`, `dflash2_real`, `dflash_prefill_real` | remain skipped: the artifact does not carry the `dflash` component, and `dflash2` misparses its argument with an uncaught `stoul` |

### Scoring is launch-bound, by arithmetic

The tool reports 9 to 10 tokens per second. The per-token cost is linear in the tool's own numbers —
2.9 s for 52 tokens against 16.0 s for 334 — which is about 46 ms per token. The arithmetic work per
token is roughly 6 GFLOP through 60 layers and 600 kernel launches, which at any plausible rate is
under a millisecond. So the cost is launch and occupancy, not arithmetic: a thousand-fold under the
hardware's ability. The fix is the one the plan already names — a batched prefill so one pass covers
many tokens, and CUDA graphs in the Program — not a faster kernel.

### Generation through the Engine is scoped, not done

`Engine::load` refuses `EnginePurpose::Generation` for this model, and that is a deliberate refusal
rather than an oversight. `EngineCore` requires fourteen methods of a Program — `abort_pending`,
`advance_prefill`, `append_forced_tokens`, `commit`, `consumes_token_masks`, `decode`,
`device_kv_lease_settlement_tokens`, `fail_all_cleanup`, `has_context_transaction`, `memory_summary`,
`plan_request`, `reset_memory_peaks`, `resource_revision`, `skip_capture` — plus a dozen contract types
for admission, leases, capture and output sessions. Of those, decode, plan and commit are fundamental to
generation; leases, capture, forced tokens and context transactions are the Qwen MoE machinery this
model has no counterpart for. The work is to implement the fundamental three and refuse the rest
loudly, which is the next thing to do rather than something to hide.

### And the rule this round earned

Twice I explained a number before checking it, and both explanations were wrong. The oracle was three
directories away the whole time. Querying the reference comes before proposing a cause.

## 23. Batched prefill: the layer stack runs many tokens per pass

`forward_layer` now takes `first_position` and `tokens` instead of a single position, and every buffer
it allocates scales with the batch: the projections are `[K,T]`, the per-head tensors `[D,H,T]`, and
the attentions take four-dimensional views of those. Decode is the same call with one token, so there is
one code path rather than two that have to agree.

Two details make a batch behave like token-at-a-time. Every token's key and value rows are written into
the cache *before* the attention runs, and each query then sees what the cache makes visible to its own
position — which is what the Op's per-query visibility rule already means. And the cache's `mark_range`
records a run of positions, with the ring's rows adjacent unless the run wraps, so the common case is
one copy per plane and only the wrapped case falls back to a copy per token.

`Program` runs the stack in batches of 128. Its arena and logits buffer are sized for that, `logits()`
returns a view of the last column so callers that want one row still get one, and `causal_score` walks
the sequence in batches, scoring the targets each batch's columns predict. A column predicts the token
after its own position, so a batch of T covers T targets and consecutive batches tile the range without
gaps or repeats.

**Measured, on the 335-token wikitext slice that had been taking 16.0 s:**

| | Before | After |
|---|---|---|
| score rate | 9.4 tok/s | **135.9 tok/s** |
| mean NLL | 9.0697 | 9.0299 |
| HF's own | 9.3332 | 9.3332 |

A 14.5-fold speedup, and the answer is not merely the same but slightly closer to the reference. The
France corpus scores 2.6717 against 2.6761 before, so the change in accumulation order costs a
fourth decimal place.

**Verified as a generalization, not a change.** All seven Gemma real tests pass — load, model, forward,
layer, score, generate, tokenizer — the suite is green at 169, and the generation is character for
character what it was, "The capital of France is Paris.", which is the guard that matters for a change
to the code every path goes through.

**Two things this does not do.** The batch is a fixed 128 rather than sized to what the device can hold,
and it is the *prefill* rate that improved: decode still runs one pass per token, so a generation
measurement will look nothing like 135 tok/s. Both are honest limits rather than oversights.

**And a mistake worth recording.** The first verification run reported the generate test and four suite
tests failing with a shape error I had just fixed. They were stale binaries: I had built only the two
targets I named, so the tests still ran the old code. Rebuilding everything showed seven green tests and
a green suite. Building every target before believing a failure is cheaper than the diagnosis it
otherwise costs.

## 24. The logits sink, and what the rows revealed

`GemmaEngineProgram` used to throw when given a `LogitsSink`, so the KLD route could not run for this
model. It now hands over the soft-capped logits it takes its own distribution from: one row per scored
target, each `vocabulary` wide, side by side, as a view of the staging buffer — the layout the KLD route
reads, and the same values the log-probabilities are computed from.

**Verified against HuggingFace's logits for the same 71 tokens**, which the oracle writes out for
exactly this purpose:

| | |
|---|---|
| rows compared | 70 targets of 262144 |
| argmax agreement | 61 of 70, including the last row |
| per-row cosine | mean 0.972, min 0.748 |
| worst single-logit difference | 27.1 |

The alignment is right: the *chosen* tokens agree, the returned log-probabilities reproduce from the
rows, and the last row's argmax matches HF exactly. What does not match is the rest of the
distribution — a few logits per row differ by up to 27, which drags the cosine down while leaving the
mean NLL within 0.2 nats of HF's.

**That is a real characterization of this artifact, and it is not good news for KLD.** Per-layer hidden
states agree to 0.2%, the sequence-level NLL agrees to 0.1 to 0.3 nats, but the logit *tails* diverge
by far more than either would predict. KLD is tail-sensitive, so a KLD against a BF16 reference would
report a much larger gap than the perplexity does — which is precisely the comparison the plan's P4
exists to make, and the reason it ranks layouts on KLD rather than perplexity.

**Not resolved:** whether that tail divergence is the quantization's own behaviour or something the
engine does differently from HF. The evidence points away from a plain quantization effect — a fraction
of a percent on hidden states should not move individual logits by 27 — but I have not isolated it, and
an artifact-level KLD measurement is the way to do that.
