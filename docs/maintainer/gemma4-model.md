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
