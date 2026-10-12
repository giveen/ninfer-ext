# Gemma 4 model reference

Gemma 4 31B (`Gemma4ForCausalLM`, text `gemma4_text`) as NInfer executes it. This is the
mathematical authority for the implementation and the oracle; it is written from the checkpoint and
the plan that preceded this work. The assistant drafter (MTP speculation) is section 12a; the image
encoder and image prompts are section 12b.

**Status.** Text generation, chat, tools, thinking with a budget, image input, MTP speculation with
the official assistant drafter, causal scoring and serving run through the public Engine. Prefix reuse and
constrained output do not exist yet; section 13 lists what is not done.

## 1. Configuration (the instruction-tuned checkpoint)

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
- **Mask:** key position `p_k` is visible to query `p_q` when `p_k ≤ high(p_q)` and
  `p_q − p_k < 1024`, where `high(p_q) = p_q` (a causal band of 1024 with the query included) except
  for an image's tokens, whose `high` is the image's last position: an image attends bidirectionally
  within itself (`use_bidirectional_attention: "vision"`, transformers' `create_masks_for_vision_model`:
  AND(window, OR(causal, same image))). Only the soft tokens form the block; `<|image>` and `<image|>`
  stay causal. The window's lower edge stays at each token.

## 5. Global attention (10 layers): K = V and proportional RoPE

```text
r = W_k x                             # [4 heads, 512]; there is no W_v
n = r · (mean(r²) + ε)^(-1/2)         # weightless shared normalization
v = n

k = rope_prop(n ⊙ w_kn)
q = rope_prop(rmsnorm(W_q x, w_qn))   # [32 heads, 512]
o = W_o · softmax(q·kᵀ · 1.0 + causal) · v
```

- Query head h reads KV head ⌊h/8⌋; always causal, images included (transformers gives the global
  layers no image overlay).
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
  0.016 rad, wider than the deviation the rotation is held to. The per-pair frequencies are
  evaluated once per geometry on the host and passed by value, so a block does one double multiply
  and range reduction per pair and a float `sincosf` on the reduced phase; blocks split a token's
  heads four at a time. This took a decode RoPE launch from 12.5 to 2.4 us with every oracle
  case's error statistics unchanged.
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

### 5.1 Why the attention Ops are Gemma's own

The shared sliding Op's geometry (`kContextQueryHeadDim` and the head counts in
`ops/softmax_attention/common/context_query.cuh`) is compile-time and shared with Qwen's dense
route, it admits only windows 2048/4096 and scale 1/√128, and its rule on the current chunk is the
symmetric image-block rule. A causal policy hook routed through its mask sites was tried and
reverted: causal cases disagreed with the oracle even with one live row, cause unresolved. Gemma
therefore has its own `sliding_causal_attention` and `causal_compact_attention` Ops, at the cost of
a second softmax core.

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
`model.embed_vision.embedding_projection`; section 12b maps it.

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
also records the source digest, and requires each reference case to decode back to its text. The
decoder replaces **every** U+2581 in a piece: `▁them` is `" them"`.

## 8. Chat and output protocol

The checkpoint is instruction-tuned (`google/gemma-4-31B-it`: its card names `google/gemma-4-31B` as
the base model, it ships `chat_template.jinja`, and its EOS set is {1 `<eos>`, 106 `<turn|>`, 50
`<|tool_response>`}). The frontend (`src/models/gemma4/frontend.{h,cpp}`) renders the artifact's own
template with the Jinja engine and decodes generated tokens by Gemma's protocol:

| Element | Form | Engine meaning |
|---|---|---|
| Turn | `<|turn>role\n … <turn|>\n`; the assistant's role is `model` | `<turn|>` is a default stop token |
| Thinking | `<|think|>` at the top of the system turn enables it; the model then opens its answer with `<|channel>thought\n … <channel|>` | reasoning channel. With thinking off the template closes an empty channel itself, so the answer is content from its first token |
| Tool declaration | `<|tool>declaration:NAME{…}<tool|>` in the system turn | from `PromptOptions::tool_jsons` (OpenAI function shape) |
| Tool call | `<|tool_call>call:NAME{key:value,…}<tool_call|>` | `GeneratedToolCall` with JSON arguments; strings are delimited by `<|"|>`, keys are bare or quoted, values nest |
| Tool result | `<|tool_response>response:NAME{…}<tool_response|>` | `<|tool_response>` is a default stop token: the model hands control to the tool runner |

Rules the decoder applies: a call that does not parse, or names a tool the prompt did not declare,
is returned to content markers and all, with the reason in `ToolCallParseDiagnostics`; a stray
`<channel|>` outside a channel carries no text; raw output (`OutputOptions::raw`) splits nothing.
When the template closes an assistant turn that both answered and called a tool without opening the
next one, the frontend appends `<|turn>model\n` (the same correction llama.cpp applies).
`PromptContinuationMode::ContinueFinalAssistant` renders without the generation prompt and removes
the final turn's close.

Sampling defaults come from `generation_config.json` (temperature 1.0, top_p 0.95) and are the
same for both modes; its `top_k` 64 is narrowed to 20, the most candidates the sampling Op keeps.

Evidence: `tests/models/gemma4/test_frontend.cpp` renders six conversations (plain, system plus
history with thinking, Unicode/whitespace, tool declaration, tool round trip, tool round trip with
thinking) to exactly the text and token ids of transformers' `apply_chat_template`
(`tests/fixtures/gemma4/chat_template.json`, written by `tools/verify/gemma4_chat_fixture.py`), and
drives the decoder through channel splitting, raw output, stop strings, the budget, a parsed call, a
malformed call, an undeclared tool and a refused `tool_choice`.

## 9. Artifacts

All are produced from the instruction-tuned BF16 checkpoint at `/mnt/storage/models/gemma/full-31b`.
The `gemma4_31b_base` recipe and the L1 artifact carry "base" in their names from before the
checkpoint was identified; the bytes are the instruction-tuned model's. Only M1 is kept on disk; L1
and G0 were deleted once M1 superseded them and are rebuilt from their recipes when needed.

| Artifact | Recipe | Layout | Size |
|---|---|---|---:|
| `out/gemma4_31b_base_nvfp4.ninfer` | `gemma4_31b_base` | plan L1: MLP NVFP4 (converter-encoded, weight-only), attention and embedding FP8 rows, head BF16, norms BF16, `layer_scalar` FP32 | 22.8 GiB |
| `out/gemma4_31b_g0.ninfer` | `gemma4_31b_g0` | gewell's G0 mask: embedding and head BF16, attention FP8, MLP FP8 in layers 0–5 and every global layer with its predecessor, NVFP4 elsewhere | 27.3 GiB |
| `models/gemma4_31b_it_m1.ninfer` | `gemma4_31b_m1`, `--components text,vision,mtp --source mtp=/mnt/storage/models/gemma/assistant-31b` | measured layout `tools/convert/layouts/gemma4_31b_m1.json`: attention and head Q6 g64, each MLP group Q6 or Q5 g64 by measured sensitivity (173 Q6, 68 Q5 of 240 plus the head), embedding BF16 in pinned host memory; the assistant drafter Q6 g64 (`mtp/`, about 0.4 GB, bound only with `--spec mtp`); the vision tower Q6 g64 with BF16 patch and output projections (`vision/`, about 0.45 GiB, bound only with `--vision`) | 24.4 GiB file; 20.9 GiB on the device, 21.7 with the drafter and the tower |

M1 is the serving artifact. The layout came from `tools/verify/gemma4_sensitivity.py` (per-tensor KLD
against the BF16 checkpoint streamed from host memory, FP32 activations; BF16 activations put a 0.15
KLD floor under every candidate) and `tools/verify/gemma4_allocate.py`, then whole layouts scored end
to end, because single-tensor KLDs are heavy-tailed and do not add. On 4 x 2048 calibration tokens:

| Layout | Device GiB | KLD |
|---|---:|---:|
| L1 | 22.7 | 0.758 |
| G0 | 26.6 | 0.708 |
| Q8 everywhere | 30.4 | 0.083 |
| Q6 everywhere | 22.3 | 0.358 |
| M1 | 20.9 | 0.419 |
| Q5 attention, Q6 MLP | 21.4 | 0.590 |

The 4-bit MLP is almost all of L1's error (BF16 attention with NVFP4 MLPs: 0.711; Q8 attention and
head with BF16 MLPs: 0.053); attention below Q6 costs a lot; FP8 rows lose to Q8 g32 at equal size;
the FP8 embedding alone costs 0.19, so M1 keeps it BF16 on the host, where the lookup reads one row
per token.

Bindings, read from the artifact: q/k/v are row slices of one FP8 parent per layer (sliding 16384 =
8192 + 4096 + 4096 rows, global 18432 = 16384 + 2048 with no `value` binding); gate and up are
separate objects, because a row slice of a swizzled NVFP4 parent cannot re-derive its scale plane;
the tied head is stored as its own BF16 object. The artifact carries `tokenizer.json`,
`tokenizer_config.json`, `generation_config.json` and `chat_template.jinja`.

G0 leaves almost no device memory for KV beside the weights on a 32 GB card (about 1 GiB with a
desktop session), so it is a quality candidate (plan P4), not a serving configuration.

## 10. Execution

- **Program** (`src/models/gemma4/program.{h,cpp}`): up to eight lanes, each with its own `KvCache`
  and sampler state (device token counts), sharing the scratch arena, logits and token buffers.
  Every allocation is made at construction; `Program::device_bytes` sizes a configuration before it
  is built. A prefill pass covers at most `kPass` = 512 tokens and applies the head to its last token
  only; scoring runs passes of `kScorePass` = 128 with every column's logits, which sizes the logits
  buffer.
- **KV cache** (`cache.{h,cpp}`): sliding layers keep a ring of `sliding_window` + slack rows, the
  slack being the MTP draft count (zero without speculation). A pass of at most slack + 1 tokens
  writes first and attends to the ring alone: the rows it replaces are a window older than its first
  query. A wider pass attends to the ring as it was plus its own keys, which the sliding Op takes as a
  second key set, and writes its last ring of keys afterwards; so no write replaces a key one of its
  own queries still sees, whatever the pass width. A verify pass writes rows for drafts that may be
  rejected: with the slack, the older row a rejected draft replaced is invisible to every later
  query, and the rejected rows themselves are recorded as stale and forgotten before a wide pass
  could see them (a narrow one overwrites them first). Global layers keep one compact row per token at
  its position, up to the lane's capacity. Unwritten slots carry position 2^30, which both attention
  Ops treat as invisible; the Ops also never load a key no row of the tile can see, so an unwritten
  or stale slot's bits cannot reach the output.
- **Attention Ops**: `sliding_causal_attention` (D 256, 32/16 heads, window, scale 1.0, optional
  second key set) and `causal_compact_attention` over the 640-value compact rows share one
  tensor-core flash kernel family (`gemma_flash_attention.cuh`): BF16 m16n8k16 MMA with FP32
  accumulation, key tiles skipped by position before they load, and an online Softmax with P carried
  as a BF16 hi + lo pair. A pass with few rows (decode, MTP verification, short prefill tails) takes
  16 query rows per block with a key split and a combine kernel. From eight row tiles per KV head
  (512 rows for the sliding geometry, 256 for the global one) a pass takes the tiled kernel instead:
  each warp keeps 16 rows' S, Softmax state and P in registers, a block holds 64 rows (32 for the
  512-wide value, where two warps split the QK dims and exchange the halves of S, then each
  accumulates half the value), and key tiles stream through two cp.async stages into swizzled rows
  read with `ldmatrix`. Both routes give the same worst-case error against the FP64 oracle on every
  test case (the limit ratio is BF16 output rounding).
- **Linears**: M1's Q6 and Q5 g64 A16 routes carry Gemma's shapes. Single-token passes use the Q6
  GEMV and single-row Q5 instances at K 5376 and 21504; 2-8 token passes (MTP verification) use one
  8-token sliced MMA block for Q5, as Q6 already did; wider passes reuse the route lists tuned for
  the nearest Qwen shapes, untuned here. The drafter's BF16 shapes have single-token GEMV routes.
  The 5376-wide RMSNorm has a fixed-width route. GeGLU is unfused (`gelu_mul` on the two halves).
- **Engine** (`src/runtime/engine/gemma_instance.{h,cpp}`): `EngineCore<GemmaInstance>` and
  `CausalScoreCore<GemmaInstance>`, the same controller Qwen uses. Admission is root-only: the
  context cache is switched off for this model, so the identity candidate is always feasible and
  never expandable and seals directly. A decode round's plain rows decode together in one pass
  (`Program::decode` over a span of lanes: every column-wise step runs once over one column per
  lane, and each lane's cache writes and attention run per `PassSegment` against its own cache);
  the rows with drafts share one MTP round (`Program::speculate` over a span of lanes: each
  drafter step runs once over the lanes still drafting, which are a prefix of the batch in
  descending draft count, one verify pass carries every lane's [anchor, drafts] as its own segment,
  and each lane's sampler accepts against its own columns). Every lane samples on device, with one
  sync a round. A round's time is every lane's round time, so the adaptive draft policy learns the
  batched cost. The prefix-cache entry points the controller is compiled against (pressure planning,
  captures, checkpoint recovery, continuations) refuse by name. Forced control tokens (the thinking
  budget's `<channel|>` close) are appended as one prefill of the lane. Contract
  types this model never constructs are the Qwen contract's pure-data types; moving them into
  `runtime/contract` is the plan's D1-b extraction and is not done.
- **Refused options**: speculative backends other than MTP (and MTP's optimized proposal head,
  prompt-lookup drafts and draft trees), a KV format other than BF16, KV streaming, and an
  explicit KV capacity below `max_context`. Video, images in tool results and images in causal scoring
  are refused per request. `ninfer-perplexity` defaults to FP8 KV, so it needs
  `--kv-dtype bf16` for this model.
- **CUDA Graphs** (`use_cuda_graph`, on by default): the decode pass of each set of lanes is
  captured once, on its first round after the Program's first (eager) decode, and replayed for every
  later round of that set; the 16 most recently used sets are kept. The lanes are laid out in
  ascending order, so a set's layout does not depend on the order a caller names them in. That
  works because a round of one-token segments issues the same launches at every position: the token
  ids and the pass's positions and ring slots (`PassIndices`) are the only host inputs, written
  before the replay; every cache write is a device-indexed `scatter`; and the attention Ops are
  handed the whole sliding ring and the whole global capacity, whose unwritten and future slots are
  invisible (the compact Op's `indexed_keys` stops at the query on the device). MTP rounds and
  prefill passes run eagerly.

## 11. Verification

| Check | Evidence | Result |
|---|---|---|
| Tokenizer | `ninfer_gemma4_tokenizer_test` | 24 cases, exact ids and round-trip decode |
| Chat template | `ninfer_gemma4_frontend_test` | 6 conversations, exact text and ids against transformers |
| Single layers | `tools/verify/gemma4_layer_reference.py` (FP32 from the BF16 checkpoint) | cosine 0.99986 sliding and global, 4 cached tokens |
| Head alone | HF's own stack output through this model's head | per-row cosine 0.999996, argmax 70/70 |
| Whole model, scoring | `ninfer_gemma4_score_test` against `transformers` BF16 on CPU, 1,501 tokens | M1: mean NLL 4.075 against HF's 4.116 (L1: 4.148); mean per-target \|difference\| from HF 0.68 (L1: 0.79) |
| Batched against token-at-a-time | `NINFER_GEMMA_SEQUENTIAL=1` on the same text | M1: mean per-target difference 0.24 nats inside the window, 0.15 past it; the worst single target varies from 4 to 8 nats between builds that only reorder sums |
| Engine routes | the score test's Engine-core and public-API modes | identical to the Program on 1,500 positions |
| Generation | `ninfer` CLI, greedy | "The capital of France is Paris."; thinking mode streams reasoning separately from the answer |
| Serving | `tools/smoke/serve_contract.py --text-only --tools` | OpenAI chat/Responses, stored continuation, Anthropic, count_tokens, one tool round trip; plus streamed reasoning, Anthropic `tool_use` and two concurrent requests checked by hand; the same with `--spec mtp`, plus a thinking budget of 24 closing the channel |
| Drafter | `ninfer_gemma4_draft_test` dumps, `tools/verify/gemma4_draft_reference.py` runs transformers' `Gemma4AssistantForCausalLM` in FP32 on the same inputs (position 1,501, rings wrapped) | four steps: logits cosine ≥ 0.99996, argmax and top-5 identical, projected-state cosine ≥ 0.99997 |
| Batched decode | `ninfer_gemma4_batch_test`: three prompts from the 1,100-token file (one longer than the ring), 33 greedy tokens a lane | lane 0 identical beside different partners and with the lanes named in either order; a lane idle through batched rounds then identical to its own solo run; three lanes identical with and without CUDA Graphs; lane 0 batched identical to lane 0 alone in this run (the test only requires a near tie at a first difference, since T=1 and T>1 take different Linear routes); two lanes in batched MTP rounds at 7 and 3 drafts identical to each lane's own rounds and to plain greedy decode |
| MTP rounds | `ninfer_gemma4_mtp_test`: 384 greedy tokens after a 1,100-token prompt at seven drafts | every ring slot and global row holds its committed position (without the slack the same run leaves 50 wrong); against one fresh scoring pass the output disagrees 1-4 times, the same as plain decode (4) |
| Image preprocessing | `ninfer_gemma4_image_processor_test` on `tools/verify/gemma4_vision_reference.py prepare` (transformers' `Gemma4ImageProcessorPil`) | the resize rule identical over 22 sizes (edge branches, both orientations, up- and downscaling); lossless images' patches within one BF16 rounding (worst 0.004 uint8 steps); a JPEG differs, see 12b |
| Image encoder | `ninfer_gemma4_vision_test`, `gemma4_vision_reference.py compare` (HF's tower in FP32 on HF's patches) | Q6: mean token cosine 0.987 / 0.978 / 0.979 on a photo and two synthetic images (Q8 gave 0.991 / 0.985 / 0.984), against HF's own BF16 tower at 0.989 / 0.985 / 0.980 |
| Image prompt | `ninfer_gemma4_image_prompt_test`, `tools/verify/gemma4_image_prompt_reference.py` (HF `Gemma4ForConditionalGeneration` BF16 on CPU), two images and a question, 540 tokens, the second image across a pass boundary | prompt ids identical; 24 greedy steps: top-1 22/24, mean KLD 0.025 (Q8 tower: 23/24, 0.026) |
| Image mask | the same, at 34 of the first image's own positions, NInfer with and without the block bound against HF with and without `mm_token_type_ids` | Q8 tower: NInfer bidirectional at mean KLD 1.35 from HF bidirectional, 3.69 from HF causal; NInfer causal 0.92 from HF causal (HF's two runs differ by 2.35). Q6 tower: 1.83 and 2.75; causal 0.86 |
| Serving with images | `tools/smoke/serve_contract.py --tools` (image request included) with `--vision --spec mtp`; captions by hand | passes; a photo, a synthetic gradient (its rectangle placed correctly) and a two-image comparison described correctly |

**Per-position divergence from HF is large even where the averages agree.** On the 1,501-token text
L1's mean per-target |difference| from HF is 0.79 nats with a 99th percentile of 7.8 and a worst of 23,
for both of this engine's routes; M1's is 0.68, 99th percentile 6.6, and the same worst target. It is concentrated on low-probability targets: of the 761 targets
HF assigns more than −0.1, both routes put three below −5. Batched and token-at-a-time passes (a
tiled SIMT linear against a GEMV) differ from each other by a similar amount, which points at the
sensitivity of this artifact's logit tails to rounding rather than at a structural error, but the
cause is not attributed; the head alone is faithful. KLD against a BF16 reference will therefore be much larger than the NLL gap
suggests, which is what plan P4 (precision) addresses.

## 12. Performance

`ninfer_bench` on an RTX 5090, CUDA 13.4 runtime, BF16 KV, one request, three repetitions:

| Test | L1, untiled attention | M1, current |
|---|---:|---:|
| pp1024 | 327 tok/s | 2,388 tok/s |
| pp4096 | 220 tok/s | 2,108 tok/s |
| pp8192 | 161 tok/s | 1,837 tok/s |
| tg128 | 41.0 tok/s | 52.6 tok/s |

Later changes, measured as alternating builds in one session (current master between two runs of
the changed build; `-n 128 -r 3`, `-p 1024,8192 -r 2`, `-pg 8192,64`, MTP at seven drafts):

| Test | before | after | change |
|---|---:|---:|---:|
| pp1024 | 2,377 | 2,469 / 2,483 | +4.2% |
| pp8192 | 1,831 | 2,184 / 2,194 | +19.5% |
| tg128 | 53.00 | 56.49 / 56.70 | +6.8% |
| decode after an 8K prompt | 49.18 | 51.55 / 51.64 | +4.9% |
| tg128, MTP | 262.3 | 268.1 / 272.1 | +2.2% / +3.7% |

Decode gains come from the RoPE coefficient change (section 5) and the single-token Q6 direct
route (12c); prefill gains from the tiled attention kernel (section 10), which took attention from
34.5% to 17.8% of pp8192's kernel time: about 57 useful TFLOP/s for the global layers and 55 for the
sliding ones over the whole run, against 19 and 31-34 before. MTP acceptance is unchanged within
run-to-run variation (0.878-0.883 on both builds).

Decode is measured alone (`-n 128 -r 5`); right after the prefill tests in the same run it reads
about 1% lower. The decode ceiling from M1's 20.9 GiB of device weights at 1.79 TB/s is about 80
tok/s, so decode ran at about two thirds of it. The figures above it were taken before decode had
CUDA Graphs.

CUDA Graphs, measured in one session (graphs, `--no-cuda-graph` on the same build, the previous
commit, graphs again): tg128 56.09 / 56.54 against 55.55 eager and 55.99 for the previous commit;
decode after an 8K prompt 52.31 / 52.35 against 51.72 and 51.77; pp1024 2,515 / 2,511 against 2,493
and 2,490. The gain is small because an eager decode token's launch gaps are only about 0.5 ms
(nsys: an 18.18 ms span over 17.66 ms of kernels), and a replayed graph still pays a per-kernel
latency over ~1,370 kernels a token. The eager path itself measures 0.8% below the previous commit
at short context, where the whole-ring sliding scan launches its full key split over a ring that is
mostly unwritten; at 8K it is level. MTP throughput is unchanged in principle (its rounds run
eagerly) and measured 267.5-282.9 tok/s across the four runs, a spread the adaptive draft policy's
round count (57-60) alone explains. Greedy decode is token-identical between the graph and eager
paths (`ninfer_gemma4_generate_test`, 48 steps after an 1,100-token prompt) and against the
previous commit.

Decode attention issues each key tile as one block-wide batch of 16-byte `cp.async` copies. It used
to copy a row per warp at a time, paying a memory latency per row: ncu put the global layers' kernel
at 332 GB/s at 8K, stalled on long scoreboard. Under ncu's base clocks the global kernel went from
148.5 to 86.2 us and the sliding one from 22.7 to 20.6 us; the arithmetic is unchanged and greedy
decode is token-identical. Measured in one session (new, previous commit, new): tg128 56.37 / 56.15
against 55.50; decode after an 8K prompt 53.91 / 52.68 against 51.87; after a 32K prompt 50.29 /
49.89 against 45.49.

The key split then became a full wave: the split count is the blocks the device holds at once
(occupancy times the SM count from the execution view, capped at 512 so the workspace bound holds
on any device) over the row blocks, so the global layers run 42 splits as 168 blocks, one per SM,
rather than 64 as 1.5 waves; the sliding layers keep 16. The combine weighs the splits once in one
warp and sums them in parallel groups of float4 columns instead of a serial loop per column. Two
sessions against the cp.async commit (new, previous, new each): tg128 58.11 / 56.96 and 56.56 /
57.14 against 57.87 and 56.28; decode after an 8K prompt 55.32 / 55.11 against 54.19; after a 32K
prompt 52.19 / 50.94 and 51.75 / 51.91 against 50.80 and 50.01. The global kernel still waits for
each tile before computing (25% of DRAM peak under ncu); section 13 lists it.

Of a decode token's GPU kernel time (nsys), the Q6 and Q5 linears are 88%, RoPE (a generic kernel)
4.6%, attention 3.6%, and norms and element-wise kernels the rest. Kernel time is about 18 ms of the
19 ms per token under nsys, so launch gaps looked like about 5% (about 3% measured more carefully,
above); the linears' distance
from the weight-streaming rate is the larger lever. An 8K prefill at 128-token passes spent 46% in the Q6 MMA, 22% in the Q5
sliced-K MMA and 29% in attention; widening the pass to 512 tokens took pp8192 from 991 to 1,837
tok/s.

Concurrent decode shares one pass across lanes (section 10). A pass of T columns costs little more
than one: in the Linear benchmark the Gemma shapes take 1.0-1.6x their T=1 time at T=8 (21504x5376
61.4 -> 67.6 us, 5376x21504 65.5 -> 86.0). Through `ninfer-serve`, C greedy requests of 512 tokens
each (`ignore_eos`, short prompts, no MTP, BF16 KV, max context 2048), against the previous commit
where every lane ran its own pass:

| C | aggregate tok/s | per request | before: aggregate | before: per request |
|---:|---:|---:|---:|---:|
| 1 | 57.5 | 57.6 | 57.1 | 57.2 |
| 2 | 92.2 | 46.3 | 56.8 | 28.5 |
| 4 | 165.8 | 41.8 | 56.5 | 14.2 |
| 8 | 291.0 | 37.1 | 56.2 | 7.1 |

The repository's serving harness (`tools/bench/run_serve_concurrency.py`) does not run this model
(it requires a `prefill_signature` in the server's start log), so these come from a scratch client
with the same request shape. Attention is still per lane (60 layers x C launches a round); its
share of a C=8 round was not measured.

MTP rounds batch the same way (section 10). Same requests with `--spec mtp --draft-tokens 7`,
against the commit where every lane drafted and verified alone:

| C | aggregate tok/s | per request | before: aggregate | before: per request |
|---:|---:|---:|---:|---:|
| 1 | 108.1 | 108.5 | 108.7 | 109.2 |
| 2 | 162.3 | 85.1 | 95.1 | 47.9 |
| 4 | 251.4 | 66.6 | 96.1 | 25.1 |

At C=8 an MTP server does not start, before and after this change: the KV allocation runs out of
device memory at max context 2048 and also at 1024. Plain decode at C=8 and max context 1024 was
level across the change (288.1 against 290.4 tok/s, one run each).

## 12a. MTP speculation

The official assistant (`google/gemma-4-31B-it-assistant`, `Gemma4AssistantForCausalLM`) is four
layers at hidden 1024: three sliding and one global, with no key or value projections, a tied
262,144-row head without soft cap, and `pre_projection` [1024, 10752] / `post_projection` [5376,
1024]. Its layers attend to the target's cache: the sliding ones to the last sliding layer's ring
(58), the global one to the last global layer's compact rows (59), whose non-rotated key is the
stored value times layer 59's key-norm weight, which the drafter's query carries exactly as the
target's own global query does.

A round, for a lane whose last accepted token (the anchor, at position p) the target has not
consumed:

1. Draft K tokens greedily. Step 1 reads [scaled target embedding(anchor); the target's
   post-final-norm state at p − 1]; step i reads [embedding(draft i − 1); the drafter's projected
   state from step i − 1]. Every step uses RoPE position p and sees exactly the target keys before p
   (transformers' generator slices the shared KV to the sequence length, which on a rejection round
   still contains the rejected draft's row at p; NInfer does not show it).
2. Verify: one target pass over [anchor, drafts] at p..p+K with every column's logits.
3. Accept with `speculative_accept_greedy_drafts` under the lane's sampler: the longest greedy-draft
   prefix the target licenses plus its own correction or bonus token, which is exact for greedy and
   speculative rejection sampling otherwise.
4. Commit the kept n tokens: the target has consumed p..p+n−1, the drafter's next input state is the
   verify pass's column n − 1, and the rows past it are stale.

The draft length is the MTP ladder's: `--draft-tokens N --fixed-draft` drafts exactly N; otherwise
each lane chooses among {2, 3, 4, 7} ∩ below N, plus N, by expected committed tokens per second
(`qwen3_5::MtpDraftPolicy`, borrowed like the other Qwen contract types until the D1-b extraction).
Rung times are measured at startup on lane 0 and refined from the rounds that run.

Measured (RTX 5090, M1 with the drafter, 384 output tokens, one request, `ninfer` CLI decode speed;
sampled is the generation-config default at seed 7):

| Prompt | No MTP | K=3 | K=5 | K=7 | adaptive ≤ 7 | acceptance (adaptive) |
|---|---:|---:|---:|---:|---:|---:|
| code, greedy | 52.0 | 136.7 | 162.4 | 167.9 | 168.1 | 49.8% |
| story, greedy | 51.9 | 99.6 | 98.7 | 104.5 | 102.2 | 33.0% |
| explain, greedy | 51.7 | 120.4 | 134.6 | 139.9 | 139.5 | 39.6% |
| math, greedy | 51.5 | 154.7 | 197.7 | 228.0 | 228.5 | 72.9% |
| translate, greedy | 53.2 | 120.6 | 125.0 | 124.2 | 125.4 | 33.6% |
| code, sampled | 51.8 | 135.4 | 158.9 | 169.4 | 169.6 | 50.2% |
| story, sampled | 51.7 | 91.1 | 96.1 | 95.5 | 101.2 | 31.6% |
| explain, sampled | 51.5 | 124.0 | 134.6 | 142.6 | 135.0 | 37.6% |
| math, sampled | 51.4 | 145.1 | 192.5 | 239.2 | 239.6 | 77.3% |
| translate, sampled | 53.2 | 120.6 | 125.3 | 124.8 | 125.4 | 33.6% |

Adaptive is within 5.3% of the best fixed length on every prompt; the worst case is the sampled
explanation (5.3% under K=7), and it beats every fixed length on the sampled story. These numbers depend on the verify pass being flat in its width: a
target pass costs 17.7 ms at one token and about 20 ms from two to eight. Before the 2-8 token Q5
route moved to one 8-token block, five to eight tokens cost 27.7-30.6 ms, K=3 was the best fixed
length and adaptive lost 3-7% on prose; that state is not what ships. From the startup rung times,
a drafter step costs 0.7-1.1 ms; its 0.94 GB of BF16 weights are about half head, and it has not
been profiled further. The artifact stores the drafter in Q6 (about 0.4 GB): the target verifies
every draft, so the format moves only acceptance and step cost. Against BF16, two alternating rounds
of adaptive MTP over the five prompts, greedy and sampled, ran 4.3% faster on average with Q6 (best
+8.5%, worst -1.9% on sampled code) and 3.9% with Q8 (worst -2.2% on sampled story); acceptance fell
on prose (greedy story 33-34% to 28-31%) and held elsewhere. The table above was measured with the
BF16 drafter. Greedy MTP output can differ from plain greedy output after a
few hundred tokens because verify passes and single-token passes round differently; the check above
compares each against one fresh pass instead.

## 12b. Images

**Encoder** (`src/models/gemma4/vision.{h,cpp}`, transformers' `Gemma4VisionModel` and
`Gemma4MultimodalEmbedder`), for one image of a W x H patch grid, W and H multiples of 3:

```text
x = W_patch · p + T_x[column] + T_y[row]          # p: the patch's 768 pixels mapped to 2u/255 − 1
27 x:  h = x + rmsnorm(W_o · attn(...), w_pa)     # q, k: rmsnorm per head with weights, then 2-D
       x = h + rmsnorm(W_d (gelu_tanh(W_g n) ⊙ W_u n), w_pf),  n = rmsnorm(h, w_pre)
       attn: v = rmsnorm(W_v n) without weight; scale 1.0; every patch sees every patch
s = (√1152 · mean₃ₓ₃(x) − std_bias) ⊙ std_scale  # one soft token per 3x3 cell, row-major
features = W_embed · rmsnorm(s)                    # weightless; [5376] per soft token
```

The 2-D RoPE (theta 100) rotates x on dims (j, j+18) and y on (36+j, 54+j), j < 18, at
`100^(−2j/36)`. The converter stores q/k rows and their norm weights with the middle two 18-dim blocks
of every head swapped, which is exactly the layout the `rope` Op's Vision 2-D mode rotates (pairs (i,
i+36), axis 0 for i < 18); scores are unchanged because q and k move together. The MLP width 4304 is
stored zero-padded to 4352 (GeGLU maps zero to zero), so the groupwise routes' K alignment holds. q, k
and v are three 1152-row linears because each is normed per head before RoPE. The dense attention is
`softmax_attention` (D72/H16) at scale 1, a second registered profile; the pooling is
`vision_pool_standardize`. One image is encoded per call with exactly its own patches: no padding.

**Formats.** Patch and output projections BF16, every layer's linears Q6 g64. The tower amplifies
weight error: simulated with the converter's encoders, against the FP32 tower on a photo the mean token
cosine is 0.956 for Q4/Q5 (the Qwen tower's formats), 0.970 Q5, 0.989 Q6 (HF's own BF16 tower) and
0.999 Q8, and nothing downstream checks image features. Q6 costs about 0.45 GiB (Q8 0.6, Q4/Q5
0.36). Executed, Q6 lands slightly under HF's BF16 tower on the synthetic images (11): its answers
match HF as closely as Q8's did, while the logits at the image's own positions moved further.

**Preprocessing** (`src/models/gemma4/image_processor.{h,cpp}`, transformers' `Gemma4ImageProcessor`):
the largest aspect-preserving size whose sides are multiples of 48 within 280 soft tokens (2,520
patches), with the reference's zero-side branches and upscaling; torchvision's antialiased bicubic
(`src/media/decode/resize.h`, shared with Qwen); pixels to `bf16(2·(u·(1/255) − 0.5))`. Only the
280-token budget is served. JPEG decoding is FFmpeg's, which upsamples 4:2:0 chroma by nearest
neighbour where Pillow's libjpeg interpolates: on a 19 MP photo the patches differ by 0.75 uint8 steps
on average (8 at worst), and HF's FP32 tower fed NInfer's decode is at mean token cosine 0.957 from
itself fed Pillow's. The tower is that sensitive throughout: lossless images that differ only in one
BF16 rounding of a few thousand values give 0.995-0.998.

**Prompts.** The template writes `<|image|>` for each image part (a message with media is passed as a
parts list, as transformers' processor does); the frontend expands it to `<|image>` + one image token
(258880) per soft token + `<image|>`. Text that itself contains the placeholder is refused, as are
video and images in tool results. The Program writes the encoder's features over the image tokens'
embeddings (no embedding scale) and gives those tokens the image's last position as their sliding
upper bound (section 4); a pass never ends inside an image, and the Engine's prefill chunks snap the
same way, so an image's keys are always in the pass that attends them. Scoring refuses images: its
128-token passes are narrower than an image block.

**Cost.** Encoding takes 27-33 ms per image of about 2,300 patches on the RTX 5090 (Q8: 26-28;
since section 12c the Q6 routes are selected for these shapes rather than borrowed from Gemma's text
list, which removes 12% of the encoder's Linear time at 2,520 patches and 1.3x to 3.6x at the shorter
extents a non-square image produces, while the Q8 tower still wins on the widest count). A one-image prompt
of 280 tokens served with `--vision --spec mtp` has a TTFT of 192 ms; a 19 MP JPEG adds about 200 ms of
single-threaded decode and resize on the request thread (13).

## 12c. Linear route tuning (Q6 and Q5)

The Q6 and Q5 A16 routes for every Gemma 4 geometry were selected by a temporary private-launcher
sweep, then verified through the public Linear benchmark and the Op conformance tests. The sweep
harness is development scratch: it includes the private launcher headers, which
[linear-benchmark](../maintainer/linear-benchmark.md) forbids the retained benchmark, and it was
deleted when the sweep ended. It reproduced `bench/ops/linear_bench.cu`'s fixture and cold-cache
timing exactly, which the table below depends on: at four of six sampled points its production row
matched the public benchmark to the last 0.001 us, and within 3% at the other two.

**What was wrong.** `select_q6_gemma4` sent every Gemma text geometry, and (via the dispatch table)
every vision and drafter geometry, to the route list tuned for `N=34816, K=5120`. That list assumes
a row extent sixteen times wider than most Gemma shapes have, so from 32 to 384 tokens it leaves the
work to sliced-K routes with too few blocks, and its tile ladder starts where the geometry wants to
stop. The vision tower is the extreme case: an encoder call uses every patch of one image as its
token extent, so the extent is 9..2520 rather than a request length, and none of those extents
resembles a language-model pass.

**Measured, on the shipped route** (public Linear benchmark, median of 20 cold-cache samples,
before -> after):

| Geometry | Role | T | before | after | gain |
|---|---|---:|---:|---:|---:|
| `5376x21504` | MLP down, Q6 | 64 | 317.4 us | 165.9 us | 1.91x |
| `4096x5376` | sliding key/value | 64 | 81.9 | 43.0 | 1.90x |
| `2048x5376` | global key = value | 128 | 124.9 | 41.0 | 3.05x |
| `5376x8192` | sliding output | 64 | 122.9 | 63.5 | 1.94x |
| `21504x5376` | MLP gate/up, Q5 | 128 | 251.9 | 168.0 | 1.50x |
| `1152x4352` | vision MLP down | 384 | 100.4 | 47.1 | 2.13x |
| `1024x10752` | drafter pre-projection | 2 | 26.6 | 18.4 | 1.44x |
| `16384x1024` | drafter attention | 1 | 16.4 | 14.3 | 1.14x |
| `262144x1024` | drafter tie head | 1 | 161.8 | 143.4 | 1.13x |

The extents that carry the bulk of a real request were already the best registered route and are
unchanged: at T=512 and T=1024 the wide shapes measure 4096x5376 139.3 us, 2048x5376 139.3 us,
8192x5376 546.8 us and 21504x5376 1206.3 us before and after, and the Q5 pair likewise. The gaps
are in the prefill *tail*, in MTP verify widths, and in vision.

**Two corrections the sweep forced.** A first attempt gave `4096x5376` and `8192x5376` an 80-token
tile across 129..768 tokens; at 512 tokens that tile ends in a masked tail and cost 1.45x, and at
192/256 it lost to the inherited 96/128-token tiles. Both shapes now hand everything above their
measured crossover back to the inherited list, keeping a 1.12x gain at 384/768 unclaimed rather than
risking the much more common prefill width. The same review removed a sliced route from
`2048x5376` above 512 tokens, where the 128-token tile is 1.36x better.

**Vision has one more finding.** For a 128-token MMA tile, a patch count that is *not* a multiple of
128 ends in a masked tail of up to 127 columns, which costs more than the tile's width is worth:
at 2376/2400/2520 patches `t128` measures 65.5 us on `1152x1152` against 53.2-55.3 for a 96-token
tile, and `1152x4352` measures 229-233 against 180-182. Those two selectors therefore choose
`t128` only when the patch count divides by 128 and `t96` otherwise. `4352x1152` has no such
preference and keeps `t128`.

**What the sweep did not fix.** Decode is 88% Linear and runs at about two thirds of the
weight-streaming rate, but the sweep found only a 1.06-1.14x spread among the registered
single-token routes: at T=1 the shapes that matter are already at or near the best *registered*
candidate. The remaining gap needs new instances rather than a new selection - `4096x5376` at T=1
streams its 16.5 MB of weights at 806 GB/s against the measured 1674.5 GB/s read ceiling, so a
single-token route with more rows in flight is the next lever. That is not in this change.

**Single-token direct route (later change).** `q6_a16_direct_simt` is Q5's direct SIMT dataflow
over the Q6 planes: one row a 128-thread block, loads straight from global memory, the row unrolled
over a static K (5376, 8192, 16384, 21504; 48 registers, no spills). ncu had put the staged GEMV's
limit at shared-memory issue (MIO throttle) rather than DRAM. Public Linear benchmark at T=1,
before -> after: `16384x5376` 55.3 -> 47.1 us, `2048x5376` 14.3 -> 10.2, `21504x5376` 67.6 -> 61.4,
`5376x8192` 28.7 -> 24.6, `5376x16384` 53.2 -> 49.2, `5376x21504` 73.7 -> 63.5. The output
projections had no GEMV route before (the staged GEMV loses to the sliced-K route there, 32.8 vs
28.7 us), and the unfused sliding shapes gain too (`4096x5376` 20.5 -> 16.4, `8192x5376` 32.8 ->
26.6). The head stays on the GEMV: 686.1 -> 675.8 us is inside the noise band. Variants with 8
warps, 2 rows, 2 phases or a different residency were none faster and several slower, by up to
2 us on `2048x5376` and `5376x8192`. With the RoPE change above, tg128 goes 53.06 -> 56.69/56.71
tok/s (master between two branch runs) with pp1024 and pp8192 unchanged (2377-2396 vs 2384-2387,
1836 vs 1840).

**Coverage and limits.** `tests/ops/linear/test_q6_a16.cpp` and `test_q5_a16.cpp` already qualify
every Gemma geometry against the FP64 oracle, at T = 1, 2, 5, 17, 64, 128 and 1000 for the text
shapes, 9, 36, 630, 2340 and 2520 for vision, and T = 1 for the drafter, so the changed intervals are
covered rather than assumed. Sweep-to-sweep repeatability at a fixed point and route measured 0-5%
(with 4096x5376 at T=512 and 1024 showing 4% and 1.5% between runs of the *unmodified* binary), so
gains below about 10% are not distinguishable and no route was changed for one. Two routes that the
sweep suggested but that measured inside that band were left alone.

**Product effect.** Two alternating A/B rounds against the unmodified build (the workflow warns that
back-to-back runs drift by about 1% from heat, so the builds were swapped rather than compared across
sessions) measure the standard matrix unchanged:

| Test | before | after | |
|---|---:|---:|---:|
| pp1024 | 2289.3 / 2261.0 | 2259.2 / 2240.0 | 0.989x |
| pp8192 | 1719.6 / 1688.7 | 1703.1 / 1688.2 | 0.995x |
| tg128 | 48.5 / 47.9 | 47.9 / 47.7 | 0.991x |

That is expected rather than disappointing: `ninfer_bench` prefills at the 1024-token chunk the
Engine picks by default and decodes at T=1, and both were already the best registered route. The
change is worth what it does elsewhere, and elsewhere is measurable: `pp1150` - one full pass and one
126-token tail - with `--spec mtp --draft-tokens 5` measures 2012.8 -> 2113.1 prefill tok/s (1.05x)
and 201.4 -> 203.5 decode tok/s (1.01x), because that tail is 11% of the request's prefill work and
runs 1.7x faster on `4096x5376`. A request whose prompt divides evenly by 1024 sees none of it.

The absolute numbers are 5-7% below the figures in section 12, which were taken in an earlier session;
a mid-session comparison of unmodified builds drifts by up to 4% on repeated points, so the section 12
table and this one should not be differenced across sessions.

## 13. Not done

| Item | Why it matters | What it needs |
|---|---|---|
| Prefix reuse | an agent loop re-prefills its whole history every turn, at the prefill rates above | the D1-b move of the continuation, shared-prefix and capture types into `runtime/contract`; ring-snapshot checkpoints as the state image |
| Decode single-token routes | measured 2026-10-11, nothing left to take: the Linear benchmark's T=1 times fit `bytes / 1.52 TB/s + 3.9 us` across the Q6 shapes (2048x5376 to 21504x5376, within its 2.048 us timer steps), so streaming is about 91% of the 1674.5 GB/s read ceiling and the shortfall of the smaller shapes is a fixed per-launch cost; Q5 21504x5376 streams the same. Inside a CUDA-Graph decode step that cost is gone: at 8K the Q6 5376-row kernels take 3,250 us a token against 3,258 predicted from 1.52 TB/s alone, and Q5 gate/up 50.9 us against 49.9 | none on the GEMVs; fewer bytes (a smaller format) or fewer kernels are the levers |
| Batched attention across lanes | a decode round shares every weight read, but each lane's attention is its own launch (60 x C a round) | per-lane caches laid out for the Ops' batch dimension, worth it only if attention's measured share of a C=8 round is large |
| Tiled attention wave quantization | a 512-token pass gives the global layers 512 blocks of near-equal work, one per SM at a time: 3.01 waves run as 4 | finer work items for the last wave (a key split there with the existing combine), or a block small enough for two per SM |
| Global decode attention at long context | one key tile in flight per block, one block per SM (68 KB shared): 25% of DRAM peak at 8K | tried and dropped (2026-10-11): 16-key tiles through three cp.async stages (88 KB), scoring split over two halves of the dims, took the global kernel from 86.2 to 67.2 us under ncu's base clocks at 8K, but the bench was level at 8K and 32K and 1.1-1.5% slower after a 64K prompt. Per-tile fixed costs (a position round trip and four barriers) double with 16-key tiles; a next attempt should keep 32-key tiles, e.g. 8 query rows instead of 16 in shared memory to fit two stages |
| Fewer decode kernels | a decode token is ~1,370 kernels; graphs removed the host side of the launch cost (section 12) but not the per-kernel latency | fusing the per-sublayer norm, residual add and layer scalar; a first attempt failed the numerics (section 15) |
| Output constraints | refused | token-mask consumption in the Program |
| FP8 KV | refused | the codecs of plan row 7 |
| Image preprocessing off the request thread | a 19 MP JPEG costs about 200 ms of single-threaded decode and resize before prefill | Qwen's media worker pool and cache, shared |
| JPEG decoding as Pillow does it | FFmpeg upsamples 4:2:0 chroma by nearest neighbour; the tower amplifies the difference (12b) | an interpolating chroma path in the shared decoder, checked for Qwen too |
| MTP at eight lanes, a cheaper drafter step | an MTP server with eight lanes fails its KV allocation at max context 1024 and 2048; drafting is 3-7 ms of a 7-draft round | the device budget of eight MTP lanes. The single-token drafter routes are now profiled and tuned (12c): the two that matter were 1.13x and 1.14x off their best registered route, and the rest were already there |

## 14. What the sibling engines settle

Checked against llama.cpp (MIT) `src/models/gemma4.cpp` and `common/parsers/gemma4.cpp`, and gewell
(Apache-2.0) `include/gewell/models/gemma4/31b/`:

- Attention scale is 1.0 with no pre-attention scaling.
- Sliding layers are a standard sliding window over a windowed cache, not a different attention.
- Proportional RoPE is a parameterization: per-pair frequency factors over the full head.
- The global K = V compression is gewell's `compact_global_cache.h` for the same card and layout.
- Tool-call syntax and the thinking channel match llama.cpp's Gemma 4 parser.

## 15. Open points

| Question | How it closes |
|---|---|
| Compact global KV rounding at 128K/256K | the Op criterion at those lengths; separate K/V rows (+60% global KV) if it fails |
| A fused sublayer boundary (post-norm + residual + scalar + next pre-norm) | an attempt was reverted after elements at small cancelling values deviated by up to 0.7% with a scalar ≠ 1; compare the fp32 `inv` and residual sum against FP64 directly before retrying |

## 16. Gemma 4 26B-A4B (the MoE variant)

**Status.** Planned, not executed: the work is on branch `gemma4-26b` and its phases are in
`docs/maintainer/gemma4-26b-plan.md`. Sections 1–15 remain the 31B authority and this section is the
26B one; where they differ, this section wins for the 26B.

### 16.1 Configuration

Measured from `/mnt/storage/models/gemma/26b-full` (`config.json`, `model.safetensors.index.json`, and
the safetensors headers of both shards): `Gemma4ForConditionalGeneration`, QAT-trained, BF16,
26,544,131,376 parameters, 1,013 tensors. Text lives under `model.language_model.`, the vision tower
(355 tensors, 27 layers) and the audio config are out of scope exactly as in §1.

| field | value |
|---|---|
| layers | 30; `full_attention` at 5, 11, 17, 23, 29, `sliding_attention` elsewhere |
| widths | hidden 2816, dense FFN 2112, sliding window 1024 |
| MoE | `enable_moe_block: true`, 128 experts, `top_k_experts: 8`, `moe_intermediate_size: 704` |
| attention | 16 query heads; 8 KV heads at head_dim 256 on sliding layers, 2 KV heads at 512 on global ones |
| RoPE | unchanged from §4/§5: sliding default θ=10000, global proportional θ=10⁶ with factor 0.25 |
| vocab / softcap | 262144, tied head, `final_logit_softcapping: 30.0` |
| per-layer input | `hidden_size_per_layer_input: 0`, i.e. §3's per-layer-input branch is dead |
| KV sharing | `num_kv_shared_layers: 0` on the target |

Parameter split: the experts are 761 M per layer, 22.8 B of the 26.5 B total (86 %), ≈3.6 B active per
token at top-8. Tensor shapes and dtypes (BF16 throughout the source):

| role | shape |
|---|---|
| `layers.N.experts.gate_up_proj` | `[128, 1408, 2816]` — gate and up fused, in that order |
| `layers.N.experts.down_proj` | `[128, 2816, 704]` |
| `layers.N.router.proj.weight` | `[128, 2816]` |
| `layers.N.router.scale` | `[2816]` |
| `layers.N.router.per_expert_scale` | `[128]` |
| `layers.N.mlp.{gate,up}_proj` | `[2112, 2816]`, `down_proj` `[2816, 2112]` |
| norms | six per layer, all `[2816]`: `input`, `post_attention`, `pre_feedforward`, `post_feedforward`, `pre_feedforward_layernorm_2`, `post_feedforward_layernorm_1`, `post_feedforward_layernorm_2` |
| `layer_scalar` | `[1]` per layer, as §3 |

`self_attn.v_proj` exists on sliding layers only, as in §1: the global layers are K = V.

**Stored widths.** Neither of the 26B's FFN widths suits the block-scaled NVFP4 layout as stored, which
needs its output rows in multiples of 128, so both are padded in the artifact and the padding is exact:
the dense MLP is 2112 → **2176** and the routed expert MLP 704 → **768**, with zero gate and up rows
and matching zero down columns, because GeGLU maps a zero gate and up to an exact zero. The artifact's
own `intermediate_size` and `moe_intermediate_size` are therefore the *stored* widths rather than the
checkpoint's — the 31B's 21504 already divides 128, which is why its artifact needs no padding at all.
The vision tower's 4304-wide MLP is stored padded to 4352 for the same reason.

### 16.2 The decoder layer with MoE

Everything outside the FFN is §3. The FFN becomes two parallel branches whose outputs are summed:

```text
residual = h
h1 = mlp(pre_feedforward_layernorm(h))                  # dense MLP, I = 2112
if enable_moe_block:
    h1 = post_feedforward_layernorm_1(h1)               # norm on the DENSE branch
    _, w, idx = router(residual)                        # router reads the PRE-FFN residual
    h2 = pre_feedforward_layernorm_2(residual)          # experts read the PRE-FFN residual
    h2 = experts(h2, idx, w)
    h2 = post_feedforward_layernorm_2(h2)               # norm on the MoE branch
    h = h1 + h2                                         # plain sum, in the output dtype
h = post_feedforward_layernorm(h)                       # shared norm on the sum
h = residual + h
h *= layer_scalar
```

Router:

```text
s  = rmsnorm_weightless(x) · router.scale · H^(-1/2)     # H is hidden 2816; the exponent is a constant
p  = softmax(router.proj(s))                             # [T, 128], no bias
(w, idx) = topk(p, k=8);  w = w / w.sum(-1)              # renormalize BEFORE the per-expert scale
w  = w * router.per_expert_scale[idx]
```

Experts, per expert hit by any token:

```text
gate, up = linear(x, gate_up_proj[e]).chunk(2, dim=-1)   # [0] is gate
y = gelu_tanh(gate) * up
y = linear(y, down_proj[e]) · w[token, slot]
out[token] += y
```

Two details that no amount of spot-checking catches, and each needs its own oracle case:

1. **The router and the experts read the pre-FFN residual**, not the normalized MLP input. The dense
   branch's `pre_feedforward_layernorm` and the experts' `pre_feedforward_layernorm_2` are different
   tensors applied to different values.
2. **The top-k weights are renormalized to sum to 1 before `per_expert_scale` is applied**, so the
   applied weights do not sum to 1 in general.

The eight expert outputs are summed in the output dtype (BF16 in the checkpoint's own reference and in
both sibling engines). FP32 accumulation is a deliberate deviation this engine may choose; it is a
numerical-semantics boundary and needs its own criterion rather than being assumed.

### 16.3 What the siblings settle

Checked against transformers **v5.6.2** `modeling_gemma4.py` (the version the checkpoint declares;
`Gemma4TextRouter` / `Gemma4TextExperts` / `Gemma4TextDecoderLayer`), llama.cpp's
`src/models/gemma4.cpp` (its `LLM_TYPE_26B_A4B` path, 30 layers), and gewell's
`src/models/gemma4/26b_a4b/sm120/moe.cu`:

- The fused `[E, 2·I, H]` expert tensor and its gate-then-up chunk order are confirmed by all three.
- llama.cpp's expert path uses exactly this norm structure (`ffn_norm`, `ffn_post_norm`,
  `ffn_pre_norm_2`, `ffn_post_norm_1`, `ffn_post_norm_2`), normalizes the MoE input from the
  post-attention residual, and computes the router as `rms_norm(residual) · (1/√n_embd) ·
  ffn_gate_inp_s`, then softmax with `norm_topk_prob` set — the same order as §16.2.
- llama.cpp loads the per-expert scale as `down_exps_s`, i.e. it scales the expert output rather than
  the stored weight; that is the same arithmetic as multiplying the reduced weight.
- gewell computes `p/sum(p) * per_expert_scale[idx]` in that order and breaks top-k ties on the
  **lower expert index**, so the tie-break is part of the observable contract.
- gewell's routed GEMM is a CUTLASS grouped block-scaled NVFP4 kernel at tile 128×128×128 with
  cluster 1×1, plus a separate hand-written 16×64 tiled path for thin decode.
- The assistant's `num_centroids` / `centroid_intermediate_top_k` / `use_ordered_embeddings` fields
  have no consumer in either sibling's 26B drafter, consistent with `use_ordered_embeddings: false`;
  confirm against this project's own 31B `mtp` component before landing the 26B drafter.

### 16.4 How NInfer executes it

`ops::gemma_moe_router` and `ops::gemma_moe_experts` implement §16.2's router and expert apply, and
`forward.cpp`'s `feed_forward_sublayer` runs the layer's two branches. The block is per layer, not a
second execution path: a sliding and a global layer differ only in their attention, and both carry
their own `MoeWeights`.

- **The router** is one CTA per token: the weightless norm, `router.scale` and `H^(-1/2)`, the
  projection, softmax, then a top-k whose tie-break is the lower expert index, renormalization, and
  `per_expert_scale`. It writes `I32 [K,T]` ids and `FP32 [K,T]` weights.
- **The experts** read a bank **in place at its stored spans**, so an expert is a row range of its
  bank and no per-expert copy is ever materialized. The bank's format is a reader policy: a BF16
  reader, and an NVFP4 reader that decodes E2M1 codes against an E4M3 sixteen-element block scale and
  divides by the source row's divisor.
- **A routed expert bank is read, never fed to a linear route**, so its input edge is `A16Only` and
  it carries no activation divisor — the one permission by which the 26B's expert banks differ from
  its dense MLP.
- **A quantized linear route resolves its geometry from a fixed registry of hand-tuned shape files**
  (`src/ops/linear/<format>/<format>_dispatch.cpp`), and the FP8, NVFP4, Q4–Q6 and BF16 registries
  hold only the Qwen and 31B geometries. The row-split int8 route is the only one whose registry
  carries a generic selector (`select_q8_generic`, already used by the EAGLE3 drafter and the vision
  tower), so every 26B projection a linear route touches is `q8_g32_fp16`: the fused qkv parent and
  its query/key/value slices at both head widths, the attention output at both, the dense MLP's
  gate/up and down, and the tied head. The same ten shapes are in `test_q8_a16.cpp`'s geometry table,
  where the T sweep meets the FP64 oracle. The routed experts are again the exception, because no
  linear route sees them: they keep NVFP4, and they are the only part of this model that spends four
  bits on this pass. Registering NVFP4 shapes for the dense MLP is what closes that difference, and
  it is a P5 item rather than a P3 one.
- **The padded width is inert**: the zero gate/up rows and matching zero down columns of §16.1 cannot
  contribute, because GeGLU maps zero to zero. The expert test pins this rather than assuming it.
- **The MoE branch adds no large scratch.** It reuses the dense branch's `[H,T]` buffers in place —
  `normed` once the gate/up projections have read it, `down` once the dense branch's post-norm has
  read it, and the attention's `projected` once its post-norm has — so the only new storage is the
  router's output, `2·K·T·4` bytes per layer per pass.
- The 26B's assistant is dense (`enable_moe_block: false`, four layers), so the drafter path is
  unchanged by the block.

