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

Both are produced from the instruction-tuned BF16 checkpoint at `/mnt/storage/models/gemma/full-31b`.
The recipe and the first artifact carry "base" in their names from before the checkpoint was
identified; the bytes are the instruction-tuned model's.

| Artifact | Recipe | Layout | Size |
|---|---|---|---:|
| `out/gemma4_31b_base_nvfp4.ninfer` | `gemma4_31b_base` | plan L1: MLP NVFP4 (converter-encoded, weight-only), attention and embedding FP8 rows, head BF16, norms BF16, `layer_scalar` FP32 | 22.8 GiB |
| `out/gemma4_31b_g0.ninfer` | `gemma4_31b_g0` | gewell's G0 mask: embedding and head BF16, attention FP8, MLP FP8 in layers 0–5 and every global layer with its predecessor, NVFP4 elsewhere | 27.3 GiB |
| `out/gemma4_31b_it_m1.ninfer` | `gemma4_31b_m1`, `--components text,vision,mtp --source mtp=/mnt/storage/models/gemma/assistant-31b` | measured layout `tools/convert/layouts/gemma4_31b_m1.json`: attention and head Q6 g64, each MLP group Q6 or Q5 g64 by measured sensitivity (173 Q6, 68 Q5 of 240 plus the head), embedding BF16 in pinned host memory; the assistant drafter Q6 g64 (`mtp/`, about 0.4 GB, bound only with `--spec mtp`); the vision tower Q6 g64 with BF16 patch and output projections (`vision/`, about 0.45 GiB, bound only with `--vision`) | 24.4 GiB file; 20.9 GiB on the device, 21.7 with the drafter and the tower |

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
  tensor-core flash kernel (`gemma_flash_attention.cuh`): BF16 m16n8k16 MMA with FP32 accumulation,
  16 query rows per block, 32-key tiles skipped by position before they load, online Softmax with P
  carried as a BF16 hi + lo pair, and a key split with a combine kernel when a pass has few rows.
- **Linears**: M1's Q6 and Q5 g64 A16 routes carry Gemma's shapes. Single-token passes use the Q6
  GEMV and single-row Q5 instances at K 5376 and 21504; 2-8 token passes (MTP verification) use one
  8-token sliced MMA block for Q5, as Q6 already did; wider passes reuse the route lists tuned for
  the nearest Qwen shapes, untuned here. The drafter's BF16 shapes have single-token GEMV routes.
  The 5376-wide RMSNorm has a fixed-width route. GeGLU is unfused (`gelu_mul` on the two halves).
- **Engine** (`src/runtime/engine/gemma_instance.{h,cpp}`): `EngineCore<GemmaInstance>` and
  `CausalScoreCore<GemmaInstance>`, the same controller Qwen uses. Admission is root-only: the
  context cache is switched off for this model, so the identity candidate is always feasible and
  never expandable and seals directly. Each lane decodes alone, one token or one MTP round per
  decode round, and samples on device. The prefix-cache entry points the controller is compiled against (pressure planning,
  captures, checkpoint recovery, continuations) refuse by name. Forced control tokens (the thinking
  budget's `<channel|>` close) are appended as one prefill of the lane. Contract
  types this model never constructs are the Qwen contract's pure-data types; moving them into
  `runtime/contract` is the plan's D1-b extraction and is not done.
- **Refused options**: speculative backends other than MTP (and MTP's optimized proposal head,
  prompt-lookup drafts and draft trees), a KV format other than BF16, KV streaming, and an
  explicit KV capacity below `max_context`. Video, images in tool results and images in causal scoring
  are refused per request. `ninfer-perplexity` defaults to FP8 KV, so it needs
  `--kv-dtype bf16` for this model. CUDA Graphs are not used whatever `use_cuda_graph` says.

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

Decode is measured alone (`-n 128 -r 5`); right after the prefill tests in the same run it reads
about 1% lower. The decode ceiling from M1's 20.9 GiB of device weights at 1.79 TB/s is about 80
tok/s, so decode runs at about two thirds of it, without CUDA Graphs. The bench header reports
`decode_path=cuda_graph`; that is the option as passed, not what this model ran.

Of a decode token's GPU kernel time (nsys), the Q6 and Q5 linears are 88%, RoPE (a generic kernel)
4.6%, attention 3.6%, and norms and element-wise kernels the rest. Kernel time is about 18 ms of the
19 ms per token, so launch gaps, which CUDA Graphs would remove, cost about 5%; the linears' distance
from the weight-streaming rate is the larger lever. An 8K prefill at 128-token passes spent 46% in the Q6 MMA, 22% in the Q5
sliced-K MMA and 29% in attention; widening the pass to 512 tokens took pp8192 from 991 to 1,837
tok/s. Two concurrent requests both progress, but each lane runs its own pass, so a second lane
re-reads the weights rather than sharing a pass.

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

**Cost.** Encoding takes 27-33 ms per image of about 2,300 patches on the RTX 5090 (Q8: 26-28; the Q6
routes borrow Gemma's text list, untuned for these shapes). A one-image prompt
of 280 tokens served with `--vision --spec mtp` has a TTFT of 192 ms; a 19 MP JPEG adds about 200 ms of
single-threaded decode and resize on the request thread (13).

## 13. Not done

| Item | Why it matters | What it needs |
|---|---|---|
| Prefix reuse | an agent loop re-prefills its whole history every turn, at the prefill rates above | the D1-b move of the continuation, shared-prefix and capture types into `runtime/contract`; ring-snapshot checkpoints as the state image |
| Tuned prefill linears | the wide Q6/Q5 routes borrow lists tuned for Qwen shapes | a route sweep at Gemma's N and K |
| Lanes in one pass | concurrency re-reads the weights per lane | per-lane caches laid out for the Ops' batch dimension; one pass over B columns |
| CUDA Graphs | decode launch overhead | graph capture per lane count |
| Output constraints | refused | token-mask consumption in the Program |
| FP8 KV | refused | the codecs of plan row 7 |
| Image preprocessing off the request thread | a 19 MP JPEG costs about 200 ms of single-threaded decode and resize before prefill | Qwen's media worker pool and cache, shared |
| JPEG decoding as Pillow does it | FFmpeg upsamples 4:2:0 chroma by nearest neighbour; the tower amplifies the difference (12b) | an interpolating chroma path in the shared decoder, checked for Qwen too |
| MTP with several lanes in one pass, a cheaper drafter step | each lane verifies alone; drafting is 3-7 ms of a 7-draft round, unprofiled | the lane batching above; a drafter profile, then e.g. a Q8 head qualified against acceptance |

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
