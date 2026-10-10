# Gemma 4 model reference

Gemma 4 31B (`Gemma4ForCausalLM`, text `gemma4_text`) as NInfer executes it. This is the
mathematical authority for the implementation and the oracle; it is written from the checkpoint and
the plan that preceded this work. Vision (P5) and the assistant drafter (P6) are not described here
yet.

**Status.** Text generation, chat, tools, thinking, causal scoring and serving run through the
public Engine. Prefix reuse, constrained output and fast prefill do not exist yet; section 13 lists
what is not done.

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
| `out/gemma4_31b_it_m1.ninfer` | `gemma4_31b_m1` | measured layout `tools/convert/layouts/gemma4_31b_m1.json`: attention and head Q6 g64, each MLP group Q6 or Q5 g64 by measured sensitivity (173 Q6, 68 Q5 of 240 plus the head), embedding BF16 in pinned host memory | 23.6 GiB file, 20.9 GiB on the device |

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
- **KV cache** (`cache.{h,cpp}`): sliding layers keep a ring of exactly `sliding_window` rows. A pass
  wider than one token attends to the ring as it was before the pass plus its own keys, which the
  sliding Op takes as a second key set, and writes its last window of keys afterwards; so no write
  replaces a key one of its own queries still sees, whatever the pass width. A single token writes
  first and attends to the ring alone, since the row it replaces (position p − window) is invisible
  to it. Global layers keep one compact row per token at its position, up to the lane's capacity.
  Unwritten slots carry position 2^30, which both attention Ops treat as invisible; the Ops also
  never load a key no row of the tile can see, so an unwritten slot's bits cannot reach the output.
- **Attention Ops**: `sliding_causal_attention` (D 256, 32/16 heads, window, scale 1.0, optional
  second key set) and `causal_compact_attention` over the 640-value compact rows share one
  tensor-core flash kernel (`gemma_flash_attention.cuh`): BF16 m16n8k16 MMA with FP32 accumulation,
  16 query rows per block, 32-key tiles skipped by position before they load, online Softmax with P
  carried as a BF16 hi + lo pair, and a key split with a combine kernel when a pass has few rows.
- **Linears**: M1's Q6 and Q5 g64 A16 routes carry Gemma's shapes. Single-token passes use the Q6
  GEMV and single-row Q5 instances at K 5376 and 21504; wider passes reuse the route lists tuned for
  the nearest Qwen shapes, untuned here. The 5376-wide RMSNorm has a fixed-width route. GeGLU is
  unfused (`gelu_mul` on the two halves).
- **Engine** (`src/runtime/engine/gemma_instance.{h,cpp}`): `EngineCore<GemmaInstance>` and
  `CausalScoreCore<GemmaInstance>`, the same controller Qwen uses. Admission is root-only: the
  context cache is switched off for this model, so the identity candidate is always feasible and
  never expandable and seals directly. Each lane decodes alone, one token per round, and samples on
  device. The prefix-cache entry points the controller is compiled against (pressure planning,
  captures, checkpoint recovery, continuations) and forced control tokens refuse by name. Contract
  types this model never constructs are the Qwen contract's pure-data types; moving them into
  `runtime/contract` is the plan's D1-b extraction and is not done.
- **Refused options**: speculative decoding, Vision, a KV format other than BF16, KV streaming, and
  an explicit KV capacity below `max_context`. `ninfer-perplexity` defaults to FP8 KV, so it needs
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
| Serving | `tools/smoke/serve_contract.py --text-only --tools` | OpenAI chat/Responses, stored continuation, Anthropic, count_tokens, one tool round trip; plus streamed reasoning, Anthropic `tool_use` and two concurrent requests checked by hand |

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

## 13. Not done

| Item | Why it matters | What it needs |
|---|---|---|
| Prefix reuse | an agent loop re-prefills its whole history every turn, at the prefill rates above | the D1-b move of the continuation, shared-prefix and capture types into `runtime/contract`; ring-snapshot checkpoints as the state image |
| Tuned prefill linears | the wide Q6/Q5 routes borrow lists tuned for Qwen shapes | a route sweep at Gemma's N and K |
| Lanes in one pass | concurrency re-reads the weights per lane | per-lane caches laid out for the Ops' batch dimension; one pass over B columns |
| CUDA Graphs | decode launch overhead | graph capture per lane count |
| Output constraints | refused | token-mask consumption in the Program |
| FP8 KV | refused | the codecs of plan row 7 |
| Vision, assistant drafter | plan P5, P6 | |

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
| Sliding mask with images (`use_bidirectional_attention: "vision"`; llama.cpp reads it as bidirectional on SWA layers only) | Transformers `create_masks_for_vision_model` and an oracle case, with Vision (P5) |
| The assistant drafter's hidden state (before or after the final norm) | `modeling_gemma4_assistant.py`, with P6 |
| Compact global KV rounding at 128K/256K | the Op criterion at those lengths; separate K/V rows (+60% global KV) if it fails |
| A fused sublayer boundary (post-norm + residual + scalar + next pre-norm) | an attempt was reverted after elements at small cancelling values deviated by up to 0.7% with a scalar ≠ 1; compare the fp32 `inv` and residual sum against FP64 directly before retrying |
