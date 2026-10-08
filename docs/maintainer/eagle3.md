# EAGLE3

EAGLE3 proposes tokens with a small **autoregressive** draft conditioned on the target's extracted
hidden features. The draft is a one-layer transformer: an *encoder* fuses three target hidden states
into a single feature per position through a projection, and a *decoder* drafts one token at a time,
reading that feature plus the previous draft token's embedding (and its own previous hidden state).
The target verifies the drafted tree causally and remains the output authority. NInfer implements
EAGLE3 as an optional component of the [Qwen3.5 model](qwen3_5-model.md).

This reference owns model mathematics, the draft graph, target conditioning and drafting/verification
semantics. Scheduling, publication and common transactions are defined by
[Engine architecture](engine-architecture.md); the DFlash backends, which are block-diffusion rather
than autoregressive, are described by [DFlash and DFlash2](dflash.md).

## Configuration and current geometry

The draft config reports `model_type=llama` and identifies `LlamaForCausalLMEagle3` (SpecForge) or
`Eagle3DraftModel` / `Eagle3LlamaForCausalLM`; the architecture field selects the parser, the model
type is only required to be present. It supplies draft hidden/intermediate width, head
counts, head dimension, one decoder layer, norm epsilon and RoPE theta, the three target layer IDs it
reads, and the draft vocabulary size with its `d2t` map. The target's hidden width and token embedding
are inherited, not stored.

The first target is the SpecForge head
[jiapingW/Qwen3.5-35B-A3B-Eagle3-Specforge](https://huggingface.co/jiapingW/Qwen3.5-35B-A3B-Eagle3-Specforge)
for Qwen3.5-35B-A3B.

| Quantity | Value |
|---|---:|
| Target hidden width `H` | 2048 |
| Fused input width `3·H` | 6144 |
| Draft layers | 1 |
| Draft hidden / intermediate | 2048 / 16384 |
| Q heads / KV heads / head dimension | 16 / 16 / 256 |
| Attention input width (`2·H`) | 4096 |
| Target layers fused | 3: `1`, `num_layers//2 - 1`, `num_layers-4` → `[1, 19, 36]` for the 40-layer target |
| Draft vocabulary | 32,000 (target 248,320) |
| Norm epsilon / RoPE theta | `1e-6` / `1e7` |
| `norm_before_residual` / `norm_before_fc` | false / false |
| Own token embedding | no (shares the target's) |
| Draft KV per token, per lane | 16 KiB: 1 layer × 16 KV heads × 256 dim × K and V, BF16 |

Draft width `K` is a runtime choice, not a weight dimension, exactly as for DFlash.

The draft cache is a second paged pool beside the Text KV: one layer at 16 KV heads and head
dimension 256, always BF16, planned at the Text capacity plus the draft window's page headroom and
never quantized with `--kv-dtype`. That is 16 KiB per token per lane, against 20 KiB for this
target's ten full-attention Text layers at BF16, so on a BF16-KV engine the draft pool adds about
80% to the KV-resident bytes; with a quantized `--kv-dtype` the draft bytes do not shrink, so the
share is larger still. It is the cost of a multi-head draft whose attention reads the whole
history at absolute positions.

## Model mathematics

Let `n = n_embd` be the draft hidden width, `H` the target hidden width, and let the three extracted
target hidden states at position `p` be `z_p = concat([h_{L0}[p], h_{L1}[p], h_{L2}[p]])` with shape
`3·H`.

**Encoder** (once per target forward, one row per target position):

```text
g_p = W_fc · rmsnorm_fc(z_p)        # [n]        (rmsnorm_fc only when norm_before_fc)
```

**Decoder** (one step per drafted position). At draft position `P` the input pair is the next token
`t_{P+1}` and the encoder feature at the *anchor* position `g_P`, with RoPE applied at `P`:

```text
e = rmsnorm(W_emb[t_{P+1}], attn_norm)          # token embedding, own or target's
u = rmsnorm(g_P,           attn_norm_2)         # fused target feature
x = concat(e, u)                                # [2n]
q,k,v = W_q x, W_k x, W_v x                     # attention input is 2n wide
q,k = rope(q, k, positions)
a = causal_attention(q, [K_ctx, k], [V_ctx, v]) # draft KV cache, shared with the target context
r = norm_before_residual ? u : g_P              # residual base
y = a + r
m = rmsnorm(y, ffn_norm)
h = y + W_down(silu(W_gate m) * (W_up m))       # draft pre-norm hidden -> next step's g
logits_full = scatter(W_head · rmsnorm(h, output_norm), d2t)   # draft vocab -> target vocab
```

Two facts drive the implementation:

- `g` is **chained**: the decoder's pre-norm hidden `h` becomes the `g` for the next draft step, so
  drafting cannot be a single masked-block forward.
- The head projects to a **32k draft vocabulary**; a `d2t` table maps each draft row onto its absolute
  target token id, leaving the remaining target rows at `-inf`.

## Target conditioning

The three `L_i` are the target's extracted hidden states, delivered through the same
[`DFlashFeatureSink`](../../src/models/qwen3_5/execution/text.cpp) mechanism the DFlash backends use:
the target forward calls `capture_layer(L_i, value)` as it walks its layers, and the sink stores the
three slabs concatenated along the feature axis (`3·H`). Unlike DFlash, EAGLE3 uses all three rows of
that buffer as the encoder input directly; it does not project per layer.

The target must expose those layers as embeddings inputs at startup
(`llama_set_embeddings_layer_inp` equivalent). Layers equal to the target layer count mean the target's
final pre-norm hidden (the MTP feature).

## Drafting and verification

Per sequence, drafting walks `K` steps:

1. Seed: encoder feature `g` at the current anchor and its target-sampled next token form the pair
   `(t, g)`, which primes the draft KV at the anchor position.
2. For each step, run the decoder on the current `(token, g)` pair, sample the next token from the
   draft logits (top-k branch, `k=10` in the reference), and feed the decoder's pre-norm hidden back
   as the next `g`. The result is a tree of candidate token paths.
3. The target verifies the tree in one causal batch; accepted tokens advance the sequence and the
   first rejected position's target sample reseeds the draft. The target output is authoritative.

The anchor feature `g_P` is produced while the target performs its own forward, so the encoder output
at the last verified position is carried as the seed.

EAGLE-3 adopts EAGLE-2's **context-aware dynamic draft tree**: the tree is grown from the draft
model's own confidence instead of being fixed in advance. The first implementation uses a top-k
chain/tree (as in the llama.cpp reference); the dynamic tree is a later refinement.

## Static draft tree (status and design)

The planned first tree is static and root-branched: Top-T first tokens from one seed hidden state
(`eagle3_propose_topk_batch`: one head projection, repeated argmax with `mask_selected_logits`),
each continued greedily, all chains verified together, winner committed. Landed enablers:
`mask_selected_logits` / `select_tree_winners` (exact-oracle unit tests), the top-k propose path,
and the `--draft-tree` option with plan/workspace/reporting plumbing (currently gated to 1). The
top-k propose path is exercised end to end at `top = 1` against the chain path
([Measured status](#measured-status)); its tree wiring, and with it any `top` above one, is not
reachable from a product route yet.

Round wiring is pending on one infra question, settled by measurement against the kernel: the
causal attention **appends** KV at its cache positions (`bf16_kv_append_attention`), so verifying
T chains that share absolute positions in one forward races on the shared slots, and a no-append
forward would break intra-forward causality (later columns must read the chain's own earlier
columns). Sequential verify-trim-reverify is correct but rereads the MoE weights T+1 times per
round, which no accept-length gain can repay. The viable batched verify gives each chain its own
text-KV row whose page table shares the physical history pages and appends to fresh pages
(metadata-only fork, no data copy, one weight read); the winner's row is adopted at commit. That
per-round row lifecycle is the remaining work; sequential chains need no draft-KV snapshot
because each chain overwrites its own window before reading it.

Reference: [EAGLE-3](https://arxiv.org/abs/2503.01840) §3.1 (inference pipeline) and §2.2 (the
EAGLE-2 tree). Two points the paper leaves loose are settled by the released weights: the fused
feature is `concat(l, m, h) -> FC -> g` (the encoder), and the decoder's `concat(embedding, g)` goes
**straight to q/k/v** with no input projection — the checkpoint's q/k/v input is `2n`, not `n`.

## Artifact and converter

The SpecForge checkpoint stores, besides `config.json`:

| Checkpoint tensor | Logical role |
|---|---|
| `fc.weight` `[n, 3H]` | feature-fusion projection (`FC`) |
| `midlayer.input_layernorm.weight` `[n]` | decoder attention input norm (`ATTN_NORM`) |
| `midlayer.hidden_norm.weight` `[n]` | fused-feature norm (`ATTN_NORM_2`) |
| `midlayer.self_attn.{q,k,v}_proj` `[2n, 2n]` | attention projections (`ATTN_Q/K/V`, input `2n`) |
| `midlayer.self_attn.o_proj` `[n, 2n]` | attention output (`ATTN_OUT`) |
| `midlayer.post_attention_layernorm.weight` `[n]` | FFN norm (`FFN_NORM`) |
| `midlayer.mlp.{gate,up}_proj` / `down_proj` | SwiGLU FFN (`FFN_*`) |
| `norm.weight` `[n]` | decoder output norm (`OUTPUT_NORM`) |
| `lm_head.weight` `[32000, n]` | draft head (`OUTPUT`, draft vocabulary) |
| `d2t` `i64 [32000]` | draft-to-target token-id map (`D2T`) |
| `t2d` `bool [248320]` | target-to-draft mask; **not used at runtime** |
| `input_norm.weight` `[3H]` | `ENC_OUTPUT_NORM`; absent here (`norm_before_fc=false`) |

The converter normalizes SpecForge's `midlayer.*` naming to the draft layer, keeps `fc`/`d2t`, drops
`t2d`, and inherits the target's token embedding and tokenizer. The `qwen3_5` model description gains
an `eagle3` draft variant; the loader binds the draft tensors exactly as it binds DFlash's.

## Integration plan

Phased, each phase independently verifiable:

1. **Aux layers and config.** Recover the three target layer IDs from the SpecForge training state and
   add the `eagle3` draft config (target layers, target hidden, draft vocab, `norm_before_*`).
2. **Converter and bindings.** Add the EAGLE3 recipe and draft model description; convert the head to a
   `.ninfer` and verify every tensor maps to its logical parameter at the stored shapes.
3. **Encoder execution.** `g = W_fc · norm(z)` over the sink buffer; qualify against an FP64 oracle on
   the represented inputs.
4. **Decoder execution.** One-layer transformer with the `2n`-wide attention input, the chained `g`,
   the draft head and the `d2t` scatter; qualify per-step logits and the pre-norm hidden against the
   oracle.
5. **Drafting and verification.** Autoregressive top-k tree drafting and batched target verification.
   The round is **MTP-shaped** (one token per step, its own KV), not the masked DFlash round, so the
   MTP decode round is the right host: the draft forward swaps in the encoder + EAGLE3 decoder while
   the draft/verify/KV/commit plumbing is shared. This is why the load reaches the DFlash dispatch
   (`decode_dflash_batch` requires the DFlash backend) and stops: EAGLE3 must not enter that round.
6. **Measure.** Acceptance, tokens/forward and end-to-end decode against MTP; see
   [Measured status](#measured-status).

## Measured status

Reference: NVIDIA GeForce RTX 5090, CUDA 13.3, `models/qwen3_6_35b_a3b_nvfp4_eagle3.ninfer`
(Qwen3.6-35B-A3B target carrying the SpecForge head, BF16 KV, greedy sampling), 2026-10-08.

One request, `ninfer_bench -pg '512,128;2048,128' -r 5 --warmup 1 --fixed-draft` (5 measured
repetitions of the benchmark corpus; MTP runs the full target head because this artifact carries no
optimized proposal head):

| Mode | decode tok/s, pp512 | decode tok/s, pp2048 | accepted/drafted, pp512 | accepted/drafted, pp2048 |
|---|---:|---:|---:|---:|
| `--spec mtp --draft-tokens 3` | 470.5 | 757.9 | 49.0% | 100% |
| `--spec eagle3 --draft-tokens 3` | 428.0 | 341.0 | 34.4% | 22.2% |
| `--spec eagle3 --draft-tokens 7` (adaptive) | 472.0 | 371.3 | 45.5% | 27.6% |

Serving, `tools/bench/run_serve_concurrency --suite decode-saturation --sampling greedy
--decode-tokens 2048 --max-context 8192 --kv-capacity auto --kv-dtype bf16` (a 293-token reasoning
prompt, 2,048 decoded tokens per request, complete full-batch intervals):

| Mode | C=1 decode tok/s | C=8 decode tok/s | C=8 scaling |
|---|---:|---:|---:|
| no `--spec` | 376.0 | 1,272.0 | 3.38x |
| `--spec eagle3 --draft-tokens 3 --fixed-draft` | 537.5 | 1,805.4 | 3.36x |
| `--spec eagle3 --draft-tokens 7` (adaptive) | 537.0 | 1,728.7 | 3.22x |

What the numbers say:

- On real long-decode prompts EAGLE3 is worth selecting: its three-draft chain is 43% faster than
  plain decode at one request and 42% faster at eight, with 52-61% accepted/drafted per request and
  no loss of batch scaling.
- On the flat benchmark corpus MTP is faster at one request (758 against 341 tok/s at pp2048)
  because its acceptance is 100% there while EAGLE3's is 22%; the adaptive EAGLE3 policy matches the
  three-draft MTP setting on the short prompt (472 against 470 tok/s) and trails on the long one.
  The corpus repeats itself, so its MTP acceptance is not representative of prose.
- No like-for-like MTP reference exists on this artifact: it was converted without `--proposal`, and
  EAGLE3 rejects the optimized proposal head anyway. The published MTP settings therefore stay the
  recommendation, with EAGLE3 the better mode only where its head is available.
- The draft's KV cost is confirmed at the planner's capacity: the reported KV payload is 82.0 MiB
  with EAGLE3 against 51.0 MiB without it at the same token capacity, about 14 KiB per token beside
  the 16 KiB per token derived above before page rounding.

The top-k propose path is verified against the chain path rather than only by its ops oracles: with
the round temporarily routed through `eagle3_propose_topk_batch(..., top = 1)`, the CLI commits
byte-identical tokens and the identical round, acceptance and draft counters.

## Open questions

- **Target version.** The head targets Qwen3.5-35B-A3B; the shipped artifact is Qwen3.6-35B-A3B. Both
  are `qwen3_5_moe`, but the hidden states were never confirmed compatible, and this is the likeliest
  remaining cause of the low flat-corpus acceptance above.
- **Target-layer capture order.** The converter derives the three taps the SpecForge rule names
  (`1`, `num_layers//2 - 1`, `num_layers-4`) and the artifact records them, but the released
  checkpoint does not say whether training read a layer's output or the input of the layer after it.
  NInfer captures the output of the named layer. A one-layer sweep found no material difference
  ([0, 18, 35] and [2, 20, 37] and the stale `num_layers//2` middle tap `[1, 20, 36]` are all within
  13% on the flat corpus and identical at pp2048), so the convention is not off by one layer in a way
  neighbouring taps can detect; confirming the exact rule needs the SpecForge training configuration.
- **Quantized target.** As with DFlash, the target's quantized hidden states condition the draft, so
  acceptance on the NVFP4 artifact may differ from a BF16 target. The artifact measured here is NVFP4.
- **Vision.** The draft runs inside the multimodal prefill chunk exactly as it does for text, but no
  test exercises `--vision` together with `--spec eagle3`.