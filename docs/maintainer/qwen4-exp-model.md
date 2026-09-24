# Qwen4Exp model reference

This reference defines the mathematics of `Qwen4ExpForCausalLM` (`qwen4_exp_text`), the
architecture of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
(`Qwen4ExpForConditionalGeneration`). It is **not** an instance of `Qwen3_5MoeForCausalLM`:
hyper-connection residual streams, Qwen Sparse Attention (QSA) and the n-gram PLE embedding
change the block and state mathematics.

**Status:** NInfer executes this architecture through `execution/qwen4_text.cpp` with routed
experts in pinned Host memory behind a Program-owned device expert cache and a file-mapped n-gram
table gathered on the Host. The FP64 oracle in
[`tests/models/qwen4_exp/reference.py`](../../tests/models/qwen4_exp/reference.py) implements
this reference. Its tests in
[`test_reference.py`](../../tests/models/qwen4_exp/test_reference.py) check it against:
- the published checkpoint constants;
- its own state semantics;
- when PyTorch and a Transformers build with `qwen4_exp` are installed, the upstream model.

The upstream authorities are:
- Transformers `models/qwen4_exp` for Text;
- SGLang `qwen4_exp_mtp.py` for MTP, because Transformers ignores `mtp.*`.

Shared pieces keep their [Qwen3.5 reference](qwen3_5-model.md) definitions:
- GDN recurrence;
- gated GQA;
- interleaved MRoPE;
- Vision tower and multimodal positions;
- tokenizer and frontend semantics.

This document lists only what differs.

## Configuration

| Quantity | Qwen3.8-Flash-Next |
|---|---:|
| Hidden width H / hyper-connection streams hc / wide width hc·H | 2560 / 4 / 10240 |
| Text layers; GDN / QSA attention | 48; 36 / 12 (every fourth layer is attention) |
| Q heads / KV heads / head dimension; rotary dimensions | 24 / 2 / 256; 64 |
| QSA indexer heads / key heads / head dimension; budget / block | 4 / 1 / 128; 2048 / 4 |
| GDN key / value heads, head dimensions, conv width | 16 / 48, 128 / 128, 4 |
| Routed experts / selected / expert width; shared width | 512 / 10 / 640; 640 |
| HC low rank | 320 |
| PLE layer (one-indexed `ple_layer_ids`) | 2, i.e. block index 1 |
| N-gram size / heads per n-gram / row width / table rows | 3 / 8 / 160 / 320,001,536 |
| Vocabulary; position capacity | 248,320; 262,144 |

Other fixed values:
- `rms_norm_eps=1e-6`
- `rope_theta=1e7`
- `mrope_section=[11,11,10]`
- `output_gate_type=sigmoid`
- `norm_topk_prob=true`
- `seed=1234`
- `eos_token_id=248044`

Embedding and output head are untied, and there is no final RMSNorm: the output mixer feeds the
head directly.

The BF16 checkpoint holds about 180B parameters, of which about 121B are routed experts and 51.2B
are the n-gram table.

## Norms

- `offset_rmsnorm(x, w) = x / rms(x) * (1 + w)`. It is used by attention Q/K norms, the indexer
  norms, HC norms, PLE norms and the MTP pre-fc norms.
- A **grouped** offset RMSNorm computes the RMS separately over each contiguous H-wide stream of a
  wide vector and then applies the hc·H-wide weight.
- The GDN output norm is `w * x / rms(x) * act(z)` with a plain (not offset) weight, and
  `act = sigmoid` here. Qwen3.5 uses SiLU in this position.

## Hyper-connection residual

The residual is a wide vector `R` of hc streams, each H wide. Embedding initializes every stream
with the token embedding (`R = [e, e, e, e]`).

Each block has two HC mixers, one around the mixer and one around the MoE. Each mixer computes:

```text
n      = grouped_offset_rmsnorm(R, hc_norm)                         # [hc·H]
m      = sigmoid(W_up · silu(W_down · n / hc))                      # [hc·H], W_down: [320, hc·H]
x      = mean_s(m_s ⊙ n_s)                                          # block input, [H]
inject = 2 · sigmoid(W_inject · n / hc)                             # [hc]
y      = block(x)                                                   # [H]
R      = R + [inject_s · y for each stream s]
```

The final `hyper_connection_mixer` has no inject weight. Its `x` is the Text hidden state that
goes to `lm_head`.

HC is per token and has no recurrent state. The wide residual is the continuation hidden that MTP
consumes.

## Block

```text
if block has PLE:  R = R + PLE(R, tokens)
x, inj = HC_attn(R);  R = R + inj ⊗ mixer(x)        # GDN or QSA gated attention
x, inj = HC_mlp(R);   R = R + inj ⊗ SparseMoE(x)
```

**GDN** is the Qwen3.5 formulation:
- `in_proj_qkv` feeds a causal depthwise conv (kernel 4) followed by SiLU;
- Q and K are L2-normalized, and Q is additionally scaled by `1/sqrt(128)`;
- each key head serves three consecutive value heads (`repeat_interleave`);
- `β = sigmoid(b)` and `g = -exp(A_log) · softplus(a + dt_bias)`;
- the delta-rule recurrence runs over `[Dk,Dv]` state per value head;
- the output passes through the gated norm with the `z` gate, then `out_proj`.

**Sparse MoE** is the [Qwen3.5 formula](qwen3_5-model.md#sparse-moe) with E = 512 and K = 10:
- softmax over router logits, top-10, and renormalization of the selected weights;
- SwiGLU experts whose fused `gate_up_proj` rows are gate first, then up;
- the shared expert scaled by `sigmoid(shared_expert_gate · x)`.

## Qwen Sparse Attention

The attention projections are unchanged from Qwen3.5:
- `q_proj` emits `[q_h, gate_h]` per head;
- offset-RMSNorm is applied to Q and K;
- partial interleaved MRoPE covers the first 64 dimensions;
- the output is `o_proj((softmax(qKᵀ/16)V) ⊙ sigmoid(gate))`.

QSA restricts which cached keys each query may see.

The indexer computes `index_qk_proj(x)`, giving `[4·128 | 128]`: the query heads and one raw key.
- **Raw key:** cached per token (a new KV-like plane), with no norm and no RoPE.
- **Query heads:** offset-RMSNorm, then RoPE on their first 64 dimensions at the query position.

For a query that sees cache indices `[0, n)`:

```text
blocks  = floor(n / 4); block b covers indices [4b, 4b+4)
k̄_b    = RoPE(offset_rmsnorm(mean(raw_key[4b .. 4b+3]), k_layernorm), position(4b))
score_b = Σ_heads relu(q_h · k̄_b) / sqrt(128)
keep    = min(512, blocks) blocks with the largest scores
visible = tokens of kept blocks ∪ tail [4·blocks, n)
```

Selection width is at most `2048 + 3` tokens. While `n ≤ 2051`, every token is visible and QSA
equals dense causal attention; beyond that, decode attention cost stays constant.

- **Positions:** pooled keys use the three-axis position of the block's first token.
- **Ties:** the oracle breaks exact score ties toward the lower block index. Production selection
  is qualified against the oracle with an explicit near-tie allowance.
- **Dtypes:** upstream stores raw keys and pooled means in the activation dtype. That is a private
  staging choice, not a semantic boundary.

## PLE n-gram embedding

PLE runs once, before block 1's attention mixer.

**N-gram rows.** For the token at position p and n-gram size n ∈ {2, 3}, the context token at
distance s is:
- `t[p-s]` when none of `t[p-s .. p-1]` is EOS;
- EOS otherwise.

An EOS therefore restarts every window, exactly like sequence start, where the two-token history
is EOS-filled. Head j of n-gram n uses row:

```text
mixed_n = XOR_{k<n} (ctx_k · m_k)            # signed 64-bit; products stay below 2^63
row_j   = mixed_n mod P_j + offset_j
```

The rows use these constants:
- **Multipliers** `m = [23703573157769, 20109073645365, 8052911324071]`. They are derived from
  the seed by SplitMix64 and also stored as the `layer_multipliers` buffer.
- **Moduli** `P_j`: the 16 consecutive primes after `ngram_vocab_size_base − 1`, from 20,000,003
  to 20,000,171.
- **Offsets:** their running sum.

The table has `320,001,536 × 160` rows, split in the checkpoint into 128 shards of 2,500,012 rows.
The 16 gathered rows are concatenated, 2-gram heads first, into the 2560-wide embedding `E`.

**Injection.** With `R` the wide residual entering the block:

```text
k_s  = grouped_offset_rmsnorm(key_proj · E, norm_key)_s            # key_proj: [hc·H, 2560]
v    = value_proj · E                                              # [H]
q_s  = grouped_offset_rmsnorm(R, norm_query)_s
g_s  = (k_s · q_s) / sqrt(H);  g_s = sign(g_s) · sqrt(max(|g_s|, 1e-6))
u    = [sigmoid(g_s) · v for each s]                               # [hc·H]
c    = grouped_offset_rmsnorm(u, norm_conv)
R    = R + u + silu(depthwise_conv(c; kernel 4, dilation 3))
```

**State.** The convolution is causal with dilation 3, so its state is the previous nine `c`
vectors (9 × 10240). Together with the two-token n-gram history, this is new per-sequence
recurrent state beside the GDN state.

## MTP

The single MTP layer is a full QSA block with its own KV and indexer-key planes and no PLE. It
consumes the **wide** residual `R_t`, not a mixed or normalized hidden state. The token-shift
alignment is the [Qwen3.5 one](qwen3_5-model.md#prefill-decode-and-mtp): hidden states and
positions stay in place while tokens shift by one.

```text
e   = fc_embedding · offset_rmsnorm(embed(x_(t+1)), pre_fc_norm_embedding)       # [H]
h   = offset_rmsnorm(R_t, pre_fc_norm_hidden)     # one RMS over all hc·H values, not grouped
R'  = [fc_hidden · h_s + e for each stream s]
R'  = QSA_block(R')                                # HC mixers, QSA attention, 512×top-10 MoE
logits = lm_head(mtp.hyper_connection_mixer(R'))
```

`R'` is the draft continuation hidden for recursive proposals. The draft MoE uses its own expert
banks (`mtp.layers.0.mlp.*`).

## Vision

The tower has the Qwen3.5 geometry (depth 27, width 1152, patch 16, merge 2). Its merger emits
2560-wide columns, which replace placeholder embeddings before the embedding is repeated into the
hc streams. PLE hashes the placeholder token IDs themselves.

## Per-sequence state

Per sequence, State holds:
- GDN conv and recurrent state (as Qwen3.5);
- PLE conv history (9 × 10240) and n-gram token history (2);
- attention K/V plus one raw indexer key per token for each of the 12 attention layers and the
  MTP layer;
- the wide continuation hidden (10240).
