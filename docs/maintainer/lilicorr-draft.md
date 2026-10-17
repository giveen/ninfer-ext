# LiLiCorr draft for Nemotron-H (Bolt 74B A9B)

The speculative drafter for the [Nemotron Bolt target](nemotron-bolt-plan.md): NVIDIA's
`NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr` checkpoint, a DFlash-style block drafter whose
projector is the LiLiCorr correlation head
([arXiv:2608.20530](https://arxiv.org/abs/2608.20530)). Milestone 5 of the port.

Two external implementations of this exact checkpoint were read for the mathematics, and they agree
on every load-bearing point: **vLLM master** (`vllm/model_executor/models/lilicorr.py`,
`vllm/v1/worker/gpu/spec_decode/{lilicorr,dflash2,dflash}/speculator.py`, `qwen3_dflash.py`,
`qwen3_dflash2.py`) and **llama.cpp master** (`src/models/dflash.cpp`, `conversion/nemotron.py`, PR
#30506 with the Bolt-side gate fix in #30520). Where they differ the difference is recorded below
rather than smoothed. The draft is a companion of the **24 GB** target, whose experts are 2-bit
`IQ2_XXS` while ours are NVFP4 at 4.5 bits: the pairing is the risk this work has to retire, and the
acceptance probe in *Verification* retires it before any of the C++ below exists.

## Geometry and tensors

962,485,872 B, 179 tensors, W4A16_NVFP4 group 16 (E2M1 codes, E4M3 per-16 scales, one F32 second
level). Packed are the five body MLPs, `fc`, and ten head projections; everything else is BF16.
1.1128 B parameters: 878.70 M packed 4-bit + 234.10 M BF16 (the 54.92 M E4M3 scales are not
parameters). There is no embedding and no LM head — both are the target's.

| Quantity | Value |
|---|---|
| Body | 5 layers, hidden 4096, 32 query / **2** KV heads, `head_dim` 128, intermediate 12288 |
| Attention | every layer `sliding_attention`, window 1024, **`causal: false`**, **attention sinks** per query head |
| RoPE | NeoX, full 128 dims, theta **10000** (not the DFlash2 draft's 1e7) |
| Convolution | grouped dynamic convolution, 2 taps, 2 sides, group 16, around both sublayers |
| Block | `block_size` 8 = 1 anchor + 7 masks; draft length is runtime, ≤ 7 |
| Taps | `target_layer_ids` `[1,18,35,51,68,85]` → six 4096-wide target states |
| Head | width 1024, 2 lattice layers, 8 heads, FFN 2048, top-8 candidates, `vector_eps` 1e-4, `logit_scale` 8.0 |
| Target's own LM head | `[131072, 4096]`, applied to the draft's final-normed hidden to pick candidates |

Binding namespace: `draft/fc`, `draft/hidden_norm`, `draft/layers/<i>/{input_norm,query,key,value,
output,post_attention_norm,gate,up,down,attention_conv_base,attention_conv_projection,mlp_conv_base,
mlp_conv_projection}`, `draft/final_norm`, and `lilicorr/...` for the head, mirroring DFlash2's
`dflash2/...`. The vLLM mapping of the head's named parts onto the checkpoint's tensors is the
authority; llama.cpp's is the cross-check.

The draft's **context** K/V is not a second projection: each layer's own `key`/`value` weights are
applied to the *same* fused target feature that the query block uses (`hidden_norm(fc(concat taps))`),
with that layer's `k_norm`/RoPE after the projection. That is why the checkpoint carries no
`context_key`/`context_value` objects, and it is the one place this draft's storage differs from the
27B DFlash2 draft ([dflash.md](dflash.md)), which materializes its context through a separate pair.

## Mathematics

**Fused target feature.** With `r_t^l` the residual output of target block `l` at position `t`
(after the mixer residual add, before the next block's input norm):

```text
s_t = concat(r_t^l for l in target_layer_ids, ascending)
c_t = plain_rmsnorm(fc(s_t), hidden_norm)
```

The six taps are blocks `[1,18,35,51,68,85]` of the target's 88; the target's final normed hidden
state is **not** among them, which both implementations state in different words (`+1` in vLLM's
`eagle3_utils`, "output of block i" in llama.cpp's converter) and the `fc` input width settles
(24576 = 6 x 4096).

**Query block.** Inputs are the target embedding of the anchor (the already-committed token) at
absolute position `F` and of the mask token (990) at `F+1..F+K`. Mask row `i` predicts the token at
its own position. Per layer:

```text
n  = plain_rmsnorm(x, input_norm)
(p, finish_a) = attention_conv.prepare(n)
q, k, v = query_key_value(p)
q, k = rope_1d(plain_head_rmsnorm(q|k, q_norm|k_norm), absolute positions)
a  = attention(q, concat(K_ctx,k), concat(V_ctx,v), window 1024, sinks, non-causal)
y  = x + attention_conv.finish(W_output a, finish_a)
m  = plain_rmsnorm(y, post_attention_norm)
(z, finish_m) = mlp_conv.prepare(m)
x' = y + mlp_conv.finish(W_down(SiLU(W_gate z) * (W_up z)), finish_m)
mask hidden = plain_rmsnorm(x', final_norm)
```

Visibility is **symmetric** within window 1024 (`abs(p_key - p_query) < 1024`) and `causal: false`
removes the upper bound entirely, so the mask rows see each other as well as the context. Sinks are a
per-query-head constant logit entering the Softmax as an extra column with **no value
contribution**: the row max includes it and the denominator gains `exp(sink_h - max)`. That is
llama.cpp's implementation (`ggml-cpu/ops.cpp`) and vLLM's `sinks=` argument, and it is an extra
term, not a learned key.

**Dynamic grouped convolution** is intra-block. With `pos = row mod W` (`W = 1 + K`, runtime),
`g = c / 16`, `side` selecting the prepare or finish half of `base_kernel[2,2,4096]` and `delta`
reshaped from `kernel_projection(hidden)` to `[rows,2,2,256]`:

```text
conv_s(X)[row,c] = (base[s,0,c] + delta[row,s,0,g]) * X[row,c]
                 + 1[pos >= 1] * (base[s,1,c] + delta[row,s,1,g]) * X[row-1,c]
```

The output side reuses the delta its branch's prepare produced. Position zero of the block has no
preceding tap, so nothing crosses a round boundary and the convolution has no history.

**Head.** For each of the 7 mask slots: project the mask hidden through the **target** LM head,
plain log-softmax over the full vocabulary, take the top 8 by logit (`K = 8`). Five scalars per
candidate, `rank` 0-based:

```text
[ log p, p, log p - log p_top1, rank/(K-1), 1{rank == 0} ]
f = feature_mlp(SiLU)(feature_norm(features))        // LayerNorm over the 5, 5 -> 1024 -> 1024
node[s,k] = token_proj(target_embed(id[s,k])) + pass_hidden_proj(mask_hidden[s])
          + f[s,k] + slot_embedding[s] + rank_embedding[k]
```

The lattice is `7 x 8 = 56` tokens, slot-major, through two pre-norm RMSNorm layers of
**bidirectional** attention (8 heads x 128) whose only mask is an additive per-head bias, plus an
ungated MLP 1024 -> 2048 -> 1024:

```text
bias[q,k,head] = relative_slot_bias[head, slot(q) - slot(k) + 7] + same_slot_bias[head] * 1{slot(q) == slot(k)}
```

then `output_norm`. The anchor is outside the lattice:

```text
anchor_state = anchor_norm(context_proj(hidden_norm(s_anchor)))   // s_anchor = fused feature at the last committed position
anchor_out   = l2norm(anchor_out_head(anchor_state), eps)
pre          = silu(factor_input_proj(cat(node, anchor_state, node ⊙ anchor_state)))
(out_vec, in_vec) = l2norm(dequant(edge_weight) @ pre + edge_bias, eps)   // per candidate
scores       = 8.0 * cosine                                    // [7, K, K], row 0 vs anchor_out broadcast
```

The walk is sequential over the 7 slots: start from predecessor 0 (the anchor's row), and for each
slot take `argmax` of the row the committed predecessor selects, then chain. Greedy `argmax` is the
default in both implementations and the only mode the paper uses; vLLM's optional probabilistic mode
makes `q = softmax(scores/temperature)` over the 8 candidates with `-inf` elsewhere, which needs no
change to the target's rejection sampler because it already resamples from `(p - q)+` on rejection.

## What the target must expose

1. **Six residual-stream taps**, blocks `[1,18,35,51,68,85]`, for every position the draft's context
   covers and for every verified/candidate position. `forward_block` already publishes exactly this
   tensor as `BlockStage::Block` (the residual output, `h + mixer(norm(h))`) through
   `BlockObserver`, so the capture is a copy of a tensor the target already produced.
2. **Its embedding table and its LM head**, both already bound: the draft's candidates are the
   target head's top-8 over the draft's hidden.
3. **A width-`W` batched verify pass** and the acceptance contract, per
   [dflash.md](dflash.md#target-verification-and-committed-prefix), plus the state rollback a partial
   accept needs (below).

## Ops: what exists, what changes

| Need | Landed today | Work |
|---|---|---|
| Draft attention: 2 KV heads, window 1024, symmetric, sinks | `sliding_causal_attention` takes any `Hq`/`Hkv` at `head_dim` 128 and a symmetric window through `position_q_high`; `sliding_window_attention` is pinned to `Hkv=8`/2048/4096 and stores FP16 V | add the **sink operand** to `sliding_causal_attention` (one extra column in the online Softmax: max and denominator only) |
| Draft convolution: H=4096, G=256, BF16 projection | `dynamic_grouped_conv` implements this algebra but pins H=5120/G=320 and requires a `Q8_G32_FP16` projection | generalize the geometry and admit the source's own BF16 projection |
| Body/head linears | bf16 has a slow generic fallback; none of the draft's 6 NVFP4 shapes is registered (plan-time `unsupported shape`); FP8's k=4096 set is the target's | one **runtime-shape** route per format (bf16, NVFP4) admitted for the draft's row domain instead of 13 hand-written entries |
| Candidate selection | `linear_topk` returns top-16 scores for target-shaped heads, with no full-vocabulary log-partition | the head's 5 features need `log p` and `p` from a **full-vocab log-softmax**: return the partition (or the top-8 log-probs) from the same pass over the head rows |
| LiLiCorr head | nothing | new: `lilicorr_candidates`, `lilicorr_nodes`, `lilicorr_lattice`, `lilicorr_edges`, `lilicorr_scores`, and the walk |
| DFlash2's geometry gates | `rope_theta != 1e7` is rejected | parameterize the draft RoPE theta |

## Storage and state

Draft context K/V is a per-layer cyclic ring of the last 1024 absolute positions, BF16 K and V:
`5 x (2+2) x 2 x 128 x 1024 = 5.24 MiB` per lane, against the target's 162 MiB of Mamba state and
8 KiB/token of attention K/V — the draft's state is not a resource question. Nothing else survives a
round except the pending target features (captured but not yet materialized into the ring) and the
target's own state.

## Round

One round per engine round, as DFlash: propose, verify, publish, with the target the output
authority. `W = K+1 <= 8` columns and one anchor. The draft's proposal is a point mass per slot
(greedy walk), so the target's acceptance is the greedy prefix match and the target's own argmax is
the correction on mismatch.

The one piece nemotron_h needs that DFlash does not: **the Mamba-2 state must roll back to the
accepted prefix**. The attention K/V rolls back by advancing the frontier by `N`, but a Mamba block's
convolution history and FP32 recurrent plane cannot be un-advanced. The fix is the shape GDN already
uses for ReplaySSM: record the per-column scan inputs for the verify block (the convolution's
`x`/`B`/`C` output and `dt` are ~10 KB per column per layer, so ~3.3 MB per round over 40 layers),
and after acceptance re-run the scan from the last committed state through only the accepted columns.
Nothing else about the round is family-specific.

## Verification

1. **Oracle** — `tools/verify/lilicorr_reference.py`: an FP32 torch implementation of the whole
   drafter from the two external sources, dumping every intermediate stage, used stage by stage by
   the op tests and the composition test, exactly as `nemotron_h_reference.py` is for the target.
   The one genuinely ambiguous choice (the mapping of `in_head`/`out_head` onto the fused edge
   weight) is switchable in the oracle so the C++ side can be checked against both readings.
2. **Per-op oracle tests** — FP64 naive oracles for the sink extension, the generalized
   convolution, the runtime-shape linear routes, and every head op, at the real geometry.
3. **Acceptance probe (first, and it needs only the taps)** — capture the six taps on a real
   generation, run the oracle's proposals against the target's own greedy stream, and report the
   accepted length. This measures the pairing the plan flags as unproven (NVFP4 target versus a draft
   trained against the 2-bit QAT variant) without any of the draft's C++ existing. The bar is the
   checkpoint's own SPEED-Bench average, **4.30 at draft length 7**. Its round loop is checked
   independently of the drafter: `--self-check` proposes the target's own tokens and asserts that
   every slot is accepted, that every position after the first frontier is committed exactly once,
   and that each round commits `accepted + 1`; it passes at draft lengths 1, 3 and 7.
4. **End to end** — `ninfer --draft-k 7` acceptance length and output tokens/s on the real artifact,
   against the target's own 90 t/s decode and 953 t/s warm prefill. The paper's own gains (acceptance
   +9..19% over a plain DFlash drafter, head cost ~3% of a block) are the reference for what the head
   is allowed to cost.

## First result: the acceptance probe, and what it leaves open

The probe runs, and its first measurement is **not** the number the card predicts. On the real
artifact's taps, with the references' own mathematics:

| | |
|---|---|
| Accepted drafts per round | **0.000** over 128 rounds |
| The target's own tokens' rank under the drafter's hidden | 10^2 .. 10^4 (uniform would be ~65000) |
| The candidate log-probabilities at a slot | top-8 sum ~0.005 (a working drafter concentrates ~0.5+) |
| The drafter's hidden scale | sane: row RMS 0.83-1.56 |
| Does the drafter read its context? | yes: ablating 1024 context rows to 1 moves the mask rows' logits by up to 35 |

**Already excluded —** the interior taps are the values the drafter asks for. Only the *last* block's convention was
   validated end to end originally (`tools/verify/lilicorr_tap_check.py`, 126 of 128 of the engine's
   own drawn tokens at rank median 0). The five interior taps were then checked against the release's
   **own** modeling code: the checkpoint ships `modeling_nemotron_h_puzzle.py`, and running it
   (meta-init from `llm_config`, shards streamed, vision tower ignored) over the same tokens gives,
   for all seven taps, **rel L2 0.023-0.105 and cos >= 0.9945** against our capture over the
   bit-stable first 32 positions -- and that is *after* subtracting the reference's own spread, so
   what remains is at or below the conversion's recorded NVFP4 expert floor of 0.081. Every
   neighbouring convention is 1.8-8.7x worse (blocks `L+-1`, `block[L].in`, both pre-norm norms, the
   mixer output, the embeddings, `norm_f`), which is what makes the winner the block output rather
   than a coincidence. The gate position is decisive in the other direction too: the wrong gate order
   rejects everything. **Our taps are the values the drafter asks for.**

**And the drafter works on the pair it shipped for.** Run in stock llama.cpp master (`6420cf5d`, CUDA
13.3, sm_120a, `--spec-draft-n-max 7`, `--temp 1.0`) against the 24 GB target, the same checkpoint
accepts **mean length 4.06** tokens per verify round -- 1859 rounds, per-request range 2.70-4.80,
pooled 3.61 over 6721 tokens, against the card's 4.30 -- and generates at 192-336 tokens/s against
165-168 without the draft. So the checkpoint is functional and our 0.000 is on our side. Two
possibilities remain, and they are distinguishable:

- **The pairing, exactly**: the BF16 release's residual stream differs enough from the QAT target's to
  kill the drafter. Note what this would have to mean, because the tolerance measurement below rules
  out the *precision* version of it: perturbing our conditioning by +10% costs the drafter 1.09x, so a
  difference of that size cannot kill it. A pairing that explains 4.06 -> 0.000 must be structural --
  a differently post-trained model's internals, not a quantization of the same one.
- **Our conditioning path is defective** -- the `fc`/context/block/head that both the BF16 and the QAT
  taps pass through, which is precisely the part the tap verification cannot see, because both sides
  of that comparison share it.

The decisive dump is the QAT target's own tapped residual stream (`[2,19,36,52,69,86]` in llama.cpp's
numbering), fed to `tools/verify/lilicorr_acceptance.py` verbatim: ~4 accepted there proves the
pairing, and ~0 localizes the defect to the conditioning path.

**That test was run, and it localizes the defect to our reference.** llama.cpp exposes exactly the
tensors the draft consumes (`llama_get_embeddings_layer_inp`, filled at the top of each block), so the
QAT target's own residual stream at the six tapped layers was dumped for a sequence it generated
itself, alongside the BF16 release's for the same tokens, and fed to our probe:

| Conditioning source | Our reference's accepted drafts per round |
|---|---|
| our converted target's taps | 0.000 |
| the **BF16 release's own** residuals | 0.008 |
| the **QAT target's own** residuals (the pair llama.cpp serves at 4.06) | 0.000 |

So it is not the taps, and not the pairing: our reference fails on conditioning that a working
runtime serves at ~4. What has since been excluded by direct comparison against llama.cpp's own
implementation, whose graph is callback-instrumented: the head's algebra (five features in the same
order, node sum, anchor through `context_proj`, the lattice with its materialized `[56,56,8]` bias,
`factor_input_proj(cat(x, anchor, x*anchor))` → silu → `in_head`/`out_head` → `l2_norm`, the
`anchor_out · in[slot 0]` and `out[s] · in[s+1]` score indexing, `logit_scale`), the block's order
(norm → conv(0) → q/k/v → q/k norms → rope → cache-aware non-causal attention with sinks → conv(1) →
residual; then norm → conv(0) → SwiGLU → conv(1) → residual), the context K/V injection, the rope
extent (full 128 at theta 1e4) and the sink's scaling (ggml adds it unscaled *after* the logit
scale), plus the tensor→role mapping verified **by value** against the vendor's GGUF (all 30 BF16
linears map to exactly the tensors this reference binds). The defect is therefore numeric inside the
reference, and the instrument that finds it is the stage-by-stage diff against llama.cpp's named
graph nodes (`inp_g_embeddings`, `Kcur_injected`, `noise_norm`, `attn_conv_in`, `Qcur`/`Kcur`/`Vcur`,
`l_out`, `result_norm`, `result_output`, `lilicorr_lattice`), which is the next step.

**And the next step found it: one missing line.** `DraftBody.forward` never applied the attention
**output projection**. llama.cpp's attention builder returns `wo(attn)` and its finish convolution
consumes *that*; our reference applied the convolution to the raw attention output, so from layer 0
on every stage was wrong in a way no invariant check could see. The evidence is arithmetic against
llama.cpp's own dumped stages: recomputing its `attn_conv_out.0` from its dumped `Qcur`/`Kcur`/`Vcur`
and the checkpoint weights gives rel L2 **2.1e-4 with the projection and 1.0004 without** (and its
`attn_conv_out.0` is 6.0e6 where ours was ~0). With the line added, **every stage agrees to 1.1e-2 or
better** (`result_norm` 4.6e-3, the draft's logits 1.1e-2) and the walk returns *exactly* the ids
llama.cpp proposed for that round.

The acceptance, before and after, per conditioning source (128 rounds, draft length 7):

| Conditioning source | accepted drafts per round, before | after |
|---|---|---|
| our converted target's taps | 0.000 | **2.122** (slot 0: 70.7%) |
| the QAT target's own residuals | 0.000 | **1.783** (slot 0: 71.7%) |
| the BF16 release's own residuals | 0.008 | -- |
| llama.cpp on the QAT pair, same draft length | 3.367 (slot 0: 68.9%) | -- |
| our reference on a *run-consistent* QAT capture | -- | 3.714 accepted, **4.714 tokens per round** |

So the drafter **does** transfer to this engine's NVFP4 W4A16 target: the reference accepts ~2.1
drafts per round on our taps, and reproduces llama.cpp's own walk ids and its acceptance once the
conditioning comes from a same-run capture. The residual gap is the BF16-versus-QAT conditioning
difference measured above (0.08-0.32 rel L2 on the high-magnitude rows), and how much of the
remaining ~1.2 accepted drafts it accounts for is the open question -- not whether the port can work.
Two lessons worth carrying: an invariant-based self-check cannot catch a *missing* operation (114 of
them passed on the broken body), and the cheap instrument that does catch one is a stage-by-stage
diff against a runtime that already implements the same mathematics.

So the conditioning is *weakly informative* rather than random, which is a different symptom from
"the wiring is wrong" and from "the tokens are unrelated". Two of the three explanations are now
ruled out by measurement and one is live:

1. **A convention or implementation error** in the reference implementation or the probe's round
   construction. *Tested*: a line-by-line audit of both files against the two sourced specs, with
   every candidate change **measured** rather than argued -- the round's anchor/slot conventions,
   offsets by +1/0/-1, the context including the bonus row, tap order reversed/zeroed/shuffled, the
   window causal/off/64/8, sinks off, theta 1e6, interleaved RoPE, the conv's sides swapped, the
   norms zero-centred or set to 1, and the tap set shifted by one. **Every one of them is equal or
   worse.** Two further independent confirmations came out of it: the trained `relative_slot_bias`
   itself confirms the offset-7 convention (columns 0 and 14 are exactly 0.0, the diagonal peak sits
   at column 7), and the vendor's own GGUF reproduces the materialized `[56,56,8]` bias from those
   two tables with a **max abs difference of 0.0**.
2. **The pair, not the code -- the surviving explanation.** The drafter shipped as the companion of
   the 24 GB target, and the plan's own reconnaissance records why that is not a mere precision
   question: that release is **QAT**, i.e. a differently *post-trained* artifact, not a quantization
   of the release we converted. A drafter conditioned on one post-trained model's residual stream and
   run against another's would degrade to exactly this. *Test*: run the drafter against the target it
   was trained against (the vendor's own recipe on the 24 GB checkpoint) and measure its acceptance
   there -- the one decisive experiment this document still lacks, and it needs a 24 GB download plus
   a vendor runtime on this machine.
3. **Our interior taps being subtly off -- excluded**, above.

The capture itself is not in question: it is deterministic, its equal-width columns are bit-identical
across the two paths, and the last block's residual survives the target's own head.

**The same audit measured how much the drafter tolerates, and that is what makes explanation 2
awkward.** Perturbing the conditioning by +10% costs it only 1.09x in rank, shuffling the whole
context 3.7x, and zeroing it entirely 3.4x. A drafter that shrugs off a ten-percent perturbation of
its own input cannot be killed by a conversion's few-percent residual difference, so a
precision-only reading of the pairing does not fit; the mismatch, if it is the pairing, has to be
identity-level (a different post-trained checkpoint rather than a quantization of the same one).
That is exactly what the vendor's own two-model split would be, and it is why the decisive
experiment is a run against the target the checkpoint actually shipped for rather than another
variation of ours.

**If the QAT taps are what fix it, the options are product decisions rather than port work**, because
none of them is inside the draft's own implementation:

| Option | What it costs |
|---|---|
| Serve the 24 GB QAT target, which is what the drafter was trained for | a 2-bit expert codec this tree does not have -- the route the NVFP4 decision was taken to avoid |
| Convert *from* the QAT checkpoint at a higher precision instead of from the BF16 release | a converter source change; the architecture is identical, so nothing in the loader moves |
| Obtain or adapt a drafter for the BF16 release | no such checkpoint is published; adapting one needs the DFlash training machinery, which lives only in unmerged upstream branches |
| Take the EXL3 route and accept the pairing risk a second time | the plan's own warning: EXL3 was the mismatch that made the Gemma 4 drafter accept almost nothing |

What the port owes in every one of those cases is unchanged -- the draft's ops, its config and
binding, the converter recipe, and the Program's round -- so that a drafter which does transfer can be
served. What it must not do is build them for a checkpoint whose target is unknown to work.

**A magnitude trail that looked like a bug, and is not one.** The reference implementation's residual
stream grows by ~1e6 per layer, reaching ~1e8 -- so it is worth writing down why that is *not* the
explanation, because it is the first thing a reader of the trace will suspect. The dynamic
convolution's coefficient is `base + delta`, and `kernel_projection`'s rows have L2 norm 12.6-20.4,
so a coefficient RMS of 2-31 is what *any* implementation gets, the vendor's included; every stage
then follows from the plain algebra (a 3.15-RMS pre-MLP input through a `gate`/`up` whose row norm is
16.6 gives the measured 52 RMS, `silu(gate) * up` squares it to 2495, and `down`'s row norm 28.8
gives 99700). Each sublayer renormalizes its input, so its contribution is a constant and the
residual grows *linearly* rather than geometrically; the vendor serves this drafter in bf16 at
acceptance 4.3, and our own trace's magnitudes are the ones the vendor's arithmetic produces. The
runaway is a property of the checkpoint, not of the port.

The vendor's own GGUF is a **pure re-container** of this checkpoint -- 102 unquantized tensors
element-wise identical, 25 NVFP4 weights within 4.4e-8 relative after dequantization, 26 scale
tensors bit-identical -- so it also settles two conventions directly: the norm gains are stored
**raw** (no `1 + w`, and they are not zero-centred: `blk.0.attn_norm` mean +0.073 / RMS 0.078,
3 of 4096 values negative), and the convolution is **not** rescaled. It also reproduces the
`[56,56,8]` materialized attention bias exactly (max abs difference 0.0) from
`relative_slot_bias` + `same_slot_bias`, which is an independent confirmation of the head's bias
construction and its offset.

## What a verify pass costs, measured

Before building the drafter, the question that decides whether it pays is what an 8-row forward
costs on this target, because that *is* the verify pass. Measured on the artifact
(`nemotron_h_74b_nvfp4.ninfer`, eager decode, no CUDA graphs, in-domain corpus):

| shape | measured | per row |
|---|---|---|
| pp8 -- one 8-row forward, i.e. one verify pass | 91.92 t/s -> **87.0 ms** | 10.9 ms |
| pp64 | 184.07 t/s -> 348 ms | 5.4 ms |
| tg64 -- decode | 61.07 t/s -> **16.4 ms per step** | 16.4 ms |

At LiLiCorr's measured acceptance (2.122 accepted drafts plus the bonus token = 3.12 tokens per
pass) an 87 ms verify pass yields **0.036 tokens/ms against the 0.061 the engine already produces**
-- a 1.7x *slowdown*. The decode step is weight-bandwidth-bound: 24.91 GiB of weights per step over
16.4 ms is 1.63 TB/s, essentially the 5090's 1.79 TB/s, so a verify pass *should* cost about a
decode step plus the arithmetic of seven more rows. The 5.3x the prefill-shaped route charges comes
from small-batch inefficiency and, for this mixture-of-experts, per-row expert routing -- visible in
pp8's 10.9 ms/row against pp64's 5.4, which is expert reuse amortizing over more rows.

**Consequence: the port's first slice is a decode-shaped 8-row forward for nemotron_h, not the
drafter's ops.** The draft pays only if a verify pass lands near the bandwidth floor: 3.12 tokens
per ~20-25 ms is 125-155 t/s against 61 (eager) or 90 (graphed) today, i.e. 1.4-2.5x. Building the
drafter first would spend the days and then find this ceiling.

The row-count curve says the ceiling is a **route boundary, not a per-row cost** (`-p 1,2,4,8,16,64`,
same artifact):

| rows | 1 | 2 | 4 | 8 | 16 | 64 | 512 | 4096 |
|---|---|---|---|---|---|---|---|---|
| ms per pass | 11.1 | 20.2 | 22.0 | 86.8 | 129.1 | 348 | 454 | 3329 |
| ms per row | 11.1 | 10.1 | 5.5 | 10.9 | 8.1 | 5.4 | 0.89 | 0.81 |

The long-prompt points reconcile with the published prefill rate (pp512 1129 t/s, pp4096 1230 t/s),
so the small-row numbers are the same engine in a different regime rather than a measurement
artifact. The amortization is 13x from eight rows to 4096 (10.9 -> 0.81 ms per row), and the
verify-sized pass sits at the **worst** point of that curve.

Up to four rows the pass sits *at* the weight-read floor -- 20-22 ms is 24.91 GiB at 1.63 TB/s, so
those rows are nearly free, which is exactly the regime speculative decoding needs. At eight rows
the engine switches to a prefill-shaped route charging four times that floor, and sixteen and
sixty-four rows stay in it but amortize. The jump is stable (pp8 is 92.18 +- 0.42 t/s), so it is
route selection rather than noise.

So a verify pass *should* cost about 22 ms, and at LiLiCorr's measured acceptance that is 3.12
tokens per 22 ms = **142 t/s against 61 (eager) or 90 (graphed), 1.6-2.3x**. The first work item is
therefore the small-row-count route for nemotron_h: measure where the boundary sits and make five to
eight rows take the decode-shaped route. That is a bounded change in route selection, unlike the
drafter, and it is what makes the drafter pay.

**And it is the expert pool, with a pool size as the fix.** The routed banks are 37.21 GB of
block-scaled NVFP4 that no 32 GB card holds, so the non-resident ones reach the device through the
pager's per-expert slot cache, and the default pool (`--expert-cache-slots 0`) is only the
structural floor. Sweeping it (3 reps, same artifact) across every regime:

| regime | slots 0 (default) | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|
| tg64 (decode), t/s | 61.41 | -- | -- | 62.24 | -- |
| pp4, t/s | 180.8 | 180.8 | 180.8 | 181.9 | -- |
| pp8, t/s | 92.18 | 93.73 | 99.70 | **318.57** | **318.43** |
| pp16, t/s | 127.5 | 127.4 | 127.4 | 122.1 | -- |
| pp512, t/s | 1129.1 | -- | -- | 1091.7 | -- |
| pp4096, t/s | 1229.7 | -- | -- | 1236.5 | -- |

An eight-row call's routed-expert working set fits the pool from 256 slots, and there the verify
pass lands on the weight-read floor -- 86.8 ms becomes **25.1 ms, a 3.5x improvement with no kernel
change** -- and nothing else pays for it: decode +1.4%, pp4 +0.6%, pp4096 +0.6%, pp16 -4.2%, pp512
-3.3% (the pool takes 0.91 GiB from the resident set, weights 24.91 -> 24.00 GiB). **The default
pool is simply undersized for verify-shaped calls, and sizing it costs the other regimes nothing
measurable.**

So the first work item is a **pool size, not a route**: raise the default, or size the pool from the
widest call the engine expects -- eight columns, which is the natural verify width. One consequence
worth stating: `docs/performance.md` and `bench/` recorded every number at zero extra slots, so
changing the default re-bases them and the affected benches have to be re-run with the change.

The draft's economics therefore flip: 3.12 tokens per 25 ms is **125 t/s against 61 eager or 90
graphed, 1.4-2x**. The working-set story fits the eight-column case exactly; it does not by itself
explain why sixteen columns loses 4% at 256 slots, which remains open and does not affect the
recommendation.

**Where the sizing lives** (`plan_experts.cpp`, `load.h`). The expert budget is the caller's cap or
the card's free memory; the two staged banks take theirs first as the pool's structural floor; the
slot cache is **the only elastic term**, granted as `min(requested, affordable)` with `affordable =
room / (pool_stride + 12 bytes)`, and every slot granted is one slot's worth of bank *not* made
resident. The default request is **zero** (`ExpertResidencyRequest::cache_slots` in
`src/models/nemotron_h/load.h`) -- maximum residency, the pool at its floor, which is the pool every
recorded number was taken with and the right choice for one-column decode. It is the wrong choice
for a five-to-eight-column call, which is the width a speculative round verifies.

So the change is a default rather than machinery: either request the count a verify-width call needs
(capped by `affordable`, which the plan already reports back as `cache_slots` / `slots_clamped`), or
have the engine state the widest call width it serves and let the plan size the pool from `top_k`
and that width. Because `docs/performance.md` and `bench/` were all taken at zero slots, the change
re-bases them and the affected benches must be re-run with it -- which is why this is a decision to
take deliberately rather than a line to slip in beside the drafter.

**The default is now 256 slots** (`ExpertCachePolicy::cache_slots` in `include/ninfer/types.h` and
the loader's own request, on the user's call). Measured through the default, with no flag:

| workload | zero slots (old default) | 256 slots (new default) |
|---|---|---|
| pp8 -- one verify pass | 92.18 t/s (86.8 ms) | **317.26 t/s (25.2 ms)** |
| pp512 | 1129.09 t/s | 1085.89 (-3.8%) |
| tg64 -- decode | 61.41 t/s | 61.27 (-0.2%) |
| resident weights | 24.91 GiB | 24.00 GiB |

So the verify-shaped pass gains 3.4x by default and nothing else moves by more than 4%. The
consequence stands: every number recorded in `docs/performance.md` and `bench/` for this family was
taken at zero slots, so those rows predate the default and need re-recording -- a benchmark pass of
its own, not a side effect of this change.

## Draft conversion contract

Everything the emitter and the C++ binding must agree on, taken from the checkpoint itself
(`/mnt/storage/models/nemo/74B/LiLiCorr`) and from the reference's role table, which is the version
that reproduces llama.cpp's walk ids.

**Normalized draft config.** `architectures: ["LiLiCorrDraftModel"]`, `model_type: "qwen3"`,
`block_size: 8`, `hidden_size: 4096`, `num_hidden_layers: 5`, `num_attention_heads: 32`,
`num_key_value_heads: 2`, `head_dim: 128`, `intermediate_size: 12288`, `rms_norm_eps: 1e-6`,
`rope_theta: 10000.0` (not the Qwen1e7 the in-tree draft path assumes), `tie_word_embeddings: false`,
`layer_types: sliding_attention x5`, `sliding_window: 1024`, `vocab_size: 131072`. From
`dflash_config`: `target_layer_ids: [1, 18, 35, 51, 68, 85]` (0-based; llama.cpp prints the same
list plus one because it counts the embedding as layer zero), `mask_token_id: 990`,
`attention_sink_bias: true`, `use_swa: true`, `swa_window_size: 1024`, `causal: false`,
`conv_kernel_size: 2`, `conv_group_size: 16`, `projector_type: "lilicorr"`, and the head's
`lilicorr_candidate_topk: 8`, `lilicorr_hidden_size: 1024`, `lilicorr_num_layers: 2`,
`lilicorr_num_heads: 8`, `lilicorr_mlp_ratio: 2.0`, `lilicorr_factor_dim: 1024`,
`lilicorr_vector_eps: 1e-4`, `lilicorr_logit_scale: 8.0`. Validation against the target mirrors
qwen3_5's `draft_config`: `hidden_size` and `vocab_size` must equal the target's, every tap must be
a distinct valid target layer index, the mask token must be inside the target vocabulary, and
`conv_group_size` must divide the hidden size -- the taps are *absolute* target layer indices, so
they are validated against `num_hidden_layers` rather than stored as offsets.

**The stored tensor inventory is 179 tensors, and the U8 shapes are byte-packed.** NVFP4 W4A16
(`quant_algo: W4A16_NVFP4`, group 16) with the standard ModelOpt triple: a `U8` code plane, an
`F8_E4M3` scale plane at one scale per 16 K values, and a scalar `F32` `weight_scale_2`. The code
plane is **byte-packed along K**, so its stored width is `logical_k / 2` -- `fc.weight [4096, 12288]`
is logically `[4096, 24576]` (six taps of 4096, matching `fuse_taps`), `layers.N.mlp.gate_proj
[12288, 2048]` is `[12288, 4096]`, `mlp.down_proj [4096, 6144]` is `[4096, 12288]`, and
`lilicorr.layers.N.attn.out_proj [1024, 512]` is `[1024, 1024]`. Reading the stored width as the
logical one silently halves every quantized projection, which is why it is written down here.

Twenty-two tensors per draft layer: `input_layernorm.weight` and `post_attention_layernorm.weight`
(4096); `self_attn.{q,k,v,o}_proj.weight` as BF16 (4096x4096, 256x4096, 256x4096, 4096x4096) with
`q_norm.weight` and `k_norm.weight` (128 each); `self_attn.attention_sink_bias` (32, one per query
head); `attention_conv.base_kernel` (2, 2, 4096) and `attention_conv.kernel_projection.weight`
(1024, 4096); the same two for `mlp_conv`; and the three NVFP4 projections
`mlp.{gate,up,down}_proj`. The head adds `hidden_norm.weight`, `norm.weight`, `fc.weight`,
`lilicorr.rank_embedding` (1, 1, 1, 8, 1024), `lilicorr.slot_embedding` (1, 1, 7, 1, 1024),
`lilicorr.relative_slot_bias` (8, 15), `lilicorr.same_slot_bias` (8), `token_proj`,
`context_proj`, `pass_hidden_proj` (all NVFP4 1024-wide with BF16 bias), `feature_norm` (5),
`feature_mlp.{up,down}_proj` (1024x5 then 1024x1024, BF16, with bias), `in_head`, `out_head`,
`anchor_out_head` (1024x1024 BF16 with bias), `anchor_norm`, `output_norm`, and two layers of
`attn.{in_proj_weight (3072, 1024), in_proj_bias (3072), out_proj, attn_norm}` plus
`mlp.{up_proj, down_proj, mlp_norm}`. The attention projections, both convolutions, every norm, and
the head's small matrices stay BF16 in the checkpoint's own exclude list, so only the thirteen
NVFP4 projections per model need the codec.

The emitter's obligations beyond names and shapes: decode each NVFP4 triple with the machinery
`nemotron_h.py` already uses for this family's own weights, emit *logical* shapes, and preserve the
convolution layout the reference depends on (`base_kernel` is `[2, taps, H]` with the side first;
`kernel_projection` rows are `side * taps * groups + tap * groups + g`). The base kernels keep the
checkpoint's own `_tap_major` treatment, which the target's recipe already implements for its
dynamic convolutions.

**Implemented.** `tools/convert/nemotron_h.py` now carries `lilicorr_config()` (the normalizer and
its validation), `_LilicorrBuilder.lilicorr()` (the body) and `.lilicorr_head()` (the 64-tensor head),
and `build_model(..., components=("text", "dflash2"), companions={"dflash2": ...})` records
`records["dflash2"] = {"config": ..., "target": "text"}`; `tools/convert/__main__.py` passes the
companion for this family instead of refusing it. Building the model against the real 917 MB draft
checkpoint declares **139 draft parameters**, every shape checked against its source -- including
`feature_projection (4096, 24576)`, which the ModelOpt reader decodes from the stored `[4096, 12288]`
U8 plane, so the byte-packing is the reader's business and the emitter never sees it. The draft stays
BF16 in the artifact (decoded once, no second quantization); the recipe needs no entry because none
of its name filters match `dflash2/...`. Seven tests in `tests/convert/test_nemotron_h.py` pin the
normalized config, the refusals (wrong width, layer count, theta, window, sinks off, other taps, a
tap past the tower) and the fixture is `tests/fixtures/nemotron_h/lilicorr_config.json`.

## Execution plan

Ordered by dependency; each step names its own check.

| # | Step | Where | Check |
|---|---|---|---|
| P1 | Shared draft-config parser: a family-neutral `DraftSpec` parsed from the `dflash2` component (the same normalized schema the converter now emits), with qwen3_5's `DraftConfig draft(value, target, dflash2)` reduced to an adapter over it and nemotron_h's `Config` gaining an optional draft | new `src/models/draft/`, `qwen3_5/config.cpp`, `nemotron_h/config.cpp:244` | affected build; qwen3_5's config tests unchanged; a nemotron test over the normalized JSON |
| P2 | Declare and bind the draft's 139 parameters; the loader accepts a `dflash2`-carrying artifact | `nemotron_h/{weights,load}.cpp` | synthetic-artifact load test |
| P3a | Sink operand on the draft's attention, unscaled after the logit scale | `sliding_causal_attention` + wrapper | FP64 oracle at the draft's geometry |
| P3b | The draft's conv geometry in the `q8_dflash2` registries (4096/256, group 16) | `ops/attn_input_proj`, `ops/linear_swiglu` | FP64 oracle cases plus the reference's conv stages |
| P4 | Body and head execution, stage by stage | `nemotron_h/execution/` | stage diff against the reference |
| P5 | The round: draft, verify, publish, with `--spec dflash2` for this family | `nemotron_h/program/` | acceptance near 2.122 drafts per round on a real prompt |
| P6 | Pool sized for verify-shaped calls; acceptance and t/s recorded | pool policy, `docs/performance.md` | measured t/s against 61 eager / 90 graphed |

P3a and P3b are independent once P1 fixes the shared struct. The two decisions the plan depends on
and this document cannot make: the pool default (P6's premise, worth 3.5x on 8-row calls on its own),
and whether qwen3_5's parser is migrated to the shared one in P1 or left as an adapter until its next
touch -- the adapter is the smaller diff on a hand-formatted file, the migration is the one-authority
answer.

**P1 landed.** `src/models/draft/draft_spec.{h,cpp}` carries the LiLiCorr `Spec` and
`parse_lilicorr(value, target_hidden, target_vocab, target_blocks)`, with exact member sets at all
three levels, the tower relations (width, vocabulary, in-range distinct taps, mask inside the
vocabulary, conv group dividing the width) and the two flags that make this draft LiLiCorr rather
than plain DFlash2 (sink required, projector `lilicorr` and enabled). It is wired into the loader:
`nemotron_h::Config` carries `std::optional<models::draft::Spec>`, and `parse_config` reads a
`dflash2` component when the artifact has one, refusing a non-`text` target. Deliberately one flavor
and no release constants -- the converter owns the release, the loader owns consistency -- and the
build is green. The binding of the 139 parameters is the next cell: the parser describes the draft,
it does not yet materialize it.

**P2 landed and verified end to end.** The converter now emits, and the loader binds, an artifact
carrying the draft: `/mnt/storage/models/nemotron-74b/nemotron_h_74b_nvfp4_draft.ninfer` is 891 objects in two files, and loading
it runs a 187-token prompt through the tower with all six tap blocks unchanged. `DraftWeights` walks
the spec (five layers x 18, `fc` at `(hidden, taps*hidden)`, coefficient planes at
`2*kernel*groups`, the head's table), `Content` carries the spec so the execution and the weights
share one copy, and the binder accepts the whole draft or none of it.

Three defects only the end-to-end run could find, all fixed: the two structural embeddings are stored
with singleton axes (`[1,1,1,8,1024]`) and needed a reshape-on-read source; the tower's recipe's
`"/attention/"` rule reached into the draft and quantized its attention *and its biases* (the draft
is now skipped wholesale and stored as its checkpoint holds it); and a dense parameter may be rank 1
or 2 only, so both conv base kernels are carried flat as `[2, taps*H]` -- the order the convolution
indexes -- on the converter and loader sides together.

What remains is execution: P3a (the sink), P3b (the draft's conv geometry in the `q8_dflash2`
registries), P4 (body and head), P5 (the round). The `flash_partial` split route adds the sink to the
merge rather than to each split, which is the one subtlety P3a carries.

**P3a landed and qualified.** `sliding_causal_attention` takes an optional per-query-head sink (BF16
`[Hq]`, a trailing operand, null by default so every existing caller -- the tower's blocks and Gemma
4's -- is byte-identical). The one-pass route seeds the running maximum and sum from it (`row_max =
sink`, `row_sum = 1`), which is the reference's `den = sum exp(w - m) + exp(sink - m)` exactly; the
split route *gates* the seed on `p.splits == 1` and the merge raises its maximum before forming the
split weights and adds `exp(sink - max)` once from thread 0. Two new op-test cases at the draft's
geometry (8 rows over window 1024, and decode width over a rotated ring) qualify it against the FP64
oracle, whose sink fold raises the same maximum and adds the same term -- `OK
sliding_causal_attention`. The first run failed precisely where the asymmetry lives: an unconditional
seed counted the sink once per split *and* the merge added it again, which reads as 0.49 relative
error at 8 rows (few visible keys) and 2% at decode width (diluted by a full ring); the one-pass
route passed both cases, which is what isolated it to the split path.

**The draft's linears need no new route.** `src/ops/linear/bf16/bf16_dispatch.cpp` ends in
`return &bf16_generic_launch` for any `(n, k)` its table does not carry, and the table already holds
`(1024, 4096)` and `(4096, 1024)` -- two of the draft's. So every draft projection computes through
the generic BF16 route today; the per-shape kernels are the intended *performance* route for a width
a model actually runs ("eight rows", the draft's block), and adding them is a tuning pass whose
correctness is already pinned by the op tests at any shape. P3b is therefore **only** the tap
application: `y[h,t] = sum_{s,k <= t mod block} (base[s*taps*H + k*H + h] + delta[(s*taps + k)*G +
h/16, t]) * x[h, t-k]` over 4096 channels in 256 groups of 16, with `delta` in the projection's own
row order so nothing is shuffled, plus its FP64 case at the draft's geometry.

**P3b landed and qualified.** `ops::grouped_tap_apply` (contract in
`include/ninfer/ops/grouped_tap_apply.h`, one thread per output element) computes the reference's
finish convolution **for one side per call** -- `conv_s(X)` in the Mathematics above -- with the
artifact's side-major `base`, `delta` in the projection's own row order, and the block-relative bound
`k <= t mod block_size`. Its four cases (the draft's geometry on the prepare half and on the finish
half, two blocks over a 16-row window, a single-token window where only tap zero is reachable, and a
block of three over a window of seven) pass against the naive FP64 oracle, `OK grouped_tap_apply`.
The side is a parameter rather than a loop because the forward applies the two halves to *different
inputs* -- the prepare half to the normed block input, the finish half to that branch's projected
output -- and the first version of this Op summed them into one output, with a test oracle that
encoded the same mistake and therefore passed; the contract and the oracle now state one side each.
With the sink, this closes the draft's op work: its linears run on the generic BF16 route
(`linear_pair` + `silu_mul` for the gated MLP, `linear_swiglu` where the artifact packs the pair),
and the two ops it needed at its own geometry now have FP64 oracles at that geometry.

## The draft body's precision, measured

The body's residual stream and norms now run FP32 end to end (widening casts at the BF16-only
operands: embedding, linears, grouped convolution, attention, and `pass_hidden` for the head),
backed by FP32 routes on `residual_add`, `rmsnorm` (plain epilogue, BF16 gain), `silu_mul`,
`add_bias` and a `cast_bf16_to_fp32`, each qualified at 8x4096 (exact for the adds and the cast,
1e-4 for rmsnorm, 1e-5 for silu). Same prompt, same 32 rounds:

||accepted/round|rank-0 chain|slot 0|
|---|---|---|---|
|wide commits, BF16 body|0.844|1.562|0.563|
|**wide commits, FP32 body**|**1.281**|**1.937**|**0.656**|
|width-1 commits, BF16 body|1.312|2.062|0.719|
|width-1 commits, FP32 body|0.937|1.625|0.594|
|reference|3.000|2.938|0.688|

The FP32 body recovers **+0.44 of the measured 0.876** at wide commits -- about half the term, as
predicted (silu, the attention output and the linears still round to BF16). The width-1 arm did not
keep its advantage after the precision change (0.937 against the wide arm's 1.281, slot 0 0.594
against 0.656); the two arms each generate their own continuation, so this is a trajectory
comparison over 32 rounds, and the wide arm now has the better first decision as well. The remaining
half of the term is the draft's own BF16 weights, which is a draft-weights precision decision.

Cost: draft forward 42.6 ms (was 41.5; the casts are ~1 ms), verify 82.7, commit 27.7, round
153.5 ms -- 2.28 tokens per round, still ~15 t/s against decode's 61.

### The graph slice's pinned gate

The draft forward is the first cost target, and the invariance target is measured on this tree, not
quoted: with the graph off, every arm must reproduce

|arm|accepted/round|rank-0|slot 0|draft|verify|commit|
|---|---|---|---|---|---|
|wide commit|1.28125|1.9375|0.65625|42.3 ms|76.5 ms|26.1 ms|
|width-1 commit|0.9375|--|--|--|--|35.0 ms|

and with the graph on, the first four columns must be *identical* plus verify row 0 exact and the
bonus agreeing with the committed pass. Three captured-work hazards must be resolved for that to be
possible, each of which would otherwise fail silently:

1. `fill_i32_positions` compiles its base into the kernel -- `body()`'s block positions and
   `context()`'s `work_first` positions would replay round 1's values.
2. the tap copy inside `context()` has both a variable source offset (`(work_first - ctx_first) *
   nb[1]`) and a variable extent -- both are baked into a captured copy.
3. the block-ids upload is pageable H2D, which is illegal inside a capture (the one failure that is
   loud).

The design: pinned host staging for ids and positions written per round; a persistent tap operand
filled by a pre-launch copy; the per-round copies *outside* the capture; the body alone captured per
lane with `DecodeGraphDefinition::capture` -> `instantiate` -> `launch`; the head outside (host
full-vocab select); bypass on a non-null observer and on `NINFER_DRAFT_NO_GRAPH=1`.

**Two prerequisites found by implementing the hoist, and they are not optional:**

1. `context()` allocates its tap operand with `work_columns`, which varies with the previous
   round's acceptance, so the arena's bump position at the start of `body()` is round-varying. The
   body needs its own arena (a second `DeviceArena`, rewound per round) or its planes sit at
   different offsets every round and a captured body writes where a later, wider context workspace
   sits.
2. `context()` assigns `anchor_hidden` from a fresh arena slice, so the body reads a pointer that
   differs every round. The anchor must be written into a fixed plane that the captured body reads.

Landed so far: the context phase is outside the body, the ids and both position planes come from
fixed staging written before the round, and the round's gate is unchanged (1.28125 / 1.9375 /
0.65625 with row 0 exact, round 140.5 ms).

**The graph is in and it changes nothing -- which is the finding.** The body now runs from its own
rewound arena, the anchor is written into a fixed plane, and a per-lane `DecodeGraphDefinition` is
captured on the first round and launched thereafter (head outside, bypass on a non-null observer and
on `NINFER_DRAFT_NO_GRAPH=1`). The gate held exactly. The draft forward is **unchanged at 41.7 ms**,
so it was never launch-bound: a graph removes host gaps, and the time did not move, meaning the 41 ms
is real GPU work -- roughly 900 MB of BF16 weights read at ~22 GB/s effective, about 70x below the
card's bandwidth. The lever is therefore the draft's small-batch GEMM efficiency (T = 8 rows across
4096x4096 and 10240x4096 weights), an Op performance task, not scheduling.

`block_size` 4 against 8 is not a test-level experiment: the slot count *and* the head's per-slot
embeddings and rank tables are shaped by the artifact's draft spec (verified in `derive`), so the
smaller block needs a regenerated artifact -- a converter change outside this slice. The slot rates
say what it would buy: slots 4-6 contribute 0.094 of the 1.281 acceptance.

**The verify's cost is PCIe expert staging, measured.** `nsys` whole-run attribution: `expert_cache_fetch_kernel` alone is 44.8% of GPU time (1728 instances, 950 us average, 3.47 ms max), and the kernel body is a plain `uint4` copy from the expert bank into the pool (`src/ops/offload_moe/kernels.cu:397`). At ~950 us and ~50 GB/s that is ~47 MB per fetch; ~54 fetches a round is ~2.5 GB of PCIe traffic per round. That is why a larger pool changes nothing (512 and 1024 slots measured identical: the proposals' experts are novel every round) and why a decode step, which hits the pool, is 16.4 ms against the verify's 75.

The consequence for this target: 74B experts do not fit device memory on a 32 GB card (NVFP4 ~37 GB), so host residency is forced and a proposal block's staging is unavoidable at block 8. Speculation pays here only if that traffic drops -- a verify that reuses an expert set, acceptance far above the reference's 3.0, or a card that holds the experts.

**The width sweep, measured, and it closes the question.** `NINFER_DRAFT_SLOTS` (1..7, the
artifact's own tables, no regeneration), 32 rounds each, same prompt:

| slots | accepted | draft | verify | commit | round | tokens | ms/token |
|---|---|---|---|---|---|---|---|
| 1 | 0.688 | 30.8 | 33.2 | 24.5 | 88.5 | 1.69 | 52.4 |
| 2 | 0.844 | 34.3 | 38.5 | 23.5 | 96.3 | 1.84 | 52.3 |
| 4 | 0.500 | 36.0 | 50.4 | 22.8 | 109.2 | 1.50 | 72.8 |
| 7 | 1.281 | 41.6 | 74.8 | 26.0 | 142.4 | 2.28 | 62.5 |

The verify falls far less than proportionally: a 2-row pass is 33.2 ms where 8 rows are 74.8, so it
splits into a **width-independent floor of ~20 ms** (one row already costs 19.7 -- all 40 layers'
weights and the state read, the same pass a decode step is) and a **marginal ~7.4 ms per verified
row** (the expert fetches). Removing the fetches entirely at two rows still leaves twice a decode
step. No width beats decode: the best is 52.3 ms/token against 16.4 (slots 1-2), and with a 3 ms
draft it is 33-36 ms/token, still ~2x.

Nor can the commit pass be dropped as a width trick: the verify advances the Mamba state by
`1 + slots` columns and no plane can be un-advanced, so the commit re-runs exactly the accepted
prefix. At one slot that is 24.5 ms for 1-2 tokens; removing it needs per-column scan replay or
per-column snapshots (8 x 162 MiB).

The arithmetic in full: a round needs two target passes; a pass costs a decode step; the block's
marginal rows cost 7.4 ms each in PCIe staging; acceptance is 0.7-1.3. Break-even needs the marginal
rows near free, which is exactly the expert-residency wall. **On this target and card the spec model
runs correctly and cannot beat decode.**

**The generic BF16 route is packed, and it is the draft's first real cost win.** Every draft matrix
(and every other fallback shape) fell through to `bf16_generic_kernel`, whose inner loop read one
BF16 element per lane with row-fastest threads -- every lane in a different sector. It now reads
32-byte packs (`load_bf16_weight_pack<16>`) while keeping one FP32 accumulator per (token,row),
ascending K and one BF16 rounding at the store, so the route is bit-identical: measured draft
forward **41.7 -> 22.9 ms**, round **142.6 -> 120.1 ms**, committed pass 25.6 -> 21.7 ms, with
acceptance 1.28125 / rank-0 1.9375 / slot 0 0.65625 unchanged and verify row 0 exact.

## Where the port stands

**The condition for the round to beat decode, stated once.** A round replaces `1 + accepted` decode
steps, so it pays only if `draft + verify + commit < (1 + accepted) x 16.4 ms`. At the measured
acceptance (1.281) that budget is 37.4 ms against a round of 143 ms. The terms:

| term | measured | in decode steps | fixable? |
|---|---|---|---|
| draft forward | 41.7 ms | 2.6 | yes -- small-T GEMM (the cost is ~22 GB/s effective, not launches; the graph proved it) |
| target verify (8 proposal rows) | 75.0 ms | 4.6 | only by expert traffic: 49 ms warm for *proposal* rows against 25.2 for 8 *prompt* rows, because consecutive prompt tokens share routing and proposals do not |
| committed pass | 25.9 ms | 1.6 | partially -- keeping the verify's accepted rows would remove it, but the taps would then come from an 8-wide pass and acceptance falls 1.281 -> 0.844 |

With the draft fixed and the commit at one row the round is ~98 ms -- 23 t/s. At the reference's
acceptance (3.0) it is ~38 t/s. **Break-even needs the verify near 20 ms (about 1.2 decode steps)**,
i.e. a block verify whose expert traffic is close to one token's; at 20 ms the round is 43 ms for
2.28 tokens (53 t/s), and with acceptance 3.0 it is 93 t/s. That is the condition the remainder of
this work has to meet, and it is a target/verify-kernel question, not a drafter one.

Every piece is in place and measured, and the verdict is arithmetic rather than opinion. The round
works (verify row 0 reproduces decode exactly; the verify pass's bonus equals the committed pass's
draw in 31 of 32 rounds). The drafter's arithmetic is validated -- slot 0 reached 0.719 against the
reference's 0.688 under width-1 conditioning with the BF16 body -- and the body's BF16 accumulation
was worth a measured +0.44 acceptance, recovered by the FP32 residual stream: **1.281 accepted drafts
per round at wide commits** (rank-0 chain 1.937), against 0.844 before that change. The width-1 arm
lost its advantage after the precision change (0.937, slot 0 0.594), so wide commits are the arm to
carry forward.

What blocks shipping is the round's cost, not its correctness:

| component | measured | target |
|---|---|---|
| draft forward | 42.6 ms, **device-bound** (a hundred small kernels over 917 MB; the earlier "26 ms host gap" is the body draining at the first sync) | ~3 ms with CUDA graphs |
| target verify (8 proposal rows) | 82.7 ms, of which ~33 ms is first-touch expert staging on proposal-shaped routing; warm floor 49 ms | ~25 ms, the prompt-row floor |
| committed pass | 27.7 ms wide | -- |

At 2.28 tokens per ~154 ms that is **~15 t/s against the engine's 61 t/s decode**: speculation is
roughly 4x slower here, and making the draft free would still leave ~109 ms for 2.28 tokens. The
causes are the three per-pass costs that a round multiplies rather than amortises, none of them in
the drafter: launch latency in a small-batch forward, expert staging for tokens whose routing is not
the prompt's, and a verify whose warm floor is twice a prompt-row pass. Those are the work that
remains before the spec model pays; the port itself, its ops, its artifact and its acceptance
measurement are done.

## Next slice, in the order that unblocks the rest

**P4 landed.** `src/models/nemotron_h/execution/draft.{h,cpp}` implements the draft: a `DraftContext`
(the context ring `[lane][layer][side][head_dim*kv_heads][window+block_size]` in BF16 with a parallel
I32 position plane, appends by 2-D copies, attention reading the whole ring and letting the position
test exclude stale slots) and the forward -- taps -> `fc` -> `context_norm` -> context K/V into the
ring; five layers of norm, `kernel_projection` + `grouped_tap_apply(side 0)`, q/k/v for block and
context, q/k norm + RoPE 1e4, `sliding_causal_attention` (window 1024, sink), `linear(output)`,
`grouped_tap_apply(side 1)`, residual, then the MLP through `linear_pair`/`silu_mul` and its own
finish tap; `final_norm`; then the head (tower log-softmax top-8, five features, feature MLP, node
embedding, the 7x8 lattice with its additive bias, the anchor branch, factor projection, 8.0 *
cosine) and the greedy walk. `DraftObserver` reports every stage under the reference's names.

Stage agreement against `tools/verify/lilicorr_reference.py` on real taps (187 context columns, 7
slots), rel L2: `body.embeddings`, `layer0.residual_in` and `lattice_bias` **bit-exact**; `fc` 2.1e-3,
both conv halves 1.8e-3 to 1.1e-2, attention 4.5e-3 to 6.8e-3, `layer4.out` 9.6e-3, `body.norm_out`
5.6e-3, `head.logits` 3.8e-3, `candidate_logits` 2.4e-3, `anchor_out` 2.9e-3 -- BF16 against the
oracle's FP32 throughout.

Two facts about the walk. It does **not** reproduce the oracle's ids, and the first divergence is the
candidate *set*: the artifact stores the tower's `text/output_head` as FP8 (the recipe's own choice)
while the oracle reads the checkpoint's BF16 `lm_head`, so top-8 sets split near ties and the integer
ids carry that through `node_token`, `factor_input`, `edges` and `score_tensor`; everything up to and
including `candidate_logits` agrees to 3.8e-3. The engine is internally consistent because its draft
and its verifier read the *same* FP8 head, and the C++ walk's slot 0 plus two more slots land on the
target's actually committed tokens -- so the ids are the wrong instrument here and acceptance at the
round is the right one. Second, the head's index arithmetic is still host-side (naive lattice
attention) and the reported 1.53 s first call includes 181 stream drains, so the draft's per-round
cost is unmeasured; a round budget of 3% of the block will need the lattice on device.

Removing the transposes the old row-major op forced changed **181 of 181 stages not at all** while
deleting 30 transposes and 240 two-byte-pitch copies per forward, which is the evidence that the op's
dim0-fastest rewrite is exact.

## Next slice, in the order that unblocks the rest

**P4 has a working template, not just a spec.** `src/models/qwen3_5/execution/draft.cpp` is the
existing DFlash2 drafter driver -- family-owned (it includes `qwen3_5/program/internal.h` and drives
that family's `ExecutionCore`/`DFlashDecodeState`), so it cannot be reused as-is, but it is the
sequence to port: `prepare_dynamic_branch` (`:244`) and `finish_dynamic_branch` (`:257`) are the
conv prepare/finish helpers (the fused `dynamic_grouped_conv` pair; the nemotron draft composes the
same mathematics through `grouped_tap_apply`), `propose_dflash2_batch` (`:269`) is the round driver
(P5's shape), and `propose_batch_impl` (`:396`) is the per-block proposal. The draft's own state and
workspace live beside the tower's, and the nemotron Program already captures the six taps, so the
port is: a draft state struct + a forward that composes the spec's stages + the two branch helpers
rewritten over `grouped_tap_apply` + the proposal loop.

**P5 landed and measured, and the round is correct but far from the reference's number.**
`Program` holds the draft -- lazily, because the artifact carries one while the runtime drives no
drafter, so a lane that never drafts pays nothing of its ~270 MB -- `device_bytes(Model|Config,
options)` carries that term (the ring and one forward's scratch from `DraftContext::plan`, one Mamba
lane image per lane, and the verify block's `[vocabulary, block_size]` logits), and `draft_round` is
the round: propose, snapshot the Mamba lanes, run the target over the block, accept the matching
prefix, roll the state back, and commit `accepted + 1` tokens as a pass of its own. Two details are
load-bearing and were both wrong on the first attempt. The target's verify block is
`[anchor, proposals...]`: the mask rows are the *drafter's* block, and feeding them to the target
makes every row past the anchor predict from a token no sequence contains -- which measured 0.53
accepted drafts and, with the later slots at exactly zero, looked like a weak drafter rather than
what it was. And the Mamba state rolls back through a snapshot plus the committed pass, since neither
the convolution history nor the FP32 recurrent plane can be un-advanced; the attention side needs only
`AttentionState::rewind`, whose discarded slots causality already hides.

Measured on `/mnt/storage/models/nemotron-74b/nemotron_h_74b_nvfp4_draft.ninfer`, 32 rounds on the tap test's own 187-token
prompt: **0.844 accepted drafts per round**, slot 0 **0.563**, per-round wall **139 ms** split
draft **40.7** (29%), target verify **73.9** (53%), committed pass **23.7**, state rollback **0.4**.
The reference implementation on the identical dump and round count gives **3.000** (slot 0 0.688);
this document's 2.122 is its 128-round run. What scopes that gap: the verify block now reproduces the
target's own decode (31 of 32 rounds' bonus identical, and row 0 matched decode exactly), so the
round's accounting is not the problem.

**The artifact's FP8 output head is not the cause**, which was the first guess and is worth recording
because the evidence for it looked strong: on one round the engine's top-8 *set* is identical to the
reference's while its *ranking* is not, and every rank-indexed term reads that order. The direct
counterfactual refutes it. Running the reference's own loop on the same dump for the same 32 rounds,
with the only change being where candidates come from -- the checkpoint's BF16 `lm_head` against the
same table quantized exactly as the artifact stores it (`fp8_e4m3fn_row_bf16`: E4M3 codes, BF16 row
scale = maxabs/448, clamp plus round-to-nearest-even, BF16 activations, FP32 accumulation; 2.65%
relative weight change) -- gives **3.000 accepted drafts and a 2.906 rank-0 chain for the FP8 head
against 3.000 and 2.938 for BF16**. The head's format costs the reference nothing.

What the same run isolates instead is the **candidate ceiling**. The no-lattice baseline (each slot's
rank-0 candidate against the target's stream) is 2.94 for the reference and **1.56 for the engine**,
so even with the lattice removed the engine's candidates carry about half the information the
reference's do.

**A/B attribution, measured.** Dumping the engine's own `body.norm_out`, `body.anchor_hidden` and
candidate ids/log-probabilities for all 32 rounds (`NINFER_DRAFT_STAGE_DUMP`, via `draft_round`'s
observer) and running `ref.CorrelatorHead` on exactly those tensors gives **1.312** accepted drafts
per round, against the engine's 0.844 and the reference's 3.000. Splitting the 2.156 gap: **1.688 is
the inputs** (3.000 -> 1.312: the draft body's hidden, and the conditioning it reads) and **0.468 is
the head's arithmetic** (1.312 -> 0.844 on identical tensors). Both walks land below the rank-0 chain
on the engine's inputs (1.312 and 0.844 against 1.562), so the lattice is not the primary defect --
on our hidden it loses in either implementation. The body and its conditioning are what to fix first;
the head's algebra is a real but secondary 0.47.

**Where the round's time goes, measured -- and one attribution corrected.** The draft's forward is
181 stage callbacks whose host gaps sum to 40.0-40.8 ms of the 40.3 ms, and the largest gap is
26.2 ms immediately before `head.candidate_ids`. That gap is **not** the host candidate selection, as
it first appeared: `head()`'s first `copy_to_host` is also the forward's first synchronize, so the gap
is the body's device work draining. The correction is measured, not argued: making the partition
float-precision and the block-id upload asynchronous changed the round by nothing (draft 40.3 ->
41.4 ms, inside noise), i.e. the host compute was never the 26 ms. The draft is **device-bound** at
~41 ms for a width-8 forward whose weights are 917 MB, and `ops::linear_topk` cannot take the
candidate selection: it is registered for 5120-wide heads and returns sixteen *scores* with no
full-vocabulary partition, so that route needs an `src/ops/**` change rather than a call. The target's verify pass is not the verify path either: the same 8 ids as an
ordinary committed pass cost 82.1 ms, so `HeadColumns::Every` and the missing capture are not the
cost; repeating that identical pass on the same expert set costs 49.0 ms, so ~33 ms is first-touch
expert staging on a proposal-shaped expert set, and one row of the same state costs 19.7 ms (the
verify is not the causal block over the live context). Staged bytes 3.68 GB against 16.37 GB resident
with all 256 cache slots granted. The doc's 25.2 ms pp8 is a prompt-row set; a verify block of
proposed ids is a different working set.

**The conditioning defect is confirmed, and it is the dominant one.** Advancing the target one row at
a time after each round -- so the draft reads taps from width-1 passes, the numerics a plain decode
produces (`NINFER_DRAFT_W1_COMMIT`, a measurement arm, never the shipping path) -- moves the same 32
rounds to **1.3125 accepted drafts per round** from 0.8438, slot 0 from **0.5625 to 0.7188** (the
reference's own slot 0 is 0.688), the rank-0 chain from 1.5625 to **2.0625**, and the per-slot curve
to 0.719 / 0.313 / 0.125 / 0.094 / 0.063 / 0 / 0. Slot 0 landing on the reference's number says the
drafter's own arithmetic is right and that pass width was destroying everything downstream of it. What
it costs: the committed pass goes from 23.6 to 42.2 ms (one pass per token instead of one per round)
and the round from 138 to 161 ms, so the decision to take is how to obtain decode-numerics taps
without a width-1 pass per token. The remaining 1.31 -> 3.00 is now a separate question: with width-1
taps and the oracle's head arithmetic on the engine's BF16 hidden, which is the next dump to take.



Every item below carries its own oracle; the acceptance bar for the finished port is this
document's own corrected reference, which on this engine's converted target accepts **2.122 drafts
per round** (slot 0: 70.7%) at draft length 7, and reproduces llama.cpp's walk ids exactly when the
conditioning comes from a same-run capture.

1. **The draft's ops already exist -- this is a geometry, not a kernel.** `draft` execution is
   qwen3_5-owned (`src/models/qwen3_5/load/dflash{,2}.cpp`, `execution/draft.cpp`,
   `program/speculative/`, `program/storage/draft_context.*`), but the *ops* a DFlash-style drafter
   needs are shared and already implement the conv-fused path: `ops::attn_input_proj`'s
   `q8_dflash2` route (`src/ops/attn_input_proj/q8/q8_dflash2_attn_input.cu`), `ops::linear_swiglu`'s
   `q8_dflash2` route (`src/ops/linear_swiglu/q8/q8_dflash2_linear_swiglu.cu`), and
   `ops::common/dflash_rope.cuh`. Their registries are keyed by shape -- gemma4's drafter already has
   its own entry (`ops/linear/bf16/shapes/gemma4_draft.cu`) -- so this draft's 4096/256 geometry is a
   registry entry plus an FP64-oracle case per op, not a new convolution. That supersedes the earlier
   "compose the conv from a generic linear" decision above: the fused drafter path is the one to
   extend, and the tap/coefficient arithmetic in those ops is the same mathematics the reference's
   `grouped_dynamic_conv` validates.
2. **Attention sinks** for the draft's own attention. One extra per-query-head term folded into the
     online Softmax's running max and denominator, added *unscaled* after the logit scale (ggml's implementation, read directly:
   `max = MAX(max, sink[h])`, `sum += expf(sink[h] - max)`), with no value contribution. That is the
   whole semantic; `sliding_causal_attention` already admits the draft's geometry (any `Hq`/`Hkv` at
   head width 128) and its symmetric window through `position_q_high`.
3. **The linear routes** the draft's shapes need: bf16 `(4096,4096)`, `(256,4096)`, `(3072,1024)`,
   `(2048,1024)`, `(1024,1024)`, `(1024,3072)`, `(1024,5)` and NVFP4 `(4096,24576)`, `(12288,4096)`,
   `(4096,12288)`, `(1024,4096)`, `(1024,1024)`, `(2048,1024)`, `(1024,2048)`. The draft's rows are 8
   per round (56 in the lattice), so the honest shape is one runtime-`{n,k}` route per format
   admitted for a bounded row count rather than fourteen hand-written entries.

Then: the draft's config parse and binding (the artifact carries the checkpoint, `lilicorr/...` names
per the namespace in this document), the converter recipe, the body and the head in C++, and finally
the Program's draft round with its verify and publication. The reference's stage trace is the
per-stage oracle for the body and head, exactly as `nemotron_h_reference.py` is for the target.

## Sequence

| Step | Observable |
|---|---|
| 1. Target taps | a pass that captures the six residuals; a capture-free pass is bit-identical to today's |
| 2. Acceptance probe | an accepted length on the real target, or the reason it is low |
| 3. Ops | each new/changed op against its oracle at the real geometry |
| 4. Draft config, binding, converter | the artifact carries the draft; the loader binds every parameter |
| 5. Draft round in the Program | `ninfer --draft-k K` generates, greedily identical to the target's own output |
| 6. Qualification | acceptance length and tokens/s measured, against the card's 4.30 |

## Open questions

- **Feature channel order** in the head's 5-vector is not stated by the training-side sources; vLLM
  and llama.cpp agree, and that agreement is what the oracle implements.
- **LayerNorm epsilon** in `feature_norm` (vLLM's default 1e-5 versus the body's 1e-6) is not
  documented; take vLLM's effective default and note it.
- `lilicorr_pos` (the anchor's ring slot) is written but never read in llama.cpp's implementation — a
  stale slot would silently feed a wrong context vector to `context_proj`. Our anchor comes from the
  last committed position directly, so the hazard does not transfer.
- The paper reports its acceptance gain as 7–19%, 9–19% and 12–19% in three places; the checkpoint's
  own SPEED-Bench table (4.30 average) is the number to match.
