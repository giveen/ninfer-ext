# Nemotron Bolt 74B A9B

Active work: adding `NemotronHPuzzleForCausalLM` to NInfer. The target is NVIDIA's
[Nemotron-3.5-Bolt-74B-A9B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16), a
hybrid Mamba-2 / LatentMoE / attention model with a vision tower, served here as text only in the
first phase.

Scope agreed at the start: **text-only target first**, converted from the **BF16** checkpoint through
this repository's own quantization routes, with the
[LiLiCorr](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr) draft
([paper](https://arxiv.org/abs/2608.20530)) taken up after the target runs.

## The three references

| Source | Use |
|---|---|
| `transformers` 5.14.1, `models/nemotron_h/modeling_nemotron_h.py` (1259 lines) | the canonical mathematics, line numbers cited below. The checkpoint's own `modeling_nemotron_h_puzzle.py` is 37 lines and subclasses it. |
| The checkpoint's `configuration_nemotron_h_puzzle.py` (151 lines) | how per-layer values are threaded: `get_nemotron_h_config_for_layer` builds one `NemotronHConfig` per block. |
| llama.cpp `cd0a5b6` | a second implementation: `src/models/nemotron-h.cpp` (352 lines) and `conversion/nemotron.py` (815 lines, including `LiLiCorrDraftModel(DFlashModel)`). It reads the **per-layer expert count from the router tensor's shape**, and its `d_in_proj` matches the reference exactly. |

## The target

88 blocks: **40 mamba, 40 moe, 8 attention**, interleaved (the first eight are
`mamba moe mamba moe mamba moe mamba attention`). Constants, all read from the checkpoint config:

| Field | Value | Note |
|---|---|---|
| `hidden_size` | 4096 | |
| `vocab_size` | 131072 | untied head; the draft shares it |
| attention | 32 heads, 2 KV heads, `head_dim` 128, no bias | plain GQA with **no positional encoding at all** — not RoPE, in any variant; see the note under the reuse survey |
| mamba | 128 heads × head_dim 64 = 8192 inner, `conv_kernel` 4, `n_groups` 8, `ssm_state_size` 128, `chunk_size` 128 | `conv_dim = 8192 + 2·8·128 = 10240`, `in_proj` out = `2·8192 + 10240 + 128 = 18560` |
| `use_conv_bias` | **true** | the depthwise conv has a bias even though `use_bias` is false |
| `time_step_min` / `_max` | 0.001 / 0.1 | **only the minimum clamps**; the reference's maximum clamp is commented out and `time_step_limit` is unused on the torch path |
| MoE | top-9, `moe_latent_size` 1024, 1 shared expert (5376), `norm_topk_prob`, `routed_scaling_factor` 5.0 | **each layer has its own expert bank**: 240 experts in 12 layers, 304 in 6, 368 in 9, 432 in 10, 496 in 3. The config's `num_experts`/`num_local_experts` of 512 is a bound, not a shared bank — llama.cpp reads the real count from each layer's router width and validates it against 512. Per-layer `moe_intermediate_size` is 1536/2048/2304/2560. |
| router grouping | `n_group = 1`, `topk_group = 1` | degenerate: the group mask selects everything, so top-k runs over the layer's whole expert bank. **`n_groups = 8` is Mamba's B/C grouping and has nothing to do with this.** |
| `hidden_act` / `mlp_hidden_act` | `relu2` | squared ReLU, for experts and the shared expert |
| norms | `layer_norm_epsilon` 1e-5 | |
| `rescale_prenorm_residual` | true | **initialisation only** (`_init_weights`, the GPT-2/Megatron 1/√N scheme). The checkpoint carries it; there is no runtime residual rule to implement. |

## The checkpoint's tensors

Verified against the released BF16 checkpoint (63 shards, 149.8 GB, 28,725 tensors), which is also where
the loader's names will come from:

- `language_model.backbone.layers.N.mixer.in_proj.weight` is `[18560, 4096]` — the projection width the
  config derives, confirmed rather than inferred;
- `mixer.conv1d.weight` is `[10240, 1, 4]` (depthwise, kernel 4) with `mixer.conv1d.bias` `[10240]`;
- `mixer.A_log`, `mixer.D` and `mixer.dt_bias` are `[128]` per layer, `mixer.norm.weight` is `[8192]`;
- **the experts are one tensor per expert per matrix** — `mixer.experts.E.{up,down}_proj.weight` — 13,824
  experts and 27,648 tensors in total, `[1024, 2560]` for a 496-expert layer's down projection;
- `mixer.gate.weight` and `mixer.gate.e_score_correction_bias` are per layer, the bias **F32** and sized
  to that layer's bank;
- `language_model.backbone.embeddings.weight` and `language_model.lm_head.weight`, plus (phase 2)
  `vision_model.*` with a three-tensor `mlp1` projector.

The config's block list and its per-layer expert counts match the tensors exactly: 40 mamba, 40 moe and 8
attention in the same order, with 240/304/368/432/496 experts in 12/6/9/10/3 layers.

## The draft (LiLiCorr)

The draft is a separate 962 MB NVFP4 file (`model.safetensors` beside the target), and its tensors say
what it is: a Qwen3-shaped five-layer block drafter (`block_size` 8) with an elaborate head.

- **What it needs from the target.** `fc.weight` is packed two values per byte, so its `[4096, 12288]` is
  really `4096 x 24576`, which its `weight_scale` of `[4096, 1536]` at group 16 confirms — that is
  **6 x 4096**, i.e. all six `target_layer_ids` (1, 18, 35, 51, 68, 85) concatenated. The target has to
  expose those six hidden states for the draft to exist at all, so that is a target-side decision even
  though the draft itself is phase 5.
- Its own attention is GQA with **per-head QK norms** (`q_norm`/`k_norm` `[128]`), **attention sinks**
  (`attention_sink_bias [32]`), a 1024 sliding window, and `causal: false` — bidirectional inside the
  block, which is what block drafting needs.
- A **grouped dynamic convolution** wraps both sublayers: `{attention,mlp}_conv.base_kernel [2,2,4096]`
  (2 taps x 2 sides) modulated by `kernel_projection [1024,4096]`, at group width 16. That is exactly
  `dynamic_grouped_conv`'s shape, so phase 5 reuses that op rather than porting it.
- The LiLiCorr head: `token_proj`, `context_proj` and `pass_hidden_proj` (4096 -> 1024), a five-feature MLP
  (`feature_norm [5]`, `up_proj [1024,5]`), a two-layer pre-norm transformer over slot x rank tokens
  (`slot_embedding [1,1,7,1,1024]` for block 8, `rank_embedding [1,1,1,8,1024]` for the top-8),
  `in_head`/`out_head` for the vectors the paper matches, `anchor_out_head`, and the pairwise terms
  `relative_slot_bias [8,15]` and `same_slot_bias [8]`.
- It has no embedding and no LM head: it shares the target's tokenizer and vocabulary, which is why every
  tensor here is a projection, a norm or a bias.

## The mathematics, with citations

**Block** (`modeling_nemotron_h.py:922`). Pre-norm RMSNorm → mixer → `residual + mixer_out`. No
scaling, no other state.

**Mamba-2 mixer** (`:115` construction, `:396` torch path):

1. `in_proj` → split `[z (8192), xBC (10240), dt (128)]`.
2. Depthwise `conv1d`, kernel 4, padding 3, **with bias**, output cropped to the sequence length.
3. **SiLU is applied to the whole conv output**, so B and C are gated along with x. This is not a
   detail to "clean up": the split into `x, B, C` happens *after* the activation.
4. `dt = softplus(dt + dt_bias)`, then `clamp(min = 0.001)`. `A = -exp(A_log)`, per head.
5. `dA = exp(dt·A)`, `dB = dt·B`, `dBx = dB·x`; B and C are `repeat_interleave`d from 8 groups to 128
   heads; the scan is chunked at 128 with a cumulative sum of A over each chunk.
6. `y = state·C + D·x`, then the **gated grouped RMSNorm** (`Zamba2RMSNormGated`,
   `zamba2/modeling_zamba2.py:52`): multiply by `silu(gate)` *first*, then RMS-normalise within groups
   of `8192/8 = 1024`, then scale by a per-element weight. Order matters.
7. `out_proj` back to 4096. The recurrent state is **fp32** (`mamba_ssm_cache_dtype`), as is the
   reference's own state arithmetic.

**MoE** (`:704`, experts `:643`, router `:748`):

1. Router in **fp32**: `sigmoid(x·W)`, then `+ e_score_correction_bias` (a learned buffer, not zeros
   in general — it must be loaded), group machinery (degenerate, see above), top-9.
2. Weights are the **uncorrected** sigmoid scores gathered at those indices, normalised by their sum
   (plus 1e-20), then multiplied by 5.0.
3. **Latent round trip**: `fc1_latent_proj` 4096 → 1024, experts run in that space, `fc2_latent_proj`
   1024 → 4096.
4. **Experts are not gated**: `down(relu2(up(x)))` — two matrices per expert, explicitly called out as
   a NemotronH difference from Mixtral/DeepSeek.
5. The **shared expert is computed on the pre-latent residual** and added after the round trip.

**Attention** (`:860`). Plain GQA, causal, no bias, no sinks, no window — and **no positional encoding**, which is the one thing the reference's own code invites you to get wrong: `NemotronHAttention` is decorated with `@use_kernelized_func(apply_rotary_pos_emb)` at `:859` and **never calls it**, and its `forward` takes no `position_embeddings`. Four things agree that the rotation is absent rather than merely unwired: the checkpoint's own `modeling_nemotron_h.py` mentions rotary nowhere, llama.cpp's `nemotron-h.cpp` has no rope at all, `NemotronHConfig` has no `rope_theta`, and the dump settles it numerically — an FP64 attend-only oracle over the dumped q/k/v reproduces the dumped `o_proj` and `mixer.0` to 1.4e-7, where any rotation gives about 6e-3. So the composition calls no rope Op, and there is nothing to port there.

## What this costs

Less is new than that first estimate suggested, and one piece costs nothing:

| Piece | Verdict |
|---|---|
| Recurrent state | **Reused unchanged.** `LinearAttentionStatePoolSpec` is parametric in exactly what Mamba-2 needs — heads, head_dim, state size, conv channels and conv width — and the state is O(1) in context. 4 MiB per layer, so 160 MiB per sequence, plus 2.34 MiB of convolution history: 162.34 MiB per sequence, 2.67x the documented 35B-A3B image. |
| Depthwise convolution | **Done.** `causal_conv1d_silu` was already depthwise, kernel-4, causal, SiLU-fused, T-any, with per-slot state, and C=10240 is a production width. It lacked only the **bias**, which is now a `const Tensor* bias` argument on every entry (null = the bias-free op). It could not be added from outside the op: the SiLU is fused, and `silu(conv)+bias` is not `silu(conv+bias)`. (`dynamic_grouped_conv` is not this op — it is a round-level, channel-mixing, two-tap dynamic convolution, which is closer to the draft model's grouped convolution than to Mamba's.) |
| Chunked scan | The SSD skeleton exists **twice**, in GDN and KDA, with fp32 state. Mamba-2 sits between them: KDA supplies the `A_log`/`dt_bias`/gate production and the chunk packet, GDN the *scalar* per-head decay cumsum (Mamba-2's A is per-head scalar), and the 8-groups-to-128-heads B/C mapping is the existing `qk_heads < value_heads` mechanism. Genuinely new: a distinct value extent (64 rather than the hardcoded 128), no delta rule, softplus dt, an unscaled readout and the `D*x` skip. |
| Dense projections | **Every projection needs a new route, in every format the family uses.** A linear here is a hand-written kernel *per shape*: `bf16`, `fp8`, `nvfp4` and `q4..q8` each dispatch through a finite registry and throw `unsupported shape` on anything else, and no format has a generic fallback (the SIMT kernels are per-schedule variants of registered shapes). Nemotron's projections are all `k = 4096`, or `k = 1024` for the latent pair, and no registered pair carries either — so **not one projection has a route today**, in BF16 or in the Q8/Q6 the artifact will use. The composition therefore lands with a generic BF16 dense route (runtime n and k, predicated row tail) as the correctness path, which is also what makes the tiny configuration runnable at all; the family's own hand-written shapes follow as the measured performance step, and its shape list is the input to that work. Worth contrasting with what *is* parametric: the expert banks are 66 of the model's 74.9 billion parameters and need no registry entry at all, because the expert op takes `{E, I, K, H}` at runtime. |
| Gated grouped norm | **Wholly new.** The gated RMSNorm here normalises the whole row and applies the gate after; this one gates first and normalises within groups. |
| Everything else | `relu2`, the non-gated expert bank with its latent round trip, the sigmoid router, and a config whose block types and expert shapes are per layer. |

Memory: NVIDIA's own 24GB variant is the only size evidence there is, and it is aggressive — see
[the serving footprint](#serving-footprint-and-what-it-implies) below, which records their per-tensor
recipe, why this repository's routes land above a 5090's 32 GB, and why that makes host offload part of
the serving configuration rather than an option.

## Serving footprint, and what it implies

NVIDIA's own 24GB variant is the only evidence about what this model costs on a small card, and it is
informative: **27,728 expert tensors at `IQ2_XXS`** (2.0625 bits, group 256, ggml packing), the mixer
projections at NVFP4 group 16, the head at Q8_0 — 24.12 GB in total. So 74B fits 32 GB only at roughly
2.5 bits per parameter on average.

This repository's routes are larger. EXL3 at 3.5 bpw puts the expert banks alone near 25 GB, and with the
dense, attention, embedding and projection parts the artifact lands around 33 GB — over a 5090's 32 GB.
Per the decision taken when this was raised: **that is the intended configuration, not a compromise.**
This model is served the way Flash-Next is — routed experts in pinned Host memory behind a Program-owned
device cache — using this repository's own quantization routes rather than NVIDIA's 2-bit recipe, which
is a note about what they accepted rather than a target to match. No low-bit expert codec is needed.

Phase 1 targets this repository's **NVFP4** route rather than EXL3, for a reason that is not about size:
the LiLiCorr draft's own checkpoint is W4A16 NVFP4, so an NVFP4 target is the precision family the draft
was trained against. Pairing it with an EXL3 target is exactly the mismatch that made the Gemma 4
drafter accept almost nothing while every mechanism still tested clean. EXL3 becomes a second route
afterwards, and the BF16 checkpoint is the conversion source for both.

Concretely, at NVFP4 this is the Flash-Next recipe applied to this model, and the sizes are worth stating
before anyone plans a machine. The routed experts are **66.14 B of the checkpoint's 74.9 B parameters**, so
NVFP4 at 4.5 bits puts the expert banks alone at **37.2 GB**; EXL3 at 3.5 bits gives 28.9 GB and NVIDIA's
`IQ2_XXS` 17.1 GB. With the dense side at Q8 the artifact lands near **46 GB** (EXL3: 38 GB). This corrects
the earlier estimate in this document of 25 GB of experts and a 33 GB artifact: that was an estimate, and
these are computed from the checkpoint's own `block_configs`.

Which makes the host side load-bearing rather than incidental. Flash-Next's mechanism pins the expert set in
Host memory behind a device cache, and **37.2 GB does not fit this machine's 32 GiB** — the pinned route is
unavailable here at NVFP4, where at EXL3's 28.9 GB it was already marginal. Two things follow. The artifact
is identical either way, because this is a property of the serving host and not of the format; and the route
needs either NVMe-resident experts streamed into the device cache instead of pinned — about 24 MB per token
(9 experts x up+down x 1024 latent x ~2300 intermediate x 0.5625 B), so ~1.2 GB/s at 50 tok/s, which makes
NVMe a requirement rather than a preference — or a host with at least 64 GB. That is a serving decision and
it does not change the converter.

Two properties the recipe inherits from `qwen3_8_flash_next_nvfp4`: each layer's bank becomes one parent per
matrix, so an expert is a fixed row range addressed as contiguous spans; and a full-precision source carries
no calibrated activation divisor, so this conversion is **W4A16**, the same precision family as the LiLiCorr
draft. One more thing the checkpoint's `block_configs` show: the per-layer geometry is more heterogeneous
than a count list suggests — **twelve** distinct (experts, intermediate width) pairs, not five — so the
offload sites key on the pair, and the expert op taking `{E, I}` at runtime is load-bearing rather than
defensive.

Sized in bytes, the split above is two independent budgets:

| Side | Holds | Size |
|---|---|---|
| **Host RAM** | the routed expert banks, pinned | **37.2 GB** at NVFP4 — so at least 64 GB of RAM, and a 32 GiB box cannot serve this route at all |
| **VRAM** | everything else | dense side 8.8 B parameters at Q8/Q6 (~9 GB); Mamba-2 recurrent state 4 MiB per layer plus 2.34 MiB of convolution history per sequence (162 MiB across the 40 layers); attention KV at 8 KB per token across the 8 attention layers (2.1 GB at the full 262144-token context in BF16, ~0.6 GB at NVFP4) |
| **VRAM, remainder** | the device expert cache | roughly 20 GB on a 32 GB card, i.e. about half the bank resident, with the rest streamed from the pinned copy |

The 64 GB host requirement is not a preference: the pinned expert set is not optional in this design, and 37.2 GB
of it does not fit 32 GiB. Everything the device holds besides the cache is small — 9 GB of dense weights plus
memory that scales with context, not with the model — which is why the expert cache is where the remaining VRAM
goes and why its size is a Program decision rather than a fixed number.

A correction from the draft's own tensors, because this argument was overstated: the draft's stated base is
the *24GB* variant, whose mixer projections are NVFP4 but whose 27,728 expert tensors are `IQ2_XXS` at
2.06 bits. The target it was qualified against therefore has **2-bit experts**, which our NVFP4 experts at
4.5 bits do not match. Higher precision should not be worse, but the pairing is unproven in both
directions, so the rank diagnostic decides it rather than an assumption. Worth watching in the same
breath: an elastic (nested) Bolt checkpoint would be device-resident on a 5090 and would remove the
host-offload requirement entirely, and the converter would take it unchanged -- the architecture is the
same.

The host-offload cache assumed a uniform bank: `slot_of` was `layers × E`, the owner→layer decode was a
division by `E`, the staged bank was `2E`, and the pool stride was one global slot size. That is now
**done.** `ExpertPoolGeometry`, built from the pool's per-layer bank shapes, derives all of it: a prefix-sum
table places the expert space, so a layer's experts stay tight in the pool instead of being padded to the
widest bank, and one stride sizes the cache slots at the widest bank's per-expert count — a cost the device
cache pays per *slot* and the pinned host banks do not, since those are the artifact's own tensors. The
uniform case is the table's equal-entry case, so Flash-Next runs the same code rather than a second path,
and it is verified end to end rather than by argument: both pool tests drive resolve, fetch, stage and
reclaim through the new geometry and pass, the EXL3 one covering the zero-tail branch, and the arithmetic has
its own test, itself checked against a library mutated to fail it three separate ways. Two things remain
unverified: a *heterogeneous* staged fill end to end, which no pool can exercise until this family's loader
and Program exist, and the Engine-level Flash-Next path, which needs its ~70 GB artifact. No artifact-format
change.

## The official recipe, and what it changes

NVIDIA publishes a ModelOpt recipe for exactly this architecture, and it is the authority to follow rather
than to approximate: `modelopt_recipes/models/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16/ptq/nvfp4-mse.yaml`
(the PTQ stage `nvfp4-mse`, updated 2026-10-15 for the Super VL checkpoint), with its
`examples/megatron_bridge/tutorials/NemotronH-Omni-W4A4-QAD` walkthrough and the released
`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`. The recipe states that its rules are not specific to
that release and "apply to NemotronH hybrids generally", which is what makes it ours. Its assignment is
ordered with the last match winning, so everything not named below is left at BF16:

| Component | Official format |
|---|---|
| MoE **routed and shared** experts | NVFP4 **W4A4**, block 16, MSE-selected static weight scales, dynamic activations |
| attention `q`/`k`/`v`/`o` (Megatron `linear_qkv`/`linear_proj`) | **FP8 W8A8** per-tensor |
| Mamba `mixer.in_proj` / `mixer.out_proj` | **FP8 W8A8** per-tensor |
| `lm_head` (Megatron `output_layer`) | **FP8 W8A8** |
| KV cache | **FP8** at a fixed range for the backbone, so no `k_scale`/`v_scale` is exported |
| latent MoE, routers, embeddings, norms, vision tower | **BF16** |

Three corrections to the route this document planned. The dense projections and the head are **FP8**, not
Q8/Q6 — this repository already has a W8A8 FP8 linear and an FP8 KV, so that is a recipe change rather than
new work. The **latent projections, the router and the token embedding stay BF16** rather than quantized,
which matters here because the router's mathematics is a place this document already refuses to approximate.
And the experts' *weights* use MSE-selected static scales while their *inputs* are dynamically quantized, so
the recipe's expert route is W4A4 — reached with a calibrated `input_quantizer` that a full-precision source
does not carry, which is why a conversion from the BF16 checkpoint gives W4A16.

The **released** artifacts do not all follow the recipe, though, and the difference is the useful part. Their
own `hf_quant_config.json` descriptors say: `…-Super-VL-120B-A12B-NVFP4` (the QAD one) reports plain `NVFP4`
for 41,043 tensors, i.e. W4A4 as the recipe specifies, while `…-Lightning-30B-A3B-NVFP4` reports
**`W4A16_NVFP4`** for 5,935 tensors — weight-only. Both are `MIXED_PRECISION` with an `FP8` KV and FP8
projections. So W4A16 is a **published** configuration rather than a fallback we invented, and it is precisely
what a full-precision source produces: our conversion from the BF16 checkpoint reproduces the Lightning
release's precision assignment, not a degraded one. W4A4 then becomes available the same way it was for them —
from an encoded source, or with an activation-calibration path this converter does not have — and their QAD
stage is training rather than conversion, so it is out of scope here.

That descriptor is also the converter's input format, and it is now known rather than guessed:
`hf_quant_config.json` with `quant_algo: MIXED_PRECISION`, `kv_cache_quant_algo: FP8`, and a
`quantized_layers` map naming every tensor as `backbone.layers.N.mixer.{in_proj,out_proj}`,
`…mixer.experts.E.{up,down}_proj` or `mixer.shared_experts.*` with its own `quant_algo` and `group_size: 16` —
the same per-tensor shape our `import_encoded` path already consumes for Flash-Next.

On disk that comes to three tensors per expert matrix, which is exactly what our loader has to consume:
`weight` (the packed E2M1 codes), `weight_scale` (the per-16-block scale) and `weight_scale_2` (the
per-tensor second level). An FP8 projection carries `weight`, `weight_scale` and — the decisive difference —
**`input_scale`**, the calibrated *activation* divisor. That is the whole of the W4A16/W4A4 distinction at the
file level: a W4A16 release's experts carry no `input_scale`, so there is nothing to quantize their
activations with, while a W4A4 one does. It is also why a conversion from the BF16 checkpoint lands on W4A16
whether or not we choose it.

The checkpoint inventory decides how we get there. Our own Bolt 74B has **no official NVFP4 release** — what
exists is `…-Bolt-74B-A9B-24GB` and `…-24GB-LiLiCorr`, the GGUF-derived 24 GB quant whose experts are 2-bit,
which is precisely the LiLiCorr draft's base and confirms it. So the NVFP4 conversion for this checkpoint is
ours to make, from `…-Bolt-74B-A9B-BF16`, using the recipe above.

But `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` does exist: the **same architecture at 30B**, an
encoded ModelOpt checkpoint with calibrated divisors, around 18 GB. That fits this machine, where the 74B's
~48 GB artifact and 37 GB of pinned experts do not, and it exercises the whole family path end to end —
conversion through `import_encoded`, the Engine, the decode loop, and W4A4 rather than W4A16 — which the real
target cannot do until it has a 64 GB host. Verification on the sibling is how the family gets tested before
the target is affordable.

## What the layout work actually is

The artifact container is layer-blind and already carries the variation: a bank is one binding of parts
with its own shape and divisors, validation is per requested parameter, and materialization has no layer
stride. Gemma 4's global layers already bind *no* `attention/value` and use different query/key row
counts, so per-layer shape and per-layer parameter *presence* are both proven. Names are per layer
already (`text/layers/<i>/moe/experts/<e>/…`), and the converter's recipe, packing groups and per-layer
layout file need no change at all.

What is uniform is the family parser (one `MoeConfig` per model), `bind_block` deciding dense-vs-moe once
for the whole model, and the expert-consuming ops. So: the config gains a per-layer FFN descriptor
(`layer_types` is the existing pattern), the loader takes it mechanically, and the MoE op is written
**new** rather than generalised — Bolt's experts are non-gated with a latent round trip and `relu2`, and
both existing offload and sparse-MoE ops bake their geometry into headers and every kernel translation
unit. The new op takes `{E, I, K, H}` at runtime; the artifact side needs nothing, because a bank is
already opaque.

One loader question is settled too: stacking a bank needs nothing new, because
`prepare_linear_weight(std::span<const WeightInput> rows)` already concatenates a bank's per-expert row
parts and propagates its divisors — it is what Gemma 4's per-expert parts use. So a bank needs one
contiguous parent per matrix in the artifact, which is the converter's packing group, and the per-layer
variation is carried entirely by the config.

## Milestones

1. **Oracle — done.** `tools/verify/nemotron_h_reference.py` dumps every stage an op has to reproduce
   (29 stages on the tiny config), from a scaled-down config or a downloaded checkpoint.
2. **Config and artifact — config done, the rest next.** The per-layer block descriptor is parsed and
   tested; what remains here is the loader threading and the converter's per-layer loops. The artifact
   format does not change.
3. **Ops, one at a time, each against the oracle**: `relu2` + RMSNorm → Mamba-2 (selective scan, fp32
   state) → router + non-gated experts → the latent round trip → the new latent-MoE op → attention →
   the block → the 88-block forward.
4. **Converter and Engine path**, then the host-offload cache work above, then the decode loop.
5. **Draft** (LiLiCorr) after the target runs, qualified against an NVFP4-family artifact.

## Open items

- The draft pairs with the **24GB NVFP4** target, not the BF16 one, so its qualification belongs on our
  NVFP4 route; the cheap decisive check is ranking the draft's token inside the target's own next-token
  distribution, the diagnostic that exposed the Gemma 4 QAT mismatch.
- The host-offload cache pins the expert set, so the maintenance machine's 32 GiB of RAM is itself a
  constraint: the pinned experts at EXL3 3.5 are roughly 25 GB, which leaves little for the host KV
  arena, the state images and the OS. Flash-Next's guide asks for about 65 GiB for a larger model; this
  one sits under that only because the model is smaller. Reducing the pinned set, or a smaller expert
  route, is the escape if it does not fit — neither is needed to start.
- **The state's key/value index is ambiguous, and both extents are 128 today**, so a swap would be
  silent: `LinearAttentionStatePool` plans `ne[0] = key_head_dim` while the GDN and KDA kernels index
  the state as `value * kStateDim + key`. Mamba-2's 128 (state size) against 64 (head_dim) makes it a
  real decision, so a test has to pin the layout rather than a comment.
- **The readout scale is locked** in the linear-attention ops: `scale = 1/sqrt(128)` is required to
  within 1e-6. Mamba-2's readout is unscaled, so that lock has to be lifted for the new instantiation.
- **`D*x` has no home yet.** It is added before the gated norm, and no op in this tree applies a
  per-head scalar over `ne[1]`, so it belongs in the scan epilogue rather than being composed — and it
  cannot be left to the norm to absorb.
- **The scan's chunk size need not be the config's 128.** Chunked SSD is chunk-size invariant in exact
  arithmetic, so keeping the existing 16 is a tolerance decision against the reference dump, not a
  correctness one.
- Vision (RADIO/InternVL tower, projector 20480, video) and MTP (`num_nextn_predict_layers` is 0 in this
  release) are out of phase 1.

## Progress

Each of these is landed, and each is verified by its own test running under the build lock, qualified
against an FP64 oracle plus — where one exists — the reference dump or the reference module itself:

| Op | Evidence |
|---|---|
| Gated grouped RMSNorm | the FP64 oracle matches the reference module to 6.8e-8; a whole-row implementation lands 160x outside the criterion on data whose group means differ, so the test discriminates |
| Conv bias | the bias-free path is bit-identical (the biased entry with a null or all-zero bias must match byte for byte); `SiLU(conv+bias)` is separated from `SiLU(conv)+bias` by about 240x the criterion |
| Mamba-2 selective scan | sequential in registers rather than chunked, so there is no chunk-boundary tolerance question; 0.54 us/token at T=4096, about 88 ms across the 40 Mamba layers of a 4096-token prefill. A swapped state extent is rejected at 15492x, an unscaled readout at 3299x, and a BF16 state image fails the fp32 criterion by 12x |
| MoE router | the correction bias selects experts while the emitted weights stay uncorrected — 130 of 130 tokens change their expert set with it — and the group mask can exclude the bank's largest score outright |
| Latent expert dispatch and `relu2` | non-gated `down(relu2(up(x)))` in the 1024-wide latent space, with `E` and `I` as runtime arguments because the banks differ per layer; fp32 accumulation per expert and per token in rank order, one BF16 rounding, and a single shared definition of the activation so the two ops cannot drift |

The config parse is verified against the released checkpoint's own tensors, and the reference dump carries
the operands an op consumes as well as its outputs, which every remaining op needs.

Remaining: the non-gated experts and the latent round trip (a new latent-MoE op), attention (largely
reuse), the block and the 88-block forward, the converter, the host-offload cache work, and then the draft.

Two caveats of the tiny oracle belong beside its results, because both are adaptors rather than production
behaviour. The scan's state axis is zero-padded from 8 to 128 and `sliding_causal_attention` runs at
`head_dim` 32 with the model's own `scale` passed in — both because the landed Ops encode production
geometry, both inert for a stated reason rather than by luck (`mamba2_scan` takes no scale argument at all,
and the attention Op takes `scale` as an argument rather than baking `1/sqrt(head_dim)`). The real model is
inside both domains: state 128 and head 128. The projection routes the family needs, in BF16 today and in
Q8/Q6 for the artifact route, are in_proj (18560, 4096), out_proj (4096, 8192), q and o (4096, 4096), k and
v (256, 4096), the router (up to 496, 4096), fc1 (1024, 4096), fc2 (4096, 1024), the shared expert's up and
down (5376, 4096) and (4096, 5376), and the head (131072, 4096) — none of which the registry carries, since
no registered pair has k = 4096 or k = 1024.
