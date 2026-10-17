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
| llama.cpp | a second implementation: `src/models/nemotron-h.cpp`, `conversion/nemotron.py` (including `LiLiCorrDraftModel(DFlashModel)`) and `tools/mtmd/models/nemotron-omni.cpp`. It reads the **per-layer expert count from the router tensor's shape**, and its `d_in_proj` matches the reference exactly. PR [#30520](https://github.com/ggml-org/llama.cpp/pull/30520) adds `NemotronH_Omni_Reasoning_V3` — this checkpoint's own architecture string — for Super 3.5 VL and Bolt 74B A9B, and was fast-tracked into their v0.7.0 release, so master is the authority to read. |

What that PR settles, and what it hands us, checked against this document's own decisions:

- **The per-layer bank is the confirmed reading, and reading it globally is a real bug.** The PR exists
  largely to stop using the config's `n_expert` (512, a bound) for the expert tensors: it now takes
  `n_expert_layer` from the router's own shape, validates `n_expert_used <= n_expert_layer <= n_expert`
  and throws `Invalid per-layer Nemotron expert count` otherwise, and passes that width to the MoE op. Its
  tensor shapes are `{latent, expert_width, experts}` for `up` and `{expert_width, latent, experts}` for
  `down`, which is our `[experts*intermediate, latent]` and `[experts*latent, intermediate]` expert-major
  banks with the axes reversed for ggml's order.
- **The MoE is non-gated relu², with a sigmoid router, a correction bias and normalised top-k.** Their
  MoE op is called with no gate and `LLM_FFN_RELU_SQR`; the router is
  `LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID` with `ffn_exp_probs_b` (the correction bias) and
  `expert_weights_norm`/`expert_weights_scale` carrying `norm_topk_prob`/`routed_scaling_factor`. That is
  every one of this family's own choices, arrived at independently.
- **`vision_model.summary_idxs` is dropped deliberately**, with the comment that it is only used by
  `RadioModel.summary` while the projector consumes `RadioModel.features`. This is what the 394 unplaced
  source tensors are: the RADIO tower, that buffer, and the projector — and dropping that buffer is the
  same choice we made, now with a reason rather than an omission.
- **An NVFP4 route for these banks exists there**, with optional per-expert `scale` and `input_scale`
  tensors (`{n_expert}`, and `{1}` for the shared expert). Two consequences: our per-bank divisor is
  justified by measurement (at most 0.0001 rel L2 against per-expert divisors), so the difference is not a
  gap; and their `input_scale` is the shape of the lead for this document's prefill item — an A4 route needs
  a calibrated activation scale, which a full-precision source does not carry. Calibrating one is the way to
  get the experts off the decode-only prefill path.
- **The LiLiCorr draft is already modelled there** (`lilicorr_vector_eps`, `lilicorr_logit_scale`,
  draft layers laid out at `block_count + index`, a `feature_norm`/`feature_mlp` at
  `block_count + num_layers`), which makes it a reference for milestone 5 rather than a from-scratch port.
  The PR also fixes that draft's hidden-projection scoring to run **only when the batch is exactly one
  draft block per sequence** (`n_tokens == dflash_block_size * n_seqs_unq`), a shape constraint our own
  route has to respect.
- **Vision, for phase 2**: the Omni model's tensors are reshaped to `(hidden, 3, patch, patch)` for images
  and `(hidden, 6, patch, patch)` for video with a temporal patch of 2, `vision_model.embeddings.
  position_embedding` is squeezed, `.layer_scale…lambda1` renames to `.weight`, the projector's
  `mlp1.{0,1,2}` are its norm, `linear1` and `linear2`, the image mean and std come from the preprocessor's
  `norm_mean`/`norm_std`, and the position table is kept whole for runtime interpolation.

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

Verified against its own 179 tensors, and reconciling exactly with its 962,485,872 bytes — which is what makes
the parameter count trustworthy rather than copied off a card:

| | |
|---|---|
| Parameters | 1.1128 B: 878.7 M packed 4-bit weights, 234.1 M at BF16, plus 54.9 M F8_E4M3 group scales that the Hub's summary omits |
| Body | 5 layers, hidden 4096, 32 query and 2 KV heads at `head_dim` 128, intermediate 12288, vocab 131072 — `model_type qwen3`, `LiLiCorrDraftModel`, with no embedding and no LM head |
| Attention | every layer `sliding_attention` with window 1024, `causal: false`, and attention sinks |
| Convolutions | a grouped convolution, kernel 2 and group 16, wrapping both sublayers |
| Precision | `W4A16_NVFP4` group 16, and only in part: the five body MLPs, `fc` and ten LiLiCorr projections are packed, while attention, the convolutions, the norms and the in/out/anchor/feature heads stay BF16 |

Those six taps land on target layers of type `[moe, mamba, mamba, mamba, mamba, moe]`, which is a concrete
pattern for the context projection to bind against. The card states acceptance 4.30 at draft length 7: that is
the number to match or beat, not a number to assume.

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
informative: **27,728 expert tensors at `IQ2_XXS`** (2.0625 bits, group 256, ggml packing), 80 NVFP4 tensors —
its Mamba `in_proj` and `out_proj` only, not the mixer projections generally — 113 at `Q8_0` including the
head, and no quantization of its KV cache; 24.12 GB in total. So 74B fits 32 GB only at roughly 2.5 bits per
parameter on average. That per-tensor split was measured from the checkpoint rather than read off a card, and
it corrects an earlier, looser line in this document.

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
Host memory behind a device cache, and the sizes decide whether that is possible. **This machine hosts it
comfortably**: 37.21 GB of pinned banks against 247 GiB of RAM, of which 206 GiB is free. A 32 GiB host could
not, and that is the constraint the vendor's own guide is answering when it asks for about 65 GiB — but it is
not this machine's constraint, and an earlier version of this section mistook the card's 32 GB for the host's.
The scarce resource here is VRAM: the dense side is 8.06 GB of a 32.6 GB card, leaving roughly 20 GB for the
device expert cache, about 59% of the 13,824 expert slots, with the remainder necessarily streamed from the
pinned copy. The route's viability is therefore a *bandwidth* question rather than a capacity one, and that
arithmetic is unforgiving. One routed layer's token pulls 9 experts x up+down x 1024 latent x ~2300 intermediate
x 0.5625 B, about 24 MB — **per layer**, and 40 such layers make **952.9 MB per token**, or 3.9 TB for a
4096-token chunk. At this tree's measured in-place SM band of 20–35 GB/s that is 27.2 ms per decode token
(36.7 tok/s) and **111.5 s of pure PCIe for a single prefill chunk**, which is not viable at any width. Decode
over UVA is a fallback, roughly 3x off the cached route's headroom. So pinning is what this machine does, and
the device cache is what makes the route *usable* rather than what rescues an impossible one; NVMe-resident
experts remain the answer only for a host that cannot pin them.

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
| **Host RAM** | the routed expert banks, pinned | **37.21 GB** at NVFP4 against this machine's 247 GiB (206 GiB free), so the pinned route is available here; a 32 GiB host could not hold it above EXL3 |
| **VRAM** | everything else | dense side **8.06 GB as converted** (FP8 5.21, BF16 1.74, FP32 0.12); Mamba-2 recurrent state 4 MiB per layer plus 2.34 MiB of convolution history per sequence (162 MiB across the 40 layers); attention KV at 8 KB per token across the 8 attention layers (2.1 GB at the full 262144-token context in BF16, ~0.6 GB at NVFP4) |
| **VRAM, remainder** | the device expert cache | measured at **34.8%** of the expert slots resident (4,816 of 13,824, 11 of 40 banks), not the ~59% an earlier version of this table assumed: the budget is 15.8 GiB once the dense side, the state and one pass's workspace, the pass's own logits buffer (1 GiB at chunk 4096 and vocab 131072, which nothing had counted), the runtime reservation and the capacity headroom are subtracted from the card |

The pinned expert set is not optional in this design, but the host requirement it implies is 37.21 GB of *free*
RAM rather than 64 GB of total: with 206 GiB available, the load is bounded by reading 45.28 GB from disk and
uploading 8.06 GB to the card, not by capacity. Everything the device holds besides the cache is small — 8.06 GB
of dense weights plus memory that scales with context rather than with the model — which is why the expert cache
is where the remaining VRAM goes and why its size is a Program decision rather than a fixed number.

**The device cache is a cache by construction, and the staged route is what makes prefill possible.** The 80 banks
sum to 40.77 GB against the roughly 20 GB a 32.6 GB card can give them, and the landed pool geometry already fits
them without change: 13,824 slot entries at a 2,949,120-byte stride, `bank_slots` 992 reserving 2.926 GB against
1.46 GB used per bank, and the banks' excess over that reservation is exactly `2 * max_tail_bytes`, which the
pool allocation's own term covers. What is missing is one entry point on `ops::nemotron_h_moe_experts` that reads
plane-addressed staged slots instead of a whole-bank `Weight`; the Qwen4 route's staged branch transfers nearly
verbatim, but onto its **A16** path, because this conversion carries no activation divisor. Staging is per bank
rather than per expert for a measured reason: the engine stages 103 columns at the wide-prefill threshold, and a
chunk that wide already touches every expert in any of these banks (9 * 103 >= E).

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

Read out of the downloaded Lightning checkpoint's own safetensors headers, so the converter's input is pinned
rather than described:

| Tensor | dtype | shape |
|---|---|---|
| expert and shared-expert `weight` | **U8** | `[I, K/2]` — two 4-bit codes per byte, packed along K |
| expert and shared-expert `weight_scale` | **F8_E4M3** | `[I, K/16]` — one scale per 16 input elements |
| expert and shared-expert `weight_scale_2` | **F32** | scalar — the per-tensor second level |
| FP8 projection | F8_E4M3 `weight`, F32 scalar `weight_scale`, F32 `[1]` `input_scale` | per-tensor W8A8, the input scale being the activation divisor |
| norms, `A_log`, `D`, `dt_bias`, `conv1d`, `gate`, embeddings | BF16 (this family's router weight F32) | unquantized |

Three of those details are load-bearing for a decoder and are worth stating exactly. The `U8` codes pack two
E2M1 values per byte with the **low nibble holding the even element**. The two scale levels compose as
`e2m1(nibble) * weight_scale * weight_scale_2`, and for a block they reduce to `block_amax / 6` — the E4M3
range the NVFP4 format is defined on. And `input_scale` is present *only* where activations are quantized, so
its presence is the W8A8 and W4A4 signal rather than a separate flag; the FP8 form never carries
`weight_scale_2`. KV scales, where a checkpoint exports them at all, are named
`<attention>.k_proj.k_scale` / `v_proj.v_scale` and disappear entirely under a constant-range cast — which is
what the backbone of the flagship does, and what our own FP8 KV does.

The naming the loader will see is `backbone.layers.N.mixer.*`, under a `language_model.` prefix in the omni
wrapper. ModelOpt's own HF↔Megatron table is `export/plugins/mcore_nemotron.py`, not the export-mapping commit
that appeared to promise it.

That is exactly the shape our `import_encoded` path already consumes, so the conversion needs no new format
work. Two details will bite a loader that assumes its own dtypes: in that checkpoint `A_log`, `D` and `dt_bias`
are **BF16** while this branch's scan Op requires FP32 for all three, so the conversion has to upcast them; and
its attention projections are BF16 although the recipe puts them at FP8 — we follow the recipe, which is what
the flagship release does.

### What is not in main yet

Main is not the whole story for this model: several pieces are open branches rather than merged code, and two
of them are a *recipe* and a *format*.

- **The search** — open PR #2582, branch `weimingc/autoquant-mtp-recipes`: an AutoQuantize recipe for the Super
  VL checkpoint. It searches every decoder module and `lm_head` over weight-MSE NVFP4, FP8 and BF16 at a 4.9
  effective-bit budget using gradient-based sensitivity scores, with the MTP experts fixed at NVFP4 W4A4 and
  the backbone KV as a fixed-range FP8 cast. Its `disabled_layers` list is the clearest statement of what stays
  BF16 by construction — `*mixer.conv1d*`, `*mlp.gate.*`, `*router*` and the whole vision/audio stack — and it
  states the accounting explicitly: "Four value bits plus one FP8 scale per 16-element block". Two caveats from
  its own README keep it in proportion: the 4.9-bit budget covers only the searched language modules rather
  than the whole checkpoint, and the search yields a *candidate*, not the published QAD weights. So it refines
  this plan's fixed route rather than replacing it, and the assignment it would choose is not published.
- **The 24 GB format** — branch `hungyuehc/omniml-5823-iq2xs-nemotron-recipe`, with **no PR at all**: the
  recipe that produced the released Bolt-24GB, carrying both a quantizer and reference reconstruction
  (`validate_iq2_xs_reconstruction.py`, `validate_iq2_xs_stock_ggml.py`). Main has no Bolt recipe directory
  whatsoever, so the two-bit route this document declines is unmerged work with reference decoders, not merely
  a released file.
- **The draft** — open PRs #2824, #2826, #2827 and the external-draft series #2637/#2687/#2688, on branches
  `haoguo/dflash-*` and `chenhany/dflash-*`: a DFlash *training* stack — packed pretraining data, a
  language-modelling anchor for a trainable base, a draft-loss weight, a training runbook, a vocabulary swap
  for cross-tokenizer drafts — plus RoPE-theta fixes for DFlash heads. That is the machinery that would let a
  drafter be trained or adapted to a target this repository actually serves, which is precisely phase 5's
  position.
- **Offload, conversion-side again** — PR #2825: a regression test for exporting NVFP4 from a CPU-offloaded
  model, from a bug where a 64 GB DGX Spark's `cudaMemGetInfo` under-counted reclaimable page cache, so
  `hf_ptq` offloaded part of the model and the NVFP4 scale arithmetic crossed devices. Main is fixed, and
  offloaded models have used a *streaming exporter* since #2008. That is a third independent confirmation that
  NVIDIA's offload story is a conversion-time mechanism rather than a serving one.
- **Expert quantization details** — #2745 (ungated and transposed fused experts), #2820 (native FP8 experts),
  #2821 (merging input amax across tied fused experts): each touches the exact shape of model we have, ungated
  experts behind a fused dispatch.

The checkpoint inventory decides how we get there. Our own Bolt 74B has **no official NVFP4 release** — what
exists is `…-Bolt-74B-A9B-24GB` and `…-24GB-LiLiCorr`, the GGUF-derived 24 GB quant whose experts are 2-bit,
which is precisely the LiLiCorr draft's base and confirms it. So the NVFP4 conversion for this checkpoint is
ours to make, from `…-Bolt-74B-A9B-BF16`, using the recipe above.

But `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` exists as a *near* sibling rather than the same
architecture, which its own config settles: it is `NemotronHForCausalLM`/`nemotron_h` with 52 layers at hidden
2688, top-6 of a uniform bank, and **no `moe_latent_size` and no `block_configs` at all** — so it has no latent
round trip, which is the newest part of our op set. What it does give us is an encoded ModelOpt checkpoint at
about 18 GB that fits this machine: the `import_encoded` path against real bytes, the Mamba mixer, the
attention, an offload pool, the Engine and the decode loop, all exercisable on hardware we have. It is a
partial end-to-end test and not the family's own shape. The only NVFP4 release that *is* our architecture is
the Super VL 120B at ~70 GB, which carries the same 64 GB-host requirement as our own conversion will.

Its descriptor also deviates from the recipe in a way worth knowing before we copy it: it leaves the attention
projections at BF16 — `exclude_modules` names every `[qkvo]_proj`, and its 46 FP8 tensors are exactly the 23
Mamba layers' in and out projections — whereas the Super VL quantizes them, its 113 being 80 Mamba plus 32
attention plus the head, which is the recipe exactly. Where the two disagree, we follow the recipe and the
flagship.

## What the vendor's own deployment does

The model cards carry the deployment recipe for this exact checkpoint, and several of its choices settle
things we would otherwise be guessing.

| Their choice | What it means here |
|---|---|
| `--kv-cache-dtype fp8` (24 GB card; the BF16 card has no such flag) | the FP8 KV the ModelOpt recipe names, and this repository already has an FP8 KV codec |
| `--mamba-ssm-cache-dtype bfloat16` / `--mamba-cache-dtype bfloat16` | their *deployment* runs the SSM state in BF16 even though the checkpoint declares `mamba_ssm_cache_dtype: float32`. Ours stays fp32, which this branch's scan test justifies: a BF16 state image misses its criterion by 12x |
| `--max-model-len 262144`, `--max-num-seqs 1` | the 256K context the config declares, served one sequence at a time — inside the one-to-eight concurrency this engine fixes at start |
| `--enable-chunked-prefill` | their long-context prefill is chunked; ours is one compact decode batch per round, so this is a difference to keep in view rather than a target to match now |
| `--tool-call-parser qwen3_coder`, `--reasoning-parser ultra_v3` | the frontend pieces this family needs: the ultra_v3 reasoning parser ships beside the checkpoint (`ultra_v3_reasoning_parser.py`) and the tool parser is the qwen3_coder one this tree already has |
| `temperature 1.0`, `top_p 0.95` | the recommended sampling defaults, inherited from the Super VL client |
| `enable_thinking=True/False` in the chat template | the reasoning mode is a template control, so the family's frontend needs that template |

The speculative setup is named exactly, in the 24 GB card's invocation: `"method":"dflash"`,
`"model":"…-24GB-LiLiCorr"`, `"num_speculative_tokens":7`. So the drafter is a **DFlash** drafter (LiLiCorr is
the checkpoint), it proposes seven tokens, and — the part that matters — it is bound to the **24 GB** target.

Two facts here change what we should expect rather than what we should do. That 24 GB release is **QAT**: its
card says "IQ2 (2-bit quantization with quantization-aware training)". So the drafter was trained against a
differently *post-trained* artifact, not merely a different precision family, which sharpens the pairing risk
this document already flags rather than resolving it. And their own benchmark table puts that 2-bit QAT release
within a few points of BF16 on most agentic benchmarks — which sets the floor for our conversion: an NVFP4
W4A16 target at 4.5 bits should be at least as good, and that is the bar the conversion gets measured against.

The same cards also name a third route for this checkpoint that we are deliberately not taking: serve the 24 GB
IQ2 target itself, which is what the drafter expects and what fits a 5090 outright. It would need a 2-bit
expert codec, which this tree does not have and which the NVFP4 decision was taken to avoid.

The technical report settles why the two releases differ, and it is the strongest support this plan has for
its own route. Verbatim: "Lightning 3.5 is designed to excel in local, agentic use cases… we quantize
Lightning 3.5 to **minimize weight fetch and raise activation precision to maintain accuracy**… deploys W4A16
**regardless of GPU SKU**, while Super 3.5 VL… deploys W4A4 on Blackwell, and W4A16 on non-Blackwell". So
W4A16 is not the cheap variant of W4A4 — it is the deliberate choice for local, agentic serving, taken to keep
activation precision, which is precisely our case and precisely what a conversion from the BF16 checkpoint
produces. The same report gives the recoveries: median accuracy 99.62% over 36 benchmarks for the Super NVFP4
(it also states 99.52 and prints 99.36/99.52 in a table, an inconsistency it does not resolve) and 99.24% for
Lightning's W4A16. Cite those as the accuracy bar, not as guarantees for this model: the report covers Super
and Lightning, **not Bolt**, and its throughput figures are GB200-class.

One negative result that matters for planning: **nothing in the vendor's serving material mentions expert
residency or CPU/host offload at all.** Their Nemotron serving keeps the experts on devices and uses expert
parallelism (EP=4 for PTQ, EP=32 for QAD); the only host-tiering they describe is a 240 GiB host **KV** tier on
Rubin and a `host_cache_size: 0` in a TRT-LLM Spark config. Our host-RAM expert bank, then, is this
repository's own design carried over from Flash-Next rather than a vendor recipe we can follow, and there is no
published counterpart to compare it against.

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
2. **Config and artifact — done.** The per-layer block descriptor is parsed and tested, `load.{h,cpp}`
   binds every declared parameter from the artifact, and the converter's per-layer loops produce it.
   The artifact format did not change.
3. **Ops, one at a time, each against the oracle**: `relu2` + RMSNorm → Mamba-2 (selective scan, fp32
   state) → router + non-gated experts → the latent round trip → the new latent-MoE op → attention →
   the block → the 88-block forward.
4. **Converter and Engine path** — the artifact is converted and independently checked, the loader and the
   Program are landing; then the host-offload cache work above, then the decode loop.
5. **Draft** (LiLiCorr) after the target runs, qualified against an NVFP4-family artifact.

### The load path, and what it stops at

The Engine keeps no architecture registry to consult. It **sniffs the artifact**: `artifact_is_gemma` is the only
family predicate, and everything else falls through to the qwen3_5 loader, which then rejects what it does not
recognize. So this family needs its own probe and its own branch — and the probe cannot go through
`models::resolve_architecture`, which an earlier version of this section claimed. The converter stores only the
keys this family's parser reads, so the artifact's `text/config` carries **neither `architectures` nor
`model_type`**, and the registry's `NemotronHPuzzle` arm — keyed on the *source* checkpoint's
`NemotronHPuzzleForCausalLM`/`nemotron_h_puzzle` — is unreachable from the artifact. The probe therefore keys on
what this family does emit and no other does: the per-layer `block_configs` list, with `mamba` and `router`
beside it. Carrying `architectures`/`model_type` in the artifact would not fix that, and the reason is worth recording, because
it says how dispatch in this tree really works: **it is bespoke per family.** `artifact_is_gemma` is itself nothing
but a config-key sniff of the artifact's own `text/config`, and `resolve_architecture` is called only from family
config parsers — the qwen one on the artifact's strings, and Gemma's as a self-consistency check on the constants
it synthesizes — so no family is *dispatched* through the registry's enum. Meanwhile this family's parser
validates an **exact key set** by design and its converter emits exactly that set, so adding the two fields would
mean loosening the parser *and* re-converting 45 GB to reach a mechanism nothing else dispatches through. The probe
therefore keys on `block_configs`, which is the same shape as the probe beside it.

A family plugs in through a fixed, small seam: `plan_load` → `materialize_model` returning the family's `Model`;
a runtime contract bundle naming the Program and the request/prompt/output types; an instance holding the model,
the frontend, the program and the capacity resolution; and arms in the Engine's variant and in its
`if constexpr` visits, which each currently end in a qwen-shaped `else`. Gemma's bundle shows how little has to
be family-specific: it reuses most of the qwen contract types, implements the rest thinly over its own Program,
and switches off the features it does not support in its engine options rather than accepting and ignoring them.

**The frontend is the one piece that had to be decided by evidence rather than by reuse, and the evidence says
reuse is impossible.** The qwen3_5 frontend cannot even construct against this artifact: it validates the
tokenizer's `pre_tokenizer` against the Qwen split regex and refuses the checkpoint's own Nanotron-v2 split, and
its tokenizer-config check requires a `pad_token` of `` that this checkpoint's vocabulary does not contain — so
there is no honest resource rewrite that would let it through. This is not a validation quibble: the split is
*implemented* per family, the converter measured the Qwen and Nemotron splits disagreeing on real text (97 ids
in 605k), and the converter itself recorded that the artifact cannot load until a runtime family implements the
checkpoint's split. So the family gets its own frontend over the two format-neutral pieces this tree already
shares — the tokenizer implementation and the Jinja engine — exactly as Gemma's does, plus one arm on the shared
`TokenizerFamily` enum, and its sampling defaults come from the artifact's own `generation_config.json`
(temperature 1.0, top_p 0.95, the vendor's own recommendation) rather than from a table.

This stage stops at **loading**, and that is a decision rather than an unfinished edge. The landed forward is a
zero-state prefill that keeps nothing between calls, so a Program that drove it through prefill and decode would
compute a *wrong* model rather than report a missing feature: the convolution history, the recurrent plane and
the attention K/V would all silently restart. The Program therefore owns that state — attention K/V for the 8
attention layers, and convolution history plus an fp32 recurrent plane for each of the 40 Mamba layers — reports
its capacity and its bytes honestly, and refuses to execute until the forward can persist what it computes.
`engine ready` is the observable end of this milestone; the decode loop is the next one.

One thing is cheaper than the decode loop and should come first: **CausalScoring needs no state at all.** It is a
single pass over a token sequence returning per-token logits, which is exactly what the landed forward already
computes, so the Program can serve `causal_score` with the composition as it stands — arena sizing from
`model_workspace_bytes`, the execution view, and the head. That makes scoring the first real execution of the
loaded model, and it is worth more as evidence than any name-or-shape check: it is the only thing that exercises
the materialized pointers, including the routed banks read over UVA from page-locked host memory. What it does
*not* come with is an independent oracle for the real 74B: the composition's oracle is the tiny-config dump, and
the weights' oracle is the checkpoint, but a second implementation of the full model at this size would mean
either running the 141 GB BF16 checkpoint on the CPU or converting the GGUF with llama.cpp's own converter — the
latter now possible because that implementation shipped. Choosing that oracle is a decision to make deliberately
before quoting a perplexity, not to assume.

### The decode path, and the one op it is missing

The decode design is settled: what matters here is its verdict. The convolution and the scan need **no signature
change and no new kernel** — both already have distinct-state forms, `T=1` is validated and tested, and the state
pool's convolution and recurrent slots are already contiguous views in the ops' exact layouts and dtypes, with the
reference's own semantics (`S <- exp(dt*A)*S + dt*outer(x,B)`, read *after* the update; the last four raw conv
inputs, taps innermost) matching what the op reads. `D*x` and the gating stay inside the scan and the gated grouped
norm. Carrying state through the composition is a nullable `BlockState*`/`PassState`, so a null state remains
bit-for-bit today's zero-state pass. One detail is a trap worth stating twice: the reference's convolution window is
taps innermost (`c*4+k`) while the op's state is tap-major and three wide (`w*C+c`), so the two are a transpose of
each other rather than the same buffer.

**Attention is the piece with a real gap.** This family needs unbounded causal K/V — no window, no rotation,
positions absolute from the K/V length, scale `1/sqrt(128)` — at head width 128 with two key/value heads, and no
landed op writes or reads that geometry's K/V state: the registered paged, ring and materialize profiles are built
for D=256 or eight key/value heads, and the wrapper refuses `window == 0` even though the flash kernel underneath
already implements causal-only visibility. So the Program has to decide that route explicitly rather than inherit
one, which is what it is being asked for.

The oracle is the landed prefill itself: a stateful decode must reproduce the zero-state pass's logits at the same
position — per stage against the dump, and at the logits, using the criteria the op tests already quote. The
two-chunk equivalence check has to compare the *second* chunk's **first** tokens; comparing the wrong columns is
how that check would pass while proving nothing.

## Open items

- The draft pairs with the **24GB NVFP4** target, not the BF16 one, so its qualification belongs on our
  NVFP4 route; the cheap decisive check is ranking the draft's token inside the target's own next-token
  distribution, the diagnostic that exposed the Gemma 4 QAT mismatch.
- The host-offload cache pins the expert set, and this machine answers that requirement: 37.21 GB of pinned
  banks against 247 GiB of RAM with 206 GiB free. An earlier version of this item called it a 32 GiB host and
  treated the pinned route as unavailable — the figure was the card's, not the host's. The real constraint is
  the 32.6 GB of VRAM, which is why the device cache is a cache by construction (the full 80-bank pool would be
  40.77 GB) and why the staged route exists at all. Reducing the pinned set, or a smaller expert route, remains
  the escape for a smaller host; neither is needed here.
- **Two count paths on the offload route are unchecked, and one of them is a real over-read.**
  `ops::moe_experts` bounds its `slots` argument by the compile-time `kOffloadMoeExperts` (512), which is larger
  than this family's *smallest* bank (240), so a wrong count silently reads past the bank instead of failing; and
  the resolve kernel indexes `used[ids[i]]` and `slot_for[ids[i]]` without validating an id against the bank's
  own width. Both need the per-layer count checked — the same class of bug llama.cpp shipped a fix for in this
  very model, and the reason the loader now validates the router's width against the bank's.
- **The staged route must be proven a byte-exact copy, not merely a fixed order.** A fixed staging order does not
  make a reduction deterministic — that is what sank the Gemma 4 gather inversion — so the check is a D2H compare
  plus bit-identity across cold, warm and lent states, and the staged result must match the direct route's bit for
  bit rather than within a tolerance.
- **The tokenizer's Unicode class tables are a version behind the reference, and this is a cross-cutting decision
  rather than a family fix.** `src/text` classifies through utf8proc 17.0.0, whose tables call 4,699 code points
  letters, marks or numbers that the reference's Rust `regex` tables call unassigned — 28 ranges (Tolong Siki,
  U+11DB0..11DDB and others) plus U+0295, with zero cases in the opposite direction. It affects only exotic
  scripts, and it is shared with Qwen and Gemma, so fixing it means a Unicode-version table in `src/text` rather
  than a change here. It is recorded so the divergence is never mistaken for a porting bug.
- **The frontend refuses four things by name rather than approximating them**, each with its reason in the code:
  a thinking budget (this model's thought block is closed by the model, not by an Engine-injected control token, so
  Qwen's canned control sentence is not injected), output constraints (the Program consumes no token masks), any
  forced or restricted tool call (the wire format is the in-tree parser's, but only `tool_choice` auto or none is
  supported), and vision (the artifact has no vision component and this family's `TextConfig` has no field for
  one). The first three are the pieces a later milestone pays for; the fourth is phase 2.
- **The state's key/value index is ambiguous, and both extents are 128 today**, so a swap would be
  silent: `LinearAttentionStatePool` plans `ne[0] = key_head_dim` while the GDN and KDA kernels index
  the state as `value * kStateDim + key`. Mamba-2's 128 (state size) against 64 (head_dim) makes it a
  real decision, so a test has to pin the layout rather than a comment.
- **No artifact-side pass runs yet, and that is now the blocking item.** `forward_model` reaches block 0 and dies
  with `fp8 linear: unsupported shape`: this family's dense projections are FP8 at K = 4096 and 1024, and the FP8
  linear route has no registered pair for them — which is the item this document already carried as "no registered
  pair has k = 4096 or k = 1024". Until those are registered there is no pass timing, no staged-versus-in-place
  comparison, and no measurement of anything that needs a forward. Two pre-existing defects had to be fixed to get
  even that far: `require_matrix`/`require_bank_latent` compared a quantized matrix's typed tensor, which is unset
  by design, and so refused every FP8 and NVFP4-bound parameter.
- **A captured pass would defeat the cache**: the pager's `snapshot_residency` still synchronizes, so under CUDA
  graph capture every expert would be staged. This family does not capture yet, which is the only reason it does not
  matter today.
- **The resolved policy is one lane and a 2048-token window**, so these numbers do not describe the long-context
  case that makes an NVFP4 KV worthwhile — at that window the KV term is 16 MiB either way. Reaching it needs the
  window set deliberately (`--max-context`), and NVFP4 KV needs a quantized K/V route for
  `sliding_causal_attention`: this family's `AttentionState` is BF16-wired and the engine refuses another dtype for
  it rather than substituting one.
- **The shared pager had a real hazard, now fixed**: bank selection was `layer & 1`, which collides as soon as one
  layer in between is cache-resident — two staged layers two apart land in the same bank. It is recorded because it
  was Qwen4Exp's mechanism too: the pager was moved to `src/models/execution/` rather than forked, so the fix
  belongs to both families.
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
| Composition | `src/models/nemotron_h/forward.cpp` composes the three mixers and the block. The tiny model is walked block by block against the dump: 37 stages compared with a per-stage rel L2, everything at the 0.0016–0.008 BF16-operand floor and the logits at 0.00633, five refusal cases, and coverage asserted (37 compared, 3 reported but not in the dump). Its loosest stage, block 1's `fc2_latent_proj` at 0.0498, is explained by a measured near-tie: 2 of 130 tokens sit where the reference's K-th and (K+1)-th corrected scores differ by 0.002x the BF16 logit perturbation, so their expert pair moves while the other 128 ids match exactly and block 3 moves not at all. Two runs agreed to every printed digit. |
| Artifact | `nemotron_h_74b_nvfp4` converts the BF16 checkpoint in 2067 s to 767 objects and **45.28 GB** (a 32.00 GB file plus a 13.28 GB part), of which the routed expert banks are **37.21 GB** — within 0.03% of what this document predicted from the checkpoint's config alone. The dense side is FP8 at 5.21 GB of a converted 8.06 GB, which is why the total is below the ~46 GB predicted when that side was assumed to be Q8. Every design choice is measured rather than asserted: one divisor per bank costs at most 0.0001 rel L2 against per-expert divisors, the BF16 and FP32 sets are bit-exact, the conv payload is bit-exact tap-major, and the experts' 0.081 rel L2 is the *format* floor — established by running the same independent check against the existing Gemma NVFP4 artifact first, where it reads 0.070. |
| Artifact completeness | `tests/convert/test_nemotron_h_artifact.py` proves the parameter set from metadata alone in about 1.3 s: 28,331 language tensors map to exactly one parameter each and every parameter is explained — 0 unmappable, 0 absent, 0 duplicate, 0 unexplained — shapes agree except for the one transform the container makes (the depthwise convolution's stored `[10240,1,4]` becoming the tap-major `[10240,4]`), format classes agree exactly (27,728 NVFP4 as 27,648 routed and 80 shared, 113 FP8, 330 BF16, 160 FP32 — the 120 per-head state vectors and the 40 correction biases), and all 40 MoE layers have their expert count confirmed **from the checkpoint's own tensors** rather than from the config: 240 in 12 layers, 304 in 6, 368 in 9, 432 in 10, 496 in 3. The artifact's stored `block_configs` agrees with what the tensors say everywhere, so this release's config does not lie. Eleven one-at-a-time perturbations of the fixture — a lost expert, a removed parameter, a vision tensor renamed into the language set, the head re-classed to BF16, the tap-major transform dropped, a bank range shifted by 1024 rows, a config claiming 304 experts for a 240-expert layer — are each caught with the expected message, which is what makes the eight passing cases evidence rather than ceremony. |
| Loader | `materialize_model` binds 763 parameters from the artifact, with **37,206,098,240 pinned bytes** — exactly the 80 routed bank parents, which are declared as whole packing groups rather than per expert, because a row range of a quantized parent cannot re-derive its own scale plane — and **8,061,946,116 device bytes**, matching the dense-side figure above independently. The plan takes 7.5 ms and materialization 14–37 s depending on page cache, with no file mapping (`mapped_file_bytes == 0`). It refuses a bank whose expert count disagrees with its router or the config, checked on all 40 MoE layers, with the message: *layer 7: the artifact's bank holds 304 experts and its router 240, where this layer's config says 240; a global expert count must not reach a layer's bank*. The loading library carries no `Model` reference at all (proven with `nm` on the archive, and by a target that links only it), so planning and execution stay on their own sides of the split. |
| Engine path | `ninfer out/nemotron_h_74b_nvfp4.ninfer --prompt hi --max-new 1` reaches **`engine ready`** for `nemotron_h_74b_nvfp4` with `total 15.5s` and `weights 7.51 GiB` — 7.51 GiB uploaded in 6.2 s at 1.22 GiB/s out of the 45.28 GB artifact — and reports the family's own `vocab 131072`, `hidden 4096` and `native context 262144` read from its config, with state 178 MiB, weights 7688 MiB and an 824 MiB runtime reservation. Generation then refuses with the reason instead of computing: *the landed forward is the zero-state prefill ... the state itself is here — the Mamba pool's slot and `AttentionState`, whose append/attend route runs the attention over a lane's own K/V — and what is missing is the composition wiring it into `forward.{h,cpp}`*. The engine stays alive and available after the refusal, which the test asserts. |
| Engine boundary | Dispatch is a third bespoke probe beside `artifact_is_gemma`, keying on this family's `block_configs`, and `NemotronProgram` implements the whole contract over the family's Program — nothing reached the Program's own surface as a missing signature. Three limits are recorded rather than smoothed over: that refusal arrives through the shared worker's `logic_error` recovery, so one `[engine] WORKER RECOVER:` line precedes the CLI's own message; `causal_score` refuses for a different reason, because the zero-state forward *would* serve scoring but the Program exposes no scoring entry point yet; and the 37.21 GB of pinned host residency is absent from the shared `MemorySummary`, which has no host-weights field for any family, so it is read from `model->storage_stats().pinned_host_bytes`. |
| Expert residency | The routed banks now take a residency split at bind time — a cache-resident bank is bound `Device` and uploaded once, the rest stay `HostPinned` — sized from the card *after* the KV/state/workspace/logits resolution and never the other way round. Measured on the artifact at `--expert-cache auto` against `--expert-cache 0`: **4,816 of 13,824 slots resident (34.8%), 13,985,906,688 B in 11 of 40 banks**, and **pinned host bytes fall from 37,206,098,240 to 23,220,191,232** — a 13.99 GB drop that is exactly the resident set, confirmed independently by the materializer's own counter. The staged route is the same reader at different addresses (one `Nvfp4BankReader` over four planes), so it is a byte copy rather than a second implementation, and its placement identity is proven bit-for-bit across four geometries together with all five refusals. The op is also verified against an independent FP64 oracle — worst deviation 0.20–0.31% of tensor magnitude against a 3e-2 limit, 0 non-finite values, across twelve in-place and staged comparisons — after the *fixture* was found to have its two divisor branches swapped: the whole-bank form was indexing elements 1..E−1 of a one-element vector, and the documented rule is `row / divisor_rows`. Two loads put residency at 11 of 40 banks (13.03 GiB) and 10 of 40 (12.27 GiB), so the figure is measured from the free VRAM at load rather than being a constant, and the bench cross-checks the pinned bytes against a geometry-derived total — the resident bank bytes plus one divisor word per plane pair — rather than trusting the counter alone. **One correction to these figures, measured 2026-10-16**: they were taken with the pool reserved *twice* — the caller's `reserve_bytes` already included it through `Program::device_bytes`, and `plan_expert_residency` subtracted it again as its `staging_bytes` — so 2,790 MiB of a 29,178 MiB card was unowned (13,176 resident + 2,790 staging against a 16,012 MiB budget, i.e. 8.7% of the card idled by accounting). The plan now takes the pool out of the reservation once (`Program::expert_pool_device_bytes`), and the same card at the same policy gives **13 of 40 banks, 5,808 of 13,824 experts (42.0%), 15.4775 GiB resident against 19.1733 GiB pinned**; the pre-fix split is reproducible on the fixed tree by capping the budget at 16,012 MiB, which returns 4,752 experts / 12.8672 GiB / 21.7837 GiB to the digit. |
| Expert cache: slots against banks | **Measured, and the trade does not pay.** The pool's size was the geometry's — `2 * max_bank_experts` = **992 slots, 2.93 GB**, the two staged banks — and is now a policy (`--expert-cache-slots N`, `ExpertResidencyPlan::cache_slots`), granted out of the same expert budget as the resident banks, one slot's bytes for one slot's worth of bank not uploaded. Four arms on one card at one automatic budget (**18.66–18.76 GiB**, the pool and the residents inside it, so the arms differ only in the split), pool slots **992 / 1984 / 2976 / 3968** (durable cache 0 / 992 / 1984 / 2976): resident banks **13 / 11 / 8 / 6**, resident experts **5,808 / 4,752 / 3,648 / 2,720** (15.4775 / 12.6694 / 10.0195 / 7.36963 GiB) against pinned **19.17 / 21.98 / 24.63 / 27.28 GiB**; warm misses per column **58 / 54 / 63 / 66** of 243 / 261 / 288 / 306 assignments; warm gathered **141.6 / 133.0 / 155.3 / 163.8 MiB**; the fetch kernel **4.17 / 3.96 / 4.42 / 5.41 ms per pass** (nsys, 24 decoded tokens with graphs off, kernel total ÷ 24, i.e. 0.154 / 0.136 / 0.137 / 0.153 ms per instance over 650 / 700 / 775 / 850 instances); the three routes at one column **narrow 12.6 / 12.5 / 12.9 / 13.3 ms, staged 392.4 / 454.9 / 505.6 / 582.3 ms, in-place 57.9 / 57.0 / 58.7 / 59.5 ms**; and the acceptance command `-n 4 -r 1 --warmup 0` reads **71.7 ± 0.5 / 69.9 ± 0.2 / 68.4 ± 0.9 / 66.3 ± 0.3 t/s** (three interleaved samples each, which is the noise control: −2.5%, −4.7%, −7.6%, monotone and reproducible to ~1%). Warm and interleaved (`-r 3 --warmup 1`, five samples in alternating order) the four are inside ±1.5% of each other against a **5% load-to-load drift**, so nothing above the structural pool is a gain. Two mechanisms, both measured: **capacity** — a slot is sized at the *widest* bank's stride (2,949,120 B per expert) while a resident expert is sized at its own layer's width (2,359,296–2,949,120 B), so the same bytes hold 0–25% more experts as banks — and **reuse, which is not cache-size-limited**: the miss *fraction* stays at 21–24% across a 4× range of cache slots, and the warm gathered bytes fall only 6% (141.6 → 133.0 MiB) at +992 slots before rising again, so the extra slots hold experts the next token does not route to rather than warming a working set. What the extra bytes do cost is exact and immediate: 2.01 GiB of pool per 992 slots, 2 banks and 1,056 experts of residency gone for the first 992, and the cold 4-token command pays that as a −2.5% rate because every non-resident bank must be filled once. The floor is structural and was checked, not assumed: `ExpertPager` requires `bank_slot + 2 * max_experts == slots`, so 992 is the smallest pool there is and the "smaller by half" arm does not exist without changing the staged route's two-bank prefetch (which was not changed). Identity is unchanged at every size — one column **131,072 logits, 0 differing, max_abs 0, rel_l2 0** and 2 / 4 / 8 columns **262,144 / 524,288 / 1,048,576 words, 0 differing, max_abs 0, rel_l2 0** in all five configurations including the pre-fix-ledger control, so a different pool moves where bytes come from and never what is computed. Artifacts: `profiles/nsys/nemotron_p1/slots/` (`run.sh`, `residency_{before,0,992,1984,2976}.log`, `tg4_*`, `tg24_*.kern.csv`, `warm_sweeps.txt`, `tg4_cold_repeats.txt`). |
| Decode | **The model generates.** A stateful decode of one token at a time from a 122-token prefill reproduces the landed zero-state pass's logits **bit for bit** — `logits rel_l2=0 max_abs=0` — and so does a two-chunk prefill compared at every column including its second chunk's first tokens, which is what makes that check non-vacuous. `ninfer out/nemotron_h_74b_nvfp4.ninfer --prompt "The capital of France is" --max-new 8` emits eight tokens, greedy reading *The user is asking for the capital of France. This is a simple factual question. The capital of France is Paris.* The 320 stage comparisons are clean except for the three router tensors at the decode positions, which is the memory order the landed test already documents as not shared with the composition (`topk(sorted=False)` defines no rank order, so the dump path matches the rank tensors as a set per token); that it is a harness rule rather than a state difference is settled by the line beside it, since the whole 88-block forward's logits are identical to zero. `forward_model` gained a frontier check, which immediately refused a harness that re-used a state after both chunks instead of rewinding it to the 122-token frontier. Timings, honestly: prefill 21 tokens in 5.1 s, then 0.5 s/token; a greedy run prefilled in 2.4 s and decoded 24 tokens at 0.7 s/token — slow, and the next thing to fix rather than to describe as working. |
| Staged-route speed | Measured on the fully linked tree, one lane, 2048-token window, best of N after a warm-up: a 512-token pass takes **3,046.8 ms with the residency split against 1,170,329.8 ms with every bank pinned over UVA — 384×** — at 11 of 40 banks resident (34.8%) and 22,059 MiB staged per pass, which is exactly the pinned set and is therefore the split's own consistency check. 4096 tokens take 21.8 s staged. The in-place route's 4096-token pass is **reported unmeasured and deliberately not taken**, correcting an earlier estimate of about 2.5 hours: its per-token cost is 0.186 s at T=8, 0.205 s at 128 and 2.29 s at 512, so the 128-to-512 segment grows like T^2.7 while the route's own traffic estimate grows linearly, and the bench's in-place GB/s column is a derivation rather than a counter — so what is claimed is the bound and the trend, not a mechanism. Auto reproduced within 1.3% across three runs, and the 128-token point is insensitive to the binary at 0.004%, which is what makes the cross-binary 512 comparison fair. All seven newly registered projection shapes are qualified against an FP64 oracle with their packed weights decoded by the test itself: 58 kernel runs per FP8 pair across token extents and activation policies, plus graph replays, and the refusal now names the offending N, K and T. |
| Narrow-route speed | **A one-column decode call now gathers instead of staging its bank.** `top_k * width >= experts` — the identity whole-bank staging was justified by — is a *prefill* identity: it holds from `ceil(max_bank_experts / top_k)` = **56 columns** for this artifact's widest bank (496 experts, the narrowest 240), while a decode call routes 9 experts of a 240-to-496-expert parent, so staging that parent copies about 27x what the call reads. `RoutedExpertPool` derives that width and the pager's two routes are chosen by it (`ExpertPager::stages`; no option, flag or policy mode), and the pool gained the narrow acquire: `expert_cache_resolve` places the routed ids in the pool's own slots and one gather kernel copies only the misses out of the pinned bank; the two scratch tensors the narrow path hands the Op (`slot_ids` I32 `[K,tokens]`, the pager's miss list I32 `[2*K*tokens+1]`) live in the block's own scratch, and `e955f8d5` then gave the plane-addressed Op its own caller-owned workspace on top of them, allocated from that same arena: what has not changed is that the Program and the harness both size it from `model_workspace_bytes`, which plans the same sequence of block allocations the pass makes, so a caller sizing at the prefill chunk still covers a narrower call — a reader should not re-derive that. The narrow route is **bit-identical** to the in-place route *and* to the staged one — at one column the residency bench reports `131072 logits, 0 differ, max_abs 0, rel_l2 0` for both comparisons, which is what W4A16 with one divisor per matrix and a byte copy of the same slot allows. Measured on the real artifact, one lane, best of two after a warm-up, 11 of 40 banks resident (13.11 GiB device, 22,059 MiB pinned): **narrow 708.1 ms per column with 632.8 MiB gathered** on the cold pass (261 of 261 assignments missed, 72 per warm column), **staged 817.5 ms per column with 22,059 MiB copied**, **in-place over UVA 1,522.8 ms** — the same words for **34.9x** less host-link traffic, out of the same slot pool and with no allocation per call. The staged path did not move: the same harness stages 512 tokens in 3,068.0 ms against the 3,046.8 ms recorded before this change (0.7%; 4,096 tokens 22,377 ms against 21.8 s) and its per-pass staged bytes are identical at 22,059.0 MiB, its own consistency check (staged bytes equal to the pinned set, every non-resident bank once) still holds. **The narrow pass is 708.1 ms against the staged route's 817.5 ms at one column — 1.15x, not 35x — and the reason is the experts' own Op**: an nsys kernel summary of the decode step puts 94% of GPU time in `nemotron_h_moe_experts_kernel<Nvfp4BankReader>` (40 instances per pass, 16.3 ms each; ncu: `<<<tokens, 128>>>`, 8.3% achieved occupancy, 0.43% SM throughput, 88.6% no-eligible, IPC 0.46 — latency-bound on scalar 4-bit loads at one block per token with the nine ranks serialized behind `__syncthreads()`), so the staged route's 22 GB of copies were hidden behind it and end to end `ninfer_bench -n 4 -r 1 --warmup 0` reads 1.42 output tokens per second against 1.36. This route is what makes fixing that Op pay, and the two are complements: at the Op's targeted ~100 us per layer a staged pass is copy-bound at about 364 ms (17.44 GB at 47.8 GB/s) where this route fetches 439 us per non-resident layer, about 10.5 ms per pass across the 24 of them. |
| Narrow-route Op | **The expert Op's narrow route runs a one-to-eight-column call across the whole GPU instead of in one 128-thread block.** Up-GEMV split by expert-local output rows — one block per (rank, row-tile) — writing `relu2` into a per-(token, rank) FP32 product; down-GEMV split by latent rows into a per-(token, rank) FP32 partial; a third kernel sums the `top_k` partials per token in ascending rank order with the single BF16 rounding. The arithmetic tree is the per-token kernel's, deliberately and exactly: four accumulator chains, the `(a0+a1)+(a2+a3)` combine, the tail appended *after* the combine, the per-row divisor hoisted out of the column loop, FP32 products, ascending-rank accumulation — which is what makes the two routes expression-identical rather than merely within tolerance, and what the SASS audit below turns into a static fact for reassociation while contraction remains nvcc's choice in one TU at identical flags. The Op's own test agrees at tokens 1..8 (1024 to 8192 words each) with the bank staged on the device: **max_abs 0, rel_l2 0, 0 differing**. The plane-addressed entry point gained a caller-owned workspace, whose carve is exact, disjoint and complete (`slots*intermediate` then `slots*latent` floats at a 256 B offset), monotone in `tokens` because the cap and the route selector are the same constant, so a Program arena sized at the pass width covers every narrower call; the query reports zero for BF16 and a too-small workspace is refused. Measured on the real artifact, same command and corpus as the baseline: **decode 1.36-1.42 output tokens/s becomes 69.89 — 49 to 51x** — with **prefill 134.80 t/s** at `-p 128`, and the staged path unmoved. One caveat on that figure's composition, because it is a comparison and not a single-variable experiment: the binary it was taken on also carried the sibling latent-projection shape entries, uncommitted beside this change, and those two projections were the second-largest term in the pass (160 instances of the dense fallback at 253 us), so the split between the two changes is being measured rather than attributed here — and what is *this* change's own evidence is the per-instance duration and occupancy of its three kernels. An independent read (source-level, read-only) proves the wide path untouched by diff — 172 insertions and **0 deletions**, a pure append, with the launch configuration and both readers byte-identical — and finds the carve exact and monotone, the stream/event/capture surface clean (three launches on the caller's stream, no synchronization, no allocation, no event), and the row-tile split disjoint, complete and free of barrier divergence, and a static audit of the emitted SASS settles the reassociation question more strongly than any output test can: in all four bodies every FADD is accounted for — three per GEMV forming literally `(a0+a1)+(a2+a3)` with the same association in both routes, plus a `FADD.FTZ` negation inside the inlined reciprocal, and none at all in the rank-sum kernel — and each chain loop decomposes into exactly four accumulator chains (the narrow pair 4 x 16 FFMAs over 64 columns, the wide main loops 4 x 1, its tails a single chain continuing the combine's result), with no fresh partial anywhere. The taxonomy is cross-checked against the same kernel's BF16 instantiation, which emits exactly two combines and no reciprocal because that bank reader's divisor is a constant. What the audit does *not* show is that a chain ascends in `k` — an FFMA site carries no column index — so ascending order rests on the source and on the runtime identity assertion, and the audit is a lower bound (no reassociation present) rather than a proof of equality. Two limits are recorded rather than smoothed: the identity comparison's metric is BF16 output words, so a pass certifies the Op's contract but not the FP32-level claim in the kernel comment — covered in practice, since a contraction difference between the two kernels would diverge by about 2e-6 against a detection floor near 1e-7 — and the narrow route newly requires 16-byte alignment of the activation and the bank planes and 4 bytes for the scales, which the wrapper does not yet validate even though a stored bank's pointers come straight from the artifact. **The model's forward test is not coverage for this route**: the walk binds a dense BF16 bank, so it executes the per-token kernel and is a no-regression check on `allocate_block`, the scratch tensors and the wrapper — the narrow route's reachability is established instead by the end-to-end rate having been produced through it on the real artifact. |
| Wide-route Op | **The same row-tiled route now takes the prefill, and it lands on the bank-traffic floor.** The route was gated at `tokens <= 8`; the token axis is now walked in tiles of 64 inside the Op, so the per-(token, rank) product and raw partial are one tile's buffers (7.88 MiB at the widest geometry this artifact carries) rather than the call's (504.0 MiB at a 4096-token chunk), and the gate admits every NVFP4 plane-addressed call up to this family's widest chunk (4096). Measured on this tree with the pre-change gate forced back in — one variable, same command, same session — prefill goes **116.70 → 262.28 t/s (2.25x)**; against the recorded baseline command it is **110.10 → 262.28 (2.38x)**, and with one warm pass first the same command reads **953.27 t/s**, i.e. a 537 ms pass of which the staged bank copy is 461 ms at 47.8 GB/s — **86%** — so the prefill is staging-bound at the link now, which is the state the earlier bound predicted would be exposed once the Op stopped hiding the copy. The Op itself is **214 ms over the pass, 5.35 ms per layer against 69.6 ms — 13.0x** — and that is at or below the 290 ms roofline this bank geometry sets (2.81 MiB per (token, rank) x 9 ranks x 512 tokens x 40 layers = 518 GiB at 1.79 TB/s), so the decomposition is at its floor rather than merely faster. Per instance under the same profiler, 8 tiles per layer: `row_tiled_down<32>` 400.7 us, `row_tiled_up<64>` 265.2 us, rank sum 2.5 us. The identity is the per-token kernel's arithmetic word for word, and the plane test pins it at 1..8, 64, 512 and 4096 against one per-token reference run at 8192 tokens (a repetition of the 4096 base, so one run serves every narrower width): **max_abs 0, rel_l2 0, 0 differing** everywhere, with a workspace probe showing which route each width took so the comparison cannot pass vacuously. Decode is untouched because it never left this route: HEAD's gate already sent `tokens <= 8` to it, the A/B build's `-n 4` reads 68.35 against 67.93 here, and the recorded `-n 4` profile lists the same 200 `narrow_up`/`narrow_down` instances — so the only token band whose route changes is 9..4096, which is prefill and batched decode at C >= 9. Residency is unmoved at 11 of 40 banks (12.87 GiB resident, card free 29178 MiB at load); the load's H2D reads 21.38 GiB against the baseline's 20.85 because the tree moved under other work, which is why the same-tree A/B is the figure to quote. Two defects were caught from a host model of the carve and the index math before any GPU time, both of one shape — a buffer addressed with two bases: the launch offset the partial buffer at the *tile's* carve while the query reported the carve capped at the call's own T (every write past the validated arena at T < 64), and the down kernel wrote its partial at the call's rank stride while the summing kernel read the tile-local one (right at T <= 64, and above it 8 of 9 ranks read never-written storage while the writes ran past the buffer). Both are single-sourced now — the cap by one function used by the query and the launch, the indexing by `tile_slot` in all three kernels — and the wide identity case is what catches the second class. |
| Latent-projection shapes | **Both latent projections now have registered BF16 routes instead of landing on the dense fallback.** `bf16_generic_kernel` is the fallback for "a problem no registered shape matches" — the file's own words — and the paragraph above says every one of this family's widths reached it: one scalar bf16 load per K step from `n * ceil(tokens/4) / 256` blocks, i.e. at one token 4 blocks of 256 threads reading 8 MB at ~32 GB/s, measured at 253 µs per instance in a pass and 401.4 µs cold in isolation. The new routes are one lane per output row walking K ascending through `bf16_gemv.cuh`'s accumulate loop with a single accumulator chain, so the arithmetic is the fallback's exactly and no reassociation is introduced. Kernel A/B cold at T=1, median of 20, against a baseline binary whose provenance is a symbol check rather than an mtime (no `bf16_latent_row_kernel`, five `bf16_generic_kernel`): **(1024, 4096) 401.4 → 47.1 µs (20.9 → 178.3 GB/s, 8.5x)**, **(4096, 1024) 104.4 → 18.4 µs (80.4 → 455.7 GB/s, 5.7x)**. Identity holds exactly at T = 1, 2, 8 and 1024 on both shapes (`max_abs 0`, `rel_l2 0`, 0 differing from 1024 to 4,194,304 elements), and the fixture was de-degenerated so that check *can* fail: the repo's patterned generators produced multiples of 2^-10 whose K = 4096 partial sums are exactly representable, so any association agreed; the new one draws full-8-bit-significand values over ~16 binades and a host control asserts a different chain order moves elements (2 of 5120 at T = 1, ~0.05% at T = 1024). End to end, with these two dispatch entries flipped and nothing else changed, the same tree reads **29.75 → 68.30 output tokens/s (+130%)** — which resolves the composition caveat on the 69.89 figure into ~21.9x from the gather route and the narrow Op, and 2.3x from these rows. Both items that were open in this row are now closed, and both by measurement rather than by argument. The `-p 128` reading that went the other way is **refuted**: a fingerprinted pair with identical hashes before both builds (`head=ade10381 sources_sha=be48d77042cc11d4 members=370`), three repetitions with a warmup and six reps per side across two harnesses, measures the route at **295.47 ± 2.49 t/s** against the fallback's **292.30 ± 4.09** — +1.1%, where the single unrepeated reading had said −7% — and the op level forbids a regression independently at prefill width in one binary (registered 131.07 µs against its unregistered `n-1` twin's 475.14 at T=128, i.e. 3.63x). The bandwidth asymmetry is **explained and shipped**: a 30-variant sweep over CTA 8 through 256 at five unroll factors found CTA 8 fastest at one token — 205.2 GB/s on the wider shape against 186.5 at CTA 32, and 512.6 against 456 on the narrower one, with `identity=exact` against the fallback in all thirty variants and all three runs, because CTA changes which block owns which row and never the order within a row — **and would have been a regression anywhere else**: at T=128 the wide route's block count is already 1024, so CTA 8 costs 137.22 → 184.32 µs (+34%) on the wider shape and 118.78 → 149.50 (+26%) on the narrower, with CTA 16 a wash. So `kThreadsPerCtaNarrow = 8` is selected only when `tokens == 1` and the multi-token route keeps CTA 32. That is the clearest case in this record of an experiment that would have shipped a regression if it had been read at one width. |
| Tokenizer | 47 fixture cases with 0 failures (boundaries, token ids and the decode round-trip, all exact), and full-corpus id agreement on the artifact's own `tokenizer.json` — 66,262/66,262 on a wikitext excerpt and 2,796,879/2,796,879 on a 10.78 MB mixed-source corpus. A 974,342-code-point Unicode sweep differs at 13 of 4.5M ids and is isolated to **table versions rather than logic**: utf8proc 17.0.0 classifies 4,699 code points as letters/marks/numbers that the reference's older Rust `regex` tables call unassigned (28 ranges, Tolong Siki and others), with zero cases in the opposite direction, which a validated Python mirror of the algorithm pinned piece-for-piece. |

One caveat the conversion reports in its own right, and belongs on the prefill work list rather than being
rediscovered: NVFP4 weights under an `A16Only` input policy — which is what a full-precision source produces —
warn that they run the decode-only path at prefill. That is a performance property of choosing W4A16 rather than
a defect, and the engine refuses `AllowA4` outright when no calibrated divisor exists.

Two Op contracts were wrong about their own payload order, and both are now reconciled — worth recording because
both were shape-valid to get wrong. `causal_conv1d_silu` documents `[C,4]` while its kernels read tap-major
(`j*C + c`), which cost the composition 1.4055 rel L2 when the binder followed the documentation; and
`nemotron_h_moe_router` writes its ids, weights and logits with the token innermost, while the shapes it declares
read as the transpose under this tree's axes convention. Each header now gives the flat index rather than a
shape alone, and names the checkpoint's own order as the transpose a binder must take. Where a wrong payload
genuinely cannot be detected the headers say so rather than implying a check exists: for the convolution the two
orders coincide at `C == 4`, and the router only ever writes those tensors, so nothing can inspect their order.

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

## Decode attribution (2026-10-16)

A single-token decode step costs 0.73 s (1.36 output tokens/s on the public route). nsys and ncu localize it
completely, and it is not where the design expected it.

| kernel | total | instances | avg |
|---|---|---|---|
| `nemotron_h_moe_experts_kernel<Nvfp4BankReader>` | 1,307.8 ms — **94.4%** | 80 | **16.35 ms** |
| `bf16_generic_kernel` | 40.4 ms — 2.9% | 160 | 253 µs |
| `expert_cache_fetch_kernel` | 24.6 ms — 1.8% | 56 | 439 µs |
| fp8 `gemv`, router, mamba scan, attention, norms | < 0.5% each | | |

80 instances is 40 MoE layers × 2 passes, so one MoE layer is 16.35 ms of a one-token pass. ncu on two
instances: **grid 1, block 128, waves per SM 0.00, achieved occupancy 8.34% (4 of 64 warps), SM throughput
0.43%, DRAM 0.06%, L2 0.04%, L1/TEX 51.13%, IPC 0.46, no-eligible 88.6%**. The launcher is
`<<<tokens, kThreads>>>` — one block per token — and the kernel loops `for (rank = 0; rank < top_k; ++rank)`
with `__syncthreads()` between each rank's up and down GEMV, so a one-token step runs ~37.7M MACs in 4 warps on
one SM, latency-bound on scalar 4-bit loads. The Op's work at T=1 is memory-bound — ~21.6 MB of routed expert
weights per layer, about 20 µs at DRAM rate — so the gap is ~1000×.

Four measurements, three of which corrected the first hypothesis that the host link was the wall:

- **The link is not the wall.** Pinned H2D `cudaMemcpyAsync` sustains 47.8 GB/s, and 42.6 GB/s at 600 KiB
  records; a kernel reading pinned memory sustains 22.7–29.2 GB/s (`profiles/nsys/nemotron_p0/h2d_probe.cu`).
  Staging is 17.44 GB per step, which is 23.8 GB/s effective — below the link, and still not the cost, for the
  next reason.
- **Gathering the routed experts buys no time while the Op is the wall.** With the pager's cache route at T=1 the step moves
  **632.8 MiB instead of 22,059 MiB (34.9×)** and its output is **bit-identical** to the in-place route
  (131072 logits, max_abs 0, rel_l2 0), yet end-to-end decode moved only 1.36 → 1.42 output tokens/s, and the
  residency bench reports 704.9 ms/pass against ~730 staged. The prefetch already hides the staging behind the
  Op — and that hiding is what makes the ordering below matter: with the Op at 654 ms it masks a 364 ms
  staging copy, and with the Op at ~2 ms the same copy is exposed, so a staged pass would then be
  copy-bound at ~364 ms (2.7 output tokens/s). The gather route and the Op fix are complements, not
  alternatives, and the staged-versus-gather choice must be re-measured once the Op is fast rather than
  settled by the numbers above.
- **CUDA Graph buys nothing at decode *in the pre-fix state***: 1.36 t/s with and 1.36 without. That is a
  property of a step whose cost was 94% one kernel, not of the graph — with the Op fixed the same pair
  reads 69.89 against 46.97, so the graph is worth 1.49x and the pager's capture-blocking
  `snapshot_residency` synchronize is worth revisiting (see the section below).
- The dose curve (`--expert-cache 0` in-place against `auto` staged) is 0.7 against 1.4 output tokens/s;
  in-place is the same Op plus ~744 ms of *unoverlapped* UVA reads, which is the one place bytes do show up.

The next wall is registered here before it is hit. `bf16_generic_kernel` is the fallback for "a problem no
registered shape matches" — the paragraph above — and both of this family's latent projections, (1024, 4096) and
(4096, 1024), land on it at 253 µs for 8 MB of weights (32 GB/s), with `bf16_gemv.cuh` sitting beside it.

That wall is closed. Both of those projections now have registered routes,
[`src/ops/linear/bf16/shapes/nemotron_h_latent.cu`](../../src/ops/linear/bf16/shapes/nemotron_h_latent.cu), and
the family's other BF16 widths are what still reach the fallback.

The route is deliberately not a `Bf16GemvSchedule`, and the reason is the accumulation order rather than the
speed. Every schedule beside it hands one row's K axis to the lanes of a warp and sums the lane partials with
`warp_reduce_sum`, which reassociates; this family's arithmetic is not free to be reassociated, because the
forward walk compares each stage element-wise against a dump that holds the dense fallback's own order — one
FP32 accumulator per output element, K ascending, one BF16 rounding at the store. That comparison is sensitive
to the order but does not fail on it: the walk's criterion for a stage is 0.08 relative L2, and the two orders
differ by ~2^-20 relative where a BF16 output ulp is 2^-8, which over a pass's 26.6M latent elements is ~1e-5.
So a reassociation would show up only as that stage's `rel_l2=0 max_abs=0` becoming nonzero, and the exactness
assertion itself lives in `tests/ops/linear/test_bf16_a16.cpp` (T = 1, 2, 8 and 1024 for both shapes, on
full-significand inputs so the assertion can fail). A warp-reduced row agrees with the reference almost
everywhere and differs wherever a partial sum lands the other side of a BF16 rounding boundary, so
the two are not interchangeable implementations. Instead each lane owns a whole output row and walks it in
ascending order through `accumulate_bf16_packs<..., 1>`, whose single-chain form is exactly the fallback's
`fmaf(w, x, acc)` chain; what the kernel takes from `bf16_gemv.cuh` is `Bf16GemvPack<16>` and its loaders, which
replace the fallback's one scalar BF16 load per FMA with a 32-byte sector per lane. That one lane per output
element is also the parallelism ceiling the order imposes — a K split is a partial sum — so the rows are the only
axis that can be widened, which is why the schedule is a block of one warp: at 1024 and 4096 rows it spreads one
warp per SM rather than concentrating them. `TokensPerThread` is then the per-token choice (1 at T=1, 4 beyond
it), sharing a weight pack across the batch without touching the order. Bit-identity with the fallback, at T=1,
2, 8 and 1024 for both shapes, is asserted in `tests/ops/linear/test_bf16_a16.cpp`.

Also recorded so nobody repeats the attempt: `nsys profile` cannot be used on this workload as it stands. It
stalls after the application's own report — 240 s and 1800 s observed, with `--stats=false` and graph tracing at
its default — and only a killed run's partial report was readable, which is what produced the table above.
`--cuda-graph-trace=none` is not a valid argument; `node` is what makes the post-processing explode.

## Decode after the fixes (2026-10-16)

The same recipe on the fixed tree — graphs disabled in the bench so the kernels trace, bounded with `--stats=false`, then `nsys stats` on the partial report — over a `-n 4` run with the in-domain corpus, five passes counted. Per-pass times are the totals divided by five:

| kernel | per pass | share |
|---|---|---|
| `expert_cache_fetch_kernel` (the gather) | **10.0 ms** | **47.6%** |
| `fp8_a16_gemv` (4096-wide) | 2.7 ms | 12.8% |
| `bf16_latent_row_kernel<1024,4096>` (new) | 1.8 ms | 8.4% |
| `narrow_down<32>` | 1.1 ms | 5.4% |
| `fp8_a16_gemv` (8192-wide) | 0.9 ms | 4.4% |
| `nemotron_router_select` | 0.8 ms | 3.7% |
| `narrow_up<64>` | 0.8 ms | 3.6% |
| everything else (nvfp4 gemvs, `bf16_latent_row<4096,1024>`, rmsnorm, router logits) | ~2.5 ms | ~12% |

- **The expert Op is fixed by measurement, not by assertion**: `narrow_up<64>` at 19 µs and `narrow_down<32>` at 28 µs per layer against the per-token kernel's 16.3 ms — about 350x per layer — so the layer's entire expert computation now costs less than one gather kernel, and the Op's share of GPU time falls from 94.4% to 9%.
- **The new first cost is the thing that feeds it.** The gather is 345 µs per instance and 29-30 instances per pass, i.e. 10.0 ms. At 632.8 MiB per pass that is about 63 GB/s, essentially this board's link rate, so it cannot be improved by making the copy faster — only by missing less, which is a cache-sizing question (`RoutedExpertPool`'s slot pool is `2 x max_bank_experts` and is not currently tunable) rather than a kernel one. **Steady state is materially better than that average**: the 10.0 ms is a five-pass blend that includes the one cold fill (261 of 261 assignments missing, 633 MiB, ~10 ms of traffic spread over four decode passes = +2.5 ms per pass), while the warm rate is 72 misses = 192 MiB = **3.0 ms per pass, 14% of the no-graph step**. So the step is now flat rather than dominated: 3.0 ms of gather, 3.6 ms of fp8 gemvs, 2.3 ms of latent shapes, 1.9 ms of expert kernels, ~2 ms of router and norms.
- **The registered latent shapes are 5.75x and 23x**: 44 µs for `(1024, 4096)` and 11 µs for `(4096, 1024)` against the dense fallback's 253 µs, measured on the real model rather than in isolation. That also settles a figure that looked impossible earlier in this record — the fallback really was ~20 ms of a pass — the error was comparing it against the fixed step's 14.3 ms instead of the 730 ms step it actually lived in.
- **The 69.89-versus-46.97 pair is not a graph difference, and this family has no graph to differ with.** `nemotron_engine_options` sets `use_cuda_graph = false` before the instance exists and the engine applies it on the path the benchmark uses; `grep -rni "cuda_graph|decode_graph" src/models/nemotron_h/` returns nothing at all, and the capture machinery (`src/core/decode_graph.*`) is included only by the Gemma 4 and Qwen3.5 programs. What the benchmark's `--no-cuda-graph` removes is its **own untimed warm pass**: `prime_decode_graph` runs a two-token `generate()` when the flag is on, and `resolve_max_context` then asks for 192 tokens instead of 4 — which is the `max_context=192` in the run headers above. So the pair measures the second decode pass against the first: the first pays the staged banks' first copy, the state's first touch and the handle/heuristic init, and that is worth 1.49x. **Ten arms in one session then settled it and refuted the graph attribution**: within one process and one command the two repetitions under the disabled flag read **42.44 and 90.80 t/s** — a 2.14x cold-to-warm difference with no cross-process or clock confound — while a *warmed eager* run reaches 86.4-88.0 and the primed arm sits between at 67.4-67.7, below both, because its prime is a two-token pass where `--warmup 1` runs a full four-token repetition. `max_context` is excluded directly: 192, 4 and 4096 move nothing (48.97 against 48.65 cold; 87.87 against 87.98 and 88.40 warm). So there is **no launch-and-gap term to recover** here — the flag's entire effect is the untimed warm pass it runs, the fastest arm is the one with no graph claimed at all, and the benchmark now prints the engine's *effective* value beside the requested one, verified against a family that keeps graphs on. One trap for anyone touching that benchmark: `--warmup` defaults to 1, so a bare `-n 4` run is warm rather than cold, and cold is the noisy state (42.4 against 48.7 on the same nominal configuration).
- **The three routes re-measured on the fixed Op settle the complementarity the earlier numbers could not.** At one column: narrow **14.5 ms/pass**, staged (width forced to 1) **451.1 ms/pass** with 22,374.0 MiB staged, in-place over UVA **60.6 ms/pass** — so the gather route is **31.1x** the staged one where the Op-bound measurement had it at 1.15x, and 4.2x the in-place one. The staged pass landed within 2% of the ~460 ms predicted for it in advance, at 49.6 GB/s effective against the 47.8 GB/s pinned-H2D ceiling measured independently: the copy is genuinely exposed now and running at the link, which is what the Op fix implied. Identity held across all three (0 of 131,072 differing) and the staged widths moved with the copy — 512 tokens now 534.0 ms against the recorded 3,068.0, and 4096 now 3,392.4 against 22,377.0 — so prefill amortizes against the Op rather than the staging.
- **The gather route's last open evidence item is closed.** At 2, 4 and 8 columns the narrow and in-place outputs agree word for word — 262,144 / 524,288 / 1,048,576 words, 0 differing, `max_abs 0`, `rel_l2 0` — with the failure path silent, so the pager's per-call slot protection and its rule that misses may not be re-read before the next resolve hold at the widths the engine actually runs (one lane to eight). One-column evidence could not see that, which is why it was asked for.
- **The wide-`T` prototype builds.** Its first real build through ninja (499/499, exit 0) is the news that its author's standalone verification holds in the tree, with all three archive gates passing after it (370/370 objects present, 278/278 older than the device link, 236/236/0 fatbinwraps). Its arithmetic is still unverified end to end, which is what the greedy-sentence gate and the plane test's wide cases are for, so it stays uncommitted until that window reports.

## Concurrency (2026-10-16)

**Decode does not scale.** One fresh server per point, a per-lane window so the attention state's capacity x lanes is constant, and the batch genuinely formed (`average_decode_batch` 8.00 at C=8, with the steady filter requiring `running == decode_ready == C` and `row_rounds == C x rounds`):

| C | window | decode tokens | per-request tok/s | aggregate tok/s | ms per round | x C=1 | TTFT mean / max |
|---|---|---|---|---|---|---|---|
| 1 | 262144 | 4095 | 44.3 | 44.3 | 22.6 | 1.00 | 47.4 / 47.4 s |
| 2 | 131072 | 8190 | 23.9 | 47.8 | 41.8 | 1.85 | 68.7 / 91.8 s |
| 4 | 65536 | 16380 | 10.7 | 42.9 | 93.1 | 4.12 | 118.1 / 190.6 s |
| 8 | 32768 | 32760 | 5.3 | **42.5** | 188.3 | 8.33 | 208.6 / 370.5 s |

- A round's cost tracks its row count within ±10%, so there is **no weight read to share**. The read per token is dominated by the routed experts, and a batch of L rows selects L x top-9 *distinct* experts per layer, so the traffic multiplies with L instead of amortising. The engine's one-compact-batch-per-round design is doing exactly what it says; what it cannot amortise is a mixture-of-experts read.
- C=8 therefore buys **0.96x the C=1 aggregate** and costs about 4.4x the first-token latency, because prefill serializes as well: every request prefills at ~195-204 t/s regardless of C and they queue, the last of eight waiting 4m39s.
- Nothing here is a lane limit. The batch forms, and the residency difference (4,080 -> 3,888 resident experts, 11.21 -> 10.42 GiB) is 4.7% of the resident set. The per-lane *state* grew 2.16 -> 3.27 GiB because the Mamba frontier is window-independent, which is why resident banks fall slightly at C=8 even though free VRAM rises.
- The implication is a property of the artifact and the card rather than of the engine: 37.21 GB of experts against ~11 GiB of residency means the majority of every routed read crosses the host link, and concurrency multiplies that instead of dividing it. **One of the two explanations is now answered from the existing record rather than by a new build**: the row-tiled kernels are already in the decode path at HEAD — the recorded `-n 4` profile carries `narrow_up<64>`, `narrow_down<32>` and `narrow_rank_sum<128>` at 200 instances each, because HEAD's gate admits them for `tokens <= 8` — so the wide route's widening to 4096 changes only the 9..4096 band and cannot account for any decode difference. A 68.30 against 90.29 t/s pair spanning that landing is a cross-harness comparison (different warmup, different context length) rather than a gain, and is recorded as such instead of as an unexplained one. What remains open is the second explanation: a per-row cost that is host-paged expert traffic rather than device-side, for which the C=8 intervals' 91-117% host exposure at batch 8.00 is what that looks like.

Not comparable with the figures above, and stated so nobody averages them: this C=1 rate (44.3 t/s) is a fresh process decoding from 9,226 tokens of context, while the 69.89 and 46.97 t/s figures are single-process one-token-context runs. The campaign's provenance carries the serve binary's sha256, the gates on the archive it was linked from, a tree note, and a greedy one-word probe on its own server that matched.

## Prefill attribution (2026-10-16)

A 512-token chunk runs at **110.10 t/s** and the same profiler localizes it to the same kernel decode used to have:

| kernel | share | instances | avg |
|---|---|---|---|
| `nemotron_h_moe_experts_kernel<Nvfp4BankReader>` (the wide per-token kernel) | **93.0%** | 40 | **69.6 ms** |
| `nvfp4_simt_kernel` (shared expert, both directions) | 2.5% | 1280 | ~58 µs |
| `expert_cache_fetch_kernel` | 0.9% | 58 | 443 µs |
| `fp8_a16_mma_kernel` (4096 and 8192 widths) | 1.0% | 112 | ~250 µs |
| `bf16_latent_row_kernel` (both, 40 each) | 0.9% | 80 | 394 / 287 µs |
| `router_logits`, the Mamba scan | 0.9% | | |

- Forty instances for a 512-token pass means **one launch per layer of 512 blocks**, and 69.6 ms per launch is about 23 ms per block across three waves — the same latency-bound per-token block that cost 16.3 ms at one token, multiplied by 512 rather than made dearer.
- The pass's own decomposition agrees: of 4.65 s, the staged bank copy is only ~0.46 s (22,059 MiB at 47.8 GB/s) and this Op is 2.78 s. Prefill is neither staging-bound nor compute-bound; it is 512 copies of a latency-bound block.
- What is *not* the problem, and is worth stating because three of these were plausible suspects: the shared expert's SIMT kernels (2.5%), the fp8 MMA kernels (1.0%), the registered latent shapes (0.9%), the router and the Mamba scan (0.9% together). The latent shapes also scale correctly at this width — 394 µs and 287 µs per instance at 512 tokens against 47 µs and 18.4 µs at one, i.e. 0.77 and 0.56 µs per token, proportional to work rather than to block count.
- The payoff of extending the row-tiled decomposition to wide calls was bounded by bank traffic rather than by blocks — of order **1000 t/s against the 110.10 measured here** — and the first implementation of it is now in the tree (uncommitted at the time of writing). A 512-token-equivalent prompt on that tree measures **295.3–297.7 t/s** (`-p 128 -n 1 -r 3 --warmup 1`, two harnesses, six repetitions per side), i.e. **2.7x the 110.10 above**, with the pair's fingerprints identical before both builds (`head=ade10381 sources_sha=be48d77042cc11d4 members=370`), both binaries postdating every non-dispatch source, and both archive gates green. So the bound is confirmed in direction and the implementation sits at ~30% of it — the honest reading of a first cut rather than a disappointment, and the reason the row above stays as the measured baseline it was.
- **The design question that bullet left open is settled, and by the bound rather than by the profile: the token axis is tiled inside the Op.** The row-tiled route now walks a call's tokens in tiles of 64 and re-uses one tile's buffers for every tile, so the per-(token, rank) activation product and raw down partial are a fixed 7.88 MiB at the widest geometry this artifact carries (1024x2560, top_k 9) where the whole 4096-token chunk's would be 504.0 MiB — against a residency budget whose unit is a whole bank of ~550 MiB, so the bound costs nothing measurable while the unbounded form would have cost a resident bank. The alternative candidate (an intermediate that never leaves shared memory) is not needed for the bound and would have put a second, block-local staging of `intermediate` on the critical path of every down row tile. The workspace query is capped by that tile and the launch takes the cap from the same function (`row_tiled_covered_tokens`), so an arena sized at the pass width covers every narrower call, decode lanes included; the route takes every NVFP4 plane-addressed call up to this family's widest chunk (4096) and leaves the per-token route above it, which is also what keeps that route reachable through the public entry point.
- **The measurement is in and the bound was right, with the copy now the cost.** With the row-tiled route taking the prefill this command reads **262.28 t/s** cold against the baseline's 110.10, and **953.27 t/s** with one warm pass — a 537 ms chunk pass of which the staged bank copy is 461 ms at 47.8 GB/s, **86%**: prefill is staging-bound at the link, which is the state the bullet above predicted would be exposed once the Op stopped hiding the copy. The Op is **214 ms** of that pass — 5.35 ms per layer against 69.6 ms — at or below the 290 ms bank-traffic roofline the geometry sets, so the decomposition is at its floor rather than merely faster. The same-tree A/B (the pre-change gate forced back in, one variable) reads 116.70 t/s against 262.28, and the identity is exact at 64, 512 and 4096 — see the Wide-route Op row for the per-instance times, the probes and the two defects the wide identity case exists to catch.
- **A kernel-level profile of the same path, for the decomposition those numbers imply rather than show.** On a cold graphs-off run with the profiler attached — so the absolute rate is 232.71 t/s against the warm 953.27 above, a different regime rather than a discrepancy — the pass reads: row-tiled expert kernels **55.8%** (down 33.4% at 397.5 µs per instance, up 22.4% at 266.6), the shared expert's SIMT pair **15.7%** where it was 2.5% while the per-token Op dominated, the registered latent shapes 5.9%, the gather 5.5%, the fp8 MMA kernels 4.9%, the router's logits 3.8% and the Mamba scan 1.9%. The transfer summary counts 1,018.8 ms of Host-to-Device memcpy across 1,249 calls (22,374 MiB, ~23 GB/s). The single kernel that was 93% of this pass is gone, and what replaced it is spread rather than concentrated.
- **And the day's three findings are one constraint, not three.** Decode's warmed gather is ~3.0 ms of a ~14 ms step; prefill's staged copy is 461 ms of a 537 ms pass, i.e. 86%; and at C=8 the same routed reads multiply with rows instead of amortising. So routed-expert traffic across the host link is not one phase's cost but the artifact's, and the lever in all three cases is residency — a question about whether the model fits, not about any kernel, which makes it a product decision about quantization or hardware rather than another optimization pass.
- **The per-instance times that the bullet above pointed at are now measured, and the profiler costs 4% in this regime.** Warm and profiled, prefill reads 913.32 t/s against 953.27 unprofiled, and the pass decomposes as row-tiled expert kernels **53.7%** (down 31.6% at 324.0 µs per instance across 920 launches, up 22.1% at 227.1), the shared expert's SIMT pair 15.4% (7.7 and 7.7 at 56.9 µs), the gather 6.2% at 301.2 µs, the fp8 MMA kernels 4.3% at 281.9 µs, the router's logits 3.8% at 98.7 µs and the latent shapes 3.4% at 396.4 µs. The transfer summary counts **1,486.9 ms of Host-to-Device memcpy in 1,486 calls** averaging 1.0 ms — against a 0.56 s pass, so the copies overlap the kernels by roughly 2.7x in total while the *staged* portion still occupies 86% of the critical path: at this width the pager's double-buffered prefetch hides part of the fill, not enough of it. Every per-instance time here sits about 20% below its cold-regime counterpart, which is the other half of the cold-noisy, warm-stable pattern the benchmark's own repetitions show (42.4 against 48.7 cold on one configuration, against sub-1% spread warm).

## Slot cache versus bank residency (2026-10-16)

A decode step's ~3.0 ms warm gather exists because 27 GB of experts live on the host, so the lever would be to miss less — and there are two ways to spend device bytes on that: whole **resident banks** (the split's current form) or **slot-cache slots** in the pool the gather fills. The pool was sized structurally at `2 * max_bank_experts` (992 slots, 2.79 GB) and had never been traded against residency, so it was made a function of the budget and swept. **The answer is no, at every setting**, and the two mechanisms are measured rather than inferred:

| durable slots | pool bytes | banks | resident experts | warm misses per assignment | gather ms/pass | `-n 4` decode |
|---|---|---|---|---|---|---|
| **0 (structural)** | 2,790 MiB | **13** | **5,808** | 58 / 243 (23.9%) | 4.17 | **71.7 ± 0.5** |
| 992 | 5,580 MiB | 11 | 4,752 | 54 / 261 (20.7%) | 3.96 | 69.9 ± 0.2 |
| 1984 | 8,370 MiB | 8 | 3,648 | 63 / 288 (21.9%) | 4.42 | 68.4 ± 0.9 |
| 2976 | 11,160 MiB | 6 | 2,720 | 66 / 306 (21.6%) | 5.41 | 66.3 ± 0.3 |

- **Capacity**: a slot is sized at the *widest* bank's per-expert stride (2,949,120 B) while a resident expert is sized at its own layer's width (2,359,296–2,949,120 B), so the same bytes buy 0–25% more experts as banks than as slots.
- **The reuse is not cache-size-limited**: the miss *fraction* holds at 21–24% across a 4x cache range and the warm gathered bytes fall only 6% (141.6 → 133.0 MiB) before *rising* to 163.8, so extra slots hold experts the next token does not route to. The first 992 slots cost two banks and 1,056 experts of residency for that 6%, and a cold command pays it immediately because every non-resident bank must be filled once. Steady-state, interleaved at `-r 3 --warmup 1`, all four arms sit within ±1.5% of each other (90.5 / 91.1 / 90.0 / 89.7 t/s) against 5% load-to-load drift.
- Identity held in all five configurations including the pre-fix control: 1 column 131,072 logits, 0 differing; 2/4/8 columns 262,144 / 524,288 / 1,048,576 words, 0 differing, `max_abs 0`, `rel_l2 0`.

**And the exercise found a real defect, which is the durable outcome.** The pool was reserved *twice* — `reserve_bytes` is `Program::device_bytes`, which already includes it, and the plan then subtracted it again as staging — leaving 2,790 of 29,178 MiB unowned, **8.7% of the card**. With `Program::expert_pool_device_bytes` subtracted at both request sites the default split moves from 11 of 40 banks to **13**, from 4,752 to **5,808** resident experts (34.375% → 42.0%), from 12.8672 to **15.4775 GiB** resident and pinned from 21.7837 to **19.1733 GiB** — 22% more residency at the same card, which is precisely the lever every other finding in this document points at. The pre-fix arm reproduces the recorded figures to the digit, which is what identifies the change as the double-count removal rather than a tuning effect.

**Read the residency figures elsewhere in this document as pre-fix.** Every one of them — the 4,080 of 13,824 in the concurrency section, the 11 of 40 banks and 12.87 GiB in the prefill rows, the 11 of 40 and 13.11 GiB in the narrow-route row — was taken with the double count in. They are correct for what they measured and understated relative to the current tree by the amounts above.
- **And that fix's end-to-end effect is measured, phase-dependent, and explained.** One binary, one build, two budgets (`--expert-cache auto` against the recorded 16,012 MiB), fingerprints byte-identical before and after, gates green, the greedy sentence identical on both arms, and the B arm's ledger reproducing the recorded figures to the digit — which is what makes it a comparator rather than a differently-loaded card. Result: **+10.8% warm prefill (1083.49 against 978.18 t/s), +2.9% cold decode, +0.8% warm decode, and nothing at a 4096-token chunk** (3311.3 against 3309.5 ms). The mechanism is that the rate followed the *staged traffic*, not the residency percentage: the larger budget unloads two more banks (29 → 27 staged, 22,306.5 → 20,322.0 MiB per pass, −8.9%), and at 512 tokens the pass *is* the copy — 20.3 GiB at 44.1 GB/s inside 483 ms — so the win tracks that (−7.6% pass time) and vanishes where the Op dominates. An in-band check that the arms genuinely differ: the throughput bench uploaded 23.38 GiB of device weights in A against 20.38 in B.

## Route-dependent per-token numerics (2026-10-16)

A stateful decode and a one-pass prefill of the same sequence do **not** agree bit for bit, and the
reason is route selection rather than any op being wrong. Measured by capturing the draft's six tap
residuals along both paths on the real artifact (128 tokens generated from a 512-token prompt of
`docs/serving.md`, one Program, one configuration):

| | |
|---|---|
| First difference | position 512, the first column a *decode* pass processes |
| Block 0 (a Mamba block) at that column | max abs 0.00195 = 2^-9, rel L2 0.0064 |
| Block 85, later in the state | max abs 30, rel L2 0.0689 |
| Prompt region (512 wide in both arms) | bit-identical, every tap, every column |
| Two captures of the same pass | bit-identical (the capture itself is deterministic) |

The mechanism: a one-column call reaches the narrow expert route and a wide one the row-tiled route,
and each format's GEMV and MMA reduce K in a different order -- which the ops' own contracts allow
("RMS reduction order, Tensor Core operand staging, projection factorization, split-K reduction, and
private intermediate precision are implementation choices"). A mixture-of-experts router then turns
that last bit into a *discrete* expert-selection difference: once one token's expert set moves, the
Mamba state carries it for the rest of the sequence, which is why block 0 stays at 2^-9 while block
85 reaches order 1. Router near-ties are not rare here -- this document already records the tiny
model's 2-of-130 case.

Two consequences, both load-bearing:

- **The Decode row's bit-for-bit claim holds for equal routes only.** It was measured when a
  multi-token pass and a one-token step took the same kernels; after the wide row-tiled route landed,
  a 512-token pass does not take the one-token route, and this is the end-to-end evidence item that
  the wide route's own note ("its arithmetic is still unverified end to end") was waiting for. The
  check itself remains right and non-vacuous -- the two-chunk comparison compares equal widths, and
  the capture test now asserts exactly where equal widths stop agreeing.
- **A speculative verify pass computes its columns at a different width than the decode it replaces**,
  so near a router tie its decisions can differ from a sequential decode's by one expert's worth.
  That is not a defect of either path, and it is the reason acceptance has to be measured on this
  pair rather than assumed from the drafter's card.

## Artifact sizes, and two numbers that look wrong but are not (2026-10-16)

A reader who compares the file against the residency figures will hit the same two apparent contradictions this record did, so both are resolved here.

- **The file is 32,000,000,000 bytes exactly** (29.80 GiB, not sparse — `du` agrees), with a 4,554,012-byte conversion report. Its *content* is **45.276 GB (42.17 GiB)**: that is the converter's own accounting of its tensors, and it matches the loader's `read 42.17 GiB` field exactly — which means that field is the content size, not an artifact of reading an mmapped file more than once, as this record previously assumed.
- **And the recorded bank figure is consistent with both.** 37.21 GB of routed expert banks plus ≈8.07 GB of dense weights is 45.28 GB, the same content the converter accounts for, so nothing in the residency arithmetic is wrong. The file is simply smaller than the sum of its tensors — a 0.71 ratio — because the container is framed and compressed, which is also why `rows_per_chunk` appears in the report.
- Consequence for reading the rest of this document: cite the *content* (45.28 GB, or 37.21 GB of banks) when arguing about what must fit, and the *file* (32.0 GB) only when talking about disk or transfer. They differ by a factor this section names.
- Also present in `out/`: a stray `nemotron_h_74b_nvfp4.ninfer.part-0001` of 13,282,126,340 bytes with no siblings, written at the same minute as the report. It is not the artifact and not this repo's publish split (which cuts at exactly 32,000,000,000 bytes), so it is either an interrupted publish or an orphan and should be identified before any upload.

## KV format, long context, and what a gibibyte is worth (2026-10-16)

**This family has exactly one KV format, and the benchmark advertises four it cannot run.** `bf16` is accepted; `int8`, `fp8`, `nvfp4` and `k8v4` are refused before the weights are read, from `nemotron_engine_options()` (the shared entry every app takes) and again from `state_geometry()` for a directly-constructed Program, with the same wording in the bench, the CLI and `ninfer-perplexity`. It is a format-level refusal rather than a size one: no context length works, including 192 tokens, so there is no "largest context that did work". **The flag is advertised where the refusal is not stated, and the surface is the benchmark's own help line rather than a document.** `bench/inference/ninfer_bench_support.cpp:319` prints `--kv-dtype <bf16|int8|fp8|nvfp4|k8v4>  KV cache storage (default: bf16)`, and nothing under `docs/` mentions the flag at all — checked, because an earlier reading of this claimed `docs/kv-and-context.md` advertised it and that is not so. So the honest description is a help line that lists four formats this family refuses at startup, which is a one-line code fix needing a build rather than a documentation gap, and the refusal itself is loud at runtime.

**The sequence arena, isolated from the context.** Every arm and format reports `kv_payload_bytes = 0` — the paged-KV instrument does not describe this family — so the K/V bytes live inside `sequence.capacity_bytes`, which is exactly **8 KiB per token per lane plus a ~163 MiB floor** (the Mamba frontier and convolution state): 0.160 GiB at 192 tokens, 0.223 at 8384, 2.160 at 262144. Decode at an 8192-token context reads **56.98 t/s** while the *same* context under `--max-ctx 262144` reads **46.88** — so the long-context gap is mostly the *arena*, not the sequence length, which is why the concurrency campaign's 44.3 t/s at 9,226 tokens and the 75 t/s at short context are not a single effect. A note for anyone repeating this: `-p 8192 -n 32` is two separate tests (a prefill and a short-context decode, because `tg` seeds with one token), and decode *at* a context requires `--prompt-gen`.

**And the price of a gibibyte, which outlives the format question.** Driven through the residency bench's own planner: 8192 tokens reserves 2,722 MiB and leaves a 18,008 MiB expert budget for 13 of 40 banks; 262,144 reserves 4,707 MiB and leaves 16,063 for 11 of 40. That is **+413 resident expert slots per GiB of arena returned, 3.0% of the 13,824-slot pool**, and by the double-count A/B's own slope (two banks ≈ 2,364 slots ≈ +2.9% cold decode and +10.8% warm prefill) a gibibyte is worth roughly 0.5% decode and 2% prefill. So an int8 KV, which would have halved a 2.0 GiB plane to 1.16 GiB with its floor, was worth about one to two percent — the refusal costs nothing. The `auto+1 GiB` arm clamps because at that window `auto` already spends what the card has, which is the same statement from the other side: the only way to give the cache a gibibyte is to take it from the arena, exactly as a smaller KV would.
