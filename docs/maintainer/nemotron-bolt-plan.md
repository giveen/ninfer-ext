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
| attention | 32 heads, 2 KV heads, `head_dim` 128, no bias | plain GQA, RoPE, no sinks or window |
| mamba | 128 heads × head_dim 64 = 8192 inner, `conv_kernel` 4, `n_groups` 8, `ssm_state_size` 128, `chunk_size` 128 | `conv_dim = 8192 + 2·8·128 = 10240`, `in_proj` out = `2·8192 + 10240 + 128 = 18560` |
| `use_conv_bias` | **true** | the depthwise conv has a bias even though `use_bias` is false |
| `time_step_min` / `_max` | 0.001 / 0.1 | **only the minimum clamps**; the reference's maximum clamp is commented out and `time_step_limit` is unused on the torch path |
| MoE | top-9, `moe_latent_size` 1024, 1 shared expert (5376), `norm_topk_prob`, `routed_scaling_factor` 5.0 | **each layer has its own expert bank**: 240 experts in 12 layers, 304 in 6, 368 in 9, 432 in 10, 496 in 3. The config's `num_experts`/`num_local_experts` of 512 is a bound, not a shared bank — llama.cpp reads the real count from each layer's router width and validates it against 512. Per-layer `moe_intermediate_size` is 1536/2048/2304/2560. |
| router grouping | `n_group = 1`, `topk_group = 1` | degenerate: the group mask selects everything, so top-k runs over the layer's whole expert bank. **`n_groups = 8` is Mamba's B/C grouping and has nothing to do with this.** |
| `hidden_act` / `mlp_hidden_act` | `relu2` | squared ReLU, for experts and the shared expert |
| norms | `layer_norm_epsilon` 1e-5 | |
| `rescale_prenorm_residual` | true | **initialisation only** (`_init_weights`, the GPT-2/Megatron 1/√N scheme). The checkpoint carries it; there is no runtime residual rule to implement. |

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

**Attention** (`:860`). Plain GQA with `rotate_half` RoPE, causal, no bias, no sinks, no window.

## What this costs

Nothing here is reused from the existing families. New: a selective-scan SSM op with an fp32 state
cache, the gated grouped norm, `relu2`, a non-gated expert bank, the latent projections, the
sigmoid/group router, and a config whose layer types and expert shapes are **per layer**. Reusable:
the linear-attention/recurrent-state machinery (GDN is the same family of problem, not the same
mathematics), MoE routing infrastructure, GQA attention, KV paging, and the host-offload expert cache
that Flash-Next already uses for a model whose weights exceed the device.

Memory: NVIDIA's own 24GB variant demonstrates the target — 74B at a mixed NVFP4 group-16 (mixer and
expert matrices) plus Q8 (head) footprint of **24.12 GB**, which is the shape of this repository's
NVFP4 and Q8 routes. Weights then fit a 5090; KV at 256K context is what the paged/host cache is for.

## Milestones

1. **Oracle.** A reference harness that instantiates the canonical implementation on tiny shapes and
   dumps per-stage tensors, then the same at real shapes. Nothing below is qualified without it.
2. **Config and artifact.** The hybrid block list, per-layer expert counts and per-layer expert widths;
   the layout question is whether the existing artifact/binding code expresses per-layer variation.
3. **Ops, one at a time, each against the oracle**: RMSNorm + `relu2` → Mamba-2 → router + experts →
   latent round trip → attention → the block → the whole 88-block forward.
4. **Converter** from the BF16 checkpoint, then the Engine path and the decode loop.

## Open items

- The draft pairs with the **24GB NVFP4** target, not the BF16 one, so its qualification belongs on
  our NVFP4 route. The cheap decisive check is ranking the draft's token inside the target's own
  next-token distribution — the diagnostic that exposed the Gemma 4 QAT mismatch.
- `e_score_correction_bias` is a buffer in the reference and must be confirmed against the released
  checkpoint's tensors.
- Vision (RADIO/InternVL-style tower, projector 20480, video) and MTP (`num_nextn_predict_layers` is 0
  in this release) are out of phase 1.
