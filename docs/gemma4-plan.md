# Gemma 4 31B in NInfer — plan

- **Status:** proposal, not started. Written 2026-10-09. Nothing described here is implemented yet.
- **Target:** `google/gemma-4-31B-it` (`Gemma4ForConditionalGeneration`, text `gemma4_text`) on one
  RTX 5090 (32 GB, sm_120a, PCIe 5.0 x16), host with 256 GB RAM (the model does not need it).
- **Weights:** NVFP4 MLPs produced by NVIDIA Model Optimizer (ModelOpt), imported unchanged; every
  other projection encoded by the NInfer converter; precision then tuned per tensor with NInfer's own
  KLD measurement.
- **Why this is "first of a kind":** every architecture NInfer runs today (`Qwen3_5ForCausalLM`,
  `Qwen3_5MoeForCausalLM`, `Qwen4ExpForCausalLM`) lives in `src/models/qwen3_5/` and shares one
  Program, one tokenizer and one tool-call grammar. Gemma 4 is the first non-Qwen family: a new model
  directory, a new tokenizer, a new chat/tool protocol, a pure-attention state model (no GDN), and
  attention geometries no current Op registers.

---

## 0. Summary

| Question | Answer |
|---|---|
| Does it fit? | Yes, with re-encoding. NVIDIA's NVFP4 checkpoint keeps attention, embeddings and vision in BF16: **29.3 GiB of text weights**, which leaves no KV room on a 32 GB card. MLP NVFP4 + attention FP8 + embedding FP8 is **20.1 GiB**; all-NVFP4 projections is **16.7 GiB** (section 5). |
| Fastest path to a first artifact | Import the MLP NVFP4 codes, scales and activation scales from `nvidia/Gemma-4-31B-IT-NVFP4` (lossless, `import_encoded`), and encode attention and embeddings with the existing `fp8_row_maxabs` method. No ModelOpt run is needed for the first artifact. |
| Where ModelOpt runs ourselves | Tuning: our own `hf_ptq.py` run on BF16 `google/gemma-4-31B-it` with a better weight algorithm (MSE / local-Hessian / AWQ-lite), chat-format calibration, and NVFP4 on `o_proj` (`nvfp4_omlp_only`), or an AutoQuantize NVFP4/FP8 search (section 6.3). |
| Decode ceiling (estimate, not measured) | About 83 tok/s single-request at short context for the 20.1 GiB layout, about 100 tok/s for all-NVFP4, from weight bytes per token at 1.79 TB/s. Long context lowers it: at 256K tokens the global KV read adds about 30% per token (section 5.3). MTP via the official assistant drafter is the main multiplier. |
| Biggest engineering items | (1) the `src/models/gemma4/` model + Program, (2) four attention routes (sliding D256 ring, global D512 compact K/V, both causal with scale 1, image-block bidirectional overlay), (3) the Gemma tokenizer and tool/thinking protocol, (4) per-shape linear instances for H=5376. |
| Decisions needed before P1 | D1 Program structure, D2 instruction vs base model, D3 first precision layout, D4 KV cache format. See section 15. |

---

## 1. Deliverable and completion conditions

The deliverable is a `Gemma4ForCausalLM` (text) execution path through the public Engine, with:

1. A converter recipe that produces a `.ninfer` from the NVIDIA NVFP4 checkpoint (and, later, from a
   ModelOpt export we produce ourselves), and a model card.
2. Text generation and CausalScoring through `ninfer`, `ninfer-serve` (OpenAI and Anthropic
   protocols, including Gemma tool calls and the thinking channel), `ninfer-perplexity` and `ninfer_bench`.
3. Prefix reuse with the Gemma 4 state model (global paged KV + sliding-window rings).
4. Quality evidence: KLD and PPL against a BF16 reference, plus the same eval set the Qwen model
   cards publish, compared with NVIDIA's own BF16/NVFP4 table where the benchmarks overlap.
5. Performance evidence under `docs/performance/methodology.md`.

Later phases (not required for the first completion): Vision, the MTP assistant drafter, precision
tuning beyond the first layout, and an EXL3 comparison.

Out of scope: audio (the 31B config has `audio_config: null`), video input, the 26B-A4B MoE sibling
(possible follow-up: section 14), the E2B/E4B per-layer-embedding variants, and multi-GPU.

---

## 2. Sources

Pinned at the HEAD of each repository on 2026-10-09.

| Source | Pin | License | What we take from it |
|---|---|---|---|
| Transformers `models/gemma4/modeling_gemma4.py`, `configuration_gemma4.py`, `modeling_rope_utils.py` | `90ef040d40` | Apache-2.0 | **Mathematical authority** for text, vision, masks and proportional RoPE; the FP64 oracle is checked against it. |
| Transformers `models/gemma4_assistant/modeling_gemma4_assistant.py` | `90ef040d40` | Apache-2.0 | Assistant (MTP) drafter mathematics. |
| [`nvidia/Gemma-4-31B-IT-NVFP4`](https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4) | HF main | Apache-2.0 (card links Gemma's Apache terms) | NVFP4 MLP codes, E4M3 block scales, FP32 global scales and activation scales; NVIDIA's accuracy table vs BF16. |
| [`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it) | HF main | Apache-2.0 | BF16 weights (re-encoding source, BF16 reference), `tokenizer.json`, `chat_template.jinja`, `generation_config.json`. |
| [`google/gemma-4-31B-it-assistant`](https://huggingface.co/google/gemma-4-31B-it-assistant) | HF main | Apache-2.0 | MTP drafter weights and config. |
| NVIDIA Model Optimizer `examples/hf_ptq/hf_ptq.py`, `modelopt/torch/models/gemma4/specs.py` | `6e6cadcbaf` | Apache-2.0 | NVFP4 PTQ production (`gemma4` is registered, `min_transformers_version="5.5"`); qformat presets, AutoQuantize. |
| vLLM `model_executor/models/gemma4.py`, `gemma4_mm.py`, `gemma4_mtp.py` | `0cc0460870` | Apache-2.0 | Production cross-check: ModelOpt NVFP4 loading, KV sharing for the drafter, how drafter inputs are wired (embedding × √5376 ‖ backbone hidden). |
| llama.cpp `src/models/gemma4.cpp`, `src/models/gemma4-assistant.cpp`, `tools/mtmd/models/gemma4v.cpp` | `609290be6b` | MIT | Independent re-implementation: confirms attention scale 1.0, V = parameter-free RMSNorm, K reused as V on global layers, no √H scaling for image embeddings. Also on the local machine at `/mnt/storage/llama.cpp`. |
| [gewell](https://github.com/leDissolution/gewell) `docs/models.md`, `docs/cache.md` | `97725bfc89` | Apache-2.0 | The closest prior art: a C++ single-GPU sm_120a Gemma 4 engine. Its **G0 mixed-precision layout**, a bundled 410-projection activation calibration, the compact global KV representation, and sliding-window checkpoint design. |
| exllamav3 `architecture/gemma4.py` | `151539c77a` | MIT | Optional EXL3 quality comparator (P7). |
| SGLang `srt/models/gemma4_causal.py` | `59eb71a831` | Apache-2.0 | Third cross-check, if the first two disagree. |
| NInfer `docs/exl3-quant-plan.md` §9, `apps/sensitivity`, `ninfer-perplexity` | in tree | — | Reference-KLD design and the sensitivity tool reused for precision tuning. |

**Precedence:** Transformers defines the mathematics. When vLLM or llama.cpp disagree with it, the
disagreement is investigated and recorded in the model reference, never resolved silently.

Two inconsistencies were found while writing this plan:
- NVIDIA's model card says ModelOpt v0.42.0, while the checkpoint's `hf_quant_config.json` says
  producer `modelopt` `0.37.0`. This affects provenance only.
- The card says "kv_cache_quant_algo: FP8", but the checkpoint stores no `k_scale`/`v_scale` tensors.
  NInfer's KV codec decides its own scales anyway (section 9).

---

## 3. Model mathematics (text)

This section is the draft of the future `docs/maintainer/gemma4-model.md`. Every statement was read
from the Transformers source at the pin above and must be confirmed by the FP64 oracle in P0.

### 3.1 Configuration (`gemma-4-31B-it`)

| Quantity | Value |
|---|---:|
| Hidden H / MLP intermediate I | 5376 / 21504 |
| Layers | 60: 50 sliding + 10 global. `layer_types` puts global at indices 5, 11, …, 59 (every sixth; the last layer is forced global). |
| Query heads | 32 in every layer |
| Sliding layers: KV heads / head dim / window | 16 / 256 / 1024 |
| Global layers: KV heads / head dim | 4 / 512, `attention_k_eq_v = true` (no `v_proj`) |
| RoPE, sliding | default, θ = 10,000, full head dim 256 |
| RoPE, global | `proportional`, θ = 1,000,000, `partial_rotary_factor` 0.25 |
| Activation | `gelu_pytorch_tanh` (GeGLU MLP) |
| RMSNorm ε | 1e-6 |
| Vocabulary | 262,144, embeddings tied to the LM head |
| Final logit soft-cap | 30.0 |
| Context | `max_position_embeddings` 262,144 |
| Not used in 31B | MoE block, per-layer inputs (PLE), KV-shared layers, double-wide MLP, audio |

Attention parameter count: 50 × 132.1M (sliding) + 10 × 187.2M (global) = 8.48B. MLP: 60 × 346.8M
= 20.81B. Embedding: 1.41B. Vision tower and projector: about 0.58B.

### 3.2 Norms

`rmsnorm(x, w) = x · (mean(x²) + ε)^(-1/2) · w`, computed in FP32. The weight is used as is. This is
**not** the Gemma 2/3 or Qwen `(1 + w)` offset, so the existing `ninfer::ops::rmsnorm` is used
with `unit_offset = false`. ModelOpt's Gemma 4 spec also deliberately omits a +1 handling.

`v_norm` and the vision projector's pre-norm are the same formula without a weight.

### 3.3 Decoder layer

```text
h  = embed(token) · √H                              # √5376 applied in the embedding's dtype
for layer ℓ:
    a  = attn_ℓ(rmsnorm(h, w_in))
    h  = h + rmsnorm(a, w_post_attn)                # sandwich norm: the norm is on the branch output
    m  = W_down · (gelu_tanh(W_gate · x) ⊙ (W_up · x)),  x = rmsnorm(h, w_pre_ff)
    h  = h + rmsnorm(m, w_post_ff)
    h  = h · s_ℓ                                    # layer_scalar, one BF16 per layer
logits = 30 · tanh((W_embᵀ · rmsnorm(h, w_final)) / 30)
```

`layer_scalar` is not a no-op. Read from the NVIDIA checkpoint: layer 0 = 0.0894, layer 1 = 0.0654,
layer 59 = 0.0364, the others 0.44–0.99. It scales the entire residual stream, so it cannot be folded
into a projection weight. It is fused into the residual-update epilogue instead (section 8).

Both post-norms reduce over the full output row, so the norm cannot live in a column-tiled GEMM
epilogue. The planned fused Op is `post-norm + residual add + layer scalar + next pre-norm`
(section 8, row 6).

### 3.4 Sliding-window attention (50 layers)

```text
q = rope₁₀ₖ(rmsnorm(W_q x, w_qn))          # [32 heads, 256]
k = rope₁₀ₖ(rmsnorm(W_k x, w_kn))          # [16 heads, 256]
v = rmsnorm(W_v x)                          # [16 heads, 256], no weight
o = W_o · softmax(q·kᵀ · 1.0 + mask) · v    # scale is exactly 1.0
```

- RoPE is `rotate_half` over the full 256 dims: pair (j, j+128), inv_freq_j = 10000^(-2j/256).
- Query head h reads KV head ⌊h/2⌋.
- **Mask:** key position p_k is visible to query position p_q when 0 ≤ p_q − p_k < 1024 (causal
  window, self included). With images, the sliding mask is `AND(window, OR(causal, same_image_block))`:
  tokens of one image block see each other bidirectionally. The window bound is one-sided (only
  `p_k > p_q − 1024`), so future keys in the same image block are visible. P0 confirms this against
  `create_masks_for_vision_model` and `sliding_window_overlay`.

### 3.5 Global attention (10 layers): K = V and proportional RoPE

```text
r = W_k x                                   # [4 heads, 512]; there is no W_v
n = r · (mean(r²) + ε)^(-1/2)               # shared normalization
v = n                                       # v_norm has no weight
k = rope_prop(n ⊙ w_kn)
q = rope_prop(rmsnorm(W_q x, w_qn))         # [32 heads, 512]
o = W_o · softmax(q·kᵀ · 1.0 + causal) · v  # query head h reads KV head ⌊h/8⌋
```

**Proportional RoPE** (`_compute_proportional_rope_parameters`): with head dim D = 512 and factor
0.25, `rope_angles = 64`. inv_freq_j = 10⁶^(−2j/512) for j < 64 (the exponent's denominator is
the **full** head dim, not 128), and 0 for j = 64..255. `rotate_half` pairs (j, j+256).

So only dims {0..63} ∪ {256..319} rotate. Note the difference from Qwen's partial RoPE, which rotates
a contiguous leading block with frequencies normalized to the rotary dim. An existing RoPE kernel
must not be reused without checking this.

Global layers have no bidirectional image overlay; they are always causal.

**Compact global KV (exact reformulation).** On the 384 non-rotated dims, `k[d] = v[d]·w_kn[d]`.
Therefore:

```text
q·k = Σ_{d∈rot} q[d]·k[d] + Σ_{d∉rot} (q[d]·w_kn[d])·v[d]
```

The cache stores, per token and KV head, only `v` (512) and the 128 rotated `k` dims: 640 values
instead of 1024, which is **37.5% less global KV** and lets one load of `v` serve both QKᵀ and PV.
The query is prescaled by `w_kn` on the non-rotated dims during Q preparation. gewell uses the same
representation ("stores the position-dependent K elements and the V vector").

The oracle remains the formula above. The reformulation changes rounding (the non-rotated k is
reconstructed from stored v), which the Op criterion must cover.

### 3.6 Vision (P5)

- **Patch embedding:** a 27-layer encoder with H = 1152, 16 heads × 72, intermediate 4304,
  `gelu_tanh`. Patches are 16×16. The input is pixels scaled to 2·(x − 0.5), then `input_proj`
  (768 → 1152), plus a learned 2-D position table [2, 10240, 1152] (x-embedding + y-embedding,
  zero for padding).
- **Attention:** each layer has q/k norms with weights and a weightless v norm, axial 2-D RoPE
  (θ = 100, the head dim split between x and y), scale 1.0, non-causal.
- **Pooling:** average pooling over k×k patch groups (k = 3 for 280 soft tokens from 2520 patches),
  ×√1152 in FP32, then standardization with `std_bias`/`std_scale`.
- **Projector:** weightless RMSNorm, then Linear 1152 → 5376. Image embeddings are **not**
  multiplied by √H (llama.cpp comments this explicitly).
- **Tokens and masks:** the default budget is 280 soft tokens per image. Image token runs use the
  sliding-layer bidirectional overlay (3.4).
- **Reuse:** NInfer already has a D = 72, H = 16 non-causal attention profile (the Qwen vision
  tower), `gelu`, `layer_norm` and `vision_pos_embed`. The pooling, the standardization and the 2-D
  RoPE variant are new.

### 3.7 Assistant drafter (P6)

`Gemma4AssistantForCausalLM`:

| Quantity | Value |
|---|---|
| Layers | 4 (3 sliding + 1 global) |
| Hidden / intermediate | 1024 / 8192 |
| Query heads | 32 |
| Own K/V projections | none (`num_kv_shared_layers = 4`) |
| Output head | tied 1024 → 262,144 |
| Soft-cap | none |
| Ordered (centroid) embeddings | `use_ordered_embeddings = false` for 31B |

Its layers are **query-only**: they attend to the **target's** KV. The sliding layers read target
layer 58 (the last sliding layer); the global layer reads target layer 59 (the last global layer).

One draft step:

```text
x0 = W_pre · [ target_embed(token) · √5376 ‖ h_prev ]   # 10752 → 1024
x  = 4 Gemma layers (q-only attention into target KV) on x0
d  = rmsnorm(x, w_final)
logits_draft = W_emb_draft · d
h_next = W_post · d                                    # 1024 → 5376, fed back as h_prev
```

`h_prev` is the target's hidden state for the first step, then the drafter's own `h_next`. P0 must
confirm whether the target hidden is taken before or after the target's final norm. The vLLM
proposer wiring is the reference for this, as Transformers' generation utilities are for timing.

This maps onto NInfer's existing MTP backend concept: concatenated embedding and hidden, one
recurrent hidden, verify in one target pass. It is simpler than Qwen MTP because it has **no draft
KV state**. About 0.45B parameters, about 0.9 GB in BF16.

---

## 4. The NVIDIA NVFP4 checkpoint

`nvidia/Gemma-4-31B-IT-NVFP4`: 4 shards, 1,728 tensors, `total_size` 32.63 GB.

| Group | Storage | Notes |
|---|---|---|
| MLP `gate_proj`, `up_proj`, `down_proj`, all 60 layers | NVFP4: U8 E2M1 pairs; `weight_scale` E4M3 [N, K/16]; `weight_scale_2` FP32 scalar; `input_scale` FP32 scalar | Group 16, W4A4. `input_scale` follows ModelOpt's convention `amax / (6·448)` (layer 0 gate: 0.03125). |
| Attention q/k/v/o, all layers | BF16 | Listed in `exclude_modules` for all 60 layers. |
| `embed_tokens` (tied head) | BF16 | 2.63 GiB. |
| Norms, `layer_scalar` | BF16 | |
| Vision tower, `embed_vision` | BF16 | |
| KV scales | absent | |

NVIDIA's published accuracy, BF16 → NVFP4 (temperature 1.0, top-p 0.95, up to 131K new tokens):

| Benchmark | BF16 | NVFP4 |
|---|---:|---:|
| GPQA Diamond | 85.80 | 85.35 |
| AIME 2025 | 87.92 | 87.60 |
| MMLU-Pro | 85.25 | 84.94 |
| LiveCodeBench | 82.49 | 82.27 |
| SciCode subtask | 33.61 | 33.18 |
| Terminal-Bench Hard | 27.08 | 27.08 |

Calibration: cnn_dailymail. The sample count and sequence length are not given.

These numbers measure MLP-only NVFP4. They say nothing about quantizing attention, which NVIDIA
chose not to do; that is the main quality risk of every layout smaller than 29 GiB (section 13).

---

## 5. Memory and bandwidth budget

These are calculations from the config shapes, not measurements. GiB = 2³⁰. NVFP4 is 4.5 bits per
weight including scales; FP8 rows are 8 bits per weight, with a negligible row scale.

### 5.1 Weights

| Layout | MLP 20.8B | q/k/v | o | Embedding/head 1.41B | Text weights | Plus vision BF16 |
|---|---|---|---|---|---:|---:|
| L0: NVIDIA checkpoint as published | NVFP4 | BF16 | BF16 | BF16 | 29.32 GiB | 30.39 GiB |
| **L1: first artifact** | NVFP4 (imported) | FP8 | FP8 | FP8 rows | **20.11 GiB** | 21.18 GiB |
| L2: ModelOpt `nvfp4_omlp_only` shape | NVFP4 | FP8 | NVFP4 | FP8 rows | 18.85 GiB | 19.92 GiB |
| L3: all projections NVFP4 | NVFP4 | NVFP4 | NVFP4 | FP8 rows | 16.65 GiB | 17.72 GiB |
| gewell G0 (reference) | FP8 in 24 layers, NVFP4 in 36 | FP8 | as MLP | BF16 in gewell (24.17 GiB); FP8 here | 22.85 GiB | 23.92 GiB |

L0 is rejected: it leaves under 2 GiB for KV, workspace, CUDA context and graphs.

The head can also use NInfer's existing Q8/Q6 vocabulary formats. The head is a full 262,144 × 5376
read per decoded token: about 1.41 GB, about 6.5% of the L1 per-token weight traffic.

### 5.2 KV cache

| | BF16 | FP8 |
|---|---:|---:|
| Global, separate K and V, per token (10 layers × 4 heads × 1024) | 80 KiB | 40 KiB |
| **Global, compact K/V** (640 values per head) | 50 KiB | 25 KiB |
| Global compact at 262,144 tokens | 12.5 GiB | **6.25 GiB** |
| Sliding rings per active request (50 layers × 16 heads × 256 × 2 × 1024) | 0.78 GiB | 0.39 GiB |

With L1, about 10 GiB remains after weights for KV, rings, workspace and CUDA overhead. That is enough
for one request at 256K tokens with FP8 compact global KV, or several concurrent requests at moderate
context. Startup capacity planning decides the exact split (`--max-ctx`, `--max-concurrency`),
as for Qwen.

### 5.3 Decode bandwidth ceilings (single request, no speculation)

| Layout | Weight bytes per token | Ceiling at 1.79 TB/s |
|---|---:|---:|
| L1 | 21.6 GB | 83 tok/s |
| L3 | 17.9 GB | 100 tok/s |

At 256K tokens, the global KV adds 6.25 GiB (FP8 compact) per decoded token, about 30% more bytes.
Real throughput will be below these ceilings. The measured gap is the first thing P3 reports.

---

## 6. Weight pipeline

### 6.1 Route R1: import NVIDIA's NVFP4 (first artifact, P1)

1. Add a `gemma4_31b_nvfp4` recipe to `tools/convert/official_recipes.py`, reading the NVIDIA
   checkpoint through the existing ModelOpt NVFP4 source (`tools/convert/sources/modelopt*`).
2. MLP: `import_encoded`, preserving codes, block scales, global divisors and `input_scale`. Gate and
   up form one fused parent: their `input_scale` values must be equal or reconciled by the existing
   rule (one divisor survives per parent; `import_encoded` documents this). Record the actual
   values in P1.
3. q/k/v/o: `fp8_row_maxabs` from the checkpoint's BF16 values (layout L1). q and k/v of one layer
   form the attention input parent (shared input). The global layer has q + k only.
4. Embedding / tied head: FP8 rows for the gather, plus a Q8 or Q6 head use, mirroring
   the Qwen recipes. Whether one stored object can serve both the gather and the head (as tied
   weights would suggest) is decided by the existing Use/Binding rules.
5. Norm weights, `layer_scalar`: BF16 as stored.
6. Frontend resources: `tokenizer.json`, `chat_template.jinja`, the generation defaults
   (`top_k` 64, `top_p` 0.95, temperature 1.0, EOS ids {1, 106, 50}; see 7.4).
7. Vision (P5) and drafter (P6) components are added to the same recipe later as optional components,
   as Qwen does with `--components`.

### 6.2 Activation scales for re-encoded projections

FP8 W8A8 or NVFP4 W4A4 attention projections need activation scales. NInfer's own `nvfp4_absmax`,
`nvfp4_mse` and `fp8_row_maxabs` are weight-only (`docs/maintainer/tensor-formats.md` §2.4 is stale
on this point and says no NVFP4 quantizer exists; a separate fix is queued). There are three options:

| Option | Source | Cost |
|---|---|---|
| A. BF16-activation routes (W8A16 / W4A16) for attention | none needed | The A16 decode routes exist (`fp8_a16_*`, `nvfp4_a16_mma`). Prefill loses the W8A8/W4A4 tensor-core speed. |
| B. gewell's `tools/default_calibration.json` | amax for all 410 text projections of the IT model, measured on 74 histories up to 32K tokens with BF16 activations and QDQ all-NVFP4 weights | Free and Apache-2.0, but measured under someone else's quantization. Use for bring-up only. |
| C. Our own calibration | a calibration observer in NInfer's forward (the EXL3 calibration Program already observes linear inputs), or ModelOpt | The correct long-term source. |

**Plan:** option A for P1/P3 correctness, then C before any performance claim on prefill.

### 6.3 Route R2: our own ModelOpt run (tuning, P4)

ModelOpt at `6e6cadcbaf` registers `gemma4` (needs Transformers ≥ 5.5). `examples/hf_ptq/hf_ptq.py`
exposes the variants we would compare:

| qformat / recipe | Meaning |
|---|---|
| `nvfp4_mlp_only` | NVIDIA's published shape |
| `nvfp4_omlp_only` | Adds `o_proj`; layout L2 |
| `nvfp4` | Everything; layout L3 |
| `nvfp4_w4a4_weight_mse_fp8_sweep` | Per-block E4M3 scale search; NInfer's `nvfp4_mse` mirrors it |
| `nvfp4_w4a4_weight_local_hessian` | Hessian-weighted weight rounding |
| `nvfp4_awq_lite` | AWQ-style pre-scaling |
| AutoQuantize, e.g. `general/auto_quantize/nvfp4_fp8_at_5p4bits` | Per-layer NVFP4/FP8 search under an effective-bits constraint |

Practicalities on this machine:

- BF16 31B is about 62 GB, so `hf_ptq.py` runs with CPU offload (`--use_seq_device_map`,
  `--gpu_max_mem_percentage`). The 256 GB RAM covers it. Expect hours, not minutes. Record the time.
- Calibration data: replace cnn_dailymail with chat-formatted, self-sampled traces. Our tooling
  already does this for EXL3 (`tools/exl3/sample_traces.py`, `ninfer-serve` on the L1 artifact).
  `--calib_size` and the sequence length are recorded in provenance.
- Use `--kv_cache_qformat none`: NInfer owns KV quantization.
- Output: a ModelOpt HF export, imported with R1's recipe (`import_encoded` for every NVFP4/FP8 tensor).

ModelOpt runs in an isolated tooling venv. It is never a product or `.venv` dependency, matching how
exllamav3 was used for the EXL3 references.

### 6.4 Tuning loop with NInfer's object model (P4)

Precision per projection is recipe data: the artifact stores per-object formats and per-Use
activation permissions. The loop:

1. Build the reference (6.5).
2. Measure per-projection sensitivity with `apps/sensitivity` generalized beyond EXL3: inject noise
   with each candidate format's error amplitude, one projection at a time, and score NLL/KLD on a
   packed trace.
3. Candidate layouts: L1, L2, L3, gewell G0, and the AutoQuantize result. Each is a recipe
   variant producing a `.ninfer`.
4. Rank by KLD against the reference, at bytes per token (the decode-speed proxy) and measured
   tok/s. Choose a Pareto point; publish only measured points.
5. Optional (P7): EXL3 4.0 bpw for the MLPs from BF16. On Qwen3.8-27B it beat NVFP4 on KLD (0.033 vs
   0.051) at a smaller size. exllamav3 supports Gemma 4, so a reference quant exists for comparison.
   The EXL3 kernels need the per-shape instances for H = 5376 (section 8).

### 6.5 BF16 reference for KLD

The BF16 model (about 62 GB) does not fit on the GPU. Options, in order of preference:

1. NInfer's streamed reference (`docs/exl3-quant-plan.md` §9 M7b, the `ninfer-reference` design).
   Gemma 4 is a good first non-Qwen consumer because it is dense and has no recurrent state.
2. Transformers on CPU/offload in the tooling venv, writing the reference distribution file that
   `ninfer-perplexity --reference` consumes (M7a). This is slow but independent of our kernels.

Either way, the reference is computed once per corpus and reused.

---

## 7. NInfer architecture

### 7.1 Model directory and ownership

New `src/models/gemma4/`, parallel to `src/models/qwen3_5/`, with its own `config`, `load`,
`model`, `execution/`, `state/`, `frontend/`, `program/`. Registry entry:
`Gemma4ForCausalLM` + `gemma4_text`. The converter writes the text architecture name, as Qwen
does for its `…ForConditionalGeneration` sources.

AGENTS.md forbids family base classes and generic model graphs. Gemma 4 is therefore an explicit
implementation, not a `qwen3_5` subclass.

### 7.2 D1: what to do with the Program

The Qwen Program (`src/models/qwen3_5/program/`, about 30K lines in 52 files) owns:
- scheduling hooks, graph capture, round buffers, transactions;
- checkpoint and context-cache work, pressure planning;
- speculative backends, vision control.

Some of it is architecture-neutral; some is shaped by GDN state (10 of its files reference
GDN/recurrent state). Options:

| Option | For | Against |
|---|---|---|
| a. Copy the Program into `gemma4/program/` and delete what Gemma lacks | Fastest start; no risk to Qwen | Two diverging copies of transaction, checkpoint and graph logic. Violates "one current authority" in spirit. |
| b. **Extract the architecture-neutral parts first**, then write a Gemma Program against them | One implementation of transactions, round buffers, graph-profile capture and pressure planning; Gemma's Program contains only Gemma state and execution | A refactor of working Qwen code before any Gemma output. Needs Qwen regression evidence (the real-model tests). |
| c. Host Gemma inside `qwen3_5/program` with branches (as Qwen4Exp did) | Smallest diff | Qwen4Exp shares most Qwen state; Gemma shares almost none. Accumulates `if gemma` branches through 30K lines. Rejected. |

**Recommendation: b, scoped by need.** Extract only what the Gemma Program actually calls, as
P2 discovers it, into `src/runtime/` (or a new `src/models/common/` if the boundary is
model-execution rather than runtime). The extraction steps are refactors validated by the existing
Qwen tests, and land before the Gemma code that uses them.

### 7.3 State model

Gemma 4 has no recurrent state. Per request, it has:

- **Global KV:** growing and paged, compact layout (3.5). 10 layers × 4 heads × 640 per token. One
  typed growing pool (`docs/maintainer/paged-kv-cache.md` §3).
- **Sliding KV:** 50 layers × 16 heads × 256 × (K, V) for the last 1024 committed positions, as a
  per-lane cyclic ring (`src/core/cyclic_kv_cache.h` already exists for DFlash's local attention).
  New query K/V stays a separate temporary segment until commit, as the existing
  `sliding_window_attention` contract does. That makes speculative rollback and chunked prefill
  safe without ring slack.

For prefix reuse, the sliding rings are a **fixed-size state image** at a token boundary. That is
exactly the role GDN state plays for Qwen (`resource-scheduling-and-context-cache.md` §5.1
StateImage). A checkpoint = global page references + one ring snapshot (0.39 GiB FP8 / 0.78 GiB BF16
at full window; less below 1024 tokens).

A later ring cannot reconstruct an earlier window: a match at 5,000 tokens with the deepest
checkpoint at 4,096 resumes at 4,096. gewell documents the same constraint. The context-cache
policy (where to drop checkpoints) is reused unchanged; only the state-image size and contents differ.

### 7.4 Frontend

| Item | Status | Work |
|---|---|---|
| Tokenizer | **New** | Gemma's `tokenizer.json`: BPE with `byte_fallback`, `fuse_unk`, normalizer `Replace(" " → "▁")`, pre-tokenizer `Split(" ", MergedWithPrevious)`, decoder `Replace("▁" → " ") + ByteFallback + Fuse`, 24 added tokens. The Qwen tokenizer rejects it by design (`tokenizer.cpp` requires NFC + ByteLevel + the Qwen split regex). Oracle: exact token ids against HF `tokenizers` over the corpus and adversarial Unicode/whitespace/byte-fallback cases. |
| Chat template | **Reuse** | `src/text/jinja` is a general engine. The Gemma template (18.7 KB) is checked for constructs our engine lacks. Exact-string oracle against Transformers' `apply_chat_template` over a message/tool corpus. |
| Turn and thinking protocol | **New** | `<|turn>`/`<turn|>` delimit turns, `<|channel>`/`<channel|>` delimit thought channels, `<|think|>` switches thinking on. Map to the serving layer's reasoning output. |
| Tool calls | **New** | `<|tool>…<tool|>` declarations, `<|tool_call>…<tool_call|>` calls, `<|tool_response>…<tool_response|>`, `<|"|>` as the string delimiter. Needs a parser and a constrained-decoding grammar. The current `tool_call_parser`/`tool_grammar` are Qwen-specific. |
| Images | P5 | `<|image>`, `<image|>`, `<|image|>` (soft token), the per-image token budget. |
| Stop ids | Config | {1 `<eos>`, 106 `<turn|>`, 50 `<|tool_response>`}. Stopping on `<|tool_response>` is how the model hands control to the tool runner, so the tool-call parser and the stop logic must agree on it. |

The OpenAI/Anthropic schemas are an external contract. Gemma's tool and reasoning output must map
onto the already-advertised fields, with schema tests updated together (AGENTS.md).

---

## 8. Ops work list

Every row qualifies against a naive FP32/FP64 oracle at the real Gemma shapes
(`docs/maintainer/op-development.md`). "Shapes" means the per-shape instantiation pattern in
`src/ops/linear/{fp8,nvfp4}/shapes/`; no Gemma shape exists today.

| # | Op | Status | Contract to add |
|---|---|---|---|
| 1 | `linear` FP8 / NVFP4 (A16 and A8/A4 routes) | Extend | Shapes: n8192/k5376, n4096/k5376 (sliding k, v), n16384/k5376 + n2048/k5376 (global q, k), n5376/k8192, n5376/k16384 (o), n5376/k21504 (down), n262144/k5376 (head). Tuned per `linear-tuning.md`. |
| 2 | Attention input projection (fused q+k+v parent) | Extend | Sliding: 8192 + 4096 + 4096. Global: 16384 + 2048. Epilogue: per-head q/k RMSNorm with weight, v RMSNorm without weight, the two RoPE kinds. The global epilogue writes compact K/V (v plus 128 rotated k dims) and prescales q by `w_kn` on the non-rotated dims. |
| 3 | `linear_swiglu` → GeGLU | Extend | Fused gate+up parent with a `gelu_tanh(gate) ⊙ up` epilogue: n2×21504/k5376, NVFP4 W4A4 + A16. Either an activation parameter on the existing Op or a sibling `linear_geglu`; decided by the Op-ownership rule in P2. |
| 4 | Sliding attention, causal, ring | New profile | D = 256, Hq = 32, Hkv = 16 (group 2), window 1024, scale 1.0, causal. Committed ring plus temporary segment; T = 1..16 decode/verify, B = 1..8; plus the image-block bidirectional overlay (block ids per position). The existing op is non-causal, D = 128. |
| 5 | Global attention, paged, compact K/V | New profile | D = 512, Hq = 32, Hkv = 4 (group 8), scale 1.0, causal, compact 640-value rows. Prefill and decode routes; the long-context decode split-K matters here (256K keys). BF16 cache first, FP8 second (row 7). |
| 6 | Post-norm + residual + layer scalar + next pre-norm | New fused | `h' = (h + rmsnorm(y, w_post)) · s`, `x = rmsnorm(h', w_next_pre)`; both outputs. One row-reduction kernel per sublayer boundary (120 per token). The `· s` is applied only after the MLP sublayer. |
| 7 | KV append codecs | Extend | Sliding K/V and global compact rows: BF16, then FP8 with a Hadamard basis. The existing H256/16 rotation covers sliding; global needs H512 on v (512) and a separate basis for the 128 rotated k dims. |
| 8 | Embedding gather × √H | Extend | FP8-row gather with the √5376 scale in BF16 (Transformers multiplies by the scale cast to the weight dtype). |
| 9 | Final norm + head + soft-cap | Extend | `30·tanh(z/30)` in the head epilogue, before sampling, `target_logprobs` and scoring. It must apply on every logits consumer: sampling, grammar masks, scoring, speculative acceptance. |
| 10 | Vision ops (P5) | New/extend | 2-D axial RoPE (θ 100, head 72), k×k position pooling, standardization, projector; attention reuses the D72/H16 profile. |
| 11 | Drafter ops (P6) | New | Q-only attention into the target's ring (layer 58) and global pages (layer 59) at D256/D512; pre/post projections; draft head GEMV 1024 → 262,144. |

The order inside P2 follows the critical path to a correct token: 8 → 2 → 4/5 (BF16 cache) → 3 →
6 → 1 → 9. Codecs (7) and tuning come after correctness.

---

## 9. KV and context cache details

- Startup capacity planning follows the existing order: weights → KV floor → rings → workspace →
  optional expert cache (none). Pools: one growing global pool (compact rows) plus per-lane rings.
- **KV formats:** BF16 for bring-up. FP8 with Hadamard as the first compressed format, qualified by
  KLD at long context (needle-style and corpus PPL at 32K/128K). NVFP4 KV is out of scope (gewell
  also refuses FP4 KV).
- **CUDA Graphs:** the decode topology per batch size B covers 60 layers of two kinds. Ring
  addressing (position mod 1024) and page tables are graph inputs, not graph keys, as today.
- **Prefill chunking:** the chunk size is a planning parameter. Each chunk's sliding attention reads
  ring + temporary segment, then commits the last min(1024, chunk) positions into the ring.
- **Speculative verify:** W ≤ 16 positions in the temporary segment; only accepted positions are
  committed to rings and pages. That is the reason the temporary-segment design matters.

---

## 10. Speculation (P6)

1. The assistant drafter is the first backend (3.7). Converter component: the assistant checkpoint,
   BF16 or FP8. The draft head is 268M parameters, about 0.27–0.54 GB.
2. Lookup drafting (`docs/maintainer/lookup-drafter.md`) is model-neutral and may come for free once
   the Program supports verify. Measure it before or alongside the assistant.
3. DFlash/EAGLE3: no Gemma checkpoints exist that we know of. Out of scope.

**Acceptance:** acceptance length and tok/s per K, per `performance/methodology.md`, against the
no-speculation baseline on the same artifact.

---

## 11. Vision (P5)

Components and math in 3.6. Work items:

1. Converter: vision tower and projector BF16 (about 1.07 GiB) or FP8 rows for the 27-layer MLPs.
2. Ops in section 8, row 10.
3. Frontend image preprocessing per `image_processing_gemma4.py`: resize to the token budget,
   patchify, position ids.
4. Image-block ids feed sliding attention (row 4); global layers stay causal.
5. Media acquisition reuses `src/product/media_acquire` and `src/media/decode`.

---

## 12. Qualification and acceptance

| Level | Evidence | Criterion |
|---|---|---|
| Oracle | FP64 Python reference `tests/models/gemma4/reference.py`, checked against Transformers on small random configs and on real layers | Matches Transformers to FP32 rounding; documents every disagreement with vLLM/llama.cpp. |
| Tokenizer / template | Exact ids and exact rendered strings vs HF | Exact. |
| Ops | Each row of section 8 vs its naive oracle at Gemma shapes and route boundaries | The Op's named criterion (op-development.md). |
| Model forward | NInfer layer-by-layer vs FP64 oracle on real weights for a few positions, then full logits | Per-layer relative error bounded; top-1 agreement and KLD vs the oracle on a short corpus. |
| Artifact quality | KLD + PPL vs BF16 reference (6.5) for L1 and each tuning candidate | Ranking reproduced on two corpus halves (the M5 practice). |
| Task quality | NInfer EvalScope set used by the Qwen cards (GPQA Diamond, AIME 2025, MMLU-Pro subset, IFBench…) at NVIDIA's sampling settings | Compared with NVIDIA's NVFP4 table where benchmarks overlap; differences explained, not tuned away. |
| Serving | Schema tests for tool calls and reasoning; streaming behavior | Existing serving test suites extended. |
| Performance | `ninfer_bench`: prefill 8K–256K, decode C = 1/2/4/8, speculative K sweep | Published only as measured. Bandwidth ceilings from section 5 reported alongside, to show the efficiency gap. |

Record for every material claim: artifact path, GPU/driver/CUDA (13.3), command, result summary.

---

## 13. Phases

| # | Deliverable | Done when |
|---|---|---|
| P0 | Model reference and oracle | `docs/maintainer/gemma4-model.md` written from section 3. FP64 oracle agrees with Transformers (config fuzz + real-weight layers). Open points (hidden before/after final norm for the drafter, window boundary, image-block mask) closed with evidence. Tokenizer exact-id tests pass. |
| P1 | Converter + L1 artifact | `gemma4_31b_nvfp4` recipe produces `out/gemma4_31b_nvfp4.ninfer` (L1) from the NVIDIA checkpoint; byte-exact NVFP4 import verified; artifact contract tests pass; model card draft. |
| P2 | Program boundary + Ops | D1 extraction steps landed with Qwen regression evidence; section 8 rows 1–9 qualified at BF16 KV. |
| P3 | Text end to end | `ninfer` generates and `ninfer-perplexity` scores L1. Model-forward and KLD checks pass. Prefix reuse with ring checkpoints works. First decode/prefill measurement. Serving with tools and thinking passes schema tests. **This is the first completion point.** |
| P4 | Precision tuning | Reference built (6.5); own calibration (6.2 C); ModelOpt R2 run(s); L1/L2/L3/G0/AutoQuantize ranked on KLD vs bytes; FP8 KV qualified; chosen layout published with evidence. |
| P5 | Vision | Image prompts through the public Engine, vision oracle, serving schema for images. |
| P6 | Assistant drafter | Drafter backend, acceptance and tok/s measured against no speculation. |
| P7 (optional) | EXL3 MLPs | EXL3 4.0 bpw quantized from BF16 with our quantizer; KLD vs L-best and vs exllamav3's quant. |

P0 and P1 are independent and can run in parallel. P2's Op work can start once P0 fixes the
mathematics. P5 and P6 are independent of each other once P3 lands.

---

## 14. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Quantized attention hurts quality more than NVIDIA's MLP-only numbers suggest | L1–L3 lose to an unfittable L0 | KLD per projection; keep sensitive layers FP8/BF16 (gewell G0 keeps the first six layers and the global layers plus their predecessors at FP8). |
| Program extraction (D1-b) destabilizes Qwen | Regressions in shipped models | Extract only what Gemma calls, one boundary at a time, each with the real-model Qwen tests. |
| Compact global KV rounding | Long-context drift | Oracle criterion at 128K/256K; fall back to separate K/V rows (+60% global KV) if it fails. |
| Global decode attention at 256K is bandwidth-heavy | Long-context decode well below the short-context ceiling | Split-K decode tuned at 64K–256K; FP8 KV. |
| Tokenizer edge cases (byte fallback, `▁` handling, added-token stripping) | Silent prompt corruption | Exact-id oracle on adversarial inputs, not just corpus text. |
| Tool/thinking protocol mapping | Client-visible serving bugs | Schema tests first; template-render exactness. |
| ModelOpt run time / memory with CPU offload | P4 delays | R1 never depends on it; record the run time; use AutoQuantize only on a subset if too slow. |
| NVIDIA checkpoint provenance ambiguity (0.37 vs 0.42) | Reproducibility of R1 | Pin the HF revision hash in the model card. |
| `layer_scalar` with very small values (0.036–0.089) on the first and last layers | Residual precision | Residual stream stays BF16/FP32 as Qwen's; check the oracle error at layers 0, 1, 59 specifically. |

---

## 15. Decisions needed

| # | Decision | Recommendation |
|---|---|---|
| D1 | Program structure (7.2) | b: extract architecture-neutral parts on demand. |
| D2 | Instruction or base model | Instruction (`-it`): NInfer is a chat/serving engine, NVIDIA's NVFP4 and the assistant drafter exist only for `-it`. Base needs our own ModelOpt run from P1 onward. |
| D3 | First precision layout | L1 (MLP NVFP4 imported, attention + embedding FP8 rows) with BF16-activation attention routes until calibrated (6.2 A). |
| D4 | First compressed KV format | FP8 with Hadamard, after BF16 bring-up. |
| D5 | Where the model reference lives | `docs/maintainer/gemma4-model.md` (permanent), this plan removed when P3 lands, per AGENTS.md's one-current-authority rule. |

---

## 16. Follow-ups (not planned here)

- `gemma-4-26B-A4B` (128 experts, top-k router with per-expert scale, MoE block parallel to the dense
  MLP): the attention, frontend and Program work transfers, the MoE path does not. NVIDIA publishes
  an NVFP4 checkpoint for it; gewell supports it.
- E2B/E4B variants (per-layer embeddings, KV-shared layers, audio).
- Video input.
