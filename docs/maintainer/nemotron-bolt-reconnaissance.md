# Nemotron Bolt: sourced reconnaissance

External facts gathered while planning this port, each tagged to the source it came from. The
authority for the port is `nemotron-bolt-plan.md`; this file is the evidence behind several of its
claims, kept so nobody re-derives it -- the draft's tensor layout, the vendor's precision rationale, and
the published recovery numbers.

Every claim below was read from the named source. Where a source contradicted itself, or where a
question stayed open, that is stated rather than smoothed over.

---

# Draft (LiLiCorr) + reference serving settings + Nemotron 3.5 report

Recon for the Nemotron-3.5-Bolt-74B-A9B serving/drafting plan. All facts below were fetched 2026-10-15
from the live Hub/GitHub; nothing here is from local repo files.

## Source key

| Tag | Source |
|---|---|
| `[D-api]` | `https://huggingface.co/api/models/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr?blobs=true` |
| `[D-card]` | `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr` → `README.md` (14,491 B) |
| `[D-cfg]` | same repo → `config.json` (2,389 B) |
| `[D-quant]` | same repo → `hf_quant_config.json` (5,594 B) |
| `[D-hdr]` | same repo → `model.safetensors` header, read directly (18,864-byte JSON header after the 8-byte length) |
| `[G-card]` | `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr-GGUF` → `README.md` |
| `[T-card]` | `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB` → `README.md` (85,216 B) + `config.json` |
| `[MO-§2/§5]` | `NVIDIA/Model-Optimizer`, `examples/megatron_bridge/tutorials/NemotronH-Omni-W4A4-QAD/README.md` (fetched raw via `gh api .../contents/... -H 'Accept: application/vnd.github.raw'`), sections 2 and 5 |
| `[MO-root]` | `NVIDIA/Model-Optimizer` → `README.md` (Latest News entry that links the report) |
| `[R-§x]` | `https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3.5-Super-and-Lightning-Technical-Report.pdf` (dated 2026-10-14, 4.78 MB), section/table numbers are the report's own |
| `[L-card]` / `[S-card]` | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` / `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` → `README.md`; quant configs from `hf_quant_config.json` |

---

## 1. The LiLiCorr draft for our target

### Files (`[D-api]`, sha `4c6a5613b6d887194a3a5493e3a18a5b3930bb86`, last modified 2026-10-15, not gated)

| File | Bytes |
|---|---:|
| `model.safetensors` | **962,485,872** (LFS sha256 `850c121c5fa444dfdebbc15d8f23a1ab2beda55863460ab44203e3cb63808562`) |
| `README.md` | 14,491 |
| `hf_quant_config.json` | 5,594 |
| `config.json` | 2,389 |
| `LICENSE` | 2,618 |
| `.gitattributes` | 1,519 |

That is the whole repo — **one 962 MB safetensors file**, no index, no shards. Hub tags: `qwen3`,
`modelopt_fp4`, `8-bit`, `latent-moe`, `LiLiCorr`, `arxiv:2608.20530`,
`base_model:…Bolt-74B-A9B-24GB` (+ the same as `base_model:quantized`), license OpenMDW-1.1 `[D-api]`.
(The automatic `8-bit` tag is a coarse Hub tag; the real algorithm is W4A16 as below.)

### Architecture — `[D-cfg]`

```
architectures        ["LiLiCorrDraftModel"]        model_type         "qwen3"
hidden_size          4096                          num_hidden_layers  5
num_attention_heads  32                            num_key_value_heads 2      (GQA 16:1)
head_dim             128                           intermediate_size 12288
hidden_act           silu                          rms_norm_eps       1e-06
vocab_size           131072                        max_position_embeddings 262144
rope_theta           10000.0                       rope_scaling       null
tie_word_embeddings  false                         torch_dtype        bfloat16
block_size           8                             num_target_layers  0
layer_types          ["sliding_attention"] x5      sliding_window     1024
attention_sink_bias  true                          use_sliding_window true
max_window_layers    0
```

`dflash_config`: `mask_token_id` 990, `target_layer_ids` **[1, 18, 35, 51, 68, 85]**,
`use_swa` true / `swa_window_size` 1024, **`causal`: false** (bidirectional inside the block),
`attention_sink_bias` true, `projector_type` `"lilicorr"`, `lilicorr_enabled` true,
`lilicorr_candidate_topk` 8, `lilicorr_hidden_size` 1024, `lilicorr_num_layers` 2,
`lilicorr_num_heads` 8, `lilicorr_mlp_ratio` 2.0, `lilicorr_factor_dim` 1024,
`lilicorr_vector_eps` 1e-4, `lilicorr_logit_scale` 8.0, `conv_kernel_size` 2, `conv_group_size` 16.

### Parameter count and where the bytes go

- Card: **1.113 B draft parameters excluding shared target embeddings and the LM head; the two-layer
  LiLiCorr head accounts for 37 M of them** `[D-card]` (identical wording in `[G-card]`).
- Recovered from the tensor header `[D-hdr]` — 179 tensors, and the dtypes reproduce the file size
  exactly (962,467,000 B of payload + 18,872 B header = 962,485,872 B):
  - **U8 439,353,344 B = 878,706,688 packed 4-bit weights**
  - BF16 234,097,194 elements = 468,194,388 B
  - F8_E4M3 54,919,168 elements = 54,919,168 B (group scales)
  - F32 25 scalars = 100 B
  - **Real parameters = 878.71 M (4-bit) + 234.10 M (BF16) = 1,112.8 M ≈ 1.113 B** — the card's number
    reconciles exactly, and the F8 scales are not parameters.
  - Note: the Hub's `safetensors` summary for this repo reports only `{"BF16": 234097194, "U8": 439353344}`
    — it silently omits the 54.9 M F8_E4M3 group scales `[D-api]`.

### What the draft's own weights are quantized to

`hf_quant_config.json` `[D-quant]`: producer `modelopt` (version string `0.0.0`),
`quant_algo` = **`W4A16_NVFP4`**, `kv_cache_quant_algo` = `null`, `group_size` = **16**.
So: **4-bit NVFP4 weights (e2m1) with 16-bit activations, E4M3 per-16-group weight scales plus an F32
per-tensor scale** — the same `W4A16_NVFP4` scheme the report applies to the parallel drafters `[R-§4.8]`.

The exclude list covers all five layers' `self_attn.*` (q/k/v/o_proj, q_norm, k_norm, attention_sink_bias),
both convs per layer (`attention_conv*`, `mlp_conv*` incl. `kernel_projection`), every layernorm,
`hidden_norm`, `norm`, `lm_head`, `*qkv_proj`, and the small LiLiCorr heads
(`in_head`, `out_head`, `anchor_out_head`, `factor_input_proj`, `feature_mlp*`). The header confirms
which tensors are *actually* packed, and it is exactly: **the five body MLPs, `fc`, and ten LiLiCorr
projections**. Everything else is BF16 `[D-hdr]`:

| Kind | Storage | Shapes (layer 0 / head) |
|---|---|---|
| `fc.weight` | U8 + F8 scale + F32 | `[4096,12288]` U8 (= **4096 × 24576** real), scale `[4096,1536]`, so group 16 and **24576 = 6 × 4096** |
| `layers.N.mlp.gate_proj` / `up_proj` | U8 | `[12288,2048]`, scale `[12288,256]` |
| `layers.N.mlp.down_proj` | U8 | `[4096,6144]`, scale `[4096,768]` |
| `lilicorr.token_proj` / `context_proj` / `pass_hidden_proj` | U8 | `[1024,2048]` (= 1024 × 4096), scale `[1024,256]` |
| `lilicorr.layers.{0,1}.attn.out_proj` | U8 | `[1024,512]`, scale `[1024,64]` |
| `lilicorr.layers.{0,1}.mlp.up_proj` / `down_proj` | U8 | `[2048,512]` / `[1024,1024]` |
| body attention `q_proj` / `o_proj` | BF16 | `[4096,4096]` |
| body attention `k_proj` / `v_proj` | BF16 | `[256,4096]` (= 2 KV heads × 128) |
| `attention_sink_bias` | BF16 | `[32]` |
| `q_norm.weight` / `k_norm.weight` | BF16 | `[128]` |
| `attention_conv.base_kernel` / `mlp_conv.base_kernel` | BF16 | `[2,2,4096]` |
| `attention_conv.kernel_projection` / `mlp_conv.kernel_projection` | BF16 | `[1024,4096]` |
| LiLiCorr head | BF16 | `factor_input_proj [1024,3072]`, `feature_mlp.up_proj [1024,5]`, `feature_mlp.down_proj [1024,1024]`, `feature_norm [5]`, `slot_embedding [1,1,7,1,1024]`, `rank_embedding [1,1,1,8,1024]`, `in_head`/`out_head [1024,1024]`, `anchor_out_head [1024,1024]`, `anchor_norm [1024]`, `output_norm [1024]`, `relative_slot_bias [8,15]`, `same_slot_bias [8]`, plus a 2-layer transformer `lilicorr.layers.N` (fused `attn.in_proj_weight [3072,1024]`, `attn.out_proj`, `mlp.up_proj/down_proj`, `attn_norm`/`mlp_norm`) |
| no embedding / no `lm_head` | — | confirmed absent from the header: every tensor is a projection, norm, bias or embedding-of-slots |

Two details the plan does not list that are in the file: `lilicorr.factor_input_proj` (a `[1024,3072]`
projection of per-candidate scalar features) and the fact that the LiLiCorr head's own two transformer
layers **are** 4-bit quantized (`lilicorr.layers.*.mlp`, `.attn.out_proj`) while its
`in_head`/`out_head`/`feature_mlp`/`factor_input_proj` are BF16.

### Which target it pairs with, and how it is used

- Pairs with **`nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB`** (the GGUF variant of the drafter pairs
  with `…-24GB-GGUF`) `[D-card][G-card]`. `base_model` in the Hub metadata is that repo
  (plus a `base_model:quantized` relation) `[D-api]`.
- "This checkpoint is **not a standalone language model**. It proposes draft token candidates that the
  compatible 24GB target verifies; the target produces the user-visible text." It "shares the target's
  tokenizer and vocabulary", has no embedding and no LM head, and "drafts tokens using representations
  from the target model" `[D-card]`.
- Runtime: **vLLM** on Blackwell SM120 only (incl. DGX Spark GB10); the GGUF drafter is for **llama.cpp**
  on RTX Spark / DGX Spark `[D-card][G-card]`.
- The exact serving invocation is on the *target* card, not the drafter card `[T-card]`:
  ```
  export VLLM_USE_V2_MODEL_RUNNER=1
  vllm serve nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB \
    --trust-remote-code --load-format auto --tensor-parallel-size 1 \
    --max-model-len 262144 --max-num-seqs 1 --enable-chunked-prefill \
    --kv-cache-dtype fp8 --linear-backend b12x --moe-backend b12x \
    --attention-backend FLASHINFER --mamba-backend flashinfer \
    --mamba-ssm-cache-dtype bfloat16 --mamba-cache-dtype bfloat16 --mamba-cache-mode align \
    --seed 42 --allowed-local-media-path / \
    --limit-mm-per-prompt '{"video":1,"image":1}' --video-pruning-rate 0.5 \
    --media-io-kwargs '{"video":{"fps":2,"num_frames":256}}' \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --reasoning-parser-plugin ultra_v3_reasoning_parser.py --reasoning-parser ultra_v3 \
    --speculative-config '{"method":"dflash","model":"nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr",
                           "num_speculative_tokens":7,"rejection_sample_method":"standard",
                           "attention_backend":"FLASHINFER","draft_load_config":{"load_format":"auto"}}'
  ```
  So the drafter is loaded through the **`dflash` speculative method with a draft length of 7**, a
  standard rejection sampler, and the drafter's own attention on FLASHINFER. The Bolt card also states
  "LiLiCorr drafter via DFlash; **7 speculative tokens** in the companion Bolt vLLM recipes" `[T-card]`.
  (Note the card's own `vllm download-kernels` / `pip install b12x` bootstrap `[T-card]`.)
- Published acceptance (drafter card, SPEED-Bench, draft length 7, `temperature=1.0, top_p=0.95`,
  explicit caveat "**not attributed to a specific serving runtime**") `[D-card][G-card]`:
  coding 5.26, rag 5.13, multilingual 5.06, summarization 4.80, math 4.59, reasoning 4.47, qa 3.91,
  stem 3.73, humanities 3.54, roleplay 3.45, writing 3.41 — **overall average 4.30**.

### Cross-check against the plan's claims

| Plan claim | Verdict | Evidence |
|---|---|---|
| "Qwen3-shaped five-layer block drafter (`block_size` 8)" | **Confirmed** | `model_type "qwen3"`, arch `LiLiCorrDraftModel`, `num_hidden_layers` 5, `block_size` 8 `[D-cfg]`; report: "The drafter uses a block of B = 8 positions, i.e., 7 drafted slots" `[R-§5.7.1]` |
| "its context is the target's residual stream at tapped layers" | **Confirmed** (with one naming nuance) | Report §5.7.1: vLLM replicas of the target "stream the hidden states of **five captured target layers, together with the final hidden state**"; and the pack's `fc.weight` real shape `4096 × 24576` with a `[4096,1536]` group-16 scale proves **six 4096-wide target states** are concatenated `[D-hdr]` |
| "6 × 4096 = all six `target_layer_ids` (1, 18, 35, 51, 68, 85)" | **Confirmed as a config fact**, but see unknown #1 | `target_layer_ids` is exactly that list and `fc` consumes 6×4096 `[D-cfg][D-hdr]`; in the Bolt target these land on layer types `[moe, mamba, mamba, mamba, mamba, moe]` (target has 88 layers: 40 mamba / 40 moe / 8 attention) `[T-card config.json]` — i.e. the taps are 5 sparse points + the last, not six of one kind |
| "GQA with per-head QK norms, attention sinks, 1024 sliding window, `causal: false`" | **Confirmed** | `q/k_norm [128]`, `attention_sink_bias [32]`, `sliding_window`/`swa_window_size` 1024, all 5 layers `sliding_attention`, `causal: false` `[D-cfg][D-hdr]`; card: "All five layers use a 1024-token per-head sliding window and attention sinks" `[D-card]` |
| "grouped dynamic convolution wraps both sublayers" | **Confirmed** | BF16 `{attention,mlp}_conv.base_kernel [2,2,4096]` + `{attention,mlp}_conv.kernel_projection [1024,4096]` in each of the 5 layers `[D-hdr]`; card and report both say a grouped dynamic convolution (kernel 2, group 16) wraps each attention and MLP sublayer `[D-card][R-Table 27]` |
| "LiLiCorr head: token/context/pass_hidden proj (4096→1024), five-feature MLP, 2-layer transformer over slot×rank, in/out heads, anchor head, `relative_slot_bias [8,15]`, `same_slot_bias [8]`" | **Confirmed, and tighter than the plan** | Report Table 27: head layers 2, width 1024, heads 8, FFN 2048 — matching `lilicorr.layers.*` (`in_proj_weight [3072,1024]`, `mlp.up_proj [2048,512]` packed) and `lilicorr_hidden_size 1024 / num_layers 2 / num_heads 8 / mlp_ratio 2.0` `[D-cfg][D-hdr]` |
| "no embedding and no LM head; shares the target's tokenizer/vocabulary" | **Confirmed** | no `embed_tokens`/`lm_head` tensor exists in the file; card states it explicitly `[D-hdr][D-card]` |
| "(implicit) drafter is a separate small file, W4A16-ish" | **Confirmed** | 962 MB single file; `W4A16_NVFP4`, group 16 `[D-quant]`; report: parallel drafters "benefit greatly from weight quantization, as this reduces the memory overhead from loading drafter parameters" `[R-§4.8]` |

Report-side detail worth carrying into the plan: for **Super**, the drafter's BF16→NVFP4 quantization of
its linear layers (activations BF16) gave **+11 % throughput on DGX Spark** with no meaningful acceptance
loss, and quantizing the LM head gave a further **+5 %** `[R-§4.8]`; and the evaluation served the drafter
at **concurrency 64 with 8,192-token sliding-window attention** because the drafter "does not extrapolate
to longer contexts" `[R-§5.7.1]` — the Bolt config instead declares a 1024 window (unknown #2).
The LiLiCorr paper itself is arXiv **2608.20530** `[D-api][D-card]` (not fetched here).

---

## 2. The reference serving settings (Model-Optimizer NemotronH-Omni-W4A4-QAD tutorial)

### The published numbers `[MO-§Results]`

All checkpoints served with the same vLLM settings (§5); quantized ones with an FP8 KV cache; BF16 is
the mean of two runs. The exported W4A4 checkpoint is **70 GB** (bytes on the Hub: 75,010,188,240 =
75.0 GB decimal = **69.8 GiB**, which reconciles the "70 GB" claim).

| Benchmark | BF16 (teacher) | W4A4 PTQ | W4A4 QAD (600 iters) |
|---|---:|---:|---:|
| AA-LCR | 68.63 | 68.62 | 67.94 |
| AA-Omniscience | 26.21 | 25.63 | 25.87 |
| AA-Omniscience (non-hallucination) | 70.03 | 71.34 | 69.65 |
| GPQA Diamond | 89.52 | 89.84 | 89.90 |
| HLE (no tools) | 31.79 | 30.49 | 30.72 |
| MMLU-ProX | 76.91 | 74.41 | 81.15 |
| SciCode (subtask) | 47.13 | 46.48 | 46.58 |
| SciCode (problem) | 15.10 | 13.65 | 15.38 |
| τ³-bench banking | 18.88 | 17.40 | 20.10 |

(MMLU-ProX scores a response only if it ends with the answer phrase in the question's language; most of
the QAD gain is fewer responses missing that phrase — 11.3 % PTQ → 5.4 % after QAD — with readable-answer
accuracy rising 83.9 → 85.8.)

### Section 5 in full — the serving settings `[MO-§5]`

> "Serve with vLLM on one 4-GPU GB300 node, **TP=4 with expert parallelism**. Quantized checkpoints use
> these engine arguments; the BF16 baseline uses `--kv-cache-dtype auto`."

| Setting | Value |
|---|---|
| `--kv-cache-dtype` | `fp8` |
| `--max-model-len` | `262144` |
| `--mamba-ssm-cache-dtype` | `float16` |
| `--mamba-conv-cache-dtype` | `float32` |
| Mamba cache rounding | stochastic rounding, 5 Philox rounds |
| Reasoning parser | `nemotron_v3` |
| Chat template | the checkpoint's `chat_template.jinja` |

That table is the entire section 5 — the README ends there (10,455 B total). The rest of the
reproduce path is §2–§4: NeMo `26.08` container on **GB300, 4 GPUs/node, aarch64**, recipe
`models/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16/ptq/nvfp4-mse`, PTQ on one 4-GPU node with
**512 `cnn_nemotron_v2_mix` calibration samples** and `--tp_size 1 --ep_size 4 --pp_size 1`
(`--tp_size 1` is *required* — static NVFP4 scales are per-shard but saved replicated, so TP>1 raises
`NotImplementedError`; "expert parallelism spreads the experts instead"), then QAD on
**8 nodes × 4 GB300 = 32 GPUs with context parallelism 4 and expert parallelism 32**
(PTQ written at EP=4 loads at EP=32), then export with `--pp_size 4` `[MO-§2–§4]`.

### Expert residency / CPU–host offload — the part that matters to us

**The tutorial README says nothing about expert residency, expert offload, or CPU/host memory. Zero
mentions** of `offload`, `cpu`, `host`, or any residency flag anywhere in the file — I grepped the whole
document, not just §5. The only mechanism it names for keeping 512-experts-per-layer MoE affordable is
**expert parallelism** (EP=4 for PTQ, EP=32 for QAD, "TP=4 with expert parallelism" at serving), i.e.
splitting the banks across GPUs, never pinning them in host RAM `[MO-§2,§5]`.

Adjacent evidence from the same family (outside that README, cited explicitly):

- The report's *only* host-memory statement is for a different thing: on Vera Rubin (11 GPUs, vLLM +
  Dynamo), "each aggregated instance and each prefill GPU also has a **240 GiB KV cache tier in host
  (CPU) memory**. This tier keeps the prefixes of earlier turns after the GPU cache evicts them"
  — a **KV** tier, not experts `[R-§5.6]`.
- The TRT-LLM DGX Spark config uses `kv_cache_config.host_cache_size: 0` and
  `max_gpu_total_bytes: 5368709120` — again KV-cache host tiering, explicitly disabled, and it is
  TensorRT-LLM, not vLLM `[S-card]`.
- The Bolt target recipe is even further from offload: **`--tensor-parallel-size 1`,
  `--max-num-seqs 1`** on a 24.1 GB checkpoint (24,083,745,496 B on the Hub), i.e. the reference
  "24GB" deployment keeps a *single* stream on one device and documents no residency policy at all
  `[T-card]`.
- Everything I found named "offload" in Model-Optimizer is **quantization-time** model loading
  (`modelopt/torch/utils/plugins/model_load_utils.py`, `accelerate.py`, `transformers.py`, etc. via
  GitHub code search), not serving-time weight placement.
- Also relevant, and undocumented as a bolt-offload knob: the report's §5.3 explains that
  `dflash`/LiLiCorr drafting needs the target's **intermediate hidden state at the last prefix token**,
  which the prefix cache does not hold (it holds attention KV + a checkpointed Mamba state), so vLLM
  replays from the nearest Mamba checkpoint and optionally keeps an extra Mamba cache page near the
  speculation boundary `[R-§5.3]`.

**Conclusion for us:** no published reference pins routed experts in host RAM. Our host-RAM expert-bank
cache is an approach the reference stack does not take and cannot be cross-checked against; the closest
published analogue is per-GPU expert parallelism plus a host KV tier.

---

## 3. The technical report (Nemotron 3.5 Super and Lightning)

Found via the Model-Optimizer root README's 2026/10/15 news entry, which links "the Quantization section
(Section 4) of the technical report": `https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3.5-Super-and-Lightning-Technical-Report.pdf`
(4.78 MB, PDF dated **2026-10-14**) `[MO-root]`. It covers Super (120B-A12B) and Lightning (30B-A3B) —
**not Bolt**.

### Published accuracy for the NVFP4 checkpoints

| Checkpoint | Scheme | Published accuracy |
|---|---|---|
| Super VL 120B-A12B NVFP4 (W4A4 on Blackwell) | aggressive recipe = NVFP4 MoE + shared experts, FP8 attention projections and Mamba linears, FP8 KV, FP16+SR SSM cache, quantized MTP; whole-model **4.82 BPE** (S = 74.987 GB, N = 124.453 B) | **"median accuracy recovery 99.62 % across 36 benchmarks"** relative to BF16 `[R-§4.1]`; verbosity +0.44 %; §4.9/Table 22 for Lightning is separate. **The conclusion says 99.52 %** and Table 18 lists 99.36 % (NVFP4 KV) / 99.52 % (FP8 KV) — the released number is stated inconsistently in the report (unknown #5) |
| Super VL NVFP4 vs BF16 (card) | — | e.g. GPQA-Diamond 89.52→89.90; MMLU-ProX 76.91→73.76; Terminal-Bench 2.1 60.67→62.78; SWE-bench Pro 51.78→51.30; DeepSWE 24.65→**18.2**; AA-Omniscience accuracy 26.20→25.86 `[S-card]` |
| Lightning 30B-A3B NVFP4 (**W4A16**) | routed + shared experts and LM head 4-bit NVFP4 W4A16, Mamba in/out proj FP8 W8A8, attention projections BF16, FP8 KV, FP32 SSM cache for eval | **median accuracy recovery 99.24 %**, verbosity **−3.63 %**; extremes: Terminal-Bench v2.1 90.22 %, Tau3 Banking 95.83 %, AA-Omniscience accuracy 93.61 %, GDPval 111.01 % `[R-§4.9, Table 22]` |
| Same Omni/Super VL W4A4+QAD as the tutorial | — | see §2 table above `[MO-Results]` |
| KV-cache ablation (Super) | NVFP4 KV vs FP8 KV, both aggressive + QAD-200 | 99.36 % vs 99.52 % median recovery; released checkpoint **defaults to FP8 KV** "for runtime-efficiency and software-readiness considerations despite NVFP4 KV's storage advantage" `[R-§4.5, Table 18]` |

Supporting recipe facts: the released aggressive tier is per-module mixed (MoE+shared NVFP4, latent
down/up proj BF16, QKV/out proj FP8, KV cache FP8, BMM1 FP8, BMM2 FP8, Mamba linear FP8, SSM cache
FP16+SR, convolution BF16, embedding BF16, lm_head FP8), and the **MTP head is quantized too** (experts
NVFP4 **W4A4**, attention BF16, K/V BMM FP8) `[R-Table 12, §4.4]`.

### Why Lightning is W4A16 while Super VL is W4A4 — stated reason

This is an explicit, deliberate deployment decision, quoted verbatim from §4 `[R-§4]`:

> "The deployment use case also influences recipe selection. For example, **Lightning 3.5 is designed to
> excel in local, agentic use cases as well as datacenter deployments. We therefore quantize Lightning
> 3.5 to minimize weight fetch and raise activation precision to maintain accuracy. Lightning 3.5
> deploys to W4A16 regardless of GPU SKU, while Super 3.5 VL's NVFP4 checkpoint deploys W4A4 on
> Blackwell, and W4A16 on non-Blackwell platforms.**"

Corroborating context: §4.9 restates "In the W4A16 NVFP4 checkpoint, the routed experts, shared experts,
and language-model output head use 4-bit NVFP4 weights with 16-bit activations"; the Lightning card
lists "NVIDIA Ampere via W4A16" and a hardware matrix of `NVFP4` stored precision with compute path
`FP4` on GB200 and `W4A16` via `marlin` on DGX Spark / `humming` on Hopper/Ampere `[L-card]`; the
Lightning card's DGX Spark recipe uses `--moe-backend marlin` (a W4A16 path — GB10 has no usable FP4
tensor-core route) `[L-card]`; and §4.8 generalizes it for drafters: "in memory-bound scenarios, e.g. on
edge devices with low concurrency, parallel drafters benefit greatly from weight quantization, as this
reduces the memory overhead from loading drafter parameters" `[R-§4.8]`.

So: **a deliberate accuracy/size trade for the target hardware, not a different pipeline artifact** —
the same PTQ+QAD stack, with the activation precision (and hence the interleaving requirement) chosen
per deployment class. One tension to keep in mind: §4's "regardless of GPU SKU" vs the Lightning card's
matrix, which shows GB200 running the FP4 compute path; the card is per-recipe and the report is per
evaluated configuration (unknown #4). The Super VL card states the SKU dependence the other way round
and matches the report: "Inference uses **W4A4 on Blackwell and W4A16 on non-Blackwell platforms**,
according to the serving configuration" `[S-card]`.

### Stated throughput / memory figures

| Figure | Value | Source |
|---|---|---|
| Super max throughput (abstract) | **up to 4,856 output tok/s/GPU on one GB200** while sustaining ≥50 tok/s/user, 20-turn proxy workload (51,200-token shared prefix, 2,048 in / 1,024 out per turn) | `[R-Abstract, §5.1, Table 24]` |
| NVFP4 vs BF16 | **2.43×** the throughput of BF16 under the evaluated interactive agentic workload | `[R-Abstract]` |
| Super vs Qwen on GB200 (TPS/GPU at ≥50 tok/s/user) | 4,331.88 vs Qwen3.5-122B-A10B 1,854.35 (**2.34×**) and Qwen3.8-Flash-Next 2,282.73 (**1.90×**) | `[R-Fig. 11]` |
| Super vs Qwen on H100 (same metric) | 988.87 vs Qwen3.5 424.94 (**2.33×**) and Qwen3.8 752.95 (**1.31×**) | `[R-Fig. 11]` |
| ReplaySSM (16-step buffer, GB200, MTP on) | output throughput **+4.7–6.4 %**, per-user **+6.4–12.5 %** | `[R-Table 24]` |
| Disaggregated serving, Vera Rubin VR200, 11 GPUs (7 prefill / 4 decode) | 3,276 TPS/GPU at 155.9 TPS/user, TTFT 1.12 s vs aggregated 2,969 / 96.6 / 0.49 s | `[R-Table 25]` |
| LiLiCorr drafter quantization on DGX Spark | **+11 %** throughput from NVFP4 (BF16 activations) linear layers, **+5 %** more from the quantized LM head | `[R-§4.8]` |
| Host memory | 240 GiB KV cache tier in host (CPU) memory per aggregated instance / prefill GPU | `[R-§5.6]` |
| Symbolic memory sizes | Super NVFP4 `S = 74.987 GB` stored payload, 4.82 BPE; released **75,010,188,240 B on the Hub**; Lightning NVFP4 **21,561,882,284 B**; Bolt-74B-24GB **24,083,745,496 B**; LiLiCorr draft **962,485,872 B** | `[R-§4.1]`, Hub blob listings |
| MTP/drafter acceptance | MTP boosting: 4.287→4.718 (greedy) and 3.787→4.612 (probabilistic) on SPEED-Bench; MTP AL at DL=5/7 with quantized MTP experts + W4A16 head: 4.025/4.471 | `[R-Abstract, Table 20]` |
| LiLiCorr acceptance (Super, draft length 7, conc. 64) | BF16 drafter 4.77 (greedy) / 4.37 (temp 1); NVFP4 drafter 4.74 / 4.36 | `[R-Table 28]` |

No absolute throughput or latency figure is published for **Lightning NVFP4** itself; its card only
states a service target ("interactive usage scenarios (achieving **40+ TPS/User**)") and per-platform
recipes `[L-card]`, and nothing at all is published for Bolt's throughput/latency `[T-card]`.

---

## What I could not determine

1. **Whether the Bolt drafter taps five intermediate layers plus the target's final hidden state, or six
   intermediate layers.** The report's §5.7.1 wording ("five captured target layers, together with the
   final hidden state") is about the *Super* drafter; the Bolt config lists six `target_layer_ids`
   `[1,18,35,51,68,85]` inside an 88-layer target whose last layer is 87, and `fc` provably consumes
   6 × 4096 `[D-cfg][D-hdr][T-card config]`. Nothing in the card, config or report resolves which
   convention Bolt uses.
2. **The Bolt drafter's real sliding-window regime.** Config/card say a 1024-token per-head window, while
   the report serves the Super drafter with 8,192-token SWA in every drafter layer to avoid
   out-of-training-distribution distances. Whether Bolt's drafter was trained at 1024 and that is correct
   at serving, or the card is stale, is unstated.
3. **Any vLLM flag list specific to `dflash`/LiLiCorr beyond the `--speculative-config` JSON**
   (`num_speculative_tokens: 7`, `rejection_sample_method: "standard"`, `attention_backend: FLASHINFER`,
   `draft_load_config.load_format: "auto"`), and any per-runtime acceptance number for Bolt — the card
   says explicitly its numbers "are not attributed to a specific serving runtime".
4. **Whether Lightning's NVFP4 release really runs W4A4 anywhere.** §4 says "W4A16 regardless of GPU SKU";
   the Lightning card's hardware matrix shows GB200 with compute path `FP4`; no text reconciles these.
5. **Which Super recovery figure is the release number** (99.62 % in §4.1, 99.52 % in the conclusion and
   Table 18, 99.36 % for the NVFP4-KV variant).
6. **Any expert-residency/offload documentation for these models** — the answer to the question we care
   about most: the tutorial README has nothing, and no other fetched source (target cards, report)
   documents host-RAM expert placement. It does not appear to exist publicly.
7. **Any technical report for Nemotron 3.5 Bolt** — the report covers Super and Lightning only; Bolt
   exists as model cards (24GB MIXED_PRECISION/GGUF/BF16/LiLiCorr), and the Model-Optimizer tutorial
   named in the task is about the *Super VL/Omni* W4A4 pipeline, not Bolt.
8. The LiLiCorr paper itself (arXiv 2608.20530) was not fetched; §5.7.1 of the report was used instead.

#
ModelOpt: recipes, per-tensor layout, and the HF/Megatron naming table

---

# ModelOpt NVFP4/FP8 export format for NemotronH checkpoints (factual reference)

Scope: what a ModelOpt (NVIDIA/Model-Optimizer `main`, read 2026-10-15) export puts on disk for a
NemotronH hybrid Mamba/MoE/attention checkpoint with **mixed NVFP4 + FP8**, and what our
`import_encoded` loader will see. Every claim cites the file, commit, or released checkpoint it
came from. Sources are of three kinds:

- **MO:<path>** — NVIDIA/Model-Optimizer at `main` (fetched with `gh api` REST; GraphQL-backed `gh`
  subcommands are broken on this box).
- **HF:<repo>/<file>** — released checkpoint file (safetensors index JSON, safetensors header, or
  config JSON, fetched by HTTPS).
- **commit 92f512eb** — `NVIDIA/Model-Optimizer` PR #2794.

---

## 0. Bottom line

- A ModelOpt NVFP4 module on disk is **three or four tensors**: `weight` (U8, two E2M1 nibbles per
  byte), `weight_scale` (F8_E4M3, one per 16-element block along the input dim), `weight_scale_2`
  (F32 scalar, per-tensor), and — only when activations are quantized — `input_scale` (F32 scalar).
- Dequant is `e2m1(nibble) * weight_scale * weight_scale_2` (MO:modelopt/torch/export/unified_export_hf.py:790
  comment; MO:modelopt/torch/quantization/qtensor/nvfp4_tensor.py:345 `dequantize`). The two scale
  levels compose to an effective per-block scale of `block_amax / 6`.
- An FP8 module is `weight` (F8_E4M3) + `weight_scale` (F32 **per-tensor**) + `input_scale` (F32
  per-tensor) ⇒ W8A8 per-tensor. `input_scale` is the activation-amax signal; `weight_scale_2` is
  **never** present on an FP8 module.
- The descriptor (`hf_quant_config.json`, and `config.json["quantization_config"]`) for a mixed
  export is `quant_algo: MIXED_PRECISION` + a **full per-module map** plus an `exclude_modules`
  list; there is no compact "recipe" description of which module is which format.
- **No ModelOpt recipe, model folder, or model_type folder targets Bolt** ("Nemotron-3.5-Bolt-74B-A9B"):
  see §1. The Bolt releases that exist are not NVFP4/FP8 (§1).
- MTP tail: present in Lightning-NVFP4 (BF16, excluded) and Super-VL-NVFP4 (NVFP4 + calibrated FP8
  KV); **absent** from Bolt (both BF16 and 24GB).

---

## 1. Which recipes target NemotronH / Bolt / Lightning

MO:modelopt_recipes/models/nvidia/ contains exactly five checkpoint folders:
`NVIDIA-Nemotron-3-Nano-4B-BF16`, `NVIDIA-Nemotron-3-Super-120B-A12B-BF16`,
`NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16`, `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`,
`NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`. MO:modelopt_recipes/model_type/ has
`nemotron_llama` and `nemotron_vl` only — **no `nemotron_h`** folder.

| Checkpoint folder | Recipe file | Scheme (per MO:modelopt_recipes/ptq.md "Checkpoint mirrors") |
|---|---|---|
| NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 | `ptq/nvfp4-mse.yaml` | routed **and** shared experts NVFP4 W4A4 block-16, static weight scales, dynamic NVFP4 activations; attention q/k/v/o + Mamba in/out_proj + lm_head FP8; BF16 latent MoE/router/embeddings/vision; FP8 KV (backbone constant amax, MTP calibrated); MSE + FP8-scale-sweep calibration |
| NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | `ptq/nvfp4-mse.yaml`, `ptq/nvfp4-max-calib.yaml` | routed experts NVFP4 W4A4; shared experts + Mamba in/out_proj FP8; KV FP8; attention q/k/v, MTP head, lm_head, latent MoE, conv1d **BF16** |
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 | `ptq/w4a16_nvfp4_4o6.yaml` | routed experts, shared experts **and lm_head** NVFP4 **W4A16** with four_over_six; Mamba in/out_proj weights+inputs FP8; KV FP8; attention BF16; `mtp.*` disabled (BF16) |
| NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 | `ptq/nvfp4-4o6.yaml` | Super-style map, routed experts NVFP4 4/6, dynamic NVFP4 activations |
| NVIDIA-Nemotron-3-Nano-4B-BF16 | `ptq/nvfp4_w4a16.yaml` | GGUF Q4_K_M mirror, per-layer NVFP4/FP8 |

**Bolt is not mentioned anywhere in the repo.** `gh api search/code` for `Bolt` and
`Nemotron-3.5-Bolt` in NVIDIA/Model-Optimizer returns `total_count: 0`; `ptq.md` (639 lines,
grep case-insensitive for `nemotron|bolt|lightning|super.vl|mamba`) has no Bolt line. The Super-VL
recipe header claims generality anyway (MO:.../Super-VL-120B-A12B-BF16/ptq/nvfp4-mse.yaml):
*"nothing here is specific to this release beyond the calibration choices, so the rules apply to
NemotronH hybrids generally. The MTP rules are inert on a checkpoint with no MTP tail."*

**The released Bolt quant is not NVFP4/FP8.** `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB`
(hf_quant_config.json, producer modelopt 0.37.0) is `MIXED_PRECISION`, `kv_cache_quant_algo: null`,
with 27,728 layers `IQ2_XXS`, 113 layers `Q8_0`, 80 layers `W4A16_NVFP4`:
- experts and shared experts → IQ2_XXS (self-contained GGML blocks, `weight` only: measured U8
  `[2048, 4, 66]` for one expert up_proj; descriptor `{group_size 256, effective_bits 2.0625,
  block_payload_bytes 66, packing: ggml}`);
- `mixer.in_proj`/`mixer.out_proj` → **W4A16_NVFP4** (U8 `[18560, 2048]` + F8_E4M3
  `[18560, 256]` + F32 `weight_scale_2 []`);
- `lm_head`, `mixer.fc1_latent_proj`/`fc2_latent_proj`, attention q/k/v/o → Q8_0;
- conv1d, norms, embeddings, router gate, shared experts' scales: BF16/F32, unquantized.

There is **no NVFP4/FP8 Bolt release from NVIDIA** on the Hub (HF API search for
`Nemotron-3.5-Bolt` lists only `-24GB`, `-24GB-LiLiCorr`, `-24GB-GGUF`, `-BF16` and third-party
GGUF/MLX conversions). The mixed NVFP4+FP8 NemotronH releases are
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` (text) and
`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` (omni). Everything below is measured on
those two (+ Bolt for contrast).

---

## 2. What the imported numerics configs mean

MO:modelopt_recipes/configs/numerics/*.yaml (body stripped to the fields; all are
`modelopt.torch.quantization.config.QuantizerAttributeConfig`):

| Import | Contents | Meaning |
|---|---|---|
| `nvfp4` | `num_bits: e2m1`, `block_sizes: {-1: 16, type: dynamic, scale_bits: e4m3}`, `effective_bits: 4.5` | NVFP4 **activations**: per-16-block E2M1 with E4M3 block scales computed at runtime from the *measured per-tensor* amax |
| `nvfp4_static` | `num_bits: e2m1`, `block_sizes: {-1: 16, type: static, scale_bits: e4m3}` | NVFP4 **weights** with **calibrated/per-block pinned** amax (`global_amax` + per-block `_amax` from MSE), not recomputed at export |
| `nvfp4_four_over_six` | `nvfp4_static` + `four_over_six: true` | per-block amax chosen by MSE between M=6 (×1.0) and M=4 (×1.5); E4M3 scale normalised to **256** instead of 448 |
| `fp8` | `num_bits: e4m3`, `axis:` (null) | **per-tensor** FP8 E4M3 weight and activation quantizer |
| `configs/ptq/units/kv_fp8_cast` | `fp8` + `use_constant_amax: true` | FP8 KV cast at a fixed range (no `_amax` buffer → no `k_scale`/`v_scale` written) |

Constants: `E2M1_MAX = 6.0`, `E4M3_MAX = 448.0`, `E4M3_MAX_46 = 256.0`
(MO:modelopt/torch/quantization/utils/numeric_utils.py:39-41,181-184).

Super-VL's `nvfp4-mse.yaml` imports **only** `configs/numerics/*` (`nvfp4`, `nvfp4_static`, `fp8`)
— no `configs/ptq/units/` snippet — and says so in a comment: *"Only configs/numerics/\* is
imported, so every rule that applies to this model is visible in this file."* Its `quant_cfg` is an
ordered list where the **last match wins**, starting with `'*': enable: false`. The Lightning
recipe imports `nvfp4_four_over_six` + `fp8` and sets `algorithm: {method: mse, start_multiplier:
1.0, stop_multiplier: 1.5, step_size: 0.5}` (= candidates [1.0, 1.5]).

---

## 3. Exact on-disk tensor layout (measured)

Numbers below are read from the **safetensors header** of the released files (dtype + shape), and
the logit tensor set from `model.safetensors.index.json`.

### 3.1 NVFP4 module — `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` (W4A16)

| Tensor | dtype | shape (measured) | Meaning |
|---|---|---|---|
| `....experts.0.up_proj.weight` | U8 | `[1856, 1344]` | packed E2M1, last dim = `in/2` |
| `....experts.0.up_proj.weight_scale` | F8_E4M3 | `[1856, 168]` | per-16-block scale, dim = `in/16` |
| `....experts.0.up_proj.weight_scale_2` | F32 | `[]` | per-tensor global scale |
| (`input_scale`) | — | **absent** | activation quantizer disabled ⇒ W4A16 |

Same shape rule for `down_proj` (`weight U8 [2688, 928]`, `weight_scale [2688, 116]`), shared
experts (`[3712, 1344]` / `[3712, 168]`) and `lm_head` (`U8 [131072, 1344]`, `F8_E4M3 [131072,168]`,
`F32 []`). `hidden_size 2688`, `moe_intermediate_size 1856`, `moe_shared_expert_intermediate_size
3712` (HF config.json).

Super-VL (NVFP4 W4A4) additionally writes `input_scale` on the same modules — measured
`language_model.backbone.layers.1.mixer.experts.0.up_proj`: `weight U8 [2688, 512]`,
`weight_scale F8_E4M3 [2688, 64]`, `weight_scale_2 F32 []`, `input_scale F32 []`
(hidden 4096 → 512 packed bytes, 64 blocks of 16). Shared experts likewise
(`weight U8 [5376, 2048]`, `weight_scale_2 F32 []`, `input_scale F32 []`).

### 3.2 Nibble packing and the E2M1 codebook

MO:modelopt/torch/quantization/qtensor/nvfp4_tensor.py:350 `_unpack_tensor`:
`unpacked[..., 1::2] = byte >> 4`, `unpacked[..., 0::2] = byte & 0x0F` — **low nibble is the
even-index element**. Codebook (same file, lines 26-27), sign in bit 3:
`[0, .5, 1, 1.5, 2, 3, 4, 6, -0, -.5, -1, -1.5, -2, -3, -4, -6]` for codes 0..15.

### 3.3 FP8 module — measured on the same checkpoint

`backbone.layers.0.mixer.in_proj` (and `.out_proj`):

| Tensor | dtype | shape |
|---|---|---|
| `weight` | **F8_E4M3** | `[10304, 2688]` |
| `weight_scale` | F32 | `[]` (per-tensor) |
| `input_scale` | F32 | `[1]` |

No `weight_scale_2`. Note the FP8 weight is stored as a real `F8_E4M3` safetensors tensor (not an
int8 view); the `int8` packing mentioned in MO:modelopt/torch/export/quant_utils.py:975
(`from_quantized_weight`) is a read-back path, not what these files contain.

### 3.4 KV cache

MO:modelopt/torch/export/quant_utils.py:992-997 `_KV_CACHE_REPLACEMENTS` renames
`k_bmm_quantizer._amax` → `k_proj.k_scale` and `v_bmm_quantizer._amax` → `v_proj.v_scale` (plus
`_bias_value` → `k_proj.k_bias`/`v_proj.v_bias` for affine KV). So the KV scales are written **as
children of the k/v projection modules**, e.g. `backbone.layers.5.mixer.k_proj.k_scale` F32 `[1]`.
Measured: Lightning has **6** attention layers (`layers_block_type` attention at
5,12,19,26,33,42) and 6 `k_scale` + 6 `v_scale`; FP8 KV scales are clamped to ≥ 1.0
(MO:modelopt/torch/export/quant_utils.py:296-340 `get_kv_cache_scaling_factor`).

With `use_constant_amax: true` there is no `_amax` buffer (MO:modelopt/torch/quantization/nn/modules/tensor_quantizer.py:256-261,738-741),
so **no `k_scale`/`v_scale` is exported at all** — measured on Super-VL: backbone attention has
`k_proj`/`v_proj` with `weight` only, while MTP has `k_proj.k_scale`/`v_proj.v_scale`. That is the
asymmetry the Super-VL recipe describes ("the backbone casts at a fixed range … MTP stays
calibrated and keeps its pair").

### 3.5 Non-quantized tensors (loader must not expect scales)

Lightning, measured: `conv1d.weight` BF16 `[6144,1,4]`, `A_log` BF16 `[64]`, `D` BF16 `[64]`,
`norm.weight`/`norm_f.weight` BF16, `embeddings.weight` BF16 `[131072,2688]`, attention
`q/k/v/o_proj.weight` BF16, all `mtp.*` weights BF16. The **router is FP32** in Lightning
(`mixer.gate.weight` F32 `[128,2688]`, `gate.e_score_correction_bias` F32 `[128]`) but BF16 in
Super-VL (`language_model.backbone.layers.1.mixer.gate.weight` BF16 `[512,4096]`) — do not assume
the router dtype, read it.

---

## 4. How the scale levels compose (exact formulas)

MO:modelopt/torch/quantization/qtensor/nvfp4_tensor.py:

- `weight_scale_2` = `global_amax / (E2M1_MAX * m)` where `m = fp8_max_for_normalization(q)` = 448,
  or **256** when `four_over_six` (lines 89-110, 181-184; static quantizers use `global_amax`,
  dynamic ones `_amax`).
- per-block `weight_scale` = `per_block_amax / 6` rescaled by `m / (global_amax/6)` and clamped to
  `[2^-9, 448]` before the E4M3 cast (lines 131-165; zero block → 1.0). The dynamic path computes
  `per_block_amax / (6 * weight_scale_2)` (lines 170-200) — algebraically the same value.
- **Composition:** `weight_scale * weight_scale_2 = per_block_amax / 6` in both paths (up to the
  E4M3 rounding of the block scale, ≤ half-ULP). For `four_over_six` the MSE-selected multiplier is
  folded into `per_block_amax`, so the effective scale is `amax/6` for M=6 blocks and `amax/4` for
  M=4 blocks — that is what "4 over 6" names (MO:.../nvfp4_four_over_six.yaml; Lightning recipe
  comment: *"MSE selects per-block between M=6 (keep amax) and M=4 (amax x 6/4)"*).
- **Activation side (NVFP4)**: `input_scale = amax_act / (maxbound * 448)` = `amax_act/2688`
  (nvfp4_tensor.py:211-228 `get_activation_scaling_factor`). MO:tensor_quantizer.py:1116-1121
  states this explicitly: *"Dynamic block quantizers keep a per-tensor amax (the NVFP4 second-level
  scale)"* — i.e. `input_scale` is the baked **second-level/global** activation scale; the per-16
  block activation scales are computed at runtime.
- **FP8**: `weight_scale = amax/448` per tensor (MO:modelopt/torch/export/unified_export_hf.py:684-700,
  `amax.item() / maxbound`, `maxbound = 448` for (4,3) per tensor_quantizer.py:315-321);
  `input_scale = amax_act/448` (unified_export_hf.py:714-746 via `get_activation_scaling_factor` →
  `get_scaling_factor` → `amax/maxbound`, quant_utils.py:163).

---

## 5. Descriptors: `hf_quant_config.json` and `config.json`

`hf_quant_config.json` is written verbatim by the exporter (MO:unified_export_hf.py:1580-1592) and,
for a mixed model, has exactly this shape (measured on Lightning):

```json
{"producer": {"name":"modelopt","version":"0.44.0rc5"},
 "quantization": {"quant_algo": "MIXED_PRECISION", "kv_cache_quant_algo": "FP8",
   "quantized_layers": {"<module path>": {"quant_algo": "...", "group_size": 16}},
   "exclude_modules": ["<module path>", ..., "mtp*"]}}
```

- `quant_algo` becomes `MIXED_PRECISION` whenever more than one distinct per-layer format exists
  (MO:modelopt/torch/export/quant_utils.py:794-796).
- Lightning counts: 5,981 `quantized_layers` = **5,935 × `{"quant_algo":"W4A16_NVFP4","group_size":16}`**
  + **46 × `{"quant_algo":"FP8"}`** (23 `in_proj` + 23 `out_proj`); `exclude_modules` has 72
  entries = `backbone.embeddings` + 23 `mixer.conv1d` + 23 `mixer.gate` + 24 attention
  `q/k/v/o_proj` + the wildcard `mtp*`.
- Super-VL counts differ in the algorithm name only: `{"quant_algo":"NVFP4","group_size":16}` for
  41,040 expert/shared-expert modules and `{"quant_algo":"FP8"}` for 113 `in_proj`/`out_proj` +
  attention q/k/v/o + `lm_head`; `ignore` additionally lists the latent projections
  (`mixer.fc1_latent_proj`, `mixer.fc2_latent_proj`), `conv1d`, `gate`, embeddings and the vision
  prefixes.
- **W4A16_NVFP4 vs NVFP4 is decided by the activation quantizer, not by the weights**
  (MO:quant_utils.py:555-570 format inference): e2m1 weights + block 16 + `scale_bits (4,3)` with the
  input quantizer absent/disabled ⇒ `W4A16_NVFP4`; with the input quantizer enabled ⇒ `NVFP4`
  (W4A4). So a descriptor saying `NVFP4` means both `weight_scale_2` **and** `input_scale` are on
  disk, while `W4A16_NVFP4` means `weight_scale_2` only.
- **An FP8 module's `input_scale` means activations really are quantized** (W8A8, static per-tensor).
  Its absence means activations stay BF16 even though the weight is FP8 — exactly the failure the
  validation skill flags: *"those input quantizers collect no amax and the export drops the input
  scale while still writing an FP8 weight scale — a layer that claims W8A8 with no activation
  scale"* (MO:plugins/modelopt/skills/ptq/references/checkpoint-validation.md; also the
  Super-VL recipe's audio-branch comment).

`config.json["quantization_config"]` is the same information converted to a compressed-tensors-ish
form (MO:modelopt/torch/export/convert_hf_config.py:150-322): `quant_method: "modelopt"`,
`quant_algo`, `kv_cache_scheme`, `ignore` (= `exclude_modules`), `config_groups` with `targets`,
and the per-layer `quantized_layers` map. Measured on Lightning: two groups —
`group_0` FP8 (`weights` + `input_activations`, `num_bits 8`, `dynamic: false`, 46 targets) and
`group_1` NVFP4 **weights only** (`num_bits 4`, `group_size 16`, `dynamic: false`, 5,935 targets);
plus `kv_cache_scheme: {"dynamic": false, "num_bits": 8, "type": "float"}`. On Super-VL
`group_1` has **both** `weights` and `input_activations` at `num_bits 4, group_size 16`
(41,040 targets) — i.e. the group entry mirrors the format inference of §5, and its
`"dynamic": false` refers to the per-tensor second-level scale being baked, not to per-block
activation scaling (MO:convert_hf_config.py:53-70 is the literal mapping).

HF file sizes (index `metadata.total_size`): Lightning 21,559,589,596 B / 18,487 tensors;
Super-VL 75,010,189,968 B / 169,505 tensors; Bolt-24GB 24,079,678,824 B / 28,885 tensors.

---

## 6. Vision tower and MTP tail per release

| Release | vision tower | MTP tail |
|---|---|---|
| Lightning-30B-A3B-NVFP4 | **absent** (text-only: keys start `backbone.` / `lm_head` / `mtp`, no `language_model.` prefix, no `vision_model.*`) | **present, 100% BF16**: 270 tensors = `mtp.layers.0` attention (q/k/v/o) + `mtp.layers.1` MoE (128 experts up/down, shared experts up/down, gate, latent absent) + `norm`/`enorm`/`hnorm`/`eh_proj`/`final_layernorm`; no scale tensor of any kind; declared excluded by `mtp*` |
| Super-VL-120B-A12B-NVFP4 | **present, 100% BF16, no scales**: 391 `vision_model.radio_model.*` tensors (patch_generator.embedder/video_embedder/cls_token, blocks.*.attn.{qkv,proj}, mlp.{fc1,fc2}, norm1/norm2, input_conditioner) + `mlp1.0/1/3` + `vision_projector.vision_final_layernorm.{weight,bias}`; also one non-weight `vision_model.summary_idxs`. No `sound_*`/`audio_*` tensor at all (config declares `sound_config: null`) | **present, quantized**: `language_model.mtp.layers.0` attention (BF16 weights **+ `k_proj.k_scale`/`v_proj.v_scale`** ⇒ calibrated FP8 KV) and `layers.1` MoE (512 experts up/down **and** shared experts, each with `weight`/`weight_scale`/`weight_scale_2`/`input_scale`), with `fc1_latent_proj`/`fc2_latent_proj` and `gate` BF16 |
| Bolt-74B-A9B (BF16 and 24GB) | present (same `vision_model.radio_model.*` 391 + `mlp1.0/1/3`), but **no `vision_projector.*`** | **absent**: BF16 `llm_config.num_nextn_predict_layers = 0`, `mtp_layers_block_type = []`, and no `mtp.*` key in either index |

The canonical MTP-presence test in ModelOpt is structural, not config-based
(MO:modelopt/torch/models/nemotron_h/mtp.py:87-113): a checkpoint has an MTP tail iff it has
`<prefix>mtp.layers.{0..N-1}.` with a single prefix in `("", "language_model.")`, contiguous layer
indices, `mtp.layers.0.eh_proj.weight` and `mtp.layers.<last>.final_layernorm.weight` present.
Both Lightning and Super-VL satisfy this with N=2; Bolt has none.

---

## 7. Module names the loader will see

### 7.1 Hugging Face spelling (what is in the safetensors)

Measured from the released indexes (patterns normalized: `layers.N`, `experts.E`). Text models use
the `backbone.` root; the omni wrapper prepends `language_model.`
(MO:modelopt/torch/export/plugins/mcore_nemotron_omni.py:27-30,
`NEMOTRON_H_OMNI_VISION_PREFIXES = ("vision_model.", "vision_projector.", "mlp1.")`).
Note also MO:modelopt/torch/models/nemotron_h/modeling_ptq.py:26-39: *"Custom remote-code checkpoint
uses model.backbone.layers; native transformers NemotronHModel uses model.model.layers"* — the
released NVFP4 files use `backbone.layers`.

| Role | HF name (per layer type) |
|---|---|
| embedding / final norm / head | `backbone.embeddings.weight`, `backbone.norm_f.weight`, `lm_head.weight` |
| per-layer norm | `backbone.layers.N.norm.weight` |
| Mamba (`layers_block_type: mamba`) | `...N.mixer.{A_log, D, dt_bias, conv1d.weight, conv1d.bias, norm.weight, in_proj, out_proj}` |
| attention (`attention`) | `...N.mixer.{q_proj, k_proj, v_proj, o_proj}` (+ `k_proj.k_scale`, `v_proj.v_scale` when KV is calibrated) |
| MoE (`moe`) | `...N.mixer.{gate.weight, gate.e_score_correction_bias, experts.E.up_proj, experts.E.down_proj, shared_experts.up_proj, shared_experts.down_proj, fc1_latent_proj, fc2_latent_proj}` (`fc*_latent_proj` only when `moe_latent_size` is set; mamba layers also carry `mixer.up_proj`/`down_proj`) |
| MTP | `mtp.layers.{0,1}.{norm.weight, enorm.weight, hnorm.weight, eh_proj.weight, final_layernorm.weight, mixer.<same as above>}` |
| vision (omni only) | `vision_model.radio_model.*`, `mlp1.{0,1,3}.weight`, `vision_projector.vision_final_layernorm.*` |

Lightning: 23 mamba + 23 moe + 6 attention (`layers_block_type` from HF config.json; the moe layers
carry `moe_latent_size: null` so no `fc*_latent_proj`). Super-VL: 40 mamba + 40 moe + 8 attention,
`moe_latent_size: 1024`, 512 routed experts, 1 shared expert. Bolt: 40 MoE layers with
**nested, per-layer expert counts** — 12 layers × 240, 6 × 304, 9 × 368, 10 × 432, 3 × 496 experts
(`experts.{E}` is a prefix range, E < N_layer; 13,824 expert matrices total = 6,912 up + 6,912
down, which is why the expert *tensor* count is what identifies the layout, not `n_routed_experts`),
8 attention layers, 40 Mamba layers, `moe_latent_size: 1024`. Bolt's HF `config.json` sits behind
an omni wrapper: per-LLM fields live under `llm_config`, and a config check must read
`llm_config.n_routed_experts`/`llm_config.num_hidden_layers`, not the top level.

### 7.2 Megatron-Core spelling

Authoritative mapping table MO:modelopt/torch/export/plugins/mcore_nemotron.py
(`nemotron_h_causal_lm_export` / `..._import`):

| MCore | HF |
|---|---|
| `embedding` (word_embeddings) | `backbone.embeddings.` |
| `final_norm` | `backbone.norm_f.` |
| `output_layer` | `lm_head.` |
| `mlp.linear_fc1` / `mlp.linear_fc2` | `backbone.layers.{}.mixer.up_proj.` / `down_proj.` |
| `self_attention.linear_qkv` | `backbone.layers.{}.mixer.{q,k,v}_proj` (QKVSlicing/-Merging) |
| `self_attention.linear_proj` | `backbone.layers.{}.mixer.o_proj.` |
| `core_attention` (KV scales) | `k_scale_name: "k_proj.k_scale"`, `v_scale_name: "v_proj.v_scale"` under `backbone.layers.{}.mixer.` |
| Mamba `mixer.in_proj` / `mixer.out_proj` / `A_log` / `D` / `dt_bias` / `conv1d` | identical names, under `backbone.layers.{}.mixer.` |
| `mlp.router` | `backbone.layers.{}.mixer.gate.` with `mapping {"expert_bias": "e_score_correction_bias"}` |
| `mlp.experts.local_experts.{E}.linear_fc1/linear_fc2` | `...mixer.experts.{E}.up_proj` / `.down_proj` |
| `mlp.experts.linear_fc1/linear_fc2` (TEGroupedMLP, fused) | same HF per-expert names, sliced/merged |
| `mlp.shared_experts.linear_fc1/linear_fc2` | `...mixer.shared_experts.up_proj` / `.down_proj` |
| `mlp.fc1_latent_proj` / `fc2_latent_proj` | `...mixer.fc1_latent_proj.` / `.fc2_latent_proj.` (latent MoE) |
| `mtp.enorm` / `hnorm` / `eh_proj` / `final_layernorm` | `mtp.layers.{}.enorm.` / `hnorm.` / `eh_proj.` / `final_layernorm.` |
| MTP inner blocks | `mtp.layers.{}.mtp_model_layer.layers.{}` reusing the backbone rules with `is_mtp=True` (recipe: *"Megatron-Core MTP block 1 is mtp.layers.0.mtp_model_layer.layers.1"*) |

Recipe-level MCore spellings in the quant configs match this: `*mlp.experts.local_experts.*.linear_fc{1,2}`,
`*mlp.shared_experts.linear_fc{1,2}`, `*self_attention.linear_qkv/linear_proj`,
`*output_layer.{weight,input}_quantizer`, `*mtp_model_layer.layers.1.mlp.experts.*`.

---

## 8. commit 92f512eb — "Fix native NemotronH export config mappings" (#2794)

Files: `CHANGELOG.rst` (+1), `modelopt/torch/utils/plugins/hf_checkpoint_utils.py` (+12),
`tests/unit/torch/utils/test_hf_checkpoint_utils.py` (+72). **It maps no tensor or module names at
all.** The whole code change is inside `sanitize_hf_config_for_deployment`:

```python
model_config = getattr(model, "config", None)
auto_map = config_data.get("auto_map")
if isinstance(auto_map, dict) and type(model_config).__module__.startswith(
    "transformers.models.nemotron_h."
):
    for auto_class in ("AutoConfig", "AutoModel", "AutoModelForCausalLM"):
        auto_map.pop(auto_class, None)
    if not auto_map:
        config_data.pop("auto_map")
```

- Why: PTQ selects the built-in Transformers implementation, which serialises `layers_block_type`,
  but the retained `auto_map` sent trusted config reloads to a legacy class exposing that field as a
  read-only property → `AttributeError: property 'layers_block_type' of 'NemotronHConfig' object has
  no setter`.
- It preserves the tokenizer mapping, genuine remote-code exports, and copy sidecars (the test
  asserts `export/configuration_legacy.py` is byte-identical to the source and that `AutoTokenizer`
  survives).
- Adjacent behaviour in the same function, **unchanged** by this commit: trimming trailing
  `layer_types` entries when the excess is exactly `num_nextn_predict_layers` (MTP/NTP tail) and
  adding llama3 `rope_theta`.
- Test asserts the exported config reloads as `transformers.NemotronHConfig` with
  `layers_block_type` intact and `quantization_config.quant_algo == "FP8"` +
  `kv_cache_scheme {"dynamic": False, "num_bits": 8, "type": "float"}`.
- Cross-check on the releases: Lightning-NVFP4 (producer 0.44.0rc5) `config.json` has **no**
  `auto_map`; Super-VL-NVFP4 (0.37.0, pre-fix) still carries `auto_map` with
  `configuration_nemotron_h_omni.*`, and Bolt (0.37.0) ships `configuration_nemotron_h*.py` /
  `modeling_nemotron_h*.py` sidecars next to its `auto_map`.

---

## 9. Things I could not determine

1. **Which recipe NVIDIA would use for a Bolt NVFP4 export.** No Bolt entry exists in
   Model-Optimizer (search + ptq.md), so Bolt's NVFP4/FP8 layer map (if one is ever published) can
   only be inferred from the Super-VL/Lightning families, not read off. Bolt's GGML release uses a
   different scheme entirely (§1).
2. **Whether the released Super-VL NVFP4's vision sidecar files match the recipe's exclusions
   exactly** — the recipe has explicit `*vision*`/`*radio_model*`/`mlp1.*` disables; I verified the
   *tensors* carry no scales, not that every disable pattern fired.
3. **Serving-side semantics of `input_scale` for NVFP4** (whether vLLM/SGLang use it as the
   per-tensor second-level activation scale and compute per-16 block scales at runtime). The claim
   above rests on MO:tensor_quantizer.py:1116-1121 and the config-group shape, not on a loader read.
4. **Why Bolt-24GB ships no `vision_projector.*`** tensors although Super-VL does, and how Bolt's
   `llm_config.block_configs` / `blockwise_members` map onto the observed nested expert counts
   (12 layers × 240, 6 × 304, 9 × 368, 10 × 432, 3 × 496) — the counts are measured, the config-level
   rule that produces them is not decoded.
5. **The exact producer version → format-version mapping.** Models produced by 0.37.0 (Super-VL,
   Bolt) and 0.44.0rc5 (Lightning) use the same tensor names in everything I measured, but I did not
   audit the CHANGELOG for intermediate format breaks; do not key parsing on `producer.version`.
6. **The DSpark/DFlash speculative variants** (`...-NVFP4-DSpark`, `...-NVFP4-DFlash`,
   `modelopt_recipes/models/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/speculative_decoding/dspark_warmstart.yaml`)
   — not examined; they may add or reshape MTP tensors.
7. **Tied-weight handling for `lm_head`.** Lightning's `lm_head` has its own NVFP4 scales and
   `tie_word_embeddings: false`; I did not check what the exporter does when embeddings are tied
   (MO:unified_export_hf.py has a name-based tied dedup over `weight`/`weight_scale`/`weight_scale_2`/
   `input_scale`, but I did not observe a tied NemotronH export).
