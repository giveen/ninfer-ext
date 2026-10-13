# Gemma 4

Two Gemma 4 models run through the same Engine, CLI and HTTP server as the Qwen models:

| Model | Architecture | Artifact |
|---|---|---|
| Gemma 4 31B IT | `Gemma4ForCausalLM` | [`gemma4-31b-it-m1/`](https://huggingface.co/jabbatheduck/ninfer-ext-models/tree/main/gemma4-31b-it-m1) |
| Gemma 4 26B-A4B IT | `Gemma4ForCausalLM`, 128 routed experts top-8 | [`gemma4-26b-a4b-qat/`](https://huggingface.co/jabbatheduck/ninfer-ext-models/tree/main/gemma4-26b-a4b-qat) |

Both generate, score and serve with chat, tools, thinking with a budget, and images (`--vision`; one
or more images per message, no video). Both ship an MTP drafter, converted with the model.
`ninfer-perplexity` scores either.

## Gemma 4 31B IT M1

Converted from [`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it) with the
`gemma4_31b_m1` recipe: a Q6/Q5 g64 body (173 Q6 and 68 Q5 of the 240 MLP objects, chosen by measured
sensitivity), Q6 attention and head, a BF16 token embedding in pinned Host memory, and Q6 drafters.
About 24.4 GiB.

| Stat | Value |
|---|---|
| Prefill `pp1024` / `pp8192` | 2,476 / 2,189 tok/s |
| Decode `tg128` | 57 tok/s (100-240 with adaptive MTP, depending on the text) |
| Concurrent decode, no MTP (aggregate at 1 / 2 / 4 / 8 requests) | 57.5 / 92.2 / 165.8 / 291.0 tok/s |
| Concurrent decode, MTP 7 drafts (aggregate at 1 / 2 / 4 requests) | 108.1 / 162.3 / 251.4 tok/s |
| Layout KLD vs the BF16 checkpoint | 0.419 (4 x 2048 calibration tokens) |
| Causal perplexity (BF16 KV) | 14.53 (mean NLL 2.676) |

Limits: **BF16 KV only** — no int8, FP8 or NVFP4 KV path, so `ninfer-perplexity` needs
`--kv-dtype bf16`; no prefix reuse and no constrained output; `top_k` is capped at 20. Concurrent
requests decode together in one pass, and with MTP they draft and verify together (eight MTP lanes do
not fit device memory yet). The artifact is published; the [model card](../model-cards/) has the full
layout comparison.

## Gemma 4 26B-A4B IT QAT

Converted from [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) with
the `gemma4_26b_a4b_qat` recipe: 30 layers, hidden 2816, 128 routed experts with top-8 routing, a
dense FFN of 2112, a 262,144-token vocabulary, and hybrid attention — a 1024-token sliding window on
25 layers and global attention on layers 5, 11, 17, 23 and 29. About 16 GiB, 927 objects.

The routed experts are **QAT int4** with group-32 learned-scale rows (181 objects); attention
projections, the dense MLP, the token embedding and the head are Q8 group-32; the vision tower is Q8
with BF16 patch and output projections; the MTP drafter is Q6 g64. The dense FFN's 2112 and the
experts' 704 intermediate widths are zero-padded to 2176 and 768, which the block-scale layouts
require — the padding is exact, because GeGLU maps zero to zero.

| Stat | Value |
|---|---|
| Decode at one lane, 300 / 8k / 31k / 123k tokens of context | 330.5 / 320.0 / 291.0 / **212.5** tok/s |
| Decode at eight lanes, 300 / 8k tokens | 674.9 / 620.7 tok/s |
| Quality (`ninfer-ppl-1m-v1` full, 1,092,299 tokens, ctx/stride 4096/2048, BF16 KV) | mean NLL **5.039971** (PPL 154.466) |
| KV storage cost (40,960-byte slice, mean NLL) | bf16 5.3987, int8 5.4194, fp8 5.4302, nvfp4 5.4501, k8v4 5.4573 |

Per-model charts, including the MTP comparison, are in the
[README's master charts](../README.md#master-charts).

Unlike the 31B, the 26B keeps prefix reuse and all five KV storages, and differs in two further ways:

- **Its KV window is reserved per lane.** Every lane claims a full context window, so context and
  concurrency trade against device memory: measured 262,144 tokens at one lane, 131,072 at two,
  65,536 at four and 32,768 at eight. A window the card cannot hold is refused at startup with the
  two numbers rather than by an allocation failure mid-request.
- **Its MTP runs on the full proposal head only** (1–7 draft tokens); the optimized head is refused,
  so `--lm-head-draft` must not be passed. MTP pays where decode is weight-bound and stops paying
  once the engine has a full batch: on this MoE at four lanes it is *slower* than plain decode
  (606 → 409 tok/s, acceptance 0.83), while at two lanes it is 1.39x and at eight 1.03x. On long
  single-request generations the same drafter reads 1.75x (288 → 500 tok/s at 0.87–0.95 acceptance).

The [26B model card](../model-cards/Gemma4-26B-A4B-QAT-NInfer/README.md) carries the full measured
set, and [the Gemma 4 reference](maintainer/gemma4-model.md) the config, per-layer mathematics and
verification record.
