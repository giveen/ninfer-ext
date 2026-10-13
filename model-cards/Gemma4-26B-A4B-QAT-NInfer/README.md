---
library_name: ninfer
pipeline_tag: text-generation
inference: false
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
base_model:
  - google/gemma-4-26B-A4B-it
base_model_relation: quantized
tags:
  - ninfer
  - gemma4
  - gemma-4
  - 26b
  - a4b
  - moe
  - qat
  - q4
  - q8
  - text-generation
  - multimodal
  - speculative-decoding
  - blackwell
  - cuda
  - rtx-5090
---

# Gemma 4 26B-A4B IT QAT (NInfer)

A NInfer v3 artifact of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it)
(`Gemma4ForCausalLM` text with the `Gemma4ForConditionalGeneration` vision tower), converted from the
BF16 checkpoint with the `gemma4_26b_a4b_qat` recipe (converter `ninfer-v3`). Contains **Text, Vision
and MTP**; the assistant drafter at this model's geometry is bound with `--spec mtp`.

> Only works with [giveen/ninfer-ext](https://github.com/giveen/ninfer-ext). `.ninfer` is NInfer's own
> artifact format; it is not a Transformers checkpoint, GGUF or safetensors file.

**Status:** measured on the development RTX 5090. Files and `SHA256SUMS` are in this folder.

## Model

A 26B-parameter mixture-of-experts Gemma 4: 30 layers, hidden 2816, **128 routed experts with top-8
routing**, a dense FFN of 2112, and a 262,144-token vocabulary. Attention is hybrid — a 1024-token
sliding window on 25 layers and global attention on layers 5, 11, 17, 23 and 29 (8 KV heads, 2 of them
global, head dim 256 / 512 global, K == V) — so the KV cache is mostly a fixed-size ring and only
those five layers grow with context. The router reads the pre-FFN residual, softmaxes, renormalises
the top-8 to sum to one, and only then applies the per-expert scale. Final logits are softcapped at 30.

## Representation

| Weights | Stored as |
|---|---|
| Routed experts (128 per layer, top-8; `text/layers/*/moe/experts/*`) | **QAT int4**, group-32 learned-scale rows (`q4_g32_fp16_rows`, 181 objects) |
| Attention projections, dense MLP, token embedding and output head | **Q8** group-32 (`q8_g32_fp16`), grouped into one object |
| Vision tower (`vision/`, bound with `--vision`) | **Q8** group-32 (135 objects), BF16 patch and output projections |
| MTP drafter (`mtp/`, bound with `--spec mtp`) | **Q6** g64 (23 objects) |
| Norms, layer scalars, small vectors | BF16 / FP32 direct |

The dense FFN's 2112 and the experts' 704 intermediate widths are zero-padded to 2176 and 768, which
the block-scale layouts require. The padding is exact: GeGLU maps zero to zero, so zero gate/up rows
and their zero down columns contribute nothing.

## Measured

Development machine: one RTX 5090 (32 GB, sm_120a), CUDA 13.3, BF16 KV, greedy, one request unless
noted. The expert banks are ~4.5 bits per weight, so the 24.9B expert elements are about 14 GB of the
file and are read on every token that routes to them.

| Metric | Value |
|---|---:|
| Prefill, 7.8k / 65.8k / 132.7k-token prompt | 6,840 / 4,095 / 2,801 tok/s |
| Decode, same contexts | 317 / 254 / 210 tok/s |
| Decode with MTP (`--spec mtp --draft-tokens 3`) | 1.39x at two lanes, 1.03x at eight, 0.67x at four (acceptance 0.82-0.85); 1.75x on long single-request generations |
| Steady decode, ~300-token prompt, 1 / 2 / 4 / 8 lanes | 330 / 474 / 586 / 675 tok/s aggregate |
| Quality (`ninfer-ppl-1m-v1` full, 1,092,299 tokens, context/stride 4096/2048, BF16 KV) | mean NLL **5.039971** (PPL 154.466) |
| Quality by domain | chinese_reference 5.184, english_long_form 5.948, english_reference 6.642, ninfer_code 2.701 |
| KV storage (separate 40,960-byte slice, mean NLL) | bf16 5.3987, int8 5.4194, fp8 5.4302, nvfp4 5.4501, k8v4 5.4573 |
| Artifact | 17,173,987,328 B (15.99 GiB), 927 objects, 12,522 parameters |

MTP pays where decode is weight-bound and stops paying once the engine already has a full batch. On
this model at a ~300-token context it is **1.39x** at two lanes (487 -> 678 tok/s), flat at eight
(687 -> 706), and *slower* than plain decode at four (606 -> 409, acceptance 0.83) -- the extra draft
passes cost more than the tokens they commit. The 1.75x is long single-request generation
(288 -> 500 tok/s, acceptance 0.87-0.95) and was measured across two sessions rather than in one, so
treat it as indicative; the per-lane ratios are same-session. Gemma 4 runs MTP on the **full** proposal
head only, 1 to 7 draft tokens, and refuses the optimized head, so `--lm-head-draft` must not be
passed for this model.

## Usage

Keep the `.ninfer` and `.conversion.json` together.

```bash
build/apps/ninfer-serve gemma4-26b-a4b-qat/gemma4_26b_a4b_qat.ninfer \
  --model-id gemma4-26b-a4b-qat --max-context 32768 --kv-capacity auto --kv-dtype bf16
```

```bash
hf download jabbatheduck/ninfer-ext-models --include "gemma4-26b-a4b-qat/*" --local-dir models
```

## Limits

- One RTX 5090, one resident model, one to eight startup-fixed active requests.
- **Every lane reserves its full context window**, so context and concurrency trade against device
  memory. Measured on this card: 262,144 tokens on one lane, 131,072 on two, 65,536 on four, 32,768 on
  eight. A window the card cannot hold is refused at startup with the two numbers rather than by an
  allocation failure mid-request.
- KV storage `bf16`, `int8`, `fp8`, `nvfp4` or `k8v4`; `bf16` is the default, and the quantized
  formats cost 0.02-0.06 nats of mean NLL on a 40,960-byte slice. Only the five global layers make a long
  context expensive — the sliding layers' ring does not grow with it.
- Images are supported (`--vision`: one or more images per message, no video). `top_k` is capped at 20.
- The full reference — config, per-layer mathematics, verification record and the open items — is
  [docs/maintainer/gemma4-model.md](https://github.com/giveen/ninfer-ext/blob/main/docs/maintainer/gemma4-model.md).

## License

The quantized weights follow the base model's [Gemma Terms of Use](https://ai.google.dev/gemma/docs/gemma_4_license).
The quantization format, kernels, and tooling are part of [giveen/ninfer-ext](https://github.com/giveen/ninfer-ext).
