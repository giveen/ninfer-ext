---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-Flash-Next
  - nvidia/Qwen3.8-Flash-Next-NVFP4
base_model_relation: quantized
tags:
  - ninfer
  - qwen3.8
  - nvfp4
  - fp8
  - moe
  - expert-offload
  - blackwell
  - multimodal
  - conversational
  - cuda
  - rtx-5090
---

# Qwen3.8-Flash-Next-NInfer

A NInfer v3 artifact of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
(`Qwen4ExpForCausalLM`), converted from
[nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) with the
`qwen3_8_flash_next_nvfp4` recipe. It contains Text, MTP and Vision.

**Status:** measured on the development RTX 5090. The artifact files and `SHA256SUMS` are in this
folder.

## Representation

| Weights | Stored as | Runtime residency |
|---|---|---|
| Routed experts (48 × 512, plus the MTP layer) | NVFP4, imported codes and scales; MTP re-encoded from block FP8 | pinned Host, fetched into a device expert cache |
| N-gram PLE table (320M × 160) | FP8 rows with BF16 multipliers | page-cache mapped or streamed from NVMe, gathered on the Host per token |
| Attention, GDN, hyper-connection, shared expert, PLE projections | Q8 | device |
| Token embedding / output head | Q8 / Q6 | device |
| Routers, shared-expert gates, norms, small vectors | BF16/FP32 direct | device |

## Measured

Development machine: one RTX 5090 (32 GB, sm_120a), CUDA 13.3, `--expert-cache auto`, fp8 KV,
MTP speculative decoding (`--spec mtp`).

| Metric | Value |
|---|---|
| Causal perplexity (`ninfer-ppl-1m-v1`, quick, fp8 KV) | **3.518** |
| Prefill, 262,144-token budget, FP8 KV, `--ngram-residency stream`, single request | **1,090 / 3,666 / 3,498 / 2,908 / 2,218** tok/s at 1.5k / 8k / 64k / 128k / 256k tokens |
| Decode (greedy, MTP K=3) | **81 tok/s** |
| Peak host RSS (`--ngram-residency stream`) | **~65 GiB** |
| Artifact size | **119 GB**, 4 sharded files |

Prefill is fastest at medium prompts: a short prompt is dominated by the fixed per-chunk expert
streaming, and a long one by the QSA selection. It also depends on how the n-gram table is read:
with it mapped through the page cache (`--ngram-residency mapped`, the default when host memory
allows) prefill is about 1.7x faster than the `stream` figures above — 6,250 tok/s at 8k — at the
cost of keeping the ~52 GB table in RAM. Long-context prefill improved 1.48x at 64k, 1.87x at 128k
and 2.53x at 256k over the previous engine, because the QSA block selection now pools each block's
index keys once per select call instead of once per query column (bit-identical). Measured with the
NIAH prompts on the development machine.

## Requirements

- one RTX 5090 (sm_120a) and CUDA 13.3;
- about 70 GB of host RAM (measured ~65 GiB peak RSS) for the pinned experts with
  `--ngram-residency stream`, which reads the n-gram table from NVMe; keeping the ~52 GB table in
  the page cache (`mapped`, chosen automatically when memory allows) needs more RAM and is faster
  once warm;
- KV storage `bf16`, `int8`, `fp8`, `nvfp4` or `k8v4`; speculative decoding `--spec mtp`.

## Serve

```bash
./build/apps/ninfer-serve qwen3_8_flash_next_nvfp4.ninfer \
  --model-id qwen3.8-flash-next --max-concurrency 2 \
  --max-context 229376 --kv-capacity 458752 --kv-dtype fp8 \
  --expert-cache auto --ngram-residency stream --spec mtp
```

`--expert-cache auto` plans the KV pool first, then gives the remaining device memory to the
routed-expert cache. The reasoning (`<think>`) and `qwen3_coder` tool-call parsers are built in.
`--ngram-residency stream` keeps the n-gram table out of RAM (about 65 GiB total); omit it to
map the table through the page cache when host memory allows.

## Convert

```bash
python3 -m tools.convert \
  --model /path/to/Qwen3.8-Flash-Next-NVFP4 \
  --recipe qwen3_8_flash_next_nvfp4 \
  --components text,vision,mtp \
  --resource chat_template.jinja=tools/chat_templates/qwen.jinja \
  --name qwen3.8-flash-next \
  --out qwen3_8_flash_next_nvfp4.ninfer
```

The mathematics is documented in the NInfer
[Qwen4Exp model reference](https://github.com/Neroued/ninfer/blob/master/docs/maintainer/qwen4-exp-model.md).
