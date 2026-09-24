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

**Status:** execution is implemented but not yet measured on hardware. This card will list the
artifact size, checksums, accuracy and throughput once they have been measured.

## Representation

| Weights | Stored as | Runtime residency |
|---|---|---|
| Routed experts (48 × 512, plus the MTP layer) | NVFP4, imported codes and scales; MTP re-encoded from block FP8 | pinned Host, fetched into a device expert cache |
| N-gram PLE table (320M × 160) | FP8 rows with BF16 multipliers | page-cache mapped or streamed from NVMe, gathered on the Host per token |
| Attention, GDN, hyper-connection, shared expert, PLE projections | Q8 | device |
| Token embedding / output head | Q8 / Q6 | device |
| Routers, shared-expert gates, norms, small vectors | BF16/FP32 direct | device |

## Requirements

- one RTX 5090 (sm_120a) and CUDA 13.3;
- about 80 GB of host RAM for the ~69 GB of pinned experts with `--ngram-residency stream`
  (the n-gram table is read from NVMe); about 128 GB to keep the 52 GB table in the page cache
  (`mapped`, chosen automatically when memory allows);
- KV storage `bf16` or `fp8`; speculative decoding `--spec mtp`.

## Serve

```bash
./build/apps/ninfer-serve qwen3_8_flash_next_nvfp4.ninfer \
  --model-id qwen3.8-flash-next --max-concurrency 2 \
  --max-context 229376 --kv-capacity 458752 --kv-dtype fp8 \
  --expert-cache auto --spec mtp
```

`--expert-cache auto` plans the KV pool first, then gives the remaining device memory to the
routed-expert cache. The reasoning (`<think>`) and `qwen3_coder` tool-call parsers are built in.

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
