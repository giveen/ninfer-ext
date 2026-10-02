---
license: apache-2.0
language:
  - en
  - zh
  - code
library_name: ninfer-ext
pipeline_tag: image-text-to-text
base_model:
  - Qwen/Qwen3.8-27B
  - Qwen/Qwen3.8-Flash-Next
  - nvidia/Qwen3.8-Flash-Next-NVFP4
tags:
  - exl3
  - trellis
  - nvfp4
  - moe
  - expert-offload
  - blackwell
  - quantization
  - ninfer
  - cuda
  - text-generation
  - vision
  - speculative-decoding
---

# NInfer models

Quantized artifacts for **NInfer Ext**, a from-scratch C++/CUDA inference engine for maximum
single-GPU performance.

> [!IMPORTANT]
> **These artifacts only work with [giveen/ninfer-ext](https://github.com/giveen/ninfer-ext).**
> They are not Transformers checkpoints, not GGUF, not safetensors weights, and cannot be loaded by
> `transformers`, `vLLM`, `llama.cpp`, or exllamav3. `.ninfer` is NInfer's own artifact format, and the
> EXL3 trellis and NVFP4 layouts are decoded by kernels that live in that repository. Loading them
> anywhere else will fail.

## Contents

### Qwen3.8-27B EXL3

| File | Size | Notes |
|---|---|---|
| [`qwen3_8_27b_exl3_4bpw.ninfer`](./qwen3_8_27b_exl3_4bpw.ninfer) | 15.68 GiB | Text + MTP + Vision, 4.0 bpw body |
| `qwen3_8_27b_exl3_4bpw.ninfer.conversion.json` | 489 KiB | Conversion provenance and per-tensor rates |
| [`qwen3_8_27b_exl3_3p5bpw.ninfer`](./qwen3_8_27b_exl3_3p5bpw.ninfer) | 14.24 GiB | Text + MTP + Vision, 3.5 bpw body |
| `qwen3_8_27b_exl3_3p5bpw.ninfer.conversion.json` | 489 KiB | Conversion provenance and per-tensor rates |

Both are the same model at two points on the size curve. **4.0 bpw is the one to lead with**: it is the
only artifact here that beats both of the engine's other Qwen3.8-27B builds on *both* perplexity and KL
divergence, while being smaller than either (see Quality). 3.5 bpw is the size-optimised tier — best PPL
per byte, but it does not carry that advantage into divergence.

### Qwen3.8-Flash-Next NVFP4

| Folder | Size | Notes |
|---|---|---|
| [`qwen3.8-flash-next/`](./qwen3.8-flash-next/) | 119 GB | Text + MTP + Vision, NVFP4 routed experts (W4A4), FP8 n-gram table, 4 shards |

Converted from the ModelOpt [`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)
checkpoint; measured numbers are in [Qwen3.8-Flash-Next](#qwen38-flash-next).

More NInfer artifacts will be added to this repository over time.

## Qwen3.8-27B EXL3

### Model

Qwen3.8-27B (`Qwen3_5ForCausalLM`): 64 layers (48 GDN linear-attention, 16 full attention), hidden
5120, intermediate 17408, vocab 248,320, plus a separate MTP layer and a Vision tower.

### Quantization

NInfer's native EXL3 format (`exl3_mul1` + `trellis_t16_v1`) — a three-instruction trellis codebook
over 16x16 tiles, with the input and output Hadamard rotations folded into the kernels. Weights are
produced by NInfer's own C++/CUDA quantizer (`ninfer-quantize`) from full-precision source tensors;
no exllamav3 checkpoint is imported.

Per-tensor rates (half bits, i.e. X.5 bpw, are first-class trellis rates):

| Scope | 4.0 bpw artifact | 3.5 bpw artifact |
|---|---|---|
| MLP and GDN projections | 4.0 bpw | 3.5 bpw |
| Attention projections (the `-hq` promotion) | 5.0 bpw | 4.5 bpw |
| Vocabulary head | 6.0 bpw | 6.0 bpw |
| MTP layer | 4.0 bpw (5.0 attention), calibrated | 3.5 bpw (4.5 attention), calibrated |
| Vision tower | groupwise Q6/Q8/Q4/Q5 (not EXL3 — its MLP intermediate is not 128-aligned) | same |

The MTP layer is calibrated from the final hidden states and next-token embeddings.

### Quality

Full-corpus perplexity over 261,167 tokens (context/stride 4096/2048, FP8 KV, greedy), and KL
divergence against the full-precision model over the same 2,940 positions of a 23.5k-token wikitext
slice:

| Artifact | Size | PPL | KL(P_BF16 ‖ P) |
|---|---|---|---|
| **EXL3 4.0 bpw** | **15.68 GiB** | **4.2939** | **0.0332** |
| Groupwise INT4 | 16.96 GiB | 4.3439 | 0.0429 |
| NVFP4 | 22.09 GiB | 4.3149 | 0.0510 |
| EXL3 3.5 bpw | 14.24 GiB | 4.3101 | 0.0624 |

The two metrics rank these differently, and both are reported for that reason: **4.0 bpw wins on both**,
but 3.5 bpw's better perplexity than INT4 and NVFP4 does *not* survive as divergence. Only the KL
*ordering* should be compared across runs — the absolute values move by roughly 2× with the text, and
the ordering was reproduced in both halves of the reference.

### Performance

Measured on one NVIDIA GeForce RTX 5090, CUDA 13.3, a single request, greedy, 64-256 output tokens,
`--prefill-chunk 1024`:

| Regime | EXL3 4.0 bpw | EXL3 3.5 bpw | Q4 | NVFP4 |
|---|---|---|---|---|
| Decode, plain | 75 tok/s | 64 tok/s | 83 tok/s | 73 tok/s |
| Decode, MTP K=3 | 137 tok/s | 130 tok/s | 134 tok/s | 142 tok/s |
| Decode, MTP K=5 + `--lm-head-draft` | 145 tok/s | 131 tok/s | 144 tok/s | 167 tok/s |
| Prefill, 0.54k / 7.6k-token prompt | 1.97k / 2.33k tok/s | 1.72k / 2.11k tok/s | 2.42k / 2.91k tok/s | 5.28k / 8.60k tok/s |

Measuring MTP at **K=3** for all four keeps the draft length equal across formats; the K=5 row is each
artifact's own best setting, which Q4 and NVFP4 reach with `--lm-head-draft`. Both EXL3 tiers carry the
indexed proposal head that flag needs, so it is available to them too.

**4.0 bpw** is close to the other native formats on every regime: prefill 1.19–1.23x behind Q4, plain
decode within 10%, and at a comparable draft length its MTP matches Q4's exactly, on an artifact
1.3 GiB smaller (15.68 against 16.96 GiB). NVFP4 remains the prefill leader — as it is for this
engine's other models — because its tensor-core contraction needs no per-weight decoding, which a
4-bit trellis does: the contraction issues exactly the same number of MMAs as Q4's, and the difference
is the funnel, bit-field extracts and `IMAD`/`DP4A` per decoded window that the trellis costs.

**3.5 bpw is slower than 4.0 bpw on every regime**, on the same kernels. Its weights are 9% smaller,
but its odd half-rates take the heavier `exl3_windows_half` window decode — two funnel shifts for the
eight windows against one funnel and five bit-field extracts — and that costs more than the bytes it
saves. Its case is size and perplexity per byte, not speed.

These are single-request spot measurements, not the engine's methodology-conforming performance
tables; [docs/performance.md](https://github.com/giveen/ninfer-ext/blob/master/docs/performance.md)
records the published coverage and the difference.

## Qwen3.8-Flash-Next

Qwen3.8-Flash-Next (`Qwen4ExpForCausalLM`) has about 180B parameters: about 121B are 512 routed
experts per layer, and 51B are an n-gram embedding table. This artifact is converted from
[nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) with the
`qwen3_8_flash_next_nvfp4` recipe, and contains Text, MTP and Vision.

### Representation

| Weights | Stored as | Runtime residency |
|---|---|---|
| Routed experts (48 × 512, plus the MTP layer) | NVFP4, imported codes and scales; MTP re-encoded from block FP8 | pinned Host, fetched into a device expert cache |
| N-gram PLE table (320M × 160) | FP8 rows with BF16 multipliers | page-cache mapped or streamed from NVMe, gathered on the Host per token |
| Attention, GDN, hyper-connection, shared expert, PLE projections | Q8 | device |
| Token embedding / output head | Q8 / Q6 | device |
| Routers, shared-expert gates, norms, small vectors | BF16/FP32 direct | device |

The routed experts run on the W4A4 tensor-core route; each MoE layer resolves its top-10 experts
against an LRU device expert cache.

### Measured

One RTX 5090 (32 GB, sm_120a), CUDA 13.3, `--expert-cache auto`, fp8 KV, `--spec mtp`:

| Metric | Value |
|---|---|
| Causal perplexity (`ninfer-ppl-1m-v1`, quick, fp8 KV) | 3.518 |
| Prefill (1,457-token prompt) | 922 tok/s |
| Prefill, long context (262,144-token budget, fp8 KV, single request) | 3,498 / 2,908 / 2,218 tok/s at 64k / 128k / 256k tokens |
| Decode (greedy, MTP K=3) | 81 tok/s |
| Peak host RSS (`--ngram-residency stream`) | ~65 GiB |
| Artifact size | 119 GB, 4 sharded files |

Long-context prefill improved 1.48x at 64k, 1.87x at 128k and 2.53x at 256k over the previous
engine: the QSA block selection now pools each block's index keys once per select call instead of
once per query column (bit-identical).

### Requirements

- One RTX 5090 (sm_120a) and CUDA 13.3.
- About 70 GB of host RAM (measured ~65 GiB peak RSS) for the pinned experts with
  `--ngram-residency stream`, which reads the n-gram table from NVMe. Keeping the ~52 GB table in the
  page cache (`mapped`, chosen automatically when memory allows) needs more RAM and is faster once
  warm.
- KV storage `bf16`, `int8`, `fp8`, `nvfp4` or `k8v4`; speculative decoding `--spec mtp`.

## Usage

Build the engine from source, then serve an artifact:

```bash
git clone https://github.com/giveen/ninfer-ext
cd ninfer-ext && cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE=$PWD/.venv/bin/python && cmake --build build -j

# Qwen3.8-27B EXL3
build/apps/ninfer-serve qwen3_8_27b_exl3_4bpw.ninfer \
  --port 8099 --spec mtp --draft-tokens 3 --fixed-draft

# Qwen3.8-Flash-Next NVFP4
build/apps/ninfer-serve qwen3.8-flash-next/qwen3_8_flash_next_nvfp4.ninfer \
  --model-id qwen3.8-flash-next --max-context 229376 --kv-capacity 458752 \
  --kv-dtype fp8 --expert-cache auto --ngram-residency stream --spec mtp
```

`--vision` enables image input; the Vision tower loads lazily. See the repository's `README.md`,
`docs/cli.md`, and `docs/serving.md` for the full command surface.

## License

The quantized weights follow the base model's license. The quantization format, kernels, and tooling
are part of [giveen/ninfer-ext](https://github.com/giveen/ninfer-ext).
