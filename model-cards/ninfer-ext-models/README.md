---
license: apache-2.0
language:
  - en
  - zh
  - code
library_name: ninfer-ext
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-27B
tags:
  - exl3
  - trellis
  - quantization
  - ninfer
  - cuda
  - text-generation
  - vision
  - speculative-decoding
---

# NInfer models — Qwen3.8-27B EXL3

Quantized artifacts for **NInfer Ext**, a from-scratch C++/CUDA inference engine for maximum
single-GPU performance.

> [!IMPORTANT]
> **These artifacts only work with [giveen/ninfer-ext](https://github.com/giveen/ninfer-ext).**
> They are not Transformers checkpoints, not GGUF, not safetensors weights, and cannot be loaded by
> `transformers`, `vLLM`, `llama.cpp`, or exllamav3. `.ninfer` is NInfer's own artifact format, and the
> EXL3 trellis layout is decoded by kernels that live in that repository. Loading them anywhere else
> will fail.

## Contents

| File | Size | Notes |
|---|---|---|
| [`qwen3_8_27b_exl3_4bpw.ninfer`](./qwen3_8_27b_exl3_4bpw.ninfer) | 15.35 GiB | Text + MTP + Vision, 4.0 bpw body |
| `qwen3_8_27b_exl3_4bpw.ninfer.conversion.json` | 489 KiB | Conversion provenance and per-tensor rates |
| [`qwen3_8_27b_exl3_3p5bpw.ninfer`](./qwen3_8_27b_exl3_3p5bpw.ninfer) | 13.91 GiB | Text + MTP + Vision, 3.5 bpw body |
| `qwen3_8_27b_exl3_3p5bpw.ninfer.conversion.json` | 489 KiB | Conversion provenance and per-tensor rates |

Both are the same model at two points on the size curve. **4.0 bpw is the one to lead with**: it is the
only artifact here that beats both of the engine's other Qwen3.8-27B builds on *both* perplexity and KL
divergence, while being smaller than either (see Quality). 3.5 bpw is the size-optimised tier — best PPL
per byte, but it does not carry that advantage into divergence.

More NInfer artifacts will be added to this repository over time.

## Model

Qwen3.8-27B (`Qwen3_5ForCausalLM`): 64 layers (48 GDN linear-attention, 16 full attention), hidden
5120, intermediate 17408, vocab 248,320, plus a separate MTP layer and a Vision tower.

## Quantization

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

## Quality

Full-corpus perplexity over 261,167 tokens (context/stride 4096/2048, FP8 KV, greedy), and KL
divergence against the full-precision model over the same 2,940 positions of a 23.5k-token wikitext
slice:

| Artifact | Size | PPL | KL(P_BF16 ‖ P) |
|---|---|---|---|
| **EXL3 4.0 bpw** | **15.35 GiB** | **4.2939** | **0.0332** |
| Groupwise INT4 | 16.96 GiB | 4.3439 | 0.0429 |
| NVFP4 | 22.09 GiB | 4.3149 | 0.0510 |
| EXL3 3.5 bpw | 13.91 GiB | 4.3101 | 0.0624 |

The two metrics rank these differently, and both are reported for that reason: **4.0 bpw wins on both**,
but 3.5 bpw's better perplexity than INT4 and NVFP4 does *not* survive as divergence. Only the KL
*ordering* should be compared across runs — the absolute values move by roughly 2× with the text, and
the ordering was reproduced in both halves of the reference.

## Performance

Measured on one NVIDIA GeForce RTX 5090, CUDA 13.3, a single request, greedy, 64-256 output tokens,
`--prefill-chunk 1024`:

| Regime | EXL3 4.0 bpw | EXL3 3.5 bpw | Q4 | NVFP4 |
|---|---|---|---|---|
| Decode, plain | 76 tok/s | 64 tok/s | 78 tok/s | 71 tok/s |
| Decode, MTP K=3 (`--spec mtp --draft-tokens 3 --fixed-draft`) | 124 tok/s | 120 tok/s | 134 tok/s | 142 tok/s |
| Prefill, 0.55k / 7.6k-token prompt | 2.01k / 2.38k tok/s | 1.69k / 2.08k tok/s | 2.35k / 2.82k tok/s | 5.18k / 8.43k tok/s |

MTP is measured at a fixed three-token draft on all four so the row compares like with like. Q4 and
NVFP4 can additionally use `--lm-head-draft`, which reaches 144 and 166 tok/s at K=5; **the EXL3
artifacts cannot — they do not carry the separate draft-head projection those artifacts have**, so
`--lm-head-draft` fails at startup with `selected proposal head is absent from artifact`, and raising
K does not help them (124.6 tok/s at K=5 against 124 at K=3).

**4.0 bpw** is close to the other native formats: prefill is 1.17–1.18x behind Q4 and decode is within
3% of it, on an artifact 1.6 GiB smaller (15.35 against 16.96 GiB). NVFP4 remains the prefill leader —
as it is for this engine's other models — because its tensor-core contraction needs no per-weight
decoding, which a 4-bit trellis does: the contraction issues exactly the same number of MMAs as Q4's,
and the difference is the funnel, bit-field extracts and `IMAD`/`DP4A` per decoded window that the
trellis costs.

**3.5 bpw is slower than 4.0 bpw on both regimes**, on the same kernels. Its weights are 9% smaller,
but its odd half-rates take the heavier `exl3_windows_half` window decode — two funnel shifts for the
eight windows against one funnel and five bit-field extracts — and that costs more than the bytes it
saves. Its case is size and perplexity per byte, not speed.

These are single-request spot measurements, not the engine's methodology-conforming performance
tables; [docs/performance.md](https://github.com/giveen/ninfer-ext/blob/master/docs/performance.md)
records the published coverage and the difference.

## Usage

Build the engine from source, then serve an artifact:

```bash
git clone https://github.com/giveen/ninfer-ext
cd ninfer-ext && cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE=$PWD/.venv/bin/python && cmake --build build -j

build/apps/ninfer-serve qwen3_8_27b_exl3_4bpw.ninfer \
  --port 8099 --spec mtp --draft-tokens 3 --fixed-draft
```

`--vision` enables image input; the Vision tower loads lazily. See the repository's `README.md`,
`docs/cli.md`, and `docs/serving.md` for the full command surface.

## License

The quantized weights follow the base model's license. The quantization format, kernels, and tooling
are part of [giveen/ninfer-ext](https://github.com/giveen/ninfer-ext).
