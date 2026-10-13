# EXL3 quantization

EXL3 is this fork's own weight format: a trellis codebook (`exl3_mul1` + `trellis_t16_v1`) over 16x16
tiles, with the input and output Hadamard rotations folded into the kernels. It is not exllamav3's
checkpoint format and imports nothing from it — `ninfer-quantize` produces the weights from
full-precision source tensors, and the loader and kernels live in this repository.

Two artifacts are published for Qwen3.8-27B, both Text + MTP + Vision (the Qwen3.8-Flash-Next
artifacts, which quantize the routed experts only, are described in
[Flash-Next EXL3](flash-next.md#flash-next-exl3)):

| Artifact | Size | PPL | KL(BF16 ‖ P) |
|---|---|---:|---:|
| `qwen3_8_27b_exl3_4bpw.ninfer` | 15.68 GiB | **4.2939** | **0.0332** |
| `qwen3_8_27b_exl3_3p5bpw.ninfer` | 14.24 GiB | 4.3101 | 0.0624 |

Perplexity is over 261,167 tokens (context/stride 4096/2048, FP8 KV, greedy) and KL divergence is
against the full-precision model. 4.0 bpw is the smallest and the best of this fork's Qwen3.8-27B
builds on **both** metrics — `groupwise-int` is 16.96 GiB at 4.3439 / 0.0429 and `nvfp4` is
22.09 GiB at 4.3149 / 0.0510. 3.5 bpw is the size tier: best perplexity per byte, but its advantage
over INT4 and NVFP4 does not survive as divergence. The two metrics rank the set differently, so both
are reported; only the KL *ordering* is comparable across runs.

Single-request speed on one RTX 5090 (CUDA 13.3, greedy, 64–256 output tokens):

| Regime | EXL3 4.0 bpw | EXL3 3.5 bpw |
|---|---|---|
| Decode, plain | 75 tok/s | 64 tok/s |
| Decode, MTP K=3 | 137 tok/s | 130 tok/s |
| Decode, MTP K=5 + `--lm-head-draft` | 145 tok/s | 131 tok/s |
| Prefill, ~0.54k / ~7.6k-token prompt | 1.97k / 2.33k tok/s | 1.72k / 2.11k tok/s |

3.5 bpw is slower on every regime: its odd half-rates take the heavier `exl3_windows_half` window
decode, which costs more than the 9% of weight bytes it saves. Both tiers carry the indexed proposal
head `--lm-head-draft` needs. Cross-format single-request numbers, the measurement protocol, and
everything else about these artifacts are in their
[model card](https://huggingface.co/jabbatheduck/ninfer-ext-models/blob/main/README.md).

The per-tensor rates behind these artifacts — where the half-bpw rates go, which projections get the
`-hq` promotion, and how the MTP layer is calibrated — are in the
[model card](https://huggingface.co/jabbatheduck/ninfer-ext-models/blob/main/README.md) and the
[conversion guide](weight-conversion.md).
