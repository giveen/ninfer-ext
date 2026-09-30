# Single-GPU serving performance

Published measurements use one NVIDIA GeForce RTX 5090 through NInfer's public HTTP serving route.
Choose a model below for its detailed results, run conditions, output limitations, and reproduction
commands. These are recorded measurements; a model/backend being supported does not
mean every workload or concurrency has a published measurement.

Read the [measurement and publication rules](performance/methodology.md) for workload definitions,
metric formulas, statistics, comparison requirements, and the standard result-page format.

## Published coverage

Each cell links to the relevant result section. “Not published” describes measurement coverage,
not product support. C is configured request concurrency; K is the number of draft tokens.

| Model / weights | MTP0 context profile | Single-request speculative decode | Corpus makespan | MTP3 decode saturation |
|---|---|---|---|---|
| Qwen3.6-27B / `groupwise-int` | [8K–256K](performance/qwen3.6-27b.md#no-speculation-context-profile) | [MTP3](performance/qwen3.6-27b.md#single-request-speculative-decode) | Not published | [C=1, 2, 4, 8](performance/qwen3.6-27b.md#decode-saturation) |
| Qwen3.6-27B / `nvfp4` | [8K–256K](performance/qwen3.6-27b.md#no-speculation-context-profile) | [MTP3](performance/qwen3.6-27b.md#single-request-speculative-decode) | Not published | [C=1, 2, 4, 8](performance/qwen3.6-27b.md#decode-saturation) |
| Qwen3.6-35B-A3B / `groupwise-int` | [8K–256K](performance/qwen3.6-35b-a3b.md#no-speculation-context-profile) | [MTP3; DFlash K=7 stochastic/greedy](performance/qwen3.6-35b-a3b.md#single-request-speculative-decode) | [MTP3 C=1, 2, 4, 8; DFlash C=1](performance/qwen3.6-35b-a3b.md#corpus-makespan) | [C=1, 2, 4, 8](performance/qwen3.6-35b-a3b.md#decode-saturation) |
| Qwen3.8-27B / `groupwise-int` | [8K–256K](performance/qwen3.8-27b.md#no-speculation-context-profile) | [MTP3; DFlash2 K=7](performance/qwen3.8-27b.md#single-request-speculative-decode) | [MTP3 C=1, 2, 4, 8; DFlash2 C=1](performance/qwen3.8-27b.md#corpus-makespan) | [C=1, 2, 4, 8](performance/qwen3.8-27b.md#decode-saturation) |
| Qwen3.8-27B / `nvfp4` | [8K–256K](performance/qwen3.8-27b.md#no-speculation-context-profile) | [MTP3; DFlash2 K=7](performance/qwen3.8-27b.md#single-request-speculative-decode) | [MTP3 C=1, 2, 4, 8; DFlash2 C=1](performance/qwen3.8-27b.md#corpus-makespan) | [C=1, 2, 4, 8](performance/qwen3.8-27b.md#decode-saturation) |
| Qwen3.8-27B / `exl3` 4.0 bpw | Not published | Not published | Not published | Not published |
| Qwen3.8-27B / `exl3` 3.5 bpw | Not published | Not published | Not published | Not published |

Qwen3.8 and Qwen3.6-35B-A3B C=1 corpus points also supply their single-request phase tables.
The Qwen3.6-27B NVFP4 MTP3 phase table comes from a corpus C=1 point whose full makespan is
not published here. Qwen3.8 measurements use FP8 E4M3 row-256 KV; the Qwen3.6 measurements
use INT8 group-64 KV. Each model page records its build and run conditions.

The `exl3` artifacts have no methodology-conforming measurement yet — the coverage row above records
that. Their single-request spot numbers and full-corpus quality figures live in the artifacts' model
card: [jabbatheduck/ninfer-ext-models](https://huggingface.co/jabbatheduck/ninfer-ext-models).

## Causal attention architecture migration

The causal-cache attention Op was replaced with the upstream per-format architecture (shared causal
primitives, one grouped/tiled plan per KV storage, split-KV prefill for FP8 and K8V4). This section
records the engineering measurement of that change. It is an **Op-level** result, not a serving
measurement, and it is not a cell in the coverage table above.

Conditions: RTX 5090, CUDA 13.3, `build` with `-O3`; baseline is the pre-migration fork attention.
Op numbers come from `ninfer_causal_softmax_attention_bench` (CUDA-graph execution, median of 11
repetitions; repeated runs of one binary agree within 1.04×, so a difference below that is noise).
End-to-end numbers come from `ninfer_bench` on Qwen3.8-27B `nvfp4`, `-p 32768 --max-ctx 32768`,
best-of-N over 15 interleaved samples per side.

Append-and-attend prefill, W=1024, fragmented mapping:

| KV storage | L=8192 | L=32768 | Op workspace |
|---|---|---|---|
| BF16 | 1.05× | 1.09× | unchanged |
| FP8-E4M3-row256 | 1.32× | 1.60× | 0 → 177.5 MB |
| INT8-G64 | 1.00× | 1.01× | unchanged |
| K8V4 | 1.29× | 1.56× | 0 → 177.5 MB |
| NVFP4-G16 | 1.00× | 1.04× | unchanged |

Decode (W=1 and W=16, B=1 and B=8, L=32768, graph execution) is 1.00–1.56×, except INT8 W=1 B=1,
which regresses 57.3 → 63.5 µs (0.90×, stable and above the noise floor; 6 µs).

End-to-end 27B prefill at 32K is unchanged: FP8 KV 0.986–0.997×, BF16 KV 0.999–1.007×. Model
workspace is unchanged at 430.8 MiB — the Op's 177 MB fits inside the existing model allocation.
The Op gain does not move end-to-end throughput because a 1024-token prefill chunk saves about
1.7 ms of a 3.4 s prefill (~1.6%), below the ~1–2% end-to-end noise floor. That noise comes from
the host: this machine runs a 475 W power limit (575 W default) with unlocked clocks, and `-lgc`
needs privileges that were unavailable. End-to-end runs were therefore interleaved and reduced
best-of-N. A single earlier FP8 reading of +6.2% was an outlier and is not reported.

## Reading the results

| Question | Metric to use |
|---|---|
| How fast is prompt processing or an individual decode phase? | Prefill phase, Server TTFT, Decode phase |
| How long does the full fixed request set take? | Corpus makespan, Corpus decode, Requests/s |
| What aggregate decode rate is sustained at a full batch? | Steady decode |

These rates use different time boundaries. Server TTFT is an internal phase sum; external
streaming TTFT has its [own benchmark contract](../tools/bench/ttft/README.md). Stochastic runs
can generate different token totals even with the same prompts and seeds. Output-limit and
repetition samples remain labeled in the measured corpus; throughput alone does not establish
successful task completion. See the [35B termination and anomalies](performance/qwen3.6-35b-a3b.md#termination-and-anomalies)
and [Qwen3.8 completion outcomes](performance/qwen3.8-27b.md#completion-outcomes).

## Related references

- [Serving benchmark runners](../tools/bench/README.md#serving-corpus-benchmark): usage and local report files.
- [Engine and Op benchmarks](../bench/README.md): their separate measurement scopes and commands.
- [Capability evaluation](../eval/README.md): evaluation workflow; published scores live in the
  [model cards](README.md#model-artifacts).
- [Perplexity](perplexity.md): offline causal-scoring measurement and comparison rules.

Model pages are the detailed result authority. README and model-card performance tables are
excerpts linked to those pages; update them together when replacing an applicable measurement.
