# NInfer documentation

Start with the [project README](../README.md) to build NInfer, download a published artifact, and
run the CLI or HTTP server.

## User guides

| Document | Purpose |
|---|---|
| [CLI](cli.md) | text, chat-history, image/video input, output streams, sampling, MTP, and common runtime options |
| [HTTP serving](serving.md) | OpenAI Responses/Chat Completions, Anthropic Messages, state, streaming, token counting, authentication, and tool calls |
| [Performance](performance.md) | RTX 5090 measurement coverage, per-model serving results, methodology, and publication rules |
| [Weight conversion](weight-conversion.md) | official recipes, custom formats and sources, conversion methods, optional components and artifact output |
| [Perplexity](perplexity.md) | fixed-corpus and custom-text causal perplexity, comparison rules, progress, and reports |
| [Optimizer](optimizer.md) | use-case runbooks and a designed-experiment sweep that picks the fastest/balanced/max-context configuration for a machine |
| [Qwen3.8-Flash-Next](flash-next.md) | the ~180B MoE with pinned-host experts: requirements, download, EXL3 variants, and speed |
| [Speculative decoding](speculative-decoding.md) | MTP, DFlash, DFlash2, EAGLE3, the prompt-lookup drafter, adaptive draft lengths, and the per-model recommended settings |
| [EXL3 quantization](quantization.md) | the trellis format at 4.0 and 3.5 bpw: quality, size, and single-request speed |
| [KV streaming and context cache](kv-and-context.md) | contexts beyond GPU memory, salvage, anchors, the Host-tier budget, and the serving extras |
| [Gemma 4](gemma4.md) | Gemma 4 31B and 26B-A4B: representation, measured numbers, and limits |
| [Versus stock NInfer](vs-stock-ninfer.md) | like-for-like comparison against upstream, and stock's numbers beside the fork's latest |
| [Engineering notes](engineering.md) | conversion throughput, C++23, and the upstream PRs merged ahead |
| [CLI examples](../examples/cli/) | committed text, multimodal, thinking, long-decode, and long-context inputs |

The executable `--help` output is the exact source for command-line option spelling and defaults.

## Model artifacts

| Model | Weights | Download | Versioned model card source |
|---|---|---|---|
| Qwen3.6-27B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) | [model card](../model-cards/Qwen3.6-27B-NInfer/README.md) |
| Qwen3.6-27B | `nvfp4` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) | [model card](../model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) |
| Qwen3.8-27B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | [model card](../model-cards/Qwen3.8-27B-NInfer/README.md) |
| Qwen3.8-27B | `nvfp4` | [Hugging Face](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | [model card](../model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) |
| Qwen3.6-35B-A3B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | [model card](../model-cards/Qwen3.6-35B-A3B-NInfer/README.md) |
| Qwen3.8-27B | `exl3` 4.0 bpw | [Hugging Face](https://huggingface.co/jabbatheduck/ninfer-ext-models) | [model card](../model-cards/ninfer-ext-models/README.md) |
| Qwen3.8-27B | `exl3` 3.5 bpw | [Hugging Face](https://huggingface.co/jabbatheduck/ninfer-ext-models) | [model card](../model-cards/ninfer-ext-models/README.md) |
| Gemma 4 31B IT | `M1` (Q6/Q5 g64) | [Hugging Face](https://huggingface.co/jabbatheduck/ninfer-ext-models/tree/main/gemma4-31b-it-m1) | [HF card](https://huggingface.co/jabbatheduck/ninfer-ext-models/blob/main/gemma4-31b-it-m1/README.md) |
| Gemma 4 26B-A4B IT | QAT int4 experts + Q8 | [Hugging Face](https://huggingface.co/jabbatheduck/ninfer-ext-models/tree/main/gemma4-26b-a4b-qat) | [model card](../model-cards/Gemma4-26B-A4B-QAT-NInfer/README.md) |

## Repository-local guides

- [Benchmarks](../bench/README.md)
- [Tests](../tests/README.md)
- [Tools](../tools/README.md)
- [Capability evaluation](../eval/README.md)

## Review records

The [review directory](review/) holds the review pass that drove the fix waves merged in PR #52, and
its triage. These are records of completed work, not active contracts.

| File | Contents |
|---|---|
| [review-triage.md](review/review-triage.md) | the triage: each finding's verdict and band, with a ledger per fix wave |
| [review-triage.json](review/review-triage.json) | the same triage in machine-readable form |
| [review-results.txt](review/review-results.txt) | the raw review output: 1,286 findings across 565 files |
| [critical-triage.txt](review/critical-triage.txt) | a separate critical-only pass over the same tree |

## Maintainer references

The active references under [`maintainer/`](maintainer/) record current architecture, model,
artifact, and maintenance contracts. These files are not additional user workflows or installed
API documentation.

[Engine architecture](maintainer/engine-architecture.md) is the single top-level reference. The
other references own narrower contracts:

| Document | Responsibility |
|---|---|
| [Engine architecture](maintainer/engine-architecture.md) | model/config/weight ownership, loading-to-execution flow, requests, scheduling, transactions and graphs |
| [Build system](maintainer/build-system.md) | CMake targets, explicit source ownership, CUDA compilation boundaries, presets and developer configuration |
| [Windows port](maintainer/windows-port.md) | the Windows toolchain floors, the platform seam, what the compile gate proves, and what is still unverified |
| [Artifact container](maintainer/artifact-container.md) | v3 directory, objects, logical bindings, Uses, resources and file framing/sharding |
| [Numeric formats](maintainer/tensor-formats.md) | represented values, codes/scales, conversion arithmetic and numerical interpretation |
| [Storage layouts](maintainer/storage-layouts.md) | packing, plane offsets, padding, encoded sizes and view addressing |
| [Qwen3.5 model](maintainer/qwen3_5-model.md) | Dense/MoE mathematics, instance config, logical parameters, MTP, Vision and state semantics |
| [Qwen4Exp model](maintainer/qwen4-exp-model.md) | Qwen3.8-Flash-Next hyper-connection, QSA, n-gram PLE, MoE and MTP mathematics and its FP64 oracle |
| [Gemma 4 model](maintainer/gemma4-model.md) | Gemma 4 31B configuration, sandwich norms with `layer_scalar`, sliding/global attention with proportional RoPE, the K = V global representation, its tokenizer and artifact |
| [DFlash and DFlash2](maintainer/dflash.md) | conditioning, masked draft computation, proposal distributions and backend state |
| [Constrained decoding](maintainer/constrained-decoding.md) | request constraints, grammar compilation and caching, per-position masks, the sampling and acceptance contract, and current coverage |
| [EAGLE3](maintainer/eagle3.md) | autoregressive draft graph, three-layer target feature fusion and draft-vocabulary mapping (chain landed and measured; static draft tree pending) |
| [Prompt-lookup suffix drafter](maintainer/lookup-drafter.md) | Host suffix index, acceptance model, per-lane choice and its phase plan (active work) |
| [Resource scheduling and context cache](maintainer/resource-scheduling-and-context-cache.md) | candidate selection, retention, materialization and Device/Host checkpoint policy |
| [Paged KV context store](maintainer/paged-kv-cache.md) | typed pools, pages, replicas, address spaces, reservations and consumer views |
| [ReplaySSM GDN](maintainer/replayssm-gdn.md) | raw transition records and faithful commitment of the verified state prefix |
| [Op development](maintainer/op-development.md) | semantic boundaries, source ownership, numerical qualification and performance evidence |
| [W4A8 prefill](maintainer/w4a8-prefill.md) | active work: the landed int8 Q4 prefill route, its enablement and measured evidence |
| [Operational logging](maintainer/logging.md) | log ownership, presentation, severity and data policy |
| [Linear benchmark](maintainer/linear-benchmark.md) | pure Linear measurement, metrics and suites |
| [Linear tuning and reports](maintainer/linear-tuning.md) | tuning ranges, priority points, dispatch tradeoffs and final performance report format |
| [Opt-in real-model Engine tests](maintainer/real-model-tests.md) | artifact selection, coverage, and the component-skip contract for the real-model subset |

Model cards contain official artifact facts and source provenance. The
[conversion guide](weight-conversion.md) is the entry point for making an artifact. Exact config
fields, parameter expansion and native supported domains are maintained by the code linked from
these references.
