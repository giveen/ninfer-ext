# NInfer

> Selected checkpoints. Maximum single-GPU inference performance.

> **ninfer-ext** is an extended fork of [Neroued/ninfer](https://github.com/Neroued/ninfer). It is
> not the upstream project. See [About this fork](#about-this-fork) for what differs.

NInfer is a from-scratch C++/CUDA inference engine for Qwen3.5 Dense and MoE architectures on a
single NVIDIA GeForce RTX 5090. It runs text, image, and video prompts through a local CLI or
OpenAI-/Anthropic-compatible HTTP APIs. The runtime is deliberately specialized: one GPU, one
resident model, and a startup-fixed capacity of one to eight active requests.

Five official artifacts are available. The quick-start commands use Qwen3.8-27B NVFP4.

| Model | Weights | Artifact | Download and model card |
|---|---|---|---|
| Qwen3.6-27B | `groupwise-int` | `qwen3_6_27b.ninfer` | [Qwen3.6-27B](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) |
| Qwen3.6-27B | `nvfp4` | `qwen3_6_27b_nvfp4.ninfer` | [Qwen3.6-27B NVFP4](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) |
| Qwen3.8-27B | `groupwise-int` | `qwen3_8_27b.ninfer` | [Qwen3.8-27B](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) |
| Qwen3.8-27B | `nvfp4` | `qwen3_8_27b_nvfp4.ninfer` | [Qwen3.8-27B NVFP4](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) |
| Qwen3.6-35B-A3B | `groupwise-int` | `qwen3_6_35b_a3b.ninfer` | [Qwen3.6-35B-A3B](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) |

Each v3 `.ninfer` artifact carries model configuration, encoded weights, logical bindings and
frontend resources. Runtime execution uses those facts with the implemented model and Op
capabilities. You can also [convert your own weights](docs/weight-conversion.md), reuse an official
recipe or choose another supported mixture of formats.

The current engine requires v3 artifacts. Existing official v2 downloads can be
[upgraded locally](docs/weight-conversion.md#upgrade-an-existing-v2-artifact) without downloading
the weights again.

## About this fork

ninfer-ext tracks [Neroued/ninfer](https://github.com/Neroued/ninfer) `master`
(`594930e7`, the `upstream` remote) and adds the following. Upstream owns the architecture and
the official artifacts; the engine design, artifact format and product scope are unchanged.

- **C++23.** The whole tree builds as C++23 and uses `std::format`/`std::println`,
  `std::ranges`, `std::flat_map`, `std::move_only_function` and deducing `this`. CMake has no
  CUDA23 dialect for nvcc, so `CMakeLists.txt` keeps `CUDA_STANDARD 20` and passes `-std=c++23`
  explicitly.
- **One chat template.** A single template, [`tools/chat_templates/qwen.jinja`](tools/chat_templates/qwen.jinja),
  serves every Qwen model. It derives from froggeric's v22.5 template with NInfer patches for
  continuation and tool results. Artifacts converted by this fork embed it; the default effort is
  `medium` and reasoning is retained.
- **Open upstream PRs, merged ahead of upstream.** Each is a merge commit on `master`:
  - NVFP4 sparse-MoE stack (#286–#290), so `Qwen3.6-35B-A3B` NVFP4 artifacts load and run.
  - GGUF as a conversion source (#282).
  - Kernels: Q5 K-split MMA for small batches (#292), Q4/Q5 group pipelining at T32 (#311),
    single-pass logsumexp for `target_logprobs` (#307), fused attention RMSNorm + NVFP4
    quantization (#305), NVFP4 SwiGLU partial last tile (#264), `rmsnorm_rope` text profile (#273),
    Q6 fused gate/up shape (#284), and two Q4 quads in flight in the sparse-MoE decode kernel
    (#199, re-ported onto the codec-based kernel).
  - Serving and frontend: `/v1/models` metadata (#162), prompt-progress timings (#163),
    `ignore_eos` (#197), `reasoning.summary` (#295), tool-call parser fixes (#299, #309), and a
    default shared-prefix catalog capacity (#274).
  - Workspace-layout fix (#297) and an inference speed-of-light estimator (#304).
- **Deliberately not merged.** #268 (sigmoid gate folded into the attention reduce epilogue)
  breaks CUDA Graph updates when MTP or ordinary profiles share a topology class, so it is
  excluded until the graph planner can tell fusable from non-fusable attention routes.

Model-quality claims, evaluation scores and the published concurrency tables below come from
upstream's runs; only the [fork versus upstream](#fork-versus-upstream) table was measured on this
fork's builds.

## Quick start

NInfer requires 64-bit Linux, an NVIDIA GeForce RTX 5090, a CUDA toolkit supporting `sm_120a`,
CMake 3.28 or newer, a C++23 host compiler, Ninja, `pkg-config`, FFmpeg development libraries
(`libavformat`, `libavcodec`, `libavutil`, and `libswscale`), and `libcurl >= 7.85`.
CUDA 13.3 is the validated development toolkit; CMake does not impose a CUDA version floor.
The build rejects CUDA architectures other than `sm_120a`.

Build the product binaries:

```bash
git clone https://github.com/giveen/ninfer-ext.git
cd ninfer-ext

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Tests and benchmarks are excluded from the default build. `cmake --preset release` configures
the same product build; `cmake --preset dev` also enables tests and benchmarks and finds a
Python 3 interpreter. Both presets use `build/` and explicitly reset the build options.
Machine-specific compiler and Python paths belong in the ignored `CMakeUserPresets.json`.
See [build organization and configuration](docs/maintainer/build-system.md) for details.

There is no install target or packaged binary distribution; run NInfer from its source build tree.
Python tools run independently of CMake; the standalone HBM probe has its own
[build command](tools/README.md#standalone-hbm-probe).

Download the artifact used by this example with the Hugging Face CLI:

```bash
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ninfer \
  --local-dir models
```

Start a long-running text/agent server with two active-request lanes and explicit Device/Host
checkpoint capacity:

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --max-context 240000 \
  --kv-capacity 240000 \
  --max-concurrency 2 \
  --kv-dtype fp8 \
  --device-state-slots 2 \
  --host-state-slots 8 \
  --host-kv-mib 8192 \
  --spec mtp --draft-tokens 7 \
  --lm-head-draft \
  --preserve-thinking
```

Each request has a 240,000-token logical ceiling. A shared 240,000-token Device KV pool serves
admitted requests; two requests run concurrently when their combined reservations fit. The cache
tiers provide two Device checkpoint slots, eight pinned Host State slots, and 8 GiB of pinned Host
KV beyond the two active StateImages.

Send an OpenAI-style request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Reply with one short sentence."}],
    "max_tokens": 64
  }'
```

Run a one-shot CLI request with a 32,768-token allocation:

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Explain prefill and decode, then give a concise conclusion." \
  --max-context 32768 \
  --max-new 8192 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 7 \
  --lm-head-draft
```

Answer content is written to stdout. Human-readable startup/runtime diagnostics and the CLI-owned
reasoning, timing, throughput, memory, and speculative-decoding report are written to stderr;
reasoning and the result report remain unprefixed product output. On a terminal, weight
materialization uses one transient progress line followed by a compact Engine-ready summary.
Redirected stderr receives persistent readable progress without terminal control sequences. Use
`--log-level debug` for complete startup detail. Option and local input errors remain direct command
diagnostics. Use `--messages FILE` and `--vision` for structured image/video input; see the
[CLI guide](docs/cli.md) and [committed examples](examples/cli/).

## Resource-aware long-context reuse

A reusable prefix checkpoint contains KV and the complete continuation state for its exact prompt
frontier. A Device-resident checkpoint resumes directly. Under pressure, the planner weighs Device
retention, pinned Host State/KV, and eviction by immediate restore work and later reuse cost. Active
requests retain their completion reservations.

See [Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
for the algorithm and [Serve TTFT benchmark](tools/bench/ttft/) for public-HTTP coverage of hot
reuse, Host resume, eviction, shared prefixes, scheduling boundaries, and multimodal load.

## Performance

Published measurements use an RTX 5090. The [performance index](docs/performance.md) links to
per-model run records and the [measurement rules](docs/performance/methodology.md). The tables
below are excerpts from those detailed results.

### Concurrent MTP3 decode

Saturated decode used INT8 group-64 KV, CUDA Graphs, MTP3, and one 8,192-token generation per active
request. Throughput uses aggregate committed decode tokens from complete intervals whose actual
decode batch equaled the configured concurrency. Acceptance covers the complete request wave;
these rates are steady decode (tok/s).

| Model profile | C=1 tok/s / accept | C=2 tok/s / accept | C=4 tok/s / accept | C=8 tok/s / accept | C8 / C1 |
|---|---:|---:|---:|---:|---:|
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#decode-saturation) `groupwise-int` | 185.8 / 68.2% | 247.0 / 69.0% | 309.5 / 68.4% | 535.0 / 68.3% | 2.88× |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#decode-saturation) `nvfp4` | 202.4 / 69.3% | 399.7 / 71.4% | 699.7 / 69.3% | 1,146.9 / 68.6% | 5.67× |
| [Qwen3.6-35B-A3B](docs/performance/qwen3.6-35b-a3b.md#decode-saturation) `groupwise-int` | 642.5 / 68.6% | 907.2 / 66.3% | 1,213.5 / 69.6% | 1,380.7 / 68.0% | 2.15× |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#decode-saturation) `nvfp4` | 143.8 / 48.9% | 267.6 / 48.1% | 461.1 / 45.8% | 766.6 / 46.0% | 5.33× |

### Single-request serving

The serial serving corpus used INT8 group-64 KV, CUDA Graphs, a 1,024-token prefill chunk, and five
fixed seeds after warm-up. The table keeps one short-prefill, one extreme-prefill, and one
structured-output MTP3 point for each published profile; the full context and scenario matrices are
linked from each model below.

| Model profile | 7,680-token prefill | 260,096-token prefill | Structured MTP3 decode |
|---|---:|---:|---:|
| [Qwen3.6-35B-A3B](docs/performance/qwen3.6-35b-a3b.md#single-request-speculative-decode) `groupwise-int` | 17,705.4 tok/s | 5,247.0 tok/s | 779.6 tok/s |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#single-request-speculative-decode) `groupwise-int` | 3,218.1 tok/s | 1,614.8 tok/s | 193.0 tok/s |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#single-request-speculative-decode) `nvfp4` | 11,191.5 tok/s | 2,510.6 tok/s | 252.2 tok/s |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#single-request-speculative-decode) `groupwise-int` | 3,274.7 tok/s | 1,609.7 tok/s | 224.4 tok/s |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#single-request-speculative-decode) `nvfp4` | 8,340.4 tok/s | 2,203.1 tok/s | 219.8 tok/s |

### Fork versus upstream

Head-to-head runs of `ninfer_bench` on one RTX 5090: upstream `594930e7` against this fork's
`master`, both built Release for `sm_120a` with CUDA 13.3 and CUDA Graphs on. Each row is the mean
of two runs per build, interleaved stock, fork, fork, stock, with 5 repetitions after 1 warmup. The
tests are `-n 128` (`tg128`) and `-pg 2048,128` (`pp2048+tg128`, decode phase reported); the
artifacts are the fork-converted local ones, and the chat template does not affect these numbers.
Speeds are decode tok/s; prefill is within ±0.4% everywhere.

Single-token decode, no speculation:

| Artifact | Test | Upstream | Fork | Change |
|---|---|---:|---:|---:|
| Qwen3.6-35B-A3B `groupwise-int` | tg128 | 362.9 | 372.1 | +2.5% |
| Qwen3.6-35B-A3B `groupwise-int` | pp2048+tg128 | 359.2 | 365.5 | +1.7% |
| Qwen3.8-27B `groupwise-int` | tg128 | 81.8 | 82.0 | +0.3% |
| Qwen3.8-27B NVFP4 | tg128 | 76.1 | 76.3 | +0.3% |

MTP (K=3) and DFlash (K=7):

| Artifact | Backend | Test | Upstream | Fork | Change |
|---|---|---|---:|---:|---:|
| Qwen3.8-27B `groupwise-int` | MTP3 | tg128 | 94.8 | 103.5 | +9.2% |
| Qwen3.8-27B `groupwise-int` | MTP3 | pp2048+tg128 | 191.4 | 212.8 | +11.2% |
| Qwen3.8-27B NVFP4 | MTP3 | tg128 | 114.1 | 114.4 | +0.3% |
| Qwen3.8-27B NVFP4 | MTP3 | pp2048+tg128 | 210.4 | 211.2 | +0.4% |
| Qwen3.6-35B-A3B `groupwise-int` | MTP3 | tg128 | 515.7 | 523.3 | +1.5% |
| Qwen3.6-35B-A3B `groupwise-int` | MTP3 | pp2048+tg128 | 738.1 | 745.0 | +0.9% |
| Qwen3.6-35B-A3B `groupwise-int` | DFlash7 | tg128 | 370.8 | 374.6 | +1.0% |
| Qwen3.6-35B-A3B `groupwise-int` | DFlash7 | pp2048+tg128 | 1,100.3 | 1,112.0 | +1.1% |

The MTP gain on Qwen3.8-27B `groupwise-int` matches PR #292's claim (+11% MTP3 decode); it applies
to Q5 small-batch projections, so NVFP4 artifacts do not see it. MTP acceptance is identical to
upstream everywhere except Qwen3.8-27B `groupwise-int` `tg128`, where the fork accepts 29.2% of
drafts against 30.2% (340 speculative rounds against 335); the cause is not yet attributed. These
are single-request Op-and-engine numbers on short prompts, not a replacement for the serving
corpus above. DFlash2, concurrency above one and NVFP4 35B-A3B were not measured.

### Adaptive MTP draft length

`--spec mtp` alone selects an adaptive policy whose longest draft is 7 tokens (`--draft-tokens N`
lowers that ceiling; `--fixed-draft` pins exactly N). A single
request drafts 2, 3, 4 or 7 tokens per round, chosen from its recent acceptance and the round times
measured at startup; rounds with several requests decoding together draft 3, where longer drafts
raised no aggregate throughput at concurrency 2, 4 or 8. The MTP3 tables above pin K=3 with
`--fixed-draft`.

Single-request greedy decode on one RTX 5090 (CUDA Graphs, default KV type), decode tok/s, mean of
three runs per cell; the 27B is the Qwen3.8-27B `groupwise-int` artifact. The nine prompts are short local ones, not
shipped with the repository: three code-edit prompts (`refactor`, `cppfix`, `edit2`), two
code-generation (`codegen`, `codegen2`), two prose (`prose`, `prose2`) and two reasoning
(`reason`, `reason2`) requests. Non-reasoning prompts use `--no-thinking --max-new 640`; reasoning
prompts use `--max-new 1200`.

| Prompt | 27B K=3 | 27B adaptive | Change | 35B-A3B K=3 | 35B-A3B adaptive | Change |
|---|---:|---:|---:|---:|---:|---:|
| refactor | 208.4 | 220.9 | +6.0% | 714.7 | 740.9 | +3.7% |
| cppfix | 215.7 | 263.4 | +22.1% | 750.1 | 802.5 | +7.0% |
| edit2 | 220.1 | 287.3 | +30.5% | 752.2 | 769.3 | +2.3% |
| codegen | 197.7 | 208.7 | +5.6% | 676.7 | 651.3 | −3.8% |
| codegen2 | 189.4 | 199.0 | +5.1% | 674.3 | 673.5 | −0.1% |
| prose | 140.2 | 139.6 | −0.4% | 496.7 | 508.4 | +2.3% |
| prose2 | 121.9 | 129.3 | +6.1% | 454.7 | 499.5 | +9.8% |
| reason | 188.7 | 188.0 | −0.4% | 587.8 | 611.1 | +4.0% |
| reason2 | 185.8 | 191.8 | +3.2% | 663.1 | 663.6 | +0.1% |
| Geometric mean | | | +8.2% | | | +2.7% |

Per class, the 27B gains +19.1% on edits, +5.3% on code generation, +2.8% on prose and +1.4% on
reasoning; the 35B-A3B gains +4.3%, −2.0%, +6.0% and +2.0%. Without speculation the 27B decodes at
about 80 tok/s and the 35B-A3B at about 370 tok/s. Server runs at concurrency 2, 4 and 8 stayed
within ±3% of fixed K=3. Each cell is one command, with `--draft-tokens 3 --fixed-draft` for the
K=3 column:

```bash
build/apps/ninfer models/qwen3_8_27b.ninfer --prompt "$(cat prompt.txt)" \
  --spec mtp --greedy --no-thinking --max-new 640
```

The summary line reports `decode speed` and, for adaptive runs, `mtp rounds by length`.

## Evaluation

Capability scores were measured through NInfer's OpenAI-compatible serving route with thinking
enabled, MTP3, and EvalScope 1.9.0 (0-shot, rule scoring, one sample per problem):

| Model profile | AIME 2025 | AIME 2026 | GPQA-Diamond | ERQA | RealWorldQA |
|---|---:|---:|---:|---:|---:|
| [Qwen3.6-27B groupwise-int](model-cards/Qwen3.6-27B-NInfer/README.md) | 86.67% | 93.33% | 86.87% | — | — |
| [Qwen3.6-27B NVFP4](model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) | 93.33% | 93.33% | 84.34% | — | — |
| [Qwen3.6-35B-A3B groupwise-int](model-cards/Qwen3.6-35B-A3B-NInfer/README.md) | 90.00% | 90.00% | 85.35% | — | — |
| [Qwen3.8-27B groupwise-int](model-cards/Qwen3.8-27B-NInfer/README.md) | 96.67% | 96.67% | 87.37% | 66.25% | 82.22% |
| [Qwen3.8-27B NVFP4](model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) | 96.67% | 96.67% | 90.40% | 66.25% | 83.53% |

The Qwen3.6 rows used temperature 0.6 and presence penalty 1.0; the Qwen3.8 rows used temperature
1.0 and presence penalty 0.0. Multimodal evaluation used `--vision` and an 81,920-token context
limit. Text evaluation used 262,144 tokens except Qwen3.8-27B NVFP4, which used 252,928 tokens to
fit the RTX 5090 after weights. Each score is one sample per problem; model cards contain the
correct/total counts and evaluation notes.

## Startup notes

GPU residency is fixed at process startup. `--spec` selects speculative decoding residency, and
`--vision` independently selects Vision residency. Qwen3.6-35B-A3B DFlash can be combined with
Vision; it accelerates generated-text decode after multimodal prefill, not Vision encode itself.

## Docker

Build the runtime image on a host with the NVIDIA Container Toolkit:

```bash
docker build --tag ninfer:local .
```

Mount the downloaded model and run the same example server profile:

```bash
docker run --rm \
  --gpus '"device=0"' \
  --publish 8080:8080 \
  --volume "$PWD/models:/models:ro" \
  ninfer:local \
  ninfer-serve /models/qwen3_8_27b_nvfp4.ninfer \
  --host 0.0.0.0 \
  --max-context 240000 \
  --kv-capacity 240000 \
  --max-concurrency 2 \
  --kv-dtype fp8 \
  --device-state-slots 2 \
  --host-state-slots 8 \
  --host-kv-mib 8192 \
  --spec mtp --draft-tokens 7 \
  --lm-head-draft \
  --preserve-thinking
```

## Capabilities and limits

The official artifacts provide the following capabilities, with optional components enabled at startup:

- text generation with thinking and non-thinking prompt modes;
- image, multi-image, video, and mixed multimodal messages;
- chunked prefill, exact-batch CUDA Graph decode, and startup-bounded batched decode;
- MTP speculative decoding up to seven draft tokens: a single request picks its draft length per round
  from measured acceptance, several requests decoding together draft three;
- BF16, INT8, FP8, NVFP4, and K8V4 KV storage;
- offline causal-perplexity scoring;
- private and shared exact-prefix reuse with Device/Host State and KV retention;
- model-aware sampling defaults and explicit sampler overrides;
- OpenAI Responses Core, OpenAI Chat Completions, and Anthropic Messages, including streaming,
  tools, local response state, token counting, and usage accounting.

The 35B-A3B target additionally supports DFlash with draft windows from one to fifteen for Text and
image/video Vision prompts. Qwen3.8-27B artifacts with the DFlash2 companion weights support
`--spec dflash2 --draft-tokens 7` for the same Text/Vision Engine path, with draft counts 1..15
and either full or optimized proposal heads.

Qwen3.8-Flash-Next (`Qwen4ExpForCausalLM`, converted with `qwen3_8_flash_next_nvfp4`) runs Text,
MTP, and Vision with its routed experts in pinned Host memory behind a device expert cache sized
by `--expert-cache`; it needs about 128 GB of host RAM and supports `bf16`/`fp8` KV.

The product boundary remains intentionally small:

- one RTX 5090 and one resident model per Engine;
- a startup-fixed capacity of one to eight active requests with bounded FIFO ingress;
- no request preemption, priority/QoS, active-request swapping, multi-GPU, or distributed
  serving; only Qwen4Exp routed experts and its n-gram table live outside device memory;
- one shared startup-fixed KV pool across active requests and retained prefixes;
- model architectures and format/shape combinations use explicitly implemented native paths;
- parsed tool calls are returned to the client; NInfer does not execute tools;
- the in-tree C++ headers are not distributed as an installed SDK.

`--max-context` is each sequence's logical limit. `--kv-capacity` sizes the shared Main Text KV pool
used by active requests and retained prefixes; `auto` resolves the largest legal capacity at
startup from the memory remaining after weights while keeping 1 GiB of sizing headroom. Explicit
capacities remain fixed for the process lifetime.

## Documentation

- [Documentation index](docs/README.md)
- [CLI](docs/cli.md)
- [HTTP serving](docs/serving.md)
- [Performance](docs/performance.md)
- [Perplexity evaluation](docs/perplexity.md)
- [Weight conversion and custom recipes](docs/weight-conversion.md)
- [Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
- [Serve TTFT benchmark](tools/bench/ttft/)
- [CLI examples](examples/cli/)
- [Contributing](CONTRIBUTING.md)

Run the relevant `--help` for the exact current option contract.

## Support

NInfer is a personal project that I develop out of interest. If you find it useful and would like
to support its continued development, you can [support the project on Ko-fi](https://ko-fi.com/neroued).

Support is entirely voluntary. It is not a purchase or investment and does not come with financial
returns, promised services or features, or a role in project decisions. The project's direction,
priorities, technical choices, and release schedule remain independently determined by the
maintainer.

## License

NInfer is licensed under the [Apache License 2.0](LICENSE).

The published artifacts are derived from
[Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B),
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), and
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). The Qwen3.6-27B NVFP4 artifact
also uses the fixed packed weights from
[rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm).
The Qwen3.8-27B NVFP4 artifact also uses the fixed mixed FP8/NVFP4 weights from
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4). These source
repositories are distributed under Apache-2.0. Vendored dependencies retain their own license files
under `third_party/`.
