# ninfer-ext

ninfer-ext is an extended fork of [Neroued/ninfer](https://github.com/Neroued/ninfer). NInfer is a
from-scratch C++/CUDA inference engine for Qwen models on one NVIDIA GeForce RTX 5090.

This fork adds:

- **Qwen3.8-Flash-Next.** The fork runs this ~180B-parameter MoE model on a single 32 GB GPU. Its
  routed experts stay in pinned Host memory behind a device expert cache.
- **Faster speculative decoding.** MTP adapts its draft length per round.
- **Serving and context-cache work.** Changes for long-running agent workloads.
- **C++23 and upstream pull requests.** The tree is ported to C++23, and a set of open upstream PRs
  is merged ahead of upstream.

Everything else is upstream's engine: the architecture, the `.ninfer` v3 artifact format, the
official artifacts, and the OpenAI- and Anthropic-compatible serving. The fork tracks upstream
`master` (currently `bace20dc`, the `upstream` remote).

- [Qwen3.8-Flash-Next](#qwen38-flash-next)
- [Performance](#performance)
- [Quick start](#quick-start)
- [What else the fork changes](#what-else-the-fork-changes)
- [Capabilities and limits](#capabilities-and-limits)
- [Documentation](#documentation)
- [Credits and license](#credits-and-license)

## Qwen3.8-Flash-Next

Qwen3.8-Flash-Next (`Qwen4ExpForCausalLM`) has about 180B parameters. About 121B of them are 512
routed experts per layer, and 51B are an n-gram embedding table. Neither fits in 32 GB, and upstream
NInfer does not implement the architecture.

ninfer-ext runs it through the same Engine, CLI, and HTTP server as every other model:

- **Experts on the Host.** Routed experts stay NVFP4 in pinned Host memory. Each MoE layer
  resolves its top-10 experts against an LRU device expert cache inside the decode CUDA Graph and
  copies misses over PCIe. `--expert-cache` sizes that cache; `auto` gives it the device memory left
  after the KV floor.
- **Prefill streams whole layers.** Chunks of 103 tokens or more stream a whole expert layer through
  a double-buffered staging bank. Experts already in the cache are copied device-to-device
  instead. Staged experts run on a W4A4 tensor-core route. The prefill chunk defaults to 4,096
  tokens for this model.
- **N-gram table off the device.** The table is file-mapped through the page cache, or streamed
  from NVMe with batched direct I/O (`--ngram-residency`).
- **Full feature set.** Qwen Sparse Attention works with every KV-cache profile (`bf16`, `int8`,
  `fp8`, `nvfp4`, `k8v4`). Hyper-connection residual streams, the per-layer embedding, MTP
  speculative decoding and Vision are all supported.

The mathematics and state semantics are in the [Qwen4Exp model reference](docs/maintainer/qwen4-exp-model.md).
An FP64 oracle checks them in `tests/models/qwen4_exp/`.

### Requirements

- One RTX 5090.
- About 128 GB of host RAM for the pinned experts.
- About 127 GB of disk for the artifact. It is split into `.part-NNNN` files next to the `.ninfer`
  file.

### Convert and serve

There is no published Flash-Next artifact. Convert it from
[nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4):

```bash
python3 -m tools.convert \
  --model /path/to/Qwen3.8-Flash-Next-NVFP4 \
  --recipe qwen3_8_flash_next_nvfp4 \
  --components text,vision,mtp \
  --resource chat_template.jinja=tools/chat_templates/qwen.jinja \
  --name qwen3.8-flash-next \
  --out models/qwen3_8_flash_next_nvfp4.ninfer
```

Serve two concurrent 229,376-token requests with FP8 KV and MTP:

```bash
./build/apps/ninfer-serve models/qwen3_8_flash_next_nvfp4.ninfer \
  --model-id qwen3.8-flash-next --max-concurrency 2 \
  --max-context 229376 --kv-capacity 458752 --kv-dtype fp8 \
  --expert-cache auto --spec mtp
```

### Flash-Next speed

All numbers are from one RTX 5090 with BF16 KV and CUDA Graphs, at fork commit `19f38b77`.
[Benchmark conditions](#benchmark-conditions) has the details.

| Test | tok/s |
|---|---:|
| Prefill, 4,096-token prompt | 2,938 |
| Prefill, 16,384-token prompt | 2,877 |
| Prefill, 512-token prompt | 371 |
| Decode, `tg128` | 71.0 |
| Decode after a 2,048-token prompt | 62.6 |
| Decode, `tg128`, adaptive MTP | 86.5 (68% accepted) |
| Decode, `tg128`, MTP K=3 | 83.1 (51% accepted) |

Serving 512-token essays per request, aggregate decode tok/s (mean of two runs):

| `ninfer-serve` | C=1 | C=2 | C=4 | C=8 |
|---|---:|---:|---:|---:|
| No speculation | 82.8 | 105.4 | 114.5 | 116.8 |
| `--spec mtp` | 93.5 | 104.7 | 111.4 | 103.3 |

Decode is bound by expert-cache misses: each miss copies a 2.6 MiB expert over PCIe. Aggregate
throughput therefore levels off at C≥4, where the requests in flight route to more distinct experts.

When Flash-Next support first landed (`e68225b7`), development runs measured about 19.5 tok/s
single-request decode and about 350 tok/s prefill on a 3k-token prompt. The main steps since then:

- a tensor-core W4A4 route for staged prefill experts;
- double-buffered layer staging, which skips cached experts;
- Tensor Core QSA attention for BF16 and FP8 KV;
- the SM-streamed router with a decode expert GEMV;
- a fixed-grid expert fetch. Profiling showed the per-miss grid falling to 20–28 GB/s over PCIe; a
  fixed 64-CTA grid holds about 36 GB/s. That raised serve throughput 13% at C=1, 24% at C=4 and 37%
  at C=8.

## Performance

These numbers were measured on this fork: one RTX 5090, CUDA 13.3, fork commit `19f38b77`.

| Artifact | Prefill 4k | Prefill 16k | `tg128` | `tg128` adaptive MTP | `tg128` MTP K=3 |
|---|---:|---:|---:|---:|---:|
| Qwen3.8-Flash-Next `nvfp4` | 2,938 | 2,877 | 71.0 | 86.5 (68%) | 83.1 (51%) |
| Qwen3.8-27B `nvfp4` | 9,208 | 8,067 | 76.8 | 114.1 (42%) | 117.4 (38%) |
| Qwen3.8-27B `groupwise-int` | 3,027 | 2,872 | 83.1 | 111.0 (41%) | 102.0 (29%) |
| Qwen3.6-35B-A3B `groupwise-int` | 18,171 | 16,463 | 393.3 | 594.9 (72%) | 545.4 (56%) |

All values are tok/s; percentages are MTP draft acceptance.

### Benchmark conditions

- **Tool.** `ninfer_bench` through the public Engine: BF16 KV, CUDA Graphs, 3 measured repetitions
  after 1 warm-up, and the default prefill chunk.
- **Commands.** `-p 512,4096,16384 -n 128 -pg 2048,128` without speculation. With speculation:
  `-n 128 -pg 2048,128 --max-ctx 4096`, plus `--spec mtp` or
  `--spec mtp --draft-tokens 3 --fixed-draft`. `--max-ctx` is no longer needed, because
  `ninfer_bench` now sizes the context for speculative runs itself.
- **`tg128` acceptance.** `tg128` decodes from a one-token seed, so its MTP acceptance depends on the
  generated text.
- **MTP after the bench prompt isn't reported.** After the 2,048-token corpus prompt, drafts are
  accepted 94–100% of the time. That flatters MTP, so those numbers are omitted.
- **Serve table.** `ninfer-serve --max-context 4096 --max-concurrency 8 --kv-capacity auto`. The
  load is `temperature=0`, one warm-up wave then one measured wave of C concurrent 512-token essays.
- **Raw reports.** The JSON reports are kept locally under `profiles/bench/readme_20260925/`.

Upstream's published results use its own methodology and artifacts. They are in the
[performance index](docs/performance.md).

### Adaptive MTP draft length

`--spec mtp` alone selects an adaptive policy with a longest draft of 7 tokens:

- A single request drafts 2, 3, 4 or 7 tokens per round. The length is chosen from its recent
  acceptance and from round times measured at startup.
- Rounds where several requests decode together draft 3. Longer drafts raised no aggregate
  throughput at C=2, 4 or 8.
- On host-resident-expert models, batches of two or more run as ordinary rounds plus an MTP KV
  append.
- `--draft-tokens N` lowers the ceiling, and `--fixed-draft` pins exactly N.

The adaptive policy was compared against fixed K=3 over nine short code-edit, code-generation,
prose and reasoning prompts. The geometric mean was +8.2% on Qwen3.8-27B `groupwise-int` (edits
+19%) and +2.7% on Qwen3.6-35B-A3B.

## Quick start

**Requirements:**

- 64-bit Linux and an RTX 5090.
- A CUDA toolkit supporting `sm_120a`. CUDA 13.3 is validated.
- CMake 3.28 or newer, a C++23 host compiler, Ninja and `pkg-config`.
- FFmpeg development libraries (`libavformat`, `libavcodec`, `libavutil`, `libswscale`) and
  `libcurl >= 7.85`.

```bash
git clone https://github.com/giveen/ninfer-ext.git
cd ninfer-ext
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

`cmake --preset dev` also enables tests and benchmarks; see
[build configuration](docs/maintainer/build-system.md). There is no install target, so run from the
build tree.

Upstream's official v3 artifacts work unchanged:

| Model | Weights | Artifact |
|---|---|---|
| [Qwen3.8-27B](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | `nvfp4` | `qwen3_8_27b_nvfp4.ninfer` |
| [Qwen3.8-27B](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | `groupwise-int` | `qwen3_8_27b.ninfer` |
| [Qwen3.6-27B](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) | `nvfp4` | `qwen3_6_27b_nvfp4.ninfer` |
| [Qwen3.6-27B](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) | `groupwise-int` | `qwen3_6_27b.ninfer` |
| [Qwen3.6-35B-A3B](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | `groupwise-int` | `qwen3_6_35b_a3b.ninfer` |

Artifacts you convert yourself embed this fork's chat template
([conversion guide](docs/weight-conversion.md)). An existing v2 download can be
[upgraded locally](docs/weight-conversion.md#upgrade-an-existing-v2-artifact).

Download an artifact and start a two-lane long-context server:

```bash
hf download neroued/Qwen3.8-27B-nvfp4-NInfer qwen3_8_27b_nvfp4.ninfer --local-dir models

./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --max-context 240000 --kv-capacity 240000 --max-concurrency 2 --kv-dtype fp8 \
  --host-cache-mib 16384 \
  --spec mtp --lm-head-draft --preserve-thinking
```

Send a request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "qwen3.8-27b",
       "messages": [{"role": "user", "content": "Reply with one short sentence."}],
       "max_tokens": 64}'
```

Or run a one-shot CLI request:

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Explain prefill and decode." \
  --max-context 32768 --kv-dtype fp8 --spec mtp --lm-head-draft
```

The answer goes to stdout, and diagnostics plus the timing report go to stderr. See the
[CLI guide](docs/cli.md), the [HTTP serving guide](docs/serving.md) and the
[examples](examples/cli/). The same server also runs in Docker: `docker build --tag ninfer:local .`,
then pass the `ninfer-serve` command with `--host 0.0.0.0` to `docker run --gpus` with the model
directory mounted.

## What else the fork changes

### Context cache and serving

- **Aborted requests keep their prefill.** A cancelled or disconnected request salvages its
  prefilled context as a reusable checkpoint.
- **Automatic anchors.** The Engine anchors the last message boundaries of a conversation. Long
  anchors are spaced geometrically, so agent loops that rewrite recent turns still reuse most of
  the prompt.
- **One Host-tier budget.** `--host-cache-mib` sizes the whole pinned Host tier and replaces
  `--host-state-slots` and `--host-kv-mib`. Its budget goes to more long anchors per owner.
- **Eviction by recency.** Prefix eviction ranks by recency and hits. Under pressure, a prefix is
  demoted to Host when there is room, instead of being destroyed.
- **Robustness.**
  - The Engine worker recovers from a CUDA OOM instead of crashing.
  - Staged prefills can run in several lanes at once.
  - Admission defers instead of throwing when a plan cannot seal.
  - The Device KV lease warns when it cannot grow.
- **Protocol additions.**
  - Anthropic `thinking.display: "omitted"`, with the reasoning carried in the signature.
  - `/v1/models` metadata.
  - Prompt-progress timings.
  - `ignore_eos` and `reasoning.summary`.

[Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
describes the planner.

### Engineering

- **C++23.** The tree uses `std::format`/`std::print`, `std::ranges`, `std::flat_map`,
  `std::move_only_function` and deducing `this`. CMake has no CUDA 23 dialect, so
  `CMakeLists.txt` keeps `CUDA_STANDARD 20` and passes `-std=c++23` to nvcc.
- **One chat template.** [`tools/chat_templates/qwen.jinja`](tools/chat_templates/qwen.jinja)
  serves every Qwen model. It is derived from froggeric's v22.5 template, with patches for
  continuation and tool results. Reasoning effort defaults to `medium`, and reasoning is retained.
- **Upstream PRs merged ahead of upstream:**
  - NVFP4 sparse MoE (#286–#290) and GGUF as a conversion source (#282).
  - Kernels: Q5 K-split MMA (#292), Q4/Q5 group pipelining (#311), single-pass logsumexp (#307),
    fused attention RMSNorm + NVFP4 quantization (#305), and two Q4 quads in flight in sparse-MoE
    decode (#199).
  - Tool-call parser fixes (#299, #309) and the inference speed-of-light estimator (#304).
- **Deliberately not merged: #268.** Folding the sigmoid gate into the attention reduce breaks CUDA
  Graph updates when MTP and ordinary profiles share a topology class.

## Capabilities and limits

**Capabilities:**

- Text with thinking and non-thinking modes.
- Image, multi-image, video and mixed multimodal input (`--vision`).
- Chunked prefill, and CUDA Graph decode of one to eight concurrent requests.
- Speculative decoding:
  - MTP on every model.
  - DFlash (draft windows 1–15) on Qwen3.6-35B-A3B.
  - DFlash2 on Qwen3.8-27B artifacts that carry the companion weights.
- BF16, INT8, FP8, NVFP4 and K8V4 KV storage.
- Private and shared exact-prefix reuse, with Device and Host retention.
- Offline perplexity scoring (`ninfer-perplexity`).
- OpenAI Responses and Chat Completions, and Anthropic Messages. These include streaming, tools,
  token counting and usage.

**Limits:**

- One RTX 5090, one resident model, and a startup-fixed one to eight active requests.
- Bounded FIFO admission.
- No preemption, priority, multi-GPU or distributed serving.
- `--max-context` is the per-request limit. `--kv-capacity` sizes the shared KV pool; `auto` sizes
  it from the memory left after weights.
- Tool calls are parsed and returned to the client; NInfer does not execute them.

## Documentation

- [Documentation index](docs/README.md)
- [CLI](docs/cli.md)
- [HTTP serving](docs/serving.md)
- [Weight conversion and custom recipes](docs/weight-conversion.md)
- [Qwen4Exp model reference](docs/maintainer/qwen4-exp-model.md)
- [Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
- [Upstream performance records](docs/performance.md)
- [Perplexity evaluation](docs/perplexity.md)
- [Contributing](CONTRIBUTING.md)

Each binary's `--help` gives the exact current options.

## Credits and license

NInfer is designed and developed by [Neroued](https://github.com/Neroued). The architecture, the
artifact format and the official artifacts are theirs. Evaluation scores for the official
artifacts are in their [model cards](model-cards/). If NInfer is useful to you, consider
[supporting upstream on Ko-fi](https://ko-fi.com/neroued).

Both projects are licensed under the [Apache License 2.0](LICENSE).

The official artifacts derive from:

- [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B)
- [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- the fixed packed weights of
  [rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm)
- the fixed mixed FP8/NVFP4 weights of
  [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)

Flash-Next artifacts are converted from
[nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4).

Vendored dependencies keep their own licenses under `third_party/`.
