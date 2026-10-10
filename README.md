# ninfer-ext

ninfer-ext is an extended fork of [Neroued/ninfer](https://github.com/Neroued/ninfer), a
from-scratch C++/CUDA inference engine for Qwen models on one NVIDIA GeForce RTX 5090.

The fork adds four things on top of upstream's engine:

- **Qwen3.8-Flash-Next on one 32 GB GPU.** A ~180B-parameter MoE model whose routed experts live in
  pinned Host memory behind a device expert cache. Stock NInfer cannot load it.
- **Constrained decoding** on the generated content: JSON object and JSON Schema through the
  standard `response_format`/`text.format` fields on all three HTTP protocols, GBNF, choice and
  regex through the `structured_outputs` object, and
  `--json-object`/`--json-schema-file`/`--grammar-file`/`--regex`/`--choice` on the CLI. The vocabulary-wide legal set is compiled once per model and applied per round on the device,
  per verify position, so speculative drafting keeps working; the supported schema subset and the
  current limits are in [constrained decoding](docs/maintainer/constrained-decoding.md).
- **Constrained tool calls**: `tool_choice` `required`/named/`allowed_tools`, `parallel_tool_calls:
  false`, and per-function `strict` schemas, so a published call is complete and its arguments
  satisfy the declared schema. Declarations always reach the prompt; the selection constrains the
  generated calls. See [constrained decoding](docs/maintainer/constrained-decoding.md).
- **Faster speculative decoding on Qwen3.8-27B `groupwise-int`.** Against stock on the same
  artifacts, DFlash2 serving is 13–48% faster at 1–4 concurrent requests, and MTP 8–32% faster.
- **Serving work for long-running agents.** Context-cache salvage and anchoring, one Host-tier
  budget, OOM recovery, and protocol additions.
- **EXL3 quantization for Qwen3.8-27B and Qwen3.8-Flash-Next.** A trellis-coded format at 4.0 and
  3.5 bpw with its own C++/CUDA quantizer and kernels ([EXL3 quantization](#exl3-quantization)). The
  27B artifacts are the smallest of this fork's Qwen3.8-27B builds and the best on both perplexity
  and KL divergence; the Flash-Next ones are about 20–26% smaller than its NVFP4 artifact.
  Nothing upstream can produce or run them.

It is **not** faster than stock everywhere. Qwen3.8-27B `nvfp4` and Qwen3.6-35B-A3B tie with stock,
and stock is ahead in two cases. [Versus stock NInfer](#versus-stock-ninfer) has both sides.

Everything else is upstream's: the architecture, the `.ninfer` v3 artifact format, the official
artifacts, and the OpenAI- and Anthropic-compatible server. The fork tracks upstream `master`
(currently `bace20dc`, the `upstream` remote), and its tree is ported to C++23.

- [Quick start](#quick-start)
- [Recommended settings](#recommended-settings)
- [Versus stock NInfer](#versus-stock-ninfer)
- [Qwen3.8-Flash-Next](#qwen38-flash-next)
- [EXL3 quantization](#exl3-quantization)
- [Performance](#performance)
- [What else the fork changes](#what-else-the-fork-changes)
- [Capabilities and limits](#capabilities-and-limits)
- [Documentation](#documentation)
- [Credits and license](#credits-and-license)

## Quick start

### 1. Requirements

- 64-bit Linux and an RTX 5090 (`sm_120a`). No other GPU is supported.
- A CUDA toolkit supporting `sm_120a`. CUDA 13.3 is validated.
- CMake 3.28 or newer, a C++23 host compiler, Ninja and `pkg-config`.
- FFmpeg development libraries (`libavformat`, `libavcodec`, `libavutil`, `libswscale`) and
  `libcurl >= 7.85`.
- For Qwen3.8-Flash-Next only: about 65 GiB of host RAM for the pinned experts and 127 GB of disk;
  `--ngram-residency stream` keeps the 51 GB n-gram table off RAM ([details](#requirements)).

### 2. Build

```bash
git clone https://github.com/giveen/ninfer-ext.git
cd ninfer-ext
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

This builds `build/apps/ninfer-serve` (HTTP server), `build/apps/ninfer` (one-shot CLI) and
`build/apps/ninfer-perplexity`. There is no install target, so run from the build tree.
`cmake --preset dev` also builds the tests and benchmarks
([build configuration](docs/maintainer/build-system.md)).

### 3. Get a model

Upstream's official v3 artifacts work unchanged:

| Model | Weights | Artifact | Download |
|---|---|---|---|
| Qwen3.8-27B | `nvfp4` | `qwen3_8_27b_nvfp4.ninfer` | [neroued/Qwen3.8-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) |
| Qwen3.8-27B | `groupwise-int` | `qwen3_8_27b.ninfer` | [neroued/Qwen3.8-27B-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) |
| Qwen3.8-27B | `exl3` 4.0 bpw | `qwen3_8_27b_exl3_4bpw.ninfer` | [jabbatheduck/ninfer-ext-models](https://huggingface.co/jabbatheduck/ninfer-ext-models) |
| Qwen3.8-27B | `exl3` 3.5 bpw | `qwen3_8_27b_exl3_3p5bpw.ninfer` | [jabbatheduck/ninfer-ext-models](https://huggingface.co/jabbatheduck/ninfer-ext-models) |
| Qwen3.6-27B | `nvfp4` | `qwen3_6_27b_nvfp4.ninfer` | [neroued/Qwen3.6-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) |
| Qwen3.6-27B | `groupwise-int` | `qwen3_6_27b.ninfer` | [neroued/Qwen3.6-27B-NInfer](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) |
| Qwen3.6-35B-A3B | `groupwise-int` | `qwen3_6_35b_a3b.ninfer` | [neroued/Qwen3.6-35B-A3B-NInfer](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) |
| Qwen3.8-Flash-Next | `nvfp4` | `qwen3.8-flash-next/qwen3_8_flash_next_nvfp4.ninfer` (+ 3 `.part` files) | [jabbatheduck/ninfer-ext-models](https://huggingface.co/jabbatheduck/ninfer-ext-models/tree/main/qwen3.8-flash-next) |
| Qwen3.8-Flash-Next | `exl3` 4.0 bpw | `qwen3.8-flash-next-exl3-4bpw/qwen3_8_flash_next_exl3_4bpw.ninfer` (+ 2 `.part` files) | [jabbatheduck/ninfer-ext-models](https://huggingface.co/jabbatheduck/ninfer-ext-models/tree/main/qwen3.8-flash-next-exl3-4bpw) |
| Qwen3.8-Flash-Next | `exl3` 3.5 bpw | `qwen3.8-flash-next-exl3-3p5bpw/qwen3_8_flash_next_exl3_3p5bpw.ninfer` (+ 2 `.part` files) | [jabbatheduck/ninfer-ext-models](https://huggingface.co/jabbatheduck/ninfer-ext-models/tree/main/qwen3.8-flash-next-exl3-3p5bpw) |

```bash
hf download neroued/Qwen3.8-27B-nvfp4-NInfer qwen3_8_27b_nvfp4.ninfer --local-dir models
hf download jabbatheduck/ninfer-ext-models qwen3_8_27b_exl3_4bpw.ninfer --local-dir models
```

The `exl3` rows and the Qwen3.8-Flash-Next rows are this fork's own artifacts, published on Hugging
Face; they download and run unchanged ([EXL3 quantization](#exl3-quantization),
[Qwen3.8-Flash-Next](#download-and-serve)). Qwen3.8-27B with DFlash2 weights has to be converted
yourself ([instructions](#converting-a-dflash2-artifact)). Converted artifacts embed this fork's chat template
([conversion guide](docs/weight-conversion.md)). An existing v2 download can be
[upgraded locally](docs/weight-conversion.md#upgrade-an-existing-v2-artifact).

Qwen3.8-27B `groupwise-int` has a Q4 W4A8 prefill route. The upstream artifact stores `A16Only` on
the MLP gate/up uses, so enable it on an existing download without reconverting (a fresh conversion
already declares `AllowA8`):

```bash
python3 -m tools.artifact.set_activation_policy models/qwen3_8_27b.ninfer models/qwen3_8_27b_a8.ninfer \
  --parameter '*/mlp/gate' --parameter '*/mlp/up' --policy AllowA8
```

The route measures ~1.6-1.7x the A16 gate/up op and ~1.16x on a 1410-token prefill
([design note](docs/maintainer/w4a8-prefill.md)).

### 4. Serve and send a request

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --max-context 32768 --kv-capacity auto --max-concurrency 2 \
  --spec mtp --draft-tokens 5 --fixed-draft --lm-head-draft
```

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "qwen3.8-27b",
       "messages": [{"role": "user", "content": "Reply with one short sentence."}],
       "max_tokens": 64}'
```

The server speaks OpenAI Chat Completions and Responses, and Anthropic Messages, including
streaming and tools. `--model-id` changes the model name it reports; `--host 0.0.0.0` exposes it
beyond localhost, and `--api-key` requires a key. `GET /metrics` serves Prometheus metrics named
after vLLM's ([Metrics](docs/serving.md#metrics)).

For a one-shot answer without a server:

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Explain prefill and decode." \
  --max-context 32768 --spec mtp --lm-head-draft
```

The answer goes to stdout, and diagnostics plus a timing report go to stderr. The
[CLI guide](docs/cli.md), the [HTTP serving guide](docs/serving.md) and the
[examples](examples/cli/) cover every option; each binary's `--help` lists the current flags.

**Docker.** `docker build --tag ninfer:local .`, then pass the `ninfer-serve` command with
`--host 0.0.0.0` to `docker run --gpus all`, with the model directory mounted.

## Recommended settings

These are the fastest measured settings per model. "Concurrency" is how many requests decode at
once; set `--max-concurrency` to that number, because every extra lane reserves KV and graph memory
whether it is used or not. Where a cell says the workload decides, the numbers behind it are in
[Performance](#performance) and [Versus stock NInfer](#versus-stock-ninfer).

| Model | 1 request | 2–4 concurrent | 8 concurrent |
|---|---|---|---|
| Qwen3.8-27B `nvfp4` | `--spec mtp --draft-tokens 5 --fixed-draft --lm-head-draft` | same | same |
| Qwen3.8-27B `groupwise-int` with DFlash2 weights | `--spec dflash2 --draft-tokens 7 --lm-head-draft` | same | same |
| Qwen3.8-27B `groupwise-int` (official artifact) | `--spec mtp --draft-tokens 5 --fixed-draft --lm-head-draft` | `--spec mtp --lm-head-draft` | `--spec mtp --draft-tokens 5 --fixed-draft --lm-head-draft` |
| Qwen3.8-27B `exl3` 4.0 / 3.5 bpw | `--spec mtp --draft-tokens 3 --fixed-draft` | not measured | not measured |
| Qwen3.6-35B-A3B | `--spec mtp --lm-head-draft` | same at 2; at 4, same for long reasoning and no `--spec` for short prose | `--spec dflash --draft-tokens 7 --lm-head-draft` for long reasoning; no `--spec` for short prose |
| Qwen3.6-27B (both) | not measured; start from the Qwen3.8-27B row of the same weights | | |
| Qwen3.8-Flash-Next | `--spec mtp --draft-tokens 15 --fixed-draft --lookup-drafts auto` | same | same |

Why these:

- **27B models.** Speculative decoding pays at every concurrency. On the long-reasoning load a fixed
  5-token MTP draft beat the adaptive default on `nvfp4` by 3–17%. On `groupwise-int` the adaptive
  default wins at 2–4 requests, and fixed K=5 at 1 and 8. DFlash2 is the fastest mode on
  `groupwise-int` whenever the artifact carries its weights.
- **EXL3.** Only the single-request settings are measured: fixed 3-token MTP drafts, the same
  speculative shape the other 27B artifacts use. Concurrency is not measured yet.
- **35B-A3B.** For one or two requests, adaptive MTP is within 5% of the best fixed draft on long
  reasoning and ahead on short prompts. Speculation hurts short prose from 4 requests up: on
  512-token essays, plain decode is 14% faster than MTP at C=4 and 32% faster at C=8. On long
  reasoning at C=8, DFlash with 7 drafts is the fastest mode.
- **Copy-heavy work (edits, refactors, quoting, repeated tool arguments).** On any MTP model use
  `--spec mtp --draft-tokens 15 --fixed-draft --lookup-drafts auto`. It is the fastest mode measured
  on that traffic and, where nothing repeats, the cost gate falls back to ordinary decode, so it
  matches plain decode rather than regressing.
- **Flash-Next.** Decode is bound by fetching experts over PCIe, so an MTP verify routes up to four
  columns and touches more experts than the accepted drafts save: on 512-token essays plain decode
  was 20% faster than MTP for a single request and 2% faster at C=8. The lookup window is the
  exception — it only runs a wide verify where the text repeats earlier context — so use it for
  copy-heavy work and plain decode otherwise.
- **The adaptive default (`--spec mtp` alone)** is a reasonable choice when the workload is unknown
  or mixed; see [Adaptive MTP draft length](#adaptive-mtp-draft-length).

Other settings that matter for speed:

- **Keep CUDA Graphs and prefix reuse on.** Both are the default; `--no-cuda-graph` and
  `--no-prefix-reuse` exist for debugging and benchmarking only.
- **Size `--max-context` to what you need.** The KV pool comes out of the same device memory as
  everything else. On Flash-Next it comes directly out of the expert cache, and a smaller cache
  means more PCIe fetches.
- **`--kv-dtype fp8`** halves KV memory for long contexts. On Flash-Next it also decoded 3% faster
  after a 2,048-token prompt; on the other models its speed effect hasn't been measured here.
- **Agent loops that resend long histories:** give the Host cache tier room with
  `--host-cache-mib` (for example `16384`) so reusable prefixes survive in pinned RAM.
- **Flash-Next:** keep `--expert-cache auto` (the default). If host RAM is tight, use
  `--ngram-residency stream` so the n-gram table is read from NVMe instead of held in the page
  cache; `auto` maps it when host memory allows.
- **Contexts larger than GPU memory:** add `--kv-stream`; see [KV streaming](#kv-streaming).

Example: Qwen3.8-27B `nvfp4` for four agents with long contexts:

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --max-context 131072 --kv-capacity auto --max-concurrency 4 --kv-dtype fp8 \
  --host-cache-mib 16384 \
  --spec mtp --draft-tokens 5 --fixed-draft --lm-head-draft --preserve-thinking
```

## KV streaming

KV streaming lets a request's context grow past the KV cache that fits in GPU memory. The recent part
of each context stays on the GPU. Older full pages of 64 tokens move to pinned host RAM, and the
attention kernels read them from there. Context length is then bounded by `--max-context` (at most the
model's `max_position_embeddings`) and the RAM you give the Host KV tier, not by VRAM.

### How it works

- Each active request owns an equal share of GPU KV, `--kv-capacity / --max-concurrency`. It borrows
  idle shares and gives them back when another request needs them.
- A request that outgrows its GPU KV moves its oldest full pages to Host KV (`--host-kv-mib`, or
  `--host-cache-mib` for the whole Host tier). The first pages and the recent tail always stay on the
  GPU.
- Each step copies the spilled pages it reads to the GPU one layer at a time, overlapped with the
  previous layer's compute. On Flash-Next the sparse attention reads only the selected tokens from
  RAM and keeps a GPU copy of the attention index.
- Admission reserves Host KV for each request's prompt plus output limit beyond its share. A request
  that does not fit waits for running ones to finish; one that could never fit gets HTTP 400.
- The context cache still works: follow-up turns, branches and shared prefixes reuse spilled KV in
  place instead of prefilling again.

### Usage

Add `--kv-stream` and size the Host KV tier for the spill:

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --max-context 262144 --kv-capacity auto --max-concurrency 2 --kv-dtype fp8 \
  --kv-stream --host-kv-mib 65536
```

- **Host KV size:** each request can spill (prompt + `max_tokens` − share) tokens of KV. KV bytes per
  token are full-attention layers × KV heads × head dimension × 2 (K and V) × bytes per element,
  plus the MTP layer when it is enabled. Size `--host-kv-mib` for the spills you run concurrently.
- **KV dtype:** `--kv-dtype fp8` halves the bytes each spilled step reads, compared with BF16.
- **`--kv-capacity`:** larger shares keep more of each context on the GPU and spill less.
- **GPU overhead:** two staging buffers hold one layer's KV each for a full `--max-context`, about
  1 GB in total at 262k with FP8 and 2 GB with BF16.
- **Pinned RAM:** Host KV is pinned for the Engine lifetime; leave room for the OS and page cache.

### Performance

Requests that fit their GPU share run at the resident speed. Once a request spills, each step reads
its spilled KV over PCIe:

| Workload (context / GPU share) | Streamed | Resident |
|---|---|---|
| 27B `nvfp4`, BF16 KV, 35k / 12k, decode | 31 tok/s | 71 tok/s |
| Flash-Next, 80k prompt / 12k share, decode | 41–44 tok/s | 46–51 tok/s |
| 27B chat, second turn over a spilled 30.6k context, TTFT | 188 ms | 170 ms (fresh prefill: 4.2 s) |

Prefill ran at the resident rate in both cases, and streamed outputs matched resident runs.
Dense-model decode is bound by host-to-GPU copy bandwidth
(~50 GB/s), so it slows as the spilled share of the context grows; Flash-Next barely slows because
it reads only the selected tokens.

### Limits

- DFlash and DFlash2 drafting and offline scoring (`ninfer-perplexity`) do not support `--kv-stream`.
- Tested up to two concurrent ~236k-token FP8 requests on 27B and 80k tokens on Flash-Next.

[Paged KV cache §6.5](docs/maintainer/paged-kv-cache.md) describes the design.

## Versus stock NInfer

Stock upstream `bace20dc` and this fork (Engine as of `19f38b77`) ran on the same RTX 5090, with
the same artifacts, under the same harness.

**Method.** Upstream's decode-saturation suite (`tools/bench/run_serve_concurrency.py`):

- **KV and sampling:** INT8 KV and upstream's stochastic sampling profile. Speculative modes use
  `--lm-head-draft`.
- **Load:** C concurrent requests, each generating 4,096 tokens from a long-reasoning (AIME) prompt.
- **Metric:** steady decode tok/s over intervals whose decode batch equals C.
- **Repeats:** each value is the mean of two runs. A single run can vary by up to about 10%, so
  differences under about 5% are ties.
- **Scheduling:** stock and fork alternated within each round, except the fork's fixed K=5 runs,
  which ran later as a separate session.

**Not yet re-measured.** Two later fork commits affect these rows: a tensor-core A16 kernel for
NVFP4 MLP projections (`e348a5d2`, +5–9% MTP on 27B `nvfp4` in fork-only runs) and the adaptive-MTP
batch draft length (`52b06393`). The tables below predate both.

### Where ninfer-ext leads

Each cell is aggregate decode tok/s from the fork, then the change against stock's best mode at that
concurrency.

| Artifact | `ninfer-serve` flags | C=1 | C=2 | C=4 | C=8 |
|---|---|---:|---:|---:|---:|
| Qwen3.8-27B `groupwise-int` + DFlash2 | `--spec dflash2 --draft-tokens 7 --lm-head-draft` | 260 **+40%** | 403 **+48%** | 522 **+13%** | 727 +3% |
| Qwen3.8-27B `groupwise-int`, MTP only | C=1 and C=8: `--spec mtp --draft-tokens 5 --fixed-draft --lm-head-draft` | 203 **+22%** | | | 658 **+8%** |
| | C=2 and C=4: `--spec mtp --lm-head-draft` | | 308 **+29%** | 520 **+32%** | |
| Qwen3.8-Flash-Next `nvfp4` | see [Qwen3.8-Flash-Next](#qwen38-flash-next) | fork only | fork only | fork only | fork only |

Stock's best, for reference: 186 / 271 / 462 / 705 tok/s with DFlash2, and 167 / 239 / 395 / 607
with MTP only (K=3 at C=1–2, K=5 at C=4–8). Stock rejects the Flash-Next artifact at load with
`tensor: unknown member divisors`.

The `groupwise-int` gains are consistent with the merged Q5 K-split MMA routes for small batches
(upstream PR #292), which target the verify rounds of speculative decoding. NVFP4 artifacts have no
Q5 weights and tie with stock. The attribution has not been isolated with an A/B.

### Where stock is ahead or even

| Artifact | Case | Stock | Fork | Change |
|---|---|---:|---:|---:|
| Qwen3.8-27B `groupwise-int` | plain decode, C=1 / C=2 | 81 / 143 | 76 / 138 | **−6% / −3%** |
| Qwen3.8-27B `nvfp4` | best mode per C (K=5 MTP on both builds) | 219 / 419 / 727 / 1,267 | 217 / 427 / 735 / 1,309 | −1% / +2% / +1% / +3% (tie) |
| Qwen3.8-27B `nvfp4` | MTP K=3, C=4 | 655 | 634 | −3% |
| Qwen3.6-35B-A3B | best mode per C: MTP K=3 at C=1–4, DFlash7 at C=8 | 644 / 926 / 1,173 / 1,516 | 657 / 951 / 1,159 / 1,545 | +2% / +3% / −1% / +2% (tie) |
| Qwen3.8-27B `nvfp4` | fork default `--spec mtp` vs stock `--draft-tokens 5` | 219 / 419 / 727 / 1,267 | 211 / 377 / 640 / 1,122 | **−4% / −10% / −12% / −11%** |
| Qwen3.6-35B-A3B | fork default `--spec mtp` vs stock `--draft-tokens 5`, C=8 | 1,488 | 1,310 | **−12%** |

- **Plain decode on 27B `groupwise-int`.** One of the fork's two runs matched stock, as did four later
  re-runs at 2,048 and 4,096 tokens. The averaged deficit stands as measured; its cause is
  unresolved.
- **The fork's adaptive MTP default was not the best setting on this batched, high-acceptance
  load.** With a fixed K=5 the fork is at least as fast as stock, which is why the
  [recommended settings](#recommended-settings) pin K=5 on 27B `nvfp4`.
- **35B-A3B and 27B `nvfp4` are ties.** The fork adds nothing measurable over stock's best mode on
  these two.

<details>
<summary>Every mode, like for like (tok/s, mean of two runs)</summary>

| Artifact | Mode | Stock C=1 / 2 / 4 / 8 | Fork C=1 / 2 / 4 / 8 |
|---|---|---|---|
| 27B `nvfp4` | plain | 73 / 136 / 238 / 490 | 77 / 146 / 250 / 504 |
| | MTP K=3 | 191 / 374 / 655 / 1,135 | 191 / 373 / 634 / 1,123 |
| | MTP K=5 | 219 / 419 / 727 / 1,267 | 217 / 427 / 735 / 1,309 |
| | MTP adaptive | — | 211 / 377 / 640 / 1,122 |
| 27B `groupwise-int` | plain | 81 / 143 / 228 / 321 | 76 / 138 / 249 / 404 |
| | MTP K=3 | 167 / 239 / 354 / 560 | 177 / 291 / 485 / 612 |
| | MTP K=5 | 151 / 210 / 395 / 607 | 203 / 299 / 465 / 658 |
| | MTP adaptive | — | 174 / 308 / 520 / 649 |
| | DFlash2 K=7 | 186 / 271 / 462 / 705 | 260 / 403 / 522 / 727 |
| 35B-A3B | plain | 376 / 605 / 958 / 1,307 | 380 / 611 / 962 / 1,321 |
| | MTP K=3 | 644 / 926 / 1,173 / 1,319 | 657 / 951 / 1,159 / 1,309 |
| | MTP K=5 | 591 / 812 / 993 / 1,488 | 592 / 812 / 1,014 / 1,499 |
| | MTP adaptive | — | 627 / 921 / 1,142 / 1,310 |
| | DFlash K=7 | 630 / 764 / 876 / 1,516 | 632 / 803 / 879 / 1,545 |

Reproduce by running `run_serve_concurrency.py --serve <build>/apps/ninfer-serve --suite
decode-saturation --decode-tokens 4096 --max-context 16384 --kv-capacity auto` once per build, then
pair the two output directories with `compare_serve_concurrency.py`
([serving benchmarks](tools/bench/README.md#concurrent-serving-benchmark)). The raw reports are kept
locally under `profiles/bench/vs_stock_20260925/`. The harness settings (INT8 KV,
`--no-prefix-reuse`, `--max-pending-requests 1`, the pinned sampling profile) match the measurement;
they are not serving advice.

</details>

### Converting a DFlash2 artifact

No published artifact carries DFlash2. Convert one from `Qwen/Qwen3.8-27B` and the DFlash2 companion
weights:

```bash
python3 -m tools.convert --model /path/to/Qwen3.8-27B --recipe qwen3_8_27b \
  --source dflash2=/path/to/Qwen3.8-27B-DFlash2 --components text,vision,mtp,dflash2 \
  --resource chat_template.jinja=tools/chat_templates/qwen.jinja \
  --proposal --name qwen3.8-27b --out models/qwen3_8_27b_dflash2.ninfer
```

### Converting an EAGLE3 artifact

No published artifact carries EAGLE3 either. Convert one from a Qwen3.6-35B-A3B checkpoint and a
SpecForge `LlamaForCausalLMEagle3` head:

```bash
python3 -m tools.convert --model /path/to/Qwen3.6-35B-A3B-NVFP4 \
  --recipe qwen3_6_35b_a3b_nvfp4 \
  --source quantized=/path/to/Qwen3.6-35B-A3B-NVFP4 \
  --source eagle3=/path/to/Qwen3.6-35B-A3B-Eagle3-Specforge \
  --components text,vision,mtp,eagle3 \
  --resource chat_template.jinja=tools/chat_templates/qwen.jinja \
  --proposal --name qwen3.6-35b-a3b-eagle3 \
  --out models/qwen3_6_35b_a3b_nvfp4_eagle3.ninfer
```

The head used here is the SpecForge EAGLE3 draft for Qwen3.5-35B-A3B
(`jiapingW/Qwen3.5-35B-A3B-Eagle3-Specforge`). The converter normalizes its `midlayer.*` weights,
derives the three target layers it fuses, stores the draft-to-target token map, and inherits the
target's token embedding and tokenizer. Heads in the `speculators`/vLLM layout (nested
`transformer_layer_config`, `layers.0.*` weights, a declared `eagle_aux_hidden_state_layer_ids` and a
partial rotary factor) convert the same way; note that a published head is not automatically better
for your traffic — the EAGLE3 reference measures the choices it knows. Serve it with `--spec eagle3 --draft-tokens N`. Keep
`--proposal`: EAGLE3 does not need the optimized proposal head (it drafts through its own 32k head
and rejects `--lm-head-draft`), but without it the file cannot run the documented
`--spec mtp --lm-head-draft` configuration at all, so EAGLE3 could only be compared against — or
fall back to — the slower full-head MTP. See [EAGLE3](docs/maintainer/eagle3.md) for the draft
graph, the target conditioning, the draft-KV cost and the measured comparison against MTP.

## Qwen3.8-Flash-Next

Qwen3.8-Flash-Next (`Qwen4ExpForCausalLM`) has about 180B parameters: about 121B are 512 routed
experts per layer, and 51B are an n-gram embedding table. Neither fits in 32 GB, and upstream NInfer
does not implement the architecture.

ninfer-ext runs it through the same Engine, CLI and HTTP server as every other model:

- **Experts on the Host.** Routed experts stay NVFP4 (or [EXL3](#flash-next-exl3)) in pinned Host
  memory. Each MoE layer resolves
  its top-10 experts against an LRU device expert cache inside the decode CUDA Graph and copies
  misses over PCIe. `--expert-cache` sizes that cache; `auto` gives it the device memory left after
  the KV floor.
- **Prefill streams whole layers.** Chunks of 103 tokens or more stream a whole expert layer through
  a double-buffered staging bank, copying experts already in the cache device-to-device instead.
  Staged experts run on a W4A4 tensor-core route. Between long chunks the staging banks serve as
  1024 more cache slots. The prefill chunk beside decode defaults to 4,096 tokens; with no decode
  waiting the idle step widens up to 16,384 as memory allows, borrowing the expert cache's top slots
  for its arena and returning them afterwards, so a long prompt streams each layer fewer times at no
  decode cost.
- **N-gram table off the device.** The table is file-mapped through the page cache, or streamed from
  NVMe with batched direct I/O (`--ngram-residency`).
- **Full feature set.** Qwen Sparse Attention with every KV-cache profile (`bf16`, `int8`, `fp8`,
  `nvfp4`, `k8v4`), hyper-connection residual streams, the per-layer embedding, MTP speculative
  decoding and Vision.

The mathematics and state semantics are in the [Qwen4Exp model reference](docs/maintainer/qwen4-exp-model.md);
an FP64 oracle checks them in `tests/models/qwen4_exp/`.

### Requirements

- One RTX 5090.
- About 65 GiB of host RAM for the pinned experts (peak RSS loading and generating one token with
  `--ngram-residency stream`). The 51 GB n-gram table is never fully resident — `mapped` faults its
  rows through the page cache and `stream` reads them from NVMe — so it does not raise that floor,
  but a warm page cache speeds the table up and competes with the pinned experts for RAM.
- About 127 GB of disk for the artifact, split into `.part-NNNN` files next to the `.ninfer` file.

### Download and serve

The artifact is published in
[jabbatheduck/ninfer-ext-models/qwen3.8-flash-next](https://huggingface.co/jabbatheduck/ninfer-ext-models/tree/main/qwen3.8-flash-next),
with its `SHA256SUMS` and conversion report. Download the whole folder; the `.part-NNNN` files must
stay next to the `.ninfer` file:

```bash
hf download jabbatheduck/ninfer-ext-models --include "qwen3.8-flash-next/*" --local-dir models
```

To build it yourself instead, convert
[nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4); it takes
about 7.6 minutes on the development machine:

```bash
python3 -m tools.convert \
  --model /path/to/Qwen3.8-Flash-Next-NVFP4 \
  --recipe qwen3_8_flash_next_nvfp4 \
  --components text,vision,mtp \
  --resource chat_template.jinja=tools/chat_templates/qwen.jinja \
  --name qwen3.8-flash-next \
  --out models/qwen3.8-flash-next/qwen3_8_flash_next_nvfp4.ninfer
```

Serve two concurrent 64k-token requests at full speed:

```bash
./build/apps/ninfer-serve models/qwen3.8-flash-next/qwen3_8_flash_next_nvfp4.ninfer \
  --model-id qwen3.8-flash-next --max-concurrency 2 \
  --max-context 65536 --kv-capacity auto --kv-dtype fp8 --expert-cache auto
```

Long contexts work too (for example `--max-context 229376 --kv-capacity 458752`), at the cost of
expert-cache room and so of decode speed.

### Flash-Next EXL3

The routed experts can also be stored as EXL3 (`exl3_mul1`) at a flat 4.0 or 3.5 bpw, quantized by
`ninfer-quantize --experts-only` from the BF16 weights with per-expert Hessians (the NVFP4 artifact is
the calibration model). Dense projections are Q6 (Q8 where K is not a multiple of 128) and the n-gram
table is 4-bit row-grouped. These artifacts hold Text and MTP, **no Vision**. Prefill runs a grouped
tensor-core route over 64-row jobs, and decode a K-split MMA route; both read the experts through the
same device expert cache. Measured on the development machine (RTX 5090, i9-285K, MTP K=3, fp8 KV,
`--expert-cache auto`):

| | NVFP4 | EXL3 4.0 bpw | EXL3 3.5 bpw |
|---|---|---|---|
| Artifact size | 119 GB | 95.82 GB | 88.11 GB |
| Perplexity, 261k-token mixed text | not run on this text | 3.525 | 3.535 |
| KL divergence vs the NVFP4 artifact | 0 (reference) | 0.0585 | 0.0629 |
| Decode, 7 mixed chat requests, E-cores | not measured | 172.7 tok/s | 186.7 tok/s |
| Expert-cache hit rate on that traffic | not measured | 90.2% | 92.0% |
| Prefill, 8k-token prompt | about 6,250 tok/s (mapped) | about 6,050 tok/s | not measured pinned |

- KL is against the NVFP4 artifact, not BF16; there is no BF16 baseline.
- **Pin the server to the efficiency cores of a hybrid CPU** (`taskset -c 8-23` on the i9-285K). The
  same runs on the P-cores were about 30% slower (121.7 and 133.0 tok/s).
- **Agentic and coding quality is not measured.** A SlopCodeBench pass (mini-swe, greedy, no thinking)
  was inconclusive and had no NVFP4 baseline; the 3.5 bpw run was stopped after a greedy repetition
  loop. Use the model's default sampling for agents, not `--greedy`.
- Long agent contexts prefill slowly (about 250 tok/s at 47k tokens, 44% prompt reuse); not
  investigated.
- Next-layer expert prefetch (the next router applied to this layer's input, fetching its predicted
  misses early) cost 4–8% decode here and is not in the tree.
- Frequency-protected cache admission (LFRU, scoring slots `freq / (age + 1)` instead of by age)
  cost 3.9–4.0% **more** misses than the LRU key and is not in the tree: 32,862 and 32,907 misses
  under LRU against 34,132 (per-expert counts) and 34,173 (sticky frame counts), on one
  23,430-token prompt and 512 greedy tokens with `qwen3_8_flash_next_nvfp4.ninfer` and 8,237
  resident slots. Every run produced byte-identical text.

```bash
hf download jabbatheduck/ninfer-ext-models --include "qwen3.8-flash-next-exl3-3p5bpw/*" --local-dir models
taskset -c 8-23 ./build/apps/ninfer-serve \
  models/qwen3.8-flash-next-exl3-3p5bpw/qwen3_8_flash_next_exl3_3p5bpw.ninfer \
  --model-id flash-next-3p5 --max-context 65536 --kv-capacity auto --kv-dtype fp8 \
  --spec mtp --draft-tokens 3
```

`--kv-capacity auto` returns the KV headroom its `--max-context` cannot use to the expert cache.
Rebuild the artifacts with `tools/convert` and the `qwen3_8_flash_next_exl3` recipe (see the folder
READMEs on Hugging Face).

### Flash-Next speed

Measured 2026-10-02 on one RTX 5090, CUDA 13.3, BF16 KV, CUDA Graphs
([benchmark conditions](#benchmark-conditions)). Fork commit `fc305acd`, which overlaps the expert
fetches with the cache-resident experts' compute, ran interleaved with the commit before it
(`31fcfd74`), alternating run for run.

Single request, `ninfer_bench -pg 2048,128` (a 2,048-token prompt, then 128 decoded tokens):

| Test | `31fcfd74` | `fc305acd` | Change |
|---|---:|---:|---:|
| Prefill, 2,048-token prompt | 2,225 | 2,226 | 0% |
| Decode, no speculation (3 runs each) | 79.1 | **84.2** | +6.5% |
| Decode, MTP K=3 (2 runs each, 97% accepted) | 129.5 | **141.9** | +9.6% |

The benchmark corpus repeats itself, so after its 2,048-token prompt MTP drafts are accepted almost
every time; that flatters MTP. Chat traffic accepts far fewer drafts, as the serving table shows.
Adaptive MTP (`--spec mtp` alone) chooses its draft length from timing and acceptance, and the
same binary can settle on different lengths from run to run: one master run took 66 verify rounds
(126.5 tok/s) and every other run 36. At 36 rounds it measured 143.1 tok/s on `31fcfd74` and
150.8–156.9 on `fc305acd`.

Serving 512-token essays with `ninfer-serve`, aggregate decode tok/s over C concurrent requests
(mean of two runs):

| Mode | Commit | C=1 | C=2 | C=4 | C=8 |
|---|---|---:|---:|---:|---:|
| No speculation | `31fcfd74` | 95.7 | 134.4 | 156.9 | 157.0 |
| | `fc305acd` | **98.5** | **141.6** | **180.6** | **185.3** |
| | change | +2.9% | +5.3% | +15.1% | +18.0% |
| `--spec mtp` | `31fcfd74` | 87.0 | 127.4 | 150.1 | 156.2 |
| | `fc305acd` | 82.2 | 136.8 | 165.0 | 182.1 |
| | change | −5.6%, inconclusive | +7.4% | +10.0% | +16.5% |

The gain grows with concurrency because more requests in flight route to more distinct experts,
so more fetches overlap. The single-request MTP point is unresolved: master's two runs were 100.0
and 74.1 tok/s and `fc305acd`'s 77.5 and 86.9, a spread larger than the difference. Throughput
still levels off from C=4, where the expert cache misses most.

Not re-measured at `fc305acd`: at `7fbabfbc`, prefill reached 3,411 tok/s for a 4,096-token prompt,
3,350 for 16,384 and 431 for 512, and FP8 KV decoded 3% faster than BF16 KV after a 2,048-token
prompt.

**Try it yourself.** Build with `cmake --preset dev` (it includes the benchmarks), download the
artifact ([Download and serve](#download-and-serve)), then:

```bash
W=models/qwen3.8-flash-next/qwen3_8_flash_next_nvfp4.ninfer

# Single request: prefill and decode, with and without MTP (the first table)
./build/bench/ninfer_bench --weights $W -pg 2048,128
./build/bench/ninfer_bench --weights $W -pg 2048,128 --spec mtp --draft-tokens 3 --fixed-draft

# Peak aggregate throughput: 1 and 8 concurrent 512-token requests, plain and adaptive MTP.
# The tool starts and stops ninfer-serve itself and writes its reports to --output.
python3 -m tools.bench.run_serve_concurrency --serve build/apps/ninfer-serve \
  --artifact flash=$W --mode mtp0 --mode mtp_adaptive --sampling greedy \
  --suite decode-saturation --concurrency 1 --concurrency 8 --decode-tokens 512 \
  --max-context 4096 --kv-capacity auto --kv-dtype bf16 --output profiles/bench/flash_try
```

Each run loads the model first (about 40 s from NVMe). The serving tool's prompts are not the
essays above, so expect figures in the same range rather than the same numbers. Your PCIe link and
host memory matter as much as the GPU, because every expert miss crosses PCIe.

**Serve it for real, at full context.** The model's whole 262,144-token window, with up to eight
requests decoding at once:

```bash
./build/apps/ninfer-serve models/qwen3.8-flash-next/qwen3_8_flash_next_nvfp4.ninfer \
  --model-id qwen3.8-flash-next --host 127.0.0.1 --port 8080 \
  --max-context 262144 --kv-capacity 262144 --kv-dtype fp8 --max-concurrency 8
```

For the highest throughput on short requests, use `--max-context 4096 --kv-capacity auto` instead
(BF16 KV, as in the serving table). Once the log prints `listening on`, send a request:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-flash-next", "max_tokens": 512,
  "chat_template_kwargs": {"enable_thinking": false},
  "messages": [{"role": "user", "content": "Write a short essay about the history of the printing press."}]
}'
```

Run verbatim with the full-context command, this first request returned a 456-token essay with
179 ms to the first token at 85.8 tok/s. The first requests after startup run slower than the
table below while the expert cache fills.

Measured at `fc305acd` with the same 512-token essays as the serving table (one run each), plus one
241,003-token prompt (79 of this repository's documentation files concatenated, with a question
about the first one):

| `ninfer-serve` settings | KV pool | C=1 | C=8 | 241k-token prompt |
|---|---|---:|---:|---|
| `--max-context 4096 --kv-capacity auto` (BF16 KV) | auto | 98.5 | 185.3 | does not fit |
| `--max-context 262144 --kv-capacity 262144 --kv-dtype fp8` | 262,144 tokens | 96.7 | 162.7 | answered correctly; 3 min 59 s to the first token |
| `--max-context 262144 --kv-capacity auto --kv-dtype fp8` | 330,880 tokens | 96.3 | 132.3 | answered correctly; 4 min 6 s |

C=1 and C=8 are aggregate decode tok/s. The KV pool is shared by every request in flight, so one
request can use the whole window, or several can split it. Leave `--kv-capacity` explicit at full
context: `auto` reserves more KV than one full-length request needs, and that memory comes out of
the expert cache, so C=8 decoded 19% slower and unevenly (its slowest request ran at 16.5 tok/s
against a mean of 31.9). Prefill slows as the prompt grows: about 1,000 tok/s averaged over the
241k-token prompt, against about 2,200 at 2,048 tokens.

The QSA block selection was the long-context bottleneck: it re-pooled, re-normalized and re-RoPE'd
each block's index keys once per query column, so a 256-column pass repeated that work 256 times.
Pooling them once per call (commit `697ff0c7`, bit-identical) moved the 64k-prompt prefill from
2,363 to 3,498 tok/s and the 256k prompt from 875 to 2,218. Single request, FP8 KV, 262,144-token
context budget, the NIAH fixtures:

| Prompt tokens | Before | After | Speedup |
|---:|---:|---:|---:|
| 7,680 | 3,571 | 3,666 | 1.03x |
| 64,512 | 2,363 | 3,498 | 1.48x |
| 130,048 | 1,553 | 2,908 | 1.87x |
| 260,096 | 875 | 2,218 | 2.53x |

```bash
# Starts ninfer-serve (262,144-token budget, FP8 KV, expert cache auto) and sends the NIAH prompts
# at C=1, three times each, reading prefill seconds from the request log.
STRATA_AB_DIR=/tmp/strata-ab python3 tools/bench/qwen4_prefill_width/bench.py auto:prefill
```

**How it got here.** When Flash-Next support first landed (`e68225b7`), it decoded about 19.5 tok/s
for a single request and prefilled about 350 tok/s. The main steps since then:

- a tensor-core W4A4 route for staged prefill experts, and double-buffered layer staging that skips
  cached experts;
- Tensor Core QSA attention for BF16 and FP8 KV;
- pooling each QSA block's index keys once per select call instead of once per query column
  (`697ff0c7`, bit-identical), which lifts long-context prefill: 64k 1.48x, 128k 1.87x, 256k 2.53x;
- the SM-streamed router with a decode expert GEMV;
- a fixed 64-CTA expert fetch, which holds about 36 GB/s over PCIe where the per-miss grid fell to
  20–28 GB/s (+13% serving at C=1, +37% at C=8);
- staging banks that double as expert-cache slots between long chunks (+6% serving at C=1, +22% at
  C=8);
- decode expert GEMVs that reduce only a job's live tokens and issue their weight loads before the
  job lookup finishes (+13% serving at C=1, +10% at C=8);
- single-column projection splits as views instead of device copies, removing about 218 graph copy
  nodes per decode token (+2.5% at C=1);
- pinned weights and Host KV on prefaulted 2 MiB pages, so the GPU's TLB covers the expert stream
  (+16% prefill, +11% serving at C=1, +32% at C=8);
- cache-route expert fetches on the pager's own stream, with the cache-resident experts' GEMVs
  running while the misses arrive (+6.5% decode after a 2,048-token prompt, +3% serving at C=1,
  +18% at C=8).

The comparison engine used during development, FreeToken with `--moe-backend offload`, measured
about 1,900 tok/s prefill and 77–79 tok/s single-request decode on the same machine (4k context).

## EXL3 quantization

EXL3 is this fork's own weight format: a trellis codebook (`exl3_mul1` + `trellis_t16_v1`) over 16x16
tiles, with the input and output Hadamard rotations folded into the kernels. It is not exllamav3's
checkpoint format and imports nothing from it — `ninfer-quantize` produces the weights from
full-precision source tensors, and the loader and kernels live in this repository.

Two artifacts are published for Qwen3.8-27B, both Text + MTP + Vision (the Qwen3.8-Flash-Next
artifacts, which quantize the routed experts only, are described [above](#flash-next-exl3)):

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
[model card](../model-cards/ninfer-ext-models/README.md).

## Performance

These numbers were measured on this fork: one RTX 5090, CUDA 13.3. The 27B and 35B rows are from
fork commits `19f38b77` and `f8106aa9` (same Engine; only benchmark sizing changed). The Flash-Next
serving rows are from `fc305acd`, built with CUDA 13.3; its single-request row is from `7fbabfbc`,
built with CUDA 13.4.

### Serving throughput

Aggregate decode tok/s from `ninfer-serve`, with C concurrent requests each writing a 512-token
essay (mean of two runs). Bold marks the faster mode at each concurrency.

| Artifact | Mode | C=1 | C=2 | C=4 | C=8 |
|---|---|---:|---:|---:|---:|
| Qwen3.8-27B `nvfp4` | plain | 78.6 | 149.0 | 254.3 | 492.3 |
| | `--spec mtp` | **127.2** | **253.1** | **423.0** | **730.4** |
| Qwen3.8-27B `groupwise-int` | plain | 81.8 | 146.0 | 260.1 | 415.9 |
| | `--spec mtp` | **127.7** | **219.3** | **340.3** | **453.2** |
| Qwen3.6-35B-A3B `groupwise-int` | plain | 386.6 | 619.8 | **972.3** | **1,324.4** |
| | `--spec mtp` | **491.9** | **663.5** | 850.9 | 1,001.0 |
| | `--spec dflash --draft-tokens 7` | 355.4 | 452.1 | 507.7 | 898.9 |
| Qwen3.8-Flash-Next `nvfp4` | plain | **98.5** | **141.6** | **180.6** | **185.3** |
| | `--spec mtp` | 82.2 | 136.8 | 165.0 | 182.1 |

On this prose load, 35B-A3B speculation loses from C=4 up. On the long-reasoning load in
[Versus stock NInfer](#versus-stock-ninfer) it wins instead: MTP K=3 beats plain at C=4 and DFlash7
is fastest at C=8. Which mode wins depends on the workload.

### Single-request benchmark

| Artifact | Prefill 4k | Prefill 16k | `tg128` | `tg128` adaptive MTP | `tg128` MTP K=3 |
|---|---:|---:|---:|---:|---:|
| Qwen3.8-Flash-Next `nvfp4` | 3,411 | 3,350 | 125.2 | 130.2 (52%) | 126.9 (45%) |
| Qwen3.8-27B `nvfp4` | 9,208 | 8,067 | 76.8 | 114.1 (42%) | 117.4 (38%) |
| Qwen3.8-27B `groupwise-int` | 3,027 | 2,872 | 83.1 | 111.0 (41%) | 102.0 (29%) |
| Qwen3.6-35B-A3B `groupwise-int` | 18,171 | 16,463 | 393.3 | 594.9 (72%) | 545.4 (56%) |

All values are tok/s; percentages are MTP draft acceptance. DFlash on Qwen3.6-35B-A3B reaches
385.1 tok/s at `tg128` with 7 drafts (20% accepted) and 219.5 with 15 (7% accepted), against 393.3
without speculation.

### Test system

Every number in this README comes from one machine:

| Component | Details |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 32 GB, PCIe Gen5 x16 |
| NVIDIA driver | 610.57.04 |
| CUDA toolkit | 13.4 (`nvcc` V13.4.92) |
| CPU | Intel Core Ultra 9 285K (24 cores, up to 5.7 GHz) |
| RAM | 247 GiB usable |
| Model storage | WD_BLACK SN850P 8 TB NVMe, ext4 |
| OS | Ubuntu 26.10 (development branch), Linux 7.3.0-5-generic |

Flash-Next keeps its routed experts in pinned host RAM and streams them over PCIe, so its numbers
also depend on host memory and the PCIe link, not only on the GPU. Model load time depends on the
NVMe drive.

### Benchmark conditions

- **Tool.** `ninfer_bench` through the public Engine: BF16 KV, CUDA Graphs, 3 measured repetitions
  after 1 warm-up, and the default prefill chunk.
- **Commands.** `-p 512,4096,16384 -n 128 -pg 2048,128` without speculation. With speculation:
  `-n 128 -pg 2048,128`, plus `--spec mtp`, `--spec mtp --draft-tokens 3 --fixed-draft` or
  `--spec dflash --draft-tokens 7|15`. The Flash-Next FP8 figures add `--kv-dtype fp8`.
- **`tg128`** decodes from a one-token seed, so its speed and MTP acceptance depend on the generated
  text. After the 2,048-token corpus prompt, drafts are accepted 94–100% of the time, which flatters
  MTP, so those numbers are omitted.
- **Serve tables.** `ninfer-serve --max-context 4096 --max-concurrency 8 --kv-capacity auto`, BF16
  KV, `temperature=0` with thinking off: one warm-up wave, then one measured wave of C concurrent
  512-token essays on eight fixed topics.
- **Not yet measured.** The Qwen3.6-27B artifacts.
- **Raw reports** are kept locally under `profiles/bench/readme_20260925/`, and the Flash-Next ones
  under `profiles/bench/readme_20260926b/` (`run.sh` reproduces them). The Flash-Next comparison
  of `fc305acd` against `31fcfd74` is under `profiles/bench/flash_overlap_20261002/` (`ab.sh` and
  `ab_serve.sh` reproduce it).

Upstream's published results use its own methodology and artifacts; they are in the
[performance index](docs/performance.md).

### Adaptive MTP draft length

`--spec mtp` alone selects an adaptive policy with a longest draft of 7 tokens:

- A single request drafts 2, 3, 4 or 7 tokens per round, chosen from its recent acceptance and from
  round times measured at startup.
- Rounds where several requests decode together choose their draft length from measured batch
  rounds.
- On host-resident-expert models, batches of two or more run as ordinary rounds plus an MTP KV
  append.
- `--draft-tokens N` lowers the ceiling, and `--fixed-draft` pins exactly N.

Against fixed K=3 over nine short code-edit, code-generation, prose and reasoning prompts, adaptive
scored a geometric mean of +8.2% on Qwen3.8-27B `groupwise-int` (edits +19%) and +2.7% on
Qwen3.6-35B-A3B. No single MTP setting wins on every model and load: on the long-reasoning load a
fixed K=5 was faster on 27B `nvfp4` and at C=8 on 35B-A3B.

## What else the fork changes

### Prompt-lookup draft source

A host-side suffix index drafts the tokens that followed an earlier occurrence of the current
suffix — a copied file, a quoted document, a repeated tool argument. It is model-free (a trigram
key with four recent positions and a backward extension), and every draft is verified by the target,
so it cannot change the output distribution.

- **`--lookup-drafts off|auto|always`** (MTP only) adds it to the MTP round; with
  **`--draft-tokens 15 --fixed-draft`** the round becomes a wide lookup-only verify that skips the
  MTP draft phases and falls back to ordinary decode when nothing repeats.
- **Cost gate.** The engine measures the wall time of a wide lookup round and of an ordinary round
  and takes lookup only when it is predicted to commit more than their ratio, so `auto` is safe by
  construction. Round times and per-match acceptance are learned per run.
- **Measured**, 400 tokens, greedy, RTX 5090: on copy-heavy edits 35B-A3B goes 390 → 1,610 tok/s
  (×4.1 over plain, +66% over MTP K=7) and Flash-Next 76 → 158 (+33% over MTP K=7); on prose the
  wide window is byte-identical to plain decode and the cost gate never fires. Request logs and the
  CLI summary carry per-request `lookup_*`/`chain_rounds` counters.
- **Rejected alternatives** (recorded so they are not repeated): the last-follower n-gram pool has
  worse recall than the trigram index, chaining MTP with the pool is dominated by the pool-only wide
  window, and the 15-draft verify ceiling is structural (target verify and GDN conv-record cap
  `T` at 16).

[Prompt-lookup suffix drafter](docs/maintainer/lookup-drafter.md) is the authority.

### Context cache and serving

- **Aborted requests keep their prefill.** A cancelled or disconnected request salvages its
  prefilled context as a reusable checkpoint.
- **Automatic anchors.** The Engine anchors the last message boundaries of a conversation. Long
  anchors are spaced geometrically, so agent loops that rewrite recent turns still reuse most of the
  prompt.
- **One Host-tier budget.** `--host-cache-mib` sizes the whole pinned Host tier and replaces
  `--host-state-slots` and `--host-kv-mib`.
- **Eviction by recency.** Prefix eviction ranks by recency and hits; under pressure a prefix is
  demoted to Host when there is room instead of being destroyed.
- **Robustness.** The Engine worker recovers from a CUDA OOM instead of crashing, staged prefills
  can run in several lanes at once, admission defers instead of throwing when a plan cannot seal,
  and the Device KV lease warns when it cannot grow.
- **Protocol additions.** Anthropic `thinking.display: "omitted"` (reasoning carried in the
  signature), `/v1/models` metadata, prompt-progress timings, `ignore_eos` and `reasoning.summary`.

[Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
describes the planner.

### Engineering

- **Faster conversion.** Encoded weights are imported in 64 MiB chunks, and the converter no longer
  does per-object work that grows with model size: Flash-Next converts 2.3× faster. Flash-Next and
  Qwen3.8-27B `groupwise-int` + DFlash2 conversions are byte-identical to the previous converter's
  apart from the random artifact id; the compressed-tensors NVFP4 import behind `qwen3_8_27b_nvfp4`
  has not been re-verified. A sharded checkpoint without its `model.safetensors.index.json` now
  converts from the shard headers.
- **Stock-comparison tooling.** The serving benchmark runners drive a stock upstream `ninfer-serve`
  as well as this fork's, and `compare_serve_concurrency.py` pairs their results.
- **C++23.** `std::format`/`std::print`, `std::ranges`, `std::flat_map`, `std::move_only_function`
  and deducing `this`. CMake has no CUDA 23 dialect, so `CMakeLists.txt` keeps `CUDA_STANDARD 20`
  and passes `-std=c++23` to nvcc.
- **One chat template.** [`tools/chat_templates/qwen.jinja`](tools/chat_templates/qwen.jinja) serves
  every Qwen model. It is derived from froggeric's v22.5 template, with patches for continuation and
  tool results. Reasoning effort defaults to `medium`, and reasoning is retained.
- **Upstream PRs merged ahead of upstream:** NVFP4 sparse MoE (#286–#290), GGUF as a conversion
  source (#282), Q5 K-split MMA (#292), Q4/Q5 group pipelining (#311), single-pass logsumexp (#307),
  fused attention RMSNorm + NVFP4 quantization (#305), two Q4 quads in flight in sparse-MoE decode
  (#199), tool-call parser fixes (#299, #309) and the inference speed-of-light estimator (#304).
  #268 is deliberately not merged: folding the sigmoid gate into the attention reduce breaks CUDA
  Graph updates when MTP and ordinary profiles share a topology class.

## Capabilities and limits

**Capabilities:**

- Text with thinking and non-thinking modes.
- Image, multi-image, video and mixed multimodal input (`--vision`).
- Chunked prefill, and CUDA Graph decode of one to eight concurrent requests.
- Speculative decoding: MTP on every model, DFlash (draft windows 1–15) on Qwen3.6-35B-A3B,
  DFlash2 on Qwen3.8-27B artifacts that carry the companion weights, and EAGLE3 (draft windows
  1–15) on artifacts that carry an EAGLE3 draft head. A host prompt-lookup (suffix)
  draft source rides any MTP model, with a measured cost gate (`--lookup-drafts`).
- BF16, INT8, FP8, NVFP4 and K8V4 KV storage.
- Private and shared exact-prefix reuse, with Device and Host retention.
- KV streaming of contexts beyond GPU memory to pinned host RAM (`--kv-stream`).
- Offline perplexity scoring (`ninfer-perplexity`).
- OpenAI Responses and Chat Completions, and Anthropic Messages, including streaming, tools, token
  counting and usage.
- Prometheus `/metrics` with vLLM-style names: request, token, latency, KV and speculative
  decoding metrics.

**Limits:**

- One RTX 5090, one resident model, and a startup-fixed one to eight active requests.
- Bounded FIFO admission; no preemption, priority, multi-GPU or distributed serving.
- `--max-context` is the per-request limit. `--kv-capacity` sizes the shared KV pool; `auto` sizes
  it from the memory left after weights. With `--kv-stream`, contexts beyond the pool spill to Host KV,
  bounded by `--host-kv-mib`.
- Tool calls are parsed and returned to the client; NInfer does not execute them.
- Gemma 4 31B (`Gemma4ForCausalLM`) generates, scores and serves with chat, tools, thinking with a
  budget, images (`--vision`, converted with `--components text,vision`; one or more images per
  message, no video), and MTP speculation with the official assistant drafter (`--spec mtp`,
  converted with `--components text,mtp --source mtp=<assistant checkpoint>`), but without prefix
  reuse, constrained output, CUDA Graphs or a KV format other than BF16 (`ninfer-perplexity`
  needs `--kv-dtype bf16`); top_k is capped at 20. Measured on the M1 artifact (RTX 5090): prefill
  2,388 tok/s at 1K and 1,837 at 8K; decode 52.6 tok/s, and 100-240 tok/s with adaptive MTP
  depending on the text. See [the Gemma 4 reference](docs/maintainer/gemma4-model.md).

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
