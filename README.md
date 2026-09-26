# ninfer-ext

ninfer-ext is an extended fork of [Neroued/ninfer](https://github.com/Neroued/ninfer). NInfer is a
from-scratch C++/CUDA inference engine for Qwen models on one NVIDIA GeForce RTX 5090.

This fork adds:

- **Qwen3.8-Flash-Next.** The fork runs this ~180B-parameter MoE model on a single 32 GB GPU. Its
  routed experts stay in pinned Host memory behind a device expert cache.
- **Faster speculative decoding on Qwen3.8-27B `groupwise-int`.** Measured against stock NInfer on
  the same artifacts, DFlash2 serving is 13–48% faster at C=1–4, and MTP 8–32% faster at C=1–8.
  Other artifacts tie with stock ([Versus stock NInfer](#versus-stock-ninfer)). MTP can also adapt
  its draft length per round.
- **Serving and context-cache work.** Changes for long-running agent workloads.
- **C++23 and upstream pull requests.** The tree is ported to C++23, and a set of open upstream PRs
  is merged ahead of upstream.

Everything else is upstream's engine: the architecture, the `.ninfer` v3 artifact format, the
official artifacts, and the OpenAI- and Anthropic-compatible serving. The fork tracks upstream
`master` (currently `bace20dc`, the `upstream` remote).

- [Versus stock NInfer](#versus-stock-ninfer)
- [Qwen3.8-Flash-Next](#qwen38-flash-next)
- [Performance](#performance)
- [Quick start](#quick-start)
- [What else the fork changes](#what-else-the-fork-changes)
- [Capabilities and limits](#capabilities-and-limits)
- [Documentation](#documentation)
- [Credits and license](#credits-and-license)

## Versus stock NInfer

Stock upstream `bace20dc` and this fork (Engine as of `19f38b77`) ran on the same RTX 5090, with the
same artifacts, under the same harness.

**Method.** The harness is upstream's decode-saturation suite
(`tools/bench/run_serve_concurrency.py`):
- **KV and sampling:** INT8 KV and upstream's stochastic sampling profile. Speculative modes use
  `--lm-head-draft`.
- **Load:** C concurrent requests, each generating 4,096 tokens from a long-reasoning (AIME)
  prompt.
- **Metric:** steady decode tok/s over intervals whose decode batch equals C.
- **Repeats:** each value is the mean of two runs. A single run can vary by up to about 10%, so
  differences under about 5% are ties.
- **Scheduling:** stock and fork alternated within each round. The exception is the fork's fixed
  K=5 runs, which ran later as a separate session and were not interleaved with stock.

### Where ninfer-ext leads

Each cell is aggregate decode tok/s from the fork's command, then the change against stock's best
mode at that concurrency.

| Artifact | `ninfer-serve` flags | C=1 | C=2 | C=4 | C=8 |
|---|---|---:|---:|---:|---:|
| Qwen3.8-27B `groupwise-int` + DFlash2 | `--spec dflash2 --draft-tokens 7 --lm-head-draft` | 260 **+40%** | 403 **+48%** | 522 **+13%** | 727 +3% |
| Qwen3.8-27B `groupwise-int`, MTP only | C=1 and C=8: `--spec mtp --draft-tokens 5 --fixed-draft --lm-head-draft` | 203 **+22%** | | | 658 **+8%** |
| | C=2 and C=4: `--spec mtp --lm-head-draft` | | 308 **+29%** | 520 **+32%** | |
| Qwen3.8-Flash-Next `nvfp4` | see [Qwen3.8-Flash-Next](#qwen38-flash-next) | fork only | fork only | fork only | fork only |

**Stock's best, for reference:**
- **27B with DFlash2:** DFlash2 at every concurrency, 186 / 271 / 462 / 705 tok/s.
- **27B MTP only:** 167 / 239 / 395 / 607 tok/s, from K=3 at C=1–2 and K=5 at C=4–8.
- **Flash-Next:** stock cannot load it (`tensor: unknown member divisors`).

The gains on `groupwise-int` are consistent with the merged Q5 K-split MMA routes for small batches
(PR #292), which target the verify rounds that speculative decoding runs. NVFP4 artifacts, which
have no Q5 weights, tie with stock. The attribution has not been isolated with an A/B.

To serve with the leading settings, use the flags from the table in a full command. For example,
Qwen3.8-27B `groupwise-int` with DFlash2 at four concurrent requests:

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_dflash2.ninfer \
  --max-context 16384 --kv-capacity auto --max-concurrency 4 --kv-dtype int8 \
  --spec dflash2 --draft-tokens 7 --lm-head-draft
```

**Getting the artifact.** No published artifact carries DFlash2. Convert one from `Qwen/Qwen3.8-27B`
and the DFlash2 companion weights:

```bash
python3 -m tools.convert --model /path/to/Qwen3.8-27B --recipe qwen3_8_27b \
  --source dflash2=/path/to/Qwen3.8-27B-DFlash2 --components text,vision,mtp,dflash2 \
  --resource chat_template.jinja=tools/chat_templates/qwen.jinja \
  --proposal --name qwen3.8-27b --out models/qwen3_8_27b_dflash2.ninfer
```

**Harness settings, not recommendations.** These match the measurements but aren't serving advice:
- INT8 KV (`--kv-dtype int8`);
- `--no-prefix-reuse`;
- `--max-pending-requests 1`;
- the pinned stochastic sampling profile.

Leave prefix reuse on and use normal admission limits for real traffic.

### Where stock is ahead or even

| Artifact | Case | Stock | Fork | Change |
|---|---|---:|---:|---:|
| Qwen3.8-27B `groupwise-int` | plain decode, C=1 / C=2 | 81 / 143 | 76 / 138 | **−6% / −3%** |
| Qwen3.8-27B `nvfp4` | best mode per C (K=5 MTP on both builds) | 219 / 419 / 727 / 1,267 | 217 / 427 / 735 / 1,309 | −1% / +2% / +1% / +3% (tie) |
| Qwen3.8-27B `nvfp4` | MTP K=3, C=4 | 655 | 634 | −3% |
| Qwen3.6-35B-A3B | best mode per C: MTP K=3 at C=1–4, DFlash7 at C=8 | 644 / 926 / 1,173 / 1,516 | 657 / 951 / 1,159 / 1,545 | +2% / +3% / −1% / +2% (tie) |
| Qwen3.8-27B `nvfp4` | fork default `--spec mtp` vs stock `--draft-tokens 5` | 219 / 419 / 727 / 1,267 | 211 / 377 / 640 / 1,122 | **−4% / −10% / −12% / −11%** |
| Qwen3.6-35B-A3B | fork default `--spec mtp` vs stock `--draft-tokens 5`, C=8 | 1,488 | 1,310 | **−12%** |

- **Plain decode on 27B `groupwise-int`.** One of the fork's two runs matched stock, as did four
  later re-runs at 2,048 and 4,096 tokens. The averaged deficit stands as measured, and its cause
  is unresolved.
- **The fork's default adaptive MTP is not the best setting for batched, high-acceptance load.**
  Its batch rounds draft 3 tokens. On this workload a fixed K=5 is faster, and the fork runs fixed
  K=5 at least as fast as stock. Use `--draft-tokens 5 --fixed-draft` there.
- **Adaptive still wins elsewhere.** It leads on 27B `groupwise-int` at C=2–4, and on the shorter
  mixed prompts in [Adaptive MTP draft length](#adaptive-mtp-draft-length).
- **35B-A3B and 27B `nvfp4` are ties.** The fork adds nothing measurable on these two over stock's
  best mode.

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
decode-saturation --decode-tokens 4096 --max-context 16384 --kv-capacity auto` once per build.
Pair the two output directories with `compare_serve_concurrency.py`
([serving benchmarks](tools/bench/README.md#concurrent-serving-benchmark)). The raw reports are
kept locally under `profiles/bench/vs_stock_20260925/`.

</details>

## Qwen3.8-Flash-Next

Qwen3.8-Flash-Next (`Qwen4ExpForCausalLM`) has about 180B parameters. About 121B of them are 512
routed experts per layer, and 51B are an n-gram embedding table. Neither fits in 32 GB, and upstream
NInfer does not implement the architecture. Stock `bace20dc` rejects the artifact at load with
`tensor: unknown member divisors`.

ninfer-ext runs it through the same Engine, CLI, and HTTP server as every other model:

- **Experts on the Host.** Routed experts stay NVFP4 in pinned Host memory. Each MoE layer
  resolves its top-10 experts against an LRU device expert cache inside the decode CUDA Graph and
  copies misses over PCIe. `--expert-cache` sizes that cache; `auto` gives it the device memory left
  after the KV floor.
- **Prefill streams whole layers.** Chunks of 103 tokens or more stream a whole expert layer through
  a double-buffered staging bank. Experts already in the cache are copied device-to-device
  instead. Staged experts run on a W4A4 tensor-core route. Between long chunks the staging banks
  serve as 1024 more cache slots. The prefill chunk defaults to 4,096
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
[nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4).
Conversion takes about 7.6 minutes on this machine (456 s for 127 GB), down from 17.6 minutes
before the fork's converter fixes:

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

All numbers are from one RTX 5090 with BF16 KV and CUDA Graphs, at fork commit `85564b25`.
[Benchmark conditions](#benchmark-conditions) has the details.

| Test | tok/s |
|---|---:|
| Prefill, 4,096-token prompt | 2,906 |
| Prefill, 16,384-token prompt | 2,865 |
| Prefill, 512-token prompt | 368 |
| Decode, `tg128` | 120.9 |
| Decode after a 2,048-token prompt | 67.2 |
| Decode, `tg128`, adaptive MTP | 112.8 (52% accepted) |
| Decode, `tg128`, MTP K=3 | 115.0 (45% accepted) |

`tg128` decodes from a one-token seed, so its speed depends on how many distinct experts the
generated text routes to; decode after a 2,048-token prompt is the steadier figure.

With FP8 KV, prefill is unchanged (2,907 tok/s at 4k, 2,884 at 16k). Decode after a 2,048-token
prompt rises to 68.9 tok/s (+3%).

Serving 512-token essays per request, aggregate decode tok/s (mean of two runs):

| `ninfer-serve` | C=1 | C=2 | C=4 | C=8 |
|---|---:|---:|---:|---:|
| No speculation | 93.1 | 128.6 | 123.0 | 129.1 |
| `--spec mtp` | 72.3 | 124.6 | 116.7 | 134.7 |

Decode is bound by expert-cache misses: each miss copies a 2.6 MiB expert over PCIe. Aggregate
throughput therefore levels off at C≥4, where the requests in flight route to more distinct experts.
On this essay load a single request now decodes faster without MTP: a verify round routes up to four
columns, which touches more experts than the accepted drafts save.

When Flash-Next support first landed (`e68225b7`), development runs measured about 19.5 tok/s
single-request decode and about 350 tok/s prefill on a 3k-token prompt. The main steps since then:

- a tensor-core W4A4 route for staged prefill experts;
- double-buffered layer staging, which skips cached experts;
- Tensor Core QSA attention for BF16 and FP8 KV;
- the SM-streamed router with a decode expert GEMV;
- a fixed-grid expert fetch. Profiling showed the per-miss grid falling to 20–28 GB/s over PCIe; a
  fixed 64-CTA grid holds about 36 GB/s. That raised serve throughput 13% at C=1, 24% at C=4 and 37%
  at C=8.
- the prefill staging banks double as expert-cache slots between long chunks (1024 more slots).
  Serve decode-saturation rose 6% at C=1, 16% at C=4 and 22% at C=8 (82.7 → 87.7, 134.8 → 156.2,
  133.8 → 162.8 tok/s, bf16 KV, no speculation).
- decode expert GEMVs that reduce only a job's live tokens, with a four-lanes-per-row down
  projection that issues its weight loads before the job lookup finishes. Serve decode-saturation
  rose to 89.9 / 132.3 / 155.4 / 172.5 tok/s at C=1/2/4/8 (from 79.7 / 113.6 / 147.8 / 156.9 in the
  same session); decode after a 2k prompt went 67.4 → 70.0 tok/s plain and 103.7 → 108.8 with MTP.

## Performance

These numbers were measured on this fork: one RTX 5090, CUDA 13.3, fork commits `19f38b77` and
`f8106aa9`. The Engine is identical in both; only the benchmark sizing changed.

### Serving throughput

Aggregate decode tok/s from `ninfer-serve`, with C concurrent requests each writing a 512-token
essay (mean of two runs):

| Artifact | Mode | C=1 | C=2 | C=4 | C=8 |
|---|---|---:|---:|---:|---:|
| Qwen3.8-27B `nvfp4` | plain | 78.6 | 149.0 | 254.3 | 492.3 |
| | `--spec mtp` | **127.2** | **253.1** | **423.0** | **730.4** |
| Qwen3.8-27B `groupwise-int` | plain | 81.8 | 146.0 | 260.1 | 415.9 |
| | `--spec mtp` | **127.7** | **219.3** | **340.3** | **453.2** |
| Qwen3.6-35B-A3B `groupwise-int` | plain | 386.6 | 619.8 | **972.3** | **1,324.4** |
| | `--spec mtp` | **491.9** | **663.5** | 850.9 | 1,001.0 |
| | `--spec dflash --draft-tokens 7` | 355.4 | 452.1 | 507.7 | 898.9 |
| Qwen3.8-Flash-Next `nvfp4` | plain | 82.8 | **105.4** | **114.5** | **116.8** |
| | `--spec mtp` | **93.5** | 104.7 | 111.4 | 103.3 |

Bold marks the faster mode at each concurrency.

- **27B models.** MTP pays at every concurrency. On NVFP4 it is still +48% at C=8.
- **35B-A3B.** On this prose load, MTP loses from C=4 up and DFlash7 is slower than plain decode. On
  the long reasoning load in [Versus stock NInfer](#versus-stock-ninfer) both help: MTP K=3 beats
  plain at C=4, and DFlash7 is the fastest mode at C=8. Which mode wins depends on the workload.
- **Flash-Next.** It is bound by expert fetches, so drafts only help a single request.

### Single-request benchmark

| Artifact | Prefill 4k | Prefill 16k | `tg128` | `tg128` adaptive MTP | `tg128` MTP K=3 |
|---|---:|---:|---:|---:|---:|
| Qwen3.8-Flash-Next `nvfp4` | 2,938 | 2,877 | 71.0 | 86.5 (68%) | 83.1 (51%) |
| Qwen3.8-27B `nvfp4` | 9,208 | 8,067 | 76.8 | 114.1 (42%) | 117.4 (38%) |
| Qwen3.8-27B `groupwise-int` | 3,027 | 2,872 | 83.1 | 111.0 (41%) | 102.0 (29%) |
| Qwen3.6-35B-A3B `groupwise-int` | 18,171 | 16,463 | 393.3 | 594.9 (72%) | 545.4 (56%) |

All values are tok/s; percentages are MTP draft acceptance.

DFlash on Qwen3.6-35B-A3B reaches 385.1 tok/s at `tg128` with 7 drafts (20% accepted) and 219.5
with 15 (7% accepted), against 393.3 without speculation.

### Benchmark conditions

- **Tool.** `ninfer_bench` through the public Engine: BF16 KV, CUDA Graphs, 3 measured repetitions
  after 1 warm-up, and the default prefill chunk.
- **Commands.** `-p 512,4096,16384 -n 128 -pg 2048,128` without speculation. With speculation:
  `-n 128 -pg 2048,128`, plus `--spec mtp`, `--spec mtp --draft-tokens 3 --fixed-draft` or
  `--spec dflash --draft-tokens 7|15`. The first MTP runs also passed `--max-ctx 4096`; the bench
  now sizes speculative contexts itself.
- **FP8 KV.** The Flash-Next FP8 figures add `--kv-dtype fp8`.
- **`tg128` acceptance.** `tg128` decodes from a one-token seed, so its MTP acceptance depends on the
  generated text.
- **MTP after the bench prompt isn't reported.** After the 2,048-token corpus prompt, drafts are
  accepted 94–100% of the time. That flatters MTP, so those numbers are omitted.
- **Serve tables.** `ninfer-serve --max-context 4096 --max-concurrency 8 --kv-capacity auto`, BF16
  KV. The load is `temperature=0` with thinking off: one warm-up wave, then one measured wave of C
  concurrent 512-token essays on eight fixed topics.
- **Not yet measured.** The Qwen3.6-27B artifacts. DFlash2 is covered in
  [Versus stock NInfer](#versus-stock-ninfer).
- **Raw reports.** The JSON reports are kept locally under `profiles/bench/readme_20260925/`, and
  the Flash-Next ones under `profiles/bench/readme_20260926/` (`run.sh` reproduces them).

Upstream's published results use its own methodology and artifacts. They are in the
[performance index](docs/performance.md).

To measure this fork against a stock upstream build on the same artifacts and workload, point
`tools/bench/run_serve_concurrency.py --serve` at each build's `ninfer-serve`, then pair the runs
with `tools/bench/compare_serve_concurrency.py`
([serving benchmarks](tools/bench/README.md#concurrent-serving-benchmark)).

### Adaptive MTP draft length

`--spec mtp` alone selects an adaptive policy with a longest draft of 7 tokens:

- A single request drafts 2, 3, 4 or 7 tokens per round. The length is chosen from its recent
  acceptance and from round times measured at startup.
- Rounds where several requests decode together draft 3. On the 512-token essay load, longer drafts
  raised no aggregate throughput at C=2, 4 or 8. That does not hold everywhere. On the long,
  high-acceptance reasoning load in [Versus stock NInfer](#versus-stock-ninfer), fixed K=5 beats
  adaptive on 27B `nvfp4` by 13–17% at C=2–8, and on 35B-A3B by 14% at C=8. Adaptive in turn beats
  fixed K=5 by 13% on 35B-A3B at C=2–4. No single MTP setting wins on every model and load.
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

- **Faster conversion.** Encoded weights are imported in 64 MiB chunks, and the converter no longer
  does per-object work that grows with model size. Flash-Next converts 2.3× faster.
  - Two conversions were re-run and compared with the previous converter's output: Flash-Next
    (ModelOpt NVFP4 and row FP8) and Qwen3.8-27B `groupwise-int` + DFlash2. Both are byte-identical
    apart from the random artifact id.
  - The compressed-tensors NVFP4 import behind `qwen3_8_27b_nvfp4` goes through the same changed
    chunking but has not been re-verified, because its source checkpoint is not on the test
    machine. A sharded checkpoint without its
  `model.safetensors.index.json` now converts from the shard headers.
- **Stock-comparison tooling.** The serving benchmark runners drive a stock upstream
  `ninfer-serve` as well as this fork's, and `compare_serve_concurrency.py` pairs their results.
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
