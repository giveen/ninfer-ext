# Qwen3.8-Flash-Next

Qwen3.8-Flash-Next (`Qwen4ExpForCausalLM`) has about 180B parameters: about 121B are 512 routed
experts per layer, and 51B are an n-gram embedding table. Neither fits in 32 GB, and upstream NInfer
does not implement the architecture.

ninfer-ext runs it through the same Engine, CLI and HTTP server as every other model:

- **Experts on the Host.** Routed experts stay NVFP4 (or [EXL3](#flash-next-exl3)) in pinned Host
  memory. Each MoE layer resolves its top-10 experts against an LRU device expert cache inside the
  decode CUDA Graph and copies misses over PCIe. `--expert-cache` sizes that cache; `auto` gives it
  the device memory left after the KV floor. A narrow decode call can split its misses:
  `--host-experts N` computes `--host-expert-share` (default 0.5) of them on N host threads from the
  pinned bank while the admitted rest cross PCIe, and the merge waits for the host rows inside the
  graph. Those experts quantize activations to Q8_0, so with a share decode output depends on which
  experts the cache held; a team larger than the process's CPU allowance minus two is clamped to
  that.
- **Prefill streams whole layers.** Chunks of 103 tokens or more stream a whole expert layer through
  a double-buffered staging bank, copying experts already in the cache device-to-device instead.
  Staged experts run on a W4A4 tensor-core route when the artifact stores the layer's NVFP4 input
  divisors, and on the same route's SIMT A16 branch otherwise. Between long chunks the staging banks serve as
  1024 more cache slots. The prefill chunk beside decode defaults to 4,096 tokens; with no decode
  waiting the idle step widens up to 16,384 as memory allows, borrowing the expert cache's top slots
  for its arena and returning them afterwards, so a long prompt streams each layer fewer times at no
  decode cost.
- **N-gram table off the device.** The table is file-mapped through the page cache, or streamed from
  NVMe with batched direct I/O (`--ngram-residency`).
- **Full feature set.** Qwen Sparse Attention with every KV-cache profile (`bf16`, `int8`, `fp8`,
  `nvfp4`, `k8v4`), hyper-connection residual streams, the per-layer embedding, MTP speculative
  decoding and Vision.

The mathematics and state semantics are in the
[Qwen4Exp model reference](maintainer/qwen4-exp-model.md); an FP64 oracle checks them in
`tests/models/qwen4_exp/`.

## Requirements

- One RTX 5090.
- About 65 GiB of host RAM for the pinned experts (peak RSS loading and generating one token with
  `--ngram-residency stream`). The 51 GB n-gram table is never fully resident — `mapped` faults its
  rows through the page cache and `stream` reads them from NVMe — so it does not raise that floor,
  but a warm page cache speeds the table up and competes with the pinned experts for RAM.
- About 127 GB of disk for the artifact, split into `.part-NNNN` files next to the `.ninfer` file.

## Download and serve

Published in
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

## Flash-Next EXL3

The routed experts can also be stored as [EXL3](quantization.md) (`exl3_mul1`) at a flat 4.0 or
3.5 bpw, quantized by `ninfer-quantize --experts-only` from the BF16 weights with per-expert
Hessians (the NVFP4 artifact is the calibration model). Dense projections are Q6 (Q8 where K is not a
multiple of 128) and the n-gram table is 4-bit row-grouped. These artifacts hold Text and MTP,
**no Vision**. Prefill runs a grouped tensor-core route over 64-row jobs, and decode a K-split MMA
route; both read the experts through the same device expert cache. Measured on the development
machine (RTX 5090, i9-285K, MTP K=3, fp8 KV, `--expert-cache auto`):

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
- **Agentic and coding quality is not measured.** A SlopCodeBench pass (mini-swe, greedy, no
  thinking) was inconclusive and had no NVFP4 baseline; the 3.5 bpw run was stopped after a greedy
  repetition loop. Use the model's default sampling for agents, not `--greedy`.
- Long agent contexts prefill slowly (about 250 tok/s at 47k tokens, 44% prompt reuse); not
  investigated.
- Next-layer expert prefetch (the next router applied to this layer's input, fetching its predicted
  misses early) cost 4–8% decode here and is not in the tree.
- **Host miss split** (`--host-experts N --host-expert-share F`). A narrow call resolves its misses
  with only a share admitted to the cache; the rest are published to a host thread team that
  computes them from the pinned bank while the admitted misses cross PCIe, and the combine waits for
  its rows inside the graph. Measured 2026-10-18, plain C=1 decode (`ninfer_bench -pg 2048,128`,
  E-cores, one session): **86.0–87.5 tok/s without the split against 91.9 with 14 threads at share
  0.5** (+5–7%); 12 threads reached 91.0 and 4 threads 90.2. On the reasoning-shaped `-pg 300,1024`
  it is **95.1–95.3 against 97.8** (+2.6%), though one of four runs there measured 80.6 — the gain
  depends on CPU headroom, because the team competes with the process's own threads. With
  `--spec mtp --draft-tokens 3` it is
  **142.2 against 164.9** at the same 96% acceptance (+16%), because a verify round's wider call has
  more misses to divide. Prefill is unchanged: the split applies only to calls at most `max_columns`
  (16) wide. Computing *every* miss on the host (share 0) remains slower than fetching — 34.35
  against 81.73 tok/s in the 2026-10-02 session — because the arithmetic, not the kernel, is the
  limit: an expert is 4.9M weights in 2.76 MB (1.8 MACs per byte), so the 36–47 GB/s fetch delivers
  about 90 GMAC/s while 16 E-cores reach 54–62 GB/s. The split divides the work until neither side is
  a layer's bottleneck. Host experts round activations to Q8_0 (about 1% relative rms against the
  exact weights), so with a share the decode output depends on what the cache held — the one
  accepted placement dependence. A team that fills the CPU allowance loses 40–60% (16 threads on a
  16-CPU allowance measured 52.4 against 87.7) because the Program's own threads lose a core; the
  lane clamps to allowance minus two and warns. `bench/ops/expert_fetch_bench.cu` measures the fetch
  pieces: SM-issued reads reach 35 GB/s from the pinned 2 MiB-page bank while the copy engine moves
  the same bytes at 47 GB/s.
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

## Speed

Measured 2026-10-02 on one RTX 5090, CUDA 13.3, BF16 KV, CUDA Graphs
([conditions](../README.md#benchmark-conditions)). Fork commit `fc305acd`, which overlaps the expert
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

Decode-width (`T <= 4`) q8 routes are chosen per shape, measured 2026-10-13 on the same box with
`ninfer_linear_bench --qtype Q8 --t 1` (cold L2, one window, an unpatched shape as the drift
control): the 320/324-row, 10,240-wide hyper-connection pair drops from 10.24 to 8.19 us per call
with eight warps per row, and the 1,280-row MoE shared gate from 10.24 to 8.19 us with two. More
warps per row is right only where K is wide — on a 320-wide K it regresses up to 5.2x, because each
warp is left with too little of the reduction — so the remaining shapes keep the generic route.

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

Measured at `fc305acd` with the same 512-token essays (one run each), plus one 241,003-token prompt
(79 of this repository's documentation files concatenated, with a question about the first one):

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

Not re-measured at `fc305acd`: at `7fbabfbc`, prefill reached 3,411 tok/s for a 4,096-token prompt,
3,350 for 16,384 and 431 for 512, and FP8 KV decoded 3% faster than BF16 KV after a 2,048-token
prompt.

### Reasoning-load numbers (decode saturation)

One RTX 5090, BF16 KV, a 300-token AIME prompt and 1,024-token generations per request — the same
protocol as the other models' master charts ([figure](../README.md#master-charts)). Aggregate decode
over intervals whose decode batch equals C, plain against MTP (K=3, full proposal head):

| C | plain | MTP | gain |
|---|---:|---:|---:|
| 1 | 69.7 | 93.4 | 1.34x |
| 2 | 116.1 | 231.3 | 1.99x |
| 4 | 169.2 | 194.7 | 1.15x |
| 8 | 252.5 | 265.5 | 1.05x |

Prefill over the same protocol: 4,861 / 5,232 / 4,468 / **3,319** tok/s at 7,680 / 64,512 / 130,048 /
260,096 tokens. It is the only model here that runs that last prompt — its 262,144-token window fits
where the others' do not.

**MTP wins here and loses on the essays above**, which is the lesson every model in this repository
teaches: the mode that wins is a property of the load, not of the model. On 512-token essays plain
decode beat MTP (98.5 against 82.2 at C=1); these 1,024-token reasoning generations gain 1.05–1.99x.
The two are not a like-for-like contrast in another way either — the essays ran `--lm-head-draft`, the
optimized proposal head, while these ran the full head — so treat the sign of the difference as
load-dependent rather than as a ranking of the two heads.

### Try it yourself

Build with `cmake --preset dev` (it includes the benchmarks), download the artifact
([above](#download-and-serve)), then drive [`ninfer_bench`](../README.md#benchmark-conditions) for
one request or [`run_serve_concurrency.py`](../tools/bench/README.md) for a wave — for example
`--mode mtp0 --mode mtp_adaptive --suite decode-saturation --concurrency 1 --concurrency 8`. Each run
loads the model first (about 40 s from NVMe), the prompts differ from the essays above, and your PCIe
link and host memory matter as much as the GPU because every expert miss crosses PCIe.

## How it got here

When Flash-Next support first landed (`e68225b7`), it decoded about 19.5 tok/s for a single request
and prefilled about 350 tok/s. The steps since then: a tensor-core W4A4 route for staged prefill
experts with double-buffered layer staging; Tensor Core QSA attention for BF16 and FP8 KV; pooling
each QSA block's index keys once per select call (`697ff0c7`, bit-identical); the SM-streamed router
with a decode expert GEMV; a fixed 64-CTA expert fetch holding about 36 GB/s over PCIe where the
per-miss grid fell to 20–28 GB/s (+13% at C=1, +37% at C=8); staging banks that double as cache slots
between long chunks (+6%/+22%); decode expert GEMVs that reduce only a job's live tokens and issue
weight loads before the lookup finishes (+13%/+10%); single-column projection splits as views instead
of device copies (−218 graph copy nodes per decode token, +2.5% at C=1); pinned weights and Host KV
on prefaulted 2 MiB pages (+16% prefill, +11%/+32%); cache-route expert fetches on the pager's own
stream (+6.5% decode after a 2,048-token prompt, +3%/+18%); and a decode miss split that computes a
share of narrow calls' expert misses on host threads beside the PCIe fetch (+5–7% plain and +16% MTP
at C=1).

The comparison engine used during development, FreeToken with `--moe-backend offload`, measured
about 1,900 tok/s prefill and 77–79 tok/s single-request decode on the same machine (4k context).
