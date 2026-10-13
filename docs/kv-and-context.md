# KV streaming, context cache and serving

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
Dense-model decode is bound by host-to-GPU copy bandwidth (~50 GB/s), so it slows as the spilled share
of the context grows; Flash-Next barely slows because it reads only the selected tokens.

### Limits

- DFlash and DFlash2 drafting and offline scoring (`ninfer-perplexity`) do not support `--kv-stream`.
- Tested up to two concurrent ~236k-token FP8 requests on 27B and 80k tokens on Flash-Next.

[Paged KV cache §6.5](maintainer/paged-kv-cache.md) describes the design.

## Context cache and serving

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

[Resource scheduling and context cache](maintainer/resource-scheduling-and-context-cache.md) describes
the planner; [HTTP serving](serving.md) covers the protocol surface.
