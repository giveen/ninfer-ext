# Versus stock NInfer

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

## Where ninfer-ext leads

Each cell is aggregate decode tok/s from the fork, then the change against stock's best mode at that
concurrency.

| Artifact | `ninfer-serve` flags | C=1 | C=2 | C=4 | C=8 |
|---|---|---:|---:|---:|---:|
| Qwen3.8-27B `groupwise-int` + DFlash2 | `--spec dflash2 --draft-tokens 7 --lm-head-draft` | 260 **+40%** | 403 **+48%** | 522 **+13%** | 727 +3% |
| Qwen3.8-27B `groupwise-int`, MTP only | C=1 and C=8: `--spec mtp --draft-tokens 5 --fixed-draft --lm-head-draft` | 203 **+22%** | | | 658 **+8%** |
| | C=2 and C=4: `--spec mtp --lm-head-draft` | | 308 **+29%** | 520 **+32%** | |
| Qwen3.8-Flash-Next `nvfp4` | see [Qwen3.8-Flash-Next](flash-next.md) | fork only | fork only | fork only | fork only |

Stock's best, for reference: 186 / 271 / 462 / 705 tok/s with DFlash2, and 167 / 239 / 395 / 607
with MTP only (K=3 at C=1–2, K=5 at C=4–8). Stock rejects the Flash-Next artifact at load with
`tensor: unknown member divisors`.

The `groupwise-int` gains are consistent with the merged Q5 K-split MMA routes for small batches
(upstream PR #292), which target the verify rounds of speculative decoding. NVFP4 artifacts have no
Q5 weights and tie with stock. The attribution has not been isolated with an A/B.

## Where stock is ahead or even

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
  [recommended settings](speculative-decoding.md#recommended-settings) pin K=5 on 27B `nvfp4`.
- **35B-A3B and 27B `nvfp4` are ties.** The fork adds nothing measurable over stock's best mode on
  these two.

## Stock's numbers beside the fork's latest

The tables above pair both builds under one harness. This one puts stock's numbers beside the fork's
**latest** runs — the decode-saturation pass that also backs the
[master charts](../README.md#master-charts). It uses the *same metric* (steady decode tok/s over
intervals whose decode batch equals C) but a **different protocol**: BF16 KV instead of INT8, a
1,024-token output budget instead of 4,096, and the saturation prompt rather than the full AIME load.
Read it as the fork's current level against stock's published one, not as a delta — the magnitudes are
not comparable across the two protocols.

Decode tok/s, aggregate over C concurrent requests. `--` is a cell that was not measured; there is no
one-lane MTP point in the latest pass for any model.

| Artifact | Mode | Stock (harness above) | Fork, latest |
|---|---|---|---|
| Qwen3.8-27B `nvfp4` | plain | 73 / 136 / 238 / 490 | -- / 153.7 / 290.8 / 559.4 |
| | MTP | 219 / 419 / 727 / 1,267 (K=5) | -- / 323.4 / 575.4 / 1,047.2 |
| Qwen3.8-27B `groupwise-int` | plain | 81 / 143 / 228 / 321 | -- / 145.2 / 258.1 / 428.0 |
| | MTP | 167 / 239 / 395 / 607 (K=3 at C=1–2, K=5 after) | -- / 321.0 / 524.9 / 763.3 |
| | DFlash2 K=7 | 186 / 271 / 462 / 705 | not re-run |
| Qwen3.6-35B-A3B | plain | 376 / 605 / 958 / 1,307 | 406.0 / 654.0 / 1,042.6 / 1,392.0 |
| | MTP K=3 | 644 / 926 / 1,173 / 1,319 | -- / 958.0 / 1,163.9 / 1,368.0 |
| Qwen3.6-35B-A3B `nvfp4` | plain | not measured against stock | 410.0 / 640.0 / 1,000.0 / 1,360.0 |
| | MTP K=3 | not measured against stock | -- / 912.0 / 1,420.0 / 2,142.0 |
| Gemma 4 26B-A4B QAT | plain | stock has no Gemma 4 support | 340.0 / 487.3 / 606.4 / 687.3 |
| | MTP K=3 | stock has no Gemma 4 support | -- / 678.0 / 408.6 / 706.0 |

Acceptance in the latest pass: 0.82–0.85 on Gemma 4 26B-A4B, 0.72–0.75 on 35B-A3B, 0.71–0.74 on
35B-A3B `nvfp4`, 0.81–0.82 on 27B `groupwise-int` and 0.59–0.63 on 27B `nvfp4`. The 26B rows are the
same session that produced its [master chart](../README.md#master-charts), where MTP at four lanes is
*slower* than plain decode — a reminder that the mode that wins is a property of the load, not of the
model.

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
([serving benchmarks](../tools/bench/README.md#concurrent-serving-benchmark)). The raw reports are
kept locally under `profiles/bench/vs_stock_20260925/`. The harness settings (INT8 KV,
`--no-prefix-reuse`, `--max-pending-requests 1`, the pinned sampling profile) match the measurement;
they are not serving advice.

</details>
