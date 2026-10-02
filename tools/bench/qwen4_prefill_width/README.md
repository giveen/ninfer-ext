# Qwen4Exp prefill-width / expert-cache A/B

Scratch harness behind the `perf(qwen4): widen idle prefill for host-resident experts` change.
It measures the tradeoff between a wider idle prefill step (fewer whole-expert-layer streams per
prompt) and the automatic routed-expert cache it takes VRAM from.

Not a maintained benchmark; it is checked in so the runs and raw logs are not lost. Delete it once
the work it supports is settled.

## Matrix

`bench.py` runs, against one resident `ninfer-serve` per server config:

- prefill (C=1, MTP off): `long_niah` 8k / 64k / 128k / 256k prompts, 3 reps
- decode (MTP fixed K3 and K7, greedy, 256 output): C = 1, 2, 4, 8 waves, 2 reps

Server profiles: `--max-concurrency 8 --max-context 262144 --kv-capacity 262144 --kv-dtype fp8
--expert-cache auto --ngram-residency stream --no-prefix-reuse --greedy --no-thinking`.
Idle widths: `auto` (the model default), `4096` (forced control), `32768` (forced, prefill only).

## Run

```bash
cmake --build build --target ninfer-serve -j
STRATA_AB_DIR=/tmp/strata-ab .venv/bin/python tools/bench/qwen4_prefill_width/bench.py \
    auto 4096 32768:prefill
# decode-only or prefill-only: append ":decode" / ":prefill" to a mode.
.venv/bin/python tools/bench/qwen4_prefill_width/summarize_bench.py results/bench-auto-none.json results/bench-rest.json
```

The earlier 64k-budget A/B is `run_one.sh <width>` / `run_ab.sh <width>` (prefill + decode phases).

## Result: 256k context budget (RTX 5090, CUDA 13.3, Flash-Next NVFP4)

Expert cache: 4096 = 6,804 slots · auto 16384 = 5,974 (−12.2 %) · 32768 = 4,897 (−28.0 %).

Prefill tok/s (C=1, mean of 3):

| prompt | 4096 | auto 16384 | 32768 |
|---:|---:|---:|---:|
| 7,680 | 2,314 | 3,571 (1.54×) | 3,669 |
| 64,512 | 1,870 | 2,363 (1.26×) | 2,483 |
| 130,048 | 1,352 | 1,553 (1.15×) | 1,611 |
| 260,096 | 815 | 875 (1.07×) | 886 |

Decode aggregate tok/s (decode tokens / wall):

| | K3 C1 | K3 C2 | K3 C4 | K3 C8 | K7 C1 | K7 C2 | K7 C4 | K7 C8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 71 | 93 | 109 | 122 | 64 | 80 | 91 | 74 |
| auto | 66 | 87 | 102 | 114 | 58 | 74 | 83 | 56 |
| ratio | 0.93 | 0.94 | 0.94 | 0.93 | 0.91 | 0.93 | 0.91 | 0.76 |

Acceptance is identical (72 % K3 / 47 % K7); the loss is fetch traffic, not quality.

Readings:

- The prefill win decays with context and is nearly gone at 256k: long-context prefill is
  QSA/attention-bound, not expert-streaming-bound. Forcing 32768 over 16384 adds ~1–3 %.
- The decode tax tracks the cache cut and is worst under wide speculation (K7 C8, −24 %), where the
  working set is widest.
- The current default is a net win for long prompts + short output and a net loss for short prompts +
  long output. Closing the gap needs the prefill workspace borrow (Strata's `Prefill::relayout`).

## P0 probe

`probe_p0.py` reads the per-interval expert-cache counters that the same change added to the serve
throughput log (`expert_cache`: hits, misses, interval hit rate, resident slots, per-layer min/max).
Steady decode hit rate at C=1 is ~0.95–0.97 and drifts to ~0.85–0.89 at C=8; `layer_min` ~40–60 vs
`layer_max` ~200–265, i.e. residency is spread across layers rather than piled into the first ones.
Raw records: `results/bench-p0-*.jsonl` (the matching `*.log` files are gitignored).

## Caveats

- One artifact, one GPU, one machine, with a 475 W power limit and unlocked clocks.
- Decode here uses a single AIME prompt with greedy sampling and fixed draft lengths; it is an A/B,
  not the published serving methodology.
- The `results/` JSON/JSONL are frozen evidence; `bench.py` reruns write to `$STRATA_AB_DIR`, not here.
