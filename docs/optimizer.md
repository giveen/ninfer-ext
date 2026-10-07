# ninfer-optimizer

Find good `ninfer` configuration for a given `.ninfer` artifact **on your machine**, automatically,
using statistically designed experiments instead of a brute-force sweep. Point it at an artifact and
a use case; it detects the hardware and model, runs a small balanced set of benchmarks, and hands
you the **fastest**, **balanced** and **max-context** configurations plus the speed-vs-context
Pareto frontier.

The method is adapted from [bigattichouse/llama-optimize](https://github.com/bigattichouse/llama-optimize)
(Morris + Taguchi design of experiments). The DOE engines are implemented here; there is no
third-party dependency.

```bash
# plan only — detects the box and the model, prints the factor matrix, the run
# count and the exact first command. Uses NO GPU.
python3 -m tools.optimizer models/qwen3_8_27b.ninfer --use-case agents

# just run it — one command, autonomous
python3 -m tools.optimizer models/qwen3_8_27b.ninfer --use-case agents --run

# shrink the sweep and prune the knobs that do not matter
python3 -m tools.optimizer models/qwen3_8_27b.ninfer --run --screen --levels 3 --min-kv fp8
```

## Use cases (`--use-case`)

Most people know *how they run the model*, not which knobs matter. A use case is a runbook: one
name that expands into the representative request shape, the driver, the concurrency, and the
objective.

| `--use-case` | Driver | Request (prompt + gen) | Streams | For |
|---|---|---:|---:|---|
| `chat` (default) | `ninfer_bench` | 512 + 256 | 1 | one interactive user |
| `code` | `ninfer_bench` | 4096 + 512 | 1 | one coding session, long prompts and replies |
| `long-context` | `ninfer_bench` | 32768 + 128 | 1 | RAG / document work, prefill-bound |
| `agents` | `ninfer-serve` | 8192 + 256 | 4 | several autonomous agents, long tool-use prompts |
| `multi-user` | `ninfer-serve` | 1024 + 256 | 8 | many concurrent chat users |
| `batch` | `ninfer-serve` | 8192 + 64 | 8 | offline batch scoring |

The **objective** is decode tokens/s for `chat`; effective tokens/s — `(P+G)/(P/pp + G/tg)`, the rate
the request actually experiences — for `code`, `agents` and `multi-user`; and prefill tokens/s for
the prefill-bound `long-context` and `batch`.

## What it tunes

Built from the artifact and the runbook, then trimmed by `--min-kv` and explicit overrides:

| Factor | Flag | Levels |
|---|---|---|
| KV precision | `--kv-dtype` | `bf16, int8, fp8, nvfp4, k8v4`, floored by `--min-kv` |
| Context depth | `--max-ctx` | fractions of the runbook's floor up to the model's native context |
| Speculative backend | `--spec` | `none`, plus `mtp` / `dflash` / `dflash2` when the artifact has one |
| Proposal head | `--lm-head-draft` | `off`, `on` |
| Prefill chunk | `--prefill-chunk` | `512, 1024, 2048, 4096` |

`--factor NAME=LEVEL[,LEVEL...]` overrides any factor, `--ctx-size N` pins the context depth, and
`--min-kv {any,k8v4,nvfp4,fp8,int8,bf16}` drops lossier KV levels (the default `any` explores all).

## Why designed experiments

With five factors at five levels a full sweep is `5^5 = 3125` GPU runs. The optimizer replaces it
with the same two-stage funnel as the reference:

1. **Morris screening** (`--screen [R]`) walks `R` one-at-a-time trajectories and ranks every knob
   by how much it moves the objective (`mu*`) and how much that depends on the others (`sigma`).
   Negligible knobs are pinned at their best-seen level so the expensive stage never spends runs on
   them.
2. **Orthogonal array** estimates every surviving knob's main effect in 8-125 balanced runs instead
   of thousands. The arrays are generated over GF(p) by projective points, which reproduces the
   standard Taguchi L4/L8/L9/L16/L25/L27/L125 designs. Factors with fewer levels than the column
   count are dummy-mapped rather than capping everyone at their size.

`--levels N` (prime) forces a coarser grid: `2 -> L8`, `3 -> L27`, `5 -> L25`. Fewer levels is a
coarser grid, not a wrong answer.

## Output

Each run is recorded as it finishes (crash-safe CSV under `--output-dir`), then the report prints:

- the **three picks** (fastest / balanced / max-context) with a suggested command;
- the **Pareto frontier** — the configurations where no other measured configuration is both deeper
  and faster;
- the **main-effects table**, ranked by range, so you can see which knobs matter;
- a **fingerprint** (machine, model, factors, run settings) written beside the CSV for sharing.

`--html report.html` writes a **self-contained** report — inline CSS and an SVG Pareto chart, no
external assets — with the picks, confirmation, main effects and every measured row.

## Time to first token

The server driver streams every request and records the time from send to the first content token,
so TTFT is reported for every server-driven run and shown in the picks, the Pareto table and the
HTML report. `--objective ttft` scores a run by TTFT (reported as `1000/ms` so higher is better);
the default objectives stay throughput.

The recommendation is read from measured configurations, not the additive model: main effects can
misattribute an interaction to a main effect.

## Measurement validity

A configuration that OOMs, crashes, times out, fails to parse, or reports a physically impossible
rate is recorded with a status (`OOM`, `SIGNAL`, `TIMEOUT`, `ERROR`, `PARSE_FAIL`, `IMPLAUSIBLE`) and
excluded from the fits and the picks — never silently averaged in as zero. High context at a slow
KV format is *expected* to OOM; that failure maps the memory cliff.

## Thermal state and confirmation

- **`--thermal-mode warm`** preheats each configuration with its own workload until the temperature
  stops rising, then measures there — the sustained rate an already-hot deployment sees.
  **`idle`** settles back toward the idle baseline between runs (the burst rate a bursty workload
  sees). **`off`** (default) measures whatever the previous run left behind. `--thermal-cap` bounds
  the wait and every row records its temperature.
- Execution order is **randomized** (`--seed` reproduces it) so thermal drift does not align with a
  factor level.
- **`--confirm`** (implied by `--full`) re-measures the additive model's predicted-optimal
  configuration and reports predicted vs actual. A small gap means the additive model held; a large
  gap means interactions or drift dominate, so trust the Pareto pick.

## Limitations (current)

- The server driver draws its prompt from the frozen TTFT corpus shape nearest the runbook (falling
  back to a synthetic prompt), streams concurrent OpenAI chat requests, and reads tokens/s and TTFT
  from the live server; it is not the audited TTFT campaign and does not reproduce its case graphs.
- Factor level generation uses the artifact config only; a `--factor` override is the escape hatch
  for anything the runbook gets wrong.

## Build

The optimizer drives the built products; build them first:

```bash
cmake --preset dev && cmake --build build -j --target ninfer_bench ninfer-serve
```

Run it from the repository root with the project Python 3.11 environment:
`python3 -m tools.optimizer MODEL.ninfer --use-case agents`.
