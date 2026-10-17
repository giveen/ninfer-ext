# tools/bench

Maintainer orchestration for the public `ninfer_bench` throughput tool, serving corpus/concurrency
runners, and the external Serve TTFT client. Correctness is owned by the affected suites under
[`tests/`](../../tests/README.md).

## External Serve TTFT

[`ttft/README.md`](ttft/README.md) defines the black-box latency benchmark. The measurement runner
uses frozen text/media requests and public streaming protocols without calling Engine. A separate
controller manages the fixed Qwen3.8-27B NVFP4/FP8 Serve profiles and fresh-process isolation.

```bash
python3 tools/bench/run_serve_ttft_campaign.py --campaign resource --samples 5
```

The controller chooses the profile, starts and stops Serve for every sample, runs the external
client, stages the NVFP4 artifact once in `/dev/shm`, stores raw/progress/Serve artifacts below
`profiles/bench/ttft/`, records structured per-request Serve diagnostics, and writes Markdown,
JSON, and CSV summaries. The case catalog, exact profiles, TTFT boundary, and fixture qualification
are documented in the dedicated README.

## Corpus baker

`ninfer_bench` benchmarks prefill at an exact length by slicing the first `P` token ids of a
committed corpus, so the corpus must be real, in-distribution text (not random tokens) and at
least as long as the largest prefill you want to run. `make_bench_corpus.py` bakes that corpus
offline with a local Hugging Face Qwen3.6 tokenizer.

Outputs (committed):

```text
bench/fixtures/bench_corpus.ids            whitespace-separated decimal token ids (exactly --tokens)
bench/fixtures/bench_corpus.manifest.json  tokenizer id, token count, and source description
```

Content sources:

- Built-in curated multi-domain prose (Chinese / English / code / math) — the default. It is
  encoded WITHOUT the chat template or special tokens, then tiled (paragraphs rotated each cycle)
  and truncated to exactly `--tokens`. Repetition only fills length; because prefill/decode
  throughput is token-count / bandwidth bound, it does not bias the numbers.
- `--source-text <file>` (repeatable) — tokenize your own long meaningful text instead, e.g. a
  downloaded public-domain book or a concatenated document set, for genuinely diverse very long
  content. The committed default is `~64k` tokens; raise `--tokens` and/or pass `--source-text`
  for more.

The binary slices `[0:P]`; the manifest is provenance only.

## Requirements

Install the tokenizer dependencies into the active Python environment:

```bash
pip install -r tools/bench/requirements.txt
```

The tokenizer is loaded locally only; the tool never downloads from the network. Pass
`--tokenizer-path` or set `NINFER_TOKENIZER_PATH`.

## Regenerate / check

```bash
# Regenerate the committed corpus from the built-in bank (default 65536 tokens).
python3 tools/bench/make_bench_corpus.py \
  --tokenizer-path /path/to/local/Qwen3.6-27B/tokenizer \
  --tokens 65536

# Bake from your own downloaded/assembled text instead (kept local; not committed).
python3 tools/bench/make_bench_corpus.py \
  --tokenizer-path /path/to/local/Qwen3.6-27B/tokenizer \
  --tokens 131072 --source-text /path/to/book.txt

# Check that the committed .ids and its descriptive manifest agree; no tokenizer or source needed.
python3 tools/bench/make_bench_corpus.py --check
```

`--tokens` is the exact committed corpus size and the ceiling on prefill length; increase it (and
optionally use `--source-text`) to benchmark longer prefills, memory permitting.

## NInfer performance matrix

`run_ninfer_bench_matrix.py` runs the layered public-Engine `ninfer_bench` matrix against the native
`.ninfer` artifact and stores its local reports under `profiles/bench/`. Its defaults are:

```text
artifact: out/qwen3_6_27b.ninfer
binary:   build/bench/ninfer_bench
corpus:   bench/fixtures/bench_corpus.ids
```

The matrix treats MTP `k=3` with the optimized proposal head as the primary path, keeps `k=0` and
`k=5` as controls, and sweeps `k=0..5` on representative context-decode cases. Decode-bearing cases
cover CUDA Graph and eager execution; prefill-only cases vary prompt length and prefill chunk.

```bash
# Configure the benchmark targets once; they are off in the default public build.
cmake -S . -B build -DNINFER_BUILD_BENCHMARKS=ON

# Inspect commands without running the model.
python3 tools/bench/run_ninfer_bench_matrix.py --preset core --dry-run

# Main run. Builds build/bench/ninfer_bench first, then writes JSON and summary.csv.
python3 tools/bench/run_ninfer_bench_matrix.py --preset core

# Longer run that adds 32k/64k prompt and context-decode points.
python3 tools/bench/run_ninfer_bench_matrix.py --preset full

# Run only the MTP draft-window sweep.
python3 tools/bench/run_ninfer_bench_matrix.py --preset full --suite mtp_sweep
```

Default outputs:

```text
profiles/bench/ninfer-<preset>-<timestamp>/
  commands.sh
  manifest.json
  json/<suite>/<case>.json
  logs/<suite>.<case>.stderr.txt
  summary.csv
  summary.json
```

Use `--resume` to skip completed JSON reports in an existing `--output-dir`, and `--preset smoke`
for a minimal script/runner check. `--no-build` uses the binary supplied by `--bench` without
building it.

Each raw report must be `ninfer_bench_report` schema v15. The flattened summary and schema-v4 matrix
manifest carry native facts from the report: architecture, public name, actual formats, prefill signature, artifact,
load/read/upload/staging values, Engine memory arenas including the non-additive Vision layout
inside the unified workspace and CUDA Graph allowance, per-test planned logical and
allocator-observed workspace peaks, KV capacity and
payload, configured proposal head and graph mode, phase timings and throughput, and speculative
rounds/drafts/acceptance/fallbacks. The matrix manifest is descriptive and records the commands and
selected local inputs; it does not make repository state part of report validity.

## Serving corpus benchmark

[Published coverage and model results](../../docs/performance.md) identify the recorded runs.
The [serving methodology](../../docs/performance/methodology.md) owns workload definitions,
metric boundaries, aggregation, comparison rules, and publication format. This section describes
runner usage and output files.

`run_serve_corpus.py` accepts explicit `--artifact LABEL=PATH` entries. Labels identify report groups;
the selected artifact supplies the architecture, public name and weight bindings.
Omitting `--mode` selects MTP0 and MTP3; repeat `--mode` to select a subset.

- **MTP modes.** `mtp3` and `mtp5` pin the draft length. `mtp_adaptive` lets the server choose each
  round's draft, up to 7.
- **DFlash modes.** Use `dflash7` for Qwen3.6-35B-A3B DFlash K=7 and `dflash2_7` for Qwen3.8-27B
  DFlash2 K=7, with companion weights in the selected artifact.
- **EAGLE3 modes.** `eagle3_3` and `eagle3_5` pin the draft length and `eagle3_adaptive` lets the
  server choose each round's draft, up to 7, on artifacts that carry the EAGLE3 draft head. EAGLE3
  drafts through its own head, so the runner never passes `--lm-head-draft` for it.
- **Sampling.** `--sampling greedy` selects exact argmax; the default is stochastic.

The runners also drive a stock upstream `ninfer-serve` (`--serve PATH`):

- **Detection.** The runner reads `--help`: a build without `--fixed-draft` has no adaptive MTP.
- **Fixed modes.** On such a build, fixed MTP modes omit the flag, because that build always drafts
  exactly `--draft-tokens`.
- **Adaptive mode.** `mtp_adaptive` and `eagle3_adaptive` are rejected on it.
- **Request log.** Both upstream's schema-v21 log and this fork's v24 are accepted.
Run commands with a selected Python 3.11 interpreter, as in the model-page reproduction entries.
Both serving runners accept `--kv-dtype bf16|int8|fp8|nvfp4|k8v4` (default: `int8`).
Specify it explicitly when recording a new campaign; the Qwen3.8 performance profile uses
`--kv-dtype fp8`. The runner verifies the loaded KV representation and records the selection in
its results. A serial run cannot resume records collected with a different KV dtype.

The serial runner writes `run.jsonl`, `summary.csv`, `summary.md`, and per-server logs under
`server/`. JSONL contains the completed requests and responses; CSV/Markdown contain fixture and
category summaries. The output directory is supplied explicitly with `--output`.

Its schema-v8 result and flattened summaries retain the KV dtype, actual `prefill_signature`, request Host
exposure, and decode Host/Device-wait time per round received from the schema-v21 serving records.
Request exposure is a latency distribution value and is never summed across concurrent requests;
worker aggregation uses the serving `throughput.host_work` interval deltas. The stochastic route pins its complete
temperature/top-p/top-k/min-p/presence/frequency profile explicitly, so model-default changes do
not alter the measurement method.

## Concurrent serving benchmark

`run_serve_concurrency.py` selects `--suite decode-saturation` or `--suite corpus-makespan`.
Their distinct time boundaries and workload dispatch are defined in the
[serving methodology](../../docs/performance/methodology.md#workloads-and-measurement-boundaries).
Repeat `--concurrency` to select C points; each point starts a fresh server. The point report
records the actual Engine configuration, automatic KV capacity, shuffle seed where applicable,
dispatch method, and per-request positions. Servers use int8 KV unless `--kv-dtype` selects
another storage; Qwen4Exp artifacts need `--kv-dtype bf16` or `fp8`.

Schema-v5 outputs include `points/*.json`, `server/*.jsonl`, and combined `summary.json`, `summary.csv`, and
`summary.md`. A fixture whose prompt exceeds `--max-context` is recorded as skipped in the point and
in every summary rather than failing the campaign. C=1 corpus runs also write complete responses in
`corpus/<point>/results.jsonl` and per-request phase summaries in that directory, which are omitted
when a fixture was refused; older campaigns may have only point reports and server logs. Historical
model pages identify the report directory associated with each table.

```bash
python3 tools/bench/run_serve_concurrency.py \
  --artifact qwen3_6_27b=out/qwen3_6_27b_nvfp4.ninfer \
  --kv-dtype int8 \
  --mode mtp3 --suite decode-saturation \
  --concurrency 1 --concurrency 2 --concurrency 4 \
  --decode-tokens 8192 \
  --output profiles/bench/concurrent-decode

python3 tools/bench/run_serve_concurrency.py \
  --artifact qwen3_6_27b=out/qwen3_6_27b_nvfp4.ninfer \
  --kv-dtype int8 \
  --mode mtp3 --suite corpus-makespan \
  --concurrency 1 --concurrency 2 \
  --output profiles/bench/concurrent-corpus
```

To compare two builds, run the same artifact labels, suites and concurrencies once per
`--serve` binary. Then match the points:

```bash
python3 tools/bench/compare_serve_concurrency.py \
  --baseline stock=profiles/bench/stock --candidate fork=profiles/bench/fork \
  --pair mtp0=mtp0 --pair mtp3=mtp3 --pair mtp5=mtp_adaptive
```

Each `--pair BASE=CANDIDATE` compares two modes; without it, every mode present in both
directories is compared with itself. Decode-saturation rows compare steady decode tok/s, and
corpus rows compare makespan.

Use `--kv-capacity auto` when the fixed corpus needs more shared KV than the default 262,144-token
pool. A point is intentionally not resumable: combining fragments from separate server processes
would not preserve either a steady interval or one continuous makespan.

### Decode-scaling campaign

`run_serve_decode_scaling.py` answers one question for one artifact: does a decode batch of L lanes
make one weight read serve L tokens? It runs the `decode-saturation` suite at C = 1, 2, 4 and 8,
one fresh server per point, and reports the suite's steady `decode_tokens_per_second` (decode
tokens over the steady interval, valid at any concurrency) as both an aggregate and a per-request
rate, with TTFT, the window each point used, free VRAM at load and the resident-expert count.

Its window rule is the reason it is not a thin wrapper: `--max-context` is one request's window and
the attention state is allocated `capacity x lanes`, so every point gets
`window = --lane-token-budget // C`. Holding that product constant keeps the state -- and therefore
the routed experts' residency plan, which is computed from whatever free VRAM is left -- comparable
across the sweep; a single `--max-context` would have multiplied the state by C and either squeezed
the resident banks or refused to load. It also refuses a corpus with ids outside the artifact's
vocabulary before spending a load on it, and it asserts only the engine values a family's own
normalization forces, because the shared runner's preflight prescribes Qwen-family defaults.

```bash
python3 tools/bench/run_serve_decode_scaling.py \
  --artifact nemotron_h_74b_nvfp4=/mnt/storage/models/nemotron-74b/nemotron_h_74b_nvfp4.ninfer \
  --corpus profiles/nsys/nemotron_p1/corpus.ids \
  --concurrency 1 --concurrency 2 --concurrency 4 --concurrency 8 \
  --lane-token-budget 262144 --pad-tokens 8192 --decode-tokens 4096 \
  --output profiles/bench/nemotron_decode_scaling
```

Outputs: `points/<key>.json` (the point report, same shape the concurrency runner writes),
`scaling.json`, `scaling.csv` and `scaling.md`. `--dry-run` prints each point's window and server
command without starting one. The measured context is the padded prompt the point records, so quote
`prompt_tokens` rather than the `--pad-tokens` request; `--pad-tokens` converts at 3.0 chars/token
and the real figure is what the server tokenized.

### Whole-model sweep and figures

`run_bench_sweep.py` drives the runner above over every local artifact at a chain of context levels
and the four concurrencies, then hands the campaigns to `plot_concurrency.py` for the figures. Both
are orchestration, not measurement: the runner owns the server, the workload and the point files, and
the plotter only reads what exists.

A campaign admits one `--max-context`, so a fixture longer than that window is skipped rather than
failing the sweep, and the driver reports which levels a model could not reach. `--dry-run` prints
every campaign without running one. The plotter draws prefill and decode against context, one line
per concurrency, as PNG and SVG; it reads the runner's per-request `corpus/<point>/results.jsonl`
where those exist (C=1, which is where a curve against context is meaningful) in preference to the
point metrics, which for a `corpus-makespan` point describe its whole fixture set at once. It also
reads `ninfer_bench --output-file` JSON, whose rows are single-request points and the only way to get
a prefill-against-context curve without a server.

```bash
# Print every campaign (seven artifacts, contexts 8192..262144, C 1/2/4/8) without running one.
python3 tools/bench/run_bench_sweep.py --dry-run

# Drive one model for real.
python3 tools/bench/run_bench_sweep.py --only gemma4_26b_a4b_qat

# Figures from whatever has been measured; needs matplotlib (here: /usr/bin/python3).
python3 tools/bench/plot_concurrency.py \
  --run profiles/bench/all_models/gemma4_26b_a4b_qat_ctx131072 \
  --bench /tmp/bench_26b.json --out /tmp/figs
```

Measurement on one card is clock-sensitive: run one campaign at a time, and do not benchmark beside
another GPU job — a contended run produced a non-monotonic decode series for the same model whose
uncontended decode is flat.
