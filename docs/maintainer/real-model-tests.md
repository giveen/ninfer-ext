# Opt-in real-model Engine tests

Most of the suite is self-contained: it builds synthetic inputs, runs an Op or a planner, and
checks an oracle. Eight tests are different. They load a complete `.ninfer` artifact, construct a
real `Program`, and drive the public Engine, so they need a model on disk, a supported GPU, and
several minutes each. This reference records what each one covers, how to configure and select
them, and why their opt-in surface is shaped the way it is.

[`tests/README.md`](../../tests/README.md) remains the entry point for running the suite; this
document is the authority for the real-model subset.

## Why they are opt-in

A real-model test cannot run from a clean checkout: artifacts are tens of gigabytes and are not in
the repository. Making them ordinary tests would fail every clone. Instead each test returns
`77` (CMake's skip code) when no artifact is configured, so a normal `ctest` reports the suite
green with eight skips.

That skip-by-default behavior is also a known hazard. A green `ctest` does **not** exercise these
tests, and a regression inside them survives an otherwise clean sweep. Run `ctest -L real` (below)
on any change that can affect Engine loading, prefill, decode, prefix reuse, the context cache,
KV storage, attention, speculative backends, Vision, or the proposal head.

## Running the suite

Each test reads its artifact from its own `NINFER_ARTIFACT_*` cache variable. The artifacts are
mutually incompatible — a DFlash2 artifact cannot serve the DFlash test, and a text-only artifact
cannot serve the MoE test — so one shared path cannot cover the suite. Configure the variables you
have and select the `real` label:

```bash
cmake -S . -B build \
  -DNINFER_ARTIFACT_LOADING=$PWD/models/qwen3_8_27b_nvfp4.ninfer \
  -DNINFER_ARTIFACT_PREFIX=$PWD/models/qwen3_8_27b_nvfp4.ninfer \
  -DNINFER_ARTIFACT_SCORE=$PWD/models/qwen3_8_27b_nvfp4.ninfer \
  -DNINFER_ARTIFACT_STREAM=$PWD/models/qwen3_8_27b_nvfp4.ninfer \
  -DNINFER_ARTIFACT_VISION=$PWD/models/qwen3_8_27b_nvfp4.ninfer \
  -DNINFER_ARTIFACT_DFLASH2=$PWD/models/qwen3_8_27b_dflash2.ninfer \
  -DNINFER_ARTIFACT_MOE=$PWD/models/qwen3_6_35b_a3b.ninfer

ctest --test-dir build -L real --output-on-failure
```

Select explicit paths, never glob order or modification time. Run the tests **serially**: each one
loads a model onto the single GPU, so parallel runs exhaust device memory.

`ctest -L real` runs whatever is configured and skips the rest, so the same command works with a
partial artifact set. Re-running `cmake` with a variable set to `""` returns that test to skipping.

Budget for runtime. Each test loads its artifact and uploads weights, so cost is dominated by model
loads, not arithmetic. Most construct one Engine. `stream_real` is the outlier: it builds about
fifteen, one per scenario, because each scenario needs different Engine options (streaming window,
concurrency, MTP). It is therefore the slowest test by a wide margin and the first to hit an
external process or job timeout. On a host with such a limit, run it on its own; its scenarios are
independent, so a subset in one process is equivalent to the whole set.

## The tests

| Variable | Test | Artifact carry | Covers |
|---|---|---|---|
| `NINFER_ARTIFACT_LOADING` | `ninfer_qwen3_5_loading_real_test` | text + MTP + Vision + optimized proposal | semantic binding and GPU materialization: every selected parent's bytes survive H2D, the tokenizer/template survive Reader teardown, native inputs resolve |
| `NINFER_ARTIFACT_PREFIX` | `ninfer_qwen3_5_prefix_real_test` | official or MTP | prefix reuse, MTP, context-cache retention and eviction, resource settlement, Vision, and long-anchor scenarios |
| `NINFER_ARTIFACT_SCORE` | `ninfer_qwen3_5_score_real_test` | official or MTP | a full 1,024-column CausalScoring tile, overlapping target suffixes, repeated-window State/KV isolation, and the exported-logits/log-probability agreement |
| `NINFER_ARTIFACT_STREAM` | `ninfer_qwen3_5_stream_real_test` | official or MTP | concurrent prefill with interleaved chunks, Host KV streaming under a small Device window, two-turn resume, and MTP streaming |
| `NINFER_ARTIFACT_VISION` | `ninfer_qwen3_5_vision_workspace_test` | text + MTP + Vision | the single-image workspace bound: raising total context past the 16K item limit does not grow the planned workspace |
| `NINFER_ARTIFACT_DFLASH2` | `ninfer_qwen3_5_dflash2_real_test` | DFlash2 draft | output budgets, speculative activity, penalty sampling, compact unequal-budget batches, same-route same-seed replay, and the retained frontier across the page boundary |
| `NINFER_ARTIFACT_MOE` | `ninfer_qwen3_5_moe_real_test` | 35B-A3B MoE | the MoE route, MTP and prefix behavior, Vision, and the maximum configuration |
| `NINFER_ARTIFACT_DFLASH` | `ninfer_qwen3_5_dflash_real_test` | plain DFlash draft | the DFlash backend: vision DFlash, concurrent full-head Graph requests, the optimized head, the boundary fixture, and partial terminal handling |

`ninfer_qwen3_5_loading_real_test` is also runnable directly, which is the fastest way to check a
new artifact without the suite:

```bash
./build/tests/ninfer_qwen3_5_loading_real_test \
  --artifact models/qwen3_8_27b_nvfp4.ninfer --vision --speculative mtp --proposal optimized
```

Add `--host-only` to check semantic binding without uploading weights; that path does not construct
a Program or establish native Op support. `--help` lists the accepted options.

`NINFER_PREFIX_REAL_SCENARIO` selects one prefix scenario such as `vision`, `pressure-resume`, or
`concurrent`; the default is `all`.

## Selecting an artifact that does not carry the component

A test's artifact variable names the model, not the component the test needs. When the artifact is
missing a requested component, the loader raises an `ArtifactError`; these tests treat that as
"not applicable" and return `77` with a `skip:` message, exactly like an unset variable. Two
messages qualify:

- `missing component <name>` — the artifact has no such component. For example the DFlash test
  needs a plain `dflash` draft, and the only local draft artifact carries `dflash2`.
- `selected proposal head is absent from artifact` — the artifact was converted without
  `--proposal`, so a test that asks for a proposal head (MTP with `--lm-head-draft`, DFlash2)
  cannot run.

Anything else still fails: a truncated or corrupt artifact, an unreachable GPU, or a numerical
mismatch returns `1`. The skip path is deliberately narrow so a broken artifact is never mistaken
for a component the build does not ship.

## Adding or changing a test

- Register the test with `ninfer_add_real_test(<target> NINFER_ARTIFACT_<NAME> ...)` in
  `tests/models/qwen3_5/tests.cmake`. It adds the `real` label, `SKIP_RETURN_CODE 77`, and the
  per-test artifact environment or command-line argument.
- Declare the matching `set(NINFER_ARTIFACT_<NAME> "" CACHE FILEPATH ...)` next to the others.
- Read the artifact from `NINFER_TEST_ARTIFACT` and skip with `77` when it is unset, then wrap the
  body so a component mismatch becomes a skip: `catch (const std::exception& error) { return
  ninfer::test::real_test_error(error); }` from
  [`tests/models/qwen3_5/real_test_artifact.h`](../../tests/models/qwen3_5/real_test_artifact.h).
- Use the artifact's declared component selection in the test's `EngineOptions` or
  `LoadOptions`; do not assume a component the artifact does not carry.
