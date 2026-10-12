# Perplexity evaluation

`ninfer-perplexity` measures the causal perplexity produced by a v3 `.ninfer` artifact.
It uses the artifact's tokenizer, Text model, selected Main KV representation, final normalization,
and main output head. It is an offline evaluator, not a serving endpoint or a logits-export API.
Only Text weights and resources are loaded; Vision and speculative components are not required.

## Run the fixed corpus

The repository includes `ninfer-ppl-1m-v1`, a fixed set of 16 independent UTF-8 streams covering
English reference text, English long-form text, Chinese reference text, and NInfer C++/CUDA code.
`full` selects all streams; `--quick` selects one stream from each domain.

```bash
./build/apps/ninfer-perplexity models/qwen3_8_27b_nvfp4.ninfer \
  --corpus eval/corpora/perplexity-1m/manifest.json \
  --quick \
  --kv-dtype fp8
```

The default evaluation uses a 4,096-token context and a 2,048-token stride. Use `--context` and
`--stride` to change that protocol, or score one UTF-8 file with `--text FILE`. The available Main
KV representations are `bf16`, `int8`, `fp8`, `nvfp4`, and `k8v4`.

```bash
./build/apps/ninfer-perplexity models/qwen3_8_27b.ninfer \
  --text notes.txt \
  --context 16384 --stride 8192 \
  --kv-dtype int8
```

Run `./build/apps/ninfer-perplexity --help` for the complete command surface. The evaluator loads
the model once, reads and tokenizes every selected stream before scoring, and writes readable
startup, corpus, scoring, and per-stream summaries to stderr. Interactive weight loading and
scoring use one transient progress line; redirected scoring emits persistent progress every ten
seconds. `--log-level debug` exposes internal startup and stream-begin detail. The final
domain/overall table remains product output on stdout; the independent full-precision machine
report is `report.json` under `profiles/perplexity/` unless `--output` supplies an empty directory.

For KV-format comparisons, the recommended long-context profile is the full corpus with
`--context 65536 --stride 32768` and without `--quick`.

## Metric

For a stream `x[0..N)`, every token after `x[0]` is scored exactly once. A window `[b,e)` with target
suffix `[s,e)` contributes:

```text
log p(x[i] | x[b], ..., x[i-1])  for i in [s,e)
```

Each window starts from empty State and Main KV, so history before `b` is deliberately excluded.
The reported metric is therefore fixed-window, truncated-context causal perplexity:

```text
mean_nll = -sum(logprob) / scored_tokens
perplexity = exp(mean_nll)
```

The first window scores `[1,min(context,N))`. Each later window advances by `stride` targets while
retaining up to `context-stride` preceding tokens as local context. Streams never share history.

## Comparing runs

For a numerical comparison, keep the corpus, context, stride, and execution settings fixed except
the variable being measured. Compare KV formats with the same artifact and weight formats with the
same KV format.

The corpus name is a workload scale, not an exact token count. Exact input and scored-token counts
are runtime results from the current artifact tokenizer and are recorded in each report. Reports
contain unrounded NLL/PPL values for every window, stream, domain, and the token-weighted overall
aggregate.

The schema-v2 report identifies the artifact's architecture, public name, actual weight formats
and prefill signature alongside the workload and numerical results.

## Reference comparison (KLD and ΔNLL)

One run can record the rows it scored and another run can be measured against them. `--write-reference
FILE` writes the selected workload's per-row reference; `--reference FILE` scores the current artifact
against that file. Rows are matched positionally, so the reference must be written and consumed with
the same tokenizer, text or corpus, context, and stride.

```bash
# 1. Record a reference from the full-precision artifact over a fixed text
./build/apps/ninfer-perplexity models/qwen3_8_27b.ninfer \
  --text path/to/fixed.txt \
  --write-reference path/to/fixed.ref

# 2. Score a candidate against it
./build/apps/ninfer-perplexity models/qwen3_8_27b_nvfp4.ninfer \
  --text path/to/fixed.txt \
  --reference path/to/fixed.ref
```

The second run adds a `reference` line and, under `report.json`, a `reference` object carrying the
reference's own numbers alongside the model's:

```text
reference: 79 tokens | mean_nll 2.244805 | ppl_reference 9.438576 | delta_nats -0.000003
```

`delta_nats = mean_nll_model - mean_nll_reference`, so a positive value means the candidate is worse
than the reference. The existing `mean_kl` (per-row KL divergence) answers "how differently
distributed?"; `delta_nats` answers "how much worse in the unit published numbers use?" — for example
a "+0.9% perplexity" claim is an NLL delta in nats, recoverable through the exponential.

**Size.** A reference stores one distribution per scored row over the full vocabulary: at vocab
248,320 a corpus-scale reference for 261,167 scored tokens is ≈129 GB, and even a 79-token one is
≈39 MB, so references are not committable artifacts. Check in the fixed **text** (kilobytes) that both
runs share, and regenerate the reference locally from it; do not store the reference.

**Sanity check.** Scoring an artifact against a reference written by the same artifact must report
`delta_nats ≈ 0` — bf16 reference rounding leaves on the order of 1e-6. Anything larger means the
text, context, or stride differ between the two runs.
