# NInfer tools

`tools/` contains artifact conversion and inspection, benchmark orchestration, and serving smoke
checks. To download and run an existing artifact, start with the [project README](../README.md).
To build your own weights, use the [weight conversion guide](../docs/weight-conversion.md).

Run commands from the repository root with a Python environment containing the dependencies
for the selected tool. The maintained environment uses Python 3.11.

Python tools are independent of CMake; there is no `NINFER_BUILD_TOOLS` option.

## Task index

| Task | Location |
|---|---|
| Convert weights with an official or custom recipe | [`convert/`](convert/); [user guide](../docs/weight-conversion.md) |
| Sample disjoint EXL3 traces through `ninfer-serve` | [`exl3/sample_traces.py`](exl3/sample_traces.py) |
| Inspect artifact metadata and objects | [`artifact/inspect.py`](artifact/inspect.py) |
| Swap an artifact's trailing chat-template resource without reconverting | [`artifact/replace_resource.py`](artifact/replace_resource.py); [usage](#artifact-workflow) |
| Enable a route on an already-downloaded artifact (`activation_policy` metadata only) | [`artifact/set_activation_policy.py`](artifact/set_activation_policy.py); [usage](#artifact-workflow) |
| Attach a donor's DFlash2 component to a v3 artifact without requantizing | [`artifact/attach_dflash2.py`](artifact/attach_dflash2.py); [guide](../docs/weight-conversion.md#attach-an-existing-dflash2-component) |
| One-time upgrade of official v2 artifacts | [`upgrade_ninfer_v2_to_v3.py`](upgrade_ninfer_v2_to_v3.py), with positional `INPUT OUTPUT` paths |
| Run benchmark matrices | [`bench/`](bench/README.md) |
| Measure external Serve TTFT | [`bench/ttft/`](bench/ttft/README.md) |
| Exercise a resident HTTP server | [`smoke/serve_contract.py`](smoke/serve_contract.py) |
| Exercise thinking preservation through a managed server | [`smoke/serve_thinking_preservation.py`](smoke/serve_thinking_preservation.py) |
| Measure the physical HBM read/copy ceiling | [`hbm_bandwidth_probe.cu`](hbm_bandwidth_probe.cu); [build command](#standalone-hbm-probe) |
| Simulate suffix/n-gram drafter designs on token ledgers | [`spec_sim/ngram.py`](spec_sim/ngram.py); see [§9.4](../docs/maintainer/lookup-drafter.md#94-pool-design-evaluation) |

## Standalone HBM probe

This maintainer probe has an explicit standalone CUDA build, independent of the CMake benchmark
targets. Build it with the project's CUDA toolkit and run it from the repository root:

```bash
mkdir -p build
nvcc -O3 -std=c++23 -arch=sm_120a tools/hbm_bandwidth_probe.cu \
  -o build/hbm_bandwidth_probe
./build/hbm_bandwidth_probe
```

## Artifact workflow

The common converter reads selected local sources and writes a `.ninfer` artifact plus its
`.conversion.json` report. These examples include the optional weights used by the official
artifacts. The input paths are placeholders for local checkpoint checkouts:

```bash
python3 -m tools.convert \
  --model /path/to/Qwen3.6-27B \
  --recipe qwen3_6_27b --components text,vision,mtp --proposal \
  --resource chat_template.jinja=tools/chat_templates/qwen.jinja \
  --name qwen3.6-27b \
  --out out/qwen3_6_27b.ninfer

python3 -m tools.convert \
  --model /path/to/Qwen3.8-27B \
  --recipe qwen3_8_27b --components text,vision,mtp,dflash2 --proposal \
  --source dflash2=/path/to/Qwen3.8-27B-DFlash2 \
  --resource chat_template.jinja=tools/chat_templates/qwen.jinja \
  --name qwen3.8-27b \
  --out out/qwen3_8_27b.ninfer

python3 -m tools.convert \
  --model /path/to/Qwen3.6-35B-A3B-base \
  --recipe qwen3_6_35b_a3b --components text,vision,mtp,dflash --proposal \
  --source dflash=/path/to/Qwen3.6-35B-A3B-DFlash \
  --resource chat_template.jinja=tools/chat_templates/qwen.jinja \
  --name qwen3.6-35b-a3b \
  --out out/qwen3_6_35b_a3b.ninfer
```

Inspect a result:

```bash
python3 -m tools.artifact.inspect out/qwen3_6_27b.ninfer --objects
```

Replace a stored resource, such as the chat template, in an existing single-file artifact without
reconverting. Every weight byte is kept, the resource must be the last object in the payload, and the
result is written to a new path with a new `artifact_id`:

```bash
python3 -m tools.artifact.replace_resource in.ninfer out.ninfer \
  --resource frontend/chat_template.jinja=tools/chat_templates/qwen.jinja
```

Enable a route that the stored weights already support without reconverting. A Use's
`activation_policy` is the permission for the activation precision at that parameter's mathematical
inputs, so rewriting it is metadata-only: every weight byte is kept, the payload start does not
move, and the result is written to a new path with a new `artifact_id`. For example, the Q4 W4A8
prefill route is enabled on the groupwise-int MLP gate/up uses:

```bash
python3 -m tools.artifact.set_activation_policy in.ninfer out.ninfer \
  --parameter '*/mlp/gate' --parameter '*/mlp/up' --policy AllowA8
```

Recipes, mixed sources, custom methods, resources and sharding are described in the
[conversion guide](../docs/weight-conversion.md). Numeric formats, layouts and framing are defined
by the references linked from the [documentation map](../docs/README.md).

## Benchmark orchestration

`tools/bench/run_ninfer_bench_matrix.py` builds and runs the public-Engine benchmark matrix and
writes ignored local reports below `profiles/bench/`:

```bash
python3 tools/bench/run_ninfer_bench_matrix.py --preset core --dry-run
python3 tools/bench/run_ninfer_bench_matrix.py --preset core
```

See [`tools/bench/README.md`](bench/README.md) and [`bench/README.md`](../bench/README.md) for the
orchestrator and executable contracts.

For request-arrival latency, use the managed Qwen3.8-27B NVFP4/FP8 TTFT campaign. Its measurement
runner remains an external-only HTTP client; the separate controller owns Serve lifecycle and
artifacts. See [`tools/bench/ttft/README.md`](bench/ttft/README.md).

## Serving smoke

After starting `ninfer-serve` in another terminal:

```bash
python3 -m tools.smoke.serve_contract \
  --base-url http://127.0.0.1:18080 \
  --model qwen3.6-27b
```

The client exercises OpenAI, Anthropic, streaming, usage, multimodal, and tool-call response
surfaces against the resident process.

## EXL3 trace preparation

`tools/exl3/sample_traces.py` starts `ninfer-serve`, samples chat continuations and a small
tool-call slice from the fixed perplexity corpus, renders conversations with the artifact's own
tokenizer and chat template, and writes a qbench-compatible trace, packed Safetensors calibration
rows, and a text stream for `ninfer-perplexity`. Tool results are deterministic local fixtures;
NInfer does not execute external tools. Calibration uses corpus shards `00`/`01`; evaluation uses
disjoint shards `02`/`03`. The sampler verifies prompt token counts against server usage and asks
`ninfer-serve` to write exact generated token IDs to a separate opt-in JSONL trace. Ordinary API
responses and request logs are unchanged.

Install its Python-only dependencies into the maintained Python 3.11 environment and run both
splits from the repository root:

```bash
uv pip install --python .venv/bin/python -r tools/exl3/requirements.txt
.venv/bin/python -m tools.exl3.sample_traces \
  --artifact models/qwen3_8_27b.ninfer \
  --corpus-manifest eval/corpora/perplexity-1m/manifest.json \
  --split calibration \
  --output-prefix profiles/exl3/traces/calibration
.venv/bin/python -m tools.exl3.sample_traces \
  --artifact models/qwen3_8_27b.ninfer \
  --corpus-manifest eval/corpora/perplexity-1m/manifest.json \
  --split evaluation \
  --rows 100 \
  --output-prefix profiles/exl3/traces/evaluation
```

The script owns the launched server lifecycle and writes its request log and console output beside
the trace files. The trace files may contain source corpus text and generated responses; keep them
local and do not commit them.

For typed rewrite-checkpoint and thinking-history behavior, the managed smoke script launches a
real server and consumes the repository fixture:

```bash
python3 tools/smoke/serve_thinking_preservation.py \
  --artifact out/qwen3_6_27b.ninfer --backend mtp
```
