# Engineering notes

What this fork changes beyond the features that have their own pages:
[speculative decoding](speculative-decoding.md), [KV streaming and context cache](kv-and-context.md),
[Qwen3.8-Flash-Next](flash-next.md), [EXL3](quantization.md), [Gemma 4](gemma4.md) and
[versus stock](vs-stock-ninfer.md).

## Conversion

- **Faster conversion.** Encoded weights are imported in 64 MiB chunks, and the converter no longer
  does per-object work that grows with model size: Flash-Next converts 2.3× faster. Flash-Next and
  Qwen3.8-27B `groupwise-int` + DFlash2 conversions are byte-identical to the previous converter's
  apart from the random artifact id; the compressed-tensors NVFP4 import behind `qwen3_8_27b_nvfp4`
  has not been re-verified. A sharded checkpoint without its `model.safetensors.index.json` now
  converts from the shard headers.
- **Stock-comparison tooling.** The serving benchmark runners drive a stock upstream `ninfer-serve`
  as well as this fork's, and `compare_serve_concurrency.py` pairs their results
  ([versus stock](vs-stock-ninfer.md)).

## Language and build

- **C++23.** `std::format`/`std::print`, `std::ranges`, `std::flat_map`, `std::move_only_function`
  and deducing `this`. CMake has no CUDA 23 dialect, so `CMakeLists.txt` keeps `CUDA_STANDARD 20`
  and passes `-std=c++23` to nvcc.
- **One chat template.** [`tools/chat_templates/qwen.jinja`](../tools/chat_templates/qwen.jinja)
  serves every Qwen model. It is derived from froggeric's v22.5 template, with patches for
  continuation and tool results. Reasoning effort defaults to `medium`, and reasoning is retained.

## Upstream PRs merged ahead of upstream

NVFP4 sparse MoE (#286–#290), GGUF as a conversion source (#282), Q5 K-split MMA (#292), Q4/Q5 group
pipelining (#311), single-pass logsumexp (#307), fused attention RMSNorm + NVFP4 quantization (#305),
two Q4 quads in flight in sparse-MoE decode (#199), tool-call parser fixes (#299, #309) and the
inference speed-of-light estimator (#304). #268 is deliberately not merged: folding the sigmoid gate
into the attention reduce breaks CUDA Graph updates when MTP and ordinary profiles share a topology
class.

## Other feature pages

- [Constrained decoding](maintainer/constrained-decoding.md), including constrained tool calls.
- [Prompt-lookup suffix drafter](maintainer/lookup-drafter.md).
- [EAGLE3](maintainer/eagle3.md).
- [W4A8 prefill](maintainer/w4a8-prefill.md).
- [Resource scheduling and context cache](maintainer/resource-scheduling-and-context-cache.md).
- [Build system](maintainer/build-system.md).
