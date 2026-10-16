#!/usr/bin/env python3
"""One artifact's decode-scaling campaign: the steady-decode suite at C = 1, 2, 4, 8.

The question is whether a decode batch of L lanes makes one weight read serve L tokens, so the
number that answers it is `metrics.steady.decode_tokens_per_second` -- decode tokens over the
steady interval, valid at any concurrency -- and never `corpus-makespan`'s
`decode_tokens_per_second`, which divides by a makespan a prefill-dominated corpus owns and whose
per-request form exists at C=1 only. This campaign runs `decode-saturation` and nothing else.

Why this is not a thin call to run_serve_concurrency.main
--------------------------------------------------------
1. **One window per point.** `--max-context` is one request's window and the attention state is
   allocated `capacity x lanes` (`nemotron_h::Program::device_bytes` ->
   `AttentionState::device_bytes(..., capacity, lanes)`), so a single invocation would give every
   point the same per-lane window and multiply the state by C along the sweep. That state is taken
   before the routed experts' residency is decided (`ExpertResidencyRequest::reserve_bytes`), so
   the higher-C points would reserve C times the memory and either squeeze the resident banks out
   or refuse to load. The campaign therefore starts one server per point with
   `window = --lane-token-budget // C`, which keeps the state -- and so the residency arithmetic --
   constant across the sweep, and records the window it used.
2. **The runner's preflight prescribes Qwen/Gemma-family facts.** `validate_server_start` there
   asserts `cuda_graph: True`, an optimized proposal head on any draft mode, and a non-empty
   `artifact.prefill_signature`. This family forces the first away in `nemotron_engine_options`
   (eager passes, context cache off, no drafter) and never assigns the third -- its `engine.cpp`
   branch resolves no prefill cost identity, where Gemma's assigns one -- so that preflight refuses
   a correct Nemotron configuration. Two assumptions in one function is the pattern worth naming:
   each is a capability note about the model rather than a defect in the harness, and neither is
   patchable from here, so this campaign keeps the runner's primitives -- `server_command`,
   `build_jobs`, `run_clients`, `load_server_events`, `analyze_point` -- and checks only what this
   family's own source says it forces, recording the rest (including the empty signature) rather
   than asserting it.
3. **The K/V cannot be quantized here** (`nemotron_h` refuses any K/V dtype other than BF16), so
   `--kv-dtype` defaults to `bf16` and any other value is refused rather than run.
4. **There is no drafter**, so only `mtp0` is a legal `--mode`.

What the prompt is, stated rather than assumed
----------------------------------------------
`decode-saturation` pins one fixture, `long_decode_aime26_15` (a thinking-enabled reasoning prompt
of ~293 tokens, output budget overridden by `--decode-tokens`), fanned out over C workers, and pads
it with the curated prose `make_bench_corpus` bakes its own corpus from (`--pad-tokens`). The serve
API takes text, so the corpus file -- whitespace-separated ids -- cannot be fed to it directly. The
campaign therefore *validates* `--corpus` (every id must be inside the artifact's vocabulary, the
rule that refuses the checked-in Qwen corpus) and records it, and it does NOT inject it: the prompt
length actually seen is the point's recorded `prompt_tokens`.

Usage
-----
    python3 tools/bench/run_serve_decode_scaling.py \\
      --artifact nemotron_h_74b_nvfp4=out/nemotron_h_74b_nvfp4.ninfer \\
      --corpus profiles/nsys/nemotron_p1/corpus.ids \\
      --concurrency 1 --concurrency 2 --concurrency 4 --concurrency 8 \\
      --lane-token-budget 262144 --pad-tokens 8192 --decode-tokens 4096 \\
      --output profiles/bench/nemotron_decode_scaling

`--dry-run` prints the per-point windows, commands and guards without starting a server.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime
import hashlib
import http.client
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.bench import run_serve_concurrency as conc  # noqa: E402
from tools.bench import run_serve_corpus as corpus  # noqa: E402
from tools.verify import check_ops_archive as gates  # noqa: E402


SCHEMA_VERSION = 1
# The pad is characters-for-tokens at 3.0 chars/token (mixed English/Chinese/code prose), which
# undershoots English prose, so a window guard has to leave room for the real prompt being longer
# than the request.
PAD_CHAR_OVERSHOOT = 1.5
# Room for the fixture's own prompt (293 tokens for the pinned one) plus the chat template.
PROMPT_FIXTURE_SLACK_TOKENS = 512
# This family's attention K/V is BF16 whatever is asked for; the campaign refuses anything else
# instead of recording a KV dtype the engine did not load.
KV_DTYPE = "bf16"
SPECULATIVE_MODE = "mtp0"
# The output-side gate. A rate cannot see an arithmetic defect: the wide-T route that briefly
# existed in this tree at 09:12 read workspace nothing had written and wrote past its buffer with
# nothing thrown, and the archive gates check packaging rather than arithmetic. The benchmark's
# instrument is a raw greedy completion, which this server cannot host (its routes are all
# chat-shaped: /v1/chat/completions, /v1/responses, /v1/messages), so the probe is a question with
# one checkable answer asked greedily with thinking off.
PROBE_PROMPT = "Reply with exactly one word: what is the capital of France?"
PROBE_EXPECTED = "paris"
PROBE_MAX_TOKENS = 64


@dataclasses.dataclass(frozen=True)
class ArtifactFacts:
    """What the conversion report says about the artifact, for the guards and the report."""

    vocab_size: int
    native_context: int | None
    moe_banks: int | None
    pool_experts: int | None


@dataclasses.dataclass(frozen=True)
class CorpusFacts:
    path: Path
    tokens: int
    max_id: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="artifact label and server model alias; the campaign admits one artifact",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="in-domain token-id corpus; validated inside the artifact's vocabulary and recorded",
    )
    parser.add_argument(
        "--concurrency",
        action="append",
        type=int,
        metavar="N",
        help="concurrency point; repeat (default: 1, 2, 4, 8)",
    )
    parser.add_argument(
        "--lane-token-budget",
        type=int,
        default=262144,
        help="total token window across all lanes: each point gets window = budget // C so the "
             "attention state and the expert-residency arithmetic stay constant across the sweep "
             "(default: 262144)",
    )
    parser.add_argument(
        "--pad-tokens",
        type=int,
        default=8192,
        help="decode-saturation padding: prefix the pinned fixture with this many tokens of curated "
             "prose, which sets the context each point decodes at (default: 8192, 0 for none)",
    )
    parser.add_argument(
        "--decode-tokens",
        type=int,
        default=4096,
        help="per-request output budget; every point must reach it or the point is not steady "
             "(default: 4096)",
    )
    parser.add_argument(
        "--sampling",
        choices=corpus.SAMPLING_MODES,
        default="stochastic",
        help="sampling profile, as the published methodology pins it (default: stochastic)",
    )
    parser.add_argument(
        "--kv-dtype",
        choices=tuple(corpus.KV_CACHE_NAMES),
        default=KV_DTYPE,
        help="K/V storage; this family refuses anything but bf16 (default: bf16)",
    )
    parser.add_argument(
        "--mode",
        choices=tuple(corpus.SPECULATIVE_MODES),
        default=SPECULATIVE_MODE,
        help="speculative mode; this family has no drafter, so only mtp0 is legal "
             "(default: mtp0)",
    )
    parser.add_argument(
        "--serve",
        type=Path,
        default=REPO_ROOT / "build/apps/ninfer-serve",
        help="ninfer-serve executable",
    )
    parser.add_argument(
        "--expert-cache",
        default=None,
        help="pass --expert-cache through to the server (auto|MiB); omitted leaves the family's "
             "own automatic plan, which is a function of free VRAM at load",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="override the vocabulary size the corpus is checked against, when the artifact's "
             "conversion report is not beside it",
    )
    parser.add_argument("--output", type=Path, required=True, help="campaign output directory")
    parser.add_argument(
        "--no-probe",
        action="store_true",
        help="skip the output probe (one greedy thinking-off request on its own server before the "
             "points, asserting the model path answers with the expected token)",
    )
    parser.add_argument(
        "--probe-expect",
        default=PROBE_EXPECTED,
        help=f"substring the probe reply must contain, case-insensitive "
             f"(default: {PROBE_EXPECTED!r})",
    )
    parser.add_argument(
        "--probe-tokens",
        type=int,
        default=PROBE_MAX_TOKENS,
        help=f"the probe's output budget (default: {PROBE_MAX_TOKENS})",
    )
    parser.add_argument(
        "--tree-note",
        default="",
        help="sentence recorded verbatim in the provenance block, e.g. what tree event the linked "
             "binary predates, so a reader can date the rates against the code that produced them",
    )
    parser.add_argument("--port", type=int, default=8080, help="loopback serving port")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument(
        "--dry-run", action="store_true", help="print per-point windows and commands only"
    )
    return parser.parse_args(argv)


def concurrency_points(args: argparse.Namespace) -> list[int]:
    points = sorted(set(args.concurrency or (1, 2, 4, 8)))
    for concurrency in points:
        if concurrency < 1 or concurrency > 8:
            raise corpus.CampaignError("--concurrency must be in [1, 8]")
    return points


def window_for(budget: int, concurrency: int) -> int:
    """The per-lane window a point gets: the lane budget divided by the lanes.

    The attention state is `capacity x lanes`, so holding `capacity x lanes` constant is what keeps
    one point comparable to another -- the resident experts' plan is computed from whatever free
    memory is left after that state plus one pass's workspace and the runtime's reservation.
    """
    if budget <= 0:
        raise corpus.CampaignError("--lane-token-budget must be positive")
    window = budget // concurrency
    if window <= 0:
        raise corpus.CampaignError(
            f"--lane-token-budget {budget} leaves concurrency {concurrency} an empty window"
        )
    return window


def read_artifact_facts(artifact: Path, vocab_override: int | None) -> ArtifactFacts:
    """Vocabulary and MoE shape from the conversion report beside the artifact, or the override."""
    report_path = Path(str(artifact) + ".conversion.json")
    config: dict[str, Any] | None = None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        candidate = report.get("components", {}).get("text", {}).get("config")
        if isinstance(candidate, dict):
            config = candidate
    except (OSError, json.JSONDecodeError):
        config = None
    if config is None and vocab_override is None:
        raise corpus.CampaignError(
            f"cannot read {report_path} for the vocabulary size; pass --vocab-size"
        )
    blocks = config.get("block_configs") if config else None
    moe_banks: int | None = None
    pool_experts: int | None = None
    if isinstance(blocks, list):
        moe = [block for block in blocks if isinstance(block, dict)
               and block.get("block_type") == "moe"]
        moe_banks = len(moe)
        pool_experts = sum(int(block.get("n_routed_experts", 0)) for block in moe)
    vocab = vocab_override if vocab_override is not None else int(config["vocab_size"])
    native_context = None
    if config and isinstance(config.get("max_position_embeddings"), int):
        native_context = int(config["max_position_embeddings"])
    return ArtifactFacts(
        vocab_size=vocab,
        native_context=native_context,
        moe_banks=moe_banks,
        pool_experts=pool_experts,
    )


def read_corpus(path: Path, vocab_size: int) -> CorpusFacts:
    """Read the id corpus and refuse anything the artifact's vocabulary cannot address.

    This is the check the checked-in Qwen bench corpus fails on this artifact (its ids reach 182531
    against a 131072-token vocabulary); the corpus is recorded for provenance and is not injected
    into the prompt, which the serve API takes as text.
    """
    try:
        raw = path.expanduser().read_text(encoding="utf-8")
    except OSError as exc:
        raise corpus.CampaignError(f"cannot read --corpus {path}: {exc}") from exc
    fields = raw.split()
    if not fields:
        raise corpus.CampaignError(f"--corpus {path} is empty")
    ids: list[int] = []
    for index, field in enumerate(fields):
        try:
            value = int(field)
        except ValueError:
            raise corpus.CampaignError(
                f"--corpus {path}: token {index} is not an integer: {field!r}"
            ) from None
        if value < 0:
            raise corpus.CampaignError(f"--corpus {path}: token {index} is negative: {value}")
        ids.append(value)
    largest = max(ids)
    if largest >= vocab_size:
        raise corpus.CampaignError(
            f"--corpus {path} holds id {largest}, outside the artifact's {vocab_size}-token "
            f"vocabulary ({sum(1 for value in ids if value >= vocab_size)} of {len(ids)} ids are "
            "out of domain); use a corpus this model's tokenizer produced"
        )
    return CorpusFacts(path=path.expanduser().resolve(), tokens=len(ids), max_id=largest)


def guard_window(
    window: int,
    concurrency: int,
    pad_tokens: int,
    decode_tokens: int,
    budget: int,
    facts: ArtifactFacts,
) -> None:
    """Refuse a point whose prompt and output cannot fit its window, before starting a server.

    The engine would answer a too-long prompt with a recorded context-length skip, which for this
    suite is a point that measured nothing; the arithmetic is the caller's to fix.
    """
    needed = (
        int(math.ceil(pad_tokens * PAD_CHAR_OVERSHOOT))
        + decode_tokens
        + PROMPT_FIXTURE_SLACK_TOKENS
    )
    if needed > window:
        raise corpus.CampaignError(
            f"C={concurrency} gets window {window} tokens (--lane-token-budget {budget} // "
            f"{concurrency}), but --pad-tokens {pad_tokens} (with the 3.0 chars/token padding "
            f"ratio, allow {(PAD_CHAR_OVERSHOOT):.1f}x) plus --decode-tokens {decode_tokens} plus "
            f"the fixture needs about {needed}; raise --lane-token-budget to at least "
            f"{needed * concurrency}, or lower --pad-tokens/--decode-tokens"
        )
    if facts.native_context is not None and window > facts.native_context:
        print(
            f"warning: C={concurrency} window {window} exceeds the model's {facts.native_context} "
            "token positions; the K/V above that is allocated and never addressed",
            file=sys.stderr,
        )


def card_state(device: int) -> str:
    """The card's own memory reading just before a load; the residency plan is derived from it."""
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--id",
                str(device),
                "--query-gpu=memory.used,memory.total,utilization.gpu,clocks.sm",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable ({exc})"
    return completed.stdout.strip()


def validate_server_start(
    event: dict[str, Any], point: conc.Point, args: argparse.Namespace, window: int
) -> tuple[str, str]:
    """Check the values this family's own source forces, and record the rest.

    Deliberately not the runner's `validate_server_start`: that one prescribes `cuda_graph: True`
    and an optimized proposal head, both of which `nemotron_engine_options` overrides -- the first
    by switching CUDA graphs off because the Program runs eager passes, the second because the
    family has no drafter at all. Asserting them here would refuse every correct configuration.
    """
    corpus.require_server_log_identity(event, "server_start")
    engine = event.get("engine", {})
    context_cache = engine.get("context_cache", {})
    required = {
        "device": args.device,
        "max_context": window,
        "kv_capacity_mode": "explicit",
        "kv_capacity": window,
        "max_concurrency": point.concurrency,
        "max_pending_requests": 1,
        "pending_timeout_ms": conc.PENDING_TIMEOUT_MS,
        "kv_cache": corpus.KV_CACHE_NAMES[point.kv_dtype],
        # Forced by nemotron_engine_options: eager passes, no context cache, no drafter.
        "cuda_graph": False,
        "prefix_reuse": False,
        "speculative_backend": "none",
    }
    actual = {name: engine.get(name) for name in required}
    if actual != required:
        raise corpus.CampaignError(
            f"server_start Engine configuration does not match this family's forced values: "
            f"{actual!r} != {required!r}"
        )
    if context_cache.get("enabled") is not False:
        raise corpus.CampaignError(
            "server_start reports a context cache, which this family disables"
        )
    if event.get("sampling_defaults", {}).get("greedy") != (point.sampling_mode == "greedy"):
        raise corpus.CampaignError("server_start sampling mode does not match the point")
    if Path(event.get("artifact", {}).get("path", "")).resolve() != point.artifact.resolve():
        raise corpus.CampaignError("loaded artifact path does not match the campaign")
    # Unlike the Qwen and Gemma paths, this family's load summary never assigns a prefill
    # signature: it resolves no prefill cost identity, so the field is empty rather than wrong.
    # The shared runner requires a non-empty one because its families provide one; requiring it
    # here would refuse every correct point, so the value is recorded as what it is.
    prefill_signature = event.get("artifact", {}).get("prefill_signature")
    if not isinstance(prefill_signature, str):
        prefill_signature = ""
    if event.get("server", {}).get("public_model_id") != point.model_id:
        raise corpus.CampaignError("server public model id does not match the campaign target")
    server_instance_id = event.get("server_instance_id")
    if not isinstance(server_instance_id, str) or not server_instance_id:
        raise corpus.CampaignError("server_start has no server_instance_id")
    return server_instance_id, prefill_signature


def ttft_seconds(events: Sequence[dict[str, Any]]) -> list[float]:
    """Per-request time to first token, as the server logged it.

    A latency exposure, not worker cost: reported as a distribution and never summed.
    """
    values: list[float] = []
    for event in events:
        if event.get("event") != "request_done":
            continue
        timings = event.get("timings_seconds")
        if not isinstance(timings, dict) or "ttft" not in timings:
            raise corpus.CampaignError("request_done has no timings_seconds.ttft")
        values.append(float(timings["ttft"]))
    if not values:
        raise corpus.CampaignError("the point recorded no completed request")
    return values


def point_row(
    point: conc.Point,
    window: int,
    report: dict[str, Any],
    events: Sequence[dict[str, Any]],
    facts: ArtifactFacts,
    card: str,
) -> dict[str, Any]:
    engine = report["engine"]
    memory = report["memory"]
    environment = report["environment"]
    steady = report["metrics"]["steady"]
    aggregate = float(steady["decode_tokens_per_second"])
    ttft = ttft_seconds(events)
    prompt_tokens = {int(request["prompt_tokens"]) for request in report["requests"]}
    return {
        "concurrency": point.concurrency,
        "window_tokens": window,
        "prompt_tokens": sorted(prompt_tokens)[0] if len(prompt_tokens) == 1 else None,
        "prompt_tokens_per_request": sorted(prompt_tokens),
        "decode_tokens": int(report["totals"]["decode_tokens"]),
        "steady_decode_tokens": int(steady["committed_decode_tokens"]),
        "steady_seconds": float(steady["seconds"]),
        "steady_intervals": int(steady["intervals"]),
        "average_decode_batch": float(steady["average_decode_batch"]),
        "aggregate_decode_tokens_per_second": aggregate,
        "per_request_decode_tokens_per_second": aggregate / point.concurrency,
        "milliseconds_per_token_per_lane": 1000.0 * point.concurrency / aggregate,
        "ttft_seconds_mean": sum(ttft) / len(ttft),
        "ttft_seconds_max": max(ttft),
        "finish_reasons": sorted({str(request["finish_reason"]) for request in report["requests"]}),
        "card_before_load": card,
        "free_vram_at_load_bytes": int(memory["available_after_weights_bytes"]),
        "free_vram_after_startup_bytes": int(memory["available_after_startup_bytes"]),
        "runtime_reservation_bytes": int(memory["runtime_reservation_bytes"]),
        "weights_bytes": int(memory["weights"]["used_bytes"]),
        "state_bytes": int(memory["sequence"]["used_bytes"]),
        "resident_experts": int(engine["expert_cache_slots"]),
        "resident_expert_bytes": int(engine["expert_cache_bytes"]),
        "pool_experts": facts.pool_experts,
        "pool_banks": facts.moe_banks,
        "prefill_chunk": int(engine["prefill_chunk"]),
        "max_concurrency": int(engine["max_concurrency"]),
        "cuda_graph": bool(engine["cuda_graph"]),
        "prefix_reuse": bool(engine["prefix_reuse"]),
        "proposal_head": engine["proposal_head"],
        "speculative_backend": engine["speculative_backend"],
        "total_device_memory_bytes": int(environment["total_device_memory_bytes"]),
        "gpu_name": environment["gpu_name"],
        "speedup_vs_c1": None,
    }


def build_provenance(serve: Path) -> dict[str, Any]:
    """What the binary under measurement was linked from, so the claim is checkable afterwards.

    A campaign run says "these servers ran on a tree whose ops objects had finished"; that is not a
    claim a reader can check once the binaries have been relinked, so the identity of the executable
    and the archive's own completeness numbers are recorded with the measurement. This is a snapshot
    taken as the campaign starts and never a lock on the build directory. The archive numbers come
    from `tools/verify/check_ops_archive.py`, so the campaign and the gate agree by construction.
    """
    provenance: dict[str, Any] = {
        "note": "snapshot at campaign start; not a lock on the build directory",
        "serve_path": str(serve),
    }
    try:
        stat = serve.stat()
        provenance["serve_bytes"] = stat.st_size
        provenance["serve_mtime"] = datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(
            timespec="seconds"
        )
        digest = hashlib.sha256()
        with serve.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        provenance["serve_sha256"] = digest.hexdigest()
    except OSError as exc:
        provenance["serve_error"] = str(exc)
    try:
        provenance["ops_archive"] = gates.archive_facts(gates.DEFAULT_BUILD_DIR / gates.DEFAULT_ARCHIVE)
    except gates.GateError as exc:
        provenance["ops_archive"] = {"error": str(exc)}
    return provenance


def probe_text(response: dict[str, Any]) -> str:
    """The generated text of one Chat Completions response, content and reasoning together."""
    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise corpus.CampaignError(f"probe response has no message: {exc}") from exc
    parts = [
        part
        for key in ("content", "reasoning_content")
        if isinstance(part := message.get(key), str)
    ]
    return "\n".join(parts)


def probe_matched(text: str, expected: str) -> bool:
    return expected.casefold() in text.casefold()


def run_probe(
    serve: Path,
    point: conc.Point,
    output_dir: Path,
    args: argparse.Namespace,
    window: int,
) -> dict[str, Any]:
    """One greedy, thinking-off request against a server loaded exactly as the points are.

    This is the output side of the gates: it asserts the model path produces the right token rather
    than mere throughput, which is what a rate cannot see. It runs on its own server and its own
    request log so the measured points' event accounting stays exactly what the harness expects
    (one `request_done` per measured request, and no stray steady interval).
    """
    probe_dir = output_dir / "probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    server_log = probe_dir / "server.jsonl"
    engine_args = SimpleNamespace(
        port=args.port,
        max_context=window,
        kv_capacity=str(window),
        prefill_chunk=None,
        device=args.device,
        kv_dtype=point.kv_dtype,
    )
    command = conc.server_command(serve, point, server_log, engine_args)
    print(f"probe: C=1 window={window} greedy, thinking off, {args.probe_tokens} tokens", flush=True)
    with corpus.RunningServer(command, "127.0.0.1", args.port, server_log) as server:
        server_start = server.wait_until_ready()
        validate_server_start(server_start, point, args, window)
        connection = http.client.HTTPConnection(
            "127.0.0.1", args.port, timeout=corpus.REQUEST_TIMEOUT_SECONDS
        )
        try:
            response = corpus.post_json(
                connection,
                {
                    "model": point.model_id,
                    "messages": [{"role": "user", "content": PROBE_PROMPT}],
                    "max_completion_tokens": args.probe_tokens,
                    "seed": 0,
                    "stream": False,
                    "enable_thinking": False,
                },
            )
        finally:
            connection.close()
    text = probe_text(response)
    matched = probe_matched(text, args.probe_expect)
    record = {
        "prompt": PROBE_PROMPT,
        "expected": args.probe_expect,
        "max_completion_tokens": args.probe_tokens,
        "thinking": False,
        "sampling": "greedy",
        "window_tokens": window,
        "text": text,
        "matched": matched,
        "usage": response.get("usage", {}),
        "server_log": str(server_log),
        "note": (
            "one request on its own server, before the measured points, so it neither warms their "
            "engines nor contributes events to their logs"
        ),
    }
    (output_dir / "probe.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"probe: text={text.strip()[:120]!r} matched={matched}", flush=True)
    if not matched:
        raise corpus.CampaignError(
            f"output probe failed: reply {text.strip()[:200]!r} does not contain "
            f"{args.probe_expect!r}; the model path is not producing the expected token, so a rate "
            "measured from it would be meaningless (see "
            f"{output_dir / 'probe.json'})"
        )
    return record


def run_point(
    serve: Path,
    point: conc.Point,
    fixtures: dict[str, corpus.Fixture],
    output_dir: Path,
    args: argparse.Namespace,
    window: int,
    facts: ArtifactFacts,
) -> tuple[dict[str, Any], dict[str, Any]]:
    jobs = conc.build_jobs(point, fixtures, args.decode_tokens, args.pad_tokens)
    server_log = output_dir / "server" / f"{point.key}.jsonl"
    server_log.parent.mkdir(parents=True, exist_ok=True)
    # server_command() reads the per-point engine values off this namespace: the window is the only
    # thing that changes between points, and it is what makes the state constant per lane.
    engine_args = SimpleNamespace(
        port=args.port,
        max_context=window,
        kv_capacity=str(window),
        prefill_chunk=None,
        device=args.device,
        kv_dtype=point.kv_dtype,
    )
    command = conc.server_command(serve, point, server_log, engine_args)
    if args.expert_cache is not None:
        command.extend(["--expert-cache", str(args.expert_cache)])

    card = card_state(args.device)
    print(
        f"start {point.target}/{point.key} C={point.concurrency} window={window} "
        f"requests={len(jobs)} card=[{card}]",
        flush=True,
    )
    with corpus.RunningServer(command, "127.0.0.1", args.port, server_log) as server:
        server_start = server.wait_until_ready()
        instance_id, prefill_signature = validate_server_start(server_start, point, args, window)
        engine = server_start.get("engine", {})
        memory = server_start.get("memory", {})
        print(
            f"loaded C={point.concurrency} free_at_load="
            f"{int(memory.get('available_after_weights_bytes', 0)) >> 20}MiB "
            f"resident_experts={engine.get('expert_cache_slots')} "
            f"({int(engine.get('expert_cache_bytes', 0)) >> 20}MiB) "
            f"weights={int(memory.get('weights', {}).get('used_bytes', 0)) >> 20}MiB "
            f"state={int(memory.get('sequence', {}).get('used_bytes', 0)) >> 20}MiB "
            f"prefill_chunk={engine.get('prefill_chunk')}",
            flush=True,
        )
        results, skips, campaign_start, campaign_end = conc.run_clients(
            point, jobs, args.port
        )
    if not prefill_signature:
        print(
            f"note: C={point.concurrency} server_start carries no prefill signature (this family "
            "resolves no prefill cost identity); recorded as the empty string",
            flush=True,
        )
    events = conc.load_server_events(server_log, instance_id)
    report = conc.analyze_point(
        point,
        command,
        server_log,
        server_start,
        prefill_signature,
        events,
        results,
        skips,
        campaign_start,
        campaign_end,
    )
    report["campaign"] = {
        "artifact_type": "ninfer_serve_decode_scaling_point",
        "schema_version": SCHEMA_VERSION,
        "lane_token_budget": args.lane_token_budget,
        "window_rule": "window = --lane-token-budget // C",
        "pad_tokens": args.pad_tokens,
        "decode_tokens": args.decode_tokens,
        "corpus": None,  # filled by the caller, which owns the corpus facts
    }
    row = point_row(point, window, report, events, facts, card)
    row["key"] = point.key
    return report, row


def tables(rows: Sequence[dict[str, Any]], corpus_facts: CorpusFacts) -> str:
    """The campaign's two tables as Markdown: rates first, then the memory the plan was derived from."""
    speedup_of = {row["concurrency"]: row["speedup_vs_c1"] for row in rows}
    sections = [
        conc.markdown_table(
            (
                "C",
                "window",
                "prompt tok",
                "decode tok",
                "steady s",
                "per-request tok/s",
                "aggregate tok/s",
                "ms/token/lane",
                "TTFT s mean/max",
                "avg batch",
                "speedup vs C=1",
            ),
            [
                (
                    str(row["concurrency"]),
                    str(row["window_tokens"]),
                    str(row["prompt_tokens"]),
                    str(row["decode_tokens"]),
                    conc.format_number(row["steady_seconds"], 1),
                    conc.format_number(row["per_request_decode_tokens_per_second"], 1),
                    conc.format_number(row["aggregate_decode_tokens_per_second"], 1),
                    conc.format_number(row["milliseconds_per_token_per_lane"], 2),
                    f"{row['ttft_seconds_mean']:.2f}/{row['ttft_seconds_max']:.2f}",
                    conc.format_number(row["average_decode_batch"], 2),
                    conc.format_number(speedup_of.get(row["concurrency"])),
                )
                for row in rows
            ],
        ),
        conc.markdown_table(
            (
                "C",
                "window",
                "free at load GiB",
                "free after startup GiB",
                "weights GiB",
                "state GiB",
                "resident experts",
                "pool experts",
                "resident bytes GiB",
                "total VRAM GiB",
                "card before load",
            ),
            [
                (
                    str(row["concurrency"]),
                    str(row["window_tokens"]),
                    conc.format_number(row["free_vram_at_load_bytes"] / 2**30),
                    conc.format_number(row["free_vram_after_startup_bytes"] / 2**30),
                    conc.format_number(row["weights_bytes"] / 2**30),
                    conc.format_number(row["state_bytes"] / 2**30),
                    str(row["resident_experts"]),
                    str(row["pool_experts"]),
                    conc.format_number(row["resident_expert_bytes"] / 2**30),
                    conc.format_number(row["total_device_memory_bytes"] / 2**30),
                    row["card_before_load"],
                )
                for row in rows
            ],
        ),
        "workload: decode-saturation (one pinned reasoning fixture per lane, padded with curated "
        f"prose), corpus {corpus_facts.path} ({corpus_facts.tokens} ids, max "
        f"{corpus_facts.max_id}); the corpus is validated in-domain and recorded, not injected. "
        "Per-request and aggregate rates are separate columns: the aggregate is C rows of one "
        "decode batch, and the per-request column is that aggregate divided by C. Every point "
        "started a fresh server, so every point pays one cold first decode pass once — the same "
        "term the benchmark's `--no-cuda-graph` removes when it skips its prime, worth ~1.49x on "
        "that single pass and well under 0.1% of a point's steady window. A campaign that reused "
        "one server across points would have folded that term into C=1 only. Each point also "
        "decodes from its own padded `prompt_tokens`, which is the context column a reader needs "
        "before comparing any rate here with a single-process `ninfer_bench` figure.",
    ]
    return "\n\n".join(sections)


def write_outputs(
    rows: Sequence[dict[str, Any]],
    output_dir: Path,
    environment: dict[str, Any],
) -> None:
    with (output_dir / "scaling.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "artifact_type": "ninfer_serve_decode_scaling_campaign",
                "schema_version": SCHEMA_VERSION,
                "environment": environment,
                "rows": list(rows),
            },
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")
    fields = [name for name in rows[0] if name not in ("prompt_tokens_per_request", "card_before_load")]
    with (output_dir / "scaling.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = __import__("csv").DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fields})


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.kv_dtype != KV_DTYPE:
        raise corpus.CampaignError(
            f"--kv-dtype {args.kv_dtype} is refused: this family's attention K/V is BF16 because "
            "that is the dtype its Op declares, and no quantized K/V codec is wired for it"
        )
    backend, draft_tokens = corpus.SPECULATIVE_MODES[args.mode]
    if backend != "none" or draft_tokens:
        raise corpus.CampaignError(
            f"--mode {args.mode} asks for speculation, which this family refuses in "
            "nemotron_engine_options (it has no drafter); use mtp0"
        )
    if args.pad_tokens < 0 or args.decode_tokens <= 0:
        raise corpus.CampaignError("--pad-tokens must be nonnegative and --decode-tokens positive")
    if args.port < 1 or args.port > 65535 or args.device < 0:
        raise corpus.CampaignError("--port must be in [1, 65535] and --device nonnegative")

    [artifact] = corpus.parse_artifacts(args.artifact)
    target, artifact_path = artifact
    facts = read_artifact_facts(artifact_path, args.vocab_size)
    corpus_facts = read_corpus(args.corpus, facts.vocab_size)
    points = concurrency_points(args)
    windows = {concurrency: window_for(args.lane_token_budget, concurrency) for concurrency in points}
    smallest_window = min(windows.values())
    for concurrency in points:
        guard_window(
            windows[concurrency],
            concurrency,
            args.pad_tokens,
            args.decode_tokens,
            args.lane_token_budget,
            facts,
        )

    fixtures = corpus.load_fixtures()
    serve = args.serve.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    campaign_points = [
        conc.Point(
            target=target,
            model_id=target,
            artifact=artifact_path,
            speculative_mode=args.mode,
            speculative_backend=backend,
            draft_tokens=draft_tokens,
            sampling_mode=args.sampling,
            kv_dtype=args.kv_dtype,
            suite="decode-saturation",
            concurrency=concurrency,
        )
        for concurrency in points
    ]

    if args.dry_run:
        print(
            f"artifact {artifact_path} ({facts.vocab_size} tokens"
            + (f", native context {facts.native_context}" if facts.native_context else "")
            + f", {facts.moe_banks} expert banks / {facts.pool_experts} pool experts)"
        )
        print(
            f"corpus {corpus_facts.path} ({corpus_facts.tokens} ids, max {corpus_facts.max_id})"
        )
        for point in campaign_points:
            window = windows[point.concurrency]
            engine_args = SimpleNamespace(
                port=args.port,
                max_context=window,
                kv_capacity=str(window),
                prefill_chunk=None,
                device=args.device,
                kv_dtype=point.kv_dtype,
            )
            jobs = conc.build_jobs(point, fixtures, args.decode_tokens, args.pad_tokens)
            command = conc.server_command(
                serve, point, output_dir / "server" / f"{point.key}.jsonl", engine_args
            )
            if args.expert_cache is not None:
                command.extend(["--expert-cache", str(args.expert_cache)])
            print(f"# {point.key}: C={point.concurrency} window={window} requests={len(jobs)}")
            print("  " + " ".join(command))
        if not args.no_probe:
            print(
                f"# output probe: own C=1 server, window={windows[points[0]]}, greedy, thinking "
                f"off, {args.probe_tokens} tokens, must contain {args.probe_expect!r}"
            )
        return 0

    if not serve.is_file():
        raise corpus.CampaignError(f"ninfer-serve executable not found: {serve}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "points").mkdir(parents=True, exist_ok=True)
    (output_dir / "server").mkdir(parents=True, exist_ok=True)

    environment = {
        "artifact": str(artifact_path),
        "artifact_bytes": artifact_path.stat().st_size,
        "artifact_vocab_size": facts.vocab_size,
        "artifact_native_context": facts.native_context,
        "artifact_expert_banks": facts.moe_banks,
        "artifact_pool_experts": facts.pool_experts,
        "corpus": str(corpus_facts.path),
        "corpus_tokens": corpus_facts.tokens,
        "corpus_max_id": corpus_facts.max_id,
        "serve": str(serve),
        "device": args.device,
        "port": args.port,
        "mode": args.mode,
        "sampling": args.sampling,
        "kv_dtype": args.kv_dtype,
        "suite": "decode-saturation",
        "lane_token_budget": args.lane_token_budget,
        "window_rule": "window = --lane-token-budget // C",
        "windows": {str(concurrency): windows[concurrency] for concurrency in points},
        "smallest_window_tokens": smallest_window,
        "pad_tokens": args.pad_tokens,
        "decode_tokens": args.decode_tokens,
        "expert_cache": args.expert_cache,
        "pad_prompt_note": (
            "the decode-saturation prompt is the pinned long_decode_aime26_15 fixture padded with "
            "make_bench_corpus prose; the corpus is validated in-domain and recorded, not injected "
            "(the serve API takes text and the corpus is token ids)"
        ),
        # Which costs this campaign's shape pays, because a reader will compare a point's rate
        # against a single-process `ninfer_bench` figure and needs to know what differs.
        "prefill_signature_note": (
            "every point's server_start reports an empty prefill_signature: nemotron_h's load "
            "summary never assigns one because the family resolves no prefill cost identity "
            "(the Qwen and Gemma paths do), so no measurement-keying signature exists for this "
            "artifact; the per-point reports carry the empty string rather than a value"
        ),
        "fresh_process_terms": [
            "every point starts a fresh ninfer-serve: artifact load, the expert-residency plan "
            "(derived from free VRAM at load) and one cold first decode pass are paid once per "
            "point and never per request; no term is shared between points, so C=1 and C=8 come "
            "from different processes and different per-lane windows by construction",
            "the cold first pass is the term a benchmark removes with its untimed prime: "
            "`ninfer_bench` defaults to priming (its measured rep is warm) and `--no-cuda-graph` "
            "skips the prime (its measured rep is the first pass, ~1.49x slower), so a bench "
            "figure taken with the default corresponds to this campaign's steady rate, not to "
            "its first interval",
            "context differs too: each point decodes from the prompt_tokens it records (padded "
            "here), while a bench `tg` case decodes from a one-token seed prefill, so the two are "
            "not the same measurement even at C=1",
        ],
    }

    rows: list[dict[str, Any]] = []
    provenance = build_provenance(serve)
    if args.tree_note:
        provenance["tree_note"] = args.tree_note
    environment["build_provenance"] = provenance
    archive = provenance.get("ops_archive", {})
    print(
        "build: serve "
        f"sha256={str(provenance.get('serve_sha256', 'unavailable'))[:16]} "
        f"bytes={provenance.get('serve_bytes')} mtime={provenance.get('serve_mtime')} | "
        f"ops archive members={archive.get('members')} bytes={archive.get('bytes')} "
        f"mtime={archive.get('mtime')} fatbinwraps referenced/defined/missing="
        f"{archive.get('fatbinwraps_referenced')}/{archive.get('fatbinwraps_defined')}/"
        f"{archive.get('fatbinwraps_missing')}",
        flush=True,
    )
    if args.no_probe:
        environment["output_probe"] = {"skipped": "--no-probe"}
    else:
        probe_point = dataclasses.replace(campaign_points[0], sampling_mode="greedy")
        environment["output_probe"] = run_probe(
            serve, probe_point, output_dir, args, windows[points[0]]
        )
    for point in campaign_points:
        report, row = run_point(
            serve, point, fixtures, output_dir, args, windows[point.concurrency], facts
        )
        report["campaign"]["corpus"] = {
            "path": str(corpus_facts.path),
            "tokens": corpus_facts.tokens,
            "max_id": corpus_facts.max_id,
            "injected": False,
        }
        (output_dir / "points" / f"{point.key}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        rows.append(row)
        print(
            f"done C={point.concurrency} window={windows[point.concurrency]} "
            f"per-request={row['per_request_decode_tokens_per_second']:.1f}tok/s "
            f"aggregate={row['aggregate_decode_tokens_per_second']:.1f}tok/s "
            f"ttft={row['ttft_seconds_mean']:.2f}s",
            flush=True,
        )

    baseline = next((row for row in rows if row["concurrency"] == 1), None)
    for row in rows:
        if baseline is not None:
            row["speedup_vs_c1"] = (
                row["aggregate_decode_tokens_per_second"]
                / baseline["aggregate_decode_tokens_per_second"]
            )
    write_outputs(rows, output_dir, environment)
    markdown = "# Decode scaling: " + target + "\n\n" + tables(rows, corpus_facts) + "\n"
    (output_dir / "scaling.md").write_text(markdown, encoding="utf-8")
    print()
    print(tables(rows, corpus_facts))
    print(f"\nsummary: {output_dir / 'scaling.json'}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except corpus.CampaignError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130) from None
