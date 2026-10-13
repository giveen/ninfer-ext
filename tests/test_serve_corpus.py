from __future__ import annotations

import contextlib
import http.server
import threading
from pathlib import Path
from argparse import Namespace
import json

import pytest

from tools.bench.run_serve_corpus import (
    SERVER_LOG_SCHEMA_VERSIONS,
    Fixture,
    RunSpec,
    build_result_record,
    parse_artifacts,
    require_server_log_identity,
)
from tools.bench.run_serve_concurrency import build_points
from tools.bench import run_serve_corpus as corpus
from tools.bench import run_serve_concurrency as concurrency


def test_result_record_parses_request_host_exposure() -> None:
    fixture = Fixture(
        name="fixture",
        messages=[],
        thinking=True,
        max_new=8,
        suite="test",
    )
    spec = RunSpec(
        target="qwen3_6_27b",
        model_id="qwen3.6-27b",
        artifact=Path("/tmp/model.ninfer"),
        speculative_mode="mtp3",
        speculative_backend="mtp",
        draft_tokens=3,
        sampling_mode="greedy",
        kv_dtype="fp8",
        fixture=fixture,
        seed=7,
    )
    payload = {"model": spec.model_id}
    response = {"usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    event = {
        "artifact_type": "ninfer_serve_request_log",
        "schema_version": SERVER_LOG_SCHEMA_VERSIONS[-1],
        "event": "request_done",
        "request": {
            "model": spec.model_id,
            "requested_output_tokens": 8,
            "enable_thinking": True,
            "sampling": {"seed": 7},
        },
        "result": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "finish_reason": "output_limit",
        },
        "timings_seconds": {
            "prepare": 0.1,
            "vision": 0.0,
            "prefill": 0.2,
            "decode": 0.4,
            "total": 0.7,
        },
        "speculative": {
            "backend": "mtp",
            "rounds": 2,
            "drafted_tokens": 6,
            "accepted_tokens": 3,
            "fallback_steps": 0,
        },
        "engine_timing": {
            "queue_wait_seconds": 0.001,
            "host_exposed_seconds": {
                "engine_boundary": 0.001,
                "program_submit": 0.002,
                "program_post": 0.003,
                "engine_commit_output": 0.004,
                "engine_maintenance": 0.005,
                "total": 0.015,
            },
            "device_wait_exposed_seconds": 0.3,
            "decode": {
                "host_exposed_seconds": 0.01,
                "device_wait_exposed_seconds": 0.2,
                "rounds": 2,
            },
        },
    }

    record = build_result_record(
        spec, "measured-prefill-bindings", payload, response, event
    )
    assert record["schema_version"] == 8
    assert record["kv_dtype"] == "fp8"
    assert record["metrics"]["engine_host_exposed_ms"] == pytest.approx(15.0)
    assert record["metrics"]["decode_host_us_per_round"] == pytest.approx(5000.0)
    assert record["metrics"]["decode_device_wait_us_per_round"] == pytest.approx(
        100000.0
    )


def test_arbitrary_artifact_labels_reach_the_requested_backend(tmp_path: Path) -> None:
    artifact = tmp_path / "custom.ninfer"
    artifact.touch()
    artifacts = parse_artifacts([f"org/custom={artifact}", f"org%2Fcustom={artifact}"])
    points = build_points(
        artifacts,
        Namespace(
            mode=["dflash7", "dflash2_7", "eagle3_adaptive"],
            suite=["decode-saturation"],
            concurrency=[1],
            sampling="greedy",
            kv_dtype="fp8",
            block_fixtures="auto",
        ),
    )
    assert [(point.target, point.speculative_backend) for point in points] == [
        ("org/custom", "dflash"),
        ("org/custom", "dflash2"),
        ("org/custom", "eagle3"),
        ("org%2Fcustom", "dflash"),
        ("org%2Fcustom", "dflash2"),
        ("org%2Fcustom", "eagle3"),
    ]
    assert all(
        point.artifact == artifact and point.model_id == point.target
        for point in points
    )
    assert len({point.key for point in points}) == len(points)
    for point in points:
        (tmp_path / f"{point.key}.json").write_text("{}")


def test_block_fixtures_override_puts_both_backends_on_the_same_prompts() -> None:
    """An A/B is only comparable if both sides run the same requests.

    The per-backend default is deliberately disjoint -- the plain path gets the long-context NIAH
    fixtures, a speculative backend the long-decode and scenario prompts -- so a plain-vs-MTP
    comparison without an override measures the workload change, not the speculation.
    """
    fixtures = corpus.load_fixtures()

    def names(backend: str, override: str) -> set[str]:
        point = concurrency.Point(
            target="t", model_id="t", artifact=Path("a.ninfer"), speculative_mode="m",
            speculative_backend=backend, draft_tokens=0, sampling_mode="greedy",
            kv_dtype="bf16", suite="corpus-makespan", concurrency=1, block_fixtures=override,
        )
        return {job.fixture.name for job in concurrency.build_jobs(point, fixtures, 128, 0)}

    plain, speculative = names("none", "auto"), names("mtp", "auto")
    assert plain and speculative and not (plain & speculative)
    assert names("none", "decode") == names("mtp", "decode")
    assert names("none", "long") == set(corpus.LONG_DECODE_FIXTURES)


def test_request_logs_of_upstream_and_this_tree_are_accepted() -> None:
    header = Path(__file__).resolve().parents[1] / "src/serve/request_log.h"
    declared = int(
        header.read_text().split("kRequestLogSchemaVersion", 1)[1].split("=", 1)[1].split(";")[0]
    )
    assert SERVER_LOG_SCHEMA_VERSIONS == (21, declared)
    for version in SERVER_LOG_SCHEMA_VERSIONS:
        require_server_log_identity(
            {"artifact_type": "ninfer_serve_request_log", "schema_version": version,
             "event": "server_start"},
            "server_start",
        )
    with pytest.raises(Exception, match="unexpected serving log identity"):
        require_server_log_identity(
            {"artifact_type": "ninfer_serve_request_log", "schema_version": 20,
             "event": "server_start"},
            "server_start",
        )

@pytest.mark.parametrize(
    "kv_dtype,cache_name", [("int8", "int8-group64"), ("fp8", "fp8-e4m3-row256")]
)
@pytest.mark.parametrize("concurrent", [False, True])
def test_selected_kv_reaches_server_and_is_verified(
    tmp_path, kv_dtype, cache_name, concurrent
):
    artifact = tmp_path / "model.ninfer"
    artifact.touch()
    common = [
        "--artifact",
        f"model={artifact}",
        "--output",
        str(tmp_path),
        "--mode",
        "mtp0",
        "--kv-dtype",
        kv_dtype,
    ]
    engine = {
        "device": 0,
        "max_context": 262144,
        "kv_capacity": 262144,
        "prefill_chunk": 1024,
        "kv_cache": cache_name,
        "cuda_graph": True,
        "prefix_reuse": False,
        "speculative_backend": "none",
        "speculative_draft_window": 0,
        "proposal_head": "full",
    }
    event = {
        "artifact_type": corpus.SERVER_LOG_ARTIFACT_TYPE,
        "schema_version": SERVER_LOG_SCHEMA_VERSIONS[-1],
        "event": "server_start",
        "engine": engine,
        "sampling_defaults": {"greedy": False},
        "artifact": {"path": str(artifact), "prefill_signature": "bindings"},
        "server": {"public_model_id": "model"},
        "server_instance_id": "instance",
    }
    if concurrent:
        args = concurrency.parse_args(
            common + ["--suite", "decode-saturation", "--concurrency", "1"]
        )
        point = build_points([("model", artifact)], args)[0]
        command = concurrency.server_command(
            Path("serve"), point, tmp_path / "log", args
        )
        engine.update(
            kv_capacity_mode="explicit",
            max_concurrency=1,
            max_pending_requests=1,
            pending_timeout_ms=concurrency.PENDING_TIMEOUT_MS,
            log_stats_interval_ms=concurrency.STATS_INTERVAL_MS,
        )
        validate = lambda: concurrency.validate_server_start(event, point, args)
    else:
        args = corpus.parse_args(common)
        spec = RunSpec(
            "model",
            "model",
            artifact,
            "mtp0",
            "none",
            0,
            "stochastic",
            args.kv_dtype,
            Fixture("fixture", [], False, 8, "test"),
            7,
        )
        command = corpus.server_command(
            Path("serve"), spec, tmp_path / "log", args.port, args.device
        )
        validate = lambda: corpus.validate_server_start(event, spec, args.device)
    assert command[command.index("--kv-dtype") + 1] == kv_dtype
    assert validate() == ("instance", "bindings")
    engine["kv_cache"] = "bf16"
    with pytest.raises(corpus.CampaignError, match="configuration mismatch"):
        validate()


def test_resume_rejects_different_kv_dtype(tmp_path):
    spec = RunSpec(
        "model",
        "model",
        tmp_path / "model.ninfer",
        "mtp0",
        "none",
        0,
        "stochastic",
        "fp8",
        Fixture("fixture", [], False, 8, "test"),
        7,
    )
    record = {
        "artifact_type": corpus.RUN_ARTIFACT_TYPE,
        "schema_version": corpus.RUN_SCHEMA_VERSION,
        "target": spec.target,
        "speculative_mode": spec.speculative_mode,
        "sampling_mode": spec.sampling_mode,
        "fixture": spec.fixture.name,
        "seed": spec.seed,
        "artifact_path": str(spec.artifact),
        "kv_dtype": "int8",
    }
    path = tmp_path / "run.jsonl"
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(corpus.CampaignError, match="KV dtype differs"):
        corpus.load_existing_records(path, {spec.key: spec})


CONTEXT_LENGTH_BODY = (
    '{"error":{"code":"context_length_exceeded","message":'
    '"prepared prompt has 67412 tokens, exceeding Engine max_context 8192"}}'
)
SKIP_MESSAGE = "prepared prompt has 67412 tokens, exceeding Engine max_context 8192"


def skip_point() -> concurrency.Point:
    return concurrency.Point(
        target="model",
        model_id="model",
        artifact=Path("/tmp/model.ninfer"),
        speculative_mode="mtp0",
        speculative_backend="none",
        draft_tokens=0,
        sampling_mode="greedy",
        kv_dtype="fp8",
        suite="corpus-makespan",
        concurrency=1,
    )


def skip_fixture(name: str, index: int) -> concurrency.SkippedFixture:
    return concurrency.SkippedFixture(
        job=concurrency.Job(
            index=index,
            case_index=index,
            fixture=Fixture(name, [], False, 8, "long_niah"),
            seed=7,
            max_tokens=8,
        ),
        message=SKIP_MESSAGE,
    )


@pytest.mark.parametrize(
    "failure,expected",
    [
        (f"HTTP 400 Bad Request: {CONTEXT_LENGTH_BODY}", SKIP_MESSAGE),
        ('HTTP 400 Bad Request: {"error":{"code":"invalid_request_error","message":"bad"}}', None),
        ('HTTP 429 Too Many Requests: {"error":{"code":"server_overloaded"}}', None),
        ('HTTP 500 Internal Server Error: {"error":{"code":"context_length_exceeded"}}', None),
        ("HTTP 400 Bad Request: not json", None),
        ("HTTP request failed: timed out", None),
        ("invalid Chat Completions response: 'usage'", None),
    ],
)
def test_only_the_context_length_refusal_is_a_skip(failure: str, expected: str | None) -> None:
    assert concurrency.context_length_exceeded_message(corpus.CampaignError(failure)) == expected


def measured_result() -> concurrency.ClientResult:
    return concurrency.ClientResult(
        job=concurrency.Job(
            index=0, case_index=0, fixture=Fixture("long_niah_8k", [], False, 8, "long_niah"),
            seed=7, max_tokens=8,
        ),
        started_at=1.0, finished_at=2.0, prompt_tokens=7000, completion_tokens=1,
        finish_reason="stop",
    )


def measured_events() -> list[dict]:
    return [
        {
            "event": "request_done",
            "result": {
                "prompt_tokens": 7000,
                "completion_tokens": 1,
                "computed_prefill_tokens": 7000,
            },
            "speculative": {
                "rounds": 0, "drafted_tokens": 0, "accepted_tokens": 0, "fallback_steps": 0,
            },
        },
        {
            "event": "throughput",
            "tokens": {"computed_prefill": 7000, "committed_decode": 0},
            "decode_batch": {"rounds": 0, "row_rounds": 0},
        },
    ]


def test_analyze_point_records_skipped_fixtures() -> None:
    events = measured_events() + [
        {
            "event": "request_rejected",
            "error": {"code": "context_length_exceeded", "message": SKIP_MESSAGE},
        }
    ]
    report = concurrency.analyze_point(
        skip_point(), ["serve"], Path("log"), {}, "bindings", events, [measured_result()],
        [skip_fixture("long_niah_64k", 1)], 0.0, 1.0,
    )
    assert report["request_count"] == 1
    assert report["fixture_count"] == 1 and report["suite_fixture_count"] == 4
    assert report["skipped_fixtures"] == [
        {"fixture": "long_niah_64k", "seed": 7, "message": SKIP_MESSAGE}
    ]
    row = concurrency.summary_row(report)
    assert row["fixture_count"] == 1 and row["suite_fixture_count"] == 4
    assert row["skipped_fixtures"] == "long_niah_64k"


def test_other_request_errors_stay_fatal_and_all_skips_fail() -> None:
    point = skip_point()
    other = {"event": "request_error", "error": {"message": "decode stopped"}}
    with pytest.raises(corpus.CampaignError, match="serving request failed: decode stopped"):
        concurrency.analyze_point(
            point, ["serve"], Path("log"), {}, "bindings", [other], [], [], 0.0, 1.0
        )
    with pytest.raises(corpus.CampaignError, match="every fixture was refused"):
        concurrency.analyze_point(
            point, ["serve"], Path("log"), {}, "bindings", [], [],
            [skip_fixture("long_niah_64k", 0)], 0.0, 0.0,
        )
    # A skip the server never logged as a context-length refusal is inconsistent, not a result.
    with pytest.raises(corpus.CampaignError, match="do not match the skipped fixtures"):
        concurrency.analyze_point(
            point, ["serve"], Path("log"), {}, "bindings", measured_events(), [measured_result()],
            [skip_fixture("long_niah_64k", 1)], 0.0, 1.0,
        )


@contextlib.contextmanager
def http_stub(status: int, body: bytes):
    """A loopback server that answers every request with one fixed response."""

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


def client_jobs(count: int) -> list[concurrency.Job]:
    return [
        concurrency.Job(
            index=index,
            case_index=index,
            fixture=Fixture("long_niah_64k", [], False, 8, "long_niah"),
            seed=7,
            max_tokens=8,
        )
        for index in range(count)
    ]


def test_run_clients_skips_only_the_context_length_refusal() -> None:
    point = skip_point()
    jobs = client_jobs(3)
    with http_stub(400, CONTEXT_LENGTH_BODY.encode("utf-8")) as port:
        results, skips, start, end = concurrency.run_clients(point, jobs, port)
    assert results == [] and [skip.message for skip in skips] == [SKIP_MESSAGE] * len(jobs)
    assert [skip.job.index for skip in skips] == [0, 1, 2]
    assert end == start
    # With every request refused there is nothing measured, so the point fails instead of
    # reporting an empty one.
    with pytest.raises(corpus.CampaignError, match="every fixture was refused"):
        concurrency.analyze_point(
            point, ["serve"], Path("log"), {}, "bindings", [], results, skips, start, end
        )
    # Any other refusal is still fatal, exactly as it was before the skip existed.
    for status, body in (
        (400, b'{"error":{"code":"invalid_request_error","message":"bad field"}}'),
        (429, b'{"error":{"code":"server_overloaded","message":"busy"}}'),
        (500, b'{"error":{"code":"context_length_exceeded","message":"too long"}}'),
        (400, b"not json"),
    ):
        with http_stub(status, body) as port:
            with pytest.raises(corpus.CampaignError):
                concurrency.run_clients(point, jobs, port)
