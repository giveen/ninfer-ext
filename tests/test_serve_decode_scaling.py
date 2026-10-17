"""The decode-scaling campaign's own arithmetic and its aggregate/per-request split.

The campaign itself is orchestration over the serving runner; what these cover is the part it adds
-- the per-lane window rule, the corpus guard, the family-forced preflight, and the row that must
not confuse an aggregate rate with a per-request one -- without starting a server or a GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.bench import run_serve_decode_scaling as scaling
from tools.bench import run_serve_concurrency as concurrency
from tools.bench import run_serve_corpus as corpus


def _facts() -> scaling.ArtifactFacts:
    return scaling.ArtifactFacts(
        vocab_size=131072, native_context=262144, moe_banks=40, pool_experts=13824
    )


def _point(instance: int) -> concurrency.Point:
    return concurrency.Point(
        target="nemotron_h_74b_nvfp4",
        model_id="nemotron_h_74b_nvfp4",
        artifact=Path("/tmp/model.ninfer"),
        speculative_mode="mtp0",
        speculative_backend="none",
        draft_tokens=0,
        sampling_mode="stochastic",
        kv_dtype="bf16",
        suite="decode-saturation",
        concurrency=instance,
    )


def _server_start(instance: int, window: int, **overrides: object) -> dict[str, object]:
    engine: dict[str, object] = {
        "device": 0,
        "max_context": window,
        "kv_capacity_mode": "explicit",
        "kv_capacity": window,
        "max_concurrency": instance,
        "max_pending_requests": 1,
        "pending_timeout_ms": concurrency.PENDING_TIMEOUT_MS,
        "kv_cache": "bf16",
        "cuda_graph": False,
        "prefix_reuse": False,
        "speculative_backend": "none",
        "context_cache": {"enabled": False},
    }
    engine.update(overrides)
    return {
        "event": "server_start",
        "artifact_type": corpus.SERVER_LOG_ARTIFACT_TYPE,
        "schema_version": corpus.SERVER_LOG_SCHEMA_VERSIONS[-1],
        "server_instance_id": "instance",
        "server": {"public_model_id": "nemotron_h_74b_nvfp4"},
        "artifact": {
            "path": "/tmp/model.ninfer",
            "prefill_signature": "nemotron-signature",
        },
        "engine": engine,
        "sampling_defaults": {"greedy": False},
    }


def test_window_keeps_the_lane_budget_constant_across_the_sweep() -> None:
    windows = {c: scaling.window_for(262144, c) for c in (1, 2, 4, 8)}
    assert windows == {1: 262144, 2: 131072, 4: 65536, 8: 32768}
    # The attention state is capacity x lanes, so the product is what must stay put.
    assert {c * w for c, w in windows.items()} == {262144}
    with pytest.raises(corpus.CampaignError):
        scaling.window_for(4, 8)


def test_corpus_guard_refuses_ids_outside_the_artifact_vocabulary(tmp_path: Path) -> None:
    in_domain = tmp_path / "in_domain.ids"
    in_domain.write_text("1010 1376 131018\n", encoding="utf-8")
    facts = scaling.read_corpus(in_domain, 131072)
    assert (facts.tokens, facts.max_id) == (3, 131018)

    out_of_domain = tmp_path / "out_of_domain.ids"
    out_of_domain.write_text("1010 182531\n", encoding="utf-8")
    with pytest.raises(corpus.CampaignError, match="outside the artifact"):
        scaling.read_corpus(out_of_domain, 131072)


def test_window_guard_refuses_a_point_that_cannot_hold_prompt_and_output() -> None:
    # C=8 of a 262144 budget: 32768 tokens, enough for the default 8192 pad and its output.
    scaling.guard_window(32768, 8, 8192, 4096, 262144, _facts())
    # The 3.0 chars/token pad undershoots English prose, so the guard allows 1.5x and refuses a
    # point whose window cannot hold that plus the fixture's own prompt.
    with pytest.raises(corpus.CampaignError, match="raise --lane-token-budget"):
        scaling.guard_window(16384, 2, 8192, 4096, 32768, _facts())


def test_artifact_facts_count_the_expert_pool_from_the_conversion_report(tmp_path: Path) -> None:
    artifact = tmp_path / "model.ninfer"
    artifact.write_bytes(b"")
    report = tmp_path / "model.ninfer.conversion.json"
    report.write_text(
        json.dumps(
            {
                "components": {
                    "text": {
                        "config": {
                            "vocab_size": 131072,
                            "max_position_embeddings": 262144,
                            "block_configs": [
                                {"block_type": "mamba"},
                                {"block_type": "moe", "n_routed_experts": 240},
                                {"block_type": "attention"},
                                {"block_type": "moe", "n_routed_experts": 432},
                            ],
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    facts = scaling.read_artifact_facts(artifact, None)
    assert (facts.vocab_size, facts.native_context) == (131072, 262144)
    assert (facts.moe_banks, facts.pool_experts) == (2, 672)
    # A conversion report with no MoE blocks reports a pool of zero rather than a missing count.
    assert scaling.read_artifact_facts(artifact, 4096).vocab_size == 4096


def test_preflight_accepts_the_forced_family_values_and_refuses_qwen_defaults() -> None:
    point = _point(8)
    args = type("Args", (), {"device": 0})()
    instance_id, signature = scaling.validate_server_start(
        _server_start(8, 32768), point, args, 32768
    )
    assert (instance_id, signature) == ("instance", "nemotron-signature")
    # cuda_graph True with a live context cache is the Qwen-family shape the shared runner asserts;
    # this family forces both off, so a start event that reports them must be refused.
    with pytest.raises(corpus.CampaignError, match="forced values"):
        scaling.validate_server_start(
            _server_start(8, 32768, cuda_graph=True), point, args, 32768
        )
    with pytest.raises(corpus.CampaignError, match="context cache"):
        scaling.validate_server_start(
            _server_start(8, 32768, context_cache={"enabled": True}), point, args, 32768
        )


def test_preflight_records_the_empty_signature_this_family_reports() -> None:
    # nemotron_h's load summary never assigns a prefill signature (it resolves no prefill cost
    # identity), so an empty one is the correct value for every point rather than a refusal.
    args = type("Args", (), {"device": 0})()
    event = _server_start(1, 262144)
    event["artifact"]["prefill_signature"] = ""
    instance_id, signature = scaling.validate_server_start(event, _point(1), args, 262144)
    assert (instance_id, signature) == ("instance", "")


def test_probe_extracts_chat_content_and_unifies_reasoning() -> None:
    # The probe reads the same response shape the client does; reasoning is included so a reply
    # that lands its answer in a reasoning field is not counted as a miss.
    assert scaling.probe_text({"choices": [{"message": {"content": "Paris"}}]}) == "Paris"
    assert scaling.probe_text(
        {"choices": [{"message": {"content": "Paris", "reasoning_content": "thinking"}}]}
    ) == "Paris\nthinking"
    with pytest.raises(corpus.CampaignError):
        scaling.probe_text({"choices": []})
    assert scaling.probe_matched("The capital is Paris.", "paris")
    assert not scaling.probe_matched("The capital is Lyon.", "paris")


def test_row_separates_the_aggregate_rate_from_the_per_request_rate() -> None:
    events = [
        {"event": "request_done", "timings_seconds": {"ttft": 1.5}},
        {"event": "request_done", "timings_seconds": {"ttft": 2.5}},
        {"event": "throughput", "timings_seconds": {}},
    ]
    report = {
        "engine": {
            "expert_cache_slots": 4608,
            "expert_cache_bytes": 4 * 2**30,
            "prefill_chunk": 1024,
            "max_concurrency": 2,
            "cuda_graph": False,
            "prefix_reuse": False,
            "proposal_head": "optimized",
            "speculative_backend": "none",
        },
        "memory": {
            "available_after_weights_bytes": 8 * 2**30,
            "available_after_startup_bytes": 6 * 2**30,
            "runtime_reservation_bytes": 3 * 2**30,
            "weights": {"used_bytes": 21 * 2**30},
            "sequence": {"used_bytes": 2 * 2**30},
        },
        "environment": {"total_device_memory_bytes": 32 * 2**30, "gpu_name": "RTX 5090"},
        "metrics": {
            "steady": {
                "decode_tokens_per_second": 160.0,
                "committed_decode_tokens": 6400,
                "seconds": 40.0,
                "intervals": 40,
                "average_decode_batch": 2.0,
            }
        },
        "totals": {"decode_tokens": 8000},
        "requests": [
            {"prompt_tokens": 8485, "finish_reason": "output_limit"},
            {"prompt_tokens": 8485, "finish_reason": "output_limit"},
        ],
    }
    row = scaling.point_row(_point(2), 131072, report, events, _facts(), "2873 MiB, 32607 MiB")
    assert row["aggregate_decode_tokens_per_second"] == 160.0
    assert row["per_request_decode_tokens_per_second"] == 80.0
    assert row["milliseconds_per_token_per_lane"] == pytest.approx(12.5)
    assert (row["ttft_seconds_mean"], row["ttft_seconds_max"]) == (2.0, 2.5)
    assert row["prompt_tokens"] == 8485
    assert row["resident_experts"] == 4608
    assert row["free_vram_at_load_bytes"] == 8 * 2**30
    assert row["finish_reasons"] == ["output_limit"]
