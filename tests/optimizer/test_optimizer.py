from __future__ import annotations

import json

import pytest

from tools.optimizer.detect import Hardware, ModelInfo
from tools.optimizer.driver import (
    STATUS_OK,
    Measurement,
    _parse_bench_report,
    _plausibility,
)
from tools.optimizer.factors import Factor, FactorOptions, Setting, build_factors, kv_dtype_levels
from tools.optimizer.report import (
    build_result,
    format_result,
    pareto,
    pick_balanced,
    pick_fastest,
    pick_max_context,
    to_json,
)
from tools.optimizer.usecases import USE_CASES, use_case


def _model(**overrides) -> ModelInfo:
    base = dict(
        name="test",
        architecture="Qwen3_5ForCausalLM",
        num_layers=4,
        hidden_size=64,
        vocab_size=128,
        is_moe=False,
        num_experts=0,
        has_mtp=True,
        has_dflash=False,
        has_dflash2=True,
        has_vision=False,
        weight_bytes=1 << 30,
        native_context=32768,
        components=("text", "mtp", "dflash2"),
    )
    base.update(overrides)
    return ModelInfo(**base)


def _hardware() -> Hardware:
    return Hardware("NVIDIA GeForce RTX 5090", 32607, 30000, 24, 32, "nvidia-smi")


def test_use_case_effective_rate_and_validation():
    chat = use_case("chat")
    assert chat.driver == "bench"
    # effective = (P+G) / (P/pp + G/tg)
    rate = chat.effective_tokens_per_second(pp_tps=1000.0, tg_tps=100.0)
    expected = (512 + 256) / (512 / 1000.0 + 256 / 100.0)
    assert rate == pytest.approx(expected)
    assert set(USE_CASES) >= {"chat", "agents", "multi-user", "long-context"}
    with pytest.raises(ValueError):
        use_case("nope")


def test_kv_floor_and_factor_building():
    assert set(kv_dtype_levels(FactorOptions(min_kv="any"))) == {
        "bf16", "int8", "fp8", "nvfp4", "k8v4"
    }
    assert set(kv_dtype_levels(FactorOptions(min_kv="fp8"))) == {"bf16", "int8", "fp8"}
    factors = build_factors(use_case("chat"), _model(), _hardware(), FactorOptions())
    names = {f.name for f in factors}
    assert {"kv_dtype", "ctx", "spec", "prefill_chunk"} <= names
    spec = next(f for f in factors if f.name == "spec").levels
    assert "none" in spec and "mtp+head" in spec and "dflash2+head" in spec


def test_setting_renders_flags():
    setting = Setting(
        {"kv_dtype": "nvfp4", "ctx": "8192", "spec": "dflash2:7+head", "prefill_chunk": "2048",
         "cuda_graph": "off"}
    )
    bench = setting.render_bench()
    assert bench == ["--kv-dtype", "nvfp4", "--max-ctx", "8192", "--spec", "dflash2",
                     "--draft-tokens", "7", "--lm-head-draft", "--prefill-chunk", "2048",
                     "--no-cuda-graph"]
    assert Setting({"spec": "none"}).render_bench() == []
    assert Setting({"spec": "mtp"}).render_bench() == ["--spec", "mtp"]


def test_parse_bench_report(tmp_path):
    report = {
        "schema_version": 15,
        "tests": [
            {"n_prompt": 512, "n_gen": 256, "prefill_seconds_mean": 0.5,
             "decode_seconds_mean": 2.56},
        ],
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    pp, tg = _parse_bench_report(path, use_case("chat"))
    assert pp == pytest.approx(1024.0)
    assert tg == pytest.approx(100.0)


def test_plausibility_rejects_impossible_rates():
    assert _plausibility(1000.0, 100.0)[0] == STATUS_OK
    assert _plausibility(1.0, 99999.0)[0] != STATUS_OK
    assert _plausibility(9e9, 1.0)[0] != STATUS_OK


def _row(ctx: int, objective: float) -> Measurement:
    return Measurement(
        Setting({"ctx": str(ctx), "kv_dtype": "fp8"}), STATUS_OK, 1000.0, objective, objective, 1.0
    )


def test_picks_and_pareto():
    rows = [
        _row(8192, 90.0),
        _row(16384, 80.0),
        _row(32768, 60.0),
        _row(32768, 40.0),
        Measurement(Setting({"ctx": "65536"}), "OOM", 0.0, 0.0, 0.0, 1.0),
    ]
    assert pick_fastest(rows).objective == 90.0
    assert pick_balanced(rows, 16384).objective == 80.0
    assert pick_max_context(rows).objective == 60.0
    frontier = pareto(rows)
    depths = [int(r.setting.values["ctx"]) for r in frontier]
    assert depths == [8192, 16384, 32768]
    assert frontier[-1].objective == 60.0  # the 40.0 row at the same depth is dominated


def test_report_round_trips_to_json():
    factors = [Factor("kv_dtype", ("fp8", "nvfp4"))]
    result = build_result(
        use_case("chat"), [_row(8192, 90.0), _row(32768, 60.0)], [], _hardware(), _model(),
        factors, reps=3,
    )
    text = format_result(result)
    assert "FASTEST" in text and "Pareto frontier" in text
    payload = json.loads(to_json(result))
    assert payload["picks"]["fastest"]["objective"] == 90.0
    assert payload["fingerprint"]["hardware"]["gpu"].startswith("NVIDIA")


def test_thermal_controller_validates_and_off_is_noop():
    from tools.optimizer.driver import ThermalController

    controller = ThermalController("off")
    calls: list[int] = []
    controller.settle(lambda: calls.append(1))
    assert calls == []
    with pytest.raises(ValueError):
        ThermalController("turbo")


def test_corpus_messages_are_available_or_synthetic():
    from tools.optimizer.driver import _corpus_messages_for

    messages = _corpus_messages_for(use_case("chat"))
    assert messages and all("role" in m and "content" in m for m in messages)


def test_confirmation_is_reported():
    result = build_result(
        use_case("chat"), [_row(8192, 90.0)], [], _hardware(), _model(), [], reps=1,
        confirmation=(100.0, _row(8192, 92.0)),
    )
    assert "Confirmation run" in format_result(result)
    payload = json.loads(to_json(result))
    assert payload["confirmation"]["predicted"] == 100.0
