"""EXL3 bit allocation: the greedy ordering and the rate table's shape.

These guard the two failures that made the first allocation useless: a signed ΔNLL and an inverted
error curve both flip the greedy, so a sensitive layer ends up *under*-allocated. The rates table is
the tool's observable output, so the assertions read it rather than the private helpers.
"""

from __future__ import annotations

import json

import pytest

from tools.exl3 import allocate

RFN = 0.145


def _write_inputs(tmp_path, layers):
    """layers: list of (layer_index, numel, [kld per member]); anchors are uniform here."""
    targets = []
    tensors = []
    for index, numel, klds in layers:
        for member, kld in enumerate(klds):
            name = _member(index, member)
            targets.append({"name": name, "numel": numel, "kld": kld})
            tensors.append({"name": name, "half_bits": 6, "proxy_error": 0.01})
    measurement = {"rfn": RFN, "clean_nll": 190000.0, "targets": targets}
    report = {"tensors": tensors}
    measurement_path = tmp_path / "measurement.json"
    report_path = tmp_path / "report.json"
    measurement_path.write_text(json.dumps(measurement))
    report_path.write_text(json.dumps(report))
    return measurement_path, report_path


def _member(layer, member):
    return f"text/layers/{layer}/mlp/{'down' if member == 0 else 'gate'}"


def _run(tmp_path, layers, bitrate, extra=()):
    measurement, report = _write_inputs(tmp_path, layers)
    out = tmp_path / "rates.json"
    allocate.main(
        [
            "--measurement", str(measurement),
            "--report", str(report),
            "--bitrate", str(bitrate),
            "--min-k", "2",
            "--max-k", "16",
            "--group-by-layer",
            "--out", str(out),
            *extra,
        ]
    )
    return json.loads(out.read_text())


def test_layer_pooling_and_sensitivity_ordering(tmp_path):
    # Layer 0 carries a large *signed* ΔNLL (the per-tensor sign is noise), layer 63 a small one,
    # and both members of a layer must share the layer's rate. The budget funds two promotions, both
    # of which the sensitive layer must win.
    rates = _run(
        tmp_path,
        layers=[(0, 1000, [-1000.0, -1200.0]), (63, 1000, [10.0, 12.0])],
        bitrate=3.0,
    )
    assert rates[_member(0, 0)] == rates[_member(0, 1)]
    assert rates[_member(63, 0)] == rates[_member(63, 1)]
    assert rates[_member(0, 0)] > rates[_member(63, 0)]
    assert rates["text/output_head"] == 12


def test_budget_is_respected_and_head_is_separate(tmp_path):
    layers = [(0, 1000, [500.0]), (1, 2000, [500.0]), (2, 3000, [500.0])]
    rates = _run(tmp_path, layers=layers, bitrate=6.0)
    body = {name: value for name, value in rates.items() if name != "text/output_head"}
    assert rates["text/output_head"] == 12
    total = sum(numel for _, numel, _ in layers)
    spent = sum(numel * body[_member(index, 0)] for index, numel, _ in layers)
    assert 2 <= min(body.values()) and max(body.values()) <= 16
    # The greedy may leave less than one unit of budget unused; nothing more.
    assert 0 <= 6.0 * total - spent < max(numel for _, numel, _ in layers)


def test_target_below_min_k_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        _run(tmp_path, layers=[(0, 1000, [1.0, 1.0])], bitrate=1.0)
