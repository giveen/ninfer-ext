"""Plot a concurrency sweep: prefill and decode throughput against context length.

Reads the point files `tools/bench/run_serve_concurrency.py` writes (`<run>/points/*.json`, plus any
`summary.json` beside them) and draws, per model, one line per concurrency: **prefill tokens/s** and
**decode tokens/s** against the context the point exercised. Decode is also drawn for
`decode-saturation` points, which fix the context and so carry no prefill number.

Context is read from the run itself rather than guessed from a fixture name: a point's context is the
largest `prompt_tokens` among its requests, which is what the model had to attend over. Throughput is
the runner's own aggregate (`computed_prefill_tokens_per_second`, and
`decode_tokens_per_second` — under `steady` for decode-saturation).

    python3 tools/bench/plot_concurrency.py --run profiles/bench/<campaign> \\
        --out profiles/bench/<campaign>/figures

Run it with an interpreter that has matplotlib (`/usr/bin/python3` on this machine carries it; the
project venv does not). Writes one PNG per model, plus an SVG beside it. `--list` prints what it found
and draws nothing, which is how to check a run before plotting it.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (backend must be chosen first)


def decode_throughput(metrics: dict) -> float | None:
    """The runner's aggregate decode rate, under either suite's shape."""
    if isinstance(metrics.get("decode_tokens_per_second"), (int, float)):
        return float(metrics["decode_tokens_per_second"])
    steady = metrics.get("steady")
    if isinstance(steady, dict) and isinstance(steady.get("decode_tokens_per_second"), (int, float)):
        return float(steady["decode_tokens_per_second"])
    return None


def point_reading(point: dict) -> dict | None:
    """What one point file says: model, suite, concurrency, context, and both rates."""
    requests = point.get("requests") or []
    if not requests:
        return None
    metrics = point.get("metrics") or {}
    contexts = [r.get("prompt_tokens") for r in requests if isinstance(r.get("prompt_tokens"), int)]
    if not contexts:
        return None
    return {
        "model": point.get("model") or point.get("artifact_path") or "unknown",
        "suite": point.get("suite") or "unknown",
        "concurrency": int(point.get("concurrency") or 1),
        "context": max(contexts),
        "prefill": metrics.get("computed_prefill_tokens_per_second"),
        "decode": decode_throughput(metrics),
        "mode": point.get("speculative_mode"),
        "path": point.get("artifact_path"),
    }


def collect(runs: list[Path]) -> list[dict]:
    readings: list[dict] = []
    for run in runs:
        for path in sorted((run / "points").glob("*.json")):
            try:
                readings.append(point_reading(json.loads(path.read_text())))
            except (json.JSONDecodeError, OSError):
                continue
        # A summary beside the points repeats the same metrics; prefer the points, and fall back to
        # the summary only when the run kept no points.
        summary = run / "summary.json"
        if summary.is_file() and not sorted((run / "points").glob("*.json")):
            try:
                payload = json.loads(summary.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            points = payload.get("points") if isinstance(payload, dict) else payload
            for point in points or []:
                if isinstance(point, dict):
                    readings.append(point_reading(point))
    return [r for r in readings if r]


def collect_bench(paths: list[Path]) -> list[dict]:
    """The same reading from `ninfer_bench --output-file` JSON, which measures one request at a time.

    Those rows are the only source of a prefill-against-context curve today: the serving runner's
    `corpus-makespan` point aggregates over its whole fixture set, so a single point cannot place a
    prefill rate at one context. Each row is single-request, so it is concurrency 1.
    """
    readings: list[dict] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        artifact = payload.get("artifact")
        model = artifact if isinstance(artifact, str) else (artifact or {}).get("name") or path.stem
        for row in payload.get("tests") or []:
            context = row.get("n_prompt")
            if not isinstance(context, int) or context < 1:
                continue
            readings.append(
                {
                    "model": model,
                    "suite": "ninfer_bench",
                    "concurrency": 1,
                    "context": context,
                    "prefill": row.get("prefill_tok_s_mean"),
                    "decode": row.get("decode_output_tok_s_mean"),
                    "path": str(path),
                }
            )
    return readings


def point_concurrency(run: Path, stem: str) -> int:
    """The lane count the record's point ran at: its own field, else the run directory's name.

    Hardcoding one here folded every campaign's records into the c1 line.
    """
    try:
        point = json.loads((run / "points" / f"{stem}.json").read_text(encoding="utf-8"))
        concurrency = point.get("concurrency")
        if isinstance(concurrency, int) and concurrency > 0:
            return concurrency
    except (OSError, json.JSONDecodeError):
        pass
    for part in run.name.split("_"):
        if part.startswith("c") and part[1:].isdigit():
            return int(part[1:])
    return 1


def collect_records(runs: list[Path]) -> list[dict]:
    """Per-request records: the only source of a prefill/decode rate at *one* context.

    A `corpus-makespan` point aggregates over its whole fixture set, so its own
    `decode_tokens_per_second` is decode tokens over the *whole makespan* -- prefill included, which
    for a long-prompt corpus is not a decode rate at all. The runner also writes
    `corpus/<point>/results.jsonl`, one record per request, whose `metrics` carry `prompt_tokens`,
    `prefill_tok_s` and `decode_tok_s` for that request alone. A curve against context has to be
    drawn from those, at every concurrency.
    """
    readings: list[dict] = []
    for run in runs:
        for path in sorted((run / "corpus").glob("*/results.jsonl")):
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            concurrency = point_concurrency(run, path.parent.name)
            for line in lines:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    metrics = record["metrics"]
                    context = int(metrics["prompt_tokens"])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
                readings.append(
                    {
                        "model": record.get("target") or record.get("model_id") or "unknown",
                        "suite": "serve request",
                        "concurrency": concurrency,
                        "context": context,
                        "prefill": metrics.get("prefill_tok_s"),
                        "decode": metrics.get("decode_tok_s"),
                        "mode": record.get("speculative_mode"),
                        "path": str(path),
                    }
                )
    return readings


def mode_label(reading: dict) -> str:
    """The plain path stays unlabelled; a speculative mode is named so one figure cannot merge it
    with its own control."""
    mode = reading.get("mode")
    return "" if mode in (None, "mtp0", "none") else f" {mode}"


def draw(model: str, readings: list[dict], out: Path) -> list[Path]:
    """One figure per model: prefill and decode against context, a line per concurrency.

    The prefill panel takes every reading, including a `corpus-makespan` point's aggregate (total
    prefill tokens over the makespan, which is a real throughput at any concurrency). The decode
    panel does not: that same aggregate divides decode tokens by the *whole* makespan, and this
    corpus stops after ~17 tokens per NIAH request, so its "decode rate" is an artifact of early
    stopping. Per-request records -- the only honest decode rate for this corpus -- exist at
    concurrency 1, because the runner can only pair a result with its server `request_done` event
    there. A decode curve at higher concurrency needs the steady-decode suite.
    """
    written: list[Path] = []
    panels = [("prefill", "prefill tok/s", frozenset()),
              ("decode", "decode tok/s", frozenset({"corpus-makespan"}))]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    drew = False
    for axis, (key, title, excluded) in zip(axes, panels):
        series: dict[tuple[int, str], list[tuple[int, float]]] = defaultdict(list)
        for reading in readings:
            if reading.get("suite") in excluded:
                continue
            value = reading.get(key)
            if isinstance(value, (int, float)) and value > 0:
                series[(reading["concurrency"], mode_label(reading))].append(
                    (reading["context"], float(value))
                )
        for concurrency, mode in sorted(series):
            points = sorted(series[(concurrency, mode)])
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            # A single context has no line to draw; the marker still shows the level.
            axis.plot(xs, ys, marker="o", label=f"c{concurrency}{mode}")
            drew = True
        axis.set_xscale("log", base=2)
        axis.set_xlabel("context (prompt tokens)")
        axis.set_ylabel(title)
        axis.set_title(f"{model}: {title} against context")
        axis.grid(True, which="both", alpha=0.3)
        handles, labels = axis.get_legend_handles_labels()
        if labels:
            axis.legend(title="concurrency")
    if not drew:
        plt.close(fig)
        return written
    stem = out / f"{model.replace('/', '_')}_pp_tg_vs_context"
    for suffix in (".png", ".svg"):
        path = stem.with_suffix(suffix)
        fig.savefig(path, dpi=140)
        written.append(path)
    plt.close(fig)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append",
                        help="a run directory holding points/ (repeatable)")
    parser.add_argument("--bench", type=Path, action="append",
                        help="a ninfer_bench --output-file JSON (repeatable)")
    parser.add_argument("--out", type=Path, help="where the figures go")
    parser.add_argument("--list", action="store_true", help="print what was found, draw nothing")
    args = parser.parse_args()

    runs = args.run or []
    records = collect_records(runs)
    recorded = {reading["concurrency"] for reading in records}
    # Records replace that concurrency's *corpus-makespan* aggregate, whose decode rate is taken over
    # the whole makespan and so is not a decode rate for this corpus. A decode-saturation point is a
    # different measurement -- steady decode at a fixed context, carrying no prefill -- and must
    # survive even for a concurrency that has records, or c1 loses its entire tg curve.
    points = [reading for reading in collect(runs)
              if reading["suite"] != "corpus-makespan" or reading["concurrency"] not in recorded]
    readings = points + records + collect_bench(args.bench or [])
    if not readings:
        print("no points found; pass the directory the runner wrote (it holds points/)")
        return 1
    for reading in sorted(readings, key=lambda r: (r["model"], r["concurrency"], r["context"])):
        prefill = "n/a" if reading["prefill"] is None else f"{reading['prefill']:.1f}"
        decode = "n/a" if reading["decode"] is None else f"{reading['decode']:.1f}"
        print(f"{reading['model']:<40} {reading['suite']:<18} c{reading['concurrency']} "
              f"context {reading['context']:>7} prefill {prefill:>9} decode {decode:>9}")
    if args.list:
        return 0
    if args.out is None:
        print("--out is required unless --list is given")
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    by_model: dict[str, list[dict]] = defaultdict(list)
    for reading in readings:
        by_model[reading["model"]].append(reading)
    for model, model_readings in sorted(by_model.items()):
        for path in draw(model, model_readings, args.out):
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
