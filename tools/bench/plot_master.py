"""Master charts: one figure per model, its two throughputs against context *and* concurrency.

Reads the same runs as `plot_concurrency.py` (a run directory holding `points/`) and draws four panels:
prefill and decode against context, each with a line per concurrency, and the same two rates against
concurrency, each with a line per context.

The two rates are deliberately taken from the two suites that measure them cleanly, so every point on a
panel shares one definition:

* prefill is the `corpus-makespan` aggregate -- every prompt token of the point divided by the whole
  makespan -- which exists at every concurrency. It therefore includes each request's decode time and
  reads a few percent under a per-request prefill rate, identically on every panel.
* decode is the `decode-saturation` steady rate, measured over complete intervals whose decode batch
  equals the concurrency, at the context its prompt was padded to. `corpus-makespan`'s own decode rate
  is not used: for this corpus it is decode tokens over the makespan, which is not a decode rate at all.

Cells are sparse on purpose. A concurrency can only reach the contexts whose windows fit the card, so
the panels show which combinations exist rather than interpolating over the ones that do not.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.bench import plot_concurrency as pc  # noqa: E402


def series(readings: list[dict[str, Any]], suite: str, key: str) -> dict[tuple[int, int], float]:
    """(concurrency, context) -> rate, for one suite and one rate."""
    out: dict[tuple[int, int], float] = {}
    for reading in readings:
        if reading.get("suite") != suite:
            continue
        value = reading.get(key)
        if isinstance(value, (int, float)) and value > 0:
            out[(reading["concurrency"], reading["context"])] = float(value)
    return out


def prefill_series(runs: list[Path], readings: list[dict[str, Any]], model: str) -> dict[tuple[int, int], float]:
    """Prefill at every concurrency, from the two sources that measure it.

    A `corpus-makespan` point reports one aggregate -- prompt tokens over the whole makespan -- which
    places a single value at that point's largest prompt, so the aggregate alone would leave one cell
    per concurrency. The per-request records exist only at concurrency 1, and they carry a true
    per-request prefill rate at each of the point's contexts. So c1 comes from the records and c>=2
    from the aggregate; the two read a few percent apart, since the aggregate includes each request's
    own decode time.
    """
    out = {key: value for key, value in series(readings, "corpus-makespan", "prefill").items()
           if key[0] != 1}
    for record in pc.collect_records(runs):
        if record.get("model") != model:
            continue
        value = record.get("prefill")
        if isinstance(value, (int, float)) and value > 0:
            out[(record["concurrency"], record["context"])] = float(value)
    return out


def decode_series(readings: list[dict[str, Any]]) -> dict[tuple[int, int, str], float]:
    """Decode rate keyed by (concurrency, context, mode), so a chart can carry MTP beside plain.

    The plain path keeps an empty mode string -- it is the baseline a chart reads against -- and any
    speculative run keeps its mode name, which is also what makes the line dashed.

    First reading wins per cell. Two runs measure the same plain cells (the decode pass and the
    speculative A/B's own plain arm, which exist to make the ratio), and they differ by one to three
    percent on this clock-sensitive box; pass the decode pass first so a chart plots the run the prose
    quotes, and let the A/B contribute only its speculative series.
    """
    out: dict[tuple[int, int, str], float] = {}
    for reading in readings:
        if reading.get("suite") != "decode-saturation":
            continue
        value = reading.get("decode")
        if not isinstance(value, (int, float)) or value <= 0:
            continue
        mode = reading.get("mode") or ""
        plain = "" if mode in ("mtp0", "none", "None") else str(mode)
        out.setdefault((reading["concurrency"], reading["context"], plain), float(value))
    return out


def human_context(tokens: int) -> str:
    return f"{tokens / 1000:.1f}k" if tokens >= 1000 else str(tokens)


def draw(model: str, readings: list[dict[str, Any]], runs: list[Path], out: Path) -> list[Path]:
    prefill = prefill_series(runs, readings, model)
    decode = decode_series(readings)
    if not prefill and not decode:
        return []

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)

    def against_context(axis, data, title, ylabel) -> None:
        grouped: dict[tuple[int, str], list[tuple[int, float]]] = collections.defaultdict(list)
        for key, value in data.items():
            concurrency, context = int(key[0]), int(key[1])
            mode = str(key[2]) if len(key) > 2 else ""
            grouped[(concurrency, mode)].append((context, value))
        for (concurrency, mode) in sorted(grouped):
            points = sorted(grouped[(concurrency, mode)])
            label = f"c{concurrency}" + (f" {mode}" if mode else "")
            axis.plot([p[0] for p in points], [p[1] for p in points], marker="o",
                      linestyle="--" if mode else "-", label=label)
        axis.set_xscale("log", base=2)
        axis.set_xlabel("context (prompt tokens)")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, which="both", alpha=0.3)
        if grouped:
            axis.legend(title="concurrency / mode", fontsize=8)

    def against_concurrency(axis, data, title, ylabel) -> None:
        grouped: dict[tuple[int, str], list[tuple[int, float]]] = collections.defaultdict(list)
        for key, value in data.items():
            concurrency, context = int(key[0]), int(key[1])
            mode = str(key[2]) if len(key) > 2 else ""
            grouped[(context, mode)].append((concurrency, value))
        for (context, mode) in sorted(grouped):
            points = sorted(grouped[(context, mode)])
            label = human_context(context) + (f" {mode}" if mode else "")
            axis.plot([p[0] for p in points], [p[1] for p in points], marker="o",
                      linestyle="--" if mode else "-", label=label)
        axis.set_xscale("log", base=2)
        axis.set_xticks([1, 2, 4, 8])
        axis.set_xticklabels(["1", "2", "4", "8"])
        axis.set_xlabel("concurrency (lanes)")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, which="both", alpha=0.3)
        if grouped:
            axis.legend(title="context / mode", fontsize=8)

    against_context(axes[0][0], prefill, "prefill against context", "prefill tok/s")
    against_context(axes[0][1], decode, "decode against context (steady)", "decode tok/s")
    against_concurrency(axes[1][0], prefill, "prefill against concurrency", "prefill tok/s")
    against_concurrency(axes[1][1], decode, "decode against concurrency (steady)", "decode tok/s")

    fig.suptitle(f"{model}: throughput across context and concurrency", fontsize=13)
    fig.text(0.005, 0.002,
             "prefill: c1 per request, c2-c8 the point aggregate (prompt tokens over the whole "
             "makespan, so it includes each request's decode)   |   "
             "decode: steady intervals whose decode batch equals the concurrency",
             fontsize=7, ha="left", va="bottom")
    written: list[Path] = []
    # Not Path.with_suffix: a model name may carry a dot itself (qwen3.8-flash-next), and with_suffix
    # would take everything after it as the suffix and write qwen3.png.
    stem = out / f"{model.replace('/', '_')}_master"
    for suffix in (".png", ".svg"):
        path = Path(f"{stem}{suffix}")
        fig.savefig(path, dpi=140)
        written.append(path)
    plt.close(fig)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True,
                        help="a run directory holding points/ (repeatable)")
    parser.add_argument("--out", type=Path, required=True, help="where the figures go")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    readings = [r for r in pc.collect(args.run) if r]
    if not readings:
        print("no points found; pass the directories the runner wrote (they hold points/)")
        return 1
    by_model: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for reading in readings:
        by_model[reading["model"]].append(reading)
    for model, model_readings in sorted(by_model.items()):
        for path in draw(model, model_readings, args.run, args.out):
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
