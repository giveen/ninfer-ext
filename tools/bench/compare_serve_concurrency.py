#!/usr/bin/env python3
"""Compare two run_serve_concurrency output directories point by point.

Typical use: one directory measured with a stock ninfer-serve build and one with this fork, the same
artifact labels, suites and concurrencies. Rows match on (target, suite, concurrency) and a mode
pair; by default each mode is compared with itself, and --pair BASE=CANDIDATE compares different
modes, for example a stock fixed MTP5 against the fork's adaptive MTP.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, metavar="LABEL=DIR")
    parser.add_argument("--candidate", required=True, metavar="LABEL=DIR")
    parser.add_argument(
        "--pair",
        action="append",
        metavar="BASE_MODE=CANDIDATE_MODE",
        help="compare these speculative modes; repeat (default: every mode present in both)",
    )
    parser.add_argument("--output", type=Path, help="write the Markdown table here too")
    return parser.parse_args(argv)


def labelled_dir(value: str) -> tuple[str, Path]:
    label, sep, path = value.partition("=")
    if not sep or not label or not path:
        raise SystemExit(f"error: expected LABEL=DIR, got {value!r}")
    return label, Path(path).expanduser()


def load_points(directory: Path) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    summary = directory / "summary.json"
    if not summary.is_file():
        raise SystemExit(f"error: {summary} not found; run run_serve_concurrency first")
    points = json.loads(summary.read_text(encoding="utf-8"))["points"]
    return {
        (p["target"], p["speculative_mode"], p["suite"], int(p["concurrency"])): p for p in points
    }


def metric(point: dict[str, Any]) -> tuple[float, bool]:
    """The point's headline value and whether higher is better."""
    if point["suite"] == "decode-saturation":
        return float(point["metrics"]["steady"]["decode_tokens_per_second"]), True
    return float(point["metrics"]["makespan_seconds"]), False


def acceptance(point: dict[str, Any]) -> str:
    rate = point["totals"].get("speculative_acceptance")
    return "—" if rate is None else f"{100.0 * float(rate):.1f}%"


def compare(
    base: dict[tuple[str, str, str, int], dict[str, Any]],
    cand: dict[tuple[str, str, str, int], dict[str, Any]],
    pairs: Sequence[tuple[str, str]],
    base_label: str,
    cand_label: str,
) -> str:
    rows: list[str] = []
    for (target, mode, suite, concurrency), base_point in sorted(base.items()):
        for base_mode, cand_mode in pairs:
            if mode != base_mode:
                continue
            cand_point = cand.get((target, cand_mode, suite, concurrency))
            if cand_point is None:
                continue
            base_value, higher_better = metric(base_point)
            cand_value, _ = metric(cand_point)
            ratio = cand_value / base_value if higher_better else base_value / cand_value
            modes = base_mode if base_mode == cand_mode else f"{base_mode} → {cand_mode}"
            unit = "tok/s" if higher_better else "s"
            rows.append(
                f"| {target} | {suite} | {modes} | {concurrency} | {base_value:.1f} {unit} "
                f"({acceptance(base_point)}) | {cand_value:.1f} {unit} "
                f"({acceptance(cand_point)}) | {100.0 * (ratio - 1.0):+.1f}% |"
            )
    if not rows:
        raise SystemExit("error: no matching points between the two directories")
    header = (
        f"| Target | Suite | Mode | C | {base_label} (accept) | {cand_label} (accept) | Change |\n"
        "|---|---|---|---:|---:|---:|---:|"
    )
    return header + "\n" + "\n".join(rows) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    base_label, base_dir = labelled_dir(args.baseline)
    cand_label, cand_dir = labelled_dir(args.candidate)
    base, cand = load_points(base_dir), load_points(cand_dir)
    if args.pair:
        pairs = []
        for value in args.pair:
            left, sep, right = value.partition("=")
            if not sep or not left or not right:
                raise SystemExit(f"error: expected BASE_MODE=CANDIDATE_MODE, got {value!r}")
            pairs.append((left, right))
    else:
        shared = {key[1] for key in base} & {key[1] for key in cand}
        pairs = [(mode, mode) for mode in sorted(shared)]
    table = compare(base, cand, pairs, base_label, cand_label)
    sys.stdout.write(table)
    if args.output:
        args.output.write_text(table, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
