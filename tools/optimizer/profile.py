"""Save an optimizer pick as a `ninfer-serve --profile` document.

The profile is the base configuration for a use case; the command line overrides it. It stores the
serve flags (not the optimizer's factor names), so what `ninfer-serve` reads is exactly what the
sweep measured.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from .report import TOOL_VERSION, OptimizerResult

PICKS = ("fastest", "balanced", "max_context")


def serve_options(row, use_case) -> dict:
    """Parse the pick's rendered serve flags into a flag -> value object."""
    args = [*row.setting.render_serve(), "--max-concurrency", str(use_case.concurrency)]
    options: dict[str, object] = {}
    index = 0
    while index < len(args):
        flag = args[index][2:]
        following = args[index + 1] if index + 1 < len(args) else None
        if following is not None and not following.startswith("--"):
            options[flag] = int(following) if following.isdigit() else following
            index += 2
        else:
            options[flag] = True
            index += 1
    return options


def _pick_row(result: OptimizerResult, pick: str):
    if pick not in PICKS:
        raise ValueError(f"unknown pick {pick!r}; known: {', '.join(PICKS)}")
    return {
        "fastest": result.fastest,
        "balanced": result.balanced,
        "max_context": result.max_context,
    }[pick]


def build_profile(result: OptimizerResult, pick: str = "balanced") -> dict:
    row = _pick_row(result, pick)
    if row is None:
        raise ValueError(f"the {pick} pick is empty; nothing to save")
    return {
        "artifact_type": "ninfer_serve_profile",
        "schema_version": 1,
        "model": result.fingerprint["model"]["name"],
        "use_case": result.use_case.name,
        "objective": result.use_case.objective,
        "pick": pick,
        "created": datetime.now(timezone.utc).isoformat(),
        "tool": "ninfer-optimizer",
        "tool_version": TOOL_VERSION,
        "options": serve_options(row, result.use_case),
        "measured": {
            "objective": row.objective,
            "pp_tps": row.pp_tps,
            "tg_tps": row.tg_tps,
            "ttft_ms": row.ttft_ms,
            "ctx": row.setting.values.get("ctx"),
        },
        "fingerprint": result.fingerprint,
    }


def write_profile(path: str | Path, document: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
