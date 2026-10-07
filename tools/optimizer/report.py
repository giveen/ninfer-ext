"""Picks, Pareto frontier, main effects and the machine fingerprint.

The recommendation is read from *measured* configurations (the Pareto frontier and the raw rows),
not from the additive main-effects model, which can misattribute an interaction to a main effect.
Main effects are reported to rank which knobs matter.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import platform
from pathlib import Path

from .design import MainEffect
from .detect import Hardware, ModelInfo
from .driver import Measurement
from .factors import Factor
from .usecases import UseCase, OBJECTIVE_LABELS

TOOL_VERSION = "0.1.0"


def _ctx(setting_values: dict[str, str]) -> int:
    try:
        return int(setting_values.get("ctx", "0"))
    except ValueError:
        return 0


def ok_rows(rows: list[Measurement]) -> list[Measurement]:
    return [r for r in rows if r.ok and r.objective > 0]


def pick_fastest(rows: list[Measurement]) -> Measurement | None:
    usable = ok_rows(rows)
    return max(usable, key=lambda r: r.objective) if usable else None


def pick_balanced(rows: list[Measurement], ctx_floor: int) -> Measurement | None:
    usable = [r for r in ok_rows(rows) if _ctx(r.setting.values) >= ctx_floor]
    return max(usable, key=lambda r: r.objective) if usable else None


def pick_max_context(rows: list[Measurement]) -> Measurement | None:
    usable = ok_rows(rows)
    if not usable:
        return None
    return max(usable, key=lambda r: (_ctx(r.setting.values), r.objective))


def pareto(rows: list[Measurement]) -> list[Measurement]:
    """Configs no other measured config beats on both context depth and objective."""
    usable = ok_rows(rows)
    frontier: list[Measurement] = []
    for candidate in usable:
        depth = _ctx(candidate.setting.values)
        if any(
            _ctx(other.setting.values) >= depth
            and other.objective > candidate.objective
            and other is not candidate
            for other in usable
        ):
            continue
        frontier.append(candidate)
    # Deduplicate by depth, keeping the best objective.
    best: dict[int, Measurement] = {}
    for row in frontier:
        depth = _ctx(row.setting.values)
        if depth not in best or row.objective > best[depth].objective:
            best[depth] = row
    return sorted(best.values(), key=lambda r: _ctx(r.setting.values))


def fingerprint(hardware: Hardware, model: ModelInfo, use_case: UseCase,
                factors: list[Factor], reps: int) -> dict:
    return {
        "tool": "ninfer-optimizer",
        "tool_version": TOOL_VERSION,
        "use_case": use_case.name,
        "objective": use_case.objective,
        "hardware": {
            "gpu": hardware.gpu_name,
            "vram_mib": hardware.vram_mib,
            "vram_free_mib": hardware.vram_free_mib,
            "physical_cores": hardware.physical_cores,
            "logical_cores": hardware.logical_cores,
            "vram_source": hardware.source,
        },
        "model": {
            "name": model.name,
            "architecture": model.architecture,
            "layers": model.num_layers,
            "is_moe": model.is_moe,
            "num_experts": model.num_experts,
            "weight_gib": round(model.weight_gi_b, 3),
            "components": list(model.components),
        },
        "os": platform.platform(),
        "reps": reps,
        "factors": [{"name": f.name, "levels": list(f.levels)} for f in factors],
    }


def format_plan(
    artifact: Path, use_case: UseCase, hardware: Hardware, model: ModelInfo,
    factors: list[Factor], run_count: int, levels: int, sample_command: list[str],
    seconds_per_run: float | None,
) -> str:
    lines = [
        f"ninfer-optimizer {TOOL_VERSION} — plan (no GPU used)",
        "",
        f"  artifact : {artifact}",
        f"  model    : {model.name} ({model.architecture}), {model.num_layers} layers, "
        f"{model.weight_gi_b:.2f} GiB weights"
        + (f", {model.num_experts} experts" if model.is_moe else ""),
        f"  GPU      : {hardware.gpu_name} ({hardware.vram_gi_b:.1f} GiB"
        + (f", {hardware.vram_free_mib / 1024.0:.1f} GiB free)" if hardware.vram_mib else ")"),
        f"  CPU      : {hardware.physical_cores} physical / {hardware.logical_cores} logical cores",
        "",
        f"  use case : {use_case.name} — {use_case.summary}",
        f"  request  : prompt {use_case.n_prompt} + gen {use_case.n_gen}, "
        f"concurrency {use_case.concurrency}, driver {use_case.driver}",
        f"  objective: {OBJECTIVE_LABELS[use_case.objective]}",
        "",
        f"  factors ({len(factors)}), orthogonal array with {levels} levels -> {run_count} runs:",
    ]
    for factor in factors:
        lines.append(f"    {factor.name:<16} {', '.join(factor.levels)}")
    if seconds_per_run is not None:
        lines.append("")
        lines.append(f"  estimated cost: {run_count} runs x ~{seconds_per_run:.0f}s "
                     f"~= {run_count * seconds_per_run / 60.0:.0f} min")
    lines += [
        "",
        "  run 1 would execute:",
        "    " + " ".join(sample_command),
        "",
        "  add --run to execute the sweep; --screen first to prune knobs, --levels N to shrink.",
    ]
    return "\n".join(lines)


@dataclass
class OptimizerResult:
    use_case: UseCase
    rows: list[Measurement]
    effects: list[MainEffect]
    fastest: Measurement | None
    balanced: Measurement | None
    max_context: Measurement | None
    frontier: list[Measurement]
    fingerprint: dict
    confirmation: tuple[float, Measurement] | None = None


def build_result(
    use_case: UseCase,
    rows: list[Measurement],
    effects: list[MainEffect],
    hardware: Hardware,
    model: ModelInfo,
    factors: list[Factor],
    reps: int,
    confirmation: tuple[float, Measurement] | None = None,
) -> OptimizerResult:
    return OptimizerResult(
        use_case=use_case,
        rows=rows,
        effects=effects,
        fastest=pick_fastest(rows),
        balanced=pick_balanced(rows, use_case.ctx_floor),
        max_context=pick_max_context(rows),
        frontier=pareto(rows),
        fingerprint=fingerprint(hardware, model, use_case, factors, reps),
        confirmation=confirmation,
    )


def format_result(result: OptimizerResult) -> str:
    use_case = result.use_case
    ok = ok_rows(result.rows)
    lines = [
        f"RESULTS: {len(ok)}/{len(result.rows)} configurations produced a number",
        f"objective: {OBJECTIVE_LABELS[use_case.objective]}",
        "",
    ]

    def show(title: str, row: Measurement | None) -> None:
        lines.append(f"### {title}")
        if row is None:
            lines.append("  (none)")
            lines.append("")
            return
        lines.append(
            f"  {use_case.objective}={row.objective:.2f}  (pp={row.pp_tps:.1f} tg={row.tg_tps:.1f})"
        )
        lines.append(f"  {row.setting.label()}")
        lines.append("  suggested command:")
        lines.append("    " + " ".join(_suggested_command(result, row)))
        lines.append("")

    show("FASTEST (best objective, any context)", result.fastest)
    show(f"BALANCED (best objective at context >= {use_case.ctx_floor})", result.balanced)
    show("MAX CONTEXT (deepest measured)", result.max_context)

    lines.append("### Pareto frontier (context vs objective)")
    if not result.frontier:
        lines.append("  (none)")
    for row in result.frontier:
        lines.append(
            f"  ctx={_ctx(row.setting.values):>7}  {use_case.objective}={row.objective:8.2f}  "
            f"tg={row.tg_tps:7.1f}  {row.setting.label()}"
        )
    lines.append("")

    lines.append("### Main effects (ranked by range; which knobs matter)")
    if not result.effects:
        lines.append("  (none)")
    for effect in result.effects:
        means = ", ".join(f"{m:.2f}" for m in effect.means)
        lines.append(
            f"  {effect.name:<16} range={effect.range:8.3f}  means=[{means}]  "
            f"best={effect.best_level()}"
        )
    lines.append("")
    if result.confirmation is not None:
        predicted, row = result.confirmation
        lines.append("### Confirmation run (predicted-optimal config)")
        lines.append(f"  predicted {use_case.objective}={predicted:.2f}")
        lines.append(f"  measured  {use_case.objective}={row.objective:.2f}  (status={row.status})")
        if predicted > 0:
            error = abs(row.objective - predicted) / predicted * 100.0
            verdict = (
                "small gap: the additive model held"
                if error <= 5.0
                else "large gap: interactions or drift — trust the Pareto pick"
            )
            lines.append(f"  prediction error: {error:.1f}% -> {verdict}")
        lines.append(f"  {row.setting.label()}")
        lines.append("")
    lines.append("Trust the Pareto frontier for the recommendation; use main effects to rank knobs.")
    return "\n".join(lines)


def _suggested_command(result: OptimizerResult, row: Measurement) -> list[str]:
    if result.use_case.driver == "server":
        return ["ninfer-serve", "<artifact>", "--max-concurrency", str(result.use_case.concurrency),
                *row.setting.render_serve()]
    return ["ninfer", "<artifact>", *row.setting.render_bench()]


def to_json(result: OptimizerResult) -> str:
    payload = {
        "fingerprint": result.fingerprint,
        "picks": {
            "fastest": _row_json(result.fastest),
            "balanced": _row_json(result.balanced),
            "max_context": _row_json(result.max_context),
        },
        "pareto": [_row_json(r) for r in result.frontier],
        "main_effects": [
            {"name": e.name, "levels": list(e.levels), "means": list(e.means), "range": e.range}
            for e in result.effects
        ],
        "confirmation": (
            {"predicted": result.confirmation[0], "measured": _row_json(result.confirmation[1])}
            if result.confirmation is not None
            else None
        ),
        "rows": [_row_json(r) for r in result.rows],
    }
    return json.dumps(payload, indent=2)


def _row_json(row: Measurement | None) -> dict | None:
    if row is None:
        return None
    data = asdict(row)
    data["setting"] = dict(row.setting.values)
    data["command"] = list(row.command)
    return data
