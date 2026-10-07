"""ninfer-optimizer: find the best ninfer configuration for how you actually run the model.

Point it at a `.ninfer` artifact and a use case; it detects the machine and model, designs a small
balanced experiment over ninfer's knobs, runs it, and prints the fastest / balanced / max-context
configurations plus the speed-vs-context Pareto frontier. Designed experiments (Morris screening ->
orthogonal arrays) instead of a brute-force sweep.

    python3 -m tools.optimizer MODEL.ninfer                    # plan only, no GPU
    python3 -m tools.optimizer MODEL.ninfer --use-case agents --run
    python3 -m tools.optimizer MODEL.ninfer --run --screen --levels 3 --min-kv fp8
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import random
import sys
import time

from tools.artifact.reader import ArtifactError

from .design import (
    additive_prediction,
    elementary_effects,
    main_effects,
    morris_trajectories,
    orthogonal_design,
    predicted_optimal,
    significant_factors,
)
from .detect import detect_hardware, detect_model
from .driver import DEFAULT_BENCH, DEFAULT_SERVE, Measurement, ThermalController, driver_for
from .html import render_html
from .profile import build_profile, write_profile
from .factors import Factor, FactorOptions, Setting, build_factors
from .report import (
    TOOL_VERSION,
    build_result,
    format_plan,
    format_result,
    to_json,
)
from .usecases import USE_CASES, use_case


def _parse_factor_overrides(values: list[str]) -> dict[str, tuple[str, ...]]:
    overrides: dict[str, tuple[str, ...]] = {}
    for item in values:
        name, _, levels = item.partition("=")
        name = name.strip()
        parsed = tuple(level.strip() for level in levels.split(",") if level.strip())
        if not name or len(parsed) < 1:
            raise argparse.ArgumentTypeError(f"--factor must be NAME=LEVEL[,LEVEL...], got {item!r}")
        overrides[name] = parsed
    return overrides


def _write_csv(path: Path, rows: list[Measurement]) -> None:
    fields = ["status", "objective", "pp_tps", "tg_tps", "seconds", "setting", "detail"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "status": row.status,
                    "objective": f"{row.objective:.4f}",
                    "pp_tps": f"{row.pp_tps:.3f}",
                    "tg_tps": f"{row.tg_tps:.3f}",
                    "seconds": f"{row.seconds:.1f}",
                    "setting": row.setting.label(),
                    "detail": row.detail,
                }
            )


def _grid_settings(factors: list[Factor], levels, point, pinned: dict[str, str]) -> Setting:
    values = dict(pinned)
    for index, factor in enumerate(factors):
        values[factor.name] = levels[index][point[index]]
    return Setting(values)


def _screen(
    driver, use_case, factors: list[Factor], pinned: dict[str, str], trajectories: int,
    reps: int, rng: random.Random, log,
):
    """Morris screening; returns the surviving factors and the best-seen pin for the rest."""
    levels = 3
    count = len(factors)
    plan = morris_trajectories(count, levels, trajectories, rng)
    selected = tuple(
        tuple(_even_levels(factor, levels)) for factor in factors
    )
    scores: list[float] = []
    for trajectory in plan:
        for point in trajectory.points:
            setting = _grid_settings(factors, selected, point, pinned)
            row = driver.measure(setting, use_case, reps)
            log(f"  screen {setting.label()} -> {row.status} objective={row.objective:.2f}")
            scores.append(row.objective)
    # Score per (factor, level) for pinning the dropped knobs at their best seen value.
    by_level: dict[tuple[str, str], list[float]] = {}
    cursor = 0
    for trajectory in plan:
        for point in trajectory.points:
            for index, factor in enumerate(factors):
                by_level.setdefault((factor.name, selected[index][point[index]]), []).append(
                    scores[cursor]
                )
            cursor += 1
    effects = elementary_effects([f.name for f in factors], plan, scores)
    survivors = significant_factors(effects)
    for factor in factors:
        if factor.name in survivors:
            continue
        candidates = {
            level: sum(vals) / len(vals)
            for (name, level), vals in by_level.items()
            if name == factor.name and vals
        }
        if candidates:
            pinned[factor.name] = max(candidates, key=candidates.get)
    log("  screening: " + ", ".join(f"{e.name}(mu*={e.mu_star:.2f})" for e in effects))
    log("  survivors: " + ", ".join(survivors))
    return [f for f in factors if f.name in survivors]


def _even_levels(factor: Factor, count: int) -> tuple[str, ...]:
    from .design import select_levels

    return select_levels(factor, count)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ninfer-optimizer", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("artifact", nargs="?", type=Path, help="a .ninfer artifact")
    parser.add_argument("--use-case", default="chat", choices=sorted(USE_CASES))
    parser.add_argument("--objective", default=None, choices=["tg", "pp", "eff", "ttft"],
                        help="override the runbook's objective")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="override the runbook's concurrency (server-driven use cases)")
    parser.add_argument("--list-use-cases", action="store_true")
    parser.add_argument("--run", action="store_true", help="execute the sweep (uses the GPU)")
    parser.add_argument("--screen", nargs="?", type=int, const=3, default=None,
                        metavar="R", help="Morris screening trajectories before the array")
    parser.add_argument("--levels", type=int, default=None, help="shared level count (prime)")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--quick", action="store_true", help="one repetition per config")
    parser.add_argument("--full", action="store_true", help="five repetitions per config")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=900.0, help="per-run seconds")
    parser.add_argument("--thermal-mode", default="off", choices=["off", "warm", "idle"],
                        help="GPU thermal state before each measurement (default off)")
    parser.add_argument("--thermal-cap", type=float, default=180.0,
                        help="maximum seconds to settle the GPU")
    parser.add_argument("--confirm", action="store_true",
                        help="re-measure the predicted-optimal config (implied by --full)")
    parser.add_argument("--min-kv", default="any",
                        choices=["any", "k8v4", "nvfp4", "fp8", "int8", "bf16"])
    parser.add_argument("--factor", action="append", default=[], metavar="NAME=LEVELS",
                        help="override a factor's levels, repeatable")
    parser.add_argument("--ctx-size", type=int, default=None, help="pin the context depth")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                        help="pin an option in every configuration, repeatable "
                             "(e.g. --set kv_capacity=524288 --set expert_cache=auto)")
    parser.add_argument("--output-dir", type=Path, default=Path("profiles/optimizer"))
    parser.add_argument("--html", type=Path, default=None, help="write a self-contained HTML report")
    parser.add_argument("--save-profile", type=Path, default=None,
                        help="write the chosen pick as a ninfer-serve --profile document")
    parser.add_argument("--save-pick", default="balanced",
                        choices=["fastest", "balanced", "max_context"],
                        help="which pick to save (default balanced)")
    parser.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--serve", type=Path, default=DEFAULT_SERVE)
    parser.add_argument("--json", action="store_true", help="emit machine-readable results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--version", action="version", version=f"ninfer-optimizer {TOOL_VERSION}")
    args = parser.parse_args(argv)

    if args.list_use_cases:
        for name, uc in sorted(USE_CASES.items()):
            print(f"  {name:<12} {uc.summary}")
        return 0
    if args.artifact is None:
        parser.error("an artifact path is required")

    case = use_case(args.use_case)
    if args.objective is not None:
        case = replace(case, objective=args.objective)
    if args.concurrency is not None:
        case = replace(case, concurrency=args.concurrency)
    hardware = detect_hardware()
    try:
        model = detect_model(args.artifact)
    except (ArtifactError, OSError, ValueError) as error:
        parser.error(f"cannot read artifact: {error}")

    overrides = _parse_factor_overrides(args.factor)
    factors = build_factors(case, model, hardware, FactorOptions(min_kv=args.min_kv))
    # Explicit --factor overrides replace a runbook factor's levels, or add a new factor.
    known = {f.name for f in factors}
    factors = [Factor(f.name, overrides.get(f.name, f.levels), f.help) for f in factors]
    for name, levels in overrides.items():
        if name not in known:
            factors.append(Factor(name, levels))

    pinned: dict[str, str] = {}
    for item in args.set:
        name, _, value = item.partition("=")
        name, value = name.strip(), value.strip()
        if not name or not value:
            parser.error(f"--set must be NAME=VALUE, got {item!r}")
        try:
            Setting({name: value}).render_serve()  # validates the option name
        except ValueError as error:
            parser.error(str(error))
        pinned[name] = value
    if args.ctx_size is not None:
        pinned["ctx"] = str(args.ctx_size)
    factors = [f for f in factors if f.name not in pinned]

    if not factors:
        parser.error("no factors to sweep; the runbook and overrides left nothing to vary")

    reps = 1 if args.quick else 5 if args.full else args.reps
    rng = random.Random(args.seed)
    log = (lambda _m: None) if args.json else (lambda m: print(m, flush=True))

    try:
        design = orthogonal_design(factors, args.levels)
    except ValueError as error:
        parser.error(str(error))

    sample = next(design.settings())
    if pinned:
        sample = Setting({**sample.values, **pinned})
    driver = driver_for(args.artifact, case, bench=args.bench, serve=args.serve,
                        warmup=args.warmup, timeout=args.timeout,
                        run_dir=args.output_dir,
                        thermal=ThermalController(args.thermal_mode, cap_seconds=args.thermal_cap))
    sample_command = (
        driver.command(sample, case, 0)  # server: a free port is chosen at run time
        if case.driver == "server"
        else driver.command(sample, case, reps) + ["--output-file", "<report.json>"]
    )

    if not args.run:
        print(format_plan(args.artifact, case, hardware, model, factors, design.run_count,
                          design.p, sample_command, None))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"ninfer-optimizer {TOOL_VERSION}: sweeping {case.name} on {hardware.gpu_name}")
    started = time.monotonic()
    rows: list[Measurement] = []

    survivors = factors
    if args.screen:
        survivors = _screen(driver, case, factors, pinned, args.screen, reps, rng, log)
        if len(survivors) < 1:
            survivors = factors
        design = orthogonal_design(survivors, args.levels)

    csv_path = args.output_dir / f"results_{case.name}.csv"
    items: list[tuple[tuple[int, ...], Setting]] = []
    for run, setting in zip(design.runs, design.settings()):
        if pinned:
            setting = Setting({**setting.values, **pinned})
        items.append((run, setting))
    rng.shuffle(items)  # randomized order fights thermal drift across the sweep
    measured: list[tuple[tuple[int, ...], Measurement]] = []
    for index, (run, setting) in enumerate(items, 1):
        row = driver.measure(setting, case, reps)
        measured.append((run, row))
        rows.append(row)
        log(f"[{index}/{design.run_count}] {setting.label()} -> {row.status} "
            f"objective={row.objective:.2f} ({row.seconds:.0f}s)")
        _write_csv(csv_path, rows)

    scores = {run: row.objective for run, row in measured}
    effects = main_effects(design, scores)
    grand_mean = sum(scores.values()) / len(scores) if scores else 0.0

    confirmation = None
    if args.confirm or args.full:
        predicted_values = predicted_optimal(effects)
        predicted_setting = Setting({**predicted_values, **pinned})
        predicted_score = additive_prediction(effects, grand_mean)
        log(f"confirmation: measuring predicted-optimal {predicted_setting.label()}")
        row = driver.measure(predicted_setting, case, reps)
        rows.append(row)
        _write_csv(csv_path, rows)
        confirmation = (predicted_score, row)

    result = build_result(case, rows, effects, hardware, model, survivors, reps, confirmation)
    elapsed = time.monotonic() - started

    print()
    print(format_result(result))
    print(f"\nelapsed {elapsed / 60.0:.1f} min; rows written to {csv_path}")
    (args.output_dir / f"results_{case.name}.json").write_text(to_json(result), encoding="utf-8")
    (args.output_dir / f"fingerprint_{case.name}.json").write_text(
        json.dumps(result.fingerprint, indent=2), encoding="utf-8"
    )
    if args.html is not None:
        args.html.parent.mkdir(parents=True, exist_ok=True)
        args.html.write_text(render_html(result), encoding="utf-8")
        print(f"HTML report written to {args.html}")
    if args.save_profile is not None:
        try:
            document = build_profile(result, args.save_pick)
        except ValueError as error:
            parser.error(str(error))
        write_profile(args.save_profile, document)
        print(f"profile written to {args.save_profile} (pick {args.save_pick})")
    if args.json:
        print(to_json(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
