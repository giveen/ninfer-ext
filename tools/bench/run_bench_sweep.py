"""Drive the all-models serving sweep: concurrency 1/2/4/8, prefill and decode against context.

Orchestrates `tools/bench/run_serve_concurrency.py` (which owns the server lifecycle, the workload
and the point files) over every local artifact, then hands the campaigns to
`tools/bench/plot_concurrency.py` for the figures. This script adds no measurement of its own.

Two facts about the runner shape everything here:

* `--mode mtp0` is `SPECULATIVE_MODES["mtp0"] == ("none", 0)`, and `block_fixture_names("none")` is
  `NIAH_FIXTURES` -- the four context lengths. So `--suite corpus-makespan --mode mtp0` is the
  context sweep, and one campaign **covers all four contexts**, not one.
* `--max-context` is fixed for a whole campaign, so a campaign can only reach the contexts its
  `--max-context` admits. A 32 GB card cannot hold a 26B/31B beside a 256K KV window, so each model
  is driven at a chain of `--max-context` levels. A fixture too long for the window is recorded as a
  *skip* inside the point (`skipped_fixtures`, and the summary's `Fixtures`/`Skipped` columns) rather
  than aborting the campaign -- unless every fixture is skipped, which is still a failure. What a
  model "can do" is therefore an observation here, not an assumption.

    python3 tools/bench/run_bench_sweep.py --dry-run                 # print every campaign
    python3 tools/bench/run_bench_sweep.py --only gemma4_26b_a4b_qat # one model, for real
    python3 tools/bench/run_bench_sweep.py                           # everything (hours)

Run it with an interpreter that has matplotlib for the plotting step (`/usr/bin/python3`), and with
`.venv/bin/python` for the runner; `--python` selects the latter. A campaign is skipped when its
output directory already holds a summary, so an interrupted sweep resumes rather than repeating
hours of measurement.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Every local artifact worth showing, with the model id the server reports for it.
ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("qwen3_6_27b", "models/qwen3_6_27b.ninfer"),
    ("qwen3_8_27b", "models/qwen3_8_27b.ninfer"),
    ("qwen3_8_27b_nvfp4", "models/qwen3_8_27b_nvfp4.ninfer"),
    ("qwen3_6_35b_a3b", "models/qwen3_6_35b_a3b.ninfer"),
    ("qwen3_6_35b_a3b_nvfp4", "models/qwen3_6_35b_a3b_nvfp4.ninfer"),
    ("gemma4_31b_it_m1", "models/gemma4_31b_it_m1.ninfer"),
    ("gemma4_26b_a4b_qat", "out/gemma4_26b_a4b_qat.ninfer"),
    ("qwen3.8-flash-next", "models/qwen3.8-flash-next/qwen3_8_flash_next_nvfp4.ninfer"),
)

# Models that take `--kv-capacity auto`: an explicit capacity would take their KV headroom out of a
# device expert cache that is on their critical path, which costs far more than the window is worth.
AUTO_CAPACITY: frozenset[str] = frozenset({"qwen3.8-flash-next"})

# Tokens of KV the card holds across all lanes. Every lane reserves a *full* window --
# gemma_engine_options forces kv_capacity = max_context and Program::device_bytes sizes the KV per
# lane -- so a campaign running C lanes gets a window of at most LANE_TOKENS // C, and a model whose
# own limit is lower needs a smaller one still. Attempts walk this ladder downward until the engine
# accepts the configuration. The four NIAH fixtures are used by every campaign; a window below a
# fixture's length makes that request fail, which is recorded.
LANE_TOKENS: int = 262144
WINDOW_LADDER: tuple[int, ...] = tuple(LANE_TOKENS >> shift for shift in range(6))
DEFAULT_CONCURRENCY: tuple[int, ...] = (1, 2, 4, 8)


@dataclass(frozen=True)
class Campaign:
    model: str
    artifact: Path
    context: int
    concurrency: tuple[int, ...]
    modes: tuple[str, ...]
    output: Path
    port: int
    stem: str = ""


def campaigns(args) -> list[Campaign]:
    selected = [pair for pair in ARTIFACTS if not args.only or pair[0] in set(args.only)]
    levels = sorted(set(args.concurrency or DEFAULT_CONCURRENCY))
    modes = tuple(args.mode) if args.mode else ("mtp0",)
    mode_label = "-".join(modes)
    made: list[Campaign] = []
    for model, relative in selected:
        artifact = REPO_ROOT / relative
        if not artifact.is_file():
            print(f"skip {model}: {artifact} is not there", file=sys.stderr)
            continue
        for level in levels:
            base = args.lane_tokens // level
            windows = [window for window in WINDOW_LADDER if window <= base]
            if not windows:
                print(f"skip {model} c{level}: no window fits {args.lane_tokens} lane tokens",
                      file=sys.stderr)
                continue
            # The default suite keeps its historical directory name, so campaigns already completed
            # resume; a decode pass is labelled so the two cannot collide.
            suite_label = "" if args.suite == "corpus-makespan" else f"{args.suite.split('-')[0]}_"
            pad_label = f"pad{args.pad_tokens}_" if args.pad_tokens else ""
            stem = f"{model}_{mode_label}_{suite_label}{pad_label}c{level}"
            made.append(
                Campaign(
                    model=model,
                    artifact=artifact,
                    context=windows[0],
                    concurrency=(level,),
                    modes=modes,
                    output=args.output / f"{stem}_ctx{windows[0]}",
                    port=args.port + len(made),
                    stem=stem,
                )
            )
    return made


def command(campaign: Campaign, args) -> list[str]:
    return [
        args.python,
        "tools/bench/run_serve_concurrency.py",
        "--serve",
        str(REPO_ROOT / "build/apps/ninfer-serve"),
        "--artifact",
        f"{campaign.model}={campaign.artifact}",
        *sum((["--mode", mode] for mode in campaign.modes), []),
        "--sampling",
        args.sampling,
        "--suite",
        args.suite,
        *(["--block-fixtures", args.block_fixtures] if args.block_fixtures != "auto" else []),
        *(["--decode-tokens", str(args.decode_tokens)] if args.suite == "decode-saturation" else []),
        *(["--pad-tokens", str(args.pad_tokens)]
          if args.suite == "decode-saturation" and args.pad_tokens else []),
        "--kv-dtype",
        args.kv_dtype,
        # Gemma forces an explicit capacity of max_context and reports it as explicit; `auto` makes
        # the runner's engine-block check disagree with it, so those campaigns state it. A model in
        # AUTO_CAPACITY wants `auto` instead; see the constant.
        "--kv-capacity",
        "auto" if campaign.model in AUTO_CAPACITY else str(campaign.context),
        "--max-context",
        str(campaign.context),
        "--prefill-chunk",
        str(args.prefill_chunk),
        "--port",
        str(campaign.port),
        "--output",
        str(campaign.output),
        *sum((["--concurrency", str(level)] for level in campaign.concurrency), []),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "profiles/bench/all_models")
    parser.add_argument("--only", action="append", help="artifact name to drive (repeatable)")
    parser.add_argument("--lane-tokens", type=int, default=LANE_TOKENS,
                        help="KV tokens the card holds across all lanes (default 262144); a "
                             "campaign's window is this divided by its lane count")
    parser.add_argument("--concurrency", type=int, action="append",
                        help="concurrency levels (default 1 2 4 8)")
    parser.add_argument("--kv-dtype", default="bf16")
    parser.add_argument("--sampling", default="greedy", choices=("greedy", "stochastic"))
    parser.add_argument("--mode", action="append",
                        choices=("mtp0", "mtp3", "mtp5", "mtp_adaptive", "mtp3_full", "mtp5_full"),
                        help="speculative mode(s) (default mtp0, the plain path); the *_full modes "
                             "skip the optimized proposal head, which Gemma 4 refuses")
    parser.add_argument("--prefill-chunk", type=int, default=1024)
    parser.add_argument("--suite", default="corpus-makespan",
                        choices=("corpus-makespan", "decode-saturation"),
                        help="corpus-makespan measures prefill against context; decode-saturation "
                             "measures steady decode at the saturation fixture's fixed context")
    parser.add_argument("--decode-tokens", type=int, default=1024,
                        help="per-request output budget for decode-saturation (default 1024)")
    parser.add_argument("--pad-tokens", type=int, default=0,
                        help="decode-saturation only: prefix the prompt with this many tokens of "
                             "in-distribution prose to move the measured context (default 0)")
    parser.add_argument("--block-fixtures", default="auto", choices=("auto", "niah", "decode"),
                        help="override the corpus prompt set; 'decode' is what makes a plain-vs-MTP "
                             "A/B comparable, since the per-backend default splits the suites")
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--python", default=str(REPO_ROOT / ".venv/bin/python"))
    parser.add_argument("--plot-python", default="python3",
                        help="interpreter with matplotlib for the figures")
    parser.add_argument("--dry-run", action="store_true", help="print the campaigns, run none")
    args = parser.parse_args()
    if not args.concurrency:
        args.concurrency = list(DEFAULT_CONCURRENCY)

    made = campaigns(args)
    if not made:
        print("no campaigns selected", file=sys.stderr)
        return 1

    failures: list[str] = []
    succeeded: list[Campaign] = []
    for campaign in made:
        levels = campaign.concurrency[0]
        ladder = [window for window in WINDOW_LADDER if window <= args.lane_tokens // levels]
        for window in ladder:
            attempt = Campaign(model=campaign.model, artifact=campaign.artifact, context=window,
                               concurrency=campaign.concurrency, modes=campaign.modes,
                               output=campaign.output.parent / f"{campaign.stem}_ctx{window}",
                               port=campaign.port, stem=campaign.stem)
            label = f"{attempt.model} ctx{window} c{levels}"
            if (attempt.output / "summary.json").is_file():
                print(f"have   {label}")
                succeeded.append(attempt)
                break
            print(f"run    {label}")
            if args.dry_run:
                print("       " + " ".join(command(attempt, args)))
                succeeded.append(attempt)
                break
            returncode = subprocess.run(command(attempt, args), cwd=REPO_ROOT).returncode
            if returncode == 0 or (attempt.output / "points").is_dir():
                # A non-zero exit with points on disk is a run that happened and had a request
                # refused, not a configuration the card cannot hold. Stepping the window down would
                # re-measure something else, and did: six times per model before this check existed.
                succeeded.append(attempt)
                break
            # A window the card cannot hold, or one above the model's own limit, is not a failure of
            # the sweep: the next rung down is tried and only exhausting the ladder is recorded.
            print(f"       did not fit at ctx{window}; trying a smaller window", file=sys.stderr)
        else:
            failures.append(f"{campaign.model} c{levels}")

    if args.dry_run:
        return 0

    by_model: dict[str, list[Path]] = {}
    for campaign in succeeded:
        by_model.setdefault(campaign.model, []).append(campaign.output)
    figures = args.output / "figures"
    for model, runs in sorted(by_model.items()):
        existing = [run for run in runs if (run / "points").is_dir()]
        if not existing:
            continue
        argv = [args.plot_python, "tools/bench/plot_concurrency.py", "--out", str(figures)]
        for run in existing:
            argv += ["--run", str(run)]
        print(f"plot   {model} over {len(existing)} campaign(s)")
        subprocess.run(argv, cwd=REPO_ROOT)
        # The master chart cuts the same readings both ways: rate against context and against
        # concurrency, prefill and decode.
        master = [args.plot_python, "tools/bench/plot_master.py", "--out", str(figures)]
        for run in existing:
            master += ["--run", str(run)]
        subprocess.run(master, cwd=REPO_ROOT)

    print()
    if failures:
        print("campaigns that did not fit or did not finish:")
        for name in failures:
            print(f"  {name}")
    print(f"figures in {figures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
