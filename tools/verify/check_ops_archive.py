#!/usr/bin/env python3
"""Gate a built CUDA/CMake tree before trusting a link or a measurement read from it.

Written after a day lost to concurrent ninjas on one build directory, where the visible symptom was
a *rotating* set of undefined symbols:

    ld.bfd: src/ops/libninfer_ops.a(cmake_device_link.o): undefined reference to
            `__fatbinwrap_a0b3a7da_9_launch_cu_aaa1fd9c_3304101'

`-rdc=true` archives carry an object (`cmake_device_link.o`) that references one `__fatbinwrap_*`
per device source, and the fatbins behind those references live in the archive's own `.cu.o`
members. Three different states produce that error and only one of them is a broken archive:

1. **A member is absent from disk.** CMake archives with `cmake -E rm -f <archive>` followed by
   `ar qc <archive> <objects>`, so one missing object leaves *no* archive at all -- and a consumer
   that then links an older archive sees rotating undefined symbols instead of the `ar` error.
   Gate 1 catches this, and it is the state that cost the most time: six `.cu.o` files of one
   target deleted by a killed or raced build, all of them declared in `build.ninja`.
2. **The archive is inconsistent with its own device-link member** -- referenced fatbinwraps that no
   member defines. Gate 3 catches it.
3. **A reader raced a writer.** Two ninjas on one build directory both re-archive; a link that runs
   between them reads a partial archive. Nothing in the products shows this afterwards, which is why
   the gates must be run with no other build in flight -- and why gate 4 exists: a linked binary
   that runs is the only end-to-end proof that the link read a complete archive.

The mtime ordering of `cmake_device_link.o` against every `.cu.o` (gate 2) is cheap and worth
keeping, but it passes on both a mid-compile tree and an inconsistent archive, so it is never
sufficient on its own.

Usage:
    python3 tools/verify/check_ops_archive.py                  # gates the default ops target
    python3 tools/verify/check_ops_archive.py --binary build/apps/ninfer-serve
    python3 tools/verify/check_ops_archive.py --quiet          # only failures and the summary

Exit status is 0 only when every gate passes.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUILD_DIR = REPO_ROOT / "build"
DEFAULT_ARCHIVE = Path("src/ops/libninfer_ops.a")
DEFAULT_OBJECT_DIR = Path("src/ops/CMakeFiles/ninfer_ops.dir")
# One `build <output>: <rule> <inputs>` line per object in a CMake ninja file.
NINJA_BUILD_LINE = re.compile(r"^build (?P<outputs>[^:]+): ")


class GateError(RuntimeError):
    """A gate could not be evaluated (as opposed to having failed)."""


def fatbinwrap_symbols(path: Path, defined: bool) -> set[str]:
    """The `__fatbinwrap_*` symbols one object or archive defines, or references."""
    try:
        completed = subprocess.run(
            ["nm", "--defined-only" if defined else "--undefined-only", str(path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise GateError(f"nm failed on {path}: {exc}") from exc
    return {
        fields[-1]
        for line in completed.stdout.splitlines()
        if "__fatbinwrap_" in line and (fields := line.split())
    }


def declared_objects(build_ninja: Path, prefix: str) -> list[str]:
    """Every object path `build.ninja` declares for the target whose objects live under `prefix`."""
    if not build_ninja.is_file():
        raise GateError(f"no ninja file at {build_ninja}")
    declared: list[str] = []
    try:
        with build_ninja.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                match = NINJA_BUILD_LINE.match(line)
                if match is None:
                    continue
                for output in match.group("outputs").split():
                    if output.startswith(prefix) and output.endswith((".o", ".obj")):
                        declared.append(output)
    except OSError as exc:
        raise GateError(f"cannot read {build_ninja}: {exc}") from exc
    return declared


def archive_facts(archive: Path) -> dict[str, Any]:
    """Member count, identity and fatbinwrap completeness of one archive, as data.

    The campaign records this beside a measurement, so the claim "the binary was linked from a
    complete archive" is checkable after the fact; the gates in `main()` assert on the same values.
    """
    facts: dict[str, Any] = {"path": str(archive)}
    if not archive.is_file():
        facts["exists"] = False
        return facts
    facts["exists"] = True
    stat = archive.stat()
    facts["bytes"] = stat.st_size
    facts["mtime"] = datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
    try:
        listed = subprocess.run(
            ["ar", "t", str(archive)], capture_output=True, text=True, check=True, timeout=600
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise GateError(f"ar t failed on {archive}: {exc}") from exc
    facts["members"] = len(listed.stdout.split())
    # The archive's own device-link member references every fatbin the archive must carry.
    with tempfile.TemporaryDirectory() as directory:
        device_link = Path(directory) / "cmake_device_link.o"
        try:
            with device_link.open("wb") as handle:
                subprocess.run(
                    ["ar", "p", str(archive), "cmake_device_link.o"],
                    stdout=handle,
                    check=True,
                    timeout=600,
                )
        except (OSError, subprocess.SubprocessError) as exc:
            raise GateError(f"cannot extract cmake_device_link.o from {archive}: {exc}") from exc
        referenced = fatbinwrap_symbols(device_link, defined=False)
    defined = fatbinwrap_symbols(archive, defined=True)
    facts["fatbinwraps_referenced"] = len(referenced)
    facts["fatbinwraps_defined"] = len(defined)
    facts["fatbinwraps_missing"] = len(referenced - defined)
    facts["fatbinwraps_missing_sample"] = sorted(referenced - defined)[:5]
    return facts


def older_than(objects: Sequence[Path], reference: Path) -> list[Path]:
    """The objects whose mtime is newer than `reference`."""
    stamp = reference.stat().st_mtime
    return [path for path in objects if path.stat().st_mtime > stamp]


def run_gates(args: argparse.Namespace, report: list[str]) -> list[str]:
    """Evaluate every gate; return the failures, appending human-readable lines to `report`."""
    failures: list[str] = []
    build_dir = args.build_dir.expanduser().resolve()
    archive = build_dir / args.archive
    object_dir = build_dir / args.object_dir

    # Gate 1: every object the graph declares exists on disk. A missing one makes `ar` delete the
    # archive outright, so this is the state a consumer sees as rotating undefined symbols.
    declared = declared_objects(build_dir / "build.ninja", str(args.object_dir))
    if not declared:
        failures.append(
            f"no objects declared for {args.object_dir} in {build_dir / 'build.ninja'}; "
            "check --object-dir"
        )
        report.append("GATE 1 declared objects      FAIL  (none declared; wrong --object-dir?)")
    else:
        missing = [path for path in declared if not (build_dir / path).is_file()]
        report.append(
            f"GATE 1 declared objects      {'FAIL' if missing else 'pass'}  "
            f"{len(declared) - len(missing)}/{len(declared)} present"
            + (f"; missing: {', '.join(Path(name).name for name in missing[:6])}"
               + (f" (+{len(missing) - 6} more)" if len(missing) > 6 else "") if missing else "")
        )
        if missing:
            failures.append(f"{len(missing)} declared object(s) absent from disk, e.g. {missing[0]}")

    # Gate 2: the device-link object is newer than every device object (cheap, necessary, not
    # sufficient — it passes on a mid-compile tree and on an inconsistent archive).
    device_link = object_dir / "cmake_device_link.o"
    cu_objects = sorted(object_dir.rglob("*.cu.o"))
    if not device_link.is_file():
        report.append("GATE 2 device-link ordering  FAIL  (no cmake_device_link.o)")
        failures.append(f"{device_link} is absent")
    elif not cu_objects:
        report.append("GATE 2 device-link ordering  FAIL  (no .cu.o objects under the target)")
        failures.append(f"no .cu.o objects under {object_dir}")
    else:
        newer = older_than(cu_objects, device_link)
        report.append(
            f"GATE 2 device-link ordering  {'FAIL' if newer else 'pass'}  "
            f"{len(cu_objects) - len(newer)}/{len(cu_objects)} .cu.o older than "
            f"cmake_device_link.o"
        )
        if newer:
            failures.append(f"{len(newer)} .cu.o file(s) newer than {device_link.name}")

    # Gate 3: the archive carries every fatbin its own device-link member references.
    try:
        facts = archive_facts(archive)
    except GateError as exc:
        report.append(f"GATE 3 archive completeness FAIL  ({exc})")
        failures.append(str(exc))
        return failures
    if not facts.get("exists"):
        report.append("GATE 3 archive completeness FAIL  (archive absent; ar deleted it?)")
        failures.append(f"{archive} is absent")
        return failures
    missing = int(facts["fatbinwraps_missing"])
    report.append(
        f"GATE 3 archive completeness {'FAIL' if missing else 'pass'}  "
        f"{facts['members']} members, fatbinwraps referenced/defined/missing "
        f"{facts['fatbinwraps_referenced']}/{facts['fatbinwraps_defined']}/{missing}, "
        f"{facts['bytes']} B, {facts['mtime']}"
    )
    if missing:
        failures.append(
            f"{missing} fatbinwrap(s) referenced by {archive.name} are undefined in it, "
            f"e.g. {facts['fatbinwraps_missing_sample']}"
        )

    # Gate 4: the linked binary runs. This is the only end-to-end proof that a link read a complete
    # archive, and it needs no GPU: --help is the cheapest command every app answers.
    if args.binary is None:
        report.append("GATE 4 binary smoke          skipped  (no --binary)")
        return failures
    binary = args.binary.expanduser().resolve()
    if not binary.is_file():
        report.append(f"GATE 4 binary smoke          FAIL  ({binary} is absent)")
        failures.append(f"{binary} is absent")
        return failures
    try:
        completed = subprocess.run(
            [str(binary), "--help"], capture_output=True, text=True, check=False, timeout=300
        )
    except (OSError, subprocess.SubprocessError) as exc:
        report.append(f"GATE 4 binary smoke          FAIL  ({exc})")
        failures.append(f"{binary} --help failed to run: {exc}")
        return failures
    report.append(
        f"GATE 4 binary smoke          {'pass' if completed.returncode == 0 else 'FAIL'}  "
        f"{binary.name} --help exited {completed.returncode}"
    )
    if completed.returncode != 0:
        failures.append(f"{binary} --help exited {completed.returncode}")
    return failures


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-dir", type=Path, default=DEFAULT_BUILD_DIR, help="CMake build directory"
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help="archive to gate, relative to --build-dir (default: the ops library)",
    )
    parser.add_argument(
        "--object-dir",
        type=Path,
        default=DEFAULT_OBJECT_DIR,
        help="the target's object directory, relative to --build-dir",
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=None,
        help="linked executable to smoke-run with --help (e.g. build/apps/ninfer-serve)",
    )
    parser.add_argument("--json", action="store_true", help="also print the facts as JSON")
    parser.add_argument("--quiet", action="store_true", help="print only failures and the summary")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report: list[str] = []
    failures = run_gates(args, report)
    if not args.quiet:
        for line in report:
            print(line)
    else:
        for line in report:
            if "FAIL" not in line:
                continue
            print(line)
    if args.json:
        try:
            print(
                json.dumps(
                    archive_facts(args.build_dir.expanduser().resolve() / args.archive),
                    indent=2,
                )
            )
        except GateError as exc:
            print(f'{{"error": "{exc}"}}')
    if failures:
        print(f"\nFAIL: {len(failures)} gate failure(s)")
        for failure in failures:
            print(f"  - {failure}")
        print(
            "  A failure with the build directory to yourself is new information about the build; "
            "with another build in flight it is contention, so re-run with no other writer."
        )
        return 1
    print("\nPASS: every gate holds")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
