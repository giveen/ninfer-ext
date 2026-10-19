#!/usr/bin/env python3
"""Syntax-check NInfer host translation units against the Windows API.

The tree is built and tuned on Linux, and its Windows port cannot be compiled here: the Windows
toolchain is MSVC (a C++23 toolset, for `<flat_map>`), and CI is the only place it runs. This tool
closes the common half of that gap locally by reusing every flag from the Linux compile database and
compiling the same translation unit as `x86_64-w64-mingw32-g++`. It catches what the port actually
breaks -- POSIX functions that do not exist on Windows, Win32 calls with the wrong signature, a
missing include, an `#if defined(_WIN32)` branch that never compiles -- in seconds instead of a CI
round trip.

What it is NOT: MSVC. MinGW provides the Windows API and the C runtime but not MSVC's compiler or
its standard library, so
  * a green result does not prove MSVC accepts the file (CI is the authority), and
  * an MSVC-only spelling (`_MSC_VER` blocks, `#pragma omp`, MSVC intrinsics) cannot be checked here.
It also cannot check CUDA: the CUDA headers are MSVC-shaped, so a unit that includes `cuda_runtime.h`
is reported as skipped rather than as broken. Device code (`.cu`) is out of scope entirely.

Requires `mingw-w64` (Ubuntu: `sudo apt-get install --yes mingw-w64`) and a configured Linux build
directory (`build/compile_commands.json`, which CMakeLists.txt enables).

Usage:
    tools/verify/win32_syntax_check.py                     # files changed on this branch
    tools/verify/win32_syntax_check.py src/artifact/file_io.cpp src/core/arena.cu
    tools/verify/win32_syntax_check.py --base origin/master
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
CXX = "x86_64-w64-mingw32-g++"
# Flags whose argument is the next argv entry, so both have to move together.
TAKES_ARGUMENT = ("-isystem", "-I", "-idirafter", "-imacros", "-include")
# A unit that includes CUDA headers cannot be checked without a Windows CUDA toolkit.
CUDA_MARKERS = ("cuda_runtime.h", "cuda.h", "cuda_fp16.h", "cublas_v2.h", "cuda_bf16.h")


def load_database(path: pathlib.Path) -> list[dict]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        sys.exit(
            f"{path} is missing. Configure a Linux build first "
            "(CMakeLists.txt sets CMAKE_EXPORT_COMPILE_COMMANDS):\n"
            "  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release"
        )


def flags_for(database: list[dict], source: pathlib.Path) -> list[str] | None:
    """The include paths and defines the Linux build uses for this source."""
    suffix = str(source)
    for entry in database:
        if entry["file"].endswith(suffix):
            arguments = entry.get("arguments") or entry["command"].split()
            flags: list[str] = []
            index = 0
            while index < len(arguments):
                argument = arguments[index]
                if argument in TAKES_ARGUMENT:
                    flags.append(f"{argument}{arguments[index + 1]}")
                    index += 2
                    continue
                if argument.startswith(("-I", "-isystem", "-D", "-U")):
                    flags.append(argument)
                index += 1
            # -D_GNU_SOURCE and friends describe the Linux build; mingw defines _WIN32 itself.
            return [flag for flag in flags if not flag.startswith(("-D_GNU_SOURCE", "-D__linux"))]
    return None


def check(source: pathlib.Path, flags: list[str]) -> tuple[str, str]:
    """Returns (status, report) with status in {pass, fail, skip}."""
    command = [CXX, "-std=c++23", "-fsyntax-only"] + flags + [str(source)]
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
    combined = result.stdout + result.stderr
    if result.returncode == 0:
        return "pass", ""
    if any(marker in combined for marker in CUDA_MARKERS):
        return "skip", "needs CUDA headers (MSVC-shaped; not checkable here)"
    diagnostics = [line for line in combined.splitlines() if ": error:" in line or ": fatal error:" in line]
    return "fail", "\n".join(diagnostics[:12] or combined.splitlines()[:12])


def changed_sources(base: str) -> list[pathlib.Path]:
    names: set[str] = set()
    for arguments in (["git", "diff", "--name-only", f"{base}...HEAD"], ["git", "diff", "--name-only", "HEAD"]):
        result = subprocess.run(arguments, cwd=REPO, capture_output=True, text=True, check=True)
        names.update(result.stdout.split())
    return [REPO / name for name in sorted(names)
            if pathlib.Path(name).suffix in (".cpp", ".cu") and (REPO / name).exists()]


def resolve_databases() -> list[pathlib.Path]:
    """Compile databases to borrow flags from, most specific first.

    A directory dedicated to this check is preferred, because it is configured so the tool can see
    files this branch adds without disturbing a working build tree. The regular build directory is
    consulted after it, because a product-only configure carries no entries for the tests.
    """
    candidates = ("build-wincheck/compile_commands.json", "build/compile_commands.json")
    return [REPO / name for name in candidates if (REPO / name).exists()]


def flags_for_any(databases: list[list[dict]], source: pathlib.Path) -> list[str] | None:
    for database in databases:
        flags = flags_for(database, source)
        if flags is not None:
            return flags
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", type=pathlib.Path)
    parser.add_argument("--base", default="origin/master", help="git base for the default file set")
    arguments = parser.parse_args()

    if shutil.which(CXX) is None:
        print(f"{CXX} not found. Install mingw-w64 to check the Windows code paths locally.", file=sys.stderr)
        return 2
    databases = [load_database(path) for path in resolve_databases()]
    if not databases:
        sys.exit("no compile database found; configure a Linux build first (see the docstring)")
    sources = arguments.paths or changed_sources(arguments.base)
    if not sources:
        print("nothing to check", file=sys.stderr)
        return 0

    failures = 0
    for source in sources:
        relative = source.relative_to(REPO) if source.is_absolute() else source
        if source.suffix == ".cu":
            print(f"skip  {relative}: device code is nvcc's business")
            continue
        if source.suffix in (".h", ".cuh"):
            print(f"skip  {relative}: check a translation unit that includes it")
            continue
        flags = flags_for_any(databases, relative)
        if flags is None:
            print(f"skip  {relative}: no compile_commands entry")
            continue
        status, report = check(source, flags)
        print(f"{status:4}  {relative}")
        if status == "fail":
            failures += 1
            print("\n".join(f"        {line}" for line in report.splitlines()))
        elif status == "skip":
            print(f"        {report}")

    print(f"\n{failures} of {len(sources)} checked units failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
