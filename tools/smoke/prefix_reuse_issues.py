"""Exercise the three recent prefix-caching changes against a real .ninfer artifact.

The GPU must be free first: stop the production server on :8080, then run this script.
It polls until the port is free and runs the real-Engine prefix scenarios serially,
plus the CPU resource-manager suite (which covers the cost-scaled search budget and
the shared-catalog reclaim at their proper planner level, without a GPU).

Covered changes:
  1. --preserved-recent-prefixes escape hatch     -> scenario "preserved-recent-prefixes"
  2. Issue #229 cost-scaled materialization budget -> ninfer_resource_manager_test (unit)
  3. Issue #251 shared-catalog saturation reclaim  -> scenario "shared-saturation-reclaim"
     plus "shared-replacement" as the adjacent explicit-candidate capacity path.

Usage:
  python tools/smoke/prefix_reuse_issues.py \
      --artifact E:/NInfer-Deploy-V3-output/qwen3_8_27b_nvfp4-quasar-proposal.ninfer
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BUILD_TESTS = REPO / "build-windows" / "tests" / "Release"
E2E_BINARY = BUILD_TESTS / "ninfer_qwen3_5_prefix_real_test.exe"
UNIT_BINARY = BUILD_TESTS / "ninfer_resource_manager_test.exe"
DEFAULT_ARTIFACT = r"E:\NInfer-Deploy-V3-output\qwen3_8_27b_nvfp4-quasar-proposal.ninfer"


@dataclass
class Issue:
    label: str
    kind: str  # "e2e" or "unit"
    name: str  # scenario name (e2e) or "all" (unit)


ISSUES = [
    Issue("preserved-recent-prefixes escape hatch", "e2e", "preserved-recent-prefixes"),
    Issue("issue #251 shared-catalog saturation reclaim", "e2e", "shared-saturation-reclaim"),
    Issue("issue #251 shared reuse at full capacity", "e2e", "shared-replacement"),
    Issue("issue #229 cost-scaled search budget (+ #251 reclaim) [unit]", "unit", "all"),
]


def port_in_use(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def wait_for_free_port(host: str, port: int, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        if not port_in_use(host, port):
            return
        if time.monotonic() >= deadline:
            raise SystemExit(
                f"port {port} was still in use after {timeout_s:.0f}s — is the production server stopped?"
            )
        print(f"  port {port} in use; waiting for the server to stop…")
        time.sleep(5)


def run_command(cmd: list[str], env: dict[str, str], timeout_s: float) -> tuple[int, str]:
    print(f"\n$ {' '.join(cmd)}")
    started = time.monotonic()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout_s)
    elapsed = time.monotonic() - started
    tail = "\n".join((proc.stdout or "").strip().splitlines()[-40:])
    err = (proc.stderr or "").strip()
    if tail:
        print(tail)
    if err:
        print(f"  [stderr] {err[:2000]}")
    print(f"  -> exit {proc.returncode} in {elapsed:.1f}s")
    return proc.returncode, ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT, help="path to the .ninfer artifact")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080, help="production server port to wait for")
    parser.add_argument("--port-timeout", type=float, default=3600, help="seconds to wait for the port to free")
    parser.add_argument("--e2e-timeout", type=float, default=1800, help="per-scenario GPU test timeout (s)")
    parser.add_argument("--unit-timeout", type=float, default=300)
    parser.add_argument("--skip-port-wait", action="store_true", help="assume the port is already free")
    args = parser.parse_args()

    artifact = args.artifact
    if not Path(artifact).is_file():
        raise SystemExit(f"artifact not found: {artifact}")
    if not E2E_BINARY.is_file():
        raise SystemExit(f"e2e test binary not found (build it first): {E2E_BINARY}")
    if not UNIT_BINARY.is_file():
        raise SystemExit(f"unit test binary not found (build it first): {UNIT_BINARY}")

    if not args.skip_port_wait:
        print(f"Waiting for the production server on {args.host}:{args.port} to stop…")
        wait_for_free_port(args.host, args.port, args.port_timeout)
        print("  port is free.")

    base_env = dict(os.environ)
    base_env["NINFER_TEST_ARTIFACT"] = str(Path(artifact).resolve())

    results: list[tuple[Issue, bool, str]] = []
    for issue in ISSUES:
        if issue.kind == "e2e":
            env = dict(base_env, NINFER_PREFIX_REAL_SCENARIO=issue.name)
            code, _ = run_command([str(E2E_BINARY)], env, args.e2e_timeout)
        else:
            env = dict(os.environ)
            code, _ = run_command([str(UNIT_BINARY)], env, args.unit_timeout)
        results.append((issue, code == 0, f"exit {code}"))

    print("\n=== summary ===")
    failed = 0
    for issue, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"  [{mark}] {issue.label}  ({detail})")
    if failed:
        print(f"\n{failed} issue(s) FAILED.")
        return 1
    print("\nAll three prefix-caching changes verified against the real artifact.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
