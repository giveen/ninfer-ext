"""Exercise the recent prefix-caching changes against a real .ninfer artifact.

The GPU must be free first: stop the production server on :8080, then run this script.
It polls until the port is free and runs the real-Engine prefix scenarios serially,
plus the CPU resource-manager suite (which covers the cost-scaled search budget and
the shared-catalog reclaim at their proper planner level, without a GPU).

Covered changes:
  1. recency-ordered prefix retention           -> scenario "recency-retention"
  2. Issue #229 cost-scaled materialization budget -> ninfer_resource_manager_test (unit)
  3. Issue #251 shared-catalog saturation reclaim  -> scenario "shared-saturation-reclaim"
     plus "shared-replacement" as the adjacent explicit-candidate capacity path.
  4. Stats-publication ordering (no leaked references after a released lane) is asserted by
     every e2e scenario, including the five realistic agent scenarios below.

The "agent-*" scenarios mirror the production agent traffic shape (E:/NInfer-Deploy-V3/log.json):
a shared system + tools prefix, a growing multi-turn conversation with tool calls, and thinking
on and preserved. They drive the 36h caching fixes into contention -- a full shared catalog,
saturated private continuations, KV-capacity pressure, and concurrent requests -- and assert the
observable guarantees (shared-prefix reuse, pressure evictions, and zero leaked references).

The test binaries are resolved from <build-dir>/tests/<config>. When --build-config is
omitted, the config whose e2e binary was built most recently (by mtime) is used, and the
script warns if that binary predates the newest commit touching the watched cache sources
(so a stale Release build that predates a recent fix is flagged instead of silently run).

Usage:
  python tools/smoke/prefix_reuse_issues.py \
      --artifact E:/NInfer-Deploy-V3-output/qwen3_8_27b_nvfp4-quasar-proposal.ninfer \
      [--build-dir E:/NInfer-V3/build-windows] [--build-config Debug]
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_BUILD_DIR = REPO / "build-windows"
E2E_BINARY_NAME = "ninfer_qwen3_5_prefix_real_test.exe"
UNIT_BINARY_NAME = "ninfer_resource_manager_test.exe"
DEFAULT_ARTIFACT = r"E:\NInfer-Deploy-V3-output\qwen3_8_27b_nvfp4-quasar-proposal.ninfer"

# Source files whose newest change gates whether a test binary is stale: a binary built
# before the latest edit to any of these will not include that change (e.g. the crash fix
# bb76238b and the issue #251 reclaim fix).
STALENESS_WATCHED_SOURCES = [
    "src/runtime/engine/context_cache/resource_manager.h",
    "src/models/qwen3_5/program/transactions/commit.cpp",
    "src/models/qwen3_5/program/transactions/capture.cpp",
    "src/models/qwen3_5/program/planning/graph_profiles.cpp",
]

DEFAULT_LOG = REPO / ".qwen" / "prefix_smoke.log"


class _Tee:
    """Write to the original stdout and a durable log file simultaneously.

    The file is flushed on every write so the log is always current and readable even
    mid-run, independent of how the process is observed (monitor, shell, nohup).
    """

    def __init__(self, stream, path: Path):
        self._stream = stream
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "a", encoding="utf-8")

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def close(self):
        try:
            self._file.close()
        except OSError:
            pass


def newest_source_commit_unix_time() -> int | None:
    """Return the unix time of the newest commit touching the watched sources, or None."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "log", "-1", "--format=%ct", *STALENESS_WATCHED_SOURCES],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if out.returncode != 0 or not (out.stdout or "").strip():
            return None
        return int(out.stdout.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def resolve_test_binaries(
    build_dir: Path, build_config: str | None
) -> tuple[Path, Path, str]:
    """Resolve the e2e + unit test binaries for a build config.

    If build_config is None, auto-detect the config whose e2e binary was built most
    recently (by mtime), so a stale config (e.g. Release built before a fix) is not
    silently used when a newer config (e.g. Debug) exists. Returns (e2e, unit, config).
    """
    tests_dir = build_dir / "tests"
    if not tests_dir.is_dir():
        raise SystemExit(f"tests dir not found (build first): {tests_dir}")
    if build_config is None:
        candidates: list[tuple[float, str]] = []
        for config_dir in tests_dir.iterdir():
            exe = config_dir / E2E_BINARY_NAME
            if exe.is_file():
                candidates.append((exe.stat().st_mtime, config_dir.name))
        if not candidates:
            raise SystemExit(f"no e2e test binary found under {tests_dir} (build it first)")
        candidates.sort(key=lambda pair: pair[0])
        build_config = candidates[-1][1]
    config_dir = tests_dir / build_config
    return config_dir / E2E_BINARY_NAME, config_dir / UNIT_BINARY_NAME, build_config


@dataclass
class Issue:
    label: str
    kind: str  # "e2e" or "unit"
    name: str  # scenario name (e2e) or "all" (unit)


ISSUES = [
    Issue("recency-ordered prefix retention", "e2e", "recency-retention"),
    Issue("issue #251 shared-catalog saturation reclaim", "e2e", "shared-saturation-reclaim"),
    Issue("issue #251 shared reuse at full capacity", "e2e", "shared-replacement"),
    # Realistic agent scenarios (system + tools shared prefix, growing multi-turn conversation
    # with tool calls, thinking on and preserved -- the shape in E:\NInfer-Deploy-V3\log.json)
    # that drive the 36h caching fixes into contention (full shared catalog, saturated private
    # continuations, KV-capacity pressure, concurrent requests) and assert the observable
    # guarantees: shared-prefix reuse, pressure evictions, and no leaked references (d9d110e3).
    Issue("agent: shared system+tools prefix across multi-turn tool calls", "e2e", "agent-multi-turn"),
    Issue("agent: private-continuation saturation preserves the most recent", "e2e", "agent-private-continuations"),
    Issue("agent: two concurrent requests settle without leaked references", "e2e", "agent-concurrent"),
    Issue("agent: long conversations overflow the KV capacity", "e2e", "agent-kv-pressure"),
    Issue("agent: automatic prefixes reclaim a saturated shared catalog (#251)", "e2e", "agent-shared-catalog"),
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


def gpu_vram_used_mb() -> int | None:
    """Return the GPU VRAM usage in MB, or None if nvidia-smi is unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode != 0:
            return None
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def wait_for_vram_release(timeout_s: float, threshold_mb: int = 2048) -> None:
    """Wait until GPU VRAM drops below threshold_mb (the server released its model).

    The port freeing is not enough: the server process can linger and still hold the 27B
    model in VRAM, which would make a fresh test load contend and crash (the abort0 the
    suite saw while running in parallel with :8080). Wait until the model is actually
    unloaded before running the suite.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        used = gpu_vram_used_mb()
        if used is not None and used < threshold_mb:
            return
        if time.monotonic() >= deadline:
            print(f"  WARNING: GPU VRAM still at {used} MB after {timeout_s:.0f}s; proceeding anyway.")
            return
        print(f"  GPU VRAM at {used} MB; waiting for the server to release the model…", flush=True)
        time.sleep(5)


def run_command(cmd: list[str], env: dict[str, str], timeout_s: float,
                heartbeat_s: float = 30.0) -> tuple[int, str]:
    """Run cmd, streaming a heartbeat while it runs and a tail of its output on exit.

    The subprocess output is redirected to a temp file (not a pipe) so a chatty binary
    cannot fill the pipe buffer and deadlock the run; a periodic heartbeat keeps a
    wrapping monitor/watcher from seeing a long silent run (model load + inference) as
    a hang and killing a legitimately-running test.
    """
    print(f"\n$ {' '.join(cmd)}", flush=True)
    started = time.monotonic()
    logf = tempfile.NamedTemporaryFile(mode="w+", suffix=".log", delete=False, encoding="utf-8")
    log_path = logf.name
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
    deadline = started + timeout_s
    timed_out = False
    try:
        while proc.poll() is None:
            if time.monotonic() >= deadline:
                proc.kill()
                timed_out = True
                break
            remaining = deadline - time.monotonic()
            time.sleep(min(heartbeat_s, max(0.0, remaining)))
            if proc.poll() is None and not timed_out:
                print(f"  … still running ({time.monotonic() - started:.0f}s)", flush=True)
        if timed_out:
            proc.wait()
    finally:
        logf.flush()
        logf.close()
    elapsed = time.monotonic() - started
    with open(log_path, encoding="utf-8", errors="replace") as f:
        content = f.read()
    try:
        os.unlink(log_path)
    except OSError:
        pass
    tail = "\n".join(content.strip().splitlines()[-40:])
    if tail:
        print(tail, flush=True)
    print(f"  -> exit {proc.returncode} in {elapsed:.1f}s" + (" (TIMEOUT)" if timed_out else ""),
          flush=True)
    return proc.returncode, ""


def main() -> int:
    # Line-buffer stdout so a wrapping monitor/watcher sees every line immediately;
    # block-buffered pipe stdout would otherwise look idle and get killed mid-wait.
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT, help="path to the .ninfer artifact")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080, help="production server port to wait for")
    parser.add_argument("--port-timeout", type=float, default=3600, help="seconds to wait for the port to free")
    parser.add_argument("--e2e-timeout", type=float, default=1800, help="per-scenario GPU test timeout (s)")
    parser.add_argument("--unit-timeout", type=float, default=300)
    parser.add_argument("--skip-port-wait", action="store_true", help="assume the port is already free")
    parser.add_argument(
        "--build-dir",
        default=str(DEFAULT_BUILD_DIR),
        help="CMake build directory (default: <repo>/build-windows)",
    )
    parser.add_argument(
        "--build-config",
        default=None,
        help="build config dir name (e.g. Debug/Release); default: the one whose e2e binary is newest",
    )
    parser.add_argument(
        "--log",
        default=str(DEFAULT_LOG),
        help="durable log file (default: <repo>/.qwen/prefix_smoke.log); append mode",
    )
    args = parser.parse_args()

    tee = _Tee(sys.stdout, Path(args.log))
    sys.stdout = tee
    print(f"\n=== prefix smoke run log: {args.log} ===", flush=True)

    artifact = args.artifact
    if not Path(artifact).is_file():
        raise SystemExit(f"artifact not found: {artifact}")

    e2e_binary, unit_binary, build_config = resolve_test_binaries(
        Path(args.build_dir), args.build_config
    )
    if not e2e_binary.is_file():
        raise SystemExit(f"e2e test binary not found (build it first): {e2e_binary}")
    if not unit_binary.is_file():
        raise SystemExit(f"unit test binary not found (build it first): {unit_binary}")

    e2e_mtime = e2e_binary.stat().st_mtime
    print(f"Build config : {build_config}")
    print(f"E2E binary   : {e2e_binary}")
    print(f"  built      : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(e2e_mtime))}")
    newest_src = newest_source_commit_unix_time()
    if newest_src is not None and e2e_mtime < newest_src:
        print(
            "  WARNING: the e2e binary is OLDER than the newest commit touching the watched "
            "cache sources; it may predate a recent fix (crash fix / issue #251). Rebuild the "
            f"'{build_config}' config before trusting a result.",
            file=sys.stderr,
        )

    if not args.skip_port_wait:
        print(f"Waiting for the production server on {args.host}:{args.port} to stop…")
        wait_for_free_port(args.host, args.port, args.port_timeout)
        print("  port is free.")
        print("Waiting for GPU VRAM to release…", flush=True)
        wait_for_vram_release(args.port_timeout)
        print("  GPU VRAM released; running the suite on a clean GPU.", flush=True)

    base_env = dict(os.environ)
    base_env["NINFER_TEST_ARTIFACT"] = str(Path(artifact).resolve())

    results: list[tuple[Issue, bool, str]] = []
    for issue in ISSUES:
        if issue.kind == "e2e":
            env = dict(base_env, NINFER_PREFIX_REAL_SCENARIO=issue.name)
            code, _ = run_command([str(e2e_binary)], env, args.e2e_timeout)
        else:
            env = dict(os.environ)
            code, _ = run_command([str(unit_binary)], env, args.unit_timeout)
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
    print(f"\nAll {len(ISSUES)} prefix-caching checks verified against the real artifact.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
