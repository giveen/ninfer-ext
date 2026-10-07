"""Run one configuration and turn it into a number the design can fit.

Two drivers mirror the reference's bench/server split:

* ``BenchDriver`` runs ``ninfer_bench`` once per configuration and reads the JSON report. It is
  exact, single-stream, and cheap to reason about.
* ``ServerDriver`` launches ``ninfer-serve`` and issues the runbook's concurrent requests, then
  reads the tokens/s the server actually delivered. It measures the concurrency and speculative
  behaviour the single-stream bench cannot.

A configuration that OOMs, crashes, times out or reports an impossible number is data, not an
abort: it is recorded with a status and excluded from the fits and the picks.
"""

from __future__ import annotations

from dataclasses import dataclass
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import threading
import time
from typing import Callable

from .factors import Setting
from .usecases import UseCase

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCH = REPO_ROOT / "build/bench/ninfer_bench"
DEFAULT_SERVE = REPO_ROOT / "build/apps/ninfer-serve"

STATUS_OK = "OK"
STATUS_OOM = "OOM"
STATUS_SIGNAL = "SIGNAL"
STATUS_TIMEOUT = "TIMEOUT"
STATUS_ERROR = "ERROR"
STATUS_PARSE_FAIL = "PARSE_FAIL"
STATUS_IMPLAUSIBLE = "IMPLAUSIBLE"

# A decode rate above this is not physically possible on a single RTX-class card and means the
# driver returned without decoding.
MAX_PLAUSIBLE_TG = 2000.0
MAX_PLAUSIBLE_PP = 200000.0


@dataclass(frozen=True)
class Measurement:
    setting: Setting
    status: str
    pp_tps: float
    tg_tps: float
    objective: float
    seconds: float
    detail: str = ""
    command: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK


def _classify_failure(returncode: int, stderr: str, timed_out: bool) -> tuple[str, str]:
    if timed_out:
        return STATUS_TIMEOUT, "timeout"
    text = stderr.lower()
    if "out of memory" in text or "cuda_error_out_of_memory" in text or "oom" in text:
        return STATUS_OOM, "out of memory"
    if returncode < 0:
        return STATUS_SIGNAL, f"killed by signal {-returncode}"
    return STATUS_ERROR, stderr.strip().splitlines()[-1] if stderr.strip() else "driver error"


class BenchDriver:
    """Single-stream measurement through the public ``ninfer_bench``."""

    def __init__(
        self,
        artifact: Path,
        *,
        bench: Path = DEFAULT_BENCH,
        warmup: int = 1,
        timeout: float = 600.0,
        run_dir: Path | None = None,
    ) -> None:
        self.artifact = Path(artifact)
        self.bench = Path(bench)
        self.warmup = warmup
        self.timeout = timeout
        self.run_dir = Path(run_dir) if run_dir else Path(tempfile.mkdtemp(prefix="ninfer-opt-"))

    def command(self, setting: Setting, use_case: UseCase, reps: int) -> list[str]:
        return [
            str(self.bench),
            "--weights",
            str(self.artifact),
            "-pg",
            f"{use_case.n_prompt},{use_case.n_gen}",
            *setting.render_bench(),
            "-r",
            str(reps),
            "--warmup",
            str(self.warmup),
            "-o",
            "json",
        ]

    def measure(self, setting: Setting, use_case: UseCase, reps: int) -> Measurement:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.run_dir / f"report_{abs(hash(setting.label())) % (1 << 32):08x}.json"
        argv = [*self.command(setting, use_case, reps), "--output-file", str(report_path)]
        started = time.monotonic()
        timed_out = False
        try:
            result = subprocess.run(
                argv, capture_output=True, text=True, timeout=self.timeout, check=False
            )
            returncode, stderr = result.returncode, result.stderr
        except subprocess.TimeoutExpired:
            timed_out, returncode, stderr = True, -1, ""
        seconds = time.monotonic() - started
        if timed_out or returncode != 0:
            status, detail = _classify_failure(returncode, stderr, timed_out)
            return Measurement(setting, status, 0.0, 0.0, 0.0, seconds, detail, tuple(argv))
        try:
            pp, tg = _parse_bench_report(report_path, use_case)
        except (OSError, ValueError, KeyError) as error:
            return Measurement(
                setting, STATUS_PARSE_FAIL, 0.0, 0.0, 0.0, seconds, str(error), tuple(argv)
            )
        status, detail = _plausibility(pp, tg)
        objective = use_case.effective_tokens_per_second(pp, tg) if use_case.objective == "eff" else (
            tg if use_case.objective == "tg" else pp
        )
        if status != STATUS_OK:
            objective = 0.0
        return Measurement(setting, status, pp, tg, objective, seconds, detail, tuple(argv))


def _parse_bench_report(path: Path, use_case: UseCase) -> tuple[float, float]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if report.get("schema_version") != 15:
        raise ValueError(f"unexpected bench schema {report.get('schema_version')!r}")
    for test in report["tests"]:
        if test["n_prompt"] == use_case.n_prompt and test["n_gen"] == use_case.n_gen:
            prefill = test.get("prefill_seconds_mean", 0.0)
            decode = test.get("decode_seconds_mean", 0.0)
            pp = use_case.n_prompt / prefill if prefill > 0 else 0.0
            tg = use_case.n_gen / decode if (decode > 0 and use_case.n_gen) else 0.0
            return pp, tg
    raise ValueError("report has no test matching the runbook request shape")


def _plausibility(pp: float, tg: float) -> tuple[str, str]:
    if pp > MAX_PLAUSIBLE_PP:
        return STATUS_IMPLAUSIBLE, f"prefill {pp:.0f} t/s is not physically possible"
    if tg > MAX_PLAUSIBLE_TG:
        return STATUS_IMPLAUSIBLE, f"decode {tg:.0f} t/s is not physically possible"
    return STATUS_OK, ""


class ServerDriver:
    """Concurrent measurement through ``ninfer-serve`` and the OpenAI chat endpoint."""

    def __init__(
        self,
        artifact: Path,
        *,
        serve: Path = DEFAULT_SERVE,
        warmup: int = 1,
        timeout: float = 900.0,
        run_dir: Path | None = None,
    ) -> None:
        self.artifact = Path(artifact)
        self.serve = Path(serve)
        self.warmup = warmup
        self.timeout = timeout
        self.run_dir = Path(run_dir) if run_dir else Path(tempfile.mkdtemp(prefix="ninfer-opt-serve-"))

    def _free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            return probe.getsockname()[1]

    def command(self, setting: Setting, use_case: UseCase, port: int) -> list[str]:
        return [
            str(self.serve),
            str(self.artifact),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--max-concurrency",
            str(use_case.concurrency),
            *setting.render_serve(),
            "--log-level",
            "warning",
        ]

    def measure(self, setting: Setting, use_case: UseCase, reps: int) -> Measurement:
        port = self._free_port()
        argv = self.command(setting, use_case, port)
        started = time.monotonic()
        process = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            if not _wait_ready(port, process, self.timeout):
                stdout, stderr = process.communicate(timeout=5)
                status, detail = _classify_failure(process.returncode or 0, stderr or stdout, False)
                return Measurement(setting, status, 0.0, 0.0, 0.0, time.monotonic() - started,
                                   detail, tuple(argv))
            _serve_once(port, use_case)  # warmup / graph prime
            for _ in range(max(0, self.warmup - 1)):
                _serve_once(port, use_case)
            samples = [_serve_once(port, use_case) for _ in range(max(1, reps))]
        except (OSError, ValueError) as error:
            return Measurement(setting, STATUS_ERROR, 0.0, 0.0, 0.0, time.monotonic() - started,
                               str(error), tuple(argv))
        finally:
            _terminate(process)
        seconds = time.monotonic() - started
        prompts = sorted(s[0] for s in samples)
        decodes = sorted(s[1] for s in samples)
        pp = prompts[len(prompts) // 2]
        tg = decodes[len(decodes) // 2]
        status, detail = _plausibility(pp, tg)
        objective = use_case.effective_tokens_per_second(pp, tg) if use_case.objective == "eff" else (
            tg if use_case.objective == "tg" else pp
        )
        if status != STATUS_OK:
            objective = 0.0
        return Measurement(setting, status, pp, tg, objective, seconds, detail, tuple(argv))


def _wait_ready(port: int, process: subprocess.Popen, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            connection.request("GET", "/v1/models")
            response = connection.getresponse()
            response.read()
            connection.close()
            if response.status == 200:
                return True
        except OSError:
            time.sleep(0.5)
    return False


def _serve_once(port: int, use_case: UseCase) -> tuple[float, float]:
    """Send ``concurrency`` simultaneous requests; return aggregate (pp t/s, tg t/s)."""
    prompt = ("The quick brown fox jumps over the lazy dog. " * (use_case.n_prompt // 9 + 1)).strip()
    body = json.dumps(
        {
            "model": "local",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max(1, use_case.n_gen),
            "temperature": 0.0,
            "stream": False,
        }
    ).encode("utf-8")
    results: list[tuple[int, int, float]] = []
    lock = threading.Lock()

    def one() -> None:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
        start = time.monotonic()
        try:
            connection.request(
                "POST", "/v1/chat/completions", body=body,
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            elapsed = time.monotonic() - start
            usage = payload.get("usage", {})
            with lock:
                results.append(
                    (int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)),
                     elapsed)
                )
        finally:
            connection.close()

    threads = [threading.Thread(target=one) for _ in range(use_case.concurrency)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if not results:
        raise ValueError("no server responses")
    prompt_tokens = sum(r[0] for r in results)
    decode_tokens = sum(r[1] for r in results)
    wall = max(r[2] for r in results)
    return (prompt_tokens / wall if wall > 0 else 0.0,
            decode_tokens / wall if wall > 0 else 0.0)


def _terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except (OSError, ProcessLookupError):
        process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (OSError, ProcessLookupError):
            process.kill()
        process.wait(timeout=5)


def driver_for(
    artifact: Path,
    use_case: UseCase,
    *,
    bench: Path = DEFAULT_BENCH,
    serve: Path = DEFAULT_SERVE,
    **kwargs,
) -> BenchDriver | ServerDriver:
    if use_case.driver == "server":
        return ServerDriver(artifact, serve=serve, **kwargs)
    return BenchDriver(artifact, bench=bench, **kwargs)
