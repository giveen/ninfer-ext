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
import re
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
    temp_c: float = 0.0
    ttft_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK


def gpu_temperature_c() -> float | None:
    """GPU temperature in Celsius, or None when no sensor is readable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


@dataclass
class ThermalController:
    """Put the GPU in a defined thermal state before each measurement.

    ``warm`` preheats a configuration with its own workload until the temperature stops rising, then
    measures there — the sustained rate a deployment gets from an already-hot card. ``idle`` settles
    back toward the idle baseline between runs — the burst rate a bursty workload sees. ``off``
    measures whatever state the previous run left behind, which is the confound the other two exist
    to remove.
    """

    mode: str = "off"
    cap_seconds: float = 180.0
    poll_seconds: float = 2.0
    idle_target_c: float = 45.0
    stable_delta_c: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in ("off", "warm", "idle"):
            raise ValueError(f"unknown thermal mode {self.mode!r}")

    def settle(self, preheat: Callable[[], None]) -> None:
        if self.mode == "off":
            return
        if gpu_temperature_c() is None:
            return
        if self.mode == "warm":
            preheat()
            self._wait_until_stable()
        else:
            self._wait_until_idle()

    def _wait_until_stable(self) -> None:
        deadline = time.monotonic() + self.cap_seconds
        previous = gpu_temperature_c()
        while time.monotonic() < deadline and previous is not None:
            time.sleep(self.poll_seconds)
            current = gpu_temperature_c()
            if current is None or abs(current - previous) <= self.stable_delta_c:
                return
            previous = current

    def _wait_until_idle(self) -> None:
        deadline = time.monotonic() + self.cap_seconds
        while time.monotonic() < deadline:
            current = gpu_temperature_c()
            if current is None or current <= self.idle_target_c:
                return
            time.sleep(self.poll_seconds)


def _classify_failure(returncode: int, stderr: str, timed_out: bool) -> tuple[str, str]:
    if timed_out:
        return STATUS_TIMEOUT, "timeout"
    text = stderr.lower()
    if "out of memory" in text or "outofmemory" in text or re.search(r"\boom\b", text):
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
        thermal: ThermalController | None = None,
    ) -> None:
        self.artifact = Path(artifact)
        self.bench = Path(bench)
        self.warmup = warmup
        self.timeout = timeout
        self.run_dir = Path(run_dir) if run_dir else Path(tempfile.mkdtemp(prefix="ninfer-opt-"))
        self.thermal = thermal or ThermalController()

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

    def _preheat(self, setting: Setting, use_case: UseCase) -> None:
        """One throwaway pass so a `warm` settle measures the configuration already hot."""
        argv = self.command(setting, use_case, 1)
        try:
            subprocess.run(argv, capture_output=True, text=True, timeout=self.timeout, check=False)
        except (OSError, subprocess.SubprocessError):
            pass

    def measure(self, setting: Setting, use_case: UseCase, reps: int) -> Measurement:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.run_dir / f"report_{abs(hash(setting.label())) % (1 << 32):08x}.json"
        argv = [*self.command(setting, use_case, reps), "--output-file", str(report_path)]
        self.thermal.settle(lambda: self._preheat(setting, use_case))
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
        temp = gpu_temperature_c() or 0.0
        if timed_out or returncode != 0:
            status, detail = _classify_failure(returncode, stderr, timed_out)
            return Measurement(setting, status, 0.0, 0.0, 0.0, seconds, detail, tuple(argv), temp)
        try:
            pp, tg = _parse_bench_report(report_path, use_case)
        except (OSError, ValueError, KeyError) as error:
            return Measurement(
                setting, STATUS_PARSE_FAIL, 0.0, 0.0, 0.0, seconds, str(error), tuple(argv), temp
            )
        status, detail = _plausibility(pp, tg)
        objective = use_case.score(pp, tg)
        if status != STATUS_OK:
            objective = 0.0
        return Measurement(setting, status, pp, tg, objective, seconds, detail, tuple(argv), temp)


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
        thermal: ThermalController | None = None,
    ) -> None:
        self.artifact = Path(artifact)
        self.serve = Path(serve)
        self.warmup = warmup
        self.timeout = timeout
        self.run_dir = Path(run_dir) if run_dir else Path(tempfile.mkdtemp(prefix="ninfer-opt-serve-"))
        self.thermal = thermal or ThermalController()
        self._messages: list[dict] | None = None

    def messages(self, use_case: UseCase) -> list[dict]:
        if self._messages is None:
            self._messages = _corpus_messages_for(use_case)
        return self._messages

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
        messages = self.messages(use_case)
        started = time.monotonic()
        process = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            model_id = _wait_ready(port, process, self.timeout)
            if model_id is None:
                stdout, stderr = process.communicate(timeout=5)
                status, detail = _classify_failure(process.returncode or 0, stderr or stdout, False)
                return Measurement(setting, status, 0.0, 0.0, 0.0, time.monotonic() - started,
                                   detail, tuple(argv), gpu_temperature_c() or 0.0)
            for _ in range(max(1, self.warmup)):
                _serve_once(port, use_case, messages, model_id)
            self.thermal.settle(lambda: _serve_once(port, use_case, messages, model_id))
            samples = [_serve_once(port, use_case, messages, model_id)
                       for _ in range(max(1, reps))]
        except (OSError, ValueError) as error:
            return Measurement(setting, STATUS_ERROR, 0.0, 0.0, 0.0, time.monotonic() - started,
                               str(error), tuple(argv), gpu_temperature_c() or 0.0)
        finally:
            _terminate(process)
        seconds = time.monotonic() - started
        prompts = sorted(s[0] for s in samples)
        decodes = sorted(s[1] for s in samples)
        ttfts = sorted(s[2] for s in samples if s[2] > 0)
        pp = prompts[len(prompts) // 2]
        tg = decodes[len(decodes) // 2]
        ttft = ttfts[len(ttfts) // 2] if ttfts else 0.0
        status, detail = _plausibility(pp, tg)
        objective = use_case.score(pp, tg, ttft)
        if status != STATUS_OK:
            objective = 0.0
        return Measurement(setting, status, pp, tg, objective, seconds, detail, tuple(argv),
                           gpu_temperature_c() or 0.0, ttft)


def _wait_ready(port: int, process: subprocess.Popen, timeout: float) -> str | None:
    """Wait for the server, returning its advertised model id, or None on failure."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return None
        try:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            connection.request("GET", "/v1/models")
            response = connection.getresponse()
            body = response.read()
            connection.close()
            if response.status == 200:
                try:
                    items = json.loads(body).get("data") or []
                    if items and isinstance(items[0], dict) and items[0].get("id"):
                        return str(items[0]["id"])
                except (json.JSONDecodeError, TypeError, AttributeError):
                    pass
                return "local"
        except OSError:
            time.sleep(0.5)
    return None


def _corpus_messages_for(use_case: UseCase) -> list[dict]:
    """The frozen TTFT corpus shape nearest the runbook's prompt, or a synthetic fallback."""
    try:
        from tools.bench.ttft.corpus import Corpus, CorpusError

        corpus = Corpus()
        shapes = corpus.manifest["shapes"]
        best = min(
            shapes.items(),
            key=lambda item: abs(int(item[1]["prompt_tokens"]) - use_case.n_prompt),
        )
        return corpus.shape_messages(best[0])
    except (ImportError, CorpusError, OSError, ValueError, KeyError):
        prompt = (
            "The quick brown fox jumps over the lazy dog. " * (use_case.n_prompt // 9 + 1)
        ).strip()
        return [{"role": "user", "content": prompt}]


@dataclass(frozen=True)
class ServeSample:
    prompt_tokens: int
    decode_tokens: int
    wall_seconds: float
    ttft_ms: float


def _serve_once(
    port: int, use_case: UseCase, messages: list[dict], model_id: str
) -> tuple[float, float, float]:
    """Release ``concurrency`` simultaneous streamed requests; return (pp t/s, tg t/s, median TTFT ms)."""
    body = json.dumps(
        {
            "model": model_id,
            "messages": messages,
            "max_tokens": max(1, use_case.n_gen),
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode("utf-8")
    samples: list[ServeSample] = []
    lock = threading.Lock()
    barrier = threading.Barrier(use_case.concurrency)

    def one() -> None:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
        try:
            barrier.wait()  # release every request together, so TTFT reflects queuing
            start = time.monotonic()
            connection.request(
                "POST", "/v1/chat/completions", body=body,
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            first: float | None = None
            usage: dict = {}
            deltas = 0
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if payload.get("usage"):
                    usage = payload["usage"]
                for choice in payload.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content") or delta.get("reasoning_content"):
                        if first is None:
                            first = time.monotonic()
                        deltas += 1
            elapsed = time.monotonic() - start
            with lock:
                samples.append(
                    ServeSample(
                        prompt_tokens=int(usage.get("prompt_tokens", 0)),
                        decode_tokens=int(usage.get("completion_tokens", deltas)),
                        wall_seconds=elapsed,
                        ttft_ms=(first - start) * 1000.0 if first is not None else 0.0,
                    )
                )
        finally:
            connection.close()

    threads = [threading.Thread(target=one) for _ in range(use_case.concurrency)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if not samples:
        raise ValueError("no server responses")
    prompt_tokens = sum(s.prompt_tokens for s in samples)
    decode_tokens = sum(s.decode_tokens for s in samples)
    wall = max(s.wall_seconds for s in samples)
    ttfts = sorted(s.ttft_ms for s in samples if s.ttft_ms > 0)
    ttft = ttfts[len(ttfts) // 2] if ttfts else 0.0
    return (
        prompt_tokens / wall if wall > 0 else 0.0,
        decode_tokens / wall if wall > 0 else 0.0,
        ttft,
    )


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
