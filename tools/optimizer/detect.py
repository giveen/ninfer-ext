"""Detect the machine and the model so the user never hand-tunes factor levels.

Hardware comes from ``nvidia-smi`` (the number the inference process can actually allocate) and
``/proc/cpuinfo``. The model comes from the v3 artifact directory only; no tensors are loaded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import subprocess

from tools.artifact.reader import Artifact

_MIB = 1024 * 1024


@dataclass(frozen=True)
class Hardware:
    gpu_name: str
    vram_mib: int
    vram_free_mib: int
    physical_cores: int
    logical_cores: int
    source: str = "nvidia-smi"

    @property
    def vram_gi_b(self) -> float:
        return self.vram_mib / 1024.0

    def thread_levels(self) -> tuple[int, ...]:
        """Thread counts bracketing the physical-core count, where throughput usually peaks."""
        base = max(2, self.physical_cores)
        candidates = {
            max(1, base // 2),
            base,
            min(self.logical_cores, base + max(2, base // 2)),
            self.logical_cores,
        }
        return tuple(sorted(c for c in candidates if 1 <= c <= max(1, self.logical_cores)))


def _run(argv: list[str]) -> str | None:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def detect_hardware() -> Hardware:
    gpu_name, total, free = "unknown GPU", 0, 0
    source = "nvidia-smi"
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    if out:
        first = out.strip().splitlines()[0]
        parts = [p.strip() for p in first.split(",")]
        if len(parts) >= 3:
            gpu_name = parts[0]
            total = int(float(parts[1]))
            free = int(float(parts[2]))
    else:
        source = "unavailable"
    physical, logical = _cpu_cores()
    return Hardware(gpu_name, total, free, physical, logical, source)


def _cpu_cores() -> tuple[int, int]:
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    except OSError:
        logical = os.cpu_count() or 1
        return max(1, logical // 2), logical
    logical = 0
    physical: set[tuple[str, str]] = set()
    physical_id = core_id = ""
    for line in text.splitlines():
        if line.startswith("processor"):
            logical += 1
        elif line.startswith("physical id"):
            physical_id = line.split(":", 1)[1].strip()
        elif line.startswith("core id"):
            core_id = line.split(":", 1)[1].strip()
            physical.add((physical_id, core_id))
    if not physical:
        return max(1, logical // 2), max(1, logical)
    return len(physical), max(1, logical)


@dataclass(frozen=True)
class ModelInfo:
    name: str
    architecture: str
    num_layers: int
    hidden_size: int
    vocab_size: int
    is_moe: bool
    num_experts: int
    has_mtp: bool
    has_dflash: bool
    has_dflash2: bool
    has_vision: bool
    weight_bytes: int
    native_context: int = 32768
    components: tuple[str, ...] = field(default_factory=tuple)

    @property
    def weight_gi_b(self) -> float:
        return self.weight_bytes / (1024.0**3)


def detect_model(path: str | Path) -> ModelInfo:
    with Artifact(path) as artifact:
        directory = artifact.directory
        components = tuple(sorted(directory.components))
        if "text" not in directory.components:
            raise ValueError(f"{path}: artifact has no text component")
        config = directory.components["text"]["config"]
        weight_bytes = sum(obj.bytes for obj in directory.objects)
    layer_types = config.get("layer_types", [])
    num_layers = config.get("num_hidden_layers") or len(layer_types)
    is_moe = "num_experts" in config
    return ModelInfo(
        name=str(config.get("model_type", "") or directory.metadata.get("name", "model")),
        architecture=(config.get("architectures") or ["unknown"])[0],
        num_layers=int(num_layers),
        hidden_size=int(config.get("hidden_size", 0)),
        vocab_size=int(config.get("vocab_size", 0)),
        is_moe=is_moe,
        num_experts=int(config.get("num_experts", 0)) if is_moe else 0,
        has_mtp=("mtp" in components) or bool(config.get("num_nextn_predict_layers")),
        has_dflash="dflash" in components,
        has_dflash2="dflash2" in components,
        has_vision="vision" in components,
        weight_bytes=weight_bytes,
        native_context=int(config.get("max_position_embeddings", 32768) or 32768),
        components=components,
    )
