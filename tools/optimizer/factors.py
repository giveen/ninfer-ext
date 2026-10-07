"""The knobs ninfer exposes, as optimizable factors with concrete levels.

A factor is a named axis with a small set of levels. Levels are complete argument values, and the
renderer turns a ``Setting`` (factor name -> level) into a driver command line. Choices that would
be invalid together (a draft length without a speculative backend, an expert cache on a dense
model) are encoded as single categorical levels, so the design never proposes a dead combination.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .detect import Hardware, ModelInfo
from .usecases import UseCase

# KV precision, cheapest/fastest first. `quality` is the ordering `--min-kv` floors against.
KV_DTYPES: dict[str, dict[str, object]] = {
    "bf16": {"quality": 4, "note": "full precision, largest KV"},
    "int8": {"quality": 3, "note": "near-lossless, 2x smaller than bf16"},
    "fp8": {"quality": 3, "note": "near-lossless, native FP8 path"},
    "nvfp4": {"quality": 2, "note": "4-bit KV, best speed/context on Blackwell"},
    "k8v4": {"quality": 2, "note": "8-bit K / 4-bit V"},
}
QUALITY_FLOORS = {"any": 0, "k8v4": 2, "nvfp4": 2, "fp8": 3, "int8": 3, "bf16": 4}


@dataclass(frozen=True)
class Factor:
    name: str
    levels: tuple[str, ...]
    help: str = ""

    def __post_init__(self) -> None:
        if len(self.levels) < 2:
            raise ValueError(f"factor {self.name!r} needs at least two levels")
        if len(set(self.levels)) != len(self.levels):
            raise ValueError(f"factor {self.name!r} has duplicate levels")


@dataclass(frozen=True)
class Setting:
    """A complete point in factor space."""

    values: dict[str, str]

    def __getitem__(self, name: str) -> str:
        return self.values[name]

    def render_bench(self) -> list[str]:
        args: list[str] = []
        for name, value in self.values.items():
            args.extend(_render(name, value, driver="bench"))
        return args

    def render_serve(self) -> list[str]:
        args: list[str] = []
        for name, value in self.values.items():
            args.extend(_render(name, value, driver="server"))
        return args

    def label(self) -> str:
        return " ".join(f"{k}={v}" for k, v in sorted(self.values.items()))


def _render(name: str, value: str, *, driver: str) -> list[str]:
    if name == "kv_dtype":
        return ["--kv-dtype", value]
    if name == "ctx":
        return ["--max-context" if driver == "server" else "--max-ctx", value]
    if name == "prefill_chunk":
        return ["--prefill-chunk", value]
    if name == "spec":
        return _render_spec(value)
    if name == "lm_head_draft":
        return ["--lm-head-draft"] if value == "on" else []
    if name == "lookup_drafts":
        return ["--lookup-drafts", value]
    if name == "cuda_graph":
        return ["--no-cuda-graph"] if value == "off" else []
    if name == "kv_capacity":
        return ["--kv-capacity", value]
    if name == "max_concurrency":
        return ["--max-concurrency", value]
    if name == "kv_stream":
        return ["--kv-stream"] if value == "on" else []
    if name == "expert_cache":
        return ["--expert-cache", value]
    if name == "ngram_residency":
        return ["--ngram-residency", value]
    raise ValueError(f"unknown factor {name!r}")


def _render_spec(value: str) -> list[str]:
    """``none`` | ``BACKEND[:DRAFT]`` with an optional ``+head``.

    Encoding the head and the draft length into the speculative level keeps the design from
    proposing a head or a draft length without a backend, which the products reject.
    """
    if value == "none":
        return []
    spec, _, head = value.partition("+")
    backend, _, draft = spec.partition(":")
    args = ["--spec", backend]
    if draft:
        args += ["--draft-tokens", draft]
    if head == "head":
        args.append("--lm-head-draft")
    return args


@dataclass
class FactorOptions:
    """User overrides that win over the runbook's defaults."""

    min_kv: str = "any"
    kv_dtypes: tuple[str, ...] | None = None
    ctx_levels: tuple[int, ...] | None = None
    spec: tuple[str, ...] | None = None
    prefill_chunks: tuple[int, ...] = (512, 1024, 2048, 4096)
    cuda_graph: bool = True
    extra: dict[str, tuple[str, ...]] = field(default_factory=dict)


def kv_dtype_levels(options: FactorOptions) -> tuple[str, ...]:
    floor = QUALITY_FLOORS[options.min_kv]
    allowed = options.kv_dtypes or tuple(
        name for name, meta in KV_DTYPES.items() if int(meta["quality"]) >= floor
    )
    return tuple(allowed)


def _spec_levels(model: ModelInfo, options: FactorOptions) -> tuple[str, ...]:
    if options.spec is not None:
        return options.spec
    levels = ["none"]
    for backend, present in (
        ("mtp", model.has_mtp),
        ("dflash", model.has_dflash),
        ("dflash2", model.has_dflash2),
    ):
        if not present:
            continue
        # DFlash/DFlash2 require an explicit draft window; MTP adapts on its own.
        spec = f"{backend}:7" if backend in ("dflash", "dflash2") else backend
        levels.append(spec)
        if model.has_proposal_head:
            levels.append(f"{spec}+head")
    return tuple(levels)


def _ctx_levels(use_case: UseCase, model: ModelInfo, options: FactorOptions) -> tuple[str, ...]:
    if options.ctx_levels is not None:
        return tuple(str(c) for c in options.ctx_levels)
    floor = max(use_case.ctx_floor, use_case.n_prompt + use_case.n_gen)
    ceiling = min(model.native_context, max(floor * 2, floor))
    levels = sorted({floor // 2, floor, min(ceiling, floor * 3 // 2), ceiling})
    return tuple(str(max(2048, c)) for c in levels if c > 0)


def build_factors(
    use_case: UseCase,
    model: ModelInfo,
    hardware: Hardware,
    options: FactorOptions,
) -> list[Factor]:
    """The factor design for one runbook. Explicit overrides replace runbook defaults."""
    factors: list[Factor] = []

    kv = kv_dtype_levels(options)
    if len(kv) >= 2:
        factors.append(
            Factor("kv_dtype", kv, "KV cache precision (quality vs size/speed)")
        )

    ctx = _ctx_levels(use_case, model, options)
    if len(ctx) >= 2:
        factors.append(Factor("ctx", ctx, "context depth (the speed-vs-context axis)"))

    spec = _spec_levels(model, options)
    if len(spec) >= 2:
        factors.append(
            Factor("spec", spec, "speculative backend, optional draft length and proposal head")
        )

    if len(options.prefill_chunks) >= 2:
        factors.append(
            Factor(
                "prefill_chunk",
                tuple(str(c) for c in options.prefill_chunks),
                "prefill chunk size (wave-aligned)",
            )
        )

    for name, levels in options.extra.items():
        if len(levels) >= 2:
            factors.append(Factor(name, levels))

    return factors
