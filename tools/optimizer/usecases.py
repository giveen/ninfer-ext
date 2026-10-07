"""Use-case runbooks: one friendly name per situation, expanding to a request profile.

The optimizer tunes for *how the model is run*, not for a single abstract score. A runbook fixes
the representative request shape (prompt, generation), the driver (single-stream bench vs
concurrent server), the concurrency, and the objective the statistics maximize. Explicit user
flags always win over the runbook (see ``factors``).
"""

from __future__ import annotations

from dataclasses import dataclass

DRIVERS = ("bench", "server")
OBJECTIVES = ("tg", "pp", "eff")

# objective -> human label
OBJECTIVE_LABELS = {
    "tg": "decode tokens/s",
    "pp": "prefill tokens/s",
    "eff": "effective tokens/s (prefill+decode as the request experiences them)",
}


@dataclass(frozen=True)
class UseCase:
    name: str
    summary: str
    n_prompt: int
    n_gen: int
    concurrency: int
    driver: str
    objective: str
    ctx_floor: int
    # Prefill-heavy workloads let the optimizer trade decode speed for prompt throughput.
    prefill_bound: bool = False

    def __post_init__(self) -> None:
        if self.driver not in DRIVERS:
            raise ValueError(f"unknown driver {self.driver!r}")
        if self.objective not in OBJECTIVES:
            raise ValueError(f"unknown objective {self.objective!r}")
        if self.n_prompt <= 0 or self.n_gen < 0 or self.concurrency < 1:
            raise ValueError(f"{self.name}: invalid request shape")

    def effective_tokens_per_second(self, pp_tps: float, tg_tps: float) -> float:
        """(P+G) / (P/pp + G/tg): the blended rate the request actually experiences."""
        if pp_tps <= 0 or tg_tps <= 0:
            return 0.0
        seconds = self.n_prompt / pp_tps + (self.n_gen / tg_tps if self.n_gen else 0.0)
        return (self.n_prompt + self.n_gen) / seconds if seconds > 0 else 0.0


USE_CASES: dict[str, UseCase] = {
    "chat": UseCase(
        name="chat",
        summary="one interactive user (chat / coding assistant), short prompts",
        n_prompt=512,
        n_gen=256,
        concurrency=1,
        driver="bench",
        objective="tg",
        ctx_floor=8192,
    ),
    "code": UseCase(
        name="code",
        summary="one coding session: long prompts, long replies, single stream",
        n_prompt=4096,
        n_gen=512,
        concurrency=1,
        driver="bench",
        objective="eff",
        ctx_floor=32768,
    ),
    "agents": UseCase(
        name="agents",
        summary="several autonomous agents: long tool-use prompts, concurrent",
        n_prompt=8192,
        n_gen=256,
        concurrency=4,
        driver="server",
        objective="eff",
        ctx_floor=32768,
        prefill_bound=True,
    ),
    "multi-user": UseCase(
        name="multi-user",
        summary="many concurrent chat users: short prompts, high concurrency",
        n_prompt=1024,
        n_gen=256,
        concurrency=8,
        driver="server",
        objective="eff",
        ctx_floor=8192,
    ),
    "long-context": UseCase(
        name="long-context",
        summary="long-context RAG / document work: very long prompts, short answers",
        n_prompt=32768,
        n_gen=128,
        concurrency=1,
        driver="bench",
        objective="pp",
        ctx_floor=65536,
        prefill_bound=True,
    ),
    "batch": UseCase(
        name="batch",
        summary="offline batch scoring: throughput over latency",
        n_prompt=8192,
        n_gen=64,
        concurrency=8,
        driver="server",
        objective="pp",
        ctx_floor=16384,
        prefill_bound=True,
    ),
}


def use_case(name: str) -> UseCase:
    try:
        return USE_CASES[name]
    except KeyError as error:
        known = ", ".join(sorted(USE_CASES))
        raise ValueError(f"unknown use case {name!r}; known: {known}") from error
