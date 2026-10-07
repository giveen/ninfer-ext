"""Design of experiments for the optimizer: screening, orthogonal arrays, refinement.

Brute force over ninfer's knobs explodes (five knobs at four levels is 4^5 = 1024 GPU runs). This
mirrors the llama-optimize funnel:

* **Morris screening** walks a few trajectories through the space and ranks every knob by how much
  it moves the objective (mu*) and how much that depends on the others (sigma). Negligible knobs
  are pinned so the expensive stage never spends runs on them.
* **Orthogonal arrays** estimate every surviving knob's main effect in 9-125 balanced runs instead
  of thousands. The arrays are generated over GF(p) by projective points, which reproduces the
  standard Taguchi L4/L8/L9/L16/L25/L27/L125 designs.

No third-party dependency: the construction is ~40 lines of arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
import random

from .factors import Factor


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


def largest_prime_at_most(n: int) -> int:
    for p in range(n, 1, -1):
        if is_prime(p):
            return p
    return 2


def _projective_points(p: int, m: int) -> list[tuple[int, ...]]:
    """One representative per 1-dimensional subspace of GF(p)^m."""
    points: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for vec in product(range(p), repeat=m):
        if all(v == 0 for v in vec):
            continue
        first = next(v for v in vec if v != 0)
        inv = pow(first, -1, p)
        canon = tuple((v * inv) % p for v in vec)
        if canon not in seen:
            seen.add(canon)
            points.append(canon)
    return points


def orthogonal_array(p: int, num_factors: int) -> list[tuple[int, ...]]:
    """A strength-2 orthogonal array with ``p``-level factors over GF(p).

    Returns one tuple of level indices (0..p-1) per run. Reproduces L4(2^3), L8(2^7), L16(2^15),
    L9(3^4), L27(3^13) and L25(5^6).
    """
    if not is_prime(p):
        raise ValueError(f"orthogonal_array needs a prime level count, got {p}")
    if num_factors < 1:
        raise ValueError("need at least one factor")
    m = 1
    while (p**m - 1) // (p - 1) < num_factors:
        m += 1
    columns = _projective_points(p, m)[:num_factors]
    rows = []
    for row in product(range(p), repeat=m):
        rows.append(
            tuple(sum(r * c for r, c in zip(row, column)) % p for column in columns)
        )
    return rows


def design_prime(factors: list[Factor], requested: int | None = None) -> int:
    """The shared column level count for an orthogonal design.

    Chosen from the widest factor (capped at 5 to bound the run count). Factors with fewer levels
    are dummy-mapped onto the same column count rather than capping everyone at their size.
    """
    if requested is not None:
        if not is_prime(requested):
            raise ValueError(f"level count must be prime for the array, got {requested}")
        return requested
    widest = max(len(f.levels) for f in factors)
    return min(5, largest_prime_at_most(widest))


def factor_column(factor: Factor, p: int) -> tuple[str, ...]:
    """A length-``p`` column of actual levels: truncate a wide factor, dummy-map a narrow one."""
    levels = factor.levels
    if len(levels) >= p:
        return select_levels(factor, p)
    return tuple(levels[value % len(levels)] for value in range(p))


def select_levels(factor: Factor, count: int) -> tuple[str, ...]:
    """Evenly spaced ``count`` levels of ``factor``, endpoints included."""
    levels = factor.levels
    if count >= len(levels):
        return levels
    if count == 1:
        return (levels[0],)
    last = len(levels) - 1
    picked = [levels[round(i * last / (count - 1))] for i in range(count)]
    # Preserve order and uniqueness.
    seen: list[str] = []
    for value in picked:
        if value not in seen:
            seen.append(value)
    return tuple(seen)


@dataclass(frozen=True)
class OrthogonalDesign:
    factors: list[Factor]
    columns: tuple[tuple[str, ...], ...]  # per-factor, each length p (levels may repeat)
    runs: tuple[tuple[int, ...], ...]  # level indices per run
    p: int

    @property
    def run_count(self) -> int:
        return len(self.runs)

    def settings(self):
        from .factors import Setting

        for run in self.runs:
            yield Setting({f.name: self.columns[i][run[i]] for i, f in enumerate(self.factors)})


def orthogonal_design(factors: list[Factor], requested_levels: int | None = None) -> OrthogonalDesign:
    if not factors:
        raise ValueError("no factors to design")
    p = design_prime(factors, requested_levels)
    columns = tuple(factor_column(f, p) for f in factors)
    runs = orthogonal_array(p, len(factors))
    return OrthogonalDesign(factors, columns, tuple(runs), p)


@dataclass(frozen=True)
class MorrisTrajectory:
    points: list[tuple[int, ...]]
    order: list[int]  # factor index stepped between points[i] and points[i+1]


def morris_trajectories(
    factor_count: int, levels: int, trajectories: int, rng: random.Random
) -> list[MorrisTrajectory]:
    """Morris one-at-a-time trajectories on a ``levels``-point grid."""
    if levels < 2:
        raise ValueError("Morris needs at least two levels")
    delta = 1  # one grid step
    out: list[MorrisTrajectory] = []
    for _ in range(trajectories):
        point = [rng.randrange(levels) for _ in range(factor_count)]
        order = list(range(factor_count))
        rng.shuffle(order)
        points = [tuple(point)]
        for index in order:
            point[index] = min(levels - 1, point[index] + delta)
            if point[index] == points[-1][index]:  # already at the top: step down instead
                point[index] = max(0, point[index] - delta)
            points.append(tuple(point))
        out.append(MorrisTrajectory(points, order))
    return out


@dataclass(frozen=True)
class Effect:
    name: str
    mu_star: float
    sigma: float


def elementary_effects(
    factor_names: list[str],
    trajectories: list[MorrisTrajectory],
    scores: list[float],
) -> list[Effect]:
    """Morris mu*/sigma from per-point scores laid out trajectory by trajectory.

    ``scores`` must be the concatenation of each trajectory's ``k+1`` point scores, in order.
    """
    width = len(factor_names) + 1
    if len(scores) != width * len(trajectories):
        raise ValueError("scores do not match the trajectory layout")
    effects: dict[int, list[float]] = {i: [] for i in range(len(factor_names))}
    for t, trajectory in enumerate(trajectories):
        base = t * width
        for step, factor_index in enumerate(trajectory.order):
            before = scores[base + step]
            after = scores[base + step + 1]
            effects[factor_index].append(after - before)
    out: list[Effect] = []
    for index, name in enumerate(factor_names):
        values = effects[index]
        if not values:
            out.append(Effect(name, 0.0, 0.0))
            continue
        mean = sum(values) / len(values)
        mu_star = sum(abs(v) for v in values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        out.append(Effect(name, mu_star, math.sqrt(variance)))
    return sorted(out, key=lambda e: e.mu_star, reverse=True)


def significant_factors(effects: list[Effect], keep_fraction: float = 0.05) -> list[str]:
    """Factors whose mu* is at least ``keep_fraction`` of the largest seen."""
    if not effects:
        return []
    top = effects[0].mu_star
    if top <= 0:
        return [e.name for e in effects]
    return [e.name for e in effects if e.mu_star >= keep_fraction * top]


@dataclass(frozen=True)
class MainEffect:
    name: str
    means: tuple[float, ...]
    levels: tuple[str, ...]

    @property
    def range(self) -> float:
        return max(self.means) - min(self.means) if self.means else 0.0

    def best_level(self) -> str:
        return self.levels[max(range(len(self.means)), key=lambda i: self.means[i])]


def main_effects(design: OrthogonalDesign, scores: dict[tuple[int, ...], float]) -> list[MainEffect]:
    """Per-factor per-level mean score, ranked by range (Taguchi main effects)."""
    out: list[MainEffect] = []
    for index, factor in enumerate(design.factors):
        column = design.columns[index]
        unique = tuple(dict.fromkeys(column))
        buckets: dict[str, list[float]] = {level: [] for level in unique}
        for run in design.runs:
            score = scores.get(run)
            if score is None:
                continue
            buckets[column[run[index]]].append(score)
        means = tuple(
            sum(buckets[level]) / len(buckets[level]) if buckets[level] else 0.0
            for level in unique
        )
        out.append(MainEffect(factor.name, means, unique))
    return sorted(out, key=lambda e: e.range, reverse=True)
