from __future__ import annotations

from itertools import combinations
import random

import pytest

from tools.optimizer.design import (
    additive_prediction,
    design_prime,
    elementary_effects,
    factor_column,
    is_prime,
    largest_prime_at_most,
    main_effects,
    morris_trajectories,
    orthogonal_array,
    orthogonal_design,
    predicted_optimal,
    select_levels,
    significant_factors,
)
from tools.optimizer.factors import Factor


@pytest.mark.parametrize(
    "p,factors,runs",
    [(2, 3, 4), (2, 7, 8), (2, 15, 16), (3, 4, 9), (3, 13, 27), (5, 6, 25)],
)
def test_orthogonal_array_sizes(p, factors, runs):
    array = orthogonal_array(p, factors)
    assert len(array) == runs
    assert all(len(row) == factors for row in array)
    assert all(0 <= value < p for row in array for value in row)


@pytest.mark.parametrize("p,factors", [(2, 3), (3, 4), (3, 13), (5, 6)])
def test_orthogonal_array_is_balanced(p, factors):
    array = orthogonal_array(p, factors)
    for left, right in combinations(range(factors), 2):
        counts: dict[tuple[int, int], int] = {}
        for row in array:
            counts[(row[left], row[right])] = counts.get((row[left], row[right]), 0) + 1
        assert len(set(counts.values())) == 1, "each level pair must appear equally often"


def test_uniform_prime_and_selection():
    assert is_prime(3) and not is_prime(4)
    assert largest_prime_at_most(5) == 5
    assert largest_prime_at_most(4) == 3
    factor = Factor("spec", ("none", "mtp", "dflash", "dflash2"))
    assert select_levels(factor, 3) == ("none", "dflash", "dflash2")
    assert select_levels(factor, 2) == ("none", "dflash2")
    assert design_prime([factor]) == 3
    # A 2-level factor is dummy-mapped onto a 5-level column, not capped.
    narrow = Factor("flag", ("off", "on"))
    assert factor_column(narrow, 5) == ("off", "on", "off", "on", "off")
    assert design_prime([Factor("kv", ("a", "b", "c", "d", "e")), narrow]) == 5


def test_orthogonal_design_expands_settings():
    design = orthogonal_design(
        [Factor("kv", ("a", "b", "c")), Factor("ctx", ("1", "2", "3"))]
    )
    settings = list(design.settings())
    assert len(settings) == design.run_count == 9
    assert all(set(s.values) == {"kv", "ctx"} for s in settings)


def test_morris_effects_rank_the_dominant_factor():
    rng = random.Random(7)
    names = ["weak", "dominant", "noise"]
    trajectories = morris_trajectories(3, 3, 8, rng)
    # Objective depends only on factor 1's grid position.
    scores = [point[1] * 10.0 for trajectory in trajectories for point in trajectory.points]
    effects = elementary_effects(names, trajectories, scores)
    assert effects[0].name == "dominant"
    assert effects[0].mu_star > effects[1].mu_star
    assert significant_factors(effects, 0.5) == ["dominant"]


def test_main_effects_find_the_best_level():
    factors = [Factor("x", ("lo", "hi")), Factor("y", ("lo", "hi"))]
    design = orthogonal_design(factors)
    scores = {
        run: (5.0 if design.columns[0][run[0]] == "hi" else 0.0)
        for run in design.runs
    }
    effects = main_effects(design, scores)
    x = next(e for e in effects if e.name == "x")
    assert x.best_level() == "hi"
    assert x.range == pytest.approx(5.0)
    assert effects[0].name == "x"


def test_predicted_optimal_and_additive_prediction():
    factors = [Factor("x", ("lo", "hi")), Factor("y", ("lo", "hi"))]
    design = orthogonal_design(factors)
    scores = {
        run: (5.0 if design.columns[0][run[0]] == "hi" else 0.0)
        + (2.0 if design.columns[1][run[1]] == "hi" else 0.0)
        for run in design.runs
    }
    effects = main_effects(design, scores)
    assert predicted_optimal(effects) == {"x": "hi", "y": "hi"}
    grand = sum(scores.values()) / len(scores)
    # The additive model recovers the true optimum for a purely additive objective.
    assert additive_prediction(effects, grand) == pytest.approx(7.0)
