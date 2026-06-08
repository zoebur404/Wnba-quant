"""Small distribution helpers used by prop models."""

from __future__ import annotations

import math


def poisson_cdf(k: int, lam: float) -> float:
    """Compute P(X <= k) for a Poisson random variable without SciPy."""

    if k < 0:
        return 0.0
    if lam <= 0:
        return 1.0

    term = math.exp(-lam)
    total = term
    for value in range(1, k + 1):
        term *= lam / value
        total += term
    return min(max(total, 0.0), 1.0)


def prop_probabilities(mean: float, line: float) -> tuple[float, float, float]:
    """Return under, push, and over probabilities for a count prop line."""

    floor_line = math.floor(line)
    under_threshold = math.ceil(line) - 1
    under = poisson_cdf(under_threshold, mean)

    push = 0.0
    if float(line).is_integer():
        push = poisson_cdf(floor_line, mean) - poisson_cdf(floor_line - 1, mean)
        over = 1.0 - poisson_cdf(floor_line, mean)
    else:
        over = 1.0 - under

    return under, max(push, 0.0), max(over, 0.0)
