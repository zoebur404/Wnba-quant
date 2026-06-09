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
    """Return mutually exclusive under, push, and over probabilities.

    For half-point lines, ``push`` is zero and ``under`` is ``P(X < line)``.
    For integer lines, ``under`` is ``P(X < line)``, ``push`` is
    ``P(X == line)``, and ``over`` is ``P(X > line)``.
    """

    floor_line = math.floor(line)
    if float(line).is_integer():
        under = poisson_cdf(floor_line - 1, mean)
        push = poisson_cdf(floor_line, mean) - under
        over = 1.0 - poisson_cdf(floor_line, mean)
    else:
        under_threshold = math.floor(line)
        under = poisson_cdf(under_threshold, mean)
        push = 0.0
        over = 1.0 - under

    return max(under, 0.0), max(push, 0.0), max(over, 0.0)
