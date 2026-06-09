"""Sportsbook odds conversion and expected-value helpers."""

from __future__ import annotations


def american_odds_profit(odds: float) -> float:
    """Return profit on a one-unit stake for American odds."""

    if odds == 0:
        raise ValueError("American odds cannot be zero")
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)


def implied_probability(odds: float) -> float:
    """Convert American odds to break-even implied probability."""

    if odds == 0:
        raise ValueError("American odds cannot be zero")
    return 100.0 / (odds + 100.0) if odds > 0 else abs(odds) / (abs(odds) + 100.0)


def expected_value(win_probability: float, loss_probability: float, odds: float) -> float:
    """Expected profit on a one-unit stake, treating pushes as refunded."""

    return win_probability * american_odds_profit(odds) - loss_probability
