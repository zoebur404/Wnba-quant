"""Polars feature engineering for WNBA player prop models."""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl

REQUIRED_GAME_LOG_COLUMNS = {
    "game_date",
    "player_id",
    "player_name",
    "team",
    "opponent",
    "minutes",
}


def _validate_columns(frame: pl.DataFrame, required: Iterable[str], frame_name: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {', '.join(missing)}")


def _with_parsed_game_date(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.schema.get("game_date") == pl.String:
        return frame.with_columns(pl.col("game_date").str.to_date(strict=False))
    return frame


def _cum_mean_entering(column: str) -> pl.Expr:
    shifted = pl.col(column).shift(1)
    return shifted.cum_sum().over("player_id") / shifted.cum_count().over("player_id")


def prepare_player_game_features(
    game_logs: pl.DataFrame,
    target: str,
    rolling_windows: tuple[int, ...] = (3, 5, 10),
) -> pl.DataFrame:
    """Create leakage-safe player form features from historical game logs.

    The function expects one row per player-game. All features for a historical
    row are based only on games that player completed before the row's
    ``game_date``. That makes the frame safe for walk-forward training and
    backtesting because the target value on a row is never used to build that
    row's predictors.
    """

    _validate_columns(game_logs, REQUIRED_GAME_LOG_COLUMNS | {target}, "game_logs")
    frame = _with_parsed_game_date(game_logs).sort(["player_id", "game_date"])

    base_exprs = [
        pl.arange(0, pl.len()).over("player_id").alias("games_played_entering"),
        pl.col("minutes").shift(1).over("player_id").alias("minutes_prev"),
        pl.col(target).shift(1).over("player_id").alias(f"{target}_prev"),
        _cum_mean_entering("minutes").alias("minutes_player_avg"),
        _cum_mean_entering(target).alias(f"{target}_player_avg"),
    ]

    rolling_exprs: list[pl.Expr] = []
    for window in rolling_windows:
        rolling_exprs.extend(
            [
                pl.col(target)
                .shift(1)
                .rolling_mean(window_size=window, min_samples=1)
                .over("player_id")
                .alias(f"{target}_roll{window}"),
                pl.col("minutes")
                .shift(1)
                .rolling_mean(window_size=window, min_samples=1)
                .over("player_id")
                .alias(f"minutes_roll{window}"),
            ]
        )

    return frame.with_columns(base_exprs + rolling_exprs)


def prepare_next_game_features(
    game_logs: pl.DataFrame,
    target: str,
    rolling_windows: tuple[int, ...] = (3, 5, 10),
) -> pl.DataFrame:
    """Create one prediction feature row per player for their next game.

    Unlike selecting the latest historical training row, this function includes
    the player's most recent completed game in lagged, average, and rolling
    features. Use this output when scoring today's prop board.
    """

    _validate_columns(game_logs, REQUIRED_GAME_LOG_COLUMNS | {target}, "game_logs")
    frame = _with_parsed_game_date(game_logs).sort(["player_id", "game_date"])

    aggregations: list[pl.Expr] = [
        pl.col("player_name").last().alias("player_name"),
        pl.col("team").last().alias("team"),
        pl.col("opponent").last().alias("opponent_prev"),
        pl.col("game_date").last().alias("last_game_date"),
        pl.len().alias("games_played_entering"),
        pl.col("minutes").last().alias("minutes_prev"),
        pl.col(target).last().alias(f"{target}_prev"),
        pl.col("minutes").mean().alias("minutes_player_avg"),
        pl.col(target).mean().alias(f"{target}_player_avg"),
    ]
    for window in rolling_windows:
        aggregations.extend(
            [
                pl.col(target).tail(window).mean().alias(f"{target}_roll{window}"),
                pl.col("minutes").tail(window).mean().alias(f"minutes_roll{window}"),
            ]
        )

    return frame.group_by("player_id", maintain_order=True).agg(aggregations)


def attach_prop_board_features(
    prop_board: pl.DataFrame,
    next_game_features: pl.DataFrame,
) -> pl.DataFrame:
    """Join today's prop board to each player's next-game feature row.

    ``prop_board`` should include ``player_id``, ``market``, and ``line``. Common
    extra columns such as sportsbook odds, team, opponent, and game start time
    are preserved.
    """

    _validate_columns(prop_board, {"player_id", "market", "line"}, "prop_board")
    return prop_board.join(next_game_features, on="player_id", how="left", suffix="_hist")
