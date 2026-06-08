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


def prepare_player_game_features(
    game_logs: pl.DataFrame,
    target: str,
    rolling_windows: tuple[int, ...] = (3, 5, 10),
) -> pl.DataFrame:
    """Create leakage-safe player form features from historical game logs.

    The function expects one row per player-game. Rolling features are shifted by
    one game within each player, so a row's target value is never used to build
    that row's predictors.
    """

    _validate_columns(game_logs, REQUIRED_GAME_LOG_COLUMNS | {target}, "game_logs")

    sort_cols = ["player_id", "game_date"]
    frame = (
        game_logs.with_columns(pl.col("game_date").str.to_date(strict=False))
        if game_logs.schema.get("game_date") == pl.String
        else game_logs
    )
    frame = frame.sort(sort_cols)

    base_exprs = [
        pl.arange(0, pl.len()).over("player_id").alias("games_played_entering"),
        pl.col("minutes").shift(1).over("player_id").alias("minutes_prev"),
        pl.col(target).shift(1).over("player_id").alias(f"{target}_prev"),
        (
            pl.col("minutes").shift(1).cum_sum().over("player_id")
            / pl.col("minutes").shift(1).cum_count().over("player_id")
        ).alias("minutes_player_avg"),
        (
            pl.col(target).shift(1).cum_sum().over("player_id")
            / pl.col(target).shift(1).cum_count().over("player_id")
        ).alias(f"{target}_player_avg"),
        pl.col(target).mean().over("team").alias(f"{target}_team_sample_avg"),
        pl.col(target).mean().over("opponent").alias(f"{target}_opp_allowed_sample_avg"),
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


def latest_player_features(feature_frame: pl.DataFrame) -> pl.DataFrame:
    """Return the most recent engineered feature row for each player."""

    _validate_columns(feature_frame, {"player_id", "game_date"}, "feature_frame")
    return feature_frame.sort(["player_id", "game_date"]).group_by("player_id").tail(1)


def attach_prop_board_features(
    prop_board: pl.DataFrame,
    latest_features: pl.DataFrame,
) -> pl.DataFrame:
    """Join today's prop board to each player's latest historical features.

    ``prop_board`` should include ``player_id``, ``market``, and ``line``. Common
    extra columns such as sportsbook odds, team, opponent, and game start time
    are preserved.
    """

    _validate_columns(prop_board, {"player_id", "market", "line"}, "prop_board")
    return prop_board.join(latest_features, on="player_id", how="left", suffix="_hist")
