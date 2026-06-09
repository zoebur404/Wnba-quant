from __future__ import annotations

import polars as pl

from wnba_quant import PlayerPropPipeline, PropModelConfig
from wnba_quant.distributions import prop_probabilities
from wnba_quant.features import prepare_next_game_features, prepare_player_game_features


def sample_game_logs() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_date": [
                "2025-05-01",
                "2025-05-03",
                "2025-05-05",
                "2025-05-07",
                "2025-05-01",
                "2025-05-03",
                "2025-05-05",
                "2025-05-07",
            ],
            "player_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "player_name": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "team": ["NYL", "NYL", "NYL", "NYL", "LVA", "LVA", "LVA", "LVA"],
            "opponent": ["LVA", "CON", "SEA", "CHI", "NYL", "SEA", "CON", "DAL"],
            "minutes": [30, 31, 28, 32, 20, 24, 22, 26],
            "points": [15, 18, 12, 21, 8, 10, 9, 11],
        }
    )


def test_feature_engineering_is_leakage_safe() -> None:
    features = prepare_player_game_features(sample_game_logs(), target="points", rolling_windows=(3,))
    player_one = features.filter(pl.col("player_id") == 1).sort("game_date")

    assert player_one["games_played_entering"].to_list() == [0, 1, 2, 3]
    assert player_one["points_prev"].to_list() == [None, 15, 18, 12]
    assert player_one["points_roll3"].to_list()[1] == 15


def test_next_game_features_include_most_recent_completed_game() -> None:
    features = prepare_next_game_features(sample_game_logs(), target="points", rolling_windows=(3,))
    player_one = features.filter(pl.col("player_id") == 1).row(0, named=True)

    assert player_one["games_played_entering"] == 4
    assert player_one["points_prev"] == 21
    assert player_one["points_roll3"] == 17


def test_poisson_pipeline_scores_prop_board() -> None:
    prop_board = pl.DataFrame(
        {
            "player_id": [1, 2],
            "player_name": ["A", "B"],
            "market": ["points", "points"],
            "line": [16.5, 9.5],
        }
    )

    pipeline = PlayerPropPipeline(
        PropModelConfig(target="points", model_type="poisson", min_history_games=1)
    )
    scored = pipeline.fit(sample_game_logs()).score_props(prop_board)

    assert {"projected_mean", "prob_over", "prob_under", "edge_to_line"}.issubset(
        scored.columns
    )
    assert scored.height == 2
    assert scored["prob_over"].min() >= 0
    assert scored["prob_over"].max() <= 1


def test_poisson_probabilities_support_push_lines() -> None:
    under, push, over = prop_probabilities(mean=10.0, line=10.0)

    assert under > 0
    assert push > 0
    assert over > 0
    assert round(under + push + over, 10) == 1.0
