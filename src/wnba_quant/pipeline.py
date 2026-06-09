"""End-to-end WNBA player prop modeling pipeline."""

from __future__ import annotations

import polars as pl

from .config import PropModelConfig
from .distributions import prop_probabilities
from .features import (
    attach_prop_board_features,
    prepare_next_game_features,
    prepare_player_game_features,
)
from .models import PoissonPropModel, XGBoostPropModel
from .odds import expected_value, implied_probability


class PlayerPropPipeline:
    """Train a prop model and score a current WNBA player prop board."""

    def __init__(self, config: PropModelConfig | None = None) -> None:
        self.config = config or PropModelConfig()
        self.model = self._build_model()

    def _build_model(self):
        if self.config.model_type == "poisson":
            return PoissonPropModel(
                target=self.config.target,
                shrinkage_games=self.config.poisson_shrinkage_games,
            )
        return XGBoostPropModel(self.config)

    def fit(self, game_logs: pl.DataFrame) -> "PlayerPropPipeline":
        """Engineer historical features and fit the selected model backend."""

        self.training_features_ = prepare_player_game_features(
            game_logs=game_logs,
            target=self.config.target,
            rolling_windows=self.config.rolling_windows,
        )
        trainable = self.training_features_.filter(
            pl.col("games_played_entering") >= self.config.min_history_games
        )
        self.model.fit(trainable)
        self.next_game_features_ = prepare_next_game_features(
            game_logs=game_logs,
            target=self.config.target,
            rolling_windows=self.config.rolling_windows,
        )
        return self

    def score_props(self, prop_board: pl.DataFrame) -> pl.DataFrame:
        """Project means and over/under probabilities for a prop board."""

        if not hasattr(self, "next_game_features_"):
            raise RuntimeError("PlayerPropPipeline must be fit before score_props")

        market_props = prop_board.filter(pl.col("market") == self.config.market_name)
        if market_props.is_empty():
            raise ValueError(
                f"prop_board contains no rows for market '{self.config.market_name}'"
            )

        scored = attach_prop_board_features(market_props, self.next_game_features_)
        means = self.model.predict(scored)
        scored = scored.with_columns(pl.Series("projected_mean", means))

        probabilities = [
            prop_probabilities(mean=float(mean), line=float(line))
            for mean, line in scored.select(["projected_mean", "line"]).iter_rows()
        ]
        scored = scored.with_columns(
            [
                pl.Series("prob_under", [item[0] for item in probabilities]),
                pl.Series("prob_push", [item[1] for item in probabilities]),
                pl.Series("prob_over", [item[2] for item in probabilities]),
                (pl.col("projected_mean") - pl.col("line")).alias("edge_to_line"),
            ]
        )
        return self._with_optional_odds_edges(scored).sort("edge_to_line", descending=True)

    def _with_optional_odds_edges(self, scored: pl.DataFrame) -> pl.DataFrame:
        """Add odds-derived break-even and EV columns when odds are present."""

        expressions: list[pl.Expr] = []
        if "over_odds" in scored.columns:
            expressions.extend(
                [
                    pl.col("over_odds")
                    .map_elements(
                        lambda odds: None if odds is None else implied_probability(odds),
                        return_dtype=pl.Float64,
                    )
                    .alias("implied_prob_over"),
                    pl.struct(["prob_over", "prob_under", "over_odds"])
                    .map_elements(
                        lambda row: None
                        if row["over_odds"] is None
                        else expected_value(
                            row["prob_over"], row["prob_under"], row["over_odds"]
                        ),
                        return_dtype=pl.Float64,
                    )
                    .alias("ev_over"),
                ]
            )
        if "under_odds" in scored.columns:
            expressions.extend(
                [
                    pl.col("under_odds")
                    .map_elements(
                        lambda odds: None if odds is None else implied_probability(odds),
                        return_dtype=pl.Float64,
                    )
                    .alias("implied_prob_under"),
                    pl.struct(["prob_under", "prob_over", "under_odds"])
                    .map_elements(
                        lambda row: None
                        if row["under_odds"] is None
                        else expected_value(
                            row["prob_under"], row["prob_over"], row["under_odds"]
                        ),
                        return_dtype=pl.Float64,
                    )
                    .alias("ev_under"),
                ]
            )
        return scored.with_columns(expressions) if expressions else scored
