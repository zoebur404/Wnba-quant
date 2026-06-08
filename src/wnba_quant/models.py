"""Model backends for WNBA player prop projections."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import numpy as np
import polars as pl

from .config import PropModelConfig


def numeric_feature_columns(frame: pl.DataFrame, target: str) -> list[str]:
    """Infer model-ready numeric feature columns from an engineered frame."""

    excluded = {target, "line"}
    numeric_types = {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    }
    return [
        name
        for name, dtype in frame.schema.items()
        if name not in excluded and dtype in numeric_types
    ]


@dataclass
class PoissonPropModel:
    """Empirical-Bayes Poisson baseline for count-like player props."""

    target: str
    shrinkage_games: float = 6.0

    def fit(self, frame: pl.DataFrame) -> "PoissonPropModel":
        if self.target not in frame.columns:
            raise ValueError(f"target column '{self.target}' is required")

        self.league_mean_ = float(frame.select(pl.col(self.target).mean()).item())
        player_rates = frame.group_by("player_id").agg(
            pl.col(self.target).sum().alias("target_sum"),
            pl.len().alias("games"),
        )
        player_rates = player_rates.with_columns(
            (
                (pl.col("target_sum") + self.shrinkage_games * self.league_mean_)
                / (pl.col("games") + self.shrinkage_games)
            ).alias("poisson_mean")
        )
        self.player_rates_ = player_rates.select(["player_id", "poisson_mean"])
        return self

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        if not hasattr(self, "player_rates_"):
            raise RuntimeError("PoissonPropModel must be fit before predict")
        joined = frame.join(self.player_rates_, on="player_id", how="left")
        return (
            joined.select(pl.col("poisson_mean").fill_null(self.league_mean_))
            .to_numpy()
            .reshape(-1)
        )


class XGBoostPropModel:
    """XGBoost regression backend for player prop mean projections."""

    def __init__(self, config: PropModelConfig) -> None:
        if importlib.util.find_spec("xgboost") is None:
            raise ImportError("Install the optional 'ml' extra to use model_type='xgboost'.")
        import xgboost as xgb

        self.target = config.target
        self.feature_columns = config.feature_columns
        self.params = config.xgboost_params
        self._model = xgb.XGBRegressor(**self.params)

    def fit(self, frame: pl.DataFrame) -> "XGBoostPropModel":
        features = list(self.feature_columns or numeric_feature_columns(frame, self.target))
        if not features:
            raise ValueError("No numeric feature columns are available for XGBoost")
        clean = frame.drop_nulls(features + [self.target])
        if clean.is_empty():
            raise ValueError("No complete training rows are available for XGBoost")

        self.feature_columns_ = features
        self._model.fit(clean.select(features).to_numpy(), clean[self.target].to_numpy())
        return self

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        if not hasattr(self, "feature_columns_"):
            raise RuntimeError("XGBoostPropModel must be fit before predict")
        clean = frame.select(self.feature_columns_).fill_null(0.0)
        return np.clip(self._model.predict(clean.to_numpy()), a_min=0.0, a_max=None)
