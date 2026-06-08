"""Configuration for WNBA player prop modeling pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PropModelConfig:
    """Runtime options for feature engineering and prop modeling.

    Attributes:
        target: Box-score column to model, such as ``points``, ``rebounds``,
            ``assists``, or a derived stat that is present in the input table.
        model_type: ``"poisson"`` for a fast count-distribution baseline or
            ``"xgboost"`` for a gradient-boosted regression model.
        rolling_windows: Game counts used for rolling player form features.
        min_history_games: Minimum player history required before a prediction
            row is considered modelable.
        poisson_shrinkage_games: Number of pseudo-games used to blend player
            rate estimates with the league average in the Poisson baseline.
        feature_columns: Optional explicit feature list. If omitted, the
            pipeline uses all numeric engineered feature columns.
        xgboost_params: Parameters forwarded to ``xgboost.XGBRegressor`` when
            ``model_type="xgboost"``.
    """

    target: str = "points"
    model_type: str = "poisson"
    rolling_windows: tuple[int, ...] = (3, 5, 10)
    min_history_games: int = 3
    poisson_shrinkage_games: float = 6.0
    feature_columns: tuple[str, ...] | None = None
    xgboost_params: dict[str, object] = field(
        default_factory=lambda: {
            "n_estimators": 300,
            "max_depth": 3,
            "learning_rate": 0.03,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "objective": "reg:squarederror",
            "random_state": 7,
        }
    )

    def __post_init__(self) -> None:
        if self.model_type not in {"poisson", "xgboost"}:
            raise ValueError("model_type must be either 'poisson' or 'xgboost'")
        if self.min_history_games < 1:
            raise ValueError("min_history_games must be positive")
        if any(window < 1 for window in self.rolling_windows):
            raise ValueError("rolling_windows values must be positive")
