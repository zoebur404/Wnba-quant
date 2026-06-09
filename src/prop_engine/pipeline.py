import polars as pl
import numpy as np
from .ingestion import fetch_odds_data, flatten_odds_data
from .math_utils import strip_market_vig
from .models import PropModel

def generate_synthetic_data(num_samples: int = 1000) -> tuple[pl.DataFrame, pl.Series]:
    """
    Generates synthetic training data to enable end-to-end pipeline execution.
    """
    np.random.seed(42)

    # Feature matrix (e.g., past moving averages of player points, minutes played)
    X_np = np.random.rand(num_samples, 3) * 10

    # Target variable (Poisson distributed based on features)
    # The true lambda is a combination of the features
    true_lambdas = np.exp(0.1 * X_np[:, 0] + 0.2 * X_np[:, 1] + 0.05 * X_np[:, 2])
    y_np = np.random.poisson(true_lambdas)

    X = pl.DataFrame({
        "feature_1": X_np[:, 0],
        "feature_2": X_np[:, 1],
        "feature_3": X_np[:, 2]
    })

    y = pl.Series("target", y_np)

    return X, y

def run_pipeline(api_key: str, sport: str = "basketball_wnba", markets: str = "player_points"):
    """
    Orchestrates the entire engine flow.
    """
    print("1. Fetching data from The Odds API...")
    raw_data = fetch_odds_data(api_key, sport, markets)

    print("2. Flattening odds data...")
    df_flat = flatten_odds_data(raw_data)

    if df_flat.height == 0:
        print("No market data returned. Pipeline ending early.")
        return df_flat

    print("3. Stripping market vig...")
    df_fair = strip_market_vig(df_flat)

    print("4. Generating synthetic training data & training model...")
    X_train, y_train = generate_synthetic_data(1000)
    model = PropModel()
    model.train(X_train, y_train)

    print("5. Generating Edge Matrix...")
    # For prediction features, we'll create synthetic random features matching our training size for each row in df_fair
    # In a real app, this would merge with a feature store matching player names.
    num_rows = df_fair.height
    np.random.seed(99)
    X_predict_np = np.random.rand(num_rows, 3) * 10
    X_predict = pl.DataFrame({
        "feature_1": X_predict_np[:, 0],
        "feature_2": X_predict_np[:, 1],
        "feature_3": X_predict_np[:, 2]
    })

    # Predict expected value (lambda)
    lambdas = model.predict_lambda(X_predict)

    # Calculate probability of crossing the threshold
    thresholds = df_fair["line_threshold"].to_numpy()
    outcomes = df_fair["outcome_name"].to_numpy()

    model_probs = model.evaluate_probability(lambdas, thresholds, outcomes)

    # Append predictions and calculate edge
    edge_matrix = df_fair.with_columns([
        pl.Series("model_projected_prob", model_probs, dtype=pl.Float64)
    ]).with_columns([
        (pl.col("model_projected_prob") - pl.col("fair_market_prob")).alias("edge")
    ])

    return edge_matrix
