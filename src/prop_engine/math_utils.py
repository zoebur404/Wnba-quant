import polars as pl

def strip_market_vig(df: pl.DataFrame) -> pl.DataFrame:
    """
    Strips market hold/vig from decimal odds and outputs fair market probabilities.
    Ensures that implied probabilities across all outcomes for a specific market block sum to 1.0.
    Eliminates orphan lines by ensuring a minimum of 2 outcomes exist per block.
    """
    if df.height == 0:
        return df.with_columns(
            pl.Series("raw_prob", dtype=pl.Float64),
            pl.Series("market_implied_sum", dtype=pl.Float64),
            pl.Series("fair_market_prob", dtype=pl.Float64)
        )

    partition_cols = ["event_id", "bookmaker", "market_prop", "player_name", "line_threshold"]

    # 1. Calculate raw probability (1 / decimal_odds)
    df = df.with_columns(
        raw_prob=1.0 / pl.col("decimal_odds")
    )

    # 2. Filter out orphan lines (we need at least 2 outcomes per block to form a proper market)
    df = df.with_columns(
        outcome_count=pl.col("outcome_name").count().over(partition_cols)
    ).filter(
        pl.col("outcome_count") >= 2
    ).drop("outcome_count")

    # 3. Calculate sum of raw probabilities per block to find the vig
    df = df.with_columns(
        market_implied_sum=pl.col("raw_prob").sum().over(partition_cols)
    )

    # 4. Strip vig by dividing raw probability by the market implied sum
    df = df.with_columns(
        fair_market_prob=pl.col("raw_prob") / pl.col("market_implied_sum")
    )

    return df
