import pytest
import polars as pl
from src.prop_engine.math_utils import strip_market_vig
from src.prop_engine.ingestion import flatten_odds_data

def test_strip_market_vig_sum_to_one():
    """
    Verifies that implied probabilities derived from `strip_market_vig` accurately sum to 1.0 per market block.
    """
    df = pl.DataFrame({
        "event_id": ["1", "1", "2", "2", "2"],
        "bookmaker": ["b1", "b1", "b2", "b2", "b2"],
        "market_prop": ["m1", "m1", "m2", "m2", "m2"],
        "player_name": ["p1", "p1", "p2", "p2", "p2"],
        "outcome_name": ["Over", "Under", "Outcome1", "Outcome2", "Outcome3"],
        "line_threshold": [1.5, 1.5, 0.0, 0.0, 0.0],
        "decimal_odds": [1.9, 1.9, 2.0, 3.0, 6.0]
    })

    result = strip_market_vig(df)

    # Calculate sum of fair market probabilities per partition
    sums = result.group_by(["event_id", "bookmaker", "market_prop", "player_name", "line_threshold"]).agg(
        pl.col("fair_market_prob").sum().alias("prob_sum")
    )

    # Assert that all sums are approximately 1.0
    for prob_sum in sums["prob_sum"]:
        assert pytest.approx(prob_sum, 0.0001) == 1.0

def test_strip_market_vig_eliminates_orphans():
    """
    Verifies that `strip_market_vig` eliminates orphan individual lines (ensure Over and Under pairs exist).
    """
    df = pl.DataFrame({
        "event_id": ["1", "2", "2"],
        "bookmaker": ["b1", "b2", "b2"],
        "market_prop": ["m1", "m2", "m2"],
        "player_name": ["p1", "p2", "p2"],
        "outcome_name": ["Over", "Over", "Under"],
        "line_threshold": [1.5, 2.5, 2.5],
        "decimal_odds": [1.9, 1.9, 1.9]
    })

    result = strip_market_vig(df)

    # The block with event_id="1" should be eliminated because it only has 1 outcome
    assert len(result.filter(pl.col("event_id") == "1")) == 0
    # The block with event_id="2" should remain because it has 2 outcomes
    assert len(result.filter(pl.col("event_id") == "2")) == 2

def test_flatten_odds_data_scalar_retention():
    """
    Verifies that the flattening logic doesn't drop scalar row elements.
    """
    mock_events = [
        {
            "id": "event_123",
            "sport_key": "basketball_wnba",
            "commence_time": "2024-01-01T00:00:00Z",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "markets": [
                        {
                            "key": "player_points",
                            "outcomes": [
                                {
                                    "name": "Over",
                                    "description": "A'ja Wilson",
                                    "price": 1.90,
                                    "point": 20.5
                                },
                                {
                                    "name": "Under",
                                    "description": "A'ja Wilson",
                                    "price": 1.90,
                                    "point": 20.5
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    ]

    result = flatten_odds_data(mock_events)

    # Verify shape
    assert result.height == 2

    # Verify scalars are retained for all rows
    assert all(eid == "event_123" for eid in result["event_id"])
    assert all(sk == "basketball_wnba" for sk in result["sport_key"])
    assert all(bm == "draftkings" for bm in result["bookmaker"])
