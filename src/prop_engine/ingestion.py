import requests
import polars as pl
from typing import Dict, Any, List

def fetch_odds_data(api_key: str, sport: str = "basketball_wnba", markets: str = "h2h") -> List[Dict[str, Any]]:
    """
    Fetches odds data from The Odds API.
    """
    url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": markets,
        "oddsFormat": "decimal"
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def flatten_odds_data(events_data: List[Dict[str, Any]]) -> pl.DataFrame:
    """
    Unpacks nested JSON from The Odds API into a flat Polars DataFrame.
    """
    if not events_data:
        # Return an empty DataFrame with the expected schema
        return pl.DataFrame({
            "event_id": pl.Series(dtype=pl.Utf8),
            "sport_key": pl.Series(dtype=pl.Utf8),
            "commence_time": pl.Series(dtype=pl.Utf8),
            "bookmaker": pl.Series(dtype=pl.Utf8),
            "market_prop": pl.Series(dtype=pl.Utf8),
            "player_name": pl.Series(dtype=pl.Utf8),
            "outcome_name": pl.Series(dtype=pl.Utf8),
            "line_threshold": pl.Series(dtype=pl.Float64),
            "decimal_odds": pl.Series(dtype=pl.Float64)
        })

    flattened_data = []

    for event in events_data:
        event_id = event.get("id")
        sport_key = event.get("sport_key")
        commence_time = event.get("commence_time")

        for bookmaker in event.get("bookmakers", []):
            bookmaker_key = bookmaker.get("key")

            for market in bookmaker.get("markets", []):
                market_key = market.get("key")

                for outcome in market.get("outcomes", []):
                    # Check if it's a player prop outcome
                    if "description" in outcome and "point" in outcome:
                        flattened_data.append({
                            "event_id": event_id,
                            "sport_key": sport_key,
                            "commence_time": commence_time,
                            "bookmaker": bookmaker_key,
                            "market_prop": market_key,
                            "player_name": outcome.get("description"),
                            "outcome_name": outcome.get("name"),
                            "line_threshold": outcome.get("point"),
                            "decimal_odds": outcome.get("price")
                        })
                    else:
                        # Sometimes general markets don't have descriptions, default to None or 'game'
                         flattened_data.append({
                            "event_id": event_id,
                            "sport_key": sport_key,
                            "commence_time": commence_time,
                            "bookmaker": bookmaker_key,
                            "market_prop": market_key,
                            "player_name": outcome.get("description", "game"),
                            "outcome_name": outcome.get("name"),
                            "line_threshold": outcome.get("point", 0.0),
                            "decimal_odds": outcome.get("price")
                        })

    if not flattened_data:
        return pl.DataFrame({
            "event_id": pl.Series(dtype=pl.Utf8),
            "sport_key": pl.Series(dtype=pl.Utf8),
            "commence_time": pl.Series(dtype=pl.Utf8),
            "bookmaker": pl.Series(dtype=pl.Utf8),
            "market_prop": pl.Series(dtype=pl.Utf8),
            "player_name": pl.Series(dtype=pl.Utf8),
            "outcome_name": pl.Series(dtype=pl.Utf8),
            "line_threshold": pl.Series(dtype=pl.Float64),
            "decimal_odds": pl.Series(dtype=pl.Float64)
        })

    df = pl.DataFrame(flattened_data)

    # Ensure correct data types
    df = df.with_columns([
        pl.col("line_threshold").cast(pl.Float64),
        pl.col("decimal_odds").cast(pl.Float64)
    ])

    return df
