# WNBA Quant

A Polars-first Python pipeline for modeling WNBA player props with a fast
Poisson baseline or an optional XGBoost regression backend.

## Pipeline design

1. **Ingest historical player game logs** with one row per player-game.
2. **Engineer leakage-safe features in Polars** by sorting by player/date and
   shifting rolling player form metrics by one game for walk-forward training.
3. **Train a model backend**:
   - `poisson`: empirical-Bayes player means shrunk toward the league average.
   - `xgboost`: optional gradient-boosted regression model for mean projections.
4. **Build next-game features** that include each player's most recent completed game, then join those rows to the current prop board.
5. **Convert projected means to probabilities** for over, under, and push outcomes.
6. **Rank props by edge to line** for downstream review or bankroll logic.

## Expected CSV inputs

`game_logs.csv` must contain at least:

- `game_date`
- `player_id`
- `player_name`
- `team`
- `opponent`
- `minutes`
- the target stat column, such as `points`, `rebounds`, or `assists`

`props.csv` must contain at least:

- `player_id`
- `market`
- `line`

The pipeline scores only the configured market, which defaults to the target
stat name. If your prop feed uses labels such as `player_points`, pass
`--market player_points` or set `PropModelConfig(market="player_points")`.
Optional `over_odds` and `under_odds` American-odds columns add break-even
probability and one-unit expected-value columns to the scored output. Any other
sportsbook, price, game, or injury-context columns are preserved.

## Usage

Install the base pipeline:

```bash
pip install -e .
```

Install XGBoost support:

```bash
pip install -e '.[ml]'
```

Score a current points prop board with the Poisson baseline:

```bash
wnba-props \
  --game-logs data/game_logs.csv \
  --prop-board data/current_props.csv \
  --target points \
  --market points \
  --model-type poisson \
  --output outputs/scored_points_props.csv
```

Use XGBoost instead:

```bash
wnba-props \
  --game-logs data/game_logs.csv \
  --prop-board data/current_props.csv \
  --target points \
  --market points \
  --model-type xgboost \
  --output outputs/scored_points_props.csv
```

## Python API

```python
import polars as pl
from wnba_quant import PlayerPropPipeline, PropModelConfig

logs = pl.read_csv("data/game_logs.csv", try_parse_dates=True)
props = pl.read_csv("data/current_props.csv", try_parse_dates=True)

pipeline = PlayerPropPipeline(
    PropModelConfig(
        target="points", market="points", model_type="poisson", min_history_games=3
    )
)
scored = pipeline.fit(logs).score_props(props)
scored.write_csv("outputs/scored_points_props.csv")
```
