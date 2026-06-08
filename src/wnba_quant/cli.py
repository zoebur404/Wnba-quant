"""Command-line entry point for scoring WNBA player props."""

from __future__ import annotations

import argparse

import polars as pl

from .config import PropModelConfig
from .pipeline import PlayerPropPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score WNBA player prop lines.")
    parser.add_argument("--game-logs", required=True, help="Historical player-game CSV path.")
    parser.add_argument("--prop-board", required=True, help="Current player prop CSV path.")
    parser.add_argument("--output", required=True, help="Output CSV path for scored props.")
    parser.add_argument("--target", default="points", help="Target stat to model.")
    parser.add_argument(
        "--model-type",
        choices=["poisson", "xgboost"],
        default="poisson",
        help="Model backend to use.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = PropModelConfig(target=args.target, model_type=args.model_type)
    pipeline = PlayerPropPipeline(config)
    game_logs = pl.read_csv(args.game_logs, try_parse_dates=True)
    prop_board = pl.read_csv(args.prop_board, try_parse_dates=True)
    scored = pipeline.fit(game_logs).score_props(prop_board)
    scored.write_csv(args.output)


if __name__ == "__main__":
    main()
