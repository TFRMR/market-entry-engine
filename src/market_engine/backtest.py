"""Deterministic setup-to-outcome backtest orchestration."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from market_engine.entry import SetupCandidate
from market_engine.execution import execute_entry
from market_engine.exits import build_exit_areas
from market_engine.outcome import TradeOutcome, evaluate_trade
from market_engine.structure import ValidSwing


@dataclass(frozen=True)
class BacktestTrade:
    candidate: SetupCandidate
    outcome: TradeOutcome


def run_backtest_trades(
    candidates: list[SetupCandidate],
    swings: list[ValidSwing],
    frame: pd.DataFrame,
    spread_price: float,
) -> list[BacktestTrade]:
    """Evaluate candidates and keep each outcome tied to its candidate."""
    trades: list[BacktestTrade] = []

    for candidate in candidates:
        exit_areas = build_exit_areas(candidate, swings, frame)
        if not exit_areas:
            continue

        execution = execute_entry(
            direction=candidate.direction,
            quoted_price=float(candidate.entry_price),
            spread_price=spread_price,
            invalidation_price=float(candidate.invalidation_price),
        )

        outcome = evaluate_trade(
            candidate=candidate,
            execution=execution,
            target=exit_areas[0],
            frame=frame,
        )
        trades.append(BacktestTrade(candidate=candidate, outcome=outcome))

    return trades


def run_backtest(
    candidates: list[SetupCandidate],
    swings: list[ValidSwing],
    frame: pd.DataFrame,
    spread_price: float,
) -> list[TradeOutcome]:
    """Evaluate setup candidates through execution and deterministic outcomes."""
    return [
        trade.outcome
        for trade in run_backtest_trades(candidates, swings, frame, spread_price)
    ]
