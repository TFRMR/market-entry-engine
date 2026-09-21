"""Deterministic setup-to-outcome backtest orchestration."""

from __future__ import annotations

import pandas as pd

from market_engine.entry import SetupCandidate
from market_engine.execution import execute_entry
from market_engine.exits import build_exit_areas
from market_engine.outcome import TradeOutcome, evaluate_trade
from market_engine.structure import ValidSwing


def run_backtest(
    candidates: list[SetupCandidate],
    swings: list[ValidSwing],
    frame: pd.DataFrame,
    spread_price: float,
) -> list[TradeOutcome]:
    """Evaluate setup candidates through execution and deterministic outcomes."""
    outcomes: list[TradeOutcome] = []

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

        outcomes.append(
            evaluate_trade(
                candidate=candidate,
                execution=execution,
                target=exit_areas[0],
                frame=frame,
            )
        )

    return outcomes
