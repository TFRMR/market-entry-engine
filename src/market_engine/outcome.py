"""Deterministic trade-outcome evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from market_engine.entry import SetupCandidate
from market_engine.execution import Execution
from market_engine.exits import ExitArea
from market_engine.structure import Direction


@dataclass(frozen=True)
class TradeOutcome:
    status: str
    exit_index: int | None
    exit_price: float | None
    pnl: float | None
    risk_multiple: float | None
    target_price: float | None


def evaluate_trade(
    candidate: SetupCandidate,
    execution: Execution,
    target: ExitArea,
    frame: pd.DataFrame,
    horizon: int | None = None,
) -> TradeOutcome:
    """Evaluate the first stop/target outcome within an optional horizon."""
    if horizon is not None and horizon <= 0:
        raise ValueError("horizon must be greater than zero.")

    end_index = len(frame)
    if horizon is not None:
        end_index = min(
            len(frame),
            candidate.setup_index + 1 + horizon,
        )

    for index in range(candidate.setup_index + 1, end_index):
        candle = frame.iloc[index]
        hit_stop = (
            candle["low"] <= execution.invalidation_price
            if candidate.direction is Direction.UP
            else candle["high"] >= execution.invalidation_price
        )
        hit_target = (
            candle["high"] >= target.price
            if candidate.direction is Direction.UP
            else candle["low"] <= target.price
        )

        if hit_stop:
            exit_price = execution.invalidation_price
            pnl = (
                exit_price - execution.entry_price
                if candidate.direction is Direction.UP
                else execution.entry_price - exit_price
            )
            return TradeOutcome(
                status="STOP",
                exit_index=index,
                exit_price=float(exit_price),
                pnl=float(pnl),
                risk_multiple=float(pnl / execution.risk),
                target_price=float(target.price),
            )

        if hit_target:
            exit_price = target.price
            pnl = (
                exit_price - execution.entry_price
                if candidate.direction is Direction.UP
                else execution.entry_price - exit_price
            )
            return TradeOutcome(
                status="TARGET",
                exit_index=index,
                exit_price=float(exit_price),
                pnl=float(pnl),
                risk_multiple=float(pnl / execution.risk),
                target_price=float(target.price),
            )

    return TradeOutcome(
        status="OPEN",
        exit_index=None,
        exit_price=None,
        pnl=None,
        risk_multiple=None,
        target_price=float(target.price),
    )
