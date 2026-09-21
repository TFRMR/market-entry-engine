"""Deterministic aggregation of backtest trade outcomes."""

from __future__ import annotations

from dataclasses import dataclass

from market_engine.backtest import BacktestTrade


@dataclass(frozen=True)
class BacktestSummary:
    total_trades: int
    target_count: int
    stop_count: int
    open_count: int
    total_pnl: float
    total_r: float


def summarize_backtest(trades: list[BacktestTrade]) -> BacktestSummary:
    """Aggregate deterministic trade outcomes without portfolio assumptions."""
    target_count = sum(
        trade.outcome.status == "TARGET" for trade in trades
    )
    stop_count = sum(
        trade.outcome.status == "STOP" for trade in trades
    )
    open_count = sum(
        trade.outcome.status == "OPEN" for trade in trades
    )

    total_pnl = sum(
        trade.outcome.pnl
        for trade in trades
        if trade.outcome.pnl is not None
    )
    total_r = sum(
        trade.outcome.risk_multiple
        for trade in trades
        if trade.outcome.risk_multiple is not None
    )

    return BacktestSummary(
        total_trades=len(trades),
        target_count=target_count,
        stop_count=stop_count,
        open_count=open_count,
        total_pnl=float(total_pnl),
        total_r=float(total_r),
    )
