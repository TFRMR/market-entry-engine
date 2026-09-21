import pandas as pd
import pytest

from market_engine.backtest import run_backtest
from market_engine.backtest import run_backtest_trades
from market_engine.entry import SetupCandidate
from market_engine.stats import summarize_backtest
from market_engine.structure import Direction, SwingType, ValidSwing


def test_run_backtest_uses_execution_and_first_exit_area():
    candidate = SetupCandidate(
        setup_index=2,
        setup_timestamp=2,
        direction=Direction.UP,
        entry_index=3,
        entry_timestamp=3,
        entry_price=100.0,
        invalidation_price=95.0,
        risk=5.0,
        invalidation_swing_index=0,
    )

    swings = [
        ValidSwing(
            swing_type=SwingType.HIGH,
            index=0,
            timestamp=0,
            confirmation_index=1,
            confirmation_timestamp=1,
            price=110.0,
        ),
    ]

    frame = pd.DataFrame([
        {"high": 101.0, "low": 99.0},
        {"high": 102.0, "low": 99.0},
        {"high": 103.0, "low": 99.0},
        {"high": 110.0, "low": 100.0},
    ])

    outcomes = run_backtest(
        candidates=[candidate],
        swings=swings,
        frame=frame,
        spread_price=0.2,
    )

    assert len(outcomes) == 1
    assert outcomes[0].status == "TARGET"
    assert outcomes[0].exit_price == 110.0
    assert outcomes[0].pnl == pytest.approx(9.8)
    assert outcomes[0].risk_multiple == pytest.approx(9.8 / 5.2)


def test_run_backtest_skips_candidate_without_exit_area():
    candidate = SetupCandidate(
        setup_index=0,
        setup_timestamp=0,
        direction=Direction.UP,
        entry_index=1,
        entry_timestamp=1,
        entry_price=100.0,
        invalidation_price=95.0,
        risk=5.0,
        invalidation_swing_index=0,
    )

    frame = pd.DataFrame([
        {"high": 101.0, "low": 99.0},
        {"high": 101.0, "low": 99.0},
    ])

    outcomes = run_backtest(
        candidates=[candidate],
        swings=[],
        frame=frame,
        spread_price=0.2,
    )

    assert outcomes == []


def test_run_backtest_trades_keeps_candidate_identity_when_some_are_skipped():
    skipped = SetupCandidate(
        setup_index=1,
        setup_timestamp=1,
        direction=Direction.DOWN,
        entry_index=2,
        entry_timestamp=2,
        entry_price=100.0,
        invalidation_price=105.0,
        risk=5.0,
        invalidation_swing_index=0,
    )
    kept = SetupCandidate(
        setup_index=2,
        setup_timestamp=2,
        direction=Direction.UP,
        entry_index=3,
        entry_timestamp=3,
        entry_price=100.0,
        invalidation_price=95.0,
        risk=5.0,
        invalidation_swing_index=0,
    )
    swings = [
        ValidSwing(
            swing_type=SwingType.HIGH,
            index=0,
            timestamp=0,
            confirmation_index=1,
            confirmation_timestamp=1,
            price=110.0,
        ),
    ]
    frame = pd.DataFrame([
        {"high": 101.0, "low": 99.0},
        {"high": 102.0, "low": 99.0},
        {"high": 103.0, "low": 99.0},
        {"high": 110.0, "low": 100.0},
    ])

    trades = run_backtest_trades(
        [skipped, kept],
        swings,
        frame,
        spread_price=0.2,
    )

    assert len(trades) == 1
    assert trades[0].candidate is kept
    assert trades[0].outcome.status == "TARGET"


def test_summarize_backtest_aggregates_closed_and_open_trades():
    candidate = SetupCandidate(
        setup_index=2,
        setup_timestamp=2,
        direction=Direction.UP,
        entry_index=3,
        entry_timestamp=3,
        entry_price=100.0,
        invalidation_price=95.0,
        risk=5.0,
        invalidation_swing_index=0,
    )
    swings = [
        ValidSwing(
            swing_type=SwingType.HIGH,
            index=0,
            timestamp=0,
            confirmation_index=1,
            confirmation_timestamp=1,
            price=110.0,
        ),
    ]
    target_frame = pd.DataFrame([
        {"high": 101.0, "low": 99.0},
        {"high": 102.0, "low": 99.0},
        {"high": 103.0, "low": 99.0},
        {"high": 110.0, "low": 100.0},
    ])
    stop_frame = pd.DataFrame([
        {"high": 101.0, "low": 99.0},
        {"high": 102.0, "low": 99.0},
        {"high": 103.0, "low": 99.0},
        {"high": 104.0, "low": 94.0},
    ])

    target_trade = run_backtest_trades(
        [candidate], swings, target_frame, spread_price=0.2
    )[0]
    stop_trade = run_backtest_trades(
        [SetupCandidate(
            setup_index=2,
            setup_timestamp=2,
            direction=Direction.UP,
            entry_index=3,
            entry_timestamp=3,
            entry_price=100.0,
            invalidation_price=95.0,
            risk=5.0,
            invalidation_swing_index=0,
        )],
        swings,
        stop_frame,
        spread_price=0.2,
    )[0]

    open_trade = target_trade.__class__(
        candidate=target_trade.candidate,
        outcome=target_trade.outcome.__class__(
            status="OPEN",
            exit_index=None,
            exit_price=None,
            pnl=None,
            risk_multiple=None,
            target_price=110.0,
        ),
    )

    summary = summarize_backtest(
        [target_trade, stop_trade, open_trade]
    )

    assert summary.total_trades == 3
    assert summary.target_count == 1
    assert summary.stop_count == 1
    assert summary.open_count == 1
    assert summary.total_pnl == pytest.approx(4.6)
    assert summary.total_r == pytest.approx(
        target_trade.outcome.risk_multiple + stop_trade.outcome.risk_multiple
    )
