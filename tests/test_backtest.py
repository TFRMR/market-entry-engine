import pandas as pd
import pytest

from market_engine.backtest import run_backtest
from market_engine.entry import SetupCandidate
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
