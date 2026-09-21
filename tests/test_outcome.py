import pandas as pd
import pytest

from market_engine.entry import SetupCandidate
from market_engine.execution import Execution
from market_engine.exits import ExitArea
from market_engine.outcome import evaluate_trade
from market_engine.structure import Direction


def make_frame(rows):
    return pd.DataFrame(rows)


def make_candidate(direction):
    return SetupCandidate(
        setup_index=0,
        setup_timestamp=0,
        direction=direction,
        entry_index=1,
        entry_timestamp=1,
        entry_price=100.0,
        invalidation_price=95.0 if direction is Direction.UP else 105.0,
        risk=5.0,
        invalidation_swing_index=10,
    )


def test_long_target_outcome():
    candidate = make_candidate(Direction.UP)
    execution = Execution(100.2, 0.2, 100.2, 95.0, 5.2)
    target = ExitArea("VALID_SWING", 20, 18, 110.0, 9.8)

    frame = make_frame([
        {"high": 101.0, "low": 99.0},
        {"high": 110.0, "low": 100.0},
    ])

    outcome = evaluate_trade(candidate, execution, target, frame)

    assert outcome.status == "TARGET"
    assert outcome.exit_index == 1
    assert outcome.exit_price == 110.0
    assert outcome.pnl == pytest.approx(9.8)
    assert outcome.risk_multiple == pytest.approx(9.8 / 5.2)


def test_short_stop_outcome():
    candidate = make_candidate(Direction.DOWN)
    execution = Execution(100.2, 0.2, 99.8, 105.0, 5.2)
    target = ExitArea("VALID_SWING", 20, 18, 90.0, 9.8)

    frame = make_frame([
        {"high": 101.0, "low": 99.0},
        {"high": 105.0, "low": 99.0},
    ])

    outcome = evaluate_trade(candidate, execution, target, frame)

    assert outcome.status == "STOP"
    assert outcome.exit_index == 1
    assert outcome.exit_price == 105.0
    assert outcome.pnl == pytest.approx(-5.2)
    assert outcome.risk_multiple == pytest.approx(-1.0)


def test_stop_wins_when_stop_and_target_are_hit_same_candle():
    candidate = make_candidate(Direction.UP)
    execution = Execution(100.0, 0.0, 100.0, 95.0, 5.0)
    target = ExitArea("VALID_SWING", 20, 18, 105.0, 5.0)

    frame = make_frame([
        {"high": 101.0, "low": 99.0},
        {"high": 106.0, "low": 94.0},
    ])

    outcome = evaluate_trade(candidate, execution, target, frame)

    assert outcome.status == "STOP"
    assert outcome.exit_index == 1
    assert outcome.risk_multiple == pytest.approx(-1.0)
    assert outcome.both_hit is True


def test_trade_remains_open_when_neither_level_is_hit():
    candidate = make_candidate(Direction.UP)
    execution = Execution(100.0, 0.0, 100.0, 95.0, 5.0)
    target = ExitArea("VALID_SWING", 20, 18, 110.0, 10.0)

    frame = make_frame([
        {"high": 101.0, "low": 99.0},
        {"high": 104.0, "low": 98.0},
    ])

    outcome = evaluate_trade(candidate, execution, target, frame)

    assert outcome.status == "OPEN"
    assert outcome.exit_index is None
    assert outcome.exit_price is None
    assert outcome.pnl is None
    assert outcome.risk_multiple is None
