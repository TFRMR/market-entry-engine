import numpy as np
import pandas as pd
import pytest

from market_engine.entry import SetupCandidate
from market_engine.realized import (
    build_realized_r_frame,
    cluster_bootstrap_mean,
    simulate_fixed_geometry_r,
)
from market_engine.structure import Direction, SwingType, ValidSwing


def swing(index, confirmation_index, price, swing_type):
    return ValidSwing(
        index=index,
        timestamp=index,
        price=price,
        swing_type=swing_type,
        confirmation_index=confirmation_index,
        confirmation_timestamp=confirmation_index,
    )


def candles(rows=12, high=101.0, low=99.0):
    return pd.DataFrame(
        {
            "open": [100.0] * rows,
            "high": [high] * rows,
            "low": [low] * rows,
            "close": [100.0] * rows,
        }
    )


def up_candidate():
    return SetupCandidate(
        setup_index=5,
        setup_timestamp=5,
        direction=Direction.UP,
        entry_index=6,
        entry_timestamp=6,
        entry_price=100.0,
        invalidation_price=95.0,
        risk=5.0,
        invalidation_swing_index=0,
    )


def test_unresolved_setup_exits_at_the_horizon_close():
    frame = candles()
    frame.loc[8, "close"] = 102.0
    swings = [swing(1, 2, 110.0, SwingType.HIGH)]

    result = build_realized_r_frame(
        [up_candidate()], swings, [], frame, spread_price=0.2, horizon=3
    )

    assert list(result["exit_kind"]) == ["TIME_STOP"]
    assert result.loc[0, "realized_r"] == pytest.approx((102.0 - 100.2) / 5.2)


def test_target_inside_the_spread_has_negative_realized_r():
    frame = candles()
    frame.loc[:5, "high"] = 100.0
    swings = [swing(1, 2, 100.1, SwingType.HIGH)]

    result = build_realized_r_frame(
        [up_candidate()], swings, [], frame, spread_price=0.3, horizon=3
    )

    assert list(result["exit_kind"]) == ["TARGET"]
    assert result.loc[0, "realized_r"] == pytest.approx((100.1 - 100.3) / 5.3)
    assert result.loc[0, "reward_r_after_spread"] < 0


def test_stop_realizes_minus_one_r():
    frame = candles(low=94.0)
    swings = [swing(1, 2, 110.0, SwingType.HIGH)]

    result = build_realized_r_frame(
        [up_candidate()], swings, [], frame, spread_price=0.2, horizon=3
    )

    assert list(result["exit_kind"]) == ["STOP"]
    assert result.loc[0, "realized_r"] == pytest.approx(-1.0)


def one_trade_r(frame, is_up, risk, reward, spread=0.0, horizon=3):
    return simulate_fixed_geometry_r(
        frame,
        np.array([0]),
        np.array([is_up]),
        np.array([risk]),
        np.array([reward]),
        spread,
        horizon,
    )[0]


def test_fixed_geometry_simulation_covers_target_stop_tie_and_time_stop():
    hit_target = candles(rows=5, high=106.0, low=99.0)
    assert one_trade_r(hit_target, True, 5.0, 5.0) == pytest.approx(1.0)

    hit_stop = candles(rows=5, high=101.0, low=94.0)
    assert one_trade_r(hit_stop, True, 5.0, 5.0) == pytest.approx(-1.0)

    same_candle_both = candles(rows=5, high=106.0, low=94.0)
    assert one_trade_r(same_candle_both, True, 5.0, 5.0) == pytest.approx(-1.0)

    unresolved = candles(rows=5)
    unresolved.loc[2, "close"] = 102.0
    assert one_trade_r(unresolved, True, 5.0, 5.0) == pytest.approx(0.4)

    short_target = candles(rows=5, high=101.0, low=94.0)
    assert one_trade_r(short_target, False, 5.0, 5.0) == pytest.approx(1.0)

    with_spread = candles(rows=5, high=106.0, low=99.0)
    assert one_trade_r(with_spread, True, 5.0, 5.0, spread=0.5) == pytest.approx(
        (5.0 - 0.5) / 5.5
    )


def test_simulation_rejects_windows_beyond_the_frame():
    with pytest.raises(ValueError):
        one_trade_r(candles(rows=2), True, 5.0, 5.0, horizon=3)


def test_cluster_bootstrap_of_constant_values_is_degenerate():
    mean, low, high = cluster_bootstrap_mean(
        np.full(6, 0.5), np.array([1, 1, 2, 2, 3, 3]), draws=200
    )

    assert mean == low == high == pytest.approx(0.5)


def test_cluster_bootstrap_of_empty_values_is_nan():
    mean, low, high = cluster_bootstrap_mean(np.array([]), np.array([]))

    assert np.isnan(mean) and np.isnan(low) and np.isnan(high)
