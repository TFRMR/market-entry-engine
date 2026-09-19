import numpy as np
import pandas as pd
import pytest

from market_engine.outcomes import (
    add_barrier_outcomes,
    add_forward_excursions,
    add_forward_returns,
)


def make_frame() -> pd.DataFrame:
    """Create a small deterministic candle set for outcome tests."""
    return pd.DataFrame(
        {
            "open": [100.0, 104.0, 103.0, 101.0, 99.0, 98.0],
            "high": [105.0, 106.0, 105.0, 102.0, 100.0, 99.0],
            "low": [99.0, 103.0, 100.0, 97.0, 96.0, 95.0],
            "close": [104.0, 103.0, 101.0, 99.0, 98.0, 96.0],
            "is_momentum_candle": [True, False, False, False, False, False],
        }
    )


def test_forward_returns_add_expected_columns() -> None:
    frame = pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0, 104.0, 108.0],
        }
    )

    result = add_forward_returns(frame, horizons=(1, 3))

    assert "forward_return_1" in result.columns
    assert "forward_return_3" in result.columns

    assert result.loc[0, "forward_return_1"] == pytest.approx(0.01)
    assert result.loc[0, "forward_return_3"] == pytest.approx(0.04)

    assert np.isnan(result.loc[4, "forward_return_1"])
    assert np.isnan(result.loc[2, "forward_return_3"])


def test_forward_returns_preserve_original_columns() -> None:
    frame = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "close": [101.0, 102.0, 103.0],
        }
    )

    result = add_forward_returns(frame, horizons=(1,))

    assert list(result.columns) == [
        "open",
        "close",
        "forward_return_1",
    ]


def test_forward_returns_reject_invalid_horizons() -> None:
    frame = pd.DataFrame({"close": [100.0, 101.0]})

    with pytest.raises(ValueError):
        add_forward_returns(frame, horizons=())

    with pytest.raises(ValueError):
        add_forward_returns(frame, horizons=(0, 1))

    with pytest.raises(ValueError):
        add_forward_returns(frame, horizons=(-1, 1))


def test_bullish_excursion_is_normalized_by_candle_risk() -> None:
    frame = make_frame()
    result = add_forward_excursions(frame, horizons=(1, 3))

    # Entry 104, SL 99 => risk 5.
    # Next candle: high 106, low 103.
    assert result.loc[0, "entry_price"] == pytest.approx(104.0)
    assert result.loc[0, "stop_price"] == pytest.approx(99.0)
    assert result.loc[0, "risk_price"] == pytest.approx(5.0)

    assert result.loc[0, "mfe_r_1"] == pytest.approx(0.4)
    assert result.loc[0, "mae_r_1"] == pytest.approx(0.2)

    # Next 3 candles: highest high 106, lowest low 96.
    assert result.loc[0, "mfe_r_3"] == pytest.approx(0.4)
    assert result.loc[0, "mae_r_3"] == pytest.approx(1.4)


def test_bearish_excursion_uses_reversed_direction() -> None:
    frame = make_frame()

    # Turn row 1 into a bearish momentum candle.
    frame.loc[1, "open"] = 104.0
    frame.loc[1, "high"] = 106.0
    frame.loc[1, "low"] = 100.0
    frame.loc[1, "close"] = 103.0
    frame.loc[1, "is_momentum_candle"] = True

    result = add_forward_excursions(frame, horizons=(1,))

    # Entry 103, SL 106 => risk 3.
    # Next candle: low 100, high 105.
    assert result.loc[1, "entry_price"] == pytest.approx(103.0)
    assert result.loc[1, "stop_price"] == pytest.approx(106.0)
    assert result.loc[1, "risk_price"] == pytest.approx(3.0)

    assert result.loc[1, "mfe_r_1"] == pytest.approx(1.0)
    assert result.loc[1, "mae_r_1"] == pytest.approx(2 / 3)


def test_non_momentum_rows_are_nan() -> None:
    frame = make_frame()
    result = add_forward_excursions(frame, horizons=(1,))

    assert np.isnan(result.loc[1, "mfe_r_1"])
    assert np.isnan(result.loc[1, "mae_r_1"])


def test_excursion_tail_is_nan_when_horizon_is_unavailable() -> None:
    frame = make_frame()
    result = add_forward_excursions(frame, horizons=(3,))

    assert np.isnan(result.loc[4, "mfe_r_3"])
    assert np.isnan(result.loc[5, "mae_r_3"])


def test_excursions_reject_invalid_input() -> None:
    frame = make_frame()

    with pytest.raises(ValueError):
        add_forward_excursions(frame, horizons=())

    with pytest.raises(ValueError):
        add_forward_excursions(frame, horizons=(0, 3))

    with pytest.raises(ValueError):
        add_forward_excursions(
            frame.drop(columns=["is_momentum_candle"])
        )


def test_barrier_outcome_detects_bullish_tp_first() -> None:
    frame = pd.DataFrame(
        {
            "open": [100.0, 104.0],
            "high": [105.0, 110.0],
            "low": [99.0, 103.0],
            "close": [104.0, 109.0],
            "is_momentum_candle": [True, False],
        }
    )

    result = add_barrier_outcomes(
        frame,
        horizons=(1,),
        targets_r=(1.0,),
    )

    assert result.loc[0, "barrier_1r_1"] == "TP_FIRST"


def test_barrier_outcome_detects_bullish_sl_first() -> None:
    frame = make_frame()

    result = add_barrier_outcomes(
        frame,
        horizons=(3,),
        targets_r=(2.0,),
    )

    assert result.loc[0, "barrier_2r_3"] == "SL_FIRST"


def test_barrier_outcome_detects_both_same_candle() -> None:
    frame = pd.DataFrame(
        {
            "open": [100.0, 100.0],
            "high": [105.0, 110.0],
            "low": [99.0, 98.0],
            "close": [104.0, 100.0],
            "is_momentum_candle": [True, False],
        }
    )

    result = add_barrier_outcomes(
        frame,
        horizons=(1,),
        targets_r=(1.0,),
    )

    assert result.loc[0, "barrier_1r_1"] == "BOTH_SAME_CANDLE"


def test_barrier_outcome_requires_valid_directional_setup() -> None:
    frame = pd.DataFrame(
        {
            "open": [100.0, 100.0],
            "high": [100.0, 101.0],
            "low": [100.0, 99.0],
            "close": [100.0, 100.0],
            "is_momentum_candle": [True, False],
        }
    )

    result = add_barrier_outcomes(
        frame,
        horizons=(1,),
        targets_r=(1.0,),
    )

    assert pd.isna(result.loc[0, "barrier_1r_1"])
