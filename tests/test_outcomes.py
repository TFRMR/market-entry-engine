import numpy as np
import pandas as pd
import pytest

from market_engine.outcomes import add_forward_returns


def make_sample_frame(rows: int = 15) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2026-01-01",
        periods=rows,
        freq="30min",
    )

    close = np.arange(100.0, 100.0 + rows)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "tick_volume": np.arange(1, rows + 1),
            "real_volume": 0,
            "spread": 200,
        }
    )


def test_forward_returns_use_future_close() -> None:
    frame = make_sample_frame()

    result = add_forward_returns(frame)

    row = 3

    for horizon in (1, 3, 5, 10):
        expected = (
            frame.loc[row + horizon, "close"]
            / frame.loc[row, "close"]
        ) - 1

        assert np.isclose(
            result.loc[row, f"forward_return_{horizon}"],
            expected,
        )


def test_forward_returns_are_nan_at_end() -> None:
    frame = make_sample_frame(rows=15)

    result = add_forward_returns(frame)

    assert result["forward_return_1"].iloc[-1:].isna().all()
    assert result["forward_return_3"].iloc[-3:].isna().all()
    assert result["forward_return_5"].iloc[-5:].isna().all()
    assert result["forward_return_10"].iloc[-10:].isna().all()


def test_forward_returns_preserve_rows() -> None:
    frame = make_sample_frame()

    result = add_forward_returns(frame)

    assert len(result) == len(frame)


def test_forward_returns_reject_invalid_horizons() -> None:
    frame = make_sample_frame()

    with pytest.raises(ValueError, match="greater than zero"):
        add_forward_returns(frame, horizons=(1, 0, 5))


def test_forward_returns_reject_empty_horizons() -> None:
    frame = make_sample_frame()

    with pytest.raises(ValueError, match="must not be empty"):
        add_forward_returns(frame, horizons=())
