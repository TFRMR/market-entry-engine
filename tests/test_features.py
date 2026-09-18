import numpy as np
import pandas as pd
import pytest

from market_engine.features import (
    add_ema_features,
    add_momentum_features,
    add_support_resistance_features,
    add_volatility_features,
    build_features,
)


def make_sample_frame(rows: int = 60) -> pd.DataFrame:
    """Create a deterministic OHLCV dataset for feature tests."""
    timestamps = pd.date_range(
        "2026-01-01 23:00:00",
        periods=rows,
        freq="30min",
    )

    close = np.arange(100.0, 100.0 + rows)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close - 0.5,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "tick_volume": np.arange(100, 100 + rows),
            "real_volume": np.zeros(rows),
            "spread": np.full(rows, 160),
        }
    )


def test_momentum_features_calculate_body_ratio() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 23:00:00",
                    "2026-01-01 23:30:00",
                ]
            ),
            "open": [100.0, 100.0],
            "high": [101.0, 110.0],
            "low": [99.0, 99.0],
            "close": [100.5, 109.5],
            "tick_volume": [100, 100],
            "real_volume": [0, 0],
            "spread": [160, 160],
        }
    )

    result = add_momentum_features(frame)

    assert result.loc[0, "body_ratio"] == 0.25
    assert np.isclose(result.loc[1, "body_ratio"], 9.5 / 11)

    assert result.loc[0, "is_momentum_candle"] == False
    assert result.loc[1, "is_momentum_candle"] == True

    assert result.loc[1, "is_bullish"] == True
    assert result.loc[1, "is_bearish"] == False


def test_momentum_features_calculate_wicks_and_close_position() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01 23:00:00"]
            ),
            "open": [100.0],
            "high": [110.0],
            "low": [90.0],
            "close": [108.0],
            "tick_volume": [100],
            "real_volume": [0],
            "spread": [160],
        }
    )

    result = add_momentum_features(frame)

    assert result.loc[0, "candle_range"] == 20.0
    assert result.loc[0, "candle_body"] == 8.0
    assert result.loc[0, "body_ratio"] == 0.4

    assert result.loc[0, "upper_wick_ratio"] == 0.1
    assert result.loc[0, "lower_wick_ratio"] == 0.5
    assert result.loc[0, "close_position"] == 0.9


def test_degenerate_candle_does_not_produce_invalid_ratios() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01 23:00:00"]
            ),
            "open": [100.0],
            "high": [100.0],
            "low": [100.0],
            "close": [100.0],
            "tick_volume": [100],
            "real_volume": [0],
            "spread": [160],
        }
    )

    result = add_momentum_features(frame)

    assert pd.isna(result.loc[0, "body_ratio"])
    assert pd.isna(result.loc[0, "upper_wick_ratio"])
    assert pd.isna(result.loc[0, "lower_wick_ratio"])
    assert pd.isna(result.loc[0, "close_position"])

    assert result.loc[0, "is_momentum_candle"] == False


def test_atr_uses_true_range_and_requires_history() -> None:
    frame = make_sample_frame()

    result = add_momentum_features(frame)
    result = add_volatility_features(result)

    assert result["atr_14"].iloc[:13].isna().all()
    assert result["atr_14"].iloc[13] > 0

    expected_atr = result["true_range"].iloc[:14].mean()

    assert np.isclose(
        result["atr_14"].iloc[13],
        expected_atr,
    )


def test_ema_features_require_expected_history() -> None:
    frame = make_sample_frame()

    result = add_ema_features(frame)

    assert result["ema_5"].iloc[:4].isna().all()
    assert result["ema_5"].iloc[4:].notna().all()

    assert result["ema_20"].iloc[:19].isna().all()
    assert result["ema_20"].iloc[19:].notna().all()

    assert result["ema_50"].iloc[:49].isna().all()
    assert result["ema_50"].iloc[49:].notna().all()


def test_ema_features_calculate_price_and_ema_relationships() -> None:
    frame = make_sample_frame()

    result = add_ema_features(frame)

    row = 59

    assert result.loc[row, "ema_5"] > 0
    assert result.loc[row, "ema_20"] > 0
    assert result.loc[row, "ema_50"] > 0

    assert np.isclose(
        result.loc[row, "price_vs_ema_5"],
        result.loc[row, "close"] - result.loc[row, "ema_5"],
    )

    assert np.isclose(
        result.loc[row, "price_vs_ema_20"],
        result.loc[row, "close"] - result.loc[row, "ema_20"],
    )

    assert np.isclose(
        result.loc[row, "price_vs_ema_50"],
        result.loc[row, "close"] - result.loc[row, "ema_50"],
    )

    assert np.isclose(
        result.loc[row, "ema_5_vs_20"],
        result.loc[row, "ema_5"] - result.loc[row, "ema_20"],
    )

    assert np.isclose(
        result.loc[row, "ema_20_vs_50"],
        result.loc[row, "ema_20"] - result.loc[row, "ema_50"],
    )


def test_ema_alignment_is_bullish_for_rising_market() -> None:
    frame = make_sample_frame()

    result = add_ema_features(frame)

    assert result.loc[59, "ema_5"] > result.loc[59, "ema_20"]
    assert result.loc[59, "ema_20"] > result.loc[59, "ema_50"]
    assert result.loc[59, "ema_alignment"] == 1


def test_support_resistance_requires_prior_history() -> None:
    frame = make_sample_frame()

    result = add_support_resistance_features(frame)

    # Current candle must not be included.
    # Therefore 20 previous candles are required before the first
    # 20-candle level becomes available.
    assert result["previous_high_20"].iloc[:20].isna().all()
    assert result["previous_low_20"].iloc[:20].isna().all()

    # Same principle for the 50-candle levels.
    assert result["previous_high_50"].iloc[:50].isna().all()
    assert result["previous_low_50"].iloc[:50].isna().all()

    assert result["previous_high_20"].iloc[20:].notna().all()
    assert result["previous_low_20"].iloc[20:].notna().all()

    assert result["previous_high_50"].iloc[50:].notna().all()
    assert result["previous_low_50"].iloc[50:].notna().all()


def test_support_resistance_uses_only_previous_candles() -> None:
    frame = make_sample_frame(rows=25)

    result = add_support_resistance_features(
        frame,
        short_window=5,
        long_window=10,
    )

    row = 10

    expected_high_5 = frame["high"].iloc[row - 5:row].max()
    expected_low_5 = frame["low"].iloc[row - 5:row].min()

    expected_high_10 = frame["high"].iloc[row - 10:row].max()
    expected_low_10 = frame["low"].iloc[row - 10:row].min()

    assert result.loc[row, "previous_high_20"] == expected_high_5
    assert result.loc[row, "previous_low_20"] == expected_low_5

    assert result.loc[row, "previous_high_50"] == expected_high_10
    assert result.loc[row, "previous_low_50"] == expected_low_10


def test_support_resistance_does_not_leak_current_candle() -> None:
    frame = make_sample_frame(rows=25)

    # Make the current candle an extreme outlier.
    row = 20
    frame.loc[row, "high"] = 1000.0
    frame.loc[row, "low"] = 1.0

    result = add_support_resistance_features(
        frame,
        short_window=5,
        long_window=10,
    )

    # The current extreme values must NOT appear in the prior levels.
    assert result.loc[row, "previous_high_20"] < 1000.0
    assert result.loc[row, "previous_low_20"] > 1.0

    assert result.loc[row, "previous_high_50"] < 1000.0
    assert result.loc[row, "previous_low_50"] > 1.0


def test_support_resistance_breakout_flags() -> None:
    frame = make_sample_frame(rows=25)

    row = 20

    previous_high = frame["high"].iloc[row - 5:row].max()
    previous_low = frame["low"].iloc[row - 5:row].min()

    frame.loc[row, "high"] = previous_high + 10.0
    frame.loc[row, "low"] = previous_low - 10.0

    result = add_support_resistance_features(
        frame,
        short_window=5,
        long_window=10,
    )

    assert result.loc[row, "breakout_above_20"] == True
    assert result.loc[row, "breakout_below_20"] == True


def test_support_resistance_rejects_invalid_windows() -> None:
    frame = make_sample_frame()

    with pytest.raises(ValueError):
        add_support_resistance_features(
            frame,
            short_window=0,
            long_window=10,
        )

    with pytest.raises(ValueError):
        add_support_resistance_features(
            frame,
            short_window=20,
            long_window=20,
        )

    with pytest.raises(ValueError):
        add_support_resistance_features(
            frame,
            short_window=30,
            long_window=20,
        )


def test_build_features_contains_support_resistance_features() -> None:
    frame = make_sample_frame()

    result = build_features(frame)

    expected_columns = {
        "previous_high_20",
        "previous_low_20",
        "previous_high_50",
        "previous_low_50",
        "distance_to_high_20",
        "distance_to_low_20",
        "distance_to_high_50",
        "distance_to_low_50",
        "breakout_above_20",
        "breakout_below_20",
        "breakout_above_50",
        "breakout_below_50",
    }

    assert expected_columns.issubset(result.columns)
    assert len(result) == len(frame)
