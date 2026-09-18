import numpy as np
import pandas as pd

from market_engine.features import (
    add_momentum_features,
    add_volatility_features,
    build_features,
)


def make_sample_frame() -> pd.DataFrame:
    """Create a small deterministic OHLCV dataset for feature tests."""
    timestamps = pd.date_range(
        "2026-01-01 23:00:00",
        periods=20,
        freq="30min",
    )

    close = np.arange(100.0, 120.0)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close - 0.5,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "tick_volume": np.arange(100, 120),
            "real_volume": np.zeros(20),
            "spread": np.full(20, 160),
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
            "timestamp": pd.to_datetime(["2026-01-01 23:00:00"]),
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
            "timestamp": pd.to_datetime(["2026-01-01 23:00:00"]),
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


def test_build_features_contains_initial_features() -> None:
    frame = make_sample_frame()

    result = build_features(frame)

    expected_columns = {
        "candle_range",
        "candle_body",
        "body_ratio",
        "upper_wick_ratio",
        "lower_wick_ratio",
        "close_position",
        "is_momentum_candle",
        "is_bullish",
        "is_bearish",
        "true_range",
        "atr_14",
        "range_to_atr",
    }

    assert expected_columns.issubset(result.columns)
    assert len(result) == len(frame)

