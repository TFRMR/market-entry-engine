import numpy as np
import pandas as pd

from market_engine.features import (
    add_ema_features,
    add_momentum_features,
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


def test_build_features_contains_initial_and_ema_features() -> None:
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
        "ema_5",
        "ema_20",
        "ema_50",
        "price_vs_ema_5",
        "price_vs_ema_20",
        "price_vs_ema_50",
        "ema_5_vs_20",
        "ema_20_vs_50",
        "ema_alignment",
    }

    assert expected_columns.issubset(result.columns)
    assert len(result) == len(frame)
