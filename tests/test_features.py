import numpy as np
import pandas as pd
import pytest

from market_engine.features import (
    add_active_structure_features,
    add_ema_features,
    add_micro_structure_features,
    add_momentum_features,
    add_recent_movement_features,
    add_volume_features,
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

    assert result["previous_high_20"].iloc[:20].isna().all()
    assert result["previous_low_20"].iloc[:20].isna().all()

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

    row = 20
    frame.loc[row, "high"] = 1000.0
    frame.loc[row, "low"] = 1.0

    result = add_support_resistance_features(
        frame,
        short_window=5,
        long_window=10,
    )

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


def test_recent_movement_features_calculate_returns() -> None:
    frame = make_sample_frame(rows=20)

    result = add_recent_movement_features(frame)

    assert result["return_3"].iloc[:3].isna().all()
    assert result["return_6"].iloc[:6].isna().all()
    assert result["return_12"].iloc[:12].isna().all()

    row = 12

    expected_return_3 = (
        frame.loc[row, "close"] / frame.loc[row - 3, "close"]
    ) - 1

    expected_return_6 = (
        frame.loc[row, "close"] / frame.loc[row - 6, "close"]
    ) - 1

    expected_return_12 = (
        frame.loc[row, "close"] / frame.loc[row - 12, "close"]
    ) - 1

    assert np.isclose(
        result.loc[row, "return_3"],
        expected_return_3,
    )

    assert np.isclose(
        result.loc[row, "return_6"],
        expected_return_6,
    )

    assert np.isclose(
        result.loc[row, "return_12"],
        expected_return_12,
    )


def test_build_features_contains_recent_movement_features() -> None:
    frame = make_sample_frame()

    result = build_features(frame)

    expected_columns = {
        "return_3",
        "return_6",
        "return_12",
    }

    assert expected_columns.issubset(result.columns)
    assert len(result) == len(frame)


def test_volume_features_calculate_context() -> None:
    frame = make_sample_frame(rows=25)
    frame["tick_volume"] = np.arange(1, 26, dtype=float)

    result = add_volume_features(frame)

    assert result["volume_ratio_20"].iloc[:20].isna().all()
    assert result["volume_change_1"].iloc[:1].isna().all()

    row = 20

    expected_average = frame.loc[0:19, "tick_volume"].mean()
    expected_ratio = frame.loc[row, "tick_volume"] / expected_average
    expected_change = (
        frame.loc[row, "tick_volume"] / frame.loc[row - 1, "tick_volume"]
    ) - 1

    assert np.isclose(
        result.loc[row, "volume_ratio_20"],
        expected_ratio,
    )

    assert np.isclose(
        result.loc[row, "volume_change_1"],
        expected_change,
    )


def test_build_features_contains_volume_features() -> None:
    frame = make_sample_frame()

    result = build_features(frame)

    expected_columns = {
        "volume_ratio_20",
        "volume_change_1",
    }

    assert expected_columns.issubset(result.columns)
    assert len(result) == len(frame)


def test_volume_features_reject_invalid_window() -> None:
    frame = make_sample_frame()

    with pytest.raises(ValueError, match="volume_window"):
        add_volume_features(frame, volume_window=0)


def test_micro_structure_features_calculate_context() -> None:
    frame = make_sample_frame(rows=10)

    result = add_micro_structure_features(frame)

    expected_columns = {
        "high_vs_previous_high_1",
        "high_vs_previous_high_2",
        "high_vs_previous_high_3",
        "low_vs_previous_low_1",
        "low_vs_previous_low_2",
        "low_vs_previous_low_3",
        "range_vs_avg_3",
        "range_vs_avg_5",
    }

    assert expected_columns.issubset(result.columns)
    assert len(result) == len(frame)

    assert result["high_vs_previous_high_1"].iloc[:1].isna().all()
    assert result["high_vs_previous_high_2"].iloc[:2].isna().all()
    assert result["high_vs_previous_high_3"].iloc[:3].isna().all()
    assert result["range_vs_avg_3"].iloc[:3].isna().all()
    assert result["range_vs_avg_5"].iloc[:5].isna().all()


def test_active_structure_features_use_positional_candle_age() -> None:
    frame = make_sample_frame(rows=30)
    shifted = frame.copy()
    shifted.index = np.arange(100, 100 + len(shifted))

    baseline = add_active_structure_features(frame)
    result = add_active_structure_features(shifted)

    assert np.allclose(
        baseline["structure_bars_since_last_swing"].to_numpy(),
        result["structure_bars_since_last_swing"].to_numpy(),
        equal_nan=True,
    )
    assert np.allclose(
        baseline["structure_bars_since_last_bos"].to_numpy(),
        result["structure_bars_since_last_bos"].to_numpy(),
        equal_nan=True,
    )


def test_build_features_contains_micro_structure_features() -> None:
    frame = make_sample_frame()

    result = build_features(frame)

    expected_columns = {
        "high_vs_previous_high_1",
        "high_vs_previous_high_2",
        "high_vs_previous_high_3",
        "low_vs_previous_low_1",
        "low_vs_previous_low_2",
        "low_vs_previous_low_3",
        "range_vs_avg_3",
        "range_vs_avg_5",
    }

    assert expected_columns.issubset(result.columns)
    assert len(result) == len(frame)
