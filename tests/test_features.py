import numpy as np
import pandas as pd
import pytest

from market_engine.features import (
    add_active_structure_features,
    add_ema_features,
    add_fvg_features,
    add_micro_structure_features,
    add_momentum_features,
    add_recent_movement_features,
    add_support_resistance_features,
    add_volatility_features,
    add_volume_features,
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
    frame = make_sample_frame(rows=60)

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










def test_active_structure_features_carry_snapshot_across_inside_candles():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=6,
                freq="30min",
            ),
            "open": [10.0, 11.0, 13.0, 13.0, 11.0, 10.0],
            "high": [12.0, 14.0, 14.5, 13.5, 13.0, 12.0],
            "low": [9.0, 10.0, 11.0, 10.0, 8.0, 8.5],
            "close": [11.0, 13.0, 14.0, 11.0, 9.0, 9.5],
            "tick_volume": [100, 101, 102, 103, 104, 105],
            "real_volume": [0.0] * 6,
            "spread": [160] * 6,
        }
    )

    result = add_active_structure_features(frame)

    # Candle 5 is inside candle 4's range. The active structural
    # high from the preceding snapshot must remain available.
    assert pd.notna(result.loc[5, "structure_distance_to_high"])
    assert result.loc[5, "structure_distance_to_high"] == (
        result.loc[4, "structure_distance_to_high"]
        + result.loc[4, "close"]
        - result.loc[5, "close"]
    )


def test_fvg_is_created_only_when_three_candle_gap_exists() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=5,
                freq="30min",
            ),
            "high": [10.0, 11.0, 12.0, 14.0, 13.0],
            "low": [8.0, 9.0, 11.0, 13.0, 12.0],
            "close": [9.0, 10.0, 11.5, 13.5, 12.5],
        }
    )

    result = add_fvg_features(frame)

    assert result.loc[0, "fvg_present"] == 0
    assert result.loc[1, "fvg_present"] == 0
    assert result.loc[2, "fvg_present"] == 1
    assert result.loc[2, "fvg_direction"] == "BULLISH"
    assert result.loc[2, "fvg_size"] == 1.0
    assert result.loc[2, "fvg_age_bars"] == 0.0
    assert result.loc[2, "fvg_creation_timestamp"] == frame.loc[2, "timestamp"]


def test_fvg_does_not_use_future_candles() -> None:
    frame = pd.DataFrame(
        {
            "high": [10.0, 11.0, 12.0, 14.0],
            "low": [8.0, 9.0, 11.0, 13.0],
            "close": [9.0, 10.0, 11.5, 13.5],
        }
    )

    result = add_fvg_features(frame)

    assert result.loc[0, "fvg_present"] == 0
    assert result.loc[1, "fvg_present"] == 0
    assert result.loc[2, "fvg_present"] == 1


def test_fvg_context_uses_latest_created_zone_and_positional_age() -> None:
    frame = pd.DataFrame(
        {
            "high": [10.0, 11.0, 12.0, 11.0, 13.0, 14.0],
            "low": [8.0, 9.0, 11.0, 10.0, 12.0, 13.0],
            "close": [9.0, 10.0, 10.5, 10.5, 12.5, 13.5],
        }
    )

    result = add_fvg_features(frame)

    assert result.loc[2, "fvg_direction"] == "BULLISH"
    assert result.loc[2, "fvg_size"] == 1.0

    # The same FVG remains active on the following raw candle.
    assert result.loc[3, "fvg_direction"] == "BULLISH"
    assert result.loc[3, "fvg_age_bars"] == 1.0
    assert result.loc[3, "fvg_distance"] == 0.0
    assert result.loc[3, "fvg_position"] == 0.5


def test_fvg_distance_is_zero_inside_zone_and_positive_outside() -> None:
    frame = pd.DataFrame(
        {
            "high": [10.0, 11.0, 12.0, 11.0, 14.0],
            "low": [8.0, 9.0, 11.0, 10.0, 13.0],
            "close": [9.0, 10.0, 10.5, 12.0, 13.5],
        }
    )

    result = add_fvg_features(frame)

    # Row 2 creates bullish FVG [10, 11].
    assert result.loc[2, "fvg_distance"] == 0.0

    # Row 3 keeps the same FVG and closes above it.
    assert result.loc[3, "fvg_distance"] == 1.0
    assert result.loc[3, "fvg_position"] == 2.0


def test_fvg_detects_bearish_gap_and_zone() -> None:
    frame = pd.DataFrame(
        {
            "high": [12.0, 11.0, 10.0],
            "low": [11.0, 9.0, 7.0],
            "close": [11.5, 10.0, 8.0],
        }
    )

    result = add_fvg_features(frame)

    assert result.loc[2, "fvg_present"] == 1
    assert result.loc[2, "fvg_direction"] == "BEARISH"
    assert result.loc[2, "fvg_size"] == 1.0
    assert result.loc[2, "fvg_age_bars"] == 0.0


def test_fvg_atr_measurements_use_creation_and_current_atr() -> None:
    frame = pd.DataFrame(
        {
            "high": [10.0, 11.0, 12.0, 11.0],
            "low": [8.0, 9.0, 11.0, 10.0],
            "close": [9.0, 10.0, 10.5, 12.0],
            "atr_14": [2.0, 2.0, 2.0, 0.5],
        }
    )

    result = add_fvg_features(frame)

    # FVG [10, 11] is created at row 2 with ATR 2.0.
    assert result.loc[2, "fvg_size_atr"] == 0.5

    # Row 3 is above the zone by 1.0, using current ATR 0.5.
    assert result.loc[3, "fvg_distance"] == 1.0
    assert result.loc[3, "fvg_distance_atr"] == 2.0


def test_fvg_latest_created_zone_replaces_previous_context() -> None:
    frame = pd.DataFrame(
        {
            "high": [10.0, 11.0, 12.0, 14.0, 15.0],
            "low": [8.0, 9.0, 11.0, 13.0, 11.5],
            "close": [9.0, 10.0, 11.5, 13.5, 11.75],
        }
    )

    result = add_fvg_features(frame)

    assert result.loc[2, "fvg_direction"] == "BULLISH"
    assert result.loc[2, "fvg_size"] == 1.0

    # Row 3 creates a newer bullish FVG [11, 13].
    assert result.loc[3, "fvg_direction"] == "BULLISH"
    assert result.loc[3, "fvg_size"] == 2.0
    assert result.loc[3, "fvg_age_bars"] == 0.0

    assert result.loc[4, "fvg_size"] == 2.0
    assert result.loc[4, "fvg_age_bars"] == 1.0


def test_build_features_includes_fvg_context() -> None:
    frame = make_sample_frame(rows=60)

    result = build_features(frame)

    expected_columns = {
        "fvg_present",
        "fvg_direction",
        "fvg_size",
        "fvg_size_atr",
        "fvg_age_bars",
        "fvg_distance",
        "fvg_distance_atr",
        "fvg_position",
        "fvg_creation_timestamp",
    }

    assert expected_columns.issubset(result.columns)


def test_fvg_requires_ohlc_context() -> None:
    frame = pd.DataFrame({"close": [100.0]})

    with pytest.raises(ValueError, match="Missing FVG context columns"):
        add_fvg_features(frame)


def test_displacement_features_calculate_normalized_candle_geometry() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01 10:00:00",
                periods=14,
                freq="h",
            ),
            "open": [100.0] * 13 + [100.0],
            "high": [105.0] * 13 + [110.0],
            "low": [100.0] * 13 + [90.0],
            "close": [105.0] * 13 + [108.0],
            "tick_volume": [100] * 14,
            "real_volume": [0] * 14,
            "spread": [160] * 14,
        }
    )

    result = add_momentum_features(frame)
    result = add_volatility_features(result)

    assert result.loc[13, "candle_range"] == 20.0
    assert result.loc[13, "candle_body"] == 8.0

    expected_atr = result.loc[13, "atr_14"]

    assert result.loc[13, "range_atr"] == (
        result.loc[13, "candle_range"] / expected_atr
    )
    assert result.loc[13, "body_atr"] == (
        result.loc[13, "candle_body"] / expected_atr
    )

    assert result.loc[13, "upper_wick_to_body"] == 0.25
    assert result.loc[13, "lower_wick_to_body"] == 1.25

    assert result.loc[13, "close_position_in_range"] == 0.9


def test_displacement_features_do_not_create_infinite_wick_body_ratios() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01 23:00:00"]),
            "open": [100.0],
            "high": [110.0],
            "low": [90.0],
            "close": [100.0],
            "tick_volume": [100],
            "real_volume": [0],
            "spread": [160],
            "atr_14": [5.0],
        }
    )

    result = add_momentum_features(frame)
    result = add_volatility_features(result)

    assert pd.isna(result.loc[0, "upper_wick_to_body"])
    assert pd.isna(result.loc[0, "lower_wick_to_body"])


def test_displacement_features_are_present_in_full_feature_build() -> None:
    frame = make_sample_frame(rows=60)

    result = build_features(frame)

    expected_columns = {
        "range_atr",
        "body_atr",
        "upper_wick_to_body",
        "lower_wick_to_body",
        "close_position_in_range",
    }

    assert expected_columns.issubset(result.columns)
