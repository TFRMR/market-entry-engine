import numpy as np
import pandas as pd
import pytest

from market_engine.features import (
    add_active_structure_features,
    add_fvg_features,
    add_micro_structure_features,
    add_pullback_features,
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
        "fvg_age_bars",
        "fvg_distance",
        "fvg_position",
        "fvg_creation_timestamp",
    }

    assert expected_columns.issubset(result.columns)


def test_fvg_requires_ohlc_context() -> None:
    frame = pd.DataFrame({"close": [100.0]})

    with pytest.raises(ValueError, match="Missing FVG context columns"):
        add_fvg_features(frame)


def test_displacement_features_are_present_in_full_feature_build() -> None:
    frame = make_sample_frame(rows=60)

    result = build_features(frame)

    expected_columns = {
        "candle_range",
        "candle_body",
        "body_ratio",
        "upper_wick_ratio",
        "lower_wick_ratio",
        "close_position_in_range",
        "upper_wick_to_body",
        "lower_wick_to_body",
    }

    assert expected_columns.issubset(result.columns)


def test_pullback_features_project_active_candidate():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=5,
                freq="30min",
            ),
            "open": [10, 11, 13, 13, 12],
            "high": [12, 14, 14.5, 14, 14.2],
            "low": [9, 10, 11, 10.5, 10.8],
            "close": [11, 13, 14, 11, 13],
        }
    )

    result = add_pullback_features(frame)

    assert result.loc[3, "pullback_active"] == 1.0
    assert result.loc[3, "pullback_direction"] == 1.0
    assert result.loc[3, "pullback_start_index"] == 3
    assert result.loc[3, "pullback_extreme_index"] == 2
    assert result.loc[3, "pullback_price"] == 10.5
    assert result.loc[3, "pullback_depth"] == 4.0
    assert result.loc[3, "pullback_bars"] == 0


def test_pullback_features_update_candidate_without_validating_swing():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=5,
                freq="30min",
            ),
            "open": [10, 11, 13, 13, 12],
            "high": [12, 14, 14.5, 14, 14.2],
            "low": [9, 10, 11, 10.5, 10.8],
            "close": [11, 13, 14, 11, 13],
        }
    )

    result = add_pullback_features(frame)

    assert result.loc[3, "pullback_active"] == 1.0
    assert result.loc[4, "pullback_active"] == 1.0
    assert result.loc[4, "pullback_start_index"] == 3
    assert result.loc[4, "pullback_extreme_index"] == 2
    assert result.loc[4, "pullback_price"] == 10.8
    assert result.loc[4, "pullback_depth"] == pytest.approx(3.7)
    assert result.loc[4, "pullback_bars"] == 1


def test_pullback_features_clear_when_candidate_breaks_and_validates_swing():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=5,
                freq="30min",
            ),
            "open": [10, 11, 13, 13, 12],
            "high": [12, 14, 14.5, 14, 13.5],
            "low": [9, 10, 11, 10.5, 10],
            "close": [11, 13, 14, 11, 10.5],
        }
    )

    result = add_pullback_features(frame)

    assert result.loc[3, "pullback_active"] == 1.0
    assert result.loc[4, "pullback_active"] == 0.0
    assert pd.isna(result.loc[4, "pullback_direction"])
    assert pd.isna(result.loc[4, "pullback_price"])
    assert pd.isna(result.loc[4, "pullback_depth"])
    assert pd.isna(result.loc[4, "pullback_bars"])
