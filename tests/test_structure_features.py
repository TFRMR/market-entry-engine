import pandas as pd

from market_engine.features import add_structure_event_features


def make_structure_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=7, freq="30min"),
            "open": [10, 11, 13, 13, 11, 10, 12],
            "high": [12, 14, 14.5, 13.5, 12, 13, 15],
            "low": [9, 10, 11, 10, 9, 10, 8],
            "close": [11, 13, 14, 11, 10, 12, 14],
        }
    )


def test_structure_features_are_available_only_from_confirmation_time():
    result = add_structure_event_features(make_structure_frame())

    assert result.loc[3, "structure_swing_high_valid"] == 0
    assert result.loc[4, "structure_swing_high_valid"] == 1
    assert pd.isna(result.loc[3, "structure_last_valid_high"])
    assert result.loc[4, "structure_last_valid_high"] == 14.5
    assert result.loc[5, "structure_last_valid_high"] == 14.5


def test_structure_features_expose_bos_and_swing_flags():
    result = add_structure_event_features(make_structure_frame())

    assert result.loc[6, "structure_bullish_bos"] == 1
    assert result.loc[6, "structure_bearish_bos"] == 0
    assert result.loc[4, "structure_internal_swing"] == 0
    assert result.loc[4, "structure_external_swing"] == 1
