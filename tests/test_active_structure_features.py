import numpy as np
import pandas as pd

from market_engine.features import add_active_structure_features


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


def test_active_structure_context_uses_confirmation_time():
    result = add_active_structure_features(make_structure_frame())

    assert result.loc[0, "structure_direction"] == 1
    assert result.loc[4, "structure_direction"] == -1

    assert pd.isna(result.loc[3, "structure_distance_to_high"])
    assert np.isclose(result.loc[4, "structure_distance_to_high"], 4.5)
    assert result.loc[4, "structure_bars_since_last_swing"] == 0
    assert result.loc[5, "structure_bars_since_last_swing"] == 1


def test_active_structure_context_tracks_bos_age_without_lookahead():
    result = add_active_structure_features(make_structure_frame())

    assert pd.isna(result.loc[5, "structure_bars_since_last_bos"])
    assert result.loc[6, "structure_bullish_bos"] == 1
    assert result.loc[6, "structure_bars_since_last_bos"] == 0
