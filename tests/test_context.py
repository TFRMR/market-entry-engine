import numpy as np
import pandas as pd
import pytest

from market_engine.features import add_trend_range_features


def make_context_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "close": [105.0, 110.0, 115.0, 125.0],
            "structure_direction": [0.0, 1.0, 1.0, -1.0],
            "structure_last_valid_high": [np.nan, 120.0, 120.0, 120.0],
            "structure_last_valid_low": [np.nan, 100.0, 100.0, 100.0],
            "structure_bullish_choch": [0, 0, 0, 1],
            "structure_bearish_choch": [0, 0, 0, 0],
        }
    )


def test_trend_regime_uses_deterministic_structure_direction() -> None:
    result = add_trend_range_features(make_context_frame())

    assert result["trend_regime"].tolist() == [
        "NEUTRAL",
        "BULLISH",
        "BULLISH",
        "BEARISH",
    ]


def test_choch_marks_trend_transition() -> None:
    result = add_trend_range_features(make_context_frame())

    assert result["trend_transition"].tolist() == [0, 0, 0, 1]
    assert np.isnan(result.loc[0, "trend_transition_direction"])
    assert result.loc[3, "trend_transition_direction"] == 1.0


def test_structural_range_position_is_as_of_current_state() -> None:
    result = add_trend_range_features(make_context_frame())

    assert result.loc[1, "range_state"] == "DEFINED"
    assert result.loc[1, "range_high"] == 120.0
    assert result.loc[1, "range_low"] == 100.0
    assert result.loc[1, "range_width"] == 20.0
    assert result.loc[1, "range_position"] == 0.5
    assert result.loc[1, "range_position_zone"] == "MID"

    assert result.loc[3, "range_position"] == 1.25
    assert result.loc[3, "range_position_zone"] == "HIGH"


def test_undefined_range_does_not_create_position() -> None:
    frame = make_context_frame().iloc[[0]].copy()

    result = add_trend_range_features(frame)

    assert result.loc[0, "range_state"] == "UNDEFINED"
    assert pd.isna(result.loc[0, "range_width"])
    assert pd.isna(result.loc[0, "range_position"])
    assert result.loc[0, "range_position_zone"] == "UNDEFINED"


def test_trend_range_requires_structural_columns() -> None:
    frame = pd.DataFrame({"close": [100.0]})

    with pytest.raises(ValueError, match="Missing structural context columns"):
        add_trend_range_features(frame)
