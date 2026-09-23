import numpy as np
import pandas as pd
import pytest

from market_engine.features import (
    add_liquidity_features,
    add_sr_location_features,
    add_trend_range_features,
)
from market_engine.structure import StructureScope, SwingType, ValidSwing


def make_context_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "close": [105.0, 110.0, 115.0, 125.0],
            "structure_direction": [0.0, 1.0, 1.0, -1.0],
            "structure_last_valid_high": [105.0, 120.0, 120.0, 120.0],
            "structure_last_valid_low": [95.0, 100.0, 100.0, 100.0],
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
    assert pd.isna(result.loc[0, "trend_transition_direction"])
    assert result.loc[3, "trend_transition_direction"] == "UP"


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


def make_liquidity_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "high": [105.0, 120.0, 121.0, 101.0],
            "low": [95.0, 100.0, 99.0, 80.0],
            "close": [100.0, 110.0, 101.0, 100.0],
            "structure_last_valid_high": [np.nan, 120.0, 120.0, 120.0],
            "structure_last_valid_low": [np.nan, 100.0, 100.0, 100.0],
        }
    )


def test_liquidity_uses_pre_candle_confirmed_levels() -> None:
    result = add_liquidity_features(make_liquidity_frame())

    # Liquidity must come from confirmed structural levels,
    # not from the previous candle's raw high/low.
    assert pd.isna(result.loc[0, "liquidity_high"])
    assert pd.isna(result.loc[1, "liquidity_high"])
    assert pd.isna(result.loc[1, "liquidity_low"])
    assert result.loc[2, "liquidity_high"] == 120.0
    assert result.loc[2, "liquidity_low"] == 100.0


def test_high_liquidity_sweep_requires_reclaim() -> None:
    result = add_liquidity_features(make_liquidity_frame())

    assert result.loc[2, "liquidity_high_sweep"] == 1
    assert result.loc[2, "liquidity_low_sweep"] == 1
    assert result.loc[2, "liquidity_sweep"] == 1
    assert result.loc[2, "liquidity_sweep_direction"] == "BOTH"
    assert result.loc[2, "liquidity_sweep_size"] == 1.0
    assert result.loc[1, "liquidity_high_sweep"] == 0


def test_low_liquidity_sweep_requires_reclaim() -> None:
    result = add_liquidity_features(make_liquidity_frame())

    assert result.loc[3, "liquidity_low_sweep"] == 1
    assert result.loc[3, "liquidity_sweep"] == 1
    assert result.loc[3, "liquidity_sweep_direction"] == "BULLISH"
    assert result.loc[3, "liquidity_sweep_size"] == 20.0


def test_liquidity_requires_structural_levels() -> None:
    frame = pd.DataFrame(
        {
            "high": [105.0],
            "low": [95.0],
            "close": [100.0],
        }
    )

    with pytest.raises(ValueError, match="Missing liquidity context columns"):
        add_liquidity_features(frame)



def make_sr_location_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=5,
                freq="30min",
            ),
            "high": [105.0, 110.0, 115.0, 120.0, 125.0],
            "low": [95.0, 100.0, 105.0, 110.0, 115.0],
            "close": [100.0, 108.0, 100.0, 100.0, 118.0],
            "structure_direction": [1.0, 1.0, 1.0, 1.0, 1.0],
            "range_position": [0.2, 0.3, 0.4, 0.6, 0.8],
        }
    )


def make_sr_swing(
    swing_type: SwingType,
    confirmation_index: int,
    price: float,
) -> ValidSwing:
    timestamp = pd.Timestamp("2026-01-01") + pd.Timedelta(
        minutes=30 * (confirmation_index - 1)
    )
    confirmation_timestamp = pd.Timestamp("2026-01-01") + pd.Timedelta(
        minutes=30 * confirmation_index
    )

    return ValidSwing(
        index=confirmation_index - 1,
        timestamp=timestamp,
        price=price,
        swing_type=swing_type,
        confirmation_index=confirmation_index,
        confirmation_timestamp=confirmation_timestamp,
        scope=StructureScope.EXTERNAL,
    )


def test_sr_location_uses_confirmed_structural_swings_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = make_sr_location_frame()

    swings = [
        make_sr_swing(SwingType.HIGH, 2, 105.0),
        make_sr_swing(SwingType.LOW, 2, 95.0),
        make_sr_swing(SwingType.HIGH, 4, 120.0),
    ]

    monkeypatch.setattr(
        "market_engine.features.process_structural_candles",
        lambda _frame: (swings, []),
    )
    monkeypatch.setattr(
        "market_engine.features.build_structural_sequence",
        lambda input_frame: input_frame.copy(),
    )

    result = add_sr_location_features(frame)

    # Swing confirmed at candle 2 is unavailable on candles 0 and 1.
    assert pd.isna(result.loc[1, "support_level"])
    assert pd.isna(result.loc[1, "resistance_level"])

    # At confirmation time, both historical levels become available.
    assert result.loc[2, "support_level"] == 95.0
    assert result.loc[2, "resistance_level"] == 105.0

    # The later high is unavailable before its confirmation candle.
    assert result.loc[3, "resistance_level"] == 105.0
    assert result.loc[4, "resistance_level"] == 120.0


def test_sr_location_selects_nearest_structural_levels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = make_sr_location_frame()

    swings = [
        make_sr_swing(SwingType.LOW, 0, 90.0),
        make_sr_swing(SwingType.LOW, 0, 100.0),
        make_sr_swing(SwingType.HIGH, 0, 115.0),
        make_sr_swing(SwingType.HIGH, 0, 130.0),
    ]

    monkeypatch.setattr(
        "market_engine.features.process_structural_candles",
        lambda _frame: (swings, []),
    )
    monkeypatch.setattr(
        "market_engine.features.build_structural_sequence",
        lambda input_frame: input_frame.copy(),
    )

    result = add_sr_location_features(frame.iloc[[0]].copy())

    assert result.loc[0, "support_level"] == 100.0
    assert result.loc[0, "resistance_level"] == 115.0
    assert result.loc[0, "distance_to_support"] == 0.0
    assert result.loc[0, "distance_to_resistance"] == 15.0


def test_sr_location_next_structure_level_follows_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = make_sr_location_frame()

    swings = [
        make_sr_swing(SwingType.LOW, 0, 95.0),
        make_sr_swing(SwingType.HIGH, 0, 105.0),
        make_sr_swing(SwingType.HIGH, 0, 125.0),
        make_sr_swing(SwingType.LOW, 0, 90.0),
    ]

    monkeypatch.setattr(
        "market_engine.features.process_structural_candles",
        lambda _frame: (swings, []),
    )
    monkeypatch.setattr(
        "market_engine.features.build_structural_sequence",
        lambda input_frame: input_frame.copy(),
    )

    up_frame = frame.iloc[[1]].copy()
    up_frame.loc[1, "close"] = 108.0
    up_frame.loc[1, "structure_direction"] = 1.0

    down_frame = frame.iloc[[1]].copy()
    down_frame.loc[1, "close"] = 108.0
    down_frame.loc[1, "structure_direction"] = -1.0

    up_result = add_sr_location_features(up_frame)
    down_result = add_sr_location_features(down_frame)

    assert up_result.loc[1, "distance_to_next_structure_level"] == 17.0
    assert down_result.loc[1, "distance_to_next_structure_level"] == 13.0


def test_sr_location_exposes_raw_structural_distances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = make_sr_location_frame().iloc[[0]].copy()

    swings = [
        make_sr_swing(SwingType.LOW, 0, 95.0),
        make_sr_swing(SwingType.HIGH, 0, 110.0),
    ]

    monkeypatch.setattr(
        "market_engine.features.process_structural_candles",
        lambda _frame: (swings, []),
    )
    monkeypatch.setattr(
        "market_engine.features.build_structural_sequence",
        lambda input_frame: input_frame.copy(),
    )

    result = add_sr_location_features(frame)

    assert result.loc[0, "distance_to_support"] == 5.0
    assert result.loc[0, "distance_to_resistance"] == 10.0
    assert result.loc[0, "leg_position"] == 0.2


def test_context_outcome_join_preserves_context_and_outcome_fields():
    context = pd.DataFrame({
        "setup_index": [10, 20],
        "direction": ["UP", "DOWN"],
        "poi_fvg_interaction": ["TESTED", "NONE"],
    })
    outcomes = pd.DataFrame({
        "setup_index": [10, 20],
        "label": ["TP_FIRST", "SL_FIRST"],
        "reward_risk": [1.5, 0.8],
    })

    from market_engine.context import combine_context_with_outcomes

    result = combine_context_with_outcomes(context, outcomes)

    assert result["setup_index"].tolist() == [10, 20]
    assert result["label"].tolist() == ["TP_FIRST", "SL_FIRST"]
    assert result["poi_fvg_interaction"].tolist() == ["TESTED", "NONE"]


def test_context_outcome_join_rejects_overlapping_non_key_columns():
    from market_engine.context import combine_context_with_outcomes

    context = pd.DataFrame({"setup_index": [10], "direction": ["UP"]})
    outcomes = pd.DataFrame({"setup_index": [10], "direction": ["UP"], "label": ["TP_FIRST"]})

    with pytest.raises(ValueError, match="overlap"):
        combine_context_with_outcomes(context, outcomes)


def test_context_outcome_join_preserves_context_and_outcome_fields():
    context = pd.DataFrame({
        "setup_index": [10, 20],
        "direction": ["UP", "DOWN"],
        "poi_fvg_interaction": ["TESTED", "NONE"],
    })
    outcomes = pd.DataFrame({
        "setup_index": [10, 20],
        "label": ["TP_FIRST", "SL_FIRST"],
        "reward_risk": [1.5, 0.8],
    })

    from market_engine.context import combine_context_with_outcomes

    result = combine_context_with_outcomes(context, outcomes)

    assert result["setup_index"].tolist() == [10, 20]
    assert result["label"].tolist() == ["TP_FIRST", "SL_FIRST"]
    assert result["poi_fvg_interaction"].tolist() == ["TESTED", "NONE"]


def test_context_outcome_join_rejects_overlapping_non_key_columns():
    from market_engine.context import combine_context_with_outcomes

    context = pd.DataFrame({"setup_index": [10], "direction": ["UP"]})
    outcomes = pd.DataFrame({"setup_index": [10], "direction": ["UP"], "label": ["TP_FIRST"]})

    with pytest.raises(ValueError, match="overlap"):
        combine_context_with_outcomes(context, outcomes)
