import math

import pandas as pd
import pytest

from market_engine.entry import SetupCandidate
from market_engine.features import (
    add_order_block_features,
    build_structural_features,
)
from market_engine.labels import build_setup_label_dataset
from market_engine.setup_facts import (
    SETUP_FACT_COLUMNS,
    SetupFactContext,
    build_setup_facts,
)
from market_engine.structure import (
    Direction,
    StructureEvent,
    StructureScope,
    SwingType,
    ValidSwing,
)


def swing(index, confirmation_index, price, swing_type):
    return ValidSwing(
        index=index,
        timestamp=index,
        price=price,
        swing_type=swing_type,
        confirmation_index=confirmation_index,
        confirmation_timestamp=confirmation_index,
    )


def candidate(direction, entry_price, invalidation_price, invalidation_swing_index):
    return SetupCandidate(
        setup_index=20,
        setup_timestamp=20,
        direction=direction,
        entry_index=21,
        entry_timestamp=21,
        entry_price=entry_price,
        invalidation_price=invalidation_price,
        risk=abs(entry_price - invalidation_price),
        invalidation_swing_index=invalidation_swing_index,
    )


def test_bullish_external_bos_facts():
    swings = [
        swing(5, 8, 100.0, SwingType.LOW),
        swing(10, 13, 110.0, SwingType.HIGH),
    ]
    events = [
        StructureEvent(
            index=20,
            timestamp=20,
            event="BULLISH_BOS",
            direction=Direction.UP,
            swing_index=10,
            swing_price=110.0,
            scope=StructureScope.EXTERNAL,
        )
    ]

    facts = build_setup_facts(candidate(Direction.UP, 112.0, 100.0, 5), swings, events)

    assert facts["setup_bos_external"] == 1.0
    assert facts["setup_broken_swing_age"] == 7.0
    assert facts["setup_entry_beyond_broken_swing_r"] == pytest.approx(2.0 / 12.0)
    assert facts["setup_invalidation_swing_age"] == 12.0


def test_bearish_internal_bos_facts():
    swings = [
        swing(5, 8, 100.0, SwingType.HIGH),
        swing(10, 13, 90.0, SwingType.LOW),
    ]
    events = [
        StructureEvent(
            index=20,
            timestamp=20,
            event="BEARISH_BOS",
            direction=Direction.DOWN,
            swing_index=10,
            swing_price=90.0,
            scope=StructureScope.INTERNAL,
        )
    ]

    facts = build_setup_facts(candidate(Direction.DOWN, 88.0, 100.0, 5), swings, events)

    assert facts["setup_bos_external"] == 0.0
    assert facts["setup_broken_swing_age"] == 7.0
    assert facts["setup_entry_beyond_broken_swing_r"] == pytest.approx(2.0 / 12.0)
    assert facts["setup_invalidation_swing_age"] == 12.0


def test_dual_bos_on_one_candle_uses_each_candidates_own_event():
    swings = [
        swing(5, 8, 100.0, SwingType.LOW),
        swing(6, 9, 100.0, SwingType.HIGH),
        swing(10, 13, 110.0, SwingType.HIGH),
        swing(11, 14, 90.0, SwingType.LOW),
    ]
    events = [
        StructureEvent(20, 20, "BULLISH_BOS", Direction.UP, 10, 110.0,
                       StructureScope.EXTERNAL),
        StructureEvent(20, 20, "BEARISH_BOS", Direction.DOWN, 11, 90.0,
                       StructureScope.INTERNAL),
    ]
    context = SetupFactContext.build(swings, events)

    up = context.facts(candidate(Direction.UP, 112.0, 100.0, 5))
    down = context.facts(candidate(Direction.DOWN, 88.0, 100.0, 6))

    assert up["setup_bos_external"] == 1.0
    assert down["setup_bos_external"] == 0.0
    assert up["setup_broken_swing_age"] == 7.0
    assert down["setup_broken_swing_age"] == 6.0


def test_missing_bos_event_returns_nan_for_every_fact():
    swings = [swing(5, 8, 100.0, SwingType.LOW), swing(10, 13, 110.0, SwingType.HIGH)]

    facts = build_setup_facts(candidate(Direction.UP, 112.0, 100.0, 5), swings, [])

    assert set(facts) == set(SETUP_FACT_COLUMNS)
    assert all(math.isnan(value) for value in facts.values())


def test_label_dataset_adds_setup_facts_only_when_events_are_given():
    swings = [
        swing(1, 3, 99.0, SwingType.HIGH),
        swing(2, 4, 95.0, SwingType.LOW),
        swing(3, 5, 110.0, SwingType.HIGH),
    ]
    events = [
        StructureEvent(10, 10, "BULLISH_BOS", Direction.UP, 1, 99.0,
                       StructureScope.EXTERNAL),
    ]
    setup = SetupCandidate(
        setup_index=10,
        setup_timestamp=10,
        direction=Direction.UP,
        entry_index=11,
        entry_timestamp=11,
        entry_price=100.0,
        invalidation_price=95.0,
        risk=5.0,
        invalidation_swing_index=2,
    )
    highs = [101.0] * 30
    highs[13] = 111.0
    frame = pd.DataFrame({"high": highs, "low": [99.0] * 30})

    with_facts = build_setup_label_dataset(
        [setup], swings, frame, None, 0.2, (), horizon=5, events=events
    )
    without_facts = build_setup_label_dataset(
        [setup], swings, frame, None, 0.2, (), horizon=5
    )

    assert list(with_facts["label"]) == ["TP_FIRST"]
    assert with_facts.loc[0, "setup_bos_external"] == 1.0
    assert with_facts.loc[0, "setup_broken_swing_age"] == 7.0
    assert with_facts.loc[0, "setup_entry_beyond_broken_swing_r"] == pytest.approx(0.2)
    assert not set(SETUP_FACT_COLUMNS) & set(without_facts.columns)


def test_order_block_feature_projection_is_directional_and_historical():
    rows = 8
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=rows,
                freq="30min",
            ),
            "open": [10, 11, 13, 13, 11, 9, 11, 12],
            "high": [12, 14, 14.5, 13.5, 13, 13.5, 14, 15],
            "low": [9, 10, 11, 10, 8, 8, 9, 11],
            "close": [11, 13, 14, 11, 9, 10, 11, 14],
        }
    )

    result = add_order_block_features(frame)

    assert result["ob_bullish_present"].iloc[:7].tolist() == [
        0, 0, 0, 0, 0, 0, 0
    ]
    assert result["ob_bullish_present"].iloc[7] == 1
    assert result["ob_bearish_present"].sum() == 0


def test_order_block_feature_projection_uses_wick_to_wick_zone():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=8,
                freq="30min",
            ),
            "open": [10, 11, 13, 13, 11, 9, 11, 12],
            "high": [12, 14, 14.5, 13.5, 13, 13.5, 14, 15],
            "low": [9, 10, 11, 10, 8, 8, 9, 11],
            "close": [11, 13, 14, 11, 9, 10, 11, 14],
            }
    )

    result = add_order_block_features(frame)

    assert result["ob_bullish_size"].iloc[7] == 5.0
    assert result["ob_bullish_age_bars"].iloc[7] == 0
    assert result["ob_bullish_contains_price"].iloc[7] == 0
    assert result["ob_bullish_distance"].iloc[7] == 1.0
