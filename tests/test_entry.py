import pandas as pd

from market_engine.entry import (
    SetupCandidate,
    _candidate_from_bos,
    build_setup_candidates,
)
from market_engine.structure import Direction, StructureEvent, StructureScope, SwingType, ValidSwing


def make_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="30min"),
            "open": [10.0, 11.0, 13.0, 12.0],
            "high": [12.0, 14.0, 15.0, 14.0],
            "low": [9.0, 10.0, 11.0, 10.0],
            "close": [11.0, 13.0, 14.0, 13.0],
        }
    )


def make_swing(swing_type: SwingType, confirmation_index: int, price: float) -> ValidSwing:
    return ValidSwing(
        index=confirmation_index - 1,
        timestamp=pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=30 * (confirmation_index - 1)),
        price=price,
        swing_type=swing_type,
        confirmation_index=confirmation_index,
        confirmation_timestamp=pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=30 * confirmation_index),
        scope=StructureScope.EXTERNAL,
    )


def test_bos_setup_uses_next_open_and_prior_opposite_swing():
    frame = make_frame()
    event = StructureEvent(
        index=1,
        timestamp=frame.loc[1, "timestamp"],
        event="BULLISH_BOS",
        direction=Direction.UP,
        swing_index=0,
        swing_price=12.0,
    )
    swings = [make_swing(SwingType.LOW, 0, 9.0)]

    candidate = _candidate_from_bos(frame, event, swings)

    assert isinstance(candidate, SetupCandidate)
    assert candidate.direction is Direction.UP
    assert candidate.setup_index == 1
    assert candidate.entry_index == 2
    assert candidate.entry_price == 13.0
    assert candidate.invalidation_price == 9.0
    assert candidate.risk == 4.0


def test_bos_does_not_use_swing_confirmed_on_same_candle():
    frame = make_frame()
    event = StructureEvent(
        index=1,
        timestamp=frame.loc[1, "timestamp"],
        event="BULLISH_BOS",
        direction=Direction.UP,
        swing_index=0,
        swing_price=12.0,
    )
    swings = [make_swing(SwingType.LOW, 1, 9.0)]

    assert _candidate_from_bos(frame, event, swings) is None


def test_build_setup_candidates_only_accepts_confirmed_bos():
    frame = make_frame()
    event = StructureEvent(
        index=1,
        timestamp=frame.loc[1, "timestamp"],
        event="BULLISH_BOS",
        direction=Direction.UP,
        swing_index=0,
        swing_price=12.0,
    )
    swing = make_swing(SwingType.LOW, 0, 9.0)

    import market_engine.entry as entry_module

    original_builder = entry_module.build_structural_sequence
    original_processor = entry_module.process_structural_candles
    try:
        entry_module.build_structural_sequence = lambda _: []
        entry_module.process_structural_candles = lambda _: ([swing], [event])
        candidates = build_setup_candidates(frame)
    finally:
        entry_module.build_structural_sequence = original_builder
        entry_module.process_structural_candles = original_processor

    assert len(candidates) == 1
    assert candidates[0].entry_index == 2
