import pandas as pd

from market_engine.structure import (
    ValidSwing,
    _label_swing,
    CandleKind,
    Direction,
    SwingType,
    build_structural_sequence,
    process_structural_candles,
)


def _frame(rows):
    return pd.DataFrame(
        rows,
        columns=["timestamp", "open", "high", "low", "close"],
    )


def test_inside_candles_are_removed_from_structural_sequence():
    frame = _frame([
        ("2026-01-01 00:00", 10, 12, 8, 11),
        ("2026-01-01 00:30", 10, 11, 9, 10.5),
        ("2026-01-01 01:00", 11, 13, 9, 12),
    ])
    seq = build_structural_sequence(frame)
    assert [x.kind for x in seq] == [CandleKind.UP, CandleKind.UP]
    assert [x.index for x in seq] == [0, 2]


def test_outside_candle_replaces_contained_reference():
    frame = _frame([
        ("2026-01-01 00:00", 10, 12, 8, 11),
        ("2026-01-01 00:30", 11, 14, 7, 12),
        ("2026-01-01 01:00", 12, 13, 9, 12.5),
    ])
    seq = build_structural_sequence(frame)
    assert seq[1].kind is CandleKind.OUTSIDE
    assert seq[1].index == 1
    assert [x.index for x in seq] == [0, 1]


def test_pullback_break_confirms_historical_extreme_as_swing():
    frame = _frame([
        ("2026-01-01 00:00", 10, 12, 9, 11),
        ("2026-01-01 00:30", 11, 14, 10, 13),
        ("2026-01-01 01:00", 13, 14.5, 11, 14),
        ("2026-01-01 01:30", 13, 13.5, 10, 11),
        ("2026-01-01 02:00", 11, 12, 9, 10),
    ])
    seq = build_structural_sequence(frame)
    swings, events = process_structural_candles(seq)

    assert swings
    swing = swings[0]
    assert swing.swing_type is SwingType.HIGH
    assert swing.index == 2
    assert swing.confirmation_index == 4
    assert events[0].event == "SWING_HIGH_VALID"
    assert events[0].direction is Direction.DOWN

def test_valid_swing_labels_compare_previous_same_type():
    previous_high = ValidSwing(
        index=10,
        timestamp="2026-01-01 05:00",
        price=100.0,
        swing_type=SwingType.HIGH,
        confirmation_index=12,
        confirmation_timestamp="2026-01-01 06:00",
    )
    higher_high = ValidSwing(
        index=20,
        timestamp="2026-01-01 10:00",
        price=105.0,
        swing_type=SwingType.HIGH,
        confirmation_index=22,
        confirmation_timestamp="2026-01-01 11:00",
    )
    lower_high = ValidSwing(
        index=30,
        timestamp="2026-01-01 15:00",
        price=98.0,
        swing_type=SwingType.HIGH,
        confirmation_index=32,
        confirmation_timestamp="2026-01-01 16:00",
    )

    assert _label_swing(higher_high, previous_high) == "HH"
    assert _label_swing(lower_high, previous_high) == "LH"
