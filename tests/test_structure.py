import pandas as pd

from market_engine.structure import (
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
    assert seq[2].kind is CandleKind.INSIDE


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
