import pandas as pd

from market_engine.structure import (
    ValidSwing,
    StructureCheckpoint,
    StructureScope,
    _label_swing,
    CandleKind,
    Direction,
    SwingType,
    StructuralCandle,
    build_structural_sequence,
    process_structural_candles,
    find_first_bos,
    process_from_first_bos,
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

def test_bos_breaks_only_previously_confirmed_valid_swing():
    candles = [
        StructuralCandle(0, "2026-01-01 00:00", 12, 9, 10, 11, CandleKind.UP),
        StructuralCandle(1, "2026-01-01 00:30", 14, 10, 11, 13, CandleKind.UP),
        StructuralCandle(2, "2026-01-01 01:00", 14.5, 11, 13, 14, CandleKind.UP),
        StructuralCandle(3, "2026-01-01 01:30", 13.5, 10, 13, 11, CandleKind.DOWN),
        StructuralCandle(4, "2026-01-01 02:00", 13, 8, 11, 9, CandleKind.DOWN),
        StructuralCandle(5, "2026-01-01 02:30", 12, 9, 9, 11, CandleKind.UP),
        StructuralCandle(6, "2026-01-01 03:00", 15, 11, 11, 14, CandleKind.UP),
    ]

    swings, events = process_structural_candles(candles)

    assert [(s.swing_type, s.index, s.confirmation_index) for s in swings] == [
        (SwingType.HIGH, 2, 4),
        (SwingType.LOW, 4, 6),
    ]

    bos = [e for e in events if e.event.endswith("_BOS")]
    assert len(bos) == 1
    assert bos[0].event == "BULLISH_BOS"
    assert bos[0].direction is Direction.UP
    assert bos[0].swing_index == 2
    assert bos[0].index == 6

def test_first_bos_is_the_initial_structure_anchor():
    candles = [
        StructuralCandle(0, "2026-01-01 00:00", 12, 9, 10, 11, CandleKind.UP),
        StructuralCandle(1, "2026-01-01 00:30", 14, 10, 11, 13, CandleKind.UP),
        StructuralCandle(2, "2026-01-01 01:00", 14.5, 11, 13, 14, CandleKind.UP),
        StructuralCandle(3, "2026-01-01 01:30", 13.5, 10, 13, 11, CandleKind.DOWN),
        StructuralCandle(4, "2026-01-01 02:00", 13, 8, 11, 9, CandleKind.DOWN),
        StructuralCandle(5, "2026-01-01 02:30", 12, 9, 9, 11, CandleKind.UP),
        StructuralCandle(6, "2026-01-01 03:00", 15, 11, 11, 14, CandleKind.UP),
    ]

    swings, events = process_structural_candles(candles)
    bos = [event for event in events if event.event.endswith("_BOS")]

    assert bos
    assert bos[0].event == "BULLISH_BOS"
    assert bos[0].index == 6
    assert bos[0].direction is Direction.UP


def test_find_first_bos_returns_first_confirmed_bos():
    candles = [
        StructuralCandle(0, "2026-01-01 00:00", 12, 9, 10, 11, CandleKind.UP),
        StructuralCandle(1, "2026-01-01 00:30", 14, 10, 11, 13, CandleKind.UP),
        StructuralCandle(2, "2026-01-01 01:00", 14.5, 11, 13, 14, CandleKind.UP),
        StructuralCandle(3, "2026-01-01 01:30", 13.5, 10, 13, 11, CandleKind.DOWN),
        StructuralCandle(4, "2026-01-01 02:00", 13, 8, 11, 9, CandleKind.DOWN),
        StructuralCandle(5, "2026-01-01 02:30", 12, 9, 9, 11, CandleKind.UP),
        StructuralCandle(6, "2026-01-01 03:00", 15, 11, 11, 14, CandleKind.UP),
    ]

    first_bos = find_first_bos(candles)

    assert first_bos is not None
    assert first_bos.event == "BULLISH_BOS"
    assert first_bos.index == 6
    assert first_bos.direction is Direction.UP


def test_find_first_bos_returns_none_when_no_bos_exists():
    candles = [
        StructuralCandle(0, "2026-01-01 00:00", 12, 9, 10, 11, CandleKind.UP),
        StructuralCandle(1, "2026-01-01 00:30", 14, 10, 11, 13, CandleKind.UP),
    ]

    assert find_first_bos(candles) is None


def test_process_from_first_bos_uses_bos_as_anchor():
    candles = [
        StructuralCandle(0, "2026-01-01 00:00", 12, 9, 10, 11, CandleKind.UP),
        StructuralCandle(1, "2026-01-01 00:30", 14, 10, 11, 13, CandleKind.UP),
        StructuralCandle(2, "2026-01-01 01:00", 14.5, 11, 13, 14, CandleKind.UP),
        StructuralCandle(3, "2026-01-01 01:30", 13.5, 10, 13, 11, CandleKind.DOWN),
        StructuralCandle(4, "2026-01-01 02:00", 13, 8, 11, 9, CandleKind.DOWN),
        StructuralCandle(5, "2026-01-01 02:30", 12, 9, 9, 11, CandleKind.UP),
        StructuralCandle(6, "2026-01-01 03:00", 15, 11, 11, 14, CandleKind.UP),
        StructuralCandle(7, "2026-01-01 03:30", 16, 12, 14, 15, CandleKind.UP),
    ]

    result = process_from_first_bos(candles)

    assert result is not None
    anchor, swings, events = result

    assert anchor.event == "BULLISH_BOS"
    assert anchor.index == 6
    assert all(event.index >= anchor.index for event in events)
    assert any(swing.index == anchor.swing_index for swing in swings)


def test_process_from_first_bos_returns_none_without_bos():
    candles = [
        StructuralCandle(0, "2026-01-01 00:00", 12, 9, 10, 11, CandleKind.UP),
        StructuralCandle(1, "2026-01-01 00:30", 14, 10, 11, 13, CandleKind.UP),
    ]

    assert process_from_first_bos(candles) is None


def test_process_from_first_bos_matches_full_run_after_anchor():
    candles = [
        StructuralCandle(0, "00:00", 12, 9, 10, 11, CandleKind.UP),
        StructuralCandle(1, "00:30", 14, 10, 11, 13, CandleKind.UP),
        StructuralCandle(2, "01:00", 14.5, 11, 13, 14, CandleKind.UP),
        StructuralCandle(3, "01:30", 13.5, 10, 13, 11, CandleKind.DOWN),
        StructuralCandle(4, "02:00", 13, 8, 11, 9, CandleKind.DOWN),
        StructuralCandle(5, "02:30", 12, 9, 9, 11, CandleKind.UP),
        StructuralCandle(6, "03:00", 15, 11, 11, 14, CandleKind.UP),
        StructuralCandle(7, "03:30", 16, 12, 14, 15, CandleKind.UP),
        StructuralCandle(8, "04:00", 15, 10, 15, 11, CandleKind.DOWN),
        StructuralCandle(9, "04:30", 14, 9, 11, 10, CandleKind.DOWN),
        StructuralCandle(10, "05:00", 13, 8, 10, 9, CandleKind.DOWN),
    ]

    full_swings, full_events = process_structural_candles(candles)
    anchored = process_from_first_bos(candles)

    assert anchored is not None
    anchor, anchored_swings, anchored_events = anchored

    assert anchored_events == [
        event for event in full_events if event.index > anchor.index
    ]

    expected_swings = [
        swing
        for swing in full_swings
        if swing.index == anchor.swing_index
        or swing.confirmation_index > anchor.index
    ]
    assert anchored_swings == expected_swings


def test_process_from_first_bos_preserves_broken_swing_context():
    candles = [
        StructuralCandle(0, "2026-01-01 00:00", 12, 9, 10, 11, CandleKind.UP),
        StructuralCandle(1, "2026-01-01 00:30", 14, 10, 11, 13, CandleKind.UP),
        StructuralCandle(2, "2026-01-01 01:00", 14.5, 11, 13, 14, CandleKind.UP),
        StructuralCandle(3, "2026-01-01 01:30", 13.5, 10, 13, 11, CandleKind.DOWN),
        StructuralCandle(4, "2026-01-01 02:00", 13, 8, 11, 9, CandleKind.DOWN),
        StructuralCandle(5, "2026-01-01 02:30", 12, 9, 9, 11, CandleKind.UP),
        StructuralCandle(6, "2026-01-01 03:00", 15, 11, 11, 14, CandleKind.UP),
        StructuralCandle(7, "2026-01-01 03:30", 16, 12, 14, 15, CandleKind.UP),
    ]

    anchor = find_first_bos(candles)
    assert anchor is not None

    _, full_events = process_structural_candles(candles)

    full_bos = [
        event for event in full_events
        if event.event.endswith("_BOS")
    ]

    assert full_bos
    assert full_bos[0].index == anchor.index
    assert full_bos[0].swing_index is not None
    assert full_bos[0].swing_index < anchor.index

    result = process_from_first_bos(candles)
    assert result is not None

    _, swings, events = result

    # The anchor's broken swing is historical context,
    # so it must remain available to the anchored structure.
    assert any(
        swing.index == full_bos[0].swing_index
        for swing in swings
    )


def test_process_from_first_bos_handles_bearish_anchor():
    candles = [
        StructuralCandle(0, "2026-01-01 00:00", 11, 8, 10, 9, CandleKind.DOWN),
        StructuralCandle(1, "2026-01-01 00:30", 10, 6, 9, 7, CandleKind.DOWN),
        StructuralCandle(2, "2026-01-01 01:00", 9, 6.5, 7, 8, CandleKind.UP),
        StructuralCandle(3, "2026-01-01 01:30", 10, 7, 8, 9, CandleKind.UP),
        StructuralCandle(4, "2026-01-01 02:00", 9, 5, 8, 6, CandleKind.DOWN),
        StructuralCandle(5, "2026-01-01 02:30", 8, 4, 6, 5, CandleKind.DOWN),
        StructuralCandle(6, "2026-01-01 03:00", 7, 3, 5, 4, CandleKind.DOWN),
    ]

    result = process_from_first_bos(candles)

    assert result is not None

    anchor, swings, events = result

    assert anchor.event == "BEARISH_BOS"
    assert anchor.direction is Direction.DOWN
    assert anchor.index == 4
    assert anchor.swing_index is not None
    assert anchor.swing_index < anchor.index

    assert all(event.index >= anchor.index for event in events)
    assert any(swing.index == anchor.swing_index for swing in swings)


def test_process_from_first_bos_does_not_expose_unconfirmed_future_swing():
    candles = [
        StructuralCandle(0, "2026-01-01 00:00", 12, 9, 10, 11, CandleKind.UP),
        StructuralCandle(1, "2026-01-01 00:30", 14, 10, 11, 13, CandleKind.UP),
        StructuralCandle(2, "2026-01-01 01:00", 14.5, 11, 13, 14, CandleKind.UP),
        StructuralCandle(3, "2026-01-01 01:30", 13.5, 10, 13, 11, CandleKind.DOWN),
        StructuralCandle(4, "2026-01-01 02:00", 13, 8, 11, 9, CandleKind.DOWN),
        StructuralCandle(5, "2026-01-01 02:30", 12, 9, 9, 11, CandleKind.UP),
        StructuralCandle(6, "2026-01-01 03:00", 15, 11, 11, 14, CandleKind.UP),
        StructuralCandle(7, "2026-01-01 03:30", 16, 12, 14, 15, CandleKind.UP),
        StructuralCandle(8, "2026-01-01 04:00", 14, 10, 15, 11, CandleKind.DOWN),
        StructuralCandle(9, "2026-01-01 04:30", 13, 9, 11, 10, CandleKind.DOWN),
    ]

    result = process_from_first_bos(candles)

    assert result is not None

    anchor, swings, _ = result

    assert anchor.index == 6

    for swing in swings:
        if swing.index != anchor.swing_index:
            assert swing.confirmation_index >= anchor.index


def test_process_from_first_bos_keeps_future_confirmed_swing_unavailable_at_bos():
    candles = [
        StructuralCandle(0, "2026-01-01 00:00", 12, 9, 10, 11, CandleKind.UP),
        StructuralCandle(1, "2026-01-01 00:30", 14, 10, 11, 13, CandleKind.UP),
        StructuralCandle(2, "2026-01-01 01:00", 14.5, 11, 13, 14, CandleKind.UP),
        StructuralCandle(3, "2026-01-01 01:30", 13.5, 10, 13, 11, CandleKind.DOWN),
        StructuralCandle(4, "2026-01-01 02:00", 13, 8, 11, 9, CandleKind.DOWN),
        StructuralCandle(5, "2026-01-01 02:30", 20, 9, 9, 11, CandleKind.UP),
        StructuralCandle(6, "2026-01-01 03:00", 16, 11, 11, 15, CandleKind.UP),
        StructuralCandle(7, "2026-01-01 03:30", 18, 12, 15, 17, CandleKind.UP),
        StructuralCandle(8, "2026-01-01 04:00", 21, 13, 17, 20, CandleKind.UP),
    ]

    result = process_from_first_bos(candles)

    assert result is not None

    anchor, swings, _ = result

    assert anchor.event == "BULLISH_BOS"
    assert anchor.index == 5

    future_confirmed = [
        swing
        for swing in swings
        if swing.index < anchor.index
        and swing.confirmation_index > anchor.index
    ]

    assert future_confirmed

def test_outside_candle_can_break_both_confirmed_swings():
    candles = [
        StructuralCandle(0, "2026-01-01 00:00", 12, 9, 10, 11, CandleKind.UP),
        StructuralCandle(1, "2026-01-01 00:30", 14, 10, 11, 13, CandleKind.UP),
        StructuralCandle(2, "2026-01-01 01:00", 14.5, 11, 13, 14, CandleKind.UP),
        StructuralCandle(3, "2026-01-01 01:30", 13.5, 10, 13, 11, CandleKind.DOWN),
        StructuralCandle(4, "2026-01-01 02:00", 13, 8, 11, 9, CandleKind.DOWN),
        StructuralCandle(5, "2026-01-01 02:30", 12, 9, 9, 11, CandleKind.UP),
        StructuralCandle(6, "2026-01-01 03:00", 15, 11, 11, 14, CandleKind.UP),
        StructuralCandle(7, "2026-01-01 03:30", 10, 7, 14, 8, CandleKind.OUTSIDE),
    ]

    swings, events = process_structural_candles(candles)

    bos = [event for event in events if event.event.endswith("_BOS")]

    assert len(bos) == 2
    assert {event.event for event in bos} == {
        "BULLISH_BOS",
        "BEARISH_BOS",
    }
    assert [(event.event, event.index) for event in bos] == [("BULLISH_BOS", 6), ("BEARISH_BOS", 7)]



def test_outside_candle_can_break_both_confirmed_swings_same_candle():
    candles = [
        StructuralCandle(0, "00:00", 12, 9, 10, 11, CandleKind.UP),
        StructuralCandle(1, "00:30", 14, 10, 11, 13, CandleKind.UP),
        StructuralCandle(2, "01:00", 14.5, 11, 13, 14, CandleKind.UP),
        StructuralCandle(3, "01:30", 13.5, 10, 13, 11, CandleKind.DOWN),
        StructuralCandle(4, "02:00", 13, 8, 11, 9, CandleKind.DOWN),
        StructuralCandle(5, "02:30", 10, 9, 9, 9.5, CandleKind.DOWN),
        StructuralCandle(6, "03:00", 11, 8.5, 9, 10.5, CandleKind.DOWN),
        StructuralCandle(7, "03:30", 16, 7, 10, 14, CandleKind.OUTSIDE),
    ]

    swings, events = process_structural_candles(candles)

    assert [(s.swing_type, s.index, s.confirmation_index) for s in swings] == [
        (SwingType.HIGH, 2, 4),
        (SwingType.LOW, 4, 6),
    ]

    bos = [event for event in events if event.event.endswith("_BOS")]

    assert [(event.event, event.index, event.swing_index) for event in bos] == [
        ("BULLISH_BOS", 7, 2),
        ("BEARISH_BOS", 7, 4),
    ]


def test_simultaneous_bos_events_have_deterministic_order():
    candles = [
        StructuralCandle(0, "00:00", 12, 9, 10, 11, CandleKind.UP),
        StructuralCandle(1, "00:30", 14, 10, 11, 13, CandleKind.UP),
        StructuralCandle(2, "01:00", 14.5, 11, 13, 14, CandleKind.UP),
        StructuralCandle(3, "01:30", 13.5, 10, 13, 11, CandleKind.DOWN),
        StructuralCandle(4, "02:00", 13, 8, 11, 9, CandleKind.DOWN),
        StructuralCandle(5, "02:30", 10, 9, 9, 9.5, CandleKind.DOWN),
        StructuralCandle(6, "03:00", 11, 8.5, 9, 10.5, CandleKind.DOWN),
        StructuralCandle(7, "03:30", 16, 7, 10, 14, CandleKind.OUTSIDE),
    ]

    _, events = process_structural_candles(candles)

    simultaneous = [event for event in events if event.index == 7]
    assert [
        (event.event, event.swing_index)
        for event in simultaneous
    ] == [
        ("BULLISH_BOS", 2),
        ("BEARISH_BOS", 4),
    ]


def test_process_from_first_bos_continues_structure_from_anchor_direction():
    candles = [
        StructuralCandle(0, "00:00", 12, 9, 10, 11, CandleKind.UP),
        StructuralCandle(1, "00:30", 14, 10, 11, 13, CandleKind.UP),
        StructuralCandle(2, "01:00", 14.5, 11, 13, 14, CandleKind.UP),
        StructuralCandle(3, "01:30", 13.5, 10, 13, 11, CandleKind.DOWN),
        StructuralCandle(4, "02:00", 13, 8, 11, 9, CandleKind.DOWN),
        StructuralCandle(5, "02:30", 12, 9, 9, 11, CandleKind.UP),
        StructuralCandle(6, "03:00", 15, 11, 11, 14, CandleKind.UP),
        StructuralCandle(7, "03:30", 16, 12, 14, 15, CandleKind.UP),
        StructuralCandle(8, "04:00", 15, 10, 15, 11, CandleKind.DOWN),
        StructuralCandle(9, "04:30", 14, 9, 11, 10, CandleKind.DOWN),
        StructuralCandle(10, "05:00", 13, 8, 10, 9, CandleKind.DOWN),
    ]

    result = process_from_first_bos(candles)

    assert result is not None

    anchor, swings, events = result

    assert anchor.event == "BULLISH_BOS"
    assert anchor.direction is Direction.UP
    assert anchor.index == 6

    future_swings = [
        swing
        for swing in swings
        if swing.confirmation_index > anchor.index
    ]

    assert future_swings
    assert future_swings[0].swing_type is SwingType.HIGH
    assert future_swings[0].confirmation_index == 9


def test_process_from_first_bos_preserves_future_swing_confirmation_time():
    candles = [
        StructuralCandle(0, "00:00", 12, 9, 10, 11, CandleKind.UP),
        StructuralCandle(1, "00:30", 14, 10, 11, 13, CandleKind.UP),
        StructuralCandle(2, "01:00", 14.5, 11, 13, 14, CandleKind.UP),
        StructuralCandle(3, "01:30", 13.5, 10, 13, 11, CandleKind.DOWN),
        StructuralCandle(4, "02:00", 13, 8, 11, 9, CandleKind.DOWN),
        StructuralCandle(5, "02:30", 20, 9, 9, 11, CandleKind.UP),
        StructuralCandle(6, "03:00", 21, 12, 11, 20, CandleKind.UP),
        StructuralCandle(7, "03:30", 20, 10, 20, 11, CandleKind.DOWN),
        StructuralCandle(8, "04:00", 19, 9, 11, 10, CandleKind.DOWN),
    ]

    result = process_from_first_bos(candles)

    assert result is not None

    anchor, swings, _ = result

    assert anchor.index == 5

    future_swings = [
        swing
        for swing in swings
        if swing.index != anchor.swing_index
        and swing.confirmation_index > anchor.index
    ]

    assert future_swings

    for swing in future_swings:
        assert swing.confirmation_index > anchor.index



def test_first_bos_checkpoint_preserves_structure_context():
    candles = [
        StructuralCandle(0, "00:00", 12, 9, 10, 11, CandleKind.UP),
        StructuralCandle(1, "00:30", 14, 10, 11, 13, CandleKind.UP),
        StructuralCandle(2, "01:00", 14.5, 11, 13, 14, CandleKind.UP),
        StructuralCandle(3, "01:30", 13.5, 10, 13, 11, CandleKind.DOWN),
        StructuralCandle(4, "02:00", 13, 8, 11, 9, CandleKind.DOWN),
        StructuralCandle(5, "02:30", 20, 9, 9, 11, CandleKind.UP),
    ]

    anchor = find_first_bos(candles)

    assert anchor is not None
    assert anchor.event == "BULLISH_BOS"
    assert anchor.index == 5
    assert anchor.direction is Direction.UP
    assert anchor.swing_index == 2
    assert anchor.swing_price == 14.5


def test_structure_checkpoint_starts_after_anchor_candle():
    checkpoint = StructureCheckpoint(
        index=5,
        direction=Direction.UP,
        extreme=StructuralCandle(
            5, "02:30", 20, 9, 9, 11, CandleKind.UP
        ),
        pullback=None,
        last_swing=ValidSwing(
            index=2,
            timestamp="01:00",
            price=14.5,
            swing_type=SwingType.HIGH,
            confirmation_index=4,
            confirmation_timestamp="02:00",
        ),
        previous_swing=None,
        scope=StructureScope.EXTERNAL,
        last_high=None,
        last_low=None,
        broken_high_index=2,
        broken_low_index=None,
    )

    assert checkpoint.index == 5
    assert checkpoint.direction is Direction.UP
    assert checkpoint.extreme.index == 5
    assert checkpoint.last_swing.index == 2
    assert checkpoint.last_swing.confirmation_index == 4
    assert checkpoint.broken_high_index == 2


def test_valid_swing_scope_is_internal_inside_external_boundaries():
    candles = [
        StructuralCandle(0, "00:00", 12, 9, 10, 11, CandleKind.UP),
        StructuralCandle(1, "00:30", 14, 10, 11, 13, CandleKind.UP),
        StructuralCandle(2, "01:00", 14.5, 11, 13, 14, CandleKind.UP),
        StructuralCandle(3, "01:30", 13.5, 10, 13, 11, CandleKind.DOWN),
        StructuralCandle(4, "02:00", 13, 8, 11, 9, CandleKind.DOWN),
        StructuralCandle(5, "02:30", 12, 9, 9, 11, CandleKind.UP),
        StructuralCandle(6, "03:00", 13, 10, 11, 12, CandleKind.UP),
        StructuralCandle(7, "03:30", 13.5, 11, 12, 13, CandleKind.UP),
        StructuralCandle(8, "04:00", 13.2, 13, 13, 13.1, CandleKind.DOWN),
        StructuralCandle(9, "04:30", 12.5, 12, 13, 12.2, CandleKind.DOWN),
    ]

    swings, events = process_structural_candles(candles)

    assert [(s.swing_type, s.index, s.scope) for s in swings] == [
        (SwingType.HIGH, 2, StructureScope.EXTERNAL),
        (SwingType.LOW, 4, StructureScope.EXTERNAL),
        (SwingType.HIGH, 7, StructureScope.INTERNAL),
    ]

    assert events[-1].event == "SWING_HIGH_VALID"
    assert events[-1].scope is StructureScope.INTERNAL


def test_external_boundary_break_rebuilds_from_new_boundary():
    candles = [
        StructuralCandle(0, "00:00", 12, 9, 10, 11, CandleKind.UP),
        StructuralCandle(1, "00:30", 14, 10, 11, 13, CandleKind.UP),
        StructuralCandle(2, "01:00", 14.5, 11, 13, 14, CandleKind.UP),
        StructuralCandle(3, "01:30", 13.5, 10, 13, 11, CandleKind.DOWN),
        StructuralCandle(4, "02:00", 13, 8, 11, 9, CandleKind.DOWN),
        StructuralCandle(5, "02:30", 12, 9, 9, 11, CandleKind.UP),
        StructuralCandle(6, "03:00", 13, 10, 11, 12, CandleKind.UP),
        StructuralCandle(7, "03:30", 13.5, 11, 12, 13, CandleKind.UP),
        StructuralCandle(8, "04:00", 13.2, 13, 13, 13.1, CandleKind.DOWN),
        StructuralCandle(9, "04:30", 12.5, 12, 13, 12.2, CandleKind.DOWN),
        StructuralCandle(10, "05:00", 15, 12, 12, 14.5, CandleKind.UP),
        StructuralCandle(11, "05:30", 14, 11, 14, 12, CandleKind.DOWN),
        StructuralCandle(12, "06:00", 13, 10, 12, 11, CandleKind.DOWN),
        StructuralCandle(13, "06:30", 12, 9, 11, 10, CandleKind.DOWN),
        StructuralCandle(14, "07:00", 11, 8, 10, 9, CandleKind.DOWN),
    ]

    swings, events = process_structural_candles(candles)

    boundary_bos = [
        event
        for event in events
        if event.event == "BULLISH_BOS" and event.index == 10
    ]

    assert len(boundary_bos) == 1
    assert boundary_bos[0].swing_index == 2
    assert boundary_bos[0].scope is StructureScope.EXTERNAL

    # The old internal swing is not reused as a BOS target after rebuild.
    later_bos = [
        event
        for event in events
        if event.event.endswith("_BOS") and event.index > 10
    ]
    assert later_bos == []
