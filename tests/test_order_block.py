import pytest

from market_engine.order_block import (
    OrderBlockCandidate,
    find_order_block_candidates,
    resolve_order_block_zone,
)
from market_engine.structure import (
    Direction,
    StructureEvent,
    StructureScope,
    SwingType,
    ValidSwing,
)


def _swing(
    index,
    price,
    swing_type,
    confirmation_index,
    label=None,
):
    return ValidSwing(
        index=index,
        timestamp=f"2026-01-01 {index:02d}:00",
        price=price,
        swing_type=swing_type,
        confirmation_index=confirmation_index,
        confirmation_timestamp=f"2026-01-01 {confirmation_index:02d}:00",
        label=label,
        scope=StructureScope.EXTERNAL,
    )


def test_bullish_bos_produces_low_swing_candidates_inside_leg():
    swings = [
        _swing(2, 100, SwingType.HIGH, 4, "HH"),
        _swing(5, 94, SwingType.LOW, 7, "HL"),
        _swing(8, 96, SwingType.LOW, 9, "HL"),
    ]
    events = [
        StructureEvent(
            index=10,
            timestamp="2026-01-01 10:00",
            event="BULLISH_BOS",
            direction=Direction.UP,
            swing_index=2,
            swing_price=100,
            scope=StructureScope.EXTERNAL,
        )
    ]

    candidates = find_order_block_candidates(swings, events)

    assert [(x.swing_index, x.swing_type) for x in candidates] == [
        (5, SwingType.LOW),
        (8, SwingType.LOW),
    ]


def test_bearish_bos_produces_high_swing_candidates_inside_leg():
    swings = [
        _swing(2, 100, SwingType.LOW, 4, "LL"),
        _swing(5, 106, SwingType.HIGH, 7, "LH"),
        _swing(8, 104, SwingType.HIGH, 9, "LH"),
    ]
    events = [
        StructureEvent(
            index=10,
            timestamp="2026-01-01 10:00",
            event="BEARISH_BOS",
            direction=Direction.DOWN,
            swing_index=2,
            swing_price=100,
            scope=StructureScope.EXTERNAL,
        )
    ]

    candidates = find_order_block_candidates(swings, events)

    assert [(x.swing_index, x.swing_type) for x in candidates] == [
        (5, SwingType.HIGH),
        (8, SwingType.HIGH),
    ]


def test_swing_alone_does_not_create_order_block_candidate():
    swings = [
        _swing(5, 94, SwingType.LOW, 7, "HL"),
    ]

    assert find_order_block_candidates(swings, []) == []


def test_unconfirmed_swing_before_bos_is_excluded():
    swings = [
        _swing(2, 100, SwingType.HIGH, 4, "HH"),
        _swing(5, 94, SwingType.LOW, 11, "HL"),
    ]
    events = [
        StructureEvent(
            index=10,
            timestamp="2026-01-01 10:00",
            event="BULLISH_BOS",
            direction=Direction.UP,
            swing_index=2,
            swing_price=100,
            scope=StructureScope.EXTERNAL,
        )
    ]

    assert find_order_block_candidates(swings, events) == []


def test_candidates_do_not_use_swing_before_broken_structure_anchor():
    swings = [
        _swing(1, 92, SwingType.LOW, 3, "LL"),
        _swing(2, 100, SwingType.HIGH, 4, "HH"),
        _swing(5, 94, SwingType.LOW, 7, "HL"),
    ]
    events = [
        StructureEvent(
            index=10,
            timestamp="2026-01-01 10:00",
            event="BULLISH_BOS",
            direction=Direction.UP,
            swing_index=2,
            swing_price=100,
            scope=StructureScope.EXTERNAL,
        )
    ]

    candidates = find_order_block_candidates(swings, events)

    assert [x.swing_index for x in candidates] == [5]


def test_real_structure_bullish_bos_maps_pullback_low_to_ob_candidate():
    from market_engine.structure import (
        CandleKind,
        StructuralCandle,
        process_structural_candles,
    )

    candles = [
        StructuralCandle(0, "00:00", 12, 9, 10, 11, CandleKind.UP),
        StructuralCandle(1, "00:30", 14, 10, 11, 13, CandleKind.UP),
        StructuralCandle(2, "01:00", 14.5, 11, 13, 14, CandleKind.UP),
        StructuralCandle(3, "01:30", 13.5, 10, 13, 11, CandleKind.DOWN),
        StructuralCandle(4, "02:00", 13, 8, 11, 9, CandleKind.DOWN),
        StructuralCandle(5, "02:30", 12, 9, 9, 11, CandleKind.UP),
        StructuralCandle(6, "03:00", 13, 10, 11, 12, CandleKind.UP),
        StructuralCandle(7, "03:30", 15, 11, 12, 14, CandleKind.UP),
    ]

    swings, events = process_structural_candles(candles)

    bos = next(
        event
        for event in events
        if event.event == "BULLISH_BOS"
    )

    candidates = find_order_block_candidates(swings, [bos])

    assert [
        (
            candidate.swing_index,
            candidate.swing_price,
            candidate.confirmation_index,
            candidate.event_index,
        )
        for candidate in candidates
    ] == [
        (4, 8, 6, 7)
    ]
    assert all(candidate.direction is Direction.UP for candidate in candidates)
    assert all(candidate.swing_type is SwingType.LOW for candidate in candidates)


def test_bearish_choch_produces_high_swing_candidate():
    swings = [
        _swing(2, 100, SwingType.LOW, 4, "LL"),
        _swing(5, 106, SwingType.HIGH, 7, "LH"),
    ]
    events = [
        StructureEvent(
            index=10,
            timestamp="2026-01-01 10:00",
            event="BEARISH_CHOCH",
            direction=Direction.DOWN,
            swing_index=2,
            swing_price=100,
            scope=StructureScope.EXTERNAL,
        )
    ]

    candidates = find_order_block_candidates(swings, events)

    assert [
        (
            candidate.swing_index,
            candidate.swing_type,
            candidate.direction,
            candidate.event,
        )
        for candidate in candidates
    ] == [
        (5, SwingType.HIGH, Direction.DOWN, "BEARISH_CHOCH")
    ]


def test_swing_confirmed_on_event_candle_is_excluded():
    swings = [
        _swing(2, 100, SwingType.HIGH, 4, "HH"),
        _swing(5, 94, SwingType.LOW, 10, "HL"),
    ]
    events = [
        StructureEvent(
            index=10,
            timestamp="2026-01-01 10:00",
            event="BULLISH_BOS",
            direction=Direction.UP,
            swing_index=2,
            swing_price=100,
            scope=StructureScope.EXTERNAL,
        )
    ]

    assert find_order_block_candidates(swings, events) == []


def test_candidate_scope_follows_swing_scope():
    swings = [
        _swing(2, 100, SwingType.HIGH, 4, "HH"),
        ValidSwing(
            index=5,
            timestamp="2026-01-01 05:00",
            price=94,
            swing_type=SwingType.LOW,
            confirmation_index=7,
            confirmation_timestamp="2026-01-01 07:00",
            label="HL",
            scope=StructureScope.INTERNAL,
        ),
    ]
    events = [
        StructureEvent(
            index=10,
            timestamp="2026-01-01 10:00",
            event="BULLISH_BOS",
            direction=Direction.UP,
            swing_index=2,
            swing_price=100,
            scope=StructureScope.EXTERNAL,
        )
    ]

    candidates = find_order_block_candidates(swings, events)

    assert len(candidates) == 1
    assert candidates[0].scope is StructureScope.INTERNAL


def test_order_block_zone_uses_full_candle_wick_to_wick_range():
    candidate = OrderBlockCandidate(
        direction=Direction.UP,
        swing_index=4,
        swing_price=8.0,
        swing_type=SwingType.LOW,
        confirmation_index=6,
        event_index=7,
        event="BULLISH_BOS",
        scope=StructureScope.EXTERNAL,
    )

    zone = resolve_order_block_zone(
        candidate,
        [
            {
                "index": 4,
                "open": 10.0,
                "high": 12.0,
                "low": 8.0,
                "close": 9.0,
            }
        ],
    )

    assert zone.low == 8.0
    assert zone.high == 12.0
    assert zone.size == 4.0


def test_order_block_zone_is_wick_to_wick_for_bearish_candidate():
    candidate = OrderBlockCandidate(
        direction=Direction.DOWN,
        swing_index=8,
        swing_price=15.0,
        swing_type=SwingType.HIGH,
        confirmation_index=10,
        event_index=11,
        event="BEARISH_BOS",
        scope=StructureScope.EXTERNAL,
    )

    zone = resolve_order_block_zone(
        candidate,
        [
            {
                "index": 8,
                "open": 14.0,
                "high": 15.0,
                "low": 11.0,
                "close": 13.0,
            }
        ],
    )

    assert zone.low == 11.0
    assert zone.high == 15.0
    assert zone.size == 4.0


def test_order_block_zone_rejects_missing_swing_candle():
    candidate = OrderBlockCandidate(
        direction=Direction.UP,
        swing_index=4,
        swing_price=8.0,
        swing_type=SwingType.LOW,
        confirmation_index=6,
        event_index=7,
        event="BULLISH_BOS",
        scope=StructureScope.EXTERNAL,
    )

    with pytest.raises(ValueError, match="Missing candle"):
        resolve_order_block_zone(candidate, [])


def test_order_block_zone_rejects_invalid_candle_range():
    candidate = OrderBlockCandidate(
        direction=Direction.UP,
        swing_index=4,
        swing_price=8.0,
        swing_type=SwingType.LOW,
        confirmation_index=6,
        event_index=7,
        event="BULLISH_BOS",
        scope=StructureScope.EXTERNAL,
    )

    with pytest.raises(ValueError, match="high < low"):
        resolve_order_block_zone(
            candidate,
            [
                {
                    "index": 4,
                    "open": 9.0,
                    "high": 8.0,
                    "low": 10.0,
                    "close": 9.0,
                }
            ],
        )
