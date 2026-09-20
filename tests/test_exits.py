from market_engine.entry import SetupCandidate
from market_engine.exits import build_exit_areas
from market_engine.structure import Direction, StructureScope, SwingType, ValidSwing


def swing(index: int, price: float, kind: SwingType, confirmation: int) -> ValidSwing:
    return ValidSwing(
        index=index,
        timestamp=index,
        price=price,
        swing_type=kind,
        confirmation_index=confirmation,
        confirmation_timestamp=confirmation,
        scope=StructureScope.EXTERNAL,
    )


def candidate(direction: Direction) -> SetupCandidate:
    return SetupCandidate(
        setup_index=5,
        setup_timestamp=5,
        direction=direction,
        entry_index=6,
        entry_timestamp=6,
        entry_price=100.0,
        invalidation_price=95.0 if direction is Direction.UP else 105.0,
        risk=5.0,
        invalidation_swing_index=4,
    )


def test_long_uses_confirmed_highs_above_entry_in_distance_order():
    areas = build_exit_areas(
        candidate(Direction.UP),
        [swing(1, 90, SwingType.LOW, 1), swing(2, 106, SwingType.HIGH, 2),
         swing(3, 112, SwingType.HIGH, 3), swing(4, 104, SwingType.HIGH, 4),
         swing(7, 120, SwingType.HIGH, 7)],
    )
    assert [area.price for area in areas] == [104.0, 106.0]
    assert [area.distance for area in areas] == [4.0, 6.0]


def test_short_uses_confirmed_lows_below_entry_and_ignores_future_confirmation():
    areas = build_exit_areas(
        candidate(Direction.DOWN),
        [swing(1, 110, SwingType.HIGH, 1), swing(2, 96, SwingType.LOW, 2),
         swing(3, 88, SwingType.LOW, 3), swing(4, 98, SwingType.LOW, 4),
         swing(7, 80, SwingType.LOW, 7)],
    )
    assert [area.price for area in areas] == [98.0, 96.0]
    assert [area.distance for area in areas] == [2.0, 4.0]


def test_exit_areas_have_explicit_source_metadata():
    areas = build_exit_areas(
        candidate(Direction.UP),
        [swing(2, 106, SwingType.HIGH, 2)],
    )
    assert areas[0].source_type == "VALID_SWING"
    assert areas[0].source_index == 2
    assert areas[0].source_confirmation_index == 2
