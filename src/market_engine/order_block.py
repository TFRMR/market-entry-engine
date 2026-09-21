"""Deterministic structural order-block candidate extraction."""

from __future__ import annotations

from dataclasses import dataclass

from .structure import (
    Direction,
    StructureEvent,
    StructureScope,
    SwingType,
    ValidSwing,
)


@dataclass(frozen=True)
class OrderBlockCandidate:
    """A structural swing candidate associated with a BOS/CHoCH leg."""

    direction: Direction
    swing_index: int
    swing_price: float
    swing_type: SwingType
    confirmation_index: int
    event_index: int
    event: str
    scope: StructureScope


@dataclass(frozen=True)
class OrderBlockZone:
    """Price zone represented by an order-block candidate candle."""

    low: float
    high: float

    @property
    def size(self) -> float:
        """Return the wick-to-wick zone width."""
        return self.high - self.low


def resolve_order_block_zone(
    candidate: OrderBlockCandidate,
    candles: list[dict],
) -> OrderBlockZone:
    """Resolve an OB candidate into its full candle wick-to-wick zone.

    The candidate's swing candle is used as the zone source. The complete
    candle range is preserved: low -> high, including both wicks.
    """
    candle = next(
        (row for row in candles if row["index"] == candidate.swing_index),
        None,
    )
    if candle is None:
        raise ValueError(
            f"Missing candle for order-block candidate "
            f"index={candidate.swing_index}"
        )

    low = float(candle["low"])
    high = float(candle["high"])

    if high < low:
        raise ValueError(
            f"Invalid candle range for order-block candidate "
            f"index={candidate.swing_index}: high < low"
        )

    return OrderBlockZone(low=low, high=high)


def find_order_block_candidates(
    swings: list[ValidSwing],
    events: list[StructureEvent],
) -> list[OrderBlockCandidate]:
    """Find structural OB candidates from confirmed swings inside BOS/CHoCH legs.

    A bullish event uses confirmed LOW swings after the broken HIGH and before
    the event. A bearish event uses confirmed HIGH swings after the broken LOW
    and before the event.

    The function preserves every qualifying swing. Selection/ranking of POIs is
    intentionally outside this primitive.
    """
    candidates: list[OrderBlockCandidate] = []

    swings_by_type = {
        SwingType.HIGH: [
            swing
            for swing in swings
            if swing.swing_type is SwingType.HIGH
        ],
        SwingType.LOW: [
            swing
            for swing in swings
            if swing.swing_type is SwingType.LOW
        ],
    }
    swings_by_index = {swing.index: swing for swing in swings}

    for event in events:
        if event.event not in {
            "BULLISH_BOS",
            "BEARISH_BOS",
            "BULLISH_CHOCH",
            "BEARISH_CHOCH",
        }:
            continue

        if event.swing_index is None:
            continue

        if event.direction is Direction.UP:
            candidate_type = SwingType.LOW
            broken_type = SwingType.HIGH
        else:
            candidate_type = SwingType.HIGH
            broken_type = SwingType.LOW

        broken = swings_by_index.get(event.swing_index)
        if broken is None or broken.swing_type is not broken_type:
            continue

        qualifying = [
            swing
            for swing in swings_by_type[candidate_type]
            if (
                swing.index > event.swing_index
                and swing.index < event.index
                and swing.confirmation_index < event.index
            )
        ]

        for swing in qualifying:
            candidates.append(
                OrderBlockCandidate(
                    direction=event.direction,
                    swing_index=swing.index,
                    swing_price=swing.price,
                    swing_type=swing.swing_type,
                    confirmation_index=swing.confirmation_index,
                    event_index=event.index,
                    event=event.event,
                    scope=swing.scope,
                )
            )

    return candidates
