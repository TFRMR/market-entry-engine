"""Deterministic POI, inducement, OBIM, and price-interaction research primitives."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from market_engine.order_block import (
    OrderBlockCandidate,
    resolve_order_block_zone,
)
from market_engine.structure import (
    Direction,
    StructureEvent,
    StructureScope,
    SwingType,
    ValidSwing,
)


class POIType(str, Enum):
    FVG = "FVG"
    ORDER_BLOCK = "ORDER_BLOCK"
    OBIM = "OBIM"
    LIQUIDITY = "LIQUIDITY"
    STRUCTURAL_SR = "STRUCTURAL_SR"


class POILifecycle(str, Enum):
    CREATED = "CREATED"
    TESTED = "TESTED"
    SWEPT = "SWEPT"
    BROKEN = "BROKEN"
    FLIPPED = "FLIPPED"
    INVALIDATED = "INVALIDATED"


class PriceInteraction(str, Enum):
    NONE = "NONE"
    TESTED = "TESTED"
    SWEPT = "SWEPT"
    BROKEN = "BROKEN"


@dataclass(frozen=True)
class POIRecord:
    poi_type: POIType
    direction: Direction | None
    created_index: int
    low: float
    high: float
    source_index: int | None = None
    scope: StructureScope | None = None


@dataclass(frozen=True)
class InducementEvent:
    index: int
    direction: Direction
    swing_index: int
    swing_price: float
    scope: StructureScope


def _zone_overlap(
    first_low: float,
    first_high: float,
    second_low: float,
    second_high: float,
) -> bool:
    return max(first_low, second_low) <= min(first_high, second_high)


def build_inducement_events(
    swings: list[ValidSwing],
    candles: pd.DataFrame,
) -> list[InducementEvent]:
    """Detect sweeps of confirmed INTERNAL swings as objective inducement events.

    This is deliberately a factual event definition: an internal swing is
    confirmed first, then a later candle trades beyond that level and closes
    back on the original side. No outcome or quality judgement is embedded.
    """
    if not {"high", "low", "close"}.issubset(candles.columns):
        raise ValueError("Inducement detection requires high, low, and close.")

    internal_highs = [
        swing for swing in swings
        if swing.scope is StructureScope.INTERNAL
        and swing.swing_type is SwingType.HIGH
    ]
    internal_lows = [
        swing for swing in swings
        if swing.scope is StructureScope.INTERNAL
        and swing.swing_type is SwingType.LOW
    ]

    events: list[InducementEvent] = []
    for position in range(len(candles)):
        high = float(candles.iloc[position]["high"])
        low = float(candles.iloc[position]["low"])
        close = float(candles.iloc[position]["close"])

        highs = [
            swing for swing in internal_highs
            if swing.confirmation_index < position
            and high > swing.price
            and close <= swing.price
        ]
        if highs:
            swing = max(highs, key=lambda item: item.confirmation_index)
            events.append(
                InducementEvent(
                    index=position,
                    direction=Direction.DOWN,
                    swing_index=swing.index,
                    swing_price=swing.price,
                    scope=swing.scope,
                )
            )

        lows = [
            swing for swing in internal_lows
            if swing.confirmation_index < position
            and low < swing.price
            and close >= swing.price
        ]
        if lows:
            swing = max(lows, key=lambda item: item.confirmation_index)
            events.append(
                InducementEvent(
                    index=position,
                    direction=Direction.UP,
                    swing_index=swing.index,
                    swing_price=swing.price,
                    scope=swing.scope,
                )
            )

    return events


def build_poi_records(
    frame: pd.DataFrame,
    swings: list[ValidSwing],
    events: list[StructureEvent],
    order_blocks: list[OrderBlockCandidate],
) -> list[POIRecord]:
    """Build point-in-time POI objects from already-confirmed references.

    OBIM is created only when a historical OB zone overlaps a historical FVG
    zone. It is a compositional fact, not a quality or probability label.
    """
    required = {"high", "low"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "POI reconstruction requires: " + ", ".join(sorted(missing))
        )

    records: list[POIRecord] = []

    # FVG records.
    for position in range(2, len(frame)):
        high_two_back = float(frame.iloc[position - 2]["high"])
        low_two_back = float(frame.iloc[position - 2]["low"])
        current_high = float(frame.iloc[position]["high"])
        current_low = float(frame.iloc[position]["low"])

        if current_low > high_two_back:
            records.append(
                POIRecord(
                    poi_type=POIType.FVG,
                    direction=Direction.UP,
                    created_index=position,
                    low=high_two_back,
                    high=current_low,
                    source_index=position,
                )
            )
        elif current_high < low_two_back:
            records.append(
                POIRecord(
                    poi_type=POIType.FVG,
                    direction=Direction.DOWN,
                    created_index=position,
                    low=current_high,
                    high=low_two_back,
                    source_index=position,
                )
            )

    # OB records.
    for candidate in order_blocks:
        candle = frame.iloc[candidate.swing_index]
        records.append(
            POIRecord(
                poi_type=POIType.ORDER_BLOCK,
                direction=candidate.direction,
                created_index=candidate.event_index,
                low=float(candle["low"]),
                high=float(candle["high"]),
                source_index=candidate.swing_index,
                scope=candidate.scope,
            )
        )

    # OBIM records: pair each OB with an overlapping FVG already known by the
    # OB event. This preserves chronology and avoids future pairing.
    fvgs = [record for record in records if record.poi_type is POIType.FVG]
    for candidate in order_blocks:
        ob = next(
            record for record in records
            if record.poi_type is POIType.ORDER_BLOCK
            and record.source_index == candidate.swing_index
            and record.created_index == candidate.event_index
            and record.direction is candidate.direction
        )
        overlaps = [
            fvg for fvg in fvgs
            if fvg.created_index <= candidate.event_index
            and fvg.direction is candidate.direction
            and _zone_overlap(ob.low, ob.high, fvg.low, fvg.high)
        ]
        for fvg in overlaps:
            records.append(
                POIRecord(
                    poi_type=POIType.OBIM,
                    direction=candidate.direction,
                    created_index=candidate.event_index,
                    low=max(ob.low, fvg.low),
                    high=min(ob.high, fvg.high),
                    source_index=candidate.swing_index,
                    scope=candidate.scope,
                )
            )

    # Structural S/R records use confirmed swing levels as deterministic
    # reference zones. They remain point levels until a separate zone model is
    # introduced.
    for swing in swings:
        records.append(
            POIRecord(
                poi_type=POIType.STRUCTURAL_SR,
                direction=(
                    Direction.DOWN
                    if swing.swing_type is SwingType.HIGH
                    else Direction.UP
                ),
                created_index=swing.confirmation_index,
                low=float(swing.price),
                high=float(swing.price),
                source_index=swing.index,
                scope=swing.scope,
            )
        )

    # Liquidity records are represented by the same confirmed swing reference.
    for swing in swings:
        records.append(
            POIRecord(
                poi_type=POIType.LIQUIDITY,
                direction=(
                    Direction.DOWN
                    if swing.swing_type is SwingType.HIGH
                    else Direction.UP
                ),
                created_index=swing.confirmation_index,
                low=float(swing.price),
                high=float(swing.price),
                source_index=swing.index,
                scope=swing.scope,
            )
        )

    return sorted(
        records,
        key=lambda record: (
            record.created_index,
            record.poi_type.value,
            record.source_index if record.source_index is not None else -1,
        ),
    )


def classify_poi_interaction(
    poi: POIRecord,
    high: float,
    low: float,
    close: float,
) -> PriceInteraction:
    """Classify one candle's objective interaction with a POI zone."""
    if poi.high < poi.low:
        raise ValueError("POI zone high must be >= low.")

    touched = low <= poi.high and high >= poi.low
    if not touched:
        return PriceInteraction.NONE

    if poi.high > poi.low:
        swept = (
            (low < poi.low and close >= poi.low)
            or (high > poi.high and close <= poi.high)
        )
        if swept:
            return PriceInteraction.SWEPT

        broken = close < poi.low or close > poi.high
        if broken:
            return PriceInteraction.BROKEN

    return PriceInteraction.TESTED


def lifecycle_after_interaction(
    previous: POILifecycle,
    interaction: PriceInteraction,
) -> POILifecycle:
    """Advance a POI lifecycle from an observed interaction."""
    if interaction is PriceInteraction.SWEPT:
        return POILifecycle.SWEPT
    if interaction is PriceInteraction.BROKEN:
        return POILifecycle.BROKEN
    if interaction is PriceInteraction.TESTED:
        return POILifecycle.TESTED
    return previous
