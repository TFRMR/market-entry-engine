"""Deterministic market-structure state engine.

The engine processes candles in chronological order and emits only information
that is knowable at the current candle. Pullbacks are validator/reference
objects; valid swings are created only when their active pullback is broken by
a reversal.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import pandas as pd


class Direction(str, Enum):
    UP = "UP"
    DOWN = "DOWN"


class CandleKind(str, Enum):
    INSIDE = "INSIDE"
    UP = "UP"
    DOWN = "DOWN"
    OUTSIDE = "OUTSIDE"


class SwingType(str, Enum):
    HIGH = "HIGH"
    LOW = "LOW"


class StructureScope(str, Enum):
    INTERNAL = "INTERNAL"
    EXTERNAL = "EXTERNAL"


@dataclass(frozen=True)
class StructuralCandle:
    index: int
    timestamp: object
    high: float
    low: float
    open: float
    close: float
    kind: CandleKind


@dataclass(frozen=True)
class PullbackCandidate:
    index: int
    timestamp: object
    price: float
    direction: Direction
    extreme_index: int
    extreme_price: float


@dataclass(frozen=True)
class ValidSwing:
    index: int
    timestamp: object
    price: float
    swing_type: SwingType
    confirmation_index: int
    confirmation_timestamp: object
    label: str | None = None


@dataclass(frozen=True)
class StructureEvent:
    index: int
    timestamp: object
    event: str
    direction: Direction | None
    swing_index: int | None = None
    swing_price: float | None = None


@dataclass
class StructureState:
    direction: Direction | None = None
    extreme: StructuralCandle | None = None
    pullback: PullbackCandidate | None = None
    last_swing: ValidSwing | None = None
    previous_swing: ValidSwing | None = None
    scope: StructureScope = StructureScope.EXTERNAL


def classify_structural_candle(
    current: pd.Series,
    reference: pd.Series,
) -> CandleKind:
    """Classify current candle against the latest structural reference."""
    breaks_high = float(current["high"]) > float(reference["high"])
    breaks_low = float(current["low"]) < float(reference["low"])

    if breaks_high and breaks_low:
        return CandleKind.OUTSIDE
    if breaks_high:
        return CandleKind.UP
    if breaks_low:
        return CandleKind.DOWN
    return CandleKind.INSIDE


def _label_swing(
    swing: ValidSwing,
    previous: ValidSwing | None,
) -> str | None:
    if previous is None or previous.swing_type != swing.swing_type:
        return None

    if swing.swing_type is SwingType.HIGH:
        return "HH" if swing.price > previous.price else "LH"
    return "HL" if swing.price > previous.price else "LL"


def _breaks_pullback(
    candle: StructuralCandle,
    pullback: PullbackCandidate,
    direction: Direction,
) -> bool:
    if direction is Direction.UP:
        return candle.low < pullback.price
    return candle.high > pullback.price


def process_structural_candles(
    candles: Iterable[StructuralCandle],
) -> tuple[list[ValidSwing], list[StructureEvent]]:
    """Run the locked swing-validation state machine over structural candles.

    This is intentionally a small deterministic core. Initial direction is
    established by the first directional structural candle; later initialization
    from a historical BOS can be layered on without changing swing semantics.
    """
    state = StructureState()
    swings: list[ValidSwing] = []
    events: list[StructureEvent] = []

    for candle in candles:
        if state.direction is None:
            if candle.kind is CandleKind.UP:
                state.direction = Direction.UP
                state.extreme = candle
            elif candle.kind is CandleKind.DOWN:
                state.direction = Direction.DOWN
                state.extreme = candle
            continue

        if state.direction is Direction.UP:
            if candle.high > state.extreme.high:  # extend current leg
                state.extreme = candle

            if state.pullback is None:
                if candle.low < state.extreme.low:
                    state.pullback = PullbackCandidate(
                        index=candle.index,
                        timestamp=candle.timestamp,
                        price=candle.low,
                        direction=Direction.UP,
                        extreme_index=state.extreme.index,
                        extreme_price=state.extreme.high,
                    )
                continue

            if candle.high > state.extreme.high:
                state.extreme = candle
                # The old candidate belongs to the superseded extreme.
                state.pullback = None
                continue

            if _breaks_pullback(candle, state.pullback, Direction.UP):
                swing = ValidSwing(
                    index=state.extreme.index,
                    timestamp=state.extreme.timestamp,
                    price=state.extreme.high,
                    swing_type=SwingType.HIGH,
                    confirmation_index=candle.index,
                    confirmation_timestamp=candle.timestamp,
                )
                swing = ValidSwing(
                    **{**swing.__dict__, "label": _label_swing(swing, state.last_swing)}
                )
                state.previous_swing = state.last_swing
                state.last_swing = swing
                swings.append(swing)
                events.append(
                    StructureEvent(
                        index=candle.index,
                        timestamp=candle.timestamp,
                        event="SWING_HIGH_VALID",
                        direction=Direction.DOWN,
                        swing_index=swing.index,
                        swing_price=swing.price,
                    )
                )
                state.direction = Direction.DOWN
                state.extreme = candle
                state.pullback = None
                continue

            if candle.low < state.pullback.price:
                # Same-direction break: candidate is invalidated, but direction
                # remains UP. This branch is defensive; the opposing-break
                # condition above normally catches it.
                state.pullback = PullbackCandidate(
                    index=candle.index,
                    timestamp=candle.timestamp,
                    price=candle.low,
                    direction=Direction.UP,
                    extreme_index=state.extreme.index,
                    extreme_price=state.extreme.high,
                )

        else:
            if candle.low < state.extreme.low:
                state.extreme = candle

            if state.pullback is None:
                if candle.high > state.extreme.high:
                    state.pullback = PullbackCandidate(
                        index=candle.index,
                        timestamp=candle.timestamp,
                        price=candle.high,
                        direction=Direction.DOWN,
                        extreme_index=state.extreme.index,
                        extreme_price=state.extreme.low,
                    )
                continue

            if candle.low < state.extreme.low:
                state.extreme = candle
                state.pullback = None
                continue

            if _breaks_pullback(candle, state.pullback, Direction.DOWN):
                swing = ValidSwing(
                    index=state.extreme.index,
                    timestamp=state.extreme.timestamp,
                    price=state.extreme.low,
                    swing_type=SwingType.LOW,
                    confirmation_index=candle.index,
                    confirmation_timestamp=candle.timestamp,
                )
                swing = ValidSwing(
                    **{**swing.__dict__, "label": _label_swing(swing, state.last_swing)}
                )
                state.previous_swing = state.last_swing
                state.last_swing = swing
                swings.append(swing)
                events.append(
                    StructureEvent(
                        index=candle.index,
                        timestamp=candle.timestamp,
                        event="SWING_LOW_VALID",
                        direction=Direction.UP,
                        swing_index=swing.index,
                        swing_price=swing.price,
                    )
                )
                state.direction = Direction.UP
                state.extreme = candle
                state.pullback = None

    return swings, events


def build_structural_sequence(frame: pd.DataFrame) -> list[StructuralCandle]:
    """Reduce raw OHLCV to structural candles using inside/outside semantics."""
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if frame.empty:
        return []

    result: list[StructuralCandle] = []
    reference = frame.iloc[0]

    first = StructuralCandle(
        index=0,
        timestamp=reference["timestamp"],
        high=float(reference["high"]),
        low=float(reference["low"]),
        open=float(reference["open"]),
        close=float(reference["close"]),
        kind=CandleKind.UP if reference["close"] > reference["open"]
        else CandleKind.DOWN if reference["close"] < reference["open"]
        else CandleKind.INSIDE,
    )
    result.append(first)

    for i in range(1, len(frame)):
        row = frame.iloc[i]
        kind = classify_structural_candle(row, reference)
        if kind is CandleKind.INSIDE:
            continue

        candle = StructuralCandle(
            index=i,
            timestamp=row["timestamp"],
            high=float(row["high"]),
            low=float(row["low"]),
            open=float(row["open"]),
            close=float(row["close"]),
            kind=kind,
        )
        result.append(candle)

        # The structural sequence advances its reference on every structural
        # candle. An OUTSIDE candle therefore replaces the contained range.
        reference = row

    return result
