"""Deterministic market-structure state engine.

The engine processes candles in chronological order and emits only information
that is knowable at the current candle. Pullbacks are validator/reference
objects; valid swings are created only when the active pullback is broken by
a reversal. A BOS event is emitted only when a previously confirmed valid swing
is broken.
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


def _confirm_swing(
    state: StructureState,
    candle: StructuralCandle,
    swing_type: SwingType,
    previous_same_type: ValidSwing | None,
) -> ValidSwing:
    assert state.extreme is not None
    swing = ValidSwing(
        index=state.extreme.index,
        timestamp=state.extreme.timestamp,
        price=(
            state.extreme.high
            if swing_type is SwingType.HIGH
            else state.extreme.low
        ),
        swing_type=swing_type,
        confirmation_index=candle.index,
        confirmation_timestamp=candle.timestamp,
    )
    return ValidSwing(
        index=swing.index,
        timestamp=swing.timestamp,
        price=swing.price,
        swing_type=swing.swing_type,
        confirmation_index=swing.confirmation_index,
        confirmation_timestamp=swing.confirmation_timestamp,
        label=_label_swing(swing, previous_same_type),
    )


def process_structural_candles(
    candles: Iterable[StructuralCandle],
) -> tuple[list[ValidSwing], list[StructureEvent]]:
    """Run deterministic pullback -> valid-swing validation.

    A directional leg owns one active extreme. A candle that extends that
    extreme invalidates the previous pullback candidate and starts a new leg
    reference. A non-extending candle creates the active pullback candidate.
    The next opposing break of that candidate validates the historical extreme
    as a swing. Confirmation is timestamped at the breaking candle.
    """
    state = StructureState()
    swings: list[ValidSwing] = []
    events: list[StructureEvent] = []
    last_high: ValidSwing | None = None
    last_low: ValidSwing | None = None
    broken_high_index: int | None = None
    broken_low_index: int | None = None

    for candle in candles:
        # Snapshot valid swings before processing this candle. A swing that is
        # confirmed on this candle is not itself eligible to be a BOS target.
        bos_high = (
            last_high
            if last_high is not None
            and candle.index > last_high.confirmation_index
            and candle.high > last_high.price
            and broken_high_index != last_high.index
            else None
        )
        bos_low = (
            last_low
            if last_low is not None
            and candle.index > last_low.confirmation_index
            and candle.low < last_low.price
            and broken_low_index != last_low.index
            else None
        )

        if bos_high is not None:
            events.append(
                StructureEvent(
                    index=candle.index,
                    timestamp=candle.timestamp,
                    event="BULLISH_BOS",
                    direction=Direction.UP,
                    swing_index=bos_high.index,
                    swing_price=bos_high.price,
                )
            )
            broken_high_index = bos_high.index

        if bos_low is not None:
            events.append(
                StructureEvent(
                    index=candle.index,
                    timestamp=candle.timestamp,
                    event="BEARISH_BOS",
                    direction=Direction.DOWN,
                    swing_index=bos_low.index,
                    swing_price=bos_low.price,
                )
            )
            broken_low_index = bos_low.index

        if state.direction is None:
            if candle.kind is CandleKind.UP:
                state.direction = Direction.UP
                state.extreme = candle
            elif candle.kind is CandleKind.DOWN:
                state.direction = Direction.DOWN
                state.extreme = candle
            continue

        assert state.extreme is not None

        if state.direction is Direction.UP:
            if candle.high > state.extreme.high:
                state.extreme = candle
                state.pullback = None
                continue

            if state.pullback is None:
                state.pullback = PullbackCandidate(
                    index=candle.index,
                    timestamp=candle.timestamp,
                    price=candle.low,
                    direction=Direction.UP,
                    extreme_index=state.extreme.index,
                    extreme_price=state.extreme.high,
                )
                continue

            if candle.low < state.pullback.price:
                swing = _confirm_swing(state, candle, SwingType.HIGH, last_high)
                state.previous_swing = state.last_swing
                state.last_swing = swing
                last_high = swing
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

            if candle.low > state.pullback.price:
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
                state.pullback = None
                continue

            if state.pullback is None:
                state.pullback = PullbackCandidate(
                    index=candle.index,
                    timestamp=candle.timestamp,
                    price=candle.high,
                    direction=Direction.DOWN,
                    extreme_index=state.extreme.index,
                    extreme_price=state.extreme.low,
                )
                continue

            if candle.high > state.pullback.price:
                swing = _confirm_swing(state, candle, SwingType.LOW, last_low)
                state.previous_swing = state.last_swing
                state.last_swing = swing
                last_low = swing
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
                continue

            if candle.high < state.pullback.price:
                state.pullback = PullbackCandidate(
                    index=candle.index,
                    timestamp=candle.timestamp,
                    price=candle.high,
                    direction=Direction.DOWN,
                    extreme_index=state.extreme.index,
                    extreme_price=state.extreme.low,
                )

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
        kind=(
            CandleKind.UP
            if reference["close"] > reference["open"]
            else CandleKind.DOWN
            if reference["close"] < reference["open"]
            else CandleKind.INSIDE
        ),
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
        reference = row

    return result


def find_first_bos(candles):
    _, events = process_structural_candles(candles)

    for event in events:
        if event.event.endswith("_BOS"):
            return event

    return None


def process_from_first_bos(candles):
    anchor = find_first_bos(candles)

    if anchor is None:
        return None

    forward_candles = [
        candle for candle in candles
        if candle.index >= anchor.index
    ]

    swings, events = process_structural_candles(forward_candles)

    return anchor, swings, events


def process_from_first_bos(candles):
    anchor = find_first_bos(candles)

    if anchor is None:
        return None

    forward_candles = [
        candle for candle in candles
        if candle.index >= anchor.index
    ]

    swings, events = process_structural_candles(forward_candles)

    return anchor, swings, events
