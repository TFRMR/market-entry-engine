"""Deterministic market-structure state engine.

The engine processes candles in chronological order and emits only information
that is knowable at the current candle. Pullbacks are validator/reference
objects; valid swings are created only when the active pullback is broken by
a reversal. A BOS event is emitted only when a previously confirmed valid swing
is broken.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

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
    scope: StructureScope = StructureScope.EXTERNAL


@dataclass(frozen=True)
class StructureEvent:
    index: int
    timestamp: object
    event: str
    direction: Direction | None
    swing_index: int | None = None
    swing_price: float | None = None
    scope: StructureScope = StructureScope.EXTERNAL


@dataclass
class StructureState:
    direction: Direction | None = None
    extreme: StructuralCandle | None = None
    pullback: PullbackCandidate | None = None
    last_swing: ValidSwing | None = None
    previous_swing: ValidSwing | None = None
    scope: StructureScope = StructureScope.EXTERNAL


@dataclass(frozen=True)
class StructureCheckpoint:
    index: int
    direction: Direction
    extreme: StructuralCandle
    pullback: PullbackCandidate | None
    last_swing: ValidSwing | None
    previous_swing: ValidSwing | None
    scope: StructureScope
    last_high: ValidSwing | None
    last_low: ValidSwing | None
    broken_high_index: int | None
    broken_low_index: int | None
    external_high: ValidSwing | None = None
    external_low: ValidSwing | None = None


@dataclass(frozen=True)
class StructureSnapshot:
    index: int
    direction: Direction | None
    last_high: ValidSwing | None
    last_low: ValidSwing | None
    historical_last_high: ValidSwing | None
    historical_last_low: ValidSwing | None
    last_swing_confirmation_index: int | None
    last_bos_index: int | None


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
    scope: StructureScope,
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
        scope=scope,
    )
    return ValidSwing(
        index=swing.index,
        timestamp=swing.timestamp,
        price=swing.price,
        swing_type=swing.swing_type,
        confirmation_index=swing.confirmation_index,
        confirmation_timestamp=swing.confirmation_timestamp,
        label=_label_swing(swing, previous_same_type),
        scope=scope,
    )


def _process_structural_candles(
    candles: Iterable[StructuralCandle],
    checkpoint: StructureCheckpoint | None = None,
    stop_after_first_bos: bool = False,
    collect_snapshots: bool = False,
) -> tuple[
    list[ValidSwing],
    list[StructureEvent],
    StructureCheckpoint | None,
]:
    state = StructureState()
    swings: list[ValidSwing] = []
    events: list[StructureEvent] = []
    snapshots: list[StructureSnapshot] = []

    last_high: ValidSwing | None = None
    last_low: ValidSwing | None = None
    broken_high_index: int | None = None
    broken_low_index: int | None = None
    external_high: ValidSwing | None = None
    external_low: ValidSwing | None = None
    historical_last_high: ValidSwing | None = None
    historical_last_low: ValidSwing | None = None
    last_swing_confirmation_index: int | None = None
    last_bos_index: int | None = None

    if checkpoint is not None:
        state.direction = checkpoint.direction
        state.extreme = checkpoint.extreme
        state.pullback = checkpoint.pullback
        state.last_swing = checkpoint.last_swing
        state.previous_swing = checkpoint.previous_swing
        state.scope = checkpoint.scope

        last_high = checkpoint.last_high
        last_low = checkpoint.last_low
        broken_high_index = checkpoint.broken_high_index
        broken_low_index = checkpoint.broken_low_index
        external_high = checkpoint.external_high
        external_low = checkpoint.external_low
        historical_last_high = checkpoint.historical_last_high
        historical_last_low = checkpoint.historical_last_low
        if checkpoint.last_swing is not None:
            last_swing_confirmation_index = checkpoint.last_swing.confirmation_index

    for candle in candles:
        high_target = (
            external_high
            if external_high is not None
            and candle.index > external_high.confirmation_index
            and candle.high > external_high.price
            and broken_high_index != external_high.index
            else last_high
        )
        bos_high = (
            high_target
            if high_target is not None
            and candle.index > high_target.confirmation_index
            and candle.high > high_target.price
            and broken_high_index != high_target.index
            else None
        )

        low_target = (
            external_low
            if external_low is not None
            and candle.index > external_low.confirmation_index
            and candle.low < external_low.price
            and broken_low_index != external_low.index
            else last_low
        )
        bos_low = (
            low_target
            if low_target is not None
            and candle.index > low_target.confirmation_index
            and candle.low < low_target.price
            and broken_low_index != low_target.index
            else None
        )

        first_bos = bos_high is not None or bos_low is not None
        external_boundary_broken = (
            (bos_high is not None and bos_high.scope is StructureScope.EXTERNAL)
            or (bos_low is not None and bos_low.scope is StructureScope.EXTERNAL)
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
                    scope=bos_high.scope,
                )
            )
            broken_high_index = bos_high.index
            last_bos_index = candle.index

        if bos_low is not None:
            events.append(
                StructureEvent(
                    index=candle.index,
                    timestamp=candle.timestamp,
                    event="BEARISH_BOS",
                    direction=Direction.DOWN,
                    swing_index=bos_low.index,
                    swing_price=bos_low.price,
                    scope=bos_low.scope,
                )
            )
            broken_low_index = bos_low.index
            last_bos_index = candle.index

        rebuild_after_candle = external_boundary_broken
        broke_high = (
            bos_high is not None
            and bos_high.scope is StructureScope.EXTERNAL
        )
        broke_low = (
            bos_low is not None
            and bos_low.scope is StructureScope.EXTERNAL
        )

        if state.direction is None:
            if candle.kind is CandleKind.UP:
                state.direction = Direction.UP
                state.extreme = candle
            elif candle.kind is CandleKind.DOWN:
                state.direction = Direction.DOWN
                state.extreme = candle

            snapshots.append(
                StructureSnapshot(
                    index=candle.index,
                    direction=state.direction,
                    last_high=last_high,
                    last_low=last_low,
                    historical_last_high=historical_last_high,
                    historical_last_low=historical_last_low,
                    last_swing_confirmation_index=last_swing_confirmation_index,
                    last_bos_index=last_bos_index,
                )
            )

            if first_bos and stop_after_first_bos:
                checkpoint_out = StructureCheckpoint(
                    index=candle.index,
                    direction=state.direction,
                    extreme=state.extreme,
                    pullback=state.pullback,
                    last_swing=state.last_swing,
                    previous_swing=state.previous_swing,
                    scope=state.scope,
                    last_high=last_high,
                    last_low=last_low,
                    broken_high_index=broken_high_index,
                    broken_low_index=broken_low_index,
                    external_high=external_high,
                    external_low=external_low,
                    historical_last_high=historical_last_high,
                    historical_last_low=historical_last_low,
                )
                if collect_snapshots:
                    return swings, events, checkpoint_out, snapshots
                return swings, events, checkpoint_out

            continue

        assert state.extreme is not None

        if state.direction is Direction.UP:
            if candle.high > state.extreme.high:
                state.extreme = candle
                state.pullback = None
            elif state.pullback is None:
                state.pullback = PullbackCandidate(
                    index=candle.index,
                    timestamp=candle.timestamp,
                    price=candle.low,
                    direction=Direction.UP,
                    extreme_index=state.extreme.index,
                    extreme_price=state.extreme.high,
                )
            elif candle.low < state.pullback.price:
                swing_scope = (
                    StructureScope.INTERNAL
                    if external_high is not None
                    and external_low is not None
                    and external_low.price < state.extreme.high < external_high.price
                    else StructureScope.EXTERNAL
                )
                swing = _confirm_swing(
                    state,
                    candle,
                    SwingType.HIGH,
                    last_high,
                    swing_scope,
                )
                state.previous_swing = state.last_swing
                state.last_swing = swing
                last_high = swing
                historical_last_high = swing
                if swing.scope is StructureScope.EXTERNAL:
                    external_high = swing
                swings.append(swing)
                last_swing_confirmation_index = candle.index
                events.append(
                    StructureEvent(
                        index=candle.index,
                        timestamp=candle.timestamp,
                        event="SWING_HIGH_VALID",
                        direction=Direction.DOWN,
                        swing_index=swing.index,
                        swing_price=swing.price,
                        scope=swing.scope,
                    )
                )
                state.direction = Direction.DOWN
                state.extreme = candle
                state.pullback = None
            elif candle.low > state.pullback.price:
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
            elif state.pullback is None:
                state.pullback = PullbackCandidate(
                    index=candle.index,
                    timestamp=candle.timestamp,
                    price=candle.high,
                    direction=Direction.DOWN,
                    extreme_index=state.extreme.index,
                    extreme_price=state.extreme.low,
                )
            elif candle.high > state.pullback.price:
                swing_scope = (
                    StructureScope.INTERNAL
                    if external_high is not None
                    and external_low is not None
                    and external_low.price < state.extreme.low < external_high.price
                    else StructureScope.EXTERNAL
                )
                swing = _confirm_swing(
                    state,
                    candle,
                    SwingType.LOW,
                    last_low,
                    swing_scope,
                )
                state.previous_swing = state.last_swing
                state.last_swing = swing
                last_low = swing
                historical_last_low = swing
                if swing.scope is StructureScope.EXTERNAL:
                    external_low = swing
                swings.append(swing)
                events.append(
                    StructureEvent(
                        index=candle.index,
                        timestamp=candle.timestamp,
                        event="SWING_LOW_VALID",
                        direction=Direction.UP,
                        swing_index=swing.index,
                        swing_price=swing.price,
                        scope=swing.scope,
                    )
                )
                state.direction = Direction.UP
                state.extreme = candle
                state.pullback = None
            elif candle.high < state.pullback.price:
                state.pullback = PullbackCandidate(
                    index=candle.index,
                    timestamp=candle.timestamp,
                    price=candle.high,
                    direction=Direction.DOWN,
                    extreme_index=state.extreme.index,
                    extreme_price=state.extreme.low,
                )

        if rebuild_after_candle:
            # Close the broken outer context only after this candle has had
            # its normal structural effect. A swing confirmed on this candle
            # belongs to the rebuilt context; older internal swings do not.
            current_high = (
                state.last_swing
                if state.last_swing is not None
                and state.last_swing.confirmation_index == candle.index
                and state.last_swing.swing_type is SwingType.HIGH
                else None
            )
            current_low = (
                state.last_swing
                if state.last_swing is not None
                and state.last_swing.confirmation_index == candle.index
                and state.last_swing.swing_type is SwingType.LOW
                else None
            )
            last_high = current_high
            last_low = current_low
            broken_high_index = None
            broken_low_index = None
            external_high = current_high
            external_low = current_low
            state.previous_swing = None
            state.scope = StructureScope.EXTERNAL
            last_swing_confirmation_index = (
                candle.index
                if current_high is not None or current_low is not None
                else None
            )
            if current_high is None and current_low is None:
                state.last_swing = None
                state.direction = (
                    Direction.UP
                    if broke_high and not broke_low
                    else Direction.DOWN
                    if broke_low and not broke_high
                    else state.direction
                )
                state.extreme = candle
                state.pullback = None

        snapshots.append(
            StructureSnapshot(
                index=candle.index,
                direction=state.direction,
                last_high=last_high,
                last_low=last_low,
                historical_last_high=historical_last_high,
                historical_last_low=historical_last_low,
                last_swing_confirmation_index=last_swing_confirmation_index,
                last_bos_index=last_bos_index,
            )
        )

        if first_bos and stop_after_first_bos:
            checkpoint_out = StructureCheckpoint(
                index=candle.index,
                direction=state.direction,
                extreme=state.extreme,
                pullback=state.pullback,
                last_swing=state.last_swing,
                previous_swing=state.previous_swing,
                scope=state.scope,
                last_high=last_high,
                last_low=last_low,
                broken_high_index=broken_high_index,
                broken_low_index=broken_low_index,
                external_high=external_high,
                external_low=external_low,
                historical_last_high=historical_last_high,
                historical_last_low=historical_last_low,
            )
            if collect_snapshots:
                return swings, events, checkpoint_out, snapshots
            return swings, events, checkpoint_out

    if collect_snapshots:
        return swings, events, None, snapshots
    return swings, events, None


def process_structural_candles_with_context(
    candles: Iterable[StructuralCandle],
) -> tuple[list[ValidSwing], list[StructureEvent], list[StructureSnapshot]]:
    """Run structure and return one post-candle snapshot per input candle."""
    swings, events, _, snapshots = _process_structural_candles(
        candles,
        collect_snapshots=True,
    )
    return swings, events, snapshots


def process_structural_candles(
    candles: Iterable[StructuralCandle],
) -> tuple[list[ValidSwing], list[StructureEvent]]:
    """Run deterministic market structure across the supplied candles."""
    swings, events, _ = _process_structural_candles(candles)
    return swings, events


def find_first_bos(candles):
    _, events, _ = _process_structural_candles(
        candles,
        stop_after_first_bos=True,
    )

    for event in events:
        if event.event.endswith("_BOS"):
            return event

    return None


def find_first_bos_checkpoint(candles):
    """Process history until the first BOS and return its actual state."""
    swings, events, checkpoint = _process_structural_candles(
        candles,
        stop_after_first_bos=True,
    )

    anchor = next(
        (event for event in events if event.event.endswith("_BOS")),
        None,
    )

    if anchor is None or checkpoint is None:
        return None

    return anchor, checkpoint, swings


def process_from_first_bos(candles):
    result = find_first_bos_checkpoint(candles)

    if result is None:
        return None

    anchor, checkpoint, historical_swings = result

    candle_list = list(candles)
    forward_candles = [
        candle
        for candle in candle_list
        if candle.index > checkpoint.index
    ]

    future_swings, future_events, _ = _process_structural_candles(
        forward_candles,
        checkpoint=checkpoint,
    )

    context_swings = [
        swing
        for swing in historical_swings
        if swing.index == anchor.swing_index
    ]
    context_swings.extend(
        swing
        for swing in future_swings
        if swing.confirmation_index > anchor.index
    )

    forward_events = [
        event
        for event in future_events
        if event.index > anchor.index
    ]

    return anchor, context_swings, forward_events

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
