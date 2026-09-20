"""Deterministic setup-candidate generation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from market_engine.structure import (
    Direction,
    StructureEvent,
    SwingType,
    ValidSwing,
    build_structural_sequence,
    process_structural_candles,
)


@dataclass(frozen=True)
class SetupCandidate:
    setup_index: int
    setup_timestamp: object
    direction: Direction
    entry_index: int
    entry_timestamp: object
    entry_price: float
    invalidation_price: float
    risk: float
    invalidation_swing_index: int


def _latest_confirmed_swing(
    swings: list[ValidSwing],
    swing_type: SwingType,
    before_index: int,
) -> ValidSwing | None:
    candidates = [
        swing
        for swing in swings
        if swing.swing_type is swing_type
        and swing.confirmation_index < before_index
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda swing: swing.confirmation_index)


def _candidate_from_bos(
    frame: pd.DataFrame,
    event: StructureEvent,
    swings: list[ValidSwing],
) -> SetupCandidate | None:
    entry_index = event.index + 1
    if entry_index >= len(frame):
        return None

    if event.event == "BULLISH_BOS":
        invalidation = _latest_confirmed_swing(
            swings, SwingType.LOW, event.index
        )
        direction = Direction.UP
    elif event.event == "BEARISH_BOS":
        invalidation = _latest_confirmed_swing(
            swings, SwingType.HIGH, event.index
        )
        direction = Direction.DOWN
    else:
        return None

    if invalidation is None:
        return None

    entry_price = float(frame.iloc[entry_index]["open"])
    invalidation_price = float(invalidation.price)

    risk = (
        entry_price - invalidation_price
        if direction is Direction.UP
        else invalidation_price - entry_price
    )
    if risk <= 0:
        return None

    return SetupCandidate(
        setup_index=event.index,
        setup_timestamp=event.timestamp,
        direction=direction,
        entry_index=entry_index,
        entry_timestamp=frame.iloc[entry_index]["timestamp"],
        entry_price=entry_price,
        invalidation_price=invalidation_price,
        risk=risk,
        invalidation_swing_index=invalidation.index,
    )


def build_setup_candidates(frame: pd.DataFrame) -> list[SetupCandidate]:
    """Build executable setup candidates from confirmed BOS events."""
    structural = build_structural_sequence(frame)
    swings, events = process_structural_candles(structural)

    candidates: list[SetupCandidate] = []
    for event in events:
        candidate = _candidate_from_bos(frame, event, swings)
        if candidate is not None:
            candidates.append(candidate)

    return candidates
