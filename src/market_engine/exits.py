"""Deterministic exit-area construction from confirmed structure."""

from __future__ import annotations

from dataclasses import dataclass

from market_engine.entry import SetupCandidate
from market_engine.structure import (
    Direction,
    StructureEvent,
    SwingType,
    ValidSwing,
)


@dataclass(frozen=True)
class ExitArea:
    source_type: str
    source_index: int
    source_confirmation_index: int
    price: float
    distance: float


def build_exit_areas(
    candidate: SetupCandidate,
    swings: list[ValidSwing],
    events: list[StructureEvent] | None = None,
) -> list[ExitArea]:
    """Return confirmed, unbroken structural levels beyond the entry price."""
    candidates: list[ExitArea] = []

    for swing in swings:
        if swing.confirmation_index >= candidate.setup_index:
            continue

        if events and any(
            event.event.endswith("_BOS")
            and event.swing_index == swing.index
            and event.index <= candidate.setup_index
            for event in events
        ):
            continue

        if candidate.direction is Direction.UP:
            if swing.swing_type is not SwingType.HIGH:
                continue
            if swing.price <= candidate.entry_price:
                continue
        else:
            if swing.swing_type is not SwingType.LOW:
                continue
            if swing.price >= candidate.entry_price:
                continue

        candidates.append(
            ExitArea(
                source_type="VALID_SWING",
                source_index=swing.index,
                source_confirmation_index=swing.confirmation_index,
                price=float(swing.price),
                distance=abs(float(swing.price) - candidate.entry_price),
            )
        )

    candidates.sort(key=lambda area: area.distance)
    return candidates[:2]
