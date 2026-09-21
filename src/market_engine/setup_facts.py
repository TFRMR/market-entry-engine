"""Deterministic setup-time facts derived from the BOS event and prior swings.

Every fact uses only the BOS event on the setup candle and swings whose
confirmation happened before the setup candle. Nothing here looks forward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from market_engine.entry import SetupCandidate
from market_engine.structure import (
    Direction,
    StructureEvent,
    StructureScope,
    SwingType,
    ValidSwing,
)

SETUP_FACT_COLUMNS: Final[tuple[str, ...]] = (
    "setup_bos_external",
    "setup_broken_swing_age",
    "setup_entry_beyond_broken_swing_r",
    "setup_invalidation_swing_age",
)

# event name -> (direction, type of the broken swing, type of the invalidation swing)
_BOS_EVENTS: Final[dict[str, tuple[Direction, SwingType, SwingType]]] = {
    "BULLISH_BOS": (Direction.UP, SwingType.HIGH, SwingType.LOW),
    "BEARISH_BOS": (Direction.DOWN, SwingType.LOW, SwingType.HIGH),
}


def _missing() -> dict[str, float]:
    return {column: math.nan for column in SETUP_FACT_COLUMNS}


@dataclass(frozen=True)
class SetupFactContext:
    """Lookup tables so facts can be computed for many candidates cheaply."""

    swings: dict[tuple[int, SwingType], ValidSwing]
    bos_events: dict[tuple[int, Direction], StructureEvent]

    @classmethod
    def build(
        cls,
        swings: list[ValidSwing],
        events: list[StructureEvent],
    ) -> SetupFactContext:
        swing_lookup = {(swing.index, swing.swing_type): swing for swing in swings}
        bos_lookup = {
            (event.index, event.direction): event
            for event in events
            if event.event in _BOS_EVENTS and event.direction is not None
        }
        return cls(swings=swing_lookup, bos_events=bos_lookup)

    def facts(self, candidate: SetupCandidate) -> dict[str, float]:
        """Return setup facts, or NaN for every fact if the BOS cannot be found."""
        bos = self.bos_events.get((candidate.setup_index, candidate.direction))
        if bos is None or bos.swing_index is None:
            return _missing()

        _, broken_type, invalidation_type = _BOS_EVENTS[bos.event]
        broken = self.swings.get((bos.swing_index, broken_type))
        invalidation = self.swings.get(
            (candidate.invalidation_swing_index, invalidation_type)
        )
        if broken is None or invalidation is None or candidate.risk <= 0:
            return _missing()

        sign = 1.0 if candidate.direction is Direction.UP else -1.0
        risk = float(candidate.risk)
        return {
            "setup_bos_external": (
                1.0 if bos.scope is StructureScope.EXTERNAL else 0.0
            ),
            "setup_broken_swing_age": float(
                candidate.setup_index - broken.confirmation_index
            ),
            "setup_entry_beyond_broken_swing_r": (
                sign * (float(candidate.entry_price) - float(broken.price)) / risk
            ),
            "setup_invalidation_swing_age": float(
                candidate.setup_index - invalidation.confirmation_index
            ),
        }


def build_setup_facts(
    candidate: SetupCandidate,
    swings: list[ValidSwing],
    events: list[StructureEvent],
) -> dict[str, float]:
    """Convenience wrapper for a single candidate."""
    return SetupFactContext.build(swings, events).facts(candidate)
