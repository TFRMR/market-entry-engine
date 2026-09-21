"""Canonical barrier-label definitions for model training."""

from __future__ import annotations

from typing import Final

TP_FIRST: Final[str] = "TP_FIRST"
SL_FIRST: Final[str] = "SL_FIRST"
BOTH_SAME_CANDLE: Final[str] = "BOTH_SAME_CANDLE"
UNRESOLVED: Final[str] = "UNRESOLVED"

MODEL_LABELS: Final[frozenset[str]] = frozenset({TP_FIRST, SL_FIRST})
ALL_BARRIER_LABELS: Final[frozenset[str]] = frozenset(
    {TP_FIRST, SL_FIRST, BOTH_SAME_CANDLE, UNRESOLVED}
)


def is_model_label(label: object) -> bool:
    """Return whether a barrier outcome is eligible as a binary model label."""
    return label in MODEL_LABELS
