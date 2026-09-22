"""Historical FVG references for transition analysis."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FVGReference:
    """Immutable historical fair-value-gap reference."""

    creation_index: int
    direction: str
    lower: float
    upper: float
    size: float
    creation_timestamp: object | None = None


def build_fvg_references(frame: pd.DataFrame) -> list[FVGReference]:
    """Build every historical FVG reference available in the frame.

    An FVG is created on candle t from candles t-2 and t. The reference is
    therefore available starting at candle t and never uses future candles.
    """
    required = {"high", "low"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "Missing FVG reference columns: " + ", ".join(sorted(missing))
        )

    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)

    references: list[FVGReference] = []

    for position in range(2, len(frame)):
        bullish_gap = low[position] > high[position - 2]
        bearish_gap = high[position] < low[position - 2]

        timestamp = (
            frame["timestamp"].iloc[position]
            if "timestamp" in frame.columns
            else None
        )

        if bullish_gap:
            lower = float(high[position - 2])
            upper = float(low[position])
            references.append(
                FVGReference(
                    creation_index=position,
                    direction="BULLISH",
                    lower=lower,
                    upper=upper,
                    size=upper - lower,
                    creation_timestamp=timestamp,
                )
            )
        elif bearish_gap:
            lower = float(high[position])
            upper = float(low[position - 2])
            references.append(
                FVGReference(
                    creation_index=position,
                    direction="BEARISH",
                    lower=lower,
                    upper=upper,
                    size=upper - lower,
                    creation_timestamp=timestamp,
                )
            )

    return references


@dataclass(frozen=True)
class FVGTransitionReferences:
    """Spatially routed FVG references available at one candle."""

    origin: FVGReference | None
    target: FVGReference | None
    next_after_target: FVGReference | None


def route_fvg_references(
    references: list[FVGReference],
    *,
    current_index: int,
    close: float,
    direction: str,
) -> FVGTransitionReferences:
    """Route the current price through spatially ordered FVG references.

    Only references created on or before ``current_index`` are available.

    Origin:
    - exactly one available FVG contains ``close``;
    - zero containing FVGs => no origin;
    - multiple overlapping containing FVGs => ambiguous, therefore no origin.

    Target:
    - UP: nearest FVG whose entire zone is strictly above ``close``;
    - DOWN: nearest FVG whose entire zone is strictly below ``close``.

    Next-after-target:
    - UP: nearest FVG strictly above the target zone;
    - DOWN: nearest FVG strictly below the target zone.

    Overlapping references are preserved but are not ordered against each
    other. This prevents creation order from being used as a hidden spatial
    assumption.
    """
    if direction not in {"UP", "DOWN"}:
        raise ValueError("direction must be 'UP' or 'DOWN'")

    available = [
        reference
        for reference in references
        if reference.creation_index <= current_index
    ]

    containing = [
        reference
        for reference in available
        if reference.lower <= close <= reference.upper
    ]

    origin = containing[0] if len(containing) == 1 else None

    if direction == "UP":
        target_candidates = [
            reference
            for reference in available
            if reference.lower > close
        ]
        target = (
            min(target_candidates, key=lambda reference: reference.lower)
            if target_candidates
            else None
        )

        next_candidates = (
            [
                reference
                for reference in available
                if target is not None and reference.lower > target.upper
            ]
            if target is not None
            else []
        )
        next_after_target = (
            min(next_candidates, key=lambda reference: reference.lower)
            if next_candidates
            else None
        )
    else:
        target_candidates = [
            reference
            for reference in available
            if reference.upper < close
        ]
        target = (
            max(target_candidates, key=lambda reference: reference.upper)
            if target_candidates
            else None
        )

        next_candidates = (
            [
                reference
                for reference in available
                if target is not None and reference.upper < target.lower
            ]
            if target is not None
            else []
        )
        next_after_target = (
            max(next_candidates, key=lambda reference: reference.upper)
            if next_candidates
            else None
        )

    return FVGTransitionReferences(
        origin=origin,
        target=target,
        next_after_target=next_after_target,
    )
