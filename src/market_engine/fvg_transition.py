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
