"""Deterministic execution-price adjustments."""

from __future__ import annotations

from dataclasses import dataclass

from market_engine.structure import Direction


@dataclass(frozen=True)
class Execution:
    quoted_price: float
    spread_price: float
    entry_price: float
    invalidation_price: float
    risk: float


def execute_entry(
    direction: Direction,
    quoted_price: float,
    spread_price: float,
    invalidation_price: float,
) -> Execution:
    """Apply spread, expressed in price units, to a quoted entry price."""
    if spread_price < 0:
        raise ValueError("Spread must not be negative.")

    if direction is Direction.UP:
        entry_price = quoted_price + spread_price
        risk = entry_price - invalidation_price
    elif direction is Direction.DOWN:
        entry_price = quoted_price - spread_price
        risk = invalidation_price - entry_price
    else:
        raise ValueError(f"Unsupported direction: {direction!r}")

    if risk <= 0:
        raise ValueError("Execution risk must be positive.")

    return Execution(
        quoted_price=float(quoted_price),
        spread_price=float(spread_price),
        entry_price=float(entry_price),
        invalidation_price=float(invalidation_price),
        risk=float(risk),
    )
