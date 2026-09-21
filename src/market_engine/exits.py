"""Deterministic exit-area construction from confirmed structure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from market_engine.entry import SetupCandidate
from market_engine.structure import (
    Direction,
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


@dataclass(frozen=True)
class ExitAreaIndex:
    """Precomputed range extrema for fast structural-level breach checks."""

    log_table: np.ndarray
    high_table: tuple[np.ndarray, ...]
    low_table: tuple[np.ndarray, ...]

    @classmethod
    def build(cls, frame: pd.DataFrame) -> "ExitAreaIndex":
        """Build O(n log n) range-max/min tables for the frame."""
        highs = frame["high"].to_numpy(dtype=float)
        lows = frame["low"].to_numpy(dtype=float)
        n = len(frame)
        if n == 0:
            return cls(
                log_table=np.zeros(1, dtype=np.int8),
                high_table=(),
                low_table=(),
            )

        log_table = np.zeros(n + 1, dtype=np.int8)
        if n >= 2:
            log_table[2:] = np.arange(1, n, dtype=np.int8).astype(float).bit_length if False else 0
            for length in range(2, n + 1):
                log_table[length] = log_table[length // 2] + 1

        high_levels = [highs]
        low_levels = [lows]
        level = 1
        while (1 << level) <= n:
            span = 1 << (level - 1)
            high_levels.append(
                np.maximum(high_levels[-1][:-span], high_levels[-1][span:])
            )
            low_levels.append(
                np.minimum(low_levels[-1][:-span], low_levels[-1][span:])
            )
            level += 1

        return cls(
            log_table=log_table,
            high_table=tuple(high_levels),
            low_table=tuple(low_levels),
        )

    def range_max(self, start: int, stop: int) -> float:
        """Return max(frame.high[start:stop]) for a non-empty half-open range."""
        if start >= stop:
            return -np.inf
        length = stop - start
        level = int(self.log_table[length])
        span = 1 << level
        return float(max(self.high_table[level][start], self.high_table[level][stop - span]))

    def range_min(self, start: int, stop: int) -> float:
        """Return min(frame.low[start:stop]) for a non-empty half-open range."""
        if start >= stop:
            return np.inf
        length = stop - start
        level = int(self.log_table[length])
        span = 1 << level
        return float(min(self.low_table[level][start], self.low_table[level][stop - span]))


def build_exit_areas(
    candidate: SetupCandidate,
    swings: list[ValidSwing],
    frame: pd.DataFrame,
    exit_index: ExitAreaIndex | None = None,
) -> list[ExitArea]:
    """Return confirmed, unbroken structural levels beyond the entry price."""
    index = exit_index or ExitAreaIndex.build(frame)
    candidates: list[ExitArea] = []

    for swing in swings:
        if swing.confirmation_index >= candidate.setup_index:
            continue

        if swing.swing_type is SwingType.HIGH:
            if index.range_max(
                swing.confirmation_index + 1,
                candidate.setup_index,
            ) > swing.price:
                continue
        else:
            if index.range_min(
                swing.confirmation_index + 1,
                candidate.setup_index,
            ) < swing.price:
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
