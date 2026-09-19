"""Outcome calculations for post-momentum analysis."""

from __future__ import annotations

import pandas as pd


def add_forward_returns(
    frame: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 3, 5, 10),
) -> pd.DataFrame:
    """Add future close-to-close returns for outcome analysis."""
    if not horizons:
        raise ValueError("horizons must not be empty.")

    if any(horizon <= 0 for horizon in horizons):
        raise ValueError("All horizons must be greater than zero.")

    result = frame.copy()

    for horizon in horizons:
        result[f"forward_return_{horizon}"] = (
            result["close"].shift(-horizon) / result["close"]
        ) - 1

    return result
