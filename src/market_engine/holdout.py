"""Chronological historical-boundary helpers.

The historical boundary is useful for reproducible partitioning, but it is not
an untouched out-of-sample holdout: the current dataset was already inspected
end-to-end before this boundary was introduced.
"""

from __future__ import annotations

from typing import Final

import pandas as pd

HISTORICAL_AUDIT_CUTOFF: Final[pd.Timestamp] = pd.Timestamp("2026-03-18 00:00:00")


def historical_boundary_index(
    frame: pd.DataFrame,
    cutoff: pd.Timestamp = HISTORICAL_AUDIT_CUTOFF,
) -> int:
    """Return the first candle index at or after the historical boundary."""
    timestamps = pd.DatetimeIndex(frame["timestamp"])
    return int(timestamps.searchsorted(pd.Timestamp(cutoff), side="left"))


def split_historical_boundary(
    labeled: pd.DataFrame,
    frame: pd.DataFrame,
    horizon: int,
    cutoff: pd.Timestamp = HISTORICAL_AUDIT_CUTOFF,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split development / historical-audit / purged rows chronologically.

    Development rows must finish their forward label horizon before the
    boundary. Historical-audit rows start at the boundary. Rows whose label
    horizon crosses the boundary are purged. The historical-audit partition is
    explicitly *not* a pristine OOS set.
    """
    if horizon <= 0:
        raise ValueError("horizon must be greater than zero.")

    first = historical_boundary_index(frame, cutoff)
    setup = labeled["setup_index"].astype(int)
    development_mask = setup + horizon < first
    audit_mask = setup >= first
    purged_mask = ~(development_mask | audit_mask)
    return (
        labeled.loc[development_mask].copy(),
        labeled.loc[audit_mask].copy(),
        labeled.loc[purged_mask].copy(),
    )
