from __future__ import annotations

import pandas as pd

from market_engine.holdout import HISTORICAL_AUDIT_CUTOFF, split_historical_boundary


def test_historical_boundary_purges_crossing_horizons() -> None:
    frame = pd.DataFrame({"time": pd.date_range("2026-03-17", periods=5, freq="D")})
    labeled = pd.DataFrame({"setup_index": [0, 1, 2, 3]})

    development, audit, purged = split_historical_boundary(labeled, frame, horizon=2)

    assert development["setup_index"].tolist() == []
    assert audit["setup_index"].tolist() == [2, 3]
    assert purged["setup_index"].tolist() == [0, 1]


def test_historical_cutoff_is_explicit() -> None:
    assert HISTORICAL_AUDIT_CUTOFF == pd.Timestamp("2026-03-18 00:00:00")
