from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

import market_engine.labels as labels_module
from market_engine.entry import SetupCandidate
from market_engine.labels import PRE_SETUP_FEATURE_COLUMNS, build_setup_label_dataset
from market_engine.structure import Direction


def test_pre_setup_features_use_previous_candle(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="h"),
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
        }
    )
    feature_frame = pd.DataFrame(
        {
            "structure_direction": ["DOWN", "UP", "UP", "UP"],
            "structure_distance_to_high": [1.0, 2.0, 3.0, 4.0],
            "structure_distance_to_low": [4.0, 5.0, 6.0, 7.0],
            "structure_bars_since_last_swing": [7.0, 8.0, 9.0, 10.0],
            "structure_bars_since_last_bos": [10.0, 11.0, 12.0, 13.0],
        }
    )
    candidate = SetupCandidate(
        setup_index=2,
        setup_timestamp=frame.iloc[2]["timestamp"],
        direction=Direction.UP,
        entry_index=3,
        entry_timestamp=frame.iloc[3]["timestamp"],
        entry_price=103.0,
        invalidation_price=101.0,
        risk=2.0,
        invalidation_swing_index=1,
    )
    monkeypatch.setattr(
        labels_module,
        "build_exit_areas",
        lambda *args, **kwargs: [SimpleNamespace(price=106.0)],
    )
    monkeypatch.setattr(
        labels_module,
        "evaluate_trade",
        lambda *args, **kwargs: SimpleNamespace(status="TARGET", both_hit=False),
    )

    labeled = build_setup_label_dataset(
        candidates=[candidate],
        swings=[],
        frame=frame,
        feature_frame=feature_frame,
        spread_price=0.0,
        feature_columns=(),
        pre_feature_columns=PRE_SETUP_FEATURE_COLUMNS,
        horizon=1,
        events=[],
    )

    row = labeled.iloc[0]
    assert row["pre_structure_direction"] == "UP"
    assert row["pre_structure_distance_to_high"] == 2.0
    assert row["pre_structure_distance_to_low"] == 5.0
    assert row["pre_structure_bars_since_last_swing"] == 8.0
    assert row["pre_structure_bars_since_last_bos"] == 11.0


def test_pre_setup_columns_are_structure_only() -> None:
    assert all(column.startswith("structure_") for column in PRE_SETUP_FEATURE_COLUMNS)
