from pathlib import Path

import pandas as pd
import pytest

from market_engine.data import DataValidationError, load_mt5_csv, validate_ohlcv


def test_load_mt5_csv_normalizes_real_export():
    frame = load_mt5_csv(Path("tests/fixtures/mt5_sample.csv"))

    assert list(frame.columns) == [
        "timestamp", "open", "high", "low", "close",
        "tick_volume", "real_volume", "spread",
    ]
    assert len(frame) == 3
    assert frame["timestamp"].is_monotonic_increasing
    assert frame["tick_volume"].tolist() == [100, 120, 140]
    assert frame["real_volume"].tolist() == [0, 0, 0]
    assert frame["spread"].tolist() == [160, 160, 180]


def test_validate_rejects_duplicate_timestamp():
    frame = pd.DataFrame({
        "timestamp": pd.to_datetime(
            ["2026-01-01 00:00:00", "2026-01-01 00:00:00"]
        ),
        "open": [100, 101], "high": [102, 103],
        "low": [99, 100], "close": [101, 102],
        "tick_volume": [10, 10], "real_volume": [0, 0], "spread": [10, 10],
    })

    with pytest.raises(DataValidationError, match="Duplicate timestamp"):
        validate_ohlcv(frame)


def test_validate_rejects_invalid_ohlc():
    frame = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-01 00:00:00"]),
        "open": [100], "high": [99], "low": [98], "close": [101],
        "tick_volume": [10], "real_volume": [0], "spread": [10],
    })

    with pytest.raises(DataValidationError, match="high below open or close"):
        validate_ohlcv(frame)
