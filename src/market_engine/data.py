"""Load and validate MT5-exported candle data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


class DataValidationError(ValueError):
    """Raised when market data violates the data contract."""


@dataclass(frozen=True)
class DatasetMetadata:
    instrument: str
    timeframe: str
    source: str = "MT5 export"
    timezone: str | None = None
    volume_type: str = "tick_volume"


_MT5_COLUMNS = {
    "<DATE>": "date",
    "<TIME>": "time",
    "<OPEN>": "open",
    "<HIGH>": "high",
    "<LOW>": "low",
    "<CLOSE>": "close",
    "<TICKVOL>": "tick_volume",
    "<VOL>": "real_volume",
    "<SPREAD>": "spread",
}

_REQUIRED = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "tick_volume",
    "real_volume",
    "spread",
]


def load_mt5_csv(path: str | Path) -> pd.DataFrame:
    """Load an MT5 tab-separated candle export into a normalized DataFrame."""
    path = Path(path)
    frame = pd.read_csv(path, sep="\t")

    missing = [column for column in _MT5_COLUMNS if column not in frame.columns]
    if missing:
        raise DataValidationError(
            f"MT5 CSV is missing required columns: {', '.join(missing)}"
        )

    frame = frame.rename(columns=_MT5_COLUMNS)
    frame["timestamp"] = pd.to_datetime(
        frame["date"].astype(str) + " " + frame["time"].astype(str),
        format="%Y.%m.%d %H:%M:%S",
        errors="coerce",
    )
    frame = frame.drop(columns=["date", "time"])
    frame = frame[["timestamp", "open", "high", "low", "close",
                   "tick_volume", "real_volume", "spread"]]

    validate_ohlcv(frame)
    return frame


def validate_ohlcv(frame: pd.DataFrame) -> None:
    """Validate normalized candle data without modifying it."""
    missing = [column for column in _REQUIRED if column not in frame.columns]
    if missing:
        raise DataValidationError(
            f"Normalized data is missing required columns: {', '.join(missing)}"
        )

    if frame.isna().any().any():
        raise DataValidationError("Missing value found in normalized market data.")

    if frame["timestamp"].duplicated().any():
        raise DataValidationError("Duplicate timestamp found.")

    if not frame["timestamp"].is_monotonic_increasing:
        raise DataValidationError("Timestamps are not in ascending order.")

    numeric = frame[["open", "high", "low", "close",
                     "tick_volume", "real_volume", "spread"]]
    if not numeric.apply(lambda col: pd.api.types.is_numeric_dtype(col)).all():
        raise DataValidationError("Non-numeric market data found.")

    if not numeric.apply(lambda col: pd.Series(col).map(pd.api.types.is_number).all()).all():
        raise DataValidationError("Invalid numeric market data found.")

    if not numeric.apply(lambda col: pd.Series(col).map(pd.api.types.is_number).all()).all():
        raise DataValidationError("Invalid numeric market data found.")

    prices = frame[["open", "high", "low", "close"]]
    if not prices.apply(lambda col: pd.Series(col).map(pd.api.types.is_number).all()).all():
        raise DataValidationError("OHLC contains non-numeric values.")

    if not prices.apply(lambda col: pd.Series(col).map(pd.notna).all()).all():
        raise DataValidationError("OHLC contains missing values.")

    if not prices.apply(lambda col: pd.Series(col).map(pd.api.types.is_finite).all()).all():
        raise DataValidationError("OHLC contains non-finite values.")

    if (prices <= 0).any().any():
        raise DataValidationError("Prices must be greater than zero.")

    if (frame["tick_volume"] < 0).any() or (frame["real_volume"] < 0).any():
        raise DataValidationError("Volume values must not be negative.")

    if (frame["spread"] < 0).any():
        raise DataValidationError("Spread values must not be negative.")

    if (frame["high"] < frame[["open", "close"]].max(axis=1)).any():
        raise DataValidationError("Found candle with high below open or close.")

    if (frame["low"] > frame[["open", "close"]].min(axis=1)).any():
        raise DataValidationError("Found candle with low above open or close.")

    if (frame["high"] < frame["low"]).any():
        raise DataValidationError("Found candle with high below low.")
