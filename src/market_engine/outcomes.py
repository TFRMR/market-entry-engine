"""Outcome calculations for post-momentum analysis."""

from __future__ import annotations

import numpy as np
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


def add_forward_excursions(
    frame: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 3, 5, 10),
) -> pd.DataFrame:
    """Add directional MFE/MAE in R using momentum-candle entry and stop."""
    if not horizons:
        raise ValueError("horizons must not be empty.")

    if any(horizon <= 0 for horizon in horizons):
        raise ValueError("All horizons must be greater than zero.")

    required = {"open", "high", "low", "close", "is_momentum_candle"}
    missing = sorted(required - set(frame.columns))

    if missing:
        raise ValueError(
            f"Frame is missing required columns: {', '.join(missing)}"
        )

    result = frame.copy()

    entry = result["close"]

    bullish = (
        result["is_momentum_candle"]
        & result["close"].gt(result["open"])
    )

    bearish = (
        result["is_momentum_candle"]
        & result["close"].lt(result["open"])
    )

    stop = pd.Series(np.nan, index=result.index, dtype=float)

    stop.loc[bullish] = result.loc[bullish, "low"]
    stop.loc[bearish] = result.loc[bearish, "high"]

    risk = (entry - stop).abs()
    valid_setup = (bullish | bearish) & risk.gt(0)

    result["entry_price"] = entry.where(valid_setup)
    result["stop_price"] = stop.where(valid_setup)
    result["risk_price"] = risk.where(valid_setup)

    future_highs = result["high"].shift(-1)
    future_lows = result["low"].shift(-1)

    for horizon in horizons:
        # rolling() normally looks backward. Reverse the series so
        # the rolling window represents candles +1 ... +horizon.
        future_high = (
            future_highs.iloc[::-1]
            .rolling(window=horizon, min_periods=horizon)
            .max()
            .iloc[::-1]
        )

        future_low = (
            future_lows.iloc[::-1]
            .rolling(window=horizon, min_periods=horizon)
            .min()
            .iloc[::-1]
        )

        favorable = pd.Series(
            np.nan,
            index=result.index,
            dtype=float,
        )

        adverse = pd.Series(
            np.nan,
            index=result.index,
            dtype=float,
        )

        # Bullish:
        # MFE = highest future price above entry
        # MAE = lowest future price below entry
        favorable.loc[bullish] = (
            future_high.loc[bullish] - entry.loc[bullish]
        ) / risk.loc[bullish]

        adverse.loc[bullish] = (
            entry.loc[bullish] - future_low.loc[bullish]
        ) / risk.loc[bullish]

        # Bearish:
        # MFE = lowest future price below entry
        # MAE = highest future price above entry
        favorable.loc[bearish] = (
            entry.loc[bearish] - future_low.loc[bearish]
        ) / risk.loc[bearish]

        adverse.loc[bearish] = (
            future_high.loc[bearish] - entry.loc[bearish]
        ) / risk.loc[bearish]

        result[f"mfe_r_{horizon}"] = favorable.where(valid_setup)
        result[f"mae_r_{horizon}"] = adverse.where(valid_setup)

    return result
