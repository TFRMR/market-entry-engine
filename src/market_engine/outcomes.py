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


def add_barrier_outcomes(
    frame: pd.DataFrame,
    horizons: tuple[int, ...] = (10,),
    targets_r: tuple[float, ...] = (1.0, 1.5, 1.618, 2.0),
) -> pd.DataFrame:
    """Add TP/SL barrier outcomes for momentum-candle setups."""
    if not horizons:
        raise ValueError("horizons must not be empty.")

    if any(horizon <= 0 for horizon in horizons):
        raise ValueError("All horizons must be greater than zero.")

    if not targets_r:
        raise ValueError("targets_r must not be empty.")

    if any(target <= 0 for target in targets_r):
        raise ValueError("All targets_r must be greater than zero.")

    required = {
        "open",
        "high",
        "low",
        "close",
        "is_momentum_candle",
    }
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

    result["barrier_entry"] = entry.where(valid_setup)
    result["barrier_stop"] = stop.where(valid_setup)
    result["barrier_risk"] = risk.where(valid_setup)

    for target_r in targets_r:
        target = pd.Series(np.nan, index=result.index, dtype=float)

        target.loc[bullish] = (
            entry.loc[bullish] + target_r * risk.loc[bullish]
        )
        target.loc[bearish] = (
            entry.loc[bearish] - target_r * risk.loc[bearish]
        )

        result[f"target_{target_r:g}r"] = target.where(valid_setup)

        for horizon in horizons:
            outcomes = pd.Series(
                pd.NA,
                index=result.index,
                dtype="string",
            )

            for index in result.index[valid_setup]:
                position = result.index.get_loc(index)
                available = len(result.index) - position - 1

                if available < horizon:
                    continue

                for offset in range(1, horizon + 1):
                    future_index = result.index[position + offset]

                    high = result.at[future_index, "high"]
                    low = result.at[future_index, "low"]

                    if bullish.loc[index]:
                        tp_hit = high >= target.loc[index]
                        sl_hit = low <= stop.loc[index]
                    else:
                        tp_hit = low <= target.loc[index]
                        sl_hit = high >= stop.loc[index]

                    if tp_hit and sl_hit:
                        outcomes.loc[index] = "BOTH_SAME_CANDLE"
                        break

                    if tp_hit:
                        outcomes.loc[index] = "TP_FIRST"
                        break

                    if sl_hit:
                        outcomes.loc[index] = "SL_FIRST"
                        break

                if pd.isna(outcomes.loc[index]):
                    outcomes.loc[index] = "UNRESOLVED"

            result[f"barrier_{target_r:g}r_{horizon}"] = outcomes

    return result
