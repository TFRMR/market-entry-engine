"""Feature engineering for the momentum-candle strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_momentum_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add basic momentum-candle structure features."""
    result = frame.copy()

    candle_range = result["high"] - result["low"]
    candle_body = (result["close"] - result["open"]).abs()

    result["candle_range"] = candle_range
    result["candle_body"] = candle_body

    # Avoid division by zero for degenerate candles.
    valid_range = candle_range.ne(0)

    result["body_ratio"] = np.nan
    result.loc[valid_range, "body_ratio"] = (
        candle_body[valid_range] / candle_range[valid_range]
    )

    upper_wick = (
        result["high"]
        - result[["open", "close"]].max(axis=1)
    )

    lower_wick = (
        result[["open", "close"]].min(axis=1)
        - result["low"]
    )

    result["upper_wick_ratio"] = np.nan
    result["lower_wick_ratio"] = np.nan

    result.loc[valid_range, "upper_wick_ratio"] = (
        upper_wick[valid_range] / candle_range[valid_range]
    )

    result.loc[valid_range, "lower_wick_ratio"] = (
        lower_wick[valid_range] / candle_range[valid_range]
    )

    result["close_position"] = np.nan
    result.loc[valid_range, "close_position"] = (
        (result.loc[valid_range, "close"] - result.loc[valid_range, "low"])
        / candle_range[valid_range]
    )

    # Core momentum-candle hypothesis:
    # body must be at least 80% of the total candle range.
    result["is_momentum_candle"] = result["body_ratio"] >= 0.80

    result["is_bullish"] = result["close"] > result["open"]
    result["is_bearish"] = result["close"] < result["open"]

    return result


def add_volatility_features(
    frame: pd.DataFrame,
    atr_period: int = 14,
) -> pd.DataFrame:
    """Add ATR and candle-range-to-ATR features."""
    if atr_period <= 0:
        raise ValueError("atr_period must be greater than zero.")

    result = frame.copy()

    previous_close = result["close"].shift(1)

    true_range = pd.concat(
        [
            result["high"] - result["low"],
            (result["high"] - previous_close).abs(),
            (result["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    result["true_range"] = true_range

    result["atr_14"] = true_range.rolling(
        window=atr_period,
        min_periods=atr_period,
    ).mean()

    result["range_to_atr"] = np.nan

    valid_atr = result["atr_14"].gt(0)

    result.loc[valid_atr, "range_to_atr"] = (
        result.loc[valid_atr, "candle_range"]
        / result.loc[valid_atr, "atr_14"]
    )

    return result


def add_ema_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add EMA 5, 20, and 50 context features."""
    result = frame.copy()

    result["ema_5"] = result["close"].ewm(
        span=5,
        adjust=False,
        min_periods=5,
    ).mean()

    result["ema_20"] = result["close"].ewm(
        span=20,
        adjust=False,
        min_periods=20,
    ).mean()

    result["ema_50"] = result["close"].ewm(
        span=50,
        adjust=False,
        min_periods=50,
    ).mean()

    # Distance between current price and each EMA.
    result["price_vs_ema_5"] = (
        result["close"] - result["ema_5"]
    )

    result["price_vs_ema_20"] = (
        result["close"] - result["ema_20"]
    )

    result["price_vs_ema_50"] = (
        result["close"] - result["ema_50"]
    )

    # EMA relationships.
    result["ema_5_vs_20"] = (
        result["ema_5"] - result["ema_20"]
    )

    result["ema_20_vs_50"] = (
        result["ema_20"] - result["ema_50"]
    )

    # Alignment:
    #  1 = bullish alignment: EMA5 > EMA20 > EMA50
    # -1 = bearish alignment: EMA5 < EMA20 < EMA50
    #  0 = mixed / not aligned
    result["ema_alignment"] = np.select(
        [
            (
                (result["ema_5"] > result["ema_20"])
                & (result["ema_20"] > result["ema_50"])
            ),
            (
                (result["ema_5"] < result["ema_20"])
                & (result["ema_20"] < result["ema_50"])
            ),
        ],
        [
            1,
            -1,
        ],
        default=0,
    )

    return result


def add_support_resistance_features(
    frame: pd.DataFrame,
    short_window: int = 20,
    long_window: int = 50,
) -> pd.DataFrame:
    """Add prior-window support/resistance context without look-ahead."""
    if short_window <= 0:
        raise ValueError("short_window must be greater than zero.")

    if long_window <= 0:
        raise ValueError("long_window must be greater than zero.")

    if short_window >= long_window:
        raise ValueError("short_window must be smaller than long_window.")

    result = frame.copy()

    # Shift by one candle first so the current candle is NEVER included
    # when calculating prior support/resistance levels.
    previous_high = result["high"].shift(1)
    previous_low = result["low"].shift(1)

    result["previous_high_20"] = previous_high.rolling(
        window=short_window,
        min_periods=short_window,
    ).max()

    result["previous_low_20"] = previous_low.rolling(
        window=short_window,
        min_periods=short_window,
    ).min()

    result["previous_high_50"] = previous_high.rolling(
        window=long_window,
        min_periods=long_window,
    ).max()

    result["previous_low_50"] = previous_low.rolling(
        window=long_window,
        min_periods=long_window,
    ).min()

    # Distance from current close to prior structural levels.
    result["distance_to_high_20"] = (
        result["previous_high_20"] - result["close"]
    )

    result["distance_to_low_20"] = (
        result["close"] - result["previous_low_20"]
    )

    result["distance_to_high_50"] = (
        result["previous_high_50"] - result["close"]
    )

    result["distance_to_low_50"] = (
        result["close"] - result["previous_low_50"]
    )

    # Breakout flags.
    # A bullish breakout occurs when the current high exceeds the
    # highest high of the previous window.
    # A bearish breakout occurs when the current low falls below the
    # lowest low of the previous window.
    result["breakout_above_20"] = (
        result["high"] > result["previous_high_20"]
    )

    result["breakout_below_20"] = (
        result["low"] < result["previous_low_20"]
    )

    result["breakout_above_50"] = (
        result["high"] > result["previous_high_50"]
    )

    result["breakout_below_50"] = (
        result["low"] < result["previous_low_50"]
    )

    return result


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the current initial feature set."""
    result = add_momentum_features(frame)
    result = add_volatility_features(result)
    result = add_ema_features(result)
    result = add_support_resistance_features(result)

    return result
