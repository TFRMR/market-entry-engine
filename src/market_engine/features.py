"""Feature engineering for the momentum-candle strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_engine.structure import (
    StructureScope,
    SwingType,
    build_structural_sequence,
    process_structural_candles,
    process_structural_candles_with_context,
)


def add_momentum_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add basic momentum-candle structure features."""
    result = frame.copy()

    candle_range = result["high"] - result["low"]
    candle_body = (result["close"] - result["open"]).abs()

    result["candle_range"] = candle_range
    result["candle_body"] = candle_body

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

    result["price_vs_ema_5"] = result["close"] - result["ema_5"]
    result["price_vs_ema_20"] = result["close"] - result["ema_20"]
    result["price_vs_ema_50"] = result["close"] - result["ema_50"]

    result["ema_5_vs_20"] = result["ema_5"] - result["ema_20"]
    result["ema_20_vs_50"] = result["ema_20"] - result["ema_50"]

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
        [1, -1],
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


def add_recent_movement_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add recent price-return context features."""
    result = frame.copy()

    result["return_3"] = result["close"].pct_change(periods=3)
    result["return_6"] = result["close"].pct_change(periods=6)
    result["return_12"] = result["close"].pct_change(periods=12)

    return result


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the current initial feature set."""
    result = add_momentum_features(frame)
    result = add_volatility_features(result)
    result = add_ema_features(result)
    result = add_support_resistance_features(result)
    result = add_recent_movement_features(result)
    result = add_micro_structure_features(result)
    result = add_volume_features(result)
    result = add_structure_event_features(result)
    result = add_active_structure_features(result)

    return result


def add_micro_structure_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add short-term price structure and range expansion features."""
    result = frame.copy()

    for window in (1, 2, 3):
        previous_high = result["high"].shift(1).rolling(
            window=window,
            min_periods=window,
        ).max()

        previous_low = result["low"].shift(1).rolling(
            window=window,
            min_periods=window,
        ).min()

        result[f"high_vs_previous_high_{window}"] = (
            result["high"] / previous_high - 1
        )

        result[f"low_vs_previous_low_{window}"] = (
            result["low"] / previous_low - 1
        )

    candle_range = result["high"] - result["low"]

    for window in (3, 5):
        previous_average_range = candle_range.shift(1).rolling(
            window=window,
            min_periods=window,
        ).mean()

        result[f"range_vs_avg_{window}"] = (
            candle_range / previous_average_range
        )

    return result


def add_structure_event_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic market-structure events aligned to confirmation time."""
    result = frame.copy()
    structural = build_structural_sequence(result)
    swings, events = process_structural_candles(structural)

    size = len(result)
    event_columns = {
        "structure_bullish_bos": "BULLISH_BOS",
        "structure_bearish_bos": "BEARISH_BOS",
        "structure_swing_high_valid": "SWING_HIGH_VALID",
        "structure_swing_low_valid": "SWING_LOW_VALID",
    }
    for column in event_columns:
        result[column] = 0

    result["structure_internal_bos"] = 0
    result["structure_external_bos"] = 0
    result["structure_internal_swing"] = 0
    result["structure_external_swing"] = 0

    last_high = np.full(size, np.nan)
    last_low = np.full(size, np.nan)

    for event in events:
        position = event.index
        if position < 0 or position >= size:
            raise ValueError("Structure event index is outside the feature frame.")

        for column, event_name in event_columns.items():
            if event.event == event_name:
                result.iloc[position, result.columns.get_loc(column)] = 1

        if event.event.endswith("_BOS"):
            column = (
                "structure_internal_bos"
                if event.scope is StructureScope.INTERNAL
                else "structure_external_bos"
            )
            result.iloc[position, result.columns.get_loc(column)] = 1

        if event.event.startswith("SWING_"):
            column = (
                "structure_internal_swing"
                if event.scope is StructureScope.INTERNAL
                else "structure_external_swing"
            )
            result.iloc[position, result.columns.get_loc(column)] = 1

    for swing in swings:
        position = swing.confirmation_index
        if position < 0 or position >= size:
            raise ValueError("Swing confirmation index is outside the feature frame.")

        if swing.swing_type is SwingType.HIGH:
            last_high[position] = swing.price
        else:
            last_low[position] = swing.price

    result["structure_last_valid_high"] = pd.Series(
        last_high,
        index=result.index,
    ).ffill()

    result["structure_last_valid_low"] = pd.Series(
        last_low,
        index=result.index,
    ).ffill()

    for label in ("HH", "HL", "LH", "LL"):
        values = np.zeros(size, dtype=int)
        for swing in swings:
            if swing.label == label:
                values[swing.confirmation_index] = 1
        result[f"structure_{label.lower()}"] = values

    return result


def add_active_structure_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add active structural direction, distances, and event ages."""
    result = frame.copy()
    structural = build_structural_sequence(result)
    _, _, snapshots = process_structural_candles_with_context(structural)

    size = len(result)
    direction = np.full(size, np.nan)
    last_high = np.full(size, np.nan)
    last_low = np.full(size, np.nan)
    last_swing_index = np.full(size, np.nan)
    last_bos_index = np.full(size, np.nan)
    snapshot_mask = np.zeros(size, dtype=bool)

    direction_code = {
        None: np.nan,
        "UP": 1.0,
        "DOWN": -1.0,
    }

    for snapshot in snapshots:
        position = snapshot.index
        if position < 0 or position >= size:
            raise ValueError("Structure snapshot index is outside the feature frame.")

        snapshot_mask[position] = True
        direction[position] = direction_code[
            snapshot.direction.value if snapshot.direction else None
        ]
        if snapshot.last_high is not None:
            last_high[position] = snapshot.last_high.price
        if snapshot.last_low is not None:
            last_low[position] = snapshot.last_low.price
        if snapshot.last_swing_confirmation_index is not None:
            last_swing_index[position] = snapshot.last_swing_confirmation_index
        if snapshot.last_bos_index is not None:
            last_bos_index[position] = snapshot.last_bos_index

    last_snapshot = np.maximum.accumulate(
        np.where(snapshot_mask, np.arange(size), -1)
    )

    def carry(values: np.ndarray) -> np.ndarray:
        valid = last_snapshot >= 0
        output = np.full(size, np.nan)
        positions = np.clip(last_snapshot, 0, None)
        output[valid] = values[positions[valid]]
        return output

    direction = carry(direction)
    last_high = carry(last_high)
    last_low = carry(last_low)
    last_swing_index = carry(last_swing_index)
    last_bos_index = carry(last_bos_index)

    result["structure_direction"] = pd.Series(
        direction,
        index=result.index,
    ).fillna(0)

    result["structure_distance_to_high"] = (
        pd.Series(last_high, index=result.index) - result["close"]
    )
    result["structure_distance_to_low"] = (
        result["close"] - pd.Series(last_low, index=result.index)
    )

    current_index = pd.Series(
        np.arange(size),
        index=result.index,
        dtype=float,
    )

    result["structure_bars_since_last_swing"] = (
        current_index - pd.Series(last_swing_index, index=result.index)
    )
    result["structure_bars_since_last_bos"] = (
        current_index - pd.Series(last_bos_index, index=result.index)
    )

    return result

def add_volume_features(
    frame: pd.DataFrame,
    volume_window: int = 20,
) -> pd.DataFrame:
    """Add tick-volume context features."""
    if volume_window <= 0:
        raise ValueError("volume_window must be greater than zero.")

    result = frame.copy()

    previous_volume = result["tick_volume"].shift(1)

    average_volume = previous_volume.rolling(
        window=volume_window,
        min_periods=volume_window,
    ).mean()

    result["volume_ratio_20"] = np.nan

    valid_average = average_volume.gt(0)

    result.loc[valid_average, "volume_ratio_20"] = (
        result.loc[valid_average, "tick_volume"]
        / average_volume[valid_average]
    )

    result["volume_change_1"] = result["tick_volume"].pct_change(periods=1)

    return result
