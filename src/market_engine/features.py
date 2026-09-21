"""Feature engineering for the momentum-candle strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_engine.order_block import find_order_block_candidates
from market_engine.structure import (
    Direction,
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


def add_order_block_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add directional historical order-block context features."""
    required = {
        "open",
        "high",
        "low",
        "close",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "Missing order-block feature columns: "
            + ", ".join(sorted(missing))
        )

    result = frame.copy()
    size = len(result)

    feature_names = (
        "present",
        "size",
        "size_atr",
        "age_bars",
        "distance",
        "distance_atr",
        "contains_price",
        "relative_position",
    )

    for direction in ("bullish", "bearish"):
        result[f"ob_{direction}_present"] = 0
        for name in feature_names[1:]:
            result[f"ob_{direction}_{name}"] = np.nan

    structural = build_structural_sequence(result)
    swings, events = process_structural_candles(structural)
    candidates = find_order_block_candidates(swings, events)

    candles = [
        {
            "index": position,
            "open": result.iloc[position]["open"],
            "high": result.iloc[position]["high"],
            "low": result.iloc[position]["low"],
            "close": result.iloc[position]["close"],
        }
        for position in range(size)
    ]

    by_direction = {"bullish": [], "bearish": []}
    for candidate in candidates:
        key = "bullish" if candidate.direction is Direction.UP else "bearish"
        by_direction[key].append(candidate)

    for direction, direction_candidates in by_direction.items():
        # The latest known candidate is the one with the latest event.
        # Swing index breaks ties deterministically.
        direction_candidates.sort(
            key=lambda candidate: (candidate.event_index, candidate.swing_index)
        )

        latest = None
        candidate_position = 0

        for position in range(size):
            while (
                candidate_position < len(direction_candidates)
                and direction_candidates[candidate_position].event_index <= position
            ):
                latest = direction_candidates[candidate_position]
                candidate_position += 1

            if latest is None:
                continue

            candle = candles[latest.swing_index]
            zone_low = float(candle["low"])
            zone_high = float(candle["high"])
            zone_size = zone_high - zone_low

            if zone_size <= 0:
                continue

            close = float(result.iloc[position]["close"])
            atr = (
                float(result.iloc[position]["atr_14"])
                if "atr_14" in result.columns
                else np.nan
            )

            distance = (
                0.0
                if zone_low <= close <= zone_high
                else min(
                    abs(close - zone_low),
                    abs(close - zone_high),
                )
            )

            relative_position = (close - zone_low) / zone_size

            result.iloc[
                position,
                result.columns.get_loc(f"ob_{direction}_present"),
            ] = 1
            result.iloc[
                position,
                result.columns.get_loc(f"ob_{direction}_size"),
            ] = zone_size
            result.iloc[
                position,
                result.columns.get_loc(f"ob_{direction}_size_atr"),
            ] = (
                zone_size / atr
                if np.isfinite(atr) and atr > 0
                else np.nan
            )
            result.iloc[
                position,
                result.columns.get_loc(f"ob_{direction}_age_bars"),
            ] = position - latest.event_index
            result.iloc[
                position,
                result.columns.get_loc(f"ob_{direction}_distance"),
            ] = distance
            result.iloc[
                position,
                result.columns.get_loc(f"ob_{direction}_distance_atr"),
            ] = (
                distance / atr
                if np.isfinite(atr) and atr > 0
                else np.nan
            )
            result.iloc[
                position,
                result.columns.get_loc(f"ob_{direction}_contains_price"),
            ] = int(zone_low <= close <= zone_high)
            result.iloc[
                position,
                result.columns.get_loc(f"ob_{direction}_relative_position"),
            ] = relative_position

    return result


def add_sr_location_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add confirmed structural S/R and deterministic location context."""
    required = {
        "high",
        "low",
        "close",
        "structure_direction",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "Missing S/R and location columns: "
            + ", ".join(sorted(missing))
        )

    result = frame.copy()
    size = len(result)

    structural = build_structural_sequence(result)
    swings, _ = process_structural_candles(structural)

    swings_by_confirmation = {}
    for swing in swings:
        swings_by_confirmation.setdefault(
            swing.confirmation_index,
            [],
        ).append(swing)

    close = result["close"].to_numpy(dtype=float)
    direction = result["structure_direction"].to_numpy(dtype=float)

    support = np.full(size, np.nan)
    resistance = np.full(size, np.nan)
    next_structure = np.full(size, np.nan)

    confirmed_highs = []
    confirmed_lows = []

    for position in range(size):
        for swing in swings_by_confirmation.get(position, []):
            if swing.swing_type is SwingType.HIGH:
                confirmed_highs.append(float(swing.price))
            else:
                confirmed_lows.append(float(swing.price))

        current_close = close[position]

        supports = [
            level
            for level in confirmed_lows
            if level <= current_close
        ]
        resistances = [
            level
            for level in confirmed_highs
            if level >= current_close
        ]

        if supports:
            support[position] = max(supports)

        if resistances:
            resistance[position] = min(resistances)

        if direction[position] > 0:
            next_levels = [
                level
                for level in confirmed_highs
                if level > current_close
            ]
            if next_levels:
                next_structure[position] = min(next_levels)

        elif direction[position] < 0:
            next_levels = [
                level
                for level in confirmed_lows
                if level < current_close
            ]
            if next_levels:
                next_structure[position] = max(next_levels)

    result["support_level"] = support
    result["resistance_level"] = resistance

    result["distance_to_support"] = close - support
    result["distance_to_resistance"] = resistance - close

    result["distance_to_support_atr"] = np.nan
    result["distance_to_resistance_atr"] = np.nan
    result["distance_to_next_structure_level"] = np.nan
    result["distance_to_next_structure_level_atr"] = np.nan

    valid_next = np.isfinite(next_structure)
    next_distance = np.full(size, np.nan)
    next_distance[valid_next] = np.abs(
        next_structure[valid_next] - close[valid_next]
    )
    result["distance_to_next_structure_level"] = next_distance

    if "atr_14" in result.columns:
        atr = result["atr_14"].to_numpy(dtype=float)
        valid_atr = np.isfinite(atr) & (atr > 0)

        support_distance = result["distance_to_support"].to_numpy(dtype=float)
        resistance_distance = result["distance_to_resistance"].to_numpy(dtype=float)

        valid_support = valid_atr & np.isfinite(support_distance)
        valid_resistance = valid_atr & np.isfinite(resistance_distance)
        valid_next_atr = valid_atr & np.isfinite(next_distance)

        result.loc[valid_support, "distance_to_support_atr"] = (
            support_distance[valid_support] / atr[valid_support]
        )
        result.loc[valid_resistance, "distance_to_resistance_atr"] = (
            resistance_distance[valid_resistance] / atr[valid_resistance]
        )
        result.loc[valid_next_atr, "distance_to_next_structure_level_atr"] = (
            next_distance[valid_next_atr] / atr[valid_next_atr]
        )

    result["leg_position"] = result.get(
        "range_position",
        pd.Series(np.nan, index=result.index),
    )

    return result


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the current initial feature set."""
    result = add_momentum_features(frame)
    result = add_volatility_features(result)
    result = add_fvg_features(result)
    result = add_ema_features(result)
    result = add_support_resistance_features(result)
    result = add_recent_movement_features(result)
    result = add_micro_structure_features(result)
    result = add_volume_features(result)
    result = add_structure_event_features(result)
    result = add_active_structure_features(result)
    result = add_trend_range_features(result)
    result = add_liquidity_features(result)
    result = add_order_block_features(result)
    result = add_sr_location_features(result)

    return result


def build_structural_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build structure-only features for the structural blueprint.

    Unlike build_features, this adds no momentum, EMA, rolling-level, or volume
    columns. build_features remains available as the legacy baseline for
    ablation studies.
    """
    result = add_structure_event_features(frame)
    result = add_active_structure_features(result)
    result = add_trend_range_features(result)
    result = add_liquidity_features(result)
    return add_order_block_features(result)


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
        "structure_bullish_choch": "BULLISH_CHOCH",
        "structure_bearish_choch": "BEARISH_CHOCH",
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
    historical_last_high = np.full(size, np.nan)
    historical_last_low = np.full(size, np.nan)
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
        if snapshot.historical_last_high is not None:
            historical_last_high[position] = snapshot.historical_last_high.price
        if snapshot.historical_last_low is not None:
            historical_last_low[position] = snapshot.historical_last_low.price
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
    historical_last_high = carry(historical_last_high)
    historical_last_low = carry(historical_last_low)
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
    result["structure_historical_distance_to_high"] = (
        pd.Series(historical_last_high, index=result.index) - result["close"]
    )
    result["structure_historical_distance_to_low"] = (
        result["close"] - pd.Series(historical_last_low, index=result.index)
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


def add_fvg_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add latest historical three-candle FVG context without look-ahead."""
    required = {"high", "low", "close"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "Missing FVG context columns: " + ", ".join(sorted(missing))
        )

    result = frame.copy()
    size = len(result)

    high = result["high"].to_numpy(dtype=float)
    low = result["low"].to_numpy(dtype=float)
    close = result["close"].to_numpy(dtype=float)

    atr = (
        result["atr_14"].to_numpy(dtype=float)
        if "atr_14" in result.columns
        else np.full(size, np.nan)
    )

    fvg_direction = np.empty(size, dtype=object)
    fvg_direction[:] = None
    fvg_lower = np.full(size, np.nan)
    fvg_upper = np.full(size, np.nan)
    fvg_size = np.full(size, np.nan)
    fvg_creation_index = np.full(size, np.nan)
    fvg_creation_atr = np.full(size, np.nan)
    fvg_creation_timestamp = np.empty(size, dtype=object)
    fvg_creation_timestamp[:] = None

    latest_fvg = None

    for position in range(size):
        if position >= 2:
            bullish_gap = low[position] > high[position - 2]
            bearish_gap = high[position] < low[position - 2]

            if bullish_gap:
                latest_fvg = {
                    "direction": "BULLISH",
                    "lower": high[position - 2],
                    "upper": low[position],
                    "size": low[position] - high[position - 2],
                    "creation_index": position,
                    "creation_atr": atr[position],
                    "creation_timestamp": (
                        result["timestamp"].iloc[position]
                        if "timestamp" in result.columns
                        else None
                    ),
                }
            elif bearish_gap:
                latest_fvg = {
                    "direction": "BEARISH",
                    "lower": high[position],
                    "upper": low[position - 2],
                    "size": low[position - 2] - high[position],
                    "creation_index": position,
                    "creation_atr": atr[position],
                    "creation_timestamp": (
                        result["timestamp"].iloc[position]
                        if "timestamp" in result.columns
                        else None
                    ),
                }

        if latest_fvg is not None:
            fvg_direction[position] = latest_fvg["direction"]
            fvg_lower[position] = latest_fvg["lower"]
            fvg_upper[position] = latest_fvg["upper"]
            fvg_size[position] = latest_fvg["size"]
            fvg_creation_index[position] = latest_fvg["creation_index"]
            fvg_creation_atr[position] = latest_fvg["creation_atr"]
            fvg_creation_timestamp[position] = latest_fvg["creation_timestamp"]

    current_index = np.arange(size, dtype=float)

    result["fvg_present"] = (~pd.isna(fvg_creation_index)).astype(int)
    result["fvg_direction"] = pd.Series(
        fvg_direction,
        index=result.index,
        dtype="object",
    )

    result["fvg_size"] = pd.Series(fvg_size, index=result.index)
    result["fvg_size_atr"] = np.nan

    valid_creation_atr = (
        ~pd.isna(fvg_size)
        & ~pd.isna(fvg_creation_atr)
        & (fvg_creation_atr > 0)
    )
    result.loc[valid_creation_atr, "fvg_size_atr"] = (
        fvg_size[valid_creation_atr]
        / fvg_creation_atr[valid_creation_atr]
    )

    result["fvg_age_bars"] = (
        current_index - fvg_creation_index
    )
    result["fvg_distance"] = np.nan
    result["fvg_position"] = np.nan

    valid_fvg = ~pd.isna(fvg_lower) & ~pd.isna(fvg_upper)
    inside_fvg = (
        valid_fvg
        & (close >= fvg_lower)
        & (close <= fvg_upper)
    )
    below_fvg = valid_fvg & (close < fvg_lower)
    above_fvg = valid_fvg & (close > fvg_upper)

    result.loc[inside_fvg, "fvg_distance"] = 0.0
    result.loc[below_fvg, "fvg_distance"] = (
        fvg_lower[below_fvg] - close[below_fvg]
    )
    result.loc[above_fvg, "fvg_distance"] = (
        close[above_fvg] - fvg_upper[above_fvg]
    )

    fvg_width = fvg_upper - fvg_lower
    valid_position = valid_fvg & (fvg_width > 0)
    result.loc[valid_position, "fvg_position"] = (
        (close[valid_position] - fvg_lower[valid_position])
        / fvg_width[valid_position]
    )

    result["fvg_distance_atr"] = np.nan
    if "atr_14" in result.columns:
        current_atr = result["atr_14"].to_numpy(dtype=float)
        distance = result["fvg_distance"].to_numpy(dtype=float)
        valid_distance_atr = (
            ~pd.isna(distance)
            & ~pd.isna(current_atr)
            & (current_atr > 0)
        )
        result.loc[valid_distance_atr, "fvg_distance_atr"] = (
            distance[valid_distance_atr]
            / current_atr[valid_distance_atr]
        )

    result["fvg_creation_timestamp"] = pd.Series(
        fvg_creation_timestamp,
        index=result.index,
        dtype="object",
    )

    return result


def add_trend_range_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic trend/regime and structural-range context."""
    required = {
        "structure_direction",
        "structure_last_valid_high",
        "structure_last_valid_low",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "Missing structural context columns: " + ", ".join(sorted(missing))
        )

    result = frame.copy()
    result["trend_regime"] = np.select(
        [result["structure_direction"] > 0, result["structure_direction"] < 0],
        ["BULLISH", "BEARISH"],
        default="NEUTRAL",
    )

    bullish_choch = result.get(
        "structure_bullish_choch", pd.Series(0, index=result.index)
    )
    bearish_choch = result.get(
        "structure_bearish_choch", pd.Series(0, index=result.index)
    )
    result["trend_transition"] = (
        bullish_choch.astype(bool) | bearish_choch.astype(bool)
    ).astype(int)
    result["trend_transition_direction"] = pd.Series(
        np.select(
            [bullish_choch.astype(bool), bearish_choch.astype(bool)],
            ["UP", "DOWN"],
            default=None,
        ),
        index=result.index,
        dtype="object",
    )

    range_high = result["structure_last_valid_high"]
    range_low = result["structure_last_valid_low"]
    range_width = range_high - range_low

    # A structural range is valid only when the canonical structure
    # direction is already established and both confirmed swing levels
    # form a positive-width range.
    valid_range = (
        result["structure_direction"].ne(0)
        & range_high.notna()
        & range_low.notna()
        & range_width.gt(0)
    )

    result["range_state"] = np.where(valid_range, "DEFINED", "UNDEFINED")
    result["range_high"] = range_high
    result["range_low"] = range_low
    result["range_width"] = range_width.where(valid_range)
    result["range_position"] = np.nan
    result.loc[valid_range, "range_position"] = (
        (result.loc[valid_range, "close"] - range_low[valid_range])
        / range_width[valid_range]
    )
    result["range_position_zone"] = np.select(
        [result["range_position"].lt(0.33), result["range_position"].gt(0.67)],
        ["LOW", "HIGH"],
        default="MID",
    )
    result.loc[~valid_range, "range_position_zone"] = "UNDEFINED"
    return result


def add_liquidity_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic liquidity levels and sweep events."""
    required = {
        "high",
        "low",
        "close",
        "structure_last_valid_high",
        "structure_last_valid_low",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "Missing liquidity context columns: " + ", ".join(sorted(missing))
        )

    result = frame.copy()

    # Only levels confirmed before the current candle are actionable.
    liquidity_high = result["structure_last_valid_high"].shift(1)
    liquidity_low = result["structure_last_valid_low"].shift(1)

    result["liquidity_high"] = liquidity_high
    result["liquidity_low"] = liquidity_low
    result["liquidity_high_present"] = liquidity_high.notna().astype(int)
    result["liquidity_low_present"] = liquidity_low.notna().astype(int)
    result["distance_to_liquidity_high"] = liquidity_high - result["close"]
    result["distance_to_liquidity_low"] = result["close"] - liquidity_low

    high_sweep = (
        liquidity_high.notna()
        & result["high"].gt(liquidity_high)
        & result["close"].le(liquidity_high)
    )
    low_sweep = (
        liquidity_low.notna()
        & result["low"].lt(liquidity_low)
        & result["close"].ge(liquidity_low)
    )

    result["liquidity_high_sweep"] = high_sweep.astype(int)
    result["liquidity_low_sweep"] = low_sweep.astype(int)
    result["liquidity_sweep"] = (high_sweep | low_sweep).astype(int)

    result["liquidity_sweep_direction"] = "NONE"
    result.loc[high_sweep & ~low_sweep, "liquidity_sweep_direction"] = "BEARISH"
    result.loc[low_sweep & ~high_sweep, "liquidity_sweep_direction"] = "BULLISH"
    result.loc[high_sweep & low_sweep, "liquidity_sweep_direction"] = "BOTH"

    result["liquidity_sweep_size"] = np.nan
    result.loc[high_sweep, "liquidity_sweep_size"] = (
        result.loc[high_sweep, "high"] - liquidity_high[high_sweep]
    )
    result.loc[low_sweep, "liquidity_sweep_size"] = (
        liquidity_low[low_sweep] - result.loc[low_sweep, "low"]
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
