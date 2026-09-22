"""Deterministic price-action and market-structure feature engineering."""

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


def add_price_action_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic candle price-action features."""
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

    result["close_position_in_range"] = result["close_position"]

    result["upper_wick_to_body"] = np.nan
    result["lower_wick_to_body"] = np.nan

    valid_body = candle_body.ne(0)

    result.loc[valid_body, "upper_wick_to_body"] = (
        upper_wick[valid_body] / candle_body[valid_body]
    )
    result.loc[valid_body, "lower_wick_to_body"] = (
        lower_wick[valid_body] / candle_body[valid_body]
    )

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
        "age_bars",
        "distance",
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
                result.columns.get_loc(f"ob_{direction}_age_bars"),
            ] = position - latest.event_index
            result.iloc[
                position,
                result.columns.get_loc(f"ob_{direction}_distance"),
            ] = distance
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
    """Add confirmed structural S/R and point-in-time location context."""
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

    swings_by_confirmation: dict[int, list] = {}
    for swing in swings:
        swings_by_confirmation.setdefault(
            swing.confirmation_index,
            [],
        ).append(swing)

    close = result["close"].to_numpy(dtype=float)
    direction = result["structure_direction"].to_numpy(dtype=float)

    support = np.full(size, np.nan)
    resistance = np.full(size, np.nan)
    support_age = np.full(size, np.nan)
    resistance_age = np.full(size, np.nan)
    support_type = np.full(size, None, dtype=object)
    resistance_type = np.full(size, None, dtype=object)
    support_label = np.full(size, None, dtype=object)
    resistance_label = np.full(size, None, dtype=object)
    support_scope = np.full(size, None, dtype=object)
    resistance_scope = np.full(size, None, dtype=object)
    next_structure = np.full(size, np.nan)

    confirmed_highs: list = []
    confirmed_lows: list = []

    for position in range(size):
        for swing in swings_by_confirmation.get(position, []):
            if swing.swing_type is SwingType.HIGH:
                confirmed_highs.append(swing)
            else:
                confirmed_lows.append(swing)

        current_close = close[position]

        supports = [
            swing
            for swing in confirmed_lows
            if swing.price <= current_close
        ]
        resistances = [
            swing
            for swing in confirmed_highs
            if swing.price >= current_close
        ]

        if supports:
            selected = max(
                supports,
                key=lambda swing: (swing.price, swing.confirmation_index),
            )
            support[position] = selected.price
            support_age[position] = position - selected.confirmation_index
            support_type[position] = selected.swing_type.value
            support_label[position] = selected.label
            support_scope[position] = selected.scope.value

        if resistances:
            selected = min(
                resistances,
                key=lambda swing: (swing.price, -swing.confirmation_index),
            )
            resistance[position] = selected.price
            resistance_age[position] = position - selected.confirmation_index
            resistance_type[position] = selected.swing_type.value
            resistance_label[position] = selected.label
            resistance_scope[position] = selected.scope.value

        if direction[position] > 0:
            next_levels = [
                swing.price
                for swing in confirmed_highs
                if swing.price > current_close
            ]
            if next_levels:
                next_structure[position] = min(next_levels)

        elif direction[position] < 0:
            next_levels = [
                swing.price
                for swing in confirmed_lows
                if swing.price < current_close
            ]
            if next_levels:
                next_structure[position] = max(next_levels)

    result["local_sr_support"] = support
    result["local_sr_resistance"] = resistance
    result["local_sr_support_present"] = np.isfinite(support).astype(int)
    result["local_sr_resistance_present"] = np.isfinite(resistance).astype(int)
    result["local_sr_support_age"] = support_age
    result["local_sr_resistance_age"] = resistance_age
    result["local_sr_support_swing_type"] = pd.Series(
        support_type,
        index=result.index,
        dtype="object",
    )
    result["local_sr_resistance_swing_type"] = pd.Series(
        resistance_type,
        index=result.index,
        dtype="object",
    )
    result["local_sr_support_label"] = pd.Series(
        support_label,
        index=result.index,
        dtype="object",
    )
    result["local_sr_resistance_label"] = pd.Series(
        resistance_label,
        index=result.index,
        dtype="object",
    )
    result["local_sr_support_scope"] = pd.Series(
        support_scope,
        index=result.index,
        dtype="object",
    )
    result["local_sr_resistance_scope"] = pd.Series(
        resistance_scope,
        index=result.index,
        dtype="object",
    )

    result["support_level"] = support
    result["resistance_level"] = resistance
    result["distance_to_support"] = close - support
    result["distance_to_resistance"] = resistance - close

    result["distance_to_next_structure_level"] = np.nan
    valid_next = np.isfinite(next_structure)
    result.loc[valid_next, "distance_to_next_structure_level"] = np.abs(
        next_structure[valid_next] - close[valid_next]
    )

    result["leg_position"] = result.get(
        "range_position",
        pd.Series(np.nan, index=result.index),
    )

    return result


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the deterministic price-action and structural feature set."""
    result = add_price_action_features(frame)
    result = add_fvg_features(result)
    result = add_structure_event_features(result)
    result = add_event_sequence_features(result)
    result = add_active_structure_features(result)
    result = add_trend_range_features(result)
    result = add_liquidity_features(result)
    result = add_order_block_features(result)
    result = add_sr_location_features(result)
    result = add_pullback_features(result)
    return result



def build_structural_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the structural feature set for the current blueprint."""

    result = add_structure_event_features(frame)
    result = add_event_sequence_features(result)
    result = add_active_structure_features(result)
    result = add_trend_range_features(result)
    result = add_liquidity_features(result)
    return add_order_block_features(result)


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


def add_event_sequence_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Project confirmed structure-event history into point-in-time features."""
    result = frame.copy()
    structural = build_structural_sequence(result)
    _, events = process_structural_candles(structural)

    size = len(result)
    events_by_index: dict[int, list] = {}

    for event in events:
        if event.index < 0 or event.index >= size:
            raise ValueError("Structure event index is outside the feature frame.")
        events_by_index.setdefault(event.index, []).append(event)

    event_count = np.zeros(size, dtype=int)
    last_event_index = np.full(size, np.nan)
    last_event_age = np.full(size, np.nan)
    last_event_type = np.full(size, None, dtype=object)
    last_event_direction = np.full(size, None, dtype=object)
    last_event_scope = np.full(size, None, dtype=object)

    previous_event_index = np.full(size, np.nan)
    previous_event_age = np.full(size, np.nan)
    previous_event_type = np.full(size, None, dtype=object)
    previous_event_direction = np.full(size, None, dtype=object)
    previous_event_scope = np.full(size, None, dtype=object)

    events_since_last_bos = np.full(size, np.nan)
    events_since_last_swing = np.full(size, np.nan)
    bars_since_last_bos = np.full(size, np.nan)
    bars_since_last_swing = np.full(size, np.nan)

    total_events = 0
    last_event = None
    previous_event = None
    events_after_bos = None
    events_after_swing = None
    last_bos_index = None
    last_swing_index = None

    for position in range(size):
        position_events = events_by_index.get(position, [])

        for event in position_events:
            previous_event = last_event
            last_event = event
            total_events += 1

            if event.event.endswith("_BOS"):
                events_after_bos = 0
                last_bos_index = position
            elif events_after_bos is not None:
                events_after_bos += 1

            if event.event.startswith("SWING_"):
                events_after_swing = 0
                last_swing_index = position
            elif events_after_swing is not None:
                events_after_swing += 1

        # Multiple events can occur on one candle. The BOS/swing event
        # itself is the zero point, so events on the same candle must not
        # increment the counter after the reset.
        if any(event.event.endswith("_BOS") for event in position_events):
            events_after_bos = 0
        if any(event.event.startswith("SWING_") for event in position_events):
            events_after_swing = 0

        event_count[position] = total_events

        if last_event is not None:
            last_event_index[position] = last_event.index
            last_event_age[position] = position - last_event.index
            last_event_type[position] = last_event.event
            last_event_direction[position] = (
                last_event.direction.value
                if last_event.direction is not None
                else None
            )
            last_event_scope[position] = last_event.scope.value

        if previous_event is not None:
            previous_event_index[position] = previous_event.index
            previous_event_age[position] = position - previous_event.index
            previous_event_type[position] = previous_event.event
            previous_event_direction[position] = (
                previous_event.direction.value
                if previous_event.direction is not None
                else None
            )
            previous_event_scope[position] = previous_event.scope.value

        if events_after_bos is not None:
            events_since_last_bos[position] = events_after_bos

        if events_after_swing is not None:
            events_since_last_swing[position] = events_after_swing

        if last_bos_index is not None:
            bars_since_last_bos[position] = position - last_bos_index

        if last_swing_index is not None:
            bars_since_last_swing[position] = position - last_swing_index

    result["structure_event_count"] = event_count
    result["structure_last_event_index"] = pd.Series(
        last_event_index,
        index=result.index,
    )
    result["structure_last_event_age"] = pd.Series(
        last_event_age,
        index=result.index,
    )
    result["structure_last_event_type"] = pd.Series(
        last_event_type,
        index=result.index,
        dtype="object",
    )
    result["structure_last_event_direction"] = pd.Series(
        last_event_direction,
        index=result.index,
        dtype="object",
    )
    result["structure_last_event_scope"] = pd.Series(
        last_event_scope,
        index=result.index,
        dtype="object",
    )

    result["structure_previous_event_index"] = pd.Series(
        previous_event_index,
        index=result.index,
    )
    result["structure_previous_event_age"] = pd.Series(
        previous_event_age,
        index=result.index,
    )
    result["structure_previous_event_type"] = pd.Series(
        previous_event_type,
        index=result.index,
        dtype="object",
    )
    result["structure_previous_event_direction"] = pd.Series(
        previous_event_direction,
        index=result.index,
        dtype="object",
    )
    result["structure_previous_event_scope"] = pd.Series(
        previous_event_scope,
        index=result.index,
        dtype="object",
    )

    result["structure_events_since_last_bos"] = pd.Series(
        events_since_last_bos,
        index=result.index,
    )
    result["structure_events_since_last_swing"] = pd.Series(
        events_since_last_swing,
        index=result.index,
    )
    result["structure_bars_since_last_bos"] = pd.Series(
        bars_since_last_bos,
        index=result.index,
    )
    result["structure_bars_since_last_swing"] = pd.Series(
        bars_since_last_swing,
        index=result.index,
    )

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

    fvg_direction = np.empty(size, dtype=object)
    fvg_direction[:] = None
    fvg_lower = np.full(size, np.nan)
    fvg_upper = np.full(size, np.nan)
    fvg_size = np.full(size, np.nan)
    fvg_creation_index = np.full(size, np.nan)
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
            fvg_creation_timestamp[position] = latest_fvg["creation_timestamp"]

    current_index = np.arange(size, dtype=float)

    result["fvg_present"] = (~pd.isna(fvg_creation_index)).astype(int)
    result["fvg_direction"] = pd.Series(
        fvg_direction,
        index=result.index,
        dtype="object",
    )

    result["fvg_size"] = pd.Series(fvg_size, index=result.index)
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

    result["fvg_creation_timestamp"] = pd.Series(
        fvg_creation_timestamp,
        index=result.index,
        dtype="object",
    )

    return result


def add_pullback_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Project the active structural pullback state into dataframe features."""
    result = frame.copy()
    structural = build_structural_sequence(result)
    _, _, snapshots = process_structural_candles_with_context(structural)

    size = len(result)
    active = np.full(size, np.nan)
    direction = np.full(size, np.nan)
    start_index = np.full(size, np.nan)
    extreme_index = np.full(size, np.nan)
    price = np.full(size, np.nan)
    depth = np.full(size, np.nan)
    bars = np.full(size, np.nan)
    snapshot_mask = np.zeros(size, dtype=bool)

    direction_code = {
        "UP": 1.0,
        "DOWN": -1.0,
    }

    for snapshot in snapshots:
        position = snapshot.index
        if position < 0 or position >= size:
            raise ValueError("Structure snapshot index is outside the feature frame.")

        snapshot_mask[position] = True

        if snapshot.pullback is None:
            active[position] = 0.0
            continue

        pullback = snapshot.pullback
        active[position] = 1.0
        direction[position] = direction_code[pullback.direction.value]
        start_index[position] = pullback.index
        extreme_index[position] = pullback.extreme_index
        price[position] = pullback.price

        if pullback.direction is Direction.UP:
            depth[position] = pullback.extreme_price - pullback.price
        else:
            depth[position] = pullback.price - pullback.extreme_price

        bars[position] = position - pullback.index

    last_snapshot = np.maximum.accumulate(
        np.where(snapshot_mask, np.arange(size), -1)
    )

    def carry(values: np.ndarray) -> np.ndarray:
        valid = last_snapshot >= 0
        output = np.full(size, np.nan)
        positions = np.clip(last_snapshot, 0, None)
        output[valid] = values[positions[valid]]
        return output

    active = carry(active)
    direction = carry(direction)
    start_index = carry(start_index)
    extreme_index = carry(extreme_index)
    price = carry(price)
    depth = carry(depth)
    bars = carry(bars)

    result["pullback_active"] = pd.Series(
        active,
        index=result.index,
    ).fillna(0.0)

    result["pullback_direction"] = pd.Series(
        direction,
        index=result.index,
    )

    result["pullback_start_index"] = pd.Series(
        start_index,
        index=result.index,
    )

    result["pullback_extreme_index"] = pd.Series(
        extreme_index,
        index=result.index,
    )

    result["pullback_price"] = pd.Series(
        price,
        index=result.index,
    )

    result["pullback_depth"] = pd.Series(
        depth,
        index=result.index,
    )

    result["pullback_bars"] = pd.Series(
        bars,
        index=result.index,
    )

    inactive = result["pullback_active"].eq(0)
    result.loc[
        inactive,
        [
            "pullback_direction",
            "pullback_start_index",
            "pullback_extreme_index",
            "pullback_price",
            "pullback_depth",
            "pullback_bars",
        ],
    ] = np.nan

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
