"""Research post-entry market journey as a structural state sequence.

Scenario:
BOS -> next same-direction valid swing -> 50% retracement entry.

The journey is descriptive. It does not rank setups or create trading signals.
A same-candle opposite structural event and invalidation are kept explicit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from market_engine.data import load_mt5_csv
from market_engine.structure import (
    Direction,
    SwingType,
    build_structural_sequence,
    process_structural_candles_with_context,
)

HORIZONS = (10, 20, 40, 80)
RETRACE = 0.50


def build_candidates(frame: pd.DataFrame):
    structural = build_structural_sequence(frame)
    swings, events, _snapshots = process_structural_candles_with_context(structural)

    rows = []
    used = set()

    for event in events:
        if event.event not in {"BULLISH_BOS", "BEARISH_BOS"}:
            continue

        direction = Direction.UP if event.event == "BULLISH_BOS" else Direction.DOWN
        wanted = SwingType.LOW if direction is Direction.UP else SwingType.HIGH
        future = [
            swing for swing in swings
            if swing.confirmation_index > event.index
            and swing.swing_type is wanted
        ]
        if not future:
            continue

        swing = min(future, key=lambda item: item.confirmation_index)
        cidx = swing.confirmation_index
        key = (event.index, cidx)
        if key in used:
            continue
        used.add(key)

        confirmation = float(frame.iloc[cidx]["close"])
        invalidation = float(swing.price)
        leg = (
            confirmation - invalidation
            if direction is Direction.UP
            else invalidation - confirmation
        )
        if leg <= 0:
            continue

        entry = (
            confirmation - RETRACE * leg
            if direction is Direction.UP
            else confirmation + RETRACE * leg
        )

        entry_index = None
        for index in range(cidx + 1, len(frame)):
            candle = frame.iloc[index]
            if direction is Direction.UP and float(candle["low"]) <= entry:
                entry_index = index
                break
            if direction is Direction.DOWN and float(candle["high"]) >= entry:
                entry_index = index
                break
        if entry_index is None:
            continue

        risk = abs(entry - invalidation)
        if risk <= 0:
            continue

        rows.append({
            "bos_index": event.index,
            "bos_timestamp": frame.iloc[event.index]["timestamp"],
            "bos_event": event.event,
            "direction": direction.value,
            "confirmation_index": cidx,
            "confirmation_timestamp": frame.iloc[cidx]["timestamp"],
            "entry_index": entry_index,
            "entry_timestamp": frame.iloc[entry_index]["timestamp"],
            "entry": entry,
            "invalidation": invalidation,
            "risk": risk,
        })

    return pd.DataFrame(rows), events


def journey(row: pd.Series, frame: pd.DataFrame, events, horizon: int) -> dict:
    entry_index = int(row["entry_index"])
    direction = row["direction"]
    invalidation = float(row["invalidation"])
    risk = float(row["risk"])
    end = min(entry_index + horizon, len(frame) - 1)

    same_bos = "BULLISH_BOS" if direction == "UP" else "BEARISH_BOS"
    opposite = (
        {"BEARISH_BOS", "BEARISH_CHOCH"}
        if direction == "UP"
        else {"BULLISH_BOS", "BULLISH_CHOCH"}
    )

    post_events = [e for e in events if entry_index < e.index <= end]
    first_event = post_events[0] if post_events else None

    failure_index = None
    for index in range(entry_index + 1, end + 1):
        candle = frame.iloc[index]
        if direction == "UP" and float(candle["low"]) <= invalidation:
            failure_index = index
            break
        if direction == "DOWN" and float(candle["high"]) >= invalidation:
            failure_index = index
            break

    first_opposite = next((e for e in post_events if e.event in opposite), None)
    first_same = next((e for e in post_events if e.event == same_bos), None)

    if failure_index is not None:
        same_candle_opposite = [
            e for e in post_events
            if e.index == failure_index and e.event in opposite
        ]
        if same_candle_opposite:
            terminal = "INVALIDATED_WITH_OPPOSITE_EVENT"
        else:
            terminal = "INVALIDATED"
    elif first_opposite is not None:
        terminal = "OPPOSITE_STRUCTURE"
    elif first_same is not None:
        terminal = "CONTINUATION"
    else:
        terminal = "NO_STRUCTURE_TRANSITION"

    mfe = 0.0
    mae = 0.0
    for index in range(entry_index + 1, end + 1):
        candle = frame.iloc[index]
        high = float(candle["high"])
        low = float(candle["low"])
        if direction == "UP":
            mfe = max(mfe, high - float(row["entry"]))
            mae = max(mae, float(row["entry"]) - low)
        else:
            mfe = max(mfe, float(row["entry"]) - low)
            mae = max(mae, high - float(row["entry"]))

    def bar_of(event):
        return event.index - entry_index if event is not None else np.nan

    failure_bar = (
        failure_index - entry_index if failure_index is not None else np.nan
    )

    return {
        "terminal_state": terminal,
        "first_event": first_event.event if first_event else None,
        "first_event_scope": first_event.scope.value if first_event else None,
        "first_event_bar": bar_of(first_event),
        "first_opposite_event": first_opposite.event if first_opposite else None,
        "first_opposite_bar": bar_of(first_opposite),
        "first_same_bos_bar": bar_of(first_same),
        "failure": failure_index is not None,
        "failure_bar": failure_bar,
        "failure_same_candle_as_opposite": bool(
            failure_index is not None and first_opposite is not None
            and first_opposite.index == failure_index
        ),
        "mfe_price": mfe,
        "mae_price": mae,
        "mfe_r": mfe / risk if risk else np.nan,
        "mae_r": mae / risk if risk else np.nan,
        "hit_1r": mfe >= risk,
        "hit_2r": mfe >= 2 * risk,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/research/xauusd_m30_market_journey_state.csv"),
    )
    args = parser.parse_args()

    frame = load_mt5_csv(args.csv)
    candidates, events = build_candidates(frame)
    if candidates.empty:
        print("No candidates.")
        return

    boundary = pd.Timestamp("2026-03-18")
    rows = []

    for _, candidate in candidates.iterrows():
        period = (
            "development"
            if pd.Timestamp(candidate["confirmation_timestamp"]) < boundary
            else "historical_oos"
        )
        for horizon in HORIZONS:
            rows.append({
                **candidate.to_dict(),
                "period": period,
                "horizon": horizon,
                **journey(candidate, frame, events, horizon),
            })

    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    summary = []
    for period in ("development", "historical_oos"):
        for direction in ("UP", "DOWN"):
            for horizon in HORIZONS:
                subset = result[
                    (result["period"] == period)
                    & (result["direction"] == direction)
                    & (result["horizon"] == horizon)
                ]
                if subset.empty:
                    continue

                states = subset["terminal_state"].value_counts().to_dict()
                summary.append({
                    "period": period,
                    "direction": direction,
                    "horizon": horizon,
                    "n": len(subset),
                    "states": states,
                    "first_event": subset["first_event"].value_counts().to_dict(),
                    "first_event_bar_median": subset["first_event_bar"].median(),
                    "failure_rate": subset["failure"].mean(),
                    "failure_same_candle_opposite_rate": subset[
                        "failure_same_candle_as_opposite"
                    ].mean(),
                    "hit_1r": subset["hit_1r"].mean(),
                    "hit_2r": subset["hit_2r"].mean(),
                    "mfe_r_median": subset["mfe_r"].median(),
                    "mae_r_median": subset["mae_r"].median(),
                })

    print("=== Market Journey State Research ===")
    print("BOS -> valid swing -> 50% retracement entry")
    print("States: CONTINUATION / OPPOSITE_STRUCTURE / INVALIDATED / INVALIDATED_WITH_OPPOSITE_EVENT / NO_STRUCTURE_TRANSITION")
    print()
    print(pd.DataFrame(summary).to_string(index=False))
    print()
    print("Candidates:", len(candidates))
    print("Artifact:", args.output)


if __name__ == "__main__":
    main()
