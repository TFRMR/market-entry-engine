"""Research empirical transitions between post-entry structural events.

Scenario:
BOS -> next same-direction valid swing -> 50% retracement entry.

This script describes event-to-event transitions. It does not rank setups
or create trading signals.
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


def build_candidates(frame):
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


def transition_rows(candidate, frame, events, horizon):
    entry_index = int(candidate["entry_index"])
    end = min(entry_index + horizon, len(frame) - 1)
    direction = candidate["direction"]
    invalidation = float(candidate["invalidation"])
    risk = float(candidate["risk"])

    structural_events = [
        event for event in events
        if entry_index < event.index <= end
    ]

    points = [{"bar": 0, "event": "ENTRY", "index": entry_index}]
    seen_invalidation = False

    for index in range(entry_index + 1, end + 1):
        candle = frame.iloc[index]
        crossed = (
            float(candle["low"]) <= invalidation
            if direction == "UP"
            else float(candle["high"]) >= invalidation
        )
        for event in [e for e in structural_events if e.index == index]:
            points.append({
                "bar": index - entry_index,
                "event": event.event,
                "scope": event.scope.value,
                "index": index,
            })
        if crossed and not seen_invalidation:
            points.append({
                "bar": index - entry_index,
                "event": "INVALIDATION",
                "scope": None,
                "index": index,
            })
            seen_invalidation = True
            break

    transitions = []
    for current, nxt in zip(points, points[1:]):
        transitions.append({
            "from_event": current["event"],
            "to_event": nxt["event"],
            "bars": nxt["bar"] - current["bar"],
            "from_bar": current["bar"],
            "to_bar": nxt["bar"],
        })

    # Path statistics up to the next event.
    for transition in transitions:
        start = entry_index + int(transition["from_bar"]) + 1
        stop = entry_index + int(transition["to_bar"])
        mfe = 0.0
        mae = 0.0
        for index in range(start, min(stop, len(frame) - 1) + 1):
            candle = frame.iloc[index]
            high = float(candle["high"])
            low = float(candle["low"])
            if direction == "UP":
                mfe = max(mfe, high - float(candidate["entry"]))
                mae = max(mae, float(candidate["entry"]) - low)
            else:
                mfe = max(mfe, float(candidate["entry"]) - low)
                mae = max(mae, high - float(candidate["entry"]))
        transition["mfe_r_to_next"] = mfe / risk if risk else np.nan
        transition["mae_r_to_next"] = mae / risk if risk else np.nan

    return transitions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/research/xauusd_m30_market_journey_transition_matrix.csv"),
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
            for transition in transition_rows(candidate, frame, events, horizon):
                rows.append({
                    "bos_index": candidate["bos_index"],
                    "entry_index": candidate["entry_index"],
                    "direction": candidate["direction"],
                    "period": period,
                    "horizon": horizon,
                    **transition,
                })

    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    print("=== Market Journey Transition Matrix ===")
    print("BOS -> valid swing -> 50% retracement entry")
    print("Transition: current event -> next structural event / invalidation")
    print()

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

                grouped = (
                    subset.groupby(["from_event", "to_event"])
                    .agg(
                        transitions=("to_event", "size"),
                        bars_median=("bars", "median"),
                        bars_p25=("bars", lambda s: s.quantile(0.25)),
                        bars_p75=("bars", lambda s: s.quantile(0.75)),
                        mfe_r_median=("mfe_r_to_next", "median"),
                        mae_r_median=("mae_r_to_next", "median"),
                    )
                    .reset_index()
                )

                totals = (
                    grouped.groupby("from_event")["transitions"]
                    .sum()
                    .rename("from_total")
                    .reset_index()
                )
                grouped = grouped.merge(totals, on="from_event", how="left")
                grouped["transition_rate"] = (
                    grouped["transitions"] / grouped["from_total"]
                )
                grouped = grouped.sort_values(
                    ["from_event", "transitions"],
                    ascending=[True, False],
                )

                print(
                    f"--- {period} {direction} H{horizon} "
                    f"n_candidates={len(subset['entry_index'].unique())} ---"
                )
                print(
                    grouped[
                        [
                            "from_event",
                            "to_event",
                            "transitions",
                            "transition_rate",
                            "bars_median",
                            "bars_p25",
                            "bars_p75",
                            "mfe_r_median",
                            "mae_r_median",
                        ]
                    ].to_string(index=False)
                )
                print()

    print("Candidates:", len(candidates))
    print("Transition rows:", len(result))
    print("Artifact:", args.output)


if __name__ == "__main__":
    main()
