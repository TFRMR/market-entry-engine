"""Research ordered post-entry market journey event sequences.

Scenario:
BOS -> next same-direction valid swing -> 50% retracement entry.

The output preserves structural events and invalidation as an ordered journey,
without ranking setups or producing trading signals.
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


def classify_event(event, direction):
    if event.event.endswith("_BOS"):
        if (
            direction == "UP" and event.event == "BULLISH_BOS"
        ) or (
            direction == "DOWN" and event.event == "BEARISH_BOS"
        ):
            return "CONTINUATION"
        return "TRANSITION"
    if event.event.endswith("_CHOCH"):
        return "TRANSITION"
    if event.event.endswith("_VALID"):
        return "SWING_UPDATE"
    return "STRUCTURE_EVENT"


def build_sequence(row, frame, events, horizon):
    entry_index = int(row["entry_index"])
    end = min(entry_index + horizon, len(frame) - 1)
    direction = row["direction"]
    invalidation = float(row["invalidation"])
    risk = float(row["risk"])

    post_events = [e for e in events if entry_index < e.index <= end]
    sequence = [{"bar": 0, "event": "ENTRY", "scope": None, "index": entry_index}]

    seen_invalidation = False
    for index in range(entry_index + 1, end + 1):
        candle = frame.iloc[index]
        crossed = (
            float(candle["low"]) <= invalidation
            if direction == "UP"
            else float(candle["high"]) >= invalidation
        )
        same_events = [e for e in post_events if e.index == index]

        for event in same_events:
            sequence.append({
                "bar": index - entry_index,
                "event": classify_event(event, direction),
                "raw_event": event.event,
                "scope": event.scope.value,
                "index": index,
            })

        if crossed and not seen_invalidation:
            sequence.append({
                "bar": index - entry_index,
                "event": "INVALIDATION",
                "raw_event": None,
                "scope": None,
                "index": index,
            })
            seen_invalidation = True
            break

    if not sequence or sequence[-1]["event"] != "INVALIDATION":
        last_bar = sequence[-1]["bar"]
    else:
        last_bar = sequence[-1]["bar"]

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

    return {
        "sequence": " > ".join(item["event"] for item in sequence),
        "sequence_raw": " > ".join(
            item.get("raw_event") or item["event"] for item in sequence
        ),
        "sequence_event_count": len(sequence) - 1,
        "first_post_entry_event": sequence[1]["event"] if len(sequence) > 1 else None,
        "first_post_entry_raw_event": (
            sequence[1].get("raw_event") or sequence[1]["event"]
            if len(sequence) > 1 else None
        ),
        "first_post_entry_bar": sequence[1]["bar"] if len(sequence) > 1 else np.nan,
        "sequence_last_bar": last_bar,
        "invalidation_in_sequence": seen_invalidation,
        "mfe_r": mfe / risk if risk else np.nan,
        "mae_r": mae / risk if risk else np.nan,
        "hit_1r": mfe >= risk,
        "hit_2r": mfe >= 2 * risk,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/research/xauusd_m30_market_journey_sequence.csv"),
    )
    args = parser.parse_args()

    frame = load_mt5_csv(args.csv)
    candidates, events = build_candidates(frame)
    if candidates.empty:
        print("No candidates.")
        return

    boundary = pd.Timestamp("2026-03-18")
    rows = []
    long_rows = []

    for _, candidate in candidates.iterrows():
        period = (
            "development"
            if pd.Timestamp(candidate["confirmation_timestamp"]) < boundary
            else "historical_oos"
        )
        for horizon in HORIZONS:
            data = build_sequence(candidate, frame, events, horizon)
            rows.append({
                **candidate.to_dict(),
                "period": period,
                "horizon": horizon,
                **data,
            })

            tokens = data["sequence"].split(" > ")
            for step, token in enumerate(tokens):
                long_rows.append({
                    "bos_index": candidate["bos_index"],
                    "entry_index": candidate["entry_index"],
                    "direction": candidate["direction"],
                    "period": period,
                    "horizon": horizon,
                    "step": step,
                    "bar": 0 if step == 0 else np.nan,
                    "event": token,
                })

    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    prefix_rows = []
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
                prefixes = (
                    subset["sequence"]
                    .str.split(" > ")
                    .str[:3]
                    .str.join(" > ")
                    .value_counts()
                    .head(8)
                    .to_dict()
                )
                prefix_rows.append({
                    "period": period,
                    "direction": direction,
                    "horizon": horizon,
                    "n": len(subset),
                    "first_event": subset["first_post_entry_event"].value_counts().to_dict(),
                    "first_event_bar_median": subset["first_post_entry_bar"].median(),
                    "prefixes_len_3": prefixes,
                    "invalidation_rate": subset["invalidation_in_sequence"].mean(),
                    "hit_1r": subset["hit_1r"].mean(),
                    "hit_2r": subset["hit_2r"].mean(),
                    "mfe_r_median": subset["mfe_r"].median(),
                    "mae_r_median": subset["mae_r"].median(),
                })

    print("=== Market Journey Sequence Research ===")
    print("BOS -> valid swing -> 50% retracement entry")
    print("Sequence: ENTRY -> SWING_UPDATE / CONTINUATION / TRANSITION -> INVALIDATION")
    print()
    print(pd.DataFrame(prefix_rows).to_string(index=False))
    print()
    print("Candidates:", len(candidates))
    print("Artifact:", args.output)


if __name__ == "__main__":
    main()
