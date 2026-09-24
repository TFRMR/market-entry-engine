"""Describe the structural context immediately before/at post-entry failure.

Scenario:
BOS -> next same-direction valid swing -> 50% retracement entry.

For each candidate, find the first confirmed-swing invalidation within the
observation horizon, then snapshot the point-in-time structural context around
that failure. No future candles are used for the snapshot.

This is descriptive research, not a trading rule.
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
    StructureScope,
    build_structural_sequence,
    process_structural_candles_with_context,
)

HORIZONS = (10, 20, 40, 80)
RETRACE = 0.50
WINDOWS = (3, 5, 10)


def build_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    structural = build_structural_sequence(frame)
    swings, events, snapshots = process_structural_candles_with_context(structural)
    rows = []
    used = set()

    for event in events:
        if event.event not in {"BULLISH_BOS", "BEARISH_BOS"}:
            continue

        direction = Direction.UP if event.event == "BULLISH_BOS" else Direction.DOWN
        future = [
            swing for swing in swings
            if swing.confirmation_index > event.index
            and (
                (direction is Direction.UP and swing.swing_type is SwingType.LOW)
                or (direction is Direction.DOWN and swing.swing_type is SwingType.HIGH)
            )
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

    return pd.DataFrame(rows), events, snapshots


def failure_context(
    row: pd.Series,
    frame: pd.DataFrame,
    events,
    snapshots,
    horizon: int,
) -> dict:
    entry_index = int(row["entry_index"])
    invalidation = float(row["invalidation"])
    end = min(entry_index + horizon, len(frame) - 1)

    if row["direction"] == "UP":
        failure_hit = frame.iloc[entry_index + 1:end + 1]["low"].astype(float).to_numpy() <= invalidation
    else:
        failure_hit = frame.iloc[entry_index + 1:end + 1]["high"].astype(float).to_numpy() >= invalidation

    hits = np.flatnonzero(failure_hit)
    if not len(hits):
        return {
            "failure": False,
            "failure_bar": np.nan,
            "failure_state_direction": None,
            "failure_last_event": None,
            "failure_last_event_direction": None,
            "failure_last_event_scope": None,
            "failure_last_event_bar": np.nan,
            "failure_events_last_3": 0,
            "failure_events_last_5": 0,
            "failure_events_last_10": 0,
            "failure_same_events_last_10": 0,
            "failure_opposite_events_last_10": 0,
            "failure_last_swing_age": np.nan,
            "failure_last_high_label": None,
            "failure_last_low_label": None,
        }

    failure_index = entry_index + int(hits[0]) + 1
    snap = next((s for s in snapshots if s.index == failure_index), None)
    if snap is None:
        return {"failure": True, "failure_bar": failure_index - entry_index}

    prior_events = [e for e in events if entry_index < e.index <= failure_index]
    last_event = prior_events[-1] if prior_events else None

    same = {"BULLISH_BOS", "SWING_HIGH_VALID"} if row["direction"] == "UP" else {"BEARISH_BOS", "SWING_LOW_VALID"}
    opposite = {"BEARISH_BOS", "BEARISH_CHOCH"} if row["direction"] == "UP" else {"BULLISH_BOS", "BULLISH_CHOCH"}

    out = {
        "failure": True,
        "failure_bar": failure_index - entry_index,
        "failure_state_direction": snap.direction.value if snap.direction else None,
        "failure_last_event": last_event.event if last_event else None,
        "failure_last_event_direction": last_event.direction.value if last_event and last_event.direction else None,
        "failure_last_event_scope": last_event.scope.value if last_event else None,
        "failure_last_event_bar": (
            failure_index - last_event.index if last_event else np.nan
        ),
        "failure_last_swing_age": (
            failure_index - snap.last_swing_confirmation_index
            if snap.last_swing_confirmation_index is not None
            else np.nan
        ),
        "failure_last_high_label": snap.last_high.label if snap.last_high else None,
        "failure_last_low_label": snap.last_low.label if snap.last_low else None,
    }

    for window in WINDOWS:
        start = failure_index - window
        recent = [e for e in events if start < e.index <= failure_index and e.index > entry_index]
        out[f"failure_events_last_{window}"] = len(recent)
        out[f"failure_same_events_last_{window}"] = sum(e.event in same for e in recent)
        out[f"failure_opposite_events_last_{window}"] = sum(e.event in opposite for e in recent)

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/research/xauusd_m30_structural_failure_context.csv"),
    )
    args = parser.parse_args()

    frame = load_mt5_csv(args.csv)
    candidates, events, snapshots = build_candidates(frame)
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
            metrics = failure_context(candidate, frame, events, snapshots, horizon)
            rows.append({
                **candidate.to_dict(),
                "period": period,
                "horizon": horizon,
                **metrics,
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
                    & result["failure"]
                ]
                if subset.empty:
                    continue

                summary.append({
                    "period": period,
                    "direction": direction,
                    "horizon": horizon,
                    "failure_n": len(subset),
                    "failure_bar_median": subset["failure_bar"].median(),
                    "last_event": subset["failure_last_event"].value_counts().to_dict(),
                    "last_event_scope": subset["failure_last_event_scope"].value_counts().to_dict(),
                    "last_event_age_median": subset["failure_last_event_bar"].median(),
                    "last_swing_age_median": subset["failure_last_swing_age"].median(),
                    "events_last_3_median": subset["failure_events_last_3"].median(),
                    "events_last_5_median": subset["failure_events_last_5"].median(),
                    "events_last_10_median": subset["failure_events_last_10"].median(),
                    "opposite_events_last_10_mean": subset["failure_opposite_events_last_10"].mean(),
                })

    print("=== Structural Failure Context Research ===")
    print("BOS -> valid swing -> 50% retracement entry")
    print("Failure = confirmed swing invalidation; snapshot context at failure")
    print()
    print(pd.DataFrame(summary).to_string(index=False))
    print()
    print("Candidates:", len(candidates))
    print("Artifact:", args.output)


if __name__ == "__main__":
    main()
