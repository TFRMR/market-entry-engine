"""Relate post-entry excursion paths to subsequent structure transitions.

Scenario:
BOS -> next same-direction valid swing -> 50% retracement entry.

For each candidate, inspect only structure events occurring after entry.
Measure whether the path reaches +1R before an opposite structural transition,
and whether failure is preceded by an opposite BOS/CHOCH.

This is descriptive research; no signal or ranking is produced.
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
    process_structural_candles,
)

HORIZONS = (10, 20, 40, 80)
RETRACE = 0.50


def build_candidates(frame: pd.DataFrame):
    structural = build_structural_sequence(frame)
    swings, events = process_structural_candles(structural)
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


def trace(row: pd.Series, events, frame: pd.DataFrame, horizon: int) -> dict:
    entry_index = int(row["entry_index"])
    entry = float(row["entry"])
    invalidation = float(row["invalidation"])
    risk = float(row["risk"])
    direction = row["direction"]
    end = min(entry_index + horizon, len(frame) - 1)

    highs = frame.iloc[entry_index + 1:end + 1]["high"].astype(float).to_numpy()
    lows = frame.iloc[entry_index + 1:end + 1]["low"].astype(float).to_numpy()
    if len(highs) == 0:
        return {}

    if direction == "UP":
        favorable = highs - entry
        failure_hit = lows <= invalidation
        opposite = {"BEARISH_BOS", "BEARISH_CHOCH"}
    else:
        favorable = entry - lows
        failure_hit = highs >= invalidation
        opposite = {"BULLISH_BOS", "BULLISH_CHOCH"}

    first_failure = np.flatnonzero(failure_hit)
    failure_index = (
        entry_index + int(first_failure[0]) + 1
        if len(first_failure)
        else None
    )

    post_events = [
        e for e in events
        if entry_index < e.index <= end
    ]
    opposite_events = [e for e in post_events if e.event in opposite]
    same_events = [
        e for e in post_events
        if (
            (direction == "UP" and e.event == "BULLISH_BOS")
            or (direction == "DOWN" and e.event == "BEARISH_BOS")
        )
    ]

    first_opposite = opposite_events[0] if opposite_events else None
    first_same = same_events[0] if same_events else None

    reached_1 = np.flatnonzero(favorable >= risk)
    first_1_index = (
        entry_index + int(reached_1[0]) + 1
        if len(reached_1)
        else None
    )

    out = {
        "failure": bool(failure_index is not None),
        "failure_bar": (
            failure_index - entry_index
            if failure_index is not None else np.nan
        ),
        "hit_1R": bool(first_1_index is not None),
        "hit_1R_before_failure": (
            first_1_index is not None
            and (failure_index is None or first_1_index < failure_index)
        ),
        "opposite_transition": bool(first_opposite),
        "opposite_before_failure": (
            first_opposite is not None
            and (failure_index is None or first_opposite.index < failure_index)
        ),
        "same_bos_before_failure": (
            first_same is not None
            and (failure_index is None or first_same.index < failure_index)
        ),
        "first_opposite_event": first_opposite.event if first_opposite else None,
        "first_opposite_scope": (
            first_opposite.scope.value
            if first_opposite and first_opposite.scope else None
        ),
        "first_opposite_bar": (
            first_opposite.index - entry_index
            if first_opposite else np.nan
        ),
        "first_same_bos_bar": (
            first_same.index - entry_index
            if first_same else np.nan
        ),
    }

    if first_opposite and first_1_index is not None:
        out["opposite_after_1R"] = first_opposite.index > first_1_index
    else:
        out["opposite_after_1R"] = False

    if first_opposite:
        out["state"] = (
            "OPPOSITE_STRUCTURE"
            if first_opposite.index <= (failure_index or end)
            else "NONE"
        )
    elif failure_index is not None:
        out["state"] = "FAILURE_WITHOUT_OPPOSITE_EVENT"
    elif first_1_index is not None:
        out["state"] = "CONTINUATION_WITHOUT_OPPOSITE_EVENT"
    else:
        out["state"] = "UNRESOLVED"

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/research/xauusd_m30_structural_transition_path.csv"),
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
            metrics = trace(candidate, events, frame, horizon)
            if metrics:
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
                ]
                if subset.empty:
                    continue

                failure = subset[subset["failure"]]
                summary.append({
                    "period": period,
                    "direction": direction,
                    "horizon": horizon,
                    "n": len(subset),
                    "failure_rate": subset["failure"].mean(),
                    "hit_1R_before_failure": subset["hit_1R_before_failure"].mean(),
                    "opposite_before_failure": subset["opposite_before_failure"].mean(),
                    "same_bos_before_failure": subset["same_bos_before_failure"].mean(),
                    "opposite_after_1R": subset["opposite_after_1R"].mean(),
                    "failure_without_opposite": (
                        (
                            failure["failure"]
                            & ~failure["opposite_before_failure"]
                        ).mean()
                        if not failure.empty else np.nan
                    ),
                    "opposite_bar_median": subset["first_opposite_bar"].median(),
                    "failure_bar_median": subset["failure_bar"].median(),
                })

    print("=== Structural Transition Path Research ===")
    print("BOS -> valid swing -> 50% retracement entry")
    print("Track post-entry opposite/same structure events")
    print()
    print(pd.DataFrame(summary).to_string(index=False))
    print()
    print("Candidates:", len(candidates))
    print("Artifact:", args.output)


if __name__ == "__main__":
    main()
