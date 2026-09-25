"""Research semantic market-journey state transitions from raw structural events.

Scenario:
BOS -> next same-direction valid swing -> 50% retracement entry.

This is descriptive empirical research. It preserves raw events while adding
a direction-aware state interpretation for transition analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

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


def event_state(event: str, direction: str) -> str:
    if event == "ENTRY":
        return "ENTRY"
    if event in {"SWING_HIGH_VALID", "SWING_LOW_VALID"}:
        return "SWING_UPDATE"
    if event == "INVALIDATION":
        return "INVALIDATED"

    continuation_event = "BULLISH_BOS" if direction == "UP" else "BEARISH_BOS"
    if event == continuation_event:
        return "CONTINUATION"

    if event in {
        "BULLISH_BOS",
        "BEARISH_BOS",
        "BULLISH_CHOCH",
        "BEARISH_CHOCH",
    }:
        return "TRANSITION"

    raise ValueError(f"Unknown journey event: {event}")


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


def build_journey(candidate, frame, events, horizon):
    entry_index = int(candidate["entry_index"])
    end = min(entry_index + horizon, len(frame) - 1)
    direction = candidate["direction"]
    invalidation = float(candidate["invalidation"])
    risk = float(candidate["risk"])
    entry = float(candidate["entry"])

    structural_events = [
        event for event in events
        if entry_index < event.index <= end
    ]

    points = [{
        "bar": 0,
        "event": "ENTRY",
        "scope": None,
        "index": entry_index,
    }]

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

        if crossed:
            points.append({
                "bar": index - entry_index,
                "event": "INVALIDATION",
                "scope": None,
                "index": index,
            })
            break

    rows = []
    for current, nxt in zip(points, points[1:]):
        start = entry_index + int(current["bar"]) + 1
        stop = entry_index + int(nxt["bar"])

        mfe = 0.0
        mae = 0.0
        for index in range(start, min(stop, len(frame) - 1) + 1):
            candle = frame.iloc[index]
            high = float(candle["high"])
            low = float(candle["low"])
            if direction == "UP":
                mfe = max(mfe, high - entry)
                mae = max(mae, entry - low)
            else:
                mfe = max(mfe, entry - low)
                mae = max(mae, high - entry)

        rows.append({
            "from_event": current["event"],
            "from_state": event_state(current["event"], direction),
            "from_scope": current["scope"],
            "to_event": nxt["event"],
            "to_state": event_state(nxt["event"], direction),
            "to_scope": nxt["scope"],
            "from_bar": current["bar"],
            "to_bar": nxt["bar"],
            "bars_to_next": nxt["bar"] - current["bar"],
            "mfe_r_to_next": mfe / risk,
            "mae_r_to_next": mae / risk,
        })

    terminal = points[-1]["event"]
    terminal_state = event_state(terminal, direction)
    terminal_bar = points[-1]["bar"]

    return rows, {
        "terminal_event": terminal,
        "terminal_state": terminal_state,
        "terminal_bar": terminal_bar,
        "horizon_reached": terminal_bar >= horizon,
        "event_count": len(points) - 1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/research/xauusd_m30_market_journey_state_transition.csv"),
    )
    args = parser.parse_args()

    frame = load_mt5_csv(args.csv)
    candidates, events = build_candidates(frame)
    if candidates.empty:
        print("No candidates.")
        return

    boundary = pd.Timestamp("2026-03-18")
    transition_rows = []
    terminal_rows = []

    for candidate_id, (_, candidate) in enumerate(candidates.iterrows()):
        period = (
            "development"
            if pd.Timestamp(candidate["confirmation_timestamp"]) < boundary
            else "historical_oos"
        )
        for horizon in HORIZONS:
            transitions, terminal = build_journey(candidate, frame, events, horizon)
            base = {
                "candidate_id": candidate_id,
                "direction": candidate["direction"],
                "period": period,
                "horizon": horizon,
            }
            for row in transitions:
                transition_rows.append({**base, **row})
            terminal_rows.append({**base, **terminal})

    transitions = pd.DataFrame(transition_rows)
    terminals = pd.DataFrame(terminal_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    transitions.to_csv(args.output, index=False)

    print("=== Market Journey State Transition Research ===")
    print("Direction-aware semantic state transitions from raw structural events")
    print()

    for period in ("development", "historical_oos"):
        for direction in ("UP", "DOWN"):
            for horizon in HORIZONS:
                subset = transitions[
                    (transitions["period"] == period)
                    & (transitions["direction"] == direction)
                    & (transitions["horizon"] == horizon)
                ]
                if subset.empty:
                    continue

                grouped = (
                    subset.groupby(["from_state", "to_state", "to_event"])
                    .agg(
                        transitions=("to_event", "size"),
                        bars_median=("bars_to_next", "median"),
                        bars_p25=("bars_to_next", lambda s: s.quantile(0.25)),
                        bars_p75=("bars_to_next", lambda s: s.quantile(0.75)),
                        mfe_r_median=("mfe_r_to_next", "median"),
                        mae_r_median=("mae_r_to_next", "median"),
                    )
                    .reset_index()
                )

                totals = (
                    grouped.groupby("from_state")["transitions"]
                    .sum()
                    .rename("observed_next_event_total")
                    .reset_index()
                )
                grouped = grouped.merge(totals, on="from_state", how="left")
                grouped["conditional_transition_rate"] = (
                    grouped["transitions"] / grouped["observed_next_event_total"]
                )
                grouped = grouped.sort_values(
                    ["from_state", "transitions"],
                    ascending=[True, False],
                )

                terminal = terminals[
                    (terminals["period"] == period)
                    & (terminals["direction"] == direction)
                    & (terminals["horizon"] == horizon)
                ]
                terminal_rate = (
                    terminal["terminal_state"].value_counts(normalize=True)
                    .sort_index()
                )

                print(
                    f"--- {period} {direction} H{horizon} "
                    f"n_candidates={len(terminal)} ---"
                )
                print(
                    grouped[
                        [
                            "from_state",
                            "to_state",
                            "to_event",
                            "transitions",
                            "conditional_transition_rate",
                            "bars_median",
                            "bars_p25",
                            "bars_p75",
                            "mfe_r_median",
                            "mae_r_median",
                        ]
                    ].to_string(index=False)
                )
                print("terminal_state_rate:")
                print(terminal_rate.to_string())
                print()

    print("Candidates:", len(candidates))
    print("Transition rows:", len(transitions))
    print("Artifact:", args.output)


if __name__ == "__main__":
    main()
