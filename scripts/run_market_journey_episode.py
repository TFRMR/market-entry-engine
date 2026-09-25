"""Research contiguous semantic episodes in the reconstructed market journey.

Scenario:
BOS -> next same-direction valid swing -> 50% retracement entry.

Episodes preserve raw structural events while grouping consecutive events that
share the same direction-aware semantic state. This is descriptive empirical
research; no signal or ranking is produced.
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

    continuation = "BULLISH_BOS" if direction == "UP" else "BEARISH_BOS"
    if event == continuation:
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


def excursion(frame, entry_index, stop_index, direction, entry, risk):
    mfe = 0.0
    mae = 0.0
    for index in range(entry_index + 1, stop_index + 1):
        candle = frame.iloc[index]
        high = float(candle["high"])
        low = float(candle["low"])
        if direction == "UP":
            mfe = max(mfe, high - entry)
            mae = max(mae, entry - low)
        else:
            mfe = max(mfe, entry - low)
            mae = max(mae, high - entry)
    return mfe / risk, mae / risk


def build_episodes(candidate, frame, events, horizon):
    entry_index = int(candidate["entry_index"])
    end = min(entry_index + horizon, len(frame) - 1)
    direction = candidate["direction"]
    entry = float(candidate["entry"])
    risk = float(candidate["risk"])
    invalidation = float(candidate["invalidation"])

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

        for event in (e for e in events if e.index == index and e.index > entry_index):
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

    # Group consecutive raw events with the same semantic state.
    episodes = []
    current = None

    for point in points:
        state = event_state(point["event"], direction)
        if current is None or state != current["state"]:
            if current is not None:
                episodes.append(current)
            current = {
                "state": state,
                "start_bar": point["bar"],
                "start_event": point["event"],
                "start_scope": point["scope"],
                "start_index": point["index"],
                "end_bar": point["bar"],
                "end_event": point["event"],
                "end_scope": point["scope"],
                "end_index": point["index"],
                "raw_events": [point["event"]],
            }
        else:
            current["end_bar"] = point["bar"]
            current["end_event"] = point["event"]
            current["end_scope"] = point["scope"]
            current["end_index"] = point["index"]
            current["raw_events"].append(point["event"])

    if current is not None:
        episodes.append(current)

    rows = []
    for episode_index, episode in enumerate(episodes):
        next_episode = episodes[episode_index + 1] if episode_index + 1 < len(episodes) else None

        state_start = entry_index + int(episode["start_bar"])
        state_end = entry_index + int(
            next_episode["start_bar"] if next_episode is not None else min(episode["end_bar"] + 1, horizon)
        )

        if next_episode is None:
            state_end = min(entry_index + horizon, len(frame) - 1)

        mfe_r, mae_r = excursion(
            frame,
            state_start,
            state_end,
            direction,
            entry,
            risk,
        )

        duration = (
            next_episode["start_bar"] - episode["start_bar"]
            if next_episode is not None
            else state_end - episode["start_bar"]
        )

        rows.append({
            "episode_index": episode_index,
            "state": episode["state"],
            "start_bar": episode["start_bar"],
            "end_bar": episode["end_bar"],
            "duration_bars": duration,
            "start_event": episode["start_event"],
            "end_event": episode["end_event"],
            "start_scope": episode["start_scope"],
            "end_scope": episode["end_scope"],
            "raw_event_count": len(episode["raw_events"]),
            "raw_event_sequence": ">".join(episode["raw_events"]),
            "next_state": next_episode["state"] if next_episode is not None else None,
            "next_event": next_episode["start_event"] if next_episode is not None else None,
            "next_scope": next_episode["start_scope"] if next_episode is not None else None,
            "bars_to_next_state": (
                next_episode["start_bar"] - episode["start_bar"]
                if next_episode is not None
                else None
            ),
            "mfe_r_during_episode": mfe_r,
            "mae_r_during_episode": mae_r,
            "terminal_reason": (
                "HORIZON_END"
                if next_episode is None and episode["state"] != "INVALIDATED"
                else "INVALIDATION"
                if next_episode is None
                else None
            ),
        })

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/research/xauusd_m30_market_journey_episode.csv"),
    )
    args = parser.parse_args()

    frame = load_mt5_csv(args.csv)
    candidates, events = build_candidates(frame)
    if candidates.empty:
        print("No candidates.")
        return

    boundary = pd.Timestamp("2026-03-18")
    rows = []

    for candidate_id, (_, candidate) in enumerate(candidates.iterrows()):
        period = (
            "development"
            if pd.Timestamp(candidate["confirmation_timestamp"]) < boundary
            else "historical_oos"
        )

        for horizon in HORIZONS:
            episodes = build_episodes(candidate, frame, events, horizon)
            for episode in episodes:
                rows.append({
                    "candidate_id": candidate_id,
                    "direction": candidate["direction"],
                    "period": period,
                    "horizon": horizon,
                    **episode,
                })

    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

    print("=== Market Journey Episode Research ===")
    print("Contiguous direction-aware semantic states; raw events preserved")
    print()

    for period in ("development", "historical_oos"):
        for direction in ("UP", "DOWN"):
            for horizon in HORIZONS:
                subset = output[
                    (output["period"] == period)
                    & (output["direction"] == direction)
                    & (output["horizon"] == horizon)
                ]
                if subset.empty:
                    continue

                print(
                    f"--- {period} {direction} H{horizon} "
                    f"candidates={subset['candidate_id'].nunique()} "
                    f"episodes={len(subset)} ---"
                )

                state_stats = (
                    subset.groupby("state")
                    .agg(
                        episodes=("state", "size"),
                        duration_median=("duration_bars", "median"),
                        duration_p25=("duration_bars", lambda s: s.quantile(0.25)),
                        duration_p75=("duration_bars", lambda s: s.quantile(0.75)),
                        mfe_r_median=("mfe_r_during_episode", "median"),
                        mae_r_median=("mae_r_during_episode", "median"),
                    )
                    .reset_index()
                )
                print(state_stats.to_string(index=False))

                transitions = subset[subset["next_state"].notna()]
                if not transitions.empty:
                    transition_counts = (
                        transitions.groupby(["state", "next_state"])
                        .size()
                        .rename("episodes")
                        .reset_index()
                    )
                    totals = (
                        transition_counts.groupby("state")["episodes"]
                        .sum()
                        .rename("observed_next_state_total")
                        .reset_index()
                    )
                    transition_counts = transition_counts.merge(totals, on="state")
                    transition_counts["conditional_rate"] = (
                        transition_counts["episodes"]
                        / transition_counts["observed_next_state_total"]
                    )
                    print("next_state:")
                    print(
                        transition_counts.sort_values(
                            ["state", "episodes"],
                            ascending=[True, False],
                        ).to_string(index=False)
                    )

                terminal = subset[subset["next_state"].isna()]
                if not terminal.empty:
                    print("terminal_reason:")
                    print(
                        terminal["terminal_reason"]
                        .value_counts(normalize=True)
                        .sort_index()
                        .to_string()
                    )
                print()

    print("Candidates:", len(candidates))
    print("Episode rows:", len(output))
    print("Artifact:", args.output)


if __name__ == "__main__":
    main()
