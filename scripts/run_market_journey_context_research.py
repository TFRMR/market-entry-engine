"""Context-conditioned empirical Market Journey research.

Consumes the Market Journey episode dataset and reconstructs two compact,
point-in-time context dimensions per candidate:
- BOS scope (INTERNAL / EXTERNAL)
- directional FVG relation at the 50% retracement entry

This is descriptive research. No signal, ranking, score, or winner is produced.
Small groups are retained in the CSV but excluded from printed summaries.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_engine.data import load_mt5_csv
from market_engine.poi import POIType, build_poi_records
from market_engine.order_block import find_order_block_candidates
from market_engine.structure import (
    Direction,
    build_structural_sequence,
    process_structural_candles_with_context,
)

HORIZON = 40
MIN_GROUP_N = 30


def build_context(frame: pd.DataFrame) -> pd.DataFrame:
    structural = build_structural_sequence(frame)
    swings, events, _ = process_structural_candles_with_context(structural)
    order_blocks = find_order_block_candidates(swings, events)
    pois = build_poi_records(frame, swings, events, order_blocks)
    fvgs = [p for p in pois if p.poi_type is POIType.FVG]

    rows = []
    used = set()

    for event in events:
        if event.event not in {"BULLISH_BOS", "BEARISH_BOS"}:
            continue

        direction = Direction.UP if event.event == "BULLISH_BOS" else Direction.DOWN
        wanted = (
            __import__("market_engine.structure", fromlist=["SwingType"]).SwingType.LOW
            if direction is Direction.UP
            else __import__("market_engine.structure", fromlist=["SwingType"]).SwingType.HIGH
        )
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
            confirmation - 0.50 * leg
            if direction is Direction.UP
            else confirmation + 0.50 * leg
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

        directional = [
            fvg for fvg in fvgs
            if fvg.direction is direction
            and fvg.created_index <= entry_index
        ]
        latest_fvg = max(directional, key=lambda item: item.created_index) if directional else None

        if latest_fvg is None:
            fvg_relation = "NONE"
            fvg_age = None
        else:
            fvg_age = entry_index - latest_fvg.created_index
            if latest_fvg.low <= entry <= latest_fvg.high:
                fvg_relation = "INSIDE"
            elif direction is Direction.UP:
                fvg_relation = "ABOVE" if entry > latest_fvg.high else "BELOW"
            else:
                fvg_relation = "BELOW" if entry < latest_fvg.low else "ABOVE"

        rows.append({
            "candidate_id": len(rows),
            "bos_index": event.index,
            "bos_scope": event.scope.value,
            "direction": direction.value,
            "confirmation_index": cidx,
            "entry_index": entry_index,
            "fvg_relation_at_entry": fvg_relation,
            "fvg_age_bars": fvg_age,
        })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument(
        "--episodes",
        type=Path,
        default=Path("data/research/xauusd_m30_market_journey_episode.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/research/xauusd_m30_market_journey_context_report.csv"),
    )
    args = parser.parse_args()

    frame = load_mt5_csv(args.csv)
    episodes = pd.read_csv(args.episodes)
    context = build_context(frame)

    if context.empty:
        print("No context candidates.")
        return

    boundary = pd.Timestamp("2026-03-18")
    context["period"] = context["confirmation_index"].map(
        lambda _: "unknown"
    )

    # Use confirmation timestamps from the source frame to preserve the same
    # development/OOS boundary used by the journey dataset.
    context["confirmation_timestamp"] = context["confirmation_index"].map(
        lambda i: frame.iloc[int(i)]["timestamp"]
    )
    context["period"] = context["confirmation_timestamp"].map(
        lambda ts: "development" if pd.Timestamp(ts) < boundary else "historical_oos"
    )

    episode = episodes[
        (episodes["horizon"] == HORIZON)
        & episodes["next_state"].notna()
        & episodes["state"].isin(["ENTRY", "SWING_UPDATE", "CONTINUATION", "TRANSITION"])
    ].copy()

    episode = episode.merge(
        context[
            [
                "candidate_id",
                "bos_scope",
                "fvg_relation_at_entry",
                "fvg_age_bars",
                "period",
            ]
        ],
        on=["candidate_id", "period"],
        how="inner",
    )

    rows = []

    def summarize(group, dimension, value):
        for state in ("ENTRY", "SWING_UPDATE"):
            current = group[group["state"] == state]
            if current.empty:
                continue
            observed = current[current["next_state"].notna()]
            if observed.empty:
                continue
            denominator = len(observed)
            for next_state, target in observed.groupby("next_state"):
                rows.append({
                    "period": group["period"].iloc[0],
                    "direction": group["direction"].iloc[0],
                    "horizon": HORIZON,
                    "dimension": dimension,
                    "context": value,
                    "state": state,
                    "next_state": next_state,
                    "n": len(target),
                    "denominator": denominator,
                    "probability": len(target) / denominator,
                    "duration_median": target["bars_to_next_state"].median(),
                    "mfe_r_median": target["mfe_r_during_episode"].median(),
                    "mae_r_median": target["mae_r_during_episode"].median(),
                })

    for dimension in ("bos_scope", "fvg_relation_at_entry"):
        for keys, group in episode.groupby(
            ["period", "direction", dimension],
            dropna=False,
        ):
            if len(group) < MIN_GROUP_N:
                continue
            summarize(group, dimension, keys[2])

    report = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False)

    print("=== Context-Conditioned Market Journey Research ===")
    print("Horizon: H40 | States: ENTRY, SWING_UPDATE")
    print(f"Minimum group size for printed summary: {MIN_GROUP_N}")
    print()

    if report.empty:
        print("No report rows.")
    else:
        for dimension in ("bos_scope", "fvg_relation_at_entry"):
            print(f"--- {dimension} ---")
            subset = report[report["dimension"] == dimension]
            for (period, direction, context), group in subset.groupby(
                ["period", "direction", "context"]
            ):
                print(f"{period} {direction} {context}")
                print(
                    group[
                        [
                            "state",
                            "next_state",
                            "n",
                            "denominator",
                            "probability",
                            "duration_median",
                            "mfe_r_median",
                            "mae_r_median",
                        ]
                    ].to_string(index=False)
                )
                print()

    print("Context candidates:", len(context))
    print("Episode rows used:", len(episode))
    print("Report rows:", len(report))
    print("Artifact:", args.output)


if __name__ == "__main__":
    main()
