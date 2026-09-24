"""Research FVG-first trigger and simple economic outcome.

Scenario:
    Trend confirmed
    -> valid swing / BOS setup
    -> directional FVG
    -> first touch
    -> first later candle closing in the opposite direction to the touch candle
       becomes the trigger
    -> entry at trigger close
    -> SL = 1R from trigger to structural invalidation
    -> TP = 2R
    -> evaluate first barrier within 40/80 M30 candles.

The FVG candidate scope is intentionally loose: it does not require the FVG
to sit between a specific pair of confirmed swings.

No engulfing or body-size threshold is used.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_engine.context import build_context_dataset
from market_engine.data import load_mt5_csv
from market_engine.entry import build_setup_candidates
from market_engine.features import build_structural_features
from market_engine.poi import (
    POIRecord,
    POIType,
    PriceInteraction,
    build_poi_records,
    classify_poi_interaction,
)
from market_engine.structure import (
    Direction,
    build_structural_sequence,
    process_structural_candles,
)


HORIZONS = (40, 80)
TP_R = 2.0
SL_R = 1.0


def candle_color(open_price: float, close_price: float) -> int:
    if close_price > open_price:
        return 1
    if close_price < open_price:
        return -1
    return 0


def fvg_candidates(
    pois: list[POIRecord],
    setup_index: int,
    direction: Direction,
) -> list[POIRecord]:
    return [
        poi
        for poi in pois
        if poi.poi_type is POIType.FVG
        and poi.direction is direction
        and poi.created_index <= setup_index
    ]


def find_trigger(
    frame: pd.DataFrame,
    poi: POIRecord,
    setup_index: int,
) -> tuple[int | None, str | None, int | None]:
    touch_index = None
    touch_color = 0

    for position in range(setup_index + 1, len(frame)):
        candle = frame.iloc[position]
        interaction = classify_poi_interaction(
            poi,
            high=float(candle["high"]),
            low=float(candle["low"]),
            close=float(candle["close"]),
        )

        if interaction is PriceInteraction.NONE:
            if touch_index is None:
                continue
        elif touch_index is None:
            touch_index = position
            touch_color = candle_color(
                float(candle["open"]),
                float(candle["close"]),
            )
            if touch_color == 0:
                continue

        if touch_index is None:
            continue

        if (
            position > touch_index
            and (
                float(candle["close"]) < poi.low
                or float(candle["close"]) > poi.high
            )
        ):
            return None, "INVALIDATED", touch_index

        current_color = candle_color(
            float(candle["open"]),
            float(candle["close"]),
        )
        if (
            position > touch_index
            and current_color != 0
            and current_color != touch_color
        ):
            return position, "TRIGGERED", touch_index

    return None, "NO_TRIGGER", touch_index


def build_trigger_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    candidates = build_setup_candidates(frame)
    features = build_structural_features(frame)
    structural = build_structural_sequence(frame)
    swings, events = process_structural_candles(structural)
    pois = build_poi_records(frame, swings, events, [])

    context = build_context_dataset(
        frame,
        candidates,
        features,
        swings=swings,
        events=events,
    )

    candidate_by_key = {
        (candidate.setup_index, candidate.direction.value): candidate
        for candidate in candidates
    }

    rows = []
    seen = set()

    for _, row in context.iterrows():
        aligned = (
            (row["direction"] == "UP" and float(row["structure_direction"]) > 0)
            or (row["direction"] == "DOWN" and float(row["structure_direction"]) < 0)
        )
        if not aligned:
            continue

        candidate = candidate_by_key[(int(row["setup_index"]), row["direction"])]
        setup_index = candidate.setup_index
        direction = candidate.direction

        fvgs = fvg_candidates(pois, setup_index, direction)
        if not fvgs:
            continue

        poi = max(
            fvgs,
            key=lambda item: (item.created_index, item.source_index or -1),
        )

        trigger_index, status, touch_index = find_trigger(
            frame,
            poi,
            setup_index,
        )
        if trigger_index is None:
            continue

        key = (setup_index, poi.created_index)
        if key in seen:
            continue
        seen.add(key)

        trigger_entry = float(frame.iloc[trigger_index]["close"])
        risk = abs(trigger_entry - float(candidate.invalidation_price))
        if risk <= 0:
            continue

        rows.append(
            {
                "setup_index": setup_index,
                "setup_timestamp": candidate.setup_timestamp,
                "direction": direction.value,
                "trend": row["trend_regime"],
                "fvg_created_index": poi.created_index,
                "fvg_age_at_setup": setup_index - poi.created_index,
                "touch_index": touch_index,
                "trigger_index": trigger_index,
                "bars_to_touch": touch_index - setup_index,
                "bars_touch_to_trigger": trigger_index - touch_index,
                "trigger_entry": trigger_entry,
                "invalidation_price": float(candidate.invalidation_price),
                "risk_price": risk,
            }
        )

    return pd.DataFrame(rows)


def evaluate_trade(
    row: pd.Series,
    frame: pd.DataFrame,
    horizon: int,
) -> tuple[str, float | None, int | None]:
    trigger_index = int(row["trigger_index"])
    entry = float(row["trigger_entry"])
    risk = float(row["risk_price"])
    direction = row["direction"]

    if direction == "UP":
        tp = entry + TP_R * risk
        sl = entry - SL_R * risk
    else:
        tp = entry - TP_R * risk
        sl = entry + SL_R * risk

    end = min(trigger_index + horizon, len(frame) - 1)
    for position in range(trigger_index + 1, end + 1):
        candle = frame.iloc[position]
        high = float(candle["high"])
        low = float(candle["low"])

        if direction == "UP":
            hit_tp = high >= tp
            hit_sl = low <= sl
        else:
            hit_tp = low <= tp
            hit_sl = high >= sl

        if hit_tp and hit_sl:
            return "AMBIGUOUS", None, position
        if hit_tp:
            return "TP_FIRST", TP_R, position
        if hit_sl:
            return "SL_FIRST", -SL_R, position

    return "UNRESOLVED", 0.0, end


def summarize_economic(
    dataset: pd.DataFrame,
    frame: pd.DataFrame,
    period: str,
    horizon: int,
) -> dict:
    subset = dataset[
        (dataset["period"] == period)
        & (dataset["horizon"] == horizon)
    ].copy()

    n = len(subset)
    if n == 0:
        return {
            "period": period,
            "horizon": horizon,
            "n": 0,
            "TP_FIRST": 0,
            "SL_FIRST": 0,
            "AMBIGUOUS": 0,
            "UNRESOLVED": 0,
            "TP_pct": 0.0,
            "SL_pct": 0.0,
            "resolved_pct": 0.0,
            "expected_r_per_entry": 0.0,
        }

    counts = subset["outcome"].value_counts()
    tp = int(counts.get("TP_FIRST", 0))
    sl = int(counts.get("SL_FIRST", 0))
    ambiguous = int(counts.get("AMBIGUOUS", 0))
    unresolved = int(counts.get("UNRESOLVED", 0))

    # Ambiguous trades are excluded from the economic expectation rather than
    # assigning an arbitrary winner.
    resolved = tp + sl
    net_r = tp * TP_R - sl * SL_R
    return {
        "period": period,
        "horizon": horizon,
        "n": n,
        "TP_FIRST": tp,
        "SL_FIRST": sl,
        "AMBIGUOUS": ambiguous,
        "UNRESOLVED": unresolved,
        "TP_pct": tp / n,
        "SL_pct": sl / n,
        "resolved_pct": resolved / n,
        "expected_r_per_entry": net_r / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/research/xauusd_m30_fvg_trigger_economic.csv"
        ),
    )
    args = parser.parse_args()

    frame = load_mt5_csv(args.csv)
    triggers = build_trigger_dataset(frame)

    if triggers.empty:
        print("No matching FVG triggers.")
        return

    boundary = pd.Timestamp("2026-03-18")
    triggers["period"] = triggers["setup_timestamp"].map(
        lambda value: "development" if pd.Timestamp(value) < boundary else "historical_oos"
    )

    rows = []
    for _, trigger in triggers.iterrows():
        for horizon in HORIZONS:
            outcome, realized_r, exit_index = evaluate_trade(
                trigger,
                frame,
                horizon,
            )
            rows.append(
                {
                    **trigger.to_dict(),
                    "horizon": horizon,
                    "outcome": outcome,
                    "realized_r": realized_r,
                    "exit_index": exit_index,
                }
            )

    dataset = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)

    print("=== Simple FVG Trigger Economic Setup ===")
    print("Entry: trigger close")
    print("SL: 1R | TP: 2R")
    print("Horizon: 40 / 80 M30 candles")
    print("Matching triggers:", len(triggers))
    print()
    print(
        pd.DataFrame(
            [
                summarize_economic(dataset, frame, period, horizon)
                for period in ("development", "historical_oos")
                for horizon in HORIZONS
            ]
        ).to_string(index=False)
    )

    print()
    print("Direction breakdown:")
    direction_rows = []
    for period in ("development", "historical_oos"):
        for direction in ("UP", "DOWN"):
            for horizon in HORIZONS:
                subset = dataset[
                    (dataset["period"] == period)
                    & (dataset["direction"] == direction)
                    & (dataset["horizon"] == horizon)
                ]
                summary = summarize_economic(
                    subset,
                    frame,
                    period,
                    horizon,
                )
                summary["direction"] = direction
                direction_rows.append(summary)

    print(pd.DataFrame(direction_rows).to_string(index=False))
    print()
    print("Artifact:", args.output)


if __name__ == "__main__":
    main()
