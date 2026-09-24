"""Research FVG-first trigger: touch, opposite-color close, then MFE/MAE.

Scenario:
    Trend confirmed
    -> valid swing / BOS setup
    -> FVG created inside the current swing leg
    -> first touch
    -> wait while FVG remains valid
    -> first later candle closing in the opposite direction to the touch candle
       becomes the trigger
    -> measure post-trigger MFE/MAE.

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
    build_poi_records,
    classify_poi_interaction,
    PriceInteraction,
)
from market_engine.structure import (
    Direction,
    SwingType,
    build_structural_sequence,
    process_structural_candles,
)


HORIZONS = (5, 10, 20, 40, 80)


def candle_color(open_price: float, close_price: float) -> int:
    if close_price > open_price:
        return 1
    if close_price < open_price:
        return -1
    return 0


def excursion_after_trigger(
    candidate,
    frame: pd.DataFrame,
    trigger_index: int,
    horizon: int,
) -> tuple[float, float]:
    entry = float(frame.iloc[trigger_index]["close"])
    risk = abs(float(candidate.entry_price) - float(candidate.invalidation_price))
    if risk <= 0:
        return float("nan"), float("nan")

    end = min(trigger_index + horizon, len(frame) - 1)
    future = frame.iloc[trigger_index + 1 : end + 1]
    if future.empty:
        return float("nan"), float("nan")

    if candidate.direction is Direction.UP:
        mfe = (future["high"].max() - entry) / risk
        mae = (entry - future["low"].min()) / risk
    else:
        mfe = (entry - future["low"].min()) / risk
        mae = (future["high"].max() - entry) / risk

    return float(mfe), float(mae)


def fvg_in_current_leg(
    pois: list[POIRecord],
    swings,
    setup_index: int,
    direction: Direction,
) -> list[POIRecord]:
    if direction is Direction.UP:
        start_type = SwingType.LOW
        end_type = SwingType.HIGH
    else:
        start_type = SwingType.HIGH
        end_type = SwingType.LOW

    starts = [
        s for s in swings
        if s.swing_type is start_type
        and s.confirmation_index < setup_index
    ]
    ends = [
        s for s in swings
        if s.swing_type is end_type
        and s.confirmation_index < setup_index
    ]
    if not starts or not ends:
        return []

    start = max(starts, key=lambda s: s.confirmation_index)
    end = max(ends, key=lambda s: s.confirmation_index)
    if start.confirmation_index >= end.confirmation_index:
        return []

    return [
        poi for poi in pois
        if poi.poi_type is POIType.FVG
        and poi.direction is direction
        and start.confirmation_index <= poi.created_index <= end.confirmation_index
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
            # After touch, a later candle may trigger even if it no longer
            # overlaps the FVG. Validity is checked separately below.
        else:
            if touch_index is None:
                touch_index = position
                touch_color = candle_color(
                    float(candle["open"]),
                    float(candle["close"]),
                )
                if touch_color == 0:
                    continue

        if touch_index is None:
            continue

        # A close outside the FVG invalidates the waiting state before trigger.
        if float(candle["close"]) < poi.low or float(candle["close"]) > poi.high:
            # A first-touch candle itself may close outside after a wick sweep;
            # keep the FVG alive unless the next waiting candle closes outside.
            if position > touch_index:
                return None, "INVALIDATED", touch_index

        current_color = candle_color(
            float(candle["open"]),
            float(candle["close"]),
        )
        if position > touch_index and current_color != 0 and current_color != touch_color:
            return position, "TRIGGERED", touch_index

    return None, "NO_TRIGGER", touch_index


def build_dataset(frame: pd.DataFrame) -> pd.DataFrame:
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
        if not (
            (row["direction"] == "UP" and float(row["structure_direction"]) > 0)
            or (row["direction"] == "DOWN" and float(row["structure_direction"]) < 0)
        ):
            continue

        candidate = candidate_by_key[(int(row["setup_index"]), row["direction"])]
        setup_index = candidate.setup_index
        direction = candidate.direction

        fvgs = fvg_in_current_leg(pois, swings, setup_index, direction)
        if not fvgs:
            continue

        # Use the most recent FVG in the current swing leg.
        poi = max(fvgs, key=lambda item: (item.created_index, item.source_index or -1))

        trigger_index, status, touch_index = find_trigger(
            frame, poi, setup_index
        )
        if trigger_index is None:
            continue

        key = (setup_index, poi.created_index)
        if key in seen:
            continue
        seen.add(key)

        touch_color = candle_color(
            float(frame.iloc[touch_index]["open"]),
            float(frame.iloc[touch_index]["close"]),
        )
        trigger_color = candle_color(
            float(frame.iloc[trigger_index]["open"]),
            float(frame.iloc[trigger_index]["close"]),
        )

        bars_to_touch = touch_index - setup_index
        bars_touch_to_trigger = trigger_index - touch_index

        for horizon in HORIZONS:
            mfe, mae = excursion_after_trigger(
                candidate,
                frame,
                trigger_index,
                horizon,
            )
            if pd.isna(mfe) or pd.isna(mae):
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
                    "bars_to_touch": bars_to_touch,
                    "bars_touch_to_trigger": bars_touch_to_trigger,
                    "touch_color": touch_color,
                    "trigger_color": trigger_color,
                    "trigger_status": status,
                    "horizon_after_trigger": horizon,
                    "mfe_r": mfe,
                    "mae_r": mae,
                }
            )

    return pd.DataFrame(rows)


def summarize(dataset: pd.DataFrame, name: str) -> dict:
    if dataset.empty:
        return {
            "scenario": name,
            "sample": 0,
            "bars_to_touch_median": float("nan"),
            "bars_touch_to_trigger_median": float("nan"),
            "mfe_median_r": float("nan"),
            "mfe_p25_r": float("nan"),
            "mfe_p75_r": float("nan"),
            "mae_median_r": float("nan"),
            "mae_p75_r": float("nan"),
            "p_mfe_ge_1r": float("nan"),
            "p_mfe_ge_2r": float("nan"),
        }

    return {
        "scenario": name,
        "sample": len(dataset),
        "bars_to_touch_median": dataset["bars_to_touch"].median(),
        "bars_touch_to_trigger_median": dataset["bars_touch_to_trigger"].median(),
        "mfe_median_r": dataset["mfe_r"].median(),
        "mfe_p25_r": dataset["mfe_r"].quantile(0.25),
        "mfe_p75_r": dataset["mfe_r"].quantile(0.75),
        "mae_median_r": dataset["mae_r"].median(),
        "mae_p75_r": dataset["mae_r"].quantile(0.75),
        "p_mfe_ge_1r": (dataset["mfe_r"] >= 1.0).mean(),
        "p_mfe_ge_2r": (dataset["mfe_r"] >= 2.0).mean(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/research/xauusd_m30_fvg_trigger_research.csv"
        ),
    )
    args = parser.parse_args()

    frame = load_mt5_csv(args.csv)
    dataset = build_dataset(frame)

    development = dataset[dataset["setup_timestamp"] < "2026-03-18"].copy()
    historical = dataset[dataset["setup_timestamp"] >= "2026-03-18"].copy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)

    print("=== Trend + Valid Swing + FVG First Touch -> Opposite Color Trigger ===")
    print("Horizons after trigger: 5, 10, 20, 40, 80")
    print("Matching FVG triggers:", dataset["setup_index"].nunique() if not dataset.empty else 0)
    print()

    rows = []
    for horizon in HORIZONS:
        for period, group in (
            ("development", development),
            ("historical_oos", historical),
        ):
            subset = group[group["horizon_after_trigger"] == horizon]
            rows.append(summarize(subset, f"{period}_h{horizon}"))

    print(pd.DataFrame(rows).to_string(index=False))

    if not dataset.empty:
        print()
        print("Trigger timing:")
        print(
            dataset[["setup_index", "bars_to_touch", "bars_touch_to_trigger"]]
            .drop_duplicates("setup_index")
            .describe()
            .to_string()
        )

    print()
    print("Artifact:", args.output)


if __name__ == "__main__":
    main()
