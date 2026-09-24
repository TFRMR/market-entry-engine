"""Simple Trend + Valid Swing + OB/FVG interaction research.

Measures the first post-setup touch of the latest active OB/FVG aligned with
the setup direction, then measures MFE/MAE after that touch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_engine.context import build_context_dataset
from market_engine.data import load_mt5_csv
from market_engine.entry import build_setup_candidates
from market_engine.execution import execute_entry
from market_engine.features import build_structural_features
from market_engine.order_block import find_order_block_candidates
from market_engine.poi import (
    POIType,
    build_poi_records,
    classify_poi_interaction,
)
from market_engine.structure import (
    Direction,
    build_structural_sequence,
    process_structural_candles,
)


HORIZONS = (5, 10, 20, 40, 80)
POI_TYPES = (POIType.FVG, POIType.ORDER_BLOCK)


def aligned_trend(row: pd.Series) -> bool:
    direction = row["direction"]
    structure_direction = row["structure_direction"]
    return (
        (direction == "UP" and float(structure_direction) > 0)
        or (direction == "DOWN" and float(structure_direction) < 0)
    )


def excursion_after_touch(
    candidate,
    frame: pd.DataFrame,
    touch_index: int,
    horizon: int,
) -> tuple[float, float]:
    execution = execute_entry(
        direction=candidate.direction,
        quoted_price=float(candidate.entry_price),
        spread_price=0.0,
        invalidation_price=float(candidate.invalidation_price),
    )
    entry = float(execution.entry_price)
    risk = float(execution.risk)
    if risk <= 0:
        return float("nan"), float("nan")

    end = min(touch_index + horizon, len(frame) - 1)
    future = frame.iloc[touch_index + 1 : end + 1]
    if future.empty:
        return float("nan"), float("nan")

    if candidate.direction is Direction.UP:
        mfe = (future["high"].max() - entry) / risk
        mae = (entry - future["low"].min()) / risk
    else:
        mfe = (entry - future["low"].min()) / risk
        mae = (future["high"].max() - entry) / risk

    return float(mfe), float(mae)


def build_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    candidates = build_setup_candidates(frame)
    features = build_structural_features(frame)
    structural = build_structural_sequence(frame)
    swings, events = process_structural_candles(structural)
    order_blocks = find_order_block_candidates(swings, events)
    pois = build_poi_records(frame, swings, events, order_blocks)

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
    for _, row in context.iterrows():
        if not aligned_trend(row):
            continue

        candidate = candidate_by_key[(int(row["setup_index"]), row["direction"])]
        setup_index = candidate.setup_index
        direction = candidate.direction

        for poi_type in POI_TYPES:
            name = "FVG" if poi_type is POIType.FVG else "OB"
            present_column = (
                "poi_fvg_present"
                if poi_type is POIType.FVG
                else "poi_ob_present"
            )
            if float(row[present_column]) <= 0:
                continue

            active = [
                poi for poi in pois
                if poi.poi_type is poi_type
                and poi.created_index <= setup_index
                and poi.direction is direction
            ]
            if not active:
                continue

            poi = max(
                active,
                key=lambda item: (item.created_index, item.source_index or -1),
            )

            approach_distance = (
                0.0
                if poi.low <= float(candidate.entry_price) <= poi.high
                else min(
                    abs(float(candidate.entry_price) - poi.low),
                    abs(float(candidate.entry_price) - poi.high),
                )
            )

            touch_index = None
            touch_interaction = None
            for position in range(setup_index + 1, len(frame)):
                interaction = classify_poi_interaction(
                    poi,
                    high=float(frame.iloc[position]["high"]),
                    low=float(frame.iloc[position]["low"]),
                    close=float(frame.iloc[position]["close"]),
                )
                if interaction.value != "NONE":
                    touch_index = position
                    touch_interaction = interaction.value
                    break

            if touch_index is None:
                continue

            bars_to_touch = touch_index - setup_index

            for horizon in HORIZONS:
                mfe, mae = excursion_after_touch(
                    candidate,
                    frame,
                    touch_index,
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
                        "poi_type": name,
                        "poi_created_index": poi.created_index,
                        "poi_age_at_setup": setup_index - poi.created_index,
                        "poi_approach_distance_r": (
                            approach_distance / abs(
                                float(candidate.entry_price)
                                - float(candidate.invalidation_price)
                            )
                            if float(candidate.entry_price)
                            != float(candidate.invalidation_price)
                            else float("nan")
                        ),
                        "touch_index": touch_index,
                        "bars_to_touch": bars_to_touch,
                        "touch_interaction": touch_interaction,
                        "horizon_after_touch": horizon,
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
            "data/research/xauusd_m30_trend_swing_ob_fvg_interaction.csv"
        ),
    )
    args = parser.parse_args()

    frame = load_mt5_csv(args.csv)
    dataset = build_dataset(frame)

    development = dataset[
        dataset["setup_timestamp"] < "2026-03-18"
    ].copy()
    historical = dataset[
        dataset["setup_timestamp"] >= "2026-03-18"
    ].copy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)

    print("=== Trend + Valid Swing + OB/FVG First Touch ===")
    print("Horizons after touch: 5, 10, 20, 40, 80")
    print(
        "Matching setup/POI interactions:",
        dataset[["setup_index", "poi_type"]].drop_duplicates().shape[0]
        if not dataset.empty
        else 0,
    )
    print()

    rows = []
    for horizon in HORIZONS:
        for poi_name in ("OB", "FVG"):
            for period, group in (
                ("development", development),
                ("historical_oos", historical),
            ):
                subset = group[
                    (group["horizon_after_touch"] == horizon)
                    & (group["poi_type"] == poi_name)
                ]
                rows.append(
                    summarize(
                        subset,
                        f"{period}_h{horizon}_{poi_name}",
                    )
                )

    print(pd.DataFrame(rows).to_string(index=False))

    print()
    print("Touch interaction:")
    print(
        dataset.groupby(
            ["poi_type", "touch_interaction"],
            dropna=False,
        ).size().to_string()
        if not dataset.empty
        else "empty"
    )

    print()
    print("Artifact:", args.output)


if __name__ == "__main__":
    main()
