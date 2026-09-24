"""Simple Trend + Valid Swing + POI MFE/MAE distribution research.

This deliberately avoids fixed TP/SL labels. For each matching setup it
measures the maximum favorable/adverse excursion in R over several horizons.
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
from market_engine.structure import (
    Direction,
    build_structural_sequence,
    process_structural_candles,
)


HORIZONS = (10, 20, 40, 80)
POI_BREAKDOWN_HORIZONS = (40, 80)
POI_NAMES = ("FVG", "OB", "OBIM", "LIQUIDITY", "SR")

POI_COLUMNS = (
    "poi_fvg_present",
    "poi_ob_present",
    "poi_obim_present",
    "poi_liquidity_present",
    "poi_sr_present",
)


def aligned_trend(row: pd.Series) -> bool:
    direction = row["direction"]
    structure_direction = row["structure_direction"]
    return (
        (direction == "UP" and float(structure_direction) > 0)
        or (direction == "DOWN" and float(structure_direction) < 0)
    )


def has_poi(row: pd.Series) -> bool:
    return any(float(row[column]) > 0 for column in POI_COLUMNS)


def excursion_r(candidate, frame: pd.DataFrame, horizon: int) -> tuple[float, float]:
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

    end = min(candidate.setup_index + horizon, len(frame) - 1)
    future = frame.iloc[candidate.setup_index + 1 : end + 1]

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
        if not aligned_trend(row) or not has_poi(row):
            continue

        candidate = candidate_by_key[(int(row["setup_index"]), row["direction"])]

        for horizon in HORIZONS:
            mfe, mae = excursion_r(candidate, frame, horizon)
            if pd.isna(mfe) or pd.isna(mae):
                continue

            poi_names = [
                name
                for name, column in zip(
                    ("FVG", "OB", "OBIM", "LIQUIDITY", "SR"),
                    POI_COLUMNS,
                )
                if float(row[column]) > 0
            ]

            rows.append(
                {
                    "setup_index": candidate.setup_index,
                    "setup_timestamp": candidate.setup_timestamp,
                    "direction": candidate.direction.value,
                    "trend": row["trend_regime"],
                    "poi": "+".join(poi_names),
                    "horizon": horizon,
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
            "mfe_median_r": float("nan"),
            "mfe_p25_r": float("nan"),
            "mfe_p75_r": float("nan"),
            "mae_median_r": float("nan"),
            "mae_p75_r": float("nan"),
            "p_mfe_ge_0_5r": float("nan"),
            "p_mfe_ge_1r": float("nan"),
            "p_mfe_ge_2r": float("nan"),
            "p_mae_le_0_5r": float("nan"),
            "p_mae_le_1r": float("nan"),
        }

    return {
        "scenario": name,
        "sample": len(dataset),
        "mfe_median_r": dataset["mfe_r"].median(),
        "mfe_p25_r": dataset["mfe_r"].quantile(0.25),
        "mfe_p75_r": dataset["mfe_r"].quantile(0.75),
        "mae_median_r": dataset["mae_r"].median(),
        "mae_p75_r": dataset["mae_r"].quantile(0.75),
        "p_mfe_ge_0_5r": (dataset["mfe_r"] >= 0.5).mean(),
        "p_mfe_ge_1r": (dataset["mfe_r"] >= 1.0).mean(),
        "p_mfe_ge_2r": (dataset["mfe_r"] >= 2.0).mean(),
        "p_mae_le_0_5r": (dataset["mae_r"] <= 0.5).mean(),
        "p_mae_le_1r": (dataset["mae_r"] <= 1.0).mean(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/research/xauusd_m30_simple_trend_swing_poi_mfe_mae.csv"),
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

    summary_rows = []
    for horizon in HORIZONS:
        for name, group in (
            ("development", development),
            ("historical_oos", historical),
        ):
            subset = group[group["horizon"] == horizon]
            summary_rows.append(
                summarize(subset, f"{name}_h{horizon}")
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)

    print("=== Simple Trend + Valid Swing + POI / MFE-MAE Distribution ===")
    print(f"Horizons: {', '.join(map(str, HORIZONS))}")
    print(f"Matching setups: {dataset['setup_index'].nunique() if not dataset.empty else 0}")
    print()
    print(pd.DataFrame(summary_rows).to_string(index=False))

    print()
    print("By horizon / direction:")
    direction_rows = []
    for horizon in HORIZONS:
        for direction in ("UP", "DOWN"):
            subset = dataset[
                (dataset["horizon"] == horizon)
                & (dataset["direction"] == direction)
                & (dataset["setup_timestamp"] < "2026-03-18")
            ]
            if not subset.empty:
                direction_rows.append(
                    summarize(subset, f"development_h{horizon}_{direction}")
                )

            subset = dataset[
                (dataset["horizon"] == horizon)
                & (dataset["direction"] == direction)
                & (dataset["setup_timestamp"] >= "2026-03-18")
            ]
            if not subset.empty:
                direction_rows.append(
                    summarize(subset, f"historical_oos_h{horizon}_{direction}")
                )

    print(pd.DataFrame(direction_rows).to_string(index=False))

    print()
    print("POI type breakdown (presence; setups may belong to multiple POI types):")
    poi_rows = []
    for horizon in POI_BREAKDOWN_HORIZONS:
        for poi_name in POI_NAMES:
            for name, group in (
                ("development", development),
                ("historical_oos", historical),
            ):
                subset = group[
                    (group["horizon"] == horizon)
                    & group["poi"].str.split("+").apply(lambda values: poi_name in values)
                ]
                if not subset.empty:
                    poi_rows.append(
                        summarize(subset, f"{name}_h{horizon}_{poi_name}")
                    )

    print(pd.DataFrame(poi_rows).to_string(index=False))
    print()
    print(f"Artifact: {args.output}")


if __name__ == "__main__":
    main()
