"""Simple trend + valid swing + POI research with fixed 1:2 RR.

The scenario stays intentionally simple:
    Trend + Valid Swing + POI + SL = 1R + TP = 2R

Only the holding horizon is swept so we can distinguish a target-distance
problem from a scenario problem.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_engine.context import build_context_dataset
from market_engine.data import load_mt5_csv
from market_engine.entry import build_setup_candidates
from market_engine.execution import execute_entry
from market_engine.holdout import HISTORICAL_AUDIT_CUTOFF
from market_engine.outcome import TradeOutcome, evaluate_trade
from market_engine.structure import (
    Direction,
    build_structural_sequence,
    process_structural_candles,
)
from market_engine.features import build_structural_features
from market_engine.exits import ExitArea


HORIZONS = (10, 20, 40, 80)
REWARD_RISK = 2.0

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


def evaluate_fixed_rr(candidate, frame: pd.DataFrame, horizon: int) -> TradeOutcome:
    execution = execute_entry(
        direction=candidate.direction,
        quoted_price=float(candidate.entry_price),
        spread_price=0.0,
        invalidation_price=float(candidate.invalidation_price),
    )

    if candidate.direction is Direction.UP:
        target_price = execution.entry_price + REWARD_RISK * execution.risk
    else:
        target_price = execution.entry_price - REWARD_RISK * execution.risk

    target = ExitArea(
        source_type="FIXED_RR_2R",
        source_index=candidate.setup_index,
        source_confirmation_index=candidate.setup_index,
        price=float(target_price),
        distance=float(abs(target_price - execution.entry_price)),
    )

    return evaluate_trade(
        candidate=candidate,
        execution=execution,
        target=target,
        frame=frame,
        horizon=horizon,
    )


def build_simple_dataset(frame: pd.DataFrame) -> pd.DataFrame:
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

    rows = []
    candidate_by_key = {
        (candidate.setup_index, candidate.direction.value): candidate
        for candidate in candidates
    }

    for _, row in context.iterrows():
        if not aligned_trend(row) or not has_poi(row):
            continue

        candidate = candidate_by_key[(int(row["setup_index"]), row["direction"])]

        for horizon in HORIZONS:
            if candidate.setup_index + horizon >= len(frame):
                continue

            outcome = evaluate_fixed_rr(candidate, frame, horizon)
            if outcome.status == "TARGET":
                label = "TP_2R"
            elif outcome.status == "STOP":
                label = "SL_1R"
            else:
                label = "UNRESOLVED"

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
                    "entry_price": float(candidate.entry_price),
                    "invalidation_price": float(candidate.invalidation_price),
                    "risk": float(abs(candidate.entry_price - candidate.invalidation_price)),
                    "target_price": float(outcome.target_price),
                    "label": label,
                    "ambiguous_barrier": outcome.both_hit,
                }
            )

    return pd.DataFrame(rows)


def summarize(dataset: pd.DataFrame, name: str) -> pd.DataFrame:
    if dataset.empty:
        return pd.DataFrame(
            [{
                "scenario": name,
                "sample": 0,
                "tp_2r": 0,
                "sl_1r": 0,
                "unresolved": 0,
                "ambiguous": 0,
                "tp_rate": float("nan"),
                "sl_rate": float("nan"),
                "expectancy_r": float("nan"),
            }]
        )

    tp = int((dataset["label"] == "TP_2R").sum())
    sl = int((dataset["label"] == "SL_1R").sum())
    unresolved = int((dataset["label"] == "UNRESOLVED").sum())
    ambiguous = int(dataset["ambiguous_barrier"].sum())
    sample = len(dataset)

    # Unresolved contributes 0R for this simple research comparison.
    expectancy = (tp * REWARD_RISK - sl) / sample

    return pd.DataFrame(
        [{
            "scenario": name,
            "sample": sample,
            "tp_2r": tp,
            "sl_1r": sl,
            "unresolved": unresolved,
            "ambiguous": ambiguous,
            "tp_rate": tp / sample,
            "sl_rate": sl / sample,
            "expectancy_r": expectancy,
        }]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/research/xauusd_m30_simple_trend_swing_poi_2r_horizons.csv"),
    )
    args = parser.parse_args()

    frame = load_mt5_csv(args.csv)
    dataset = build_simple_dataset(frame)

    development = dataset[
        dataset["setup_timestamp"] < HISTORICAL_AUDIT_CUTOFF
    ].copy()
    historical = dataset[
        dataset["setup_timestamp"] >= HISTORICAL_AUDIT_CUTOFF
    ].copy()

    summary_frames = []
    for horizon in HORIZONS:
        summary_frames.append(
            summarize(
                development[development["horizon"] == horizon],
                f"development_h{horizon}",
            )
        )
        summary_frames.append(
            summarize(
                historical[historical["horizon"] == horizon],
                f"historical_oos_h{horizon}",
            )
        )

    summary = pd.concat(summary_frames, ignore_index=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)

    print("=== Simple Trend + Valid Swing + POI / RR 1:2 / Horizon Sweep ===")
    print(f"Horizons: {', '.join(map(str, HORIZONS))}")
    print(f"Matching setups: {dataset['setup_index'].nunique() if not dataset.empty else 0}")
    print()
    print(summary.to_string(index=False))

    print()
    print("By horizon / direction:")
    if dataset.empty:
        print("No matching setups.")
    else:
        direction_frames = []
        for (horizon in HORIZONS):
            for direction in ("UP", "DOWN"):
                group = dataset[
                    (dataset["horizon"] == horizon)
                    & (dataset["direction"] == direction)
                ]
                if not group.empty:
                    direction_frames.append(
                        summarize(group, f"h{horizon}_{direction}")
                    )

        if direction_frames:
            print(pd.concat(direction_frames, ignore_index=True).to_string(index=False))
        else:
            print("No matching direction groups.")

    print()
    print(f"Artifact: {args.output}")


if __name__ == "__main__":
    main()