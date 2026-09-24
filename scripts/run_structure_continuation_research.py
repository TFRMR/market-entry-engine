"""Research structure-first BOS continuation.

Scenario:
    trend-aligned market structure
    -> bullish/bearish BOS
    -> wait for the next valid swing/pullback confirmation
    -> measure post-confirmation excursion.

No FVG/OB is required. POIs can be layered later as supporting context.
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
    process_structural_candles,
)


HORIZONS = (10, 20, 40, 80)


def build_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    structural = build_structural_sequence(frame)
    swings, events = process_structural_candles(structural)

    rows = []
    used = set()

    for event in events:
        if event.event not in {"BULLISH_BOS", "BEARISH_BOS"}:
            continue

        direction = (
            Direction.UP if event.event == "BULLISH_BOS" else Direction.DOWN
        )

        future_swings = [
            swing
            for swing in swings
            if swing.confirmation_index > event.index
            and (
                (direction is Direction.UP and swing.swing_type is SwingType.LOW)
                or (direction is Direction.DOWN and swing.swing_type is SwingType.HIGH)
            )
        ]
        if not future_swings:
            continue

        swing = min(future_swings, key=lambda item: item.confirmation_index)
        confirmation_index = swing.confirmation_index

        key = (event.index, confirmation_index)
        if key in used:
            continue
        used.add(key)

        entry = float(frame.iloc[confirmation_index]["close"])
        invalidation = float(swing.price)
        risk = abs(entry - invalidation)
        if risk <= 0:
            continue

        rows.append(
            {
                "bos_index": event.index,
                "bos_timestamp": frame.iloc[event.index]["timestamp"],
                "bos_event": event.event,
                "bos_scope": event.scope.value if event.scope else None,
                "direction": direction.value,
                "confirmation_index": confirmation_index,
                "confirmation_timestamp": frame.iloc[confirmation_index]["timestamp"],
                "confirmation_swing_type": swing.swing_type.value,
                "entry": entry,
                "invalidation": invalidation,
                "risk_price": risk,
            }
        )

    return pd.DataFrame(rows)


def excursion(
    row: pd.Series,
    frame: pd.DataFrame,
    horizon: int,
) -> tuple[float, float]:
    index = int(row["confirmation_index"])
    entry = float(row["entry"])
    risk = float(row["risk_price"])

    end = min(index + horizon, len(frame) - 1)
    highs = frame.iloc[index + 1 : end + 1]["high"].astype(float)
    lows = frame.iloc[index + 1 : end + 1]["low"].astype(float)

    if row["direction"] == "UP":
        mfe = (highs.max() - entry) / risk
        mae = (entry - lows.min()) / risk
    else:
        mfe = (entry - lows.min()) / risk
        mae = (highs.max() - entry) / risk

    return float(mfe), float(mae)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/research/xauusd_m30_structure_continuation_research.csv"
        ),
    )
    args = parser.parse_args()

    frame = load_mt5_csv(args.csv)
    dataset = build_dataset(frame)

    if dataset.empty:
        print("No structure continuation candidates.")
        return

    boundary = pd.Timestamp("2026-03-18")
    dataset["period"] = dataset["confirmation_timestamp"].map(
        lambda value: (
            "development" if pd.Timestamp(value) < boundary else "historical_oos"
        )
    )

    rows = []
    for _, row in dataset.iterrows():
        for horizon in HORIZONS:
            mfe, mae = excursion(row, frame, horizon)
            rows.append(
                {
                    **row.to_dict(),
                    "horizon": horizon,
                    "mfe_r": mfe,
                    "mae_r": mae,
                }
            )

    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    summary_rows = []
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

                summary_rows.append(
                    {
                        "period": period,
                        "direction": direction,
                        "horizon": horizon,
                        "n": len(subset),
                        "mfe_median": subset["mfe_r"].median(),
                        "mfe_ge_0.5R": (subset["mfe_r"] >= 0.5).mean(),
                        "mfe_ge_1R": (subset["mfe_r"] >= 1.0).mean(),
                        "mfe_ge_2R": (subset["mfe_r"] >= 2.0).mean(),
                        "mfe_ge_3R": (subset["mfe_r"] >= 3.0).mean(),
                        "mae_median": subset["mae_r"].median(),
                        "mae_le_1R": (subset["mae_r"] <= 1.0).mean(),
                    }
                )

    print("=== Structure-First BOS Continuation Research ===")
    print("Scenario: BOS -> next same-direction valid swing confirmation")
    print("No FVG / OB filter")
    print("Entry: confirmation candle close")
    print("Invalidation: confirmed swing price")
    print()
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print()
    print("Candidates:", len(dataset))
    print("Artifact:", args.output)


if __name__ == "__main__":
    main()
