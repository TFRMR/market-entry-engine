"""Research simple entry triggers after structure-first BOS continuation.

Scenario:
    BOS
    -> next same-direction valid swing confirmation
    -> compare simple trigger timing:
       1. confirmation close
       2. first later candle closing in continuation direction
       3. first later candle breaking confirmation candle high/low
    -> measure MFE / MAE from trigger entry.

No FVG/OB filter.
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
TRIGGERS = ("CONFIRM_CLOSE", "COLOR_CONTINUATION", "BREAK_CONFIRM")


def candle_color(open_price: float, close_price: float) -> int:
    if close_price > open_price:
        return 1
    if close_price < open_price:
        return -1
    return 0


def build_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    structural = build_structural_sequence(frame)
    swings, events = process_structural_candles(structural)

    rows = []
    used = set()

    for event in events:
        if event.event not in {"BULLISH_BOS", "BEARISH_BOS"}:
            continue

        direction = Direction.UP if event.event == "BULLISH_BOS" else Direction.DOWN
        future = [
            swing
            for swing in swings
            if swing.confirmation_index > event.index
            and (
                (direction is Direction.UP and swing.swing_type is SwingType.LOW)
                or (direction is Direction.DOWN and swing.swing_type is SwingType.HIGH)
            )
        ]
        if not future:
            continue

        swing = min(future, key=lambda item: item.confirmation_index)
        confirmation_index = swing.confirmation_index
        key = (event.index, confirmation_index)
        if key in used:
            continue
        used.add(key)

        entry = float(frame.iloc[confirmation_index]["close"])
        risk = abs(entry - float(swing.price))
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
                "confirmation_high": float(frame.iloc[confirmation_index]["high"]),
                "confirmation_low": float(frame.iloc[confirmation_index]["low"]),
                "invalidation": float(swing.price),
                "risk_price": risk,
            }
        )

    return pd.DataFrame(rows)


def find_trigger(row: pd.Series, frame: pd.DataFrame, trigger: str) -> int | None:
    confirmation = int(row["confirmation_index"])
    direction = row["direction"]

    if trigger == "CONFIRM_CLOSE":
        return confirmation

    for index in range(confirmation + 1, len(frame)):
        candle = frame.iloc[index]
        color = candle_color(float(candle["open"]), float(candle["close"]))

        if trigger == "COLOR_CONTINUATION":
            if (direction == "UP" and color == 1) or (
                direction == "DOWN" and color == -1
            ):
                return index

        if trigger == "BREAK_CONFIRM":
            if (direction == "UP" and float(candle["high"]) > float(row["confirmation_high"])) or (
                direction == "DOWN" and float(candle["low"]) < float(row["confirmation_low"])
            ):
                return index

    return None


def excursion(row: pd.Series, frame: pd.DataFrame, horizon: int) -> tuple[float, float]:
    index = int(row["trigger_index"])
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
        default=Path("data/research/xauusd_m30_structure_entry_research.csv"),
    )
    args = parser.parse_args()

    frame = load_mt5_csv(args.csv)
    candidates = build_candidates(frame)
    if candidates.empty:
        print("No structure continuation candidates.")
        return

    boundary = pd.Timestamp("2026-03-18")
    rows = []

    for _, candidate in candidates.iterrows():
        for trigger in TRIGGERS:
            trigger_index = find_trigger(candidate, frame, trigger)
            if trigger_index is None:
                continue

            entry = float(frame.iloc[trigger_index]["close"])
            risk = float(candidate["risk_price"])
            for horizon in HORIZONS:
                mfe, mae = excursion(
                    pd.Series(
                        {
                            **candidate.to_dict(),
                            "trigger_index": trigger_index,
                            "entry": entry,
                        }
                    ),
                    frame,
                    horizon,
                )
                rows.append(
                    {
                        **candidate.to_dict(),
                        "period": (
                            "development"
                            if pd.Timestamp(candidate["confirmation_timestamp"]) < boundary
                            else "historical_oos"
                        ),
                        "trigger": trigger,
                        "trigger_index": trigger_index,
                        "bars_after_confirmation": trigger_index
                        - int(candidate["confirmation_index"]),
                        "entry": entry,
                        "horizon": horizon,
                        "mfe_r": mfe,
                        "mae_r": mae,
                    }
                )

    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    summary = []
    for period in ("development", "historical_oos"):
        for trigger in TRIGGERS:
            for direction in ("UP", "DOWN"):
                for horizon in HORIZONS:
                    subset = result[
                        (result["period"] == period)
                        & (result["trigger"] == trigger)
                        & (result["direction"] == direction)
                        & (result["horizon"] == horizon)
                    ]
                    if subset.empty:
                        continue
                    summary.append(
                        {
                            "period": period,
                            "trigger": trigger,
                            "direction": direction,
                            "horizon": horizon,
                            "n": len(subset),
                            "bars_median": subset["bars_after_confirmation"].median(),
                            "mfe_median": subset["mfe_r"].median(),
                            "mfe_ge_0.5R": (subset["mfe_r"] >= 0.5).mean(),
                            "mfe_ge_1R": (subset["mfe_r"] >= 1.0).mean(),
                            "mfe_ge_2R": (subset["mfe_r"] >= 2.0).mean(),
                            "mae_median": subset["mae_r"].median(),
                            "mae_le_1R": (subset["mae_r"] <= 1.0).mean(),
                        }
                    )

    print("=== Structure-First Entry Trigger Research ===")
    print("BOS -> valid swing confirmation -> simple trigger")
    print("No FVG / OB filter")
    print()
    print(pd.DataFrame(summary).to_string(index=False))
    print()
    print("Candidates:", len(candidates))
    print("Artifact:", args.output)


if __name__ == "__main__":
    main()
