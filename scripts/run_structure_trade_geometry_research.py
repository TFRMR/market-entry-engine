"""Research structural trade geometry after BOS + pullback.

Measures price-space geometry instead of normalizing everything immediately
by a moving R denominator.

Scenario:
    BOS -> same-direction valid swing confirmation -> 50% retracement entry.

For each entry measure:
- entry to confirmed swing invalidation
- entry to BOS close
- entry to confirmation close
- continuation excursion in price
- adverse excursion in price
- structural target distance

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
RETRACE = 0.50


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
            swing for swing in swings
            if swing.confirmation_index > event.index
            and (
                (direction is Direction.UP and swing.swing_type is SwingType.LOW)
                or (direction is Direction.DOWN and swing.swing_type is SwingType.HIGH)
            )
        ]
        if not future:
            continue

        swing = min(future, key=lambda item: item.confirmation_index)
        cidx = swing.confirmation_index
        key = (event.index, cidx)
        if key in used:
            continue
        used.add(key)

        bos_close = float(frame.iloc[event.index]["close"])
        confirmation = float(frame.iloc[cidx]["close"])
        swing_price = float(swing.price)

        leg = (
            confirmation - swing_price
            if direction is Direction.UP
            else swing_price - confirmation
        )
        if leg <= 0:
            continue

        entry_target = (
            confirmation - RETRACE * leg
            if direction is Direction.UP
            else confirmation + RETRACE * leg
        )

        entry_index = None
        for index in range(cidx + 1, len(frame)):
            candle = frame.iloc[index]
            if direction is Direction.UP and float(candle["low"]) <= entry_target:
                entry_index = index
                break
            if direction is Direction.DOWN and float(candle["high"]) >= entry_target:
                entry_index = index
                break

        if entry_index is None:
            continue

        entry = entry_target
        risk = abs(entry - swing_price)
        if risk <= 0:
            continue

        rows.append({
            "bos_index": event.index,
            "bos_timestamp": frame.iloc[event.index]["timestamp"],
            "bos_event": event.event,
            "direction": direction.value,
            "confirmation_index": cidx,
            "confirmation_timestamp": frame.iloc[cidx]["timestamp"],
            "entry_index": entry_index,
            "entry_timestamp": frame.iloc[entry_index]["timestamp"],
            "bos_close": bos_close,
            "confirmation_close": confirmation,
            "entry": entry,
            "confirmed_swing": swing_price,
            "leg_size": leg,
            "entry_to_swing": risk,
            "entry_to_bos": abs(entry - bos_close),
            "entry_to_confirmation": abs(entry - confirmation),
            "entry_wait_bars": entry_index - cidx,
        })

    return pd.DataFrame(rows)


def excursion(row: pd.Series, frame: pd.DataFrame, horizon: int) -> tuple[float, float]:
    index = int(row["entry_index"])
    entry = float(row["entry"])
    end = min(index + horizon, len(frame) - 1)
    highs = frame.iloc[index + 1:end + 1]["high"].astype(float)
    lows = frame.iloc[index + 1:end + 1]["low"].astype(float)

    if row["direction"] == "UP":
        mfe = highs.max() - entry
        mae = entry - lows.min()
    else:
        mfe = entry - lows.min()
        mae = highs.max() - entry
    return float(mfe), float(mae)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/research/xauusd_m30_structure_trade_geometry.csv"),
    )
    args = parser.parse_args()

    frame = load_mt5_csv(args.csv)
    candidates = build_candidates(frame)
    if candidates.empty:
        print("No candidates.")
        return

    boundary = pd.Timestamp("2026-03-18")
    rows = []

    for _, candidate in candidates.iterrows():
        for horizon in HORIZONS:
            mfe, mae = excursion(candidate, frame, horizon)
            rows.append({
                **candidate.to_dict(),
                "period": (
                    "development"
                    if pd.Timestamp(candidate["confirmation_timestamp"]) < boundary
                    else "historical_oos"
                ),
                "horizon": horizon,
                "mfe_price": mfe,
                "mae_price": mae,
                "mfe_r": mfe / candidate["entry_to_swing"],
                "mae_r": mae / candidate["entry_to_swing"],
            })

    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    summary = []
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

                summary.append({
                    "period": period,
                    "direction": direction,
                    "horizon": horizon,
                    "n": len(subset),
                    "entry_to_swing_median": subset["entry_to_swing"].median(),
                    "entry_to_bos_median": subset["entry_to_bos"].median(),
                    "entry_wait_median": subset["entry_wait_bars"].median(),
                    "mfe_price_median": subset["mfe_price"].median(),
                    "mae_price_median": subset["mae_price"].median(),
                    "mfe_ge_1R": (subset["mfe_r"] >= 1.0).mean(),
                    "mfe_ge_2R": (subset["mfe_r"] >= 2.0).mean(),
                    "mae_le_1R": (subset["mae_r"] <= 1.0).mean(),
                })

    print("=== Structure Trade Geometry Research ===")
    print("BOS -> valid swing -> 50% retracement entry")
    print("No FVG / OB filter")
    print()
    print(pd.DataFrame(summary).to_string(index=False))
    print()
    print("Candidates:", len(candidates))
    print("Artifact:", args.output)


if __name__ == "__main__":
    main()
