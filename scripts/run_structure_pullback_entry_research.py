"""Research structure-first pullback entry locations.

Scenario:
    BOS
    -> next same-direction valid swing confirmation
    -> wait for retracement toward the confirmed swing
    -> compare entry at fixed retracement fractions of the BOS continuation leg.

The structure defines direction and invalidation. No FVG/OB filter.
This is a distribution study, not a trading signal.
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
RETRACE_LEVELS = (0.25, 0.50, 0.75)


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

        bos_price = float(frame.iloc[event.index]["close"])
        confirmation_price = float(frame.iloc[cidx]["close"])
        swing_price = float(swing.price)

        if direction is Direction.UP:
            leg = confirmation_price - swing_price
        else:
            leg = swing_price - confirmation_price
        if leg <= 0:
            continue

        rows.append({
            "bos_index": event.index,
            "bos_timestamp": frame.iloc[event.index]["timestamp"],
            "bos_event": event.event,
            "bos_scope": event.scope.value if event.scope else None,
            "direction": direction.value,
            "confirmation_index": cidx,
            "confirmation_timestamp": frame.iloc[cidx]["timestamp"],
            "bos_close": bos_price,
            "confirmation_close": confirmation_price,
            "confirmed_swing": swing_price,
            "leg_size": leg,
        })

    return pd.DataFrame(rows)


def find_entry(row: pd.Series, frame: pd.DataFrame, retrace: float) -> tuple[int, float] | None:
    cidx = int(row["confirmation_index"])
    direction = row["direction"]
    swing = float(row["confirmed_swing"])
    confirmation = float(row["confirmation_close"])
    target = (
        confirmation - retrace * (confirmation - swing)
        if direction == "UP"
        else confirmation + retrace * (swing - confirmation)
    )

    for index in range(cidx + 1, len(frame)):
        candle = frame.iloc[index]
        low = float(candle["low"])
        high = float(candle["high"])

        if direction == "UP" and low <= target:
            return index, target
        if direction == "DOWN" and high >= target:
            return index, target

    return None


def excursion(row: pd.Series, frame: pd.DataFrame, horizon: int) -> tuple[float, float]:
    index = int(row["entry_index"])
    entry = float(row["entry"])
    risk = abs(entry - float(row["confirmed_swing"]))
    end = min(index + horizon, len(frame) - 1)

    highs = frame.iloc[index + 1:end + 1]["high"].astype(float)
    lows = frame.iloc[index + 1:end + 1]["low"].astype(float)

    if direction := row["direction"] == "UP":
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
        default=Path("data/research/xauusd_m30_structure_pullback_entry_research.csv"),
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
        for retrace in RETRACE_LEVELS:
            found = find_entry(candidate, frame, retrace)
            if found is None:
                continue
            entry_index, entry = found
            risk = abs(entry - float(candidate["confirmed_swing"]))
            if risk <= 0:
                continue

            for horizon in HORIZONS:
                mfe, mae = excursion(
                    pd.Series({
                        **candidate.to_dict(),
                        "entry_index": entry_index,
                        "entry": entry,
                    }),
                    frame,
                    horizon,
                )
                rows.append({
                    **candidate.to_dict(),
                    "period": (
                        "development"
                        if pd.Timestamp(candidate["confirmation_timestamp"]) < boundary
                        else "historical_oos"
                    ),
                    "retrace": retrace,
                    "entry_index": entry_index,
                    "entry_timestamp": frame.iloc[entry_index]["timestamp"],
                    "bars_after_confirmation": entry_index - int(candidate["confirmation_index"]),
                    "entry": entry,
                    "risk_price": risk,
                    "horizon": horizon,
                    "mfe_r": mfe,
                    "mae_r": mae,
                })

    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    summary = []
    for period in ("development", "historical_oos"):
        for retrace in RETRACE_LEVELS:
            for direction in ("UP", "DOWN"):
                for horizon in HORIZONS:
                    subset = result[
                        (result["period"] == period)
                        & (result["retrace"] == retrace)
                        & (result["direction"] == direction)
                        & (result["horizon"] == horizon)
                    ]
                    if subset.empty:
                        continue
                    summary.append({
                        "period": period,
                        "retrace": retrace,
                        "direction": direction,
                        "horizon": horizon,
                        "n": len(subset),
                        "entry_wait_median": subset["bars_after_confirmation"].median(),
                        "mfe_median": subset["mfe_r"].median(),
                        "mfe_ge_0.5R": (subset["mfe_r"] >= 0.5).mean(),
                        "mfe_ge_1R": (subset["mfe_r"] >= 1.0).mean(),
                        "mfe_ge_2R": (subset["mfe_r"] >= 2.0).mean(),
                        "mae_median": subset["mae_r"].median(),
                        "mae_le_1R": (subset["mae_r"] <= 1.0).mean(),
                    })

    print("=== Structure-First Pullback Entry Research ===")
    print("BOS -> valid swing confirmation -> retracement entry")
    print("No FVG / OB filter")
    print()
    print(pd.DataFrame(summary).to_string(index=False))
    print()
    print("Candidates:", len(candidates))
    print("Artifact:", args.output)


if __name__ == "__main__":
    main()
