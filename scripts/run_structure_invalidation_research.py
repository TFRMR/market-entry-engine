"""Compare structural invalidation references after BOS + 50% pullback.

Direction comes from BOS. Entry is the first touch of the 50% retracement
between confirmation close and the confirmed same-direction swing.

Compare risk references:
- CONFIRMED_SWING: confirmed pullback swing
- BOS_LEVEL: structural price broken by the BOS event
- BOS_CLOSE: BOS candle close

The study keeps the entry fixed and only changes the invalidation reference,
so R-normalized outcomes can be compared without changing the entry sample.
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
INVALIDATIONS = ("CONFIRMED_SWING", "BOS_LEVEL", "BOS_CLOSE")


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

        bos_candle = frame.iloc[event.index]
        confirmation = float(frame.iloc[cidx]["close"])
        swing_price = float(swing.price)
        bos_close = float(bos_candle["close"])
        bos_level = (
            float(event.swing_price)
            if event.swing_price is not None
            else bos_close
        )

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

        rows.append({
            "bos_index": event.index,
            "bos_timestamp": bos_candle["timestamp"],
            "bos_event": event.event,
            "bos_scope": event.scope.value if event.scope else None,
            "direction": direction.value,
            "confirmation_index": cidx,
            "confirmation_timestamp": frame.iloc[cidx]["timestamp"],
            "entry_index": entry_index,
            "entry_timestamp": frame.iloc[entry_index]["timestamp"],
            "entry": entry_target,
            "confirmed_swing": swing_price,
            "bos_level": bos_level,
            "bos_close": bos_close,
            "leg_size": leg,
            "entry_wait_bars": entry_index - cidx,
        })

    return pd.DataFrame(rows)


def invalidation_price(row: pd.Series, kind: str) -> float:
    if kind == "CONFIRMED_SWING":
        return float(row["confirmed_swing"])
    if kind == "BOS_LEVEL":
        return float(row["bos_level"])
    return float(row["bos_close"])


def excursion(
    row: pd.Series,
    frame: pd.DataFrame,
    horizon: int,
    invalidation: float,
) -> tuple[float, float, float]:
    index = int(row["entry_index"])
    entry = float(row["entry"])
    risk = abs(entry - invalidation)
    end = min(index + horizon, len(frame) - 1)
    highs = frame.iloc[index + 1:end + 1]["high"].astype(float)
    lows = frame.iloc[index + 1:end + 1]["low"].astype(float)

    if row["direction"] == "UP":
        mfe = highs.max() - entry
        mae = entry - lows.min()
    else:
        mfe = entry - lows.min()
        mae = highs.max() - entry

    return float(mfe), float(mae), float(risk)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/research/xauusd_m30_structure_invalidation_research.csv"),
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
        for kind in INVALIDATIONS:
            invalidation = invalidation_price(candidate, kind)
            risk = abs(float(candidate["entry"]) - invalidation)
            if risk <= 0:
                continue

            for horizon in HORIZONS:
                mfe, mae, _ = excursion(candidate, frame, horizon, invalidation)
                rows.append({
                    **candidate.to_dict(),
                    "period": (
                        "development"
                        if pd.Timestamp(candidate["confirmation_timestamp"]) < boundary
                        else "historical_oos"
                    ),
                    "invalidation": kind,
                    "invalidation_price": invalidation,
                    "risk_price": risk,
                    "horizon": horizon,
                    "mfe_price": mfe,
                    "mae_price": mae,
                    "mfe_r": mfe / risk,
                    "mae_r": mae / risk,
                })

    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    summary = []
    for period in ("development", "historical_oos"):
        for kind in INVALIDATIONS:
            for direction in ("UP", "DOWN"):
                for horizon in HORIZONS:
                    subset = result[
                        (result["period"] == period)
                        & (result["invalidation"] == kind)
                        & (result["direction"] == direction)
                        & (result["horizon"] == horizon)
                    ]
                    if subset.empty:
                        continue
                    summary.append({
                        "period": period,
                        "invalidation": kind,
                        "direction": direction,
                        "horizon": horizon,
                        "n": len(subset),
                        "risk_median": subset["risk_price"].median(),
                        "mfe_price_median": subset["mfe_price"].median(),
                        "mae_price_median": subset["mae_price"].median(),
                        "mfe_ge_1R": (subset["mfe_r"] >= 1.0).mean(),
                        "mfe_ge_2R": (subset["mfe_r"] >= 2.0).mean(),
                        "mae_le_1R": (subset["mae_r"] <= 1.0).mean(),
                        "mae_le_2R": (subset["mae_r"] <= 2.0).mean(),
                    })

    print("=== Structure Invalidation Research ===")
    print("BOS -> valid swing -> 50% retracement entry")
    print("Compare structural invalidation references")
    print()
    print(pd.DataFrame(summary).to_string(index=False))
    print()
    print("Candidates:", len(candidates))
    print("Artifact:", args.output)


if __name__ == "__main__":
    main()
